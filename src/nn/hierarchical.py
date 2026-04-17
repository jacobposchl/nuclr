from typing import Iterable, Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from .attention import BlockDiagonalMask, RotarySelfAttention


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float, pre_norm: bool):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.in_proj = nn.Linear(dim, 2 * dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)
        self.dp = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y, gate = self.in_proj(y).chunk(2, dim=-1)
        y = self.act(gate) * y
        y = self.dp(y)
        return self.out_proj(y)


class TemporalMerge(nn.Module):
    def __init__(
        self,
        dim: int,
        merge_factor: int,
        merge_norm: bool,
        merge_activation: str | None,
    ):
        super().__init__()
        if merge_factor < 1:
            raise ValueError(f"merge_factor must be >= 1, got {merge_factor}")

        self.dim = dim
        self.merge_factor = merge_factor
        self.proj = nn.Linear(dim * merge_factor, dim)
        self.norm = nn.LayerNorm(dim) if merge_norm else nn.Identity()

        if merge_activation is None or merge_activation == "none":
            self.act = nn.Identity()
        elif merge_activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported merge_activation: {merge_activation}")

    def forward(self, x: Tensor) -> Tensor:
        if self.merge_factor == 1:
            return x

        num_units, num_tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")

        pad = (-num_tokens) % self.merge_factor
        if pad:
            x = torch.cat((x, x.new_zeros(num_units, pad, dim)), dim=1)

        x = rearrange(x, "u (l f) d -> u l (f d)", f=self.merge_factor)
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)


class HierarchicalStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        atn_dropout: float,
        lin_dropout: float,
    ):
        super().__init__()
        self.blocks: Iterable[Iterable[nn.Module]] = nn.ModuleList(  # type: ignore
            [
                nn.ModuleList(
                    [
                        RotarySelfAttention(
                            dim,
                            heads,
                            dim_head,
                            atn_dropout,
                            rotate_value=True,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        RotarySelfAttention(
                            dim,
                            heads,
                            dim_head,
                            atn_dropout,
                            rotate_value=True,
                        ),
                        FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.dp = nn.Dropout(lin_dropout)

    def forward(
        self,
        x: Tensor,
        unit_seqlen: Tensor,
        rotary: Tensor,
    ) -> Tensor:
        num_units, num_tokens, _ = x.shape
        spatial_attn_bias = BlockDiagonalMask.from_seqlens(
            q_seqlen=unit_seqlen.tolist() * num_tokens,
            device=x.device,
        )

        for t_attn, t_ffn, s_attn, s_ffn in self.blocks:
            x = x + self.dp(t_attn(x=x, rotary=rotary))
            x = x + self.dp(t_ffn(x))

            x = rearrange(x, "u l d -> (l u) d", l=num_tokens, u=num_units)
            x = x + self.dp(s_attn(x, attn_bias=spatial_attn_bias))
            x = x + self.dp(s_ffn(x))
            x = rearrange(x, "(l u) d -> u l d", l=num_tokens, u=num_units)

        return x


class StageSummary(nn.Module):
    def __init__(
        self,
        dim: int,
        fusion_dim: int,
        summary_type: Literal["attention_pool", "mean", "mean_proj"],
    ):
        super().__init__()
        self.summary_type = summary_type
        self.norm = nn.LayerNorm(dim)

        if summary_type == "attention_pool":
            self.score = nn.Linear(dim, 1)
            self.proj = nn.Linear(dim, fusion_dim)
        elif summary_type == "mean_proj":
            self.score = None
            self.proj = nn.Linear(dim, fusion_dim)
        elif summary_type == "mean":
            self.score = None
            self.proj = (
                nn.Identity() if dim == fusion_dim else nn.Linear(dim, fusion_dim)
            )
        else:
            raise ValueError(f"Unsupported stage_summary_type: {summary_type}")

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        if self.summary_type == "attention_pool":
            assert self.score is not None
            weights = self.score(y).squeeze(-1).softmax(dim=1)
            pooled = torch.sum(x * weights[..., None], dim=1)
        else:
            pooled = x.mean(dim=1)
        return self.proj(pooled)


class CrossScaleFusion(nn.Module):
    def __init__(
        self,
        num_stages: int,
        dim: int,
        depth: int,
        heads: int,
        dropout: float,
        use_cls: bool,
        use_scale_embeddings: bool,
        fusion_type: Literal["transformer", "mean", "gated", "last_scale"],
    ):
        super().__init__()
        self.num_stages = num_stages
        self.dim = dim
        self.use_cls = use_cls
        self.fusion_type = fusion_type

        if use_scale_embeddings:
            self.scale_emb = nn.Parameter(torch.empty(num_stages, dim))
            nn.init.normal_(self.scale_emb, std=0.02)
        else:
            self.register_parameter("scale_emb", None)

        if fusion_type == "transformer":
            if use_cls:
                self.cls = nn.Parameter(torch.empty(1, 1, dim))
                nn.init.normal_(self.cls, std=0.02)
            else:
                self.register_parameter("cls", None)

            if depth > 0:
                layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=4 * dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
            else:
                self.transformer = nn.Identity()
            self.gate = None
        elif fusion_type == "gated":
            self.register_parameter("cls", None)
            self.transformer = nn.Identity()
            self.gate = nn.Linear(dim, 1)
        elif fusion_type in {"mean", "last_scale"}:
            self.register_parameter("cls", None)
            self.transformer = nn.Identity()
            self.gate = None
        else:
            raise ValueError(f"Unsupported cross-scale fusion_type: {fusion_type}")

    def forward(self, summaries: list[Tensor] | Tensor) -> Tensor:
        if isinstance(summaries, list):
            x = torch.stack(summaries, dim=1)
        else:
            x = summaries

        if x.size(1) != self.num_stages:
            raise ValueError(f"Expected {self.num_stages} stages, got {x.size(1)}")

        if self.scale_emb is not None:
            x = x + self.scale_emb[None, :, :]

        if self.fusion_type == "mean":
            return x.mean(dim=1)
        if self.fusion_type == "last_scale":
            return x[:, -1]
        if self.fusion_type == "gated":
            assert self.gate is not None
            weights = self.gate(x).squeeze(-1).softmax(dim=1)
            return torch.sum(x * weights[..., None], dim=1)

        if self.use_cls:
            assert self.cls is not None
            cls = self.cls.expand(x.size(0), -1, -1)
            x = torch.cat((cls, x), dim=1)
            x = self.transformer(x)
            return x[:, 0]

        x = self.transformer(x)
        return x.mean(dim=1)
