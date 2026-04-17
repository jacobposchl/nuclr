import math
from typing import Literal

import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor, nn

from torch_brain.data import chain

from . import BaseNeuronEncoder
from ..dataset import SpikeData, ViewData
from ..nn.hierarchical import (
    CrossScaleFusion,
    HierarchicalStage,
    StageSummary,
    TemporalMerge,
)
from ..nn.rotary_embedding import RotaryEmbedding
from ..utils import Precision, is_divisible


class NuclrV2gaHierBin(BaseNeuronEncoder):
    def __init__(
        self,
        ctx_duration: float,
        latent_step: float,
        bin_size: float,
        precision: Precision,
        dim: int,
        fusion_dim: int,
        stage_depths: list[int],
        stage_merge_factors: list[int],
        self_heads: int,
        dim_head: int,
        cross_scale_heads: int,
        cross_scale_depth: int,
        atn_dropout: float,
        lin_dropout: float,
        merge_norm: bool,
        merge_activation: str | None,
        stage_summary_type: Literal["attention_pool", "mean", "mean_proj"],
        use_scale_embeddings: bool,
        cross_scale_use_cls: bool,
        fusion_type: Literal[
            "transformer", "mean", "gated", "last_scale"
        ] = "transformer",
        rot_ratio: float = 0.5,
    ):
        super().__init__()

        if len(stage_depths) == 0:
            raise ValueError("stage_depths must contain at least one stage")
        if len(stage_merge_factors) != len(stage_depths) - 1:
            raise ValueError(
                "stage_merge_factors must have one fewer element than stage_depths"
            )

        assert is_divisible(latent_step, bin_size)
        assert is_divisible(ctx_duration, latent_step)

        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.bin_size = bin_size
        self.precision = precision
        self.dim = dim
        self.emb_dim = fusion_dim
        self.stage_depths = [int(x) for x in stage_depths]
        self.stage_merge_factors = [int(x) for x in stage_merge_factors]

        self.bins_per_latent = int(latent_step / bin_size)
        self.num_latents = int(math.ceil(ctx_duration / latent_step))
        self.stage_lengths = self._compute_stage_lengths()

        t_min, t_max = 1.0, 8.0 * self.num_latents
        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.read_in = nn.Linear(self.bins_per_latent, dim)
        self.stages = nn.ModuleList(
            [
                HierarchicalStage(
                    dim=dim,
                    depth=depth,
                    heads=self_heads,
                    dim_head=dim_head,
                    atn_dropout=atn_dropout,
                    lin_dropout=lin_dropout,
                )
                for depth in self.stage_depths
            ]
        )
        self.merges = nn.ModuleList(
            [
                TemporalMerge(
                    dim=dim,
                    merge_factor=merge_factor,
                    merge_norm=merge_norm,
                    merge_activation=merge_activation,
                )
                for merge_factor in self.stage_merge_factors
            ]
        )
        self.stage_summaries = nn.ModuleList(
            [
                StageSummary(
                    dim=dim,
                    fusion_dim=fusion_dim,
                    summary_type=stage_summary_type,
                )
                for _ in self.stage_depths
            ]
        )
        self.cross_scale = CrossScaleFusion(
            num_stages=len(self.stage_depths),
            dim=fusion_dim,
            depth=cross_scale_depth,
            heads=cross_scale_heads,
            dropout=lin_dropout,
            use_cls=cross_scale_use_cls,
            use_scale_embeddings=use_scale_embeddings,
            fusion_type=fusion_type,
        )

    def _compute_stage_lengths(self) -> list[int]:
        lengths = [self.num_latents]
        cur_len = self.num_latents
        for merge_factor in self.stage_merge_factors:
            cur_len = math.ceil(cur_len / merge_factor)
            lengths.append(cur_len)
        return lengths

    @staticmethod
    def _merge_positions(positions: Tensor, merge_factor: int) -> Tensor:
        if merge_factor == 1:
            return positions

        num_tokens = positions.numel()
        pad = (-num_tokens) % merge_factor
        mask = positions.new_ones(num_tokens)
        if pad:
            positions = torch.cat((positions, positions.new_zeros(pad)), dim=0)
            mask = torch.cat((mask, mask.new_zeros(pad)), dim=0)

        positions = rearrange(positions, "(l f) -> l f", f=merge_factor)
        mask = rearrange(mask, "(l f) -> l f", f=merge_factor)
        return (positions * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _rotary_for_positions(self, positions: Tensor, num_units: int) -> Tensor:
        rotary = self.rotary_emb(positions)
        return repeat(rotary, "l d -> u l d", u=num_units)

    def forward(
        self,
        bins: Tensor,
        unit_seqlen: Tensor,
    ) -> Tensor:
        device = bins.device
        if not torch.is_tensor(unit_seqlen):
            unit_seqlen = torch.tensor([unit_seqlen], device=device, dtype=torch.long)
        else:
            unit_seqlen = unit_seqlen.to(device=device, dtype=torch.long)

        num_units = int(unit_seqlen.sum())
        expected_tokens = num_units * self.num_latents
        if bins.size(0) != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} tokens for {num_units} units and "
                f"{self.num_latents} latents, got {bins.size(0)}"
            )

        x = self.read_in(bins.float())
        x = rearrange(x, "(u l) d -> u l d", u=num_units, l=self.num_latents)

        positions = torch.arange(
            self.num_latents,
            dtype=torch.float32,
            device=device,
        )
        positions = positions + 0.5

        summaries = []
        for stage_idx, stage in enumerate(self.stages):
            rotary = self._rotary_for_positions(positions, num_units)
            x = stage(x=x, unit_seqlen=unit_seqlen, rotary=rotary)
            summaries.append(self.stage_summaries[stage_idx](x))

            if stage_idx < len(self.merges):
                merge_factor = self.stage_merge_factors[stage_idx]
                x = self.merges[stage_idx](x)
                positions = self._merge_positions(positions, merge_factor)

        return self.cross_scale(summaries)

    def tokenize(self, view: SpikeData) -> ViewData:
        spike_times = torch.tensor(view.spikes.timestamps, dtype=torch.float32)
        spike_units = torch.tensor(view.spikes.unit_index, dtype=torch.long)
        active_units = spike_units.unique()
        assert torch.all(
            active_units[:-1] <= active_units[1:]
        ), "active_units must be sorted"

        num_bins = int(self.ctx_duration / self.bin_size)
        rate = 1 / self.bin_size
        bins = torch.zeros((len(view.units), num_bins + 1), dtype=torch.int16)
        bins.index_put_(
            indices=(spike_units, torch.floor(spike_times * rate).long()),
            values=torch.ones(len(spike_times), dtype=torch.int16),
            accumulate=True,
        )
        bins = bins[active_units][:, : self.num_latents * self.bins_per_latent]
        bins = rearrange(
            bins,
            "u (l b) -> (u l) b",
            l=self.num_latents,
            b=self.bins_per_latent,
        )

        enc_input = {
            "bins": chain(bins),
            "unit_seqlen": len(active_units),
        }

        recording_ids = np.array([view.session.id for _ in range(len(active_units))])  # type: ignore
        unit_ids = view.units.id[active_units]  # type: ignore

        data = {"enc_input": enc_input}
        return ViewData(data=data, unit_ids=unit_ids, recording_ids=recording_ids)
