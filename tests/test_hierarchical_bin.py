from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.nuclr_v2ga_hier_bin import NuclrV2gaHierBin
from src.nn.hierarchical import TemporalMerge
from src.utils import Precision, instantiate


def make_model(**overrides):
    kwargs = dict(
        ctx_duration=10.0,
        latent_step=1.0,
        bin_size=0.5,
        precision=Precision("fp32"),
        dim=16,
        fusion_dim=16,
        stage_depths=[1, 1, 1],
        stage_merge_factors=[2, 2],
        self_heads=2,
        dim_head=8,
        cross_scale_heads=2,
        cross_scale_depth=1,
        atn_dropout=0.0,
        lin_dropout=0.0,
        merge_norm=True,
        merge_activation="gelu",
        stage_summary_type="attention_pool",
        use_scale_embeddings=True,
        cross_scale_use_cls=True,
        fusion_type="transformer",
    )
    kwargs.update(overrides)
    model = NuclrV2gaHierBin(**kwargs)
    model.eval()
    return model


def make_bins(model, unit_count, offset=0):
    total = unit_count * model.num_latents * model.bins_per_latent
    values = torch.arange(offset, offset + total, dtype=torch.float32)
    values = values.remainder(3).to(torch.int16)
    return values.reshape(unit_count * model.num_latents, model.bins_per_latent)


def test_forward_handles_odd_stage_lengths():
    torch.manual_seed(0)
    model = make_model()
    bins = make_bins(model, unit_count=5)
    unit_seqlen = torch.tensor([2, 3])

    with torch.inference_mode():
        y = model(bins=bins, unit_seqlen=unit_seqlen)

    assert model.stage_lengths == [10, 5, 3]
    assert y.shape == (5, 16)
    assert torch.isfinite(y).all()


def test_temporal_merge_zero_pads_final_group():
    merge = TemporalMerge(
        dim=4,
        merge_factor=2,
        merge_norm=True,
        merge_activation="gelu",
    )
    x = torch.randn(3, 5, 4)

    y = merge(x)

    assert y.shape == (3, 3, 4)
    assert torch.isfinite(y).all()


def test_unit_permutation_equivariance():
    torch.manual_seed(1)
    model = make_model()
    bins = make_bins(model, unit_count=4).reshape(4, model.num_latents, -1)
    unit_seqlen = torch.tensor([4])
    perm = torch.tensor([2, 0, 3, 1])
    inv_perm = torch.argsort(perm)

    with torch.inference_mode():
        y = model(bins=bins.reshape(-1, model.bins_per_latent), unit_seqlen=unit_seqlen)
        y_perm = model(
            bins=bins[perm].reshape(-1, model.bins_per_latent),
            unit_seqlen=unit_seqlen,
        )

    assert torch.allclose(y, y_perm[inv_perm], atol=1e-5, rtol=1e-5)


def test_block_diagonal_spatial_isolation():
    torch.manual_seed(2)
    model = make_model()
    unit_seqlen = torch.tensor([2, 3])
    bins = make_bins(model, unit_count=5).reshape(5, model.num_latents, -1)
    changed = bins.clone()
    changed[2:] = make_bins(model, unit_count=3, offset=100).reshape(
        3,
        model.num_latents,
        -1,
    )

    with torch.inference_mode():
        y = model(bins=bins.reshape(-1, model.bins_per_latent), unit_seqlen=unit_seqlen)
        y_changed = model(
            bins=changed.reshape(-1, model.bins_per_latent),
            unit_seqlen=unit_seqlen,
        )

    assert torch.allclose(y[:2], y_changed[:2], atol=1e-5, rtol=1e-5)


def test_hydra_model_configs_instantiate():
    config_names = [
        "nuclr_v2ga_hier_bin.yaml",
        "nuclr_v2ga_hier_bin_mean_summary.yaml",
        "nuclr_v2ga_hier_bin_no_scale_embed.yaml",
        "nuclr_v2ga_hier_bin_no_xscale.yaml",
        "nuclr_v2ga_hier_bin_last_scale.yaml",
    ]
    config_root = Path("configs/model")

    for config_name in config_names:
        cfg = OmegaConf.load(config_root / config_name)
        cfg.ctx_duration = 10.0
        cfg.bin_size = 0.5
        cfg.dim = 16
        cfg.fusion_dim = 16
        cfg.self_heads = 2
        cfg.dim_head = 8
        cfg.cross_scale_heads = 2

        model = instantiate(cfg, precision=Precision("fp32"))

        assert isinstance(model, NuclrV2gaHierBin)
        assert model.stage_lengths == [10, 5, 3]
