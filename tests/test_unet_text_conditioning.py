import types

import pytest
import torch

from con_stable_diff_model.models.MultiHeadAttention import MultiHeadCrossAttention
from con_stable_diff_model.models.UNetBlocks import (
    BottleNeck,
    DownSamplingBlock,
    UpSamplingBlock,
)
from con_stable_diff_model.modules.UNet import UNet


TEXT_EMBED_DIM = 48


def _stub_cross_attention(attn_module: MultiHeadCrossAttention, seen_calls, monkeypatch):
    """Replace forward with a minimal stub that records query/context shapes."""

    def _forward(self, query, context):
        seen_calls.append((tuple(query.shape), tuple(context.shape)))
        return torch.zeros_like(query)

    monkeypatch.setattr(
        attn_module, "forward", types.MethodType(_forward, attn_module)
    )


def test_downsampling_block_uses_text_context(monkeypatch):
    batch, height, width = 2, 8, 8
    in_channels, out_channels = 16, 32
    time_dim = 32
    context = torch.randn(batch, 5, TEXT_EMBED_DIM)

    block = DownSamplingBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        time_emb_dim=time_dim,
        down_sample=False,
        num_layers=1,
        use_attn=False,
        cross_attn=True,
        cross_cont_dim=TEXT_EMBED_DIM,
    )

    seen = []
    for attn in block.cross_attention:
        _stub_cross_attention(attn, seen, monkeypatch)

    x = torch.randn(batch, in_channels, height, width)
    t = torch.randn(batch, time_dim)

    out = block(x, time_emb=t, context=context)

    assert out.shape == (batch, out_channels, height, width)
    assert seen and seen[0][1] == (batch, 5, TEXT_EMBED_DIM)


def test_bottleneck_uses_text_context(monkeypatch):
    batch, height, width = 2, 4, 4
    channels = 32
    time_dim = 32
    context = torch.randn(batch, 6, TEXT_EMBED_DIM)

    block = BottleNeck(
        in_channels=channels,
        out_channels=channels,
        time_emb_dim=time_dim,
        num_layers=1,
        cross_attn=True,
        cross_cont_dim=TEXT_EMBED_DIM,
    )

    seen = []
    for attn in block.cross_attention:
        _stub_cross_attention(attn, seen, monkeypatch)

    x = torch.randn(batch, channels, height, width)
    t = torch.randn(batch, time_dim)

    out = block(x, time_emb=t, context=context)

    assert out.shape == (batch, channels, height, width)
    assert seen and seen[0][1] == (batch, 6, TEXT_EMBED_DIM)


def test_upsampling_block_uses_text_context(monkeypatch):
    batch, height, width = 2, 4, 4
    in_channels, skip_channels, out_channels = 32, 16, 16
    time_dim = 32
    context = torch.randn(batch, 4, TEXT_EMBED_DIM)

    block = UpSamplingBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        skip_channels=skip_channels,
        time_emb_dim=time_dim,
        up_sample=False,
        num_layers=1,
        use_attn=False,
        cross_attn=True,
        cross_cont_dim=TEXT_EMBED_DIM,
    )

    seen = []
    for attn in block.cross_attention:
        _stub_cross_attention(attn, seen, monkeypatch)

    x = torch.randn(batch, in_channels, height, width)
    skip = torch.randn(batch, skip_channels, height, width)
    t = torch.randn(batch, time_dim)

    out = block(x, time_emb=t, out_down=skip, context=context)

    assert out.shape == (batch, out_channels, height, width)
    assert seen and seen[0][1] == (batch, 4, TEXT_EMBED_DIM)


def _build_text_conditioned_unet():
    return UNet(
        UnetParams={
            "down_channels": [16, 32],
            "mid_channels": [32, 16],
            "down_sample": [False],
            "attn_down": [False],
            "time_emb_dim": 32,
            "im_channels": 4,
            "num_down_layers": 1,
            "num_mid_layers": 1,
            "num_up_layers": 1,
            "model_config": {
                "condition_config": {
                    "condition_types": ["text"],
                    "text_condition_config": {"text_embed_dim": TEXT_EMBED_DIM},
                }
            },
        }
    )


def test_unet_requires_text_cond_input():
    model = _build_text_conditioned_unet()
    x = torch.randn(1, 4, 8, 8)
    t = torch.tensor(0.0)

    with pytest.raises(AssertionError):
        _ = model(x, t, cond_input={})


def test_unet_forward_with_text_condition(monkeypatch):
    batch, height, width = 2, 8, 8
    seq_len = 6
    model = _build_text_conditioned_unet()

    seen = []
    for module in model.modules():
        if isinstance(module, MultiHeadCrossAttention):
            _stub_cross_attention(module, seen, monkeypatch)

    x = torch.randn(batch, model.in_channels, height, width)
    t = torch.tensor([0.0, 1.0])
    context = torch.randn(batch, seq_len, TEXT_EMBED_DIM)

    out = model(x, t, cond_input={"text": context})

    assert out.shape == x.shape
    assert seen
    assert all(ctx_shape == (batch, seq_len, TEXT_EMBED_DIM) for _, ctx_shape in seen)
