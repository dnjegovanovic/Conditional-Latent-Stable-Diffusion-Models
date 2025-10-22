import pytest
import torch

from ddpm_model.models.MultiHeadAttention import (
    MultiHeadSelfAttention,
    MultiHeadCrossAttention,
)


DEVICE_LIST = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _zero_linear(layer: torch.nn.Linear):
    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


@pytest.mark.parametrize("E,H", [(8, 1), (8, 2), (8, 4), (16, 1), (16, 4)])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_mhsa_output_shape_and_device(device: str, E: int, H: int):
    B, L = 2, 5
    x = torch.randn(B, L, E, device=device)

    attn = MultiHeadSelfAttention(num_heads=H, embed_dim=E).to(device)
    y = attn(x)

    assert y.shape == (B, L, E)
    assert y.device.type == device


def _set_mhsa_as_controlled_identity(attn: MultiHeadSelfAttention):
    # Configure projections so that:
    # - q and k depend only on the first half of channels (constant across positions)
    # - v depends only on the second half of channels (varies with position)
    E = attn.output_proj.in_features
    half = E // 2

    # input_proj maps E -> 3E stacked [Wq; Wk; Wv]
    _zero_linear(attn.input_proj)
    with torch.no_grad():
        # Wq block occupies rows [0:E). Make Q depend only on first half inputs.
        # Set first-half rows to identity on first-half columns; rest remain zero.
        attn.input_proj.weight[:half, :half] = torch.eye(half)

        # Wk block occupies rows [E:2E). Make K depend only on first half inputs.
        attn.input_proj.weight[E : E + half, :half] = torch.eye(half)

        # Wv block occupies rows [2E:3E). Make V depend only on second half inputs
        # by setting last-half rows to identity on last-half columns.
        attn.input_proj.weight[2 * E + half : 3 * E, half:] = torch.eye(half)

    # output as identity
    _zero_linear(attn.output_proj)
    with torch.no_grad():
        attn.output_proj.weight.copy_(torch.eye(E))


@pytest.mark.parametrize("E,H", [(8, 1), (8, 2), (8, 4), (16, 2)])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_mhsa_causal_mask_changes_outputs(device: str, E: int, H: int):
    # Setup deterministic attention where weights are uniform across all keys
    # (through constant Q, K), and V carries position-dependent signal.
    B, L = 1, 4
    half = E // 2

    # Build input where first half channels are constant (drive Q,K),
    # and second half encode position index (drive V).
    x = torch.zeros(B, L, E, device=device)
    x[:, :, :half] = 1.0  # constant across positions
    for i in range(L):
        x[:, i, half] = float(i)  # vary one dim in the V-part

    attn = MultiHeadSelfAttention(num_heads=H, embed_dim=E).to(device)
    _set_mhsa_as_controlled_identity(attn)
    attn.eval()

    with torch.no_grad():
        y_nomask = attn(x, apply_causal_mask=False)  # uniform over all L
        y_mask = attn(x, apply_causal_mask=True)  # uniform over <= i

    # Only the second half (V channels) should carry signal
    # Unmasked: all positions should equal the global mean of indices [0,1,2,3] = 1.5
    expected_global = 1.5
    assert torch.allclose(
        y_nomask[0, 0, half], torch.tensor(expected_global, device=device)
    )
    assert torch.allclose(
        y_nomask[0, 3, half], torch.tensor(expected_global, device=device)
    )

    # Masked: position i equals mean([0..i]) = i/2 for our simple index encoding
    for i in range(L):
        expected = i / 2.0
        assert torch.allclose(y_mask[0, i, half], torch.tensor(expected, device=device))


def test_mhsa_invalid_heads_raises():
    # embed_dim not divisible by num_heads -> assertion on init
    with pytest.raises(AssertionError):
        _ = MultiHeadSelfAttention(num_heads=3, embed_dim=8)


@pytest.mark.parametrize(
    "E,H,cross_dim",
    [
        (8, 1, 8),
        (8, 2, 8),
        (8, 4, 8),
        (8, 2, 12),  # mixed dim (cross_dim > E)
        (16, 4, 16),
        (16, 8, 16),
        (16, 2, 8),  # mixed dim (cross_dim < E)
        (16, 4, 24),  # mixed dim (cross_dim > E)
    ],
)
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_mhca_output_shape_and_device(device: str, E: int, H: int, cross_dim: int):
    B, Lq, Lkv = 2, 3, 5
    query = torch.randn(B, Lq, E, device=device)
    context = torch.randn(B, Lkv, cross_dim, device=device)

    attn = MultiHeadCrossAttention(num_heads=H, embed_dim=E, cross_dim=cross_dim).to(
        device
    )
    y = attn(query, context)

    assert y.shape == (B, Lq, E)
    assert y.device.type == device


def _set_mhca_uniform_weights_and_value_mean(attn: MultiHeadCrossAttention):
    # Set Q and K to constant (thus uniform attention), V passes through last-half.
    E = attn.output_proj.in_features
    half = E // 2

    _zero_linear(attn.query_proj)
    _zero_linear(attn.key_proj)
    _zero_linear(attn.value_proj)
    _zero_linear(attn.output_proj)

    with torch.no_grad():
        # Q depends only on first half (constant across positions)
        attn.query_proj.weight[:half, :half] = torch.eye(half)
        # K depends only on first half (constant across positions)
        attn.key_proj.weight[:half, :half] = torch.eye(half)
        # V: route a single cross feature (column=half) into output channel (row=half).
        # This works for cross_dim >= half+1.
        attn.value_proj.weight.zero_()
        attn.value_proj.weight[half, half] = 1.0
        # Output identity
        attn.output_proj.weight.copy_(torch.eye(E))


@pytest.mark.parametrize("E,H,cross_dim", [(8, 2, 8), (8, 2, 12), (16, 4, 16)])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_mhca_uniform_attention_means_values(
    device: str, E: int, H: int, cross_dim: int
):
    # Make attention uniform over context tokens and verify output equals context V-mean
    B, Lq, Lkv = 1, 2, 4
    half = E // 2

    # Queries: first half ones (drives Q), second half zeros
    query = torch.zeros(B, Lq, E, device=device)
    query[:, :, :half] = 1.0

    # Context: first half ones (drives K), second half encodes index in one dim (drives V)
    context = torch.zeros(B, Lkv, cross_dim, device=device)
    context[:, :, :half] = 1.0
    for i in range(Lkv):
        context[:, i, half] = float(i)

    attn = MultiHeadCrossAttention(num_heads=H, embed_dim=E, cross_dim=cross_dim).to(
        device
    )
    _set_mhca_uniform_weights_and_value_mean(attn)
    attn.eval()

    with torch.no_grad():
        y = attn(query, context)

    expected_mean = sum(range(Lkv)) / Lkv
    # Each query position should get the same mean value in the selected V channel
    for i in range(Lq):
        assert torch.allclose(y[0, i, half], torch.tensor(expected_mean, device=device))
