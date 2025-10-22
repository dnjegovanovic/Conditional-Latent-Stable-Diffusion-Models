import pytest
import torch

from ddpm_model.models.TimeEmbedding import time_embedding_fun, TimeEmbedding


DEVICE_LIST = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICE_LIST)
def test_time_embedding_fun_shape_and_zero_values(device: str):
    # Given zero timesteps, sin(0)=0 and cos(0)=1
    B = 3
    dim = 320  # even
    t = torch.zeros(B, dtype=torch.float32, device=device)

    emb = time_embedding_fun(t, time_embedding_dim=dim)

    assert emb.shape == (B, dim)
    # First half are sin components -> zeros
    first_half = emb[:, : dim // 2]
    # Second half are cos components -> ones
    second_half = emb[:, dim // 2 :]

    assert torch.allclose(first_half, torch.zeros_like(first_half))
    assert torch.allclose(second_half, torch.ones_like(second_half))
    assert emb.device.type == device


def test_time_embedding_fun_odd_dimension_raises():
    t = torch.tensor([0.0])
    with pytest.raises(AssertionError):
        _ = time_embedding_fun(t, time_embedding_dim=321)  # not divisible by 2


@pytest.mark.parametrize("use_direct_map", [True, False])
@pytest.mark.parametrize("device", DEVICE_LIST)
def test_time_embedding_module_output_shapes(use_direct_map: bool, device: str):
    n_embd = 128
    B = 4
    x = torch.randn(B, n_embd, device=device)

    m = TimeEmbedding(n_embd=n_embd, use_direct_map=use_direct_map).to(device)
    m.eval()

    with torch.no_grad():
        y = m(x)

    expected_dim = n_embd if use_direct_map else 4 * n_embd
    assert y.shape == (B, expected_dim)
    assert y.device.type == device
