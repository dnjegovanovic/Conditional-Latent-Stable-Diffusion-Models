import pytest
import torch
from torch import Tensor
from ddpm_model.models.LinearNoiseScheduler import LinearNoiseScheduler
from ddpm_model.config.core import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Fixtures for test setup
@pytest.fixture
def scheduler():
    return LinearNoiseScheduler(**vars(config), device=DEVICE)


@pytest.fixture
def test_data():
    return {
        "x_start": torch.randn(2, 3, 16, 16).to(DEVICE),  # Batch of 2 images
        "noise": torch.randn(2, 3, 16, 16).to(DEVICE),
        "t": torch.randint(0, 10, (2,)).to(DEVICE),
    }


def test_forward_process(scheduler, test_data):
    """Test forward process properties"""
    x_start = test_data["x_start"]
    noise = test_data["noise"]
    t = test_data["t"]

    # Test basic functionality
    x_noisy = scheduler.add_noise(x_start, noise, t)

    # Shape preservation
    assert x_noisy.shape == x_start.shape

    # Test boundary conditions
    # At t=0 (first timestep)
    t_zero = torch.tensor([0, 0])
    x_noisy_t0 = scheduler.add_noise(x_start, noise, t_zero)
    assert torch.allclose(
        x_noisy_t0, x_start, atol=1e-1
    ), "t=0 should have minimal noise"

    # At max timestep
    t_max = torch.full((x_start.shape[0],), scheduler.num_timesteps - 1)
    x_noisy_tmax = scheduler.add_noise(x_start, noise, t_max)
    assert torch.allclose(
        x_noisy_tmax, noise, atol=1e1
    ), "At max t, output should be nearly pure noise"


def test_reverse_process(scheduler, test_data):
    """Test reverse process properties"""
    x_t = test_data["x_start"]  # Using clean image as x_t for testing
    noise_pred = torch.zeros_like(x_t)  # Simulate perfect noise prediction

    # Test basic functionality
    for t in range(scheduler.num_timesteps - 1, -1, -1):
        x_prev, x0_pred = scheduler.sample_prev_timestep(x_t, noise_pred, t)

        # Shape preservation
        assert x_prev.shape == x_t.shape
        assert x0_pred.shape == x_t.shape

        # Clamping check
        assert torch.all(x0_pred >= -1.0) and torch.all(
            x0_pred <= 1.0
        ), "x0 should be clamped"

        # Special case handling
        if t == 0:
            # Should return mean without added noise
            assert not torch.isnan(x_prev).any(), "t=0 should produce valid output"

    # Test stochastic vs deterministic
    t_mid = scheduler.num_timesteps // 2
    x_prev_stochastic, _ = scheduler.sample_prev_timestep(x_t, noise_pred, t_mid)
    x_prev_deterministic, _ = scheduler.sample_prev_timestep(x_t, noise_pred, 0)

    # Verify stochastic sampling adds noise
    assert not torch.allclose(
        x_prev_stochastic, x_t
    ), "Intermediate steps should add noise"

    # Verify deterministic sampling (t=0)
    assert torch.allclose(
        x_prev_deterministic, x_prev_deterministic
    ), "Final step should be deterministic"


def test_scheduler_initialization(scheduler):
    """Test scheduler parameter calculations"""
    # Validate beta schedule
    assert torch.all(scheduler.betas >= 0), "All betas should be non-negative"
    assert scheduler.betas[0].cpu() == pytest.approx(0.0001, abs=1e-5)
    assert scheduler.betas[-1].cpu() == pytest.approx(0.02, abs=1e-5)

    # Validate alpha calculations
    assert torch.all(scheduler.alphas == 1 - scheduler.betas)
    assert torch.allclose(
        scheduler.alphas_cumprod, torch.cumprod(scheduler.alphas, dim=0)
    )

    # Validate precomputed terms
    assert torch.allclose(
        scheduler.sqrt_alphas_cumprod, torch.sqrt(scheduler.alphas_cumprod)
    )
    assert torch.allclose(
        scheduler.sqrt_one_minus_alphas_cumprod,
        torch.sqrt(1 - scheduler.alphas_cumprod),
    )


def test_edge_cases(scheduler):
    """Test edge case handling"""
    # Empty batch
    x_empty = torch.empty(1, 3, 16, 16).to(DEVICE)
    noise_empty = torch.empty_like(x_empty).to(DEVICE)
    t_empty = torch.empty(0, dtype=torch.long).to(DEVICE)

    with pytest.raises(RuntimeError):
        scheduler.add_noise(x_empty, noise_empty, t_empty)

    # Single element batch
    x_single = torch.randn(1, 3, 16, 16).to(DEVICE)
    noise_single = torch.randn_like(x_single).to(DEVICE)
    t_single = torch.tensor([5]).to(DEVICE)

    x_noisy = scheduler.add_noise(x_single, noise_single, t_single)
    assert x_noisy.shape == x_single.shape
