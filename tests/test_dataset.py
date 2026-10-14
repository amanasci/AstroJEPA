import torch
import pytest
from astrojepa.dataset import normalise

def test_normalise_2d_basic():
    """Test normalisation on a standard 2D tensor, typical for patch outputs."""
    # Shape: (num_patches, patch_dim)
    x = torch.tensor([
        [1.0, 2.0, 3.0],   # mean=2, std=1
        [4.0, 4.0, 4.0],   # mean=4, std=0
        [10.0, 20.0, 30.0] # mean=20, std=10
    ])

    out = normalise(x)

    # Check shape
    assert out.shape == x.shape

    # Check mean and std for first row
    assert torch.allclose(out[0], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5)

    # Check mean and std for second row (zero variance)
    assert torch.allclose(out[1], torch.tensor([0.0, 0.0, 0.0]), atol=1e-5)

    # Check mean and std for third row
    assert torch.allclose(out[2], torch.tensor([-1.0, 0.0, 1.0]), atol=1e-5)

def test_normalise_zero_variance():
    """Ensure normalisation does not divide by zero or result in NaN for identical elements."""
    x = torch.ones((10, 256)) * 5.0
    out = normalise(x)

    # Since all elements are the same, the output should be strictly zeroes
    assert not torch.isnan(out).any()
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

def test_normalise_shape_preservation():
    """Ensure normalisation handles typical output dimensions without altering the shape."""
    x = torch.randn(1024, 768)
    out = normalise(x)

    assert out.shape == x.shape

def test_normalise_properties():
    """Ensure output of normalise has zero mean and unit variance along dim=1."""
    x = torch.randn(512, 128) * 10 + 5  # Mean 5, std 10
    out = normalise(x)

    std, mean = torch.std_mean(out, dim=1)

    # Check mean is approx 0
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)

    # Check std is approx 1
    # We use roughly 1e-4 tolerance since floating point std_mean on 128 elements can have minor precision differences
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)
