import math
import torch
import torch.nn.functional as F

from astrojepa.model import new_gelu

def test_new_gelu_zeros():
    """Test that new_gelu returns zeros for zero input."""
    x = torch.zeros(3)
    out = new_gelu(x)
    assert torch.allclose(out, torch.zeros(3))

def test_new_gelu_values():
    """Test new_gelu values match standard PyTorch GELU with tanh approximation."""
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    out = new_gelu(x)

    expected = F.gelu(x, approximate='tanh')

    # We use a small tolerance since this is an exact specific approximation
    # matching OpenAI GPT/Google BERT (which F.gelu approximate='tanh' also targets)
    assert torch.allclose(out, expected, atol=1e-5)

def test_new_gelu_large_positive():
    """Test that new_gelu returns approximately x for very large positive x."""
    x = torch.tensor([10.0, 20.0, 100.0])
    out = new_gelu(x)
    assert torch.allclose(out, x, atol=1e-4)

def test_new_gelu_large_negative():
    """Test that new_gelu returns approximately 0 for very large negative x."""
    x = torch.tensor([-10.0, -20.0, -100.0])
    out = new_gelu(x)
    assert torch.allclose(out, torch.zeros_like(x), atol=1e-4)

def test_new_gelu_multidimensional():
    """Test that new_gelu handles multidimensional tensors correctly."""
    x = torch.randn(2, 3, 4)
    out = new_gelu(x)
    expected = F.gelu(x, approximate='tanh')

    assert out.shape == x.shape
    assert torch.allclose(out, expected, atol=1e-5)

def test_new_gelu_preserves_dtype():
    """Test that new_gelu preserves the tensor data type."""
    # Test float32
    x_float32 = torch.tensor([1.0, -1.0], dtype=torch.float32)
    out_float32 = new_gelu(x_float32)
    assert out_float32.dtype == torch.float32

    # Test float64
    x_float64 = torch.tensor([1.0, -1.0], dtype=torch.float64)
    out_float64 = new_gelu(x_float64)
    assert out_float64.dtype == torch.float64

def test_new_gelu_requires_grad():
    """Test that new_gelu properly propagates gradients."""
    x = torch.tensor([1.0, -1.0], requires_grad=True)
    out = new_gelu(x)

    # Check that out requires grad
    assert out.requires_grad

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Check that gradients were populated
    assert x.grad is not None
    assert x.grad.shape == x.shape
