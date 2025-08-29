import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

# Test if basic SDPA works
query = torch.rand(1, 8, 32, 32, dtype=torch.float16, device="cuda")
key = torch.rand(1, 8, 32, 32, dtype=torch.float16, device="cuda")
value = torch.rand(1, 8, 32, 32, dtype=torch.float16, device="cuda")

# Try math backend first
try:
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(query, key, value)
    print("Math backend: OK")
except Exception as e:
    print(f"Math backend failed: {e}")

# Then try Flash Attention
try:
    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION], set_priority=True
    ):
        output = F.scaled_dot_product_attention(query, key, value)
    print("Flash Attention: OK")
except Exception as e:
    print(f"Flash Attention failed: {e}")
