# EVOLVE-BLOCK-START
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def elementwise_add_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    output = a + b
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        output = torch.empty_like(a)
        assert a.is_cuda and b.is_cuda and output.is_cuda
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        elementwise_add_kernel[grid](
            a, b, output,
            n_elements,
            BLOCK_SIZE=1024,
        )
        return output
# EVOLVE-BLOCK-END

# Helper functions (not evolved by OpenEvolve)
def get_inputs():
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]

def get_init_inputs():
    return []