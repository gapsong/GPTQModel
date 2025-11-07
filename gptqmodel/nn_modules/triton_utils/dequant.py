# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import itertools
from typing import List

import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd

from ...utils.torch import HAS_XPU


def make_dequant_configs(block_sizes: List[int], num_warps: List[int], num_stages: List[int]):
    configs = []
    for bs, ws, ns in itertools.product(block_sizes, num_warps, num_stages):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws, num_stages=ns))
    return configs

# tested on A100 with [Llama 3.2 1B and Falcon 7B] bits:4, group_size:128
DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([1024], [1], [1])
#DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8], [2]) <- slower
@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    not_packed,  # 1 if zeros are unpacked float [num_groups, out_features], 0 if packed int
    out_ptr,
    out_dtype: tl.constexpr,
    numels,
    pack_bits: tl.constexpr,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    out_features: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    """
    Triton kernel for dequantizing GPTQ/SPQR weights with outlier-aware support.
    
    Dequantization formulas:
    - If not_packed != 0 (unpacked float zeros): w_fp = w_int * scale - zero_float
    - If not_packed == 0 (packed int zeros):     w_fp = (w_int - zero_int) * scale
    
    The unpacked format is used when zeros are stored as floats (e.g., from outlier-aware
    quantization or LoRA merging). The packed format stores zeros as quantized integers.
    """
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // out_features
    col_idx = x_index % out_features

    pack_scale: tl.constexpr = pack_bits // bits
    # ceil_div für korrekte Spaltenanzahl bei nicht-teilbaren out_features
    qzero_ncols: tl.constexpr = (out_features + pack_scale - 1) // pack_scale

    g_idx = tl.load(g_idx_ptr + row_idx, mask=xmask, eviction_policy="evict_last")
    groups = tl.where(g_idx < 0, g_idx + num_groups, g_idx)

    scales = tl.cast(
        tl.load(scales_ptr + (col_idx + out_features * groups), mask=xmask, eviction_policy="evict_last"),
        tl.float32
    )

    # === Zeros laden und auf tl.float32 casten ===
    if not_packed != 0:
        # Unpacked: [num_groups, out_features]
        zeros = tl.load(qzeros_ptr + (groups * out_features + col_idx), mask=xmask, eviction_policy="evict_last")
        zeros = tl.cast(zeros, tl.float32)
    elif bits == 3:
        # 3-bit packed: flat uint32 stream
        zero_bit_pos = (groups * out_features + col_idx) * 3
        zero_word_idx = zero_bit_pos // 32
        zero_bit_offset = zero_bit_pos % 32

        zero_word = tl.load(qzeros_ptr + zero_word_idx, mask=xmask, eviction_policy="evict_last").to(tl.uint32)
        within = (zero_bit_offset <= 29)
        zeros_within = (zero_word >> zero_bit_offset) & 0x7

        next_zero_word = tl.load(qzeros_ptr + zero_word_idx + 1, mask=xmask & ~within, eviction_policy="evict_last").to(tl.uint32)
        combined = (zero_word >> zero_bit_offset) | (next_zero_word << (32 - zero_bit_offset))
        zeros = tl.where(within, zeros_within, combined & 0x7).to(tl.float32)
    else:
        # 2/4/8-bit packed: [num_groups, qzero_ncols]
        qzero_word = tl.load(qzeros_ptr + (groups * qzero_ncols + col_idx // pack_scale), mask=xmask, eviction_policy="evict_last").to(tl.uint32)
        shift = (col_idx % pack_scale) * bits
        zeros = ((qzero_word >> shift) & maxq).to(tl.float32)

    # === Weights laden ===
    if bits == 3:
        weight_bit_pos = (row_idx * out_features + col_idx) * 3
        weight_word_idx = weight_bit_pos // 32
        weight_bit_offset = weight_bit_pos % 32

        weight_word = tl.load(qweight_ptr + weight_word_idx, mask=xmask, eviction_policy="evict_last").to(tl.uint32)
        within = (weight_bit_offset <= 29)
        weights_within = (weight_word >> weight_bit_offset) & 0x7

        next_weight_word = tl.load(qweight_ptr + weight_word_idx + 1, mask=xmask & ~within, eviction_policy="evict_last").to(tl.uint32)
        combined = (weight_word >> weight_bit_offset) | (next_weight_word << (32 - weight_bit_offset))
        weights = tl.where(within, weights_within, combined & 0x7).to(tl.float32)
    else:
        qweight_word = tl.load(qweight_ptr + (col_idx + out_features * (row_idx // pack_scale)), mask=xmask, eviction_policy="evict_last").to(tl.uint32)
        shift = (row_idx % pack_scale) * bits
        weights = ((qweight_word >> shift) & maxq).to(tl.float32)

    # === Dequantize: formula depends on not_packed ===
    if not_packed != 0:
        # Unpacked: zeros are float, use w_int * scale - zero_float
        weights = weights * scales - zeros
    else:
        # Packed: zeros are int, use (w_int - zero_int) * scale
        weights = (weights - zeros) * scales
    
    tl.store(out_ptr + x_index, tl.cast(weights, out_dtype), mask=xmask)



def torch_dtype_to_triton(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.int32:
        return tl.int32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def dequant(dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq):
    assert bits in [2, 3, 4, 8], "Only 2, 3, 4, 8 bits are supported"

    num_groups = scales.shape[0]
    out_features = scales.shape[1]
    in_features = g_idx.shape[0]

    # === not_packed aus Shape ableiten ===
    pack_scale = pack_bits // bits
    qzero_ncols = (out_features + pack_scale - 1) // pack_scale  # ceil_div

    if qzeros.dim() == 2 and qzeros.shape == (num_groups, out_features):
        not_packed = 1  # unpacked
    elif bits == 3 and qzeros.dim() == 1:
        not_packed = 0  # 3-bit packed (flat stream)
    elif qzeros.dim() == 2 and qzeros.shape[1] == qzero_ncols:
        not_packed = 0  # 2/4/8-bit packed
    else:
        raise ValueError(
            f"qzeros shape {qzeros.shape} incompatible: "
            f"expected unpacked ({num_groups}, {out_features}) or packed ({num_groups}, {qzero_ncols}) for bits={bits}"
        )

    # === Dtype-Check (optional, für Robustheit) ===
    qzeros = qzeros.contiguous()
    if not_packed == 0:
        # Packed sollte int32/uint32 sein
        if qzeros.dtype not in (torch.int32, getattr(torch, "uint32", torch.int32)):
            qzeros = qzeros.to(torch.int32)
    # Unpacked kann float16/int8/int16 sein (Kernel castet auf float32)

    out = torch.empty((in_features, out_features), device=qweight.device, dtype=dtype)
    numels = out.numel()
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)

    dequant_kernel[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        int(not_packed),  # shape-basiert, nicht dtype
        out,
        torch_dtype_to_triton(dtype),
        numels,
        pack_bits=pack_bits,
        maxq=maxq,
        bits=bits,
        out_features=out_features,
        num_groups=num_groups,
    )
    return out


def quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=False):
    W = dequant(input.dtype, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
    if transpose:
        return input @ W.t()
    return input @ W

class QuantLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type="xpu" if HAS_XPU else "cuda")
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq):
        output = quant_matmul(input, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq, ctx.pack_bits = bits, maxq, pack_bits
        return output

    @staticmethod
    @custom_bwd(device_type="xpu" if HAS_XPU else "cuda")
    def backward(ctx, grad_output):
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq, pack_bits = ctx.bits, ctx.maxq, ctx.pack_bits
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_matmul(grad_output, qweight, scales, qzeros, g_idx, bits, pack_bits, maxq, transpose=True)
        return grad_input, None, None, None, None, None, None
