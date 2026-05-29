# -*- coding: utf-8 -*-
"""W4A8 量化 Linear 算子(AWQ 风格权重布局,纯 PyTorch 实现)。

## AWQ 4-bit 权重布局

与 GPTQ 沿 K 方向 pack 不同,AWQ 为了让 dequant kernel 一次性取出 low/high
两组 nibble,采用沿 **N 方向** pack 并带有 **interleave 重排**的约定:

- ``qweight``: int32, shape ``[K, N // 8]``。
  沿 **N 方向** 每 8 个 4-bit 权重打包进一个 int32;**nibble 的 pack 顺序为
  interleaved**:``[0, 2, 4, 6, 1, 3, 5, 7]``。即:

  * 第 0 个 nibble(bits [0, 4)) 放 N 维第 0 列
  * 第 1 个 nibble(bits [4, 8)) 放 N 维第 2 列
  * 第 2 个 nibble(bits [8, 12))放 N 维第 4 列
  * 第 3 个 nibble(bits [12, 16))放 N 维第 6 列
  * 第 4 个 nibble(bits [16, 20))放 N 维第 1 列
  * 第 5 个 nibble(bits [20, 24))放 N 维第 3 列
  * 第 6 个 nibble(bits [24, 28))放 N 维第 5 列
  * 第 7 个 nibble(bits [28, 32))放 N 维第 7 列

- ``qzeros``:  int32, shape ``[K // group_size, N // 8]``。
  沿 **N 方向** 每 8 个 4-bit 零点打包进一个 int32,interleave 顺序与 ``qweight``
  相同。
- ``scales``:  float16, shape ``[K // group_size, N]``,与 ``qweight`` 的 N 维
  **不做 interleave**(scales 本身就是按列排布的)。

反量化语义:``w_fp[k, j] = (w_q[k, j] - w_z[k // g, j]) * scales[k // g, j]``。

## 计算流程(per-token int8 激活 + per-group int8 权重 + int8 GEMM)

1. 对输入 ``x : [M, K] bf16`` 做 per-token 动态对称 int8 量化,得到
   ``x_q : [M, K] int8`` 与 ``x_scale : [M] float32``。
2. 将 ``qweight`` / ``qzeros`` 解包并反 interleave,得到
   ``w_q : [K, N] int8`` 与 ``w_z : [K//g, N] int8``。
3. 按 K 维每个 group 独立做 int8 GEMM
   ``acc_g = x_q[:, k0:k1] @ (w_q[k0:k1] - w_z[g]) (int32)``,随后以
   ``x_scale * w_scale[g]`` 反量化并累加至 fp32。
4. 转回 bfloat16。

本实现不依赖 vLLM / C++ 扩展,可作为 AWQ 等价性对比的参考路径。
"""
from __future__ import annotations

import torch

__all__ = [
    "AWQ_REVERSE_ORDER",
    "unpack_awq_qweight",
    "unpack_awq_qzeros",
    "w4a8_linear",
]


# AWQ 原始 interleave 顺序:第 i 个 nibble 对应 N 维的 AWQ_ORDER[i] 列
# 权威来源:
#   https://github.com/casper-hansen/AutoAWQ/blob/main/awq/utils/packing_utils.py
#   https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py
AWQ_ORDER: tuple[int, ...] = (0, 2, 4, 6, 1, 3, 5, 7)

# 逆序:给定输出 N 维第 j 列,应当取第 AWQ_REVERSE_ORDER[j] 个 nibble
# 由 AWQ_ORDER 反推:[0, 4, 1, 5, 2, 6, 3, 7]
AWQ_REVERSE_ORDER: tuple[int, ...] = tuple(
    AWQ_ORDER.index(j) for j in range(8)
)

_BIT_SHIFTS = torch.arange(0, 32, 4, dtype=torch.int32)


# ── 形状校验 ──

def _check_shapes(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[int, int, int, int]:
    """校验输入张量形状并返回 ``(M, K, N, group_size)``。"""
    if x.dtype != torch.bfloat16:
        raise RuntimeError(f"x 必须为 bfloat16,当前 dtype={x.dtype}")
    if qweight.dtype != torch.int32:
        raise RuntimeError(f"qweight 必须为 int32,当前 dtype={qweight.dtype}")
    if qzeros.dtype != torch.int32:
        raise RuntimeError(f"qzeros 必须为 int32,当前 dtype={qzeros.dtype}")
    if scales.dtype != torch.float16:
        raise RuntimeError(f"scales 必须为 float16,当前 dtype={scales.dtype}")

    if qweight.dim() != 2 or qzeros.dim() != 2 or scales.dim() != 2:
        raise RuntimeError("qweight / qzeros / scales 必须均为 2D")

    k, n_over_8 = qweight.shape
    n = n_over_8 * 8

    groups_z, n_over_8_z = qzeros.shape
    groups_s, n_s = scales.shape

    if n_over_8_z != n_over_8:
        raise RuntimeError(
            f"qzeros N 维应与 qweight 一致: {n_over_8_z} vs {n_over_8}"
        )
    if groups_z != groups_s:
        raise RuntimeError(
            f"qzeros 和 scales 的 group 维不一致: {groups_z} vs {groups_s}"
        )
    if n_s != n:
        raise RuntimeError(f"scales 的 N 维应为 {n},当前为 {n_s}")
    if k % groups_z != 0:
        raise RuntimeError(f"K={k} 不能被 group 数 {groups_z} 整除")

    group_size = k // groups_z
    if x.shape[-1] != k:
        raise RuntimeError(f"x 最后一维应为 K={k},当前为 {x.shape[-1]}")

    m = x.numel() // k
    return m, k, n, group_size


# ── AWQ 解包 ──

def _unpack_int4_along_n_awq(packed: torch.Tensor) -> torch.Tensor:
    """沿 N 方向解包 AWQ int32 → int8(含 interleave 反重排)。

    输入 ``packed : [..., N // 8] int32``,输出 ``[..., N] int8``,值域 ``[0, 15]``。
    """
    if packed.dtype != torch.int32:
        raise RuntimeError(f"packed 必须为 int32,当前 dtype={packed.dtype}")

    lead = packed.shape[:-1]
    n_over_8 = packed.shape[-1]

    shifts = _BIT_SHIFTS.to(packed.device)                        # [8]
    # 逐 nibble 提取 → [..., N//8, 8](第 8 维按 pack 内顺序,即 AWQ_ORDER)
    unpacked = (packed.unsqueeze(-1) >> shifts.view(*([1] * len(lead)), 1, 8)) & 0xF

    # 反 interleave:按 AWQ_REVERSE_ORDER 取 nibble,使得输出沿 N 维是自然顺序
    reverse_idx = torch.tensor(
        AWQ_REVERSE_ORDER, dtype=torch.long, device=packed.device,
    )
    unpacked = unpacked.index_select(-1, reverse_idx)             # [..., N//8, 8]
    return unpacked.reshape(*lead, n_over_8 * 8).to(torch.int8)


def unpack_awq_qweight(qweight: torch.Tensor) -> torch.Tensor:
    """解包 AWQ 的 ``qweight [K, N // 8] int32`` 为 ``[K, N] int8``。"""
    if qweight.dim() != 2:
        raise RuntimeError(f"qweight 必须为 2D,当前 {qweight.dim()}D")
    return _unpack_int4_along_n_awq(qweight)


def unpack_awq_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
    """解包 AWQ 的 ``qzeros [G, N // 8] int32`` 为 ``[G, N] int8``。"""
    if qzeros.dim() != 2:
        raise RuntimeError(f"qzeros 必须为 2D,当前 {qzeros.dim()}D")
    return _unpack_int4_along_n_awq(qzeros)


# ── 激活量化 ──

def _quantize_activation_per_token(x: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
    """对 ``x [M, K]`` 做 per-token 动态对称 int8 量化。

    :returns: ``(x_q [M, K] int8, x_scale [M] float32)``
    """
    x_fp32 = x.to(torch.float32)
    max_abs = x_fp32.abs().amax(dim=-1)                           # [M]
    # max_abs 为 0 的行 scale 置 1.0 以避免除零(此时 x_q 为全 0)
    scale = torch.where(
        max_abs > 0,
        max_abs / 127.0,
        torch.ones_like(max_abs),
    )
    x_q = torch.round(x_fp32 / scale.unsqueeze(-1)).clamp_(-128, 127).to(torch.int8)
    return x_q, scale


# ── 主算子 ──

def w4a8_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """W4A8 量化 Linear(AWQ 布局):per-token int8 GEMM 实现。

    在数值上等价于:

    .. code-block:: python

        w_q = unpack_awq_qweight(qweight).float()                 # [K, N]
        w_z = unpack_awq_qzeros(qzeros).float()                   # [G, N]
        w_z_full = w_z.repeat_interleave(group_size, dim=0)       # [K, N]
        s_full = scales.float().repeat_interleave(group_size, 0)  # [K, N]
        w_fp = (w_q - w_z_full) * s_full
        out = (x.float() @ w_fp).to(bfloat16)

    但内部以 **per-token int8 激活 + per-group int8 权重 + int8 GEMM** 路径计算,
    贴近生产侧 W4A8 算子的数值特征。

    :param x:       bfloat16 张量,形状 ``[M, K]``。
    :param qweight: int32 张量,形状 ``[K, N // 8]``,AWQ 沿 N 方向 interleaved pack。
    :param qzeros:  int32 张量,形状 ``[K // group_size, N // 8]``,AWQ interleaved pack。
    :param scales:  float16 张量,形状 ``[K // group_size, N]``。
    :param bias:    可选 bias,形状 ``[N]``,输出 dtype 为 bfloat16。
    :returns:       bfloat16 张量,形状 ``[*, N]``。
    """
    _m, k, n, group_size = _check_shapes(x, qweight, qzeros, scales)

    # 空输入短路
    if x.numel() == 0:
        return torch.empty(
            (*x.shape[:-1], n), dtype=torch.bfloat16, device=x.device,
        )

    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, k)                          # [M, K] bf16

    # 1) 激活 per-token int8 量化
    x_q, x_scale = _quantize_activation_per_token(x_2d)           # int8, fp32

    # 2) AWQ 权重 / 零点解包(一次性到 int8 自然排布的 [K, N] / [G, N])
    w_q = unpack_awq_qweight(qweight.contiguous())                # [K, N] int8
    w_z = unpack_awq_qzeros(qzeros.contiguous())                  # [G, N] int8
    w_scale = scales.contiguous().to(torch.float32)               # [G, N] fp32

    # 3) per-group int8 GEMM,fp32 累加
    groups_k = k // group_size
    out_fp32 = torch.zeros(
        (x_q.shape[0], n), dtype=torch.float32, device=x.device,
    )
    for g in range(groups_k):
        k0 = g * group_size
        k1 = k0 + group_size

        # 本 group 的中心化权重:w_q - w_z ∈ [-15, 15],可容纳于 int8
        w_block_i32 = (
            w_q[k0:k1].to(torch.int32)
            - w_z[g].to(torch.int32).unsqueeze(0)
        )                                                         # [gs, N]

        # torch.matmul 不支持 int8;以 int32 提升保证位宽不溢出(int8*int8+ 累加)
        # 替换为int8 gemm
        acc = torch.matmul(
            x_q[:, k0:k1].to(torch.int32),
            w_block_i32,
        )                                                         # [M, N] int32

        out_fp32.add_(acc.to(torch.float32) * w_scale[g].unsqueeze(0))

    # 激活 scale 在最外层统一乘一次,避免 per-group 重复广播
    out_fp32.mul_(x_scale.unsqueeze(-1))

    if bias is not None:
        if bias.shape != (n,):
            raise RuntimeError(
                f"bias 形状应为 ({n},),当前为 {tuple(bias.shape)}"
            )
        out_fp32.add_(bias.to(torch.float32))

    return out_fp32.to(torch.bfloat16).reshape(*orig_shape[:-1], n)
