#pragma once

#include <cstdint>
#include <cstddef>

typedef uint16_t bfloat16_t;

//  CBLAS 枚举
enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans  = 111, CblasTrans    = 112 };

//  BF16 <=> FP32 转换
static inline float bf16_to_fp32(bfloat16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline bfloat16_t fp32_to_bf16(float f) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    return static_cast<bfloat16_t>(bits >> 16);
}

//  INT4 解包工具 (两个 INT4 打包在一个 INT8 中)
//  低 4 bit = 偶数索引元素, 高 4 bit = 奇数索引元素
static inline int8_t unpack_int4_lo(int8_t packed) {
    return static_cast<int8_t>((packed << 4)) >> 4;
}

static inline int8_t unpack_int4_hi(int8_t packed) {
    return packed >> 4;
}

/**
 * @brief BF16 × INT4 混合精度矩阵乘法 (GEMM)
 *
 * 计算: C = alpha * op(A) * dequant(B) + beta * C
 * 其中 dequant(B) = scale * (B_int4 - zero_point)
 *
 * @param order      行/列主序: CblasRowMajor 或 CblasColMajor
 * @param transA     A 的转置模式: CblasNoTrans / CblasTrans
 * @param transB     B 的转置模式: CblasNoTrans / CblasTrans
 * @param M          矩阵 C 的行数 (op(A) 的行数)
 * @param N          矩阵 C 的列数 (dequant(B) 的列数)
 * @param K          op(A) 的列数 / dequant(B) 的行数
 * @param alpha      标量乘子 alpha (float)
 * @param A          输入矩阵 A 指针, BF16 类型
 * @param lda        A 的 leading dimension
 * @param B_packed   输入矩阵 B 指针, INT4 打包为 INT8 类型
 *                   物理存储大小: 若 transB=NoTrans 则为 K × ceil(N/2) 字节
 *                                若 transB=Trans   则为 N × ceil(K/2) 字节
 * @param ldb        B 的 leading dimension (以 INT4 元素计, 非字节数)
 * @param scale      反量化缩放因子指针, BF16 类型
 * @param zero_point 反量化零点指针, INT8 类型 (存储 INT4 零点值), 可为 NULL (默认零点为 0)
 * @param group_size 量化分组大小, 0 表示 per-channel 量化
 * @param beta       标量乘子 beta (float)
 * @param C          输出矩阵 C 指针, BF16 类型
 * @param ldc        C 的 leading dimension
 *
 * @return           0 表示成功, 非 0 表示错误码
 */
int cblas_bf16int4_gemm(
    const enum CBLAS_ORDER     order,
    const enum CBLAS_TRANSPOSE transA,
    const enum CBLAS_TRANSPOSE transB,
    const int                  M,
    const int                  N,
    const int                  K,
    const float                alpha,
    const bfloat16_t          *A,
    const int                  lda,
    const int8_t              *B_packed,
    const int                  ldb,
    const bfloat16_t          *scale,
    const int8_t              *zero_point,
    const int                  group_size,
    const float                beta,
    bfloat16_t                *C,
    const int                  ldc
);
