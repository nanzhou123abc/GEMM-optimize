#include "blas_gemm_bf16_int4.hpp"

//   读取 op(A) 的一个元素 -> FP32
static inline float get_A(const bfloat16_t *A, int lda,
                          enum CBLAS_TRANSPOSE trans, int i, int k)
{
    return (trans == CblasNoTrans)
        ? bf16_to_fp32(A[i * lda + k])
        : bf16_to_fp32(A[k * lda + i]);
}

//  反量化 B 的一个元素 -> FP32
static inline float get_B(const int8_t *B_packed, int ldb,
                          enum CBLAS_TRANSPOSE trans,
                          const bfloat16_t *scale,
                          const int8_t *zero_point,
                          int group_size,
                          int row, int col, int K_dim)
{
    // 物理坐标
    int pr = (trans == CblasNoTrans) ? row : col;
    int pc = (trans == CblasNoTrans) ? col : row;

    // 解包 INT4
    int byte_idx = pr * ((ldb + 1) / 2) + pc / 2;
    int8_t packed = B_packed[byte_idx];
    int8_t val = (pc % 2 == 0) ? unpack_int4_lo(packed) : unpack_int4_hi(packed);

    // scale / zero_point 索引
    int scale_idx;
    if (group_size == 0) {
        scale_idx = col;
    } else {
        int gid = row / group_size;
        int ng  = (K_dim + group_size - 1) / group_size;
        scale_idx = col * ng + gid;
    }

    float s  = bf16_to_fp32(scale[scale_idx]);
    float zp = 0.0f;
     if (zero_point) {
        int zb = scale_idx / 2;
        zp = (scale_idx % 2 == 0)
            ? (float)unpack_int4_lo(zero_point[zb])
            : (float)unpack_int4_hi(zero_point[zb]);
    }

    return s * ((float)val - zp);
}

//naive实现
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
    const int                  ldc)
{
    if (M <= 0 || N <= 0 || K <= 0)           return -1;
    if (!A || !B_packed || !scale || !C)       return -2;

    // ColMajor: C = alpha * op(A) * deq(B) + beta * C
    // 等价于行主序下: C^T = alpha * deq(B)^T * op(A)^T + beta * C^T
    // 即交换 A<->B, M<->N, 翻转转置标志
    if (order == CblasColMajor) {
        //  ColMajor 下 C[i,j] = C_ptr[j * ldc + i]
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    // ColMajor: A 按列存储
                    // NoTrans: op(A)[i,k] = A[k*lda + i]
                    // Trans:   op(A)[i,k] = A[i*lda + k]
                    float a = (transA == CblasNoTrans)
                        ? bf16_to_fp32(A[k * lda + i])
                        : bf16_to_fp32(A[i * lda + k]);
                    float b = get_B(B_packed, ldb, transB,
                                    scale, zero_point, group_size,
                                    k, j, K);
                    acc += a * b;
                }
                float c_old = bf16_to_fp32(C[j * ldc + i]);
                C[j * ldc + i] = fp32_to_bf16(alpha * acc + beta * c_old);
            }
        }
        return 0;
    }

    // RowMajor
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = get_A(A, lda, transA, i, k);
                float b = get_B(B_packed, ldb, transB,
                                scale, zero_point, group_size,
                                k, j, K);
                acc += a * b;
            }
            float c_old = bf16_to_fp32(C[i * ldc + j]);
            C[i * ldc + j] = fp32_to_bf16(alpha * acc + beta * c_old);
        }
    }
    return 0;
}
