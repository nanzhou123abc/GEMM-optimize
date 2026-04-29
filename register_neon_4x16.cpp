#include<cstdio> //printf
#include <cstdlib> //aligned_alloc()、free()、atoi()、srand()、rand()、RAND_MAX
#include <ctime> //time(NULL)
#include <cstring> //memset
#include <math.h> //fabsf()
#include <algorithm> //min
#include <arm_neon.h> 
#include "Timer.hpp"

#define MAT_A(i,j) A[ (i)*lda + (j) ]
#define MAT_B(i,j) B[ (i)*ldb + (j) ]
#define MAT_C(i,j) C[ (i)*ldc + (j) ]

constexpr int Mc = 64;
constexpr int Nc = 128;
constexpr int Kc = 64;
constexpr int Mr = 4;
constexpr int Nr = 16;

// 4×16 微内核: 16 个累加寄存器 + 4个B向量 + 4个A广播 = 24 寄存器
// 每个 k 步: 16 次 FMA / (4+4) 次 load = 计算访存比 2.0
// 对比 4×8: 8 次 FMA / (2+4) 次 load = 计算访存比 1.33

void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        memset(C+i*ldc, 0, N*sizeof(float));
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++) {
                MAT_C(i,j) += MAT_A(i, k) * MAT_B(k, j);
            }
    }
}

void check(int M, int N, float *C_ref, int ldc_ref, float *C_opt, int ldc_opt) {
    float max_error = 0.0f;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float err = fabsf(C_ref[i * ldc_ref + j] - C_opt[i * ldc_opt + j]);
            if (err > max_error) max_error = err;
        }
    printf("  最大误差: %.6f\n", max_error);
}

// 外层循环在调用处按 Mr 步进
static inline void pack_A(int Kc, const float *A, int lda, float *A_pack) {
    for (int kc = 0; kc < Kc; kc++) {
        A_pack[0] = A[0 * lda + kc];
        A_pack[1] = A[1 * lda + kc];
        A_pack[2] = A[2 * lda + kc];
        A_pack[3] = A[3 * lda + kc];
        A_pack += Mr;
    }
}

void pack_B(int Kc, int Nc, float *B, int ldb, float *B_pack, int k0, int j0) {
    for (int jc = 0; jc < Nc; jc += Nr) {
        for (int kc = 0; kc < Kc; kc++) {
            for (int jr = 0; jr < Nr; jr++) {
                B_pack[jr] = MAT_B(k0 + kc, j0 + jc + jr);
            }
            B_pack += Nr;
        }
    }
}

// 微内核 4×16: 4行 × 4个NEON向量(16列), k方向不展开
// 累加寄存器: cv[row][col_vec], row=0..3, col_vec=0..3
// 每个 k 步: load 4个B向量 + 4个A标量广播, 做 16 次 FMA
// 寄存器分配: 16 个 C 累加器 + 4个B + 4个A = 24 个

static inline void micro_kernel(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc)
{
    // 加载 C 的 4×16 块到 16 个累加寄存器
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));
    float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));
    float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));
    float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));
    float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    for (int kr = 0; kr < Kc; kr++) {
        float32x4_t bv0 = vld1q_f32(bp + 0);
        float32x4_t bv1 = vld1q_f32(bp + 4);
        float32x4_t bv2 = vld1q_f32(bp + 8);
        float32x4_t bv3 = vld1q_f32(bp + 12);
        float32x4_t a0 = vld1q_dup_f32(ap + 0);
        float32x4_t a1 = vld1q_dup_f32(ap + 1);
        float32x4_t a2 = vld1q_dup_f32(ap + 2);
        float32x4_t a3 = vld1q_dup_f32(ap + 3);

        cv00 = vfmaq_f32(cv00, bv0, a0);  cv01 = vfmaq_f32(cv01, bv1, a0);
        cv02 = vfmaq_f32(cv02, bv2, a0);  cv03 = vfmaq_f32(cv03, bv3, a0);
        cv10 = vfmaq_f32(cv10, bv0, a1);  cv11 = vfmaq_f32(cv11, bv1, a1);
        cv12 = vfmaq_f32(cv12, bv2, a1);  cv13 = vfmaq_f32(cv13, bv3, a1);
        cv20 = vfmaq_f32(cv20, bv0, a2);  cv21 = vfmaq_f32(cv21, bv1, a2);
        cv22 = vfmaq_f32(cv22, bv2, a2);  cv23 = vfmaq_f32(cv23, bv3, a2);
        cv30 = vfmaq_f32(cv30, bv0, a3);  cv31 = vfmaq_f32(cv31, bv1, a3);
        cv32 = vfmaq_f32(cv32, bv2, a3);  cv33 = vfmaq_f32(cv33, bv3, a3);

        ap += Mr;
        bp += Nr;
    }

    // 写回 C 的 4×16 块
    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);
    vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);
    vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);
    vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);
    vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
}

void register_neon_unroll_gemm(int M, int N, int K,
                               float * __restrict__ A, int lda,
                               float * __restrict__ B, int ldb,
                               float * __restrict__ C, int ldc){
    float * __restrict__ A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float * __restrict__ B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));

    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);

        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);

            // Pack A: 整个 Mc×Kc 块，按 Mr 一组
            for (int ic = 0; ic < i_len; ic += Mr) {
                pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
            }

            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);

                pack_B(k_len, j_len, B, ldb, B_pack, k, j);

                if (k == 0) {
                    for (int ic = i; ic < i + i_len; ic++)
                        for (int jc = j; jc < j + j_len; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }

                // 内核
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        micro_kernel(
                            k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }

    free(A_pack);
    free(B_pack);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("用法: %s M N K\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    if (M % Mr != 0 || N % Nr != 0) {
        printf("错误: M 必须是 %d 的倍数, N 必须是 %d 的倍数\n", Mr, Nr);
        return 1;
    }

    int lda = K, ldb = N, ldc = N;

    float *A        = (float *)aligned_alloc(64, M * K * sizeof(float));
    float *B        = (float *)aligned_alloc(64, K * N * sizeof(float));
    float *C_naive  = (float *)aligned_alloc(64, M * N * sizeof(float));
    float *C_opt    = (float *)aligned_alloc(64, M * N * sizeof(float));

    std::srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;

    GemmTimer::bench("naive",                    M, N, K, 20,  [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("register_neon_4x16",       M, N, K, 100, [&](){ register_neon_unroll_gemm(M, N, K, A, lda, B, ldb, C_opt, ldc); });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
