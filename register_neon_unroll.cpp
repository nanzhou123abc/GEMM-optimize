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
constexpr int Nr = 8;

//在微内核中p循环展开
//修改大块循环，pji -》pij 先pack A再 B   因为B > A 
//把 pack 放在能最大化复用的循环层级，大块的少 pack，小块的多 pack。
//修改pack a   将i留在主函数 按 MR panel 排

void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += MAT_A(i, k) * MAT_B(k, j);
            MAT_C(i, j) = sum;
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

// 微内核 4×8: NEON + k方向×4展开
//   未展开: 每次循环处理 1 个 k，load A/B → FMA  下一次循环
//   展开×4: 每次循环处理 4 个 k，一次性 load 4组 A/B   4组 FMA
//   减少循环开销 ：循环次数变为 1/4
//    寄存器利用  32 个 NEON 寄存器
//      8 个给 C 累加器，4组×(2个B + 4个A) = 24 个给 A/B

static inline void micro_kernel(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc)
{
    float32x4_t cv00 = vld1q_f32(&MAT_C(0, 0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0, 4));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1, 0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1, 4));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2, 0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2, 4));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3, 0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3, 4));

    const float *ap = A_pack;
    const float *bp = B_pack;

    // 主循环: 每次处理 4 个 k
    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        // ---- k+0 ----
        float32x4_t bv0_lo = vld1q_f32(bp + 0 * Nr + 0);
        float32x4_t bv0_hi = vld1q_f32(bp + 0 * Nr + 4);
        float32x4_t av0_0 = vld1q_dup_f32(ap + 0 * Mr + 0);
        float32x4_t av0_1 = vld1q_dup_f32(ap + 0 * Mr + 1);
        float32x4_t av0_2 = vld1q_dup_f32(ap + 0 * Mr + 2);
        float32x4_t av0_3 = vld1q_dup_f32(ap + 0 * Mr + 3);

        // ---- k+1 ----
        float32x4_t bv1_lo = vld1q_f32(bp + 1 * Nr + 0);
        float32x4_t bv1_hi = vld1q_f32(bp + 1 * Nr + 4);
        float32x4_t av1_0 = vld1q_dup_f32(ap + 1 * Mr + 0);
        float32x4_t av1_1 = vld1q_dup_f32(ap + 1 * Mr + 1);
        float32x4_t av1_2 = vld1q_dup_f32(ap + 1 * Mr + 2);
        float32x4_t av1_3 = vld1q_dup_f32(ap + 1 * Mr + 3);

        // ---- k+2 ----
        float32x4_t bv2_lo = vld1q_f32(bp + 2 * Nr + 0);
        float32x4_t bv2_hi = vld1q_f32(bp + 2 * Nr + 4);
        float32x4_t av2_0 = vld1q_dup_f32(ap + 2 * Mr + 0);
        float32x4_t av2_1 = vld1q_dup_f32(ap + 2 * Mr + 1);
        float32x4_t av2_2 = vld1q_dup_f32(ap + 2 * Mr + 2);
        float32x4_t av2_3 = vld1q_dup_f32(ap + 2 * Mr + 3);

        // ---- k+3 ----
        float32x4_t bv3_lo = vld1q_f32(bp + 3 * Nr + 0);
        float32x4_t bv3_hi = vld1q_f32(bp + 3 * Nr + 4);
        float32x4_t av3_0 = vld1q_dup_f32(ap + 3 * Mr + 0);
        float32x4_t av3_1 = vld1q_dup_f32(ap + 3 * Mr + 1);
        float32x4_t av3_2 = vld1q_dup_f32(ap + 3 * Mr + 2);
        float32x4_t av3_3 = vld1q_dup_f32(ap + 3 * Mr + 3);

        ap += 4 * Mr;
        bp += 4 * Nr;

        // FMA k+0
        cv00 = vfmaq_f32(cv00, bv0_lo, av0_0);  cv01 = vfmaq_f32(cv01, bv0_hi, av0_0);
        cv10 = vfmaq_f32(cv10, bv0_lo, av0_1);  cv11 = vfmaq_f32(cv11, bv0_hi, av0_1);
        cv20 = vfmaq_f32(cv20, bv0_lo, av0_2);  cv21 = vfmaq_f32(cv21, bv0_hi, av0_2);
        cv30 = vfmaq_f32(cv30, bv0_lo, av0_3);  cv31 = vfmaq_f32(cv31, bv0_hi, av0_3);

        // FMA k+1
        cv00 = vfmaq_f32(cv00, bv1_lo, av1_0);  cv01 = vfmaq_f32(cv01, bv1_hi, av1_0);
        cv10 = vfmaq_f32(cv10, bv1_lo, av1_1);  cv11 = vfmaq_f32(cv11, bv1_hi, av1_1);
        cv20 = vfmaq_f32(cv20, bv1_lo, av1_2);  cv21 = vfmaq_f32(cv21, bv1_hi, av1_2);
        cv30 = vfmaq_f32(cv30, bv1_lo, av1_3);  cv31 = vfmaq_f32(cv31, bv1_hi, av1_3);

        // FMA k+2
        cv00 = vfmaq_f32(cv00, bv2_lo, av2_0);  cv01 = vfmaq_f32(cv01, bv2_hi, av2_0);
        cv10 = vfmaq_f32(cv10, bv2_lo, av2_1);  cv11 = vfmaq_f32(cv11, bv2_hi, av2_1);
        cv20 = vfmaq_f32(cv20, bv2_lo, av2_2);  cv21 = vfmaq_f32(cv21, bv2_hi, av2_2);
        cv30 = vfmaq_f32(cv30, bv2_lo, av2_3);  cv31 = vfmaq_f32(cv31, bv2_hi, av2_3);

        // FMA k+3
        cv00 = vfmaq_f32(cv00, bv3_lo, av3_0);  cv01 = vfmaq_f32(cv01, bv3_hi, av3_0);
        cv10 = vfmaq_f32(cv10, bv3_lo, av3_1);  cv11 = vfmaq_f32(cv11, bv3_hi, av3_1);
        cv20 = vfmaq_f32(cv20, bv3_lo, av3_2);  cv21 = vfmaq_f32(cv21, bv3_hi, av3_2);
        cv30 = vfmaq_f32(cv30, bv3_lo, av3_3);  cv31 = vfmaq_f32(cv31, bv3_hi, av3_3);
    }

    // 尾部处理: 剩余不足 4 个的 k
    for (; kr < Kc; kr++) {
        float32x4_t bv_lo = vld1q_f32(bp + 0);
        float32x4_t bv_hi = vld1q_f32(bp + 4);
        float32x4_t a0 = vld1q_dup_f32(ap + 0);
        float32x4_t a1 = vld1q_dup_f32(ap + 1);
        float32x4_t a2 = vld1q_dup_f32(ap + 2);
        float32x4_t a3 = vld1q_dup_f32(ap + 3);

        cv00 = vfmaq_f32(cv00, bv_lo, a0);  cv01 = vfmaq_f32(cv01, bv_hi, a0);
        cv10 = vfmaq_f32(cv10, bv_lo, a1);  cv11 = vfmaq_f32(cv11, bv_hi, a1);
        cv20 = vfmaq_f32(cv20, bv_lo, a2);  cv21 = vfmaq_f32(cv21, bv_hi, a2);
        cv30 = vfmaq_f32(cv30, bv_lo, a3);  cv31 = vfmaq_f32(cv31, bv_hi, a3);

        ap += Mr;
        bp += Nr;
    }

    vst1q_f32(&MAT_C(0, 0), cv00);  vst1q_f32(&MAT_C(0, 4), cv01);
    vst1q_f32(&MAT_C(1, 0), cv10);  vst1q_f32(&MAT_C(1, 4), cv11);
    vst1q_f32(&MAT_C(2, 0), cv20);  vst1q_f32(&MAT_C(2, 4), cv21);
    vst1q_f32(&MAT_C(3, 0), cv30);  vst1q_f32(&MAT_C(3, 4), cv31);
}

void register_neon_unroll_gemm(int M, int N, int K,
                               float * __restrict__ A, int lda,
                               float * __restrict__ B, int ldb,
                               float * __restrict__ C, int ldc)
{
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

                //内核
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

    GemmTimer::bench("naive",                M, N, K, 20,  [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("register_neon_unroll", M, N, K, 100, [&](){ register_neon_unroll_gemm(M, N, K, A, lda, B, ldb, C_opt, ldc); });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
