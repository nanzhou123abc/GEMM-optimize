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

constexpr int Mc = 66;  // 必须是 Mr=6 的倍数
constexpr int Nc = 128;
constexpr int Kc = 64;
constexpr int Mr = 6;
constexpr int Nr = 16;

// 6×16 微内核: C:24  + 4个B向量 + 6个A广播 = 34 寄存器
// 复用a的寄存器，减少+*指令（bp += 16;写在外面）
// k方向×4展开

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

// 外层循环在调用处按 Mr=6 步进
static inline void pack_A(int Kc, const float *A, int lda, float *A_pack) {
    for (int kc = 0; kc < Kc; kc++) {
        A_pack[0] = A[0 * lda + kc];
        A_pack[1] = A[1 * lda + kc];
        A_pack[2] = A[2 * lda + kc];
        A_pack[3] = A[3 * lda + kc];
        A_pack[4] = A[4 * lda + kc];
        A_pack[5] = A[5 * lda + kc];
        A_pack += Mr;
    }
}

void pack_B(int Kc, int Nc, float *B, int ldb, float *B_pack, int k0, int j0) {
    for (int jc = 0; jc < Nc; jc += 16) {
        for (int kc = 0; kc < Kc; kc++) {
            for (int jr = 0; jr < 16; jr++) {
                B_pack[jr] = MAT_B(k0 + kc, j0 + jc + jr);
            }
            B_pack += 16;
        }
    }
}

static inline void register_6x16(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc)
{
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));
    float32x4_t cv40 = vld1q_f32(&MAT_C(4,  0));  float32x4_t cv41 = vld1q_f32(&MAT_C(4,  4));  float32x4_t cv42 = vld1q_f32(&MAT_C(4,  8));  float32x4_t cv43 = vld1q_f32(&MAT_C(4, 12));
    float32x4_t cv50 = vld1q_f32(&MAT_C(5,  0));  float32x4_t cv51 = vld1q_f32(&MAT_C(5,  4));  float32x4_t cv52 = vld1q_f32(&MAT_C(5,  8));  float32x4_t cv53 = vld1q_f32(&MAT_C(5, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        // ---- k+0 ----
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);
        cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);
        cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;

        // ---- k+1 ----
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);
        cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);
        cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;

        // ---- k+2 ----
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);
        cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);
        cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;

        // ---- k+3 ----
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);
        cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);
        cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;
    }

    for (; kr < Kc; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);
        cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);
        cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
    vst1q_f32(&MAT_C(4,  0), cv40);  vst1q_f32(&MAT_C(4,  4), cv41);  vst1q_f32(&MAT_C(4,  8), cv42);  vst1q_f32(&MAT_C(4, 12), cv43);
    vst1q_f32(&MAT_C(5,  0), cv50);  vst1q_f32(&MAT_C(5,  4), cv51);  vst1q_f32(&MAT_C(5,  8), cv52);  vst1q_f32(&MAT_C(5, 12), cv53);
}

void register_neon_unroll_gemm(int M, int N, int K,
                               float * __restrict__ A, int lda,
                               float * __restrict__ B, int ldb,
                               float * __restrict__ C, int ldc){
    float * __restrict__ A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float * __restrict__ B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));

    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);
        
        for (int i = 0; i < M; i++)
            memset(&MAT_C(i, j), 0, j_len * sizeof(float));

        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            pack_B(k_len, j_len, B, ldb, B_pack, k, j);

            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);

            // Pack A: 整个 Mc×Kc 块，按 4 一组
                for (int ic = 0; ic < i_len; ic += 4) {
                    pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }

                // 内核
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_6x16(
                            k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / 16) * k_len * 16],
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

    if (M % Mr != 0 || N % 16 != 0) {
        printf("错误: M 必须是 %d 的倍数, N 必须是 %d 的倍数\n", Mr, 16);
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
    GemmTimer::bench("register_neon_unroll_6x16", M, N, K, 100, [&](){ register_neon_unroll_gemm(M, N, K, A, lda, B, ldb, C_opt, ldc); });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
