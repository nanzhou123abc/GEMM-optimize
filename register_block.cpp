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
constexpr int Mr = 4;   // 微内核行数
constexpr int Nr = 4;   // 微内核列数

//引入微内核： 4*4
//pack a做了转置。a的一列*b的一行

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

// pack_A: 按 Mr 行一组，转置存储（列主序）
// 原始 A 是行主序: A(i, p) = a[i * lda + p]
//   对于 Mr=4, p_len=3 的例子:
//   原始 A (行主序):          pack 后 (列主序):
//   row0: a00 a01 a02         [a00 a10 a20 a30]  k=0 的 4 行
//   row1: a10 a11 a12         [a01 a11 a21 a31]  k=1 的 4 行
//   row2: a20 a21 a22         [a02 a12 a22 a32]  k=2 的 4 行
//   row3: a30 a31 a32
// 后续升级到 NEON 时，一条 vld1q_f32 就能加载 4 个值
void pack_A(int Mc, int Kc, float *A, int lda, float *A_pack, int i0, int k0) {
    for (int ic = 0; ic < Mc; ic += Mr) {
        for (int kc = 0; kc < Kc; kc++) {
            for (int ir = 0; ir < Mr; ir++) {
                A_pack[ir] = MAT_A(i0 + ic + ir, k0 + kc);
            }
            A_pack += Mr;
        }
    }
}

// B_pack 布局: BK 行 × BN 列，B_pack[p * BN + j]
void pack_B(int Kc, int Nc, float *B, int ldb, float *B_pack, int k0, int j0) {
    for (int kc = 0; kc < Kc; kc++) {
        for (int jc = 0; jc < Nc; jc++) {
            B_pack[kc * Nc + jc] = MAT_B(k0 + kc, j0 + jc);
        }
    }
}

static inline void micro_kernel(
    int Kc,
    const float *A_pack,                // 列主序，步长 MR
    const float *B_pack, int ldb,   // 行主序，步长 j_len
    float *C, int ldc)
{
    // Mr×Nr = 4×4 = 16 个寄存器变量
    float c00 = 0, c01 = 0, c02 = 0, c03 = 0;
    float c10 = 0, c11 = 0, c12 = 0, c13 = 0;
    float c20 = 0, c21 = 0, c22 = 0, c23 = 0;
    float c30 = 0, c31 = 0, c32 = 0, c33 = 0;

    for (int kr = 0; kr < Kc; kr++) {
       
        float a0 = A_pack[kr * Mr + 0];
        float a1 = A_pack[kr * Mr + 1];
        float a2 = A_pack[kr * Mr + 2];
        float a3 = A_pack[kr * Mr + 3];

     
        float b0 = B_pack[kr * ldb + 0];
        float b1 = B_pack[kr * ldb + 1];
        float b2 = B_pack[kr * ldb + 2];
        float b3 = B_pack[kr * ldb + 3];

     
        c00 += a0 * b0;  c01 += a0 * b1;  c02 += a0 * b2;  c03 += a0 * b3;
        c10 += a1 * b0;  c11 += a1 * b1;  c12 += a1 * b2;  c13 += a1 * b3;
        c20 += a2 * b0;  c21 += a2 * b1;  c22 += a2 * b2;  c23 += a2 * b3;
        c30 += a3 * b0;  c31 += a3 * b1;  c32 += a3 * b2;  c33 += a3 * b3;
    }

    // 累加写回 C（+=，因为外层 p 循环分了多个 BK 块）
    MAT_C(0,0) += c00;  MAT_C(0,1) += c01;  MAT_C(0,2) += c02;  MAT_C(0,3) += c03;
    MAT_C(1,0) += c10;  MAT_C(1,1) += c11;  MAT_C(1,2) += c12;  MAT_C(1,3) += c13;
    MAT_C(2,0) += c20;  MAT_C(2,1) += c21;  MAT_C(2,2) += c22;  MAT_C(2,3) += c23;
    MAT_C(3,0) += c30;  MAT_C(3,1) += c31;  MAT_C(3,2) += c32;  MAT_C(3,3) += c33;
}

void register_gemm(int M, int N, int K,
                   float * __restrict__ A, int lda,
                   float * __restrict__ B, int ldb,
                   float * __restrict__ C, int ldc)
{
    float * __restrict__ A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float * __restrict__ B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));

    memset(C, 0, M * ldc * sizeof(float));
    // B_pack 在 j 循环里 pack 一次，对所有 i 块复用
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);

        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);

            // Pack B: 当前 (k, j) 块，对所有 i 块共享
            pack_B(k_len, j_len, B, ldb, B_pack, k, j);

            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);

                // Pack A: 当前 (i, k) 块
                pack_A(i_len, k_len, A, lda, A_pack, i, k);

                if (k == 0) {
                    for (int ic = i; ic < i + i_len; ic++)
                        for (int jc = j; jc < j + j_len; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }

                // 微内核调度: 把 Mc×Nc 块切成 Mr×Nr 的小块
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        micro_kernel(
                            k_len,
                            &A_pack[ir * k_len],           // A_pack 列主序，每 MR 组
                            &B_pack[jr], j_len,    // B_pack 行主序
                            C[(i+ir) * ldc + (j+jr)], ldc
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

    GemmTimer::bench("naive",    M, N, K, 20,  [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("register", M, N, K, 100, [&](){ register_gemm(M, N, K, A, lda, B, ldb, C_opt, ldc); });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
