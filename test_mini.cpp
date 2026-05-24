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
void packA(int M, int N, int K, float* A, float* pack_A, int lda) {
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            pack_A[i * K + k] = MAT_A(k, i);
        }
    }
}

void opt(
    int K,
    const float* __restrict A, int lda,
    const float* __restrict B, int ldb,
    float* __restrict C, int ldc) {

    float32x4_t c00 = vdupq_n_f32(0.0f);
    float32x4_t c01 = vdupq_n_f32(0.0f);
    float32x4_t c02 = vdupq_n_f32(0.0f);
    float32x4_t c03 = vdupq_n_f32(0.0f);

    float32x4_t c10 = vdupq_n_f32(0.0f);
    float32x4_t c11 = vdupq_n_f32(0.0f);
    float32x4_t c12 = vdupq_n_f32(0.0f);
    float32x4_t c13 = vdupq_n_f32(0.0f);

    float32x4_t c20 = vdupq_n_f32(0.0f);
    float32x4_t c21 = vdupq_n_f32(0.0f);
    float32x4_t c22 = vdupq_n_f32(0.0f);
    float32x4_t c23 = vdupq_n_f32(0.0f);

    float32x4_t c30 = vdupq_n_f32(0.0f);
    float32x4_t c31 = vdupq_n_f32(0.0f);
    float32x4_t c32 = vdupq_n_f32(0.0f);
    float32x4_t c33 = vdupq_n_f32(0.0f);

    const float* a0p = A + 0 * 4;
    const float* a1p = A + 1 * 4;
    const float* a2p = A + 2 * 4;
    const float* a3p = A + 3 * 4;
    const float* bp  = B;

    for (int k = 0; k < K; ++k) {
        float32x4_t b0 = vld1q_f32(bp + 0);
        float32x4_t b1 = vld1q_f32(bp + 4);
        float32x4_t b2 = vld1q_f32(bp + 8);
        float32x4_t b3 = vld1q_f32(bp + 12);

        float a0 = *a0p++;
        float a1 = *a1p++;
        float a2 = *a2p++;
        float a3 = *a3p++;

        c00 = vfmaq_n_f32(c00, b0, a0);
        c01 = vfmaq_n_f32(c01, b1, a0);
        c02 = vfmaq_n_f32(c02, b2, a0);
        c03 = vfmaq_n_f32(c03, b3, a0);

        c10 = vfmaq_n_f32(c10, b0, a1);
        c11 = vfmaq_n_f32(c11, b1, a1);
        c12 = vfmaq_n_f32(c12, b2, a1);
        c13 = vfmaq_n_f32(c13, b3, a1);

        c20 = vfmaq_n_f32(c20, b0, a2);
        c21 = vfmaq_n_f32(c21, b1, a2);
        c22 = vfmaq_n_f32(c22, b2, a2);
        c23 = vfmaq_n_f32(c23, b3, a2);

        c30 = vfmaq_n_f32(c30, b0, a3);
        c31 = vfmaq_n_f32(c31, b1, a3);
        c32 = vfmaq_n_f32(c32, b2, a3);
        c33 = vfmaq_n_f32(c33, b3, a3);

        bp += 16;
    }

    vst1q_f32(&MAT_C(0,  0), c00);  vst1q_f32(&MAT_C(0,  4), c01);  vst1q_f32(&MAT_C(0,  8), c02);  vst1q_f32(&MAT_C(0, 12), c03);
    vst1q_f32(&MAT_C(1,  0), c10);  vst1q_f32(&MAT_C(1,  4), c11);  vst1q_f32(&MAT_C(1,  8), c12);  vst1q_f32(&MAT_C(1, 12), c13);
    vst1q_f32(&MAT_C(2,  0), c20);  vst1q_f32(&MAT_C(2,  4), c21);  vst1q_f32(&MAT_C(2,  8), c22);  vst1q_f32(&MAT_C(2, 12), c23);
    vst1q_f32(&MAT_C(3,  0), c30);  vst1q_f32(&MAT_C(3,  4), c31);  vst1q_f32(&MAT_C(3,  8), c32);  vst1q_f32(&MAT_C(3, 12), c33);
}

int main(int argc, char *argv[]) {
    // if (argc != 4) {
    //     printf("用法: %s M N K\n", argv[0]);
    //     return 1;
    // }

    int M = 4;
    int N = 16;
    int K = 4;

    if (M % 4 != 0 || N % 8 != 0) {
        printf("错误: M 必须是 %d 的倍数, N 必须是 %d 的倍数\n", 4, 8);
        return 1;
    }

    int lda = K, ldb = N, ldc = N;
    float *A       = (float *)aligned_alloc(64, M * K * sizeof(float));
    float *B       = (float *)aligned_alloc(64, K * N * sizeof(float));
    float *C_naive = (float *)aligned_alloc(64, M * N * sizeof(float));
    float *C_opt   = (float *)aligned_alloc(64, M * N * sizeof(float));

    std::srand(time(NULL));

    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;
    memset(C_opt, 0, M * N * sizeof(float));
    GemmTimer::bench("naive", M, N, K, 200000, [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });

    GemmTimer::bench("opt", M, N, K, 1000000, [&](){ opt(K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);
    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
