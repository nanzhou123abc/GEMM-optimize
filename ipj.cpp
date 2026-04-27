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
//A: M*K   B: K*N    C: M*N 行主序
//拿a的一个点刷b的一行，返回C
void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
   
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += MAT_A(i,k) * MAT_B(k,j);
            MAT_C(i,j) = sum;
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
void ipj_gemm(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        memset(&MAT_C(i, 0), 0, N * sizeof(float));
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                MAT_C(i, j) += MAT_A(i, k) * MAT_B(k, j);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("用法: %s M N K\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    
    int lda = K, ldb = N, ldc = N;

    float *A      = (float *)aligned_alloc(64, M * K  * sizeof(float));
    float *B      = (float *)aligned_alloc(64, K * N  * sizeof(float));
    float *C_naive  = (float *)aligned_alloc(64, M * N  * sizeof(float));
    float *C_opt    = (float *)aligned_alloc(64, M * N  * sizeof(float));

    std::srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;

    // 性能基准测试

    GemmTimer::bench("naive",     M, N, K, 20, [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("ipj",     M, N, K, 20, [&](){ ipj_gemm(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    

    check(M, N, C_naive, ldc, C_opt, ldc);
    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
