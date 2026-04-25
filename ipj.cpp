#include<cstdio> //printf
#include <cstdlib> //aligned_alloc()、free()、atoi()、srand()、rand()、RAND_MAX
#include <ctime> //time(NULL)
#include <cstring> //memset
#include <math.h> //fabsf()
#include <algorithm> //min
#include <arm_neon.h> 
#include "Timer.hpp"

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]

//A: M*K   B: K*N    C: M*N 行主序
//拿a的一个点刷b的一行，返回C
void naive(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
   
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++)
                sum += A(i,p) * B(p,j);
            C(i,j) = sum;
        }
}
void check(int m, int n, float *c_ref, int ldc_ref, float *c_opt, int ldc_opt) {
    float max_error = 0.0f;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float err = fabsf(c_ref[i * ldc_ref + j] - c_opt[i * ldc_opt + j]);
            if (err > max_error) max_error = err;
        }
    printf("  最大误差: %.6f\n", max_error);
}
void ipj_gemm(int m, int n, int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
   memset(c, 0, m * ldc * sizeof(float));
    for (int i = 0; i < m; i++) {
        
        for (int p = 0; p < k; p++) {
            float a_val = A(i,p);
            for (int j = 0; j < n; j++) {
                C(i,j) += a_val * B(p,j);
               
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("用法: %s m n k\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    
    int lda = k, ldb = n, ldc = n;

    float *a      = (float *)aligned_alloc(64, m * k  * sizeof(float));
    float *b      = (float *)aligned_alloc(64, k * n  * sizeof(float));
    float *c_naive  = (float *)aligned_alloc(64, m * n  * sizeof(float));
    float *c_opt    = (float *)aligned_alloc(64, m * n  * sizeof(float));

    std :: srand(time(NULL));
    for (int i = 0; i < m * k; i++) a[i] = (float)std :: rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) b[i] = (float)std :: rand() / RAND_MAX;

    // 性能基准测试

    GemmTimer::bench("naive",     m, n, k, 20,  [&](){ naive(m, n, k, a, lda, b, ldb, c_naive, ldc); });
    GemmTimer::bench("ipj",     m, n, k, 20,  [&](){ ipj_gemm(m, n, k, a, lda, b, ldb, c_opt, ldc); });
    

    check(m, n, c_naive, ldc, c_opt, ldc);
    free(a); free(b); free(c_naive); free(c_opt);
    return 0;
}
