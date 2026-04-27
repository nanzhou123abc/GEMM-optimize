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
const int Mc = 128;
const int Nc = 256;
const int Kc = 128;
//与ipj不同的是：切割了MNK三个维度。
void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        memset(&MAT_C(i, 0), 0, N * sizeof(float));
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                MAT_C(i, j) += MAT_A(i, k) * MAT_B(k, j);
            }
        }
    }
}

void naive_loopreorder(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++) {
	memset(C+i*ldc, 0, N*sizeof(float));	
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++) {
                MAT_C(i,j) += MAT_A(i, k) * MAT_B(k, j);
        }
    }
}

void check(int M, int N, float *c_ref, int ldc_ref, float *c_opt, int ldc_opt) {
    float max_error = 0.0f;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float err = fabsf(c_ref[i * ldc_ref + j] - c_opt[i * ldc_opt + j]);
            if (err > max_error) max_error = err;
        }
    printf("  最大误差: %.6f\n", max_error);
}
void cache_block(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    //cache block
    // memset(C, 0, M * ldc * sizeof(float));
    for(int i = 0; i < M; i += Mc) {
        int i_end = std::min(i + Mc, M);
        for(int k = 0; k < K; k += Kc) {
            int k_end = std::min(k + Kc, K);
            for(int j = 0; j < N; j += Nc) {
                int j_end = std::min(j + Nc, N);
            
                if (k == 0) {
                    for (int ic = i; ic < i_end; ic++)
                        for (int jc = j; jc < j_end; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }
                //micro kernel
                for(int ic = i; ic < i_end; ic++) {
                    for(int kc = k; kc < k_end; kc++) {
                        float a_val = MAT_A(ic,kc);
                        for(int jc = j; jc < j_end; jc++) {
                            MAT_C(ic,jc) += a_val * MAT_B(kc,jc);
                        }
                    }
                }
                
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("用法: %s m n k\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    
    int lda = K, ldb = N, ldc = N;

    float *A      = (float *)aligned_alloc(64, M * K  * sizeof(float));
    float *B      = (float *)aligned_alloc(64, K * N  * sizeof(float));
    float *C_naive  = (float *)aligned_alloc(64, M * N  * sizeof(float));
    float *C_loopreorder    = (float *)aligned_alloc(64, M * N  * sizeof(float));
    float *C_opt    = (float *)aligned_alloc(64, M * N  * sizeof(float));

    std::srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;

    // 性能基准测试
    GemmTimer::bench("naive",     	M, N, K, 20, [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("loopreorder",     M, N, K, 20, [&](){ naive_loopreorder(M, N, K, A, lda, B, ldb, C_loopreorder, ldc); });
    GemmTimer::bench("cache",     	M, N, K, 20, [&](){ cache_block(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    
    // 正确性验证
    check(M, N, C_naive, ldc, C_loopreorder, ldc);
    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt); free(C_loopreorder);
    return 0;
}
