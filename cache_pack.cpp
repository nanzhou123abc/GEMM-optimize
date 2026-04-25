#include<iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <arm_neon.h> 
#include "Timer.hpp"

#define MAT_A(i,j) A[ (i)*lda + (j) ]
#define MAT_B(i,j) B[ (i)*ldb + (j) ]
#define MAT_C(i,j) C[ (i)*ldc + (j) ]
const int Mc = 64;
const int Nc = 128;
const int Kc = 64;

//为了内存连续统一进行pack，使块的内存连续。
//并调整大块的循环 从ipj改为pji，先pack b再pack a

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
void pack_A(int Mc, int Kc, float *A, int lda, float *A_pack, int i0, int k0) {
    for(int ic = 0; ic < Mc; ic++) {
        for(int kc = 0; kc < Kc; kc++) {
            A_pack[ic * Kc + kc] = MAT_A(i0 + ic, k0 + kc);
        }
    }
}
           
void pack_B(int Kc, int Nc, float *B, int ldb, float *B_pack, int k0, int j0) {
    for(int kc = 0; kc < Kc; kc++) {
        for(int jc = 0; jc < Nc; jc++) {
            B_pack[kc * Nc + jc] = MAT_B(k0 + kc, j0 + jc);
        }
    }
}

void cache_block_pack(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    float *B_pack = (float*)aligned_alloc(64, Kc * Nc * sizeof(float));
    float *A_pack = (float*)aligned_alloc(64, Mc * Kc * sizeof(float));


    for(int k = 0; k < K; k += Kc) {
        int k_end = std::min(k + Kc, K);
        int k_len = std::min(Kc, K - k);
        for(int j = 0; j < N; j += Nc) {
            int j_end = std::min(j + Nc, N);
            int j_len = std::min(Nc, N - j);
            pack_B(k_len, j_len, B, ldb, B_pack, k, j);
            
            for(int i = 0; i < M; i += Mc) {
                int i_end = std::min(i + Mc, M);
                int i_len = std::min(Mc, M - i);
                pack_A(i_len, k_len, A, lda, A_pack, i, k);

                if (k == 0) {
                    for (int ic = i; ic < i_end; ic++)
                        for (int jc = j; jc < j_end; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }

                //内核
                for(int ic = i; ic < i_end; ic++) {
                    for(int kc = k; kc < k_end; kc++) {
                        float a_val = A_pack[(ic-i)*k_len + (kc-k)];
                        for(int jc = j; jc < j_end; jc++) {
                            MAT_C(ic,jc) += a_val * B_pack[(kc - k) * j_len + (jc - j)];
                        }
                    }
                }
                
            }
        }
    }
    free(A_pack); free(B_pack);
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

    srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;

    // 性能基准测试
    GemmTimer::bench("naive",     M, N, K, 20,  [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("cache_pack",     M, N, K, 100,  [&](){ cache_block_pack(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    
    // 正确性验证
    check(M, N, C_naive, ldc, C_opt, ldc);
    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
