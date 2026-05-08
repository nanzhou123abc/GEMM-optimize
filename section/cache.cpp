#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#include "cache.hpp"
#include "register.hpp"
#define MAT_A(i,j) A[ (i)*lda + (j) ]
#define MAT_B(i,j) B[ (i)*ldb + (j) ]
#define MAT_C(i,j) C[ (i)*ldc + (j) ]

static inline void pack_A(int Kc, const float *A, int lda, float *A_pack) {
    for (int kc = 0; kc < Kc; kc++) {
        A_pack[0] = A[0 * lda + kc];
        A_pack[1] = A[1 * lda + kc];
        A_pack[2] = A[2 * lda + kc];
        A_pack[3] = A[3 * lda + kc];
        A_pack += 4;
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

void cache_kji(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);
        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);
            pack_B(k_len, j_len, B, ldb, B_pack, k, j);
            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                for (int ic = 0; ic < i_len; ic += 4) {
                    pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                }
                for (int ir = 0; ir < i_len; ir += 4) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_block_4x4(
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
    free(A_pack); free(B_pack);
}

// k -> i -> j
void cache_kij(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);
            for (int ic = 0; ic < i_len; ic += 4) {
                pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
            }
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);
                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                }
                for (int ir = 0; ir < i_len; ir += 4) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_block_4x4(
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
    free(A_pack); free(B_pack);
}

// i -> j -> k
void cache_ijk(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int i = 0; i < M; i += Mc) {
        int i_len = std::min(Mc, M - i);
        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);

            for (int ic = i; ic < i+i_len; ic++) 
                for (int jc = j; jc < j+j_len; jc++) 
                    MAT_C(ic,jc) = 0.0f;

            for (int k = 0; k < K; k += Kc) {
                int k_len = std::min(Kc, K - k);
                for (int ic = 0; ic < i_len; ic += 4) {
                    pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);
                for (int ir = 0; ir < i_len; ir += 4) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_block_4x4(
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
    free(A_pack); free(B_pack);
}

// i -> k -> j
void cache_ikj(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int i = 0; i < M; i += Mc) {
        int i_len = std::min(Mc, M - i);
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            for (int ic = 0; ic < i_len; ic += 4) {
                pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
            }
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);

                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                }

                for (int ir = 0; ir < i_len; ir += 4) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_block_4x4(
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
    free(A_pack); free(B_pack);
}

// j -> i -> k
void cache_jik(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);

            for (int ic = i; ic < i+i_len; ic++) 
                for (int jc = j; jc < j+j_len; jc++) 
                    MAT_C(ic,jc) = 0.0f;

            for (int k = 0; k < K; k += Kc) {
                int k_len = std::min(Kc, K - k);
                for (int ic = 0; ic < i_len; ic += 4) {
                    pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);

                for (int ir = 0; ir < i_len; ir += 4) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_block_4x4(
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
    free(A_pack); free(B_pack);
}

// j -> k -> i
void cache_jki(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
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


            for (int ir = 0; ir < i_len; ir += 4) {
                for (int jr = 0; jr < j_len; jr += 16) {
                    register_block_4x4(
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
