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

template<int Mr>
void pack_A(int Kc, const float *A, int lda, float *A_pack) {
    for (int kc = 0; kc < Kc; kc++) {
        for (int ir = 0; ir < Mr; ir++) {
            A_pack[ir] = A[ir * lda + kc];
        }
        A_pack += Mr;
    }
}

template<int Nr>
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


template<int Mr, int Nr>
void cache_kji (int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);
        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);
            pack_B<Nr>(k_len, j_len, B, ldb, B_pack, k, j);
            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                for (int ic = 0; ic < i_len; ic += Mr) {
                    pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                if (k == 0) {
                    for (int ic = i; ic < i + i_len; ic++)
                        for (int jc = j; jc < j + j_len; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        register_block(
                            Mr, Nr, 0, k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }
    free(A_pack); free(B_pack);
}

template<int Mr, int Nr>
void cache_kij (int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);
            for (int ic = 0; ic < i_len; ic += Mr) {
                pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
            }
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                pack_B<Nr>(k_len, j_len, B, ldb, B_pack, k, j);
                if (k == 0) {
                    for (int ic = i; ic < i + i_len; ic++)
                        for (int jc = j; jc < j + j_len; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        register_block(
                            Mr, Nr, 0, k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }
    free(A_pack); free(B_pack);
}

template<int Mr, int Nr>
void cache_ijk (int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int i = 0; i < M; i += Mc) {
        int i_len = std::min(Mc, M - i);
        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);
            for (int ic = i; ic < i + i_len; ic++)
                for (int jc = j; jc < j + j_len; jc++)
                    MAT_C(ic, jc) = 0.0f;
            for (int k = 0; k < K; k += Kc) {
                int k_len = std::min(Kc, K - k);
                for (int ic = 0; ic < i_len; ic += Mr) {
                    pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                pack_B<Nr>(k_len, j_len, B, ldb, B_pack, k, j);
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        register_block(
                            Mr, Nr, 0, k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }
    free(A_pack); free(B_pack);
}

template<int Mr, int Nr>
void cache_ikj (int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int i = 0; i < M; i += Mc) {
        int i_len = std::min(Mc, M - i);
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            for (int ic = 0; ic < i_len; ic += Mr) {
                pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
            }
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                pack_B<Nr>(k_len, j_len, B, ldb, B_pack, k, j);
                if (k == 0) {
                    for (int ic = i; ic < i + i_len; ic++)
                        for (int jc = j; jc < j + j_len; jc++)
                            MAT_C(ic, jc) = 0.0f;
                }
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        register_block(
                            Mr, Nr, 0, k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }
    free(A_pack); free(B_pack);
}

template<int Mr, int Nr>
void cache_jik (int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);
            for (int ic = i; ic < i + i_len; ic++)
                for (int jc = j; jc < j + j_len; jc++)
                    MAT_C(ic, jc) = 0.0f;
            for (int k = 0; k < K; k += Kc) {
                int k_len = std::min(Kc, K - k);
                for (int ic = 0; ic < i_len; ic += Mr) {
                    pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                pack_B<Nr>(k_len, j_len, B, ldb, B_pack, k, j);
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        register_block(
                            Mr, Nr, 0, k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }
    free(A_pack); free(B_pack);
}

template<int Mr, int Nr>
void cache_jki (int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);
        for (int i = 0; i < M; i++)
            memset(&MAT_C(i, j), 0, j_len * sizeof(float));
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            pack_B<Nr>(k_len, j_len, B, ldb, B_pack, k, j);
            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                for (int ic = 0; ic < i_len; ic += Mr) {
                    pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);
                }
                for (int ir = 0; ir < i_len; ir += Mr) {
                    for (int jr = 0; jr < j_len; jr += Nr) {
                        register_block(
                            Mr, Nr, 0, k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / Nr) * k_len * Nr],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }
    free(A_pack); free(B_pack);
}

template<int Mr, int Nr>
void cache_op(int op, int M, int N, int K,
                                  int Mc, int Nc, int Kc,
                                  float * __restrict__ A, int lda,
                                  float * __restrict__ B, int ldb,
                                  float * __restrict__ C, int ldc) {
    switch (op) {
        case 0: cache_kji <Mr, Nr>(M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); break;
        case 1: cache_kij <Mr, Nr>(M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); break;
        case 2: cache_ijk <Mr, Nr>(M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); break;
        case 3: cache_ikj <Mr, Nr>(M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); break;
        case 4: cache_jik <Mr, Nr>(M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); break;
        case 5: cache_jki <Mr, Nr>(M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); break;
        default: std::printf("错误: 不支持的循环顺序 op=%d\n", op); break;
    }
}



void cache(int op, int M, int N, int K,
           int Mc, int Nc, int Kc, int Mr, int Nr,
           float * __restrict__ A, int lda,
           float * __restrict__ B, int ldb,
           float * __restrict__ C, int ldc) {
    if (Mr == 4 && Nr == 16) { cache_op<4, 16>(op, M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); return; }
    if (Mr == 5 && Nr == 16) { cache_op<5, 16>(op, M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); return; }
    if (Mr == 4 && Nr == 20) { cache_op<4, 20>(op, M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); return; }
    if (Mr == 6 && Nr == 16) { cache_op<6, 16>(op, M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); return; }
    if (Mr == 4 && Nr == 24) { cache_op<4, 24>(op, M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); return; }
    if (Mr == 3 && Nr == 16) { cache_op<3, 16>(op, M, N, K, Mc, Nc, Kc, A, lda, B, ldb, C, ldc); return; }
    std::printf("错误: 当前 cache 不支持 Mr=%d, Nr=%d\n", Mr, Nr);
}
