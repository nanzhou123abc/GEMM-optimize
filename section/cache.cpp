#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#include "cache.hpp"
#include "register.hpp"
#include "Timer.hpp"
#define MAT_A(i,j) A[ (i)*lda + (j) ]
#define MAT_B(i,j) B[ (i)*ldb + (j) ]
#define MAT_C(i,j) C[ (i)*ldc + (j) ]
//bug ： 当N < Nr时会出错。此时不能pack B
void  naive_compute(int i_len, int j_len, int k_len,
                const float *A_blk, int lda,
                const float *B_blk, int ldb,
                float *C_blk, int ldc) {
    for (int i = 0; i < i_len; i++) {
        for (int k = 0; k < k_len; k++) {
            for (int j = 0; j < j_len; j++) {
                C_blk[i * ldc + j] += A_blk[i * lda + k] * B_blk[k * ldb + j];
            }
        }
    }
}

template<int Mr, int Nr>
void tail_block(int i0, int j0, int k0,
                int i_len, int j_len, int k_len,
                int i_full, int j_full,
                float * __restrict__ A, int lda,
                float * __restrict__ B, int ldb,
                float * __restrict__ C, int ldc) {
    if (i_full < i_len) {
         naive_compute(
            i_len - i_full, j_len, k_len,
            &MAT_A(i0 + i_full, k0), lda,
            &MAT_B(k0, j0), ldb,
            &MAT_C(i0 + i_full, j0), ldc
        );
    }
    if (j_full < j_len && i_full > 0) {
         naive_compute(
            i_full, j_len - j_full, k_len,
            &MAT_A(i0, k0), lda,
            &MAT_B(k0, j0 + j_full), ldb,
            &MAT_C(i0, j0 + j_full), ldc
        );
    }
}

template<int Mr>
void pack_A(int k_len, const float *A, int lda, float *A_pack) {
    for (int k_cur = 0; k_cur < k_len; k_cur++) {
        for (int ir = 0; ir < Mr; ir++) {
            A_pack[ir] = MAT_A(ir, k_cur);
        }
        A_pack += Mr;
    }
}

template<int Nr>
void pack_B(int k_len, int j_len, float *B, int ldb, float *B_pack) {
    for (int j_cur = 0; j_cur < j_len; j_cur += Nr) {
        for (int k_cur = 0; k_cur < k_len; k_cur++) {
            for (int jr = 0; jr < Nr; jr++) {
                B_pack[jr] = MAT_B(k_cur, j_cur + jr);
            }
            B_pack += Nr;
        }
    }
}

//| | * -> 
//V V * ->
template<int Mr, int Nr>
void cache_kji (int M, int N, int K,
                int Mc, int Nc, int Kc,
                float * __restrict__ A, int lda,
                float * __restrict__ B, int ldb,
                float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, M * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    memset(C, 0, M*N * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);

        int M_full = (M / Mr) * Mr;
        
            for (int ic = 0; ic < M_full; ic += Mr) {
                GemmTimer::bench_pack([&](){
                    pack_A<Mr>(k_len, &MAT_A(ic, k), lda, &A_pack[ic * k_len]);
                });
            }
        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);
            int j_full = (j_len / Nr) * Nr;
            GemmTimer::bench_pack([&](){pack_B<Nr>(k_len, j_full, &MAT_B(k, j), ldb, B_pack);});
            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                int i_full = (i_len / Mr) * Mr;
                
                GemmTimer::bench_kernel([&](){
                    for (int ir = 0; ir < i_full; ir += Mr) {
                        for (int jr = 0; jr < j_full; jr += Nr) {
                            register_block(
                                Mr, Nr, 0, k_len,
                                &A_pack[(i + ir) * k_len],
                                &B_pack[(jr / Nr) * k_len * Nr],
                                &MAT_C(i + ir, j + jr), ldc
                            );
                        }
                    }
                });
                tail_block<Mr, Nr>(
                    i, j, k, i_len, j_len, k_len, i_full, j_full,
                    A, lda, B, ldb, C, ldc
                );
            }
        }
    }
    free(A_pack); free(B_pack);
}
//| | * -> 
//V V * ->
template<int Mr, int Nr>
void cache_kij (int M, int N, int K,
                int Mc, int Nc, int Kc,
                float * __restrict__ A, int lda,
                float * __restrict__ B, int ldb,
                float * __restrict__ C, int ldc) {
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    memset(C, 0, M*N * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);
            int i_full = (i_len / Mr) * Mr;
            for (int ic = 0; ic < i_full; ic += Mr) {
                GemmTimer::bench_pack([&](){pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);});
            }
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                int j_full = (j_len / Nr) * Nr;
                
                GemmTimer::bench_pack([&](){pack_B<Nr>(k_len, j_full, &MAT_B(k, j), ldb, B_pack);});
                
                for (int ir = 0; ir < i_full; ir += Mr) {
                    for (int jr = 0; jr < j_full; jr += Nr) {
                        GemmTimer::bench_kernel([&](){
                            register_block(
                                Mr, Nr, 0, k_len,
                                &A_pack[ir * k_len],
                                &B_pack[(jr / Nr) * k_len * Nr],
                                &MAT_C(i + ir, j + jr), ldc
                            );
                        });
                    }
                }
                tail_block<Mr, Nr>(
                    i, j, k, i_len, j_len, k_len, i_full, j_full,
                    A, lda, B, ldb, C, ldc
                );
            }
        }
    }
    free(A_pack); free(B_pack);
}

// -> * | |
// -> * V V
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
        for (int ic = 0; ic < i_len; ic++) {
            memset(&MAT_C(i + ic, 0), 0, N * sizeof(float));
        }
        int i_full = (i_len / Mr) * Mr;
        for (int j = 0; j < N; j += Nc) {
            int j_len = std::min(Nc, N - j);
            int j_full = (j_len / Nr) * Nr;
            
            for (int k = 0; k < K; k += Kc) {
                int k_len = std::min(Kc, K - k);
                for (int ic = 0; ic < i_full; ic += Mr) {
                    GemmTimer::bench_pack([&](){pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);});
                }
                GemmTimer::bench_pack([&](){pack_B<Nr>(k_len, j_full, &MAT_B(k, j), ldb, B_pack);});
                
                for (int ir = 0; ir < i_full; ir += Mr) {
                    for (int jr = 0; jr < j_full; jr += Nr) {
                        GemmTimer::bench_kernel([&](){
                            register_block(
                                Mr, Nr, 0, k_len,
                                &A_pack[ir * k_len],
                                &B_pack[(jr / Nr) * k_len * Nr],
                                &MAT_C(i + ir, j + jr), ldc
                            );
                        });
                    }
                }
                tail_block<Mr, Nr>(
                    i, j, k, i_len, j_len, k_len, i_full, j_full,
                    A, lda, B, ldb, C, ldc
                );
            }
        }
    }
    free(A_pack); free(B_pack);
}


//-> * ->
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
        for (int ic = 0; ic < i_len; ic++) {
            memset(&MAT_C(i + ic, 0), 0, N * sizeof(float));
        }
        int i_full = (i_len / Mr) * Mr;
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            
            for (int ic = 0; ic < i_full; ic += Mr) {
                GemmTimer::bench_pack([&](){pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);});
            }
            
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                int j_full = (j_len / Nr) * Nr;
                
                GemmTimer::bench_pack([&](){pack_B<Nr>(k_len, j_full, &MAT_B(k, j), ldb, B_pack);});
                
                
                for (int ir = 0; ir < i_full; ir += Mr) {
                    for (int jr = 0; jr < j_full; jr += Nr) {
                        GemmTimer::bench_kernel([&](){
                            register_block(
                                Mr, Nr, 0, k_len,
                                &A_pack[ir * k_len],
                                &B_pack[(jr / Nr) * k_len * Nr],
                                &MAT_C(i + ir, j + jr), ldc
                            );
                        });
                        
                    }
                }
                tail_block<Mr, Nr>(
                    i, j, k, i_len, j_len, k_len, i_full, j_full,
                    A, lda, B, ldb, C, ldc
                );
            }
        }
    }
    free(A_pack); free(B_pack);
}
// -> * | |
// -> * V V
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
        int j_full = (j_len / Nr) * Nr;
        for(int i = 0; i < M; i++) {
            memset(&MAT_C(i,j), 0, j_len*sizeof(float));
        }
        
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);
            int i_full = (i_len / Mr) * Mr;
            
            for (int k = 0; k < K; k += Kc) {
                int k_len = std::min(Kc, K - k);
                for (int ic = 0; ic < i_full; ic += Mr) {
                    GemmTimer::bench_pack([&](){pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);});
                }
                GemmTimer::bench_pack([&](){pack_B<Nr>(k_len, j_full, &MAT_B(k, j), ldb, B_pack);});
                for (int ir = 0; ir < i_full; ir += Mr) {
                    for (int jr = 0; jr < j_full; jr += Nr) {
                        GemmTimer::bench_kernel([&](){
                            register_block(
                                Mr, Nr, 0, k_len,
                                &A_pack[ir * k_len],
                                &B_pack[(jr / Nr) * k_len * Nr],
                                &MAT_C(i + ir, j + jr), ldc
                            );
                        });
                    }
                }
                tail_block<Mr, Nr>(
                    i, j, k, i_len, j_len, k_len, i_full, j_full,
                    A, lda, B, ldb, C, ldc
                );
            }
        }
    }
    free(A_pack); free(B_pack);
}
// | | * | |
// V V * V V
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
        int j_full = (j_len / Nr) * Nr;
        
        for (int i = 0; i < M; i++)
            memset(&MAT_C(i, j), 0, j_len * sizeof(float));
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            
            GemmTimer::bench_pack([&](){pack_B<Nr>(k_len, j_full, &MAT_B(k, j), ldb, B_pack);});
            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                int i_full = (i_len / Mr) * Mr;
                for (int ic = 0; ic < i_full; ic += Mr) {
                    GemmTimer::bench_pack([&](){pack_A<Mr>(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len]);});
                }
                for (int ir = 0; ir < i_full; ir += Mr) {
                    for (int jr = 0; jr < j_full; jr += Nr) {
                        GemmTimer::bench_kernel([&](){
                            register_block(
                                Mr, Nr, 0, k_len,
                                &A_pack[ir * k_len],
                                &B_pack[(jr / Nr) * k_len * Nr],
                                &MAT_C(i + ir, j + jr), ldc
                            );
                        });
                    }
                }
                tail_block<Mr, Nr>(
                    i, j, k, i_len, j_len, k_len, i_full, j_full,
                    A, lda, B, ldb, C, ldc
                );
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
