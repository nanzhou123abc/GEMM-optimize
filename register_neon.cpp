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


//微内核改为4*8
//修改pack B的写法保证一行 8个，方便neon
//pack B 的结构：B_pack[panel][kc][jr]     panel 的大小是 k_len * 8
//C使用8个寄存器。
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

void pack_A(int Mc, int Kc, float *A, int lda, float *A_pack, int i0, int k0) {
    for (int ic = 0; ic < Mc; ic += 4) {
        for (int kc = 0; kc < Kc; kc++) {
            for (int ir = 0; ir < 4; ir++) {
                A_pack[ir] = MAT_A(i0 + ic + ir, k0 + kc);
            }
            A_pack += 4;
        }
    }
}


// 两条 vld1q_f32 加载完整的 8=8 个值
void pack_B(int Kc, int Nc, float *B, int ldb, float *B_pack, int k0, int j0) {
    for (int jc = 0; jc < Nc; jc += 8) {
        for (int kc = 0; kc < Kc; kc++) {
            for (int jr = 0; jr < 8; jr++) {
                B_pack[jr] = MAT_B(k0 + kc, j0 + jc + jr);
            }
            B_pack += 8;
        }
    }
}


//   标量: c_ij += a_i * b_j
//   NEON: cv_i_lo += a_i_broadcast * bv_lo
//         cv_i_hi += a_i_broadcast * bv_hi
static inline void micro_kernel(
    int Kc,
    const float *A_pack,    // 列主序，步长 4
    const float *B_pack,    // 行主序，步长 8
    float *C, int ldc){
    //前四个                                      后四个
    float32x4_t cv00 = vld1q_f32(&MAT_C(0, 0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0, 4));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1, 0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1, 4));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2, 0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2, 4));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3, 0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3, 4));

    for (int kr = 0; kr < Kc; kr++) {
        // 加载 B 的 8=8 个值 
        float32x4_t bv_lo = vld1q_f32(B_pack + kr * 8 + 0);  // 前 4 个
        float32x4_t bv_hi = vld1q_f32(B_pack + kr * 8 + 4);  // 后 4 个

        // 加载 A 的 4=4 个值并广播
        
        float32x4_t av0 = vld1q_dup_f32(A_pack + kr * 4 + 0);
        float32x4_t av1 = vld1q_dup_f32(A_pack + kr * 4 + 1);
        float32x4_t av2 = vld1q_dup_f32(A_pack + kr * 4 + 2);
        float32x4_t av3 = vld1q_dup_f32(A_pack + kr * 4 + 3);

        // FMA 
        cv00 = vfmaq_f32(cv00, bv_lo, av0);  cv01 = vfmaq_f32(cv01, bv_hi, av0);
        cv10 = vfmaq_f32(cv10, bv_lo, av1);  cv11 = vfmaq_f32(cv11, bv_hi, av1);
        cv20 = vfmaq_f32(cv20, bv_lo, av2);  cv21 = vfmaq_f32(cv21, bv_hi, av2);
        cv30 = vfmaq_f32(cv30, bv_lo, av3);  cv31 = vfmaq_f32(cv31, bv_hi, av3);
    }

    // 写回 C
    vst1q_f32(&MAT_C(0, 0), cv00);  vst1q_f32(&MAT_C(0, 4), cv01);
    vst1q_f32(&MAT_C(1, 0), cv10);  vst1q_f32(&MAT_C(1, 4), cv11);
    vst1q_f32(&MAT_C(2, 0), cv20);  vst1q_f32(&MAT_C(2, 4), cv21);
    vst1q_f32(&MAT_C(3, 0), cv30);  vst1q_f32(&MAT_C(3, 4), cv31);
}

void register_block(int i_len, int j_len, int k_len,
                    float * __restrict__ A_pack,
                    float * __restrict__ B_pack,
                    float * __restrict__ C, int ldc) {
    for (int ir = 0; ir < i_len; ir += 4) {
        for (int jr = 0; jr < j_len; jr += 8) {
            micro_kernel(
                k_len,
                &A_pack[ir * k_len],
                &B_pack[(jr / 8) * k_len * 8],
                &MAT_C(ir, jr), ldc
            );
        }
    }
}

// 每种内部都是: pack_A, pack_B, 清零, register_block
// 区别只在三层循环的嵌套顺序

//  k -> j -> i
void cache_kji(int M, int N, int K,
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
                pack_A(i_len, k_len, A, lda, A_pack, i, k);
                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                }
                register_block(i_len, j_len, k_len, A_pack, B_pack, &MAT_C(i,j), ldc);
            }
        }
    }
    free(A_pack); free(B_pack);
}

// k -> i -> j
void cache_kij(int M, int N, int K,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int k = 0; k < K; k += Kc) {
        int k_len = std::min(Kc, K - k);
        for (int i = 0; i < M; i += Mc) {
            int i_len = std::min(Mc, M - i);
            pack_A(i_len, k_len, A, lda, A_pack, i, k);
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);
                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                }
                register_block(i_len, j_len, k_len, A_pack, B_pack, &MAT_C(i,j), ldc);
            }
        }
    }
    free(A_pack); free(B_pack);
}

// i -> j -> k
void cache_ijk(int M, int N, int K,
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
                pack_A(i_len, k_len, A, lda, A_pack, i, k);
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);
                register_block(i_len, j_len, k_len, A_pack, B_pack, &MAT_C(i,j), ldc);
            }
        }
    }
    free(A_pack); free(B_pack);
}

// i -> k -> j
void cache_ikj(int M, int N, int K,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int i = 0; i < M; i += Mc) {
        int i_len = std::min(Mc, M - i);
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            pack_A(i_len, k_len, A, lda, A_pack, i, k);
            for (int j = 0; j < N; j += Nc) {
                int j_len = std::min(Nc, N - j);
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);

                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                }

                register_block(i_len, j_len, k_len, A_pack, B_pack, &MAT_C(i,j), ldc);
            }
        }
    }
    free(A_pack); free(B_pack);
}

// j -> i -> k
void cache_jik(int M, int N, int K,
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
                pack_A(i_len, k_len, A, lda, A_pack, i, k);
                pack_B(k_len, j_len, B, ldb, B_pack, k, j);
                register_block(i_len, j_len, k_len, A_pack, B_pack, &MAT_C(i,j), ldc);
            }
        }
    }
    free(A_pack); free(B_pack);
}

// j -> k -> i
void cache_jki(int M, int N, int K,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc){
    float *A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float *B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));
    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);
        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            pack_B(k_len, j_len, B, ldb, B_pack, k, j);
            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                pack_A(i_len, k_len, A, lda, A_pack, i, k);
                
                if (k == 0) { 
                    for (int ic = i; ic < i+i_len; ic++) 
                        for (int jc = j; jc < j+j_len; jc++) 
                            MAT_C(ic,jc) = 0.0f; 
                
                        }
                register_block(i_len, j_len, k_len, A_pack, B_pack, &MAT_C(i,j), ldc);
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

    GemmTimer::bench("naive", M, N, K, 20, [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });

    GemmTimer::bench("kji", M, N, K, 100, [&](){ cache_kji(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);

    GemmTimer::bench("kij", M, N, K, 100, [&](){ cache_kij(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);

    GemmTimer::bench("ijk", M, N, K, 100, [&](){ cache_ijk(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);

    GemmTimer::bench("ikj", M, N, K, 100, [&](){ cache_ikj(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);

    GemmTimer::bench("jik", M, N, K, 100, [&](){ cache_jik(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);

    GemmTimer::bench("jki", M, N, K, 100, [&](){ cache_jki(M, N, K, A, lda, B, ldb, C_opt, ldc); });
    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
