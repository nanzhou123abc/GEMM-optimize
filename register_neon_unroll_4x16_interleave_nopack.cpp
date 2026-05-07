#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
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

/*
 * ============================================================================
 * register_block_4x4  — 微内核 4行×16列, k方向×4展开
 * ============================================================================
 *
 * 计算: C[4×16] += A[4×Kc] × B[Kc×16]
 *
 *
 * │                        整体数据流                                     
 * │                                                                     
 * │   A[4×Kc]              B[Kc×16]              C[4×16]                
 * │   ┌──────────┐         ┌──────────┐         ┌──────────┐            
 * │   │ a00 a01..│         │ b00..b0f │         │ c00..c0f │            
 * │   │ a10 a11..│    ×    │ b10..b1f │    +=   │ c10..c1f │            
 * │   │ a20 a21..│         │  ......  │         │ c20..c2f │            
 * │   │ a30 a31..│         │ bk0..bkf │         │ c30..c3f │            
 * │   └──────────┘         └──────────┘         └──────────┘            
 * │    4 rows × Kc cols      Kc rows × 16 cols    4 rows × 16 cols      
 * 
 *
 * │                     C 累加器寄存器布局 (16个 float32x4_t)              
 * │                                                                     
 * │   C 的 4×16 块被 16 个 NEON 寄存器覆盖, 每个寄存器存 4 个 float:         
 * │                                                                     
 * │        col 0-3    col 4-7    col 8-11   col 12-15                  
 * │   row0 [cv00]     [cv01]     [cv02]     [cv03]                     
 * │   row1 [cv10]     [cv11]     [cv12]     [cv13]                     
 * │   row2 [cv20]     [cv21]     [cv22]     [cv23]                     
 * │   row3 [cv30]     [cv31]     [cv32]     [cv33]                     
 * │                                                                    
 * │   命名规则: cv{row}{col_vec}  row=0..3, col_vec=0..3                 
 * │   例如 cv12 = C 第 1 行, 第 8~11 列 (4个float)                        
 *
 * │              单个 k 步的计算模式 (以 k=kr 为例)                         
 * │                                                                     
 * │   Step 1: 从 B_pack 加载 4 个 NEON 向量 (B 的一整行, 16个float)         
 * │                                                                     
 * │     B_pack 布局 (按行优先排列, 每行16个float连续):                      
 * │     ┌──────────────────────────────────────────┐                    
 * │     │ bv0 = B[kr][0..3]   = bp[0..3]           │                    
 * │     │ bv1 = B[kr][4..7]   = bp[4..7]           │                    
 * │     │ bv2 = B[kr][8..11]  = bp[8..11]          │                    
 * │     │ bv3 = B[kr][12..15] = bp[12..15]         │                    
 * │     └──────────────────────────────────────────┘                    
 * │                                                                     
 * │   Step 2: 4 个独立指针沿 A 的 4 行步进, 加载标量并广播               
 * │                                                                     
 * │     A 矩阵 (row-major, lda=K), 4个指针各自沿行方向++:                
 * │     ┌──────────────────────────────────────────┐                    
 * │     │ ap0 → A[0][k] → A[0][k+1] → A[0][k+2]... │                    
 * │     │ ap1 → A[1][k] → A[1][k+1] → A[1][k+2]... │                    
 * │     │ ap2 → A[2][k] → A[2][k+1] → A[2][k+2]... │                    
 * │     │ ap3 → A[3][k] → A[3][k+1] → A[3][k+2]... │                    
 * │     └──────────────────────────────────────────┘                    
 * │                                                                     
 * │     每次: a_reg = vld1q_dup_f32(apN); apN++;                       
 * │                                                                     
 * │   Step 3: 16 次 FMA (Fused Multiply-Add)                            
 * │                                                                     
 * │     对于 row 0:                                                     
 * │       cv00 += A[0][kr] * bv0    (即 C[0][0..3]   += a * B[kr][0..3])   
 * │       cv01 += A[0][kr] * bv1    (即 C[0][4..7]   += a * B[kr][4..7])   
 * │       cv02 += A[0][kr] * bv2    (即 C[0][8..11]  += a * B[kr][8..11])   
 * │       cv03 += A[0][kr] * bv3    (即 C[0][12..15] += a * B[kr][12..15])   
 * │                                                                     
 * │     对于 row 1,2,3 同理, 共 4行 × 4向量 = 16 次 FMA                    
 * │                                                                     
 * │   可视化 (以 row 0 为例, 一次外积):                                    
 * │                                                                     
 * │          B[kr][0..15] →                                             
 * │          ┌────────────────────────┐                                 
 * │   A↓     │ b0  b1  b2  b3  ... bf │                                 
 * │   ┌───┐  ├────────────────────────┤                                 
 * │   │a00│  │a00*b0  a00*b1  ...  a00*bf│  → 累加到 C[0][0..15]         
 * │   └───┘  └────────────────────────┘                                 
 *
 * │                   k 方向 ×4 展开 (loop unrolling)                    
 * │                                                                     
 * │   不展开: 每次循环处理 1 个 k, 循环 Kc 次                               
 * │   展开×4: 每次循环处理 4 个 k, 循环 Kc/4 次                             
 * │                                                                     
 * │   一次展开迭代处理: k=kr, kr+1, kr+2, kr+3                             
 * │                                                                     
 * │   寄存器复用策略:                                                     
 * │     - bv0..bv3: 4个B向量, 每个k步复用 (先加载k+0的B,用完再加载k+1的B)     
 * │     - a_reg:   1个A标量, 每个row复用 (先广播row0,用完再广播row1)         
 * │     - cv00..cv33: 16个C累加器, 全程驻留寄存器                         
 * │                                                                     
 * │   寄存器用量: 16(C) + 4(B) + 1(A) = 21 个 (NEON共32个, 充裕)           
 * │                                                                     
 * │   时间线 (一次展开迭代, 4个k步):                                        
 * │                                                                     
 * │   k+0: load B → load A[0..3] → 16 FMA → bp+=16                     
 * │   k+1: load B → load A[0..3] → 16 FMA → bp+=16                     
 * │   k+2: load B → load A[0..3] → 16 FMA → bp+=16                     
 * │   k+3: load B → load A[0..3] → 16 FMA → bp+=16                     
 * │                                                                     
 * │   总计: 4×16 = 64 次 FMA / 展开迭代                                   
 *
 * │                      B_pack 内存布局                                   
 * │                                                                       
 * │   B_pack 按 16列 一组排列, 每组内按 k 方向连续:                            
 * │                                                                       
 * │   ┌──────────┬──────────┬──────────┬─────┐                            
 * │   │ col 0-15 │ col 0-15 │ col 0-15 │ ... │  ← 每组16列                 
 * │   │  k = 0   │  k = 1   │  k = 2   │     │  ← k方向连续                
 * │   └──────────┴──────────┴──────────┴─────┘                            
 * │                                                                       
 * │   对于 jr=0 的 16 列: B_pack[0..Kc*16-1]                               
 * │   对于 jr=16 的 16 列: B_pack[Kc*16 .. 2*Kc*16-1]                      
 * │                                                                       
 * │   微内核调用时传入: &B_pack[(jr/16) * k_len * 16]                        
 * │   即跳到对应 16 列组的起始位置                                            
 */
static inline void register_block_4x4(
    int Kc,
    const float *A, int lda,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));

    const float *bp = B_pack;

    const float *ap0 = &A[0 * lda + 0];
    const float *ap1 = &A[1 * lda + 0];
    const float *ap2 = &A[2 * lda + 0];
    const float *ap3 = &A[3 * lda + 0];

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap0); ap0++;
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap1); ap1++;
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap2); ap2++;
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap3); ap3++;
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap0); ap0++;
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap1); ap1++;
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap2); ap2++;
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap3); ap3++;
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap0); ap0++;
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap1); ap1++;
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap2); ap2++;
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap3); ap3++;
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap0); ap0++;
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap1); ap1++;
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap2); ap2++;
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap3); ap3++;
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;
    }

    for (; kr < Kc; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap0); ap0++;
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap1); ap1++;
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap2); ap2++;
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap3); ap3++;
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
}

void register_neon_u16oll_gemm(int M, int N, int K,
                               float * __restrict__ A, int lda,
                               float * __restrict__ B, int ldb,
                               float * __restrict__ C, int ldc) {
    float * __restrict__ B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));


    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);

        for (int i = 0; i < M; i++)
            memset(&MAT_C(i, j), 0, j_len * sizeof(float));

        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);
            pack_B(k_len, j_len, B, ldb, B_pack, k, j);

            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);

                for (int ir = 0; ir < i_len; ir += 4) {
                    for (int jr = 0; jr < j_len; jr += 16) {
                        register_block_4x4(
                            k_len,
                            &MAT_A(i + ir, k), lda,
                            &B_pack[(jr / 16) * k_len * 16],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }
            }
        }
    }

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

    if (M % 4 != 0 || N % 16 != 0) {
        printf("错误: M 必须是 %d 的倍数, N 必须是 %d 的倍数\n", 4, 16);
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

    GemmTimer::bench("naive",                    M, N, K, 20,  [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("register_neon_4x16_nopack", M, N, K, 200, [&](){ register_neon_u16oll_gemm(M, N, K, A, lda, B, ldb, C_opt, ldc); });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
