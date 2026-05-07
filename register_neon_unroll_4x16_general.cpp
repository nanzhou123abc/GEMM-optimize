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

static inline void pack_A(int Kc, const float *A, int lda, float *A_pack, int mr) {
    for (int kc = 0; kc < Kc; kc++) {
        for (int r = 0; r < mr; r++) {
            A_pack[r] = A[r * lda + kc];
        }
        A_pack += mr;
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

static inline void register_block_4x16(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;  ap += 4;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;  ap += 4;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;  ap += 4;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;  ap += 4;
    }

    for (; kr < Kc; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);
        cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        bp += 16;  ap += 4;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
}

static inline void register_block_1x16(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));
    float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        bp += 16;  ap += 1;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        bp += 16;  ap += 1;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        bp += 16;  ap += 1;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        bp += 16;  ap += 1;
    }

    for (; kr < Kc; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        bp += 16;  ap += 1;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);
    vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
}

static inline void register_block_2x16(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        bp += 16;  ap += 2;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        bp += 16;  ap += 2;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        bp += 16;  ap += 2;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        bp += 16;  ap += 2;
    }

    for (; kr < Kc; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        bp += 16;  ap += 2;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
}

static inline void register_block_3x16(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;
    }

    for (; kr < Kc; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);
        cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);
        cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
}

static inline void register_block_Mx16(
    int mr, int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    switch (mr) {
        case 1: register_block_1x16(Kc, A_pack, B_pack, C, ldc); break;
        case 2: register_block_2x16(Kc, A_pack, B_pack, C, ldc); break;
        case 3: register_block_3x16(Kc, A_pack, B_pack, C, ldc); break;
        case 4: register_block_4x16(Kc, A_pack, B_pack, C, ldc); break;
    }
}

static inline void scalar_tail_N(
    int i_start, int i_len,
    int j_start, int j_len,
    int k_start, int k_len,
    float *A, int lda,
    float *B, int ldb,
    float *C, int ldc) {
    for (int ir = 0; ir < i_len; ir++) {
        for (int kr = 0; kr < k_len; kr++) {
            float a_val = MAT_A(i_start + ir, k_start + kr);
            for (int jr = 0; jr < j_len; jr++) {
                MAT_C(i_start + ir, j_start + jr) += a_val * MAT_B(k_start + kr, j_start + jr);
            }
        }
    }
}

void register_4x16_general(int M, int N, int K,
                           float * __restrict__ A, int lda,
                           float * __restrict__ B, int ldb,
                           float * __restrict__ C, int ldc) {
    float * __restrict__ A_pack = (float *)aligned_alloc(64, Mc * Kc * sizeof(float));
    float * __restrict__ B_pack = (float *)aligned_alloc(64, Kc * Nc * sizeof(float));

    for (int j = 0; j < N; j += Nc) {
        int j_len = std::min(Nc, N - j);
        int j_main = (j_len / 16) * 16;
        int j_tail = j_len - j_main;

        for (int i = 0; i < M; i++)
            memset(&MAT_C(i, j), 0, j_len * sizeof(float));

        for (int k = 0; k < K; k += Kc) {
            int k_len = std::min(Kc, K - k);

            if (j_main > 0)
                pack_B(k_len, j_main, B, ldb, B_pack, k, j);

            for (int i = 0; i < M; i += Mc) {
                int i_len = std::min(Mc, M - i);
                int i_main = (i_len / 4) * 4;
                int i_tail = i_len - i_main;

                for (int ic = 0; ic < i_main; ic += 4)
                    pack_A(k_len, &MAT_A(i + ic, k), lda, &A_pack[ic * k_len], 4);

                if (i_tail > 0)
                    pack_A(k_len, &MAT_A(i + i_main, k), lda, &A_pack[i_main * k_len], i_tail);

                for (int ir = 0; ir < i_main; ir += 4) {
                    for (int jr = 0; jr < j_main; jr += 16) {
                        register_block_4x16(
                            k_len,
                            &A_pack[ir * k_len],
                            &B_pack[(jr / 16) * k_len * 16],
                            &MAT_C(i + ir, j + jr), ldc
                        );
                    }
                }

                if (i_tail > 0 && j_main > 0) {
                    for (int jr = 0; jr < j_main; jr += 16) {
                        register_block_Mx16(
                            i_tail, k_len,
                            &A_pack[i_main * k_len],
                            &B_pack[(jr / 16) * k_len * 16],
                            &MAT_C(i + i_main, j + jr), ldc
                        );
                    }
                }

                if (j_tail > 0) {
                    scalar_tail_N(
                        i, i_len,
                        j + j_main, j_tail,
                        k, k_len,
                        A, lda, B, ldb, C, ldc
                    );
                }
            }
        }
    }

    free(A_pack);
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

    int lda = K, ldb = N, ldc = N;

    float *A        = (float *)aligned_alloc(64, M * K * sizeof(float));
    float *B        = (float *)aligned_alloc(64, K * N * sizeof(float));
    float *C_naive  = (float *)aligned_alloc(64, M * N * sizeof(float));
    float *C_opt    = (float *)aligned_alloc(64, M * N * sizeof(float));

    std::srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;

    GemmTimer::bench("naive",                      M, N, K, 20,  [&](){ naive(M, N, K, A, lda, B, ldb, C_naive, ldc); });
    GemmTimer::bench("register_4x16_general",      M, N, K, 200, [&](){ register_4x16_general(M, N, K, A, lda, B, ldb, C_opt, ldc); });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
