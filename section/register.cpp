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
void register_block_4x4(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc){
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    // 主循环: 每次处理 4 个 k
    int kr = 0;
    for (; kr + 3 < Kc; kr += 4) {
        // ---- k+0 ----
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

        // ---- k+1 ----
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

        // ---- k+2 ----
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

        // ---- k+3 ----
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

    // 尾部处理: 剩余不足 4 个的 k
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

    // 写回 C 的 4×16 块
    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
}
