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
template<int Mr, int Nr>
void register_block_vv_4x4(
    int k_len,
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
    for (; kr + 3 < k_len; kr += 4) {
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
    for (; kr < k_len; kr++) {
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

template<int Mr, int Nr>
void register_block_vv_5x4(
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));
    float32x4_t cv40 = vld1q_f32(&MAT_C(4,  0));  float32x4_t cv41 = vld1q_f32(&MAT_C(4,  4));  float32x4_t cv42 = vld1q_f32(&MAT_C(4,  8));  float32x4_t cv43 = vld1q_f32(&MAT_C(4, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < k_len; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        bp += 16;  ap += 5;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        bp += 16;  ap += 5;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        bp += 16;  ap += 5;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        bp += 16;  ap += 5;
    }

    for (; kr < k_len; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        bp += 16;  ap += 5;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
    vst1q_f32(&MAT_C(4,  0), cv40);  vst1q_f32(&MAT_C(4,  4), cv41);  vst1q_f32(&MAT_C(4,  8), cv42);  vst1q_f32(&MAT_C(4, 12), cv43);
}

template<int Mr, int Nr>
void register_block_vv_4x5(
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));  float32x4_t cv04 = vld1q_f32(&MAT_C(0, 16));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));  float32x4_t cv14 = vld1q_f32(&MAT_C(1, 16));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));  float32x4_t cv24 = vld1q_f32(&MAT_C(2, 16));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));  float32x4_t cv34 = vld1q_f32(&MAT_C(3, 16));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3, bv4;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < k_len; kr += 4) {
        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);
        bp += 20;  ap += 4;

        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);
        bp += 20;  ap += 4;

        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);
        bp += 20;  ap += 4;

        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);
        bp += 20;  ap += 4;
    }

    for (; kr < k_len; kr++) {
        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);
        bp += 20;  ap += 4;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);  vst1q_f32(&MAT_C(0, 16), cv04);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);  vst1q_f32(&MAT_C(1, 16), cv14);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);  vst1q_f32(&MAT_C(2, 16), cv24);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);  vst1q_f32(&MAT_C(3, 16), cv34);
}
template<int Mr, int Nr>
void register_block_vv_6x4(
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));
    float32x4_t cv40 = vld1q_f32(&MAT_C(4,  0));  float32x4_t cv41 = vld1q_f32(&MAT_C(4,  4));  float32x4_t cv42 = vld1q_f32(&MAT_C(4,  8));  float32x4_t cv43 = vld1q_f32(&MAT_C(4, 12));
    float32x4_t cv50 = vld1q_f32(&MAT_C(5,  0));  float32x4_t cv51 = vld1q_f32(&MAT_C(5,  4));  float32x4_t cv52 = vld1q_f32(&MAT_C(5,  8));  float32x4_t cv53 = vld1q_f32(&MAT_C(5, 12));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < k_len; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);  cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);  cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);  cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);  cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;
    }

    for (; kr < k_len; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 4);  cv40 = vfmaq_f32(cv40, a_reg, bv0);  cv41 = vfmaq_f32(cv41, a_reg, bv1);  cv42 = vfmaq_f32(cv42, a_reg, bv2);  cv43 = vfmaq_f32(cv43, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 5);  cv50 = vfmaq_f32(cv50, a_reg, bv0);  cv51 = vfmaq_f32(cv51, a_reg, bv1);  cv52 = vfmaq_f32(cv52, a_reg, bv2);  cv53 = vfmaq_f32(cv53, a_reg, bv3);
        bp += 16;  ap += 6;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);
    vst1q_f32(&MAT_C(4,  0), cv40);  vst1q_f32(&MAT_C(4,  4), cv41);  vst1q_f32(&MAT_C(4,  8), cv42);  vst1q_f32(&MAT_C(4, 12), cv43);
    vst1q_f32(&MAT_C(5,  0), cv50);  vst1q_f32(&MAT_C(5,  4), cv51);  vst1q_f32(&MAT_C(5,  8), cv52);  vst1q_f32(&MAT_C(5, 12), cv53);
}
template<int Mr, int Nr>
void register_block_vv_4x6(
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00 = vld1q_f32(&MAT_C(0,  0));  float32x4_t cv01 = vld1q_f32(&MAT_C(0,  4));  float32x4_t cv02 = vld1q_f32(&MAT_C(0,  8));  float32x4_t cv03 = vld1q_f32(&MAT_C(0, 12));  float32x4_t cv04 = vld1q_f32(&MAT_C(0, 16));  float32x4_t cv05 = vld1q_f32(&MAT_C(0, 20));
    float32x4_t cv10 = vld1q_f32(&MAT_C(1,  0));  float32x4_t cv11 = vld1q_f32(&MAT_C(1,  4));  float32x4_t cv12 = vld1q_f32(&MAT_C(1,  8));  float32x4_t cv13 = vld1q_f32(&MAT_C(1, 12));  float32x4_t cv14 = vld1q_f32(&MAT_C(1, 16));  float32x4_t cv15 = vld1q_f32(&MAT_C(1, 20));
    float32x4_t cv20 = vld1q_f32(&MAT_C(2,  0));  float32x4_t cv21 = vld1q_f32(&MAT_C(2,  4));  float32x4_t cv22 = vld1q_f32(&MAT_C(2,  8));  float32x4_t cv23 = vld1q_f32(&MAT_C(2, 12));  float32x4_t cv24 = vld1q_f32(&MAT_C(2, 16));  float32x4_t cv25 = vld1q_f32(&MAT_C(2, 20));
    float32x4_t cv30 = vld1q_f32(&MAT_C(3,  0));  float32x4_t cv31 = vld1q_f32(&MAT_C(3,  4));  float32x4_t cv32 = vld1q_f32(&MAT_C(3,  8));  float32x4_t cv33 = vld1q_f32(&MAT_C(3, 12));  float32x4_t cv34 = vld1q_f32(&MAT_C(3, 16));  float32x4_t cv35 = vld1q_f32(&MAT_C(3, 20));

    const float *ap = A_pack;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3, bv4, bv5;
    float32x4_t a_reg;

    int kr = 0;
    for (; kr + 3 < k_len; kr += 4) {
        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);  bv5 = vld1q_f32(bp + 20);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);  cv05 = vfmaq_f32(cv05, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);  cv15 = vfmaq_f32(cv15, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);  cv25 = vfmaq_f32(cv25, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);  cv35 = vfmaq_f32(cv35, a_reg, bv5);
        bp += 24;  ap += 4;

        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);  bv5 = vld1q_f32(bp + 20);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);  cv05 = vfmaq_f32(cv05, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);  cv15 = vfmaq_f32(cv15, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);  cv25 = vfmaq_f32(cv25, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);  cv35 = vfmaq_f32(cv35, a_reg, bv5);
        bp += 24;  ap += 4;

        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);  bv5 = vld1q_f32(bp + 20);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);  cv05 = vfmaq_f32(cv05, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);  cv15 = vfmaq_f32(cv15, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);  cv25 = vfmaq_f32(cv25, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);  cv35 = vfmaq_f32(cv35, a_reg, bv5);
        bp += 24;  ap += 4;

        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);  bv5 = vld1q_f32(bp + 20);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);  cv05 = vfmaq_f32(cv05, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);  cv15 = vfmaq_f32(cv15, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);  cv25 = vfmaq_f32(cv25, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);  cv35 = vfmaq_f32(cv35, a_reg, bv5);
        bp += 24;  ap += 4;
    }

    for (; kr < k_len; kr++) {
        bv0 = vld1q_f32(bp +  0);  bv1 = vld1q_f32(bp +  4);  bv2 = vld1q_f32(bp +  8);  bv3 = vld1q_f32(bp + 12);  bv4 = vld1q_f32(bp + 16);  bv5 = vld1q_f32(bp + 20);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);  cv04 = vfmaq_f32(cv04, a_reg, bv4);  cv05 = vfmaq_f32(cv05, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);  cv14 = vfmaq_f32(cv14, a_reg, bv4);  cv15 = vfmaq_f32(cv15, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);  cv24 = vfmaq_f32(cv24, a_reg, bv4);  cv25 = vfmaq_f32(cv25, a_reg, bv5);
        a_reg = vld1q_dup_f32(ap + 3);  cv30 = vfmaq_f32(cv30, a_reg, bv0);  cv31 = vfmaq_f32(cv31, a_reg, bv1);  cv32 = vfmaq_f32(cv32, a_reg, bv2);  cv33 = vfmaq_f32(cv33, a_reg, bv3);  cv34 = vfmaq_f32(cv34, a_reg, bv4);  cv35 = vfmaq_f32(cv35, a_reg, bv5);
        bp += 24;  ap += 4;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);  vst1q_f32(&MAT_C(0, 16), cv04);  vst1q_f32(&MAT_C(0, 20), cv05);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);  vst1q_f32(&MAT_C(1, 16), cv14);  vst1q_f32(&MAT_C(1, 20), cv15);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);  vst1q_f32(&MAT_C(2, 16), cv24);  vst1q_f32(&MAT_C(2, 20), cv25);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);  vst1q_f32(&MAT_C(3, 16), cv34);  vst1q_f32(&MAT_C(3, 20), cv35);
}
template<int Mr, int Nr>
void register_block_vv_3x4(
    int k_len,
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
    for (; kr + 3 < k_len; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;
    }

    for (; kr < k_len; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg = vld1q_dup_f32(ap + 0);  cv00 = vfmaq_f32(cv00, a_reg, bv0);  cv01 = vfmaq_f32(cv01, a_reg, bv1);  cv02 = vfmaq_f32(cv02, a_reg, bv2);  cv03 = vfmaq_f32(cv03, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 1);  cv10 = vfmaq_f32(cv10, a_reg, bv0);  cv11 = vfmaq_f32(cv11, a_reg, bv1);  cv12 = vfmaq_f32(cv12, a_reg, bv2);  cv13 = vfmaq_f32(cv13, a_reg, bv3);
        a_reg = vld1q_dup_f32(ap + 2);  cv20 = vfmaq_f32(cv20, a_reg, bv0);  cv21 = vfmaq_f32(cv21, a_reg, bv1);  cv22 = vfmaq_f32(cv22, a_reg, bv2);  cv23 = vfmaq_f32(cv23, a_reg, bv3);
        bp += 16;  ap += 3;
    }

    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);
}



template<int Mr, int Nr>
static inline void register_op(
    int op_reg,
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C,
    int ldc) {
    switch (op_reg) {
        case 0: register_block_vv_4x4<Mr, Nr>(k_len, A_pack, B_pack, C, ldc); break;
        case 1: register_block_vv_5x4<Mr, Nr>(k_len, A_pack, B_pack, C, ldc); break;
        case 2: register_block_vv_4x5<Mr, Nr>(k_len, A_pack, B_pack, C, ldc); break;
        case 3: register_block_vv_6x4<Mr, Nr>(k_len, A_pack, B_pack, C, ldc); break;
        case 4: register_block_vv_4x6<Mr, Nr>(k_len, A_pack, B_pack, C, ldc); break;
        case 5: register_block_vv_3x4<Mr, Nr>(k_len, A_pack, B_pack, C, ldc); break;
        default: std::printf("错误: 不支持的循环顺序 op=%d\n", op_reg); break;
    }
}

void register_block(
    int Mr,
    int Nr,
    int op_reg,
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C,
    int ldc) {
    if (op_reg != 0) {
        std::printf("错误: 不支持的循环顺序 op=%d\n", op_reg);
        return;
    }
    if (Mr == 4 && Nr == 16) {
        register_block_vv_4x4<4, 16>(k_len, A_pack, B_pack, C, ldc);
        return;
    }
    if (Mr == 5 && Nr == 16) {
        register_block_vv_5x4<5, 16>(k_len, A_pack, B_pack, C, ldc);
        return;
    }
    if (Mr == 4 && Nr == 20) {
        register_block_vv_4x5<4, 20>(k_len, A_pack, B_pack, C, ldc);
        return;
    }
    if (Mr == 6 && Nr == 16) {
        register_block_vv_6x4<6, 16>(k_len, A_pack, B_pack, C, ldc);
        return;
    }
    if (Mr == 4 && Nr == 24) {
        register_block_vv_4x6<4, 24>(k_len, A_pack, B_pack, C, ldc);
        return;
    }
    if (Mr == 3 && Nr == 16) {
        register_block_vv_3x4<3, 16>(k_len, A_pack, B_pack, C, ldc);
        return;
    }
    std::printf("错误: 当前 register 不支持 Mr=%d, Nr=%d\n", Mr, Nr);
}
