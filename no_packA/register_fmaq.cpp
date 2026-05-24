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
    const float *A, int lda,
    const float *B_pack,
    float *C, int ldc){

    float32x4_t cv00, cv01, cv02, cv03;
    float32x4_t cv10, cv11, cv12, cv13;
    float32x4_t cv20, cv21, cv22, cv23;
    float32x4_t cv30, cv31, cv32, cv33;
    const float *ap0 = A;
    const float *ap1 = A + lda;
    const float *ap2 = A + 2 * lda;
    const float *ap3 = A + 3 * lda;
    const float *bp = B_pack;
    float *cp = C;

    float32x4_t av0, av1, av2, av3;
    float32x4_t bv0, bv1, bv2, bv3;
    bv0 = vld1q_f32(bp + 0);  
    bv1 = vld1q_f32(bp + 4);  
    bv2 = vld1q_f32(bp + 8);  
    bv3 = vld1q_f32(bp + 12);
    
    av0 = vld1q_dup_f32(ap0); ap0++;
    av1 = vld1q_dup_f32(ap1); ap1++;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp + 4);  cv02 = vld1q_f32(cp + 8);  cv03 = vld1q_f32(cp + 12);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp + 4);  cv12 = vld1q_f32(cp + 8);  cv13 = vld1q_f32(cp + 12);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp + 4);  cv22 = vld1q_f32(cp + 8);  cv23 = vld1q_f32(cp + 12);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp + 4);  cv32 = vld1q_f32(cp + 8);  cv33 = vld1q_f32(cp + 12);

    
    int kr = 0;
    for (; kr + 1 < k_len; kr += 2) {
        //k1
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        // k2
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        // //k3
        // av2 = vld1q_dup_f32(ap2); ap2++;
        // av3 = vld1q_dup_f32(ap3); ap3++;
        // cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        // cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        // av0 = vld1q_dup_f32(ap0); ap0++;
        // av1 = vld1q_dup_f32(ap1); ap1++;
        // cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        // cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);

        // bp += 16;
        // bv0 = vld1q_f32(bp + 0);
        // bv1 = vld1q_f32(bp + 4);
        // bv2 = vld1q_f32(bp + 8);
        // bv3 = vld1q_f32(bp + 12);

        // // k4
        // av2 = vld1q_dup_f32(ap2); ap2++;
        // av3 = vld1q_dup_f32(ap3); ap3++;
        // cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        // cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        // av0 = vld1q_dup_f32(ap0); ap0++;
        // av1 = vld1q_dup_f32(ap1); ap1++;
        // cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        // cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);

        // bp += 16;
        // bv0 = vld1q_f32(bp + 0);
        // bv1 = vld1q_f32(bp + 4);
        // bv2 = vld1q_f32(bp + 8);
        // bv3 = vld1q_f32(bp + 12);
    }

    for (; kr < k_len; kr++) {
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);   
        bv1 = vld1q_f32(bp + 4);  
        bv2 = vld1q_f32(bp + 8);  
        bv3 = vld1q_f32(bp + 12);
    }
    cp = C;
    vst1q_f32(cp, cv00);  vst1q_f32(cp+4, cv01);  vst1q_f32(cp+8, cv02);  vst1q_f32(cp+12, cv03); 
    cp += ldc;
    vst1q_f32(cp, cv10);  vst1q_f32(cp+4, cv11);  vst1q_f32(cp+8, cv12);  vst1q_f32(cp+12, cv13); 
    cp +=  ldc;
    vst1q_f32(cp, cv20);  vst1q_f32(cp+4, cv21);  vst1q_f32(cp+8, cv22);  vst1q_f32(cp+12, cv23);  
    cp += ldc;
    vst1q_f32(cp, cv30);  vst1q_f32(cp+4, cv31);  vst1q_f32(cp+8, cv32);  vst1q_f32(cp+12, cv33);  
}

template<int Mr, int Nr>
void register_block_vv_5x4(
    int k_len,
    const float *A, int lda,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00, cv01, cv02, cv03;
    float32x4_t cv10, cv11, cv12, cv13;
    float32x4_t cv20, cv21, cv22, cv23;
    float32x4_t cv30, cv31, cv32, cv33;
    float32x4_t cv40, cv41, cv42, cv43;

    const float *ap0 = A;
    const float *ap1 = A + lda;
    const float *ap2 = A + 2 * lda;
    const float *ap3 = A + 3 * lda;
    const float *ap4 = A + 4 * lda;
    const float *bp = B_pack;
    float *cp = C;

    float32x4_t av0, av1, av2, av3, av4;
    float32x4_t bv0, bv1, bv2, bv3;

    bv0 = vld1q_f32(bp + 0);
    bv1 = vld1q_f32(bp + 4);
    bv2 = vld1q_f32(bp + 8);
    bv3 = vld1q_f32(bp + 12);

    av0 = vld1q_dup_f32(ap0); ap0++;
    av1 = vld1q_dup_f32(ap1); ap1++;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp + 4);  cv02 = vld1q_f32(cp + 8);  cv03 = vld1q_f32(cp + 12);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp + 4);  cv12 = vld1q_f32(cp + 8);  cv13 = vld1q_f32(cp + 12);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp + 4);  cv22 = vld1q_f32(cp + 8);  cv23 = vld1q_f32(cp + 12);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp + 4);  cv32 = vld1q_f32(cp + 8);  cv33 = vld1q_f32(cp + 12);
    cp += ldc;
    cv40 = vld1q_f32(cp);  cv41 = vld1q_f32(cp + 4);  cv42 = vld1q_f32(cp + 8);  cv43 = vld1q_f32(cp + 12);

    int kr = 0;
    for (; kr + 1 < k_len; kr += 2) {
        // k1
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        av4 = vld1q_dup_f32(ap4); ap4++;
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        // k2
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        av4 = vld1q_dup_f32(ap4); ap4++;
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        // // k3
        // av2 = vld1q_dup_f32(ap2); ap2++;
        // av3 = vld1q_dup_f32(ap3); ap3++;
        // cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        // cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        // av4 = vld1q_dup_f32(ap4); ap4++;
        // cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        // cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        // av0 = vld1q_dup_f32(ap0); ap0++;
        // av1 = vld1q_dup_f32(ap1); ap1++;
        // cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);

        // bp += 16;
        // bv0 = vld1q_f32(bp + 0);
        // bv1 = vld1q_f32(bp + 4);
        // bv2 = vld1q_f32(bp + 8);
        // bv3 = vld1q_f32(bp + 12);

        // // k4
        // av2 = vld1q_dup_f32(ap2); ap2++;
        // av3 = vld1q_dup_f32(ap3); ap3++;
        // cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        // cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        // av4 = vld1q_dup_f32(ap4); ap4++;
        // cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        // cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        // av0 = vld1q_dup_f32(ap0); ap0++;
        // av1 = vld1q_dup_f32(ap1); ap1++;
        // cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);

        // bp += 16;
        // bv0 = vld1q_f32(bp + 0);
        // bv1 = vld1q_f32(bp + 4);
        // bv2 = vld1q_f32(bp + 8);
        // bv3 = vld1q_f32(bp + 12);
    }

    for (; kr < k_len; kr++) {
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        av4 = vld1q_dup_f32(ap4); ap4++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);

        bp += 16;
    
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        
    }

    cp = C;
    vst1q_f32(cp, cv00);  vst1q_f32(cp + 4, cv01);  vst1q_f32(cp + 8, cv02);  vst1q_f32(cp + 12, cv03);
    cp += ldc;
    vst1q_f32(cp, cv10);  vst1q_f32(cp + 4, cv11);  vst1q_f32(cp + 8, cv12);  vst1q_f32(cp + 12, cv13);
    cp += ldc;
    vst1q_f32(cp, cv20);  vst1q_f32(cp + 4, cv21);  vst1q_f32(cp + 8, cv22);  vst1q_f32(cp + 12, cv23);
    cp += ldc;
    vst1q_f32(cp, cv30);  vst1q_f32(cp + 4, cv31);  vst1q_f32(cp + 8, cv32);  vst1q_f32(cp + 12, cv33);
    cp += ldc;
    vst1q_f32(cp, cv40);  vst1q_f32(cp + 4, cv41);  vst1q_f32(cp + 8, cv42);  vst1q_f32(cp + 12, cv43);
}


template<int Mr, int Nr>
void register_block_vv_6x4(
    int k_len,
    const float *A, int lda,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00, cv01, cv02, cv03;
    float32x4_t cv10, cv11, cv12, cv13;
    float32x4_t cv20, cv21, cv22, cv23;
    float32x4_t cv30, cv31, cv32, cv33;
    float32x4_t cv40, cv41, cv42, cv43;
    float32x4_t cv50, cv51, cv52, cv53;

    const float *ap0 = A;
    const float *ap1 = A + lda;
    const float *ap2 = A + 2 * lda;
    const float *ap3 = A + 3 * lda;
    const float *ap4 = A + 4 * lda;
    const float *ap5 = A + 5 * lda;
    const float *bp = B_pack;
    float *cp = C;

    float32x4_t av0, av1, av2, av3, av4, av5;
    float32x4_t bv0, bv1, bv2, bv3;

    bv0 = vld1q_f32(bp + 0);
    bv1 = vld1q_f32(bp + 4);
    bv2 = vld1q_f32(bp + 8);
    bv3 = vld1q_f32(bp + 12);

    av0 = vld1q_dup_f32(ap0); ap0++;
    av1 = vld1q_dup_f32(ap1); ap1++;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp + 4);  cv02 = vld1q_f32(cp + 8);  cv03 = vld1q_f32(cp + 12);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp + 4);  cv12 = vld1q_f32(cp + 8);  cv13 = vld1q_f32(cp + 12);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp + 4);  cv22 = vld1q_f32(cp + 8);  cv23 = vld1q_f32(cp + 12);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp + 4);  cv32 = vld1q_f32(cp + 8);  cv33 = vld1q_f32(cp + 12);
    cp += ldc;
    cv40 = vld1q_f32(cp);  cv41 = vld1q_f32(cp + 4);  cv42 = vld1q_f32(cp + 8);  cv43 = vld1q_f32(cp + 12);
    cp += ldc;
    cv50 = vld1q_f32(cp);  cv51 = vld1q_f32(cp + 4);  cv52 = vld1q_f32(cp + 8);  cv53 = vld1q_f32(cp + 12);

    int kr = 0;
    for (; kr + 1 < k_len; kr += 2) {
        // k1
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        av4 = vld1q_dup_f32(ap4); ap4++;
        av5 = vld1q_dup_f32(ap5); ap5++;
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);
        cv50 = vfmaq_f32(cv50, av5, bv0);  cv51 = vfmaq_f32(cv51, av5, bv1);  cv52 = vfmaq_f32(cv52, av5, bv2);  cv53 = vfmaq_f32(cv53, av5, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        // k2
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        av4 = vld1q_dup_f32(ap4); ap4++;
        av5 = vld1q_dup_f32(ap5); ap5++;
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);
        cv50 = vfmaq_f32(cv50, av5, bv0);  cv51 = vfmaq_f32(cv51, av5, bv1);  cv52 = vfmaq_f32(cv52, av5, bv2);  cv53 = vfmaq_f32(cv53, av5, bv3);

        bp += 16;
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        // //k3
        // av2 = vld1q_dup_f32(ap2); ap2++;
        // av3 = vld1q_dup_f32(ap3); ap3++;
        // cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        // cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        // av4 = vld1q_dup_f32(ap4); ap4++;
        // av5 = vld1q_dup_f32(ap5); ap5++;
        // cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        // cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        // av0 = vld1q_dup_f32(ap0); ap0++;
        // av1 = vld1q_dup_f32(ap1); ap1++;
        // cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);
        // cv50 = vfmaq_f32(cv50, av5, bv0);  cv51 = vfmaq_f32(cv51, av5, bv1);  cv52 = vfmaq_f32(cv52, av5, bv2);  cv53 = vfmaq_f32(cv53, av5, bv3);

        // bp += 16;
        // bv0 = vld1q_f32(bp + 0);
        // bv1 = vld1q_f32(bp + 4);
        // bv2 = vld1q_f32(bp + 8);
        // bv3 = vld1q_f32(bp + 12);

        // //k4
        // av2 = vld1q_dup_f32(ap2); ap2++;
        // av3 = vld1q_dup_f32(ap3); ap3++;
        // cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        // cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        // av4 = vld1q_dup_f32(ap4); ap4++;
        // av5 = vld1q_dup_f32(ap5); ap5++;
        // cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        // cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        // av0 = vld1q_dup_f32(ap0); ap0++;
        // av1 = vld1q_dup_f32(ap1); ap1++;
        // cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);
        // cv50 = vfmaq_f32(cv50, av5, bv0);  cv51 = vfmaq_f32(cv51, av5, bv1);  cv52 = vfmaq_f32(cv52, av5, bv2);  cv53 = vfmaq_f32(cv53, av5, bv3);

        // bp += 16;
        // bv0 = vld1q_f32(bp + 0);
        // bv1 = vld1q_f32(bp + 4);
        // bv2 = vld1q_f32(bp + 8);
        // bv3 = vld1q_f32(bp + 12);
    }

    for (; kr < k_len; kr++) {
        av2 = vld1q_dup_f32(ap2); ap2++;
        av3 = vld1q_dup_f32(ap3); ap3++;
        av4 = vld1q_dup_f32(ap4); ap4++;
        av5 = vld1q_dup_f32(ap5); ap5++;
        cv00 = vfmaq_f32(cv00, av0, bv0);  cv01 = vfmaq_f32(cv01, av0, bv1);  cv02 = vfmaq_f32(cv02, av0, bv2);  cv03 = vfmaq_f32(cv03, av0, bv3);
        cv10 = vfmaq_f32(cv10, av1, bv0);  cv11 = vfmaq_f32(cv11, av1, bv1);  cv12 = vfmaq_f32(cv12, av1, bv2);  cv13 = vfmaq_f32(cv13, av1, bv3);
        cv20 = vfmaq_f32(cv20, av2, bv0);  cv21 = vfmaq_f32(cv21, av2, bv1);  cv22 = vfmaq_f32(cv22, av2, bv2);  cv23 = vfmaq_f32(cv23, av2, bv3);
        cv30 = vfmaq_f32(cv30, av3, bv0);  cv31 = vfmaq_f32(cv31, av3, bv1);  cv32 = vfmaq_f32(cv32, av3, bv2);  cv33 = vfmaq_f32(cv33, av3, bv3);
        cv40 = vfmaq_f32(cv40, av4, bv0);  cv41 = vfmaq_f32(cv41, av4, bv1);  cv42 = vfmaq_f32(cv42, av4, bv2);  cv43 = vfmaq_f32(cv43, av4, bv3);
        cv50 = vfmaq_f32(cv50, av5, bv0);  cv51 = vfmaq_f32(cv51, av5, bv1);  cv52 = vfmaq_f32(cv52, av5, bv2);  cv53 = vfmaq_f32(cv53, av5, bv3);

        bp += 16;
       
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);
        av0 = vld1q_dup_f32(ap0); ap0++;
        av1 = vld1q_dup_f32(ap1); ap1++;
        
    }

    cp = C;
    vst1q_f32(cp, cv00);  vst1q_f32(cp + 4, cv01);  vst1q_f32(cp + 8, cv02);  vst1q_f32(cp + 12, cv03);
    cp += ldc;
    vst1q_f32(cp, cv10);  vst1q_f32(cp + 4, cv11);  vst1q_f32(cp + 8, cv12);  vst1q_f32(cp + 12, cv13);
    cp += ldc;
    vst1q_f32(cp, cv20);  vst1q_f32(cp + 4, cv21);  vst1q_f32(cp + 8, cv22);  vst1q_f32(cp + 12, cv23);
    cp += ldc;
    vst1q_f32(cp, cv30);  vst1q_f32(cp + 4, cv31);  vst1q_f32(cp + 8, cv32);  vst1q_f32(cp + 12, cv33);
    cp += ldc;
    vst1q_f32(cp, cv40);  vst1q_f32(cp + 4, cv41);  vst1q_f32(cp + 8, cv42);  vst1q_f32(cp + 12, cv43);
    cp += ldc;
    vst1q_f32(cp, cv50);  vst1q_f32(cp + 4, cv51);  vst1q_f32(cp + 8, cv52);  vst1q_f32(cp + 12, cv53);
}


template<int Mr, int Nr>
void register_block_vv_4x5(
    int k_len,
    const float *A,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00, cv01, cv02, cv03, cv04;
    float32x4_t cv10, cv11, cv12, cv13, cv14;
    float32x4_t cv20, cv21, cv22, cv23, cv24;
    float32x4_t cv30, cv31, cv32, cv33, cv34;
    const float *ap = A;
    const float *bp = B_pack;
    float *cp = C;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp+4);  cv02 = vld1q_f32(cp+8);  cv03 = vld1q_f32(cp+12);  cv04 = vld1q_f32(cp+16);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp+4);  cv12 = vld1q_f32(cp+8);  cv13 = vld1q_f32(cp+12);  cv14 = vld1q_f32(cp+16);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp+4);  cv22 = vld1q_f32(cp+8);  cv23 = vld1q_f32(cp+12);  cv24 = vld1q_f32(cp+16);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp+4);  cv32 = vld1q_f32(cp+8);  cv33 = vld1q_f32(cp+12);  cv34 = vld1q_f32(cp+16);

    float32x4_t a_reg0, a_reg1;
    float32x4_t bv0, bv1, bv2, bv3, bv4;
    int kr = 0;

    a_reg0 = vld1q_f32(ap);
    bv0 = vld1q_f32(bp +  0);
    bv1 = vld1q_f32(bp +  4);
    bv2 = vld1q_f32(bp +  8);

    for(; kr + 1 < k_len; kr += 2) {
        bv3 = vld1q_f32(bp + 12);
        bv4 = vld1q_f32(bp + 16);
    	a_reg1 = vld1q_f32(ap+4);
	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg0, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg0, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg0, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg0, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg0, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg0, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg0, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg0, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg0, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg0, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg0, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg0, 3);

        bp += 20;  ap += 4;
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg0, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg0, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg0, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg0, 3);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg0, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg0, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg0, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg0, 3);
	

        bv3 = vld1q_f32(bp + 12);
        bv4 = vld1q_f32(bp + 16);
	a_reg0 = vld1q_f32(ap+4);
	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg1, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg1, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg1, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg1, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg1, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg1, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg1, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg1, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg1, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg1, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg1, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg1, 3);

        bp += 20;  ap += 4;
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg1, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg1, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg1, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg1, 3);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg1, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg1, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg1, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg1, 3);

    }

    for (; kr < k_len; kr++) {
        bv3 = vld1q_f32(bp + 12);
        bv4 = vld1q_f32(bp + 16);
	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg0, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg0, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg0, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg0, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg0, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg0, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg0, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg0, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg0, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg0, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg0, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg0, 3);
       
        bp += 20;  ap += 4;
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
	cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg0, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg0, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg0, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg0, 3);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg0, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg0, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg0, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg0, 3);

        a_reg0 = vld1q_f32(ap);
    }

    cp = C;
    vst1q_f32(cp, cv00);  vst1q_f32(cp+4, cv01);  vst1q_f32(cp+8, cv02);  vst1q_f32(cp+12, cv03);  vst1q_f32(cp+16, cv04);
    cp += ldc;
    vst1q_f32(cp, cv10);  vst1q_f32(cp+4, cv11);  vst1q_f32(cp+8, cv12);  vst1q_f32(cp+12, cv13);  vst1q_f32(cp+16, cv14);
    cp +=  ldc;
    vst1q_f32(cp, cv20);  vst1q_f32(cp+4, cv21);  vst1q_f32(cp+8, cv22);  vst1q_f32(cp+12, cv23);  vst1q_f32(cp+16, cv24);
    cp += ldc;
    vst1q_f32(cp, cv30);  vst1q_f32(cp+4, cv31);  vst1q_f32(cp+8, cv32);  vst1q_f32(cp+12, cv33);  vst1q_f32(cp+16, cv34);
}

template<int Mr, int Nr>
void register_block_vv_4x6(
    int k_len,
    const float *A,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00, cv01, cv02, cv03, cv04, cv05;
    float32x4_t cv10, cv11, cv12, cv13, cv14, cv15;
    float32x4_t cv20, cv21, cv22, cv23, cv24, cv25;
    float32x4_t cv30, cv31, cv32, cv33, cv34, cv35;
    const float *ap = A;
    const float *bp = B_pack;
    float *cp = C;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp+4);  cv02 = vld1q_f32(cp+8);  cv03 = vld1q_f32(cp+12);  cv04 = vld1q_f32(cp+16); cv05 = vld1q_f32(cp+20);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp+4);  cv12 = vld1q_f32(cp+8);  cv13 = vld1q_f32(cp+12);  cv14 = vld1q_f32(cp+16); cv15 = vld1q_f32(cp+20);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp+4);  cv22 = vld1q_f32(cp+8);  cv23 = vld1q_f32(cp+12);  cv24 = vld1q_f32(cp+16); cv25 = vld1q_f32(cp+20);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp+4);  cv32 = vld1q_f32(cp+8);  cv33 = vld1q_f32(cp+12);  cv34 = vld1q_f32(cp+16); cv35 = vld1q_f32(cp+20);

    /*
    cv00 = vld1q_f32(&MAT_C(0,  0));  cv01 = vld1q_f32(&MAT_C(0,  4));  cv02 = vld1q_f32(&MAT_C(0,  8));  cv03 = vld1q_f32(&MAT_C(0, 12));  cv04 = vld1q_f32(&MAT_C(0, 16));  cv05 = vld1q_f32(&MAT_C(0, 20));
    cv10 = vld1q_f32(&MAT_C(1,  0));  cv11 = vld1q_f32(&MAT_C(1,  4));  cv12 = vld1q_f32(&MAT_C(1,  8));  cv13 = vld1q_f32(&MAT_C(1, 12));  cv14 = vld1q_f32(&MAT_C(1, 16));  cv15 = vld1q_f32(&MAT_C(1, 20));
    cv20 = vld1q_f32(&MAT_C(2,  0));  cv21 = vld1q_f32(&MAT_C(2,  4));  cv22 = vld1q_f32(&MAT_C(2,  8));  cv23 = vld1q_f32(&MAT_C(2, 12));  cv24 = vld1q_f32(&MAT_C(2, 16));  cv25 = vld1q_f32(&MAT_C(2, 20));
    cv30 = vld1q_f32(&MAT_C(3,  0));  cv31 = vld1q_f32(&MAT_C(3,  4));  cv32 = vld1q_f32(&MAT_C(3,  8));  cv33 = vld1q_f32(&MAT_C(3, 12));  cv34 = vld1q_f32(&MAT_C(3, 16));  cv35 = vld1q_f32(&MAT_C(3, 20));
*/
    float32x4_t bv0, bv1, bv2, bv3, bv4, bv5;
    float32x4_t a_reg;

    int kr = 0;

    a_reg = vld1q_f32(ap);
    bv0 = vld1q_f32(bp +  0);
    bv1 = vld1q_f32(bp +  4);
    bv2 = vld1q_f32(bp +  8);
    bv3 = vld1q_f32(bp + 12);

    for (; kr + 3 < k_len; kr += 4) {
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);
	
	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg, 3);
	bp += 24;  ap += 4;
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
        bv3 = vld1q_f32(bp + 12);
	
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg, 3);
	cv05 = vfmaq_laneq_f32(cv05, bv5, a_reg, 0);cv15 = vfmaq_laneq_f32(cv15, bv5, a_reg, 1);cv25 = vfmaq_laneq_f32(cv25, bv5, a_reg, 2);cv35 = vfmaq_laneq_f32(cv35, bv5, a_reg, 3);

        a_reg = vld1q_f32(ap);
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);

	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg, 3);
	bp += 24;  ap += 4;
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
        bv3 = vld1q_f32(bp + 12);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg, 3);
	cv05 = vfmaq_laneq_f32(cv05, bv5, a_reg, 0);cv15 = vfmaq_laneq_f32(cv15, bv5, a_reg, 1);cv25 = vfmaq_laneq_f32(cv25, bv5, a_reg, 2);cv35 = vfmaq_laneq_f32(cv35, bv5, a_reg, 3);

        a_reg = vld1q_f32(ap);
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);

	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg, 3);
	bp += 24;  ap += 4;
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
        bv3 = vld1q_f32(bp + 12);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg, 3);
	cv05 = vfmaq_laneq_f32(cv05, bv5, a_reg, 0);cv15 = vfmaq_laneq_f32(cv15, bv5, a_reg, 1);cv25 = vfmaq_laneq_f32(cv25, bv5, a_reg, 2);cv35 = vfmaq_laneq_f32(cv35, bv5, a_reg, 3);
        
	a_reg = vld1q_f32(ap);
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);

	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg, 3);
	bp += 24;  ap += 4;
    	bv0 = vld1q_f32(bp +  0);
    	bv1 = vld1q_f32(bp +  4);
    	bv2 = vld1q_f32(bp +  8);
    	bv3 = vld1q_f32(bp + 12);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg, 3);
	cv05 = vfmaq_laneq_f32(cv05, bv5, a_reg, 0);cv15 = vfmaq_laneq_f32(cv15, bv5, a_reg, 1);cv25 = vfmaq_laneq_f32(cv25, bv5, a_reg, 2);cv35 = vfmaq_laneq_f32(cv35, bv5, a_reg, 3);

    	a_reg = vld1q_f32(ap);
    }

    for (; kr < k_len; kr++) {
        a_reg = vld1q_f32(ap);
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        bv2 = vld1q_f32(bp +  8);
        bv3 = vld1q_f32(bp + 12);
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);

	cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg, 0);cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg, 1);cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg, 2);cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg, 3);
	cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg, 0);cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg, 1);cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg, 2);cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg, 3);
	cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg, 0);cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg, 1);cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg, 2);cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg, 0);cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg, 1);cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg, 2);cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg, 3);
	cv04 = vfmaq_laneq_f32(cv04, bv4, a_reg, 0);cv14 = vfmaq_laneq_f32(cv14, bv4, a_reg, 1);cv24 = vfmaq_laneq_f32(cv24, bv4, a_reg, 2);cv34 = vfmaq_laneq_f32(cv34, bv4, a_reg, 3);
	cv05 = vfmaq_laneq_f32(cv05, bv5, a_reg, 0);cv15 = vfmaq_laneq_f32(cv15, bv5, a_reg, 1);cv25 = vfmaq_laneq_f32(cv25, bv5, a_reg, 2);cv35 = vfmaq_laneq_f32(cv35, bv5, a_reg, 3);
        bp += 24;  ap += 4;
    }
/*
    vst1q_f32(&MAT_C(0,  0), cv00); vst1q_f32(&MAT_C(1,  0), cv10); vst1q_f32(&MAT_C(2,  0), cv20); vst1q_f32(&MAT_C(3,  0), cv30);
    vst1q_f32(&MAT_C(0,  4), cv01); vst1q_f32(&MAT_C(1,  4), cv11); vst1q_f32(&MAT_C(2,  4), cv21); vst1q_f32(&MAT_C(3,  4), cv31);
    vst1q_f32(&MAT_C(0,  8), cv02); vst1q_f32(&MAT_C(1,  8), cv12); vst1q_f32(&MAT_C(2,  8), cv22); vst1q_f32(&MAT_C(3,  8), cv32); 
    vst1q_f32(&MAT_C(0, 12), cv03); vst1q_f32(&MAT_C(1, 12), cv13); vst1q_f32(&MAT_C(2, 12), cv23); vst1q_f32(&MAT_C(3, 12), cv33);
    vst1q_f32(&MAT_C(0, 16), cv04); vst1q_f32(&MAT_C(1, 16), cv14); vst1q_f32(&MAT_C(2, 16), cv24); vst1q_f32(&MAT_C(3, 16), cv34);
    vst1q_f32(&MAT_C(0, 20), cv05); vst1q_f32(&MAT_C(1, 20), cv15); vst1q_f32(&MAT_C(2, 20), cv25); vst1q_f32(&MAT_C(3, 20), cv35);
*/
    vst1q_f32(&MAT_C(0,  0), cv00);  vst1q_f32(&MAT_C(0,  4), cv01);  vst1q_f32(&MAT_C(0,  8), cv02);  vst1q_f32(&MAT_C(0, 12), cv03);  vst1q_f32(&MAT_C(0, 16), cv04);  vst1q_f32(&MAT_C(0, 20), cv05);
    vst1q_f32(&MAT_C(1,  0), cv10);  vst1q_f32(&MAT_C(1,  4), cv11);  vst1q_f32(&MAT_C(1,  8), cv12);  vst1q_f32(&MAT_C(1, 12), cv13);  vst1q_f32(&MAT_C(1, 16), cv14);  vst1q_f32(&MAT_C(1, 20), cv15);
    vst1q_f32(&MAT_C(2,  0), cv20);  vst1q_f32(&MAT_C(2,  4), cv21);  vst1q_f32(&MAT_C(2,  8), cv22);  vst1q_f32(&MAT_C(2, 12), cv23);  vst1q_f32(&MAT_C(2, 16), cv24);  vst1q_f32(&MAT_C(2, 20), cv25);
    vst1q_f32(&MAT_C(3,  0), cv30);  vst1q_f32(&MAT_C(3,  4), cv31);  vst1q_f32(&MAT_C(3,  8), cv32);  vst1q_f32(&MAT_C(3, 12), cv33);  vst1q_f32(&MAT_C(3, 16), cv34);  vst1q_f32(&MAT_C(3, 20), cv35);
}


template<int Mr, int Nr>
void register_block_vv_3x4(
    int k_len,
    const float *A,
    const float *B_pack,
    float *C, int ldc) {
    float32x4_t cv00, cv01, cv02, cv03;
    float32x4_t cv10, cv11, cv12, cv13;
    float32x4_t cv20, cv21, cv22, cv23;

    cv00 = vld1q_f32(&MAT_C(0,  0));  cv01 = vld1q_f32(&MAT_C(0,  4));  cv02 = vld1q_f32(&MAT_C(0,  8));  cv03 = vld1q_f32(&MAT_C(0, 12));
    cv10 = vld1q_f32(&MAT_C(1,  0));  cv11 = vld1q_f32(&MAT_C(1,  4));  cv12 = vld1q_f32(&MAT_C(1,  8));  cv13 = vld1q_f32(&MAT_C(1, 12));
    cv20 = vld1q_f32(&MAT_C(2,  0));  cv21 = vld1q_f32(&MAT_C(2,  4));  cv22 = vld1q_f32(&MAT_C(2,  8));  cv23 = vld1q_f32(&MAT_C(2, 12));

    const float *ap = A;
    const float *bp = B_pack;

    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t a_reg0, a_reg1;

    int kr = 0;
    for (; kr + 3 < k_len; kr += 4) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg0 = vld1q_dup_f32(ap + 0); a_reg1 = vld1q_dup_f32(ap + 1);
        cv00 = vfmaq_f32(cv00, a_reg0, bv0);  cv01 = vfmaq_f32(cv01, a_reg0, bv1);  cv02 = vfmaq_f32(cv02, a_reg0, bv2);  cv03 = vfmaq_f32(cv03, a_reg0, bv3);
        cv10 = vfmaq_f32(cv10, a_reg1, bv0);  cv11 = vfmaq_f32(cv11, a_reg1, bv1);  cv12 = vfmaq_f32(cv12, a_reg1, bv2);  cv13 = vfmaq_f32(cv13, a_reg1, bv3);
        a_reg0 = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg0, bv0);  cv21 = vfmaq_f32(cv21, a_reg0, bv1);  cv22 = vfmaq_f32(cv22, a_reg0, bv2);  cv23 = vfmaq_f32(cv23, a_reg0, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg0 = vld1q_dup_f32(ap + 0); a_reg1 = vld1q_dup_f32(ap + 1);
        cv00 = vfmaq_f32(cv00, a_reg0, bv0);  cv01 = vfmaq_f32(cv01, a_reg0, bv1);  cv02 = vfmaq_f32(cv02, a_reg0, bv2);  cv03 = vfmaq_f32(cv03, a_reg0, bv3);
        cv10 = vfmaq_f32(cv10, a_reg1, bv0);  cv11 = vfmaq_f32(cv11, a_reg1, bv1);  cv12 = vfmaq_f32(cv12, a_reg1, bv2);  cv13 = vfmaq_f32(cv13, a_reg1, bv3);
        a_reg0 = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg0, bv0);  cv21 = vfmaq_f32(cv21, a_reg0, bv1);  cv22 = vfmaq_f32(cv22, a_reg0, bv2);  cv23 = vfmaq_f32(cv23, a_reg0, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg0 = vld1q_dup_f32(ap + 0); a_reg1 = vld1q_dup_f32(ap + 1);
        cv00 = vfmaq_f32(cv00, a_reg0, bv0);  cv01 = vfmaq_f32(cv01, a_reg0, bv1);  cv02 = vfmaq_f32(cv02, a_reg0, bv2);  cv03 = vfmaq_f32(cv03, a_reg0, bv3);
        cv10 = vfmaq_f32(cv10, a_reg1, bv0);  cv11 = vfmaq_f32(cv11, a_reg1, bv1);  cv12 = vfmaq_f32(cv12, a_reg1, bv2);  cv13 = vfmaq_f32(cv13, a_reg1, bv3);
        a_reg0 = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg0, bv0);  cv21 = vfmaq_f32(cv21, a_reg0, bv1);  cv22 = vfmaq_f32(cv22, a_reg0, bv2);  cv23 = vfmaq_f32(cv23, a_reg0, bv3);
        bp += 16;  ap += 3;

        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg0 = vld1q_dup_f32(ap + 0); a_reg1 = vld1q_dup_f32(ap + 1);
        cv00 = vfmaq_f32(cv00, a_reg0, bv0);  cv01 = vfmaq_f32(cv01, a_reg0, bv1);  cv02 = vfmaq_f32(cv02, a_reg0, bv2);  cv03 = vfmaq_f32(cv03, a_reg0, bv3);
        cv10 = vfmaq_f32(cv10, a_reg1, bv0);  cv11 = vfmaq_f32(cv11, a_reg1, bv1);  cv12 = vfmaq_f32(cv12, a_reg1, bv2);  cv13 = vfmaq_f32(cv13, a_reg1, bv3);
        a_reg0 = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg0, bv0);  cv21 = vfmaq_f32(cv21, a_reg0, bv1);  cv22 = vfmaq_f32(cv22, a_reg0, bv2);  cv23 = vfmaq_f32(cv23, a_reg0, bv3);
        bp += 16;  ap += 3;
    }

    for (; kr < k_len; kr++) {
        bv0 = vld1q_f32(bp + 0);  bv1 = vld1q_f32(bp + 4);  bv2 = vld1q_f32(bp + 8);  bv3 = vld1q_f32(bp + 12);
        a_reg0 = vld1q_dup_f32(ap + 0); a_reg1 = vld1q_dup_f32(ap + 1);
        cv00 = vfmaq_f32(cv00, a_reg0, bv0);  cv01 = vfmaq_f32(cv01, a_reg0, bv1);  cv02 = vfmaq_f32(cv02, a_reg0, bv2);  cv03 = vfmaq_f32(cv03, a_reg0, bv3);
        cv10 = vfmaq_f32(cv10, a_reg1, bv0);  cv11 = vfmaq_f32(cv11, a_reg1, bv1);  cv12 = vfmaq_f32(cv12, a_reg1, bv2);  cv13 = vfmaq_f32(cv13, a_reg1, bv3);
        a_reg0 = vld1q_dup_f32(ap + 2);
        cv20 = vfmaq_f32(cv20, a_reg0, bv0);  cv21 = vfmaq_f32(cv21, a_reg0, bv1);  cv22 = vfmaq_f32(cv22, a_reg0, bv2);  cv23 = vfmaq_f32(cv23, a_reg0, bv3);
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
    const float *A, int lda,
    const float *B_pack,
    float *C,
    int ldc) {
    switch (op_reg) {
        case 0: register_block_vv_4x4<Mr, Nr>(k_len, A, lda, B_pack, C, ldc); break;
        case 1: register_block_vv_5x4<Mr, Nr>(k_len, A, lda, B_pack, C, ldc); break;
        case 2: register_block_vv_4x5<Mr, Nr>(k_len, A, B_pack, C, ldc); break;
        case 3: register_block_vv_6x4<Mr, Nr>(k_len, A, lda, B_pack, C, ldc); break;
        case 4: register_block_vv_4x6<Mr, Nr>(k_len, A, B_pack, C, ldc); break;
        case 5: register_block_vv_3x4<Mr, Nr>(k_len, A, B_pack, C, ldc); break;
        default: std::printf("错误: 不支持的循环顺序 op=%d\n", op_reg); break;
    }
}

void register_block(
    int Mr,
    int Nr,
    int op_reg,
    int k_len,
    const float *A, int lda,
    const float *B_pack,
    float *C,
    int ldc) {
    if (op_reg != 0) {
        std::printf("错误: 不支持的循环顺序 op=%d\n", op_reg);
        return;
    }
    if (Mr == 4 && Nr == 16) {
        register_block_vv_4x4<4, 16>(k_len, A, lda, B_pack, C, ldc);
        return;
    }
    if (Mr == 5 && Nr == 16) {
        register_block_vv_5x4<5, 16>(k_len, A, lda, B_pack, C, ldc);
        return;
    }
    if (Mr == 4 && Nr == 20) {
        register_block_vv_4x5<4, 20>(k_len, A, B_pack, C, ldc);
        return;
    }
    if (Mr == 6 && Nr == 16) {
        register_block_vv_6x4<6, 16>(k_len, A, lda, B_pack, C, ldc);
        return;
    }
    if (Mr == 4 && Nr == 24) {
        register_block_vv_4x6<4, 24>(k_len, A, B_pack, C, ldc);
        return;
    }
    if (Mr == 3 && Nr == 16) {
        register_block_vv_3x4<3, 16>(k_len, A, B_pack, C, ldc);
        return;
    }
    std::printf("错误: 当前 register 不支持 Mr=%d, Nr=%d\n", Mr, Nr);
}
