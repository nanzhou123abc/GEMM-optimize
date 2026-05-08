#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#ifndef CACHE_HPP
#define CACHE_HPP



void cache_kji(int M, int N, int K, 
               int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc);


void cache_kij(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc);

void cache_ijk(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc);

void cache_ikj(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc);

void cache_jik(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc);

void cache_jki(int M, int N, int K,
             int Mc, int Nc, int Kc,
               float * __restrict__ A, int lda,
               float * __restrict__ B, int ldb,
               float * __restrict__ C, int ldc);

void cache(int order, int M, int N, int K,
           int Mc, int Nc, int Kc,
           float * __restrict__ A, int lda,
           float * __restrict__ B, int ldb,
           float * __restrict__ C, int ldc);


// void cache (int op,int M, int N, int K,
//              int Mc, int Nc, int Kc,
//                float * __restrict__ A, int lda,
//                float * __restrict__ B, int ldb,
//                float * __restrict__ C, int ldc) {
    
//     switch (op)
//     {
//     case 0:
//         void cache_kji(int M, int N, int K,
//              int Mc, int Nc, int Kc,
//                float * __restrict__ A, int lda,
//                float * __restrict__ B, int ldb,
//                float * __restrict__ C, int ldc);
//         break;

//         case 1:
//             void cache_kij(int M, int N, int K,
//                 int Mc, int Nc, int Kc,
//                 float * __restrict__ A, int lda,
//                 float * __restrict__ B, int ldb,
//                 float * __restrict__ C, int ldc);
//         break;
        
//         case 2:

//             void cache_ijk(int M, int N, int K,
//                         int Mc, int Nc, int Kc,
//                         float * __restrict__ A, int lda,
//                         float * __restrict__ B, int ldb,
//                         float * __restrict__ C, int ldc);
//             break;
//         case 3:
//             void cache_ikj(int M, int N, int K,
//                         int Mc, int Nc, int Kc,
//                         float * __restrict__ A, int lda,
//                         float * __restrict__ B, int ldb,
//                         float * __restrict__ C, int ldc);
//             break;
//         case 4:
//             void cache_jik(int M, int N, int K,
//                         int Mc, int Nc, int Kc,
//                         float * __restrict__ A, int lda,
//                         float * __restrict__ B, int ldb,
//                         float * __restrict__ C, int ldc);
//             break;
        
//         case 5:
//             void cache_jki(int M, int N, int K,
//                         int Mc, int Nc, int Kc,
//                         float * __restrict__ A, int lda,
//                         float * __restrict__ B, int ldb,
//                         float * __restrict__ C, int ldc);
//         break;
    
//     default:
//         break;
//     }
// }
#endif