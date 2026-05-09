#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#ifndef CACHE_HPP
#define CACHE_HPP

void cache(int order, int M, int N, int K,
          int Mc, int Nc, int Kc, int Mr, int Nr,
           float * __restrict__ A, int lda,
           float * __restrict__ B, int ldb,
           float * __restrict__ C, int ldc);



#endif
