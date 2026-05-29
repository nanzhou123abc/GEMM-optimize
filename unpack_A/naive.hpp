#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#ifndef NAIVE_HPP
#define NAIVE_HPP

void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc);
void check(int M, int N, float *C_ref, int ldc_ref, float *C_opt, int ldc_opt);

#endif