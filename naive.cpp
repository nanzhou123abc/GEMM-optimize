#include<iostream>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <arm_neon.h> 
void naive(int m, int n, int k, float *A, float *B, float *C) {
    memset(C, 0, m * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}