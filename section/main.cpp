#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#include "Timer.hpp"
#include "naive.hpp"
#include "cache.hpp"
#include "register.hpp"

int main(int argc, char *argv[]) {
    if (argc != 9) {
        printf("用法: %s M N K Mc Nc Kc Mr Nr\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int Mc = atoi(argv[4]);
    int Nc = atoi(argv[5]);
    int Kc = atoi(argv[6]);
    int Mr = atoi(argv[7]);
    int Nr = atoi(argv[8]);
    if (Mr != 4 || Nr != 16) {
        printf("错误: 当前 section 微内核只支持 Mr=%d, Nr=%d\n", 4, 16);
        return 1;
    }
    if (M % Mr != 0 || N % Nr != 0 || Mc % Mr != 0 || Nc % Nr != 0) {
        printf("错误: M/Mc 必须是 %d 的倍数, N/Nc 必须是 %d 的倍数\n", Mr, Nr);
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

    //0:kji
    //1:kij
    //2:ijk
    //3:ikj
    //4:jik
    //5:jki
    GemmTimer::bench("naive", M, N, K, 20, [&](){
        naive(M, N, K, A, lda, B, ldb, C_naive, ldc);
    });
    GemmTimer::bench("opt", M, N, K, 100, [&](){
        cache(0, M, N, K, Mc, Nc, Kc, Mr, Nr, A, lda, B, ldb, C_opt, ldc);
    });

    check(M, N, C_naive, ldc, C_opt, ldc);

    free(A); free(B); free(C_naive); free(C_opt);
    return 0;
}
