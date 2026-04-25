#pragma once
#include <chrono>
#include <cstdio>
#include <time.h>

class GemmTimer {
public:

    template<typename Fn>
    static double bench(const char* name, int m, int n, int k,
                        int run_times, Fn&& fn) {
  
        for(int i = 0; i < 100; i++)
            fn();

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);  //start
        for (int i = 0; i < run_times; i++) fn();
        clock_gettime(CLOCK_MONOTONIC, &end);

        double sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        sec /= run_times;
        double gflops = (2.0 * m * n * k) / (sec * 1e9);
        std::printf("Timer: %-16s %8.3f GFLOPS  (%.4f ms)   %8.3f \n", name, gflops, sec * 1e3, gflops / 102.8 * 100);
        return gflops;
    }
};
