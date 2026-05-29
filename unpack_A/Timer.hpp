#pragma once
#include <chrono>
#include <cstdio>
#include <time.h>
#include <iostream>

class GemmTimer {
public:

    static double& get_pack_time() {
        static double time = 0;
        return time;
    }

    static double& get_kernel_time() {
        static double time = 0;
        return time;
    }

    // 记录最近一次 bench 的问题规模，供 report_times 计算 kernel GFLOPS 使用
    static int& get_M() { static int v = 0; return v; }
    static int& get_N() { static int v = 0; return v; }
    static int& get_K() { static int v = 0; return v; }

    template<typename Fn>
    static double bench(const char* name, int m, int n, int k,
                        int run_times, Fn&& fn) {
  
        for(int i = 0; i < 10; i++)
            fn();

        get_pack_time() = 0;
        get_kernel_time() = 0;
        get_M() = m; get_N() = n; get_K() = k;
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);  //start
        for (int i = 0; i < run_times; i++) fn();
        clock_gettime(CLOCK_MONOTONIC, &end);

        double sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        sec /= run_times;
        double gflops = (2.0 * m * n * k) / (sec * 1e9);
        std::printf("Timer: %-16s %8.3f GFLOPS  (%.4f ms)   Efficiency=%8.3f \n", name, gflops, sec * 1e3, gflops / 41.6 * 100);
        return gflops;
    }

    template<typename Fn>
    static void bench_pack(Fn && fn) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        fn();
        clock_gettime(CLOCK_MONOTONIC, &end);
        
        get_pack_time() += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    }

    template<typename Fn>
    static void bench_kernel(Fn && fn) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        fn();
        clock_gettime(CLOCK_MONOTONIC, &end);
        get_kernel_time() += (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    }

    static void report_times(int run_time) {
        double pack_time = get_pack_time() / run_time;
        double kernel_time = get_kernel_time() / run_time;
        int m = get_M(), n = get_N(), k = get_K();
        double kernel_gflops = (kernel_time > 0 && m > 0 && n > 0 && k > 0)
                                   ? (2.0 * m * n * k) / (kernel_time * 1e9)
                                   : 0.0;
        std::printf("  - Pack time:   %8.4f ms\n", pack_time * 1e3);
        std::printf("  - Kernel time: %8.4f ms   %8.3f GFLOPS   Efficiency=%8.3f%%\n \n",
                    kernel_time * 1e3, kernel_gflops, kernel_gflops / 41.6 * 100);
        get_pack_time() = 0;
        get_kernel_time() = 0;
    }
};
