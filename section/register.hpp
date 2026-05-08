#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#ifndef REGISTER_HPP
#define REGISTER_HPP
static inline void register_block_4x4(
    int Kc,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc);
#endif