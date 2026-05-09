#include<cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <algorithm>
#include <arm_neon.h>
#ifndef REGISTER_HPP
#define REGISTER_HPP

void register_block(
    int Mr,
    int Nr,
    int op_reg,
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C,
    int ldc);

#endif
