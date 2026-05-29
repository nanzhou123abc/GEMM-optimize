#!/bin/sh
echo 1020 1024 1024 40 32 256 4 16 
g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
./gemm 1020 1024 1024 40 32 256 4 16 0

echo 1020 1020 1024 40 32 256 4 20
g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
./gemm 1020 1020 1024 40 40 256 4 20 0

echo 1020 1152 1024 40 32 256 4 24
g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
./gemm 1020 1152 1024 40 48 256 4 24 0
