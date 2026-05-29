#!/bin/sh
echo 1020 1024 1024 40 32 256 4 16 0
g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
./gemm 1020 1024 1024 40 32 256 4 16 0

echo 1020 1024 1024 40 32 256 5 16 
g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
./gemm 1020 1024 1024 40 32 256 5 16 0

echo 1026 1024 1024 36 32 256 6 16 0
g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
./gemm 1026 1024 1024 36 32 256 6 16 0
