all:
	g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
	./gemm 1024 1024 1024 64 128 64 4 16

clean:
	rm -f gemm