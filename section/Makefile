all:
	g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
	./gemm 1024 1440 1024 64 48 256 4 24 0

clean:
	rm -f gemm
