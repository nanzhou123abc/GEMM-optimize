all:
	g++ -O2 -march=armv8-a+simd+nosve -funroll-loops -std=c++11 main.cpp naive.cpp cache.cpp register.cpp -o gemm
	./gemm 1024 1104 1024 64 48 256 4 16 0

clean:
	rm -f gemm
