CXX      = g++
CXXFLAGS = -O2 -march=native -funroll-loops -ffast-math -std=c++11
SIZE     = 1024 1024 1024

SRCS = ipj.cpp cache.cpp cache_pack.cpp cache_pack_unroll.cpp \
       register_block.cpp register_neon.cpp register_neon_unroll.cpp

BINS = $(SRCS:.cpp=)

all: $(BINS)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test: $(BINS)
	@for bin in $(BINS); do \
		echo "========== $$bin =========="; \
		./$$bin $(SIZE); \
		echo ""; \
	done

clean:
	rm -f $(BINS)
