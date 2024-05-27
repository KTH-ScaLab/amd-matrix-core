CXX=hipcc
CXXFLAGS=-std=c++14 -O2 -lrocblas -Iinclude/

BIN=./bin
SRC=./src

ifdef GEMM_OP
	CXXFLAGS += -D$(GEMM_OP)
endif

ifeq ($(notdir $(shell which nvcc 2> /dev/null)),nvcc)
	# NVIDIA flags
	CXXFLAGS += -arch=native
	GENASMFLAGS=-ptx
else
	# AMD flags
	CXXFLAGS += --offload-arch=gfx90a
	GENASMFLAGS=-S -mllvm --x86-asm-syntax=intel
endif

$(BIN)/gemm_blas: $(SRC)/gemm_blas.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

$(BIN)/gemm_blas.s: $(SRC)/gemm_blas.cpp
	$(CXX) $(CXXFLAGS) $(GENASMFLAGS) -o $@ $? 