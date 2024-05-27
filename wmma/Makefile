HIPCC=hipcc
HIPFLAGS=-std=c++14 -O2
INCLUDES=-Iinclude/

BIN=./bin
SRC=./src

ifeq ($(notdir $(shell which nvcc 2> /dev/null)),nvcc)
	# NVIDIA flags
	ARCHFLAGS=-arch=sm_80
	GENASMFLAGS=-ptx
else
	# AMD flags
	ARCHFLAGS=--offload-arch=gfx90a
	GENASMFLAGS=-S -mllvm --x86-asm-syntax=intel
endif

$(BIN)/wmma_gemm: $(SRC)/wmma_gemm.cpp
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(ARCHFLAGS) $(INCLUDES) -o $@ $?

$(BIN)/wmma_gemm.s: $(SRC)/wmma_gemm.cpp
	$(HIPCC) $(HIPFLAGS) $(ARCHFLAGS) $(INCLUDES) $(GENASMFLAGS) -o $@.s $? # Generate assembly

$(BIN)/kernel_overhead: $(SRC)/kernel_overhead.cpp
	$(HIPCC) $(HIPFLAGS) $(CPPFLAGS) $(ARCHFLAGS) $(INCLUDES) -o $@ $?
