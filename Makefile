# voxtral.c — Voxtral Realtime 4B Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# OpenMP (Linux only). Helps a lot for CPU-side attention + elementwise ops.
# Disable with: make <target> OPENMP=0
OPENMP ?= 0

# CUDA toolkit location (override with: make cuda CUDA_HOME=/path/to/cuda)
CUDA_HOME ?= /usr/local/cuda

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Enable OpenMP by default on Linux if available.
ifeq ($(UNAME_S),Linux)
ifneq ($(OPENMP),0)
CFLAGS_BASE += -fopenmp
LDFLAGS += -fopenmp
endif
endif

# Source files
SRCS = voxtral.c voxtral_kernels.c voxtral_cuda.c voxtral_audio.c voxtral_encoder.c voxtral_decoder.c voxtral_tokenizer.c voxtral_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = voxtral

# CUDA kernels (built only for CUDA target). We embed a cubin to avoid relying
# on driver-side PTX JIT compatibility (WSL2 driver/toolkit mismatches can
# break PTX loading).
CUDA_CUBIN = voxtral_cuda_kernels.cubin
CUDA_CUBIN_HDR = voxtral_cuda_kernels_cubin.h
CUDA_ARCH ?= sm_86

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas cuda cuda-check mps inspect test

# Default: show available targets
all: help

help:
	@echo "voxtral.c — Voxtral Realtime 4B - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo "  make cuda     - NVIDIA CUDA + cuBLAS (Linux/WSL2)"
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
	@echo "  make mps      - Apple Silicon with Metal GPU (fastest)"
endif
endif
	@echo ""
	@echo "Other targets:"
	@echo "  make test     - Run regression tests (slow, needs fast GPU)"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make inspect  - Build safetensors weight inspector"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./voxtral -d voxtral-model -i audio.wav"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
BLAS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
BLAS_LDFLAGS = $(LDFLAGS) -framework Accelerate
else
BLAS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
BLAS_LDFLAGS = $(LDFLAGS) -lopenblas
endif
blas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(BLAS_CFLAGS)" LDFLAGS="$(BLAS_LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: cuda (NVIDIA CUDA + cuBLAS)
# =============================================================================
CUDA_CFLAGS = $(CFLAGS_BASE) -DUSE_CUDA -I$(CUDA_HOME)/include -DVOX_CUDA_ARCH=$(CUDA_ARCH)
# Prefer toolkit libs, but allow linking libcuda via toolkit stubs in non-driver CI.
CUDA_LDFLAGS = $(LDFLAGS) -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lcublasLt -lcublas -lcuda

cuda: cuda-check
	@$(MAKE) clean
	@$(MAKE) $(CUDA_CUBIN_HDR)
	@$(MAKE) $(TARGET) CFLAGS="$(CUDA_CFLAGS)" LDFLAGS="$(CUDA_LDFLAGS)"
	@echo ""
	@echo "Built with CUDA backend (cuBLAS)"

cuda-check:
	@test -f "$(CUDA_HOME)/include/cuda.h" || (echo "Error: cuda.h not found under $(CUDA_HOME)/include" && echo "Tip: install CUDA toolkit in WSL2 and/or set CUDA_HOME" && exit 1)
	@test -f "$(CUDA_HOME)/include/cublas_v2.h" || (echo "Error: cublas_v2.h not found under $(CUDA_HOME)/include" && exit 1)
	@test -f "$(CUDA_HOME)/include/cublasLt.h" || (echo "Error: cublasLt.h not found under $(CUDA_HOME)/include" && exit 1)
	@test -x "$(CUDA_HOME)/bin/nvcc" || (echo "Error: nvcc not found under $(CUDA_HOME)/bin" && echo "Tip: install CUDA toolkit (nvcc) and/or set CUDA_HOME" && exit 1)

# =============================================================================
# Backend: mps (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation

mps:
	@$(MAKE) clean
	@$(MAKE) mps-build
	@echo ""
	@echo "Built with MPS backend (Metal GPU acceleration)"

mps-build: $(SRCS:.c=.mps.o) voxtral_metal.o main.mps.o
	$(CC) $(MPS_CFLAGS) -o $(TARGET) $^ $(MPS_LDFLAGS)

%.mps.o: %.c voxtral.h voxtral_kernels.h
	$(CC) $(MPS_CFLAGS) -c -o $@ $<

# Embed Metal shader source as C array (runtime compilation, no Metal toolchain needed)
voxtral_shaders_source.h: voxtral_shaders.metal
	xxd -i $< > $@

voxtral_metal.o: voxtral_metal.m voxtral_metal.h voxtral_shaders_source.h
	$(CC) $(MPS_OBJCFLAGS) -c -o $@ $<

else
mps:
	@echo "Error: MPS backend requires Apple Silicon (arm64)"
	@exit 1
endif
else
mps:
	@echo "Error: MPS backend requires macOS"
	@exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c voxtral.h voxtral_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(DEBUG_CFLAGS)" LDFLAGS="$(LDFLAGS) -fsanitize=address"

# =============================================================================
# Weight inspector utility
# =============================================================================
inspect: CFLAGS = $(CFLAGS_BASE)
inspect: inspect_weights.o voxtral_safetensors.o
	$(CC) $(CFLAGS) -o inspect_weights $^ $(LDFLAGS)

# =============================================================================
# Test
# =============================================================================
test:
	@./runtest.sh

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) *.mps.o voxtral_metal.o main.o inspect_weights.o $(TARGET) inspect_weights
	rm -f voxtral_shaders_source.h
	rm -f $(CUDA_CUBIN) $(CUDA_CUBIN_HDR)

info:
	@echo "Platform: $(UNAME_S) $(UNAME_M)"
	@echo "Compiler: $(CC)"
	@echo ""
	@echo "Available backends for this platform:"
ifeq ($(UNAME_S),Darwin)
	@echo "  blas    - Apple Accelerate"
ifeq ($(UNAME_M),arm64)
	@echo "  mps     - Metal GPU (recommended)"
endif
else
	@echo "  blas    - OpenBLAS (requires libopenblas-dev)"
	@echo "  cuda    - NVIDIA CUDA + cuBLAS"
endif

# =============================================================================
# Dependencies
# =============================================================================
voxtral.o: voxtral.c voxtral.h voxtral_kernels.h voxtral_safetensors.h voxtral_audio.h voxtral_tokenizer.h
voxtral_kernels.o: voxtral_kernels.c voxtral_kernels.h voxtral_cuda.h
voxtral_cuda.o: voxtral_cuda.c voxtral_cuda.h
voxtral_audio.o: voxtral_audio.c voxtral_audio.h
voxtral_encoder.o: voxtral_encoder.c voxtral.h voxtral_kernels.h voxtral_safetensors.h
voxtral_decoder.o: voxtral_decoder.c voxtral.h voxtral_kernels.h voxtral_safetensors.h
voxtral_tokenizer.o: voxtral_tokenizer.c voxtral_tokenizer.h
voxtral_safetensors.o: voxtral_safetensors.c voxtral_safetensors.h
main.o: main.c voxtral.h voxtral_kernels.h
inspect_weights.o: inspect_weights.c voxtral_safetensors.h

# =============================================================================
# CUDA kernels: compile .cu -> CUBIN -> embed as C header
# =============================================================================
$(CUDA_CUBIN): voxtral_cuda_kernels.cu
	$(CUDA_HOME)/bin/nvcc -O3 --std=c++14 -cubin -arch=$(CUDA_ARCH) -lineinfo -o $@ $<

$(CUDA_CUBIN_HDR): $(CUDA_CUBIN)
	xxd -i $< > $@
