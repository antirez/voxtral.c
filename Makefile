# voxtral.c — Voxtral Realtime 4B Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Source files
SRCS = voxtral.c voxtral_kernels.c voxtral_audio.c voxtral_mic.c voxtral_encoder.c voxtral_decoder.c voxtral_tokenizer.c voxtral_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = voxtral

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas mps inspect test

# Default: show available targets
all: help

help:
	@echo "voxtral.c — Voxtral Realtime 4B - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
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
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate -framework AudioToolbox -framework CoreFoundation
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: mps (Apple Silicon Metal GPU)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
ifeq ($(UNAME_M),arm64)
MPS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_METAL -DACCELERATE_NEW_LAPACK
MPS_OBJCFLAGS = $(MPS_CFLAGS) -fobjc-arc
MPS_LDFLAGS = $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Foundation -framework AudioToolbox -framework CoreFoundation

mps: clean mps-build
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
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug: clean $(TARGET)

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
endif

# =============================================================================
# Dependencies
# =============================================================================
voxtral.o: voxtral.c voxtral.h voxtral_kernels.h voxtral_safetensors.h voxtral_audio.h voxtral_tokenizer.h
voxtral_kernels.o: voxtral_kernels.c voxtral_kernels.h
voxtral_audio.o: voxtral_audio.c voxtral_audio.h
voxtral_mic.o: voxtral_mic.c voxtral_mic.h
voxtral_encoder.o: voxtral_encoder.c voxtral.h voxtral_kernels.h voxtral_safetensors.h
voxtral_decoder.o: voxtral_decoder.c voxtral.h voxtral_kernels.h voxtral_safetensors.h
voxtral_tokenizer.o: voxtral_tokenizer.c voxtral_tokenizer.h
voxtral_safetensors.o: voxtral_safetensors.c voxtral_safetensors.h
main.o: main.c voxtral.h voxtral_kernels.h voxtral_audio.h voxtral_mic.h
inspect_weights.o: inspect_weights.c voxtral_safetensors.h
