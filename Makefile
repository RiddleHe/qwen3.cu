# Compiler
NVCC = nvcc
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O3
NVCC_LDFLAGS = -lcublas -lcublasLt
INCLUDES = -I.

# GPU
GPU_COMPUTE_CAPABILITY ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
ifneq ($(GPU_COMPUTE_CAPABILITY),)
	NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# Build dir
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))

# Targets
all: qwen3_inference

qwen3_inference: qwen3_inference.cu qwen3_model.cuh kernels/moe.cuh utils/weight_loader.h
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) qwen3_inference.cu -o qwen3_inference $(NVCC_LDFLAGS)

clean:
	rm -f qwen3_inference
	rm -rf $(BUILD_DIR)

.PHONY: all clean