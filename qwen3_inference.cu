/*
Main entry point for text generation with Qwen3-30B-A3B-Thinking-2507
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "qwen3_model.cuh"
#include "kernels/moe.cuh"
#include "utils/weight_loader.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
        cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

void print_usage(const char* program_name) {
    printf("Usage: %s <model_path> <prompt>\n", program_name);
    printf("\n Example:\n");
    printf("  %s ./models/qwen3-30b-a3b \"What is the capital of Thailand?\"\n", program_name);

}

int main(int argc, char* argv[]) {
    printf("=== Qwen3-30B-A3B-Thinking-2507 in CUDA ===\n");

    if (argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = argv[2];

    printf("Model path: %s\n", model_path);
    printf("Prompt: %s\n", prompt);

    // Gather GPU stats
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device %s\n", prop.name);

    float gpu_memory_gb = prop.totalGlobalmem / (1024.0 * 1024.0 * 1024.0); // GB
    printf("GPU memory: %.1f GB\n", gpu_memory_gb);

    // Load config
    Qwen3Config config;
    // TODO: finish the rest
}