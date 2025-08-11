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

    float gpu_memory_gb = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0); // GB
    printf("GPU memory: %.1f GB\n", gpu_memory_gb);

    // Load config
    Qwen3Config config;
    qwen3_init_30b_config(&config);
    qwen3_print_config(&config);

    // TODO: load weights
    // TODO: allocate memory
    // TODO: set up moe router
    // TODO: running inference
    // TODO: generate text

    printf("\n=== Inference Complete ===\n");
    return 0;
}

void qwen3_init_30b_config(Qwen3Config* config) {
    config->vocab_size = 151936;
    config->hidden_size = 2048;
    config->intermediate_size = 6144;
    config->moe_intermediate_size = 768;
    config->num_hidden_layers = 48;
    config->num_attention_heads = 32;
    config->num_key_value_heads = 4;
    config->head_dim = 64;
    config->num_experts = 128;
    config->num_experts_per_tok = 8;
    config->router_aux_loss_coef = 0.001f;
    config->max_seq_len = 262144;
    config->max_batch_size = 1;
}

void qwen3_print_config(Qwen3Config* config) {
    printf("\n=== Qwen3-30B-A3B-Thinking-2507 Config ===\n");
    printf("Vocabuarly size: %d\n", config->vocab_size);
    printf("Hidden size: %d\n", config->hidden_size);
    printf("Intermediate size: %d\n", config->intermediate_size);
    printf("Num hidden layers: %d\n", config->num_hidden_layers);
    printf("Num attention heads: %d\n", config->num_attention_heads);
    printf("Num experts: %d\n", config->num_experts);
    printf("Num experts per token: %d\n", config->num_experts_per_tok);
    printf("Max seq len: %d\n", config->max_seq_len);
    printf("\n");
}

void qwen3_forward(
    Qwen3Config* config,
    Qwen3Weights* weights,
    int* input_tokens,  // (batch_size, seq_len)
    floatX* output_logits, // (batch_size, seq_len, vocab_size)
    int batch_size,
    int seq_len
) {
    // TODO: implement forward pass
}

void qwen3_free_weights(Qwen3Weights* weights) {
    // TODO: implement
}

void moe_forward(
    Qwen3Config* config,
    floatX* input, 
    floatX* router_weights, 
    ExpertWeights* experts, 
    floatX* output, 
    int batch_size,
    int seq_len
) {
    // TODO: implement
}