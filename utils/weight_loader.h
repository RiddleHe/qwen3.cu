/*
Load weight for Qwen3 models using safetensors
*/
#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "../qwen3_model.cuh"

int load_qwen3_config(const char* model_path, Qwen3Config* config);

int load_qwen3_weights(
    const char* model_path,
    Qwen3Config* config,
    Qwen3Weights* weights
);

int allocate_qwen3_weights(Qwen3Config* config, Qwen3Weights* weights);

int load_tensor_from_safetensors(
    const char* safetensors_path,
    const char* tensor_name,
    floatX* destination,
    size_t expected_size
);

void print_30b_config(Qwen3Config* config);

void print_30b_memory_usage(Qwen3Config* config);

void* allocate_gpu_memory(size_t bytes, const char* description);

void free_gpu_memory(void* ptr, const char* description);

#endif