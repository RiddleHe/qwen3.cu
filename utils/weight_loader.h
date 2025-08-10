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

void print_config(Qwen3Config* config);

void print_memory_usage(Qwen3Config* config);

#endif