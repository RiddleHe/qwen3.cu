/*
MoE kernel for qwen3-30B-A3B-Thinking-2507
128 experts, 8 per token
*/

#ifndef MOE_CUH
#define MOE_CUH

#include "../qwen3_model.cuh"

__global__ void moe_router_kernel(
    floatX* router_logits, // (batch_size, seq_len, 128)
    int* selected_experts, // (batch_size, seq_len, 8)
    floatX* routing_weights, // (batch_size, seq_len, 8)
    int batch_size,
    int seq_len
);

__global__ void expert_gate_proj_kernel(
    floatX* input, // (tokens, 2048)
    floatX* gate_weights, // (expert_id, 2048, 768)
    floatX* gate_output, // (tokens, 768)
    int* expert_ids, // (tokens,)
    int num_tokens
);

__global__ void expert_up_proj_kernel(
    floatX* input, // (tokens, 2048)
    floatX* up_weights, // (expert_id, 2048, 768)
    floatX* up_output, // (tokens, 768)
    int* expert_ids, // (tokens,)
    int num_tokens
);

__global__ void silu_gate_kernel(
    floatX* gate_output, // all three are (tokens, 768)
    floatX* up_output,
    floatX* gated_output,
    int num_tokens
);

__global__ void expert_down_proj_kernel(
    floatX* gated_input, // (tokens, 768)
    floatX* down_weights, // (expert_id, 768, 2048)
    floatX* down_output, // (tokens, 2048)
    int* expert_ids, // (tokens,)
    int num_tokens
);

__global__ void moe_aggregate_kernel(
    floatX* expert_outputs, // (batch_size, seq_len, 8, 2048)
    floatX* routing_weights, // (batch_size, seq_len, 8)
    floatX* final_output, // (batch_size, seq_len, 2048)
    int batch_size,
    int seq_len
);

void moe_forward(
    Qwen3Config* config,
    floatX* input, // (batch_size, seq_len, 2048)
    floatX* router_weights, // (2048, 128)
    ExpertWeights* experts, // (128, expert weight arrays)
    floatX* output, // (batch_size, seq_len, 2048)
    int batch_size,
    int seq_len
);

#endif