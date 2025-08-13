/*
MoE kernel for qwen3-30B-A3B-Thinking-2507
128 experts, 8 per token
*/

#ifndef MOE_CUH
#define MOE_CUH

#include <assert.h>
#include <float.h>
#include <cub/cub.cuh>

#include "../qwen3_model.cuh"
// import something like llmc/cuda_common.h and cuda_utils.cuh

__device__ __forceinline__ float silu_activation(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_forward_kernel(
    floatX* out,
    const floatX* inp,
    int N
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N) return;

    x128 packed_inp = load128cs(inp+idx);
    x128 packed_out;

    for (int k=0; k<x128::size; ++k) {
        float xi = (float)packed_inp[k];
        packed_out[k] = (floatX)silu_activation(xi);
    }
    store128(out+idx, packed_out);
}

__global__ void moe_router_softmax_topk_kernel(
    floatX* router_logits,  // (bs * seq_len, 128)
    int* selected_experts,  // (bs * seq_len, 8)
    floatX* routing_weights,  // (bs * seq_len, 8)
    int num_tokens,
    int num_experts,
    int num_experts_per_tok
) {
    int token_idx = blockIdx.x;  // each block processes one token 
    if (token_idx >= num_tokens) return;

    // for each block/token, allocate num_experts space for ind/scores in shared memory
    extern __shared__ float shared_mem[]; 
    float* scores = shared_mem;
    int* indices = (int*)&scores[num_experts]; 

    float max_val = -FLT_MAX;
    // traverse across all experts for one token
    // each thread processes one expert
    // each iteration processes one "expert chunk", chunk size is blockDim.x
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) { 
        float val = (float)router_logits[token_idx * num_experts + i]; // get logit for this expert
        scores[i] = val;
        max_val = fmaxf(max_val, val);
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage; // allocate shared memory for the token block
    max_val = BlockReduce(temp_storage).Reduce(max_val, cub::Max()); // reduce max value on all experts for this token

    if (threadIdx.x == 0) {
        // store max value for softmax
        shared_mem[num_experts] = max_val; // will be overwritten by indices[0]
    }
    __syncthreads();
    max_val = shared_mem[num_experts];

    // compute softmax with max_val offset
    float sum = 0.0f;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        float exp_val = expf(scores[i] - max_val);
        scores[i] = exp_val;
        sum += exp_val;
    }

    __syncthreads();
    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        shared_mem[num_experts + 1] = sum; // will be overwritten by indices[1]
    }

    __syncthreads();
    sum = shared_mem[num_experts + 1];
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        scores[i] = scores[i] / sum;
        indices[i] = i;
    }

    __syncthreads();

    // parallel topk
    // TODO: change to cub:BlockRadixSort for efficiency
    if (threadIdx.x == 0) {
        // bubble sort
        for (int i = 0; i < num_experts_per_tok; i++) {
            int max_idx = i;
            float max_score = scores[i];
            for (int j = i + 1; j < num_experts; j++) {
                if (scores[j] > max_score) {
                    max_score = scores[j];
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                float temp_score = scores[i];
                int temp_idx = indices[i];
                scores[i] = scores[max_idx];
                indices[i] = indices[max_idx];
                scores[max_idx] = temp_score;
                indices[max_idx] = temp_idx;
            }
        } // first k indices in indices are now topk
    }

    // Normalize topk and write to selected_experts
    float tpok_sum = 0.0f;
    for (int i = 0; i < num_experts_per_tok; i++) {
        topk_sum += scores[i];
    }

    for (int i = 0; i < num_experts_per_tok; i++) {
        selected_experts[token_idx * num_experts_per_tok + i] = indices[i];
        routing_weights[token_idx * num_experts_per_tok + i] = (floatX)(scores[i] / topk_sum); 
    }

}

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