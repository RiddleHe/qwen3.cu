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

__global__ void moe_expert_gated_proj_kernel(
    floatX* input, // (bs * seq_len, hidden_size)
    floatX* gate_proj_weights, // (num_experts, hidden_size, moe_intermediate_size)
    floatX* up_proj_weights, // (num_experts, hidden_size, moe_intermediate_size)
    floatX* gated_output, // (bs * seq_len, num_experts_per_tok, moe_intermediate_size)
    int* token_expert_pairs, // (bs * seq_len * num_experts_per_tok, 2) // token_idx, expert_idx
    int num_pairs,
    int hidden_size,
    int moe_intermediate_size,
    int num_experts_per_tok
) {
    int pair_idx = blockIdx.x; // each block processes one token-expert pair
    if (pair_idx >= num_pairs) return;

    int token_idx = token_expert_pairs[pair_idx * 2];
    int expert_idx = token_expert_pairs[pair_idx * 2 + 1]; // global expert idx for this pair (0-127)
    int local_expert_idx = pair_idx % num_experts_per_tok; // local expert idx for this pair (0-7), maintain order as they are sorted by topk

    // for each token-expert pair, transform (hidden_size,) -> (moe_intermediate_size,)
    // each thread processes one dimension in moe_intermediate_size
    // each iteration processes one "moe intermediate dim chunk"
    for (int j = threadIdx.x; j < moe_intermediate_size; j += blockDim.x) { 
        float gate_sum = 0.0f;
        float up_sum = 0.0f;

        for (int i = 0; i < hidden_size; i++) {  // for each token-expert pair, calculate a single dim in moe_intermediate_size
            // input_idx is (token_idx, i) in (num_toknens, hidden_size)
            float input_val = (float)input[token_idx * hidden_size + i];

            // gate idx is [expert_idx, i, j] in (num_experts, hidden_size, moe_intermediate_size)
            int gate_idx = expert_idx * hidden_size * moe_intermediate_size + i * moe_intermediate_size + j;
            gate_sum += input_val * (float)gate_proj_weights[gate_idx];

            // up_idx is also [expert_idx, i, j] in (num_experts, hidden_size, moe_intermediate_size)
            int up_idx = expert_idx * hidden_size * moe_intermediate_size + i * moe_intermediate_size + j;
            up_sum += input_val * (float)up_proj_weights[up_idx];
        }

        float gated = silu_activation(gate_sum) * up_sum; // gate + up

        // out_idx is (token_idx, local_expert_idx, j) in (num_tokens, num_experts_per_tok, moe_intermediate_size)
        int out_idx = token_idx * num_experts_per_tok * moe_intermediate_size + local_expert_idx * moe_intermediate_size + j;
        gated_output[out_idx] = (floatX)gated;
    }
}

__global__ void moe_expert_down_proj_kernel(
    floatX* gated_input, // (bs * seq_len, num_experts_per_tok, moe_intermediate_size)
    floatX* down_proj_weights, // (num_experts, moe_intermediate_size, hidden_size)
    floatX* down_output, // (num_tokens, num_experts_per_tok, hidden_size)
    int* token_expert_pairs, // (num_token * num_experts_per_tok, 2)
    int num_pairs,
    int moe_intermediate_size,
    int hidden_size,
    int num_experts_per_tok
) {
    int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    int token_idx = token_expert_pairs[pair_idx * 2];
    int expert_idx = token_expert_pairs[pair_idx * 2 + 1];
    int local_expert_idx = pair_idx % num_experts_per_tok;

    // for each token-expert pair, transform (moe_intermediate_size,) -> (hidden_size,)
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        float sum = 0.0f;

        for (int i = 0; i < moe_intermediate_size; i++) { 
            // input_idx is [token_idx, local_expert_idx, i])
            int input_idx = token_idx * num_experts_per_tok * moe_intermediate_size + local_expert_idx * moe_intermediate_size + i;
            float input_val = (float)gated_input[input_idx];

            int weight_idx = expert_idx * moe_intermediate_size * hidden_size + i * hidden_size + j;
            sum += input_val * (float)down_proj_weights[weight_idx];
        }
        // out_idx is (token_idx, local_expert_idx, j)
        int out_idx = token_idx * num_experts_per_tok * hidden_size + local_expert_idx * hidden_size + j;
        down_output[out_idx] = (floatX)sum;
    }
}

__global__ void moe_aggregate_kernel(
    floatX* down_output, // (bs * seq_len, num_experts_per_tok, hidden_size)
    floatX* routing_weights, // (bs * seq_len, num_experts_per_tok)
    floatX* final_output, // (bs * seq_len, hidden_size)
    int num_tokens,
    int num_experts_per_tok,
    int hidden_size
) {
    int token_idx = blockIdx.x; // each block processes one token
    // TODO: verify that thread size is larger than hidden_size, if not use chunking
    int element_idx = threadIdx.x; // each thread processes one dim in hidden_size for this token
    
    if (token_idx >= num_tokens || element_idx >= hidden_size) return;

    float sum = 0.0f;
    for (int e = 0; e < num_experts_per_tok; e++) { // iterate over all experts for this token
        float weight = (float)routing_weights[token_idx * num_experts_per_tok + e];
        // expert_idx is (token_idx, e, element_idx)
        int expert_idx = token_idx * num_experts_per_tok * hidden_size + e * hidden_size + element_idx;
        sum += (float)down_output[expert_idx] * weight;
    }
    final_output[token_idx * hidden_size + element_idx] = (floatX)sum;
}

void moe_forward(
    Qwen3Config* config,
    floatX* input, // (batch_size * seq_len, hidden_size)
    floatX* router_weights, // (2048, 128)
    ExpertWeights* experts, // (128, expert weight arrays)
    floatX* output, // (batch_size, seq_len, 2048)
    floatX* workspace,
    cudaStream_t stream
) {
    int num_tokens = config->batch_size * config->seq_len;
    int hidden_size = config->hidden_size;
    int num_experts = config->num_experts;
    int num_experts_per_tok = config->num_experts_per_tok;
    int moe_intermediate_size = config->moe_intermediate_size;

    // Pre-allocate GPU memory for intermediate tensors
    // TODO: understand where the weight loading can go
    floatX* router_logits = workspace;
    int* selected_experts = (int*)(router_logits + num_tokens * num_experts); // add the shape of previous token to point to next token
    floatX* routing_weights_out = (floatX*)(selected_experts + num_tokens * num_experts_per_tok);
    intX* token_expert_pairs = (intX*)(routing_weights_out + num_tokens * num_experts_per_tok);
    floatX* gated_outputs = (floatX*)(token_expert_pairs + num_tokens * num_experts_per_tok * 2);
    floatX* expert_outputs = gated_outputs + num_tokens * num_experts_per_tok * moe_intermediate_size; // now point to expert outputs

    // Compute router logits
    // TODO: is there a specific reason for using cublas here and nowhere else?
    cublasHandle_t handle = cublas_handle;
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_experts, num_tokens, hidden_size,
        &alpha,
        router_weights, CUBLAS_LOWP, hidden_size,
        input, CUBLAS_LOWP, hidden_size,
        &beta,
        router_logits, CUBLAS_LOWP, num_experts,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    // Softmax and topK
    // TODO: is there a specific reason for using 256 here regardless of GPU hardware?
    int threads = 256;
    // TODO: understand this
    int shared_size = (num_experts * sizeof((float) + num_experts * sizeof(int) + 2 * sizeof(float)));
    moe_router_softmax_topk_kernel<<<num_tokens, threads, shared_size, stream>>>(
        router_logits, selected_experts, routing_weights_out,
        num_tokens, num_experts, num_experts_per_tok
    );

    // Create token-expert pairs
    // TODO: do it on GPU only

    // Expert forward pass

    // Aggregate


}

#endif