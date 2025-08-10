/*
This models after Qwen3-30B-A3B-Thinking-2507
*/
#ifndef QWEN3_MODEL_CUH
#define QWEN3_MODEL_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef ENABLE_BF16
typedef __nv_bfloat16 floatX;
#else
typedef float floatX
#endif

typedef struct {
    int vocab_size; // 151936
    int hidden_size; // 2048
    int intermediate_size; // 6144
    int moe_intermediate_size; // 768
    int num_hidden_layers; // 48
    int num_attention_heads; // 32
    int num_key_value_heads; // 4, GQA
    int head_dim; // 64 = hidden_size / num_attention_heads

    int num_experts; // 128
    int num_experts_per_tok; // 8
    float router_aux_loss_coef; 

    int max_seq_len;
    int max_batch_size;
} Qwen3Config;

typedef struct {
    floatX* wte; // (vocab_size, hidden_size)

    struct {
        floatX* attn_qkv; 
        // (hidden_size, hidden_size * (num_attention_heads + num_key_value_heads * 2) / num_attention_heads)
        floatX* attn_out; // (hidden_size, hidden_size)
        floatX* attn_norm; // (hidden_size)

        floatX* router; // (hidden_size, num_experts)
        floatX* experts; // (num_experts, expert_weight_arrays)
        floatX* ffn_norm; // (hidden_size)
    } layers[48]; // 48 layers

    floatX* final_norm; // (hidden_size)
    floatX* lm_head; // (hidden_size, vocab_size)
} Qwen3Weights;

typedef struct {
    floatX* gate_proj; // (hidden_size, moe_intermediate_size)
    floatX* up_proj; // (hidden_size, moe_intermediate_size)
    floatX* down_proj; // (moe_intermediate_size, hidden_size)
} ExpertWeights;

void qwen3_forward(
    Qwen3Config* config,
    Qwen3Weights* weights,
    int* input_tokens,  // (batch_size, seq_len)
    floatX* output_logits, // (batch_size, seq_len, vocab_size)
    int batch_size,
    int seq_len
);

void qwen3_init_30b_config(Qwen3Config* config);
void qwen3_print_config(Qwen3Config* config);
void qwen3_free_weights(Qwen3Weights* weights);
size_t qwen3_calculate_memory_usage(Qwen3Config* config);

#endif