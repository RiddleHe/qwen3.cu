# qwen3.cu

A cuda implementation of Qwen3-30B-A3B-Thinking-2507, architecture and inference code

## Inference
```
make
./qwen3_inference path/to/qwen3-30b "What is the capital of Thailand?"
```

## Test

### Environment Test
```
make qwen3_inference
./qwen3_inference test
```

### Compilation Test
```
make clean 
make qwen3_inference
```