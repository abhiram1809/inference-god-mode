# Quantization and full-context capacity

## Discover weights on the live Hub

Resolve the user's model to an exact `organization/repository` and commit. Search Hub models and files for the base name and actual variants: NVFP4/ModelOpt, AWQ, GPTQ, GGUF, MLX, other 4-bit formats, plus a higher-precision reference. Include EXL2/EXL3 only when a compatible [community loader/build](community-builds.md) exists for an allowed engine. Inspect `config.json`, `quantization_config`, safetensors metadata, tokenizer files, model card, recent commits, gating, license, and trust requirements. Do not treat a filename or search ranking as validation. Record the producing organization and revision. Check whether the engine supports the model architecture *and* the exact quantization method in that release. Use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/package_reference/cli) or [`HfApi.list_models` / `model_info`](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api) when available; a web search alone is not a complete inventory.

Avoid blindly promoting a community checkpoint. Prefer model-author or reputable producer artifacts with reproducible recipes, appropriate calibration, and evidence of quality. Compare a small task-specific eval against the original model, including long-context retrieval, tool calls, structured output, and any reasoning mode the user needs.

For each candidate, record three separate facts: the on-disk encoding, the in-memory representation in the **chosen runtime**, and the kernels actually used for prefill and decode. Quantized weights can remain packed, be unpacked inside a kernel, or be converted to a wider representation. A label such as GGUF, Q4, or NVFP4 alone cannot predict peak memory or native arithmetic. In particular, GGUF size and llama.cpp fit results are runtime-specific; vLLM's GGUF path has its own support and memory behavior. Compare measured load-time allocations, KV cache, compute buffers, and actual resident memory. On Blackwell, inspect the exact GPU architecture and checkpoint's weight/activation quantization scheme before claiming native FP4 tensor-core work.

## Capacity estimate

Four-bit raw *packed* weight storage is approximately `parameters × 0.5 bytes`; scales, zero points, unquantized layers, embeddings, alignment, and runtime conversion can increase resident memory. For MoE, resident weight sizing generally uses **total** parameters, not only active parameters per token; account explicitly for any expert offload. The KV cache is independent of weight quantization. For a conventional full-attention decoder, a rough per-sequence KV estimate is:

`2 × layers × context_tokens × KV_heads × head_dim × bytes_per_KV_element`

Multiply by simultaneous active sequences, allowing for prefix sharing only when the workload and engine actually achieve it. Sliding-window, hybrid, MLA, MoE, recurrent, and multimodal architectures require their actual cache scheme; the simple formula may be wrong. Add runtime/workspace, CUDA graphs, attention buffers, activations during prefill, fragmentation, host staging, and a safety margin. In tensor parallel serving, calculate placement and communication per GPU, not merely total GPU memory.

Check `native_context >= worst_case_input + requested_output + template/system/tool overhead`. Test a request close to that bound and one at peak concurrency. “Full context” refers to the model's native supported window unless the user explicitly accepts extrapolation. Prefer a supported [FP8 KV candidate](fp8-kv-cache.md), measuring actual cache allocation and long-context quality; preserve a BF16/FP16 reference. Quantizing KV does not imply quantized weights or native FP8 attention arithmetic.

## If no suitable quantized checkpoint exists

1. Confirm that the unquantized model and an appropriate quantizer support the architecture and target format. For Blackwell NVFP4, check current NVIDIA ModelOpt / backend workflow; for other NVIDIA 4-bit formats, check current AWQ/GPTQ or llm-compressor paths; for Apple Silicon use MLX conversion/quantization; for llama.cpp use its supported GGUF conversion/quantization path. Do not assume every source precision or architecture is convertible.
2. Choose calibration prompts representative of the real domain, languages, long context, and tool/reasoning use. Reserve a held-out quality set.
3. Ensure disk, host RAM, and GPU memory can hold the source, intermediate, and output; quantify time and cost before running. Preserve original weights and revisions.
4. Export the full artifact with tokenizer, template, quantization metadata, and reproducible recipe. Load it in the target backend and run quality/API/full-context tests.
5. Keep generated artifacts local unless the user asks to publish them and the license permits it.

Primary references: [vLLM quantization](https://docs.vllm.ai/en/stable/features/quantization/index.html), [vLLM GGUF support](https://docs.vllm.ai/en/latest/features/quantization/gguf/), [MLX-LM repository](https://github.com/ml-explore/mlx-lm), [NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer), [llm-compressor](https://github.com/vllm-project/llm-compressor), [llama.cpp quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).
