# Backend routing

Use this reference when selecting an engine. Recheck the cited upstream docs for the installed version; support is model-, format-, hardware-, and release-specific.

## Legacy GPUs: compatibility before workload defaults

When the GPU falls outside a current engine's hardware requirements or lacks the kernels needed by the chosen model/quantization, shift the first candidate to **llama.cpp** with a compatible GGUF. Age alone is insufficient: inspect the exact GPU, driver, runtime, and backend support. On older NVIDIA hardware, compare a supported llama.cpp CUDA build with Vulkan or CPU/hybrid execution as appropriate; do not assume a current CUDA toolkit still targets that GPU. Preserve the full-context, concurrency, quality, and API requirements while establishing what this hardware can deliver.

For older AMD Radeon/GCN cards, including **Radeon RX 560X**, start by evaluating **llama.cpp with Vulkan**, particularly when ROCm/HIP serving support is absent. If the user says “Ryzen 560X,” confirm the GPU from device inventory; distinguish a Ryzen CPU or integrated GPU from the discrete RX 560X. AMD's [RX 560X specifications](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-600-500-400/radeon-rx-500x-series/radeon-rx-560x.html) identify the card and Vulkan support; they do not guarantee a particular llama.cpp kernel works on the installed driver.

Follow the current [llama.cpp Vulkan build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#vulkan). Inspect `vulkaninfo --summary` and full feature information where needed, select the intended device, and verify server logs show Vulkan execution and the actual layers/buffers offloaded. A software Vulkan device or CPU fallback is not GPU acceleration. Fit weights, KV, and prefill buffers within the usable memory budget, accounting for display use. Tune partial layer offload if necessary, and benchmark GPU/hybrid execution against CPU-only; Vulkan is the preferred compatibility candidate here, not a guaranteed speed winner. Use a supported cache dtype rather than imposing FP8 on a legacy path. If the full native window cannot fit, report the gap and ask which constraint may change.

## NVIDIA

Use these workload defaults after checking the legacy compatibility route.

| Workload | First candidate | Reason to compare another |
| --- | --- | --- |
| Several active requests or high throughput | vLLM | SGLang or TensorRT-LLM may win on a supported model/shape; measure. |
| One interactive request | SGLang | vLLM may have better endpoint coverage, model support, or measured latency. |
| Stable production shape and supported engine build | TensorRT-LLM | Build time, compilation, model support, and API feature coverage may erase the gain. |
| GGUF or model partly offloaded | llama.cpp | Compare native GPU kernels and long-context cost before settling; avoid it as the default for a multi-GPU performance deployment. |

For Blackwell, distinguish GPU SKU and compute capability, genuine NVFP4 checkpoint metadata, and serving kernel support. A model name containing `FP4` does not prove NVFP4. H100/H200 are Hopper: do not assume native NVFP4. On Ampere/Ada/Hopper, test optimized AWQ/GPTQ paths (including Marlin where supported) rather than assuming the generic implementation. FP8 or BF16 may still win when the model fits and quality or speed matters more than 4-bit memory.

If this exact model/quant/GPU path is missing or falls back to slow kernels, investigate [official specialized builds and community images/plugins](community-builds.md). Identify the precise gap and source changes. A100, RTX PRO 6000, B200, and DGX Spark need separate qualification; a Blackwell family name does not establish binary or kernel compatibility.

## Apple Silicon

Use the exact chip and unified-memory budget, allowing for the operating system and other applications. MLX-LM is a straightforward path for an MLX checkpoint and a single user. Its documented server supplies an OpenAI-like Chat Completions route, has limited production safeguards, and 4-bit KV quantization disables batching; verify before choosing it for concurrent service. vLLM-Metal is a vLLM hardware plugin using MLX; confirm that the model, quantization, and required endpoints work on the installed release. llama.cpp is the GGUF alternative, especially when API dialects or partial offload matter. Benchmark actual prompt and decode lengths rather than relying on a generic tokens/s claim.

## Other accelerators and large fleets

On AMD/Intel/other accelerators, inspect the exact backend and model/quantization support in the installed vLLM or SGLang build. Consider llama.cpp where its hardware backend and GGUF path are better supported. Do not transfer NVIDIA kernel, NVFP4, or benchmark assumptions to another vendor. If none of the allowed engines meets the API and performance requirements, report that boundary rather than silently choosing a new engine. For organizational fleet or Kubernetes serving, read [cluster serving](cluster-serving.md): distinguish engine selection, multi-node model parallelism, replica routing, and prefill/decode disaggregation. Evaluate llm-d above a supported engine without expanding the engine set.

## API fit can override performance defaults

As of the referenced vLLM documentation, vLLM lists `/v1/chat/completions`, `/v1/responses`, and Anthropic `/v1/messages`. TensorRT-LLM documents OpenAI Chat and Responses. SGLang and llama.cpp routes evolve; inspect the installed server and test all required calls. Claude Code compatibility requires more than a successful `/v1/messages` text response: tools, streaming events, token counts, reasoning content, and error behavior may matter.

Primary references:

- [vLLM online serving and routes](https://docs.vllm.ai/en/latest/serving/online_serving/)
- [vLLM quantization hardware matrix](https://docs.vllm.ai/en/stable/features/quantization/index.html)
- [SGLang upstream repository](https://github.com/sgl-project/sglang)
- [TensorRT-LLM serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [MLX-LM server](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [vLLM-Metal](https://github.com/vllm-project/vllm-metal)
