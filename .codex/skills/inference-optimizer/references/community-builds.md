# Model and GPU specific builds

Read when stock support is missing, a quantization loader is external, or profiling shows a supported upstream path underuses the hardware. Search the exact model, quant subtype, GPU SKU/compute capability, host architecture, and engine version. Community images, out-of-tree plugins, and source forks are candidates within the existing engine policy.

## Discover and compare

Check a current stable build first, then official model recipes and upstream merged changes or nightlies. Search model cards/discussions, maintainer repositories, and reproducible community recipes for the remaining gap. A model can be supported while its quantization, attention mode, TP layout, or GPU kernel is not. Distinguish loader support, graph correctness, kernel availability, and proven performance. Benchmark a community build against the best working official candidate where one exists.

Resolve each candidate into:

- Model/checkpoint revision, exact tensor encoding, supported modalities, context, and parallelism layout.
- Image registry and immutable digest, Dockerfile/build recipe, engine/fork commit, patch diff, plugin commit, and native extension/wheel hashes.
- CPU architecture (x86-64 versus aarch64), driver minimum, CUDA/ROCm, PyTorch, and actual attention/GEMM/MoE library versions; check the binaries cover the GPU's compute capability.
- What is added: model implementation, quant loader, architecture-specific kernels, uneven sharding, or a serving adapter. Identify emulation/dequantization and eager fallbacks.
- Maintainer test evidence, open regressions, license, and a way to reproduce and return to the working baseline. A tag containing the GPU name is weak evidence.

Inspect the source and launch recipe before using it. Preserve the baseline environment. Carry forward only settings justified for this machine; a recipe's privileged container, disabled correctness assertions, omitted memory profiling, or extreme memory reservation does not become a default. Changes to attention-head padding, routing, or tensor partitioning require numerical parity checks against a trusted implementation. Report unverified binaries or unavailable source explicitly.

## EXL formats inside vLLM

Search for the **specific** EXL2/EXL3 format and codebook, mixed per-layer bit widths, target model, and compatible plugin revision. An ExLlama-related name does not make EXL2, EXL3, AWQ, and GPTQ interchangeable. Verify packed loader, dense/MoE coverage, extension ABI, supported GPU/TP shapes, graph capture, and prefill/decode kernel dispatch. A plugin registering a quantization method does not supply a missing model architecture. Conversion or repacking creates a distinct artifact; preserve metadata, pin it, and evaluate it again.

Research examples, **not preselected deployments**:

- [vllm-exl3](https://github.com/vcruz305/vllm-exl3) registers an external EXL3 quantization path and documents model/runtime qualification boundaries; some development paths still need GPU qualification.
- [cuda-exl3](https://github.com/Zeuss5/cuda-exl3) supplies specialized EXL3 kernels and a vLLM integration. Check its current GPU, shape, and model limits.
- [MiniMax M3 on three RTX PRO 6000 GPUs](https://github.com/Bitman-Sachs/minimax-m3-tp3-rtx6000) demonstrates a community fork and TP-specific patching; its advertised 240K context does not meet a request for the model's full native window. Recheck the [official MiniMax M3 recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3) before choosing a fork: support may have moved upstream.

## Qualify the selected build

Verify the actual loaded commits, extension ABI, cache format, and launched kernels in the serving workers. Test output/numerical parity, template and parser behavior, all required APIs, modalities if requested, full native context, and target concurrency. Measure startup/graph memory, sustained performance, OOM/errors, and fallbacks using the same workload. Smoke tests and a successful load are preliminary evidence. Pin the winning artifacts and launch command; record regressions and requalify after upgrades.
