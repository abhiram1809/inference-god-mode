# Prefer FP8 KV with a verified attention path

Read when selecting KV precision or addressing long-context memory pressure. Prefer FP8 as the optimized candidate where supported, with BF16/FP16 as the quality control. Do not infer KV dtype from the weight checkpoint.

## Evidence and boundary

The [vLLM April 2026 study](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) traces the old Hopper FA3 long-context regression to intermediate accumulation precision. Its fix brought the 128K needle score from 13% to 89%, near the 91% BF16 control. This supports evaluating FP8, rather than rejecting it based on old results; it does not certify every model or attention implementation. Verify the installed attention library includes the relevant fixes. A recent engine version alone does not identify the kernel binary.

FP8 halves the bytes per quantized KV element; total serving memory savings depend on cache layout, scales, unquantized state, weights, and workspace. FP8 storage and native FP8 attention arithmetic are separate capabilities. The validated Hopper FA3 and B200 FlashInfer results do not establish the behavior of A100, RTX PRO 6000, RTX 5090, GB10, Apple, or another backend.

## Selection procedure

1. Inspect the installed backend's [KV quantization documentation](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/), supported dtypes, model cache scheme, scale granularity, and calibration recipe. Pin engine and attention-library versions. For vLLM, `--kv-cache-dtype fp8` is a candidate flag; inspect which FP8 format and attention implementation it selects.
2. Record the actual Q/K/V scale handling. Use available checkpoint scales or representative calibration when needed. Unit scales are a hypothesis to test. Per-head scales require a compatible kernel; inspect current support before exporting them. Calibrate and retest if quality drifts.
3. Compare at multiple context lengths and needle positions, including the full native window with generation headroom. Add multi-needle retrieval or MRCR-style evaluation and the user's real reasoning/tool tasks; one needle test is insufficient. Hold weights, template, sampling, and workload constant. If a BF16 full-window control cannot fit, use a validated reference host or report that verification gap.
4. Benchmark TTFT, decode latency, throughput, and peak memory at one user and target concurrency. Small sliding-window layers can favor BF16; where supported, test `--kv-cache-dtype-skip-layers sliding_window` or selected sensitive layer indices. Large head dimensions can regress prefill even while decode improves. Keep accurate accumulation enabled while tuning.
5. Adopt full FP8 or a measured mixed-layer policy only if quality and service targets pass. Otherwise calibrate, change to a supported attention path, or retain BF16/FP16. Recheck speculation, prefix reuse, and offload connector compatibility with the selected cache layout; old caches may need invalidation.
