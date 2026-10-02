# Hardware throughput ceilings and measured efficiency

Read before performance experiments. Calculate an optimistic roofline envelope for the exact hardware, model, quantization, context, batch, and layout; compare benchmark results with that envelope throughout optimization. This is an estimated upper limit under explicit execution/traffic assumptions, not a guarantee that a serving engine can attain it. Keep capacity, quality, API, and latency requirements as independent acceptance gates.

First read [architecture-specific cost models](architecture-cost-models.md) and classify every component/phase from its implementation. Derive work, traffic, state, execution frequency, and output units for that graph. The resource-bound method applies broadly; the decoder weight/KV/operation formulas below apply only to the stated dense autoregressive transformer case. MoE, lookup tables, encoders/cross-attention, recurrent layers, and TTS/iterative generators require their respective component models.

## Establish bandwidth and compute budgets

- Inventory the exact GPU SKU, enabled SMs/compute units and tensor resources, clocks, power limits, memory bandwidth, and topology. Account for partitions, shared devices, throttling, and reserved resources. Obtain advertised peak bandwidth and precision-specific dense compute rates from the device vendor. If deriving compute from core count, use the architecture's supported operations per unit per clock and the relevant clock; distinguish tensor instructions from scalar/vector instructions. State the operation convention, normally two FLOPs per multiply-add. Do not substitute a generic core count for arithmetic throughput.
- Match each operator to its actual arithmetic and accumulation path. W4A16 weights can reduce memory traffic while executing wider arithmetic; do not assign them the GPU's native FP4 rate. Use native FP4/FP8/INT8 rates only for kernels that execute those instructions on this GPU. Use advertised sparse rates only when the model's sparsity and kernel qualify. Include unpacking, scaling, conversion, and non-matmul work where they bind.
- Keep two budgets: **theoretical**, using applicable vendor peaks and idealized traffic; and **calibrated**, using independently measured sustained bandwidth and representative shape/dtype kernel performance under the intended power/clock conditions. A large GEMM or memory-copy microbenchmark is not evidence that batch-one decoding reaches that rate. Label unavailable rates and uncertain inputs; show a range instead of inventing a utilization factor or using the candidate's own tokens/s to define its target.

Use bytes/s and operations/s consistently; state decimal GB/TB versus binary GiB/TiB. Memory capacity determines whether a candidate fits, while bandwidth and operation rates constrain its speed.

## Dense autoregressive decoder example

For one ordinary autoregressive decode step on one GPU, let `B` be the actual active decode batch, `N = B` the emitted target tokens, `D` bytes transferred from/to GPU DRAM, `F` operations executed, `BW` applicable memory bandwidth, and `C` the matched compute rate. Peak outstanding requests can exceed `B`; use the planned batch for screening and observed batch/context distributions for benchmark comparisons.

```text
t_memory = D / BW
t_compute = F / C
t_step_floor = max(t_memory, t_compute)
decode_ceiling_tokens_per_second = N / t_step_floor
arithmetic_intensity = F / D
```

The dominant term predicts a memory or compute bottleneck. These are resource lower bounds on time; launches, dependencies, communication, and CPU/server work can lower achievable throughput further. For mixed arithmetic, model operator-specific work and rates instead of dividing all operations by the largest advertised TFLOPS. Sum lower bounds for sequential kernels/stages that cannot overlap; account for concurrent work using shared-resource limits and the execution critical path.

Estimate `D` from **transferred bytes**, not allocated VRAM or checkpoint file size:

- **Weights:** For a dense decode batch, a useful first approximation is one read of the executed weights per step, shared across `B` tokens. Packed storage starts at `P_executed × weight_bits / 8`; add scales, zero points, wider layers, and any expanded/repeated reads in the actual path. State the ideal batch reuse assumption and model cache residency when weights fit on chip. For MoE, resident capacity uses total experts, but step traffic uses shared weights plus the union of routed experts touched by the batch; active parameters per token alone do not describe batch weight traffic.
- **KV:** For conventional full attention with ideal GQA reuse, approximate decode reads as `sum_i(2 × layers × context_i × KV_heads × head_dim × KV_bytes)` and new-token writes as `B × 2 × layers × KV_heads × head_dim × KV_bytes`. Add scales and layout/kernel overhead. FP8 can reduce this term independently of weight precision. Use actual sliding-window, hybrid, MLA, recurrent, or multimodal cache behavior; capacity formulas are not automatically traffic formulas, and attention kernels may reread data.
- **Other traffic:** Include activations, scratch buffers, conversions, and host/device transfers as applicable. Prefix caching avoids some repeated prefill, but does not automatically eliminate decode attention reads of that prefix.

For dense linear layers, `F_linear ≈ 2 × P_linear_executed × B` is a first-order decode estimate; use the actual executed matrix dimensions for refinement. For conventional attention, add approximately `4 × layers × sum_i(context_i) × query_heads × head_dim` for QK and AV, plus softmax, normalization, sampling, and other operations. GQA reduces KV storage/traffic but does not replace query-head count with KV-head count in this attention-work estimate. For MoE and other architectures, count the actual routed operators and cache scheme.

Model **prefill separately** using uncached prompt tokens, matrix shapes, causal/attention work, weight reuse, and activation/KV traffic. Report input-token throughput and a compute/traffic latency floor; do not reuse the decode weight-bandwidth shortcut for TTFT. For mixed serving, combine phase demands over the actual arrival trace and execution schedule, including queue/client/gateway time. A steady decode ceiling is not a p95/p99 service-latency bound.

## Quantization and layout comparisons

Calculate a row for each feasible weight/KV combination, such as BF16/FP16, FP8/INT8, AWQ/GPTQ/GGUF W4, and native NVFP4 where supported. Record weight traffic, KV traffic at the tested context, executed arithmetic, operation count, memory and compute time bounds, and the predicted limiting resource. Do not assume halving weight bits doubles tokens/s: KV traffic, wider arithmetic, dequantization, or overhead may dominate.

For TP/PP/EP, calculate work, traffic, and limits **per rank**. Model collective/all-to-all payloads, link bandwidth and latency, routing imbalance, pipeline dependencies/bubbles, and supported overlap. Independent DP replicas can contribute aggregate capacity, but do not speed one stream by summing GPU bandwidth. Use resource bounds for aggregate throughput and critical-path bounds for per-stream latency; do not multiply a single-GPU ceiling by GPU count without the layout analysis. CPU/SSD/remote offload adds host, storage, and transport budgets to the model.

For speculation, replace `N = B` with expected **verified target-token advancement** per draft/verify cycle; include draft and target verification work/traffic, accepted lengths, rejections, and extra cache/communication. Count emitted target tokens, not proposed draft tokens. Recalculate the ceiling for this algorithm rather than treating the ordinary one-token-per-step estimate as a universal hardware limit.

## Compare measurements and decide when to stop

Report, per matched workload and configuration:

| Architecture, configuration, workload, and metric unit | Theoretical ceiling | Calibrated estimate/range | Measured throughput | Fraction of theoretical ceiling | Fraction of calibrated estimate | Predicted/profiled bottleneck | Quality/SLO result |
| --- | --- | --- | --- | --- | --- | --- | --- |

Compute each fraction as `measured / matching_estimate`; distinguish per-stream output, aggregate output, input-token, request, and audio-duration metrics. For latency or RTF, where smaller is better, compare `estimated_floor / measured_value` instead of applying the throughput ratio. Preserve the initial assumptions and their sources. Revise estimates only with documented evidence about architecture, phase invocation counts, batch/context, traffic, clocks, arithmetic, or topology; do not lower a target just to make a result pass. Values above the theoretical envelope indicate that its assumptions or metric scope need review, such as cache reuse, speculative advancement, or incorrect byte/operation counts. Do not clamp the fraction to 100%.

Use the gap to choose the next experiment: profile memory stalls and redundant reads for a bandwidth gap; occupancy, shapes, conversion, and instruction dispatch for a compute gap; collectives/imbalance for a communication gap; and launch, scheduler, tokenization, or HTTP gaps when device work is efficient. Match profiling evidence to the predicted limiter before declaring the hardware saturated.

Treat proximity to the ceiling as a measurable optimization objective. Set any target fraction from the user's objective and independent, shape-matched achievable-rate evidence, with repeated-run uncertainty; there is no universal required utilization percentage. When no defensible target exists, report the gap and uncertainty. Stop when requirements pass and remaining gains are within measurement noise or the run budget is exhausted; mark an exhausted run as the best tested result if a material unexplained gap remains. Identify both the closest-to-ceiling configuration and the fastest/SLO-compliant configuration on the same workload and resource budget. A slower configuration can achieve a larger fraction of a smaller ceiling.

### Illustrative calculation

Assume one stream on a hypothetical GPU with `BW = 1 TB/s` and an applicable dense compute rate of `100 TFLOP/s`. An 8B dense model has ideal step weight traffic of `16.2 GB` at 16-bit or `4.2 GB` at packed 4-bit, including assumed metadata/wider layers. At the chosen context, assume total KV read/write traffic is `2 GB` at 16-bit or `1 GB` at FP8, and executed work is `32 GFLOP/step` for all three wider-arithmetic paths (`0.32 ms` compute floor). Ignoring other traffic and overhead gives:

| Weight / KV path | DRAM bytes per step | Memory time floor | Optimistic decode ceiling |
| --- | --- | --- | --- |
| W16 / KV16 | 18.2 GB | 18.2 ms | ~55 tokens/s |
| W4A16 / KV16 | 6.2 GB | 6.2 ms | ~161 tokens/s |
| W4A16 / KV8 | 5.2 GB | 5.2 ms | ~192 tokens/s |

All three are memory-bound in this simplified model. A measured 120 tokens/s on W4A16/KV8 is about 62% of its envelope; 50 tokens/s on W16/KV16 is about 91%. The latter is closer to its ceiling while the former is faster. These are hypothetical arithmetic examples, not a claim about an actual GPU, model, or attainable quality.

Primary references: [NVIDIA GPU performance and arithmetic intensity](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html), [NVIDIA LLM inference optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/). Use the actual device vendor's specifications and the installed engine/kernel documentation for each run; the formulas above apply the resource-bound method to the specified inference workload.
