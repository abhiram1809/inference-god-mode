# Architecture-specific capacity and throughput models

Read before applying capacity or throughput formulas. Identify **what executes, how often, and on which resource** from the exact checkpoint configuration, model-author documentation/source, installed engine implementation, and a representative trace when available. A task name, parameter count, or `model_type` alone is insufficient. Architecture features below are composable: an encoder-decoder can contain MoE layers, lookup modules, and recurrent layers.

## Build a component and phase inventory

For each component, record input/output shapes and lengths, executed operators, resident versus accessed parameters, quantization/arithmetic, placement, persistent state, transient buffers, and invocation frequency: once per request, per token/frame/chunk, per routed assignment, or per refinement evaluation. Mark the evidence and unresolved implementation details. Use the native length/rate limits of each component; text tokens, audio frames, codec tokens, and encoder patches are different units.

| Architecture feature | Cost model to derive | Essential distinction |
| --- | --- | --- |
| Dense autoregressive transformer | Prefill and repeated decode; linear, attention, KV, and output-head work | Dense decoder formulas apply only to qualifying layers/phases. |
| MoE layers | Router/shared work, routed expert compute, accessed expert weights, dispatch/combine | Resident parameters, per-token active parameters, and the batch's expert union differ. |
| Learned n-gram/conditional-memory tables | Addressing, row gathers, cache/memory/transport, fusion operators | Large table capacity does not imply a full table scan or dense matmul per token. |
| Encoder-only or encoder-decoder | Encoding, projections, self-attention, cross-attention, decoder work | One-time encoding and reusable source state differ from repeated target decoding. |
| Partial encoder / partial decoder / multimodal | Each encoder, adapter/resampler, decoder, and output stage | Follow the actual graph: source features may use cross-attention or become decoder-prefix tokens. |
| Recurrent / SSM / hybrid attention | State updates, scans/convolutions, attention only where present | Recurrent state and transformer KV have different shapes and scaling. |
| TTS/audio generation | Text/conditioning, acoustic or codec generation, iterative refinement, vocoder | The task can use autoregressive, parallel, or iterative architectures; audio throughput needs its own units. |

For unsupported/new architectures, derive operator work and memory access from the implementation rather than borrowing a neighboring model's formula. If exact modeling is unavailable, provide a partial bound/range with the missing terms and collect a trace; do not claim a verified maximum. Apply quantization and FP8 KV only to supported components and validate the full pipeline in the chosen engine.

## MoE: three parameter counts and a routing distribution

Calculate resident capacity from all resident shared/expert weights and metadata, accounting for sharding/offload. For each routed layer and batch, let `n_e` be token assignments to expert `e`, `P_e` its executed linear parameters, `P_shared` the executed shared linear parameters, and `B` the number of tokens processed. With multiply-add counted as two operations:

```text
F_linear_step ≈ 2 × B × P_shared + 2 × sum_e(n_e × P_e)
D_weight_step ≈ shared_weight_bytes + sum_{e: n_e > 0}(expert_weight_bytes_e)
```

The traffic approximation assumes one DRAM read of each touched weight with ideal batch reuse and no persistent on-chip weight residency. Refine for real tiling/repeated reads, shared experts, routing/top-k, token drops/padding, expert replication, dequantization, and placement. Add attention, router, dispatch/combine, and other work separately. Compute expert occupancy and the batch union from measured routing or explicit best/worst-case assumptions; never estimate batch traffic merely as active parameters per token times batch size. For EP, use rank-local assignments and link traffic, including imbalance and all-to-all latency.

## Learned n-gram tables and conditional memory

Distinguish a learned lookup module from draft-free n-gram speculation and from prefix KV reuse. Inventory table entries, row width/dtype, hash/order/head structure, accessed row IDs, cache residency, and where the table lives (GPU, host, or another tier). Capacity uses resident rows and metadata; transfer demand uses cache-missing rows and the actual gather granularity. Model address generation, projection/gating/convolution/fusion work, and transport independently.

Use random-access/gather performance at representative locality, row size, and concurrency, not only sequential DRAM bandwidth. Account for misses, access latency, transfer batching, and overlap with backbone execution; an algorithmic O(1) lookup is not constant wall-clock latency or free traffic. Derive an accessed-row byte estimate instead of multiplying all table parameters by two FLOPs per token. Inspect the actual serving implementation: the [official Engram example](https://github.com/deepseek-ai/Engram) demonstrates the lookup data flow but mocks other backbone components, so it does not qualify a complete serving path.

## Encoders, decoders, and mixed graphs

Separate encoder execution and source-state construction from decoder prefill and each decode step. Track source length `S_enc` and target length `S_dec` independently. Count encoder matrix/attention work at its own sequence shapes; retain encoder outputs if needed. A conventional decoder may cache source K/V projections once per request while reading cross-attention state repeatedly. For that implementation, a first-order per-sequence cross-attention cache estimate is `2 × decoder_cross_attention_layers × S_enc × cross_KV_heads × cross_head_dim × cache_bytes`, distinct from target self-attention KV.

For ordinary dot-product cross-attention, QK and AV work per generated target token is approximately `4 × decoder_cross_attention_layers × S_enc × cross_query_heads × cross_head_dim`; add projections, softmax, and other operators. Recompute the term for every generated token, but charge source projection/encoding only at its actual invocation frequency. Check whether the runtime caches, shares, expands for beams, or recomputes source state; do not assume reuse from the architecture name alone. See [encoder-decoder state semantics](https://huggingface.co/docs/transformers/model_doc/encoder-decoder).

For encoder-only models, use requests/s or encoded tokens/s at fixed length, without autoregressive decode or a growing decoder KV term. For partial/multimodal graphs, enumerate modality encoders and preprocessing, projection/resampling, decoder stages, and output heads. If encoder features become decoder-prefix tokens, include their resulting decoder context cost; if they feed cross-attention, model that path. Avoid counting the same representation in both paths unless the implementation uses both.

## Recurrent, hybrid, and iterative components

Count SSM/recurrent state read/write and update work at the implemented state dimensions, including convolution buffers and scans where used. Apply full/sliding-window attention costs only to the layers that implement them; recurrent-only layers do not acquire a context-length-sized transformer KV cache. For iterative diffusion/flow/refinement, count the actual network function evaluations, shapes, solver schedule, conditioning/guidance branches, and reuse. An inference-step flag is not necessarily the number of model calls. These components need implementation-specific operator counts rather than `2 × total_parameters × output_tokens`.

## TTS and speech pipelines

Classify the acoustic/codec generator as autoregressive, parallel, or iterative, and inventory text/conditioning encoders, duration/alignment stages, codebooks, and vocoder/waveform stages. These stages may use different devices and precisions. SpeechT5's [encoder-decoder and vocoder path](https://huggingface.co/docs/transformers/model_doc/speecht5) and [F5-TTS's flow-matching implementation](https://github.com/SWivid/F5-TTS) are examples of different execution graphs, not interchangeable formulas.

For autoregressive audio, determine frames/codec tokens advanced per call, multiple codebooks or delay patterns, chunking, and valid audio duration per output. For parallel generation, count work at the predicted output-frame length; for iterative generation, count all evaluations at their actual lengths plus vocoding. Fix sample/frame rates, output duration, reference-audio workload, chunking, and quality-acceptable solver settings when comparing candidates; changing steps or codecs changes both work and potential quality.

Report **time to first playable audio**, chunk delivery gaps/deadlines, end-to-end latency, generated audio seconds per wall-clock second, and `RTF = processing_seconds / generated_audio_seconds` with explicit per-request or aggregate scope. Smaller RTF is better; its reciprocal is audio seconds/s only for matching scope. Internal codec tokens/s requires a validated duration conversion and is not comparable to text tokens/s. Include all required encoders and the vocoder in the end-to-end measurement; a fast acoustic stage alone does not prove a real-time service.

## Compose the hardware bounds

Apply the [hardware roofline method](hardware-throughput-ceilings.md) to each phase using its own work, traffic, and arithmetic. For a request or output unit, aggregate demands of all stages sharing a resource (GPU compute/DRAM, host memory, interconnect, lookup path). Divide each demand by the corresponding supported resource rate to obtain a resource time floor. The largest aggregate resource floor bounds steady throughput; serial dependencies bound latency via the execution graph's critical path.

Sum dependent stage times that cannot overlap; use overlap only when supported by the actual schedule. Pipeline stages on independent resources can overlap across requests, but stages sharing one GPU cannot each claim that GPU's full bandwidth/compute simultaneously. Distinguish startup/one-time work, first-output latency, sustained throughput, and full-request latency. Normalize stage units before composing the result, retain per-stage bottlenecks, and compare measured efficiency only against the bound for the same graph, shapes, invocation counts, and task metric.

These equations are conditional cost-model derivations, not universal architecture guarantees. Additional primary reference for state-space implementation: [Mamba](https://github.com/state-spaces/mamba). Refresh model-author and engine sources for the exact deployed revisions.
