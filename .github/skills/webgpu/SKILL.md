---
name: webgpu
description: 'Use when: designing, implementing, debugging, or validating WebGPU compute acceleration for neural-network inference in NeatapticTS.'
argument-hint: 'Specify the WebGPU topic (lifecycle, WGSL kernel, buffer layout, pipeline, error handling, testing, CPU parity), the target files, the network/demos in scope, and the acceptance criteria (fallback behavior, f32 tolerance, browser support).'
user-invocable: true
disable-model-invocation: false
skills:
  - implementation-standards
  - reproducibility-contracts
  - performance-optimization
  - chrome-devtools-mcp
  - research-methodology
tools:
  - web_fetch
  - neataptic-cortex-mcp-search_corpus
  - neataptic-cortex-mcp-search_context
  - neataptic-cortex-mcp-load_document
  - neataptic-devtools-mcp
  - neataptic-validation-mcp-run_allowlisted_validation
model: kimi-k2.7-code:cloud
compatibility: 'All agents working on GPU inference, compute kernels, or WebGPU integration in NeatapticTS.'
---

# WebGPU Compute Acceleration Playbook for NeatapticTS

Use this skill when an agent task touches WebGPU compute acceleration for neural-network inference in NeatapticTS. It consolidates the API lifecycle, WGSL patterns, buffer mapping, CPU parity rules, fallback strategy, and testing approach needed to keep the GPU path safe, deterministic, and optional.

## When to Use

- Adding or modifying a WebGPU forward-pass kernel for `Network.activate()`.
- Mapping the existing CPU slab/SoA/CSR layout to `GPUBuffer` bindings.
- Deciding whether a network is eligible for GPU inference or must fall back to CPU.
- Debugging `device.lost`, validation errors, or CPU-vs-GPU tolerance failures.
- Writing mocked-device unit tests or browser smoke tests for the GPU path.
- Benchmarking CPU-vs-GPU throughput to pick an enablement threshold.

## When NOT to Use

Do NOT use for general GPU graphics, WebGL, or rendering work. Do NOT use for CPU-only network training, mutation, or evolution work unless the task specifically compares CPU/GPU parity. Do NOT use for CUDA, Vulkan, or native GPU APIs outside the WebGPU browser context.

## Core Promise

WebGPU in NeatapticTS is an **optional, transparent inference fast path**:

- CPU remains the source of truth for topology, training, mutation, and evolution.
- GPU inference is used only when the browser supports WebGPU, the network is eligible, and the workload justifies the overhead.
- The same network + same inputs + same GPU device must produce deterministic outputs within documented `f32` tolerance.
- Any failure path must fall back to the CPU slab or legacy `activate()` seamlessly.

## WebGPU Lifecycle Checklist

Always probe and set up the device defensively:

```text
Flowchart summary: "Secure context?" → "navigator.gpu exists?" → "requestAdapter()" → "requestDevice()" → "handle device.lost" → "create pipeline / buffers" → "fallback on any failure".
```

1. **Secure context** — WebGPU requires HTTPS/localhost or a secure worker. In insecure contexts, fall back immediately.
2. **Adapter request** — `navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })`. If `null`, fallback.
3. **Device request** — `adapter.requestDevice({ requiredLimits, requiredFeatures })`. Catch rejection and fallback.
4. **Device loss** — Subscribe to `device.lost` and route subsequent calls to CPU. Do not retry automatically in a tight loop.
5. **Pipeline compilation** — Prefer `createComputePipelineAsync` to avoid blocking the main thread. Cache pipelines per unique topology/shader.

## Buffer and Binding Contract

The CPU slab (`src/architecture/network/slab/`) stores networks as parallel typed arrays. Map them to WebGPU buffers with the smallest possible binding surface:

| CPU slab array               | GPU buffer role       | WGSL type                  | Notes                                         |
| ---------------------------- | --------------------- | -------------------------- | --------------------------------------------- |
| `_connWeights`               | `weightBuffer`        | `var<storage, read>`       | `f32` per edge.                               |
| `_connTo`                    | `toBuffer`            | `var<storage, read>`       | Destination node per edge.                    |
| `_outStart`                  | `outStartBuffer`      | `var<storage, read>`       | CSR row starts, length `nodeCount + 1`.       |
| `_outOrder`                  | `outOrderBuffer`      | `var<storage, read>`       | Edge permutation for outgoing edges.          |
| Node bias + activation index | `nodeBuffer`          | `var<storage, read>`       | Interleaved or separate; upload once.         |
| `_topoOrder`                 | `topoOrderBuffer`     | `var<storage, read>`       | Topological order indices.                    |
| `activation`                 | `activationBuffer`    | `var<storage, read_write>` | Per-node output cache, reset each tick.       |
| `state`                      | `stateBuffer`         | `var<storage, read_write>` | Per-node accumulator, reset each tick.        |
| Inputs / outputs             | `inputBuffer` / reuse | `var<storage, read>`       | Uploaded each tick or reused from activation. |

### Upload rules

- Structural buffers (topology, weights, CSR) are uploaded once when the slab is rebuilt.
- Weights can be updated incrementally with `device.queue.writeBuffer` if topology is unchanged.
- Inputs are uploaded each tick; outputs are read back only when needed.
- Always check `device.limits.maxStorageBufferBindingSize` and `maxBufferSize` before uploading large networks.

## WGSL Compute Kernel Anatomy

A minimal forward-pass kernel follows this shape:

```wgsl
@group(0) @binding(0) var<storage, read> topoOrder: array<u32>;
@group(0) @binding(1) var<storage, read> nodeBias: array<f32>;
@group(0) @binding(2) var<storage, read> nodeActivationIndex: array<u32>;
@group(0) @binding(3) var<storage, read> outStart: array<u32>;
@group(0) @binding(4) var<storage, read> outOrder: array<u32>;
@group(0) @binding(5) var<storage, read> connTo: array<u32>;
@group(0) @binding(6) var<storage, read> connWeight: array<f32>;
@group(0) @binding(7) var<storage, read_write> state: array<f32>;
@group(0) @binding(8) var<storage, read_write> activation: array<f32>;

@compute @workgroup_size(64)
fn forward(@builtin(global_invocation_id) id: vec3u) {
  let nodeCount = arrayLength(&topoOrder);
  let i = id.x;
  if (i >= nodeCount) { return; }

  let nodeIndex = topoOrder[i];

  if (nodeIndex >= inputCount) {
    let sum = state[nodeIndex] + nodeBias[nodeIndex];
    let act = applyActivation(sum, nodeActivationIndex[nodeIndex]);
    activation[nodeIndex] = act;
    state[nodeIndex] = 0.0;
  }

  let start = outStart[nodeIndex];
  let end = outStart[nodeIndex + 1u];
  let src = activation[nodeIndex];
  for (var e = start; e < end; e = e + 1u) {
    let edge = outOrder[e];
    let dst = connTo[edge];
    let w = connWeight[edge];
    state[dst] = state[dst] + src * w;
  }
}
```

### Kernel design rules

- One thread per node is the simplest correct model for DAG topologies.
- The topological order must guarantee that no two threads write to the same destination concurrently. The CPU slab path already enforces this.
- For deep or wide graphs, prefer sequential layered dispatches over workgroup barriers, because barriers do not span dispatches.
- Keep the activation-function `switch` in WGSL aligned with the CPU `ACTIVATION_FUNCTIONS` registry (`src/multithreading/multi.utils.ts`) by the same numeric index.

## Pipeline and Bind Group Design

- Use explicit `GPUPipelineLayout`/`GPUBindGroupLayout` for kernels that are created frequently; use `layout: 'auto'` only for one-off prototypes.
- Cache the compute pipeline per unique shader and bind group per unique buffer set.
- Command encoding pattern:
  ```text
  createCommandEncoder → beginComputePass → setPipeline → setBindGroup → dispatchWorkgroups → end → finish → queue.submit
  ```
- Batch multiple agent dispatches into one command encoder when each network has its own buffers.
- For identical-topology networks, consider a batched single-buffer dispatch with lane offsets.

## Activation Function Mapping

WGSL supports most built-in activations directly. The GPU path is **ineligible** if any node uses a custom activation not representable in WGSL.

### Direct WGSL implementations

| Activation       | WGSL                          |
| ---------------- | ----------------------------- |
| `identity`       | `x`                           |
| `step`           | `select(0.0, 1.0, x > 0.0)`   |
| `relu`           | `max(0.0, x)`                 |
| `tanh`           | `tanh(x)`                     |
| `logistic`       | `1.0 / (1.0 + exp(-x))`       |
| `softsign`       | `x / (1.0 + abs(x))`          |
| `hardTanh`       | `clamp(x, -1.0, 1.0)`         |
| `absolute`       | `abs(x)`                      |
| `bipolar`        | `select(-1.0, 1.0, x > 0.0)`  |
| `bipolarSigmoid` | `2.0 / (1.0 + exp(-x)) - 1.0` |
| `inverse`        | `1.0 - x`                     |

### Implementable with constants or care

| Activation     | Notes                                               |
| -------------- | --------------------------------------------------- |
| `sinusoid`     | `sin(x)`                                            |
| `gaussian`     | `exp(-x * x)`                                       |
| `bentIdentity` | `sqrt(x*x + 1.0)` and division                      |
| `selu`         | Needs α and λ constants; `exp` for negative branch. |
| `softplus`     | Piecewise approximation to avoid overflow.          |
| `swish`        | `x * logistic(x)`                                   |
| `gelu`         | Approximation with `tanh` and polynomial constants. |
| `mish`         | `x * tanh(softplus(x))`                             |

Pass constants such as SELU/GELU parameters via a small uniform buffer. Custom or closure-dependent activations force CPU fallback.

## CPU Fallback Strategy

The GPU path must be transparent to callers of `network.activate(input)`:

1. **No WebGPU context** → fallback.
2. **No adapter / no device** → fallback.
3. **Device lost** → mark GPU disabled for that network and fallback.
4. **Network ineligible** → fallback. Ineligibility includes:
   - cyclic, gated, recurrent, or non-fast-slab topology.
   - custom activation not in the WGSL registry.
   - `f64` precision requirement.
   - buffer size exceeds `device.limits`.
5. **Runtime tick failure** → fallback for subsequent ticks; log the failure.

Keep the fallback path hot-swappable so a demo continues running even if WebGPU becomes unavailable mid-session.

## Determinism and CPU Parity

- **Same-device replay**: identical inputs on the same GPU device must produce identical outputs.
- **CPU-vs-GPU tolerance**: GPU uses `f32`; CPU defaults to `f64` JavaScript numbers. Expect relative errors around `1e-4` to `1e-5` for typical activations; do not assert bitwise equality.
- **Order of operations**: thread ordering on the GPU differs from the CPU, introducing tiny rounding differences in weighted sums.
- **Transcendentals**: `sin`, `exp`, `tanh` may vary slightly by driver. Use tolerance-based assertions.
- **Benchmark mode**: offer an opt-in `forceCpu` policy for evaluation packs so results are comparable across machines.

## Error Handling

- Wrap dispatches in `device.pushErrorScope('validation')` / `popErrorScope()` during development to catch binding/pipeline mistakes.
- Listen to `device.addEventListener('uncapturederror', ...)` in production to detect async failures.
- Treat validation errors as contagious invalidity: recover by recreating resources or falling back to CPU.
- Do not silently swallow `device.lost`; always emit a telemetry/fallback event.

## Limits and Feature Detection

Key limits to inspect on the device:

- `maxComputeWorkgroupSizeX` / `Y` / `Z`
- `maxComputeInvocationsPerWorkgroup`
- `maxStorageBufferBindingSize`
- `maxBufferSize`
- `maxComputePerDimensionDispatchSize`
- Optional features: `shader-f16`, `subgroups`

For an 8,000-node / 32,000-edge network, static GPU memory is well under 1 MB, so standard desktop limits are sufficient. Mobile or integrated GPUs may need smaller batch sizes or CPU fallback.

## Testing Strategy

### Unit tests

- Mock `navigator.gpu`, `GPUAdapter`, `GPUDevice`, `GPUBuffer`, `GPUComputePipeline`, `GPUCommandEncoder`, and the command queue.
- Test the eligibility predicate, buffer upload sizes, and fallback branches.
- Use the mock to verify WGSL source contains expected bindings and activation functions.

### Browser smoke tests

- Launch a browser test that creates a small network, runs one CPU and one GPU activation, and asserts tolerance.
- Use Chrome DevTools MCP to capture GPU process activity if available.
- Run on the target machine (e.g., RTX 4070 SUPER) to validate real adapter behavior.

### Performance tests

- Compare CPU slab vs GPU inference across network sizes (1k, 2k, 4k, 8k nodes).
- Measure per-tick latency and CPU↔GPU transfer cost.
- Define an empirical threshold above which GPU is enabled by default.

## Real-Device Validation Requirement

Any green-validation of WebGPU code MUST include real GPU measurement on a
visible browser window. Mock GPU tests (mock `GPUAdapter`/`GPUDevice` in Jest)
are a pre-flight check only; they are NOT a green gate for slices that touch
`src/architecture/network/gpu/*`.

Validation must:

1. Run the browser-harness-specialist or Chrome DevTools MCP against a real
   browser page such as `docs/browser-tests/webgpu-nge-tier-benchmark.html`.
2. Use a **visible, non-headless, foreground** browser window; headless or
   minimized measurements are invalid.
3. Record GPU adapter info, CPU reference output, GPU output, and the maximum
   absolute difference between them (or latency/throughput for performance slices).
4. Report `browserVisibility: visible-foreground` in the trace summary.

If real-device validation cannot be performed, the slice is NOT green and must
be routed back to implementation through the orchestrator.

## Performance Basics

- Create pipelines asynchronously with `createComputePipelineAsync`.
- Cache pipelines and bind groups; do not recreate them every tick.
- Minimize `writeBuffer` and `readBuffer` calls; keep intermediate buffers GPU-resident.
- Batch agent dispatches into one command encoder when possible.
- Only enable GPU when the combined agent count and network size justify the fixed overhead.

## NeatapticTS-Specific Integration

- Hook into the existing CPU slab eligibility check (`_canUseFastSlab()`).
- Reuse the worker activation registry numeric indices for the WGSL switch.
- Place the GPU path behind an opt-in or auto-detected flag in the network/worker controller.
- Do not change the CPU `Network.activate()` signature; the GPU path is an internal implementation detail.
- Preserve the existing deterministic seed contract for CPU evaluation; GPU determinism is scoped to the same device.

## References

- `docs/architecture/webgpu/webgpu.docs.md` — synthesized WebGPU master reference.
- `docs/architecture/webgpu/webgpu.architecture.md` — NeatapticTS GPU target architecture and risk register.
- `src/architecture/network/slab/network.slab.utils.ts` — CPU slab layout.
- `src/architecture/network/slab/network.slab.fast-path.helpers.utils.ts` — CPU fast-path eligibility and helpers.
- `src/multithreading/multi.utils.ts` — worker activation registry.
