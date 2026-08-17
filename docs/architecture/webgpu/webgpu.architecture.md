# WebGPU acceleration architecture for NeatapticTS

> Target context: optional, transparent GPU inference fast path for `Network.activate()` on acyclic, non-gated, non-recurrent networks that already qualify for the CPU typed-array slab fast path. Training, mutation, and evolution stay on the CPU.
>
> This document builds on the WebGPU master reference (`webgpu.docs.md`) and the repository's existing CPU slab implementation (`src/architecture/network/slab/`).

## 1. Design goal

Add a GPU inference fast path that:

1. Reuses the existing SoA/CSR slab layout (`_connWeights`, `_connFrom`, `_connTo`, `_outStart`, `_outOrder`, node flags/bias/activation index) without re-serialization.
2. Supports the built-in activation registry used by the worker serialization contract (`src/multithreading/multi.utils.ts`).
3. Falls back to CPU automatically when WebGPU is unavailable, the device is lost, or the network is ineligible.
4. Preserves a deterministic replay contract: same GPU device + same inputs → same outputs, matching CPU results within a documented `f32` tolerance.
5. Is exercised first through the racing-curriculum worker controller, where per-car `network.activate()` is already isolated and inference-heavy.

## 2. Mapping the CPU slab forward pass to a compute pipeline

The CPU fast path (`fastSlabActivate`) works in three phases:

1. **Seed inputs**: copy the input vector into the activation buffer at the input-node indices.
2. **Topological propagation**: for each node in `_topoOrder`:
   - If non-input: `weightedSum = state[node] + bias`; `activation = squash(weightedSum)`; store in activation buffer.
   - Fan out to outgoing edges via CSR adjacency: `state[to] += activation * weight`.
3. **Collect outputs**: read the tail of the activation buffer.

A GPU compute kernel can implement the same logic with a few adjustments:

- The topological order and node metadata are uploaded once as buffers.
- Each tick we upload the input vector.
- One or more compute dispatches run the propagation in topological waves (a node can only be activated after all its incoming contributions have landed).
- The output vector is read back to a staging buffer.

Because the slab graph is a directed acyclic graph (DAG), a single shader can iterate over the topological node list and, for each node, accumulate incoming weighted contributions from a CSR adjacency structure. For dense or shallow graphs, one dispatch covering all nodes with a guarded workgroup-size loop is sufficient. For very deep graphs, multiple dispatches with barriers are unnecessary because the workgroup barrier does not span dispatches; instead, we can split the topology into layers and dispatch each layer sequentially, or simply run the whole forward pass in one dispatch with enough threads per node to avoid divergence.

For NeatapticTS, the simplest viable kernel is a **single compute pass with one thread per node**:

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

  // Skip input-layer activation (bias/activation not defined for inputs).
  if (nodeIndex >= inputCount) {
    let sum = state[nodeIndex] + nodeBias[nodeIndex];
    let act = applyActivation(sum, nodeActivationIndex[nodeIndex]);
    activation[nodeIndex] = act;
    state[nodeIndex] = 0.0; // reset for next tick
  }

  // Fan out. Each thread writes to destination states directly.
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

Caveat: because all threads write to the same `state` array concurrently, the simple kernel relies on the topological order guaranteeing no two threads target the same destination node at the same time. This is true for a DAG when each node has exactly one topological slot and edges go from earlier to later nodes. The CPU slab path already enforces this.

## 3. SoA/CSR slab layout on the GPU

The CPU slab stores the network as parallel typed arrays:

| CPU slab array                     | GPU buffer role               | WGSL binding type                       | Notes                                          |
| ---------------------------------- | ----------------------------- | --------------------------------------- | ---------------------------------------------- |
| `_connWeights`                     | `weightBuffer`                | `storage, read`                         | One `f32` per edge.                            |
| `_connFrom`                        | (optional debug)              | —                                       | Not needed at runtime; CSR is source-oriented. |
| `_connTo`                          | `toBuffer`                    | `storage, read`                         | Destination node per edge.                     |
| `_connGain`                        | `gainBuffer`                  | `storage, read`                         | Optional; if all gains are neutral, omit.      |
| `_outStart`                        | `outStartBuffer`              | `storage, read`                         | CSR row starts, length `nodeCount + 1`.        |
| `_outOrder`                        | `outOrderBuffer`              | `storage, read`                         | Edge permutation for outgoing edges.           |
| Node bias, activation index, flags | `nodeBuffer`                  | `storage, read`                         | Interleaved or separate.                       |
| `_topoOrder`                       | `topoOrderBuffer`             | `storage, read`                         | Node indices in topological order.             |
| Working `activation`               | `activationBuffer`            | `storage, read_write`                   | One `f32` per node; seeded from inputs.        |
| Working `state`                    | `stateBuffer`                 | `storage, read_write`                   | One `f32` per node; reset each tick.           |
| Inputs / outputs                   | `inputBuffer`, `outputBuffer` | `storage, read` / `storage, read_write` | Or reuse the activation buffer.                |

Buffer sizes for a network with `N` nodes and `E` edges:

- `weightBuffer`, `toBuffer`, `gainBuffer`: `E * 4` bytes each (`f32`).
- `outStartBuffer`: `(N + 1) * 4` bytes (`u32`).
- `outOrderBuffer`: `E * 4` bytes (`u32`).
- `nodeBuffer`: `N * 8` bytes if bias (`f32`) + activation index (`u32`) + padding.
- `activationBuffer`, `stateBuffer`: `N * 4` bytes each (`f32`).

Total static GPU memory for a large network (8k nodes, 32k edges):

- Edge data: `32k * 4 * 3 ≈ 384 KB` (weights, to, order).
- Node data + buffers: `8k * 4 * 5 ≈ 160 KB`.
- Total: well under typical `maxStorageBufferBindingSize` and `maxBufferSize` limits.

Upload strategy:

- Structural buffers (topology, weights, CSR) are uploaded once when the slab is rebuilt.
- Weights can be updated incrementally with `device.queue.writeBuffer` if the topology is unchanged.
- Input vectors are uploaded each tick.
- Outputs are either read back every tick or kept GPU-resident for further GPU-side evaluation.

## 4. Activation functions: WGSL support vs CPU fallback

The worker-compatible activation registry (`ACTIVATION_FUNCTIONS` in `src/multithreading/multi.utils.ts`) encodes each node's activation by a stable numeric index. The WGSL kernel uses the same ordered `switch`.

### 4.1 Straightforward in WGSL

These map directly to WGSL built-ins or simple arithmetic:

| Activation           | WGSL implementation           |
| -------------------- | ----------------------------- |
| `identity`           | `x`                           |
| `step`               | `select(0.0, 1.0, x > 0.0)`   |
| `relu`               | `max(0.0, x)`                 |
| `tanh`               | `tanh(x)` (built-in)          |
| `logistic` (sigmoid) | `1.0 / (1.0 + exp(-x))`       |
| `softsign`           | `x / (1.0 + abs(x))`          |
| `hardTanh`           | `clamp(x, -1.0, 1.0)`         |
| `absolute`           | `abs(x)`                      |
| `bipolar`            | `select(-1.0, 1.0, x > 0.0)`  |
| `bipolarSigmoid`     | `2.0 / (1.0 + exp(-x)) - 1.0` |
| `inverse`            | `1.0 - x`                     |

### 4.2 Implementable but need constants or numerical care

These can be written in WGSL but require passing constants (e.g., via a uniform buffer) or piecewise approximations:

| Activation     | Notes                                                         |
| -------------- | ------------------------------------------------------------- |
| `sinusoid`     | `sin(x)` built-in; trivial.                                   |
| `gaussian`     | `exp(-x*x)`.                                                  |
| `bentIdentity` | `sqrt(x*x + 1.0)` and division.                               |
| `selu`         | Needs α and λ constants; `exp` for negative branch.           |
| `softplus`     | Piecewise approximation to avoid overflow; needs `exp`/`log`. |
| `swish`        | `x * logistic(x)`.                                            |
| `gelu`         | Approximation with `tanh` and polynomial constants.           |
| `mish`         | `x * tanh(softplus(x))`; depends on `softplus`.               |

For the first implementation slice, we support all functions by inlining the formulas and passing the SELU/GELU constants as a small uniform buffer. If a network uses a custom activation not in the registry, the GPU path is ineligible and falls back to CPU.

### 4.3 CPU fallback rule

The GPU path is only used when **every node in the network** uses an activation function present in the WGSL registry. The registry starts as the 19 built-in activations and can be extended. Fallback triggers for:

- Custom activation functions.
- Runtime-configurable activations that depend on closure state not representable in WGSL.
- Any precision mode that requires `f64` (WGSL has no `f64`).

## 5. Batch inference across agents / demes

The racing curriculum and future Pac-Man-like demos need to evaluate many networks each tick. Two batching strategies:

### 5.1 One network per dispatch

- Each agent's network has its own static buffers.
- Encode one compute pass per agent into the same command encoder.
- Submit once per tick. Overhead: `O(agentCount)` compute passes and dispatches.

Best when networks differ in topology/weights and cannot share buffers.

### 5.2 Batched single-buffer dispatch

- Networks with identical topology are packed into one SoA buffer set, with an offset/lane per network.
- One dispatch handles all lanes in parallel.
- Best for demes or clones that share structure but have different weights.

For the racing-curriculum first pass, **one network per dispatch** is simpler and sufficient. A future optimization can batch identical-topology agents.

## 6. CPU fallback strategy

WebGPU availability is not guaranteed. The GPU path must be transparent:

1. **Probe at startup**: `if (!navigator.gpu) fallback`.
2. **Request adapter**: if `requestAdapter()` returns `null`, fallback.
3. **Request device**: if `requestDevice()` fails or `device.lost` fires later, fallback.
4. **Network eligibility**: reuse `_canUseFastSlab()` predicates; add WebGPU-specific checks:
   - network uses only supported activations.
   - network precision is `f32` (or f64 is accepted with tolerance).
   - buffer sizes fit within `device.limits`.
5. **Runtime fallback**: if a tick fails (lost device, validation error, OOM), mark the GPU path disabled for that network and route subsequent calls through `fastSlabActivate` or legacy `activate()`.

The fallback must be **seamless** — callers continue to use `network.activate(input)` and the runtime decides which path to take.

## 7. Determinism considerations

- **Same-device replay**: a given network, given identical inputs, must produce identical outputs on the same GPU device and driver. This is the contract used by deterministic evaluation packs.
- **CPU vs GPU tolerance**: CPU uses JavaScript `number` (`f64` by default, but can be configured to `f32`). GPU uses `f32`. Expected tolerance for large networks is on the order of `1e-4` to `1e-5` relative error for typical activations.
- **Order of operations**: weighted sums on GPU are performed in a different thread order than CPU; this introduces tiny `f32` rounding differences. The application must not expect bitwise equality.
- **Avoid non-deterministic activations**: step/bipolar are deterministic but discontinuous; relu is exact. Transcendental functions (`sin`, `exp`, `tanh`) may vary slightly by driver.
- **Replay guard**: for benchmark evaluation packs, offer an opt-in `forceCpu` policy so results are comparable across machines.

## 8. Risk register

| Risk                                              | Likelihood | Impact | Mitigation                                                                    |
| ------------------------------------------------- | ---------- | ------ | ----------------------------------------------------------------------------- |
| WebGPU unavailable in target browser              | Medium     | High   | Transparent CPU fallback.                                                     |
| Device loss during long-running demo              | Low-Medium | High   | Listen to `device.lost`; recreate or fall back.                               |
| `f32` vs `f64` mismatch breaks tests              | Medium     | Medium | Document tolerance; add parity tests.                                         |
| Buffer size exceeds `maxStorageBufferBindingSize` | Low        | High   | Split large tensors or fall back to CPU.                                      |
| Pipeline compilation stalls main thread           | Medium     | Medium | Use `createComputePipelineAsync`; cache pipeline per topology.                |
| Transfer cost dominates for small networks        | High       | Medium | Only enable GPU when network and batch are large enough; benchmark threshold. |
| Driver-specific `tanh`/`exp` differences          | Low        | Low    | Tolerance-based assertions, not bitwise equality.                             |
| Worker context lacks WebGPU                       | Low        | High   | Probe `WorkerNavigator.gpu`; fall back to CPU worker path.                    |
| WGSL activation switch grows large                | Low        | Low    | Compile one kernel per activation subset if needed.                           |

## 9. Skill recommendations

A WebGPU skill for agents working on NeatapticTS should cover the following topics. A dedicated specialist sub-agent would be useful once the GPU path matures beyond the first slice.

### 9.1 Topics a WebGPU skill should cover

1. **Lifecycle checklist**: `navigator.gpu` → `requestAdapter` → `requestDevice` → `device.lost`.
2. **Buffer contract**: `GPUBufferUsage`, `writeBuffer`, `mapAsync`, `getMappedRange`, `unmap`.
3. **WGSL compute kernel anatomy**: `@compute`, `@workgroup_size`, `@group`/`@binding`, storage buffers, builtins.
4. **Bind group and pipeline layout design**: explicit vs `layout: 'auto'`, compatibility, multiple `@group`s.
5. **Command encoding pattern**: `createCommandEncoder` → `beginComputePass` → `setPipeline`/`setBindGroup`/`dispatchWorkgroups` → `end` → `finish` → `queue.submit`.
6. **Error handling**: error scopes, `uncapturederror`, contagious invalidity.
7. **Limits and feature detection**: `GPUSupportedLimits`, `maxComputeWorkgroupSize*`, `maxStorageBufferBindingSize`, `shader-f16`, `subgroups`.
8. **Performance basics**: async pipeline creation, batching dispatches, minimizing CPU↔GPU transfers, keeping buffers GPU-resident.
9. **Determinism and precision**: `f32` semantics, order-of-operation differences, tolerance-based validation.
10. **Testing strategies**: mocked device unit tests, browser smoke tests, Chrome DevTools MCP trace/memory checks.

### 9.2 Should we create a specialist sub-agent?

Yes, once the first GPU slice is merged. A `webgpu-specialist` or `browser-runtime-scout` sub-agent would:

- Diagnose `device.lost` and adapter request failures.
- Review WGSL kernels for correctness, binding limits, and dispatch size compliance.
- Compare CPU vs GPU traces and tolerance regressions.
- Maintain the activation-function WGSL registry as the CPU registry evolves.

Until then, the research briefs in `docs/architecture/webgpu/` plus the `browser-runtime-scout` and `browser-ui-specialist` agents are sufficient for the feasibility phase.
