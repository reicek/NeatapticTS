# Flappy Bird WebGPU Feasibility Research

## Question

Can `examples/flappy_bird` in NeatapticTS use the existing WebGPU inference
fast path for all of its network architectures, especially heavy LSTM+dense
controllers, or does GPU acceleration require structural changes?

## Evidence

### 1. Network creation and activation surfaces in Flappy

- **Creation** happens through the shared builder `examples/architectureProfiles.ts`
  (`buildExampleArchitectureProfileNetwork`, lines 323–375). The Flappy demo
  passes `'flappy-bird'` and resolves one of five profiles: `mlp`,
  `random-sparse`, `narx`, `gru`, `lstm`.
- **Direct activation** is in exactly two hot paths:
  - `examples/flappy_bird/evaluation/rollout/evaluation.rollout.services.ts:391`
    (`resolveRolloutFrameFlapDecision`) calls `network.activate(observation)`.
  - `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.frame.service.ts:168`
    (`resolveBirdControlActions`) calls `bird.network.activate(observation.observationVector)`.
- **Worker / channel inference** uses `openInferenceChannel` and
  `SharedInferenceWorker.infer`, but the payloads are produced from the same
  CPU `Network` objects; no GPU opt-in is wired today.
- **No `{ useGPU: true }` call exists in Flappy.** `Network.activate` only
  dispatches to the GPU path when the caller passes `useGPU: true` and the
  network is eligible (`src/architecture/network/network.ts:1096–1108`).

### 2. GPU eligibility rules

`src/architecture/network/gpu/network.gpu.capability.ts:42–65` (`canUseGPU`)
rejects networks with:

- `network.gates.length > 0`
- `network.selfconns.length > 0`
- any connection where `connection.from === connection.to`
- any node whose squash index is not in the supported WGSL set

The supported activations are indices `0,1,2,3,4,5,9,10,11,12,13`
(`src/architecture/network/gpu/network.gpu.activation.wgsl.ts`), mapping to
logistic, tanh, identity, step, relu, softsign, bipolar, bipolarSigmoid,
hardTanh, absolute, and inverse. Activations such as sinusoid, gaussian,
selu, softplus, swish, gelu, and mish are unsupported.

`isGPUEligible` additionally requires a usable `GPUDevice` and
`network._useFloat32Weights === true`
(`src/architecture/network/gpu/network.gpu.fallback.ts:58–68`).

### 3. Architecture-by-architecture verdict

| Profile | Family | GPU today | Verdict | Blocker |
|---|---|---|---|---|
| `mlp` | feed-forward dense | **Yes** | YES | Must opt in with `useGPU: true` and `gpuDevice` |
| `random-sparse` | feed-forward sparse | **Yes** | YES | Must opt in; mutations could later introduce unsupported squashes or topology |
| `narx` | recurrent delay-line | **No** | NO (silent) | Directed cycle `output → outputMemory → processing → output` is not rejected by `canUseGPU` but cannot be scheduled by the acyclic Kahn-level GPU kernel |
| `gru` | gated recurrent | **No** | NO | `network.gates.length > 0` |
| `lstm` | gated recurrent with cell state | **No** | NO | Gated connections plus memory-cell self-connections; also `network.gates.length > 0` and `network.selfconns.length > 0` |

### 4. Kernel structure

`src/architecture/network/gpu/network.gpu.kernel.ts` generates a single WGSL
compute shader that:

- dispatches one thread per node,
- skips nodes whose topological level does not match the current dispatch,
- gathers incoming weighted activations,
- applies **one** global activation function,
- writes the result back to node state and an output buffer.

There is no support for gate modulation, recurrent-state registers,
self-connections, or multi-iteration recurrent schedules. CPU recurrent
activation uses a compiled schedule with `recurrent-component` steps
(`src/architecture/network/activate/network.activate.schedule.utils.ts:133–167`);
no GPU equivalent exists.

### 5. Reference GPU opt-in pattern

`examples/racing_curriculum/gpu-enabled-racing.example.ts:35–58` shows the
working contract: assign `network.gpuDevice = device`, then call
`await network.activate(observation, { useGPU: true })`. The same contract is
available to Flappy but is not currently used.

## Decision

Only the non-recurrent Flappy profiles (`mlp` and `random-sparse`) can use
WebGPU acceleration with the current implementation. The recurrent profiles
(`narx`, `gru`, `lstm`) cannot. The immediate path to GPU speed-ups for
Flappy is to wire the existing opt-in contract into the Flappy evaluation and
worker playback paths for eligible profiles, not to extend the GPU kernel to
recurrent networks.

## Risks

1. **Silent wrong-output for NARX.** `canUseGPU` does not detect arbitrary
directed cycles. A caller that forces `useGPU: true` on a NARX profile could
get incorrect results instead of a clean fallback. Owner: any implementation
step that adds GPU opt-in must also add an explicit directed-cycle guard or
restrict GPU opt-in to known non-recurrent profiles.
2. **Mutation drift.** NEAT mutation can change activation functions or add
self/gate connections during evolution, making an initially GPU-eligible
population ineligible. Owner: the GPU opt-in site should re-check
eligibility before each dispatch or rely on the existing fallback in
`Network.activate`.
3. **Worker payload mismatch.** Flappy currently exports transferable
inference payloads separately from JSON. Adding GPU state may require
clarifying whether `gpuDevice` is transferred or recreated in workers.
Owner: a worker-payload specialist should validate the seam before the
worker playback path is wired for GPU.
4. **Recurrent GPU is out of scope.** Enabling LSTM/GRU/NARX on GPU would
require a new recurrent GPU schedule/state path, not a small patch.
Owner: a separate planning phase if recurrent GPU inference becomes a goal.
