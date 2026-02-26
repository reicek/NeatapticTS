# Standalone Inference Export Plan (Dependency-Free Runtime)

## Purpose

Enable exporting a trained/evolved network into a **standalone inference artifact** that:

- has no NeatapticTS runtime dependency
- runs in Node and browsers
- is deterministic and easy to deploy

This is aimed at production deployment and sharing models.

## Goals

- G1: Export a network to a compact, self-contained JS/TS module that exposes `predict(inputs)`.
- G2: Ensure exported inference matches the library runtime within a defined numeric tolerance.
- G3: Support the most common activations and network types first (feed-forward; limited recurrent support).

## Non-goals

- Exporting training/evolution code.
- Achieving maximal minification/compression (we can optimize later).
- Full parity with every exotic network feature in v1.

## Output formats

### Format A — Source module export

- `export function createModel(): { predict(input: number[]): number[]; reset?(): void }`
- Implementation embeds all weights/biases/constants.

Pros: easy to inspect; easy to bundle.
Cons: larger than binary.

### Format B — JSON + tiny runtime

- `model.json` with a minimal runtime `predict(json, input)`.

Pros: good for transport.
Cons: requires shipping the tiny runtime.

Start with Format A (best UX), optionally add Format B.

## Supported subset (v1)

- Node activations: identity, sigmoid, tanh, relu, leakyRelu (whatever is currently in repo).
- Dense connections.
- Deterministic activation schedule (depends on schedule plan).

Optional/restricted:

- Recurrent graphs: only if we define exported state semantics (`reset()` support).

## Export IR (intermediate representation)

Define an internal, versioned IR that is stable across builds:

```ts
export interface InferenceIRv1 {
  version: 1;
  inputNodeIds: number[];
  outputNodeIds: number[];
  activationSteps: number[][];
  nodes: Array<{ id: number; bias: number; activation: string }>;
  edges: Array<{ from: number; to: number; weight: number }>;
}
```

Notes:

- `activationSteps` encodes deterministic ordering and avoids recomputing schedules.
- `activation` is a string key mapped to a runtime function.

## Public API sketch

```ts
export interface ExportStandaloneOptions {
  format: 'esm' | 'cjs' | 'iife';
  name?: string;
  numericPrecision?: 'full' | 'f32';
}

export function exportStandaloneInference(
  network: Network,
  options: ExportStandaloneOptions,
): string;
```

If the project prefers file writing, provide a helper that writes to disk, but keep the core function returning a string.

## Implementation steps

### Step 1 — Build the IR

- Extract I/O sets and deterministic activation schedule.
- Extract nodes (bias, activation) and edges (from, to, weight).
- Validate that the network is exportable (clear error for unsupported features).

Acceptance:

- IR generation is deterministic (same network → same JSON).

### Step 2 — Runtime kernel generator

- Generate a small runtime implementation:
  - allocate arrays for node values
  - inject inputs
  - execute `activationSteps`
  - read outputs

Acceptance:

- Exported module runs and returns outputs.

### Step 3 — Numeric precision option

- Support optional `f32` by storing weights/biases as `Math.fround(...)` constants.

Acceptance:

- `f32` output remains close to full precision for typical networks.

### Step 4 — Minimal recurrent support (optional v1.1)

- If schedule mode is recurrent:
  - define whether node values persist across calls
  - include `reset()` to clear state

Acceptance:

- A simple recurrent example behaves as documented.

### Step 5 — Docs + examples

- Add documentation:
  - what is supported
  - how to export
  - how to use in browser and Node

Acceptance:

- Example is small and copy-pasteable.

## Testing strategy

- Golden equivalence tests:
  - build small networks with fixed weights
  - compare `network.activate(input)` vs exported `predict(input)`
  - use one `expect(...)` per test (split tests by case)
- Snapshot tests:
  - exported code is stable for a fixed IR

## Risks and mitigations

- Risk: schedule/activation semantics drift from runtime.
  - Mitigation: exported runtime uses the same schedule representation; add equivalence tests.
- Risk: users expect ONNX-level portability.
  - Mitigation: position this as a lightweight JS deployment option; ONNX is a separate roadmap.

## Success criteria

- Users can export and run inference in a fresh project without NeatapticTS installed.
- Exported inference matches library inference within tolerance.
- Clear errors for unsupported networks.
