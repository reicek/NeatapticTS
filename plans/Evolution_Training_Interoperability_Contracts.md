# Evolution–Training Interoperability Contracts Plan

**Status:** [PLANNED]

## Purpose

Make hybrid workflows (evolution + gradient fine-tuning) predictable and well-typed by defining **clear contracts** between:

- evolved network representations
- trainable parameter vectors
- optimizer/training loops

This prevents “special-case glue” and improves correctness, reproducibility, and docs.

## Goals

- G1: Define a stable way to map a network to/from a parameter vector (“weights + biases”).
- G2: Define training hooks that can be used during evaluation (optional fine-tune) without mutating shared state accidentally.
- G3: Provide deterministic behavior when seed/state handling is enabled.

## Non-goals

- Implementing a full deep learning training framework.
- Supporting automatic differentiation for arbitrary graph operations.

## Key contracts

### Contract A — Parameter vector mapping

A network must be able to:

- export its parameters in a deterministic order
- import parameters back

```ts
export interface ParameterVector {
  values: Float64Array;
  layout: {
    version: 1;
    entries: Array<{
      kind: 'weight' | 'bias';
      from?: number;
      to?: number;
      nodeId?: number;
    }>;
  };
}

export function toParameterVector(network: Network): ParameterVector;
export function fromParameterVector(
  network: Network,
  vector: ParameterVector,
): void;
```

Notes:

- Layout is explicit and versioned.
- Deterministic ordering is required (stable node/edge order).

### Contract B — Trainable view / evaluation isolation

Training during evaluation must not accidentally leak into other candidates.

Proposed model:

- training operates on a cloned network, or
- training operates on a parameter vector and returns a new vector

```ts
export interface FineTuneOptions {
  steps: number;
  learningRate: number;
  seed?: number;
}

export interface FineTuneResult {
  trainedVector: ParameterVector;
  metrics?: Record<string, number>;
}

export async function fineTuneVector(
  vector: ParameterVector,
  dataset: TrainingDataset,
  options: FineTuneOptions,
): Promise<FineTuneResult>;
```

This is compatible with worker evaluation and deterministic checkpointing.

### Contract C — Fitness evaluation semantics

Define an explicit policy for when fine-tuning happens:

- never
- always
- conditional (e.g., only top-K or above threshold)

And whether the trained parameters:

- are thrown away (fitness-only)
- are persisted back into the genome/network (Lamarckian)

## Proposed public API (high-level)

```ts
export interface HybridEvaluationPolicy {
  fineTune: 'never' | 'always' | 'conditional';
  persistTrainedWeights: boolean;
}

export interface EvaluateCandidateOptions {
  policy: HybridEvaluationPolicy;
  fineTuneOptions?: FineTuneOptions;
}

export async function evaluateCandidate(
  network: Network,
  dataset: TrainingDataset,
  options: EvaluateCandidateOptions,
): Promise<{ fitness: number; trainedNetwork?: Network }>;
```

Exact naming should align with existing code; the important part is to make policies explicit.

## Implementation steps

### Step 1 — Stabilize deterministic parameter ordering

- Require stable node/edge ordering.
- Define a canonical “parameter layout v1”.

Acceptance:

- Same network always produces the same layout ordering.

### Step 2 — Implement vector export/import

- Implement `toParameterVector` and `fromParameterVector`.
- Validate vector/layout compatibility.

Acceptance:

- Roundtrip preserves inference outputs.

### Step 3 — Add training isolation helpers

- Provide helpers to clone or to operate on vectors.

Acceptance:

- Fine-tuning cannot mutate shared population state accidentally.

### Step 4 — Define hybrid evaluation policy integration

- Integrate policies into NEAT evaluation loop via clear hooks.

Acceptance:

- Users can choose fitness-only fine-tuning or Lamarckian persistence.

### Step 5 — Docs + examples

- Document:
  - what “Lamarckian” means in this library
  - reproducibility considerations
  - recommended defaults

Acceptance:

- Docs contain a small example and clear warnings.

## Testing strategy

- Parameter mapping tests:
  - roundtrip equality on outputs
- Hybrid evaluation tests:
  - fine-tuning improves loss for a tiny dataset (separate tests)
  - persistence policy changes whether weights are retained

## Risks and mitigations

- Risk: API complexity.
  - Mitigation: keep high-level defaults; expose advanced policies separately.
- Risk: determinism is hard with floating-point + parallelism.
  - Mitigation: clearly document determinism scope; keep ordering fixed.

## Success criteria

- Hybrid workflows are well-defined and reproducible.
- Training integration does not require ad-hoc per-project glue.
- Parameter mapping becomes a later unification seam for export- and worker-adjacent tooling once those Phase 4 foundations exist.
