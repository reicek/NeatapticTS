# Stable Activation Ordering + Explicit I/O Roles Plan

## Purpose

Make network execution **deterministic, explainable, and architecture-independent** by standardizing:

1. **Explicit input/output roles** (no “mystery nodes” inferred implicitly)
2. **Stable activation ordering** (same graph → same execution schedule)

This is foundational for:

- primitives/graph assembly
- preconfigured architecture builders
- consistent serialization
- worker-friendly evaluation
- reproducible telemetry

## Goals

- G1: Define an explicit representation of I/O node sets for every built network.
- G2: Ensure deterministic activation scheduling for both acyclic and recurrent graphs.
- G3: Provide actionable diagnostics when a graph cannot be scheduled under the chosen mode.

## Non-goals

- Supporting arbitrary dynamic graph mutation at activation time.
- Achieving maximal runtime performance in the first iteration (correctness + determinism first).

## Key concepts

### Concept A — I/O roles as first-class metadata

Every network must have:

- `inputNodeIds: number[]`
- `outputNodeIds: number[]`

Optional (future): named ports and grouped inputs.

Constraints:

- Inputs must be disjoint from outputs.
- The order of `inputNodeIds` defines input vector semantics.
- The order of `outputNodeIds` defines output vector semantics.

### Concept B — Stable activation scheduling

We define an `ActivationSchedule` that captures the order and grouping of node activations.

Modes:

- **Acyclic mode**: schedule via topological sorting; reject cycles.
- **Recurrent mode**: schedule via strongly-connected component (SCC) condensation graph; within SCCs use a deterministic iteration policy.

## Proposed internal API

```ts
export type ActivationMode = 'acyclic' | 'recurrent';

export interface ExplicitIORoles {
  inputNodeIds: number[];
  outputNodeIds: number[];
}

export interface ActivationSchedule {
  mode: ActivationMode;
  steps: ReadonlyArray<ReadonlyArray<number>>;
  outputNodeIds: number[];
}

export interface BuildScheduleOptions {
  mode: ActivationMode;
  stableTieBreak?: 'nodeId' | 'creationIndex';
  maxRecurrentIterations?: number;
}

export function buildActivationSchedule(
  graph: GraphDefinition,
  io: ExplicitIORoles,
  options: BuildScheduleOptions,
): ActivationSchedule;
```

Notes:

- `steps` is a list of “parallelizable groups” (a group may be executed in any order, but we still fix an internal deterministic order).
- Tie-breaking must be deterministic and documented.

## Behavior specification

### Acyclic mode

- Reject cycles with a clear error that includes:
  - the mode
  - a minimal cycle trace (or at least a set of involved node IDs)
  - suggestions: “switch to recurrent mode” or “remove back-connections”

### Recurrent mode

- Build SCCs and schedule SCCs topologically.
- For SCCs of size > 1 or self-loops:
  - execute nodes in stable order for `k` iterations (default `k = 1` for compatibility; configurable)
  - define initial state semantics (e.g., node value reset vs carried state)

### Input injection and output readout

- Input nodes are set before schedule execution.
- Output values are read after schedule completion.

## Implementation steps

### Step 1 — Define I/O role plumbing

- Add an internal structure (or extend `Network`) to store explicit I/O node lists.
- Ensure all builders and graph assembly code fill these lists.

Acceptance:

- All constructed networks have explicit, ordered I/O sets.

### Step 2 — Deterministic acyclic schedule

- Implement topological scheduling with stable tie-breaks.
- Ensure it is consistent across runtimes.

Acceptance:

- Same graph always yields identical schedule output.

### Step 3 — Deterministic recurrent schedule

- Implement SCC condensation scheduling.
- Define iteration policy and defaults.

Acceptance:

- Recurrent graphs run with deterministic ordering and predictable semantics.

### Step 4 — Integrate schedule into activation path

- Ensure the runtime activation uses the computed schedule.
- Provide safe fallbacks for legacy networks (if any) with a deprecation path.

Acceptance:

- Legacy paths continue to work; new paths are the default for constructed architectures.

### Step 5 — Diagnostics and docs

- Add human-friendly errors and doc sections:
  - “What is activation ordering?”
  - “Acyclic vs recurrent mode”
  - “State clearing”

Acceptance:

- Users can understand and fix scheduling errors quickly.

## Testing strategy

- Determinism tests:
  - schedule output is identical for repeated builds
  - tie-break behavior is stable
- Correctness tests:
  - acyclic: known DAG produces correct ordering
  - cyclic: acyclic mode rejects
  - recurrent: SCC graphs run and match expected iteration semantics

## Risks and mitigations

- Risk: behavior changes for existing recurrent networks.
  - Mitigation: keep legacy behavior behind a compatibility mode; document differences.
- Risk: performance overhead.
  - Mitigation: schedule computed once and cached; keep schedule representation compact.

## Success criteria

- All architecture builders produce networks with explicit I/O roles.
- Activation order is deterministic across runs.
- Scheduling failures are actionable and well documented.
