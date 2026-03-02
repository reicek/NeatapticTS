# Construct From Parts (Deterministic Graph Assembly) Plan

## Purpose

Provide a deterministic, validated way to build a runnable `Network` from user-defined parts (nodes/groups/layers), with:

- explicit input/output handling
- a stable activation schedule
- clear error messages for ambiguous/invalid graphs

This is the correctness backbone for the primitives API and preconfigured architectures.

## Goals

- G1: Deterministic construction (same parts + seed ⇒ same built network).
- G2: Robust validation (fail fast, descriptive errors).
- G3: Stable activation ordering for acyclic graphs; well-defined handling for recurrence.
- G4: Constructed networks seamlessly work with evolution/training.

## Non-goals

- Replacing NEAT genome operators.
- Fully general static scheduling for arbitrary cyclic graphs without explicit recurrence semantics.

## Key design decisions

### 1) Input/output nodes must be explicit by default

We should not silently guess I/O roles from degree heuristics except behind an explicit opt-in flag.

### 2) Two construction modes

- **Acyclic mode** (default):
  - require that edges can be topologically sorted
  - compute a stable activation order
- **Recurrent mode** (opt-in):
  - allow cycles
  - define recurrence handling policy and state reset policy

### 3) Stable node identity

Construction uses stable node ids (string/number). Internal network indices are derived deterministically.

## API sketch

```ts
export interface ConstructOptions {
  mode?: 'acyclic' | 'recurrent';
  inputNodes?: string[];
  outputNodes?: string[];
  allowIsolatedHiddenNodes?: boolean;
  validate?: {
    forbidDuplicateEdges?: boolean;
    forbidSelfEdges?: boolean;
  };
}

export interface ConstructDiagnostics {
  nodeCount: number;
  edgeCount: number;
  detectedCycles: boolean;
  activationOrder: number[]; // network indices
}

export function constructNetwork(
  parts: Array<GraphNode | NodeGroup | Layer>,
  options?: ConstructOptions,
): { network: Network; diagnostics: ConstructDiagnostics };
```

## Implementation steps

### Step 1 — Canonical graph extraction

- Flatten parts into canonical sets:
  - nodes (unique by id)
  - edges (unique by (fromId,toId,kind))
- Resolve conflicts deterministically:
  - same node id with differing specs ⇒ error (unless exact-equal)

Acceptance:

- A graph can be built from mixed inputs (nodes + groups + layers).

### Step 2 — Validation layer

Implement validations with high-quality errors:

- missing nodes referenced by edges
- duplicate edges (optional)
- self edges (policy-controlled)
- acyclic mode: detect cycles and explain a minimal cycle path
- input/output constraints:
  - at least one input and output
  - inputs cannot have incoming edges (optional rule)
  - outputs cannot have outgoing edges (optional rule)

Acceptance:

- Invalid graphs produce errors that tell users what to change.

### Step 3 — Activation scheduling

**Acyclic mode**

- Determine activation order by stable topological sort.
- Tie-breaker must be deterministic (e.g., by node id).

**Recurrent mode**

- Require explicit `mode:'recurrent'`.
- Define a policy:
  - either use a fixed order that is stable but not a topological order
  - and define how many “passes” per activation call (usually 1)

Acceptance:

- Acyclic graphs run with a proper topo order.
- Recurrent graphs run deterministically given the same initial state.

### Step 4 — Materialize into `Network`

- Create `Network` with correct input/output sizes.
- Map node ids to network indices.
- Add nodes to `Network` with role, bias, activation.
- Create connections with correct weights and enabled flags.

Acceptance:

- Built networks activate and return the expected output length.

### Step 5 — Diagnostics + developer tools

Add optional diagnostics to help users debug:

- export a compact graph JSON
- print a human-readable summary (counts, I/O ids, cycle info)

Acceptance:

- Users can introspect why a graph failed.

## Testing strategy

- Unit tests:
  - stable topo order (same ids ⇒ same order)
  - cycle detection yields a helpful cycle path
  - recurrent mode requires explicit opt-in
  - deterministic mapping from ids → indices
- Property tests (optional):
  - generate DAGs, ensure schedule covers all nodes

## Risks and mitigations

- Risk: scheduling semantics for recurrent networks confuse users.
  - Mitigation: require explicit mode, document state/clear behavior.
- Risk: graph builder becomes a second “Network”.
  - Mitigation: builder only compiles into `Network`; no parallel runtime.

## Success criteria

- Users can assemble networks safely and deterministically.
- Activation order is stable and correct for DAGs.
- Recurrent graphs are supported with explicit, documented semantics.
