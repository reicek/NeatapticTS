# Architecture Primitives (Node / Group / Layer) Plan

## Purpose

Introduce a **first-class architecture-building API** that lets users construct networks from composable primitives (nodes, groups, layers), while staying compatible with:

- Evolution (topology + weights)
- Optional gradient fine-tuning
- Telemetry and reproducibility

This plan focuses on **ergonomic graph construction** and **strong typing**, not on changing the core evolutionary algorithms.

## Why this matters

Many users want to:

- Build a custom graph (including gating/modulatory patterns) before evolving.
- Swap parts of a network (e.g., replace an encoder) without rewriting controllers.
- Teach the library “this is input / this is output” without relying on fragile indexing.

A primitives API also becomes the foundation for:

- preconfigured architectures (separate plan)
- visualization export (separate plan)
- browser examples (separate plan)

## Goals

- G1: Provide minimal primitives that can express arbitrary directed graphs.
- G2: Make input/output designation explicit and robust.
- G3: Ensure primitives can be compiled into the existing `Network` execution model.
- G4: Preserve current public API; primitives are additive.

## Non-goals

- Replacing the existing `Network` class.
- Designing a full deep-learning layers DSL (Conv/Attention/etc).
- Introducing any breaking changes in NEAT controller behavior.

## Proposed primitives

### `NeuronNode` (or `Node`)

A lightweight object that represents a neuron-like unit with:

- id (stable)
- type (`input` | `hidden` | `output` | `constant`)
- bias, activation function, optional metadata

### `ConnectionEdge`

A connection between nodes:

- fromId, toId
- weight
- enabled flag
- optional gating/modulation reference

### `NodeGroup`

A collection of nodes with helper operations:

- create N nodes
- connect to another group with a connection policy
- gate a set of edges (optional, if supported)

### `Layer`

A specialization of group with explicit `inputNodes` and `outputNodes` to simplify connecting composite blocks.

## API sketch

```ts
export type NodeRole = 'input' | 'hidden' | 'output' | 'constant';

export interface NodeSpec {
  role: NodeRole;
  bias?: number;
  activation?: Activation;
  label?: string;
}

export class GraphNode {
  readonly id: string;
  role: NodeRole;
  bias: number;
  activation: Activation;
}

export interface ConnectOptions {
  weightInit?: WeightInitializer;
  allowSelf?: boolean;
  allowRecurrent?: boolean;
}

export class NodeGroup {
  readonly nodes: readonly GraphNode[];
  connect(
    target: NodeGroup | GraphNode,
    policy?: ConnectPolicy,
    options?: ConnectOptions,
  ): GraphEdge[];
}

export class Layer extends NodeGroup {
  readonly inputNodes: readonly GraphNode[];
  readonly outputNodes: readonly GraphNode[];
}

export interface GraphBuildResult {
  network: Network;
  mapping: {
    nodeIdToNetworkIndex: Map<string, number>;
  };
}

export function constructNetwork(
  parts: Array<NodeGroup | Layer | GraphNode>,
  options?: ConstructOptions,
): GraphBuildResult;
```

The library should keep naming consistent with existing exports, but the key is:

- stable ids
- explicit roles
- a single “construct” function that produces a `Network`.

## Implementation steps

### Step 1 — Define types and minimal classes

- Add new module folder, e.g. `src/architecture/primitives/`.
- Implement minimal `GraphNode`, `GraphEdge`, `NodeGroup`, and `Layer`.

Acceptance:

- Can create groups and connect them; edges are produced.

### Step 2 — Connection policies

Implement a small set of connection policies:

- `oneToOne`
- `allToAll`
- `allToAllForward` (acyclic-by-construction)
- `allToElse` (exclude same-node connections)

Acceptance:

- Policies are deterministic under seeded RNG/weight initializer.

### Step 3 — Define role inference rules (conservative)

Construction must not guess aggressively.

Rules:

- If any nodes are explicitly marked `input`/`output`, use those.
- If not, require the user to specify via options.
- Provide a helper that can infer only when unambiguous, but default to explicit.

Acceptance:

- Ambiguous graphs throw descriptive errors.

### Step 4 — Compile primitives into a `Network`

- Implement `constructNetwork(...)`:
  - resolve nodes in stable order
  - create a `Network` with correct input/output sizes
  - add nodes in the correct order
  - materialize connections and optional gating references

Acceptance:

- Constructed network activates and produces outputs.

### Step 5 — JSDoc + docs map

- Document the primitives as an educational feature.
- Provide 2–3 short examples:
  - MLP via groups
  - custom graph via nodes
  - layer chaining

Acceptance:

- Docs generator includes the new APIs cleanly.

## Testing strategy

- Unit tests for policies and construct:
  - builds correct node counts
  - preserves explicit input/output roles
  - rejects ambiguous graphs
  - deterministic construction under seed
- Typecheck: `npx tsc --noEmit -p tsconfig.json`

## Risks and mitigations

- Risk: duplicating concepts already in `Network`.
  - Mitigation: primitives compile _into_ `Network`; do not fork execution.
- Risk: confusion about "graph" vs "genome" semantics.
  - Mitigation: docs clarify: primitives are for building initial phenotypes; NEAT still operates on genomes.

## Success criteria

- Users can construct non-trivial architectures without touching internal arrays.
- The constructed network works with evolution and training flows.
- No breaking changes to existing imports.
