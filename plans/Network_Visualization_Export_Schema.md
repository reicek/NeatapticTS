# Network Visualization Export Schema Plan

## Purpose

Make networks easy to inspect and visualize by exporting a **stable, versioned graph schema** that external tools can render.

This avoids bundling a UI into the library while enabling:

- debugging architectures
- sharing networks in issues/docs
- building small demos and visualizations

## Goals

- G1: Provide `exportVisualizationGraph(network)` that returns a JSON object with nodes/edges and metadata.
- G2: Keep it deterministic and versioned.
- G3: Include enough metadata for useful rendering (I/O roles, activations, weights, group/layer labels).

## Non-goals

- Building an interactive visualization app in-repo.
- Supporting huge graphs with advanced layout out-of-the-box.

## Schema (v1)

```ts
export interface VisualizationGraphV1 {
  version: 1;
  nodes: Array<{
    id: number;
    label?: string;
    role?: 'input' | 'output' | 'hidden';
    activation?: string;
    bias?: number;
    groupId?: string;
    layerIndex?: number;
  }>;
  edges: Array<{
    from: number;
    to: number;
    weight: number;
    enabled?: boolean;
    kind?: 'forward' | 'recurrent' | 'self';
  }>;
  io: {
    inputNodeIds: number[];
    outputNodeIds: number[];
  };
  metadata?: {
    name?: string;
    mode?: 'acyclic' | 'recurrent';
    createdAtIso?: string;
  };
}
```

Notes:

- `groupId` and `layerIndex` become useful once primitives/builders standardize them.
- `kind` can be inferred where possible.

## Proposed public API

```ts
export interface ExportVisualizationOptions {
  includeBiases?: boolean;
  includeWeights?: boolean;
  includeDisabledEdges?: boolean;
}

export function exportVisualizationGraph(
  network: Network,
  options?: ExportVisualizationOptions,
): VisualizationGraphV1;
```

Optional convenience outputs (secondary):

- `toDot(graph)` → Graphviz DOT string (small helper)

## Implementation steps

### Step 1 — Deterministic graph extraction

- Extract nodes and edges in stable order (by node ID, then edge endpoints).
- Include explicit `io` ordering.

Acceptance:

- Same network exports identical JSON.

### Step 2 — Role and metadata mapping

- Determine node roles from explicit I/O sets.
- Add activation and bias if enabled.

Acceptance:

- Inputs/outputs are correctly labeled.

### Step 3 — Optional DOT exporter

- Implement a tiny DOT renderer that maps:
  - inputs/outputs to distinct shapes
  - edge weights to labels

Acceptance:

- DOT output can be pasted into Graphviz to render.

### Step 4 — Docs + examples

- Show how to export JSON and render it externally.

Acceptance:

- Example is clear and minimal.

## Testing strategy

- Snapshot test:
  - fixed network exports stable JSON
- Ordering test:
  - edges/nodes sorted deterministically

## Risks and mitigations

- Risk: schema churn.
  - Mitigation: versioned schema with migration notes.
- Risk: large graph size.
  - Mitigation: options to omit weights/biases.

## Success criteria

- Users can export a visualization-ready graph in one call.
- Schema is stable, deterministic, and documented.
