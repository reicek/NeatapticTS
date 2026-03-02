# Preconfigured Architectures (MLP + Sequence Builders) Plan

## Purpose

Provide a set of **high-quality, preconfigured architecture builders** that are:

- easy to use (one-liners)
- deterministic
- compatible with evolution and optional gradient fine-tuning
- implemented using the primitives + construct pipeline (or existing internal helpers) so we avoid “two ways to build everything”

## Goals

- G1: Ship ergonomic builders for common architectures.
- G2: Ensure all builders produce networks with explicit I/O roles and stable activation order.
- G3: Provide clear docs with small examples and expected usage patterns.

## Non-goals

- Implementing every deep learning layer type.
- Claiming strict equivalence with specialized frameworks.
- Making these architectures ONNX-exportable by default (that’s constrained by ONNX plan).

## Architecture set (initial)

### 1) `Perceptron` / MLP

- `Perceptron(input, ...hidden, output)`
- Fully-connected feed-forward.

### 2) `RandomSparse`

- `RandomSparse(input, hidden, output, { connections, backConnections, selfConnections, gates })`
- Focus on evolution-friendly sparse initializations.

### 3) Sequence / time-series family

These are intended for tasks where state matters.

- `NARX(input, hiddenSizes, output, inputMemory, outputMemory)`
  - “Remember last N inputs/outputs” via identity memory blocks.
- `LSTM(input, ...blockSizes, output, options)`
  - A pedagogical LSTM-like gating structure.
- `GRU(input, ...unitSizes, output)`
  - A pedagogical GRU-like structure.

Important note: these are educational implementations intended to interoperate with evolution; they should be clearly documented as such.

## Design principles

- Prefer **small, explicit graphs** rather than clever magic.
- Use **named substructures** (input gate, forget gate, memory cell, etc.) to support visualization and debugging.
- Ensure deterministic initial weights when a seed/initializer is provided.

## API sketch

```ts
export interface BuildOptions {
  weightInit?: WeightInitializer;
  biasInit?: BiasInitializer;
  activation?: Activation;
  seed?: number;
}

export const Architect = {
  Perceptron(...sizes: number[]): Network,
  RandomSparse(input: number, hidden: number, output: number, options?: RandomSparseOptions): Network,
  NARX(input: number, hidden: number | number[], output: number, inputMemory: number, outputMemory: number, options?: BuildOptions): Network,
  LSTM(...sizesAndMaybeOptions: Array<number | LSTMOptions>): Network,
  GRU(...sizes: number[]): Network,
};
```

Naming should match the repo’s existing surface (if `Architect` already exists, extend it; if not, introduce a new top-level namespace carefully).

## Implementation steps

### Step 1 — MLP builder hardening

- Ensure MLP builder:
  - uses explicit input/output nodes
  - is fully connected
  - produces a stable activation order

Acceptance:

- MLP outputs match expectations and works with both evolution and training.

### Step 2 — Random sparse builder

- Provide a builder that creates a sparse graph with configurable counts.
- Must avoid invalid connection requests (e.g., more connections than possible) with clear errors.

Acceptance:

- Builds quickly; respects constraints; deterministic under seed.

### Step 3 — NARX builder

- Build memory layers using identity/constant nodes.
- Document the “clear state” behavior and when to use it.

Acceptance:

- Works on a small sequence prediction example.

### Step 4 — LSTM/GRU builders (pedagogical)

- Implement as explicit gated graphs using primitives.
- Provide options to toggle extra connections (e.g., input-to-output direct connections).

Acceptance:

- Produces stable graphs; can be evolved and optionally trained.

### Step 5 — Docs and examples

- For each builder, include:
  - small example
  - recommended training/evolution knobs
  - pitfalls (state clearing, dataset shuffling)

Acceptance:

- Documentation is clear and runnable.

## Testing strategy

- Snapshot structural tests:
  - node/edge counts for known sizes
  - roles for I/O nodes
  - deterministic output under seed
- Runtime sanity tests:
  - XOR for MLP
  - tiny sequence task for NARX/GRU/LSTM (1–2 minutes max)

## Risks and mitigations

- Risk: user expects “framework-grade” LSTM/GRU.
  - Mitigation: label as pedagogical/evolution-friendly; show when to prefer simpler NARX.
- Risk: recurrent graphs break activation assumptions.
  - Mitigation: require recurrent-mode construction semantics and stable ordering.

## Success criteria

- Users can create common architectures with one line.
- Builders are deterministic and documented.
- Architectures integrate with evolution and training without special cases.
