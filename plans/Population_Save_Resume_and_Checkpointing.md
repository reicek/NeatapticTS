# Population Save/Resume + Checkpointing Plan

## Purpose

Allow long-running evolution to be paused/resumed and moved across machines by introducing a **versioned checkpoint format** that captures:

- population state
- species assignment (if applicable)
- genome/network weights
- RNG state / deterministic seed context
- generation counters and metadata

## Goals

- G1: Provide a stable checkpoint format with explicit versioning.
- G2: Ensure a resumed run reproduces the same future trajectory when deterministic settings are enabled.
- G3: Offer both “full checkpoint” (exact resume) and “light checkpoint” (best-effort resume) modes.

## Non-goals

- Perfect compactness in v1 (we can add compression later).
- Serializing every possible custom user callback.

## Checkpoint modes

### Mode A — Full checkpoint (exact resume)

Captures everything needed for deterministic continuation:

- generation index
- population genomes with full network parameters
- species state
- mutation rates/adaptive controllers
- RNG snapshot/state
- any cached innovation/ID counters needed for consistent future IDs

### Mode B — Light checkpoint (approx resume)

Captures:

- best genomes/networks
- high-level config
- seed

Use this when the user only needs to continue “roughly” from a good state.

## Format sketch (v1)

```ts
export interface EvolutionCheckpointV1 {
  version: 1;
  createdAtIso: string;
  mode: 'full' | 'light';
  config: Record<string, unknown>;
  generation: number;
  rng?: {
    kind: string;
    state: unknown;
  };
  population: {
    genomes: Array<{
      id: string;
      score?: number;
      network: unknown; // use existing network serialization, or a new stable one
      meta?: Record<string, unknown>;
    }>;
    species?: Array<{
      id: string;
      memberGenomeIds: string[];
      representativeGenomeId?: string;
      fitness?: number;
    }>;
  };
  counters?: {
    nextGenomeId?: number;
    nextNodeId?: number;
    nextEdgeId?: number;
    // any NEAT-specific innovation counters
  };
}
```

Notes:

- Keep the structure explicit and versioned.
- Prefer JSON-serializable fields in v1.

## Proposed public API

```ts
export interface SaveCheckpointOptions {
  mode: 'full' | 'light';
}

export function saveEvolutionCheckpoint(
  evolution: EvolutionController,
  options: SaveCheckpointOptions,
): EvolutionCheckpointV1;

export function loadEvolutionCheckpoint(
  checkpoint: EvolutionCheckpointV1,
): EvolutionController;
```

The exact types depend on current architecture (`Population`, `NEAT`, etc.). The key is that save/load sit at the orchestration layer.

## Determinism requirements

To guarantee exact resume:

- checkpoint must include RNG state snapshot
- all ID/innovation counters must be captured
- any adaptive mutation state must be captured
- evaluation order must be deterministic

If any requirement is missing, load should:

- either throw (strict mode)
- or warn and continue best-effort (non-strict)

## Implementation steps

### Step 1 — Inventory required state

- Identify the minimal set of fields needed for exact resume.
- Document which features affect determinism.

Acceptance:

- A written checklist of required state exists.

### Step 2 — Implement full checkpoint

- Implement `saveEvolutionCheckpoint(..., { mode: "full" })`.
- Include validation to ensure required state is present.

Acceptance:

- Full checkpoint roundtrip loads successfully.

### Step 3 — Implement load/resume

- Implement `loadEvolutionCheckpoint`.
- Ensure IDs and counters are restored.

Acceptance:

- Resumed run continues without errors.

### Step 4 — Light checkpoint

- Implement light mode focusing on “resume from good solutions”.

Acceptance:

- Users can resume evolution with good initial genomes.

### Step 5 — Docs + examples

- Add docs covering:
  - when to use full vs light
  - determinism expectations
  - recommended checkpoint cadence

Acceptance:

- Copy-paste example for saving/loading.

## Testing strategy

- Roundtrip tests:
  - save → load preserves population size, scores, and network parameters
- Determinism tests (where feasible):
  - fixed seed + fixed evaluation dataset
  - run N generations → checkpoint → resume → compare best score trajectory

## Risks and mitigations

- Risk: checkpoint becomes huge.
  - Mitigation: light mode; later add compression or typed/binary formats.
- Risk: breaking changes.
  - Mitigation: strict versioning + migration notes.

## Success criteria

- Users can pause/resume long runs reliably.
- Deterministic resume works when configured.
- Checkpoint format is documented and versioned.
