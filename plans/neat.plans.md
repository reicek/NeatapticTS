# Proper NEAT (No-Compromise) — Gap Analysis + Implementation Plan

This document is the engineering plan to upgrade NeatapticTS from “NEAT‑inspired” topology evolution to **canonical NEAT with historical markings**, i.e. **proper innovation tracking, correct crossover alignment, and speciation that remains meaningful across the entire run**.

It also includes an “ultimate” tier: improvements that go beyond the 2002 NEAT paper while keeping NEAT’s core guarantees.

## Executive Summary (What’s broken vs NEAT)

NeatapticTS already has a strong NEAT _controller_ layer in [src/neat/](../../../neat/) (speciation, fitness sharing, mutation helpers, RNG/telemetry, export/import of innovation tables).

However, the **genome representation and genetic operators at the `Network` level** (notably `Network.crossOver` implemented via [src/architecture/network/genetic/](./)) are still “NEAT-ish” and currently violate key NEAT invariants:

- **Historical markings are not preserved through crossover**
  - Offspring nodes are cloned without preserving `geneId`.
  - Offspring connections are re-created without preserving `connection.innovation`.
  - Therefore offspring genomes become “new, unrelated genomes” from the perspective of speciation/compatibility.
- **Initial population genomes do not share stable gene IDs / innovations** when created via repeated `new Network(...)`.
  - This prevents meaningful innovation reuse and causes compatibility distance to treat identical structures as disjoint.
- **Recurrent and self connections are implicitly dropped by crossover**
  - Materialization filters only `from < to`, which excludes `from === to` (self) and `from > to` (recurrent).
- **Determinism is leaky**
  - Some crossover decisions use `Math.random()` instead of the controller-provided RNG.

These gaps mean we are not yet offering “proper NEAT”. Fixing them is non-trivial, but it’s very achievable because much of the controller/telemetry infrastructure already exists.

## Canonical NEAT Requirements (No Compromise)

These are the minimum invariants we must satisfy to claim “proper NEAT”:

### R1) Stable node identity (node genes)

- Every node must have a **stable identifier** across the entire evolutionary history.
- In this repo, `Node.geneId` already exists and is the right idea — but it must be:
  - **preserved** during cloning and crossover,
  - **consistent** across the initial population,
  - **restorable** from serialization,
  - and **globally unique** when genuinely new nodes arise.

### R2) Stable connection identity (connection genes)

- Every connection gene must have a stable **innovation number** (“historical marking”).
- In this repo, `Connection.innovation` already exists — but we must ensure:
  - when a connection gene is inherited in crossover, the offspring connection keeps the same `innovation`.
  - when a connection gene is created by mutation, it is assigned a deterministic global innovation number via the controller’s innovation registry (not via per-instance auto-increment semantics).

### R3) Crossover aligns by innovation number

- Matching genes are those with the same innovation number.
- Disjoint/excess genes are determined by innovation ordering.
- Offspring inherits:
  - matching genes randomly (or via fitness bias),
  - disjoint/excess from fitter parent (or both when `equal` / tie).
- Disabled genes follow canonical re-enable semantics (commonly: inherit disabled if either disabled; sometimes allow re-enable with probability ~0.25).

### R4) Compatibility distance uses innovation numbers

- Speciation must compute distance on **innovation numbers**, not on node indices.
- We should be able to guarantee that for genomes produced by our NEAT flow:
  - innovations are present and meaningful,
  - any fallback heuristic is only for importing legacy/foreign genomes.

### R5) Recurrent/self connections are supported (at least configurable)

- Canonical NEAT supports recurrence. Many modern NEAT libs allow the user to enable/disable recurrence.
- Our genetic pipeline must not silently drop these genes.

## Current Implementation Audit (Where we are)

### What’s already strong

- Controller-level innovation bookkeeping exists:
  - `_connInnovations: Map<string, number>`
  - `_nodeSplitInnovations: Map<string, NodeSplitRecord>`
  - `_nextGlobalInnovation: number`
  - exported/restored via [src/neat/neat.export.ts](../../../neat/neat.export.ts).
- Mutation helpers attempt innovation reuse:
  - add-conn reuses keys derived from node geneIds.
  - add-node reuses a split record based on `from.geneId -> to.geneId`.
- Compatibility distance is implemented (and cached) and already supports a fallback innovation for missing innovation ids.

### What blocks “proper NEAT” right now

#### B1) `Network.crossOver` currently does not preserve historical markings

The crossover implementation in [network.genetic.setup.utils.ts](./network.genetic.setup.utils.ts) clones nodes with:

- `const clonedNode = new Node(sourceNode.type)`
- Copies bias and squash
- **Does not copy `geneId`**

Connections are materialized in [network.genetic.materialize.utils.ts](./network.genetic.materialize.utils.ts) by calling `offspring.connect(...)`, which creates brand-new `Connection` objects with brand-new `innovation` numbers.

Net effect:

- Offspring loses node identity and connection identity.
- Compatibility distance becomes meaningless after the first sexual reproduction.

#### B2) Initial pool genomes aren’t identity-aligned

The pool creation path in [src/neat/neat.helpers.ts](../../../neat/neat.helpers.ts) constructs each genome with `new Network(...)` (when no seed is supplied).

That means:

- Each genome gets different `Node.geneId` values (because `geneId` allocation is global/static).
- Each genome gets different `Connection.innovation` values (because `Connection` innovations auto-increment globally).

Even if the networks are structurally identical, they will appear completely unrelated.

#### B3) Recurrent/self genes are dropped by crossover

The filter `from < to` in genetic materialization excludes:

- self connections (`from === to`)
- recurrent/back edges (`from > to`)

This is not acceptable for NEAT correctness; it should be controlled by a clear policy (e.g., `allowRecurrent`), not silently filtered.

#### B4) RNG determinism is inconsistent

Some logic uses `Math.random()` instead of `randomGenerator`.
For deterministic runs, **every stochastic decision in genetic operators must use the same RNG source**.

## Gap to “Full Proper NEAT” (What we need to build)

This section is deliberately strict: “proper NEAT” means the genome is a stable, comparable object across generations.

### Gap G1 — Make `Network` a correct NEAT genome

We need a single, coherent contract:

- Node gene identity: `Node.geneId` is the stable node gene ID.
- Connection gene identity: `Connection.innovation` is the stable connection gene ID.

And the entire lifecycle must preserve those:

1. Initial genome creation
2. Mutation (add node, add connection, enable/disable)
3. Crossover (gene alignment, inheritance)
4. Serialization/import/export
5. Pooling (nodePool/connection pooling)

### Gap G2 — Rewrite crossover to align by connection innovations (not indices)

The current crossover uses:

- innovation key = `Connection.innovationID(from.index, to.index)`
- endpoints stored as node indices
- nodes selected by position

Proper NEAT requires:

- innovation key = `connection.innovation`
- endpoints stored as **node geneIds** (or as innovations referencing node genes)
- offspring node set derived from inherited genes + required IO nodes

### Gap G3 — Make speciation distance operate on correct innovations

Compatibility already prefers `connection.innovation`. That’s good, but we must:

- ensure those innovations are meaningful and consistent.
- reserve `_fallbackInnov` for legacy import only.

### Gap G4 — Fix population bootstrapping

We must guarantee that the initial population uses:

- identical IO node geneIds across all genomes
- identical initial connection innovation ids across all genomes

Simplest canonical approach:

- build a single template genome once, then clone it N times.

## Implementation Plan (Phased)

The plan is split into **Phase 0 (diagnostics + invariants)**, **Phase 1 (correctness)**, and **Phase 2+ (ultimate NEAT)**.

### Phase 0 — Freeze the invariants (1–2 days)

Goal: create an explicit contract + tests so we can refactor safely.

- [ ] Define the “NEAT Genome Contract” in one place (doc + types):
  - Node identity: `geneId`
  - Connection identity: `innovation`
  - Connection endpoints reference nodes by `geneId` (NOT by `index`).
- [ ] Add a small set of correctness tests:
  - Crossover preserves node geneIds.
  - Crossover preserves connection innovation IDs.
  - Compatibility distance between identical genomes is 0 (or near 0 depending on weight noise), and remains stable after crossover/mutation.
  - Self/recurrent genes are preserved when allowed.
- [ ] Add a debug validator (dev-only) that can be run on a genome:
  - verifies unique geneIds, innovation numbers, endpoints valid, gater references valid.

### Phase 1 — Make “proper NEAT” the default behavior (core lift) (3–10 days)

This is the big lift. It’s mostly surgical but touches foundational code.

#### 1.1 Fix pool bootstrapping (must-do)

Target: [src/neat/neat.helpers.ts](../../../neat/neat.helpers.ts)

- [ ] When `seedNetwork` is null, create exactly **one** template genome (e.g., `new Network(input, output, ...)`) and then clone it `popsize` times using `toJSON()` / `Network.fromJSON()`.
- [ ] Ensure template cloning preserves:
  - node geneIds
  - connection innovation numbers
  - enabled flags
  - gates / selfconns

Why this matters:

- It guarantees identical initial genomes have identical historical markings.
- It makes innovation reuse maps based on `geneId` actually work.

#### 1.2 Rewrite `Network.crossOver` to be true NEAT crossover (must-do)

Targets:

- [src/architecture/network/genetic/network.genetic.selection.utils.ts](./network.genetic.selection.utils.ts)
- [src/architecture/network/genetic/network.genetic.materialize.utils.ts](./network.genetic.materialize.utils.ts)
- [src/architecture/network/genetic/network.genetic.setup.utils.ts](./network.genetic.setup.utils.ts)
- Types: [src/architecture/network/network.types.ts](../network.types.ts)

Key changes:

- [ ] Collect parent connection genes keyed by `connection.innovation` (not `Connection.innovationID(from.index,to.index)`).
- [ ] Connection gene descriptor must include:
  - `innovation: number`
  - `fromGeneId: number`
  - `toGeneId: number`
  - `weight`, `enabled`
  - `gaterGeneId?: number` (or sentinel)
  - plus an explicit `isRecurrent?: boolean` if we want to preserve recurrent classification.
- [ ] Offspring node set must be derived as:
  - always include all IO nodes (input + output), preserving their geneIds
  - include any hidden nodes referenced by inherited connection genes
  - optionally include isolated hidden nodes if we decide to preserve them (configurable)
- [ ] Node inheritance:
  - for nodes present in both parents (same `geneId`): choose bias/squash from one parent (or average/weighted)
  - for nodes present in only one parent: inherit from the parent that contributed the structural gene
- [ ] Connection materialization:
  - create runtime connections between the resolved offspring nodes
  - explicitly set `createdConnection.innovation = gene.innovation`
  - explicitly set enabled flag
  - attach gater by `gaterGeneId` mapping
- [ ] Recurrent/self policy:
  - do NOT filter `from < to` unconditionally
  - apply policy based on:
    - genome/network flag `_enforceAcyclic`
    - controller option `allowRecurrent`
  - self connections (`fromGeneId === toGeneId`) must be representable and preserved when recurrence is allowed.
- [ ] RNG determinism:
  - replace all `Math.random()` calls inside genetic operators with the injected RNG.

#### 1.3 Ensure mutation operators keep innovations/geneIds consistent (tighten)

Targets: [src/neat/neat.mutation.\*](../../../neat/)

- [ ] Make connection innovation keys directional (ordered pair) when recurrence is supported.
  - The current unordered/symmetric key (`min::max`) is only valid if recurrence/back edges are impossible.
- [ ] Ensure that re-adding a previously seen connection reuses the original innovation (persistent registry, not generation-local).
- [ ] Ensure add-node split record is keyed robustly:
  - preferred: key by the **split connection’s innovation id** (canonical) or by ordered endpoint geneIds + (optionally) connection innovation.

#### 1.4 Tighten compatibility distance expectations (optional but recommended)

Targets: [src/neat/neat.compat.ts](../../../neat/neat.compat.ts)

- [ ] For genomes produced by our NEAT engine, require `connection.innovation` to exist and be a number.
- [ ] Keep `_fallbackInnov` only for imported legacy genomes.

### Phase 2 — “Ultimate NEAT” upgrades (2–6+ weeks, modular)

These are improvements that modern NEAT users expect, without compromising the core NEAT guarantees.

#### 2.1 First-class genotype layer (recommended long-term)

Problem: using the full `Network` runtime object as the genotype is convenient, but slow and easy to corrupt.

Plan:

- [ ] Introduce an explicit `Genome` representation:
  - `NodeGene[]` and `ConnectionGene[]` sorted by innovation
  - tiny, immutable-ish, easy to clone
- [ ] Provide compilation step `Genome -> Network` (phenotype build)
- [ ] Keep `Network` as the phenotype/runtime; keep NEAT ops on `Genome`

Benefits:

- Faster crossover/mutation
- Clear invariants
- Easy compatibility distance
- Easy export/import and reproducible runs

#### 2.2 Canonical reproduction pipeline knobs

- [ ] Per-species elitism (champion always survives)
- [ ] Explicit offspring allocation by adjusted fitness (already partially present; validate vs canonical)
- [ ] Species stagnation pruning consistent with NEAT paper

#### 2.3 Operator completeness for topology and parameters

- [ ] Mutate activation function per node gene (with compatibility impact)
- [ ] Mutate bias and response parameters per node gene
- [ ] Structured weight mutation (perturb vs reset) per canonical NEAT configs
- [ ] Toggle enable/disable mutations for connection genes
- [ ] Optional gate mutation operators (if gating is a featured extension)

#### 2.4 Deterministic replay and “experiment state”

- [ ] Ensure export/import includes:
  - RNG state
  - innovation registries
  - generation
  - options snapshot
  - population genomes
- [ ] Provide a deterministic replay harness (same seed => same evolution).

#### 2.5 Performance: compatibility distance at scale

- [ ] Keep innovation lists cached per genome and invalidate only on structural mutation
- [ ] SIMD-ish / typed-array adjacency for genome comparisons when popsize is large
- [ ] Optional “fast mode” approximate speciation (already some fast diversity metrics exist)

## Definition of Done (for “Proper NEAT”)

We should consider NEAT “proper” when all are true:

- Initial population genomes share the same gene IDs for equivalent nodes, and the same innovation IDs for equivalent connections.
- After crossover, offspring nodes retain parent gene IDs; offspring connections retain inherited innovation IDs.
- Compatibility distance produces sensible results (identical ≈ 0, small mutations => small distance, major topology divergence => larger distance).
- Recurrent/self genes are preserved when allowed and explicitly disallowed when acyclic mode is enabled.
- All randomness in NEAT operations is routed through the controller RNG.

## Rough Lift Estimate

Assuming we want “proper NEAT” first (Phase 1) before the genotype refactor:

- Phase 0: **Small** (1–2 days)
- Phase 1: **Medium–Large** (3–10 days) depending on how much API surface we keep stable and how many tests we add
- Phase 2 (Ultimate): **Large** (weeks), but can be shipped incrementally

The highest risk items are:

- rewriting crossover without breaking existing users’ expectations
- reconciling recurrent/self/gating semantics across mutation/crossover/speciation
- ensuring geneId/innovation counters remain consistent across pooling and serialization

## Controller Folderization Progress

- Completed species step after speciation: extracted current-species summaries into `src/neat/species/stats/species.stats.ts` and kept `src/neat/species/species.ts` focused on the public reporting flow.
- Species history JSONL export remains isolated in `src/neat/species/history/species.history.ts`.
- Next species follow-up: narrow `src/neat/species/core/species.core.ts` into clearer augmentation/shared chapters so extended-history policy and innovation-summary mechanics stop living in one file.
