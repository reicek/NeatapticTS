# neat/nge-main-agent

## neat/nge-main-agent/neat.nge-main-agent.types.ts

### NgeMainAgentAdult

Main agent adult state after maturation and structural pruning are complete.

### NgeMainAgentAttentionHead

Attention-head motif descriptor for threat prioritization.

This is a typed tag used by the main agent allowlist; the actual realization
logic lives in later NGE phases.

### NgeMainAgentEmbryo

Main agent embryo state.

The embryo is the smallest materialized stage. It carries the full motif
allowlist as archetypes but keeps node and edge counts tiny so that later
growth has headroom within the tier budget.

### NgeMainAgentEmbryoArchetypeDescriptor

Archetype descriptor specialized for the embryo stage.

Every embryo archetype receives a deterministic substrate coordinate and zone
assignment from the coordinate allocator so that downstream materialization is
reproducible and zone-aware.

### NgeMainAgentEpisodicSlot

Episodic slot motif descriptor for spawn-pattern memory.

This is a typed tag used by the main agent allowlist; the actual realization
logic lives in later NGE phases.

### NgeMainAgentEquilibrium

Equilibrium candidate produced by adult optimization when the network stabilizes.

### NgeMainAgentGatedRecurrentCell

Gated recurrent cell motif descriptor for aim/strafe state retention.

This is a typed tag used by the main agent allowlist; the actual realization
logic lives in later NGE phases.

### NgeMainAgentJuvenile

Main agent juvenile state during local growth and focus gating.

### NgeMainAgentLifecycleConfig

Configuration governing the main agent lifecycle and tier budget.

The same config plus the same seed must always advance through the lifecycle
in the same order, so the seed is stored here rather than inferred from the
environment.

### NgeMainAgentLifecycleStage

Ordered lifecycle stages for the NGE main agent.

The stage machine is intentionally distinct from the Racing Curriculum
juvenile/adult stage contract so that main-agent morphogenesis can evolve its
own cadence without being coupled to a different demo's lifecycle policy.

### NgeMainAgentLifecycleState

Shared state fields across all main-agent lifecycle stages for the NGE extension.

### NgeMainAgentReproducing

Main agent reproducing stage, ready to emit the next generation's embryo.

### NgeMainAgentTopologyBudget

Topology budget enforced at every lifecycle transition.

The budget is always positive and never exceeds the configured tier cap.

## neat/nge-main-agent/nge-to-network.ts

### assembleNetworkFromParts

```ts
assembleNetworkFromParts(
  nodes: default[],
  seed: number,
): default
```

Assemble a `Network` from pre-wired nodes via the construct-from-parts API.

Input and output node labels are collected from the node array and passed to
`Network.construct` with `mode: 'recurrent'` and relaxed output-edge
validation so that output self-connections and feedback edges are permitted.

`Network.construct` separates self-connections into `network.selfconns` and
non-self connections into `network.connections`. The bridge preserves that
invariant — callers should inspect both arrays. The total edge count is
`network.connections.length + network.selfconns.length`, matching the bridge
contract that this sum equals `state.edgeCount`.

Parameters:
- `nodes` - Pre-wired node array with labels on I/O nodes.
- `seed` - Determinism seed forwarded to the network constructor.

Returns: A materialized `Network` instance.

### createMaterializationNodes

```ts
createMaterializationNodes(
  nodeCount: number,
  seed: number,
  inputCount: number,
  outputCount: number,
): default[]
```

Create `nodeCount` deterministic `Node` instances with role-based labels.

The first `inputCount` nodes are typed `'input'`, the last `outputCount`
nodes `'output'`, and every interior node `'hidden'`. A seeded PRNG supplies
bias values so the same seed always produces the same initial biases.

Parameters:
- `nodeCount` - Total nodes to create (must be ≥ 2 for a valid I/O pair).
- `seed` - Determinism seed forwarded to the PRNG.
- `inputCount` - Number of input nodes (first `inputCount` indices).
- `outputCount` - Number of output nodes (last `outputCount` indices).

Returns: An ordered array of `Node` instances.

### createSeededRng

```ts
createSeededRng(
  seed: number,
): () => number
```

Create a deterministic mulberry32 PRNG from a numeric seed.

The same seed always produces the same sequence of pseudo-random floats in
`[0, 1)`, which keeps node bias initialization reproducible.

Parameters:
- `seed` - Numeric seed (coerced to a 32-bit unsigned integer).

Returns: A stateful RNG function returning floats in `[0, 1)`.

### enumerateCandidateEdges

```ts
enumerateCandidateEdges(
  nodeCount: number,
  recurrentFirst: boolean,
  inputCount: number,
): [number, number][]
```

Enumerate all candidate directed edges for `nodeCount` nodes in deterministic
order.

Self-edges are listed separately from non-self edges. When `recurrentFirst`
is true, self-edges precede non-self edges so that a `slice(0, edgeCount)`
selection guarantees at least one recurrent connection.

Parameters:
- `nodeCount` - Total number of nodes.
- `recurrentFirst` - Whether to prioritize self-connections.
- `inputCount` - Number of input nodes (indices 0..inputCount-1) to
protect from incoming edges.

Returns: An ordered array of `[sourceIndex, targetIndex]` pairs.

### materializeFromNgeState

```ts
materializeFromNgeState(
  state: NgeMaterializableState,
  options: MaterializeOptions | undefined,
): default
```

Materialize a live {@link Network} runtime from an NGE main-agent lifecycle state.

The bridge reads the deterministic topology counts (`nodeCount`, `edgeCount`),
motif archetypes, and seed from any lifecycle stage — embryo, juvenile, adult,
or reproducing — and produces a fully wired `Network` instance whose:

- `nodes.length` equals `state.nodeCount`,
- `connections.length + selfconns.length` equals `state.edgeCount`
  (self-connections remain in `selfconns`, matching the `Network.construct`
  invariant), and
- `connections` or `selfconns` contains at least one self-connection when the
  state's archetypes include a recurrent motif (`GatedRecurrentCell` or
  `EpisodicSlot`).

The same state always produces the same node and connection counts, and the
same innovation IDs, making the bridge reproducible across calls.

Parameters:
- `state` - Any materializable NGE main-agent lifecycle state.
- `options` - Optional I/O configuration. When omitted the bridge defaults
to 1 input and 1 output node (the first and last nodes respectively). Pass
`{ inputCount, outputCount }` to produce a network with the correct number
of input and output nodes — e.g. to be a drop-in replacement for
`new Network(22, 5)`.

Returns: A `Network` instance wired with the state's topology.

Example:

```ts
const embryo = buildMainAgentEmbryo({ seed: 42, maxNodes: 1024, maxEdges: 4096 });
const network = materializeFromNgeState(embryo);
console.log(network.nodes.length);       // 3
console.log(network.connections.length); // non-self edges
console.log(network.selfconns.length);   // self-edges
```

### MaterializeOptions

Optional I/O configuration for {@link materializeFromNgeState}.

Allows the caller to specify how many of the materialized nodes should be
designated as input and output nodes, making the bridge a drop-in replacement
for `new Network(inputCount, outputCount)`.

### NgeMaterializableState

Union of every NGE main-agent lifecycle stage that carries enough topology
metadata to materialize a `Network`.

All four stages share `nodeCount`, `edgeCount`, `seed`, and `archetypes`
(each with a `computationType`), which are the only fields the bridge reads.

### resolveNodeLabel

```ts
resolveNodeLabel(
  role: string,
  index: number,
): string | null
```

Resolve a human-readable label for a node so it can be referenced by string
id in `Network.construct`.

Parameters:
- `role` - Runtime role assigned to the node.
- `index` - Zero-based node position.

Returns: A label string for input/output nodes, or `null` for hidden nodes.

### resolveNodeRole

```ts
resolveNodeRole(
  index: number,
  nodeCount: number,
  inputCount: number,
  outputCount: number,
): string
```

Resolve the runtime role string for a node at a given index.

The first `inputCount` nodes are inputs and the last `outputCount` nodes are
outputs. When the two ranges would overlap (`inputCount + outputCount >
nodeCount`) the output range takes precedence so every requested output node
is guaranteed a slot.

Parameters:
- `index` - Zero-based node position.
- `nodeCount` - Total node count in the materialization set.
- `inputCount` - Number of input nodes.
- `outputCount` - Number of output nodes.

Returns: `'input'`, `'output'`, or `'hidden'`.

### wireMaterializationEdges

```ts
wireMaterializationEdges(
  nodes: default[],
  edgeCount: number,
  hasRecurrentMotif: boolean,
  inputCount: number,
): void
```

Wire `edgeCount` deterministic connections between the provided nodes.

Candidate edges are enumerated in a stable order. When `hasRecurrentMotif`
is true, self-connections are prioritized so at least one recurrent edge is
selected (provided `edgeCount` ≥ 1). Otherwise, non-self edges come first and
self-connections fill remaining slots only when needed.

The first `inputCount` nodes are never targets of any edge, preserving the
pure-source invariant required by `Network.construct`.

Parameters:
- `nodes` - Ordered node array to wire.
- `edgeCount` - Number of edges to create.
- `hasRecurrentMotif` - Whether recurrent motifs are present in the state.
- `inputCount` - Number of input nodes to protect from incoming edges.

## neat/nge-main-agent/neat.nge-main-agent.adult.ts

### buildDefaultFitnessMetrics

```ts
buildDefaultFitnessMetrics(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibriumFitnessMetrics
```

Build deterministic default fitness metrics from the adult topology.

Combat fields start neutral; structural bonuses/penalties are derived from the
adult's share of the tier budget so the candidate always carries a
deterministic, reproducible fitness shape.

Parameters:
- `adult` - Adult state at equilibrium.
- `config` - Lifecycle config with the tier budget.

Returns: Neutral-but-shaped fitness metrics.

### buildSnapshotCandidate

```ts
buildSnapshotCandidate(
  candidate: NgeMainAgentStableCandidate,
  adult: NgeMainAgentAdult,
): NgeMainAgentEquilibriumSnapshot
```

Build a frozen snapshot from a stable candidate.

The snapshot is a deep-cloned, frozen copy so enemies evaluate against an
immutable view of the main agent at equilibrium.

Parameters:
- `candidate` - Stable candidate to freeze.
- `adult` - Adult state that produced the candidate.

Returns: A frozen snapshot that enemies can evaluate against.

### buildStableCandidate

```ts
buildStableCandidate(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentStableCandidate
```

Build a stable candidate from the adult state.

The candidate packages the genome state, structural qualifiers, and a
deterministic fitness placeholder into one reproduction-ready object.

Parameters:
- `adult` - Adult state at equilibrium.
- `config` - Lifecycle config with the tier budget.

Returns: A stable candidate suitable for reproduction.

### evaluateAdultTopologyBudget

```ts
evaluateAdultTopologyBudget(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): { withinBudget: boolean; }
```

Evaluate whether an adult topology is within the configured tier budget.

Parameters:
- `adult` - Adult state to evaluate.
- `config` - Lifecycle config with the tier budget.

Returns: An evaluation object whose `withinBudget` flag is true when the adult respects both caps.

### NgeMainAgentEquilibriumFitnessMetrics

Fitness metrics captured at adult equilibrium.

These fields mirror the combat quality signal used by the harness, but the
first-pass equilibrium stage only records structural defaults because the
lifecycle module does not yet run episodes. Later phases will overwrite them
with real episode telemetry.

### NgeMainAgentEquilibriumResult

Equilibrium result produced by {@link runAdultEquilibrium}.

Extends the base equilibrium contract with a stable reproduction candidate
and a frozen enemy-evaluable snapshot.

### NgeMainAgentEquilibriumSnapshot

Frozen snapshot that enemy populations evaluate against.

The snapshot is intentionally clone-safe and deterministic: the same adult
and config always produce the same snapshot, and callers can safely pass it
across worker or barrier boundaries.

### NgeMainAgentStableCandidate

Genome and structural snapshot of a stable adult at equilibrium.

This is the reproduction-ready view of the adult: it captures the genome
state, deterministic structural qualifiers, and the fitness metric shape
that downstream reproduction and barrier logic expect.

### pruneAdultTopology

```ts
pruneAdultTopology(
  juvenile: NgeMainAgentJuvenile,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult
```

Prune a juvenile topology down to adult limits while remaining within budget.

The adult stage may reduce or cap node and edge counts, but it never allows
them to exceed the configured tier budget.

Parameters:
- `juvenile` - Juvenile state to prune.
- `config` - Lifecycle config with the tier budget cap.

Returns: Adult state whose counts are bounded by the tier budget.

### runAdultEquilibrium

```ts
runAdultEquilibrium(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibriumResult
```

Run adult optimization until an equilibrium candidate is stable.

Produces a stable reproduction candidate and a frozen enemy-evaluable
snapshot. The result is deterministic: the same adult and config always
yield the same stable candidate and snapshot.

Parameters:
- `adult` - Adult state to optimize.
- `config` - Lifecycle config with the deterministic seed and tier budget.

Returns: Equilibrium result wrapping the stable adult, candidate, and snapshot.

## neat/nge-main-agent/neat.nge-main-agent.embryo.ts

### buildMainAgentEmbryo

```ts
buildMainAgentEmbryo(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEmbryo
```

Build a deterministic main-agent embryo state.

The embryo carries the three allowed motif archetypes, each allocated a
deterministic substrate coordinate and zone, plus an initial parthenogenetic
reproduction mode that the hysteresis policy may later update. Node and edge
counts stay within the tier budget. The same config always produces the same
embryo, which is required for reproducible generation barriers.

Parameters:
- `config` - Lifecycle config with seed and tier budget.

Returns: A deterministic embryo state ready for juvenile growth.

### computeEmbryoTopologyBudget

```ts
computeEmbryoTopologyBudget(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentTopologyBudget
```

Compute the topology budget for an embryo, capping it at the tier budget.

The embryo is intentionally small relative to the tier cap so that juvenile
growth and adult pruning have meaningful headroom without risking overflow.

Parameters:
- `config` - Lifecycle config with the configured tier cap.

Returns: A positive topology budget bounded by the tier cap.

### resolveMainAgentMotifAllowlist

```ts
resolveMainAgentMotifAllowlist(): readonly ("DenseFeedForward" | "AttentionHead" | "GatedRecurrentCell" | "EpisodicSlot" | "ModulatorBroadcaster" | "GatingRouter")[]
```

Resolve the exact three existing catalogue motifs allowed for the main agent.

The main-agent motif set is deliberately minimal and does not introduce any
new computation types or schema versions. All returned values are already
present in {@link NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE}.

Returns: A readonly array containing AttentionHead, GatedRecurrentCell, and EpisodicSlot.

## neat/nge-main-agent/neat.nge-main-agent.juvenile.ts

### evaluateJuvenileGrowGate

```ts
evaluateJuvenileGrowGate(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): boolean
```

Evaluate the deterministic hysteresis grow gate for a juvenile embryo.

The gate opens every `MIN_JUVENILE_GROW_COOLDOWN + (seed % 3)` generations,
starting from generation 0. This creates a predictable but lineage-specific
cadence that avoids synchronized population-wide growth bursts.

Parameters:
- `embryo` - Embryo state carrying the current generation and seed.
- `config` - Lifecycle config with the deterministic seed.

Returns: True when the juvenile is permitted to grow this generation.

Example:

```ts
const canGrow = evaluateJuvenileGrowGate(embryo, { seed: 7, maxNodes: 64, maxEdges: 256 });
expect(canGrow).toBe(embryo.generation === 0 || embryo.generation % 4 === 0);
```

### growJuvenileTopology

```ts
growJuvenileTopology(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentJuvenile
```

Grow the embryo topology into a juvenile state while staying within budget.

Juvenile growth is deterministic and never shrinks the network below the
embryo size. The growth factor is derived from the embryo's seed so that the
same embryo always produces the same juvenile.

Parameters:
- `embryo` - Embryo state to grow.
- `config` - Lifecycle config with the tier budget cap.

Returns: Juvenile state with node and edge counts within the tier budget.

### growJuvenileTopologyWithInternalAssimilation

```ts
growJuvenileTopologyWithInternalAssimilation(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
  candidate: NgeAssimilationCandidate | undefined,
): JuvenileGrowPassResult
```

Grow the juvenile topology and, if available, assimilate internal priors.

This orchestration applies the hysteresis grow gate first. When the gate is
open, {@link growJuvenileTopology} is used exactly as defined. When the gate is
closed, a minimal juvenile snapshot is produced so the stage contract is
preserved without forcing a topology change. If an equilibrium candidate is
provided, weak, decaying structural priors are written back to the main
agent's own genome.

Parameters:
- `embryo` - Embryo state to grow.
- `config` - Lifecycle config with the tier budget cap and seed.
- `candidate` - Optional equilibrium candidate from the main agent's own
adult boundary. Enemy-derived fields are ignored by internal assimilation.

Returns: Juvenile state plus grow-gate and internal-assimilation metadata.

Example:

```ts
const pass = growJuvenileTopologyWithInternalAssimilation(embryo, config, candidate);
expect(pass.juvenile.stage).toBe('juvenile');
expect(pass.assimilation?.enemyWeightsIncorporated).toBe(false);
```

### JuvenileGrowPassResult

Result of a combined juvenile grow and internal assimilation pass.

The result surfaces whether the topology actually grew this generation and
any structural priors that were weakly written back from the main agent's
own equilibrium candidate.

Example:

```ts
const pass: JuvenileGrowPassResult = growJuvenileTopologyWithInternalAssimilation(
  embryo,
  config,
  candidate,
);
expect(pass.didGrow).toBe(true);
```

### transitionJuvenileToAdult

```ts
transitionJuvenileToAdult(
  juvenile: NgeMainAgentJuvenile | NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult
```

Mature a juvenile or embryo state into an adult state.

The transition is a deterministic stage change that preserves the input
topology. Adult pruning is handled separately by {@link pruneAdultTopology}.

Parameters:
- `juvenile` - Juvenile or embryo state to mature.
- `config` - Lifecycle config with the deterministic seed.

Returns: Adult state with the same topology as the input.

## neat/nge-main-agent/neat.nge-main-agent.lifecycle.ts

### advanceMainAgentLifecycle

```ts
advanceMainAgentLifecycle(
  state: NgeMainAgentLifecycleState,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentLifecycleState
```

Advance one main-agent lifecycle state to the next stage.

The stage machine follows the fixed order
Embryo → Juvenile → Adult → Reproducing → Embryo. The generation counter
increments on every call so that the full cycle is observable and
deterministic for the same config.

Parameters:
- `state` - Current lifecycle state.
- `config` - Lifecycle config that supplies the deterministic seed.

Returns: The next lifecycle state with the stage advanced and generation incremented.

### createMainAgentLifecycleRunner

```ts
createMainAgentLifecycleRunner(
  config: NgeMainAgentLifecycleConfig,
): (state: NgeMainAgentLifecycleState) => NgeMainAgentLifecycleState
```

Create a deterministic lifecycle runner bound to the supplied config.

The runner is a pure function: the same input state always yields the same
output state, which makes generation replay and barrier tests stable.

Parameters:
- `config` - Lifecycle config to bind to every runner invocation.

Returns: A function that advances a lifecycle state using the bound config.

## neat/nge-main-agent/neat.nge-main-agent.reproduction.ts

### enforceMainAgentSeedPolicy

```ts
enforceMainAgentSeedPolicy(
  policy: NgeReproductionPolicy,
): NgeReproductionPolicy
```

Enforce the main-agent seed policy on a hysteresis-selected policy.

Parameters:
- `policy` - Policy returned by the hysteresis selector.

Returns: The same policy with the seed policy slot forced to the main-agent contract.

### exhaustiveFallback

```ts
exhaustiveFallback(
  mode: never,
): never
```

Fall back for unexpected reproduction modes.

The switch above covers every known {@link NgeReproductionPolicyMode}, so a
runtime mismatch is treated as an implementation bug rather than a user error.

Parameters:
- `mode` - Unexpected mode value.

Returns: Never; always throws.

### NgeMainAgentReproductionStageResult

Result of running the main-agent reproduction stage.

Carries the canonical offspring genome, the mode that produced it, the
fingerprints of every parent that contributed genetic material, and the
determinism seed so callers can replay or audit the generation.

### reproduceParthenogenesisPath

```ts
reproduceParthenogenesisPath(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  parentDna: NgeDnaCanonicalEnvelope,
  policy: NgeReproductionPolicy,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproductionStageResult
```

Run the parthenogenesis branch of the reproduction stage.

Parameters:
- `equilibriumResult` - Stable adult equilibrium for the primary parent.
- `parentDna` - Canonical DNA of the primary parent.
- `policy` - Hysteresis-selected policy with enforced seed policy.
- `config` - Lifecycle config with the deterministic seed.

Returns: Reproduction stage result for the asexual path.

### reproducePolyandricPath

```ts
reproducePolyandricPath(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  parentDna: NgeDnaCanonicalEnvelope,
  matePool: readonly NgeDnaCanonicalEnvelope[],
  policy: NgeReproductionPolicy,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproductionStageResult
```

Run the polyandric branch of the reproduction stage.

Selects up to `policy.polyandricDroneCount` mates from the pool, reusing the
primary parent as a fallback when the pool is empty.

Parameters:
- `equilibriumResult` - Stable adult equilibrium for the queen parent.
- `parentDna` - Canonical DNA of the queen parent.
- `matePool` - Optional secondary DNAs used as drones.
- `policy` - Hysteresis-selected policy with enforced seed policy.
- `config` - Lifecycle config with the deterministic seed.

Returns: Reproduction stage result for the multi-parent path.

### reproduceSexualPath

```ts
reproduceSexualPath(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  parentDna: NgeDnaCanonicalEnvelope,
  matePool: readonly NgeDnaCanonicalEnvelope[],
  policy: NgeReproductionPolicy,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproductionStageResult
```

Run the sexual crossover branch of the reproduction stage.

Uses the first mate in the pool as the second parent, falling back to the
primary parent when no pool is supplied.

Parameters:
- `equilibriumResult` - Stable adult equilibrium for the first parent.
- `parentDna` - Canonical DNA of the first parent.
- `matePool` - Optional secondary DNAs used as the second parent.
- `policy` - Hysteresis-selected policy with enforced seed policy.
- `config` - Lifecycle config with the deterministic seed.

Returns: Reproduction stage result for the sexual path.

### resolveMatePool

```ts
resolveMatePool(
  matePool: readonly NgeDnaCanonicalEnvelope[],
  parentDna: NgeDnaCanonicalEnvelope,
): NgeDnaCanonicalEnvelope[]
```

Resolve the pool of secondary mates used by polyandric and sexual modes.

When no external mate pool is supplied the primary parent is reused as a safe
fallback so the stage always produces deterministic output.

Parameters:
- `matePool` - Optional secondary DNAs supplied by the caller.
- `parentDna` - Primary parent DNA to fall back to.

Returns: A non-empty list of mate DNAs.

### resolveParentScore

```ts
resolveParentScore(
  equilibriumResult: NgeMainAgentEquilibriumResult,
): number
```

Extract a deterministic parent score from the stable candidate.

Falls back to the neutral mate fitness when the candidate metrics are absent.

Parameters:
- `equilibriumResult` - Equilibrium result carrying the stable candidate.

Returns: A numeric score suitable for reproduction operators.

### runReproductionStage

```ts
runReproductionStage(
  equilibriumResult: NgeMainAgentEquilibriumResult,
  pressureHistory: readonly ReproductionModePressureSignal[],
  parentDna: NgeDnaCanonicalEnvelope,
  config: NgeMainAgentLifecycleConfig,
  matePool: readonly NgeDnaCanonicalEnvelope[],
): NgeMainAgentReproductionStageResult
```

Run the main-agent reproduction stage from a stable adult equilibrium.

The stage consults the 3-generation combat-pressure hysteresis policy to
choose a reproduction mode, then dispatches to the matching NGE operator:
parthenogenesis, polyandric, or sexual crossover. The resulting offspring
always uses the canonical seed policy `{ siblingsDifferBySeed: true,
twinsAllowed: false }` so siblings diverge by seed and exact twins are
disallowed.

Parameters:
- `equilibriumResult` - Stable adult equilibrium carrying the reproduction-ready candidate.
- `pressureHistory` - Last-generation combat-pressure window (oldest to newest).
- `parentDna` - Canonical DNA envelope of the primary parent (queen/first parent).
- `config` - Lifecycle config with the deterministic seed.
- `matePool` - Optional secondary DNAs used as drones or sexual partners.

Returns: Reproduction stage result with offspring, mode, parent fingerprints, and seed.

### transitionAdultToReproducing

```ts
transitionAdultToReproducing(
  adult: NgeMainAgentAdult,
  equilibrium: NgeMainAgentEquilibrium,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproducing
```

Transition a stable adult equilibrium into the reproducing stage.

The reproducing state inherits its topology from the stable equilibrium
adult. It is the final stage before the lifecycle runner loops back to embryo
for the next generation.

Parameters:
- `adult` - Adult state entering reproduction.
- `equilibrium` - Stable equilibrium candidate produced by adult optimization.
- `config` - Lifecycle config with the deterministic seed.

Returns: Reproducing state ready to emit the next generation.
