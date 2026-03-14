# neat

Speciation options for the NEAT speciation controller.

Extends {@link NeatOptions} with speciation-specific configuration used by:
- Compatibility-threshold based species assignment
- Adaptive threshold controllers (PID-like)
- Species allocation telemetry (history snapshots)

## neat/neat.types.ts

### AnyObj

Generic map type used as a stop‑gap where the precise shape is still in flux.
Prefer a specific interface once the surface stabilises.

### ComplexityMetrics

Aggregate structural complexity metrics capturing size & growth pressure.

### ConnectionLike

Lightweight connection representation used by telemetry and structural helpers.

### DiversityStats

Diversity statistics captured each generation. Individual fields may be
omitted in telemetry output if diversity tracking is partially disabled to
reduce runtime cost.

### GenomeDetailed

More concrete genome surface used by telemetry and lineage helpers.
Extends the minimal `GenomeLike` with node/connection shapes and a few
internal bookkeeping fields used by telemetry.

### GenomeLike

Minimal genome structural surface used by several helpers (incrementally expanded).

NOTE: `nodes` and `connections` remain intentionally structural/opaque
until a stable public abstraction is finalised.

### LineageSnapshot

Snapshot of lineage & ancestry statistics for the current generation.

### NeatLike

Minimal surface every helper currently expects from a NEAT instance while
extraction continues. Kept intentionally loose; prefer concrete fields
when helpers are stabilised. Represented as a simple record to avoid an
empty interface that duplicates its supertype.

### NeatOptions

Options subset used by telemetry helpers. Kept narrow to avoid leaking
full runtime options into the helper type surface.

### NodeLike

Lightweight node representation used by telemetry and structural helpers.

### ObjAges

Map of objective key to age in generations since introduction.

### ObjectiveDescriptor

Descriptor for a single optimisation objective (single or multi‑objective runs).

Examples:

Add a maximisation objective for accuracy
```ts
const accuracyObj: ObjectiveDescriptor = {
  key: 'accuracy',
  direction: 'max',
  accessor: g => g.score ?? 0
};
```

Add a minimisation objective for network complexity
```ts
const complexityObj: ObjectiveDescriptor = {
  key: 'complexity',
  direction: 'min',
  accessor: g => (g.nodes.length + g.connections.length)
};
```

### ObjectiveEvent

Objective add/remove lifecycle event for telemetry and auditing.

### ObjEvent

Dynamic objective lifecycle event (addition or removal).

**Deprecated:** Use `ObjectiveEvent` instead.

### ObjImportance

Map of objective key to its importance metrics (range / variance).

### ObjImportanceEntry

Contribution / dispersion metrics for an objective over a recent window.
Used to gauge whether an objective meaningfully influences selection.

### OperatorStat

Per-generation statistic for a genetic operator.

Success is operator‑specific (e.g. produced a structurally valid mutation).
A high attempt count with low success can indicate constraints becoming tight
(e.g. structural budgets reached) – useful for adaptive operator scheduling.

### OperatorStatsRecord

Aggregated success / attempt counters over a window or entire run.

### ParetoArchiveEntry

Pareto archive entry capturing a genome plus its objective values.

### PerformanceMetrics

Timing metrics for coarse evolutionary phases (milliseconds).

### SpeciationHarnessContext

Minimal runtime surface required by speciation helpers.
Tests and harnesses can narrow the options type via the generic parameter.

### SpeciationOptions

Speciation options for the NEAT speciation controller.

Extends {@link NeatOptions} with speciation-specific configuration used by:
- Compatibility-threshold based species assignment
- Adaptive threshold controllers (PID-like)
- Species allocation telemetry (history snapshots)

### SpeciesAlloc

Offspring allocation for a species during reproduction.

### SpeciesHistoryEntry

Species statistics captured for a particular generation.

### SpeciesHistoryStat

Species statistics at a single historical snapshot (generation boundary).

### SpeciesHistoryStatExtended

Extended per-species historical snapshot with optional backfilled metrics
that may be computed lazily (innovationRange, enabledRatio).

### SpeciesLastStats

Rolling statistics tracked for each species between generations.
These values inform stagnation heuristics and adaptive controllers.

### SpeciesLike

Internal species representation used by helpers. Kept minimal and structural.

### TelemetryEntry

Telemetry summary for one generation.

Optional properties are feature‑dependent; consumers MUST test for presence.

Example:

```ts
function logSummary(t: TelemetryEntry) {
  console.log(`Gen ${t.gen} best=${t.best.toFixed(4)} species=${t.species}`);
  if (t.diversity) console.log('Mean compat', t.diversity.meanCompat);
}
```

## neat/neat.evolve.types.ts

### GenomeWithMetadata

Runtime interface for a genome carrying evolution metadata.

This mirrors the dynamic properties attached at runtime during evolution,
without pulling in the full Genome class to avoid circular dependencies.

### MultiObjectiveOptions

Multi-objective configuration block.

### MutationMethod

Mutation method descriptor used by runtime mutation hooks.

### NeatControllerForEvolution

NEAT controller subset used by evolve orchestrations.

### ObjectiveDescriptor

Objective descriptor for multi-objective evaluation.

### SpeciesHistoryRecord

Species history snapshot record used for telemetry/exports.

### SpeciesWithMetadata

Runtime interface for species metadata used in allocation and stats.

## neat/neat.harness.types.ts

Type helpers for test harnesses exercising Neat lineage behaviour.

### LineageTrackedNetwork

Network subtype that surfaces lineage metadata fields for assertions.

Example:

const lineageAware = child as LineageTrackedNetwork;
console.log(lineageAware._parents);

### NeatLineageHarness

Narrow Neat surface exposing lineage helper methods used in tests.

Example:

const helper: NeatLineageHarness = neat as NeatLineageHarness;
const child = helper.spawnFromParent(parent, 1);

### PhasedComplexityHarness

Minimal surface exposing phased complexity internals for testing.

## neat/neat.mutation.types.ts

Type definitions for NEAT mutation operations.
Extracted to avoid circular dependencies.

### ConnectionWithMetadata

Runtime interface for a connection within a genome.

### GenomeWithMetadata

Runtime interface for a genome with mutation-related metadata.
Avoids circular dependencies by defining only the properties accessed in mutation modules.

### MutationMethod

Runtime interface for a mutation method descriptor.

### NeatControllerForMutation

Runtime interface for the NEAT controller used in mutation operations.
Avoids circular dependencies by defining only properties accessed in mutation modules.

### NodeSplitRecord

Runtime interface for node-split innovation records.

### NodeWithMetadata

Runtime interface for a node within a genome.

### OperatorStats

Runtime interface for operator statistics tracking.

## neat/neat.telemetry.types.ts

### OperatorStatsMap

Operator stats map shape for telemetry extraction.

### TelemetryBufferContext

Minimal telemetry buffer context shape.

### TelemetryCoreFields

Core telemetry field keys used by selection helpers.

### TelemetryDiversityOptions

Diversity telemetry options for sampling and novelty defaults.

### TelemetryEntryRecord

Telemetry entry shape used for constructing snapshots.

### TelemetryGenome

Minimal genome shape used by telemetry helpers.

### TelemetrySelectContext

Minimal telemetry selection context shape.

### TelemetryStreamOptions

Minimal telemetry stream options for streaming helpers.

## neat/neat.rng.ts

### exportRngState

```ts
exportRngState(
  host: RngHost,
): number | undefined
```

Export the current RNG state for persistence.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when not set.

### getOrCreateRng

```ts
getOrCreateRng(
  host: RngHost,
): () => number
```

Return a cached RNG or create a deterministic xorshift RNG when absent.

The helper respects a user-provided RNG at `options.rng` when present.
Otherwise it seeds a xorshift32 RNG using the current time and population
size, guarding against the invalid zero seed.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

### importRngState

```ts
importRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Alias for restoring RNG state kept for compatibility with prior surface.

### restoreRngState

```ts
restoreRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Restore a previously captured RNG state.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### RNG_DEFAULT_SEED_FALLBACK

Fallback seed used when the derived seed would be zero (xorshift cannot use 0).

### RNG_NORMALIZATION_DIVISOR

Divisor used to normalize the 32-bit integer state into [0, 1).

### RNG_POPULATION_OFFSET

Minimum population offset added before scrambling to avoid zero seeds.

### RNG_SHIFT_LEFT_PRIMARY

Bit-shift values for the xorshift32 variant.

### RNG_SHIFT_LEFT_SECONDARY

### RNG_SHIFT_RIGHT_PRIMARY

### RNG_TIME_SCRAMBLE_CONSTANT

Constants used by the deterministic xorshift RNG helper.

### RngHost

Minimal host surface required by the RNG utilities.

### sampleRandomSequence

```ts
sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce a sequence of random samples using the host RNG.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

### snapshotRngState

```ts
snapshotRngState(
  host: RngHost,
): number | undefined
```

Snapshot the current RNG state for deterministic replay.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.

## neat/neat.cache.ts

### invalidateGenomeCaches

```ts
invalidateGenomeCaches(
  genomeCandidate: unknown,
): void
```

Invalidate per-genome caches used across compatibility and forward-pass logic.

Parameters:
- `genomeCandidate` - - Genome object whose caches should be cleared.

## neat/neat.compat.ts

### _compatibilityDistance

```ts
_compatibilityDistance(
  genomeA: GenomeLike,
  genomeB: GenomeLike,
): number
```

Compute the NEAT compatibility distance between two genomes (networks).

The compatibility distance is used for speciation in NEAT. It combines the
number of excess and disjoint genes with the average weight difference of
matching genes. A generation-scoped cache is used to avoid recomputing the
same pair distances repeatedly within a generation.

Formula:
distance = (c1 * excess + c2 * disjoint) / N + c3 * avgWeightDiff
where N = max(number of genes in genomeA, number of genes in genomeB)
and c1,c2,c3 are coefficients provided in `this.options`.

Example:
const d = _compatibilityDistance.call(neatInstance, genomeA, genomeB);
if (d < neatInstance.options.compatibilityThreshold) { // same species }

Parameters:
- `this` - - The NEAT instance / context which holds generation, options, and caches.
- `genomeA` - - First genome (network) to compare. Expected to expose `_id` and `connections`.
- `genomeB` - - Second genome (network) to compare. Expected to expose `_id` and `connections`.

Returns: A numeric compatibility distance; lower means more similar.

### _fallbackInnov

```ts
_fallbackInnov(
  connection: ConnectionLike,
): number
```

Generate a deterministic fallback innovation id for a connection when the
connection does not provide an explicit innovation number.

This function encodes the (from.index, to.index) pair into a single number
by multiplying the `from` index by a large base and adding the `to` index.
The large base reduces collisions between different pairs and keeps the id
stable and deterministic across runs. It is intended as a fallback only —
explicit innovation numbers (when present) should be preferred.

Example:
const conn = { from: { index: 2 }, to: { index: 5 } };
const id = _fallbackInnov.call(neatContext, conn); // 200005

Notes:
- Not globally guaranteed unique, but deterministic for the same indices.
- Useful during compatibility checks when some connections are missing innovation ids.

Parameters:
- `this` - - The NEAT instance / context (kept for symmetry with other helpers).
- `connection` - - Connection object expected to contain `from.index` and `to.index`.

Returns: A numeric innovation id derived from the (from, to) index pair.

## neat/neat.evolve.ts

### evolve

```ts
evolve(): Promise<default>
```

Run a single evolution step for this NEAT population.

This method performs a full generation update: evaluation (if needed),
adaptive hooks, speciation and fitness sharing, multi-objective
processing, elitism/provenance, offspring allocation (within or without
species), mutation, pruning, and telemetry recording. It mutates the
controller state (`this.population`, `this.generation`, and telemetry
caches) and returns a copy of the best discovered `Network` for the
generation.

Important side-effects:
- Replaces `this.population` with the newly constructed generation.
- Increments `this.generation`.
- May register or remove dynamic objectives via adaptive controllers.

Example:
// assuming `neat` is an instance with configured population/options
await neat.evolve();
console.log('generation:', neat.generation);

Returns: a deep-cloned Network representing the best genome
 in the previous generation (useful for evaluation)

### EVOLVE_AUTO_COMPAT_ADJUST_RATE

Default auto-compatibility adjust rate.

### EVOLVE_AUTO_COMPAT_MAX_COEFF

Default maximum compatibility coefficient.

### EVOLVE_AUTO_COMPAT_MIN_COEFF

Default minimum compatibility coefficient.

### EVOLVE_AUTO_COMPAT_RANDOM_SCALE

Random scale factor used when auto-compatibility has zero error.

### EVOLVE_AUTO_COMPAT_TARGET_MIN

Minimum target species when auto-tuning compatibility coefficients.

### EVOLVE_AUTO_ENTROPY_ADD_AT

Default auto-entropy activation generation.

### EVOLVE_CROSS_SPECIES_GUARD_LIMIT

Guard limit for cross-species mating selection retries.

### EVOLVE_DEFAULT_EPSILON_ADJUST

Default adjustment step for dominance epsilon.

### EVOLVE_DEFAULT_EPSILON_COOLDOWN

Default cooldown (generations) between epsilon adjustments.

### EVOLVE_DEFAULT_EPSILON_MAX

Default maximum dominance epsilon.

### EVOLVE_DEFAULT_EPSILON_MIN

Default minimum dominance epsilon.

### EVOLVE_GLOBAL_STAGNATION_REPLACE_FRACTION

Fraction of population to replace during global stagnation injection.

### EVOLVE_MIN_OFFSPRING_DEFAULT

Default minimum offspring per species.

### EVOLVE_OLD_MULTIPLIER_DEFAULT

Default old species fitness multiplier.

### EVOLVE_OLD_THRESHOLD_DEFAULT

Default old species threshold (generations).

### EVOLVE_PARETO_ARCHIVE_MAX

Maximum number of Pareto archive snapshots to retain.

### EVOLVE_PRUNE_RANGE_EPS_DEFAULT

Default inactive objective range epsilon.

### EVOLVE_PRUNE_WINDOW_DEFAULT

Default prune window (generations) for inactive objectives.

### EVOLVE_REENABLE_DELTA_SCALE

Scale factor for re-enable probability adjustment.

### EVOLVE_REENABLE_MAX

Maximum re-enable probability.

### EVOLVE_REENABLE_MIN

Minimum re-enable probability.

### EVOLVE_REENABLE_MIN_SAMPLES

Minimum samples required to adjust re-enable probability.

### EVOLVE_REENABLE_TARGET

Target re-enable success ratio.

### EVOLVE_SPECIES_HISTORY_MAX

Maximum number of species history snapshots to retain.

### EVOLVE_SURVIVAL_THRESHOLD_DEFAULT

Default survival threshold for parent selection.

### EVOLVE_TARGET_FRONT_LOWER_RATIO

Lower ratio threshold for Pareto front size vs target.

### EVOLVE_TARGET_FRONT_MIN

Minimum target front size used for adaptive epsilon tuning.

### EVOLVE_TARGET_FRONT_UPPER_RATIO

Upper ratio threshold for Pareto front size vs target.

### EVOLVE_YOUNG_MULTIPLIER_DEFAULT

Default young species fitness multiplier.

### EVOLVE_YOUNG_THRESHOLD_DEFAULT

Default young species threshold (generations).

## neat/neat.export.ts

### exportPopulation

```ts
exportPopulation(): GenomeJSON[]
```

Export the current population (array of genomes) into plain JSON objects.
Each genome is converted via its `toJSON()` method. You can persist this
result (e.g. to disk, a database, or localStorage) and later rehydrate it
with {@link importPopulation}.

Why export population only? Sometimes you want to snapshot *just* the set of
candidate solutions (e.g. for ensemble evaluation) without freezing the
innovation counters or hyper‑parameters.

Example:

```ts
// Assuming `neat` is an instance exposing this helper
const popSnapshot = neat.exportPopulation();
fs.writeFileSync('population.json', JSON.stringify(popSnapshot, null, 2));
```

Returns: Array of genome JSON objects.

### exportState

```ts
exportState(): NeatStateJSON
```

Convenience helper that returns a full evolutionary snapshot: both NEAT meta
information and the serialized population array. Use this when you want a
truly *pause‑and‑resume* capability including innovation bookkeeping.

Example:

```ts
const state = neat.exportState();
fs.writeFileSync('state.json', JSON.stringify(state));
// ...later / elsewhere...
const raw = JSON.parse(fs.readFileSync('state.json','utf8')) as NeatStateJSON;
const neat2 = Neat.importState(raw, fitnessFn); // identical evolutionary context
```

Returns: A  {@link NeatStateJSON} bundle containing meta + population.

### fromJSONImpl

```ts
fromJSONImpl(
  neatJSON: NeatMetaJSON,
  fitnessFunction: (network: GenomeWithSerialization) => number | Promise<number>,
): NeatControllerForExport
```

Static-style implementation that rehydrates a NEAT instance from previously
exported meta JSON produced by {@link toJSONImpl}. This does *not* restore a
population; callers typically follow up with `importPopulation` or use
{@link importStateImpl} for a complete restore.

Example:

```ts
const meta: NeatMetaJSON = JSON.parse(fs.readFileSync('neat-meta.json','utf8'));
const neat = Neat.fromJSONImpl(meta, fitnessFn); // empty population, same innovations
neat.importPopulation(popSnapshot); // optional
```

Parameters:
- `neatJSON` - Serialized meta (no population).
- `fitnessFunction` - Fitness callback used to construct the new instance.

Returns: Fresh NEAT instance with restored innovation history.

### GenomeJSON

JSON representation of an individual genome (network). The concrete shape is
produced by `Network#toJSON()` and re‑hydrated via `Network.fromJSON()`. We use
an open record signature here because the network architecture may evolve with
plugins / future features (e.g. CPPNs, substrate metadata, ONNX export tags).

### GenomeWithSerialization

Genome with toJSON serialization method.

### importPopulation

```ts
importPopulation(
  populationJSON: GenomeJSON[],
): Promise<void>
```

Import (replace) the current population from an array of serialized genomes.
This does not touch NEAT meta state (generation, innovations, etc.)—only the
population array and implied `popsize` are updated.

Example:

```ts
const populationData: GenomeJSON[] = JSON.parse(fs.readFileSync('population.json','utf8'));
neat.importPopulation(populationData); // population replaced
neat.evolve(); // continue evolving with new starting genomes
```

Edge cases handled:
- Empty array => becomes an empty population (popsize=0).
- Malformed entries will throw if `Network.fromJSON` rejects them.

Parameters:
- `populationJSON` - Array of serialized genome objects.

### importStateImpl

```ts
importStateImpl(
  stateBundle: NeatStateJSON,
  fitnessFunction: (network: GenomeWithSerialization) => number | Promise<number>,
): Promise<NeatControllerForExport>
```

Static-style helper that rehydrates a full evolutionary state previously
produced by {@link exportState}. Invoke this with the NEAT *class* (not an
instance) bound as `this`, e.g. `Neat.importStateImpl(bundle, fitnessFn)`.
It constructs a new NEAT instance using the meta data, then imports the
population (if present).

Safety & validation:
- Throws if the bundle is not an object.
- Silently skips population import if `population` is missing or not an array.

Example:

```ts
const bundle: NeatStateJSON = JSON.parse(fs.readFileSync('state.json','utf8'));
const neat = Neat.importStateImpl(bundle, fitnessFn);
neat.evolve();
```

Parameters:
- `stateBundle` - Full state bundle from  {@link exportState} .
- `fitnessFunction` - Fitness evaluation callback used for new instance.

Returns: Rehydrated NEAT instance ready to continue evolving.

### InnovationMapEntry

Connection innovation map entry.

### NeatConstructor

NEAT class constructor interface.

### NeatControllerForExport

NEAT controller interface for export operations.

### NeatMetaJSON

Serialized meta information describing a NEAT run, excluding the concrete
population genomes. This allows you to persist & resume experiment context
(innovation history, current generation, IO sizes, hyper‑parameters) without
committing to a particular population snapshot.

### NeatStateJSON

Top‑level bundle containing both NEAT meta information and the full array of
serialized genomes (population). This is what you get from `exportState()` and
feed into `importStateImpl()` to resume exactly where you left off.

### NetworkClass

Network class with static fromJSON method.

### toJSONImpl

```ts
toJSONImpl(): NeatMetaJSON
```

Serialize NEAT meta (excluding the mutable population) for persistence of
innovation history and experiment configuration. This is sufficient to
recreate a *blank* NEAT run at the same evolutionary generation with the
same innovation counters, enabling deterministic continuation when combined
later with a saved population.

Example:

```ts
const meta = neat.toJSONImpl();
fs.writeFileSync('neat-meta.json', JSON.stringify(meta));
// ... later ...
const metaLoaded = JSON.parse(fs.readFileSync('neat-meta.json','utf8')) as NeatMetaJSON;
const neat2 = Neat.fromJSONImpl(metaLoaded, fitnessFn); // empty population
```

## neat/neat.helpers.ts

Helper utilities that augment the core NEAT (NeuroEvolution of Augmenting Topologies)
implementation. These functions are kept separate from the main class so they can
be tree‑shaken when unused and independently documented for educational purposes.

The helpers focus on three core lifecycle operations:
1. Spawning children from an existing parent genome with mutation ("sexual" reproduction not handled here).
2. Registering externally created genomes so lineage & invariants remain consistent.
3. Creating the initial population pool (bootstrapping evolution) either from a seed
   network or by synthesizing fresh minimal networks.

All helpers expect to be invoked with a `this` context that matches `NeatLike`.
They intentionally use defensive try/catch blocks to avoid aborting broader
evolutionary runs when an individual genome operation fails; this mirrors the
tolerant/robust nature of many historical NEAT library implementations.

### addGenome

```ts
addGenome(
  genome: GenomeWithMetadata,
  parents: number[] | undefined,
): void
```

Register an externally constructed genome (e.g., deserialized, custom‑built,
or imported from another run) into the active population. Ensures lineage
metadata and structural invariants are consistent with internally spawned
genomes.

Defensive design: If invariant enforcement fails, the genome is still added
(best effort) so experiments remain reproducible and do not abort mid‑run.
Caller can optionally inspect or prune later during evaluation.

Parameters:
- `this` - Bound NEAT instance.
- `genome` - Genome / network object to insert. Mutated in place to add
internal metadata fields (`_id`, `_parents`, `_depth`, `_reenableProb`).
- `parents` - Optional explicit list of parent genome IDs (e.g., 2 parents
for crossover). If omitted, lineage metadata is left empty.

Example:

```ts
const imported = Network.fromJSON(saved);
neat.addGenome(imported, [parentA._id, parentB._id]);
```

### createPool

```ts
createPool(
  seedNetwork: GenomeWithMetadata | null,
): void
```

Create (or reset) the initial population pool for a NEAT run.

If a `seedNetwork` is supplied, every genome is a structural + weight clone
of that seed. This is useful for transfer learning or continuing evolution
from a known good architecture. When omitted, brand‑new minimal networks are
synthesized using the configured input/output sizes (and optional minimum
hidden layer size).

Design notes:
- Population size is derived from `options.popsize` (default 50).
- Each genome gets a unique sequential `_id` for reproducible lineage.
- When lineage tracking is enabled (`_lineageEnabled`), parent & depth fields
  are initialized for later analytics.
- Structural invariant checks are best effort. A single failure should not
  prevent other genomes from being created, hence broad try/catch blocks.

Parameters:
- `this` - Bound NEAT instance.
- `seedNetwork` - Optional prototype network to clone for every initial genome.

Example:

```ts
// Basic: create 50 fresh minimal networks
neat.createPool(null);

// Seeded: start with a known topology
const seed = new Network(neat.input, neat.output, { minHidden: 4 });
neat.createPool(seed);
```

### GenomeWithMetadata

Genome with NEAT-specific metadata and methods.

### MutationMethod

Mutation method with optional name.

### NeatControllerForHelpers

NEAT controller interface for helper functions.

### spawnFromParent

```ts
spawnFromParent(
  parentGenome: GenomeWithMetadata,
  mutateCount: number,
): Promise<GenomeWithMetadata>
```

Spawn (clone & mutate) a child genome from an existing parent genome.

The returned child is intentionally NOT auto‑inserted into the population;
call {@link addGenome} (or the class method wrapper) once you decide to
keep it. This separation allows callers to perform custom validation or
scoring heuristics before committing the child genome.

Evolutionary rationale:
- Cloning preserves the full topology & weights of the parent.
- A configurable number of mutation passes are applied sequentially; each
  pass may alter structure (add/remove nodes / connections) or weights.
- Lineage annotations (`_parents`, `_depth`) enable later analytics (e.g.,
  diversity statistics, genealogy visualization, pruning heuristics).

Robustness philosophy: individual mutation failures are silently ignored so
a single stochastic edge case (e.g., no valid structural mutation) does not
derail evolutionary progress.

Parameters:
- `this` - Bound NEAT instance (inferred when used as a method).
- `parentGenome` - Parent genome/network to clone. Must implement either
`clone()` OR a pair of `toJSON()` / static `fromJSON()` for deep copying.
- `mutateCount` - Number of sequential mutation operations to attempt; each
iteration chooses a mutation method using the instance's selection logic.
Defaults to 1 for conservative structural drift.

Returns: A new genome (unregistered) whose score is reset and whose lineage
metadata references the parent.

Example:

```ts
// Assume `neat` is an instance implementing NeatLike and `parent` is a genome in neat.population
const child = neat.spawnFromParent(parent, 3); // apply 3 mutation passes
// Optionally inspect / filter the child before adding
neat.addGenome(child, [parent._id]);
```

## neat/neat.lineage.ts

Lineage / ancestry analysis helpers for NEAT populations.

These utilities were migrated from the historical implementation inside `src/neat.ts`
to keep core NEAT logic lean while still exposing educational metrics for users who
want to introspect evolutionary diversity.

Glossary:
 - Genome: An individual network encoding (has a unique `_id` and optional `_parents`).
 - Ancestor Window: A shallow breadth‑first window (default depth = 4) over the lineage graph.
 - Jaccard Distance: 1 - |A ∩ B| / |A ∪ B|, measuring dissimilarity between two sets.

### buildAnc

```ts
buildAnc(
  genome: GenomeLike,
): Set<number>
```

Build the (shallow) ancestor ID set for a single genome using breadth‑first traversal.

Traversal Strategy:
1. Seed queue with the genome's parent IDs (depth = 1).
2. Repeatedly dequeue, record its ID, and enqueue its parents with incremented depth.
3. Stop exploring a branch once the configured depth window is exceeded.

This bounded BFS gives a quick, memory‑friendly approximation of a genome's lineage neighborhood
that works well for diversity/uniqueness metrics without the expense of full historical graphs.

Edge Cases:
 - Missing or empty `_parents` array ⇒ returns an empty set.
 - Orphan parent IDs (not found in population) are still added (their ID), but no further expansion occurs.

Complexity (worst case): O(B^D) where B is average branching factor of parent links (usually <= 2)
and D = ANCESTOR_DEPTH_WINDOW (default 4) – so effectively constant for typical NEAT usage.

Parameters:
- `this` - NEAT / evolutionary context; must provide `population` (array) for ID lookups.
- `genome` - Genome whose shallow ancestor set you want to compute.

Returns: A Set of numeric ancestor IDs (deduplicated).

Example:

// Assuming `neat` is your NEAT instance and `g` a genome inside `neat.population`:
import { buildAnc } from 'neataptic';
const ancestorIds = buildAnc.call(neat, g);
console.log([...ancestorIds]); // -> e.g. [12, 4, 9]

### computeAncestorUniqueness

```ts
computeAncestorUniqueness(): number
```

Compute an "ancestor uniqueness" diversity metric for the current population.

The metric = mean Jaccard distance between shallow ancestor sets of randomly sampled genome pairs.
A higher value indicates that individuals trace back to more distinct recent lineages (i.e. less
overlap in their ancestor windows), while a lower value indicates convergence toward similar ancestry.

Why Jaccard Distance? It is scale‑independent: adding unrelated ancestors to both sets simultaneously
does not change the proportion of shared ancestry, and distance stays within [0,1].

Sampling Strategy:
 - Uniformly sample up to N = min(30, populationPairs) distinct unordered pairs (with replacement on pair selection, but indices are adjusted to avoid self‑pairs).
 - For each pair, construct ancestor sets via `buildAnc` and accumulate their Jaccard distance.
 - Return the average (rounded to 3 decimal places) or 0 if insufficient samples.

Edge Cases:
 - Population < 2 ⇒ returns 0 (cannot form pairs).
 - Both ancestor sets empty ⇒ pair skipped (no information about uniqueness).

Performance: O(S * W) where S is sampled pair count (≤ 30) and W is bounded ancestor set size
(kept small by the depth window). This is intentionally lightweight for per‑generation telemetry.

Parameters:
- `this` - NEAT context (`population` and `_getRNG` must exist).

Returns: Mean Jaccard distance in [0,1]. Higher ⇒ more lineage uniqueness / diversity.

Example:

import { computeAncestorUniqueness } from 'neataptic';
// inside an evolutionary loop, with `neat` as your NEAT instance:
const uniqueness = computeAncestorUniqueness.call(neat);
console.log('Ancestor uniqueness:', uniqueness); // e.g. 0.742

### GenomeLike

Minimal shape assumed for a genome inside the NEAT population. Additional properties are
intentionally left open (index signature) because user implementations may extend genomes.

### NeatLineageContext

Expected `this` context for lineage helpers (a subset of the NEAT instance).

## neat/neat.pruning.ts

### applyAdaptivePruning

```ts
applyAdaptivePruning(): void
```

Adaptive pruning controller.

This function monitors a population-level metric (average nodes or
average connections) and adjusts a global pruning level so the
population converges to a target sparsity automatically.

It updates `this._adaptivePruneLevel` on the Neat instance and calls
each genome's `pruneToSparsity` with the new level when adjustment
is required.

Example:

```ts
// options.adaptivePruning = { enabled: true, metric: 'connections', targetSparsity: 0.6 }
neat.applyAdaptivePruning();
```

### applyEvolutionPruning

```ts
applyEvolutionPruning(): void
```

Apply evolution-time pruning to the current population.

This method is intended to be called from the evolve loop. It reads
pruning parameters from `this.options.evolutionPruning` and, when
appropriate for the current generation, instructs each genome to
prune its connections/nodes to reach a target sparsity.

The pruning target can be ramped in over a number of generations so
sparsification happens gradually instead of abruptly.

Example (in a Neat instance):
```ts
// options.evolutionPruning = { startGeneration: 10, targetSparsity: 0.5 }
neat.applyEvolutionPruning();
```

Notes for docs:
- `method` is passed through to each genome's `pruneToSparsity` and
  commonly is `'magnitude'` (prune smallest-weight connections first).
- This function performs no changes if pruning options are not set or
  the generation is before `startGeneration`.

## neat/neat.species.ts

### getSpeciesHistory

```ts
getSpeciesHistory(): SpeciesHistoryEntry[]
```

Retrieve the recorded species history across generations.

Each entry in the returned array corresponds to a recorded generation and
contains a snapshot of statistics for every species at that generation.
This is useful for plotting species sizes over time, tracking innovation
spread, or implementing population-level diagnostics.

The shape of each entry is defined by `SpeciesHistoryEntry` in the public
types. When `options.speciesAllocation.extendedHistory` is enabled the
library attempts to include additional metrics such as `innovationRange`
and `enabledRatio`. When those extended metrics are missing they are
computed lazily from a representative genome to ensure historical data is
still useful for analysis.

Example:

```ts
const history = neat.getSpeciesHistory();
// history => [{ generation: 0, stats: [{ id:1, size:10, innovationRange:5, enabledRatio:0.9 }, ...] }, ...]
```

Notes for documentation:
- The function tries to avoid heavy computation. Extended metrics are
  computed only when explicitly requested via options.
- Computed extended metrics are conservative fallbacks; they use the
  available member connections and a fallback innovation extractor when
  connection innovation IDs are not present.

Returns: Array of generation-stamped species statistic snapshots.

### getSpeciesStats

```ts
getSpeciesStats(): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Get lightweight per-species statistics for the current population.

This method intentionally returns a small, immutable-friendly summary per
species rather than exposing internal member lists. This avoids accidental
mutation of the library's internal state while still providing useful
telemetry for UIs, dashboards, or logging.

Example:

```ts
const stats = neat.getSpeciesStats();
// stats => [{ id: 1, size: 12, bestScore: 0.85, lastImproved: 42 }, ...]
```

Success criteria:
- Returns an array of objects each containing `id`, `size`, `bestScore`,
  and `lastImproved`.
- Does not expose or return references to internal member arrays.

Returns: Array of per-species summaries suitable for reporting.

## neat/neat.adaptive.ts

### ANNEAL_BASELINE_GENERATIONS

Baseline generations for annealing progress.

### ANNEAL_PROGRESS_MAX

Maximum progress ratio used in annealing.

### applyAdaptiveMutation

```ts
applyAdaptiveMutation(): void
```

Self-adaptive per-genome mutation tuning.

This function implements several strategies to adjust each genome's
internal mutation rate (`g._mutRate`) and optionally its mutation
amount (`g._mutAmount`) over time. Strategies include:
- `twoTier`: push top and bottom halves in opposite directions to
  create exploration/exploitation balance.
- `exploreLow`: preferentially increase mutation for lower-scoring
  genomes to promote exploration.
- `anneal`: gradually reduce mutation deltas over time.

The method reads `this.options.adaptiveMutation` for configuration
and mutates genomes in-place.

Example:

// configuration example:
// options.adaptiveMutation = { enabled: true, initialRate: 0.5, adaptEvery: 1, strategy: 'twoTier', minRate: 0.01, maxRate: 1 }
engine.applyAdaptiveMutation();

### applyAncestorUniqAdaptive

```ts
applyAncestorUniqAdaptive(): void
```

Adaptive adjustments based on ancestor uniqueness telemetry.

This helper inspects the most recent telemetry lineage block (if
available) for an `ancestorUniq` metric indicating how unique
ancestry is across the population. If ancestry uniqueness drifts
outside configured thresholds, the method will adjust either the
multi-objective dominance epsilon (if `mode === 'epsilon'`) or the
lineage pressure strength (if `mode === 'lineagePressure'`).

Typical usage: keep population lineage diversity within a healthy
band. Low ancestor uniqueness means too many genomes share ancestors
(risking premature convergence); high uniqueness might indicate
excessive divergence.

Example:

// Adjusts `options.multiObjective.dominanceEpsilon` when configured
engine.applyAncestorUniqAdaptive();

### applyComplexityBudget

```ts
applyComplexityBudget(): void
```

Apply complexity budget scheduling to the evolving population.

This routine updates `this.options.maxNodes` (and optionally
`this.options.maxConns`) according to a configured complexity budget
strategy. Two modes are supported:

- `adaptive`: reacts to recent population improvement (or stagnation)
  by increasing or decreasing the current complexity cap using
  heuristics such as slope (linear trend) of recent best scores,
  novelty, and configured increase/stagnation factors.
- `linear` (default behaviour when not `adaptive`): linearly ramps
  the budget from `maxNodesStart` to `maxNodesEnd` over a horizon.

Internal state used/maintained on the `this` object:
- `_cbHistory`: rolling window of best scores used to compute trends.
- `_cbMaxNodes`: current complexity budget for nodes.
- `_cbMaxConns`: current complexity budget for connections (optional).

The method is intended to be called on the NEAT engine instance with
`this` bound appropriately (i.e. a NeatapticTS `Neat`-like object).

Returns: Updates `this.options.maxNodes` and possibly
`this.options.maxConns` in-place; no value is returned.

Example:

// inside a training loop where `engine` is your Neat instance:
engine.applyComplexityBudget();
// engine.options.maxNodes now holds the adjusted complexity cap

### applyMinimalCriterionAdaptive

```ts
applyMinimalCriterionAdaptive(): void
```

Apply adaptive minimal criterion (MC) acceptance.

This method maintains an MC threshold used to decide whether an
individual genome is considered acceptable. It adapts the threshold
based on the proportion of the population that meets the current
threshold, trying to converge to a target acceptance rate.

Behavior summary:
- Initializes `_mcThreshold` from configuration if undefined.
- Computes the proportion of genomes with score >= threshold.
- Adjusts threshold multiplicatively by `adjustRate` to move the
  observed proportion towards `targetAcceptance`.
- Sets `g.score = 0` for genomes that fall below the final threshold
  — effectively rejecting them from selection.

Example:

// Example config snippet used by the engine
// options.minimalCriterionAdaptive = { enabled: true, initialThreshold: 0.1, targetAcceptance: 0.5, adjustRate: 0.1 }
engine.applyMinimalCriterionAdaptive();

### applyOperatorAdaptation

```ts
applyOperatorAdaptation(): void
```

Decay operator adaptation statistics (success/attempt counters).

Many adaptive operator-selection schemes keep running tallies of how
successful each operator has been. This helper applies an exponential
moving-average style decay to those counters so older outcomes
progressively matter less.

The `_operatorStats` map on `this` is expected to contain values of
the shape `{ success: number, attempts: number }` keyed by operator
id/name.

Example:

engine.applyOperatorAdaptation();

### applyPhasedComplexity

```ts
applyPhasedComplexity(): void
```

Toggle phased complexity mode between 'complexify' and 'simplify'.

Phased complexity supports alternating periods where the algorithm
is encouraged to grow (complexify) or shrink (simplify) network
structures. This can help escape local minima or reduce bloat.

The current phase and its start generation are stored on `this` as
`_phase` and `_phaseStartGeneration` so the state persists across
generations.

Returns: Mutates `this._phase` and `this._phaseStartGeneration`.

Example:

// Called once per generation to update the phase state
engine.applyPhasedComplexity();

### DEFAULT_ADAPT_EVERY

Default adapt-every cadence for adaptive mutation.

### DEFAULT_ANCESTOR_UNIQ_ADJUST

Default adjustment magnitude for uniqueness nudges.

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

Default cooldown (generations) for ancestor-uniqueness adjustments.

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

Default upper bound for acceptable ancestor uniqueness.

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

Default lower bound for acceptable ancestor uniqueness.

### DEFAULT_INITIAL_MUTATION_RATE

Default initial mutation rate used for balance checks.

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

Default lineage pressure strength when initializing the option.

### DEFAULT_MAX_MUTATION_AMOUNT

Default maximum mutation amount.

### DEFAULT_MAX_MUTATION_RATE

Default maximum per-genome mutation rate.

### DEFAULT_MIN_MUTATION_AMOUNT

Default minimum mutation amount.

### DEFAULT_MIN_MUTATION_RATE

Default minimum per-genome mutation rate.

### DEFAULT_MUTATION_AMOUNT

Default mutation amount when genome value is missing.

### DEFAULT_MUTATION_AMOUNT_SIGMA

Default mutation amount sigma for perturbations.

### DEFAULT_MUTATION_SIGMA

Default mutation sigma for adaptive mutation.

### EXPLORE_LOW_DECREASE_MULTIPLIER

Multiplicative decay for explore-low strategy (top half).

### EXPLORE_LOW_INCREASE_MULTIPLIER

Multiplicative boost for explore-low strategy (bottom half).

### HALF_INDEX_DIVISOR

Divisor used to split populations in half.

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

Multiplier when decreasing lineage pressure strength.

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

Multiplier when increasing lineage pressure strength.

### MUTATION_SIGMA_SCALE

Scale applied to mutation sigma for perturbations.

### MUTATION_STRATEGY_ANNEAL

Strategy identifier for annealed mutation.

### MUTATION_STRATEGY_EXPLORE_LOW

Strategy identifier for explore-low mutation.

### MUTATION_STRATEGY_TWO_TIER

Strategy identifier for two-tier mutation.

### NeatLikeWithAdaptive

Minimal interface for NEAT instances with adaptive features.
Exported for use in tests and type-safe function calls.

### RNG_CENTER_OFFSET

Random offset for signed deltas.

### RNG_SPREAD_MULTIPLIER

Random range multiplier for signed deltas.

## neat/neat.evaluate.ts

### AUTO_COEFF_ADJUST_DEFAULT

Default adjustment rate for auto distance coefficient tuning.

### AUTO_COEFF_MAX_DEFAULT

Default maximum coefficient for auto distance coefficient tuning.

### AUTO_COEFF_MIN_DEFAULT

Default minimum coefficient for auto distance coefficient tuning.

### COMPAT_MAX_THRESHOLD_DEFAULT

Default maximum compatibility threshold.

### COMPAT_MIN_THRESHOLD_DEFAULT

Default minimum compatibility threshold.

### COMPAT_THRESHOLD_DEFAULT

Default compatibility threshold when not provided.

### DISTANCE_COEFF_DEFAULT

Default coefficient value when not provided.

### ENTROPY_ADJUST_DEFAULT

Default adjustment rate for compatibility tuning.

### ENTROPY_DEADBAND_DEFAULT

Default deadband for compatibility tuning.

### ENTROPY_TARGET_DEFAULT

Default target entropy for compatibility tuning.

### ENTROPY_VAR_ADJUST_DEFAULT

Default adjustment rate for entropy sharing.

### ENTROPY_VAR_HIGH_BAND

Upper band multiplier for entropy variance tuning.

### ENTROPY_VAR_LOW_BAND

Lower band multiplier for entropy variance tuning.

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

Default maximum sigma for entropy sharing.

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

Default minimum sigma for entropy sharing.

### ENTROPY_VAR_TARGET_DEFAULT

Default target variance for entropy sharing.

### evaluate

```ts
evaluate(): Promise<void>
```

Evaluate the population or population-wide fitness delegate.

This function mirrors the legacy `evaluate` behaviour used by NeatapticTS
but adds documentation and clearer local variable names for readability.

Top-level responsibilities (method steps descriptions):
1) Run fitness either on each genome or once for the population depending
   on `options.fitnessPopulation`.
2) Optionally clear genome internal state before evaluation when
   `options.clear` is set.
3) After scoring, apply optional novelty blending using a user-supplied
   descriptor function. Novelty is blended into scores using a blend
   factor and may be archived.
4) Apply several adaptive tuning behaviors (entropy-sharing, compatibility
   threshold tuning, auto-distance coefficient tuning) guarded by options.
5) Trigger light-weight speciation when speciation-related controller
   options are enabled so tests that only call evaluate still exercise
   threshold tuning.

Example usage:
// await evaluate.call(controller); // where controller has `population`, `fitness` etc.

Returns: Promise<void> resolves after evaluation and adaptive updates complete.

### NOVELTY_ARCHIVE_CAP

Maximum number of entries stored in the novelty archive.

### NOVELTY_DEFAULT_BLEND

Default blend factor for novelty vs. fitness.

### NOVELTY_DEFAULT_NEIGHBORS

Default neighbor count for novelty calculation.

### VARIANCE_DECREASE_THRESHOLD

Variance decrease threshold multiplier.

### VARIANCE_INCREASE_THRESHOLD

Variance increase threshold multiplier.

## neat/neat.mutation.ts

### DEFAULT_CONNECTION_WEIGHT

Default connection weight used for bootstrap and split in-edges.

### DEFAULT_GENE_ID

Default gene id value when a node has no gene id.

### DEFAULT_INNOVATION_ID

Default innovation id value when a connection has none.

### ensureMinHiddenNodes

```ts
ensureMinHiddenNodes(
  network: GenomeWithMetadata,
  multiplierOverride: number | undefined,
): Promise<void>
```

Ensure the network has a minimum number of hidden nodes and connectivity.

### ensureNoDeadEnds

```ts
ensureNoDeadEnds(
  network: GenomeWithMetadata,
): void
```

Ensure there are no dead-end nodes (input/output isolation) in the network.

### mutate

```ts
mutate(): Promise<void>
```

Mutate every genome in the population according to configured policies.

This is the high-level mutation driver used by NeatapticTS. It iterates the
current population and, depending on the configured mutation rate and
(optional) adaptive mutation controller, applies one or more mutation
operators to each genome.

Educational notes:
- Adaptive mutation allows per-genome mutation rates/amounts to evolve so
  that successful genomes can reduce or increase plasticity over time.
- Structural mutations (ADD_NODE, ADD_CONN, etc.) may update global
  innovation bookkeeping; this function attempts to reuse specialized
  helper routines that preserve innovation ids across the population.

Example:

```ts
// called on a Neat instance after a generation completes
neat.mutate();
```

### mutateAddConnReuse

```ts
mutateAddConnReuse(
  genome: GenomeWithMetadata,
): void
```

Add a connection between two previously unconnected nodes, reusing a
stable innovation id per unordered node pair when possible.

Notes on behavior:
- The search space consists of node pairs (from, to) where `from` is not
  already projecting to `to` and respects the input/output ordering used by
  the genome representation.
- When a historical innovation exists for the unordered pair, the
  previously assigned innovation id is reused to keep different genomes
  compatible for downstream crossover and speciation.

Steps:
- Build a list of all legal (from,to) pairs that don't currently have a
  connection.
- Prefer pairs which already have a recorded innovation id (reuse
  candidates) to maximize reuse; otherwise use the full set.
- If the genome enforces acyclicity, simulate whether adding the connection
  would create a cycle; abort if it does.
- Create the connection and set its innovation id, either from the
  historical table or by allocating a new global innovation id.

Parameters:
- `genome` - - genome to modify in-place

### mutateAddNodeReuse

```ts
mutateAddNodeReuse(
  genome: GenomeWithMetadata,
): Promise<void>
```

Split a randomly chosen enabled connection and insert a hidden node.

This routine attempts to reuse a historical "node split" innovation record
so that identical splits across different genomes share the same
innovation ids. This preservation of innovation information is important
for NEAT-style speciation and genome alignment.

Method steps (high-level):
- If the genome has no connections, connect an input to an output to
  bootstrap connectivity.
- Filter enabled connections and choose one at random.
- Disconnect the chosen connection and either reuse an existing split
  innovation record or create a new hidden node + two connecting
  connections (in->new, new->out) assigning new innovation ids.
- Insert the newly created node into the genome's node list at the
  deterministic position to preserve ordering for downstream algorithms.

Example:

```ts
neat._mutateAddNodeReuse(genome);
```

Parameters:
- `genome` - - genome to modify in-place

### selectMutationMethod

```ts
selectMutationMethod(
  genome: GenomeWithMetadata,
  rawReturnForTest: boolean,
): Promise<MutationMethod | MutationMethod[] | null>
```

Select a mutation method respecting structural constraints and adaptive controllers.
Mirrors legacy implementation from `neat.ts` to preserve test expectations.
`rawReturnForTest` retains historical behavior where the full FFW array is
returned for identity checks in tests.

## neat/neat.constants.ts

Shared numerical / heuristic constants for NEAT modules.

Keeping these in a single dependency‑free module avoids scattering magic
numbers and simplifies tuning while refactoring.

### EPSILON

Numerical stability offset used inside log / division expressions.

### EXTRA_CONNECTION_PROBABILITY

Probability of performing an opportunistic extra ADD_CONN mutation.

### NORM_EPSILON

Epsilon used in normalization layers (variance smoothing).

### PROB_EPSILON

Extremely small epsilon for log/ratio protections in probability losses.

## neat/neat.diversity.ts

### computeDiversityStats

```ts
computeDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined
```

Compute diversity statistics for a NEAT population.
This is a pure helper used by reporting and diagnostics. It intentionally
samples pairwise computations to keep cost bounded for large populations.

Notes for documentation:
- Lineage metrics rely on genomes exposing a numeric `_depth` property.
- Compatibility distances are computed via the provided compatComputer
  which mirrors legacy code and may use historical marker logic.

Parameters:
- `population` - - array of genome-like objects (nodes, connections, optional _depth)
- `compatibilityComputer` - - object exposing _compatibilityDistance(a,b)

Returns: DiversityStats object with all computed aggregates, or undefined if input empty

Example:

const stats = computeDiversityStats(population, compatImpl);
console.log(`Mean nodes: ${stats?.meanNodes}`);

### DiversityStats

Diversity statistics returned by computeDiversityStats.
Each field represents an aggregate metric for a NEAT population.

### MAX_COMPATIBILITY_SAMPLE

Maximum population sample size for compatibility comparisons.

### MAX_LINEAGE_PAIR_SAMPLE

Maximum lineage sample size for pairwise depth comparisons.

### structuralEntropy

```ts
structuralEntropy(
  graph: default,
): number
```

Compute the Shannon-style entropy of a network's out-degree distribution.
This is a lightweight, approximate structural dispersion metric used to
characterise how 'spread out' connections are across nodes.

Educational note: structural entropy here is simply H = -sum(p_i log p_i)
over the normalized out-degree histogram. It does not measure information
content of weights or dynamics, but provides a quick structural fingerprint.

Example:

// network-like object shape expected by this helper:
// const net = { nodes: [ { connections: { out: [] } }, ... ] };
// const h = structuralEntropy(net);

## neat/neat.selection.ts

### DEFAULT_POWER

Default power exponent for POWER selection when none is configured.

### DEFAULT_SCORE

Default score when a genome has no explicit score.

### DEFAULT_TOURNAMENT_PROBABILITY

Default tournament win probability when none is configured.

### DEFAULT_TOURNAMENT_SIZE

Default tournament size when none is configured.

### FIRST_INDEX

Index of the first element in an array.

### getAverage

```ts
getAverage(): number
```

Compute the average (mean) fitness across the population.

If genomes have not been evaluated yet this will call `evaluate()` so
that scores exist. Missing scores are treated as 0.

Example:
const avg = neat.getAverage();
console.log(`Average fitness: ${avg}`);

Returns: The mean fitness as a number.

### getFittest

```ts
getFittest(): GenomeWithScore
```

Return the fittest genome in the population.

This will trigger an `evaluate()` if genomes have not been scored yet, and
will ensure the population is sorted so index 0 contains the fittest.

Example:
const best = neat.getFittest();
console.log(best.score);

Returns: The genome object judged to be the fittest (highest score).

### getParent

```ts
getParent(): GenomeWithScore
```

Select a parent genome according to the configured selection strategy.

Supported strategies (via `options.selection.name`):
- 'POWER'              : biased power-law selection (exploits best candidates)
- 'FITNESS_PROPORTIONATE': roulette-wheel style selection proportional to fitness
- 'TOURNAMENT'         : pick N random competitors and select the best with probability p

This function intentionally makes no changes to the population except in
the POWER path where a quick sort may be triggered to ensure descending
order.

Examples:
// POWER selection (higher power => more exploitation)
neat.options.selection = { name: 'POWER', power: 2 };
const parent = neat.getParent();

// Tournament selection (size 3, 75% probability to take top of tournament)
neat.options.selection = { name: 'TOURNAMENT', size: 3, probability: 0.75 };
const parent2 = neat.getParent();

Returns: A genome object chosen as the parent according to the selection strategy

### INITIAL_CUMULATIVE_FITNESS

Initial cumulative fitness value for threshold scans.

### INITIAL_MOST_NEGATIVE_SCORE

Initial most-negative score sentinel for fitness scans.

### INITIAL_TOTAL_FITNESS

Initial total fitness accumulator value.

### LAST_ELEMENT_INDEX

Index used with `at()` to access the last element.

### LAST_INDEX_OFFSET

Offset for retrieving the last element via length arithmetic.

### LOOP_INDEX_INCREMENT

Step size for index-based loops.

### SECOND_INDEX

Index of the second element in an array.

### sort

```ts
sort(): void
```

Sorts the internal population in place by descending fitness.

This method mutates the `population` array on the Neat instance so that
the genome with the highest `score` appears at index 0. It treats missing
scores as 0.

Example:
const neat = new Neat(...);
neat.sort();
console.log(neat.population[0].score); // highest score

Notes for documentation generators: this is a small utility used by many
selection and evaluation routines; it intentionally sorts in-place for
performance and to preserve references to genome objects.

## neat/neat.telemetry.ts

### applyTelemetrySelect

```ts
applyTelemetrySelect(
  entry: Record<string, unknown>,
): Record<string, unknown>
```

Apply a telemetry selection whitelist to a telemetry entry.

This helper inspects a per-instance Set of telemetry keys stored at
`this._telemetrySelect`. If present, only keys included in the set are
retained on the produced entry. Core fields (generation, best score and
species count) are always preserved.

Example:

Parameters:
- `entry` - - Raw telemetry object to be filtered in-place.

Returns: The filtered telemetry object (same reference as input).

Example:

// keep only 'gen', 'best', 'species' and 'diversity' fields
neat._telemetrySelect = new Set(['diversity']);
applyTelemetrySelect.call(neat, entry);

### buildTelemetryEntry

```ts
buildTelemetryEntry(
  fittest: Record<string, unknown>,
): TelemetryEntry
```

Build a comprehensive telemetry entry for the current generation.

The returned object contains a snapshot of population statistics, multi-
objective front sizes, operator statistics, lineage summaries and optional
complexity/performance metrics depending on configured telemetry options.

This function intentionally mirrors the legacy in-loop telemetry construction
to preserve behavior relied upon by tests and consumers.

Example:

Parameters:
- `fittest` - - The currently fittest genome (used to report `best` score).

Returns: A TelemetryEntry object suitable for recording/streaming.

Example:

// build a telemetry snapshot for the current generation
const snapshot = neat.buildTelemetryEntry(neat.population[0]);
neat.recordTelemetryEntry(snapshot);

### computeDiversityStats

```ts
computeDiversityStats(): void
```

Compute several diversity statistics used by telemetry reporting.

This helper is intentionally conservative in runtime: when `fastMode` is enabled it will automatically tune a few sampling defaults to keep the computation cheap. The computed statistics are written to `this._diversityStats` as an object with keys like `meanCompat` and `graphletEntropy`.

Example:

// compute and store diversity stats onto the neat instance
neat.options.diversityMetrics = { enabled: true };
neat.computeDiversityStats();
console.log(neat._diversityStats.meanCompat);

### createTelemetryEntryBase

```ts
createTelemetryEntryBase(
  generationIndex: number,
  bestScore: number,
  speciesCount: number,
): TelemetryEntry
```

Create a strict baseline telemetry entry with required fields populated.

This helper centralizes defaults so downstream telemetry producers can
extend the entry while keeping the strict `TelemetryEntry` contract.

Parameters:
- `generationIndex` - Generation index for the telemetry snapshot.
- `bestScore` - Best fitness value observed in the generation.
- `speciesCount` - Number of extant species.

Returns: A strict telemetry entry with required fields populated.

### recordTelemetryEntry

```ts
recordTelemetryEntry(
  entry: TelemetryEntry,
): void
```

Record a telemetry entry into the instance buffer and optionally stream it.

Steps:
This method performs the following steps to persist and optionally stream telemetry:
1. Apply `applyTelemetrySelect` to filter fields according to user selection.
2. Ensure `this._telemetry` buffer exists and push the entry.
3. If a telemetry stream callback is configured, call it.
4. Trim the buffer to a conservative max size (500 entries).

Example:

Parameters:
- `entry` - - Telemetry entry to record.

Example:

// record a simple telemetry entry from inside the evolve loop
neat.recordTelemetryEntry({ gen: neat.generation, best: neat.population[0].score });

### structuralEntropy

```ts
structuralEntropy(
  graph: { [key: string]: unknown; nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; },
): number
```

Lightweight proxy for structural entropy based on degree-distribution.

This function computes an approximate entropy of a graph topology by
counting node degrees and computing the entropy of the degree histogram.
The result is cached on the graph object for the current generation in
`_entropyVal` to avoid repeated expensive recomputation.

Example:

Parameters:
- `graph` - - A genome-like object with `nodes` and `connections` arrays.

Returns: A non-negative number approximating structural entropy.

Example:

const H = structuralEntropy.call(neat, genome);
console.log(`Structure entropy: ${H.toFixed(3)}`);

### TelemetryContext

Context view used within telemetry helpers to access optional internal
fields with descriptive names rather than repeated inline casts.

## neat/neat.objectives.ts

### _getObjectives

```ts
_getObjectives(): ObjectiveDescriptor[]
```

Build and return the list of registered objectives for this NEAT instance.

This function lazily builds `this._objectivesList` from the built-in
fitness objective (unless suppressed) and any user-registered multi-
objective descriptors found on `this.options.multiObjective.objectives`.

Typical use: the evolution loop calls this to know which objectives to
evaluate and whether each objective should be maximized or minimized.

Example:

```ts
const objectives = neatInstance._getObjectives();
// objectives: Array<ObjectiveDescriptor>
```

Returns: Array of objective descriptors in the
order they should be applied. If multi-objective support is disabled or
no objectives are registered, this will contain only the built-in
fitness objective (unless suppressed).

### clearObjectives

```ts
clearObjectives(): void
```

Clear all registered multi-objectives.

This resets `this.options.multiObjective.objectives` to an empty array and
clears the cached objectives list so that subsequent calls will reflect the
cleared state.

Example:

```ts
neat.clearObjectives();
// now only the default fitness objective (unless suppressed) will remain
```

### registerObjective

```ts
registerObjective(
  key: string,
  direction: "max" | "min",
  accessor: (genome: GenomeLike) => number,
): void
```

Register a new objective descriptor.

This adds or replaces an objective with the given `key`. The objective is a
lightweight descriptor with a `key`, `direction` ('min' | 'max'), and an
`accessor` function that maps a genome to a numeric objective value.

Example:

```ts
// register an objective that measures model sparsity (lower is better)
neat.registerObjective('sparsity', 'min', genome => computeSparsity(genome));
```

Notes:
- If `this.options.multiObjective` doesn't exist it will be created and
  enabled.
- Registering an objective replaces any previous objective with the same
  `key`.

## neat/neat.speciation.ts

Assign genomes into species based on compatibility distance and maintain species structures.
This function creates new species for unassigned genomes, prunes empty species, updates
dynamic compatibility threshold controllers, performs optional auto coefficient tuning, and
records per‑species history statistics used by telemetry and adaptive controllers.

Implementation notes:

### _applyFitnessSharing

```ts
_applyFitnessSharing(): void
```

Apply fitness sharing to penalize similarity within species.

Parameters:
- `this` - - Neat instance context with species array and compatibility distance function.

### _sortSpeciesMembers

```ts
_sortSpeciesMembers(
  species: SpeciesLike,
): void
```

Sort species members by descending score.

Parameters:
- `this` - - Neat instance context.
- `sp` - - Species to sort.

### _speciate

```ts
_speciate(): void
```

Assign genomes into species based on compatibility distance.

Parameters:
- `this` - - Speciation harness context.

Returns: Nothing.

### _updateSpeciesStagnation

```ts
_updateSpeciesStagnation(): void
```

Update stagnation counters for all species.

Parameters:
- `this` - - Neat instance context with species array and generation counter.

## neat/neat.rng.constants.ts

Constants used by the deterministic xorshift RNG helper.

### RNG_DEFAULT_SEED_FALLBACK

Fallback seed used when the derived seed would be zero (xorshift cannot use 0).

### RNG_NORMALIZATION_DIVISOR

Divisor used to normalize the 32-bit integer state into [0, 1).

### RNG_POPULATION_OFFSET

Minimum population offset added before scrambling to avoid zero seeds.

### RNG_SHIFT_LEFT_PRIMARY

Bit-shift values for the xorshift32 variant.

### RNG_SHIFT_LEFT_SECONDARY

### RNG_SHIFT_RIGHT_PRIMARY

### RNG_TIME_SCRAMBLE_CONSTANT

Constants used by the deterministic xorshift RNG helper.

## neat/neat.multiobjective.ts

Multi-objective helpers (fast non-dominated sorting + crowding distance).
Extracted from `neat.ts` to keep the core class slimmer.

### fastNonDominated

```ts
fastNonDominated(
  pop: default[],
): default[][]
```

Perform fast non-dominated sorting and compute crowding distances for a
population of networks (genomes). This implements a standard NSGA-II style
non-dominated sorting followed by crowding distance assignment.

The function annotates genomes with two fields used elsewhere in the codebase:
- `_moRank`: integer Pareto front rank (0 = best/frontier)
- `_moCrowd`: numeric crowding distance (higher is better; Infinity for
  boundary solutions)

Example
```ts
// inside a Neat class that exposes `_getObjectives()` and `options`
const fronts = fastNonDominated.call(neatInstance, population);
// fronts[0] is the Pareto-optimal set
```

Notes for documentation generation:
- Each objective descriptor returned by `_getObjectives()` must have an
  `accessor(genome: Network): number` function and may include
  `direction: 'max' | 'min'` to indicate optimization direction.
- Accessor failures are guarded and will yield a default value of 0.

Parameters:
- `this` - - Neat instance providing `_getObjectives()`, `options` and
`_paretoArchive` fields (function is meant to be invoked using `.call`)
- `pop` - - population array of `Network` genomes to be ranked

Returns: Array of Pareto fronts; each front is an array of `Network` genomes.

## neat/neat.adaptive.shared.ts

Constant: zero value.

### ACCEPTANCE_LOWER_MULTIPLIER

Lower acceptance multiplier.

### ACCEPTANCE_UPPER_MULTIPLIER

Upper acceptance multiplier.

### AdaptiveMutationConfig

### ADJUST_RATE_DEFAULT

Default adjustment rate in minimal criterion.

### ANCESTOR_UNIQ_MODE_EPSILON

Ancestor uniqueness epsilon mode.

### ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE

Ancestor uniqueness lineage pressure mode.

### AncestorUniqAdaptiveConfig

### ANNEAL_BASELINE_GENERATIONS

Baseline generations for annealing progress.

### ANNEAL_PROGRESS_MAX

Maximum progress ratio used in annealing.

### BUDGET_GROWTH_MULTIPLIER

Default budget growth multiplier.

### COMPLEXITY_MODE_ADAPTIVE

Complexity budget adaptive mode string.

### COMPLEXITY_MODE_LINEAR

Complexity budget linear mode string.

### ComplexityBudgetConfig

### DEFAULT_ADAPT_EVERY

Default adapt-every cadence for adaptive mutation.

### DEFAULT_ANCESTOR_UNIQ_ADJUST

Default adjustment magnitude for uniqueness nudges.

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

Default cooldown (generations) for ancestor-uniqueness adjustments.

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

Default upper bound for acceptable ancestor uniqueness.

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

Default lower bound for acceptable ancestor uniqueness.

### DEFAULT_CB_INCREASE_FACTOR

Default increase factor for adaptive schedule.

### DEFAULT_CB_STAGNATION_FACTOR

Default stagnation factor for adaptive schedule.

### DEFAULT_IMPROVEMENT_WINDOW

Default score history window.

### DEFAULT_INITIAL_MUTATION_RATE

Default initial mutation rate used for balance checks.

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

Default lineage pressure strength when initializing the option.

### DEFAULT_MAX_MUTATION_AMOUNT

Default maximum mutation amount.

### DEFAULT_MAX_MUTATION_RATE

Default maximum per-genome mutation rate.

### DEFAULT_MIN_MUTATION_AMOUNT

Default minimum mutation amount.

### DEFAULT_MIN_MUTATION_RATE

Default minimum per-genome mutation rate.

### DEFAULT_MUTATION_AMOUNT

Default mutation amount when genome value is missing.

### DEFAULT_MUTATION_AMOUNT_SIGMA

Default mutation amount sigma for perturbations.

### DEFAULT_MUTATION_SIGMA

Default mutation sigma for adaptive mutation.

### DENOMINATOR_FALLBACK

Fallback denominator to avoid divide-by-zero.

### EXPLORE_LOW_DECREASE_MULTIPLIER

Multiplicative decay for explore-low strategy (top half).

### EXPLORE_LOW_INCREASE_MULTIPLIER

Multiplicative boost for explore-low strategy (bottom half).

### FIVE

Constant: five value.

### FOUR

Constant: four value.

### Genome

### HALF_INDEX_DIVISOR

Divisor used to split populations in half.

### HISTORY_MIN_IMPROVEMENT_COUNT

Minimum history length to compute improvement.

### HISTORY_MIN_SLOPE_COUNT

Minimum history length to compute slope.

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

Multiplier when decreasing lineage pressure strength.

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

Multiplier when increasing lineage pressure strength.

### LINEAGE_PRESSURE_MODE_SPREAD

Lineage pressure spread mode.

### LINEAR_HORIZON_DEFAULT

Default horizon for linear schedule.

### MINIMAL_TOPOLOGY_OFFSET

Offset added to input/output for minimal topology.

### MinimalCriterionAdaptiveConfig

### MUTATION_SIGMA_SCALE

Scale applied to mutation sigma for perturbations.

### MUTATION_STRATEGY_ANNEAL

Strategy identifier for annealed mutation.

### MUTATION_STRATEGY_EXPLORE_LOW

Strategy identifier for explore-low mutation.

### MUTATION_STRATEGY_TWO_TIER

Strategy identifier for two-tier mutation.

### MutationOutcome

### MutationPartitions

### MutationSettings

### NeatLikeWithAdaptive

Minimal interface for NEAT instances with adaptive features.
Exported for use in tests and type-safe function calls.

### NEGATIVE_ONE

Constant: negative one for last index.

### NOVELTY_ARCHIVE_MIN_SIZE

Novelty archive minimum size.

### NOVELTY_FACTOR_DEFAULT

Novelty factor when archive is sufficient.

### NOVELTY_FACTOR_SMALL

Novelty factor when archive is small.

### ONE

Constant: one value.

### ONE_HUNDRED

Constant: one hundred value.

### OPERATOR_DECAY_DEFAULT

Default operator decay factor.

### OperatorAdaptationConfig

### PHASE_COMPLEXIFY

Phase label for complexify.

### PHASE_LENGTH_DEFAULT

Default phase length in generations.

### PHASE_SIMPLIFY

Phase label for simplify.

### PhasedComplexityConfig

### PROGRESS_RATIO_MAX

Maximum progress ratio for scheduling.

### RNG_CENTER_OFFSET

Random offset for signed deltas.

### RNG_SPREAD_MULTIPLIER

Random range multiplier for signed deltas.

### SLOPE_BOOST_MULTIPLIER

Slope boost multiplier for adaptive increase factor.

### SLOPE_NORMALIZE_CLAMP

Clamp magnitude for slope normalization.

### SLOPE_PENALTY_MULTIPLIER

Slope penalty multiplier for stagnation factor.

### TARGET_ACCEPTANCE_DEFAULT

Default target acceptance in minimal criterion.

### TEN

Constant: ten value.

### THREE

Constant: three value.

### TWO

Constant: two value.

### ZERO

Constant: zero value.

## neat/neat.telemetry.exports.ts

### buildSpeciesHistoryCsv

```ts
buildSpeciesHistoryCsv(
  recentHistory: SpeciesHistoryEntry[],
  headers: string[],
): string
```

Build the full CSV string for species history given ordered headers and
a slice of history entries.

Implementation notes:
- The history is a 2‑level structure (generation entry -> species stats[]).
- We emit one CSV row per species stat, repeating the generation value.
- Values are JSON.stringify'd to remain safe for commas/quotes.

### buildTelemetryHeaders

```ts
buildTelemetryHeaders(
  info: TelemetryHeaderInfo,
): string[]
```

Build the ordered list of CSV headers from collected metadata.
Flattened nested metrics are emitted using group prefixes (group.key).

### collectTelemetryHeaderInfo

```ts
collectTelemetryHeaderInfo(
  entries: TelemetryEntry[],
): TelemetryHeaderInfo
```

Collect header metadata from the raw telemetry entries.
- Discovers base (top‑level) keys excluding grouped objects.
- Discovers nested keys inside complexity, perf, lineage, diversity groups.
- Tracks presence of optional multi-value structures (ops, objectives, etc.).

### DEFAULT_SPECIES_BEST_SCORE

Default fallback best score when missing.

### DEFAULT_SPECIES_HISTORY_GENERATION

Default fallback generation when missing.

### DEFAULT_SPECIES_HISTORY_MAX_ENTRIES

Default max entries for species history CSV exports.

### DEFAULT_SPECIES_ID

Default fallback species id when missing.

### DEFAULT_SPECIES_LAST_IMPROVED

Default fallback last improved when missing.

### DEFAULT_SPECIES_SIZE

Default fallback species size when missing.

### exportSpeciesHistoryCSV

```ts
exportSpeciesHistoryCSV(
  maxEntries: number,
): string
```

Export species history snapshots to CSV.

Each row represents a single species at a specific generation; the generation
value is repeated per species. Dynamically discovers species stat keys so
custom metadata added at runtime is preserved.

Behavior:
- If `_speciesHistory` is absent/empty but `_species` exists, synthesizes a
  minimal snapshot to ensure deterministic headers early in a run.
- Returns a header-only CSV when there is no history or species.

Parameters:
- `this` - Neat instance (expects `_speciesHistory` and optionally `_species`).
- `maxEntries` - Maximum number of most recent history snapshots (generations) to include (default 200).

Returns: CSV string (headers + rows) describing species evolution timeline.

### exportTelemetryCSV

```ts
exportTelemetryCSV(
  maxEntries: number,
): string
```

Export recent telemetry entries to a CSV string.

Responsibilities:
- Collect a bounded slice (`maxEntries`) of recent telemetry records.
- Discover and flatten dynamic header keys (top-level + grouped metrics).
- Serialize each entry into a CSV row with stable, parseable values.

Flattening Rules:
- Nested groups (complexity, perf, lineage, diversity) become group.key columns.
- Optional arrays/maps (ops, objectives, objAges, speciesAlloc, objEvents, objImportance, fronts) included only if present.

Parameters:
- `this` - Neat instance (expects `_telemetry` array field).
- `maxEntries` - Maximum number of most recent telemetry entries to include (default 500).

Returns: CSV string (headers + rows) or empty string when no telemetry.

### exportTelemetryJSONL

```ts
exportTelemetryJSONL(): string
```

Telemetry export helpers extracted from `neat.ts`.

This module exposes small helpers intended to serialize the internal
telemetry gathered by the NeatapticTS `Neat` runtime into common
data-export formats (JSONL and CSV). The functions intentionally
operate against `this` so they can be attached to instances.

### serializeTelemetryEntry

```ts
serializeTelemetryEntry(
  entry: TelemetryEntry,
  headers: string[],
): string
```

Serialize one telemetry entry into a CSV row using previously computed headers.
Uses a `switch(true)` pattern instead of a long if/else chain to reduce
cognitive complexity while preserving readability of each scenario.

### TelemetryHeaderInfo

Shape describing collected telemetry header discovery info.

## neat/neat.evolve.offspring.constants.ts

Index used when falling back to the first genome in the population.

### LINEAGE_BASE_DEPTH

Baseline lineage depth when parent depth metadata is missing.

### LINEAGE_DEPTH_INCREMENT

Depth increment applied when deriving a child from its parents.

### OFFSPRING_FALLBACK_INDEX

Index used when falling back to the first genome in the population.

## neat/neat.evaluate.utils.types.ts

Genome with score, novelty, and clearing capabilities.

This interface describes the minimal genome shape required by evaluation
helpers. It intentionally stays permissive for compatibility with legacy
genome variants while documenting the expected properties.

### DiversityStats

Diversity statistics tracked during evaluation.

The values are optional because different evaluations may only compute a
subset of metrics.

### GenomeForEvaluation

Genome with score, novelty, and clearing capabilities.

This interface describes the minimal genome shape required by evaluation
helpers. It intentionally stays permissive for compatibility with legacy
genome variants while documenting the expected properties.

### NeatControllerForEval

NEAT controller interface for evaluation.

This interface models the subset of a NEAT controller used by the evaluation
helpers. It includes options, population data, and optional adaptive tuning
hooks.

### NoveltyArchiveEntry

Novelty archive entry with descriptor and novelty score.

Entries store a descriptor vector alongside the computed novelty so the
archive can seed future novelty calculations.

### ObjectiveDef

Objective definition for multi-objective optimization.

Objectives are registered dynamically to guide evaluation and selection.

## neat/neat.multiobjective.utils.types.ts

### NeatLikeWithMultiObjective

Minimal Neat-like interface required by the multi-objective helpers.

This intentionally models only the fields used for archiving Pareto fronts
and retrieving objective descriptors. It allows these helpers to be used
without depending on the full Neat class type.

### NetworkWithMOAnnotations

Extends a genome/network with multi-objective annotations.

These properties are used as transient metadata during selection.

- `_moRank`: Pareto front rank (0 = best front)
- `_moCrowd`: crowding distance within the front (higher = more isolated;
  boundary genomes are typically `Infinity`)
- `_id`: optional stable identifier used for compact archiving

### ObjectiveDescriptor

Describes how to evaluate a single objective for a genome.

The order of objective descriptors defines the order of each genome’s
objective vector and therefore the columns of the values matrix.

Notes:
- `accessor` should be deterministic for a given genome state.
- `direction` controls Pareto dominance comparisons:
  - `'max'`: higher is better
  - `'min'`: lower is better
- If `direction` is omitted, it defaults to `'max'`.

Example:

```ts
const objectives: ObjectiveDescriptor[] = [
  { accessor: (g) => g.score ?? 0, direction: 'max' },
  { accessor: (g) => g.cost ?? 0, direction: 'min' },
];
```

## neat/neat.rng.utils.ts

### exportRngState

```ts
exportRngState(
  host: RngHost,
): number | undefined
```

Export the current RNG state for persistence.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when not set.

### getOrCreateRng

```ts
getOrCreateRng(
  host: RngHost,
): () => number
```

Return a cached RNG or create a deterministic xorshift RNG when absent.

The helper respects a user-provided RNG at `options.rng` when present.
Otherwise it seeds a xorshift32 RNG using the current time and population
size, guarding against the invalid zero seed.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

### importRngState

```ts
importRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Alias for restoring RNG state kept for compatibility with prior surface.

### restoreRngState

```ts
restoreRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Restore a previously captured RNG state.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### RngHost

Minimal host surface required by the RNG utilities.

### sampleRandomSequence

```ts
sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce a sequence of random samples using the host RNG.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

### snapshotRngState

```ts
snapshotRngState(
  host: RngHost,
): number | undefined
```

Snapshot the current RNG state for deterministic replay.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.

## neat/neat.cache.utils.ts

Invalidate per-genome caches used across compatibility and forward-pass logic.

### invalidateGenomeCaches

```ts
invalidateGenomeCaches(
  genomeCandidate: unknown,
): void
```

Invalidate per-genome caches used across compatibility and forward-pass logic.

Parameters:
- `genomeCandidate` - - Genome object whose caches should be cleared.

## neat/neat.compat.utils.ts

Compatibility-distance helper utilities.

### buildPairKey

```ts
buildPairKey(
  firstGenome: GenomeLike,
  secondGenome: GenomeLike,
): string
```

Build a stable cache key for a genome pair.

Parameters:
- `firstGenome` - - First genome in the pair.
- `secondGenome` - - Second genome in the pair.

Returns: Stable cache key in the form "minId|maxId".

### compareInnovationLists

```ts
compareInnovationLists(
  firstList: [number, number][],
  secondList: [number, number][],
): ComparisonMetrics
```

Compare two sorted innovation lists and derive comparison metrics.

Parameters:
- `firstList` - - Sorted innovation list for the first genome.
- `secondList` - - Sorted innovation list for the second genome.

Returns: Aggregated comparison metrics for distance computation.

### ComparisonMetrics

Aggregated comparison metrics for compatibility calculations.

### computeCompatibilityDistance

```ts
computeCompatibilityDistance(
  neatContext: NeatLikeForCompat,
  metrics: ComparisonMetrics,
): number
```

Compute the compatibility distance from comparison metrics.

Parameters:
- `neatContext` - - NEAT context providing coefficients.
- `metrics` - - Aggregated comparison metrics.

Returns: Final compatibility distance for the genome pair.

### ConnectionLike

Shape of a connection entry used during compatibility checks.

### ensureGenerationCache

```ts
ensureGenerationCache(
  neatContext: NeatLikeForCompat,
): void
```

Ensure generation-scoped compatibility caches exist.

Parameters:
- `neatContext` - - Current NEAT context holding generation and caches.

Returns: void

### GenomeLike

Minimal genome shape used for compatibility distance calculations.

### getDistanceCacheMap

```ts
getDistanceCacheMap(
  neatContext: NeatLikeForCompat,
): Map<string, number>
```

Retrieve the generation-scoped cache map for pairwise distances.

Parameters:
- `neatContext` - - Current NEAT context with the cache map.

Returns: Map storing cached distances for genome pairs this generation.

### getSortedInnovationCache

```ts
getSortedInnovationCache(
  neatContext: NeatLikeForCompat,
  genome: GenomeLike,
): [number, number][]
```

Retrieve or build a sorted innovation list for a genome.

Parameters:
- `neatContext` - - NEAT context used for fallback innovation numbers.
- `genome` - - Genome to derive sorted innovation list for.

Returns: Array of [innovationNumber, weight] sorted by innovationNumber.

### NeatLikeForCompat

Minimal NEAT context required by compatibility helpers.

### resolveMaxInnovation

```ts
resolveMaxInnovation(
  list: [number, number][],
): number
```

Resolve the highest innovation id from a sorted list.

Parameters:
- `list` - - Sorted innovation list for a genome.

Returns: Highest innovation id or 0 if list is empty.

## neat/neat.evolve.utils.ts

### adaptReenableProbability

```ts
adaptReenableProbability(
  internal: NeatControllerForEvolution,
  config: { minSamples: number; target: number; min: number; max: number; deltaScale: number; },
): void
```

Adapt the re-enable probability based on recent success ratios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### addOffspring

```ts
addOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  helpers: { addSpeciatedOffspring: (nextPopulation: default[], remainingSlots: number) => Promise<void>; addUnspeciatedOffspring: (nextPopulation: default[], remainingSlots: number) => Promise<void>; },
): Promise<void>
```

Add offspring to fill remaining population slots.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `helpers` - - Helper callbacks for offspring selection.
- `helpers` - - Speciated offspring helper.
- `helpers` - - Unspeciated offspring helper.

Returns: void.

### addSpeciatedOffspring

```ts
addSpeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  remainingSlots: number,
  config: { minOffspringDefault: number; survivalThresholdDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; crossSpeciesGuardLimit: number; },
): Promise<void>
```

Add offspring when speciation is enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Offspring allocation constants.

Returns: void.

### addUnspeciatedOffspring

```ts
addUnspeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  remainingSlots: number,
): Promise<void>
```

Add offspring when speciation is disabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.

Returns: void.

### applyAdaptiveComplexityControllers

```ts
applyAdaptiveComplexityControllers(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply adaptive complexity controllers if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAncestorUniqAdaptiveSafe

```ts
applyAncestorUniqAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply ancestor uniqueness adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAutoCompatibilityTuning

```ts
applyAutoCompatibilityTuning(
  internal: NeatControllerForEvolution,
  config: { targetMin: number; adjustRate: number; minCoeff: number; maxCoeff: number; randomScale: number; },
): void
```

Apply auto-compatibility tuning if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Tuning constants.

Returns: void.

### applyDynamicObjectiveSchedule

```ts
applyDynamicObjectiveSchedule(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  config: { autoEntropyAddAt: number; },
): void
```

Apply dynamic objective scheduling and entropy rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Keys of active objectives.
- `config` - - Scheduling constants.

Returns: void.

### applyElitism

```ts
applyElitism(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): void
```

Apply elitism for the next generation.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyFitnessSuppressionForTests

```ts
applyFitnessSuppressionForTests(
  internal: NeatControllerForEvolution,
): void
```

Suppress fitness objective for specific test scenarios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyGlobalStagnationInjectionIfNeeded

```ts
applyGlobalStagnationInjectionIfNeeded(
  internal: NeatControllerForEvolution,
  helpers: { buildFreshGenomeForStagnation: () => Promise<GenomeWithMetadata>; replaceFraction: number; },
): Promise<void>
```

Apply global stagnation injection if configured.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for stagnation injection.
- `helpers` - - Genome builder for injection.

Returns: void.

### applyMinimalCriterionAdaptiveSafe

```ts
applyMinimalCriterionAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply minimal criterion adaptive controller if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyOperatorAdaptationSafe

```ts
applyOperatorAdaptationSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply operator adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyProvenance

```ts
applyProvenance(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): void
```

Add provenance genomes into the next population.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyPruningAndMutation

```ts
applyPruningAndMutation(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply pruning and mutation phases.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applySpeciationAndSharingIfEnabled

```ts
applySpeciationAndSharingIfEnabled(
  internal: NeatControllerForEvolution,
  helpers: { applyAutoCompatibilityTuning: () => void; recordSpeciesHistorySnapshot: () => void; },
): Promise<void>
```

Apply speciation, fitness sharing, and related side effects.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used for tuning and history.
- `helpers` - - Auto-compatibility adjustment helper.
- `helpers` - - Species history snapshot helper.

Returns: void.

### buildFittestSnapshot

```ts
buildFittestSnapshot(
  internal: NeatControllerForEvolution,
): default
```

Build a cloned Network from the current best genome.

Parameters:
- `internal` - - NEAT controller instance.

Returns: best network snapshot.

### buildFreshGenomeForStagnation

```ts
buildFreshGenomeForStagnation(
  internal: NeatControllerForEvolution,
): Promise<GenomeWithMetadata>
```

Build a fresh genome for stagnation injection.

Parameters:
- `internal` - - NEAT controller instance.

Returns: new genome with minimum constraints.

### buildNextPopulation

```ts
buildNextPopulation(
  internal: NeatControllerForEvolution,
  helpers: { applyElitism: (nextPopulation: default[]) => void; applyProvenance: (nextPopulation: default[]) => void; addOffspring: (nextPopulation: default[]) => Promise<void>; },
): Promise<default[]>
```

Build the next population (elitism, provenance, offspring).

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for population construction.
- `helpers` - - Elitism helper.
- `helpers` - - Provenance helper.
- `helpers` - - Offspring helper.

Returns: next population array.

### captureObjectiveImportanceSnapshot

```ts
captureObjectiveImportanceSnapshot(
  internal: NeatControllerForEvolution,
): void
```

Capture objective importance stats for telemetry.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### clearPopulationScores

```ts
clearPopulationScores(
  internal: NeatControllerForEvolution,
): void
```

Clear genome scores to force re-evaluation.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeDiversityStatsSafely

```ts
computeDiversityStatsSafely(
  internal: NeatControllerForEvolution,
): void
```

Compute diversity stats safely if the hook exists.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeElapsedTime

```ts
computeElapsedTime(
  startTimestamp: number,
): number
```

Compute elapsed time since the start of evolve().

Parameters:
- `startTimestamp` - - Start time resolved earlier.

Returns: elapsed time.

### createOffspring

```ts
createOffspring(
  context: OffspringContext,
  selectParent: () => default,
): default
```

Create a child genome by crossing two parents selected via the provided callback.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.

Returns: Newly created offspring genome.

### enforcePopulationConstraints

```ts
enforcePopulationConstraints(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): Promise<void>
```

Ensure new population meets structural constraints.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Population to validate.

Returns: void.

### ensurePopulationEvaluated

```ts
ensurePopulationEvaluated(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Ensure the population is evaluated before evolution operations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### ensureSpeciesHistorySnapshot

```ts
ensureSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void
```

Ensure a minimal species history snapshot exists for exports.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### EVOLVE_NO_BEST_GENOME_WARNING

Warning emitted when evolution finishes without a best genome.

### invalidateCompatibilityCaches

```ts
invalidateCompatibilityCaches(
  internal: NeatControllerForEvolution,
): void
```

Invalidate compatibility caches after mutations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### LINEAGE_BASE_DEPTH

Baseline lineage depth when parent depth metadata is missing.

### LINEAGE_DEPTH_INCREMENT

Depth increment applied when deriving a child from its parents.

### OFFSPRING_FALLBACK_INDEX

Index used when falling back to the first genome in the population.

### OffspringContext

Minimal surface needed for offspring generation.

### processMultiObjective

```ts
processMultiObjective(
  internal: NeatControllerForEvolution,
  config: { paretoArchiveMax: number; targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; pruneWindowDefault: number; pruneRangeEpsDefault: number; },
): void
```

Run multi-objective ranking, crowding distance, and archives.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Multi-objective tuning constants.

Returns: void.

### recordSpeciesHistorySnapshot

```ts
recordSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void
```

Record a species history snapshot when needed.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### recordTelemetryIfEnabled

```ts
recordTelemetryIfEnabled(
  internal: NeatControllerForEvolution,
  snapshot: default,
): Promise<void>
```

Record telemetry if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot for the generation.

Returns: void.

### resetObjectivesCache

```ts
resetObjectivesCache(
  internal: NeatControllerForEvolution,
): void
```

Clear cached objectives so dynamic schedules can rebuild them.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### resolveStartTime

```ts
resolveStartTime(): number
```

Resolve the start time for an evolution step.

Returns: timestamp in milliseconds or high-resolution units.

### trackGlobalImprovement

```ts
trackGlobalImprovement(
  internal: NeatControllerForEvolution,
  snapshot: default,
): void
```

Track global best improvement for stagnation logic.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot.

Returns: void.

### updateGlobalBestTracking

```ts
updateGlobalBestTracking(
  internal: NeatControllerForEvolution,
): void
```

Update generation-level best score tracking.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### updateObjectiveScheduleAndAges

```ts
updateObjectiveScheduleAndAges(
  internal: NeatControllerForEvolution,
  helpers: { applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) => void; },
): Promise<void>
```

Update objective schedule, pending adds/removes, and objective ages.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used by scheduling logic.
- `helpers` - - Dynamic objective scheduler.

Returns: void.

### updateSpeciesStagnationIfEnabled

```ts
updateSpeciesStagnationIfEnabled(
  internal: NeatControllerForEvolution,
): void
```

Update species stagnation status when speciation enabled.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### warnIfNoBestGenome

```ts
warnIfNoBestGenome(): void
```

Emit the standard warning for runs that end without a valid best genome.

## neat/neat.lineage.utils.ts

Lineage / ancestry helper utilities for NEAT populations.

This module centralizes helper logic used by the public lineage APIs to keep
the main entry file small and orchestration-focused.

### AncestorQueueEntry

Queue entry for ancestor traversal.

### calculateMaxSamplePairs

```ts
calculateMaxSamplePairs(
  size: number,
): number
```

Parameters:
- `size` - Population size.

Returns: Upper bound on the number of sampled pairs.

### collectAncestorIds

```ts
collectAncestorIds(
  queueEntries: AncestorQueueEntry[],
  population: GenomeLike[],
): Set<number>
```

Parameters:
- `queueEntries` - BFS queue seeded with direct parents.
- `population` - Current population for lookups.

Returns: Unique ancestor IDs encountered within the window.

### computeAverageDistance

```ts
computeAverageDistance(
  distances: number[],
): number
```

Parameters:
- `distances` - Jaccard distances to average.

Returns: Mean distance rounded to the configured decimal places.

### computeJaccardDistance

```ts
computeJaccardDistance(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): number
```

Parameters:
- `ancestorSetA` - First ancestor set.
- `ancestorSetB` - Second ancestor set.

Returns: Jaccard distance for the two sets.

### computePairDistance

```ts
computePairDistance(
  pair: GenomeIndexPair,
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number | undefined
```

Parameters:
- `pair` - Index pair for comparison.
- `population` - Current population.
- `buildAncestorSet` - Helper to build ancestor sets.

Returns: Jaccard distance or undefined when skipped.

### computePairDistances

```ts
computePairDistances(
  pairs: GenomeIndexPair[],
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number[]
```

Parameters:
- `pairs` - Sampled index pairs.
- `population` - Current population.
- `buildAncestorSet` - Helper to build ancestor sets.

Returns: Jaccard distances for valid pairs.

### countIntersection

```ts
countIntersection(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): number
```

Parameters:
- `ancestorSetA` - First ancestor set.
- `ancestorSetB` - Second ancestor set.

Returns: Size of intersection between the sets.

### createInitialQueue

```ts
createInitialQueue(
  parentIds: number[],
  population: GenomeLike[],
): AncestorQueueEntry[]
```

Parameters:
- `parentIds` - Direct parent IDs to seed the queue.
- `population` - Current population for lookups.

Returns: Queue entries at depth 1.

### enqueueParentEntries

```ts
enqueueParentEntries(
  queueEntries: AncestorQueueEntry[],
  currentEntry: AncestorQueueEntry,
  population: GenomeLike[],
): void
```

Parameters:
- `queueEntries` - Mutable queue array to append to.
- `currentEntry` - Current ancestor entry being expanded.
- `population` - Current population for lookups.

### findGenomeById

```ts
findGenomeById(
  population: GenomeLike[],
  genomeId: number,
): GenomeLike | undefined
```

Parameters:
- `population` - Current population for lookup.
- `genomeId` - Genome identifier to match.

Returns: Genome reference if found.

### GenomeIndexPair

Index pair representing a sampled genome pair.

### GenomeLike

Minimal shape assumed for a genome inside the NEAT population. Additional properties are
intentionally left open (index signature) because user implementations may extend genomes.

### hasMinimumPopulation

```ts
hasMinimumPopulation(
  size: number,
): boolean
```

Parameters:
- `size` - Population size.

Returns: True when at least two genomes exist.

### isEmptyAncestorPair

```ts
isEmptyAncestorPair(
  ancestorSetA: Set<number>,
  ancestorSetB: Set<number>,
): boolean
```

Parameters:
- `ancestorSetA` - First ancestor set.
- `ancestorSetB` - Second ancestor set.

Returns: True when both sets are empty.

### isWithinDepthWindow

```ts
isWithinDepthWindow(
  depth: number,
): boolean
```

Parameters:
- `depth` - Current depth value.

Returns: True when within the configured depth window.

### NeatLineageContext

Expected `this` context for lineage helpers (a subset of the NEAT instance).

### normalizeParentIds

```ts
normalizeParentIds(
  value: GenomeLike,
): number[]
```

Parameters:
- `value` - Genome to read parents from.

Returns: Parent ID list (empty when absent).

### pickDistinctIndex

```ts
pickDistinctIndex(
  randomNumber: () => number,
  size: number,
  firstIndex: number,
): number
```

Parameters:
- `randomNumber` - RNG function returning [0,1).
- `size` - Population size for bounds.
- `firstIndex` - Index to avoid.

Returns: Random index not equal to the first index.

### pickRandomIndex

```ts
pickRandomIndex(
  randomNumber: () => number,
  size: number,
): number
```

Parameters:
- `randomNumber` - RNG function returning [0,1).
- `size` - Population size for bounds.

Returns: Random index within bounds.

### resolveParentIds

```ts
resolveParentIds(
  value: GenomeLike | undefined,
): number[]
```

Parameters:
- `value` - Optional genome reference.

Returns: Parent IDs when available.

### sampleGenomePairs

```ts
sampleGenomePairs(
  sampleCount: number,
  size: number,
  rngFactory: () => () => number,
): GenomeIndexPair[]
```

Parameters:
- `sampleCount` - Number of pairs to sample.
- `size` - Population size for index bounds.
- `rngFactory` - RNG provider to obtain a random function.

Returns: Array of sampled index pairs.

## neat/neat.novelty.utils.ts

Return the current size of the novelty archive.

### getNoveltyArchiveSize

```ts
getNoveltyArchiveSize(
  host: { _noveltyArchive?: unknown[] | undefined; },
): number
```

Return the current size of the novelty archive.

### resetNoveltyArchive

```ts
resetNoveltyArchive(
  host: { _noveltyArchive?: unknown[] | undefined; },
): void
```

Reset the novelty archive in place.

## neat/neat.pruning.utils.ts

Minimal Neat instance contract required by pruning helpers.

Example:

const host: NeatLikeForPruning = {
  options: { evolutionPruning: { startGeneration: 5, targetSparsity: 0.4 } },
  generation: 10,
  population: [],
} as NeatLikeForPruning;

### AdaptivePruningOptions

Adaptive pruning options extracted from the Neat instance.

### applyAdaptivePruneLevelToPopulation

```ts
applyAdaptivePruneLevelToPopulation(
  host: NeatLikeForPruning,
  pruneLevel: number,
): void
```

Parameters:
- `host` - - Neat instance with population.
- `pruneLevel` - - Prune level to apply.

### applyPruningToPopulation

```ts
applyPruningToPopulation(
  host: NeatLikeForPruning,
  options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; },
  targetSparsity: number,
): void
```

Parameters:
- `host` - - Neat instance with population.
- `options` - - Evolution pruning options.
- `targetSparsity` - - Target sparsity to apply.

### computeMeanConnectionCount

```ts
computeMeanConnectionCount(
  host: NeatLikeForPruning,
): number
```

Parameters:
- `host` - - Neat instance with population.

Returns: Average number of connections per genome.

### computeMeanNodeCount

```ts
computeMeanNodeCount(
  host: NeatLikeForPruning,
): number
```

Parameters:
- `host` - - Neat instance with population.

Returns: Average number of nodes per genome.

### computeNextAdaptivePruneLevel

```ts
computeNextAdaptivePruneLevel(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  currentPruneLevel: number,
  currentMetricValue: number,
  targetRemainingMetric: number,
): number
```

Parameters:
- `options` - - Adaptive pruning options.
- `currentPruneLevel` - - Current global prune level.
- `currentMetricValue` - - Current observed metric value.
- `targetRemainingMetric` - - Target remaining metric value.

Returns: Updated prune level.

### computePopulationMetrics

```ts
computePopulationMetrics(
  host: NeatLikeForPruning,
): PopulationMetrics
```

Parameters:
- `host` - - Neat instance with population.

Returns: Population metric summary.

### computeRampFraction

```ts
computeRampFraction(
  host: NeatLikeForPruning,
  options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; },
): number
```

Parameters:
- `host` - - Neat instance with generation state.
- `options` - - Evolution pruning options.

Returns: Fraction in [0,1] indicating ramp completion.

### computeTargetRemainingMetric

```ts
computeTargetRemainingMetric(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  adaptivePruneBaseline: number,
): number
```

Parameters:
- `options` - - Adaptive pruning options.
- `adaptivePruneBaseline` - - Baseline metric value.

Returns: Target remaining metric value.

### computeTargetSparsityNow

```ts
computeTargetSparsityNow(
  host: NeatLikeForPruning,
  options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; },
): number
```

Parameters:
- `host` - - Neat instance with generation state.
- `options` - - Evolution pruning options.

Returns: Target sparsity to apply for this generation.

### EvolutionPruningOptions

Evolution pruning options extracted from the Neat instance.

### initializeAdaptivePruningState

```ts
initializeAdaptivePruningState(
  host: NeatLikeForPruning,
): void
```

Parameters:
- `host` - - Neat instance with adaptive pruning state.

### NeatLikeForPruning

Minimal Neat instance contract required by pruning helpers.

Example:

const host: NeatLikeForPruning = {
  options: { evolutionPruning: { startGeneration: 5, targetSparsity: 0.4 } },
  generation: 10,
  population: [],
} as NeatLikeForPruning;

### PopulationMetrics

Summary of population metrics used by adaptive pruning.

### resolveActiveAdaptivePruningOptions

```ts
resolveActiveAdaptivePruningOptions(
  host: NeatLikeForPruning,
): { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; } | null
```

Parameters:
- `host` - - Neat instance with adaptive pruning options.

Returns: Adaptive pruning options when enabled, otherwise null.

### resolveActiveEvolutionPruningOptions

```ts
resolveActiveEvolutionPruningOptions(
  host: NeatLikeForPruning,
): { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; } | null
```

Parameters:
- `host` - - Neat instance with generation state.

Returns: Evolution pruning options when active, otherwise null.

### resolveAdaptivePruneBaseline

```ts
resolveAdaptivePruneBaseline(
  host: NeatLikeForPruning,
  currentMetricValue: number,
): number
```

Parameters:
- `host` - - Neat instance with adaptive baseline state.
- `currentMetricValue` - - Current observed metric value.

Returns: Baseline metric value used for adaptation.

### resolveObservedMetricValue

```ts
resolveObservedMetricValue(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  metrics: PopulationMetrics,
): number
```

Parameters:
- `options` - - Adaptive pruning options.
- `metrics` - - Population metric summary.

Returns: Current observed metric value used for adaptation.

### shouldAdjustAdaptivePruning

```ts
shouldAdjustAdaptivePruning(
  options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; },
  currentMetricValue: number,
  targetRemainingMetric: number,
  adaptivePruneBaseline: number,
): boolean
```

Parameters:
- `options` - - Adaptive pruning options.
- `currentMetricValue` - - Current observed metric value.
- `targetRemainingMetric` - - Target remaining metric value.
- `adaptivePruneBaseline` - - Baseline metric value.

Returns: True when pruning should be adjusted.

## neat/neat.species.utils.ts

### backfillExtendedHistory

```ts
backfillExtendedHistory(
  history: SpeciesHistoryEntry[],
  context: { _species?: SpeciesLike[] | undefined; _fallbackInnov?: ((c: ConnectionLike) => number) | undefined; },
): void
```

Parameters:
- `history` - - Recorded history to enrich in place.
- `context` - - Neat instance context for lookups.

### shouldAugmentExtendedHistory

```ts
shouldAugmentExtendedHistory(
  options: NeatOptions | undefined,
): boolean
```

Parameters:
- `options` - - Current Neat options.

Returns: True when extended history is enabled.

### SPECIES_HISTORY_DEFAULT_ENABLED_RATIO

Default enabled ratio when no connections exist.

### SPECIES_HISTORY_DEFAULT_INNOVATION_ID

Default innovation id when none is present.

### SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE

Default innovation range when data is missing.

### SPECIES_HISTORY_INITIAL_MAX_INNOVATION

Initial max tracker for innovation range aggregation.

### SPECIES_HISTORY_INITIAL_MIN_INNOVATION

Initial min tracker for innovation range aggregation.

### SPECIES_HISTORY_ZERO

Shared zero value for counters and defaults.

## neat/neat.adaptive.utils.ts

### ACCEPTANCE_LOWER_MULTIPLIER

Lower acceptance multiplier.

### ACCEPTANCE_UPPER_MULTIPLIER

Upper acceptance multiplier.

### AdaptiveMutationConfig

### ADJUST_RATE_DEFAULT

Default adjustment rate in minimal criterion.

### adjustConnectionBudget

```ts
adjustConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  factors: { increaseFactor: number; stagnationFactor: number; },
  noveltyFactor: number,
  history: number[],
): void
```

Adjust connection budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### adjustNodeBudget

```ts
adjustNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  factors: { increaseFactor: number; stagnationFactor: number; },
  noveltyFactor: number,
  history: number[],
): void
```

Adjust node budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### ANCESTOR_UNIQ_MODE_EPSILON

Ancestor uniqueness epsilon mode.

### ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE

Ancestor uniqueness lineage pressure mode.

### AncestorUniqAdaptiveConfig

### ANNEAL_BASELINE_GENERATIONS

Baseline generations for annealing progress.

### ANNEAL_PROGRESS_MAX

Maximum progress ratio used in annealing.

### applyAdaptiveSchedule

```ts
applyAdaptiveSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply adaptive complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance with adaptive state.
- `config` - - Complexity budget configuration.

### applyAnnealDelta

```ts
applyAnnealDelta(
  baseDelta: number,
  settings: MutationSettings,
): number
```

Apply annealing adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `settings` - - Resolved settings.

Returns: Adjusted delta.

### applyComplexityBudgetSchedule

```ts
applyComplexityBudgetSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply the complexity budget schedule for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyEpsilonAdjustment

```ts
applyEpsilonAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply dominance-epsilon adjustments when configured.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### applyExploreLowDelta

```ts
applyExploreLowDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply explore-low adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyLineagePressureAdjustment

```ts
applyLineagePressureAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
): void
```

Apply lineage pressure strength adjustments.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.

### applyLinearSchedule

```ts
applyLinearSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply linear complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyMutationAmount

```ts
applyMutationAmount(
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  settings: MutationSettings,
  randomSource: () => number,
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): void
```

Apply mutation-amount adjustments to a genome.

Parameters:
- `genome` - - Current genome.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

### applyMutationsToPopulation

```ts
applyMutationsToPopulation(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  partitions: MutationPartitions,
  settings: MutationSettings,
  randomSource: () => number,
): MutationOutcome
```

Apply mutation updates to the population.

Parameters:
- `population` - - Full population to mutate.
- `partitions` - - Scored partitions.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.

Returns: Mutation outcome flags.

### applyOperatorDecay

```ts
applyOperatorDecay(
  stats: Map<string, { success: number; attempts: number; }>,
  entries: [string, { success: number; attempts: number; }][],
  decay: number,
): void
```

Apply exponential decay to each operator statistic entry.

Parameters:
- `stats` - - Operator statistics map.
- `entries` - - Operator stat entries to update.
- `decay` - - Decay factor.

### applyRejection

```ts
applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void
```

Zero scores below the final threshold.

Parameters:
- `engine` - - NEAT engine instance.
- `threshold` - - Final MC threshold.

### applyTwoTierAmountDelta

```ts
applyTwoTierAmountDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply two-tier adjustments to amount delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierDelta

```ts
applyTwoTierDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply two-tier adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierFallback

```ts
applyTwoTierFallback(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  settings: MutationSettings,
): void
```

Apply two-tier fallback balancing.

Parameters:
- `population` - - Population of genomes.
- `settings` - - Resolved settings.

### applyUniquenessAdjustment

```ts
applyUniquenessAdjustment(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply an adjustment for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### BUDGET_GROWTH_MULTIPLIER

Default budget growth multiplier.

### clampNodeBudget

```ts
clampNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Clamp node budget to configured minimum.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### clampValue

```ts
clampValue(
  value: number,
  min: number,
  max: number,
): number
```

Clamp a value between min and max bounds.

Parameters:
- `value` - - Value to clamp.
- `min` - - Minimum bound.
- `max` - - Maximum bound.

Returns: Clamped value.

### collectOperatorStatsEntries

```ts
collectOperatorStatsEntries(
  stats: Map<string, { success: number; attempts: number; }>,
): [string, { success: number; attempts: number; }][]
```

Collect operator statistic entries for processing.

Parameters:
- `stats` - - Operator statistics map.

Returns: Array of operator stat entries.

### collectScoredGenomes

```ts
collectScoredGenomes(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]
```

Collect genomes with numeric scores.

Parameters:
- `population` - - Population of genomes.

Returns: Scored genomes.

### collectScores

```ts
collectScores(
  engine: NeatLikeWithAdaptive,
): number[]
```

Collect population scores into a snapshot array.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Array of scores (missing scores treated as 0).

### COMPLEXITY_MODE_ADAPTIVE

Complexity budget adaptive mode string.

### COMPLEXITY_MODE_LINEAR

Complexity budget linear mode string.

### ComplexityBudgetConfig

### computeAcceptance

```ts
computeAcceptance(
  scores: number[],
  threshold: number,
): number
```

Compute acceptance metrics for the current threshold.

Parameters:
- `scores` - - Population score snapshot.
- `threshold` - - Current MC threshold.

Returns: Acceptance proportion.

### computeAdjustmentFactors

```ts
computeAdjustmentFactors(
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  history: number[],
): { increaseFactor: number; stagnationFactor: number; }
```

Compute adjustment factors for budget growth and decay.

Parameters:
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `history` - - Rolling history of best scores.

Returns: Adjustment factors (increase and stagnation multipliers).

### computeNoveltyFactor

```ts
computeNoveltyFactor(
  engine: NeatLikeWithAdaptive,
): number
```

Compute novelty factor based on archive size.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Novelty multiplier (0.9 if archive small, 1.0 otherwise).

### computeSlope

```ts
computeSlope(
  history: number[],
): number
```

Compute linear regression slope using ordinary least squares.

Parameters:
- `history` - - Rolling history of best scores.

Returns: OLS slope estimate.

### computeTrends

```ts
computeTrends(
  history: number[],
): { improvement: number; slope: number; }
```

Compute improvement and slope trends from score history.

Parameters:
- `history` - - Rolling history of best scores.

Returns: Trend metrics (improvement and slope).

### createRandomDelta

```ts
createRandomDelta(
  sigmaBase: number,
  randomSource: () => number,
): number
```

Create a signed random delta scaled by sigma.

Parameters:
- `sigmaBase` - - Sigma scaling factor.
- `randomSource` - - Random number provider.

Returns: Signed delta.

### decayOperatorStat

```ts
decayOperatorStat(
  operatorStat: { success: number; attempts: number; },
  decay: number,
): { success: number; attempts: number; }
```

Apply decay to a single operator statistic record.

Parameters:
- `operatorStat` - - Operator statistic record.
- `decay` - - Decay factor.

Returns: Decayed operator statistic record.

### DEFAULT_ADAPT_EVERY

Default adapt-every cadence for adaptive mutation.

### DEFAULT_ANCESTOR_UNIQ_ADJUST

Default adjustment magnitude for uniqueness nudges.

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

Default cooldown (generations) for ancestor-uniqueness adjustments.

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

Default upper bound for acceptable ancestor uniqueness.

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

Default lower bound for acceptable ancestor uniqueness.

### DEFAULT_CB_INCREASE_FACTOR

Default increase factor for adaptive schedule.

### DEFAULT_CB_STAGNATION_FACTOR

Default stagnation factor for adaptive schedule.

### DEFAULT_IMPROVEMENT_WINDOW

Default score history window.

### DEFAULT_INITIAL_MUTATION_RATE

Default initial mutation rate used for balance checks.

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

Default lineage pressure strength when initializing the option.

### DEFAULT_MAX_MUTATION_AMOUNT

Default maximum mutation amount.

### DEFAULT_MAX_MUTATION_RATE

Default maximum per-genome mutation rate.

### DEFAULT_MIN_MUTATION_AMOUNT

Default minimum mutation amount.

### DEFAULT_MIN_MUTATION_RATE

Default minimum per-genome mutation rate.

### DEFAULT_MUTATION_AMOUNT

Default mutation amount when genome value is missing.

### DEFAULT_MUTATION_AMOUNT_SIGMA

Default mutation amount sigma for perturbations.

### DEFAULT_MUTATION_SIGMA

Default mutation sigma for adaptive mutation.

### DENOMINATOR_FALLBACK

Fallback denominator to avoid divide-by-zero.

### ensureLineagePressureState

```ts
ensureLineagePressureState(
  engine: NeatLikeWithAdaptive,
): { enabled?: boolean | undefined; mode?: string | undefined; strength?: number | undefined; }
```

Ensure lineage pressure state is available.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Lineage pressure configuration object.

### EXPLORE_LOW_DECREASE_MULTIPLIER

Multiplicative decay for explore-low strategy (top half).

### EXPLORE_LOW_INCREASE_MULTIPLIER

Multiplicative boost for explore-low strategy (bottom half).

### extractAncestorUniqueness

```ts
extractAncestorUniqueness(
  engine: NeatLikeWithAdaptive,
): number | undefined
```

Extract the latest ancestor-uniqueness metric from telemetry.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Ancestor uniqueness value or undefined when missing.

### FIVE

Constant: five value.

### FOUR

Constant: four value.

### Genome

### HALF_INDEX_DIVISOR

Divisor used to split populations in half.

### HISTORY_MIN_IMPROVEMENT_COUNT

Minimum history length to compute improvement.

### HISTORY_MIN_SLOPE_COUNT

Minimum history length to compute slope.

### initializeConnectionBudget

```ts
initializeConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Initialize connection budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializeNodeBudget

```ts
initializeNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Initialize node budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializePhaseState

```ts
initializePhaseState(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Ensure phase state is initialized.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### initializeThreshold

```ts
initializeThreshold(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): void
```

Initialize MC threshold if missing.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Minimal-criterion adaptive configuration.

### isCooldownSatisfied

```ts
isCooldownSatisfied(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): boolean
```

Determine whether the cooldown window has elapsed.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: True when adjustment is allowed.

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

Multiplier when decreasing lineage pressure strength.

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

Multiplier when increasing lineage pressure strength.

### LINEAGE_PRESSURE_MODE_SPREAD

Lineage pressure spread mode.

### LINEAR_HORIZON_DEFAULT

Default horizon for linear schedule.

### MINIMAL_TOPOLOGY_OFFSET

Offset added to input/output for minimal topology.

### MinimalCriterionAdaptiveConfig

### MUTATION_SIGMA_SCALE

Scale applied to mutation sigma for perturbations.

### MUTATION_STRATEGY_ANNEAL

Strategy identifier for annealed mutation.

### MUTATION_STRATEGY_EXPLORE_LOW

Strategy identifier for explore-low mutation.

### MUTATION_STRATEGY_TWO_TIER

Strategy identifier for two-tier mutation.

### MutationOutcome

### MutationPartitions

### MutationSettings

### NeatLikeWithAdaptive

Minimal interface for NEAT instances with adaptive features.
Exported for use in tests and type-safe function calls.

### NEGATIVE_ONE

Constant: negative one for last index.

### normalizeSlope

```ts
normalizeSlope(
  slope: number,
  initialScore: number,
): number
```

Normalize slope magnitude relative to initial score.

Parameters:
- `slope` - - Raw OLS slope.
- `initialScore` - - First score in history window.

Returns: Normalized slope clamped to [-2, 2].

### NOVELTY_ARCHIVE_MIN_SIZE

Novelty archive minimum size.

### NOVELTY_FACTOR_DEFAULT

Novelty factor when archive is sufficient.

### NOVELTY_FACTOR_SMALL

Novelty factor when archive is small.

### ONE

Constant: one value.

### ONE_HUNDRED

Constant: one hundred value.

### OPERATOR_DECAY_DEFAULT

Default operator decay factor.

### OperatorAdaptationConfig

### PHASE_COMPLEXIFY

Phase label for complexify.

### PHASE_LENGTH_DEFAULT

Default phase length in generations.

### PHASE_SIMPLIFY

Phase label for simplify.

### PhasedComplexityConfig

### PROGRESS_RATIO_MAX

Maximum progress ratio for scheduling.

### recordAdjustment

```ts
recordAdjustment(
  engine: NeatLikeWithAdaptive,
): void
```

Record the generation when an adjustment is applied.

Parameters:
- `engine` - - NEAT engine instance.

### resolveAdjustmentMagnitude

```ts
resolveAdjustmentMagnitude(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): number
```

Resolve adjustment magnitude for nudging controlled parameters.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Adjustment magnitude.

### resolveAmountDelta

```ts
resolveAmountDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Resolve mutation-amount delta based on strategy.

Parameters:
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Signed mutation amount delta.

### resolveMutationSettings

```ts
resolveMutationSettings(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): MutationSettings
```

Resolve mutation settings derived from configuration and engine state.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Adaptive mutation configuration.

Returns: Resolved mutation settings.

### resolveNextPhase

```ts
resolveNextPhase(
  currentPhase: string,
): string
```

Resolve next phase name.

Parameters:
- `currentPhase` - - Current phase label.

Returns: Next phase label.

### resolveOperatorDecay

```ts
resolveOperatorDecay(
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; alpha?: number | undefined; decay?: number | undefined; },
): number
```

Resolve the decay factor for operator statistics.

Parameters:
- `config` - - Operator adaptation configuration.

Returns: Decay factor for exponential smoothing.

### resolveRandomSource

```ts
resolveRandomSource(
  engine: NeatLikeWithAdaptive,
): () => number
```

Resolve a random source that matches the legacy RNG usage.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Random number provider.

### resolveRateDelta

```ts
resolveRateDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Resolve mutation-rate delta based on strategy.

Parameters:
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Signed mutation rate delta.

### resolveTargetSettings

```ts
resolveTargetSettings(
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): { targetAcceptance: number; adjustRate: number; }
```

Resolve target acceptance and adjust rate settings.

Parameters:
- `config` - - Minimal-criterion adaptive configuration.

Returns: Target settings.

### resolveUniquenessThresholds

```ts
resolveUniquenessThresholds(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): { lowThreshold: number; highThreshold: number; }
```

Resolve thresholds for ancestor-uniqueness decisions.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Threshold bounds.

### RNG_CENTER_OFFSET

Random offset for signed deltas.

### RNG_SPREAD_MULTIPLIER

Random range multiplier for signed deltas.

### shouldAdaptThisGeneration

```ts
shouldAdaptThisGeneration(
  generation: number,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): boolean
```

Check whether mutation adaptation should run this generation.

Parameters:
- `generation` - - Current generation index.
- `config` - - Adaptive mutation configuration.

Returns: True if adaptation should run.

### shouldApplyTwoTierFallback

```ts
shouldApplyTwoTierFallback(
  strategy: string,
  outcome: MutationOutcome,
): boolean
```

Determine whether a two-tier fallback is needed.

Parameters:
- `strategy` - - Mutation strategy identifier.
- `outcome` - - Mutation outcome flags.

Returns: True if fallback should run.

### SLOPE_BOOST_MULTIPLIER

Slope boost multiplier for adaptive increase factor.

### SLOPE_NORMALIZE_CLAMP

Clamp magnitude for slope normalization.

### SLOPE_PENALTY_MULTIPLIER

Slope penalty multiplier for stagnation factor.

### sortScoredGenomes

```ts
sortScoredGenomes(
  scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]
```

Sort scored genomes in ascending score order.

Parameters:
- `scoredGenomes` - - Scored genomes.

Returns: Sorted genomes.

### splitScoredGenomes

```ts
splitScoredGenomes(
  scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): MutationPartitions
```

Split scored genomes into top and bottom halves.

Parameters:
- `scoredGenomes` - - Sorted scored genomes.

Returns: Partitions used by strategy rules.

### TARGET_ACCEPTANCE_DEFAULT

Default target acceptance in minimal criterion.

### TEN

Constant: ten value.

### THREE

Constant: three value.

### togglePhaseIfNeeded

```ts
togglePhaseIfNeeded(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Toggle phase if the current phase has exceeded its length.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### TWO

Constant: two value.

### updateScoreHistory

```ts
updateScoreHistory(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): number[]
```

Update rolling score history with current best score.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

Returns: Rolling history array after update.

### updateThreshold

```ts
updateThreshold(
  engine: NeatLikeWithAdaptive,
  acceptance: number,
  tuning: { targetAcceptance: number; adjustRate: number; },
): void
```

Update the MC threshold based on acceptance proportion.

Parameters:
- `engine` - - NEAT engine instance.
- `acceptance` - - Observed acceptance proportion.
- `tuning` - - Target acceptance and adjustment settings.

### ZERO

Constant: zero value.

## neat/neat.evaluate.utils.ts

### AUTO_COEFF_ADJUST_DEFAULT

Default adjustment rate for auto distance coefficient tuning.

### AUTO_COEFF_MAX_DEFAULT

Default maximum coefficient for auto distance coefficient tuning.

### AUTO_COEFF_MIN_DEFAULT

Default minimum coefficient for auto distance coefficient tuning.

### COMPAT_MAX_THRESHOLD_DEFAULT

Default maximum compatibility threshold.

### COMPAT_MIN_THRESHOLD_DEFAULT

Default minimum compatibility threshold.

### COMPAT_THRESHOLD_DEFAULT

Default compatibility threshold when not provided.

### DISTANCE_COEFF_DEFAULT

Default coefficient value when not provided.

### DiversityStats

Diversity statistics tracked during evaluation.

The values are optional because different evaluations may only compute a
subset of metrics.

### ensureDiversityStatsContainer

```ts
ensureDiversityStatsContainer(
  controller: NeatControllerForEval,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.

Returns: void.

### ENTROPY_ADJUST_DEFAULT

Default adjustment rate for compatibility tuning.

### ENTROPY_DEADBAND_DEFAULT

Default deadband for compatibility tuning.

### ENTROPY_TARGET_DEFAULT

Default target entropy for compatibility tuning.

### ENTROPY_VAR_ADJUST_DEFAULT

Default adjustment rate for entropy sharing.

### ENTROPY_VAR_HIGH_BAND

Upper band multiplier for entropy variance tuning.

### ENTROPY_VAR_LOW_BAND

Lower band multiplier for entropy variance tuning.

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

Default maximum sigma for entropy sharing.

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

Default minimum sigma for entropy sharing.

### ENTROPY_VAR_TARGET_DEFAULT

Default target variance for entropy sharing.

### GenomeForEvaluation

Genome with score, novelty, and clearing capabilities.

This interface describes the minimal genome shape required by evaluation
helpers. It intentionally stays permissive for compatibility with legacy
genome variants while documenting the expected properties.

### NeatControllerForEval

NEAT controller interface for evaluation.

This interface models the subset of a NEAT controller used by the evaluation
helpers. It includes options, population data, and optional adaptive tuning
hooks.

### NOVELTY_ARCHIVE_CAP

Maximum number of entries stored in the novelty archive.

### NOVELTY_DEFAULT_BLEND

Default blend factor for novelty vs. fitness.

### NOVELTY_DEFAULT_NEIGHBORS

Default neighbor count for novelty calculation.

### NoveltyArchiveEntry

Novelty archive entry with descriptor and novelty score.

Entries store a descriptor vector alongside the computed novelty so the
archive can seed future novelty calculations.

### ObjectiveDef

Objective definition for multi-objective optimization.

Objectives are registered dynamically to guide evaluation and selection.

### runAutoDistanceCoefficientTuning

```ts
runAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runAutoEntropyObjectiveInjection

```ts
runAutoEntropyObjectiveInjection(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runEntropyCompatibilityTuning

```ts
runEntropyCompatibilityTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runEntropySharingTuning

```ts
runEntropySharingTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runFitnessEvaluation

```ts
runFitnessEvaluation(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): Promise<void>
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Promise<void> after fitness evaluation completes.

### runLightweightSpeciation

```ts
runLightweightSpeciation(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runNoveltyBlendAndArchive

```ts
runNoveltyBlendAndArchive(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### VARIANCE_DECREASE_THRESHOLD

Variance decrease threshold multiplier.

### VARIANCE_INCREASE_THRESHOLD

Variance increase threshold multiplier.

## neat/neat.mutation.utils.ts

### applyAddConnMutation

```ts
applyAddConnMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply an ADD_CONN mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyAddNodeMutation

```ts
applyAddNodeMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply an ADD_NODE mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyMutationOperator

```ts
applyMutationOperator(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply a mutation operator to a genome and invalidate caches as needed.

Parameters:
- `genome` - - genome to mutate
- `mutationMethod` - - mutation operator to apply
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyOperatorAdaptationForSelect

```ts
applyOperatorAdaptationForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[]
```

Apply operator adaptation weighting to the pool when enabled.

Parameters:
- `pool` - - base pool
- `internal` - - neat controller context

Returns: augmented pool

### applyOperatorBanditForSelect

```ts
applyOperatorBanditForSelect(
  pool: MutationMethod[],
  fallbackMethod: MutationMethod,
  internal: NeatControllerForMutation,
): MutationMethod
```

Apply operator bandit selection if enabled.

Parameters:
- `pool` - - operator pool
- `fallbackMethod` - - method used when bandit is disabled
- `internal` - - neat controller context

Returns: selected method

### applyPhasedComplexityForSelect

```ts
applyPhasedComplexityForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[]
```

Apply phased complexity adjustments to the pool when enabled.

Parameters:
- `pool` - - base operator pool
- `internal` - - neat controller context

Returns: pool with phased complexity adjustments

### applySplitWithExistingRecord

```ts
applySplitWithExistingRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number; },
  splitRecord: { newNodeGeneId: number; inInnov: number; outInnov: number; },
  NodeClass: new (type: "input" | "output" | "hidden") => unknown,
): void
```

Apply a split using an existing innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `splitRecord` - - existing innovation record
- `NodeClass` - - node constructor

Returns: void

### applySplitWithNewRecord

```ts
applySplitWithNewRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number; },
  NodeClass: new (type: "input" | "output" | "hidden") => unknown,
  internal: NeatControllerForMutation,
): void
```

Apply a split and create a new innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `NodeClass` - - node constructor
- `internal` - - neat controller context

Returns: void

### assignInnovationForConnection

```ts
assignInnovationForConnection(
  connection: ConnectionWithMetadata,
  pairNodes: { symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; },
  internal: NeatControllerForMutation,
): void
```

Assign an innovation id for a new connection, reusing when possible.

Parameters:
- `connection` - - newly created connection
- `pairNodes` - - resolved pair metadata
- `internal` - - neat controller context

Returns: void

### assignInnovationsForNewSplit

```ts
assignInnovationsForNewSplit(
  newNode: NodeWithMetadata,
  splitConnections: { incomingConnection?: ConnectionWithMetadata | undefined; outgoingConnection?: ConnectionWithMetadata | undefined; },
  internal: NeatControllerForMutation,
): { newNodeGeneId: number; inInnov: number; outInnov: number; }
```

Assign new innovations for a split and build the innovation record.

Parameters:
- `newNode` - - newly created hidden node
- `splitConnections` - - incoming/outgoing connections
- `internal` - - neat controller context

Returns: innovation record for the split

### buildLegacyKeyForConn

```ts
buildLegacyKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string
```

Build a legacy directional innovation key.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: directional innovation key

### buildSplitDescriptor

```ts
buildSplitDescriptor(
  connectionToSplit: ConnectionWithMetadata,
): { splitKey: string; originalWeight: number; }
```

Build the split descriptor used for innovation lookup and connection creation.

Parameters:
- `connectionToSplit` - - connection being split

Returns: split descriptor

### buildSymmetricKeyForConn

```ts
buildSymmetricKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string
```

Build a symmetric innovation key for an unordered node pair.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: symmetric innovation key

### captureStructuralSizes

```ts
captureStructuralSizes(
  genome: GenomeWithMetadata,
): { beforeNodes: number; beforeConns: number; }
```

Capture structural sizes used to evaluate operator success.

Parameters:
- `genome` - - genome to inspect

Returns: structural size snapshot

### chooseConnectionForSplit

```ts
chooseConnectionForSplit(
  enabledConnectionsList: ConnectionWithMetadata[],
  internal: NeatControllerForMutation,
): ConnectionWithMetadata | null
```

Choose a random enabled connection to split.

Parameters:
- `enabledConnectionsList` - - candidate connections
- `internal` - - neat controller context

Returns: selected connection or null

### choosePairForConn

```ts
choosePairForConn(
  pairs: [NodeWithMetadata, NodeWithMetadata][],
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata] | null
```

Choose a pair deterministically when only one candidate exists.

Parameters:
- `pairs` - - selection pool
- `internal` - - neat controller context

Returns: chosen pair or null

### chooseRandomNodeForDeadEnds

```ts
chooseRandomNodeForDeadEnds(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null
```

Choose a random node from candidates for dead-end repair.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### chooseRandomNodeForMinHidden

```ts
chooseRandomNodeForMinHidden(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null
```

Choose a random node from a candidate list.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectCandidatePairsForConn

```ts
collectCandidatePairsForConn(
  genomeToInspect: GenomeWithMetadata,
): [NodeWithMetadata, NodeWithMetadata][]
```

Collect legal (from,to) node pairs not already connected.

Parameters:
- `genomeToInspect` - - genome to scan

Returns: candidate node pairs

### collectEnabledConnections

```ts
collectEnabledConnections(
  genomeToInspect: GenomeWithMetadata,
): ConnectionWithMetadata[]
```

Collect all enabled connections from a genome.

Parameters:
- `genomeToInspect` - - genome to inspect

Returns: enabled connections list

### collectNodeGroupsForDeadEnds

```ts
collectNodeGroupsForDeadEnds(
  networkToInspect: GenomeWithMetadata,
): { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; }
```

Collect categorized node arrays for dead-end repair.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### collectNodeGroupsForMinHidden

```ts
collectNodeGroupsForMinHidden(
  networkToInspect: GenomeWithMetadata,
): { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; }
```

Collect categorized node arrays for the network.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### computeMinimumHiddenSize

```ts
computeMinimumHiddenSize(
  inputCount: number,
  outputCount: number,
  explicitMinimumHidden: number | undefined,
  hiddenMultiplier: number | undefined,
): number
```

Compute the minimum hidden node count using explicit or multiplier-based settings.

Parameters:
- `inputCount` - - Number of input nodes in the network.
- `outputCount` - - Number of output nodes in the network.
- `explicitMinimumHidden` - - Optional explicit minimum hidden count.
- `hiddenMultiplier` - - Optional multiplier used when explicit minimum is absent.

Returns: Minimum hidden node requirement.

### connectChosenPair

```ts
connectChosenPair(
  genomeToEdit: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; },
): ConnectionWithMetadata | undefined
```

Create the connection for the chosen pair.

Parameters:
- `genomeToEdit` - - genome to edit
- `pairNodes` - - resolved pair nodes

Returns: created connection or undefined

### connectIfCandidatesExistForDeadEnds

```ts
connectIfCandidatesExistForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  anchorNode: NodeWithMetadata,
  candidates: NodeWithMetadata[],
  reverse: boolean,
  internal: NeatControllerForMutation,
): void
```

Connect a node to a random candidate if candidates exist.

Parameters:
- `networkToEdit` - - network to edit
- `anchorNode` - - node to connect from/to
- `candidates` - - candidate nodes for connection
- `reverse` - - whether to connect candidate -> anchor
- `internal` - - neat controller context

Returns: void

### connectSplitEdges

```ts
connectSplitEdges(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  newNode: NodeWithMetadata,
  originalWeight: number,
): { incomingConnection?: ConnectionWithMetadata | undefined; outgoingConnection?: ConnectionWithMetadata | undefined; }
```

Create the incoming and outgoing split connections.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `newNode` - - newly created hidden node
- `originalWeight` - - weight to preserve on the outgoing connection

Returns: incoming/outgoing connection handles

### createsCycle

```ts
createsCycle(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): boolean
```

Detect whether adding a connection would create a cycle.

Parameters:
- `sourceNode` - - source node of the new connection
- `targetNode` - - target node of the new connection

Returns: true when a cycle is detected

### disconnectOriginalConnection

```ts
disconnectOriginalConnection(
  genomeToEdit: GenomeWithMetadata,
  connectionToRemove: ConnectionWithMetadata,
): void
```

Disconnect the original connection before inserting the split node.

Parameters:
- `genomeToEdit` - - genome to edit
- `connectionToRemove` - - original connection to remove

Returns: void

### ensureBootstrapConnection

```ts
ensureBootstrapConnection(
  genomeToSeed: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure the genome has at least one connection by linking input to output.

Parameters:
- `genomeToSeed` - - genome that may need a bootstrap connection
- `internal` - - neat controller context retained for compatibility with existing callers

Returns: void

### ensureHiddenConnectivityForDeadEnds

```ts
ensureHiddenConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenConnectivityForMinHidden

```ts
ensureHiddenConnectivityForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenNodeCountForMinHidden

```ts
ensureHiddenNodeCountForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToEdit: { hiddenNodes: NodeWithMetadata[]; },
  minimumHidden: number,
  maxNodesLimit: number,
): Promise<void>
```

Ensure the network has at least the minimum number of hidden nodes.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToEdit` - - grouped node arrays
- `minimumHidden` - - minimum hidden nodes required
- `maxNodesLimit` - - maximum allowed nodes

Returns: Promise resolving when nodes are created

### ensureIncomingConnectionForMinHidden

```ts
ensureIncomingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure a hidden node has at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureInputConnectivityForDeadEnds

```ts
ensureInputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure all input nodes have at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureOutgoingConnectionForMinHidden

```ts
ensureOutgoingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure a hidden node has at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureOutputConnectivityForDeadEnds

```ts
ensureOutputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure all output nodes have at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### filterPairsWithInnovations

```ts
filterPairsWithInnovations(
  pairs: [NodeWithMetadata, NodeWithMetadata][],
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata][]
```

Filter candidate pairs that already have innovation reuse keys.

Parameters:
- `pairs` - - candidate node pairs
- `internal` - - neat controller context

Returns: reuse candidates

### findFirstNodeByType

```ts
findFirstNodeByType(
  genomeToSearch: GenomeWithMetadata,
  nodeType: "input" | "output" | "hidden",
): NodeWithMetadata | undefined
```

Find the first node of a given type.

Parameters:
- `genomeToSearch` - - genome whose nodes are searched
- `nodeType` - - node type to match

Returns: the first matching node or undefined

### hasIncomingForDeadEnds

```ts
hasIncomingForDeadEnds(
  node: NodeWithMetadata,
): boolean
```

Check whether a node has any incoming connections.

Parameters:
- `node` - - node to inspect

Returns: true when incoming connections exist

### hasOutgoingForDeadEnds

```ts
hasOutgoingForDeadEnds(
  node: NodeWithMetadata,
): boolean
```

Check whether a node has any outgoing connections.

Parameters:
- `node` - - node to inspect

Returns: true when outgoing connections exist

### hasRequiredEndpointsForMinHidden

```ts
hasRequiredEndpointsForMinHidden(
  nodeGroupsToCheck: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; },
): boolean
```

Check whether the network has at least one input and output node.

Parameters:
- `nodeGroupsToCheck` - - grouped node arrays

Returns: true when inputs and outputs are present

### initializeAdaptiveMutation

```ts
initializeAdaptiveMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Initialize per-genome adaptive mutation parameters if configured.

Parameters:
- `genome` - - genome to initialize
- `internal` - - neat controller context

Returns: void

### isBlockedByRecurrentPolicyForSelect

```ts
isBlockedByRecurrentPolicyForSelect(
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): boolean
```

Check whether a mutation is blocked by recurrent connection policy.

Parameters:
- `mutationMethod` - - mutation operator to check
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isBlockedByStructuralLimitsForSelect

```ts
isBlockedByStructuralLimitsForSelect(
  mutationMethod: MutationMethod,
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): boolean
```

Check whether a mutation is blocked by structural limits.

Parameters:
- `mutationMethod` - - mutation operator to check
- `genome` - - genome to inspect
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isLegacyFFWPoolForSelect

```ts
isLegacyFFWPoolForSelect(
  configuredPool: MutationMethod[],
  methods: { mutation: unknown; },
): boolean
```

Check whether a pool matches the legacy FFW operator ordering.

Parameters:
- `configuredPool` - - configured operator pool
- `methods` - - methods module

Returns: true when the pool matches FFW

### isOperatorNamePrefixedForSelect

```ts
isOperatorNamePrefixedForSelect(
  method: MutationMethod,
  prefix: string,
): boolean
```

Check whether an operator name uses a specific prefix.

Parameters:
- `method` - - mutation operator
- `prefix` - - name prefix to match

Returns: true when the operator name matches the prefix

### maybeAddExtraConnection

```ts
maybeAddExtraConnection(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Optionally add an extra connection to increase exploration.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context

Returns: void

### MINIMUM_HIDDEN_BASELINE

Baseline minimum hidden nodes when no configuration is provided.

### mutateGenome

```ts
mutateGenome(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): Promise<void>
```

Mutate a single genome based on configured mutation policies.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: Promise resolving after mutation attempts complete

### normalizeMutationPoolForSelect

```ts
normalizeMutationPoolForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
  rawReturnForTest: boolean,
): MutationMethod[]
```

Normalize the configured mutation pool to a flat operator list.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW for tests

Returns: normalized mutation pool

### rebuildNetworkConnectionsForMinHidden

```ts
rebuildNetworkConnectionsForMinHidden(
  networkToEdit: GenomeWithMetadata,
): Promise<void>
```

Rebuild connection caches after structural edits.

Parameters:
- `networkToEdit` - - network to rebuild

Returns: Promise resolving after rebuild completes

### resolveEffectiveAmount

```ts
resolveEffectiveAmount(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number
```

Resolve the effective mutation amount for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation amount

### resolveEffectiveRate

```ts
resolveEffectiveRate(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number
```

Resolve the effective mutation rate for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation rate

### resolveFFWPolicyForSelect

```ts
resolveFFWPolicyForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
  rawReturnForTest: boolean,
): MutationMethod | MutationMethod[] | null
```

Resolve legacy FFW policy behavior, including test-specific returns.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW array for tests

Returns: mutation method or null when not handled

### resolveInsertIndex

```ts
resolveInsertIndex(
  genomeToEdit: GenomeWithMetadata,
  targetNode: NodeWithMetadata,
): number
```

Resolve the insertion index for a new node, keeping outputs at the end.

Parameters:
- `genomeToEdit` - - genome whose node list is updated
- `targetNode` - - original target node of the split connection

Returns: insertion index

### resolveMaxNodesForMinHidden

```ts
resolveMaxNodesForMinHidden(
  internal: NeatControllerForMutation,
): number
```

Resolve the maximum node limit for the network.

Parameters:
- `internal` - - neat controller context

Returns: maximum node limit

### resolveMinHiddenForMinHidden

```ts
resolveMinHiddenForMinHidden(
  networkToInspect: GenomeWithMetadata,
  maxNodesLimit: number,
  multiplier: number | undefined,
  internal: NeatControllerForMutation,
): number
```

Resolve the minimum hidden node requirement for the network.

Parameters:
- `networkToInspect` - - network to inspect
- `maxNodesLimit` - - maximum allowed nodes
- `multiplier` - - optional size multiplier
- `internal` - - neat controller context

Returns: minimum hidden node count

### resolvePairNodes

```ts
resolvePairNodes(
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
): { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }
```

Resolve nodes and innovation key details for a chosen pair.

Parameters:
- `chosenPair` - - pair to connect

Returns: resolved pair metadata

### sampleFromPoolForSelect

```ts
sampleFromPoolForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod | null
```

Sample a random method from the pool.

Parameters:
- `pool` - - operator pool
- `internal` - - neat controller context

Returns: sampled method or null

### selectConcreteMutationMethod

```ts
selectConcreteMutationMethod(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): Promise<MutationMethod | null>
```

Select a concrete mutation method, resolving any legacy arrays.

Parameters:
- `genome` - - genome to select for
- `internal` - - neat controller context

Returns: resolved mutation method or null

### selectPairPool

```ts
selectPairPool(
  allPairs: [NodeWithMetadata, NodeWithMetadata][],
  reusePairs: [NodeWithMetadata, NodeWithMetadata][],
): [NodeWithMetadata, NodeWithMetadata][]
```

Build the final selection pool based on reuse and hidden-node preference.

Parameters:
- `allPairs` - - all candidate pairs
- `reusePairs` - - pairs with historical innovations

Returns: selection pool

### shouldAbortForCycle

```ts
shouldAbortForCycle(
  genomeToInspect: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; },
): boolean
```

Determine whether adding the connection would create a cycle.

Parameters:
- `genomeToInspect` - - genome to inspect
- `pairNodes` - - resolved pair nodes

Returns: true if the connection should be aborted

### shouldInvalidateCaches

```ts
shouldInvalidateCaches(
  mutationMethod: MutationMethod,
  methods: { mutation: unknown; },
): boolean
```

Determine whether a mutation method invalidates cached structures.

Parameters:
- `mutationMethod` - - mutation operator to inspect
- `methods` - - mutation methods module

Returns: true when caches should be invalidated

### shouldMutateGenome

```ts
shouldMutateGenome(
  effectiveRate: number,
  internal: NeatControllerForMutation,
): boolean
```

Decide whether a genome should be mutated based on probability.

Parameters:
- `effectiveRate` - - effective mutation probability
- `internal` - - neat controller context

Returns: true when the genome should be mutated

### updateOperatorStatsIfNeeded

```ts
updateOperatorStatsIfNeeded(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  beforeSizes: { beforeNodes: number; beforeConns: number; },
  internal: NeatControllerForMutation,
): void
```

Update operator statistics when adaptation is enabled.

Parameters:
- `genome` - - genome used to compute after-sizes
- `mutationMethod` - - operator being recorded
- `beforeSizes` - - structural sizes captured before mutation
- `internal` - - neat controller context

Returns: void

### warnMissingEndpointsForMinHidden

```ts
warnMissingEndpointsForMinHidden(): void
```

Emit a warning when the network lacks input or output nodes.

Returns: void

## neat/neat.diversity.utils.ts

### arrayMean

```ts
arrayMean(
  values: number[],
): number
```

Compute the arithmetic mean of a numeric array. Returns 0 for empty arrays.

Parameters:
- `values` - - Values to average.

Returns: Arithmetic mean of the values.

### arrayVariance

```ts
arrayVariance(
  values: number[],
): number
```

Compute the variance (population variance) of a numeric array.
Returns 0 for empty arrays. Uses arrayMean internally.

Parameters:
- `values` - - Values to evaluate.

Returns: Population variance.

### calculateDiversityStats

```ts
calculateDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined
```

Compute diversity statistics for a NEAT population.

Parameters:
- `population` - - array of genome-like objects (nodes, connections, optional _depth)
- `compatibilityComputer` - - object exposing _compatibilityDistance(a,b)

Returns: DiversityStats object with all computed aggregates, or undefined if input empty

### calculateStructuralEntropy

```ts
calculateStructuralEntropy(
  graph: default,
): number
```

Compute the Shannon-style entropy of a network's out-degree distribution.

Parameters:
- `graph` - - Network instance to evaluate.

Returns: Shannon-style entropy value.

### CompatComputer

Minimal interface that provides a compatibility distance function.
Implementors should expose a compatible signature with legacy NEAT code.

### DiversityStats

Diversity statistics returned by computeDiversityStats.
Each field represents an aggregate metric for a NEAT population.

### GenomeWithMetrics

Minimal genome interface for diversity computations.

### MAX_COMPATIBILITY_SAMPLE

Maximum population sample size for compatibility comparisons.

### MAX_LINEAGE_PAIR_SAMPLE

Maximum lineage sample size for pairwise depth comparisons.

### NodeWithConnections

Minimal node interface with connections.

## neat/neat.selection.utils.ts

### calculateFitnessTotals

```ts
calculateFitnessTotals(
  population: GenomeWithScore[],
): { totalFitness: number; minFitnessShift: number; }
```

Compute the total fitness and minimal score shift for roulette selection.

Parameters:
- `population` - - Genomes in the current population.

Returns: Aggregated fitness totals.

### calculateTotalScore

```ts
calculateTotalScore(
  population: GenomeWithScore[],
): number
```

Calculate the total fitness across the population.

Parameters:
- `population` - - Genomes in the current population.

Returns: The sum of all scores.

### DEFAULT_POWER

Default power exponent for POWER selection when none is configured.

### DEFAULT_SCORE

Default score when a genome has no explicit score.

### DEFAULT_TOURNAMENT_PROBABILITY

Default tournament win probability when none is configured.

### DEFAULT_TOURNAMENT_SIZE

Default tournament size when none is configured.

### ensurePopulationEvaluated

```ts
ensurePopulationEvaluated(
  internal: NeatLikeWithSelection,
): void
```

Ensure population scores exist by running evaluation if needed.

Parameters:
- `internal` - - The Neat instance containing `population` and `evaluate`.

Returns: void

### ensurePopulationSortedDescending

```ts
ensurePopulationSortedDescending(
  internal: NeatLikeWithSelection,
): void
```

Ensure the population is sorted descending by score when out of order.

Parameters:
- `internal` - - The Neat instance containing `population`.

Returns: void

### ensurePopulationSortedDescendingForPower

```ts
ensurePopulationSortedDescendingForPower(
  selectionContext: SelectionContext,
): void
```

Ensure the population is sorted descending by score if the first two
entries are out of order.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: void

### FIRST_INDEX

Index of the first element in an array.

### GenomeWithScore

Genome with a fitness score and arbitrary additional metadata.

Example:

const genome: GenomeWithScore = { score: 42, id: 'g-1' };

### getRandomPopulationMember

```ts
getRandomPopulationMember(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a random population member using the configured RNG.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: A randomly chosen genome.

### INITIAL_CUMULATIVE_FITNESS

Initial cumulative fitness value for threshold scans.

### INITIAL_MOST_NEGATIVE_SCORE

Initial most-negative score sentinel for fitness scans.

### INITIAL_TOTAL_FITNESS

Initial total fitness accumulator value.

### LAST_ELEMENT_INDEX

Index used with `at()` to access the last element.

### LAST_INDEX_OFFSET

Offset for retrieving the last element via length arithmetic.

### LOOP_INDEX_INCREMENT

Step size for index-based loops.

### NeatLikeWithSelection

NEAT-like instance extended with selection-specific state and helpers.

Example:

const selectionHost: NeatLikeWithSelection = {
  population: [],
  options: { selection: { name: 'TOURNAMENT', size: 3 } },
  _getRNG: () => Math.random,
  sort: () => undefined,
} as NeatLikeWithSelection;

### pickByShiftedThreshold

```ts
pickByShiftedThreshold(
  population: GenomeWithScore[],
  selectionThreshold: number,
  minFitnessShift: number,
): GenomeWithScore | undefined
```

Pick the first genome whose shifted cumulative fitness exceeds the threshold.

Parameters:
- `population` - - Genomes in the current population.
- `selectionThreshold` - - Random threshold in shifted fitness space.
- `minFitnessShift` - - Amount added to each score to shift negatives.

Returns: The chosen genome if one crosses the threshold.

### pickTournamentWinner

```ts
pickTournamentWinner(
  selectionContext: SelectionContext,
  sortedParticipants: GenomeWithScore[],
): GenomeWithScore
```

Select a winner from sorted tournament participants.

Parameters:
- `selectionContext` - - Shared selection state.
- `sortedParticipants` - - Participants sorted by descending score.

Returns: The chosen tournament winner.

### resolveTournamentOverflow

```ts
resolveTournamentOverflow(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Resolve what happens when the tournament size exceeds population size.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: A fallback parent genome.

### sampleTournamentParticipants

```ts
sampleTournamentParticipants(
  selectionContext: SelectionContext,
  tournamentSize: number,
): GenomeWithScore[]
```

Sample a list of tournament participants (with possible repeats).

Parameters:
- `selectionContext` - - Shared selection state.
- `tournamentSize` - - Number of competitors to sample.

Returns: Sampled participants.

### SECOND_INDEX

Index of the second element in an array.

### selectParentByFitnessProportionate

```ts
selectParentByFitnessProportionate(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a parent using roulette-wheel fitness proportionate selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

### selectParentByPower

```ts
selectParentByPower(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a parent by power-law distribution on the sorted population.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

### selectParentByStrategy

```ts
selectParentByStrategy(
  internal: NeatLikeWithSelection,
): GenomeWithScore
```

Select a parent genome according to configured selection strategy.

Parameters:
- `internal` - - The Neat instance containing population and options.

Returns: A genome object chosen as the parent.

### selectParentByTournament

```ts
selectParentByTournament(
  selectionContext: SelectionContext,
): GenomeWithScore
```

Select a parent by tournament selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

## neat/neat.telemetry.utils.ts

### applyComplexityStatsMonoObjective

```ts
applyComplexityStatsMonoObjective(
  telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  populationSnapshot: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Attach complexity stats for mono-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `entry` - - Telemetry entry to update.

### applyComplexityStatsMultiObjective

```ts
applyComplexityStatsMultiObjective(
  telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Attach complexity stats for multi-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyFastModeDefaults

```ts
applyFastModeDefaults(
  telemetryContext: { _fastModeTuned?: boolean | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
): void
```

Apply fast-mode tuning to diversity sampling and novelty defaults.

Parameters:
- `telemetryContext` - - Context object storing fast-mode tuning flag.
- `telemetryOptions` - - Options with diversity and novelty settings.

### applyHypervolumeTelemetry

```ts
applyHypervolumeTelemetry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  hyperVolumeProxy: number,
  entry: TelemetryEntryRecord,
): void
```

Attach hypervolume scalar when requested.

Parameters:
- `telemetryOptions` - - Options controlling telemetry fields.
- `hyperVolumeProxy` - - Hypervolume proxy value.
- `entry` - - Telemetry entry to update.

### applyLineageStatsMonoObjective

```ts
applyLineageStatsMonoObjective(
  telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; },
  populationSnapshot: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Apply lineage stats for mono-objective mode using sampled ancestors.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyLineageStatsMultiObjective

```ts
applyLineageStatsMultiObjective(
  telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; },
  population: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Apply lineage stats for multi-objective mode using ancestor uniqueness.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyObjectiveAges

```ts
applyObjectiveAges(
  telemetryContext: { _objectiveAges?: Map<string, number> | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply objective age snapshots to the entry.

Parameters:
- `telemetryContext` - - Neat-like context with objective ages.
- `entry` - - Telemetry entry to update.

### applyObjectiveEvents

```ts
applyObjectiveEvents(
  telemetryContext: { _pendingObjectiveAdds?: string[] | undefined; _pendingObjectiveRemoves?: string[] | undefined; _objectiveEvents?: ObjectiveEvent[] | undefined; },
  entry: TelemetryEntryRecord,
  generation: number,
): void
```

Apply and flush objective lifecycle events.

Parameters:
- `telemetryContext` - - Neat-like context holding objective events.
- `entry` - - Telemetry entry to update.
- `generation` - - Generation index for event records.

### applyObjectiveImportance

```ts
applyObjectiveImportance(
  telemetryContext: { _lastObjImportance?: ObjImportance | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply the most recent objective importance snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with objective importance.
- `entry` - - Telemetry entry to update.

### applyObjectivesSnapshot

```ts
applyObjectivesSnapshot(
  telemetryContext: { _getObjectives?: (() => { key: string; }[]) | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply objectives list snapshot (keys only).

Parameters:
- `telemetryContext` - - Neat-like context with objective provider.
- `entry` - - Telemetry entry to update.

### applyPerformanceStats

```ts
applyPerformanceStats(
  telemetryContext: { _lastEvalDuration?: number | undefined; _lastEvolveDuration?: number | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void
```

Attach performance stats when configured.

Parameters:
- `telemetryContext` - - Neat-like context with performance data.
- `telemetryOptions` - - Options controlling performance telemetry.
- `entry` - - Telemetry entry to update.

### applyRngState

```ts
applyRngState(
  telemetryContext: { _rngState?: unknown; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void
```

Attach RNG state when configured.

Parameters:
- `telemetryContext` - - Neat-like context with RNG state.
- `telemetryOptions` - - Options controlling RNG telemetry.
- `entry` - - Telemetry entry to update.

### applySpeciesAllocation

```ts
applySpeciesAllocation(
  telemetryContext: { _lastOffspringAlloc?: SpeciesAlloc[] | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply per-species offspring allocation snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with allocation snapshot.
- `entry` - - Telemetry entry to update.

### buildComplexityEntry

```ts
buildComplexityEntry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  meanCounts: { meanNodes: number; meanConns: number; },
  maxCounts: { maxNodes: number; maxConns: number; },
  meanEnabledRatio: number,
  growthValues: { growthNodes: number; growthConns: number; },
): { meanNodes: number; meanConns: number; maxNodes: number; maxConns: number; meanEnabledRatio: number; growthNodes: number; growthConns: number; budgetMaxNodes: number; budgetMaxConns: number; }
```

Build the complexity entry payload for multi-objective mode.

Parameters:
- `telemetryOptions` - - Options controlling complexity telemetry.
- `meanCounts` - - Mean node/connection counts.
- `maxCounts` - - Max node/connection counts.
- `meanEnabledRatio` - - Mean enabled ratio.
- `growthValues` - - Growth deltas.

Returns: Complexity entry payload.

### buildDegreeHistogram

```ts
buildDegreeHistogram(
  counts: Record<number, number>,
): Record<number, number>
```

Build a histogram of degree frequencies from a degree-count table.

Parameters:
- `counts` - - Map geneId -> degree count.

Returns: Map degree -> number of nodes with that degree.

### buildLineageContext

```ts
buildLineageContext(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSnapshot: GenomeDetailed[],
): NeatLineageContext
```

Build a lineage helper context for ancestor operations.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Lineage helper context.

### buildLineageEntry

```ts
buildLineageEntry(
  context: { _prevInbreedingCount?: number | undefined; },
  bestGenomeSnapshot: GenomeDetailed,
  meanDepthValue: number,
  ancestorUniquenessScore: number,
): { parents: number[]; depthBest: number; meanDepth: number; inbreeding: number; ancestorUniq: number; }
```

Build the lineage entry payload.

Parameters:
- `context` - - Neat-like context with lineage info.
- `bestGenomeSnapshot` - - Best genome snapshot.
- `meanDepthValue` - - Mean lineage depth.
- `ancestorUniquenessScore` - - Ancestor uniqueness score.

Returns: Lineage entry payload.

### buildLineageSnapshot

```ts
buildLineageSnapshot(
  population: { _id?: number | undefined; _parents?: number[] | undefined; }[],
  limit: number,
): { id: number; parents: number[]; }[]
```

Snapshot lineage metadata for the first `limit` genomes.

### clearTelemetryBuffer

```ts
clearTelemetryBuffer(
  host: TelemetryAccessorHost,
): void
```

Clear the telemetry buffer in place.

### collectDepths

```ts
collectDepths(
  populationSnapshot: GenomeDetailed[],
): number[]
```

Collect depth values for the current population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of depth values (defaults to 0).

### collectPopulationCounts

```ts
collectPopulationCounts(
  populationSnapshot: GenomeDetailed[],
): { nodeCounts: number[]; connectionCounts: number[]; }
```

Collect node and connection counts for the population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Node and connection counts arrays.

### computeAncestorUniquenessSampled

```ts
computeAncestorUniquenessSampled(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSnapshot: GenomeDetailed[],
): number
```

Compute ancestor uniqueness using sampled Jaccard distance.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Rounded ancestor uniqueness score.

### computeAndStoreGrowthValues

```ts
computeAndStoreGrowthValues(
  context: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; },
  meanCounts: { meanNodes: number; meanConns: number; },
): { growthNodes: number; growthConns: number; }
```

Compute growth values and store the latest means on the context.

Parameters:
- `context` - - Neat-like context with previous mean values.
- `meanCounts` - - Current mean node/connection counts.

Returns: Growth values for nodes and connections.

### computeCompatibilityStats

```ts
computeCompatibilityStats(
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
  compatibilityDistance: ((a: TelemetryGenome, b: TelemetryGenome) => number) | undefined,
): { meanCompat: number; varCompat: number; }
```

Compute pairwise compatibility statistics via sampling.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.
- `compatibilityDistance` - - Optional compatibility distance function.

Returns: Mean and variance of sampled compatibilities.

### computeDegreeCounts

```ts
computeDegreeCounts(
  entropyGraph: { nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; },
): Record<number, number>
```

Compute per-node degree counts for enabled connections.

Parameters:
- `entropyGraph` - - Genome-like graph object.

Returns: Map geneId -> degree count.

### computeEnabledRatios

```ts
computeEnabledRatios(
  populationSnapshot: GenomeDetailed[],
): number[]
```

Compute enabled ratios per genome.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of enabled ratios.

### computeEntropyFromHistogram

```ts
computeEntropyFromHistogram(
  histogram: Record<number, number>,
  totalNodes: number,
): number
```

Compute entropy from a degree-frequency histogram.

Parameters:
- `histogram` - - Map degree -> number of nodes.
- `totalNodes` - - Total node count used to normalize into probabilities.

Returns: Entropy value (non-negative).

### computeEntropyStats

```ts
computeEntropyStats(
  genomes: TelemetryGenome[],
  structuralEntropyFn: (genome: TelemetryGenome) => number,
): { meanEntropy: number; varEntropy: number; }
```

Compute structural entropy mean and variance across the population.

Parameters:
- `genomes` - - Population snapshot.
- `structuralEntropyFn` - - Function to compute entropy for a genome.

Returns: Mean and variance of entropy values.

### computeGraphletEntropy

```ts
computeGraphletEntropy(
  genomes: TelemetryGenome[],
  size: number,
  graphletSampleCount: number,
  rngFactoryFn: () => () => number,
): number
```

Sample graphlet motifs and compute entropy over their edge counts.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `graphletSampleCount` - - Number of graphlets to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Graphlet entropy value.

### computeHyperVolumeProxy

```ts
computeHyperVolumeProxy(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
): number
```

Compute a hypervolume-like proxy for the Pareto front.

Parameters:
- `telemetryOptions` - - Options controlling complexity metric.
- `population` - - Population snapshot.

Returns: Hypervolume proxy value.

### computeLineageStats

```ts
computeLineageStats(
  lineageEnabled: boolean,
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
): { lineageMeanDepth: number; lineageMeanPairDist: number; }
```

Compute lineage depth and pairwise depth-distance statistics.

Parameters:
- `lineageEnabled` - - Whether lineage metrics are enabled.
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Lineage mean depth and pairwise distance.

### computeMaxCounts

```ts
computeMaxCounts(
  counts: { nodeCounts: number[]; connectionCounts: number[]; },
): { maxNodes: number; maxConns: number; }
```

Compute max node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Max node and connection counts.

### computeMeanCounts

```ts
computeMeanCounts(
  counts: { nodeCounts: number[]; connectionCounts: number[]; },
): { meanNodes: number; meanConns: number; }
```

Compute mean node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Mean node and connection counts.

### computeMeanDepth

```ts
computeMeanDepth(
  depthValues: number[],
): number
```

Compute the mean depth from a depth list.

Parameters:
- `depthValues` - - Depth values to average.

Returns: Mean depth value.

### computeMeanEnabledRatio

```ts
computeMeanEnabledRatio(
  enabledRatios: number[],
): number
```

Compute mean of enabled ratios.

Parameters:
- `enabledRatios` - - Enabled ratios per genome.

Returns: Mean enabled ratio.

### computeOperatorStatsSnapshot

```ts
computeOperatorStatsSnapshot(
  operatorStats: OperatorStatsMap | undefined,
): { op: string; succ: number; att: number; }[]
```

Snapshot operator statistics into a telemetry-friendly array.

Parameters:
- `operatorStats` - - Operator stats map (opName -> success/attempts).

Returns: Operator stats snapshot array.

### computePairJaccardDistance

```ts
computePairJaccardDistance(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSnapshot: GenomeDetailed[],
  firstIndex: number,
  secondIndex: number,
): number | undefined
```

Compute Jaccard distance between ancestor sets for a pair.

Parameters:
- `context` - - Neat-like context for lineage helpers.
- `populationSnapshot` - - Population snapshot.
- `firstIndex` - - First genome index.
- `secondIndex` - - Second genome index.

Returns: Jaccard distance or undefined when both sets are empty.

### computeParetoFrontSizes

```ts
computeParetoFrontSizes(
  population: GenomeDetailed[],
): number[]
```

Compute sizes of early Pareto fronts.

Parameters:
- `population` - - Population snapshot.

Returns: Array of front sizes (rank 0..4).

### countAncestorIntersection

```ts
countAncestorIntersection(
  ancestorsA: Set<number>,
  ancestorsB: Set<number>,
): number
```

Count the size of an ancestor intersection.

Parameters:
- `ancestorsA` - - First ancestor set.
- `ancestorsB` - - Second ancestor set.

Returns: Intersection count.

### countEnabledEdges

```ts
countEnabledEdges(
  genome: TelemetryGenome,
  selectedNodes: NodeLike[],
): number
```

Count enabled edges between the selected nodes in a genome.

Parameters:
- `genome` - - Genome with connections to inspect.
- `selectedNodes` - - Nodes forming the graphlet sample.

Returns: Edge count capped at 3.

### ensureTelemetryBuffer

```ts
ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[]
```

Ensure the telemetry buffer is initialized.

Parameters:
- `telemetryContext` - - Neat-like context holding telemetry buffer.

Returns: A mutable telemetry buffer.

### getCachedDiversityStats

```ts
getCachedDiversityStats(
  host: TelemetryAccessorHost,
): DiversityStats | undefined
```

Read cached diversity statistics.

### getCachedEntropy

```ts
getCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
): number | undefined
```

Read a cached entropy value if it exists and belongs to the current
generation.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.

Returns: Cached entropy number, or undefined when not available.

### getObjectiveEventsSnapshot

```ts
getObjectiveEventsSnapshot(
  host: TelemetryAccessorHost,
): { gen: number; type: "add" | "remove"; key: string; }[]
```

Return a shallow copy of recent objective events.

### getPerformanceStatsSnapshot

```ts
getPerformanceStatsSnapshot(
  host: TelemetryAccessorHost,
): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Snapshot performance timings for evaluation and evolution steps.

### getTelemetryBuffer

```ts
getTelemetryBuffer(
  host: TelemetryAccessorHost,
): TelemetryEntry[]
```

Return the telemetry buffer, defaulting to an empty array when missing.

### getTelemetryCoreSnapshot

```ts
getTelemetryCoreSnapshot(
  sourceEntry: Record<string, unknown>,
  fields: TelemetryCoreFields,
): Partial<Record<string, unknown>>
```

Build a snapshot of the core telemetry fields present on the entry; does
not mutate the source entry.

Parameters:
- `sourceEntry` - - Source telemetry object.
- `fields` - - Core telemetry field keys to preserve.

Returns: Shallow snapshot of core fields that exist on the entry.

### isLineageEligible

```ts
isLineageEligible(
  context: { _lineageEnabled?: boolean | undefined; },
  populationSnapshot: GenomeDetailed[],
): boolean
```

Check whether lineage metrics should be computed.

Parameters:
- `context` - - Neat-like context with lineage flag.
- `populationSnapshot` - - Population snapshot to validate.

Returns: True when lineage stats should be computed.

### LINEAGE_SNAPSHOT_DEFAULT_LIMIT

Default limit for lineage snapshots to avoid large payloads.

### mergeTelemetryCoreFields

```ts
mergeTelemetryCoreFields(
  sourceEntry: Record<string, unknown>,
  coreSnapshot: Partial<Record<string, unknown>>,
): Record<string, unknown>
```

Re-attach core fields to the filtered entry.
Mutates the entry so the caller keeps the original reference.

Parameters:
- `sourceEntry` - - Filtered telemetry entry to update.
- `coreSnapshot` - - Snapshot of core fields to ensure presence.

Returns: The same entry reference with core fields restored.

### OperatorStatsMap

Operator stats map shape for telemetry extraction.

### pickDistinctIndices

```ts
pickDistinctIndices(
  upperBound: number,
  count: number,
  rng: () => number,
): number[]
```

Pick a fixed number of distinct random indices.

Parameters:
- `upperBound` - - Exclusive upper bound for random indices.
- `count` - - Number of distinct indices to pick.
- `rng` - - RNG function returning values in [0,1).

Returns: Array of distinct indices.

### pickDistinctPairIndices

```ts
pickDistinctPairIndices(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSize: number,
): { firstIndex: number; secondIndex: number; }
```

Pick two distinct indices using the context RNG.

Parameters:
- `context` - - Neat-like context with RNG factory.
- `populationSize` - - Population size for index bounds.

Returns: Pair of distinct indices.

### readOperatorStats

```ts
readOperatorStats(
  operatorStats: OperatorStatsMap | undefined,
): { name: string; success: number; attempts: number; }[]
```

Convert operator stats map into the public accessor shape.

### safelyApplyTelemetrySelect

```ts
safelyApplyTelemetrySelect(
  telemetryContext: TContext,
  telemetryEntry: TelemetryEntry,
  applyTelemetrySelectFn: (this: TContext, entry: Record<string, unknown>) => Record<string, unknown>,
): void
```

Apply telemetry selection while swallowing any selection errors.

Parameters:
- `telemetryContext` - - Neat-like context with telemetry selection.
- `telemetryEntry` - - Entry to filter in place.
- `applyTelemetrySelectFn` - - Selection helper to invoke.

### safelyStreamTelemetryEntry

```ts
safelyStreamTelemetryEntry(
  telemetryContext: { options?: TelemetryStreamOptions | undefined; },
  telemetryEntry: TelemetryEntry,
): void
```

Stream telemetry entry when a stream callback is configured.

Parameters:
- `telemetryContext` - - Neat-like context with stream settings.
- `telemetryEntry` - - Entry to stream.

### setCachedEntropy

```ts
setCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
  entropyValue: number,
): void
```

Cache an entropy value for the current generation on the graph object.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.
- `entropyValue` - - Entropy value to cache.

### stripUnselectedTelemetryKeys

```ts
stripUnselectedTelemetryKeys(
  sourceEntry: Record<string, unknown>,
  selection: Set<string>,
  fields: TelemetryCoreFields,
): Record<string, unknown>
```

Remove non-core keys that are not whitelisted by the selection set.
Mutates the provided entry in-place for efficiency.

Parameters:
- `sourceEntry` - - Telemetry entry being filtered.
- `selection` - - Whitelist of additional telemetry keys.
- `fields` - - Core telemetry field keys that must be preserved.

Returns: The same entry reference after filtering.

### TelemetryAccessorHost

Minimal host surface needed by telemetry accessors.

### TelemetryBufferContext

Minimal telemetry buffer context shape.

### TelemetryCoreFields

Core telemetry field keys used by selection helpers.

### TelemetryDiversityOptions

Diversity telemetry options for sampling and novelty defaults.

### TelemetryEntryRecord

Telemetry entry shape used for constructing snapshots.

### TelemetryGenome

Minimal genome shape used by telemetry helpers.

### TelemetrySelectContext

Minimal telemetry selection context shape.

### TelemetryStreamOptions

Minimal telemetry stream options for streaming helpers.

### trimTelemetryBuffer

```ts
trimTelemetryBuffer(
  telemetryBufferRef: TelemetryEntry[],
  maxEntries: number,
): void
```

Trim the telemetry buffer to a maximum size.

Parameters:
- `telemetryBufferRef` - - Buffer to trim in-place.
- `maxEntries` - - Maximum entries to keep.

## neat/neat.objectives.utils.ts

### buildDefaultFitnessObjective

```ts
buildDefaultFitnessObjective(): ObjectiveDescriptor
```

Returns: Default fitness objective descriptor.

### collectDefaultObjectives

```ts
collectDefaultObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[]
```

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Default objectives when fitness is not suppressed.

### collectUserObjectives

```ts
collectUserObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[]
```

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Valid user-registered objectives when multi-objective is enabled.

### ensureMultiObjectiveOptions

```ts
ensureMultiObjectiveOptions(
  neatInstance: NeatLikeWithObjectives,
): { enabled?: boolean | undefined; objectives?: ObjectiveDescriptor[] | undefined; }
```

Parameters:
- `neatInstance` - - Instance receiving the multi-objective container.

Returns: Initialized multi-objective options.

### ensureObjectivesList

```ts
ensureObjectivesList(
  multiObjectiveOptions: { enabled?: boolean | undefined; objectives?: ObjectiveDescriptor[] | undefined; },
): ObjectiveDescriptor[]
```

Parameters:
- `multiObjectiveOptions` - - Multi-objective container to hydrate.

Returns: Objectives list for mutation-free operations.

### getObjectiveCandidates

```ts
getObjectiveCandidates(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[]
```

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Candidate objectives from configuration.

### isMultiObjectiveEnabled

```ts
isMultiObjectiveEnabled(
  neatInstance: NeatLikeWithObjectives,
): boolean
```

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Whether multi-objective mode is enabled with a candidate list.

### isValidObjective

```ts
isValidObjective(
  candidateObjective: ObjectiveDescriptor | undefined,
): boolean
```

Parameters:
- `candidateObjective` - - Candidate descriptor to validate.

Returns: True when the descriptor has the required shape.

### NeatLikeWithObjectives

Minimal interface for NEAT instances using objective management.

This shape is intentionally small and only includes the pieces needed by
`_getObjectives`, `registerObjective`, and `clearObjectives`.

Example:

```ts
const neatLike: NeatLikeWithObjectives = {
  options: { multiObjective: { enabled: true, objectives: [] } },
};
```

### replaceObjectiveByKey

```ts
replaceObjectiveByKey(
  objectivesList: ObjectiveDescriptor[],
  objectiveKey: string,
  objectiveDirection: "max" | "min",
  objectiveAccessor: (genome: GenomeLike) => number,
): ObjectiveDescriptor[]
```

Parameters:
- `objectivesList` - - Existing objectives to update.
- `objectiveKey` - - Key to replace.
- `objectiveDirection` - - Direction for the new objective.
- `objectiveAccessor` - - Accessor for the new objective.

Returns: Updated objectives list with the new descriptor appended.

## neat/neat.speciation.utils.ts

Utility helpers for NEAT speciation orchestration.

### adjustCompatibilityThreshold

```ts
adjustCompatibilityThreshold(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: { smoothingWindow?: number | undefined; decay?: number | undefined; kp?: number | undefined; ki?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; },
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void
```

Update the adaptive compatibility threshold and clamp to bounds.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `compatAdjust` - - Compatibility adjustment settings.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Nothing.

### applyAgeProtection

```ts
applyAgeProtection(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void
```

Apply age protection penalties to old species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### applyFitnessSharing

```ts
applyFitnessSharing(
  speciationContext: FitnessSharingContext,
  sharingSigma: number,
): void
```

Apply fitness sharing to penalize similarity within species.

Parameters:
- `speciationContext` - - Neat instance context with species and distance function.
- `sharingSigma` - - Sharing radius used for distance weighting.

Returns: Nothing.

### assignPopulationToSpecies

```ts
assignPopulationToSpecies(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void
```

Assign each genome in the population to a compatible species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### averageNumbers

```ts
averageNumbers(
  values: number[],
): number
```

Average a list of numbers, returning zero when empty.

Parameters:
- `values` - - Numeric values to average.

Returns: Mean of the values or zero.

### buildExtendedHistoryStats

```ts
buildExtendedHistoryStats(
  speciationContext: SpeciationHarnessContext<TOptions>,
  species: SpeciesLike,
): Record<string, unknown>
```

Build extended history stats for a species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `species` - - Species to snapshot.

Returns: Extended history entry.

### clampCompatibilityThreshold

```ts
clampCompatibilityThreshold(
  options: SpeciationOptions,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void
```

Clamp the compatibility threshold to configured bounds.

Parameters:
- `options` - - Speciation options.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Nothing.

### CompatAdjust

Resolved compatibility-threshold adjustment settings.

This is the non-nullable form of {@link SpeciationOptions.compatAdjust} used
by the speciation PID controller.

### computePidThreshold

```ts
computePidThreshold(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: { smoothingWindow?: number | undefined; decay?: number | undefined; kp?: number | undefined; ki?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; },
  currentThreshold: number,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): number
```

Compute a PID-based threshold update and clamp when needed.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `compatAdjust` - - Compatibility adjustment settings.
- `currentThreshold` - - Current compatibility threshold.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Updated threshold.

### createSpeciesForGenome

```ts
createSpeciesForGenome(
  speciationContext: SpeciationHarnessContext<TOptions>,
  genome: GenomeDetailed,
): void
```

Create a new species for the provided genome.

Parameters:
- `speciationContext` - - Speciation harness context.
- `genome` - - Genome that starts a new species.

Returns: Nothing.

### DEFAULT_COMPAT_INTEGRAL

Default integral accumulator value.

### DEFAULT_COMPATIBILITY_INTEGRAL_GAIN

Default integral gain for compatibility PID.

### DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN

Default proportional gain for compatibility PID.

### DEFAULT_COMPATIBILITY_THRESHOLD

Default compatibility threshold when unspecified.

### DEFAULT_LAST_IMPROVED_GENERATION

Default last improved generation when missing.

### DEFAULT_MAX_COMPATIBILITY_THRESHOLD

Default maximum compatibility threshold.

### DEFAULT_MEMBER_COUNT_FALLBACK

Fallback divisor when member count is zero.

### DEFAULT_MIN_COMPATIBILITY_THRESHOLD

Default minimum compatibility threshold.

### DEFAULT_SCORE_FALLBACK

Fallback numeric score when missing.

### DEFAULT_SHARING_SIGMA

Default sigma for fitness sharing.

### DEFAULT_SPECIES_AGE_GRACE

Default grace period for young species.

### DEFAULT_SPECIES_OLD_PENALTY

Default penalty applied to old species.

### DEFAULT_STAGNATION_WINDOW

Default stagnation window in generations.

### DEFAULT_TARGET_SPECIES

Default target number of species for PID controller.

### findCompatibleSpecies

```ts
findCompatibleSpecies(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  genome: GenomeDetailed,
): SpeciesLike | undefined
```

Find a compatible species representative for the given genome.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `genome` - - Genome to match.

Returns: Matching species or undefined.

### FitnessSharingContext

Minimal context required to apply fitness sharing.

Fitness sharing normalizes per-genome fitness within each species to reduce
selection pressure toward dense clusters of very similar genomes.

### HISTORY_BUFFER_MAX_ENTRIES

Max number of history entries to keep.

### InnovationAccumulator

Accumulator for innovation-id statistics across a set of connections.

Used for extended history telemetry (mean innovation, innovation range, and
enabled/disabled ratios).

### NEGATIVE_INFINITY

Shared negative infinity constant for score initialization.

### PENALTY_NO_EFFECT_THRESHOLD

Penalty cutoff where no reduction should occur.

### recordHistory

```ts
recordHistory(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
): void
```

Record the current species history snapshot.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### refreshSpeciesRepresentatives

```ts
refreshSpeciesRepresentatives(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Refresh representatives and remove empty species.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### resetSpeciesMembers

```ts
resetSpeciesMembers(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Clear member lists for all species.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### SHARING_MAX_CONTRIBUTION

Maximum sharing contribution per peer.

### SHARING_SELF_DISTANCE

Distance used when comparing a member with itself.

### SHARING_SUM_FLOOR

Fallback divisor when sharing sum is zero.

### snapshotPreviousMembers

```ts
snapshotPreviousMembers(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Snapshot current species memberships for telemetry.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### SPECIES_AGE_GRACE_MULTIPLIER

Multiplier used to convert grace generations to age threshold.

### StagnationContext

Minimal context required to update species stagnation.

Stagnation pruning removes species that have not improved their best score
within a configured number of generations.

### summarizeInnovations

```ts
summarizeInnovations(
  speciationContext: SpeciationHarnessContext<TOptions>,
  members: GenomeDetailed[],
): { meanInnovation: number; innovationRange: number; enabledRatio: number; }
```

Summarize innovation statistics for a set of members.

Parameters:
- `speciationContext` - - Speciation harness context.
- `members` - - Members to summarize.

Returns: Innovation summary statistics.

### trimHistory

```ts
trimHistory(
  speciationContext: SpeciationHarnessContext<TOptions>,
): void
```

Trim species history to the maximum buffer size.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### updateSpeciesStagnation

```ts
updateSpeciesStagnation(
  speciationContext: StagnationContext,
  stagnationWindow: number,
  sortSpeciesMembers: (species: SpeciesLike) => void,
): void
```

Update stagnation counters and prune stagnant species.

Parameters:
- `speciationContext` - - Neat instance context with species array and generation counter.
- `stagnationWindow` - - Allowed stagnation window.
- `sortSpeciesMembers` - - Sort function for species members.

Returns: Nothing.

## neat/neat.mutation.flow.utils.ts

### applyAddConnMutation

```ts
applyAddConnMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply an ADD_CONN mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyAddNodeMutation

```ts
applyAddNodeMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply an ADD_NODE mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyMutationOperator

```ts
applyMutationOperator(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply a mutation operator to a genome and invalidate caches as needed.

Parameters:
- `genome` - - genome to mutate
- `mutationMethod` - - mutation operator to apply
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### captureStructuralSizes

```ts
captureStructuralSizes(
  genome: GenomeWithMetadata,
): { beforeNodes: number; beforeConns: number; }
```

Capture structural sizes used to evaluate operator success.

Parameters:
- `genome` - - genome to inspect

Returns: structural size snapshot

### initializeAdaptiveMutation

```ts
initializeAdaptiveMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Initialize per-genome adaptive mutation parameters if configured.

Parameters:
- `genome` - - genome to initialize
- `internal` - - neat controller context

Returns: void

### maybeAddExtraConnection

```ts
maybeAddExtraConnection(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Optionally add an extra connection to increase exploration.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context

Returns: void

### mutateGenome

```ts
mutateGenome(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): Promise<void>
```

Mutate a single genome based on configured mutation policies.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: Promise resolving after mutation attempts complete

### resolveEffectiveAmount

```ts
resolveEffectiveAmount(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number
```

Resolve the effective mutation amount for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation amount

### resolveEffectiveRate

```ts
resolveEffectiveRate(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number
```

Resolve the effective mutation rate for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation rate

### selectConcreteMutationMethod

```ts
selectConcreteMutationMethod(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): Promise<MutationMethod | null>
```

Select a concrete mutation method, resolving any legacy arrays.

Parameters:
- `genome` - - genome to select for
- `internal` - - neat controller context

Returns: resolved mutation method or null

### shouldInvalidateCaches

```ts
shouldInvalidateCaches(
  mutationMethod: MutationMethod,
  methods: { mutation: unknown; },
): boolean
```

Determine whether a mutation method invalidates cached structures.

Parameters:
- `mutationMethod` - - mutation operator to inspect
- `methods` - - mutation methods module

Returns: true when caches should be invalidated

### shouldMutateGenome

```ts
shouldMutateGenome(
  effectiveRate: number,
  internal: NeatControllerForMutation,
): boolean
```

Decide whether a genome should be mutated based on probability.

Parameters:
- `effectiveRate` - - effective mutation probability
- `internal` - - neat controller context

Returns: true when the genome should be mutated

### updateOperatorStatsIfNeeded

```ts
updateOperatorStatsIfNeeded(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  beforeSizes: { beforeNodes: number; beforeConns: number; },
  internal: NeatControllerForMutation,
): void
```

Update operator statistics when adaptation is enabled.

Parameters:
- `genome` - - genome used to compute after-sizes
- `mutationMethod` - - operator being recorded
- `beforeSizes` - - structural sizes captured before mutation
- `internal` - - neat controller context

Returns: void

## neat/neat.telemetry.rng.utils.ts

### applyRngState

```ts
applyRngState(
  telemetryContext: { _rngState?: unknown; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void
```

Attach RNG state when configured.

Parameters:
- `telemetryContext` - - Neat-like context with RNG state.
- `telemetryOptions` - - Options controlling RNG telemetry.
- `entry` - - Telemetry entry to update.

## neat/neat.evolve.runtime.utils.ts

### buildFittestSnapshot

```ts
buildFittestSnapshot(
  internal: NeatControllerForEvolution,
): default
```

Build a cloned Network from the current best genome.

Parameters:
- `internal` - - NEAT controller instance.

Returns: best network snapshot.

### clearPopulationScores

```ts
clearPopulationScores(
  internal: NeatControllerForEvolution,
): void
```

Clear genome scores to force re-evaluation.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeElapsedTime

```ts
computeElapsedTime(
  startTimestamp: number,
): number
```

Compute elapsed time since the start of evolve().

Parameters:
- `startTimestamp` - - Start time resolved earlier.

Returns: elapsed time.

### ensurePopulationEvaluated

```ts
ensurePopulationEvaluated(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Ensure the population is evaluated before evolution operations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### resolveStartTime

```ts
resolveStartTime(): number
```

Resolve the start time for an evolution step.

Returns: timestamp in milliseconds or high-resolution units.

### trackGlobalImprovement

```ts
trackGlobalImprovement(
  internal: NeatControllerForEvolution,
  snapshot: default,
): void
```

Track global best improvement for stagnation logic.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot.

Returns: void.

### updateGlobalBestTracking

```ts
updateGlobalBestTracking(
  internal: NeatControllerForEvolution,
): void
```

Update generation-level best score tracking.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

## neat/neat.multiobjective.utils.ts

Barrel exports for multi-objective utilities.

### accumulateCrowdingForObjective

```ts
accumulateCrowdingForObjective(
  sortedFront: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): void
```

Accumulates crowding distance contributions for a single objective.

Pre-conditions / expectations:
- `sortedFront` must be sorted ascending by the selected objective.
- {@link initializeCrowding} has already set `_moCrowd = 0` for the front.
- {@link markBoundaryCrowding} is typically called before this to set the
  boundary genomes to `Infinity`.

Edge cases:
- If the front has fewer than 2 genomes, this is a no-op.
- If the objective range is `0`, a range of `1` is used (see
  {@link resolveObjectiveRange}).

Parameters:
- `sortedFront` - - Front sorted by objective.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

### archiveParetoFrontsIfEnabled

```ts
archiveParetoFrontsIfEnabled(
  neatInstance: NeatLikeWithMultiObjective,
  fronts: default[][],
): void
```

Archives a compact snapshot of the current Pareto fronts when
multi-objective mode is enabled.

This is intended for visualization/debugging:
- Stores only genome `_id` values (not full genomes).
- Keeps only the top {@link MAX_PARETO_ARCHIVE_FRONTS} fronts.
- Maintains a ring-buffer-like cap of {@link MAX_PARETO_ARCHIVE_LENGTH}
  snapshots by shifting the oldest entry.

Behavior note:
- This currently gates only on `neatInstance.options.multiObjective?.enabled`.
  If you want a separate archive toggle, ensure the caller configures
  `enabled` accordingly.

Parameters:
- `neatInstance` - - Neat instance.
- `fronts` - - Pareto fronts to archive.

### assignCrowdingDistances

```ts
assignCrowdingDistances(
  fronts: default[][],
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  population: default[],
): void
```

Assigns crowding-distance annotations for each Pareto front.

This implements the crowding distance component of NSGA-II selection. Each
genome in each front receives a `_moCrowd` value representing how isolated
it is in objective space within its front.

Notes:
- This function sorts each front by each objective (ascending raw values).
  Objective direction (min vs max) does not affect the computed spacing
  magnitude; extrema are treated as boundaries either way.
- Empty fronts are skipped.

Side effects:
- Writes `_moCrowd` on each genome in each front.

Parameters:
- `fronts` - - Pareto fronts.
- `valuesMatrixInput` - - Values matrix.
- `descriptors` - - Objective descriptors (provides objective count).
- `population` - - Population to resolve indices.

### buildDominanceState

```ts
buildDominanceState(
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
): DominanceState
```

Builds dominance bookkeeping structures used by fast non-dominated sorting.

This computes (pairwise):
- `dominationCounts[i]`: how many genomes dominate genome `i`.
- `dominatedIndicesByIndex[i]`: which genomes are dominated by genome `i`.
- `firstFrontIndices`: genomes with `dominationCounts[i] === 0`.

Complexity:
- Time: $O(n^2 \cdot m)$ where $n$ is population size and $m$ is objective
  count.
- Space: $O(n^2)$ in the worst case for the dominated adjacency lists.

Assumptions:
- Each row in `valuesMatrixInput` is a vector aligned with `descriptors`.
- Genome ordering in later steps is expected to match the matrix ordering.

Parameters:
- `valuesMatrixInput` - - Matrix of objective values (row = genome).
- `descriptors` - - Objective descriptors (direction semantics).

Returns: Dominance bookkeeping structures for ranking.

### buildGenomeIndexByReference

```ts
buildGenomeIndexByReference(
  population: default[],
): Map<default, number>
```

Builds a stable mapping from genome object references to their population
index.

This relies on object identity (reference equality), not structural
equality. It is used to resolve objective values from a values matrix when
working with reordered views (e.g., sorted fronts).

Parameters:
- `population` - - Genomes in population order.

Returns: Map from genome references to their index.

### buildGenomeValues

```ts
buildGenomeValues(
  genomeItem: default,
  descriptors: ObjectiveDescriptor[],
): number[]
```

Builds an objective vector for a single genome.

The resulting array order matches the `descriptors` order exactly.
Each component is read via {@link readObjectiveValue} so individual
objective accessors are fault-tolerant.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptors` - - Objective descriptors (vector schema).

Returns: Objective value vector (length equals `descriptors.length`).

### buildMultiObjectiveMetrics

```ts
buildMultiObjectiveMetrics(
  population: default[],
): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Build lightweight multi-objective metrics for each genome in the population.

### buildParetoFronts

```ts
buildParetoFronts(
  population: default[],
  dominanceState: DominanceState,
  maxFrontRankGuard: number,
): default[][]
```

Builds Pareto fronts from a precomputed dominance state.

This performs the “peeling” phase of fast non-dominated sorting:
- Start with the first front (all non-dominated genomes).
- For each front, reduce domination counts of the genomes it dominates.
- Any genome whose domination count becomes zero moves to the next front.

Side effects:
- Annotates each genome in `population` with `_moRank` (0 = best front).

Guard:
- Stops when `currentFrontRank > maxFrontRankGuard` to avoid pathological
  infinite/degenerate runs. If the guard triggers, the returned fronts may
  be incomplete.

Parameters:
- `population` - - Genome population (same ordering used by dominance
bookkeeping).
- `dominanceState` - - Dominance bookkeeping.
- `maxFrontRankGuard` - - Safety guard for ranking iterations.

Returns: Ordered Pareto fronts (rank order).

### buildValuesMatrix

```ts
buildValuesMatrix(
  population: default[],
  descriptors: ObjectiveDescriptor[],
): number[][]
```

Builds a population-wide objective value matrix.

The resulting matrix is indexed as `[genomeIndex][objectiveIndex]` where
`genomeIndex` matches the input `population` order.

Parameters:
- `population` - - Genomes to evaluate (population order is preserved).
- `descriptors` - - Objective descriptors (column schema).

Returns: Objective values matrix.

### DEFAULT_MAX_PARETO_FRONTS

Default number of Pareto fronts returned by accessors.

### DEFAULT_PARETO_ARCHIVE_JSONL_MAX

Default slice size when exporting Pareto archive as JSONL.

### DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES

Default slice size when reading Pareto archive entries.

### DominanceState

Dominance bookkeeping structures for fast non-dominated sorting.

These structures are typically produced once per generation (from the values
matrix) and then consumed to build Pareto fronts.

### exportParetoArchiveJsonl

```ts
exportParetoArchiveJsonl(
  archive: unknown[],
  maxEntries: number,
): string
```

Export a Pareto archive slice as JSON Lines.

### initializeCrowding

```ts
initializeCrowding(
  front: default[],
): void
```

Initializes crowding-distance annotations for a front.

This sets each genome’s `_moCrowd` to `0`. Later steps accumulate per-
objective spacing deltas.

Parameters:
- `front` - - Pareto front.

### markBoundaryCrowding

```ts
markBoundaryCrowding(
  sortedFront: default[],
): void
```

Marks the boundary genomes of a sorted front as infinitely crowded.

In NSGA-II style crowding distance, boundary solutions (extremes for the
objective) are assigned an infinite crowding distance to ensure they are
always preferred when ranks tie.

Parameters:
- `sortedFront` - - Front sorted by the current objective.

### MAX_PARETO_ARCHIVE_FRONTS

Maximum number of top Pareto fronts to retain per archive snapshot.

Archival stores a compact representation (IDs only) for visualization or
debugging.

### MAX_PARETO_ARCHIVE_LENGTH

Maximum number of archive snapshots to retain.

When the archive exceeds this length, the oldest snapshot is dropped.

### MAX_PARETO_FRONT_RANK_GUARD

Maximum number of Pareto fronts to allow during ranking before aborting.

This is a defensive guard against pathological conditions (e.g., corrupted
dominance bookkeeping) that could otherwise cause long/infinite loops.

### NeatLikeWithMultiObjective

Minimal Neat-like interface required by the multi-objective helpers.

This intentionally models only the fields used for archiving Pareto fronts
and retrieving objective descriptors. It allows these helpers to be used
without depending on the full Neat class type.

### NetworkWithMOAnnotations

Extends a genome/network with multi-objective annotations.

These properties are used as transient metadata during selection.

- `_moRank`: Pareto front rank (0 = best front)
- `_moCrowd`: crowding distance within the front (higher = more isolated;
  boundary genomes are typically `Infinity`)
- `_id`: optional stable identifier used for compact archiving

### ObjectiveDescriptor

Describes how to evaluate a single objective for a genome.

The order of objective descriptors defines the order of each genome’s
objective vector and therefore the columns of the values matrix.

Notes:
- `accessor` should be deterministic for a given genome state.
- `direction` controls Pareto dominance comparisons:
  - `'max'`: higher is better
  - `'min'`: lower is better
- If `direction` is omitted, it defaults to `'max'`.

Example:

```ts
const objectives: ObjectiveDescriptor[] = [
  { accessor: (g) => g.score ?? 0, direction: 'max' },
  { accessor: (g) => g.cost ?? 0, direction: 'min' },
];
```

### readObjectiveValue

```ts
readObjectiveValue(
  genomeItem: default,
  descriptor: ObjectiveDescriptor,
): number
```

Safely reads a single objective value for a given genome.

This wraps the descriptor `accessor` in a `try/catch` so that a buggy
objective function cannot crash multi-objective ranking.

Notes:
- If the accessor throws, this returns `0` (a neutral-ish fallback).
- Callers should prefer to surface accessor errors during development;
  this helper is intentionally defensive for long-running training loops.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptor` - - Objective descriptor providing an accessor.

Returns: Numeric objective value; `0` if the accessor throws.

Example:

```ts
const score = readObjectiveValue(genome, { accessor: (g) => g.score ?? 0 });
```

### reconstructParetoFronts

```ts
reconstructParetoFronts(
  population: default[],
  maxFronts: number,
  isMultiObjectiveEnabled: boolean,
): default[][]
```

Reconstruct Pareto fronts from stored rank annotations.

### resolveGenomeIndex

```ts
resolveGenomeIndex(
  genomeIndexByReference: Map<default, number>,
  genomeItem: default,
): number
```

Resolves a genome’s index using a reference-based map.

Parameters:
- `genomeIndexByReference` - - Lookup map created by
 *  {@link buildGenomeIndexByReference} .
- `genomeItem` - - Genome to resolve.

Returns: The population index of the genome.

### resolveObjectiveRange

```ts
resolveObjectiveRange(
  minValue: number,
  maxValue: number,
): number
```

Resolves a non-zero objective range used to normalize crowding deltas.

If all genomes have the same objective value, the raw range is `0`. This
returns `1` in that case to avoid division by zero while still producing a
well-defined crowding delta of `0`.

Parameters:
- `minValue` - - Minimum objective value.
- `maxValue` - - Maximum objective value.

Returns: Normalized range with a non-zero floor.

### resolveObjectiveValue

```ts
resolveObjectiveValue(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  genomeItem: default,
  objectiveIndex: number,
): number
```

Resolves an objective value for a genome from a values matrix.

This is a convenience helper for working with sorted/reordered views of the
population while keeping objective values in a dense matrix.

Parameters:
- `valuesMatrixInput` - - Values matrix indexed by population index.
- `genomeIndexByReference` - - Lookup map from genome reference to index.
- `genomeItem` - - Genome to resolve.
- `objectiveIndex` - - Objective column index.

Returns: The objective value for the genome.

### sliceParetoArchive

```ts
sliceParetoArchive(
  archive: T[],
  maxEntries: number,
): T[]
```

Return the most recent Pareto archive entries up to the provided limit.

### vectorDominates

```ts
vectorDominates(
  valuesA: number[],
  valuesB: number[],
  descriptors: ObjectiveDescriptor[],
): boolean
```

Determines whether vector A Pareto-dominates vector B.

A dominates B iff:
- A is **no worse** than B in every objective (respecting each objective’s
  direction: maximize/minimize), and
- A is **strictly better** in at least one objective.

Assumptions:
- `valuesA` and `valuesB` are aligned and have the same length.
- `descriptors` provides a descriptor for each objective index.
- If a descriptor has no `direction`, it defaults to `'max'`.

Parameters:
- `valuesA` - - Objective values for candidate A.
- `valuesB` - - Objective values for candidate B.
- `descriptors` - - Objective descriptors defining direction semantics.

Returns: `true` if A dominates B; otherwise `false`.

Example:

```ts
// Maximize accuracy, minimize latency:
vectorDominates([0.9, 120], [0.9, 150], [
  { accessor: () => 0, direction: 'max' },
  { accessor: () => 0, direction: 'min' },
]);
// => true (equal accuracy, lower latency)
```

## neat/neat.adaptive.phases.utils.ts

### initializePhaseState

```ts
initializePhaseState(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Ensure phase state is initialized.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### resolveNextPhase

```ts
resolveNextPhase(
  currentPhase: string,
): string
```

Resolve next phase name.

Parameters:
- `currentPhase` - - Current phase label.

Returns: Next phase label.

### togglePhaseIfNeeded

```ts
togglePhaseIfNeeded(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Toggle phase if the current phase has exceeded its length.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

## neat/neat.evolve.adaptive.utils.ts

### adaptReenableProbability

```ts
adaptReenableProbability(
  internal: NeatControllerForEvolution,
  config: { minSamples: number; target: number; min: number; max: number; deltaScale: number; },
): void
```

Adapt the re-enable probability based on recent success ratios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAdaptiveComplexityControllers

```ts
applyAdaptiveComplexityControllers(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply adaptive complexity controllers if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAncestorUniqAdaptiveSafe

```ts
applyAncestorUniqAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply ancestor uniqueness adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAutoCompatibilityTuning

```ts
applyAutoCompatibilityTuning(
  internal: NeatControllerForEvolution,
  config: { targetMin: number; adjustRate: number; minCoeff: number; maxCoeff: number; randomScale: number; },
): void
```

Apply auto-compatibility tuning if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Tuning constants.

Returns: void.

### applyMinimalCriterionAdaptiveSafe

```ts
applyMinimalCriterionAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply minimal criterion adaptive controller if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyOperatorAdaptationSafe

```ts
applyOperatorAdaptationSafe(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply operator adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyPruningAndMutation

```ts
applyPruningAndMutation(
  internal: NeatControllerForEvolution,
): Promise<void>
```

Apply pruning and mutation phases.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### invalidateCompatibilityCaches

```ts
invalidateCompatibilityCaches(
  internal: NeatControllerForEvolution,
): void
```

Invalidate compatibility caches after mutations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

## neat/neat.evolve.warnings.utils.ts

Warning emitted when evolution finishes without a best genome.

### EVOLVE_NO_BEST_GENOME_WARNING

Warning emitted when evolution finishes without a best genome.

### warnIfNoBestGenome

```ts
warnIfNoBestGenome(): void
```

Emit the standard warning for runs that end without a valid best genome.

## neat/neat.mutation.select.utils.ts

### applyOperatorAdaptationForSelect

```ts
applyOperatorAdaptationForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[]
```

Apply operator adaptation weighting to the pool when enabled.

Parameters:
- `pool` - - base pool
- `internal` - - neat controller context

Returns: augmented pool

### applyOperatorBanditForSelect

```ts
applyOperatorBanditForSelect(
  pool: MutationMethod[],
  fallbackMethod: MutationMethod,
  internal: NeatControllerForMutation,
): MutationMethod
```

Apply operator bandit selection if enabled.

Parameters:
- `pool` - - operator pool
- `fallbackMethod` - - method used when bandit is disabled
- `internal` - - neat controller context

Returns: selected method

### applyPhasedComplexityForSelect

```ts
applyPhasedComplexityForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[]
```

Apply phased complexity adjustments to the pool when enabled.

Parameters:
- `pool` - - base operator pool
- `internal` - - neat controller context

Returns: pool with phased complexity adjustments

### isBlockedByRecurrentPolicyForSelect

```ts
isBlockedByRecurrentPolicyForSelect(
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): boolean
```

Check whether a mutation is blocked by recurrent connection policy.

Parameters:
- `mutationMethod` - - mutation operator to check
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isBlockedByStructuralLimitsForSelect

```ts
isBlockedByStructuralLimitsForSelect(
  mutationMethod: MutationMethod,
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): boolean
```

Check whether a mutation is blocked by structural limits.

Parameters:
- `mutationMethod` - - mutation operator to check
- `genome` - - genome to inspect
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isLegacyFFWPoolForSelect

```ts
isLegacyFFWPoolForSelect(
  configuredPool: MutationMethod[],
  methods: { mutation: unknown; },
): boolean
```

Check whether a pool matches the legacy FFW operator ordering.

Parameters:
- `configuredPool` - - configured operator pool
- `methods` - - methods module

Returns: true when the pool matches FFW

### isOperatorNamePrefixedForSelect

```ts
isOperatorNamePrefixedForSelect(
  method: MutationMethod,
  prefix: string,
): boolean
```

Check whether an operator name uses a specific prefix.

Parameters:
- `method` - - mutation operator
- `prefix` - - name prefix to match

Returns: true when the operator name matches the prefix

### normalizeMutationPoolForSelect

```ts
normalizeMutationPoolForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
  rawReturnForTest: boolean,
): MutationMethod[]
```

Normalize the configured mutation pool to a flat operator list.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW for tests

Returns: normalized mutation pool

### resolveFFWPolicyForSelect

```ts
resolveFFWPolicyForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
  rawReturnForTest: boolean,
): MutationMethod | MutationMethod[] | null
```

Resolve legacy FFW policy behavior, including test-specific returns.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW array for tests

Returns: mutation method or null when not handled

### sampleFromPoolForSelect

```ts
sampleFromPoolForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod | null
```

Sample a random method from the pool.

Parameters:
- `pool` - - operator pool
- `internal` - - neat controller context

Returns: sampled method or null

## neat/neat.species.history.utils.ts

Default slice size when exporting species history as JSONL.

### exportSpeciesHistoryJsonl

```ts
exportSpeciesHistoryJsonl(
  speciesHistory: unknown[],
  maxEntries: number,
): string
```

Export species history records as JSON Lines.

### SPECIES_HISTORY_JSONL_MAX_DEFAULT

Default slice size when exporting species history as JSONL.

## neat/neat.topology-intent.utils.ts

### isGenomeEligibleForFeedForwardIntentPromotion

```ts
isGenomeEligibleForFeedForwardIntentPromotion(
  genome: TopologyIntentGenome,
): boolean
```

Check whether a genome can safely adopt feed-forward topology intent.

Eligibility is intentionally conservative: the graph must already be free of
gates/self-connections and all normal connections must follow the current
node ordering. This avoids reinterpreting arbitrary legacy seeds as ordered
feed-forward graphs when that would change structural semantics.

Parameters:
- `genome` - Genome candidate.

Returns: True when the genome can safely adopt feed-forward intent.

### matchesCanonicalFeedForwardPool

```ts
matchesCanonicalFeedForwardPool(
  configuredPool: TopologyIntentMutationMethod[],
  canonicalPool: TopologyIntentMutationMethod[],
): boolean
```

Check whether a configured mutation pool matches the canonical FFW pool.

Parameters:
- `configuredPool` - Mutation pool configured on the NEAT instance.
- `canonicalPool` - Canonical feed-forward mutation pool.

Returns: True when both pools align by operator name and order.

### promoteGenomeToFeedForwardIntentWhenEligible

```ts
promoteGenomeToFeedForwardIntentWhenEligible(
  genome: TopologyIntentGenome,
  shouldPromote: boolean,
): void
```

Promote a genome to feed-forward topology intent when the structure is eligible.

Parameters:
- `genome` - Genome candidate being inserted into a population.
- `shouldPromote` - Whether the active NEAT options request FFW semantics.

Returns: Nothing.

### TopologyIntentGenome

Minimal genome surface required to promote feed-forward intent safely.

### TopologyIntentMutationMethod

Minimal mutation descriptor used by topology-intent helpers.

### usesFeedForwardMutationPolicy

```ts
usesFeedForwardMutationPolicy(
  mutationConfig: unknown,
): boolean
```

Determine whether the configured mutation policy communicates feed-forward intent.

Parameters:
- `mutationConfig` - Configured mutation option.

Returns: True when the option expresses canonical FFW intent.

## neat/neat.evaluate.fitness.utils.ts

### clearGenomeStateIfRequested

```ts
clearGenomeStateIfRequested(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
  clearAction: (genome: GenomeForEvaluation) => void,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.
- `clearAction` - - Action that clears a genome's internal state.

Returns: void.

### runFitnessEvaluation

```ts
runFitnessEvaluation(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): Promise<void>
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Promise<void> after fitness evaluation completes.

## neat/neat.evaluate.novelty.utils.ts

### addGenomeToNoveltyArchive

```ts
addGenomeToNoveltyArchive(
  controller: NeatControllerForEval,
  descriptor: number[],
  novelty: number,
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `descriptor` - - Genome descriptor.
- `novelty` - - Novelty score.
- `noveltyOptions` - - Novelty configuration.

Returns: void.

### applyNoveltyToPopulation

```ts
applyNoveltyToPopulation(
  controller: NeatControllerForEval,
  descriptors: number[][],
  distanceMatrix: number[][],
  kNeighbors: number,
  blendFactor: number,
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `descriptors` - - Descriptor vectors for each genome.
- `distanceMatrix` - - Distance matrix.
- `kNeighbors` - - Neighbor count.
- `blendFactor` - - Blend factor.
- `noveltyOptions` - - Novelty configuration.

Returns: void.

### blendNoveltyIntoScore

```ts
blendNoveltyIntoScore(
  genome: GenomeForEvaluation,
  novelty: number,
  blendFactor: number,
): void
```

Parameters:
- `genome` - - Genome to update.
- `novelty` - - Novelty value.
- `blendFactor` - - Blend factor.

Returns: void.

### buildDistanceMatrix

```ts
buildDistanceMatrix(
  descriptors: number[][],
): number[][]
```

Parameters:
- `descriptors` - - Descriptor vectors.

Returns: Distance matrix.

### buildNoveltyDescriptors

```ts
buildNoveltyDescriptors(
  controller: NeatControllerForEval,
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): number[][]
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `noveltyOptions` - - Novelty configuration.

Returns: Descriptor vectors for each genome.

### computeDescriptorDistance

```ts
computeDescriptorDistance(
  leftDescriptor: number[],
  rightDescriptor: number[],
  isSame: boolean,
): number
```

Parameters:
- `left` - - Left descriptor.
- `right` - - Right descriptor.
- `isSame` - - Whether the descriptors are the same index.

Returns: Euclidean distance.

### computeNoveltyScore

```ts
computeNoveltyScore(
  distanceRow: number[],
  kNeighbors: number,
): number
```

Parameters:
- `distanceRow` - - Distance values for a single genome.
- `kNeighbors` - - Neighbor count.

Returns: Novelty score.

### getNoveltyBlendFactor

```ts
getNoveltyBlendFactor(
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): number
```

Parameters:
- `noveltyOptions` - - Novelty configuration.

Returns: Blend factor for novelty vs. fitness.

### getNoveltyNeighborCount

```ts
getNoveltyNeighborCount(
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): number
```

Parameters:
- `noveltyOptions` - - Novelty configuration.

Returns: Number of neighbors to consider.

### runNoveltyBlendAndArchive

```ts
runNoveltyBlendAndArchive(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.evolve.offspring.utils.ts

### createOffspring

```ts
createOffspring(
  context: OffspringContext,
  selectParent: () => default,
): default
```

Create a child genome by crossing two parents selected via the provided callback.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.

Returns: Newly created offspring genome.

### OffspringContext

Minimal surface needed for offspring generation.

## neat/neat.evolve.telemetry.utils.ts

### computeDiversityStatsSafely

```ts
computeDiversityStatsSafely(
  internal: NeatControllerForEvolution,
): void
```

Compute diversity stats safely if the hook exists.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### recordTelemetryIfEnabled

```ts
recordTelemetryIfEnabled(
  internal: NeatControllerForEvolution,
  snapshot: default,
): Promise<void>
```

Record telemetry if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot for the generation.

Returns: void.

## neat/neat.telemetry.buffer.utils.ts

### ensureTelemetryBuffer

```ts
ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[]
```

Ensure the telemetry buffer is initialized.

Parameters:
- `telemetryContext` - - Neat-like context holding telemetry buffer.

Returns: A mutable telemetry buffer.

### safelyStreamTelemetryEntry

```ts
safelyStreamTelemetryEntry(
  telemetryContext: { options?: TelemetryStreamOptions | undefined; },
  telemetryEntry: TelemetryEntry,
): void
```

Stream telemetry entry when a stream callback is configured.

Parameters:
- `telemetryContext` - - Neat-like context with stream settings.
- `telemetryEntry` - - Entry to stream.

### trimTelemetryBuffer

```ts
trimTelemetryBuffer(
  telemetryBufferRef: TelemetryEntry[],
  maxEntries: number,
): void
```

Trim the telemetry buffer to a maximum size.

Parameters:
- `telemetryBufferRef` - - Buffer to trim in-place.
- `maxEntries` - - Maximum entries to keep.

## neat/neat.adaptive.mutation.utils.ts

### applyAnnealDelta

```ts
applyAnnealDelta(
  baseDelta: number,
  settings: MutationSettings,
): number
```

Apply annealing adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `settings` - - Resolved settings.

Returns: Adjusted delta.

### applyExploreLowDelta

```ts
applyExploreLowDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply explore-low adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyMutationAmount

```ts
applyMutationAmount(
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  settings: MutationSettings,
  randomSource: () => number,
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): void
```

Apply mutation-amount adjustments to a genome.

Parameters:
- `genome` - - Current genome.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

### applyMutationsToPopulation

```ts
applyMutationsToPopulation(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  partitions: MutationPartitions,
  settings: MutationSettings,
  randomSource: () => number,
): MutationOutcome
```

Apply mutation updates to the population.

Parameters:
- `population` - - Full population to mutate.
- `partitions` - - Scored partitions.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.

Returns: Mutation outcome flags.

### applyTwoTierAmountDelta

```ts
applyTwoTierAmountDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply two-tier adjustments to amount delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierDelta

```ts
applyTwoTierDelta(
  baseDelta: number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Apply two-tier adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierFallback

```ts
applyTwoTierFallback(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
  settings: MutationSettings,
): void
```

Apply two-tier fallback balancing.

Parameters:
- `population` - - Population of genomes.
- `settings` - - Resolved settings.

### clampValue

```ts
clampValue(
  value: number,
  min: number,
  max: number,
): number
```

Clamp a value between min and max bounds.

Parameters:
- `value` - - Value to clamp.
- `min` - - Minimum bound.
- `max` - - Maximum bound.

Returns: Clamped value.

### collectScoredGenomes

```ts
collectScoredGenomes(
  population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]
```

Collect genomes with numeric scores.

Parameters:
- `population` - - Population of genomes.

Returns: Scored genomes.

### createRandomDelta

```ts
createRandomDelta(
  sigmaBase: number,
  randomSource: () => number,
): number
```

Create a signed random delta scaled by sigma.

Parameters:
- `sigmaBase` - - Sigma scaling factor.
- `randomSource` - - Random number provider.

Returns: Signed delta.

### resolveAmountDelta

```ts
resolveAmountDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Resolve mutation-amount delta based on strategy.

Parameters:
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Signed mutation amount delta.

### resolveMutationSettings

```ts
resolveMutationSettings(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): MutationSettings
```

Resolve mutation settings derived from configuration and engine state.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Adaptive mutation configuration.

Returns: Resolved mutation settings.

### resolveRandomSource

```ts
resolveRandomSource(
  engine: NeatLikeWithAdaptive,
): () => number
```

Resolve a random source that matches the legacy RNG usage.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Random number provider.

### resolveRateDelta

```ts
resolveRateDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; },
  genomeIndex: number,
  topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
  bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>,
): number
```

Resolve mutation-rate delta based on strategy.

Parameters:
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Signed mutation rate delta.

### shouldAdaptThisGeneration

```ts
shouldAdaptThisGeneration(
  generation: number,
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; },
): boolean
```

Check whether mutation adaptation should run this generation.

Parameters:
- `generation` - - Current generation index.
- `config` - - Adaptive mutation configuration.

Returns: True if adaptation should run.

### shouldApplyTwoTierFallback

```ts
shouldApplyTwoTierFallback(
  strategy: string,
  outcome: MutationOutcome,
): boolean
```

Determine whether a two-tier fallback is needed.

Parameters:
- `strategy` - - Mutation strategy identifier.
- `outcome` - - Mutation outcome flags.

Returns: True if fallback should run.

### sortScoredGenomes

```ts
sortScoredGenomes(
  scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]
```

Sort scored genomes in ascending score order.

Parameters:
- `scoredGenomes` - - Scored genomes.

Returns: Sorted genomes.

### splitScoredGenomes

```ts
splitScoredGenomes(
  scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[],
): MutationPartitions
```

Split scored genomes into top and bottom halves.

Parameters:
- `scoredGenomes` - - Sorted scored genomes.

Returns: Partitions used by strategy rules.

## neat/neat.adaptive.operator.utils.ts

### applyOperatorDecay

```ts
applyOperatorDecay(
  stats: Map<string, { success: number; attempts: number; }>,
  entries: [string, { success: number; attempts: number; }][],
  decay: number,
): void
```

Apply exponential decay to each operator statistic entry.

Parameters:
- `stats` - - Operator statistics map.
- `entries` - - Operator stat entries to update.
- `decay` - - Decay factor.

### collectOperatorStatsEntries

```ts
collectOperatorStatsEntries(
  stats: Map<string, { success: number; attempts: number; }>,
): [string, { success: number; attempts: number; }][]
```

Collect operator statistic entries for processing.

Parameters:
- `stats` - - Operator statistics map.

Returns: Array of operator stat entries.

### decayOperatorStat

```ts
decayOperatorStat(
  operatorStat: { success: number; attempts: number; },
  decay: number,
): { success: number; attempts: number; }
```

Apply decay to a single operator statistic record.

Parameters:
- `operatorStat` - - Operator statistic record.
- `decay` - - Decay factor.

Returns: Decayed operator statistic record.

### resolveOperatorDecay

```ts
resolveOperatorDecay(
  config: { enabled?: boolean | undefined; learningRate?: number | undefined; alpha?: number | undefined; decay?: number | undefined; },
): number
```

Resolve the decay factor for operator statistics.

Parameters:
- `config` - - Operator adaptation configuration.

Returns: Decay factor for exponential smoothing.

## neat/neat.evolve.objectives.utils.ts

### applyDynamicObjectiveSchedule

```ts
applyDynamicObjectiveSchedule(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  config: { autoEntropyAddAt: number; },
): void
```

Apply dynamic objective scheduling and entropy rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Keys of active objectives.
- `config` - - Scheduling constants.

Returns: void.

### applyFitnessSuppressionForTests

```ts
applyFitnessSuppressionForTests(
  internal: NeatControllerForEvolution,
): void
```

Suppress fitness objective for specific test scenarios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### captureObjectiveImportanceSnapshot

```ts
captureObjectiveImportanceSnapshot(
  internal: NeatControllerForEvolution,
): void
```

Capture objective importance stats for telemetry.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### handleEntropyDropAndReadd

```ts
handleEntropyDropAndReadd(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  dynamicConfig: { enabled?: boolean | undefined; addComplexityAt?: number | undefined; addEntropyAt?: number | undefined; dropEntropyOnStagnation?: number | undefined; readdEntropyAfter?: number | undefined; } | undefined,
): void
```

Handle entropy removal and re-addition rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Active objective keys.
- `dynamicConfig` - - Dynamic objective config.

Returns: void.

### resetObjectivesCache

```ts
resetObjectivesCache(
  internal: NeatControllerForEvolution,
): void
```

Clear cached objectives so dynamic schedules can rebuild them.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### updateObjectiveScheduleAndAges

```ts
updateObjectiveScheduleAndAges(
  internal: NeatControllerForEvolution,
  helpers: { applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) => void; },
): Promise<void>
```

Update objective schedule, pending adds/removes, and objective ages.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used by scheduling logic.
- `helpers` - - Dynamic objective scheduler.

Returns: void.

## neat/neat.evolve.population.utils.ts

### addOffspring

```ts
addOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  helpers: { addSpeciatedOffspring: (nextPopulation: default[], remainingSlots: number) => Promise<void>; addUnspeciatedOffspring: (nextPopulation: default[], remainingSlots: number) => Promise<void>; },
): Promise<void>
```

Add offspring to fill remaining population slots.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `helpers` - - Helper callbacks for offspring selection.
- `helpers` - - Speciated offspring helper.
- `helpers` - - Unspeciated offspring helper.

Returns: void.

### addSpeciatedOffspring

```ts
addSpeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  remainingSlots: number,
  config: { minOffspringDefault: number; survivalThresholdDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; crossSpeciesGuardLimit: number; },
): Promise<void>
```

Add offspring when speciation is enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Offspring allocation constants.

Returns: void.

### addUnspeciatedOffspring

```ts
addUnspeciatedOffspring(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
  remainingSlots: number,
): Promise<void>
```

Add offspring when speciation is disabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.

Returns: void.

### applyElitism

```ts
applyElitism(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): void
```

Apply elitism for the next generation.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyProvenance

```ts
applyProvenance(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): void
```

Add provenance genomes into the next population.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### buildNextPopulation

```ts
buildNextPopulation(
  internal: NeatControllerForEvolution,
  helpers: { applyElitism: (nextPopulation: default[]) => void; applyProvenance: (nextPopulation: default[]) => void; addOffspring: (nextPopulation: default[]) => Promise<void>; },
): Promise<default[]>
```

Build the next population (elitism, provenance, offspring).

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for population construction.
- `helpers` - - Elitism helper.
- `helpers` - - Provenance helper.
- `helpers` - - Offspring helper.

Returns: next population array.

### buildSpeciesOffspring

```ts
buildSpeciesOffspring(
  internal: NeatControllerForEvolution,
  survivors: GenomeWithMetadata[],
  speciesIndex: number,
  crossSpeciesProbability: number,
  crossSpeciesGuardLimit: number,
  survivalThresholdDefault: number,
): GenomeWithMetadata
```

Build a single offspring within a species.

Parameters:
- `internal` - - NEAT controller instance.
- `survivors` - - Survivors pool for selection.
- `speciesIndex` - - Species index.
- `crossSpeciesProbability` - - Cross-species mating probability.
- `crossSpeciesGuardLimit` - - Retry guard for cross-species selection.

Returns: offspring genome.

### computeOffspringAllocation

```ts
computeOffspringAllocation(
  internal: NeatControllerForEvolution,
  remainingSlots: number,
  config: { minOffspringDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; },
): number[]
```

Compute offspring allocation per species.

Parameters:
- `internal` - - NEAT controller instance.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Allocation constants.

Returns: allocation per species index.

### distributeRemainingSlots

```ts
distributeRemainingSlots(
  allocation: number[],
  rawShares: number[],
  remainingSlots: number,
): void
```

Distribute leftover slots by fractional remainders.

Parameters:
- `allocation` - - Allocation array to adjust.
- `rawShares` - - Raw fractional shares.
- `remainingSlots` - - Total slots available.

Returns: void.

### enforceMinimumOffspring

```ts
enforceMinimumOffspring(
  internal: NeatControllerForEvolution,
  allocation: number[],
  remainingSlots: number,
  minOffspringDefault: number,
): void
```

Enforce minimum offspring per species when possible.

Parameters:
- `internal` - - NEAT controller instance.
- `allocation` - - Allocation array to adjust.
- `remainingSlots` - - Total slots available.
- `minOffspringDefault` - - Default minimum offspring.

Returns: void.

### enforcePopulationConstraints

```ts
enforcePopulationConstraints(
  internal: NeatControllerForEvolution,
  nextPopulation: default[],
): Promise<void>
```

Ensure new population meets structural constraints.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Population to validate.

Returns: void.

### selectSecondParent

```ts
selectSecondParent(
  internal: NeatControllerForEvolution,
  survivors: GenomeWithMetadata[],
  speciesIndex: number,
  crossSpeciesProbability: number,
  crossSpeciesGuardLimit: number,
  survivalThresholdDefault: number,
): GenomeWithMetadata
```

Select a second parent, optionally from another species.

Parameters:
- `internal` - - NEAT controller instance.
- `survivors` - - Survivors pool from the current species.
- `speciesIndex` - - Current species index.
- `crossSpeciesProbability` - - Probability to cross species.
- `crossSpeciesGuardLimit` - - Retry guard for cross-species selection.

Returns: chosen parent genome.

### trimOversubscription

```ts
trimOversubscription(
  internal: NeatControllerForEvolution,
  allocation: number[],
  remainingSlots: number,
  minOffspringDefault: number,
): void
```

Trim allocations when oversubscribed.

Parameters:
- `internal` - - NEAT controller instance.
- `allocation` - - Allocation array to adjust.
- `remainingSlots` - - Total slots available.
- `minOffspringDefault` - - Default minimum offspring.

Returns: void.

## neat/neat.evolve.speciation.utils.ts

### applyGlobalStagnationInjectionIfNeeded

```ts
applyGlobalStagnationInjectionIfNeeded(
  internal: NeatControllerForEvolution,
  helpers: { buildFreshGenomeForStagnation: () => Promise<GenomeWithMetadata>; replaceFraction: number; },
): Promise<void>
```

Apply global stagnation injection if configured.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for stagnation injection.
- `helpers` - - Genome builder for injection.

Returns: void.

### applySpeciationAndSharingIfEnabled

```ts
applySpeciationAndSharingIfEnabled(
  internal: NeatControllerForEvolution,
  helpers: { applyAutoCompatibilityTuning: () => void; recordSpeciesHistorySnapshot: () => void; },
): Promise<void>
```

Apply speciation, fitness sharing, and related side effects.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used for tuning and history.
- `helpers` - - Auto-compatibility adjustment helper.
- `helpers` - - Species history snapshot helper.

Returns: void.

### buildFreshGenomeForStagnation

```ts
buildFreshGenomeForStagnation(
  internal: NeatControllerForEvolution,
): Promise<GenomeWithMetadata>
```

Build a fresh genome for stagnation injection.

Parameters:
- `internal` - - NEAT controller instance.

Returns: new genome with minimum constraints.

### ensureHiddenNodeVariance

```ts
ensureHiddenNodeVariance(
  internal: NeatControllerForEvolution,
  genome: GenomeWithMetadata,
): Promise<void>
```

Ensure a minimal hidden-node variance in injected genomes.

Parameters:
- `internal` - - NEAT controller instance.
- `genome` - - Genome to adjust.

Returns: void.

### ensureSpeciesHistorySnapshot

```ts
ensureSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void
```

Ensure a minimal species history snapshot exists for exports.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### recordSpeciesHistorySnapshot

```ts
recordSpeciesHistorySnapshot(
  internal: NeatControllerForEvolution,
  maxHistory: number,
): void
```

Record a species history snapshot when needed.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### updateSpeciesStagnationIfEnabled

```ts
updateSpeciesStagnationIfEnabled(
  internal: NeatControllerForEvolution,
): void
```

Update species stagnation status when speciation enabled.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

## neat/neat.mutation.add-conn.utils.ts

### assignInnovationForConnection

```ts
assignInnovationForConnection(
  connection: ConnectionWithMetadata,
  pairNodes: { symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; },
  internal: NeatControllerForMutation,
): void
```

Assign an innovation id for a new connection, reusing when possible.

Parameters:
- `connection` - - newly created connection
- `pairNodes` - - resolved pair metadata
- `internal` - - neat controller context

Returns: void

### buildLegacyKeyForConn

```ts
buildLegacyKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string
```

Build a legacy directional innovation key.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: directional innovation key

### buildSymmetricKeyForConn

```ts
buildSymmetricKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string
```

Build a symmetric innovation key for an unordered node pair.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: symmetric innovation key

### choosePairForConn

```ts
choosePairForConn(
  pairs: [NodeWithMetadata, NodeWithMetadata][],
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata] | null
```

Choose a pair deterministically when only one candidate exists.

Parameters:
- `pairs` - - selection pool
- `internal` - - neat controller context

Returns: chosen pair or null

### collectCandidatePairsForConn

```ts
collectCandidatePairsForConn(
  genomeToInspect: GenomeWithMetadata,
): [NodeWithMetadata, NodeWithMetadata][]
```

Collect legal (from,to) node pairs not already connected.

Parameters:
- `genomeToInspect` - - genome to scan

Returns: candidate node pairs

### connectChosenPair

```ts
connectChosenPair(
  genomeToEdit: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; },
): ConnectionWithMetadata | undefined
```

Create the connection for the chosen pair.

Parameters:
- `genomeToEdit` - - genome to edit
- `pairNodes` - - resolved pair nodes

Returns: created connection or undefined

### createsCycle

```ts
createsCycle(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): boolean
```

Detect whether adding a connection would create a cycle.

Parameters:
- `sourceNode` - - source node of the new connection
- `targetNode` - - target node of the new connection

Returns: true when a cycle is detected

### filterPairsWithInnovations

```ts
filterPairsWithInnovations(
  pairs: [NodeWithMetadata, NodeWithMetadata][],
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata][]
```

Filter candidate pairs that already have innovation reuse keys.

Parameters:
- `pairs` - - candidate node pairs
- `internal` - - neat controller context

Returns: reuse candidates

### resolvePairNodes

```ts
resolvePairNodes(
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
): { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }
```

Resolve nodes and innovation key details for a chosen pair.

Parameters:
- `chosenPair` - - pair to connect

Returns: resolved pair metadata

### selectPairPool

```ts
selectPairPool(
  allPairs: [NodeWithMetadata, NodeWithMetadata][],
  reusePairs: [NodeWithMetadata, NodeWithMetadata][],
): [NodeWithMetadata, NodeWithMetadata][]
```

Build the final selection pool based on reuse and hidden-node preference.

Parameters:
- `allPairs` - - all candidate pairs
- `reusePairs` - - pairs with historical innovations

Returns: selection pool

### shouldAbortForCycle

```ts
shouldAbortForCycle(
  genomeToInspect: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata; },
): boolean
```

Determine whether adding the connection would create a cycle.

Parameters:
- `genomeToInspect` - - genome to inspect
- `pairNodes` - - resolved pair nodes

Returns: true if the connection should be aborted

## neat/neat.mutation.add-node.utils.ts

### applySplitWithExistingRecord

```ts
applySplitWithExistingRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number; },
  splitRecord: { newNodeGeneId: number; inInnov: number; outInnov: number; },
  NodeClass: new (type: "input" | "output" | "hidden") => unknown,
): void
```

Apply a split using an existing innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `splitRecord` - - existing innovation record
- `NodeClass` - - node constructor

Returns: void

### applySplitWithNewRecord

```ts
applySplitWithNewRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number; },
  NodeClass: new (type: "input" | "output" | "hidden") => unknown,
  internal: NeatControllerForMutation,
): void
```

Apply a split and create a new innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `NodeClass` - - node constructor
- `internal` - - neat controller context

Returns: void

### assignInnovationsForNewSplit

```ts
assignInnovationsForNewSplit(
  newNode: NodeWithMetadata,
  splitConnections: { incomingConnection?: ConnectionWithMetadata | undefined; outgoingConnection?: ConnectionWithMetadata | undefined; },
  internal: NeatControllerForMutation,
): { newNodeGeneId: number; inInnov: number; outInnov: number; }
```

Assign new innovations for a split and build the innovation record.

Parameters:
- `newNode` - - newly created hidden node
- `splitConnections` - - incoming/outgoing connections
- `internal` - - neat controller context

Returns: innovation record for the split

### buildSplitDescriptor

```ts
buildSplitDescriptor(
  connectionToSplit: ConnectionWithMetadata,
): { splitKey: string; originalWeight: number; }
```

Build the split descriptor used for innovation lookup and connection creation.

Parameters:
- `connectionToSplit` - - connection being split

Returns: split descriptor

### chooseConnectionForSplit

```ts
chooseConnectionForSplit(
  enabledConnectionsList: ConnectionWithMetadata[],
  internal: NeatControllerForMutation,
): ConnectionWithMetadata | null
```

Choose a random enabled connection to split.

Parameters:
- `enabledConnectionsList` - - candidate connections
- `internal` - - neat controller context

Returns: selected connection or null

### collectEnabledConnections

```ts
collectEnabledConnections(
  genomeToInspect: GenomeWithMetadata,
): ConnectionWithMetadata[]
```

Collect all enabled connections from a genome.

Parameters:
- `genomeToInspect` - - genome to inspect

Returns: enabled connections list

### connectSplitEdges

```ts
connectSplitEdges(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  newNode: NodeWithMetadata,
  originalWeight: number,
): { incomingConnection?: ConnectionWithMetadata | undefined; outgoingConnection?: ConnectionWithMetadata | undefined; }
```

Create the incoming and outgoing split connections.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `newNode` - - newly created hidden node
- `originalWeight` - - weight to preserve on the outgoing connection

Returns: incoming/outgoing connection handles

### disconnectOriginalConnection

```ts
disconnectOriginalConnection(
  genomeToEdit: GenomeWithMetadata,
  connectionToRemove: ConnectionWithMetadata,
): void
```

Disconnect the original connection before inserting the split node.

Parameters:
- `genomeToEdit` - - genome to edit
- `connectionToRemove` - - original connection to remove

Returns: void

### ensureBootstrapConnection

```ts
ensureBootstrapConnection(
  genomeToSeed: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure the genome has at least one connection by linking input to output.

Parameters:
- `genomeToSeed` - - genome that may need a bootstrap connection
- `internal` - - neat controller context retained for compatibility with existing callers

Returns: void

### findFirstNodeByType

```ts
findFirstNodeByType(
  genomeToSearch: GenomeWithMetadata,
  nodeType: "input" | "output" | "hidden",
): NodeWithMetadata | undefined
```

Find the first node of a given type.

Parameters:
- `genomeToSearch` - - genome whose nodes are searched
- `nodeType` - - node type to match

Returns: the first matching node or undefined

### resolveInsertIndex

```ts
resolveInsertIndex(
  genomeToEdit: GenomeWithMetadata,
  targetNode: NodeWithMetadata,
): number
```

Resolve the insertion index for a new node, keeping outputs at the end.

Parameters:
- `genomeToEdit` - - genome whose node list is updated
- `targetNode` - - original target node of the split connection

Returns: insertion index

## neat/neat.telemetry.entropy.utils.ts

### buildDegreeHistogram

```ts
buildDegreeHistogram(
  counts: Record<number, number>,
): Record<number, number>
```

Build a histogram of degree frequencies from a degree-count table.

Parameters:
- `counts` - - Map geneId -> degree count.

Returns: Map degree -> number of nodes with that degree.

### computeDegreeCounts

```ts
computeDegreeCounts(
  entropyGraph: { nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; },
): Record<number, number>
```

Compute per-node degree counts for enabled connections.

Parameters:
- `entropyGraph` - - Genome-like graph object.

Returns: Map geneId -> degree count.

### computeEntropyFromHistogram

```ts
computeEntropyFromHistogram(
  histogram: Record<number, number>,
  totalNodes: number,
): number
```

Compute entropy from a degree-frequency histogram.

Parameters:
- `histogram` - - Map degree -> number of nodes.
- `totalNodes` - - Total node count used to normalize into probabilities.

Returns: Entropy value (non-negative).

### getCachedEntropy

```ts
getCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
): number | undefined
```

Read a cached entropy value if it exists and belongs to the current
generation.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.

Returns: Cached entropy number, or undefined when not available.

### setCachedEntropy

```ts
setCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
  entropyValue: number,
): void
```

Cache an entropy value for the current generation on the graph object.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.
- `entropyValue` - - Entropy value to cache.

## neat/neat.telemetry.exports.utils.ts

### buildSpeciesHistoryStats

```ts
buildSpeciesHistoryStats(
  speciesList: SpeciesHistoryStat[],
  defaultSpeciesId: number,
  defaultSpeciesSize: number,
  defaultBestScore: number,
  defaultLastImproved: number,
): SpeciesHistoryStat[]
```

Normalize raw species records into exportable history stats.

Parameters:
- `speciesList` - - Raw species records to normalize.
- `defaultSpeciesId` - - Default species id when missing.
- `defaultSpeciesSize` - - Default species size when missing.
- `defaultBestScore` - - Default best score when missing.
- `defaultLastImproved` - - Default last improved when missing.

Returns: Normalized stats for CSV export.

### collectBaseKeys

```ts
collectBaseKeys(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
  frontsHeader: string,
): void
```

Collect base (top-level) telemetry keys for a single entry.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.
- `frontsHeader` - - Header label for fronts column.

Returns: void. Mutates `state.baseKeys`.

### collectDiversityLineageMetrics

```ts
collectDiversityLineageMetrics(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void
```

Collect curated diversity lineage metrics for stable CSV exports.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates diversity lineage key set.

### collectGroupedMetricKeys

```ts
collectGroupedMetricKeys(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void
```

Collect nested metric keys for grouped telemetry fields.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates complexity/perf/lineage key sets.

### collectOptionalColumnPresence

```ts
collectOptionalColumnPresence(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void
```

Collect presence flags for optional telemetry columns.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates optional-column flags.

### collectSpeciesHistoryHeaders

```ts
collectSpeciesHistoryHeaders(
  history: SpeciesHistoryEntry[],
  generationHeader: string,
): string[]
```

Collect ordered header keys for species history CSV export.

Parameters:
- `history` - - Recent species history entries.
- `generationHeader` - - Header label for generation column.

Returns: Ordered header list for CSV output.

### ensureMinimalSpeciesSnapshot

```ts
ensureMinimalSpeciesSnapshot(
  neatInstance: NeatLike & { _speciesHistory?: SpeciesHistoryEntry[] | undefined; _species?: SpeciesHistoryStat[] | undefined; generation?: number | undefined; },
  history: SpeciesHistoryEntry[],
  fallbackGeneration: number,
  defaultSpeciesId: number,
  defaultSpeciesSize: number,
  defaultBestScore: number,
  defaultLastImproved: number,
): void
```

Ensure a minimal species snapshot exists for deterministic CSV headers.

Parameters:
- `neatInstance` - - Neat instance with optional species history and species.
- `history` - - Species history backing array.
- `fallbackGeneration` - - Generation fallback when missing.
- `defaultSpeciesId` - - Default species id when missing.
- `defaultSpeciesSize` - - Default species size when missing.
- `defaultBestScore` - - Default best score when missing.
- `defaultLastImproved` - - Default last improved when missing.

Returns: void. Mutates history when a minimal snapshot is needed.

### ensureSpeciesHistoryArray

```ts
ensureSpeciesHistoryArray(
  neatInstance: NeatLike & { _speciesHistory?: SpeciesHistoryEntry[] | undefined; },
): SpeciesHistoryEntry[]
```

Ensure the species history array exists on the Neat instance.

Parameters:
- `neatInstance` - - Neat instance holding species history.

Returns: Species history backing array (ensured on instance).

### resolveSpeciesHistoryCellValue

```ts
resolveSpeciesHistoryCellValue(
  historyEntry: SpeciesHistoryEntry,
  speciesStat: SpeciesHistoryStat,
  headerName: string,
  generationHeader: string,
): string
```

Resolve a single species history cell value for the provided header.

Parameters:
- `historyEntry` - - A single generation snapshot.
- `speciesStat` - - A single species stat record.
- `headerName` - - Column header name.
- `generationHeader` - - Column header name for generation.

Returns: Serialized cell (JSON) or empty string for missing values.

### safeStringifyCell

```ts
safeStringifyCell(
  value: unknown,
): string
```

Serialize a CSV cell with JSON.stringify safeguards.

Parameters:
- `value` - - Any value to stringify.

Returns: JSON string or empty string when JSON.stringify returns undefined.

### serializeSpeciesHistoryRow

```ts
serializeSpeciesHistoryRow(
  historyEntry: SpeciesHistoryEntry,
  speciesStat: SpeciesHistoryStat,
  orderedHeaders: string[],
  generationHeader: string,
): string
```

Serialize one species history row using the provided headers.

Parameters:
- `historyEntry` - - A single generation snapshot.
- `speciesStat` - - A single species stat record for that generation.
- `orderedHeaders` - - Ordered header list for stable CSV.
- `generationHeader` - - Column header name for generation.

Returns: CSV row string matching the provided header order.

### TelemetryHeaderCollectionState

Mutable state container used while collecting telemetry header metadata.

## neat/neat.telemetry.lineage.utils.ts

### applyLineageStatsMonoObjective

```ts
applyLineageStatsMonoObjective(
  telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; },
  populationSnapshot: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Apply lineage stats for mono-objective mode using sampled ancestors.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyLineageStatsMultiObjective

```ts
applyLineageStatsMultiObjective(
  telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; },
  population: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Apply lineage stats for multi-objective mode using ancestor uniqueness.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### buildLineageContext

```ts
buildLineageContext(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSnapshot: GenomeDetailed[],
): NeatLineageContext
```

Build a lineage helper context for ancestor operations.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Lineage helper context.

### buildLineageEntry

```ts
buildLineageEntry(
  context: { _prevInbreedingCount?: number | undefined; },
  bestGenomeSnapshot: GenomeDetailed,
  meanDepthValue: number,
  ancestorUniquenessScore: number,
): { parents: number[]; depthBest: number; meanDepth: number; inbreeding: number; ancestorUniq: number; }
```

Build the lineage entry payload.

Parameters:
- `context` - - Neat-like context with lineage info.
- `bestGenomeSnapshot` - - Best genome snapshot.
- `meanDepthValue` - - Mean lineage depth.
- `ancestorUniquenessScore` - - Ancestor uniqueness score.

Returns: Lineage entry payload.

### collectDepths

```ts
collectDepths(
  populationSnapshot: GenomeDetailed[],
): number[]
```

Collect depth values for the current population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of depth values (defaults to 0).

### computeAncestorUniquenessSampled

```ts
computeAncestorUniquenessSampled(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSnapshot: GenomeDetailed[],
): number
```

Compute ancestor uniqueness using sampled Jaccard distance.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Rounded ancestor uniqueness score.

### computeLineageStats

```ts
computeLineageStats(
  lineageEnabled: boolean,
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
): { lineageMeanDepth: number; lineageMeanPairDist: number; }
```

Compute lineage depth and pairwise depth-distance statistics.

Parameters:
- `lineageEnabled` - - Whether lineage metrics are enabled.
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Lineage mean depth and pairwise distance.

### computeMeanDepth

```ts
computeMeanDepth(
  depthValues: number[],
): number
```

Compute the mean depth from a depth list.

Parameters:
- `depthValues` - - Depth values to average.

Returns: Mean depth value.

### computePairJaccardDistance

```ts
computePairJaccardDistance(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSnapshot: GenomeDetailed[],
  firstIndex: number,
  secondIndex: number,
): number | undefined
```

Compute Jaccard distance between ancestor sets for a pair.

Parameters:
- `context` - - Neat-like context for lineage helpers.
- `populationSnapshot` - - Population snapshot.
- `firstIndex` - - First genome index.
- `secondIndex` - - Second genome index.

Returns: Jaccard distance or undefined when both sets are empty.

### countAncestorIntersection

```ts
countAncestorIntersection(
  ancestorsA: Set<number>,
  ancestorsB: Set<number>,
): number
```

Count the size of an ancestor intersection.

Parameters:
- `ancestorsA` - - First ancestor set.
- `ancestorsB` - - Second ancestor set.

Returns: Intersection count.

### isLineageEligible

```ts
isLineageEligible(
  context: { _lineageEnabled?: boolean | undefined; },
  populationSnapshot: GenomeDetailed[],
): boolean
```

Check whether lineage metrics should be computed.

Parameters:
- `context` - - Neat-like context with lineage flag.
- `populationSnapshot` - - Population snapshot to validate.

Returns: True when lineage stats should be computed.

### pickDistinctPairIndices

```ts
pickDistinctPairIndices(
  context: { _getRNG?: (() => () => number) | undefined; },
  populationSize: number,
): { firstIndex: number; secondIndex: number; }
```

Pick two distinct indices using the context RNG.

Parameters:
- `context` - - Neat-like context with RNG factory.
- `populationSize` - - Population size for index bounds.

Returns: Pair of distinct indices.

## neat/neat.evaluate.constants.utils.ts

Default neighbor count for novelty calculation.

### AUTO_COEFF_ADJUST_DEFAULT

Default adjustment rate for auto distance coefficient tuning.

### AUTO_COEFF_MAX_DEFAULT

Default maximum coefficient for auto distance coefficient tuning.

### AUTO_COEFF_MIN_DEFAULT

Default minimum coefficient for auto distance coefficient tuning.

### COMPAT_MAX_THRESHOLD_DEFAULT

Default maximum compatibility threshold.

### COMPAT_MIN_THRESHOLD_DEFAULT

Default minimum compatibility threshold.

### COMPAT_THRESHOLD_DEFAULT

Default compatibility threshold when not provided.

### DISTANCE_COEFF_DEFAULT

Default coefficient value when not provided.

### ENTROPY_ADJUST_DEFAULT

Default adjustment rate for compatibility tuning.

### ENTROPY_DEADBAND_DEFAULT

Default deadband for compatibility tuning.

### ENTROPY_TARGET_DEFAULT

Default target entropy for compatibility tuning.

### ENTROPY_VAR_ADJUST_DEFAULT

Default adjustment rate for entropy sharing.

### ENTROPY_VAR_HIGH_BAND

Upper band multiplier for entropy variance tuning.

### ENTROPY_VAR_LOW_BAND

Lower band multiplier for entropy variance tuning.

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

Default maximum sigma for entropy sharing.

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

Default minimum sigma for entropy sharing.

### ENTROPY_VAR_TARGET_DEFAULT

Default target variance for entropy sharing.

### NOVELTY_ARCHIVE_CAP

Maximum number of entries stored in the novelty archive.

### NOVELTY_DEFAULT_BLEND

Default blend factor for novelty vs. fitness.

### NOVELTY_DEFAULT_NEIGHBORS

Default neighbor count for novelty calculation.

### VARIANCE_DECREASE_THRESHOLD

Variance decrease threshold multiplier.

### VARIANCE_INCREASE_THRESHOLD

Variance increase threshold multiplier.

## neat/neat.mutation.dead-ends.utils.ts

### chooseRandomNodeForDeadEnds

```ts
chooseRandomNodeForDeadEnds(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null
```

Choose a random node from candidates for dead-end repair.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectNodeGroupsForDeadEnds

```ts
collectNodeGroupsForDeadEnds(
  networkToInspect: GenomeWithMetadata,
): { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; }
```

Collect categorized node arrays for dead-end repair.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### connectIfCandidatesExistForDeadEnds

```ts
connectIfCandidatesExistForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  anchorNode: NodeWithMetadata,
  candidates: NodeWithMetadata[],
  reverse: boolean,
  internal: NeatControllerForMutation,
): void
```

Connect a node to a random candidate if candidates exist.

Parameters:
- `networkToEdit` - - network to edit
- `anchorNode` - - node to connect from/to
- `candidates` - - candidate nodes for connection
- `reverse` - - whether to connect candidate -> anchor
- `internal` - - neat controller context

Returns: void

### ensureHiddenConnectivityForDeadEnds

```ts
ensureHiddenConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureInputConnectivityForDeadEnds

```ts
ensureInputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure all input nodes have at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureOutputConnectivityForDeadEnds

```ts
ensureOutputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure all output nodes have at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### hasIncomingForDeadEnds

```ts
hasIncomingForDeadEnds(
  node: NodeWithMetadata,
): boolean
```

Check whether a node has any incoming connections.

Parameters:
- `node` - - node to inspect

Returns: true when incoming connections exist

### hasOutgoingForDeadEnds

```ts
hasOutgoingForDeadEnds(
  node: NodeWithMetadata,
): boolean
```

Check whether a node has any outgoing connections.

Parameters:
- `node` - - node to inspect

Returns: true when outgoing connections exist

## neat/neat.telemetry.operator.utils.ts

### computeOperatorStatsSnapshot

```ts
computeOperatorStatsSnapshot(
  operatorStats: OperatorStatsMap | undefined,
): { op: string; succ: number; att: number; }[]
```

Snapshot operator statistics into a telemetry-friendly array.

Parameters:
- `operatorStats` - - Operator stats map (opName -> success/attempts).

Returns: Operator stats snapshot array.

### readOperatorStats

```ts
readOperatorStats(
  operatorStats: OperatorStatsMap | undefined,
): { name: string; success: number; attempts: number; }[]
```

Convert operator stats map into the public accessor shape.

## neat/neat.adaptive.complexity.utils.ts

### adjustConnectionBudget

```ts
adjustConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  factors: { increaseFactor: number; stagnationFactor: number; },
  noveltyFactor: number,
  history: number[],
): void
```

Adjust connection budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### adjustNodeBudget

```ts
adjustNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  factors: { increaseFactor: number; stagnationFactor: number; },
  noveltyFactor: number,
  history: number[],
): void
```

Adjust node budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### applyAdaptiveSchedule

```ts
applyAdaptiveSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply adaptive complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance with adaptive state.
- `config` - - Complexity budget configuration.

### applyComplexityBudgetSchedule

```ts
applyComplexityBudgetSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply the complexity budget schedule for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyLinearSchedule

```ts
applyLinearSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply linear complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### clampNodeBudget

```ts
clampNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Clamp node budget to configured minimum.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### computeAdjustmentFactors

```ts
computeAdjustmentFactors(
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  history: number[],
): { increaseFactor: number; stagnationFactor: number; }
```

Compute adjustment factors for budget growth and decay.

Parameters:
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `history` - - Rolling history of best scores.

Returns: Adjustment factors (increase and stagnation multipliers).

### computeNoveltyFactor

```ts
computeNoveltyFactor(
  engine: NeatLikeWithAdaptive,
): number
```

Compute novelty factor based on archive size.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Novelty multiplier (0.9 if archive small, 1.0 otherwise).

### computeSlope

```ts
computeSlope(
  history: number[],
): number
```

Compute linear regression slope using ordinary least squares.

Parameters:
- `history` - - Rolling history of best scores.

Returns: OLS slope estimate.

### computeTrends

```ts
computeTrends(
  history: number[],
): { improvement: number; slope: number; }
```

Compute improvement and slope trends from score history.

Parameters:
- `history` - - Rolling history of best scores.

Returns: Trend metrics (improvement and slope).

### initializeConnectionBudget

```ts
initializeConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Initialize connection budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializeNodeBudget

```ts
initializeNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Initialize node budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### normalizeSlope

```ts
normalizeSlope(
  slope: number,
  initialScore: number,
): number
```

Normalize slope magnitude relative to initial score.

Parameters:
- `slope` - - Raw OLS slope.
- `initialScore` - - First score in history window.

Returns: Normalized slope clamped to [-2, 2].

### updateScoreHistory

```ts
updateScoreHistory(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): number[]
```

Update rolling score history with current best score.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

Returns: Rolling history array after update.

## neat/neat.evaluate.objectives.utils.ts

### registerEntropyObjective

```ts
registerEntropyObjective(
  controller: NeatControllerForEval,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.

Returns: void.

### runAutoEntropyObjectiveInjection

```ts
runAutoEntropyObjectiveInjection(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### shouldAutoInjectEntropy

```ts
shouldAutoInjectEntropy(
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): boolean
```

Parameters:
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Whether entropy objective should be injected.

## neat/neat.evaluate.speciation.utils.ts

### runLightweightSpeciation

```ts
runLightweightSpeciation(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### shouldRunSpeciation

```ts
shouldRunSpeciation(
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): boolean
```

Parameters:
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Whether speciation should be run.

## neat/neat.mutation.min-hidden.utils.ts

### chooseRandomNodeForMinHidden

```ts
chooseRandomNodeForMinHidden(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null
```

Choose a random node from a candidate list.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectNodeGroupsForMinHidden

```ts
collectNodeGroupsForMinHidden(
  networkToInspect: GenomeWithMetadata,
): { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; }
```

Collect categorized node arrays for the network.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### computeMinimumHiddenSize

```ts
computeMinimumHiddenSize(
  inputCount: number,
  outputCount: number,
  explicitMinimumHidden: number | undefined,
  hiddenMultiplier: number | undefined,
): number
```

Compute the minimum hidden node count using explicit or multiplier-based settings.

Parameters:
- `inputCount` - - Number of input nodes in the network.
- `outputCount` - - Number of output nodes in the network.
- `explicitMinimumHidden` - - Optional explicit minimum hidden count.
- `hiddenMultiplier` - - Optional multiplier used when explicit minimum is absent.

Returns: Minimum hidden node requirement.

### ensureHiddenConnectivityForMinHidden

```ts
ensureHiddenConnectivityForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  internal: NeatControllerForMutation,
): void
```

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenNodeCountForMinHidden

```ts
ensureHiddenNodeCountForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToEdit: { hiddenNodes: NodeWithMetadata[]; },
  minimumHidden: number,
  maxNodesLimit: number,
): Promise<void>
```

Ensure the network has at least the minimum number of hidden nodes.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToEdit` - - grouped node arrays
- `minimumHidden` - - minimum hidden nodes required
- `maxNodesLimit` - - maximum allowed nodes

Returns: Promise resolving when nodes are created

### ensureIncomingConnectionForMinHidden

```ts
ensureIncomingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { inputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure a hidden node has at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureOutgoingConnectionForMinHidden

```ts
ensureOutgoingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: { outputNodes: NodeWithMetadata[]; hiddenNodes: NodeWithMetadata[]; },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Ensure a hidden node has at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### hasRequiredEndpointsForMinHidden

```ts
hasRequiredEndpointsForMinHidden(
  nodeGroupsToCheck: { inputNodes: NodeWithMetadata[]; outputNodes: NodeWithMetadata[]; },
): boolean
```

Check whether the network has at least one input and output node.

Parameters:
- `nodeGroupsToCheck` - - grouped node arrays

Returns: true when inputs and outputs are present

### MINIMUM_HIDDEN_BASELINE

Baseline minimum hidden nodes when no configuration is provided.

### rebuildNetworkConnectionsForMinHidden

```ts
rebuildNetworkConnectionsForMinHidden(
  networkToEdit: GenomeWithMetadata,
): Promise<void>
```

Rebuild connection caches after structural edits.

Parameters:
- `networkToEdit` - - network to rebuild

Returns: Promise resolving after rebuild completes

### resolveMaxNodesForMinHidden

```ts
resolveMaxNodesForMinHidden(
  internal: NeatControllerForMutation,
): number
```

Resolve the maximum node limit for the network.

Parameters:
- `internal` - - neat controller context

Returns: maximum node limit

### resolveMinHiddenForMinHidden

```ts
resolveMinHiddenForMinHidden(
  networkToInspect: GenomeWithMetadata,
  maxNodesLimit: number,
  multiplier: number | undefined,
  internal: NeatControllerForMutation,
): number
```

Resolve the minimum hidden node requirement for the network.

Parameters:
- `networkToInspect` - - network to inspect
- `maxNodesLimit` - - maximum allowed nodes
- `multiplier` - - optional size multiplier
- `internal` - - neat controller context

Returns: minimum hidden node count

### warnMissingEndpointsForMinHidden

```ts
warnMissingEndpointsForMinHidden(): void
```

Emit a warning when the network lacks input or output nodes.

Returns: void

## neat/neat.telemetry.accessors.utils.ts

### buildLineageSnapshot

```ts
buildLineageSnapshot(
  population: { _id?: number | undefined; _parents?: number[] | undefined; }[],
  limit: number,
): { id: number; parents: number[]; }[]
```

Snapshot lineage metadata for the first `limit` genomes.

### clearTelemetryBuffer

```ts
clearTelemetryBuffer(
  host: TelemetryAccessorHost,
): void
```

Clear the telemetry buffer in place.

### getCachedDiversityStats

```ts
getCachedDiversityStats(
  host: TelemetryAccessorHost,
): DiversityStats | undefined
```

Read cached diversity statistics.

### getObjectiveEventsSnapshot

```ts
getObjectiveEventsSnapshot(
  host: TelemetryAccessorHost,
): { gen: number; type: "add" | "remove"; key: string; }[]
```

Return a shallow copy of recent objective events.

### getPerformanceStatsSnapshot

```ts
getPerformanceStatsSnapshot(
  host: TelemetryAccessorHost,
): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Snapshot performance timings for evaluation and evolution steps.

### getTelemetryBuffer

```ts
getTelemetryBuffer(
  host: TelemetryAccessorHost,
): TelemetryEntry[]
```

Return the telemetry buffer, defaulting to an empty array when missing.

### LINEAGE_SNAPSHOT_DEFAULT_LIMIT

Default limit for lineage snapshots to avoid large payloads.

### TelemetryAccessorHost

Minimal host surface needed by telemetry accessors.

## neat/neat.telemetry.diversity.utils.ts

### applyFastModeDefaults

```ts
applyFastModeDefaults(
  telemetryContext: { _fastModeTuned?: boolean | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
): void
```

Apply fast-mode tuning to diversity sampling and novelty defaults.

Parameters:
- `telemetryContext` - - Context object storing fast-mode tuning flag.
- `telemetryOptions` - - Options with diversity and novelty settings.

### computeCompatibilityStats

```ts
computeCompatibilityStats(
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
  compatibilityDistance: ((a: TelemetryGenome, b: TelemetryGenome) => number) | undefined,
): { meanCompat: number; varCompat: number; }
```

Compute pairwise compatibility statistics via sampling.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.
- `compatibilityDistance` - - Optional compatibility distance function.

Returns: Mean and variance of sampled compatibilities.

### computeEntropyStats

```ts
computeEntropyStats(
  genomes: TelemetryGenome[],
  structuralEntropyFn: (genome: TelemetryGenome) => number,
): { meanEntropy: number; varEntropy: number; }
```

Compute structural entropy mean and variance across the population.

Parameters:
- `genomes` - - Population snapshot.
- `structuralEntropyFn` - - Function to compute entropy for a genome.

Returns: Mean and variance of entropy values.

### computeGraphletEntropy

```ts
computeGraphletEntropy(
  genomes: TelemetryGenome[],
  size: number,
  graphletSampleCount: number,
  rngFactoryFn: () => () => number,
): number
```

Sample graphlet motifs and compute entropy over their edge counts.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `graphletSampleCount` - - Number of graphlets to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Graphlet entropy value.

### countEnabledEdges

```ts
countEnabledEdges(
  genome: TelemetryGenome,
  selectedNodes: NodeLike[],
): number
```

Count enabled edges between the selected nodes in a genome.

Parameters:
- `genome` - - Genome with connections to inspect.
- `selectedNodes` - - Nodes forming the graphlet sample.

Returns: Edge count capped at 3.

### pickDistinctIndices

```ts
pickDistinctIndices(
  upperBound: number,
  count: number,
  rng: () => number,
): number[]
```

Pick a fixed number of distinct random indices.

Parameters:
- `upperBound` - - Exclusive upper bound for random indices.
- `count` - - Number of distinct indices to pick.
- `rng` - - RNG function returning values in [0,1).

Returns: Array of distinct indices.

## neat/neat.telemetry.selection.utils.ts

### getTelemetryCoreSnapshot

```ts
getTelemetryCoreSnapshot(
  sourceEntry: Record<string, unknown>,
  fields: TelemetryCoreFields,
): Partial<Record<string, unknown>>
```

Build a snapshot of the core telemetry fields present on the entry; does
not mutate the source entry.

Parameters:
- `sourceEntry` - - Source telemetry object.
- `fields` - - Core telemetry field keys to preserve.

Returns: Shallow snapshot of core fields that exist on the entry.

### mergeTelemetryCoreFields

```ts
mergeTelemetryCoreFields(
  sourceEntry: Record<string, unknown>,
  coreSnapshot: Partial<Record<string, unknown>>,
): Record<string, unknown>
```

Re-attach core fields to the filtered entry.
Mutates the entry so the caller keeps the original reference.

Parameters:
- `sourceEntry` - - Filtered telemetry entry to update.
- `coreSnapshot` - - Snapshot of core fields to ensure presence.

Returns: The same entry reference with core fields restored.

### safelyApplyTelemetrySelect

```ts
safelyApplyTelemetrySelect(
  telemetryContext: TContext,
  telemetryEntry: TelemetryEntry,
  applyTelemetrySelectFn: (this: TContext, entry: Record<string, unknown>) => Record<string, unknown>,
): void
```

Apply telemetry selection while swallowing any selection errors.

Parameters:
- `telemetryContext` - - Neat-like context with telemetry selection.
- `telemetryEntry` - - Entry to filter in place.
- `applyTelemetrySelectFn` - - Selection helper to invoke.

### stripUnselectedTelemetryKeys

```ts
stripUnselectedTelemetryKeys(
  sourceEntry: Record<string, unknown>,
  selection: Set<string>,
  fields: TelemetryCoreFields,
): Record<string, unknown>
```

Remove non-core keys that are not whitelisted by the selection set.
Mutates the provided entry in-place for efficiency.

Parameters:
- `sourceEntry` - - Telemetry entry being filtered.
- `selection` - - Whitelist of additional telemetry keys.
- `fields` - - Core telemetry field keys that must be preserved.

Returns: The same entry reference after filtering.

## neat/neat.telemetry.complexity.utils.ts

### applyComplexityStatsMonoObjective

```ts
applyComplexityStatsMonoObjective(
  telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  populationSnapshot: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Attach complexity stats for mono-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `entry` - - Telemetry entry to update.

### applyComplexityStatsMultiObjective

```ts
applyComplexityStatsMultiObjective(
  telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void
```

Attach complexity stats for multi-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### buildComplexityEntry

```ts
buildComplexityEntry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  meanCounts: { meanNodes: number; meanConns: number; },
  maxCounts: { maxNodes: number; maxConns: number; },
  meanEnabledRatio: number,
  growthValues: { growthNodes: number; growthConns: number; },
): { meanNodes: number; meanConns: number; maxNodes: number; maxConns: number; meanEnabledRatio: number; growthNodes: number; growthConns: number; budgetMaxNodes: number; budgetMaxConns: number; }
```

Build the complexity entry payload for multi-objective mode.

Parameters:
- `telemetryOptions` - - Options controlling complexity telemetry.
- `meanCounts` - - Mean node/connection counts.
- `maxCounts` - - Max node/connection counts.
- `meanEnabledRatio` - - Mean enabled ratio.
- `growthValues` - - Growth deltas.

Returns: Complexity entry payload.

### collectPopulationCounts

```ts
collectPopulationCounts(
  populationSnapshot: GenomeDetailed[],
): { nodeCounts: number[]; connectionCounts: number[]; }
```

Collect node and connection counts for the population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Node and connection counts arrays.

### computeAndStoreGrowthValues

```ts
computeAndStoreGrowthValues(
  context: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; },
  meanCounts: { meanNodes: number; meanConns: number; },
): { growthNodes: number; growthConns: number; }
```

Compute growth values and store the latest means on the context.

Parameters:
- `context` - - Neat-like context with previous mean values.
- `meanCounts` - - Current mean node/connection counts.

Returns: Growth values for nodes and connections.

### computeEnabledRatios

```ts
computeEnabledRatios(
  populationSnapshot: GenomeDetailed[],
): number[]
```

Compute enabled ratios per genome.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of enabled ratios.

### computeMaxCounts

```ts
computeMaxCounts(
  counts: { nodeCounts: number[]; connectionCounts: number[]; },
): { maxNodes: number; maxConns: number; }
```

Compute max node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Max node and connection counts.

### computeMeanCounts

```ts
computeMeanCounts(
  counts: { nodeCounts: number[]; connectionCounts: number[]; },
): { meanNodes: number; meanConns: number; }
```

Compute mean node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Mean node and connection counts.

### computeMeanEnabledRatio

```ts
computeMeanEnabledRatio(
  enabledRatios: number[],
): number
```

Compute mean of enabled ratios.

Parameters:
- `enabledRatios` - - Enabled ratios per genome.

Returns: Mean enabled ratio.

## neat/neat.telemetry.objectives.utils.ts

### applyHypervolumeTelemetry

```ts
applyHypervolumeTelemetry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  hyperVolumeProxy: number,
  entry: TelemetryEntryRecord,
): void
```

Attach hypervolume scalar when requested.

Parameters:
- `telemetryOptions` - - Options controlling telemetry fields.
- `hyperVolumeProxy` - - Hypervolume proxy value.
- `entry` - - Telemetry entry to update.

### applyObjectiveAges

```ts
applyObjectiveAges(
  telemetryContext: { _objectiveAges?: Map<string, number> | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply objective age snapshots to the entry.

Parameters:
- `telemetryContext` - - Neat-like context with objective ages.
- `entry` - - Telemetry entry to update.

### applyObjectiveEvents

```ts
applyObjectiveEvents(
  telemetryContext: { _pendingObjectiveAdds?: string[] | undefined; _pendingObjectiveRemoves?: string[] | undefined; _objectiveEvents?: ObjectiveEvent[] | undefined; },
  entry: TelemetryEntryRecord,
  generation: number,
): void
```

Apply and flush objective lifecycle events.

Parameters:
- `telemetryContext` - - Neat-like context holding objective events.
- `entry` - - Telemetry entry to update.
- `generation` - - Generation index for event records.

### applyObjectiveImportance

```ts
applyObjectiveImportance(
  telemetryContext: { _lastObjImportance?: ObjImportance | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply the most recent objective importance snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with objective importance.
- `entry` - - Telemetry entry to update.

### applyObjectivesSnapshot

```ts
applyObjectivesSnapshot(
  telemetryContext: { _getObjectives?: (() => { key: string; }[]) | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply objectives list snapshot (keys only).

Parameters:
- `telemetryContext` - - Neat-like context with objective provider.
- `entry` - - Telemetry entry to update.

### applySpeciesAllocation

```ts
applySpeciesAllocation(
  telemetryContext: { _lastOffspringAlloc?: SpeciesAlloc[] | undefined; },
  entry: TelemetryEntryRecord,
): void
```

Apply per-species offspring allocation snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with allocation snapshot.
- `entry` - - Telemetry entry to update.

### computeHyperVolumeProxy

```ts
computeHyperVolumeProxy(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
): number
```

Compute a hypervolume-like proxy for the Pareto front.

Parameters:
- `telemetryOptions` - - Options controlling complexity metric.
- `population` - - Population snapshot.

Returns: Hypervolume proxy value.

### computeParetoFrontSizes

```ts
computeParetoFrontSizes(
  population: GenomeDetailed[],
): number[]
```

Compute sizes of early Pareto fronts.

Parameters:
- `population` - - Population snapshot.

Returns: Array of front sizes (rank 0..4).

## neat/neat.multiobjective.fronts.utils.ts

### annotateGenomeRank

```ts
annotateGenomeRank(
  population: default[],
  genomeIndex: number,
  frontRank: number,
): void
```

Annotates a genome with its Pareto front rank.

Parameters:
- `population` - - Genome population.
- `genomeIndex` - - Index of the genome to annotate.
- `frontRank` - - Pareto front rank (0 = best front).

### appendFront

```ts
appendFront(
  paretoFronts: default[][],
  population: default[],
  currentFrontIndices: number[],
): void
```

Appends the current front (index list) as genome references to the
`paretoFronts` accumulator.

Parameters:
- `paretoFronts` - - Accumulator for Pareto fronts.
- `population` - - Genome population.
- `currentFrontIndices` - - Indices for the current front.

### buildNextFrontIndices

```ts
buildNextFrontIndices(
  population: default[],
  dominanceState: DominanceState,
  currentFrontIndices: number[],
  currentFrontRank: number,
): number[]
```

Builds the next front by applying rank annotations and dominance updates.

Parameters:
- `population` - - Genome population.
- `dominanceState` - - Dominance bookkeeping.
- `currentFrontIndices` - - Indices for the current front.
- `currentFrontRank` - - Rank to assign to the current front.

Returns: Indices for the next front.

### buildParetoFronts

```ts
buildParetoFronts(
  population: default[],
  dominanceState: DominanceState,
  maxFrontRankGuard: number,
): default[][]
```

Builds Pareto fronts from a precomputed dominance state.

This performs the “peeling” phase of fast non-dominated sorting:
- Start with the first front (all non-dominated genomes).
- For each front, reduce domination counts of the genomes it dominates.
- Any genome whose domination count becomes zero moves to the next front.

Side effects:
- Annotates each genome in `population` with `_moRank` (0 = best front).

Guard:
- Stops when `currentFrontRank > maxFrontRankGuard` to avoid pathological
  infinite/degenerate runs. If the guard triggers, the returned fronts may
  be incomplete.

Parameters:
- `population` - - Genome population (same ordering used by dominance
bookkeeping).
- `dominanceState` - - Dominance bookkeeping.
- `maxFrontRankGuard` - - Safety guard for ranking iterations.

Returns: Ordered Pareto fronts (rank order).

### collectNextFrontIndices

```ts
collectNextFrontIndices(
  dominanceState: DominanceState,
  genomeIndex: number,
  nextFrontIndices: number[],
): void
```

Collects indices that become non-dominated after removing the current
genome’s dominance influence.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `genomeIndex` - - Index of the current genome.
- `nextFrontIndices` - - Accumulator for the next front.

### incrementFrontRank

```ts
incrementFrontRank(
  currentFrontRank: number,
): number
```

Increments the front rank counter.

Parameters:
- `currentFrontRank` - - Current front rank.

Returns: Incremented front rank.

### MAX_PARETO_FRONT_RANK_GUARD

Maximum number of Pareto fronts to allow during ranking before aborting.

This is a defensive guard against pathological conditions (e.g., corrupted
dominance bookkeeping) that could otherwise cause long/infinite loops.

### shouldStopFrontRanking

```ts
shouldStopFrontRanking(
  currentFrontRank: number,
  maxFrontRankGuard: number,
): boolean
```

Determines whether ranking should stop due to a safety guard.

Parameters:
- `currentFrontRank` - - Current front rank after increment.
- `maxFrontRankGuard` - - Safety guard for ranking iterations.

Returns: `true` if ranking should stop.

## neat/neat.telemetry.performance.utils.ts

### applyPerformanceStats

```ts
applyPerformanceStats(
  telemetryContext: { _lastEvalDuration?: number | undefined; _lastEvolveDuration?: number | undefined; },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void
```

Attach performance stats when configured.

Parameters:
- `telemetryContext` - - Neat-like context with performance data.
- `telemetryOptions` - - Options controlling performance telemetry.
- `entry` - - Telemetry entry to update.

## neat/neat.evaluate.auto-distance.utils.ts

### applyAutoDistanceCoefficientTuning

```ts
applyAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  autoDistanceCoeffOptions: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; },
  connectionVariance: number,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `autoDistanceCoeffOptions` - - Tuning options.
- `connectionVariance` - - Variance of connection counts.

Returns: void.

### applyDistanceCoefficientDecrease

```ts
applyDistanceCoefficientDecrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number; },
  adjustRate: number,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `bounds` - - Min/max coefficients.
- `adjustRate` - - Adjustment rate.

Returns: void.

### applyDistanceCoefficientIncrease

```ts
applyDistanceCoefficientIncrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number; },
  adjustRate: number,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `bounds` - - Min/max coefficients.
- `adjustRate` - - Adjustment rate.

Returns: void.

### computeMean

```ts
computeMean(
  values: number[],
): number
```

Parameters:
- `values` - - Input values.

Returns: Mean of the values.

### computeVariance

```ts
computeVariance(
  values: number[],
  meanValue: number,
): number
```

Parameters:
- `values` - - Input values.
- `meanValue` - - Precomputed mean.

Returns: Variance of the values.

### getDistanceCoefficientBounds

```ts
getDistanceCoefficientBounds(
  autoDistanceCoeffOptions: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; },
): { minCoeff: number; maxCoeff: number; }
```

Parameters:
- `autoDistanceCoeffOptions` - - Tuning options.

Returns: Bounds for coefficients.

### initializeConnectionVarianceBootstrap

```ts
initializeConnectionVarianceBootstrap(
  controller: NeatControllerForEval,
  connectionVariance: number,
  bounds: { minCoeff: number; maxCoeff: number; },
  adjustRate: number,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `connectionVariance` - - Current connection variance.
- `bounds` - - Min/max coefficients.
- `adjustRate` - - Adjustment rate.

Returns: void.

### runAutoDistanceCoefficientTuning

```ts
runAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.multiobjective.archive.utils.ts

### archiveParetoFrontsIfEnabled

```ts
archiveParetoFrontsIfEnabled(
  neatInstance: NeatLikeWithMultiObjective,
  fronts: default[][],
): void
```

Archives a compact snapshot of the current Pareto fronts when
multi-objective mode is enabled.

This is intended for visualization/debugging:
- Stores only genome `_id` values (not full genomes).
- Keeps only the top {@link MAX_PARETO_ARCHIVE_FRONTS} fronts.
- Maintains a ring-buffer-like cap of {@link MAX_PARETO_ARCHIVE_LENGTH}
  snapshots by shifting the oldest entry.

Behavior note:
- This currently gates only on `neatInstance.options.multiObjective?.enabled`.
  If you want a separate archive toggle, ensure the caller configures
  `enabled` accordingly.

Parameters:
- `neatInstance` - - Neat instance.
- `fronts` - - Pareto fronts to archive.

### MAX_PARETO_ARCHIVE_FRONTS

Maximum number of top Pareto fronts to retain per archive snapshot.

Archival stores a compact representation (IDs only) for visualization or
debugging.

### MAX_PARETO_ARCHIVE_LENGTH

Maximum number of archive snapshots to retain.

When the archive exceeds this length, the oldest snapshot is dropped.

## neat/neat.multiobjective.metrics.utils.ts

### buildMultiObjectiveMetrics

```ts
buildMultiObjectiveMetrics(
  population: default[],
): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Build lightweight multi-objective metrics for each genome in the population.

### DEFAULT_MAX_PARETO_FRONTS

Default number of Pareto fronts returned by accessors.

### DEFAULT_PARETO_ARCHIVE_JSONL_MAX

Default slice size when exporting Pareto archive as JSONL.

### DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES

Default slice size when reading Pareto archive entries.

### exportParetoArchiveJsonl

```ts
exportParetoArchiveJsonl(
  archive: unknown[],
  maxEntries: number,
): string
```

Export a Pareto archive slice as JSON Lines.

### reconstructParetoFronts

```ts
reconstructParetoFronts(
  population: default[],
  maxFronts: number,
  isMultiObjectiveEnabled: boolean,
): default[][]
```

Reconstruct Pareto fronts from stored rank annotations.

### sliceParetoArchive

```ts
sliceParetoArchive(
  archive: T[],
  maxEntries: number,
): T[]
```

Return the most recent Pareto archive entries up to the provided limit.

## neat/neat.evaluate.entropy-compat.utils.ts

### computeNextCompatibilityThreshold

```ts
computeNextCompatibilityThreshold(
  entropyCompatOptions: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; },
  meanEntropy: number,
  currentThreshold: number,
): number
```

Parameters:
- `entropyCompatOptions` - - Tuning options.
- `meanEntropy` - - Current mean entropy.
- `currentThreshold` - - Current compatibility threshold.

Returns: Next compatibility threshold.

### runEntropyCompatibilityTuning

```ts
runEntropyCompatibilityTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.multiobjective.category.utils.ts

### adaptDominanceEpsilon

```ts
adaptDominanceEpsilon(
  internal: NeatControllerForEvolution,
  paretoFronts: GenomeWithMetadata[][],
  config: { targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; },
): void
```

Adapt dominance epsilon based on Pareto front size.

Parameters:
- `internal` - - NEAT controller instance.
- `paretoFronts` - - Non-dominated fronts.
- `config` - - Epsilon tuning constants.

Returns: void.

### computeCrowdingDistances

```ts
computeCrowdingDistances(
  internal: NeatControllerForEvolution,
  populationSnapshot: GenomeWithMetadata[],
  paretoFronts: GenomeWithMetadata[][],
  objectives: ObjectiveDescriptor[],
): number[]
```

Compute crowding distances for multi-objective fronts.

Parameters:
- `internal` - - NEAT controller instance.
- `populationSnapshot` - - Current population reference.
- `paretoFronts` - - Non-dominated fronts.
- `objectives` - - Active objectives.

Returns: crowding distances aligned with population order.

### processMultiObjective

```ts
processMultiObjective(
  internal: NeatControllerForEvolution,
  config: { paretoArchiveMax: number; targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; pruneWindowDefault: number; pruneRangeEpsDefault: number; },
): void
```

Run multi-objective ranking, crowding distance, and archives.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Multi-objective tuning constants.

Returns: void.

### pruneInactiveObjectives

```ts
pruneInactiveObjectives(
  internal: NeatControllerForEvolution,
  config: { pruneWindowDefault: number; pruneRangeEpsDefault: number; },
): void
```

Prune objectives that have collapsed ranges over a window.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Pruning constants.

Returns: void.

### recordParetoArchives

```ts
recordParetoArchives(
  internal: NeatControllerForEvolution,
  paretoFronts: GenomeWithMetadata[][],
  objectives: ObjectiveDescriptor[],
  archiveMax: number,
): void
```

Record Pareto front archives for telemetry.

Parameters:
- `internal` - - NEAT controller instance.
- `paretoFronts` - - Non-dominated fronts.
- `objectives` - - Active objectives.
- `archiveMax` - - Maximum archive size.

Returns: void.

### sortPopulationByPareto

```ts
sortPopulationByPareto(
  internal: NeatControllerForEvolution,
  populationSnapshot: GenomeWithMetadata[],
  crowdingDistances: number[],
): void
```

Sort population by Pareto rank and crowding distance.

Parameters:
- `internal` - - NEAT controller instance.
- `populationSnapshot` - - Current population reference.
- `crowdingDistances` - - Crowding distances aligned with population order.

Returns: void.

## neat/neat.multiobjective.crowding.utils.ts

### accumulateCrowdingForObjective

```ts
accumulateCrowdingForObjective(
  sortedFront: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): void
```

Accumulates crowding distance contributions for a single objective.

Pre-conditions / expectations:
- `sortedFront` must be sorted ascending by the selected objective.
- {@link initializeCrowding} has already set `_moCrowd = 0` for the front.
- {@link markBoundaryCrowding} is typically called before this to set the
  boundary genomes to `Infinity`.

Edge cases:
- If the front has fewer than 2 genomes, this is a no-op.
- If the objective range is `0`, a range of `1` is used (see
  {@link resolveObjectiveRange}).

Parameters:
- `sortedFront` - - Front sorted by objective.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

### accumulateInteriorCrowding

```ts
accumulateInteriorCrowding(
  sortedFront: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
  valueRange: number,
): void
```

Accumulates crowding deltas for the interior genomes of a sorted front.

Interior genomes receive a normalized spacing delta:
`delta = (nextValue - previousValue) / valueRange`.

Parameters:
- `sortedFront` - - Front sorted by objective.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.
- `valueRange` - - Normalized objective range.

### applyCrowdingDelta

```ts
applyCrowdingDelta(
  currentGenome: NetworkWithMOAnnotations,
  previousValue: number,
  nextValue: number,
  valueRange: number,
): void
```

Applies a normalized crowding-distance delta to a genome.

If the genome’s crowding distance is `Infinity`, it will remain `Infinity`.
This helper only updates when `_moCrowd` is initialized.

Parameters:
- `currentGenome` - - Genome to update.
- `previousValue` - - Objective value of previous genome.
- `nextValue` - - Objective value of next genome.
- `valueRange` - - Normalized objective range.

### applyCrowdingForObjective

```ts
applyCrowdingForObjective(
  front: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): void
```

Applies crowding-distance accumulation for a single objective within a
single front.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

### assignCrowdingDistances

```ts
assignCrowdingDistances(
  fronts: default[][],
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  population: default[],
): void
```

Assigns crowding-distance annotations for each Pareto front.

This implements the crowding distance component of NSGA-II selection. Each
genome in each front receives a `_moCrowd` value representing how isolated
it is in objective space within its front.

Notes:
- This function sorts each front by each objective (ascending raw values).
  Objective direction (min vs max) does not affect the computed spacing
  magnitude; extrema are treated as boundaries either way.
- Empty fronts are skipped.

Side effects:
- Writes `_moCrowd` on each genome in each front.

Parameters:
- `fronts` - - Pareto fronts.
- `valuesMatrixInput` - - Values matrix.
- `descriptors` - - Objective descriptors (provides objective count).
- `population` - - Population to resolve indices.

### assignCrowdingForFront

```ts
assignCrowdingForFront(
  front: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndices: number[],
): void
```

Assigns crowding distances for a single front across all objectives.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndices` - - Objective indices to process.

### buildGenomeIndexByReference

```ts
buildGenomeIndexByReference(
  population: default[],
): Map<default, number>
```

Builds a stable mapping from genome object references to their population
index.

This relies on object identity (reference equality), not structural
equality. It is used to resolve objective values from a values matrix when
working with reordered views (e.g., sorted fronts).

Parameters:
- `population` - - Genomes in population order.

Returns: Map from genome references to their index.

### buildInteriorIndexRange

```ts
buildInteriorIndexRange(
  frontLength: number,
): number[]
```

Builds the index range for interior genomes of a front.

Boundary genomes are excluded because their crowding distance is treated as
infinite.

Parameters:
- `frontLength` - - Length of the sorted front.

Returns: Interior indices excluding boundary genomes.

### buildObjectiveIndexRange

```ts
buildObjectiveIndexRange(
  objectiveCount: number,
): number[]
```

Builds a stable objective index range.

Parameters:
- `objectiveCount` - - Number of objectives.

Returns: Objective indices `0..objectiveCount-1`.

### buildSortedFrontByObjective

```ts
buildSortedFrontByObjective(
  front: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): default[]
```

Builds a copy of the front sorted by the specified objective.

Sorting is ascending by the raw objective value. This ordering is used for
computing neighbor spacing in objective space.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

Returns: Front sorted by objective value.

### compareObjectiveValuesForCrowding

```ts
compareObjectiveValuesForCrowding(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
  leftGenome: default,
  rightGenome: default,
): number
```

Comparator used to sort genomes by a specific objective value.

Parameters:
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.
- `leftGenome` - - Left genome.
- `rightGenome` - - Right genome.

Returns: Numeric sort comparison value (ascending).

### initializeCrowding

```ts
initializeCrowding(
  front: default[],
): void
```

Initializes crowding-distance annotations for a front.

This sets each genome’s `_moCrowd` to `0`. Later steps accumulate per-
objective spacing deltas.

Parameters:
- `front` - - Pareto front.

### markBoundaryCrowding

```ts
markBoundaryCrowding(
  sortedFront: default[],
): void
```

Marks the boundary genomes of a sorted front as infinitely crowded.

In NSGA-II style crowding distance, boundary solutions (extremes for the
objective) are assigned an infinite crowding distance to ensure they are
always preferred when ranks tie.

Parameters:
- `sortedFront` - - Front sorted by the current objective.

### resolveBoundaryGenomes

```ts
resolveBoundaryGenomes(
  sortedFront: default[],
): { firstGenome: default; lastGenome: default; } | null
```

Resolves the boundary (first/last) genomes for a sorted front.

Parameters:
- `sortedFront` - - Front sorted by objective.

Returns: Boundary genomes, or `null` if the front is empty.

### resolveGenomeIndex

```ts
resolveGenomeIndex(
  genomeIndexByReference: Map<default, number>,
  genomeItem: default,
): number
```

Resolves a genome’s index using a reference-based map.

Parameters:
- `genomeIndexByReference` - - Lookup map created by
 *  {@link buildGenomeIndexByReference} .
- `genomeItem` - - Genome to resolve.

Returns: The population index of the genome.

### resolveNeighborPair

```ts
resolveNeighborPair(
  sortedFront: default[],
  sortedIndex: number,
): { previousGenome: default; nextGenome: default; }
```

Resolves the neighbor genomes for an interior element of a sorted front.

Parameters:
- `sortedFront` - - Front sorted by objective.
- `sortedIndex` - - Current index in sorted front.

Returns: Previous and next neighbor genomes.

### resolveObjectiveRange

```ts
resolveObjectiveRange(
  minValue: number,
  maxValue: number,
): number
```

Resolves a non-zero objective range used to normalize crowding deltas.

If all genomes have the same objective value, the raw range is `0`. This
returns `1` in that case to avoid division by zero while still producing a
well-defined crowding delta of `0`.

Parameters:
- `minValue` - - Minimum objective value.
- `maxValue` - - Maximum objective value.

Returns: Normalized range with a non-zero floor.

### resolveObjectiveRangeFromBoundaries

```ts
resolveObjectiveRangeFromBoundaries(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  boundaryGenomes: { firstGenome: default; lastGenome: default; },
  objectiveIndex: number,
): number
```

Resolves the normalized objective range for a front from its boundary
genomes.

Because `sortedFront` is sorted by objective, the first and last genomes are
the extrema used for range normalization.

Parameters:
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `boundaryGenomes` - - Boundary genomes for the front.
- `objectiveIndex` - - Objective column index.

Returns: Normalized value range for the objective.

### resolveObjectiveValue

```ts
resolveObjectiveValue(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  genomeItem: default,
  objectiveIndex: number,
): number
```

Resolves an objective value for a genome from a values matrix.

This is a convenience helper for working with sorted/reordered views of the
population while keeping objective values in a dense matrix.

Parameters:
- `valuesMatrixInput` - - Values matrix indexed by population index.
- `genomeIndexByReference` - - Lookup map from genome reference to index.
- `genomeItem` - - Genome to resolve.
- `objectiveIndex` - - Objective column index.

Returns: The objective value for the genome.

### shouldSkipCrowdingFront

```ts
shouldSkipCrowdingFront(
  front: default[],
): boolean
```

Determines whether crowding-distance processing should be skipped for a
front.

Parameters:
- `front` - - Pareto front.

Returns: `true` if the front should be skipped.

## neat/neat.evaluate.entropy-sharing.utils.ts

### computeNextSharingSigma

```ts
computeNextSharingSigma(
  entropySharingOptions: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; },
  currentVarEntropy: number,
  currentSigma: number,
): number
```

Parameters:
- `entropySharingOptions` - - Tuning options.
- `currentVarEntropy` - - Current variance of entropy.
- `currentSigma` - - Current sigma value.

Returns: Next sigma value.

### ensureDiversityStatsContainer

```ts
ensureDiversityStatsContainer(
  controller: NeatControllerForEval,
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.

Returns: void.

### runEntropySharingTuning

```ts
runEntropySharingTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.multiobjective.dominance.utils.ts

### applyPairwiseDominance

```ts
applyPairwiseDominance(
  dominanceState: DominanceState,
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  candidateIndex: number,
  opponentIndex: number,
): void
```

Applies a single pairwise dominance update between candidate and opponent.

If the candidate dominates the opponent, the opponent index is appended to
`dominatedIndicesByIndex[candidateIndex]`. If the candidate is dominated by
the opponent, `dominationCounts[candidateIndex]` is incremented.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `valuesMatrixInput` - - Matrix of objective values.
- `descriptors` - - Objective descriptors.
- `candidateIndex` - - Candidate genome index.
- `opponentIndex` - - Opponent genome index.

### buildDominanceState

```ts
buildDominanceState(
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
): DominanceState
```

Builds dominance bookkeeping structures used by fast non-dominated sorting.

This computes (pairwise):
- `dominationCounts[i]`: how many genomes dominate genome `i`.
- `dominatedIndicesByIndex[i]`: which genomes are dominated by genome `i`.
- `firstFrontIndices`: genomes with `dominationCounts[i] === 0`.

Complexity:
- Time: $O(n^2 \cdot m)$ where $n$ is population size and $m$ is objective
  count.
- Space: $O(n^2)$ in the worst case for the dominated adjacency lists.

Assumptions:
- Each row in `valuesMatrixInput` is a vector aligned with `descriptors`.
- Genome ordering in later steps is expected to match the matrix ordering.

Parameters:
- `valuesMatrixInput` - - Matrix of objective values (row = genome).
- `descriptors` - - Objective descriptors (direction semantics).

Returns: Dominance bookkeeping structures for ranking.

### buildIndexRange

```ts
buildIndexRange(
  populationSize: number,
): number[]
```

Builds a stable index range for iterating the population.

Parameters:
- `populationSize` - - Number of genomes.

Returns: Array of indices `0..populationSize-1`.

### compareObjectiveValues

```ts
compareObjectiveValues(
  direction: "max" | "min",
  candidateValue: number,
  opponentValue: number,
): { isDominated: boolean; isStrictlyBetter: boolean; }
```

Compares a candidate and opponent value for a single objective.

This does not compute full Pareto dominance; it returns per-objective flags
used by the vector-level dominance check.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: Comparison flags for this objective.

### createEmptyDominanceState

```ts
createEmptyDominanceState(
  populationSize: number,
): DominanceState
```

Creates an empty dominance state container sized to the population.

Parameters:
- `populationSize` - - Number of genomes.

Returns: An initialized dominance state with zeroed counts.

### DominanceState

Dominance bookkeeping structures for fast non-dominated sorting.

These structures are typically produced once per generation (from the values
matrix) and then consumed to build Pareto fronts.

### isCandidateDominatedByObjective

```ts
isCandidateDominatedByObjective(
  direction: "max" | "min",
  candidateValue: number,
  opponentValue: number,
): boolean
```

Checks if the candidate is worse than the opponent for a single objective.

For dominance, being worse on any objective makes the candidate unable to
dominate the opponent.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: `true` if the candidate is dominated for this objective.

### isCandidateStrictlyBetterForObjective

```ts
isCandidateStrictlyBetterForObjective(
  direction: "max" | "min",
  candidateValue: number,
  opponentValue: number,
): boolean
```

Checks if the candidate is strictly better than the opponent for a single
objective.

Strict improvement in at least one objective is required for Pareto
dominance when the candidate is not worse in any objective.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: `true` if the candidate is strictly better for this objective.

### isNonDominatedCandidate

```ts
isNonDominatedCandidate(
  dominanceState: DominanceState,
  candidateIndex: number,
): boolean
```

Determines whether a candidate has zero domination count.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `candidateIndex` - - Candidate genome index.

Returns: `true` if the candidate is currently non-dominated.

### resolveDominanceOutcome

```ts
resolveDominanceOutcome(
  candidateVector: number[],
  opponentVector: number[],
  descriptors: ObjectiveDescriptor[],
): "dominates" | "dominated" | "indifferent"
```

Resolves dominance outcome between two objective vectors.

Outcome meanings:
- `'dominates'`: candidate dominates opponent.
- `'dominated'`: candidate is dominated by opponent.
- `'indifferent'`: neither dominates the other.

Parameters:
- `candidateVector` - - Candidate objective values.
- `opponentVector` - - Opponent objective values.
- `descriptors` - - Objective descriptors.

Returns: Dominance outcome between candidate and opponent.

### resolveObjectiveDirection

```ts
resolveObjectiveDirection(
  descriptors: ObjectiveDescriptor[],
  objectiveIndex: number,
): "max" | "min"
```

Resolves the objective direction for a given objective index.

If a descriptor omits `direction`, it is treated as maximization.

Parameters:
- `descriptors` - - Objective descriptors.
- `objectiveIndex` - - Objective index.

Returns: Normalized objective direction.

### shouldSkipSelfComparison

```ts
shouldSkipSelfComparison(
  candidateIndex: number,
  opponentIndex: number,
): boolean
```

Determines whether a pairwise comparison should be skipped.

Currently this skips only self-comparisons.

Parameters:
- `candidateIndex` - - Candidate genome index.
- `opponentIndex` - - Opponent genome index.

Returns: `true` if the pair should be skipped.

### updateDominanceForCandidate

```ts
updateDominanceForCandidate(
  dominanceState: DominanceState,
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  candidateIndex: number,
  candidateIndices: number[],
): void
```

Updates dominance bookkeeping for a candidate against all opponents.

This iterates every opponent index and applies a pairwise dominance update.
Self-comparisons are ignored.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `valuesMatrixInput` - - Matrix of objective values.
- `descriptors` - - Objective descriptors.
- `candidateIndex` - - Candidate genome index.
- `candidateIndices` - - Indices to compare against.

### updateStrictImprovement

```ts
updateStrictImprovement(
  hasStrictImprovement: boolean,
  isStrictlyBetter: boolean,
): boolean
```

Accumulates whether the candidate has any strict improvement across
objectives.

Parameters:
- `hasStrictImprovement` - - Current strict-improvement flag.
- `isStrictlyBetter` - - Whether the candidate strictly improves on the
current objective.

Returns: Updated strict-improvement flag.

### vectorDominates

```ts
vectorDominates(
  valuesA: number[],
  valuesB: number[],
  descriptors: ObjectiveDescriptor[],
): boolean
```

Determines whether vector A Pareto-dominates vector B.

A dominates B iff:
- A is **no worse** than B in every objective (respecting each objective’s
  direction: maximize/minimize), and
- A is **strictly better** in at least one objective.

Assumptions:
- `valuesA` and `valuesB` are aligned and have the same length.
- `descriptors` provides a descriptor for each objective index.
- If a descriptor has no `direction`, it defaults to `'max'`.

Parameters:
- `valuesA` - - Objective values for candidate A.
- `valuesB` - - Objective values for candidate B.
- `descriptors` - - Objective descriptors defining direction semantics.

Returns: `true` if A dominates B; otherwise `false`.

Example:

```ts
// Maximize accuracy, minimize latency:
vectorDominates([0.9, 120], [0.9, 150], [
  { accessor: () => 0, direction: 'max' },
  { accessor: () => 0, direction: 'min' },
]);
// => true (equal accuracy, lower latency)
```

## neat/neat.multiobjective.objectives.utils.ts

### buildGenomeValues

```ts
buildGenomeValues(
  genomeItem: default,
  descriptors: ObjectiveDescriptor[],
): number[]
```

Builds an objective vector for a single genome.

The resulting array order matches the `descriptors` order exactly.
Each component is read via {@link readObjectiveValue} so individual
objective accessors are fault-tolerant.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptors` - - Objective descriptors (vector schema).

Returns: Objective value vector (length equals `descriptors.length`).

### buildValuesMatrix

```ts
buildValuesMatrix(
  population: default[],
  descriptors: ObjectiveDescriptor[],
): number[][]
```

Builds a population-wide objective value matrix.

The resulting matrix is indexed as `[genomeIndex][objectiveIndex]` where
`genomeIndex` matches the input `population` order.

Parameters:
- `population` - - Genomes to evaluate (population order is preserved).
- `descriptors` - - Objective descriptors (column schema).

Returns: Objective values matrix.

### readObjectiveValue

```ts
readObjectiveValue(
  genomeItem: default,
  descriptor: ObjectiveDescriptor,
): number
```

Safely reads a single objective value for a given genome.

This wraps the descriptor `accessor` in a `try/catch` so that a buggy
objective function cannot crash multi-objective ranking.

Notes:
- If the accessor throws, this returns `0` (a neutral-ish fallback).
- Callers should prefer to surface accessor errors during development;
  this helper is intentionally defensive for long-running training loops.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptor` - - Objective descriptor providing an accessor.

Returns: Numeric objective value; `0` if the accessor throws.

Example:

```ts
const score = readObjectiveValue(genome, { accessor: (g) => g.score ?? 0 });
```

## neat/neat.adaptive.minimal-criterion.utils.ts

### applyRejection

```ts
applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void
```

Zero scores below the final threshold.

Parameters:
- `engine` - - NEAT engine instance.
- `threshold` - - Final MC threshold.

### collectScores

```ts
collectScores(
  engine: NeatLikeWithAdaptive,
): number[]
```

Collect population scores into a snapshot array.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Array of scores (missing scores treated as 0).

### computeAcceptance

```ts
computeAcceptance(
  scores: number[],
  threshold: number,
): number
```

Compute acceptance metrics for the current threshold.

Parameters:
- `scores` - - Population score snapshot.
- `threshold` - - Current MC threshold.

Returns: Acceptance proportion.

### initializeThreshold

```ts
initializeThreshold(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): void
```

Initialize MC threshold if missing.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Minimal-criterion adaptive configuration.

### resolveTargetSettings

```ts
resolveTargetSettings(
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): { targetAcceptance: number; adjustRate: number; }
```

Resolve target acceptance and adjust rate settings.

Parameters:
- `config` - - Minimal-criterion adaptive configuration.

Returns: Target settings.

### updateThreshold

```ts
updateThreshold(
  engine: NeatLikeWithAdaptive,
  acceptance: number,
  tuning: { targetAcceptance: number; adjustRate: number; },
): void
```

Update the MC threshold based on acceptance proportion.

Parameters:
- `engine` - - NEAT engine instance.
- `acceptance` - - Observed acceptance proportion.
- `tuning` - - Target acceptance and adjustment settings.

## neat/neat.adaptive.ancestor-uniqueness.utils.ts

### applyEpsilonAdjustment

```ts
applyEpsilonAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply dominance-epsilon adjustments when configured.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### applyLineagePressureAdjustment

```ts
applyLineagePressureAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
): void
```

Apply lineage pressure strength adjustments.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.

### applyUniquenessAdjustment

```ts
applyUniquenessAdjustment(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply an adjustment for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### ensureLineagePressureState

```ts
ensureLineagePressureState(
  engine: NeatLikeWithAdaptive,
): { enabled?: boolean | undefined; mode?: string | undefined; strength?: number | undefined; }
```

Ensure lineage pressure state is available.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Lineage pressure configuration object.

### extractAncestorUniqueness

```ts
extractAncestorUniqueness(
  engine: NeatLikeWithAdaptive,
): number | undefined
```

Extract the latest ancestor-uniqueness metric from telemetry.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Ancestor uniqueness value or undefined when missing.

### isCooldownSatisfied

```ts
isCooldownSatisfied(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): boolean
```

Determine whether the cooldown window has elapsed.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: True when adjustment is allowed.

### recordAdjustment

```ts
recordAdjustment(
  engine: NeatLikeWithAdaptive,
): void
```

Record the generation when an adjustment is applied.

Parameters:
- `engine` - - NEAT engine instance.

### resolveAdjustmentMagnitude

```ts
resolveAdjustmentMagnitude(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): number
```

Resolve adjustment magnitude for nudging controlled parameters.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Adjustment magnitude.

### resolveUniquenessThresholds

```ts
resolveUniquenessThresholds(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): { lowThreshold: number; highThreshold: number; }
```

Resolve thresholds for ancestor-uniqueness decisions.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Threshold bounds.
