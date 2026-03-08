# neat

## neat/neat.types.ts

### neat.types

Speciation options for the NEAT speciation controller.

Extends {@link NeatOptions} with speciation-specific configuration used by:
- Compatibility-threshold based species assignment
- Adaptive threshold controllers (PID-like)
- Species allocation telemetry (history snapshots)

### AnyObj

Shared lightweight structural types for modular NEAT components.

These are deliberately kept small & structural (duck-typed) so that helper
modules can interoperate without importing the concrete (heavier) `Neat`
class, avoiding circular references while the codebase is being
progressively extracted / refactored.

Guidelines:
- Prefer adding narrowly scoped interfaces instead of widening existing ones.
- Avoid leaking implementation details; keep contracts minimal.
- Feature‑detect optional telemetry fields – they may be omitted to save cost.

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

### neat.harness.types

Type helpers for test harnesses exercising Neat lineage behaviour.

### LineageTrackedNetwork

Network subtype that surfaces lineage metadata fields for assertions.

### NeatLineageHarness

Narrow Neat surface exposing lineage helper methods used in tests.

### PhasedComplexityHarness

Minimal surface exposing phased complexity internals for testing.

## neat/neat.mutation.types.ts

### neat.mutation.types

Type definitions for NEAT mutation operations.
Extracted to avoid circular dependencies.

### ConnectionWithMetadata

Runtime interface for a connection within a genome.

### GenomeWithMetadata

Type definitions for NEAT mutation operations.
Extracted to avoid circular dependencies.

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

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost) => number | undefined`

Export the current RNG state for persistence.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when not set.

### getOrCreateRng

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost) => () => number`

Return a cached RNG or create a deterministic xorshift RNG when absent.

The helper respects a user-provided RNG at `options.rng` when present.
Otherwise it seeds a xorshift32 RNG using the current time and population
size, guarding against the invalid zero seed.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

### importRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost, state: string | number | undefined) => void`

Alias for restoring RNG state kept for compatibility with prior surface.

### restoreRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost, state: string | number | undefined) => void`

Restore a previously captured RNG state.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### RNG_DEFAULT_SEED_FALLBACK

### RNG_NORMALIZATION_DIVISOR

### RNG_POPULATION_OFFSET

### RNG_SHIFT_LEFT_PRIMARY

### RNG_SHIFT_LEFT_SECONDARY

### RNG_SHIFT_RIGHT_PRIMARY

### RNG_TIME_SCRAMBLE_CONSTANT

### RngHost

Minimal host surface required by the RNG utilities.

### sampleRandomSequence

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost, sampleCount: number) => number[]`

Produce a sequence of random samples using the host RNG.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

### snapshotRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost) => number | undefined`

Snapshot the current RNG state for deterministic replay.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.

## neat/neat.cache.ts

### invalidateGenomeCaches

`(genomeCandidate: unknown) => void`

Invalidate per-genome caches used across compatibility and forward-pass logic.

Parameters:
- `genomeCandidate` - - Genome object whose caches should be cleared.

## neat/neat.compat.ts

### _compatibilityDistance

`(genomeA: import("C:/NeatapticTS/src/neat/neat.compat.utils").GenomeLike, genomeB: import("C:/NeatapticTS/src/neat/neat.compat.utils").GenomeLike) => number`

### _fallbackInnov

`(connection: import("C:/NeatapticTS/src/neat/neat.compat.utils").ConnectionLike) => number`

## neat/neat.evolve.ts

### evolve

`() => Promise<import("C:/NeatapticTS/src/architecture/network").default>`

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

### EVOLVE_AUTO_COMPAT_MAX_COEFF

### EVOLVE_AUTO_COMPAT_MIN_COEFF

### EVOLVE_AUTO_COMPAT_RANDOM_SCALE

### EVOLVE_AUTO_COMPAT_TARGET_MIN

### EVOLVE_AUTO_ENTROPY_ADD_AT

### EVOLVE_CROSS_SPECIES_GUARD_LIMIT

### EVOLVE_DEFAULT_EPSILON_ADJUST

### EVOLVE_DEFAULT_EPSILON_COOLDOWN

### EVOLVE_DEFAULT_EPSILON_MAX

### EVOLVE_DEFAULT_EPSILON_MIN

### EVOLVE_GLOBAL_STAGNATION_REPLACE_FRACTION

### EVOLVE_MIN_OFFSPRING_DEFAULT

### EVOLVE_OLD_MULTIPLIER_DEFAULT

### EVOLVE_OLD_THRESHOLD_DEFAULT

### EVOLVE_PARETO_ARCHIVE_MAX

### EVOLVE_PRUNE_RANGE_EPS_DEFAULT

### EVOLVE_PRUNE_WINDOW_DEFAULT

### EVOLVE_REENABLE_DELTA_SCALE

### EVOLVE_REENABLE_MAX

### EVOLVE_REENABLE_MIN

### EVOLVE_REENABLE_MIN_SAMPLES

### EVOLVE_REENABLE_TARGET

### EVOLVE_SPECIES_HISTORY_MAX

### EVOLVE_SURVIVAL_THRESHOLD_DEFAULT

### EVOLVE_TARGET_FRONT_LOWER_RATIO

### EVOLVE_TARGET_FRONT_MIN

### EVOLVE_TARGET_FRONT_UPPER_RATIO

### EVOLVE_YOUNG_MULTIPLIER_DEFAULT

### EVOLVE_YOUNG_THRESHOLD_DEFAULT

## neat/neat.export.ts

### exportPopulation

`() => import("C:/NeatapticTS/src/neat/neat.export").GenomeJSON[]`

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

`() => import("C:/NeatapticTS/src/neat/neat.export").NeatStateJSON`

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

`(neatJSON: import("C:/NeatapticTS/src/neat/neat.export").NeatMetaJSON, fitnessFunction: (network: GenomeWithSerialization) => number | Promise<number>) => NeatControllerForExport`

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

`(populationJSON: import("C:/NeatapticTS/src/neat/neat.export").GenomeJSON[]) => Promise<void>`

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

`(stateBundle: import("C:/NeatapticTS/src/neat/neat.export").NeatStateJSON, fitnessFunction: (network: GenomeWithSerialization) => number | Promise<number>) => Promise<NeatControllerForExport>`

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
 *
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

`() => import("C:/NeatapticTS/src/neat/neat.export").NeatMetaJSON`

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

### addGenome

`(genome: GenomeWithMetadata, parents: number[] | undefined) => void`

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

### createPool

`(seedNetwork: GenomeWithMetadata | null) => void`

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

### GenomeWithMetadata

Genome with NEAT-specific metadata and methods.

### MutationMethod

Mutation method with optional name.

### NeatControllerForHelpers

NEAT controller interface for helper functions.

### spawnFromParent

`(parentGenome: GenomeWithMetadata, mutateCount: number) => Promise<GenomeWithMetadata>`

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

## neat/neat.lineage.ts

### neat.lineage

Lineage / ancestry analysis helpers for NEAT populations.

These utilities were migrated from the historical implementation inside `src/neat.ts`
to keep core NEAT logic lean while still exposing educational metrics for users who
want to introspect evolutionary diversity.

Glossary:
 - Genome: An individual network encoding (has a unique `_id` and optional `_parents`).
 - Ancestor Window: A shallow breadth‑first window (default depth = 4) over the lineage graph.
 - Jaccard Distance: 1 - |A ∩ B| / |A ∪ B|, measuring dissimilarity between two sets.

### buildAnc

`(genome: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike) => Set<number>`

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

### computeAncestorUniqueness

`() => number`

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

### GenomeLike

Lineage / ancestry helper utilities for NEAT populations.

This module centralizes helper logic used by the public lineage APIs to keep
the main entry file small and orchestration-focused.

### NeatLineageContext

Expected `this` context for lineage helpers (a subset of the NEAT instance).

## neat/neat.pruning.ts

### applyAdaptivePruning

`() => void`

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

`() => void`

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

`() => import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[]`

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

`() => { id: number; size: number; bestScore: number; lastImproved: number; }[]`

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

### ANNEAL_PROGRESS_MAX

### applyAdaptiveMutation

`() => void`

### applyAncestorUniqAdaptive

`() => void`

### applyComplexityBudget

`() => void`

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

### applyMinimalCriterionAdaptive

`() => void`

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

### applyOperatorAdaptation

`() => void`

### applyPhasedComplexity

`() => void`

Toggle phased complexity mode between 'complexify' and 'simplify'.

Phased complexity supports alternating periods where the algorithm
is encouraged to grow (complexify) or shrink (simplify) network
structures. This can help escape local minima or reduce bloat.

The current phase and its start generation are stored on `this` as
`_phase` and `_phaseStartGeneration` so the state persists across
generations.

Returns: Mutates `this._phase` and `this._phaseStartGeneration`.

### DEFAULT_ADAPT_EVERY

### DEFAULT_ANCESTOR_UNIQ_ADJUST

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

### DEFAULT_INITIAL_MUTATION_RATE

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

### DEFAULT_MAX_MUTATION_AMOUNT

### DEFAULT_MAX_MUTATION_RATE

### DEFAULT_MIN_MUTATION_AMOUNT

### DEFAULT_MIN_MUTATION_RATE

### DEFAULT_MUTATION_AMOUNT

### DEFAULT_MUTATION_AMOUNT_SIGMA

### DEFAULT_MUTATION_SIGMA

### EXPLORE_LOW_DECREASE_MULTIPLIER

### EXPLORE_LOW_INCREASE_MULTIPLIER

### HALF_INDEX_DIVISOR

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

### MUTATION_SIGMA_SCALE

### MUTATION_STRATEGY_ANNEAL

### MUTATION_STRATEGY_EXPLORE_LOW

### MUTATION_STRATEGY_TWO_TIER

### NeatLikeWithAdaptive

Minimal interface for NEAT instances with adaptive features.
Exported for use in tests and type-safe function calls.

### RNG_CENTER_OFFSET

### RNG_SPREAD_MULTIPLIER

## neat/neat.evaluate.ts

### AUTO_COEFF_ADJUST_DEFAULT

### AUTO_COEFF_MAX_DEFAULT

### AUTO_COEFF_MIN_DEFAULT

### COMPAT_MAX_THRESHOLD_DEFAULT

### COMPAT_MIN_THRESHOLD_DEFAULT

### COMPAT_THRESHOLD_DEFAULT

### DISTANCE_COEFF_DEFAULT

### ENTROPY_ADJUST_DEFAULT

### ENTROPY_DEADBAND_DEFAULT

### ENTROPY_TARGET_DEFAULT

### ENTROPY_VAR_ADJUST_DEFAULT

### ENTROPY_VAR_HIGH_BAND

### ENTROPY_VAR_LOW_BAND

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

### ENTROPY_VAR_TARGET_DEFAULT

### evaluate

`() => Promise<void>`

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

### NOVELTY_DEFAULT_BLEND

### NOVELTY_DEFAULT_NEIGHBORS

### VARIANCE_DECREASE_THRESHOLD

### VARIANCE_INCREASE_THRESHOLD

## neat/neat.mutation.ts

### DEFAULT_CONNECTION_WEIGHT

### DEFAULT_GENE_ID

### DEFAULT_INNOVATION_ID

### ensureMinHiddenNodes

`(network: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, multiplierOverride: number | undefined) => Promise<void>`

Ensure the network has a minimum number of hidden nodes and connectivity.

### ensureNoDeadEnds

`(network: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => void`

Ensure there are no dead-end nodes (input/output isolation) in the network.

### mutate

`() => Promise<void>`

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

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => void`

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

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => Promise<void>`

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

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, rawReturnForTest: boolean) => Promise<import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[] | null>`

Select a mutation method respecting structural constraints and adaptive controllers.
Mirrors legacy implementation from `neat.ts` to preserve test expectations.
`rawReturnForTest` retains historical behavior where the full FFW array is
returned for identity checks in tests.

## neat/neat.constants.ts

### neat.constants

Shared numerical / heuristic constants for NEAT modules.

Keeping these in a single dependency‑free module avoids scattering magic
numbers and simplifies tuning while refactoring.

### EPSILON

### EXTRA_CONNECTION_PROBABILITY

### NORM_EPSILON

### PROB_EPSILON

## neat/neat.diversity.ts

### computeDiversityStats

`(population: import("C:/NeatapticTS/src/neat/neat.diversity.utils").GenomeWithMetrics[], compatibilityComputer: import("C:/NeatapticTS/src/neat/neat.diversity.utils").CompatComputer) => import("C:/NeatapticTS/src/neat/neat.diversity.utils").DiversityStats | undefined`

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

### DiversityStats

Diversity statistics returned by computeDiversityStats.
Each field represents an aggregate metric for a NEAT population.

### MAX_COMPATIBILITY_SAMPLE

### MAX_LINEAGE_PAIR_SAMPLE

### structuralEntropy

`(graph: import("C:/NeatapticTS/src/architecture/network").default) => number`

Compute the Shannon-style entropy of a network's out-degree distribution.
This is a lightweight, approximate structural dispersion metric used to
characterise how 'spread out' connections are across nodes.

Educational note: structural entropy here is simply H = -sum(p_i log p_i)
over the normalized out-degree histogram. It does not measure information
content of weights or dynamics, but provides a quick structural fingerprint.

## neat/neat.selection.ts

### DEFAULT_POWER

### DEFAULT_SCORE

### DEFAULT_TOURNAMENT_PROBABILITY

### DEFAULT_TOURNAMENT_SIZE

### FIRST_INDEX

### getAverage

`() => number`

Compute the average (mean) fitness across the population.

If genomes have not been evaluated yet this will call `evaluate()` so
that scores exist. Missing scores are treated as 0.

Example:
const avg = neat.getAverage();
console.log(`Average fitness: ${avg}`);

Returns: The mean fitness as a number.

### getFittest

`() => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Return the fittest genome in the population.

This will trigger an `evaluate()` if genomes have not been scored yet, and
will ensure the population is sorted so index 0 contains the fittest.

Example:
const best = neat.getFittest();
console.log(best.score);

Returns: The genome object judged to be the fittest (highest score).

### getParent

`() => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

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

### INITIAL_MOST_NEGATIVE_SCORE

### INITIAL_TOTAL_FITNESS

### LAST_ELEMENT_INDEX

### LAST_INDEX_OFFSET

### LOOP_INDEX_INCREMENT

### SECOND_INDEX

### sort

`() => void`

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

`(entry: Record<string, unknown>) => Record<string, unknown>`

Apply a telemetry selection whitelist to a telemetry entry.

This helper inspects a per-instance Set of telemetry keys stored at
`this._telemetrySelect`. If present, only keys included in the set are
retained on the produced entry. Core fields (generation, best score and
species count) are always preserved.

Example:

Parameters:
- `entry` - - Raw telemetry object to be filtered in-place.

Returns: The filtered telemetry object (same reference as input).

### buildTelemetryEntry

`(fittest: Record<string, unknown>) => import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry`

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

### computeDiversityStats

`() => void`

Compute several diversity statistics used by telemetry reporting.

This helper is intentionally conservative in runtime: when `fastMode` is enabled it will automatically tune a few sampling defaults to keep the computation cheap. The computed statistics are written to `this._diversityStats` as an object with keys like `meanCompat` and `graphletEntropy`.

### createTelemetryEntryBase

`(generationIndex: number, bestScore: number, speciesCount: number) => import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry`

Create a strict baseline telemetry entry with required fields populated.

This helper centralizes defaults so downstream telemetry producers can
extend the entry while keeping the strict `TelemetryEntry` contract.

Parameters:
- `generationIndex` - Generation index for the telemetry snapshot.
- `bestScore` - Best fitness value observed in the generation.
- `speciesCount` - Number of extant species.

Returns: A strict telemetry entry with required fields populated.

### recordTelemetryEntry

`(entry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry) => void`

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

### structuralEntropy

`(graph: { [key: string]: unknown; nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; }) => number`

Lightweight proxy for structural entropy based on degree-distribution.

This function computes an approximate entropy of a graph topology by
counting node degrees and computing the entropy of the degree histogram.
The result is cached on the graph object for the current generation in
`_entropyVal` to avoid repeated expensive recomputation.

Example:

Parameters:
- `graph` - - A genome-like object with `nodes` and `connections` arrays.

Returns: A non-negative number approximating structural entropy.

### TelemetryContext

Context view used within telemetry helpers to access optional internal
fields with descriptive names rather than repeated inline casts.

## neat/neat.objectives.ts

### _getObjectives

`() => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

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

`() => void`

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

`(key: string, direction: "max" | "min", accessor: (genome: import("C:/NeatapticTS/src/neat/neat.types").GenomeLike) => number) => void`

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

### neat.speciation

Assign genomes into species based on compatibility distance and maintain species structures.
This function creates new species for unassigned genomes, prunes empty species, updates
dynamic compatibility threshold controllers, performs optional auto coefficient tuning, and
records per‑species history statistics used by telemetry and adaptive controllers.

Implementation notes:

### _applyFitnessSharing

`() => void`

Apply fitness sharing to penalize similarity within species.

Parameters:
- `this` - - Neat instance context with species array and compatibility distance function.

### _sortSpeciesMembers

`(species: import("C:/NeatapticTS/src/neat/neat.types").SpeciesLike) => void`

Sort species members by descending score.

Parameters:
- `this` - - Neat instance context.
- `sp` - - Species to sort.

### _speciate

`() => void`

Assign genomes into species based on compatibility distance.

Parameters:
- `this` - - Speciation harness context.

Returns: Nothing.

### _updateSpeciesStagnation

`() => void`

Update stagnation counters for all species.

Parameters:
- `this` - - Neat instance context with species array and generation counter.

## neat/neat.rng.constants.ts

### neat.rng.constants

Constants used by the deterministic xorshift RNG helper.

### RNG_DEFAULT_SEED_FALLBACK

### RNG_NORMALIZATION_DIVISOR

### RNG_POPULATION_OFFSET

### RNG_SHIFT_LEFT_PRIMARY

### RNG_SHIFT_LEFT_SECONDARY

### RNG_SHIFT_RIGHT_PRIMARY

### RNG_TIME_SCRAMBLE_CONSTANT

## neat/neat.multiobjective.ts

### neat.multiobjective

Multi-objective helpers (fast non-dominated sorting + crowding distance).
Extracted from `neat.ts` to keep the core class slimmer.

### fastNonDominated

`(pop: import("C:/NeatapticTS/src/architecture/network").default[]) => import("C:/NeatapticTS/src/architecture/network").default[][]`

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

### neat.adaptive.shared

Constant: zero value.

### ACCEPTANCE_LOWER_MULTIPLIER

### ACCEPTANCE_UPPER_MULTIPLIER

### AdaptiveMutationConfig

### ADJUST_RATE_DEFAULT

### ANCESTOR_UNIQ_MODE_EPSILON

### ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE

### AncestorUniqAdaptiveConfig

### ANNEAL_BASELINE_GENERATIONS

### ANNEAL_PROGRESS_MAX

### BUDGET_GROWTH_MULTIPLIER

### COMPLEXITY_MODE_ADAPTIVE

### COMPLEXITY_MODE_LINEAR

### ComplexityBudgetConfig

### DEFAULT_ADAPT_EVERY

### DEFAULT_ANCESTOR_UNIQ_ADJUST

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

### DEFAULT_CB_INCREASE_FACTOR

### DEFAULT_CB_STAGNATION_FACTOR

### DEFAULT_IMPROVEMENT_WINDOW

### DEFAULT_INITIAL_MUTATION_RATE

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

### DEFAULT_MAX_MUTATION_AMOUNT

### DEFAULT_MAX_MUTATION_RATE

### DEFAULT_MIN_MUTATION_AMOUNT

### DEFAULT_MIN_MUTATION_RATE

### DEFAULT_MUTATION_AMOUNT

### DEFAULT_MUTATION_AMOUNT_SIGMA

### DEFAULT_MUTATION_SIGMA

### DENOMINATOR_FALLBACK

### EXPLORE_LOW_DECREASE_MULTIPLIER

### EXPLORE_LOW_INCREASE_MULTIPLIER

### FIVE

### FOUR

### Genome

### HALF_INDEX_DIVISOR

### HISTORY_MIN_IMPROVEMENT_COUNT

### HISTORY_MIN_SLOPE_COUNT

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

### LINEAGE_PRESSURE_MODE_SPREAD

### LINEAR_HORIZON_DEFAULT

### MINIMAL_TOPOLOGY_OFFSET

### MinimalCriterionAdaptiveConfig

### MUTATION_SIGMA_SCALE

### MUTATION_STRATEGY_ANNEAL

### MUTATION_STRATEGY_EXPLORE_LOW

### MUTATION_STRATEGY_TWO_TIER

### MutationOutcome

### MutationPartitions

### MutationSettings

### NeatLikeWithAdaptive

Minimal interface for NEAT instances with adaptive features.
Exported for use in tests and type-safe function calls.

### NEGATIVE_ONE

### NOVELTY_ARCHIVE_MIN_SIZE

### NOVELTY_FACTOR_DEFAULT

### NOVELTY_FACTOR_SMALL

### ONE

### ONE_HUNDRED

### OPERATOR_DECAY_DEFAULT

### OperatorAdaptationConfig

### PHASE_COMPLEXIFY

### PHASE_LENGTH_DEFAULT

### PHASE_SIMPLIFY

### PhasedComplexityConfig

### PROGRESS_RATIO_MAX

### RNG_CENTER_OFFSET

### RNG_SPREAD_MULTIPLIER

### SLOPE_BOOST_MULTIPLIER

### SLOPE_NORMALIZE_CLAMP

### SLOPE_PENALTY_MULTIPLIER

### TARGET_ACCEPTANCE_DEFAULT

### TEN

### THREE

### TWO

### ZERO

## neat/neat.telemetry.exports.ts

### buildSpeciesHistoryCsv

`(recentHistory: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[], headers: string[]) => string`

Build the full CSV string for species history given ordered headers and
a slice of history entries.

Implementation notes:
- The history is a 2‑level structure (generation entry -> species stats[]).
- We emit one CSV row per species stat, repeating the generation value.
- Values are JSON.stringify'd to remain safe for commas/quotes.

### buildTelemetryHeaders

`(info: TelemetryHeaderInfo) => string[]`

Build the ordered list of CSV headers from collected metadata.
Flattened nested metrics are emitted using group prefixes (group.key).

### collectTelemetryHeaderInfo

`(entries: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[]) => TelemetryHeaderInfo`

Collect header metadata from the raw telemetry entries.
- Discovers base (top‑level) keys excluding grouped objects.
- Discovers nested keys inside complexity, perf, lineage, diversity groups.
- Tracks presence of optional multi-value structures (ops, objectives, etc.).

### DEFAULT_SPECIES_BEST_SCORE

### DEFAULT_SPECIES_HISTORY_GENERATION

### DEFAULT_SPECIES_HISTORY_MAX_ENTRIES

### DEFAULT_SPECIES_ID

### DEFAULT_SPECIES_LAST_IMPROVED

### DEFAULT_SPECIES_SIZE

### exportSpeciesHistoryCSV

`(maxEntries: number) => string`

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

`(maxEntries: number) => string`

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

`() => string`

Telemetry export helpers extracted from `neat.ts`.

This module exposes small helpers intended to serialize the internal
telemetry gathered by the NeatapticTS `Neat` runtime into common
data-export formats (JSONL and CSV). The functions intentionally
operate against `this` so they can be attached to instances.

### serializeTelemetryEntry

`(entry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, headers: string[]) => string`

Serialize one telemetry entry into a CSV row using previously computed headers.
Uses a `switch(true)` pattern instead of a long if/else chain to reduce
cognitive complexity while preserving readability of each scenario.

### TelemetryHeaderInfo

Shape describing collected telemetry header discovery info.

## neat/neat.evolve.offspring.constants.ts

### neat.evolve.offspring.constants

Index used when falling back to the first genome in the population.

### LINEAGE_BASE_DEPTH

### LINEAGE_DEPTH_INCREMENT

### OFFSPRING_FALLBACK_INDEX

## neat/neat.evaluate.utils.types.ts

### neat.evaluate.utils.types

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

## neat/neat.rng.utils.ts

### exportRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost) => number | undefined`

Export the current RNG state for persistence.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when not set.

### getOrCreateRng

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost) => () => number`

Return a cached RNG or create a deterministic xorshift RNG when absent.

The helper respects a user-provided RNG at `options.rng` when present.
Otherwise it seeds a xorshift32 RNG using the current time and population
size, guarding against the invalid zero seed.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

### importRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost, state: string | number | undefined) => void`

Alias for restoring RNG state kept for compatibility with prior surface.

### restoreRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost, state: string | number | undefined) => void`

Restore a previously captured RNG state.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### RngHost

Minimal host surface required by the RNG utilities.

### sampleRandomSequence

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost, sampleCount: number) => number[]`

Produce a sequence of random samples using the host RNG.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

### snapshotRngState

`(host: import("C:/NeatapticTS/src/neat/neat.rng.utils").RngHost) => number | undefined`

Snapshot the current RNG state for deterministic replay.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.

## neat/neat.cache.utils.ts

### neat.cache.utils

Invalidate per-genome caches used across compatibility and forward-pass logic.

@param genomeCandidate - Genome object whose caches should be cleared.

### invalidateGenomeCaches

`(genomeCandidate: unknown) => void`

Invalidate per-genome caches used across compatibility and forward-pass logic.

Parameters:
- `genomeCandidate` - - Genome object whose caches should be cleared.

## neat/neat.compat.utils.ts

### neat.compat.utils

Compatibility-distance helper utilities.

@remarks
This module is intentionally dependency-free and provides small, focused
helpers for use by the compatibility orchestration layer.

### buildPairKey

`(firstGenome: import("C:/NeatapticTS/src/neat/neat.compat.utils").GenomeLike, secondGenome: import("C:/NeatapticTS/src/neat/neat.compat.utils").GenomeLike) => string`

Build a stable cache key for a genome pair.

Parameters:
- `firstGenome` - - First genome in the pair.
- `secondGenome` - - Second genome in the pair.

Returns: Stable cache key in the form "minId|maxId".

### compareInnovationLists

`(firstList: [number, number][], secondList: [number, number][]) => import("C:/NeatapticTS/src/neat/neat.compat.utils").ComparisonMetrics`

Compare two sorted innovation lists and derive comparison metrics.

Parameters:
- `firstList` - - Sorted innovation list for the first genome.
- `secondList` - - Sorted innovation list for the second genome.

Returns: Aggregated comparison metrics for distance computation.

### ComparisonMetrics

Aggregated comparison metrics for compatibility calculations.

### computeCompatibilityDistance

`(neatContext: import("C:/NeatapticTS/src/neat/neat.compat.utils").NeatLikeForCompat, metrics: import("C:/NeatapticTS/src/neat/neat.compat.utils").ComparisonMetrics) => number`

Compute the compatibility distance from comparison metrics.

Parameters:
- `neatContext` - - NEAT context providing coefficients.
- `metrics` - - Aggregated comparison metrics.

Returns: Final compatibility distance for the genome pair.

### ConnectionLike

Compatibility-distance helper utilities.

### ensureGenerationCache

`(neatContext: import("C:/NeatapticTS/src/neat/neat.compat.utils").NeatLikeForCompat) => void`

Ensure generation-scoped compatibility caches exist.

Parameters:
- `neatContext` - - Current NEAT context holding generation and caches.

Returns: void

### GenomeLike

Minimal genome shape used for compatibility distance calculations.

### getDistanceCacheMap

`(neatContext: import("C:/NeatapticTS/src/neat/neat.compat.utils").NeatLikeForCompat) => Map<string, number>`

Retrieve the generation-scoped cache map for pairwise distances.

Parameters:
- `neatContext` - - Current NEAT context with the cache map.

Returns: Map storing cached distances for genome pairs this generation.

### getSortedInnovationCache

`(neatContext: import("C:/NeatapticTS/src/neat/neat.compat.utils").NeatLikeForCompat, genome: import("C:/NeatapticTS/src/neat/neat.compat.utils").GenomeLike) => [number, number][]`

Retrieve or build a sorted innovation list for a genome.

Parameters:
- `neatContext` - - NEAT context used for fallback innovation numbers.
- `genome` - - Genome to derive sorted innovation list for.

Returns: Array of [innovationNumber, weight] sorted by innovationNumber.

### NeatLikeForCompat

Minimal NEAT context required by compatibility helpers.

### resolveMaxInnovation

`(list: [number, number][]) => number`

Resolve the highest innovation id from a sorted list.

Parameters:
- `list` - - Sorted innovation list for a genome.

Returns: Highest innovation id or 0 if list is empty.

## neat/neat.evolve.utils.ts

### adaptReenableProbability

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { minSamples: number; target: number; min: number; max: number; deltaScale: number; }) => void`

Adapt the re-enable probability based on recent success ratios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### addOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], helpers: { addSpeciatedOffspring: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number) => Promise<void>; addUnspeciatedOffspring: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number) => Promise<void>; }) => Promise<void>`

Add offspring to fill remaining population slots.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `helpers` - - Helper callbacks for offspring selection.
- `helpers` - - Speciated offspring helper.
- `helpers` - - Unspeciated offspring helper.

Returns: void.

### addSpeciatedOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number, config: { minOffspringDefault: number; survivalThresholdDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; crossSpeciesGuardLimit: number; }) => Promise<void>`

Add offspring when speciation is enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Offspring allocation constants.

Returns: void.

### addUnspeciatedOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number) => Promise<void>`

Add offspring when speciation is disabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.

Returns: void.

### applyAdaptiveComplexityControllers

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply adaptive complexity controllers if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAncestorUniqAdaptiveSafe

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply ancestor uniqueness adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAutoCompatibilityTuning

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { targetMin: number; adjustRate: number; minCoeff: number; maxCoeff: number; randomScale: number; }) => void`

Apply auto-compatibility tuning if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Tuning constants.

Returns: void.

### applyDynamicObjectiveSchedule

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, currentObjectiveKeys: string[], config: { autoEntropyAddAt: number; }) => void`

Apply dynamic objective scheduling and entropy rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Keys of active objectives.
- `config` - - Scheduling constants.

Returns: void.

### applyElitism

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Apply elitism for the next generation.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyFitnessSuppressionForTests

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Suppress fitness objective for specific test scenarios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyGlobalStagnationInjectionIfNeeded

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { buildFreshGenomeForStagnation: () => Promise<import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata>; replaceFraction: number; }) => Promise<void>`

Apply global stagnation injection if configured.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for stagnation injection.
- `helpers` - - Genome builder for injection.

Returns: void.

### applyMinimalCriterionAdaptiveSafe

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply minimal criterion adaptive controller if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyOperatorAdaptationSafe

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply operator adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyProvenance

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Add provenance genomes into the next population.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyPruningAndMutation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply pruning and mutation phases.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applySpeciationAndSharingIfEnabled

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { applyAutoCompatibilityTuning: () => void; recordSpeciesHistorySnapshot: () => void; }) => Promise<void>`

Apply speciation, fitness sharing, and related side effects.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used for tuning and history.
- `helpers` - - Auto-compatibility adjustment helper.
- `helpers` - - Species history snapshot helper.

Returns: void.

### buildFittestSnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => import("C:/NeatapticTS/src/architecture/network").default`

Build a cloned Network from the current best genome.

Parameters:
- `internal` - - NEAT controller instance.

Returns: best network snapshot.

### buildFreshGenomeForStagnation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata>`

Build a fresh genome for stagnation injection.

Parameters:
- `internal` - - NEAT controller instance.

Returns: new genome with minimum constraints.

### buildNextPopulation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { applyElitism: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void; applyProvenance: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void; addOffspring: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => Promise<void>; }) => Promise<import("C:/NeatapticTS/src/architecture/network").default[]>`

Build the next population (elitism, provenance, offspring).

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for population construction.
- `helpers` - - Elitism helper.
- `helpers` - - Provenance helper.
- `helpers` - - Offspring helper.

Returns: next population array.

### captureObjectiveImportanceSnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Capture objective importance stats for telemetry.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### clearPopulationScores

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Clear genome scores to force re-evaluation.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeDiversityStatsSafely

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Compute diversity stats safely if the hook exists.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeElapsedTime

`(startTimestamp: number) => number`

Compute elapsed time since the start of evolve().

Parameters:
- `startTimestamp` - - Start time resolved earlier.

Returns: elapsed time.

### createOffspring

`(context: import("C:/NeatapticTS/src/neat/neat.evolve.offspring.utils").OffspringContext, selectParent: () => import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network").default`

Create a child genome by crossing two parents selected via the provided callback.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.

Returns: Newly created offspring genome.

### enforcePopulationConstraints

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => Promise<void>`

Ensure new population meets structural constraints.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Population to validate.

Returns: void.

### ensurePopulationEvaluated

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Ensure the population is evaluated before evolution operations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### ensureSpeciesHistorySnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, maxHistory: number) => void`

Ensure a minimal species history snapshot exists for exports.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### EVOLVE_NO_BEST_GENOME_WARNING

### invalidateCompatibilityCaches

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Invalidate compatibility caches after mutations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### LINEAGE_BASE_DEPTH

### LINEAGE_DEPTH_INCREMENT

### OFFSPRING_FALLBACK_INDEX

### OffspringContext

Minimal surface needed for offspring generation.

### processMultiObjective

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { paretoArchiveMax: number; targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; pruneWindowDefault: number; pruneRangeEpsDefault: number; }) => void`

Run multi-objective ranking, crowding distance, and archives.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Multi-objective tuning constants.

Returns: void.

### recordSpeciesHistorySnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, maxHistory: number) => void`

Record a species history snapshot when needed.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### recordTelemetryIfEnabled

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, snapshot: import("C:/NeatapticTS/src/architecture/network").default) => Promise<void>`

Record telemetry if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot for the generation.

Returns: void.

### resetObjectivesCache

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Clear cached objectives so dynamic schedules can rebuild them.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### resolveStartTime

`() => number`

Resolve the start time for an evolution step.

Returns: timestamp in milliseconds or high-resolution units.

### trackGlobalImprovement

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, snapshot: import("C:/NeatapticTS/src/architecture/network").default) => void`

Track global best improvement for stagnation logic.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot.

Returns: void.

### updateGlobalBestTracking

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Update generation-level best score tracking.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### updateObjectiveScheduleAndAges

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) => void; }) => Promise<void>`

Update objective schedule, pending adds/removes, and objective ages.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used by scheduling logic.
- `helpers` - - Dynamic objective scheduler.

Returns: void.

### updateSpeciesStagnationIfEnabled

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Update species stagnation status when speciation enabled.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### warnIfNoBestGenome

`() => void`

Emit the standard warning for runs that end without a valid best genome.

## neat/neat.lineage.utils.ts

### neat.lineage.utils

Lineage / ancestry helper utilities for NEAT populations.

This module centralizes helper logic used by the public lineage APIs to keep
the main entry file small and orchestration-focused.

### AncestorQueueEntry

Queue entry for ancestor traversal.

### calculateMaxSamplePairs

`(size: number) => number`

Parameters:
- `size` - Population size.

Returns: Upper bound on the number of sampled pairs.

### collectAncestorIds

`(queueEntries: AncestorQueueEntry[], population: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike[]) => Set<number>`

Parameters:
- `queueEntries` - BFS queue seeded with direct parents.
- `population` - Current population for lookups.

Returns: Unique ancestor IDs encountered within the window.

### computeAverageDistance

`(distances: number[]) => number`

Parameters:
- `distances` - Jaccard distances to average.

Returns: Mean distance rounded to the configured decimal places.

### computeJaccardDistance

`(ancestorSetA: Set<number>, ancestorSetB: Set<number>) => number`

Parameters:
- `ancestorSetA` - First ancestor set.
- `ancestorSetB` - Second ancestor set.

Returns: Jaccard distance for the two sets.

### computePairDistance

`(pair: GenomeIndexPair, population: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike[], buildAncestorSet: (genome: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike) => Set<number>) => number | undefined`

Parameters:
- `pair` - Index pair for comparison.
- `population` - Current population.
- `buildAncestorSet` - Helper to build ancestor sets.

Returns: Jaccard distance or undefined when skipped.

### computePairDistances

`(pairs: GenomeIndexPair[], population: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike[], buildAncestorSet: (genome: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike) => Set<number>) => number[]`

Parameters:
- `pairs` - Sampled index pairs.
- `population` - Current population.
- `buildAncestorSet` - Helper to build ancestor sets.

Returns: Jaccard distances for valid pairs.

### countIntersection

`(ancestorSetA: Set<number>, ancestorSetB: Set<number>) => number`

Parameters:
- `ancestorSetA` - First ancestor set.
- `ancestorSetB` - Second ancestor set.

Returns: Size of intersection between the sets.

### createInitialQueue

`(parentIds: number[], population: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike[]) => AncestorQueueEntry[]`

Parameters:
- `parentIds` - Direct parent IDs to seed the queue.
- `population` - Current population for lookups.

Returns: Queue entries at depth 1.

### enqueueParentEntries

`(queueEntries: AncestorQueueEntry[], currentEntry: AncestorQueueEntry, population: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike[]) => void`

Parameters:
- `queueEntries` - Mutable queue array to append to.
- `currentEntry` - Current ancestor entry being expanded.
- `population` - Current population for lookups.

### findGenomeById

`(population: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike[], genomeId: number) => import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike | undefined`

Parameters:
- `population` - Current population for lookup.
- `genomeId` - Genome identifier to match.

Returns: Genome reference if found.

### GenomeIndexPair

Index pair representing a sampled genome pair.

### GenomeLike

Lineage / ancestry helper utilities for NEAT populations.

This module centralizes helper logic used by the public lineage APIs to keep
the main entry file small and orchestration-focused.

### hasMinimumPopulation

`(size: number) => boolean`

Parameters:
- `size` - Population size.

Returns: True when at least two genomes exist.

### isEmptyAncestorPair

`(ancestorSetA: Set<number>, ancestorSetB: Set<number>) => boolean`

Parameters:
- `ancestorSetA` - First ancestor set.
- `ancestorSetB` - Second ancestor set.

Returns: True when both sets are empty.

### isWithinDepthWindow

`(depth: number) => boolean`

Parameters:
- `depth` - Current depth value.

Returns: True when within the configured depth window.

### NeatLineageContext

Expected `this` context for lineage helpers (a subset of the NEAT instance).

### normalizeParentIds

`(value: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike) => number[]`

Parameters:
- `value` - Genome to read parents from.

Returns: Parent ID list (empty when absent).

### pickDistinctIndex

`(randomNumber: () => number, size: number, firstIndex: number) => number`

Parameters:
- `randomNumber` - RNG function returning [0,1).
- `size` - Population size for bounds.
- `firstIndex` - Index to avoid.

Returns: Random index not equal to the first index.

### pickRandomIndex

`(randomNumber: () => number, size: number) => number`

Parameters:
- `randomNumber` - RNG function returning [0,1).
- `size` - Population size for bounds.

Returns: Random index within bounds.

### resolveParentIds

`(value: import("C:/NeatapticTS/src/neat/neat.lineage.utils").GenomeLike | undefined) => number[]`

Parameters:
- `value` - Optional genome reference.

Returns: Parent IDs when available.

### sampleGenomePairs

`(sampleCount: number, size: number, rngFactory: () => () => number) => GenomeIndexPair[]`

Parameters:
- `sampleCount` - Number of pairs to sample.
- `size` - Population size for index bounds.
- `rngFactory` - RNG provider to obtain a random function.

Returns: Array of sampled index pairs.

## neat/neat.novelty.utils.ts

### neat.novelty.utils

Return the current size of the novelty archive.

### getNoveltyArchiveSize

`(host: { _noveltyArchive?: unknown[] | undefined; }) => number`

Return the current size of the novelty archive.

### resetNoveltyArchive

`(host: { _noveltyArchive?: unknown[] | undefined; }) => void`

Reset the novelty archive in place.

## neat/neat.pruning.utils.ts

### neat.pruning.utils

Minimal Neat instance contract required by pruning helpers.

@example
const host: NeatLikeForPruning = {
  options: { evolutionPruning: { startGeneration: 5, targetSparsity: 0.4 } },
  generation: 10,
  population: [],
} as NeatLikeForPruning;

### AdaptivePruningOptions

Adaptive pruning options extracted from the Neat instance.

### applyAdaptivePruneLevelToPopulation

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning, pruneLevel: number) => void`

Parameters:
- `host` - - Neat instance with population.
- `pruneLevel` - - Prune level to apply.

### applyPruningToPopulation

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning, options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; }, targetSparsity: number) => void`

Parameters:
- `host` - - Neat instance with population.
- `options` - - Evolution pruning options.
- `targetSparsity` - - Target sparsity to apply.

### computeMeanConnectionCount

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning) => number`

Parameters:
- `host` - - Neat instance with population.

Returns: Average number of connections per genome.

### computeMeanNodeCount

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning) => number`

Parameters:
- `host` - - Neat instance with population.

Returns: Average number of nodes per genome.

### computeNextAdaptivePruneLevel

`(options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; }, currentPruneLevel: number, currentMetricValue: number, targetRemainingMetric: number) => number`

Parameters:
- `options` - - Adaptive pruning options.
- `currentPruneLevel` - - Current global prune level.
- `currentMetricValue` - - Current observed metric value.
- `targetRemainingMetric` - - Target remaining metric value.

Returns: Updated prune level.

### computePopulationMetrics

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning) => import("C:/NeatapticTS/src/neat/neat.pruning.utils").PopulationMetrics`

Parameters:
- `host` - - Neat instance with population.

Returns: Population metric summary.

### computeRampFraction

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning, options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; }) => number`

Parameters:
- `host` - - Neat instance with generation state.
- `options` - - Evolution pruning options.

Returns: Fraction in [0,1] indicating ramp completion.

### computeTargetRemainingMetric

`(options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; }, adaptivePruneBaseline: number) => number`

Parameters:
- `options` - - Adaptive pruning options.
- `adaptivePruneBaseline` - - Baseline metric value.

Returns: Target remaining metric value.

### computeTargetSparsityNow

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning, options: { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; }) => number`

Parameters:
- `host` - - Neat instance with generation state.
- `options` - - Evolution pruning options.

Returns: Target sparsity to apply for this generation.

### EvolutionPruningOptions

Evolution pruning options extracted from the Neat instance.

### initializeAdaptivePruningState

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning) => void`

Parameters:
- `host` - - Neat instance with adaptive pruning state.

### NeatLikeForPruning

Minimal Neat instance contract required by pruning helpers.

### PopulationMetrics

Summary of population metrics used by adaptive pruning.

### resolveActiveAdaptivePruningOptions

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning) => { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; } | null`

Parameters:
- `host` - - Neat instance with adaptive pruning options.

Returns: Adaptive pruning options when enabled, otherwise null.

### resolveActiveEvolutionPruningOptions

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning) => { startGeneration?: number | undefined; interval?: number | undefined; rampGenerations?: number | undefined; targetSparsity?: number | undefined; method?: string | undefined; } | null`

Parameters:
- `host` - - Neat instance with generation state.

Returns: Evolution pruning options when active, otherwise null.

### resolveAdaptivePruneBaseline

`(host: import("C:/NeatapticTS/src/neat/neat.pruning.utils").NeatLikeForPruning, currentMetricValue: number) => number`

Parameters:
- `host` - - Neat instance with adaptive baseline state.
- `currentMetricValue` - - Current observed metric value.

Returns: Baseline metric value used for adaptation.

### resolveObservedMetricValue

`(options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; }, metrics: import("C:/NeatapticTS/src/neat/neat.pruning.utils").PopulationMetrics) => number`

Parameters:
- `options` - - Adaptive pruning options.
- `metrics` - - Population metric summary.

Returns: Current observed metric value used for adaptation.

### shouldAdjustAdaptivePruning

`(options: { enabled?: boolean | undefined; metric?: string | undefined; targetSparsity?: number | undefined; learningRate?: number | undefined; tolerance?: number | undefined; adjustRate?: number | undefined; }, currentMetricValue: number, targetRemainingMetric: number, adaptivePruneBaseline: number) => boolean`

Parameters:
- `options` - - Adaptive pruning options.
- `currentMetricValue` - - Current observed metric value.
- `targetRemainingMetric` - - Target remaining metric value.
- `adaptivePruneBaseline` - - Baseline metric value.

Returns: True when pruning should be adjusted.

## neat/neat.species.utils.ts

### backfillExtendedHistory

`(history: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[], context: { _species?: import("C:/NeatapticTS/src/neat/neat.types").SpeciesLike[] | undefined; _fallbackInnov?: ((c: import("C:/NeatapticTS/src/neat/neat.types").ConnectionLike) => number) | undefined; }) => void`

Parameters:
- `history` - - Recorded history to enrich in place.
- `context` - - Neat instance context for lookups.

### shouldAugmentExtendedHistory

`(options: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions | undefined) => boolean`

Parameters:
- `options` - - Current Neat options.

Returns: True when extended history is enabled.

### SPECIES_HISTORY_DEFAULT_ENABLED_RATIO

### SPECIES_HISTORY_DEFAULT_INNOVATION_ID

### SPECIES_HISTORY_DEFAULT_INNOVATION_RANGE

### SPECIES_HISTORY_INITIAL_MAX_INNOVATION

### SPECIES_HISTORY_INITIAL_MIN_INNOVATION

### SPECIES_HISTORY_ZERO

## neat/neat.adaptive.utils.ts

### ACCEPTANCE_LOWER_MULTIPLIER

### ACCEPTANCE_UPPER_MULTIPLIER

### AdaptiveMutationConfig

### ADJUST_RATE_DEFAULT

### adjustConnectionBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }, trends: { improvement: number; slope: number; }, factors: { increaseFactor: number; stagnationFactor: number; }, noveltyFactor: number, history: number[]) => void`

Adjust connection budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### adjustNodeBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }, trends: { improvement: number; slope: number; }, factors: { increaseFactor: number; stagnationFactor: number; }, noveltyFactor: number, history: number[]) => void`

Adjust node budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### ANCESTOR_UNIQ_MODE_EPSILON

### ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE

### AncestorUniqAdaptiveConfig

### ANNEAL_BASELINE_GENERATIONS

### ANNEAL_PROGRESS_MAX

### applyAdaptiveSchedule

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Apply adaptive complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance with adaptive state.
- `config` - - Complexity budget configuration.

### applyAnnealDelta

`(baseDelta: number, settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings) => number`

Apply annealing adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `settings` - - Resolved settings.

Returns: Adjusted delta.

### applyComplexityBudgetSchedule

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Apply the complexity budget schedule for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyEpsilonAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, ancestorUniq: number, thresholds: { lowThreshold: number; highThreshold: number; }, adjustMagnitude: number) => void`

Apply dominance-epsilon adjustments when configured.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### applyExploreLowDelta

`(baseDelta: number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

Apply explore-low adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyLineagePressureAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, ancestorUniq: number, thresholds: { lowThreshold: number; highThreshold: number; }) => void`

Apply lineage pressure strength adjustments.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.

### applyLinearSchedule

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Apply linear complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyMutationAmount

`(genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => void`

Apply mutation-amount adjustments to a genome.

Parameters:
- `genome` - - Current genome.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

### applyMutationsToPopulation

`(population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[], partitions: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationPartitions, settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number) => import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationOutcome`

Apply mutation updates to the population.

Parameters:
- `population` - - Full population to mutate.
- `partitions` - - Scored partitions.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.

Returns: Mutation outcome flags.

### applyOperatorDecay

`(stats: Map<string, { success: number; attempts: number; }>, entries: [string, { success: number; attempts: number; }][], decay: number) => void`

Apply exponential decay to each operator statistic entry.

Parameters:
- `stats` - - Operator statistics map.
- `entries` - - Operator stat entries to update.
- `decay` - - Decay factor.

### applyRejection

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, threshold: number) => void`

Zero scores below the final threshold.

Parameters:
- `engine` - - NEAT engine instance.
- `threshold` - - Final MC threshold.

### applyTwoTierAmountDelta

`(baseDelta: number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

Apply two-tier adjustments to amount delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierDelta

`(baseDelta: number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

Apply two-tier adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierFallback

`(population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[], settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings) => void`

Apply two-tier fallback balancing.

Parameters:
- `population` - - Population of genomes.
- `settings` - - Resolved settings.

### applyUniquenessAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }, ancestorUniq: number, thresholds: { lowThreshold: number; highThreshold: number; }, adjustMagnitude: number) => void`

Apply an adjustment for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### BUDGET_GROWTH_MULTIPLIER

### clampNodeBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Clamp node budget to configured minimum.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### clampValue

`(value: number, min: number, max: number) => number`

Clamp a value between min and max bounds.

Parameters:
- `value` - - Value to clamp.
- `min` - - Minimum bound.
- `max` - - Maximum bound.

Returns: Clamped value.

### collectOperatorStatsEntries

`(stats: Map<string, { success: number; attempts: number; }>) => [string, { success: number; attempts: number; }][]`

Collect operator statistic entries for processing.

Parameters:
- `stats` - - Operator statistics map.

Returns: Array of operator stat entries.

### collectScoredGenomes

`(population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]) => { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]`

Collect genomes with numeric scores.

Parameters:
- `population` - - Population of genomes.

Returns: Scored genomes.

### collectScores

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => number[]`

Collect population scores into a snapshot array.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Array of scores (missing scores treated as 0).

### COMPLEXITY_MODE_ADAPTIVE

### COMPLEXITY_MODE_LINEAR

### ComplexityBudgetConfig

### computeAcceptance

`(scores: number[], threshold: number) => number`

Compute acceptance metrics for the current threshold.

Parameters:
- `scores` - - Population score snapshot.
- `threshold` - - Current MC threshold.

Returns: Acceptance proportion.

### computeAdjustmentFactors

`(config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }, trends: { improvement: number; slope: number; }, history: number[]) => { increaseFactor: number; stagnationFactor: number; }`

Compute adjustment factors for budget growth and decay.

Parameters:
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `history` - - Rolling history of best scores.

Returns: Adjustment factors (increase and stagnation multipliers).

### computeNoveltyFactor

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => number`

Compute novelty factor based on archive size.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Novelty multiplier (0.9 if archive small, 1.0 otherwise).

### computeSlope

`(history: number[]) => number`

Compute linear regression slope using ordinary least squares.

Parameters:
- `history` - - Rolling history of best scores.

Returns: OLS slope estimate.

### computeTrends

`(history: number[]) => { improvement: number; slope: number; }`

Compute improvement and slope trends from score history.

Parameters:
- `history` - - Rolling history of best scores.

Returns: Trend metrics (improvement and slope).

### createRandomDelta

`(sigmaBase: number, randomSource: () => number) => number`

Create a signed random delta scaled by sigma.

Parameters:
- `sigmaBase` - - Sigma scaling factor.
- `randomSource` - - Random number provider.

Returns: Signed delta.

### decayOperatorStat

`(operatorStat: { success: number; attempts: number; }, decay: number) => { success: number; attempts: number; }`

Apply decay to a single operator statistic record.

Parameters:
- `operatorStat` - - Operator statistic record.
- `decay` - - Decay factor.

Returns: Decayed operator statistic record.

### DEFAULT_ADAPT_EVERY

### DEFAULT_ANCESTOR_UNIQ_ADJUST

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

### DEFAULT_CB_INCREASE_FACTOR

### DEFAULT_CB_STAGNATION_FACTOR

### DEFAULT_IMPROVEMENT_WINDOW

### DEFAULT_INITIAL_MUTATION_RATE

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

### DEFAULT_MAX_MUTATION_AMOUNT

### DEFAULT_MAX_MUTATION_RATE

### DEFAULT_MIN_MUTATION_AMOUNT

### DEFAULT_MIN_MUTATION_RATE

### DEFAULT_MUTATION_AMOUNT

### DEFAULT_MUTATION_AMOUNT_SIGMA

### DEFAULT_MUTATION_SIGMA

### DENOMINATOR_FALLBACK

### ensureLineagePressureState

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => { enabled?: boolean | undefined; mode?: string | undefined; strength?: number | undefined; }`

Ensure lineage pressure state is available.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Lineage pressure configuration object.

### EXPLORE_LOW_DECREASE_MULTIPLIER

### EXPLORE_LOW_INCREASE_MULTIPLIER

### extractAncestorUniqueness

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => number | undefined`

Extract the latest ancestor-uniqueness metric from telemetry.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Ancestor uniqueness value or undefined when missing.

### FIVE

### FOUR

### Genome

### HALF_INDEX_DIVISOR

### HISTORY_MIN_IMPROVEMENT_COUNT

### HISTORY_MIN_SLOPE_COUNT

### initializeConnectionBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Initialize connection budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializeNodeBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Initialize node budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializePhaseState

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; }) => void`

Ensure phase state is initialized.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### initializeThreshold

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; }) => void`

Initialize MC threshold if missing.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Minimal-criterion adaptive configuration.

### isCooldownSatisfied

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }) => boolean`

Determine whether the cooldown window has elapsed.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: True when adjustment is allowed.

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

### LINEAGE_PRESSURE_MODE_SPREAD

### LINEAR_HORIZON_DEFAULT

### MINIMAL_TOPOLOGY_OFFSET

### MinimalCriterionAdaptiveConfig

### MUTATION_SIGMA_SCALE

### MUTATION_STRATEGY_ANNEAL

### MUTATION_STRATEGY_EXPLORE_LOW

### MUTATION_STRATEGY_TWO_TIER

### MutationOutcome

### MutationPartitions

### MutationSettings

### NeatLikeWithAdaptive

Minimal interface for NEAT instances with adaptive features.
Exported for use in tests and type-safe function calls.

### NEGATIVE_ONE

### normalizeSlope

`(slope: number, initialScore: number) => number`

Normalize slope magnitude relative to initial score.

Parameters:
- `slope` - - Raw OLS slope.
- `initialScore` - - First score in history window.

Returns: Normalized slope clamped to [-2, 2].

### NOVELTY_ARCHIVE_MIN_SIZE

### NOVELTY_FACTOR_DEFAULT

### NOVELTY_FACTOR_SMALL

### ONE

### ONE_HUNDRED

### OPERATOR_DECAY_DEFAULT

### OperatorAdaptationConfig

### PHASE_COMPLEXIFY

### PHASE_LENGTH_DEFAULT

### PHASE_SIMPLIFY

### PhasedComplexityConfig

### PROGRESS_RATIO_MAX

### recordAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => void`

Record the generation when an adjustment is applied.

Parameters:
- `engine` - - NEAT engine instance.

### resolveAdjustmentMagnitude

`(config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }) => number`

Resolve adjustment magnitude for nudging controlled parameters.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Adjustment magnitude.

### resolveAmountDelta

`(settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

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

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; }) => import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings`

Resolve mutation settings derived from configuration and engine state.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Adaptive mutation configuration.

Returns: Resolved mutation settings.

### resolveNextPhase

`(currentPhase: string) => string`

Resolve next phase name.

Parameters:
- `currentPhase` - - Current phase label.

Returns: Next phase label.

### resolveOperatorDecay

`(config: { enabled?: boolean | undefined; learningRate?: number | undefined; alpha?: number | undefined; decay?: number | undefined; }) => number`

Resolve the decay factor for operator statistics.

Parameters:
- `config` - - Operator adaptation configuration.

Returns: Decay factor for exponential smoothing.

### resolveRandomSource

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => () => number`

Resolve a random source that matches the legacy RNG usage.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Random number provider.

### resolveRateDelta

`(settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

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

`(config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; }) => { targetAcceptance: number; adjustRate: number; }`

Resolve target acceptance and adjust rate settings.

Parameters:
- `config` - - Minimal-criterion adaptive configuration.

Returns: Target settings.

### resolveUniquenessThresholds

`(config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }) => { lowThreshold: number; highThreshold: number; }`

Resolve thresholds for ancestor-uniqueness decisions.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Threshold bounds.

### RNG_CENTER_OFFSET

### RNG_SPREAD_MULTIPLIER

### shouldAdaptThisGeneration

`(generation: number, config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; }) => boolean`

Check whether mutation adaptation should run this generation.

Parameters:
- `generation` - - Current generation index.
- `config` - - Adaptive mutation configuration.

Returns: True if adaptation should run.

### shouldApplyTwoTierFallback

`(strategy: string, outcome: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationOutcome) => boolean`

Determine whether a two-tier fallback is needed.

Parameters:
- `strategy` - - Mutation strategy identifier.
- `outcome` - - Mutation outcome flags.

Returns: True if fallback should run.

### SLOPE_BOOST_MULTIPLIER

### SLOPE_NORMALIZE_CLAMP

### SLOPE_PENALTY_MULTIPLIER

### sortScoredGenomes

`(scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]) => { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]`

Sort scored genomes in ascending score order.

Parameters:
- `scoredGenomes` - - Scored genomes.

Returns: Sorted genomes.

### splitScoredGenomes

`(scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]) => import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationPartitions`

Split scored genomes into top and bottom halves.

Parameters:
- `scoredGenomes` - - Sorted scored genomes.

Returns: Partitions used by strategy rules.

### TARGET_ACCEPTANCE_DEFAULT

### TEN

### THREE

### togglePhaseIfNeeded

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; }) => void`

Toggle phase if the current phase has exceeded its length.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### TWO

### updateScoreHistory

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => number[]`

Update rolling score history with current best score.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

Returns: Rolling history array after update.

### updateThreshold

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, acceptance: number, tuning: { targetAcceptance: number; adjustRate: number; }) => void`

Update the MC threshold based on acceptance proportion.

Parameters:
- `engine` - - NEAT engine instance.
- `acceptance` - - Observed acceptance proportion.
- `tuning` - - Target acceptance and adjustment settings.

### ZERO

## neat/neat.evaluate.utils.ts

### AUTO_COEFF_ADJUST_DEFAULT

### AUTO_COEFF_MAX_DEFAULT

### AUTO_COEFF_MIN_DEFAULT

### COMPAT_MAX_THRESHOLD_DEFAULT

### COMPAT_MIN_THRESHOLD_DEFAULT

### COMPAT_THRESHOLD_DEFAULT

### DISTANCE_COEFF_DEFAULT

### DiversityStats

Diversity statistics tracked during evaluation.

The values are optional because different evaluations may only compute a
subset of metrics.

### ensureDiversityStatsContainer

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.

Returns: void.

### ENTROPY_ADJUST_DEFAULT

### ENTROPY_DEADBAND_DEFAULT

### ENTROPY_TARGET_DEFAULT

### ENTROPY_VAR_ADJUST_DEFAULT

### ENTROPY_VAR_HIGH_BAND

### ENTROPY_VAR_LOW_BAND

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

### ENTROPY_VAR_TARGET_DEFAULT

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

### NOVELTY_DEFAULT_BLEND

### NOVELTY_DEFAULT_NEIGHBORS

### NoveltyArchiveEntry

Novelty archive entry with descriptor and novelty score.

Entries store a descriptor vector alongside the computed novelty so the
archive can seed future novelty calculations.

### ObjectiveDef

Objective definition for multi-objective optimization.

Objectives are registered dynamically to guide evaluation and selection.

### runAutoDistanceCoefficientTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runAutoEntropyObjectiveInjection

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runEntropyCompatibilityTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runEntropySharingTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runFitnessEvaluation

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => Promise<void>`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Promise<void> after fitness evaluation completes.

### runLightweightSpeciation

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### runNoveltyBlendAndArchive

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### VARIANCE_DECREASE_THRESHOLD

### VARIANCE_INCREASE_THRESHOLD

## neat/neat.mutation.utils.ts

### applyAddConnMutation

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => void`

Apply an ADD_CONN mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyAddNodeMutation

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => void`

Apply an ADD_NODE mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyMutationOperator

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => void`

Apply a mutation operator to a genome and invalidate caches as needed.

Parameters:
- `genome` - - genome to mutate
- `mutationMethod` - - mutation operator to apply
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyOperatorAdaptationForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[]`

Apply operator adaptation weighting to the pool when enabled.

Parameters:
- `pool` - - base pool
- `internal` - - neat controller context

Returns: augmented pool

### applyOperatorBanditForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], fallbackMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod`

Apply operator bandit selection if enabled.

Parameters:
- `pool` - - operator pool
- `fallbackMethod` - - method used when bandit is disabled
- `internal` - - neat controller context

Returns: selected method

### applyPhasedComplexityForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[]`

Apply phased complexity adjustments to the pool when enabled.

Parameters:
- `pool` - - base operator pool
- `internal` - - neat controller context

Returns: pool with phased complexity adjustments

### applySplitWithExistingRecord

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, splitDescriptor: { splitKey: string; originalWeight: number; }, splitRecord: { newNodeGeneId: number; inInnov: number; outInnov: number; }, NodeClass: new (type: "input" | "output" | "hidden") => unknown) => void`

Apply a split using an existing innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `splitRecord` - - existing innovation record
- `NodeClass` - - node constructor

Returns: void

### applySplitWithNewRecord

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, splitDescriptor: { splitKey: string; originalWeight: number; }, NodeClass: new (type: "input" | "output" | "hidden") => unknown, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Apply a split and create a new innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `NodeClass` - - node constructor
- `internal` - - neat controller context

Returns: void

### assignInnovationForConnection

`(connection: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, pairNodes: { symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Assign an innovation id for a new connection, reusing when possible.

Parameters:
- `connection` - - newly created connection
- `pairNodes` - - resolved pair metadata
- `internal` - - neat controller context

Returns: void

### assignInnovationsForNewSplit

`(newNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, splitConnections: { incomingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; outgoingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => { newNodeGeneId: number; inInnov: number; outInnov: number; }`

Assign new innovations for a split and build the innovation record.

Parameters:
- `newNode` - - newly created hidden node
- `splitConnections` - - incoming/outgoing connections
- `internal` - - neat controller context

Returns: innovation record for the split

### buildLegacyKeyForConn

`(sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => string`

Build a legacy directional innovation key.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: directional innovation key

### buildSplitDescriptor

`(connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata) => { splitKey: string; originalWeight: number; }`

Build the split descriptor used for innovation lookup and connection creation.

Parameters:
- `connectionToSplit` - - connection being split

Returns: split descriptor

### buildSymmetricKeyForConn

`(sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => string`

Build a symmetric innovation key for an unordered node pair.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: symmetric innovation key

### captureStructuralSizes

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => { beforeNodes: number; beforeConns: number; }`

Capture structural sizes used to evaluate operator success.

Parameters:
- `genome` - - genome to inspect

Returns: structural size snapshot

### chooseConnectionForSplit

`(enabledConnectionsList: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | null`

Choose a random enabled connection to split.

Parameters:
- `enabledConnectionsList` - - candidate connections
- `internal` - - neat controller context

Returns: selected connection or null

### choosePairForConn

`(pairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata] | null`

Choose a pair deterministically when only one candidate exists.

Parameters:
- `pairs` - - selection pool
- `internal` - - neat controller context

Returns: chosen pair or null

### chooseRandomNodeForDeadEnds

`(candidates: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata | null`

Choose a random node from candidates for dead-end repair.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### chooseRandomNodeForMinHidden

`(candidates: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata | null`

Choose a random node from a candidate list.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectCandidatePairsForConn

`(genomeToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]`

Collect legal (from,to) node pairs not already connected.

Parameters:
- `genomeToInspect` - - genome to scan

Returns: candidate node pairs

### collectEnabledConnections

`(genomeToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata[]`

Collect all enabled connections from a genome.

Parameters:
- `genomeToInspect` - - genome to inspect

Returns: enabled connections list

### collectNodeGroupsForDeadEnds

`(networkToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }`

Collect categorized node arrays for dead-end repair.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### collectNodeGroupsForMinHidden

`(networkToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }`

Collect categorized node arrays for the network.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### computeMinimumHiddenSize

`(inputCount: number, outputCount: number, explicitMinimumHidden: number | undefined, hiddenMultiplier: number | undefined) => number`

Compute the minimum hidden node count using explicit or multiplier-based settings.

Parameters:
- `inputCount` - - Number of input nodes in the network.
- `outputCount` - - Number of output nodes in the network.
- `explicitMinimumHidden` - - Optional explicit minimum hidden count.
- `hiddenMultiplier` - - Optional multiplier used when explicit minimum is absent.

Returns: Minimum hidden node requirement.

### connectChosenPair

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, pairNodes: { sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; }) => import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined`

Create the connection for the chosen pair.

Parameters:
- `genomeToEdit` - - genome to edit
- `pairNodes` - - resolved pair nodes

Returns: created connection or undefined

### connectIfCandidatesExistForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, anchorNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, candidates: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[], reverse: boolean, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Connect a node to a random candidate if candidates exist.

Parameters:
- `networkToEdit` - - network to edit
- `anchorNode` - - node to connect from/to
- `candidates` - - candidate nodes for connection
- `reverse` - - whether to connect candidate -> anchor
- `internal` - - neat controller context

Returns: void

### connectSplitEdges

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, newNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, originalWeight: number) => { incomingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; outgoingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; }`

Create the incoming and outgoing split connections.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `newNode` - - newly created hidden node
- `originalWeight` - - weight to preserve on the outgoing connection

Returns: incoming/outgoing connection handles

### createsCycle

`(sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => boolean`

Detect whether adding a connection would create a cycle.

Parameters:
- `sourceNode` - - source node of the new connection
- `targetNode` - - target node of the new connection

Returns: true when a cycle is detected

### disconnectOriginalConnection

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToRemove: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata) => void`

Disconnect the original connection before inserting the split node.

Parameters:
- `genomeToEdit` - - genome to edit
- `connectionToRemove` - - original connection to remove

Returns: void

### ensureBootstrapConnection

`(genomeToSeed: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure the genome has at least one connection by linking input to output.

Parameters:
- `genomeToSeed` - - genome that may need a bootstrap connection
- `internal` - - neat controller context

Returns: void

### ensureHiddenConnectivityForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenConnectivityForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenNodeCountForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToEdit: { hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, minimumHidden: number, maxNodesLimit: number) => Promise<void>`

Ensure the network has at least the minimum number of hidden nodes.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToEdit` - - grouped node arrays
- `minimumHidden` - - minimum hidden nodes required
- `maxNodesLimit` - - maximum allowed nodes

Returns: Promise resolving when nodes are created

### ensureIncomingConnectionForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, hiddenNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure a hidden node has at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureInputConnectivityForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure all input nodes have at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureOutgoingConnectionForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, hiddenNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure a hidden node has at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureOutputConnectivityForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure all output nodes have at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### filterPairsWithInnovations

`(pairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]`

Filter candidate pairs that already have innovation reuse keys.

Parameters:
- `pairs` - - candidate node pairs
- `internal` - - neat controller context

Returns: reuse candidates

### findFirstNodeByType

`(genomeToSearch: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeType: "input" | "output" | "hidden") => import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata | undefined`

Find the first node of a given type.

Parameters:
- `genomeToSearch` - - genome whose nodes are searched
- `nodeType` - - node type to match

Returns: the first matching node or undefined

### hasIncomingForDeadEnds

`(node: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => boolean`

Check whether a node has any incoming connections.

Parameters:
- `node` - - node to inspect

Returns: true when incoming connections exist

### hasOutgoingForDeadEnds

`(node: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => boolean`

Check whether a node has any outgoing connections.

Parameters:
- `node` - - node to inspect

Returns: true when outgoing connections exist

### hasRequiredEndpointsForMinHidden

`(nodeGroupsToCheck: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }) => boolean`

Check whether the network has at least one input and output node.

Parameters:
- `nodeGroupsToCheck` - - grouped node arrays

Returns: true when inputs and outputs are present

### initializeAdaptiveMutation

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Initialize per-genome adaptive mutation parameters if configured.

Parameters:
- `genome` - - genome to initialize
- `internal` - - neat controller context

Returns: void

### isBlockedByRecurrentPolicyForSelect

`(mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => boolean`

Check whether a mutation is blocked by recurrent connection policy.

Parameters:
- `mutationMethod` - - mutation operator to check
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isBlockedByStructuralLimitsForSelect

`(mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => boolean`

Check whether a mutation is blocked by structural limits.

Parameters:
- `mutationMethod` - - mutation operator to check
- `genome` - - genome to inspect
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isLegacyFFWPoolForSelect

`(configuredPool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], methods: { mutation: unknown; }) => boolean`

Check whether a pool matches the legacy FFW operator ordering.

Parameters:
- `configuredPool` - - configured operator pool
- `methods` - - methods module

Returns: true when the pool matches FFW

### isOperatorNamePrefixedForSelect

`(method: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, prefix: string) => boolean`

Check whether an operator name uses a specific prefix.

Parameters:
- `method` - - mutation operator
- `prefix` - - name prefix to match

Returns: true when the operator name matches the prefix

### maybeAddExtraConnection

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Optionally add an extra connection to increase exploration.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context

Returns: void

### MINIMUM_HIDDEN_BASELINE

### mutateGenome

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => Promise<void>`

Mutate a single genome based on configured mutation policies.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: Promise resolving after mutation attempts complete

### normalizeMutationPoolForSelect

`(internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }, rawReturnForTest: boolean) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[]`

Normalize the configured mutation pool to a flat operator list.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW for tests

Returns: normalized mutation pool

### rebuildNetworkConnectionsForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => Promise<void>`

Rebuild connection caches after structural edits.

Parameters:
- `networkToEdit` - - network to rebuild

Returns: Promise resolving after rebuild completes

### resolveEffectiveAmount

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the effective mutation amount for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation amount

### resolveEffectiveRate

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the effective mutation rate for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation rate

### resolveFFWPolicyForSelect

`(internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }, rawReturnForTest: boolean) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[] | null`

Resolve legacy FFW policy behavior, including test-specific returns.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW array for tests

Returns: mutation method or null when not handled

### resolveInsertIndex

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => number`

Resolve the insertion index for a new node, keeping outputs at the end.

Parameters:
- `genomeToEdit` - - genome whose node list is updated
- `targetNode` - - original target node of the split connection

Returns: insertion index

### resolveMaxNodesForMinHidden

`(internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the maximum node limit for the network.

Parameters:
- `internal` - - neat controller context

Returns: maximum node limit

### resolveMinHiddenForMinHidden

`(networkToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, maxNodesLimit: number, multiplier: number | undefined, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the minimum hidden node requirement for the network.

Parameters:
- `networkToInspect` - - network to inspect
- `maxNodesLimit` - - maximum allowed nodes
- `multiplier` - - optional size multiplier
- `internal` - - neat controller context

Returns: minimum hidden node count

### resolvePairNodes

`(chosenPair: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata]) => { sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }`

Resolve nodes and innovation key details for a chosen pair.

Parameters:
- `chosenPair` - - pair to connect

Returns: resolved pair metadata

### sampleFromPoolForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | null`

Sample a random method from the pool.

Parameters:
- `pool` - - operator pool
- `internal` - - neat controller context

Returns: sampled method or null

### selectConcreteMutationMethod

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => Promise<import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | null>`

Select a concrete mutation method, resolving any legacy arrays.

Parameters:
- `genome` - - genome to select for
- `internal` - - neat controller context

Returns: resolved mutation method or null

### selectPairPool

`(allPairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][], reusePairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]`

Build the final selection pool based on reuse and hidden-node preference.

Parameters:
- `allPairs` - - all candidate pairs
- `reusePairs` - - pairs with historical innovations

Returns: selection pool

### shouldAbortForCycle

`(genomeToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, pairNodes: { sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; }) => boolean`

Determine whether adding the connection would create a cycle.

Parameters:
- `genomeToInspect` - - genome to inspect
- `pairNodes` - - resolved pair nodes

Returns: true if the connection should be aborted

### shouldInvalidateCaches

`(mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, methods: { mutation: unknown; }) => boolean`

Determine whether a mutation method invalidates cached structures.

Parameters:
- `mutationMethod` - - mutation operator to inspect
- `methods` - - mutation methods module

Returns: true when caches should be invalidated

### shouldMutateGenome

`(effectiveRate: number, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => boolean`

Decide whether a genome should be mutated based on probability.

Parameters:
- `effectiveRate` - - effective mutation probability
- `internal` - - neat controller context

Returns: true when the genome should be mutated

### updateOperatorStatsIfNeeded

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, beforeSizes: { beforeNodes: number; beforeConns: number; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Update operator statistics when adaptation is enabled.

Parameters:
- `genome` - - genome used to compute after-sizes
- `mutationMethod` - - operator being recorded
- `beforeSizes` - - structural sizes captured before mutation
- `internal` - - neat controller context

Returns: void

### warnMissingEndpointsForMinHidden

`() => void`

Emit a warning when the network lacks input or output nodes.

Returns: void

## neat/neat.diversity.utils.ts

### arrayMean

`(values: number[]) => number`

Compute the arithmetic mean of a numeric array. Returns 0 for empty arrays.

Parameters:
- `values` - - Values to average.

Returns: Arithmetic mean of the values.

### arrayVariance

`(values: number[]) => number`

Compute the variance (population variance) of a numeric array.
Returns 0 for empty arrays. Uses arrayMean internally.

Parameters:
- `values` - - Values to evaluate.

Returns: Population variance.

### calculateDiversityStats

`(population: import("C:/NeatapticTS/src/neat/neat.diversity.utils").GenomeWithMetrics[], compatibilityComputer: import("C:/NeatapticTS/src/neat/neat.diversity.utils").CompatComputer) => import("C:/NeatapticTS/src/neat/neat.diversity.utils").DiversityStats | undefined`

Compute diversity statistics for a NEAT population.

Parameters:
- `population` - - array of genome-like objects (nodes, connections, optional _depth)
- `compatibilityComputer` - - object exposing _compatibilityDistance(a,b)

Returns: DiversityStats object with all computed aggregates, or undefined if input empty

### calculateStructuralEntropy

`(graph: import("C:/NeatapticTS/src/architecture/network").default) => number`

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

### MAX_LINEAGE_PAIR_SAMPLE

### NodeWithConnections

Minimal node interface with connections.

## neat/neat.selection.utils.ts

### calculateFitnessTotals

`(population: import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore[]) => { totalFitness: number; minFitnessShift: number; }`

Compute the total fitness and minimal score shift for roulette selection.

Parameters:
- `population` - - Genomes in the current population.

Returns: Aggregated fitness totals.

### calculateTotalScore

`(population: import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore[]) => number`

Calculate the total fitness across the population.

Parameters:
- `population` - - Genomes in the current population.

Returns: The sum of all scores.

### DEFAULT_POWER

### DEFAULT_SCORE

### DEFAULT_TOURNAMENT_PROBABILITY

### DEFAULT_TOURNAMENT_SIZE

### ensurePopulationEvaluated

`(internal: import("C:/NeatapticTS/src/neat/neat.selection.utils").NeatLikeWithSelection) => void`

Ensure population scores exist by running evaluation if needed.

Parameters:
- `internal` - - The Neat instance containing `population` and `evaluate`.

Returns: void

### ensurePopulationSortedDescending

`(internal: import("C:/NeatapticTS/src/neat/neat.selection.utils").NeatLikeWithSelection) => void`

Ensure the population is sorted descending by score when out of order.

Parameters:
- `internal` - - The Neat instance containing `population`.

Returns: void

### ensurePopulationSortedDescendingForPower

`(selectionContext: SelectionContext) => void`

Ensure the population is sorted descending by score if the first two
entries are out of order.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: void

### FIRST_INDEX

### GenomeWithScore

Genome with a fitness score and arbitrary additional metadata.

### getRandomPopulationMember

`(selectionContext: SelectionContext) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Select a random population member using the configured RNG.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: A randomly chosen genome.

### INITIAL_CUMULATIVE_FITNESS

### INITIAL_MOST_NEGATIVE_SCORE

### INITIAL_TOTAL_FITNESS

### LAST_ELEMENT_INDEX

### LAST_INDEX_OFFSET

### LOOP_INDEX_INCREMENT

### NeatLikeWithSelection

NEAT-like instance extended with selection-specific state and helpers.

### pickByShiftedThreshold

`(population: import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore[], selectionThreshold: number, minFitnessShift: number) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore | undefined`

Pick the first genome whose shifted cumulative fitness exceeds the threshold.

Parameters:
- `population` - - Genomes in the current population.
- `selectionThreshold` - - Random threshold in shifted fitness space.
- `minFitnessShift` - - Amount added to each score to shift negatives.

Returns: The chosen genome if one crosses the threshold.

### pickTournamentWinner

`(selectionContext: SelectionContext, sortedParticipants: import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore[]) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Select a winner from sorted tournament participants.

Parameters:
- `selectionContext` - - Shared selection state.
- `sortedParticipants` - - Participants sorted by descending score.

Returns: The chosen tournament winner.

### resolveTournamentOverflow

`(selectionContext: SelectionContext) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Resolve what happens when the tournament size exceeds population size.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: A fallback parent genome.

### sampleTournamentParticipants

`(selectionContext: SelectionContext, tournamentSize: number) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore[]`

Sample a list of tournament participants (with possible repeats).

Parameters:
- `selectionContext` - - Shared selection state.
- `tournamentSize` - - Number of competitors to sample.

Returns: Sampled participants.

### SECOND_INDEX

### selectParentByFitnessProportionate

`(selectionContext: SelectionContext) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Select a parent using roulette-wheel fitness proportionate selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

### selectParentByPower

`(selectionContext: SelectionContext) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Select a parent by power-law distribution on the sorted population.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

### selectParentByStrategy

`(internal: import("C:/NeatapticTS/src/neat/neat.selection.utils").NeatLikeWithSelection) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Select a parent genome according to configured selection strategy.

Parameters:
- `internal` - - The Neat instance containing population and options.

Returns: A genome object chosen as the parent.

### selectParentByTournament

`(selectionContext: SelectionContext) => import("C:/NeatapticTS/src/neat/neat.selection.utils").GenomeWithScore`

Select a parent by tournament selection.

Parameters:
- `selectionContext` - - Shared selection state.

Returns: The chosen parent genome.

## neat/neat.telemetry.utils.ts

### applyComplexityStatsMonoObjective

`(telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach complexity stats for mono-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `entry` - - Telemetry entry to update.

### applyComplexityStatsMultiObjective

`(telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach complexity stats for multi-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyFastModeDefaults

`(telemetryContext: { _fastModeTuned?: boolean | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions) => void`

Apply fast-mode tuning to diversity sampling and novelty defaults.

Parameters:
- `telemetryContext` - - Context object storing fast-mode tuning flag.
- `telemetryOptions` - - Options with diversity and novelty settings.

### applyHypervolumeTelemetry

`(telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, hyperVolumeProxy: number, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach hypervolume scalar when requested.

Parameters:
- `telemetryOptions` - - Options controlling telemetry fields.
- `hyperVolumeProxy` - - Hypervolume proxy value.
- `entry` - - Telemetry entry to update.

### applyLineageStatsMonoObjective

`(telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply lineage stats for mono-objective mode using sampled ancestors.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyLineageStatsMultiObjective

`(telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; }, population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply lineage stats for multi-objective mode using ancestor uniqueness.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyObjectiveAges

`(telemetryContext: { _objectiveAges?: Map<string, number> | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply objective age snapshots to the entry.

Parameters:
- `telemetryContext` - - Neat-like context with objective ages.
- `entry` - - Telemetry entry to update.

### applyObjectiveEvents

`(telemetryContext: { _pendingObjectiveAdds?: string[] | undefined; _pendingObjectiveRemoves?: string[] | undefined; _objectiveEvents?: import("C:/NeatapticTS/src/neat/neat.types").ObjectiveEvent[] | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord, generation: number) => void`

Apply and flush objective lifecycle events.

Parameters:
- `telemetryContext` - - Neat-like context holding objective events.
- `entry` - - Telemetry entry to update.
- `generation` - - Generation index for event records.

### applyObjectiveImportance

`(telemetryContext: { _lastObjImportance?: import("C:/NeatapticTS/src/neat/neat.types").ObjImportance | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply the most recent objective importance snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with objective importance.
- `entry` - - Telemetry entry to update.

### applyObjectivesSnapshot

`(telemetryContext: { _getObjectives?: (() => { key: string; }[]) | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply objectives list snapshot (keys only).

Parameters:
- `telemetryContext` - - Neat-like context with objective provider.
- `entry` - - Telemetry entry to update.

### applyPerformanceStats

`(telemetryContext: { _lastEvalDuration?: number | undefined; _lastEvolveDuration?: number | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach performance stats when configured.

Parameters:
- `telemetryContext` - - Neat-like context with performance data.
- `telemetryOptions` - - Options controlling performance telemetry.
- `entry` - - Telemetry entry to update.

### applyRngState

`(telemetryContext: { _rngState?: unknown; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach RNG state when configured.

Parameters:
- `telemetryContext` - - Neat-like context with RNG state.
- `telemetryOptions` - - Options controlling RNG telemetry.
- `entry` - - Telemetry entry to update.

### applySpeciesAllocation

`(telemetryContext: { _lastOffspringAlloc?: import("C:/NeatapticTS/src/neat/neat.types").SpeciesAlloc[] | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply per-species offspring allocation snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with allocation snapshot.
- `entry` - - Telemetry entry to update.

### buildComplexityEntry

`(telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, meanCounts: { meanNodes: number; meanConns: number; }, maxCounts: { maxNodes: number; maxConns: number; }, meanEnabledRatio: number, growthValues: { growthNodes: number; growthConns: number; }) => { meanNodes: number; meanConns: number; maxNodes: number; maxConns: number; meanEnabledRatio: number; growthNodes: number; growthConns: number; budgetMaxNodes: number; budgetMaxConns: number; }`

Build the complexity entry payload for multi-objective mode.

Parameters:
- `telemetryOptions` - - Options controlling complexity telemetry.
- `meanCounts` - - Mean node/connection counts.
- `maxCounts` - - Max node/connection counts.
- `meanEnabledRatio` - - Mean enabled ratio.
- `growthValues` - - Growth deltas.

Returns: Complexity entry payload.

### buildDegreeHistogram

`(counts: Record<number, number>) => Record<number, number>`

Build a histogram of degree frequencies from a degree-count table.

Parameters:
- `counts` - - Map geneId -> degree count.

Returns: Map degree -> number of nodes with that degree.

### buildLineageContext

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => import("C:/NeatapticTS/src/neat/neat.lineage.utils").NeatLineageContext`

Build a lineage helper context for ancestor operations.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Lineage helper context.

### buildLineageEntry

`(context: { _prevInbreedingCount?: number | undefined; }, bestGenomeSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed, meanDepthValue: number, ancestorUniquenessScore: number) => { parents: number[]; depthBest: number; meanDepth: number; inbreeding: number; ancestorUniq: number; }`

Build the lineage entry payload.

Parameters:
- `context` - - Neat-like context with lineage info.
- `bestGenomeSnapshot` - - Best genome snapshot.
- `meanDepthValue` - - Mean lineage depth.
- `ancestorUniquenessScore` - - Ancestor uniqueness score.

Returns: Lineage entry payload.

### buildLineageSnapshot

`(population: { _id?: number | undefined; _parents?: number[] | undefined; }[], limit: number) => { id: number; parents: number[]; }[]`

Snapshot lineage metadata for the first `limit` genomes.

### clearTelemetryBuffer

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => void`

Clear the telemetry buffer in place.

### collectDepths

`(populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number[]`

Collect depth values for the current population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of depth values (defaults to 0).

### collectPopulationCounts

`(populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => { nodeCounts: number[]; connectionCounts: number[]; }`

Collect node and connection counts for the population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Node and connection counts arrays.

### computeAncestorUniquenessSampled

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number`

Compute ancestor uniqueness using sampled Jaccard distance.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Rounded ancestor uniqueness score.

### computeAndStoreGrowthValues

`(context: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; }, meanCounts: { meanNodes: number; meanConns: number; }) => { growthNodes: number; growthConns: number; }`

Compute growth values and store the latest means on the context.

Parameters:
- `context` - - Neat-like context with previous mean values.
- `meanCounts` - - Current mean node/connection counts.

Returns: Growth values for nodes and connections.

### computeCompatibilityStats

`(genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], size: number, pairSampleCount: number, rngFactoryFn: () => () => number, compatibilityDistance: ((a: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome, b: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome) => number) | undefined) => { meanCompat: number; varCompat: number; }`

Compute pairwise compatibility statistics via sampling.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.
- `compatibilityDistance` - - Optional compatibility distance function.

Returns: Mean and variance of sampled compatibilities.

### computeDegreeCounts

`(entropyGraph: { nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; }) => Record<number, number>`

Compute per-node degree counts for enabled connections.

Parameters:
- `entropyGraph` - - Genome-like graph object.

Returns: Map geneId -> degree count.

### computeEnabledRatios

`(populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number[]`

Compute enabled ratios per genome.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of enabled ratios.

### computeEntropyFromHistogram

`(histogram: Record<number, number>, totalNodes: number) => number`

Compute entropy from a degree-frequency histogram.

Parameters:
- `histogram` - - Map degree -> number of nodes.
- `totalNodes` - - Total node count used to normalize into probabilities.

Returns: Entropy value (non-negative).

### computeEntropyStats

`(genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], structuralEntropyFn: (genome: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome) => number) => { meanEntropy: number; varEntropy: number; }`

Compute structural entropy mean and variance across the population.

Parameters:
- `genomes` - - Population snapshot.
- `structuralEntropyFn` - - Function to compute entropy for a genome.

Returns: Mean and variance of entropy values.

### computeGraphletEntropy

`(genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], size: number, graphletSampleCount: number, rngFactoryFn: () => () => number) => number`

Sample graphlet motifs and compute entropy over their edge counts.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `graphletSampleCount` - - Number of graphlets to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Graphlet entropy value.

### computeHyperVolumeProxy

`(telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number`

Compute a hypervolume-like proxy for the Pareto front.

Parameters:
- `telemetryOptions` - - Options controlling complexity metric.
- `population` - - Population snapshot.

Returns: Hypervolume proxy value.

### computeLineageStats

`(lineageEnabled: boolean, genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], size: number, pairSampleCount: number, rngFactoryFn: () => () => number) => { lineageMeanDepth: number; lineageMeanPairDist: number; }`

Compute lineage depth and pairwise depth-distance statistics.

Parameters:
- `lineageEnabled` - - Whether lineage metrics are enabled.
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Lineage mean depth and pairwise distance.

### computeMaxCounts

`(counts: { nodeCounts: number[]; connectionCounts: number[]; }) => { maxNodes: number; maxConns: number; }`

Compute max node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Max node and connection counts.

### computeMeanCounts

`(counts: { nodeCounts: number[]; connectionCounts: number[]; }) => { meanNodes: number; meanConns: number; }`

Compute mean node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Mean node and connection counts.

### computeMeanDepth

`(depthValues: number[]) => number`

Compute the mean depth from a depth list.

Parameters:
- `depthValues` - - Depth values to average.

Returns: Mean depth value.

### computeMeanEnabledRatio

`(enabledRatios: number[]) => number`

Compute mean of enabled ratios.

Parameters:
- `enabledRatios` - - Enabled ratios per genome.

Returns: Mean enabled ratio.

### computeOperatorStatsSnapshot

`(operatorStats: import("C:/NeatapticTS/src/neat/neat.telemetry.types").OperatorStatsMap | undefined) => { op: string; succ: number; att: number; }[]`

Snapshot operator statistics into a telemetry-friendly array.

Parameters:
- `operatorStats` - - Operator stats map (opName -> success/attempts).

Returns: Operator stats snapshot array.

### computePairJaccardDistance

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], firstIndex: number, secondIndex: number) => number | undefined`

Compute Jaccard distance between ancestor sets for a pair.

Parameters:
- `context` - - Neat-like context for lineage helpers.
- `populationSnapshot` - - Population snapshot.
- `firstIndex` - - First genome index.
- `secondIndex` - - Second genome index.

Returns: Jaccard distance or undefined when both sets are empty.

### computeParetoFrontSizes

`(population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number[]`

Compute sizes of early Pareto fronts.

Parameters:
- `population` - - Population snapshot.

Returns: Array of front sizes (rank 0..4).

### countAncestorIntersection

`(ancestorsA: Set<number>, ancestorsB: Set<number>) => number`

Count the size of an ancestor intersection.

Parameters:
- `ancestorsA` - - First ancestor set.
- `ancestorsB` - - Second ancestor set.

Returns: Intersection count.

### countEnabledEdges

`(genome: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome, selectedNodes: import("C:/NeatapticTS/src/neat/neat.types").NodeLike[]) => number`

Count enabled edges between the selected nodes in a genome.

Parameters:
- `genome` - - Genome with connections to inspect.
- `selectedNodes` - - Nodes forming the graphlet sample.

Returns: Edge count capped at 3.

### ensureTelemetryBuffer

`(telemetryContext: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryBufferContext) => import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[]`

Ensure the telemetry buffer is initialized.

Parameters:
- `telemetryContext` - - Neat-like context holding telemetry buffer.

Returns: A mutable telemetry buffer.

### getCachedDiversityStats

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => import("C:/NeatapticTS/src/neat/neat.diversity.utils").DiversityStats | undefined`

Read cached diversity statistics.

### getCachedEntropy

`(generation: number | undefined, entropyGraph: Record<string, unknown>) => number | undefined`

Read a cached entropy value if it exists and belongs to the current
generation.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.

Returns: Cached entropy number, or undefined when not available.

### getObjectiveEventsSnapshot

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => { gen: number; type: "add" | "remove"; key: string; }[]`

Return a shallow copy of recent objective events.

### getPerformanceStatsSnapshot

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }`

Snapshot performance timings for evaluation and evolution steps.

### getTelemetryBuffer

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[]`

Return the telemetry buffer, defaulting to an empty array when missing.

### getTelemetryCoreSnapshot

`(sourceEntry: Record<string, unknown>, fields: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryCoreFields) => Partial<Record<string, unknown>>`

Build a snapshot of the core telemetry fields present on the entry; does
not mutate the source entry.

Parameters:
- `sourceEntry` - - Source telemetry object.
- `fields` - - Core telemetry field keys to preserve.

Returns: Shallow snapshot of core fields that exist on the entry.

### isLineageEligible

`(context: { _lineageEnabled?: boolean | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => boolean`

Check whether lineage metrics should be computed.

Parameters:
- `context` - - Neat-like context with lineage flag.
- `populationSnapshot` - - Population snapshot to validate.

Returns: True when lineage stats should be computed.

### LINEAGE_SNAPSHOT_DEFAULT_LIMIT

### mergeTelemetryCoreFields

`(sourceEntry: Record<string, unknown>, coreSnapshot: Partial<Record<string, unknown>>) => Record<string, unknown>`

Re-attach core fields to the filtered entry.
Mutates the entry so the caller keeps the original reference.

Parameters:
- `sourceEntry` - - Filtered telemetry entry to update.
- `coreSnapshot` - - Snapshot of core fields to ensure presence.

Returns: The same entry reference with core fields restored.

### OperatorStatsMap

Operator stats map shape for telemetry extraction.

### pickDistinctIndices

`(upperBound: number, count: number, rng: () => number) => number[]`

Pick a fixed number of distinct random indices.

Parameters:
- `upperBound` - - Exclusive upper bound for random indices.
- `count` - - Number of distinct indices to pick.
- `rng` - - RNG function returning values in [0,1).

Returns: Array of distinct indices.

### pickDistinctPairIndices

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSize: number) => { firstIndex: number; secondIndex: number; }`

Pick two distinct indices using the context RNG.

Parameters:
- `context` - - Neat-like context with RNG factory.
- `populationSize` - - Population size for index bounds.

Returns: Pair of distinct indices.

### readOperatorStats

`(operatorStats: import("C:/NeatapticTS/src/neat/neat.telemetry.types").OperatorStatsMap | undefined) => { name: string; success: number; attempts: number; }[]`

Convert operator stats map into the public accessor shape.

### safelyApplyTelemetrySelect

`(telemetryContext: TContext, telemetryEntry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, applyTelemetrySelectFn: (this: TContext, entry: Record<string, unknown>) => Record<string, unknown>) => void`

Apply telemetry selection while swallowing any selection errors.

Parameters:
- `telemetryContext` - - Neat-like context with telemetry selection.
- `telemetryEntry` - - Entry to filter in place.
- `applyTelemetrySelectFn` - - Selection helper to invoke.

### safelyStreamTelemetryEntry

`(telemetryContext: { options?: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryStreamOptions | undefined; }, telemetryEntry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry) => void`

Stream telemetry entry when a stream callback is configured.

Parameters:
- `telemetryContext` - - Neat-like context with stream settings.
- `telemetryEntry` - - Entry to stream.

### setCachedEntropy

`(generation: number | undefined, entropyGraph: Record<string, unknown>, entropyValue: number) => void`

Cache an entropy value for the current generation on the graph object.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.
- `entropyValue` - - Entropy value to cache.

### stripUnselectedTelemetryKeys

`(sourceEntry: Record<string, unknown>, selection: Set<string>, fields: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryCoreFields) => Record<string, unknown>`

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

`(telemetryBufferRef: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[], maxEntries: number) => void`

Trim the telemetry buffer to a maximum size.

Parameters:
- `telemetryBufferRef` - - Buffer to trim in-place.
- `maxEntries` - - Maximum entries to keep.

## neat/neat.objectives.utils.ts

### buildDefaultFitnessObjective

`() => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor`

Returns: Default fitness objective descriptor.

### collectDefaultObjectives

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.objectives.utils").NeatLikeWithObjectives) => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Default objectives when fitness is not suppressed.

### collectUserObjectives

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.objectives.utils").NeatLikeWithObjectives) => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Valid user-registered objectives when multi-objective is enabled.

### ensureMultiObjectiveOptions

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.objectives.utils").NeatLikeWithObjectives) => { enabled?: boolean | undefined; objectives?: import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[] | undefined; }`

Parameters:
- `neatInstance` - - Instance receiving the multi-objective container.

Returns: Initialized multi-objective options.

### ensureObjectivesList

`(multiObjectiveOptions: { enabled?: boolean | undefined; objectives?: import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[] | undefined; }) => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Parameters:
- `multiObjectiveOptions` - - Multi-objective container to hydrate.

Returns: Objectives list for mutation-free operations.

### getObjectiveCandidates

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.objectives.utils").NeatLikeWithObjectives) => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Candidate objectives from configuration.

### isMultiObjectiveEnabled

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.objectives.utils").NeatLikeWithObjectives) => boolean`

Parameters:
- `neatInstance` - - Instance providing objective settings.

Returns: Whether multi-objective mode is enabled with a candidate list.

### isValidObjective

`(candidateObjective: import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor | undefined) => boolean`

Parameters:
- `candidateObjective` - - Candidate descriptor to validate.

Returns: True when the descriptor has the required shape.

### NeatLikeWithObjectives

Minimal interface for NEAT instances using objective management.

This shape is intentionally small and only includes the pieces needed by
`_getObjectives`, `registerObjective`, and `clearObjectives`.

### replaceObjectiveByKey

`(objectivesList: import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[], objectiveKey: string, objectiveDirection: "max" | "min", objectiveAccessor: (genome: import("C:/NeatapticTS/src/neat/neat.types").GenomeLike) => number) => import("C:/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Parameters:
- `objectivesList` - - Existing objectives to update.
- `objectiveKey` - - Key to replace.
- `objectiveDirection` - - Direction for the new objective.
- `objectiveAccessor` - - Accessor for the new objective.

Returns: Updated objectives list with the new descriptor appended.

## neat/neat.speciation.utils.ts

### neat.speciation.utils

Utility helpers for NEAT speciation orchestration.

### adjustCompatibilityThreshold

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, options: TOptions, compatAdjust: { smoothingWindow?: number | undefined; decay?: number | undefined; kp?: number | undefined; ki?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; }, minCompatibilityThreshold: number, maxCompatibilityThreshold: number) => void`

Update the adaptive compatibility threshold and clamp to bounds.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `compatAdjust` - - Compatibility adjustment settings.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Nothing.

### applyAgeProtection

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, options: TOptions) => void`

Apply age protection penalties to old species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### applyFitnessSharing

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.speciation.utils").FitnessSharingContext, sharingSigma: number) => void`

Apply fitness sharing to penalize similarity within species.

Parameters:
- `speciationContext` - - Neat instance context with species and distance function.
- `sharingSigma` - - Sharing radius used for distance weighting.

Returns: Nothing.

### assignPopulationToSpecies

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, options: TOptions) => void`

Assign each genome in the population to a compatible species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### averageNumbers

`(values: number[]) => number`

Average a list of numbers, returning zero when empty.

Parameters:
- `values` - - Numeric values to average.

Returns: Mean of the values or zero.

### buildExtendedHistoryStats

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, species: import("C:/NeatapticTS/src/neat/neat.types").SpeciesLike) => Record<string, unknown>`

Build extended history stats for a species.

Parameters:
- `speciationContext` - - Speciation harness context.
- `species` - - Species to snapshot.

Returns: Extended history entry.

### clampCompatibilityThreshold

`(options: import("C:/NeatapticTS/src/neat/neat.types").SpeciationOptions, minCompatibilityThreshold: number, maxCompatibilityThreshold: number) => void`

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

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, options: TOptions, compatAdjust: { smoothingWindow?: number | undefined; decay?: number | undefined; kp?: number | undefined; ki?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; }, currentThreshold: number, minCompatibilityThreshold: number, maxCompatibilityThreshold: number) => number`

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

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, genome: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed) => void`

Create a new species for the provided genome.

Parameters:
- `speciationContext` - - Speciation harness context.
- `genome` - - Genome that starts a new species.

Returns: Nothing.

### DEFAULT_COMPAT_INTEGRAL

### DEFAULT_COMPATIBILITY_INTEGRAL_GAIN

### DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN

### DEFAULT_COMPATIBILITY_THRESHOLD

### DEFAULT_LAST_IMPROVED_GENERATION

### DEFAULT_MAX_COMPATIBILITY_THRESHOLD

### DEFAULT_MEMBER_COUNT_FALLBACK

### DEFAULT_MIN_COMPATIBILITY_THRESHOLD

### DEFAULT_SCORE_FALLBACK

### DEFAULT_SHARING_SIGMA

### DEFAULT_SPECIES_AGE_GRACE

### DEFAULT_SPECIES_OLD_PENALTY

### DEFAULT_STAGNATION_WINDOW

### DEFAULT_TARGET_SPECIES

### findCompatibleSpecies

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, options: TOptions, genome: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed) => import("C:/NeatapticTS/src/neat/neat.types").SpeciesLike | undefined`

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

### InnovationAccumulator

Accumulator for innovation-id statistics across a set of connections.

Used for extended history telemetry (mean innovation, innovation range, and
enabled/disabled ratios).

### NEGATIVE_INFINITY

### PENALTY_NO_EFFECT_THRESHOLD

### recordHistory

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, options: TOptions) => void`

Record the current species history snapshot.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.

Returns: Nothing.

### refreshSpeciesRepresentatives

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>) => void`

Refresh representatives and remove empty species.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### resetSpeciesMembers

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>) => void`

Clear member lists for all species.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### SHARING_MAX_CONTRIBUTION

### SHARING_SELF_DISTANCE

### SHARING_SUM_FLOOR

### snapshotPreviousMembers

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>) => void`

Snapshot current species memberships for telemetry.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### SPECIES_AGE_GRACE_MULTIPLIER

### StagnationContext

Minimal context required to update species stagnation.

Stagnation pruning removes species that have not improved their best score
within a configured number of generations.

### summarizeInnovations

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>, members: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => { meanInnovation: number; innovationRange: number; enabledRatio: number; }`

Summarize innovation statistics for a set of members.

Parameters:
- `speciationContext` - - Speciation harness context.
- `members` - - Members to summarize.

Returns: Innovation summary statistics.

### trimHistory

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.types").SpeciationHarnessContext<TOptions>) => void`

Trim species history to the maximum buffer size.

Parameters:
- `speciationContext` - - Speciation harness context.

Returns: Nothing.

### updateSpeciesStagnation

`(speciationContext: import("C:/NeatapticTS/src/neat/neat.speciation.utils").StagnationContext, stagnationWindow: number, sortSpeciesMembers: (species: import("C:/NeatapticTS/src/neat/neat.types").SpeciesLike) => void) => void`

Update stagnation counters and prune stagnant species.

Parameters:
- `speciationContext` - - Neat instance context with species array and generation counter.
- `stagnationWindow` - - Allowed stagnation window.
- `sortSpeciesMembers` - - Sort function for species members.

Returns: Nothing.

## neat/neat.mutation.flow.utils.ts

### applyAddConnMutation

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => void`

Apply an ADD_CONN mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyAddNodeMutation

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => void`

Apply an ADD_NODE mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyMutationOperator

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => void`

Apply a mutation operator to a genome and invalidate caches as needed.

Parameters:
- `genome` - - genome to mutate
- `mutationMethod` - - mutation operator to apply
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### captureStructuralSizes

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => { beforeNodes: number; beforeConns: number; }`

Capture structural sizes used to evaluate operator success.

Parameters:
- `genome` - - genome to inspect

Returns: structural size snapshot

### initializeAdaptiveMutation

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Initialize per-genome adaptive mutation parameters if configured.

Parameters:
- `genome` - - genome to initialize
- `internal` - - neat controller context

Returns: void

### maybeAddExtraConnection

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Optionally add an extra connection to increase exploration.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context

Returns: void

### mutateGenome

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => Promise<void>`

Mutate a single genome based on configured mutation policies.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: Promise resolving after mutation attempts complete

### resolveEffectiveAmount

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the effective mutation amount for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation amount

### resolveEffectiveRate

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the effective mutation rate for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation rate

### selectConcreteMutationMethod

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => Promise<import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | null>`

Select a concrete mutation method, resolving any legacy arrays.

Parameters:
- `genome` - - genome to select for
- `internal` - - neat controller context

Returns: resolved mutation method or null

### shouldInvalidateCaches

`(mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, methods: { mutation: unknown; }) => boolean`

Determine whether a mutation method invalidates cached structures.

Parameters:
- `mutationMethod` - - mutation operator to inspect
- `methods` - - mutation methods module

Returns: true when caches should be invalidated

### shouldMutateGenome

`(effectiveRate: number, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => boolean`

Decide whether a genome should be mutated based on probability.

Parameters:
- `effectiveRate` - - effective mutation probability
- `internal` - - neat controller context

Returns: true when the genome should be mutated

### updateOperatorStatsIfNeeded

`(genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, beforeSizes: { beforeNodes: number; beforeConns: number; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Update operator statistics when adaptation is enabled.

Parameters:
- `genome` - - genome used to compute after-sizes
- `mutationMethod` - - operator being recorded
- `beforeSizes` - - structural sizes captured before mutation
- `internal` - - neat controller context

Returns: void

## neat/neat.telemetry.rng.utils.ts

### applyRngState

`(telemetryContext: { _rngState?: unknown; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach RNG state when configured.

Parameters:
- `telemetryContext` - - Neat-like context with RNG state.
- `telemetryOptions` - - Options controlling RNG telemetry.
- `entry` - - Telemetry entry to update.

## neat/neat.evolve.runtime.utils.ts

### buildFittestSnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => import("C:/NeatapticTS/src/architecture/network").default`

Build a cloned Network from the current best genome.

Parameters:
- `internal` - - NEAT controller instance.

Returns: best network snapshot.

### clearPopulationScores

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Clear genome scores to force re-evaluation.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### computeElapsedTime

`(startTimestamp: number) => number`

Compute elapsed time since the start of evolve().

Parameters:
- `startTimestamp` - - Start time resolved earlier.

Returns: elapsed time.

### ensurePopulationEvaluated

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Ensure the population is evaluated before evolution operations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### resolveStartTime

`() => number`

Resolve the start time for an evolution step.

Returns: timestamp in milliseconds or high-resolution units.

### trackGlobalImprovement

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, snapshot: import("C:/NeatapticTS/src/architecture/network").default) => void`

Track global best improvement for stagnation logic.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot.

Returns: void.

### updateGlobalBestTracking

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Update generation-level best score tracking.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

## neat/neat.multiobjective.utils.ts

### neat.multiobjective.utils

Barrel exports for multi-objective utilities.

### accumulateCrowdingForObjective

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[], valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndex: number) => void`

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

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").NeatLikeWithMultiObjective, fronts: import("C:/NeatapticTS/src/architecture/network").default[][]) => void`

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

`(fronts: import("C:/NeatapticTS/src/architecture/network").default[][], valuesMatrixInput: number[][], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[], population: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

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

`(valuesMatrixInput: number[][], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState`

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

`(population: import("C:/NeatapticTS/src/architecture/network").default[]) => Map<import("C:/NeatapticTS/src/architecture/network").default, number>`

Builds a stable mapping from genome object references to their population
index.

This relies on object identity (reference equality), not structural
equality. It is used to resolve objective values from a values matrix when
working with reordered views (e.g., sorted fronts).

Parameters:
- `population` - - Genomes in population order.

Returns: Map from genome references to their index.

### buildGenomeValues

`(genomeItem: import("C:/NeatapticTS/src/architecture/network").default, descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => number[]`

Builds an objective vector for a single genome.

The resulting array order matches the `descriptors` order exactly.
Each component is read via {@link readObjectiveValue} so individual
objective accessors are fault-tolerant.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptors` - - Objective descriptors (vector schema).

Returns: Objective value vector (length equals `descriptors.length`).

### buildMultiObjectiveMetrics

`(population: import("C:/NeatapticTS/src/architecture/network").default[]) => { rank: any; crowding: any; score: number; nodes: number; connections: number; }[]`

Build lightweight multi-objective metrics for each genome in the population.

### buildParetoFronts

`(population: import("C:/NeatapticTS/src/architecture/network").default[], dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, maxFrontRankGuard: number) => import("C:/NeatapticTS/src/architecture/network").default[][]`

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

`(population: import("C:/NeatapticTS/src/architecture/network").default[], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => number[][]`

Builds a population-wide objective value matrix.

The resulting matrix is indexed as `[genomeIndex][objectiveIndex]` where
`genomeIndex` matches the input `population` order.

Parameters:
- `population` - - Genomes to evaluate (population order is preserved).
- `descriptors` - - Objective descriptors (column schema).

Returns: Objective values matrix.

### DEFAULT_MAX_PARETO_FRONTS

### DEFAULT_PARETO_ARCHIVE_JSONL_MAX

### DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES

### DominanceState

Dominance bookkeeping structures for fast non-dominated sorting.

These structures are typically produced once per generation (from the values
matrix) and then consumed to build Pareto fronts.

### exportParetoArchiveJsonl

`(archive: unknown[], maxEntries: number) => string`

Export a Pareto archive slice as JSON Lines.

### initializeCrowding

`(front: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Initializes crowding-distance annotations for a front.

This sets each genome’s `_moCrowd` to `0`. Later steps accumulate per-
objective spacing deltas.

Parameters:
- `front` - - Pareto front.

### markBoundaryCrowding

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Marks the boundary genomes of a sorted front as infinitely crowded.

In NSGA-II style crowding distance, boundary solutions (extremes for the
objective) are assigned an infinite crowding distance to ensure they are
always preferred when ranks tie.

Parameters:
- `sortedFront` - - Front sorted by the current objective.

### MAX_PARETO_ARCHIVE_FRONTS

### MAX_PARETO_ARCHIVE_LENGTH

### MAX_PARETO_FRONT_RANK_GUARD

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

### readObjectiveValue

`(genomeItem: import("C:/NeatapticTS/src/architecture/network").default, descriptor: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor) => number`

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

### reconstructParetoFronts

`(population: import("C:/NeatapticTS/src/architecture/network").default[], maxFronts: number, isMultiObjectiveEnabled: boolean) => import("C:/NeatapticTS/src/architecture/network").default[][]`

Reconstruct Pareto fronts from stored rank annotations.

### resolveGenomeIndex

`(genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, genomeItem: import("C:/NeatapticTS/src/architecture/network").default) => number`

Resolves a genome’s index using a reference-based map.

Parameters:
- `genomeIndexByReference` - - Lookup map created by
 *  {@link buildGenomeIndexByReference} .
 *
- `genomeItem` - - Genome to resolve.

Returns: The population index of the genome.

### resolveObjectiveRange

`(minValue: number, maxValue: number) => number`

Resolves a non-zero objective range used to normalize crowding deltas.

If all genomes have the same objective value, the raw range is `0`. This
returns `1` in that case to avoid division by zero while still producing a
well-defined crowding delta of `0`.

Parameters:
- `minValue` - - Minimum objective value.
- `maxValue` - - Maximum objective value.

Returns: Normalized range with a non-zero floor.

### resolveObjectiveValue

`(valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, genomeItem: import("C:/NeatapticTS/src/architecture/network").default, objectiveIndex: number) => number`

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

`(archive: T[], maxEntries: number) => T[]`

Return the most recent Pareto archive entries up to the provided limit.

### vectorDominates

`(valuesA: number[], valuesB: number[], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => boolean`

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

## neat/neat.adaptive.phases.utils.ts

### initializePhaseState

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; }) => void`

Ensure phase state is initialized.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### resolveNextPhase

`(currentPhase: string) => string`

Resolve next phase name.

Parameters:
- `currentPhase` - - Current phase label.

Returns: Next phase label.

### togglePhaseIfNeeded

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; }) => void`

Toggle phase if the current phase has exceeded its length.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

## neat/neat.evolve.adaptive.utils.ts

### adaptReenableProbability

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { minSamples: number; target: number; min: number; max: number; deltaScale: number; }) => void`

Adapt the re-enable probability based on recent success ratios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAdaptiveComplexityControllers

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply adaptive complexity controllers if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAncestorUniqAdaptiveSafe

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply ancestor uniqueness adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyAutoCompatibilityTuning

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { targetMin: number; adjustRate: number; minCoeff: number; maxCoeff: number; randomScale: number; }) => void`

Apply auto-compatibility tuning if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Tuning constants.

Returns: void.

### applyMinimalCriterionAdaptiveSafe

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply minimal criterion adaptive controller if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyOperatorAdaptationSafe

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply operator adaptation if available.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### applyPruningAndMutation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<void>`

Apply pruning and mutation phases.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### invalidateCompatibilityCaches

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Invalidate compatibility caches after mutations.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

## neat/neat.evolve.warnings.utils.ts

### neat.evolve.warnings.utils

Warning emitted when evolution finishes without a best genome.

### EVOLVE_NO_BEST_GENOME_WARNING

### warnIfNoBestGenome

`() => void`

Emit the standard warning for runs that end without a valid best genome.

## neat/neat.mutation.select.utils.ts

### applyOperatorAdaptationForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[]`

Apply operator adaptation weighting to the pool when enabled.

Parameters:
- `pool` - - base pool
- `internal` - - neat controller context

Returns: augmented pool

### applyOperatorBanditForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], fallbackMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod`

Apply operator bandit selection if enabled.

Parameters:
- `pool` - - operator pool
- `fallbackMethod` - - method used when bandit is disabled
- `internal` - - neat controller context

Returns: selected method

### applyPhasedComplexityForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[]`

Apply phased complexity adjustments to the pool when enabled.

Parameters:
- `pool` - - base operator pool
- `internal` - - neat controller context

Returns: pool with phased complexity adjustments

### isBlockedByRecurrentPolicyForSelect

`(mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => boolean`

Check whether a mutation is blocked by recurrent connection policy.

Parameters:
- `mutationMethod` - - mutation operator to check
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isBlockedByStructuralLimitsForSelect

`(mutationMethod: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, genome: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }) => boolean`

Check whether a mutation is blocked by structural limits.

Parameters:
- `mutationMethod` - - mutation operator to check
- `genome` - - genome to inspect
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isLegacyFFWPoolForSelect

`(configuredPool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], methods: { mutation: unknown; }) => boolean`

Check whether a pool matches the legacy FFW operator ordering.

Parameters:
- `configuredPool` - - configured operator pool
- `methods` - - methods module

Returns: true when the pool matches FFW

### isOperatorNamePrefixedForSelect

`(method: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod, prefix: string) => boolean`

Check whether an operator name uses a specific prefix.

Parameters:
- `method` - - mutation operator
- `prefix` - - name prefix to match

Returns: true when the operator name matches the prefix

### normalizeMutationPoolForSelect

`(internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }, rawReturnForTest: boolean) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[]`

Normalize the configured mutation pool to a flat operator list.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW for tests

Returns: normalized mutation pool

### resolveFFWPolicyForSelect

`(internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation, methods: { mutation: unknown; }, rawReturnForTest: boolean) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[] | null`

Resolve legacy FFW policy behavior, including test-specific returns.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW array for tests

Returns: mutation method or null when not handled

### sampleFromPoolForSelect

`(pool: import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").MutationMethod | null`

Sample a random method from the pool.

Parameters:
- `pool` - - operator pool
- `internal` - - neat controller context

Returns: sampled method or null

## neat/neat.species.history.utils.ts

### neat.species.history.utils

Default slice size when exporting species history as JSONL.

### exportSpeciesHistoryJsonl

`(speciesHistory: unknown[], maxEntries: number) => string`

Export species history records as JSON Lines.

### SPECIES_HISTORY_JSONL_MAX_DEFAULT

## neat/neat.evaluate.fitness.utils.ts

### clearGenomeStateIfRequested

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }, clearAction: (genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => void) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.
- `clearAction` - - Action that clears a genome's internal state.

Returns: void.

### runFitnessEvaluation

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => Promise<void>`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Promise<void> after fitness evaluation completes.

## neat/neat.evaluate.novelty.utils.ts

### addGenomeToNoveltyArchive

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, descriptor: number[], novelty: number, noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `descriptor` - - Genome descriptor.
- `novelty` - - Novelty score.
- `noveltyOptions` - - Novelty configuration.

Returns: void.

### applyNoveltyToPopulation

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, descriptors: number[][], distanceMatrix: number[][], kNeighbors: number, blendFactor: number, noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `descriptors` - - Descriptor vectors for each genome.
- `distanceMatrix` - - Distance matrix.
- `kNeighbors` - - Neighbor count.
- `blendFactor` - - Blend factor.
- `noveltyOptions` - - Novelty configuration.

Returns: void.

### blendNoveltyIntoScore

`(genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation, novelty: number, blendFactor: number) => void`

Parameters:
- `genome` - - Genome to update.
- `novelty` - - Novelty value.
- `blendFactor` - - Blend factor.

Returns: void.

### buildDistanceMatrix

`(descriptors: number[][]) => number[][]`

Parameters:
- `descriptors` - - Descriptor vectors.

Returns: Distance matrix.

### buildNoveltyDescriptors

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; }) => number[][]`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `noveltyOptions` - - Novelty configuration.

Returns: Descriptor vectors for each genome.

### computeDescriptorDistance

`(leftDescriptor: number[], rightDescriptor: number[], isSame: boolean) => number`

Parameters:
- `left` - - Left descriptor.
- `right` - - Right descriptor.
- `isSame` - - Whether the descriptors are the same index.

Returns: Euclidean distance.

### computeNoveltyScore

`(distanceRow: number[], kNeighbors: number) => number`

Parameters:
- `distanceRow` - - Distance values for a single genome.
- `kNeighbors` - - Neighbor count.

Returns: Novelty score.

### getNoveltyBlendFactor

`(noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; }) => number`

Parameters:
- `noveltyOptions` - - Novelty configuration.

Returns: Blend factor for novelty vs. fitness.

### getNoveltyNeighborCount

`(noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; }) => number`

Parameters:
- `noveltyOptions` - - Novelty configuration.

Returns: Number of neighbors to consider.

### runNoveltyBlendAndArchive

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.evolve.offspring.utils.ts

### createOffspring

`(context: import("C:/NeatapticTS/src/neat/neat.evolve.offspring.utils").OffspringContext, selectParent: () => import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network").default`

Create a child genome by crossing two parents selected via the provided callback.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.

Returns: Newly created offspring genome.

### OffspringContext

Minimal surface needed for offspring generation.

## neat/neat.evolve.telemetry.utils.ts

### computeDiversityStatsSafely

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Compute diversity stats safely if the hook exists.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### recordTelemetryIfEnabled

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, snapshot: import("C:/NeatapticTS/src/architecture/network").default) => Promise<void>`

Record telemetry if enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `snapshot` - - Best network snapshot for the generation.

Returns: void.

## neat/neat.telemetry.buffer.utils.ts

### ensureTelemetryBuffer

`(telemetryContext: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryBufferContext) => import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[]`

Ensure the telemetry buffer is initialized.

Parameters:
- `telemetryContext` - - Neat-like context holding telemetry buffer.

Returns: A mutable telemetry buffer.

### safelyStreamTelemetryEntry

`(telemetryContext: { options?: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryStreamOptions | undefined; }, telemetryEntry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry) => void`

Stream telemetry entry when a stream callback is configured.

Parameters:
- `telemetryContext` - - Neat-like context with stream settings.
- `telemetryEntry` - - Entry to stream.

### trimTelemetryBuffer

`(telemetryBufferRef: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[], maxEntries: number) => void`

Trim the telemetry buffer to a maximum size.

Parameters:
- `telemetryBufferRef` - - Buffer to trim in-place.
- `maxEntries` - - Maximum entries to keep.

## neat/neat.adaptive.mutation.utils.ts

### applyAnnealDelta

`(baseDelta: number, settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings) => number`

Apply annealing adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `settings` - - Resolved settings.

Returns: Adjusted delta.

### applyExploreLowDelta

`(baseDelta: number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

Apply explore-low adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyMutationAmount

`(genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => void`

Apply mutation-amount adjustments to a genome.

Parameters:
- `genome` - - Current genome.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

### applyMutationsToPopulation

`(population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[], partitions: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationPartitions, settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number) => import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationOutcome`

Apply mutation updates to the population.

Parameters:
- `population` - - Full population to mutate.
- `partitions` - - Scored partitions.
- `settings` - - Resolved settings.
- `randomSource` - - Random number provider.

Returns: Mutation outcome flags.

### applyTwoTierAmountDelta

`(baseDelta: number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

Apply two-tier adjustments to amount delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierDelta

`(baseDelta: number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

Apply two-tier adjustments to a delta.

Parameters:
- `baseDelta` - - Base random delta.
- `genome` - - Current genome.
- `genomeIndex` - - Genome index.
- `topHalfSet` - - Lookup for top-half genomes.
- `bottomHalfSet` - - Lookup for bottom-half genomes.

Returns: Adjusted delta.

### applyTwoTierFallback

`(population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[], settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings) => void`

Apply two-tier fallback balancing.

Parameters:
- `population` - - Population of genomes.
- `settings` - - Resolved settings.

### clampValue

`(value: number, min: number, max: number) => number`

Clamp a value between min and max bounds.

Parameters:
- `value` - - Value to clamp.
- `min` - - Minimum bound.
- `max` - - Maximum bound.

Returns: Clamped value.

### collectScoredGenomes

`(population: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]) => { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]`

Collect genomes with numeric scores.

Parameters:
- `population` - - Population of genomes.

Returns: Scored genomes.

### createRandomDelta

`(sigmaBase: number, randomSource: () => number) => number`

Create a signed random delta scaled by sigma.

Parameters:
- `sigmaBase` - - Sigma scaling factor.
- `randomSource` - - Random number provider.

Returns: Signed delta.

### resolveAmountDelta

`(settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

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

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; }) => import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings`

Resolve mutation settings derived from configuration and engine state.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Adaptive mutation configuration.

Returns: Resolved mutation settings.

### resolveRandomSource

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => () => number`

Resolve a random source that matches the legacy RNG usage.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Random number provider.

### resolveRateDelta

`(settings: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationSettings, randomSource: () => number, genome: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }, genomeIndex: number, topHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>, bottomHalfSet: Set<{ [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }>) => number`

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

`(generation: number, config: { enabled?: boolean | undefined; learningRate?: number | undefined; min?: number | undefined; max?: number | undefined; adaptEvery?: number | undefined; sigma?: number | undefined; minRate?: number | undefined; maxRate?: number | undefined; strategy?: string | undefined; adaptAmount?: boolean | undefined; minAmount?: number | undefined; maxAmount?: number | undefined; initialRate?: number | undefined; amountSigma?: number | undefined; }) => boolean`

Check whether mutation adaptation should run this generation.

Parameters:
- `generation` - - Current generation index.
- `config` - - Adaptive mutation configuration.

Returns: True if adaptation should run.

### shouldApplyTwoTierFallback

`(strategy: string, outcome: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationOutcome) => boolean`

Determine whether a two-tier fallback is needed.

Parameters:
- `strategy` - - Mutation strategy identifier.
- `outcome` - - Mutation outcome flags.

Returns: True if fallback should run.

### sortScoredGenomes

`(scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]) => { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]`

Sort scored genomes in ascending score order.

Parameters:
- `scoredGenomes` - - Scored genomes.

Returns: Sorted genomes.

### splitScoredGenomes

`(scoredGenomes: { [key: string]: unknown; score?: number | undefined; _mutRate?: number | null | undefined; _mutAmount?: number | null | undefined; }[]) => import("C:/NeatapticTS/src/neat/neat.adaptive.shared").MutationPartitions`

Split scored genomes into top and bottom halves.

Parameters:
- `scoredGenomes` - - Sorted scored genomes.

Returns: Partitions used by strategy rules.

## neat/neat.adaptive.operator.utils.ts

### applyOperatorDecay

`(stats: Map<string, { success: number; attempts: number; }>, entries: [string, { success: number; attempts: number; }][], decay: number) => void`

Apply exponential decay to each operator statistic entry.

Parameters:
- `stats` - - Operator statistics map.
- `entries` - - Operator stat entries to update.
- `decay` - - Decay factor.

### collectOperatorStatsEntries

`(stats: Map<string, { success: number; attempts: number; }>) => [string, { success: number; attempts: number; }][]`

Collect operator statistic entries for processing.

Parameters:
- `stats` - - Operator statistics map.

Returns: Array of operator stat entries.

### decayOperatorStat

`(operatorStat: { success: number; attempts: number; }, decay: number) => { success: number; attempts: number; }`

Apply decay to a single operator statistic record.

Parameters:
- `operatorStat` - - Operator statistic record.
- `decay` - - Decay factor.

Returns: Decayed operator statistic record.

### resolveOperatorDecay

`(config: { enabled?: boolean | undefined; learningRate?: number | undefined; alpha?: number | undefined; decay?: number | undefined; }) => number`

Resolve the decay factor for operator statistics.

Parameters:
- `config` - - Operator adaptation configuration.

Returns: Decay factor for exponential smoothing.

## neat/neat.evolve.objectives.utils.ts

### applyDynamicObjectiveSchedule

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, currentObjectiveKeys: string[], config: { autoEntropyAddAt: number; }) => void`

Apply dynamic objective scheduling and entropy rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Keys of active objectives.
- `config` - - Scheduling constants.

Returns: void.

### applyFitnessSuppressionForTests

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Suppress fitness objective for specific test scenarios.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### captureObjectiveImportanceSnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Capture objective importance stats for telemetry.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### handleEntropyDropAndReadd

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, currentObjectiveKeys: string[], dynamicConfig: { enabled?: boolean | undefined; addComplexityAt?: number | undefined; addEntropyAt?: number | undefined; dropEntropyOnStagnation?: number | undefined; readdEntropyAfter?: number | undefined; } | undefined) => void`

Handle entropy removal and re-addition rules.

Parameters:
- `internal` - - NEAT controller instance.
- `currentObjectiveKeys` - - Active objective keys.
- `dynamicConfig` - - Dynamic objective config.

Returns: void.

### resetObjectivesCache

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Clear cached objectives so dynamic schedules can rebuild them.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

### updateObjectiveScheduleAndAges

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) => void; }) => Promise<void>`

Update objective schedule, pending adds/removes, and objective ages.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used by scheduling logic.
- `helpers` - - Dynamic objective scheduler.

Returns: void.

## neat/neat.evolve.population.utils.ts

### addOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], helpers: { addSpeciatedOffspring: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number) => Promise<void>; addUnspeciatedOffspring: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number) => Promise<void>; }) => Promise<void>`

Add offspring to fill remaining population slots.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `helpers` - - Helper callbacks for offspring selection.
- `helpers` - - Speciated offspring helper.
- `helpers` - - Unspeciated offspring helper.

Returns: void.

### addSpeciatedOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number, config: { minOffspringDefault: number; survivalThresholdDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; crossSpeciesGuardLimit: number; }) => Promise<void>`

Add offspring when speciation is enabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Offspring allocation constants.

Returns: void.

### addUnspeciatedOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[], remainingSlots: number) => Promise<void>`

Add offspring when speciation is disabled.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.
- `remainingSlots` - - Slots remaining to fill.

Returns: void.

### applyElitism

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Apply elitism for the next generation.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### applyProvenance

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Add provenance genomes into the next population.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Target population array.

Returns: void.

### buildNextPopulation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { applyElitism: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void; applyProvenance: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => void; addOffspring: (nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => Promise<void>; }) => Promise<import("C:/NeatapticTS/src/architecture/network").default[]>`

Build the next population (elitism, provenance, offspring).

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for population construction.
- `helpers` - - Elitism helper.
- `helpers` - - Provenance helper.
- `helpers` - - Offspring helper.

Returns: next population array.

### buildSpeciesOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, survivors: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[], speciesIndex: number, crossSpeciesProbability: number, crossSpeciesGuardLimit: number, survivalThresholdDefault: number) => import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata`

Build a single offspring within a species.

Parameters:
- `internal` - - NEAT controller instance.
- `survivors` - - Survivors pool for selection.
- `speciesIndex` - - Species index.
- `crossSpeciesProbability` - - Cross-species mating probability.
- `crossSpeciesGuardLimit` - - Retry guard for cross-species selection.

Returns: offspring genome.

### computeOffspringAllocation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, remainingSlots: number, config: { minOffspringDefault: number; youngThresholdDefault: number; youngMultiplierDefault: number; oldThresholdDefault: number; oldMultiplierDefault: number; }) => number[]`

Compute offspring allocation per species.

Parameters:
- `internal` - - NEAT controller instance.
- `remainingSlots` - - Slots remaining to fill.
- `config` - - Allocation constants.

Returns: allocation per species index.

### distributeRemainingSlots

`(allocation: number[], rawShares: number[], remainingSlots: number) => void`

Distribute leftover slots by fractional remainders.

Parameters:
- `allocation` - - Allocation array to adjust.
- `rawShares` - - Raw fractional shares.
- `remainingSlots` - - Total slots available.

Returns: void.

### enforceMinimumOffspring

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, allocation: number[], remainingSlots: number, minOffspringDefault: number) => void`

Enforce minimum offspring per species when possible.

Parameters:
- `internal` - - NEAT controller instance.
- `allocation` - - Allocation array to adjust.
- `remainingSlots` - - Total slots available.
- `minOffspringDefault` - - Default minimum offspring.

Returns: void.

### enforcePopulationConstraints

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, nextPopulation: import("C:/NeatapticTS/src/architecture/network").default[]) => Promise<void>`

Ensure new population meets structural constraints.

Parameters:
- `internal` - - NEAT controller instance.
- `nextPopulation` - - Population to validate.

Returns: void.

### selectSecondParent

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, survivors: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[], speciesIndex: number, crossSpeciesProbability: number, crossSpeciesGuardLimit: number, survivalThresholdDefault: number) => import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata`

Select a second parent, optionally from another species.

Parameters:
- `internal` - - NEAT controller instance.
- `survivors` - - Survivors pool from the current species.
- `speciesIndex` - - Current species index.
- `crossSpeciesProbability` - - Probability to cross species.
- `crossSpeciesGuardLimit` - - Retry guard for cross-species selection.

Returns: chosen parent genome.

### trimOversubscription

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, allocation: number[], remainingSlots: number, minOffspringDefault: number) => void`

Trim allocations when oversubscribed.

Parameters:
- `internal` - - NEAT controller instance.
- `allocation` - - Allocation array to adjust.
- `remainingSlots` - - Total slots available.
- `minOffspringDefault` - - Default minimum offspring.

Returns: void.

## neat/neat.evolve.speciation.utils.ts

### applyGlobalStagnationInjectionIfNeeded

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { buildFreshGenomeForStagnation: () => Promise<import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata>; replaceFraction: number; }) => Promise<void>`

Apply global stagnation injection if configured.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks for stagnation injection.
- `helpers` - - Genome builder for injection.

Returns: void.

### applySpeciationAndSharingIfEnabled

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, helpers: { applyAutoCompatibilityTuning: () => void; recordSpeciesHistorySnapshot: () => void; }) => Promise<void>`

Apply speciation, fitness sharing, and related side effects.

Parameters:
- `internal` - - NEAT controller instance.
- `helpers` - - Helper callbacks used for tuning and history.
- `helpers` - - Auto-compatibility adjustment helper.
- `helpers` - - Species history snapshot helper.

Returns: void.

### buildFreshGenomeForStagnation

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => Promise<import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata>`

Build a fresh genome for stagnation injection.

Parameters:
- `internal` - - NEAT controller instance.

Returns: new genome with minimum constraints.

### ensureHiddenNodeVariance

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, genome: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata) => Promise<void>`

Ensure a minimal hidden-node variance in injected genomes.

Parameters:
- `internal` - - NEAT controller instance.
- `genome` - - Genome to adjust.

Returns: void.

### ensureSpeciesHistorySnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, maxHistory: number) => void`

Ensure a minimal species history snapshot exists for exports.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### recordSpeciesHistorySnapshot

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, maxHistory: number) => void`

Record a species history snapshot when needed.

Parameters:
- `internal` - - NEAT controller instance.
- `maxHistory` - - Maximum history length.

Returns: void.

### updateSpeciesStagnationIfEnabled

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution) => void`

Update species stagnation status when speciation enabled.

Parameters:
- `internal` - - NEAT controller instance.

Returns: void.

## neat/neat.mutation.add-conn.utils.ts

### assignInnovationForConnection

`(connection: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, pairNodes: { symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Assign an innovation id for a new connection, reusing when possible.

Parameters:
- `connection` - - newly created connection
- `pairNodes` - - resolved pair metadata
- `internal` - - neat controller context

Returns: void

### buildLegacyKeyForConn

`(sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => string`

Build a legacy directional innovation key.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: directional innovation key

### buildSymmetricKeyForConn

`(sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => string`

Build a symmetric innovation key for an unordered node pair.

Parameters:
- `sourceNode` - - source node
- `targetNode` - - target node

Returns: symmetric innovation key

### choosePairForConn

`(pairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata] | null`

Choose a pair deterministically when only one candidate exists.

Parameters:
- `pairs` - - selection pool
- `internal` - - neat controller context

Returns: chosen pair or null

### collectCandidatePairsForConn

`(genomeToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]`

Collect legal (from,to) node pairs not already connected.

Parameters:
- `genomeToInspect` - - genome to scan

Returns: candidate node pairs

### connectChosenPair

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, pairNodes: { sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; }) => import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined`

Create the connection for the chosen pair.

Parameters:
- `genomeToEdit` - - genome to edit
- `pairNodes` - - resolved pair nodes

Returns: created connection or undefined

### createsCycle

`(sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => boolean`

Detect whether adding a connection would create a cycle.

Parameters:
- `sourceNode` - - source node of the new connection
- `targetNode` - - target node of the new connection

Returns: true when a cycle is detected

### filterPairsWithInnovations

`(pairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]`

Filter candidate pairs that already have innovation reuse keys.

Parameters:
- `pairs` - - candidate node pairs
- `internal` - - neat controller context

Returns: reuse candidates

### resolvePairNodes

`(chosenPair: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata]) => { sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; symmetricKey: string; legacyForwardKey: string; legacyReverseKey: string; }`

Resolve nodes and innovation key details for a chosen pair.

Parameters:
- `chosenPair` - - pair to connect

Returns: resolved pair metadata

### selectPairPool

`(allPairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][], reusePairs: [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]) => [import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata][]`

Build the final selection pool based on reuse and hidden-node preference.

Parameters:
- `allPairs` - - all candidate pairs
- `reusePairs` - - pairs with historical innovations

Returns: selection pool

### shouldAbortForCycle

`(genomeToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, pairNodes: { sourceNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata; }) => boolean`

Determine whether adding the connection would create a cycle.

Parameters:
- `genomeToInspect` - - genome to inspect
- `pairNodes` - - resolved pair nodes

Returns: true if the connection should be aborted

## neat/neat.mutation.add-node.utils.ts

### applySplitWithExistingRecord

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, splitDescriptor: { splitKey: string; originalWeight: number; }, splitRecord: { newNodeGeneId: number; inInnov: number; outInnov: number; }, NodeClass: new (type: "input" | "output" | "hidden") => unknown) => void`

Apply a split using an existing innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `splitRecord` - - existing innovation record
- `NodeClass` - - node constructor

Returns: void

### applySplitWithNewRecord

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, splitDescriptor: { splitKey: string; originalWeight: number; }, NodeClass: new (type: "input" | "output" | "hidden") => unknown, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Apply a split and create a new innovation record.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `splitDescriptor` - - metadata for the split
- `NodeClass` - - node constructor
- `internal` - - neat controller context

Returns: void

### assignInnovationsForNewSplit

`(newNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, splitConnections: { incomingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; outgoingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => { newNodeGeneId: number; inInnov: number; outInnov: number; }`

Assign new innovations for a split and build the innovation record.

Parameters:
- `newNode` - - newly created hidden node
- `splitConnections` - - incoming/outgoing connections
- `internal` - - neat controller context

Returns: innovation record for the split

### buildSplitDescriptor

`(connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata) => { splitKey: string; originalWeight: number; }`

Build the split descriptor used for innovation lookup and connection creation.

Parameters:
- `connectionToSplit` - - connection being split

Returns: split descriptor

### chooseConnectionForSplit

`(enabledConnectionsList: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | null`

Choose a random enabled connection to split.

Parameters:
- `enabledConnectionsList` - - candidate connections
- `internal` - - neat controller context

Returns: selected connection or null

### collectEnabledConnections

`(genomeToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata[]`

Collect all enabled connections from a genome.

Parameters:
- `genomeToInspect` - - genome to inspect

Returns: enabled connections list

### connectSplitEdges

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToSplit: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata, newNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, originalWeight: number) => { incomingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; outgoingConnection?: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata | undefined; }`

Create the incoming and outgoing split connections.

Parameters:
- `genomeToEdit` - - genome being modified
- `connectionToSplit` - - connection being split
- `newNode` - - newly created hidden node
- `originalWeight` - - weight to preserve on the outgoing connection

Returns: incoming/outgoing connection handles

### disconnectOriginalConnection

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, connectionToRemove: import("C:/NeatapticTS/src/neat/neat.mutation.types").ConnectionWithMetadata) => void`

Disconnect the original connection before inserting the split node.

Parameters:
- `genomeToEdit` - - genome to edit
- `connectionToRemove` - - original connection to remove

Returns: void

### ensureBootstrapConnection

`(genomeToSeed: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure the genome has at least one connection by linking input to output.

Parameters:
- `genomeToSeed` - - genome that may need a bootstrap connection
- `internal` - - neat controller context

Returns: void

### findFirstNodeByType

`(genomeToSearch: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeType: "input" | "output" | "hidden") => import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata | undefined`

Find the first node of a given type.

Parameters:
- `genomeToSearch` - - genome whose nodes are searched
- `nodeType` - - node type to match

Returns: the first matching node or undefined

### resolveInsertIndex

`(genomeToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, targetNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => number`

Resolve the insertion index for a new node, keeping outputs at the end.

Parameters:
- `genomeToEdit` - - genome whose node list is updated
- `targetNode` - - original target node of the split connection

Returns: insertion index

## neat/neat.telemetry.entropy.utils.ts

### buildDegreeHistogram

`(counts: Record<number, number>) => Record<number, number>`

Build a histogram of degree frequencies from a degree-count table.

Parameters:
- `counts` - - Map geneId -> degree count.

Returns: Map degree -> number of nodes with that degree.

### computeDegreeCounts

`(entropyGraph: { nodes: { geneId: number; }[]; connections: { from: { geneId: number; }; to: { geneId: number; }; enabled: boolean; }[]; }) => Record<number, number>`

Compute per-node degree counts for enabled connections.

Parameters:
- `entropyGraph` - - Genome-like graph object.

Returns: Map geneId -> degree count.

### computeEntropyFromHistogram

`(histogram: Record<number, number>, totalNodes: number) => number`

Compute entropy from a degree-frequency histogram.

Parameters:
- `histogram` - - Map degree -> number of nodes.
- `totalNodes` - - Total node count used to normalize into probabilities.

Returns: Entropy value (non-negative).

### getCachedEntropy

`(generation: number | undefined, entropyGraph: Record<string, unknown>) => number | undefined`

Read a cached entropy value if it exists and belongs to the current
generation.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.

Returns: Cached entropy number, or undefined when not available.

### setCachedEntropy

`(generation: number | undefined, entropyGraph: Record<string, unknown>, entropyValue: number) => void`

Cache an entropy value for the current generation on the graph object.

Parameters:
- `generation` - - Current generation number.
- `entropyGraph` - - Genome-like graph object.
- `entropyValue` - - Entropy value to cache.

## neat/neat.telemetry.exports.utils.ts

### buildSpeciesHistoryStats

`(speciesList: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryStat[], defaultSpeciesId: number, defaultSpeciesSize: number, defaultBestScore: number, defaultLastImproved: number) => import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryStat[]`

Normalize raw species records into exportable history stats.

Parameters:
- `speciesList` - - Raw species records to normalize.
- `defaultSpeciesId` - - Default species id when missing.
- `defaultSpeciesSize` - - Default species size when missing.
- `defaultBestScore` - - Default best score when missing.
- `defaultLastImproved` - - Default last improved when missing.

Returns: Normalized stats for CSV export.

### collectBaseKeys

`(entry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, state: import("C:/NeatapticTS/src/neat/neat.telemetry.exports.utils").TelemetryHeaderCollectionState, frontsHeader: string) => void`

Collect base (top-level) telemetry keys for a single entry.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.
- `frontsHeader` - - Header label for fronts column.

Returns: void. Mutates `state.baseKeys`.

### collectDiversityLineageMetrics

`(entry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, state: import("C:/NeatapticTS/src/neat/neat.telemetry.exports.utils").TelemetryHeaderCollectionState) => void`

Collect curated diversity lineage metrics for stable CSV exports.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates diversity lineage key set.

### collectGroupedMetricKeys

`(entry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, state: import("C:/NeatapticTS/src/neat/neat.telemetry.exports.utils").TelemetryHeaderCollectionState) => void`

Collect nested metric keys for grouped telemetry fields.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates complexity/perf/lineage key sets.

### collectOptionalColumnPresence

`(entry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, state: import("C:/NeatapticTS/src/neat/neat.telemetry.exports.utils").TelemetryHeaderCollectionState) => void`

Collect presence flags for optional telemetry columns.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates optional-column flags.

### collectSpeciesHistoryHeaders

`(history: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[], generationHeader: string) => string[]`

Collect ordered header keys for species history CSV export.

Parameters:
- `history` - - Recent species history entries.
- `generationHeader` - - Header label for generation column.

Returns: Ordered header list for CSV output.

### ensureMinimalSpeciesSnapshot

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.types").NeatLike & { _speciesHistory?: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[] | undefined; _species?: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryStat[] | undefined; generation?: number | undefined; }, history: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[], fallbackGeneration: number, defaultSpeciesId: number, defaultSpeciesSize: number, defaultBestScore: number, defaultLastImproved: number) => void`

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

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.types").NeatLike & { _speciesHistory?: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[] | undefined; }) => import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[]`

Ensure the species history array exists on the Neat instance.

Parameters:
- `neatInstance` - - Neat instance holding species history.

Returns: Species history backing array (ensured on instance).

### resolveSpeciesHistoryCellValue

`(historyEntry: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry, speciesStat: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryStat, headerName: string, generationHeader: string) => string`

Resolve a single species history cell value for the provided header.

Parameters:
- `historyEntry` - - A single generation snapshot.
- `speciesStat` - - A single species stat record.
- `headerName` - - Column header name.
- `generationHeader` - - Column header name for generation.

Returns: Serialized cell (JSON) or empty string for missing values.

### safeStringifyCell

`(value: unknown) => string`

Serialize a CSV cell with JSON.stringify safeguards.

Parameters:
- `value` - - Any value to stringify.

Returns: JSON string or empty string when JSON.stringify returns undefined.

### serializeSpeciesHistoryRow

`(historyEntry: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry, speciesStat: import("C:/NeatapticTS/src/neat/neat.types").SpeciesHistoryStat, orderedHeaders: string[], generationHeader: string) => string`

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

`(telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply lineage stats for mono-objective mode using sampled ancestors.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### applyLineageStatsMultiObjective

`(telemetryContext: { _lineageEnabled?: boolean | undefined; _getRNG?: (() => () => number) | undefined; _lastMeanDepth?: number | undefined; _prevInbreedingCount?: number | undefined; }, population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply lineage stats for multi-objective mode using ancestor uniqueness.

Parameters:
- `telemetryContext` - - Neat-like context with lineage settings.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### buildLineageContext

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => import("C:/NeatapticTS/src/neat/neat.lineage.utils").NeatLineageContext`

Build a lineage helper context for ancestor operations.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Lineage helper context.

### buildLineageEntry

`(context: { _prevInbreedingCount?: number | undefined; }, bestGenomeSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed, meanDepthValue: number, ancestorUniquenessScore: number) => { parents: number[]; depthBest: number; meanDepth: number; inbreeding: number; ancestorUniq: number; }`

Build the lineage entry payload.

Parameters:
- `context` - - Neat-like context with lineage info.
- `bestGenomeSnapshot` - - Best genome snapshot.
- `meanDepthValue` - - Mean lineage depth.
- `ancestorUniquenessScore` - - Ancestor uniqueness score.

Returns: Lineage entry payload.

### collectDepths

`(populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number[]`

Collect depth values for the current population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of depth values (defaults to 0).

### computeAncestorUniquenessSampled

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number`

Compute ancestor uniqueness using sampled Jaccard distance.

Parameters:
- `context` - - Neat-like context with RNG helpers.
- `populationSnapshot` - - Population snapshot.

Returns: Rounded ancestor uniqueness score.

### computeLineageStats

`(lineageEnabled: boolean, genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], size: number, pairSampleCount: number, rngFactoryFn: () => () => number) => { lineageMeanDepth: number; lineageMeanPairDist: number; }`

Compute lineage depth and pairwise depth-distance statistics.

Parameters:
- `lineageEnabled` - - Whether lineage metrics are enabled.
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Lineage mean depth and pairwise distance.

### computeMeanDepth

`(depthValues: number[]) => number`

Compute the mean depth from a depth list.

Parameters:
- `depthValues` - - Depth values to average.

Returns: Mean depth value.

### computePairJaccardDistance

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], firstIndex: number, secondIndex: number) => number | undefined`

Compute Jaccard distance between ancestor sets for a pair.

Parameters:
- `context` - - Neat-like context for lineage helpers.
- `populationSnapshot` - - Population snapshot.
- `firstIndex` - - First genome index.
- `secondIndex` - - Second genome index.

Returns: Jaccard distance or undefined when both sets are empty.

### countAncestorIntersection

`(ancestorsA: Set<number>, ancestorsB: Set<number>) => number`

Count the size of an ancestor intersection.

Parameters:
- `ancestorsA` - - First ancestor set.
- `ancestorsB` - - Second ancestor set.

Returns: Intersection count.

### isLineageEligible

`(context: { _lineageEnabled?: boolean | undefined; }, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => boolean`

Check whether lineage metrics should be computed.

Parameters:
- `context` - - Neat-like context with lineage flag.
- `populationSnapshot` - - Population snapshot to validate.

Returns: True when lineage stats should be computed.

### pickDistinctPairIndices

`(context: { _getRNG?: (() => () => number) | undefined; }, populationSize: number) => { firstIndex: number; secondIndex: number; }`

Pick two distinct indices using the context RNG.

Parameters:
- `context` - - Neat-like context with RNG factory.
- `populationSize` - - Population size for index bounds.

Returns: Pair of distinct indices.

## neat/neat.evaluate.constants.utils.ts

### neat.evaluate.constants.utils

Default neighbor count for novelty calculation.

### AUTO_COEFF_ADJUST_DEFAULT

### AUTO_COEFF_MAX_DEFAULT

### AUTO_COEFF_MIN_DEFAULT

### COMPAT_MAX_THRESHOLD_DEFAULT

### COMPAT_MIN_THRESHOLD_DEFAULT

### COMPAT_THRESHOLD_DEFAULT

### DISTANCE_COEFF_DEFAULT

### ENTROPY_ADJUST_DEFAULT

### ENTROPY_DEADBAND_DEFAULT

### ENTROPY_TARGET_DEFAULT

### ENTROPY_VAR_ADJUST_DEFAULT

### ENTROPY_VAR_HIGH_BAND

### ENTROPY_VAR_LOW_BAND

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

### ENTROPY_VAR_TARGET_DEFAULT

### NOVELTY_ARCHIVE_CAP

### NOVELTY_DEFAULT_BLEND

### NOVELTY_DEFAULT_NEIGHBORS

### VARIANCE_DECREASE_THRESHOLD

### VARIANCE_INCREASE_THRESHOLD

## neat/neat.mutation.dead-ends.utils.ts

### chooseRandomNodeForDeadEnds

`(candidates: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata | null`

Choose a random node from candidates for dead-end repair.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectNodeGroupsForDeadEnds

`(networkToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }`

Collect categorized node arrays for dead-end repair.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### connectIfCandidatesExistForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, anchorNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, candidates: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[], reverse: boolean, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Connect a node to a random candidate if candidates exist.

Parameters:
- `networkToEdit` - - network to edit
- `anchorNode` - - node to connect from/to
- `candidates` - - candidate nodes for connection
- `reverse` - - whether to connect candidate -> anchor
- `internal` - - neat controller context

Returns: void

### ensureHiddenConnectivityForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureInputConnectivityForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure all input nodes have at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureOutputConnectivityForDeadEnds

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure all output nodes have at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### hasIncomingForDeadEnds

`(node: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => boolean`

Check whether a node has any incoming connections.

Parameters:
- `node` - - node to inspect

Returns: true when incoming connections exist

### hasOutgoingForDeadEnds

`(node: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata) => boolean`

Check whether a node has any outgoing connections.

Parameters:
- `node` - - node to inspect

Returns: true when outgoing connections exist

## neat/neat.telemetry.operator.utils.ts

### computeOperatorStatsSnapshot

`(operatorStats: import("C:/NeatapticTS/src/neat/neat.telemetry.types").OperatorStatsMap | undefined) => { op: string; succ: number; att: number; }[]`

Snapshot operator statistics into a telemetry-friendly array.

Parameters:
- `operatorStats` - - Operator stats map (opName -> success/attempts).

Returns: Operator stats snapshot array.

### readOperatorStats

`(operatorStats: import("C:/NeatapticTS/src/neat/neat.telemetry.types").OperatorStatsMap | undefined) => { name: string; success: number; attempts: number; }[]`

Convert operator stats map into the public accessor shape.

## neat/neat.adaptive.complexity.utils.ts

### adjustConnectionBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }, trends: { improvement: number; slope: number; }, factors: { increaseFactor: number; stagnationFactor: number; }, noveltyFactor: number, history: number[]) => void`

Adjust connection budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### adjustNodeBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }, trends: { improvement: number; slope: number; }, factors: { increaseFactor: number; stagnationFactor: number; }, noveltyFactor: number, history: number[]) => void`

Adjust node budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### applyAdaptiveSchedule

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Apply adaptive complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance with adaptive state.
- `config` - - Complexity budget configuration.

### applyComplexityBudgetSchedule

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Apply the complexity budget schedule for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyLinearSchedule

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Apply linear complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### clampNodeBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Clamp node budget to configured minimum.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### computeAdjustmentFactors

`(config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }, trends: { improvement: number; slope: number; }, history: number[]) => { increaseFactor: number; stagnationFactor: number; }`

Compute adjustment factors for budget growth and decay.

Parameters:
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `history` - - Rolling history of best scores.

Returns: Adjustment factors (increase and stagnation multipliers).

### computeNoveltyFactor

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => number`

Compute novelty factor based on archive size.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Novelty multiplier (0.9 if archive small, 1.0 otherwise).

### computeSlope

`(history: number[]) => number`

Compute linear regression slope using ordinary least squares.

Parameters:
- `history` - - Rolling history of best scores.

Returns: OLS slope estimate.

### computeTrends

`(history: number[]) => { improvement: number; slope: number; }`

Compute improvement and slope trends from score history.

Parameters:
- `history` - - Rolling history of best scores.

Returns: Trend metrics (improvement and slope).

### initializeConnectionBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Initialize connection budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializeNodeBudget

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => void`

Initialize node budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### normalizeSlope

`(slope: number, initialScore: number) => number`

Normalize slope magnitude relative to initial score.

Parameters:
- `slope` - - Raw OLS slope.
- `initialScore` - - First score in history window.

Returns: Normalized slope clamped to [-2, 2].

### updateScoreHistory

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; }) => number[]`

Update rolling score history with current best score.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

Returns: Rolling history array after update.

## neat/neat.evaluate.objectives.utils.ts

### registerEntropyObjective

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.

Returns: void.

### runAutoEntropyObjectiveInjection

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### shouldAutoInjectEntropy

`(evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => boolean`

Parameters:
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Whether entropy objective should be injected.

## neat/neat.evaluate.speciation.utils.ts

### runLightweightSpeciation

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

### shouldRunSpeciation

`(evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => boolean`

Parameters:
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: Whether speciation should be run.

## neat/neat.mutation.min-hidden.utils.ts

### chooseRandomNodeForMinHidden

`(candidates: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[], internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata | null`

Choose a random node from a candidate list.

Parameters:
- `candidates` - - candidate nodes
- `internal` - - neat controller context

Returns: selected node or null

### collectNodeGroupsForMinHidden

`(networkToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }`

Collect categorized node arrays for the network.

Parameters:
- `networkToInspect` - - network to inspect

Returns: grouped node arrays

### computeMinimumHiddenSize

`(inputCount: number, outputCount: number, explicitMinimumHidden: number | undefined, hiddenMultiplier: number | undefined) => number`

Compute the minimum hidden node count using explicit or multiplier-based settings.

Parameters:
- `inputCount` - - Number of input nodes in the network.
- `outputCount` - - Number of output nodes in the network.
- `explicitMinimumHidden` - - Optional explicit minimum hidden count.
- `hiddenMultiplier` - - Optional multiplier used when explicit minimum is absent.

Returns: Minimum hidden node requirement.

### ensureHiddenConnectivityForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure hidden nodes have both incoming and outgoing connections.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `internal` - - neat controller context

Returns: void

### ensureHiddenNodeCountForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToEdit: { hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, minimumHidden: number, maxNodesLimit: number) => Promise<void>`

Ensure the network has at least the minimum number of hidden nodes.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToEdit` - - grouped node arrays
- `minimumHidden` - - minimum hidden nodes required
- `maxNodesLimit` - - maximum allowed nodes

Returns: Promise resolving when nodes are created

### ensureIncomingConnectionForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, hiddenNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure a hidden node has at least one incoming connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### ensureOutgoingConnectionForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, nodeGroupsToUse: { outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; hiddenNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }, hiddenNode: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => void`

Ensure a hidden node has at least one outgoing connection.

Parameters:
- `networkToEdit` - - network to edit
- `nodeGroupsToUse` - - grouped node arrays
- `hiddenNode` - - hidden node to connect
- `internal` - - neat controller context

Returns: void

### hasRequiredEndpointsForMinHidden

`(nodeGroupsToCheck: { inputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; outputNodes: import("C:/NeatapticTS/src/neat/neat.mutation.types").NodeWithMetadata[]; }) => boolean`

Check whether the network has at least one input and output node.

Parameters:
- `nodeGroupsToCheck` - - grouped node arrays

Returns: true when inputs and outputs are present

### MINIMUM_HIDDEN_BASELINE

### rebuildNetworkConnectionsForMinHidden

`(networkToEdit: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata) => Promise<void>`

Rebuild connection caches after structural edits.

Parameters:
- `networkToEdit` - - network to rebuild

Returns: Promise resolving after rebuild completes

### resolveMaxNodesForMinHidden

`(internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the maximum node limit for the network.

Parameters:
- `internal` - - neat controller context

Returns: maximum node limit

### resolveMinHiddenForMinHidden

`(networkToInspect: import("C:/NeatapticTS/src/neat/neat.mutation.types").GenomeWithMetadata, maxNodesLimit: number, multiplier: number | undefined, internal: import("C:/NeatapticTS/src/neat/neat.mutation.types").NeatControllerForMutation) => number`

Resolve the minimum hidden node requirement for the network.

Parameters:
- `networkToInspect` - - network to inspect
- `maxNodesLimit` - - maximum allowed nodes
- `multiplier` - - optional size multiplier
- `internal` - - neat controller context

Returns: minimum hidden node count

### warnMissingEndpointsForMinHidden

`() => void`

Emit a warning when the network lacks input or output nodes.

Returns: void

## neat/neat.telemetry.accessors.utils.ts

### buildLineageSnapshot

`(population: { _id?: number | undefined; _parents?: number[] | undefined; }[], limit: number) => { id: number; parents: number[]; }[]`

Snapshot lineage metadata for the first `limit` genomes.

### clearTelemetryBuffer

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => void`

Clear the telemetry buffer in place.

### getCachedDiversityStats

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => import("C:/NeatapticTS/src/neat/neat.diversity.utils").DiversityStats | undefined`

Read cached diversity statistics.

### getObjectiveEventsSnapshot

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => { gen: number; type: "add" | "remove"; key: string; }[]`

Return a shallow copy of recent objective events.

### getPerformanceStatsSnapshot

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }`

Snapshot performance timings for evaluation and evolution steps.

### getTelemetryBuffer

`(host: import("C:/NeatapticTS/src/neat/neat.telemetry.accessors.utils").TelemetryAccessorHost) => import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry[]`

Return the telemetry buffer, defaulting to an empty array when missing.

### LINEAGE_SNAPSHOT_DEFAULT_LIMIT

### TelemetryAccessorHost

Minimal host surface needed by telemetry accessors.

## neat/neat.telemetry.diversity.utils.ts

### applyFastModeDefaults

`(telemetryContext: { _fastModeTuned?: boolean | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions) => void`

Apply fast-mode tuning to diversity sampling and novelty defaults.

Parameters:
- `telemetryContext` - - Context object storing fast-mode tuning flag.
- `telemetryOptions` - - Options with diversity and novelty settings.

### computeCompatibilityStats

`(genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], size: number, pairSampleCount: number, rngFactoryFn: () => () => number, compatibilityDistance: ((a: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome, b: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome) => number) | undefined) => { meanCompat: number; varCompat: number; }`

Compute pairwise compatibility statistics via sampling.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `pairSampleCount` - - Number of pairs to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.
- `compatibilityDistance` - - Optional compatibility distance function.

Returns: Mean and variance of sampled compatibilities.

### computeEntropyStats

`(genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], structuralEntropyFn: (genome: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome) => number) => { meanEntropy: number; varEntropy: number; }`

Compute structural entropy mean and variance across the population.

Parameters:
- `genomes` - - Population snapshot.
- `structuralEntropyFn` - - Function to compute entropy for a genome.

Returns: Mean and variance of entropy values.

### computeGraphletEntropy

`(genomes: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome[], size: number, graphletSampleCount: number, rngFactoryFn: () => () => number) => number`

Sample graphlet motifs and compute entropy over their edge counts.

Parameters:
- `genomes` - - Population snapshot.
- `size` - - Population size.
- `graphletSampleCount` - - Number of graphlets to sample.
- `rngFactoryFn` - - RNG factory returning a uniform random function.

Returns: Graphlet entropy value.

### countEnabledEdges

`(genome: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryGenome, selectedNodes: import("C:/NeatapticTS/src/neat/neat.types").NodeLike[]) => number`

Count enabled edges between the selected nodes in a genome.

Parameters:
- `genome` - - Genome with connections to inspect.
- `selectedNodes` - - Nodes forming the graphlet sample.

Returns: Edge count capped at 3.

### pickDistinctIndices

`(upperBound: number, count: number, rng: () => number) => number[]`

Pick a fixed number of distinct random indices.

Parameters:
- `upperBound` - - Exclusive upper bound for random indices.
- `count` - - Number of distinct indices to pick.
- `rng` - - RNG function returning values in [0,1).

Returns: Array of distinct indices.

## neat/neat.telemetry.selection.utils.ts

### getTelemetryCoreSnapshot

`(sourceEntry: Record<string, unknown>, fields: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryCoreFields) => Partial<Record<string, unknown>>`

Build a snapshot of the core telemetry fields present on the entry; does
not mutate the source entry.

Parameters:
- `sourceEntry` - - Source telemetry object.
- `fields` - - Core telemetry field keys to preserve.

Returns: Shallow snapshot of core fields that exist on the entry.

### mergeTelemetryCoreFields

`(sourceEntry: Record<string, unknown>, coreSnapshot: Partial<Record<string, unknown>>) => Record<string, unknown>`

Re-attach core fields to the filtered entry.
Mutates the entry so the caller keeps the original reference.

Parameters:
- `sourceEntry` - - Filtered telemetry entry to update.
- `coreSnapshot` - - Snapshot of core fields to ensure presence.

Returns: The same entry reference with core fields restored.

### safelyApplyTelemetrySelect

`(telemetryContext: TContext, telemetryEntry: import("C:/NeatapticTS/src/neat/neat.types").TelemetryEntry, applyTelemetrySelectFn: (this: TContext, entry: Record<string, unknown>) => Record<string, unknown>) => void`

Apply telemetry selection while swallowing any selection errors.

Parameters:
- `telemetryContext` - - Neat-like context with telemetry selection.
- `telemetryEntry` - - Entry to filter in place.
- `applyTelemetrySelectFn` - - Selection helper to invoke.

### stripUnselectedTelemetryKeys

`(sourceEntry: Record<string, unknown>, selection: Set<string>, fields: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryCoreFields) => Record<string, unknown>`

Remove non-core keys that are not whitelisted by the selection set.
Mutates the provided entry in-place for efficiency.

Parameters:
- `sourceEntry` - - Telemetry entry being filtered.
- `selection` - - Whitelist of additional telemetry keys.
- `fields` - - Core telemetry field keys that must be preserved.

Returns: The same entry reference after filtering.

## neat/neat.telemetry.complexity.utils.ts

### applyComplexityStatsMonoObjective

`(telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach complexity stats for mono-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `entry` - - Telemetry entry to update.

### applyComplexityStatsMultiObjective

`(telemetryContext: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[], entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach complexity stats for multi-objective mode.

Parameters:
- `telemetryContext` - - Neat-like context with population state.
- `telemetryOptions` - - Options controlling complexity telemetry.
- `population` - - Population snapshot.
- `entry` - - Telemetry entry to update.

### buildComplexityEntry

`(telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, meanCounts: { meanNodes: number; meanConns: number; }, maxCounts: { maxNodes: number; maxConns: number; }, meanEnabledRatio: number, growthValues: { growthNodes: number; growthConns: number; }) => { meanNodes: number; meanConns: number; maxNodes: number; maxConns: number; meanEnabledRatio: number; growthNodes: number; growthConns: number; budgetMaxNodes: number; budgetMaxConns: number; }`

Build the complexity entry payload for multi-objective mode.

Parameters:
- `telemetryOptions` - - Options controlling complexity telemetry.
- `meanCounts` - - Mean node/connection counts.
- `maxCounts` - - Max node/connection counts.
- `meanEnabledRatio` - - Mean enabled ratio.
- `growthValues` - - Growth deltas.

Returns: Complexity entry payload.

### collectPopulationCounts

`(populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => { nodeCounts: number[]; connectionCounts: number[]; }`

Collect node and connection counts for the population.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Node and connection counts arrays.

### computeAndStoreGrowthValues

`(context: { _lastMeanNodes?: number | undefined; _lastMeanConns?: number | undefined; }, meanCounts: { meanNodes: number; meanConns: number; }) => { growthNodes: number; growthConns: number; }`

Compute growth values and store the latest means on the context.

Parameters:
- `context` - - Neat-like context with previous mean values.
- `meanCounts` - - Current mean node/connection counts.

Returns: Growth values for nodes and connections.

### computeEnabledRatios

`(populationSnapshot: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number[]`

Compute enabled ratios per genome.

Parameters:
- `populationSnapshot` - - Population snapshot.

Returns: Array of enabled ratios.

### computeMaxCounts

`(counts: { nodeCounts: number[]; connectionCounts: number[]; }) => { maxNodes: number; maxConns: number; }`

Compute max node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Max node and connection counts.

### computeMeanCounts

`(counts: { nodeCounts: number[]; connectionCounts: number[]; }) => { meanNodes: number; meanConns: number; }`

Compute mean node and connection counts.

Parameters:
- `counts` - - Node and connection counts arrays.

Returns: Mean node and connection counts.

### computeMeanEnabledRatio

`(enabledRatios: number[]) => number`

Compute mean of enabled ratios.

Parameters:
- `enabledRatios` - - Enabled ratios per genome.

Returns: Mean enabled ratio.

## neat/neat.telemetry.objectives.utils.ts

### applyHypervolumeTelemetry

`(telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, hyperVolumeProxy: number, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach hypervolume scalar when requested.

Parameters:
- `telemetryOptions` - - Options controlling telemetry fields.
- `hyperVolumeProxy` - - Hypervolume proxy value.
- `entry` - - Telemetry entry to update.

### applyObjectiveAges

`(telemetryContext: { _objectiveAges?: Map<string, number> | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply objective age snapshots to the entry.

Parameters:
- `telemetryContext` - - Neat-like context with objective ages.
- `entry` - - Telemetry entry to update.

### applyObjectiveEvents

`(telemetryContext: { _pendingObjectiveAdds?: string[] | undefined; _pendingObjectiveRemoves?: string[] | undefined; _objectiveEvents?: import("C:/NeatapticTS/src/neat/neat.types").ObjectiveEvent[] | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord, generation: number) => void`

Apply and flush objective lifecycle events.

Parameters:
- `telemetryContext` - - Neat-like context holding objective events.
- `entry` - - Telemetry entry to update.
- `generation` - - Generation index for event records.

### applyObjectiveImportance

`(telemetryContext: { _lastObjImportance?: import("C:/NeatapticTS/src/neat/neat.types").ObjImportance | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply the most recent objective importance snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with objective importance.
- `entry` - - Telemetry entry to update.

### applyObjectivesSnapshot

`(telemetryContext: { _getObjectives?: (() => { key: string; }[]) | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply objectives list snapshot (keys only).

Parameters:
- `telemetryContext` - - Neat-like context with objective provider.
- `entry` - - Telemetry entry to update.

### applySpeciesAllocation

`(telemetryContext: { _lastOffspringAlloc?: import("C:/NeatapticTS/src/neat/neat.types").SpeciesAlloc[] | undefined; }, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Apply per-species offspring allocation snapshot.

Parameters:
- `telemetryContext` - - Neat-like context with allocation snapshot.
- `entry` - - Telemetry entry to update.

### computeHyperVolumeProxy

`(telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number`

Compute a hypervolume-like proxy for the Pareto front.

Parameters:
- `telemetryOptions` - - Options controlling complexity metric.
- `population` - - Population snapshot.

Returns: Hypervolume proxy value.

### computeParetoFrontSizes

`(population: import("C:/NeatapticTS/src/neat/neat.types").GenomeDetailed[]) => number[]`

Compute sizes of early Pareto fronts.

Parameters:
- `population` - - Population snapshot.

Returns: Array of front sizes (rank 0..4).

## neat/neat.multiobjective.fronts.utils.ts

### annotateGenomeRank

`(population: import("C:/NeatapticTS/src/architecture/network").default[], genomeIndex: number, frontRank: number) => void`

Annotates a genome with its Pareto front rank.

Parameters:
- `population` - - Genome population.
- `genomeIndex` - - Index of the genome to annotate.
- `frontRank` - - Pareto front rank (0 = best front).

### appendFront

`(paretoFronts: import("C:/NeatapticTS/src/architecture/network").default[][], population: import("C:/NeatapticTS/src/architecture/network").default[], currentFrontIndices: number[]) => void`

Appends the current front (index list) as genome references to the
`paretoFronts` accumulator.

Parameters:
- `paretoFronts` - - Accumulator for Pareto fronts.
- `population` - - Genome population.
- `currentFrontIndices` - - Indices for the current front.

### buildNextFrontIndices

`(population: import("C:/NeatapticTS/src/architecture/network").default[], dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, currentFrontIndices: number[], currentFrontRank: number) => number[]`

Builds the next front by applying rank annotations and dominance updates.

Parameters:
- `population` - - Genome population.
- `dominanceState` - - Dominance bookkeeping.
- `currentFrontIndices` - - Indices for the current front.
- `currentFrontRank` - - Rank to assign to the current front.

Returns: Indices for the next front.

### buildParetoFronts

`(population: import("C:/NeatapticTS/src/architecture/network").default[], dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, maxFrontRankGuard: number) => import("C:/NeatapticTS/src/architecture/network").default[][]`

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

`(dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, genomeIndex: number, nextFrontIndices: number[]) => void`

Collects indices that become non-dominated after removing the current
genome’s dominance influence.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `genomeIndex` - - Index of the current genome.
- `nextFrontIndices` - - Accumulator for the next front.

### incrementFrontRank

`(currentFrontRank: number) => number`

Increments the front rank counter.

Parameters:
- `currentFrontRank` - - Current front rank.

Returns: Incremented front rank.

### MAX_PARETO_FRONT_RANK_GUARD

### shouldStopFrontRanking

`(currentFrontRank: number, maxFrontRankGuard: number) => boolean`

Determines whether ranking should stop due to a safety guard.

Parameters:
- `currentFrontRank` - - Current front rank after increment.
- `maxFrontRankGuard` - - Safety guard for ranking iterations.

Returns: `true` if ranking should stop.

## neat/neat.telemetry.performance.utils.ts

### applyPerformanceStats

`(telemetryContext: { _lastEvalDuration?: number | undefined; _lastEvolveDuration?: number | undefined; }, telemetryOptions: import("C:/NeatapticTS/src/neat/neat.types").NeatOptions & import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryDiversityOptions, entry: import("C:/NeatapticTS/src/neat/neat.telemetry.types").TelemetryEntryRecord) => void`

Attach performance stats when configured.

Parameters:
- `telemetryContext` - - Neat-like context with performance data.
- `telemetryOptions` - - Options controlling performance telemetry.
- `entry` - - Telemetry entry to update.

## neat/neat.evaluate.auto-distance.utils.ts

### applyAutoDistanceCoefficientTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, autoDistanceCoeffOptions: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; }, connectionVariance: number) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `autoDistanceCoeffOptions` - - Tuning options.
- `connectionVariance` - - Variance of connection counts.

Returns: void.

### applyDistanceCoefficientDecrease

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, bounds: { minCoeff: number; maxCoeff: number; }, adjustRate: number) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `bounds` - - Min/max coefficients.
- `adjustRate` - - Adjustment rate.

Returns: void.

### applyDistanceCoefficientIncrease

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, bounds: { minCoeff: number; maxCoeff: number; }, adjustRate: number) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `bounds` - - Min/max coefficients.
- `adjustRate` - - Adjustment rate.

Returns: void.

### computeMean

`(values: number[]) => number`

Parameters:
- `values` - - Input values.

Returns: Mean of the values.

### computeVariance

`(values: number[], meanValue: number) => number`

Parameters:
- `values` - - Input values.
- `meanValue` - - Precomputed mean.

Returns: Variance of the values.

### getDistanceCoefficientBounds

`(autoDistanceCoeffOptions: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; }) => { minCoeff: number; maxCoeff: number; }`

Parameters:
- `autoDistanceCoeffOptions` - - Tuning options.

Returns: Bounds for coefficients.

### initializeConnectionVarianceBootstrap

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, connectionVariance: number, bounds: { minCoeff: number; maxCoeff: number; }, adjustRate: number) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `connectionVariance` - - Current connection variance.
- `bounds` - - Min/max coefficients.
- `adjustRate` - - Adjustment rate.

Returns: void.

### runAutoDistanceCoefficientTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.multiobjective.archive.utils.ts

### archiveParetoFrontsIfEnabled

`(neatInstance: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").NeatLikeWithMultiObjective, fronts: import("C:/NeatapticTS/src/architecture/network").default[][]) => void`

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

### MAX_PARETO_ARCHIVE_LENGTH

## neat/neat.multiobjective.metrics.utils.ts

### buildMultiObjectiveMetrics

`(population: import("C:/NeatapticTS/src/architecture/network").default[]) => { rank: any; crowding: any; score: number; nodes: number; connections: number; }[]`

Build lightweight multi-objective metrics for each genome in the population.

### DEFAULT_MAX_PARETO_FRONTS

### DEFAULT_PARETO_ARCHIVE_JSONL_MAX

### DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES

### exportParetoArchiveJsonl

`(archive: unknown[], maxEntries: number) => string`

Export a Pareto archive slice as JSON Lines.

### reconstructParetoFronts

`(population: import("C:/NeatapticTS/src/architecture/network").default[], maxFronts: number, isMultiObjectiveEnabled: boolean) => import("C:/NeatapticTS/src/architecture/network").default[][]`

Reconstruct Pareto fronts from stored rank annotations.

### sliceParetoArchive

`(archive: T[], maxEntries: number) => T[]`

Return the most recent Pareto archive entries up to the provided limit.

## neat/neat.evaluate.entropy-compat.utils.ts

### computeNextCompatibilityThreshold

`(entropyCompatOptions: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; }, meanEntropy: number, currentThreshold: number) => number`

Parameters:
- `entropyCompatOptions` - - Tuning options.
- `meanEntropy` - - Current mean entropy.
- `currentThreshold` - - Current compatibility threshold.

Returns: Next compatibility threshold.

### runEntropyCompatibilityTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.multiobjective.category.utils.ts

### adaptDominanceEpsilon

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, paretoFronts: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[][], config: { targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; }) => void`

Adapt dominance epsilon based on Pareto front size.

Parameters:
- `internal` - - NEAT controller instance.
- `paretoFronts` - - Non-dominated fronts.
- `config` - - Epsilon tuning constants.

Returns: void.

### computeCrowdingDistances

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[], paretoFronts: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[][], objectives: import("C:/NeatapticTS/src/neat/neat.evolve.types").ObjectiveDescriptor[]) => number[]`

Compute crowding distances for multi-objective fronts.

Parameters:
- `internal` - - NEAT controller instance.
- `populationSnapshot` - - Current population reference.
- `paretoFronts` - - Non-dominated fronts.
- `objectives` - - Active objectives.

Returns: crowding distances aligned with population order.

### processMultiObjective

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { paretoArchiveMax: number; targetFrontMin: number; targetFrontUpperRatio: number; targetFrontLowerRatio: number; defaultEpsilonAdjust: number; defaultEpsilonMin: number; defaultEpsilonMax: number; defaultEpsilonCooldown: number; pruneWindowDefault: number; pruneRangeEpsDefault: number; }) => void`

Run multi-objective ranking, crowding distance, and archives.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Multi-objective tuning constants.

Returns: void.

### pruneInactiveObjectives

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, config: { pruneWindowDefault: number; pruneRangeEpsDefault: number; }) => void`

Prune objectives that have collapsed ranges over a window.

Parameters:
- `internal` - - NEAT controller instance.
- `config` - - Pruning constants.

Returns: void.

### recordParetoArchives

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, paretoFronts: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[][], objectives: import("C:/NeatapticTS/src/neat/neat.evolve.types").ObjectiveDescriptor[], archiveMax: number) => void`

Record Pareto front archives for telemetry.

Parameters:
- `internal` - - NEAT controller instance.
- `paretoFronts` - - Non-dominated fronts.
- `objectives` - - Active objectives.
- `archiveMax` - - Maximum archive size.

Returns: void.

### sortPopulationByPareto

`(internal: import("C:/NeatapticTS/src/neat/neat.evolve.types").NeatControllerForEvolution, populationSnapshot: import("C:/NeatapticTS/src/neat/neat.evolve.types").GenomeWithMetadata[], crowdingDistances: number[]) => void`

Sort population by Pareto rank and crowding distance.

Parameters:
- `internal` - - NEAT controller instance.
- `populationSnapshot` - - Current population reference.
- `crowdingDistances` - - Crowding distances aligned with population order.

Returns: void.

## neat/neat.multiobjective.crowding.utils.ts

### accumulateCrowdingForObjective

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[], valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndex: number) => void`

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

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[], valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndex: number, valueRange: number) => void`

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

`(currentGenome: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").NetworkWithMOAnnotations, previousValue: number, nextValue: number, valueRange: number) => void`

Applies a normalized crowding-distance delta to a genome.

If the genome’s crowding distance is `Infinity`, it will remain `Infinity`.
This helper only updates when `_moCrowd` is initialized.

Parameters:
- `currentGenome` - - Genome to update.
- `previousValue` - - Objective value of previous genome.
- `nextValue` - - Objective value of next genome.
- `valueRange` - - Normalized objective range.

### applyCrowdingForObjective

`(front: import("C:/NeatapticTS/src/architecture/network").default[], valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndex: number) => void`

Applies crowding-distance accumulation for a single objective within a
single front.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

### assignCrowdingDistances

`(fronts: import("C:/NeatapticTS/src/architecture/network").default[][], valuesMatrixInput: number[][], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[], population: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

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

`(front: import("C:/NeatapticTS/src/architecture/network").default[], valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndices: number[]) => void`

Assigns crowding distances for a single front across all objectives.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndices` - - Objective indices to process.

### buildGenomeIndexByReference

`(population: import("C:/NeatapticTS/src/architecture/network").default[]) => Map<import("C:/NeatapticTS/src/architecture/network").default, number>`

Builds a stable mapping from genome object references to their population
index.

This relies on object identity (reference equality), not structural
equality. It is used to resolve objective values from a values matrix when
working with reordered views (e.g., sorted fronts).

Parameters:
- `population` - - Genomes in population order.

Returns: Map from genome references to their index.

### buildInteriorIndexRange

`(frontLength: number) => number[]`

Builds the index range for interior genomes of a front.

Boundary genomes are excluded because their crowding distance is treated as
infinite.

Parameters:
- `frontLength` - - Length of the sorted front.

Returns: Interior indices excluding boundary genomes.

### buildObjectiveIndexRange

`(objectiveCount: number) => number[]`

Builds a stable objective index range.

Parameters:
- `objectiveCount` - - Number of objectives.

Returns: Objective indices `0..objectiveCount-1`.

### buildSortedFrontByObjective

`(front: import("C:/NeatapticTS/src/architecture/network").default[], valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndex: number) => import("C:/NeatapticTS/src/architecture/network").default[]`

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

`(valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, objectiveIndex: number, leftGenome: import("C:/NeatapticTS/src/architecture/network").default, rightGenome: import("C:/NeatapticTS/src/architecture/network").default) => number`

Comparator used to sort genomes by a specific objective value.

Parameters:
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.
- `leftGenome` - - Left genome.
- `rightGenome` - - Right genome.

Returns: Numeric sort comparison value (ascending).

### initializeCrowding

`(front: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Initializes crowding-distance annotations for a front.

This sets each genome’s `_moCrowd` to `0`. Later steps accumulate per-
objective spacing deltas.

Parameters:
- `front` - - Pareto front.

### markBoundaryCrowding

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[]) => void`

Marks the boundary genomes of a sorted front as infinitely crowded.

In NSGA-II style crowding distance, boundary solutions (extremes for the
objective) are assigned an infinite crowding distance to ensure they are
always preferred when ranks tie.

Parameters:
- `sortedFront` - - Front sorted by the current objective.

### resolveBoundaryGenomes

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[]) => { firstGenome: import("C:/NeatapticTS/src/architecture/network").default; lastGenome: import("C:/NeatapticTS/src/architecture/network").default; } | null`

Resolves the boundary (first/last) genomes for a sorted front.

Parameters:
- `sortedFront` - - Front sorted by objective.

Returns: Boundary genomes, or `null` if the front is empty.

### resolveGenomeIndex

`(genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, genomeItem: import("C:/NeatapticTS/src/architecture/network").default) => number`

Resolves a genome’s index using a reference-based map.

Parameters:
- `genomeIndexByReference` - - Lookup map created by
 *  {@link buildGenomeIndexByReference} .
 *
- `genomeItem` - - Genome to resolve.

Returns: The population index of the genome.

### resolveNeighborPair

`(sortedFront: import("C:/NeatapticTS/src/architecture/network").default[], sortedIndex: number) => { previousGenome: import("C:/NeatapticTS/src/architecture/network").default; nextGenome: import("C:/NeatapticTS/src/architecture/network").default; }`

Resolves the neighbor genomes for an interior element of a sorted front.

Parameters:
- `sortedFront` - - Front sorted by objective.
- `sortedIndex` - - Current index in sorted front.

Returns: Previous and next neighbor genomes.

### resolveObjectiveRange

`(minValue: number, maxValue: number) => number`

Resolves a non-zero objective range used to normalize crowding deltas.

If all genomes have the same objective value, the raw range is `0`. This
returns `1` in that case to avoid division by zero while still producing a
well-defined crowding delta of `0`.

Parameters:
- `minValue` - - Minimum objective value.
- `maxValue` - - Maximum objective value.

Returns: Normalized range with a non-zero floor.

### resolveObjectiveRangeFromBoundaries

`(valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, boundaryGenomes: { firstGenome: import("C:/NeatapticTS/src/architecture/network").default; lastGenome: import("C:/NeatapticTS/src/architecture/network").default; }, objectiveIndex: number) => number`

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

`(valuesMatrixInput: number[][], genomeIndexByReference: Map<import("C:/NeatapticTS/src/architecture/network").default, number>, genomeItem: import("C:/NeatapticTS/src/architecture/network").default, objectiveIndex: number) => number`

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

`(front: import("C:/NeatapticTS/src/architecture/network").default[]) => boolean`

Determines whether crowding-distance processing should be skipped for a
front.

Parameters:
- `front` - - Pareto front.

Returns: `true` if the front should be skipped.

## neat/neat.evaluate.entropy-sharing.utils.ts

### computeNextSharingSigma

`(entropySharingOptions: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; }, currentVarEntropy: number, currentSigma: number) => number`

Parameters:
- `entropySharingOptions` - - Tuning options.
- `currentVarEntropy` - - Current variance of entropy.
- `currentSigma` - - Current sigma value.

Returns: Next sigma value.

### ensureDiversityStatsContainer

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.

Returns: void.

### runEntropySharingTuning

`(controller: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").NeatControllerForEval, evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: import("C:/NeatapticTS/src/neat/neat.evaluate.utils.types").GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; }) => void`

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.

Returns: void.

## neat/neat.multiobjective.dominance.utils.ts

### applyPairwiseDominance

`(dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, valuesMatrixInput: number[][], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[], candidateIndex: number, opponentIndex: number) => void`

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

`(valuesMatrixInput: number[][], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState`

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

`(populationSize: number) => number[]`

Builds a stable index range for iterating the population.

Parameters:
- `populationSize` - - Number of genomes.

Returns: Array of indices `0..populationSize-1`.

### compareObjectiveValues

`(direction: "max" | "min", candidateValue: number, opponentValue: number) => { isDominated: boolean; isStrictlyBetter: boolean; }`

Compares a candidate and opponent value for a single objective.

This does not compute full Pareto dominance; it returns per-objective flags
used by the vector-level dominance check.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: Comparison flags for this objective.

### createEmptyDominanceState

`(populationSize: number) => import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState`

Creates an empty dominance state container sized to the population.

Parameters:
- `populationSize` - - Number of genomes.

Returns: An initialized dominance state with zeroed counts.

### DominanceState

Dominance bookkeeping structures for fast non-dominated sorting.

These structures are typically produced once per generation (from the values
matrix) and then consumed to build Pareto fronts.

### isCandidateDominatedByObjective

`(direction: "max" | "min", candidateValue: number, opponentValue: number) => boolean`

Checks if the candidate is worse than the opponent for a single objective.

For dominance, being worse on any objective makes the candidate unable to
dominate the opponent.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: `true` if the candidate is dominated for this objective.

### isCandidateStrictlyBetterForObjective

`(direction: "max" | "min", candidateValue: number, opponentValue: number) => boolean`

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

`(dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, candidateIndex: number) => boolean`

Determines whether a candidate has zero domination count.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `candidateIndex` - - Candidate genome index.

Returns: `true` if the candidate is currently non-dominated.

### resolveDominanceOutcome

`(candidateVector: number[], opponentVector: number[], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => "dominates" | "dominated" | "indifferent"`

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

`(descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[], objectiveIndex: number) => "max" | "min"`

Resolves the objective direction for a given objective index.

If a descriptor omits `direction`, it is treated as maximization.

Parameters:
- `descriptors` - - Objective descriptors.
- `objectiveIndex` - - Objective index.

Returns: Normalized objective direction.

### shouldSkipSelfComparison

`(candidateIndex: number, opponentIndex: number) => boolean`

Determines whether a pairwise comparison should be skipped.

Currently this skips only self-comparisons.

Parameters:
- `candidateIndex` - - Candidate genome index.
- `opponentIndex` - - Opponent genome index.

Returns: `true` if the pair should be skipped.

### updateDominanceForCandidate

`(dominanceState: import("C:/NeatapticTS/src/neat/neat.multiobjective.dominance.utils").DominanceState, valuesMatrixInput: number[][], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[], candidateIndex: number, candidateIndices: number[]) => void`

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

`(hasStrictImprovement: boolean, isStrictlyBetter: boolean) => boolean`

Accumulates whether the candidate has any strict improvement across
objectives.

Parameters:
- `hasStrictImprovement` - - Current strict-improvement flag.
- `isStrictlyBetter` - - Whether the candidate strictly improves on the
current objective.

Returns: Updated strict-improvement flag.

### vectorDominates

`(valuesA: number[], valuesB: number[], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => boolean`

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

## neat/neat.multiobjective.objectives.utils.ts

### buildGenomeValues

`(genomeItem: import("C:/NeatapticTS/src/architecture/network").default, descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => number[]`

Builds an objective vector for a single genome.

The resulting array order matches the `descriptors` order exactly.
Each component is read via {@link readObjectiveValue} so individual
objective accessors are fault-tolerant.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptors` - - Objective descriptors (vector schema).

Returns: Objective value vector (length equals `descriptors.length`).

### buildValuesMatrix

`(population: import("C:/NeatapticTS/src/architecture/network").default[], descriptors: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor[]) => number[][]`

Builds a population-wide objective value matrix.

The resulting matrix is indexed as `[genomeIndex][objectiveIndex]` where
`genomeIndex` matches the input `population` order.

Parameters:
- `population` - - Genomes to evaluate (population order is preserved).
- `descriptors` - - Objective descriptors (column schema).

Returns: Objective values matrix.

### readObjectiveValue

`(genomeItem: import("C:/NeatapticTS/src/architecture/network").default, descriptor: import("C:/NeatapticTS/src/neat/neat.multiobjective.utils.types").ObjectiveDescriptor) => number`

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

## neat/neat.adaptive.minimal-criterion.utils.ts

### applyRejection

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, threshold: number) => void`

Zero scores below the final threshold.

Parameters:
- `engine` - - NEAT engine instance.
- `threshold` - - Final MC threshold.

### collectScores

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => number[]`

Collect population scores into a snapshot array.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Array of scores (missing scores treated as 0).

### computeAcceptance

`(scores: number[], threshold: number) => number`

Compute acceptance metrics for the current threshold.

Parameters:
- `scores` - - Population score snapshot.
- `threshold` - - Current MC threshold.

Returns: Acceptance proportion.

### initializeThreshold

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; }) => void`

Initialize MC threshold if missing.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Minimal-criterion adaptive configuration.

### resolveTargetSettings

`(config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; }) => { targetAcceptance: number; adjustRate: number; }`

Resolve target acceptance and adjust rate settings.

Parameters:
- `config` - - Minimal-criterion adaptive configuration.

Returns: Target settings.

### updateThreshold

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, acceptance: number, tuning: { targetAcceptance: number; adjustRate: number; }) => void`

Update the MC threshold based on acceptance proportion.

Parameters:
- `engine` - - NEAT engine instance.
- `acceptance` - - Observed acceptance proportion.
- `tuning` - - Target acceptance and adjustment settings.

## neat/neat.adaptive.ancestor-uniqueness.utils.ts

### applyEpsilonAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, ancestorUniq: number, thresholds: { lowThreshold: number; highThreshold: number; }, adjustMagnitude: number) => void`

Apply dominance-epsilon adjustments when configured.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### applyLineagePressureAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, ancestorUniq: number, thresholds: { lowThreshold: number; highThreshold: number; }) => void`

Apply lineage pressure strength adjustments.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.

### applyUniquenessAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }, ancestorUniq: number, thresholds: { lowThreshold: number; highThreshold: number; }, adjustMagnitude: number) => void`

Apply an adjustment for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### ensureLineagePressureState

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => { enabled?: boolean | undefined; mode?: string | undefined; strength?: number | undefined; }`

Ensure lineage pressure state is available.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Lineage pressure configuration object.

### extractAncestorUniqueness

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => number | undefined`

Extract the latest ancestor-uniqueness metric from telemetry.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Ancestor uniqueness value or undefined when missing.

### isCooldownSatisfied

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive, config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }) => boolean`

Determine whether the cooldown window has elapsed.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: True when adjustment is allowed.

### recordAdjustment

`(engine: import("C:/NeatapticTS/src/neat/neat.adaptive.shared").NeatLikeWithAdaptive) => void`

Record the generation when an adjustment is applied.

Parameters:
- `engine` - - NEAT engine instance.

### resolveAdjustmentMagnitude

`(config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }) => number`

Resolve adjustment magnitude for nudging controlled parameters.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Adjustment magnitude.

### resolveUniquenessThresholds

`(config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; }) => { lowThreshold: number; highThreshold: number; }`

Resolve thresholds for ancestor-uniqueness decisions.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Threshold bounds.
