# 

## config.ts

### config

### NeatapticConfig

Global NeatapticTS configuration contract & default instance.

WHY THIS EXISTS
--------------
A central `config` object offers a convenient, documented surface for end-users (and tests)
to tweak library behaviour without digging through scattered constants. Centralization also
lets us validate & evolve feature flags in a single place.

USAGE PATTERN
------------
  import { config } from 'neataptic-ts';
  config.warnings = true;              // enable runtime warnings
  config.deterministicChainMode = true // opt into deterministic deep path construction

Adjust BEFORE constructing networks / invoking evolutionary loops so that subsystems read
the intended values while initializing internal buffers / metadata.

DESIGN NOTES
------------
- We intentionally avoid setters / proxies to keep this a plain serializable object.
- Optional flags are conservative by default (disabled) to preserve legacy stochastic
  behaviour unless a test or user explicitly opts in.

## neat.ts

### neat

### NeatOptions

### Options

Configuration options for Neat evolutionary runs.

Each property is optional and the class applies sensible defaults when a
field is not provided. Options control population size, mutation rates,
compatibility coefficients, selection strategy and other behavioral knobs.

Example:
const opts: NeatOptions = { popsize: 100, mutationRate: 0.5 };
const neat = new Neat(3, 1, fitnessFn, opts);

Note: this type is intentionally permissive to support staged migration and
legacy callers; prefer providing a typed options object where possible.

### default

#### _adaptivePruneLevel

Adaptive prune level for complexity control (optional).

#### _applyFitnessSharing

`() => void`

Apply fitness sharing within species. When `sharingSigma` > 0 this uses a kernel-based
sharing; otherwise it falls back to classic per-species averaging. Sharing reduces
effective fitness for similar genomes to promote diversity.

#### _bestScoreLastGen

Best score observed in the last generation (used for improvement detection).

#### _compatIntegral

Integral accumulator used by adaptive compatibility controllers.

#### _compatSpeciesEMA

Exponential moving average for compatibility threshold (adaptive speciation).

#### _computeDiversityStats

`() => void`

Compute and cache diversity statistics used by telemetry & tests.

#### _connInnovations

Map of connection innovations keyed by a string identifier.

#### _diversityStats

Cached diversity metrics (computed lazily).

#### _getObjectives

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Internal: return cached objective descriptors, building if stale.

#### _invalidateGenomeCaches

`(genome: any) => void`

Invalidate per-genome caches (compatibility distance, forward pass, etc.).

#### _lastAncestorUniqAdjustGen

Generation when ancestor uniqueness adjustment was last applied.

#### _lastEpsilonAdjustGen

Generation when epsilon compatibility was last adjusted.

#### _lastEvalDuration

Duration of the last evaluation run (ms).

#### _lastEvolveDuration

Duration of the last evolve run (ms).

#### _lastGlobalImproveGeneration

Generation index where the last global improvement occurred.

#### _lastInbreedingCount

Last observed count of inbreeding (used for detecting excessive cloning).

#### _lastOffspringAlloc

Last allocated offspring set (used by adaptive allocators).

#### _lineageEnabled

Whether lineage metadata should be recorded on genomes.

#### _mcThreshold

Adaptive minimal criterion threshold (optional).

#### _nextGenomeId

Counter for assigning unique genome ids.

#### _nextGlobalInnovation

Counter for issuing global innovation numbers when explicit numbers are used.

#### _nodeSplitInnovations

Map of node-split innovations used to reuse innovation ids for node splits.

#### _noveltyArchive

Novelty archive used by novelty search (behavior representatives).

#### _objectiveAges

Map tracking ages for objectives by key.

#### _objectiveEvents

Queue of recent objective activation/deactivation events for telemetry.

#### _objectivesList

Cached list of registered objectives.

#### _objectiveStale

Map tracking stale counts for objectives by key.

#### _operatorStats

Operator statistics used by adaptive operator selection.

#### _paretoArchive

Archive of Pareto front metadata for multi-objective tracking.

#### _paretoObjectivesArchive

Archive storing Pareto objectives snapshots.

#### _pendingObjectiveAdds

Pending objective keys to add during safe phases.

#### _pendingObjectiveRemoves

Pending objective keys to remove during safe phases.

#### _phase

Optional phase marker for multi-stage experiments.

#### _prevInbreedingCount

Previous inbreeding count snapshot.

#### _prevSpeciesMembers

Map of species id -> set of member genome ids from previous generation.

#### _rng

Cached RNG function; created lazily and seeded from `_rngState` when used.

#### _rngState

Internal numeric state for the deterministic xorshift RNG when no user RNG
is provided. Stored as a 32-bit unsigned integer.

#### _sortSpeciesMembers

`(sp: { members: import("D:/code-practice/NeatapticTS/src/architecture/network").default[]; }) => void`

Sort members of a species in-place by descending score.

Parameters:
- `sp` - - Species object with `members` array.

#### _speciate

`() => void`

Assign genomes into species based on compatibility distance and maintain species structures.
This function creates new species for unassigned genomes and prunes empty species.
It also records species-level history used for telemetry and adaptive controllers.

#### _species

Array of current species (internal representation).

#### _speciesCreated

Map of speciesId -> creation generation for bookkeeping.

#### _speciesHistory

Time-series history of species stats (for exports/telemetry).

#### _speciesLastStats

Last recorded stats per species id.

#### _structuralEntropy

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number`

Compatibility wrapper retained for tests that reference (neat as any)._structuralEntropy

#### _telemetry

Telemetry buffer storing diagnostic snapshots per generation.

#### _updateSpeciesStagnation

`() => void`

Update species stagnation tracking and remove species that exceeded the allowed stagnation.

#### _warnIfNoBestGenome

`() => void`

Emit a standardized warning when evolution loop finds no valid best genome (test hook).

#### addGenome

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default, parents: number[] | undefined) => void`

Register an externally-created genome into the `Neat` population.

Use this method when code constructs or mutates a `Network` outside of the
usual reproduction pipeline and needs to insert it into `neat.population`
while preserving lineage, id assignment, and structural invariants. The
method performs best-effort safety actions and falls back to pushing the
genome even if invariant enforcement throws, which mirrors the forgiving
behavior used in dynamic population expansion.

Behavior summary:
- Clears the genome's `score` and assigns `_id` using Neat's counter.
- When lineage is enabled, attaches the provided `parents` array (copied)
  and estimates `_depth` as `max(parent._depth) + 1` when parent ids are
  resolvable from the current population.
- Enforces structural invariants (`ensureMinHiddenNodes` and
  `ensureNoDeadEnds`) and invalidates caches via
  `_invalidateGenomeCaches(genome)`.
- Pushes the genome into `this.population`.

Note: Because depth estimation requires parent objects to be discoverable
in `this.population`, callers that generate intermediate parent genomes
should register them via `addGenome` before relying on automatic depth
estimation for their children.

Parameters:
- `genome` - - The external `Network` to add.
- `parents` - - Optional array of parent ids to record on the genome.

#### applyAdaptivePruning

`() => void`

Run the adaptive pruning controller once. This adjusts the internal
`_adaptivePruneLevel` based on the configured metric (nodes or
connections) and invokes per-genome pruning when an adjustment is
warranted.

Educational usage: Allows step-wise observation of how the adaptive
controller converges population complexity toward a target sparsity.

#### applyEvolutionPruning

`() => void`

Manually apply evolution-time pruning once using the current generation
index and configuration in `options.evolutionPruning`.

Educational usage: While pruning normally occurs automatically inside
the evolve loop, exposing this method lets learners trigger the pruning
logic in isolation to observe its effect on network sparsity.

Implementation detail: Delegates to the migrated helper in
`neat.pruning.ts` so the core class surface remains thin.

#### clearObjectives

`() => void`

Register a custom objective for multi-objective optimization.

Educational context: multi-objective optimization lets you optimize for
multiple, potentially conflicting goals (e.g., maximize fitness while
minimizing complexity). Each objective is identified by a unique key and
an accessor function mapping a genome to a numeric score. Registering an
objective makes it visible to the internal MO pipeline and clears any
cached objective list so changes take effect immediately.

Parameters:
- `key` - Unique objective key.
- `direction` - 'min' or 'max' indicating optimization direction.
- `accessor` - Function mapping a genome to a numeric objective value.

#### clearParetoArchive

`() => void`

Clear the Pareto archive.

Removes any stored Pareto-front snapshots retained by the algorithm.

#### clearTelemetry

`() => void`

Export telemetry as CSV with flattened columns for common nested fields.

#### createPool

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default | null) => void`

Create initial population pool. Delegates to helpers if present.

#### ensureMinHiddenNodes

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, multiplierOverride: number | undefined) => void`

Ensure a network has the minimum number of hidden nodes according to
configured policy. Delegates to migrated helper implementation.

Parameters:
- `network` - Network instance to adjust.
- `multiplierOverride` - Optional multiplier to override configured policy.

#### ensureNoDeadEnds

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => void`

Delegate ensureNoDeadEnds to mutation module (added for backward compat).

#### evolve

`() => Promise<import("D:/code-practice/NeatapticTS/src/architecture/network").default>`

Evolves the population by selecting, mutating, and breeding genomes.
This method is delegated to `src/neat/neat.evolve.ts` during the migration.

#### export

`() => any[]`

Exports the current population as an array of JSON objects.
Useful for saving the state of the population for later use.

Returns: An array of JSON representations of the population.

#### exportParetoFrontJSONL

`(maxEntries: number) => string`

Export Pareto front archive as JSON Lines for external analysis.

Each line is a JSON object representing one archived Pareto snapshot.

Parameters:
- `maxEntries` - Maximum number of entries to include (default: 100).

Returns: Newline-separated JSON objects.

#### exportRNGState

`() => number | undefined`

Export the current RNG state for external persistence or tests.

#### exportSpeciesHistoryCSV

`(maxEntries: number) => string`

Return an array of {id, parents} for the first `limit` genomes in population.

#### exportSpeciesHistoryJSONL

`(maxEntries: number) => string`

Export species history as JSON Lines for storage and analysis.

Each line is a JSON object containing a generation index and per-species
stats recorded at that generation. Useful for long-term tracking.

Parameters:
- `maxEntries` - Maximum history entries to include (default: 200).

Returns: Newline-separated JSON objects.

#### exportState

`() => any`

Convenience: export full evolutionary state (meta + population genomes).
Combines innovation registries and serialized genomes for easy persistence.

#### exportTelemetryCSV

`(maxEntries: number) => string`

Export recent telemetry entries as CSV.

The exporter attempts to flatten commonly-used nested fields (complexity,
perf, lineage) into columns. This is a best-effort exporter intended for
human inspection and simple ingestion.

Parameters:
- `maxEntries` - Maximum number of recent telemetry entries to include.

Returns: CSV string (may be empty when no telemetry present).

#### exportTelemetryJSONL

`() => string`

Export telemetry as JSON Lines (one JSON object per line).

Useful for piping telemetry to external loggers or analysis tools.

Returns: A newline-separated string of JSON objects.

#### getAverage

`() => number`

Calculates the average fitness score of the population.
Ensures that the population is evaluated before calculating the average.

Returns: The average fitness score of the population.

#### getDiversityStats

`() => any`

Return the latest cached diversity statistics.

Educational context: diversity metrics summarize how genetically and
behaviorally spread the population is. They can include lineage depth,
pairwise genetic distances, and other aggregated measures used by
adaptive controllers, novelty search, and telemetry. This accessor returns
whatever precomputed diversity object the Neat instance holds (may be
undefined if not computed for the current generation).

Returns: Arbitrary diversity summary object or undefined.

#### getFittest

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Retrieves the fittest genome from the population.
Ensures that the population is evaluated and sorted before returning the result.

Returns: The fittest genome in the population.

#### getLineageSnapshot

`(limit: number) => { id: number; parents: number[]; }[]`

Get recent objective add/remove events.

#### getMinimumHiddenSize

`(multiplierOverride: number | undefined) => number`

Minimum hidden size considering explicit minHidden or multiplier policy.

#### getMultiObjectiveMetrics

`() => { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]`

Returns compact multi-objective metrics for each genome in the current
population. The metrics include Pareto rank and crowding distance (if
computed), along with simple size and score measures useful in
instructional contexts.

Returns: Array of per-genome MO metric objects.

#### getNoveltyArchiveSize

`() => number`

Returns the number of entries currently stored in the novelty archive.

Educational context: The novelty archive stores representative behaviors
used by behavior-based novelty search. Monitoring its size helps teach
how behavioral diversity accumulates over time and can be used to
throttle archive growth.

Returns: Number of archived behaviors.

#### getObjectiveKeys

`() => string[]`

Public helper returning just the objective keys (tests rely on).

#### getObjectives

`() => { key: string; direction: "max" | "min"; }[]`

Clear all collected telemetry entries.

#### getOffspring

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Generates an offspring by crossing over two parent networks.
Uses the crossover method described in the Instinct algorithm.

Returns: A new network created from two parents.

#### getOperatorStats

`() => { name: string; success: number; attempts: number; }[]`

Returns a summary of mutation/operator statistics used by operator
adaptation and bandit selection.

Educational context: Operator statistics track how often mutation
operators are attempted and how often they succeed. These counters are
used by adaptation mechanisms to bias operator selection towards
successful operators.

Returns: Array of { name, success, attempts } objects.

#### getParent

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Selects a parent genome for breeding based on the selection method.
Supports multiple selection strategies, including POWER, FITNESS_PROPORTIONATE, and TOURNAMENT.

Returns: The selected parent genome.

#### getParetoArchive

`(maxEntries: number) => any[]`

Get recent Pareto archive entries (meta information about archived fronts).

Educational context: when performing multi-objective search we may store
representative Pareto-front snapshots over time. This accessor returns the
most recent archive entries up to the provided limit.

Parameters:
- `maxEntries` - Maximum number of entries to return (default: 50).

Returns: Array of archived Pareto metadata entries.

#### getParetoFronts

`(maxFronts: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default[][]`

Export species history as CSV.

Produces rows for each recorded per-species stat entry within the
specified window. Useful for quick inspection or spreadsheet analysis.

Parameters:
- `maxEntries` - Maximum history entries to include (default: 200).

Returns: CSV string (may be empty).

#### getPerformanceStats

`() => { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }`

Return recent performance statistics (durations in milliseconds) for the
most recent evaluation and evolve operations.

Provides wall-clock timing useful for profiling and teaching how runtime
varies with network complexity or population settings.

Returns: Object with { lastEvalMs, lastEvolveMs }.

#### getSpeciesHistory

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[]`

Returns the historical species statistics recorded each generation.

Educational context: Species history captures per-generation snapshots
of species-level metrics (size, best score, last improvement) and is
useful for plotting trends, teaching about speciation dynamics, and
driving adaptive controllers.

The returned array contains entries with a `generation` index and a
`stats` array containing per-species summaries recorded at that
generation.

Returns: An array of generation-stamped species stat snapshots.

#### getSpeciesStats

`() => { id: number; size: number; bestScore: number; lastImproved: number; }[]`

Return a concise summary for each current species.

Educational context: In NEAT, populations are partitioned into species based
on genetic compatibility. Each species groups genomes that are similar so
selection and reproduction can preserve diversity between groups. This
accessor provides a lightweight view suitable for telemetry, visualization
and teaching examples without exposing full genome objects.

The returned array contains objects with these fields:
- id: numeric species identifier
- size: number of members currently assigned to the species
- bestScore: the best observed fitness score for the species
- lastImproved: generation index when the species last improved its best score

Notes for learners:
- Species sizes and lastImproved are typical signals used to detect
  stagnation and apply protective or penalizing measures.
- This function intentionally avoids returning full member lists to
  prevent accidental mutation of internal state; use `getSpeciesHistory`
  for richer historical data.

Returns: An array of species summary objects.

#### getTelemetry

`() => any[]`

Return the internal telemetry buffer.

Telemetry entries are produced per-generation when telemetry is enabled
and include diagnostic metrics (diversity, performance, lineage, etc.).
This accessor returns the raw buffer for external inspection or export.

Returns: Array of telemetry snapshot objects.

#### import

`(json: any[]) => void`

Imports a population from an array of JSON objects.
Replaces the current population with the imported one.

Parameters:
- `json` - - An array of JSON objects representing the population.

#### importRNGState

`(state: any) => void`

Import an RNG state (alias for restore; kept for compatibility).

Parameters:
- `state` - Numeric RNG state.

#### importState

`(bundle: any, fitness: (n: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number) => import("D:/code-practice/NeatapticTS/src/neat").default`

Convenience: restore full evolutionary state previously produced by exportState().

Parameters:
- `bundle` - Object with shape { neat, population }
- `fitness` - Fitness function to attach

#### mutate

`() => void`

Applies mutations to the population based on the mutation rate and amount.
Each genome is mutated using the selected mutation methods.
Slightly increases the chance of ADD_CONN mutation for more connectivity.

#### resetNoveltyArchive

`() => void`

Reset the novelty archive (clear entries).

The novelty archive is used to keep representative behaviors for novelty
search. Clearing it removes stored behaviors.

#### restoreRNGState

`(state: any) => void`

Restore a previously-snapshotted RNG state. This restores the internal
seed but does not re-create the RNG function until next use.

Parameters:
- `state` - Opaque numeric RNG state produced by `snapshotRNGState()`.

#### sampleRandom

`(count: number) => number[]`

Produce `count` deterministic random samples using instance RNG.

#### selectMutationMethod

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default, rawReturnForTest: boolean) => any`

Selects a mutation method for a given genome based on constraints.
Ensures that the mutation respects the maximum nodes, connections, and gates.

Parameters:
- `genome` - - The genome to mutate.

Returns: The selected mutation method or null if no valid method is available.

#### snapshotRNGState

`() => number | undefined`

Return the current opaque RNG numeric state used by the instance.
Useful for deterministic test replay and debugging.

#### sort

`() => void`

Sorts the population in descending order of fitness scores.
Ensures that the fittest genomes are at the start of the population array.

#### spawnFromParent

`(parent: import("D:/code-practice/NeatapticTS/src/architecture/network").default, mutateCount: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Spawn a new genome derived from a single parent while preserving Neat bookkeeping.

This helper performs a canonical "clone + slight mutation" workflow while
keeping `Neat`'s internal invariants intact. It is intended for callers that
want a child genome derived from a single parent but do not want to perform the
bookkeeping and registration steps manually. The function deliberately does NOT
add the returned child to `this.population` so callers are free to inspect or
further modify the child and then register it via `addGenome()` (or push it
directly if they understand the consequences).

Behavior summary:
- Clone the provided `parent` (`parent.clone()` when available, else JSON round-trip).
- Clear fitness/score on the child and assign a fresh unique `_id`.
- If lineage tracking is enabled, set `(child as any)._parents = [parent._id]`
  and `(child as any)._depth = (parent._depth ?? 0) + 1`.
- Enforce structural invariants by calling `ensureMinHiddenNodes(child)` and
  `ensureNoDeadEnds(child)` so the child is valid for subsequent mutation/evaluation.
- Apply `mutateCount` mutations selected via `selectMutationMethod` and driven by
  the instance RNG (`_getRNG()`); mutation exceptions are caught and ignored to
  preserve best-effort behavior during population seeding/expansion.
- Invalidate per-genome caches with `_invalidateGenomeCaches(child)` before return.

Important: the returned child is not registered in `Neat.population` — call
`addGenome(child, [parentId])` to insert it and keep telemetry/lineage consistent.

Parameters:
- `parent` - - Source genome to derive from. Must be a `Network` instance.
- `mutateCount` - - Number of mutation operations to apply to the spawned child (default: 1).

Returns: A new `Network` instance derived from `parent`. The child is unregistered.

#### toJSON

`() => any`

Import a previously exported state bundle and rehydrate a Neat instance.

## neataptic.ts

### neataptic

Network (Evolvable / Trainable Graph)
=====================================
Represents a directed neural computation graph used both as a NEAT genome
phenotype and (optionally) as a gradient‑trainable model. The class binds
together specialized modules (topology, pruning, serialization, slab packing)
to keep the core surface approachable for learners.

Educational Highlights:
 - Structural Mutation: functions like `addNodeBetween()` and evolutionary
   helpers (in higher-level `Neat`) mutate topology to explore architectures.
 - Fast Execution Paths: a Structure‑of‑Arrays (SoA) slab (`rebuildConnectionSlab`)
   packs connection data into typed arrays to improve cache locality.
 - Memory Optimization: node pooling & typed array pooling demonstrate how
   allocation patterns affect performance and GC pressure.
 - Determinism: RNG snapshot/restore methods allow reproducible experiments.
 - Hybrid Workflows: dropout, stochastic depth, weight noise and mixed precision
   illustrate gradient‑era regularization applied to evolved topologies.

Typical Usage:
```ts
const net = new Network(4, 2);           // create network
const out = net.activate([0.1,0.3,0.2,0.9]);
net.addNodeBetween();                    // structural mutation
const slab = (net as any).getConnectionSlab(); // inspect packed arrays
const clone = net.clone();               // deep copy
```

Performance Guidance:
 - Invoke `activate()` normally; the class auto‑selects slab vs object path.
 - Batch structural mutations then call `rebuildConnectionSlab(true)` if you
   need an immediate fast‑path (it is invoked lazily otherwise).
 - Keep input array length exactly equal to `input`; mismatches throw early.

Serialization:
 - `toJSON()` / `fromJSON()` support experiment checkpointing.
 - ONNX export (`exportToONNX`) enables interoperability with other tools.

### default

#### _activateCore

`(withTrace: boolean, input: number | undefined) => number`

Internal shared implementation for activate/noTraceActivate.

Parameters:
- `withTrace` - Whether to update eligibility traces.
- `input` - Optional externally supplied activation (bypasses weighted sum if provided).

#### _adaptivePruneLevel

Adaptive prune level for complexity control (optional).

#### _applyFitnessSharing

`() => void`

Apply fitness sharing within species. When `sharingSigma` > 0 this uses a kernel-based
sharing; otherwise it falls back to classic per-species averaging. Sharing reduces
effective fitness for similar genomes to promote diversity.

#### _applyGradientClipping

`(cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; }) => void`

Trains the network on a given dataset subset for one pass (epoch or batch).
Performs activation and backpropagation for each item in the set.
Updates weights based on batch size configuration.

Parameters:
- `` - - The training dataset subset (e.g., a batch or the full set for one epoch).
- `` - - The number of samples to process before updating weights.
- `` - - The learning rate to use for this training pass.
- `` - - The momentum factor to use.
- `` - - The regularization configuration (L1, L2, or custom function).
- `` - - The function used to calculate the error between target and output.

Returns: The average error calculated over the provided dataset subset.

#### _bestScoreLastGen

Best score observed in the last generation (used for improvement detection).

#### _compatIntegral

Integral accumulator used by adaptive compatibility controllers.

#### _compatSpeciesEMA

Exponential moving average for compatibility threshold (adaptive speciation).

#### _computeDiversityStats

`() => void`

Compute and cache diversity statistics used by telemetry & tests.

#### _connInnovations

Map of connection innovations keyed by a string identifier.

#### _diversityStats

Cached diversity metrics (computed lazily).

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### _getObjectives

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

Internal: return cached objective descriptors, building if stale.

#### _globalNodeIndex

Global index counter for assigning unique indices to nodes.

#### _invalidateGenomeCaches

`(genome: any) => void`

Invalidate per-genome caches (compatibility distance, forward pass, etc.).

#### _la_shadowWeight

**Deprecated:** Use lookaheadShadowWeight instead.

#### _lastAncestorUniqAdjustGen

Generation when ancestor uniqueness adjustment was last applied.

#### _lastEpsilonAdjustGen

Generation when epsilon compatibility was last adjusted.

#### _lastEvalDuration

Duration of the last evaluation run (ms).

#### _lastEvolveDuration

Duration of the last evolve run (ms).

#### _lastGlobalImproveGeneration

Generation index where the last global improvement occurred.

#### _lastInbreedingCount

Last observed count of inbreeding (used for detecting excessive cloning).

#### _lastOffspringAlloc

Last allocated offspring set (used by adaptive allocators).

#### _lineageEnabled

Whether lineage metadata should be recorded on genomes.

#### _mcThreshold

Adaptive minimal criterion threshold (optional).

#### _nextGenomeId

Counter for assigning unique genome ids.

#### _nextGlobalInnovation

Counter for issuing global innovation numbers when explicit numbers are used.

#### _nodeSplitInnovations

Map of node-split innovations used to reuse innovation ids for node splits.

#### _noveltyArchive

Novelty archive used by novelty search (behavior representatives).

#### _objectiveAges

Map tracking ages for objectives by key.

#### _objectiveEvents

Queue of recent objective activation/deactivation events for telemetry.

#### _objectivesList

Cached list of registered objectives.

#### _objectiveStale

Map tracking stale counts for objectives by key.

#### _operatorStats

Operator statistics used by adaptive operator selection.

#### _paretoArchive

Archive of Pareto front metadata for multi-objective tracking.

#### _paretoObjectivesArchive

Archive storing Pareto objectives snapshots.

#### _pendingObjectiveAdds

Pending objective keys to add during safe phases.

#### _pendingObjectiveRemoves

Pending objective keys to remove during safe phases.

#### _phase

Optional phase marker for multi-stage experiments.

#### _prevInbreedingCount

Previous inbreeding count snapshot.

#### _prevSpeciesMembers

Map of species id -> set of member genome ids from previous generation.

#### _rng

Cached RNG function; created lazily and seeded from `_rngState` when used.

#### _rngState

Internal numeric state for the deterministic xorshift RNG when no user RNG
is provided. Stored as a 32-bit unsigned integer.

#### _safeUpdateWeight

`(connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default, delta: number) => void`

Internal helper to safely update a connection weight with clipping and NaN checks.

#### _sortSpeciesMembers

`(sp: { members: import("D:/code-practice/NeatapticTS/src/architecture/network").default[]; }) => void`

Sort members of a species in-place by descending score.

Parameters:
- `sp` - - Species object with `members` array.

#### _speciate

`() => void`

Assign genomes into species based on compatibility distance and maintain species structures.
This function creates new species for unassigned genomes and prunes empty species.
It also records species-level history used for telemetry and adaptive controllers.

#### _species

Array of current species (internal representation).

#### _speciesCreated

Map of speciesId -> creation generation for bookkeeping.

#### _speciesHistory

Time-series history of species stats (for exports/telemetry).

#### _speciesLastStats

Last recorded stats per species id.

#### _structuralEntropy

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number`

Compatibility wrapper retained for tests that reference (neat as any)._structuralEntropy

#### _telemetry

Telemetry buffer storing diagnostic snapshots per generation.

#### _updateSpeciesStagnation

`() => void`

Update species stagnation tracking and remove species that exceeded the allowed stagnation.

#### _warnIfNoBestGenome

`() => void`

Emit a standardized warning when evolution loop finds no valid best genome (test hook).

#### acquire

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default`

Acquire a `Connection` from the pool (or construct new). Fields are fully reset & given
a fresh sequential `innovation` id. Prefer this in evolutionary algorithms that mutate
topology frequently to reduce GC pressure.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### activate

`(input: number[], training: boolean, maxActivationDepth: number) => number[]`

Activates the network using the given input array.
Performs a forward pass through the network, calculating the activation of each node.

Parameters:
- `` - - An array of numerical values corresponding to the network's input nodes.
- `` - - Flag indicating if the activation is part of a training process.
- `` - - Maximum allowed activation depth to prevent infinite loops/cycles.

Returns: An array of numerical values representing the activations of the network's output nodes.

#### activate

`(input: number | undefined) => number`

Activates the node, calculating its output value based on inputs and state.
This method also calculates eligibility traces (`xtrace`) used for training recurrent connections.

The activation process involves:
1. Calculating the node's internal state (`this.state`) based on:
   - Incoming connections' weighted activations.
   - The recurrent self-connection's weighted state from the previous timestep (`this.old`).
   - The node's bias.
2. Applying the activation function (`this.squash`) to the state to get the activation (`this.activation`).
3. Applying the dropout mask (`this.mask`).
4. Calculating the derivative of the activation function.
5. Updating the gain of connections gated by this node.
6. Calculating and updating eligibility traces for incoming connections.

Parameters:
- `input` - Optional input value. If provided, sets the node's activation directly (used for input nodes).

Returns: The calculated activation value of the node.

#### activate

`(value: number[] | undefined, training: boolean) => number[]`

Activates all nodes within the layer, computing their output values.

If an input `value` array is provided, it's used as the initial activation
for the corresponding nodes in the layer. Otherwise, nodes compute their
activation based on their incoming connections.

During training, layer-level dropout is applied, masking all nodes in the layer together.
During inference, all masks are set to 1.

Parameters:
- `value` - - An optional array of activation values to set for the layer's nodes. The length must match the number of nodes.
- `training` - - A boolean indicating whether the layer is in training mode. Defaults to false.

Returns: An array containing the activation value of each node in the layer after activation.

#### activate

`(value: number[] | undefined) => number[]`

Activates all nodes in the group. If input values are provided, they are assigned
sequentially to the nodes before activation. Otherwise, nodes activate based on their
existing states and incoming connections.

Parameters:
- `` - - An optional array of input values. If provided, its length must match the number of nodes in the group.

Returns: An array containing the activation value of each node in the group, in order.

#### activateBatch

`(inputs: number[][], training: boolean) => number[][]`

Activate the network over a batch of input vectors (micro-batching).

Currently iterates sample-by-sample while reusing the network's internal
fast-path allocations. Outputs are cloned number[] arrays for API
compatibility. Future optimizations can vectorize this path.

Parameters:
- `inputs` - Array of input vectors, each length must equal this.input
- `training` - Whether to run with training-time stochastic features

Returns: Array of output vectors, each length equals this.output

#### activateRaw

`(input: number[], training: boolean, maxActivationDepth: number) => any`

Raw activation that can return a typed array when pooling is enabled (zero-copy).
If reuseActivationArrays=false falls back to standard activate().

#### activation

The output value of the node after applying the activation function. This is the value transmitted to connected nodes.

#### addGenome

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default, parents: number[] | undefined) => void`

Register an externally-created genome into the `Neat` population.

Use this method when code constructs or mutates a `Network` outside of the
usual reproduction pipeline and needs to insert it into `neat.population`
while preserving lineage, id assignment, and structural invariants. The
method performs best-effort safety actions and falls back to pushing the
genome even if invariant enforcement throws, which mirrors the forgiving
behavior used in dynamic population expansion.

Behavior summary:
- Clears the genome's `score` and assigns `_id` using Neat's counter.
- When lineage is enabled, attaches the provided `parents` array (copied)
  and estimates `_depth` as `max(parent._depth) + 1` when parent ids are
  resolvable from the current population.
- Enforces structural invariants (`ensureMinHiddenNodes` and
  `ensureNoDeadEnds`) and invalidates caches via
  `_invalidateGenomeCaches(genome)`.
- Pushes the genome into `this.population`.

Note: Because depth estimation requires parent objects to be discoverable
in `this.population`, callers that generate intermediate parent genomes
should register them via `addGenome` before relying on automatic depth
estimation for their children.

Parameters:
- `genome` - - The external `Network` to add.
- `parents` - - Optional array of parent ids to record on the genome.

#### adjustRateForAccumulation

`(rate: number, accumulationSteps: number, reduction: "average" | "sum") => number`

Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average').

#### applyAdaptivePruning

`() => void`

Run the adaptive pruning controller once. This adjusts the internal
`_adaptivePruneLevel` based on the configured metric (nodes or
connections) and invokes per-genome pruning when an adjustment is
warranted.

Educational usage: Allows step-wise observation of how the adaptive
controller converges population complexity toward a target sparsity.

#### applyBatchUpdates

`(momentum: number) => void`

Applies accumulated batch updates to incoming and self connections and this node's bias.
Uses momentum in a Nesterov-compatible way: currentDelta = accumulated + momentum * previousDelta.
Resets accumulators after applying. Safe to call on any node type.

Parameters:
- `momentum` - Momentum factor (0 to disable)

#### applyBatchUpdatesWithOptimizer

`(opts: { type: "sgd" | "rmsprop" | "adagrad" | "adam" | "adamw" | "amsgrad" | "adamax" | "nadam" | "radam" | "lion" | "adabelief" | "lookahead"; momentum?: number | undefined; beta1?: number | undefined; beta2?: number | undefined; eps?: number | undefined; weightDecay?: number | undefined; lrScale?: number | undefined; t?: number | undefined; baseType?: any; la_k?: number | undefined; la_alpha?: number | undefined; }) => void`

Extended batch update supporting multiple optimizers.

Applies accumulated (batch) gradients stored in `totalDeltaWeight` / `totalDeltaBias` to the
underlying weights and bias using the selected optimization algorithm. Supports both classic
SGD (with Nesterov-style momentum via preceding propagate logic) and a collection of adaptive
optimizers. After applying an update, gradient accumulators are reset to 0.

Supported optimizers (type):
 - 'sgd'      : Standard gradient descent with optional momentum.
 - 'rmsprop'  : Exponential moving average of squared gradients (cache) to normalize step.
 - 'adagrad'  : Accumulate squared gradients; learning rate effectively decays per weight.
 - 'adam'     : Bias‑corrected first (m) & second (v) moment estimates.
 - 'adamw'    : Adam with decoupled weight decay (applied after adaptive step).
 - 'amsgrad'  : Adam variant maintaining a maximum of past v (vhat) to enforce non‑increasing step size.
 - 'adamax'   : Adam variant using the infinity norm (u) instead of second moment.
 - 'nadam'    : Adam + Nesterov momentum style update (lookahead on first moment).
 - 'radam'    : Rectified Adam – warms up variance by adaptively rectifying denominator when sample size small.
 - 'lion'     : Uses sign of combination of two momentum buffers (beta1 & beta2) for update direction only.
 - 'adabelief': Adam-like but second moment on (g - m) (gradient surprise) for variance reduction.
 - 'lookahead': Wrapper; performs k fast optimizer steps then interpolates (alpha) towards a slow (shadow) weight.

Options:
 - momentum     : (SGD) momentum factor (Nesterov handled in propagate when update=true).
 - beta1/beta2  : Exponential decay rates for first/second moments (Adam family, Lion, AdaBelief, etc.).
 - eps          : Numerical stability epsilon added to denominator terms.
 - weightDecay  : Decoupled weight decay (AdamW) or additionally applied after main step when adamw selected.
 - lrScale      : Learning rate scalar already scheduled externally (passed as currentRate).
 - t            : Global step (1-indexed) for bias correction / rectification.
 - baseType     : Underlying optimizer for lookahead (not itself lookahead).
 - la_k         : Lookahead synchronization interval (number of fast steps).
 - la_alpha     : Interpolation factor towards slow (shadow) weights/bias at sync points.

Internal per-connection temp fields (created lazily):
 - firstMoment / secondMoment / maxSecondMoment / infinityNorm : Moment / variance / max variance / infinity norm caches.
 - gradientAccumulator : Single accumulator (RMSProp / AdaGrad).
 - previousDeltaWeight : For classic SGD momentum.
 - lookaheadShadowWeight / _la_shadowBias : Lookahead shadow copies.

Safety: We clip extreme weight / bias magnitudes and guard against NaN/Infinity.

Parameters:
- `opts` - Optimizer configuration (see above).

#### applyEvolutionPruning

`() => void`

Manually apply evolution-time pruning once using the current generation
index and configuration in `options.evolutionPruning`.

Educational usage: While pruning normally occurs automatically inside
the evolve loop, exposing this method lets learners trigger the pruning
logic in isolation to observe its effect on network sparsity.

Implementation detail: Delegates to the migrated helper in
`neat.pruning.ts` so the core class surface remains thin.

#### attention

`(size: number, heads: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a multi-head self-attention layer (stub implementation).

Parameters:
- `size` - - Number of output nodes.
- `heads` - - Number of attention heads (default 1).

Returns: A new Layer instance representing an attention layer.

#### batchNorm

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a batch normalization layer.
Applies batch normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a batch normalization layer.

#### bias

The bias value of the node. Added to the weighted sum of inputs before activation.
Input nodes typically have a bias of 0.

#### clear

`() => void`

Clears the internal state of all nodes in the network.
Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.

#### clearObjectives

`() => void`

Register a custom objective for multi-objective optimization.

Educational context: multi-objective optimization lets you optimize for
multiple, potentially conflicting goals (e.g., maximize fitness while
minimizing complexity). Each objective is identified by a unique key and
an accessor function mapping a genome to a numeric score. Registering an
objective makes it visible to the internal MO pipeline and clears any
cached objective list so changes take effect immediately.

Parameters:
- `key` - Unique objective key.
- `direction` - 'min' or 'max' indicating optimization direction.
- `accessor` - Function mapping a genome to a numeric objective value.

#### clearParetoArchive

`() => void`

Clear the Pareto archive.

Removes any stored Pareto-front snapshots retained by the algorithm.

#### clearTelemetry

`() => void`

Export telemetry as CSV with flattened columns for common nested fields.

#### clone

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a deep copy of the network.

Returns: A new Network instance that is a clone of the current network.

#### connect

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]`

Creates a connection between two nodes in the network.
Handles both regular connections and self-connections.
Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).

Parameters:
- `` - - The source node of the connection.
- `` - - The target node of the connection.
- `` - - Optional weight for the connection. If not provided, a random weight is usually assigned by the underlying `Node.connect` method.

Returns: An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.

#### connect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | { nodes: import("D:/code-practice/NeatapticTS/src/architecture/node").default[]; }, weight: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]`

Creates a connection from this node to a target node or all nodes in a group.

Parameters:
- `target` - The target Node or a group object containing a `nodes` array.
- `weight` - The weight for the new connection(s). If undefined, a default or random weight might be assigned by the Connection constructor (currently defaults to 0, consider changing).

Returns: An array containing the newly created Connection object(s).

#### connect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, method: any, weight: number | undefined) => any[]`

Connects this layer's output to a target component (Layer, Group, or Node).

This method delegates the connection logic primarily to the layer's `output` group
or the target layer's `input` method. It establishes the forward connections
necessary for signal propagation.

Parameters:
- `target` - - The destination Layer, Group, or Node to connect to.
- `method` - - The connection method (e.g., `ALL_TO_ALL`, `ONE_TO_ONE`) defining the connection pattern. See `methods.groupConnection`.
- `weight` - - An optional fixed weight to assign to all created connections.
- `` - - The destination entity (Group, Layer, or Node) to connect to.

Returns: An array containing the newly created connection objects.

#### connections

Stores incoming, outgoing, gated, and self-connections for this node.

#### construct

`(list: (import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default)[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Constructs a Network instance from an array of interconnected Layers, Groups, or Nodes.

This method processes the input list, extracts all unique nodes, identifies connections,
gates, and self-connections, and determines the network's input and output sizes based
on the `type` property ('input' or 'output') set on the nodes. It uses Sets internally
for efficient handling of unique elements during construction.

Parameters:
- `` - - An array containing the building blocks (Nodes, Layers, Groups) of the network, assumed to be already interconnected.

Returns: A Network object representing the constructed architecture.

#### conv1d

`(size: number, kernelSize: number, stride: number, padding: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a 1D convolutional layer (stub implementation).

Parameters:
- `size` - - Number of output nodes (filters).
- `kernelSize` - - Size of the convolution kernel.
- `stride` - - Stride of the convolution (default 1).
- `padding` - - Padding (default 0).

Returns: A new Layer instance representing a 1D convolutional layer.

#### createMLP

`(inputCount: number, hiddenCounts: number[], outputCount: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a fully connected, strictly layered MLP network.

Parameters:
- `` - - Number of input nodes
- `` - - Array of hidden layer sizes (e.g. [2,3] for two hidden layers)
- `` - - Number of output nodes

Returns: A new, fully connected, layered MLP

#### createPool

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default | null) => void`

Create initial population pool. Delegates to helpers if present.

#### crossOver

`(network1: import("D:/code-practice/NeatapticTS/src/architecture/network").default, network2: import("D:/code-practice/NeatapticTS/src/architecture/network").default, equal: boolean) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a new offspring network by performing crossover between two parent networks.
This method implements the crossover mechanism inspired by the NEAT algorithm and described
in the Instinct paper, combining genes (nodes and connections) from both parents.
Fitness scores can influence the inheritance process. Matching genes are inherited randomly,
while disjoint/excess genes are typically inherited from the fitter parent (or randomly if fitness is equal or `equal` flag is set).

Parameters:
- `` - - The first parent network.
- `` - - The second parent network.
- `` - - If true, disjoint and excess genes are inherited randomly regardless of fitness.
   If false (default), they are inherited from the fitter parent.

Returns: A new Network instance representing the offspring.

#### dcMask

DropConnect active mask: 1 = not dropped (active), 0 = dropped for this stochastic pass.

#### dense

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a standard fully connected (dense) layer.

All nodes in the source layer/group will connect to all nodes in this layer
when using the default `ALL_TO_ALL` connection method via `layer.input()`.

Parameters:
- `size` - - The number of nodes (neurons) in this layer.

Returns: A new Layer instance configured as a dense layer.

#### derivative

The derivative of the activation function evaluated at the node's current state. Used in backpropagation.

#### deserialize

`(data: any[], inputSize: number | undefined, outputSize: number | undefined) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Network instance from serialized data produced by `serialize()`.
Reconstructs the network structure and state based on the provided arrays.

Parameters:
- `` - - The serialized network data array, typically obtained from `network.serialize()`.
  Expected format: `[activations, states, squashNames, connectionData, inputSize, outputSize]`.
- `` - - Optional input size override.
- `` - - Optional output size override.

Returns: A new Network instance reconstructed from the serialized data.

#### disconnect

`(from: import("D:/code-practice/NeatapticTS/src/architecture/node").default, to: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Disconnects two nodes, removing the connection between them.
Handles both regular connections and self-connections.
If the connection being removed was gated, it is also ungated.

Parameters:
- `` - - The source node of the connection to remove.
- `` - - The target node of the connection to remove.

#### disconnect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default, twosided: boolean) => void`

Removes the connection from this node to the target node.

Parameters:
- `target` - The target node to disconnect from.
- `twosided` - If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.

#### disconnect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, twosided: boolean | undefined) => void`

Removes connections between this layer's nodes and a target Group or Node.

Parameters:
- `target` - - The Group or Node to disconnect from.
- `twosided` - - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.

#### disconnect

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, twosided: boolean) => void`

Removes connections between nodes in this group and a target Group or Node.

Parameters:
- `` - - The Group or Node to disconnect from.
- `` - - If true, also removes connections originating from the `target` and ending in this group. Defaults to false (only removes connections from this group to the target).

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### dropout

Dropout rate for this layer (0 to 1). If > 0, all nodes in the layer are masked together during training.
Layer-level dropout takes precedence over node-level dropout for nodes in this layer.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene (connection) is currently expressed (participates in forward pass).

#### enableWeightNoise

`(stdDev: number | { perHiddenLayer: number[]; }) => void`

Enable weight noise. Provide a single std dev number or { perHiddenLayer: number[] }.

#### enforceMinimumHiddenLayerSizes

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Enforces the minimum hidden layer size rule on a network.

This ensures that all hidden layers have at least min(input, output) + 1 nodes,
which is a common heuristic to ensure networks have adequate representation capacity.

Parameters:
- `` - - The network to enforce minimum hidden layer sizes on

Returns: The same network with properly sized hidden layers

#### ensureMinHiddenNodes

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default, multiplierOverride: number | undefined) => void`

Ensure a network has the minimum number of hidden nodes according to
configured policy. Delegates to migrated helper implementation.

Parameters:
- `network` - Network instance to adjust.
- `multiplierOverride` - Optional multiplier to override configured policy.

#### ensureNoDeadEnds

`(network: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => void`

Delegate ensureNoDeadEnds to mutation module (added for backward compat).

#### error

Stores error values calculated during backpropagation.

#### evolve

`() => Promise<import("D:/code-practice/NeatapticTS/src/architecture/network").default>`

Evolves the population by selecting, mutating, and breeding genomes.
This method is delegated to `src/neat/neat.evolve.ts` during the migration.

#### export

`() => any[]`

Exports the current population as an array of JSON objects.
Useful for saving the state of the population for later use.

Returns: An array of JSON representations of the population.

#### exportParetoFrontJSONL

`(maxEntries: number) => string`

Export Pareto front archive as JSON Lines for external analysis.

Each line is a JSON object representing one archived Pareto snapshot.

Parameters:
- `maxEntries` - Maximum number of entries to include (default: 100).

Returns: Newline-separated JSON objects.

#### exportRNGState

`() => number | undefined`

Export the current RNG state for external persistence or tests.

#### exportSpeciesHistoryCSV

`(maxEntries: number) => string`

Return an array of {id, parents} for the first `limit` genomes in population.

#### exportSpeciesHistoryJSONL

`(maxEntries: number) => string`

Export species history as JSON Lines for storage and analysis.

Each line is a JSON object containing a generation index and per-species
stats recorded at that generation. Useful for long-term tracking.

Parameters:
- `maxEntries` - Maximum history entries to include (default: 200).

Returns: Newline-separated JSON objects.

#### exportState

`() => any`

Convenience: export full evolutionary state (meta + population genomes).
Combines innovation registries and serialized genomes for easy persistence.

#### exportTelemetryCSV

`(maxEntries: number) => string`

Export recent telemetry entries as CSV.

The exporter attempts to flatten commonly-used nested fields (complexity,
perf, lineage) into columns. This is a best-effort exporter intended for
human inspection and simple ingestion.

Parameters:
- `maxEntries` - Maximum number of recent telemetry entries to include.

Returns: CSV string (may be empty when no telemetry present).

#### exportTelemetryJSONL

`() => string`

Export telemetry as JSON Lines (one JSON object per line).

Useful for piping telemetry to external loggers or analysis tools.

Returns: A newline-separated string of JSON objects.

#### fastSlabActivate

`(input: number[]) => number[]`

Public wrapper for fast slab forward pass (primarily for tests / benchmarking).
Prefer using standard activate(); it will auto dispatch when eligible.
Falls back internally if prerequisites not met.

#### firstMoment

First moment estimate (Adam / AdamW) (was opt_m).

#### from

The source (pre-synaptic) node supplying activation.

#### fromJSON

`(json: any) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Reconstructs a network from a JSON object (latest standard).
Handles formatVersion, robust error handling, and index-based references.

Parameters:
- `` - - The JSON object representing the network.

Returns: The reconstructed network.

#### fromJSON

`(json: { bias: number; type: string; squash: string; mask: number; }) => import("D:/code-practice/NeatapticTS/src/architecture/node").default`

Creates a Node instance from a JSON object.

Parameters:
- `json` - The JSON object containing node configuration.

Returns: A new Node instance configured according to the JSON object.

#### gain

Multiplicative modulation applied *after* weight. Default is `1` (neutral). We only store an
internal symbol-keyed property when the gain is non-neutral, reducing memory usage across
large populations where most connections are ungated.

#### gate

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default, connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Gates a connection with a specified node.
The activation of the `node` (gater) will modulate the weight of the `connection`.
Adds the connection to the network's `gates` list.

Parameters:
- `` - - The node that will act as the gater. Must be part of this network.
- `` - - The connection to be gated.

#### gate

`(connections: import("D:/code-practice/NeatapticTS/src/architecture/connection").default | import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]) => void`

Makes this node gate the provided connection(s).
The connection's gain will be controlled by this node's activation value.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to be gated.

#### gate

`(connections: any[], method: any) => void`

Applies gating to a set of connections originating from this layer's output group.

Gating allows the activity of nodes in this layer (specifically, the output group)
to modulate the flow of information through the specified `connections`.

Parameters:
- `connections` - - An array of connection objects to be gated.
- `method` - - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.

#### gate

`(connections: any, method: any) => void`

Configures nodes within this group to act as gates for the specified connection(s).
Gating allows the output of a node in this group to modulate the flow of signal through the gated connection.

Parameters:
- `` - - A single connection object or an array of connection objects to be gated. Consider using a more specific type like `Connection | Connection[]`.
- `` - - The gating mechanism to use (e.g., `methods.gating.INPUT`, `methods.gating.OUTPUT`, `methods.gating.SELF`). Specifies which part of the connection is influenced by the gater node.

#### gater

Optional gating node whose activation can modulate effective weight (symbol-backed).

#### gates

**Deprecated:** Use connections.gated; retained for legacy tests

#### geneId

Stable per-node gene identifier for NEAT innovation reuse

#### getAverage

`() => number`

Calculates the average fitness score of the population.
Ensures that the population is evaluated before calculating the average.

Returns: The average fitness score of the population.

#### getDiversityStats

`() => any`

Return the latest cached diversity statistics.

Educational context: diversity metrics summarize how genetically and
behaviorally spread the population is. They can include lineage depth,
pairwise genetic distances, and other aggregated measures used by
adaptive controllers, novelty search, and telemetry. This accessor returns
whatever precomputed diversity object the Neat instance holds (may be
undefined if not computed for the current generation).

Returns: Arbitrary diversity summary object or undefined.

#### getFittest

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Retrieves the fittest genome from the population.
Ensures that the population is evaluated and sorted before returning the result.

Returns: The fittest genome in the population.

#### getLastGradClipGroupCount

`() => number`

Returns last gradient clipping group count (0 if no clipping yet).

#### getLineageSnapshot

`(limit: number) => { id: number; parents: number[]; }[]`

Get recent objective add/remove events.

#### getLossScale

`() => number`

Returns current mixed precision loss scale (1 if disabled).

#### getMinimumHiddenSize

`(multiplierOverride: number | undefined) => number`

Minimum hidden size considering explicit minHidden or multiplier policy.

#### getMultiObjectiveMetrics

`() => { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]`

Returns compact multi-objective metrics for each genome in the current
population. The metrics include Pareto rank and crowding distance (if
computed), along with simple size and score measures useful in
instructional contexts.

Returns: Array of per-genome MO metric objects.

#### getNoveltyArchiveSize

`() => number`

Returns the number of entries currently stored in the novelty archive.

Educational context: The novelty archive stores representative behaviors
used by behavior-based novelty search. Monitoring its size helps teach
how behavioral diversity accumulates over time and can be used to
throttle archive growth.

Returns: Number of archived behaviors.

#### getObjectiveKeys

`() => string[]`

Public helper returning just the objective keys (tests rely on).

#### getObjectives

`() => { key: string; direction: "max" | "min"; }[]`

Clear all collected telemetry entries.

#### getOffspring

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Generates an offspring by crossing over two parent networks.
Uses the crossover method described in the Instinct algorithm.

Returns: A new network created from two parents.

#### getOperatorStats

`() => { name: string; success: number; attempts: number; }[]`

Returns a summary of mutation/operator statistics used by operator
adaptation and bandit selection.

Educational context: Operator statistics track how often mutation
operators are attempted and how often they succeed. These counters are
used by adaptation mechanisms to bias operator selection towards
successful operators.

Returns: Array of { name, success, attempts } objects.

#### getParent

`() => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Selects a parent genome for breeding based on the selection method.
Supports multiple selection strategies, including POWER, FITNESS_PROPORTIONATE, and TOURNAMENT.

Returns: The selected parent genome.

#### getParetoArchive

`(maxEntries: number) => any[]`

Get recent Pareto archive entries (meta information about archived fronts).

Educational context: when performing multi-objective search we may store
representative Pareto-front snapshots over time. This accessor returns the
most recent archive entries up to the provided limit.

Parameters:
- `maxEntries` - Maximum number of entries to return (default: 50).

Returns: Array of archived Pareto metadata entries.

#### getParetoFronts

`(maxFronts: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default[][]`

Export species history as CSV.

Produces rows for each recorded per-species stat entry within the
specified window. Useful for quick inspection or spreadsheet analysis.

Parameters:
- `maxEntries` - Maximum history entries to include (default: 200).

Returns: CSV string (may be empty).

#### getPerformanceStats

`() => { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }`

Return recent performance statistics (durations in milliseconds) for the
most recent evaluation and evolve operations.

Provides wall-clock timing useful for profiling and teaching how runtime
varies with network complexity or population settings.

Returns: Object with { lastEvalMs, lastEvolveMs }.

#### getRawGradientNorm

`() => number`

Returns last recorded raw (pre-update) gradient L2 norm.

#### getSpeciesHistory

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[]`

Returns the historical species statistics recorded each generation.

Educational context: Species history captures per-generation snapshots
of species-level metrics (size, best score, last improvement) and is
useful for plotting trends, teaching about speciation dynamics, and
driving adaptive controllers.

The returned array contains entries with a `generation` index and a
`stats` array containing per-species summaries recorded at that
generation.

Returns: An array of generation-stamped species stat snapshots.

#### getSpeciesStats

`() => { id: number; size: number; bestScore: number; lastImproved: number; }[]`

Return a concise summary for each current species.

Educational context: In NEAT, populations are partitioned into species based
on genetic compatibility. Each species groups genomes that are similar so
selection and reproduction can preserve diversity between groups. This
accessor provides a lightweight view suitable for telemetry, visualization
and teaching examples without exposing full genome objects.

The returned array contains objects with these fields:
- id: numeric species identifier
- size: number of members currently assigned to the species
- bestScore: the best observed fitness score for the species
- lastImproved: generation index when the species last improved its best score

Notes for learners:
- Species sizes and lastImproved are typical signals used to detect
  stagnation and apply protective or penalizing measures.
- This function intentionally avoids returning full member lists to
  prevent accidental mutation of internal state; use `getSpeciesHistory`
  for richer historical data.

Returns: An array of species summary objects.

#### getTelemetry

`() => any[]`

Return the internal telemetry buffer.

Telemetry entries are produced per-generation when telemetry is enabled
and include diagnostic metrics (diversity, performance, lineage, etc.).
This accessor returns the raw buffer for external inspection or export.

Returns: Array of telemetry snapshot objects.

#### getTrainingStats

`() => { gradNorm: number; gradNormRaw: number; lossScale: number; optimizerStep: number; mp: { good: number; bad: number; overflowCount: number; scaleUps: number; scaleDowns: number; lastOverflowStep: number; }; }`

Consolidated training stats snapshot.

#### gradientAccumulator

Generic gradient accumulator (RMSProp / AdaGrad) (was opt_cache).

#### gru

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a Gated Recurrent Unit (GRU) layer.

GRUs are another type of recurrent neural network cell, often considered
simpler than LSTMs but achieving similar performance on many tasks.
They use an update gate and a reset gate to manage information flow.

Parameters:
- `size` - - The number of GRU units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as a GRU layer.

#### gru

`(layers: number[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Gated Recurrent Unit (GRU) network.
GRUs are another type of recurrent neural network, similar to LSTMs but often simpler.
This constructor uses `Layer.gru` to create the core GRU blocks.

Parameters:
- `` - - A sequence of numbers representing the size (number of units) of each layer: input layer size, hidden GRU layer sizes..., output layer size. Must include at least input, one hidden, and output layer sizes.

Returns: The constructed GRU network.

#### hasGater

Whether a gater node is assigned (modulates gain); true if the gater symbol field is present.

#### hopfield

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Hopfield network.
Hopfield networks are a form of recurrent neural network often used for associative memory tasks.
This implementation creates a simple, fully connected structure.

Parameters:
- `` - - The number of nodes in the network (input and output layers will have this size).

Returns: The constructed Hopfield network.

#### import

`(json: any[]) => void`

Imports a population from an array of JSON objects.
Replaces the current population with the imported one.

Parameters:
- `json` - - An array of JSON objects representing the population.

#### importRNGState

`(state: any) => void`

Import an RNG state (alias for restore; kept for compatibility).

Parameters:
- `state` - Numeric RNG state.

#### importState

`(bundle: any, fitness: (n: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number) => import("D:/code-practice/NeatapticTS/src/neat").default`

Convenience: restore full evolutionary state previously produced by exportState().

Parameters:
- `bundle` - Object with shape { neat, population }
- `fitness` - Fitness function to attach

#### index

Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.

#### infinityNorm

Adamax: Exponential moving infinity norm (was opt_u).

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

`(sourceNodeId: number, targetNodeId: number) => number`

Deterministic Cantor pairing function for a (sourceNodeId, targetNodeId) pair.
Useful when you want a stable innovation id without relying on global mutable counters
(e.g., for hashing or reproducible experiments).

NOTE: For large indices this can overflow 53-bit safe integer space; keep node indices reasonable.

Parameters:
- `sourceNodeId` - Source node integer id / index.
- `targetNodeId` - Target node integer id / index.

Returns: Unique non-negative integer derived from the ordered pair.

#### input

`(from: import("D:/code-practice/NeatapticTS/src/architecture/layer").default | import("D:/code-practice/NeatapticTS/src/architecture/group").default, method: any, weight: number | undefined) => any[]`

Handles the connection logic when this layer is the *target* of a connection.

It connects the output of the `from` layer or group to this layer's primary
input mechanism (which is often the `output` group itself, but depends on the layer type).
This method is usually called by the `connect` method of the source layer/group.

Parameters:
- `from` - - The source Layer or Group connecting *to* this layer.
- `method` - - The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
- `weight` - - An optional fixed weight for the connections.

Returns: An array containing the newly created connection objects.

#### isActivating

Internal flag to detect cycles during activation

#### isConnectedTo

`(target: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Checks if this node is connected to another node.

Parameters:
- `target` - The target node to check the connection with.

Returns: True if connected, otherwise false.

#### isGroup

`(obj: any) => boolean`

Type guard to check if an object is likely a `Group`.

This is a duck-typing check based on the presence of expected properties
(`set` method and `nodes` array). Used internally where `layer.nodes`
might contain `Group` instances (e.g., in `Memory` layers).

Parameters:
- `obj` - - The object to inspect.

Returns: `true` if the object has `set` and `nodes` properties matching a Group, `false` otherwise.

#### isProjectedBy

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Checks if the given node has a direct outgoing connection to this node.
Considers both regular incoming connections and the self-connection.

Parameters:
- `node` - The potential source node.

Returns: True if the given node projects to this node, false otherwise.

#### isProjectingTo

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => boolean`

Checks if this node has a direct outgoing connection to the given node.
Considers both regular outgoing connections and the self-connection.

Parameters:
- `node` - The potential target node.

Returns: True if this node projects to the target node, false otherwise.

#### layerNorm

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a layer normalization layer.
Applies layer normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - - The number of nodes in this layer.

Returns: A new Layer instance configured as a layer normalization layer.

#### lookaheadShadowWeight

Lookahead: shadow (slow) weight parameter (was _la_shadowWeight).

#### lstm

`(size: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a Long Short-Term Memory (LSTM) layer.

LSTMs are a type of recurrent neural network (RNN) cell capable of learning
long-range dependencies. This implementation uses standard LSTM architecture
with input, forget, and output gates, and a memory cell.

Parameters:
- `size` - - The number of LSTM units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as an LSTM layer.

#### lstm

`(layerArgs: (number | { inputToOutput?: boolean | undefined; })[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Long Short-Term Memory (LSTM) network.
LSTMs are a type of recurrent neural network (RNN) capable of learning long-range dependencies.
This constructor uses `Layer.lstm` to create the core LSTM blocks.

Parameters:
- `` - - A sequence of arguments defining the network structure:
- Numbers represent the size (number of units) of each layer: input layer size, hidden LSTM layer sizes..., output layer size.
- An optional configuration object can be provided as the last argument.
- `` - - Configuration options (if passed as the last argument).

Returns: The constructed LSTM network.

#### mask

A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.

#### maxSecondMoment

AMSGrad: Maximum of past second moment (was opt_vhat).

#### memory

`(size: number, memory: number) => import("D:/code-practice/NeatapticTS/src/architecture/layer").default`

Creates a Memory layer, designed to hold state over a fixed number of time steps.

This layer consists of multiple groups (memory blocks), each holding the state
from a previous time step. The input connects to the most recent block, and
information propagates backward through the blocks. The layer's output
concatenates the states of all memory blocks.

Parameters:
- `size` - - The number of nodes in each memory block (must match the input size).
- `memory` - - The number of time steps to remember (number of memory blocks).

Returns: A new Layer instance configured as a Memory layer.

#### mutate

`() => void`

Applies mutations to the population based on the mutation rate and amount.
Each genome is mutated using the selected mutation methods.
Slightly increases the chance of ADD_CONN mutation for more connectivity.

#### mutate

`(method: any) => void`

Mutates the network's structure or parameters according to the specified method.
This is a core operation for neuro-evolutionary algorithms (like NEAT).
The method argument should be one of the mutation types defined in `methods.mutation`.

Parameters:
- `` - - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
  Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).
- `method` - A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).

#### narx

`(inputSize: number, hiddenLayers: number | number[], outputSize: number, previousInput: number, previousOutput: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a Nonlinear AutoRegressive network with eXogenous inputs (NARX).
NARX networks are recurrent networks often used for time series prediction.
They predict the next value of a time series based on previous values of the series
and previous values of external (exogenous) input series.

Parameters:
- `` - - The number of input nodes for the exogenous inputs at each time step.
- `` - - The size of the hidden layer(s). Can be a single number for one hidden layer, or an array of numbers for multiple hidden layers. Use 0 or [] for no hidden layers.
- `` - - The number of output nodes (predicting the time series).
- `` - - The number of past time steps of the exogenous input to feed back into the network.
- `` - - The number of past time steps of the network's own output to feed back into the network (autoregressive part).

Returns: The constructed NARX network.

#### nodes

An array containing all the nodes (neurons or groups) that constitute this layer.
The order of nodes might be relevant depending on the layer type and its connections.

**Deprecated:** Placeholder kept for legacy structural algorithms. No longer populated.

#### noTraceActivate

`(input: number[]) => number[]`

Activates the network without calculating eligibility traces.
This is a performance optimization for scenarios where backpropagation is not needed,
such as during testing, evaluation, or deployment (inference).

Parameters:
- `` - - An array of numerical values corresponding to the network's input nodes.
  The length must match the network's `input` size.

Returns: An array of numerical values representing the activations of the network's output nodes.

#### noTraceActivate

`(input: number | undefined) => number`

Activates the node without calculating eligibility traces (`xtrace`).
This is a performance optimization used during inference (when the network
is just making predictions, not learning) as trace calculations are only needed for training.

Parameters:
- `input` - Optional input value. If provided, sets the node's activation directly (used for input nodes).

Returns: The calculated activation value of the node.

#### old

The node's state from the previous activation cycle. Used for recurrent self-connections.

#### opt_cache

**Deprecated:** Use gradientAccumulator instead.

#### opt_m

**Deprecated:** Use firstMoment instead.

#### opt_m2

**Deprecated:** Use secondMomentum instead.

#### opt_u

**Deprecated:** Use infinityNorm instead.

#### opt_v

**Deprecated:** Use secondMoment instead.

#### opt_vhat

**Deprecated:** Use maxSecondMoment instead.

#### output

Represents the primary output group of nodes for this layer.
This group is typically used when connecting this layer *to* another layer or group.
It might be null if the layer is not yet fully constructed or is an input layer.

#### perceptron

`(layers: number[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a standard Multi-Layer Perceptron (MLP) network.
An MLP consists of an input layer, one or more hidden layers, and an output layer,
fully connected layer by layer.

Parameters:
- `` - - A sequence of numbers representing the size (number of nodes) of each layer, starting with the input layer, followed by hidden layers, and ending with the output layer. Must include at least input, one hidden, and output layer sizes.

Returns: The constructed MLP network.

#### plastic

Whether this connection participates in plastic adaptation (rate > 0).

#### plasticityRate

Per-connection plasticity / learning rate (0 means non-plastic). Setting >0 marks plastic flag.

#### previousDeltaBias

The change in bias applied in the previous training iteration. Used for calculating momentum.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### propagate

`(rate: number, momentum: number, update: boolean, target: number[], regularization: number, costDerivative: ((target: number, output: number) => number) | undefined) => void`

Propagates the error backward through the network (backpropagation).
Calculates the error gradient for each node and connection.
If `update` is true, it adjusts the weights and biases based on the calculated gradients,
learning rate, momentum, and optional L2 regularization.

The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).

Parameters:
- `` - - The learning rate (controls the step size of weight adjustments).
- `` - - The momentum factor (helps overcome local minima and speeds up convergence). Typically between 0 and 1.
- `` - - If true, apply the calculated weight and bias updates. If false, only calculate gradients (e.g., for batch accumulation).
- `` - - An array of target values corresponding to the network's output nodes.
  The length must match the network's `output` size.
- `` - - The L2 regularization factor (lambda). Helps prevent overfitting by penalizing large weights.
- `` - - Optional derivative of the cost function for output nodes.

#### propagate

`(rate: number, momentum: number, update: boolean, regularization: number | { type: "L1" | "L2"; lambda: number; } | ((weight: number) => number), target: number | undefined) => void`

Back-propagates the error signal through the node and calculates weight/bias updates.

This method implements the backpropagation algorithm, including:
1. Calculating the node's error responsibility based on errors from subsequent nodes (`projected` error)
   and errors from connections it gates (`gated` error).
2. Calculating the gradient for each incoming connection's weight using eligibility traces (`xtrace`).
3. Calculating the change (delta) for weights and bias, incorporating:
   - Learning rate.
   - L1/L2/custom regularization.
   - Momentum (using Nesterov Accelerated Gradient - NAG).
4. Optionally applying the calculated updates immediately or accumulating them for batch training.

Parameters:
- `rate` - The learning rate (controls the step size of updates).
- `momentum` - The momentum factor (helps accelerate learning and overcome local minima). Uses NAG.
- `update` - If true, apply the calculated weight/bias updates immediately. If false, accumulate them in `totalDelta*` properties for batch updates.
- `regularization` - The regularization setting. Can be:
- number (L2 lambda)
- { type: 'L1'|'L2', lambda: number }
- (weight: number) => number (custom function)
- `target` - The target output value for this node. Only used if the node is of type 'output'.

#### propagate

`(rate: number, momentum: number, target: number[] | undefined) => void`

Propagates the error backward through all nodes in the layer.

This is a core step in the backpropagation algorithm used for training.
If a `target` array is provided (typically for the output layer), it's used
to calculate the initial error for each node. Otherwise, nodes calculate
their error based on the error propagated from subsequent layers.

Parameters:
- `rate` - - The learning rate, controlling the step size of weight adjustments.
- `momentum` - - The momentum factor, used to smooth weight updates and escape local minima.
- `target` - - An optional array of target values (expected outputs) for the layer's nodes. The length must match the number of nodes.
- `` - - The learning rate to apply during weight updates.

#### pruneToSparsity

`(targetSparsity: number, method: "magnitude" | "snip") => void`

Immediately prune connections to reach (or approach) a target sparsity fraction.
Used by evolutionary pruning (generation-based) independent of training iteration schedule.

Parameters:
- `targetSparsity` - fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
- `method` - 'magnitude' | 'snip'

#### random

`(input: number, hidden: number, output: number, options: { connections?: number | undefined; backconnections?: number | undefined; selfconnections?: number | undefined; gates?: number | undefined; }) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Creates a randomly structured network based on specified node counts and connection options.

This method allows for the generation of networks with a less rigid structure than MLPs.
It initializes a network with input and output nodes and then iteratively adds hidden nodes
and various types of connections (forward, backward, self) and gates using mutation methods.
This approach is inspired by neuro-evolution techniques where network topology evolves.

Parameters:
- `` - - The number of input nodes.
- `` - - The number of hidden nodes to add.
- `` - - The number of output nodes.
- `` - - Optional configuration for the network structure.

Returns: The constructed network with a randomized topology.

#### rebuildConnections

`(net: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => void`

Rebuilds the network's connections array from all per-node connections.
This ensures that the network.connections array is consistent with the actual
outgoing connections of all nodes. Useful after manual wiring or node manipulation.

Parameters:
- `` - - The network instance to rebuild connections for.

Returns: Example usage:
  Network.rebuildConnections(net);

#### release

`(conn: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Return a `Connection` to the internal pool for later reuse. Do NOT use the instance again
afterward unless re-acquired (treat as surrendered). Optimizer / trace fields are not
scrubbed here (they're overwritten during `acquire`).

Parameters:
- `conn` - The connection instance to recycle.

#### remove

`(node: import("D:/code-practice/NeatapticTS/src/architecture/node").default) => void`

Removes a node from the network.
This involves:
1. Disconnecting all incoming and outgoing connections associated with the node.
2. Removing any self-connections.
3. Removing the node from the `nodes` array.
4. Attempting to reconnect the node's direct predecessors to its direct successors
   to maintain network flow, if possible and configured.
5. Handling gates involving the removed node (ungating connections gated *by* this node,
   and potentially re-gating connections that were gated *by other nodes* onto the removed node's connections).

Parameters:
- `` - - The node instance to remove. Must exist within the network's `nodes` list.

#### resetDropoutMasks

`() => void`

Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
Should be called after training to ensure inference is unaffected by previous dropout.

#### resetInnovationCounter

`(value: number) => void`

Reset the monotonic auto-increment innovation counter (used for newly constructed / pooled instances).
You normally only call this at the start of an experiment or when deserializing a full population.

Parameters:
- `value` - New starting value (default 1).

#### resetNoveltyArchive

`() => void`

Reset the novelty archive (clear entries).

The novelty archive is used to keep representative behaviors for novelty
search. Clearing it removes stored behaviors.

#### restoreRNGState

`(state: any) => void`

Restore a previously-snapshotted RNG state. This restores the internal
seed but does not re-create the RNG function until next use.

Parameters:
- `state` - Opaque numeric RNG state produced by `snapshotRNGState()`.

#### sampleRandom

`(count: number) => number[]`

Produce `count` deterministic random samples using instance RNG.

#### secondMoment

Second raw moment estimate (Adam family) (was opt_v).

#### secondMomentum

Secondary momentum (Lion variant) (was opt_m2).

#### selectMutationMethod

`(genome: import("D:/code-practice/NeatapticTS/src/architecture/network").default, rawReturnForTest: boolean) => any`

Selects a mutation method for a given genome based on constraints.
Ensures that the mutation respects the maximum nodes, connections, and gates.

Parameters:
- `genome` - - The genome to mutate.

Returns: The selected mutation method or null if no valid method is available.

#### serialize

`() => any[]`

Lightweight tuple serializer delegating to network.serialize.ts

#### set

`(values: { bias?: number | undefined; squash?: any; }) => void`

Sets specified properties (e.g., bias, squash function) for all nodes in the network.
Useful for initializing or resetting node properties uniformly.

Parameters:
- `` - - An object containing the properties and values to set.

#### set

`(values: { bias?: number | undefined; squash?: any; type?: string | undefined; }) => void`

Configures properties for all nodes within the layer.

Allows batch setting of common node properties like bias, activation function (`squash`),
or node type. If a node within the `nodes` array is actually a `Group` (e.g., in memory layers),
the configuration is applied recursively to the nodes within that group.

Parameters:
- `values` - - An object containing the properties and their values to set.
  Example: `{ bias: 0.5, squash: methods.Activation.ReLU }`
- `` - - An object containing the properties and their new values. Only provided properties are updated.
`bias`: Sets the bias term for all nodes.
`squash`: Sets the activation function (squashing function) for all nodes.
`type`: Sets the node type (e.g., 'input', 'hidden', 'output') for all nodes.

#### setActivation

`(fn: (x: number, derivate?: boolean | undefined) => number) => void`

Sets a custom activation function for this node at runtime.

Parameters:
- `fn` - The activation function (should handle derivative if needed).

#### setStochasticDepth

`(survival: number[]) => void`

Configure stochastic depth with survival probabilities per hidden layer (length must match hidden layer count when using layered network).

#### snapshotRNGState

`() => number | undefined`

Return the current opaque RNG numeric state used by the instance.
Useful for deterministic test replay and debugging.

#### sort

`() => void`

Sorts the population in descending order of fitness scores.
Ensures that the fittest genomes are at the start of the population array.

#### spawnFromParent

`(parent: import("D:/code-practice/NeatapticTS/src/architecture/network").default, mutateCount: number) => import("D:/code-practice/NeatapticTS/src/architecture/network").default`

Spawn a new genome derived from a single parent while preserving Neat bookkeeping.

This helper performs a canonical "clone + slight mutation" workflow while
keeping `Neat`'s internal invariants intact. It is intended for callers that
want a child genome derived from a single parent but do not want to perform the
bookkeeping and registration steps manually. The function deliberately does NOT
add the returned child to `this.population` so callers are free to inspect or
further modify the child and then register it via `addGenome()` (or push it
directly if they understand the consequences).

Behavior summary:
- Clone the provided `parent` (`parent.clone()` when available, else JSON round-trip).
- Clear fitness/score on the child and assign a fresh unique `_id`.
- If lineage tracking is enabled, set `(child as any)._parents = [parent._id]`
  and `(child as any)._depth = (parent._depth ?? 0) + 1`.
- Enforce structural invariants by calling `ensureMinHiddenNodes(child)` and
  `ensureNoDeadEnds(child)` so the child is valid for subsequent mutation/evaluation.
- Apply `mutateCount` mutations selected via `selectMutationMethod` and driven by
  the instance RNG (`_getRNG()`); mutation exceptions are caught and ignored to
  preserve best-effort behavior during population seeding/expansion.
- Invalidate per-genome caches with `_invalidateGenomeCaches(child)` before return.

Important: the returned child is not registered in `Neat.population` — call
`addGenome(child, [parentId])` to insert it and keep telemetry/lineage consistent.

Parameters:
- `parent` - - Source genome to derive from. Must be a `Network` instance.
- `mutateCount` - - Number of mutation operations to apply to the spawned child (default: 1).

Returns: A new `Network` instance derived from `parent`. The child is unregistered.

#### squash

`(x: number, derivate: boolean | undefined) => number`

The activation function (squashing function) applied to the node's state.
Maps the internal state to the node's output (activation).

Parameters:
- `x` - The node's internal state (sum of weighted inputs + bias).
- `derivate` - If true, returns the derivative of the function instead of the function value.

Returns: The activation value or its derivative.

#### state

The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.

#### test

`(set: { input: number[]; output: number[]; }[], cost: any) => { error: number; time: number; }`

Tests the network's performance on a given dataset.
Calculates the average error over the dataset using a specified cost function.
Uses `noTraceActivate` for efficiency as gradients are not needed.
Handles dropout scaling if dropout was used during training.

Parameters:
- `` - - The test dataset, an array of objects with `input` and `output` arrays.
- `` - - The cost function to evaluate the error. Defaults to Mean Squared Error.

Returns: An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

`() => any`

Import a previously exported state bundle and rehydrate a Neat instance.

#### toJSON

`() => object`

Converts the network into a JSON object representation (latest standard).
Includes formatVersion, and only serializes properties needed for full reconstruction.
All references are by index. Excludes runtime-only properties (activation, state, traces).

Returns: A JSON-compatible object representing the network.

#### toJSON

`() => { index: number | undefined; bias: number; type: string; squash: string | null; mask: number; }`

Converts the node's essential properties to a JSON object for serialization.
Does not include state, activation, error, or connection information, as these
are typically transient or reconstructed separately.

Returns: A JSON representation of the node's configuration.

#### toJSON

`() => { size: number; nodeIndices: (number | undefined)[]; connections: { in: number; out: number; self: number; }; }`

Serializes the group into a JSON-compatible format, avoiding circular references.
Only includes node indices and connection counts.

Returns: A JSON-compatible representation of the group.

#### toONNX

`() => import("D:/code-practice/NeatapticTS/src/architecture/network/network.onnx").OnnxModel`

Exports the network to ONNX format (JSON object, minimal MLP support).
Only standard feedforward architectures and standard activations are supported.
Gating, custom activations, and evolutionary features are ignored or replaced with Identity.

Returns: ONNX model as a JSON object.

#### totalDeltaBias

Accumulates changes in bias over a mini-batch during batch training. Reset after each weight update.

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### type

The type of the node: 'input', 'hidden', or 'output'.
Determines behavior (e.g., input nodes don't have biases modified typically, output nodes calculate error differently).

#### ungate

`(connection: import("D:/code-practice/NeatapticTS/src/architecture/connection").default) => void`

Removes the gate from a specified connection.
The connection will no longer be modulated by its gater node.
Removes the connection from the network's `gates` list.

Parameters:
- `` - - The connection object to ungate.

#### ungate

`(connections: import("D:/code-practice/NeatapticTS/src/architecture/connection").default | import("D:/code-practice/NeatapticTS/src/architecture/connection").default[]) => void`

Removes this node's gating control over the specified connection(s).
Resets the connection's gain to 1 and removes it from the `connections.gated` list.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to ungate.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.
