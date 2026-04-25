# src

Root orchestration surface for NeuroEvolution of Augmenting Topologies (NEAT) in NeatapticTS.

This root chapter is the public control desk for the library. The
architecture surfaces explain what a network graph is. The `src/neat/**`
chapters explain how evaluation, reproduction, speciation, telemetry, and
persistence work in detail. This file sits between those two layers and
answers the first practical reader question: how do I run one evolutionary
experiment without learning every subsystem in the same breath?

That boundary matters because a useful NEAT run is more than "mutate a
network." A controller has to keep a population alive, protect enough
diversity to avoid early collapse, score genomes fairly, record what
happened, and give the caller a deterministic way to pause or resume the
search. The root surface is where those responsibilities become one readable
workflow instead of a pile of helper calls.

One helpful mental model is to read the controller as four shelves. The setup
shelf decides population size, defaults, and reproducibility. The search
shelf drives `evaluate()`, `evolve()`, and the public mutation hooks. The
observability shelf exposes telemetry, lineage, diversity, species, and
Pareto views. The persistence shelf turns a live run into replayable state
through `toJSON()`, `exportState()`, and RNG snapshots.

The chapter also exists to keep the public class orchestration-first after
the internal split. `src/neat/**` now owns the heavier policy chapters:
`evaluate/` scores genomes, `evolve/` creates the next generation,
`speciation/` manages compatibility pressure, `telemetry/` records what the
run did, and `export/` plus `rng/` keep experiments reproducible. If you can
read the root workflow first, the subchapters become "why does this step
work?" reads instead of "where do I even start?" reads.

The guiding historical idea comes from Stanley and Miikkulainen's NEAT
paper: evolve both weights and topology while protecting innovation long
enough for new structures to prove useful. See Stanley and Miikkulainen,
[Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02),
for the compact background behind the controller vocabulary that appears all
through this surface.

Read this root when you want to answer one of three questions quickly: how to
start a run, how to move it forward generation by generation, or how to
inspect and persist it without diving into every implementation chapter. Read
the narrower `src/neat/**` READMEs when the next question becomes about one
specific policy family rather than the whole experiment loop.

```mermaid
flowchart LR
  classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
  classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;

  Configure["Configure run<br/>sizes + fitness + options"]:::accent --> Seed["Seed or restore<br/>constructor / createPool / import"]:::base
  Seed --> Score["Score population<br/>evaluate()"]:::base
  Score --> Breed["Breed next generation<br/>evolve() / mutate()"]:::accent
  Breed --> Observe["Observe pressure<br/>telemetry / species / diversity / Pareto"]:::base
  Observe --> Persist["Persist or replay<br/>exportState() / toJSON() / RNG state"]:::base
  Breed --> Score
```

```mermaid
flowchart TD
  classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
  classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;

  Root["src root controller"]:::accent --> Setup["Defaults and initialization<br/>constructor / createPool / import"]:::base
  Root --> Search["Search loop<br/>evaluate / evolve / mutate"]:::base
  Root --> Observe["Diagnostics<br/>telemetry / species / lineage / Pareto"]:::base
  Root --> Replay["Persistence<br/>toJSON / exportState / RNG"]:::base
  Search --> Chapters["Detailed policy chapters<br/>src/neat/**"]:::base
```

Example: start a small deterministic run and inspect the best score after one
generation.

```ts
const neat = new Neat(2, 1, fitness, {
  popsize: 50,
  seed: 7,
  fastMode: true,
});

await neat.evaluate();
const bestGenome = await neat.evolve();

console.log(bestGenome.score);
```

Example: capture telemetry and a replayable snapshot after the current run
step.

```ts
const latestTelemetry = neat.getTelemetry().at(-1);
const exportedState = neat.exportState();

console.log(latestTelemetry?.generation);
console.log(exportedState.neat.generation);
```

Recommended reading after this root chapter:
- `./neat/evaluate/README.md` for scoring flow and objective handling
- `./neat/evolve/README.md` for reproduction orchestration
- `./neat/speciation/README.md` for compatibility distance and sharing
- `./neat/telemetry/README.md` for diagnostics and export surfaces

## neat.ts

### DEFAULT_COMPATIBILITY_THRESHOLD

Default compatibility threshold controlling speciation distance.

This starts the speciation-pressure family of defaults. It is the neutral
boundary the controller uses before adaptive tuning or custom settings make
species splits stricter or more permissive.

### DEFAULT_DISJOINT_COEFF

Default disjoint coefficient for NEAT compatibility distance.

Matching the excess coefficient by default gives the root controller a
balanced structural view: excess and disjoint innovation gaps both count as
first-class evidence during compatibility comparisons.

### DEFAULT_DIVERSITY_GRAPHLET_SAMPLE

Default graphlet sample size used by diversity metrics in fast mode.

Read this beside {@link DEFAULT_DIVERSITY_PAIR_SAMPLE}: pair samples give the
controller quick distance evidence, while graphlet samples provide a small
structural texture read without forcing whole-population analysis.

### DEFAULT_DIVERSITY_PAIR_SAMPLE

Default pair-sample size used by diversity metrics in fast mode.

This starts the observability-sampling family. The root controller uses a
bounded sample instead of exhaustive pair checks so diversity reads stay
cheap enough for ordinary runs.

### DEFAULT_ELITISM

Default elitism count applied when unspecified.

Read this beside {@link DEFAULT_POPULATION_SIZE} and
{@link DEFAULT_PROVENANCE}: the trio defines how much of each generation is
reserved for carry-over, how much is freshly injected, and how much capacity
remains for ordinary offspring.

### DEFAULT_EXCESS_COEFF

Default excess coefficient for NEAT compatibility distance.

This begins the root compatibility-weight family. These coefficients explain
which kinds of genome disagreement matter most when the controller decides
whether two genomes still belong in the same species neighborhood.

### DEFAULT_MAX_CONNS

Default maximum allowed connections where `Infinity` means unbounded growth.

This preserves the same baseline policy as {@link DEFAULT_MAX_NODES}: the
controller does not impose a fixed connection ceiling unless the caller wants
one.

### DEFAULT_MAX_GATES

Default maximum allowed gates where `Infinity` means unbounded growth.

Gate limits stay in the same family as node and connection limits so the
whole structural-cap story remains consistent at the root surface.

### DEFAULT_MAX_NODES

Default maximum allowed nodes where `Infinity` means unbounded growth.

Read the three `DEFAULT_MAX_*` exports as one structural-ceiling family.
Leaving them unbounded by default tells the root controller to rely on
mutation policy, pruning, and adaptive limits instead of an immediate hard
cap.

### DEFAULT_MUTATION_AMOUNT

Default number of mutation operations applied per genome.

The default keeps the baseline search policy conservative: most runs mutate
often enough to keep topology moving, but each genome usually pays for only
one structural or parametric change per mutation pass.

### DEFAULT_MUTATION_RATE

Default mutation rate used by the root controller when no explicit rate is supplied.

This belongs to the same search-tempo family as
{@link DEFAULT_MUTATION_AMOUNT}. Together they define how often mutation is
attempted and how many mutation steps a genome can receive once mutation is
active.

### DEFAULT_NOVELTY_K

Default neighbor count for novelty search when `k` is unspecified.

This closes the root observability-and-exploration shelf. It controls how
many nearby behaviors contribute to novelty before the caller tunes novelty
search more explicitly.

### DEFAULT_POPULATION_SIZE

Default population size when caller does not specify `popsize`.

This opens the root defaults shelf's search-volume family. It controls how
many genomes compete in each generation before elitism, provenance, or
mutation pressure begin to reshape the population.

### DEFAULT_PROVENANCE

Default provenance count applied when unspecified.

Provenance is the root controller's small "fresh seed" policy. A value of
`0` means the default run does not spend population budget on extra
generation-zero style injections unless the caller asks for them.

### DEFAULT_WEIGHT_DIFF_COEFF

Default average weight difference coefficient for compatibility distance.

This keeps parameter drift relevant without letting weight deltas dominate
the whole speciation read. In the default family, topology disagreement still
carries more weight than modest edge-weight differences.

### Neat

High-level NEAT controller that keeps the public workflow linear while the implementation stays chaptered.

If you are learning the library, this is the class to read first. It owns the
practical experiment loop and answers the first questions most users ask:
how to seed a population, when to call `evaluate()` versus `evolve()`, how to
inspect species and telemetry, and how to export or replay a run deterministically.

Design-wise, `Neat` is intentionally orchestration-first. Mutation operators,
speciation rules, telemetry formatting, archive management, cache invalidation,
and pruning policies all live in dedicated modules so this top-level surface can
stay readable even as the underlying algorithm becomes richer.

Minimal workflow:

```ts
const neat = new Neat(2, 1, fitness, {
  popsize: 50,
  seed: 7,
  fastMode: true,
});

await neat.evaluate();
const bestGenome = await neat.evolve();

console.log(bestGenome.score);
console.log(neat.getTelemetry().at(-1));
```

#### _applyFitnessSharing

```ts
_applyFitnessSharing(): void
```

Apply fitness sharing adjustments within each species.

Returns: Adjusted species fitness data.

#### _compatibilityDistance

```ts
_compatibilityDistance(
  netA: default,
  netB: default,
): number
```

Compute compatibility distance between two networks (delegates to compat module).

Parameters:
- `netA` - First network for comparison.
- `netB` - Second network for comparison.

Returns: Compatibility distance scalar.

#### _computeDiversityStats

```ts
_computeDiversityStats(): DiversityStats
```

Compute and cache diversity statistics used by telemetry and tests.

Returns: Cached diversity statistics snapshot.

#### _diversityStats

Cached diversity metrics (computed lazily).

#### _fallbackInnov

```ts
_fallbackInnov(
  conn: ConnectionLike,
): number
```

Fallback innovation id resolver used when reuse mapping is absent.

Parameters:
- `conn` - Connection metadata used to derive the innovation id.

Returns: Innovation id for the connection.

#### _getObjectives

```ts
_getObjectives(): ObjectiveDescriptor[]
```

Internal: return cached objective descriptors, building if stale.

Returns: Cached or freshly built objective descriptors.

#### _getRNG

```ts
_getRNG(): () => number
```

Provide a memoized RNG function, initializing from internal state if needed.

Returns: RNG function bound to this instance.

#### _innovationTracker

Explicit owner for global innovation ids and generation-local reuse state.

#### _invalidateGenomeCaches

```ts
_invalidateGenomeCaches(
  genome: unknown,
): void
```

Invalidate per-genome caches (compatibility distance, forward pass, etc.).

Parameters:
- `genome` - Genome instance whose caches should be cleared.

#### _lastEvalDuration

Duration of the last evaluation run (ms).

#### _lastEvolveDuration

Duration of the last evolve run (ms).

#### _lastInbreedingCount

Last observed count of inbreeding (used for detecting excessive cloning).

#### _lineageEnabled

Whether lineage metadata should be recorded on genomes.

#### _mutateAddConnReuse

```ts
_mutateAddConnReuse(
  genome: default,
): void
```

Add-connection mutation that reuses global innovation ids when possible.

Parameters:
- `genome` - Genome receiving the mutation.

Returns: Mutated genome with added connection.

#### _mutateAddNodeReuse

```ts
_mutateAddNodeReuse(
  genome: default,
): Promise<void>
```

Add-node mutation that reuses global innovation ids when possible.

Parameters:
- `genome` - Genome receiving the mutation.

Returns: Mutated genome with added node.

#### _nextGenomeId

Counter for assigning unique genome ids.

#### _noveltyArchive

Novelty archive used by novelty search (behavior representatives).

#### _objectiveEvents

Queue of recent objective activation/deactivation events for telemetry.

#### _operatorStats

Operator statistics used by adaptive operator selection.

#### _paretoArchive

Archive of Pareto front metadata for multi-objective tracking.

#### _paretoObjectivesArchive

Archive storing Pareto objectives snapshots.

#### _prepareInnovationTrackerGeneration

```ts
_prepareInnovationTrackerGeneration(
  targetGeneration: number,
): void
```

Prepare the innovation tracker for a specific mutation-generation window.

#### _rng

Cached RNG function; created lazily and seeded from `_rngState` when used.

#### _rngState

Internal numeric state for the deterministic xorshift RNG when no user RNG is provided.

#### _sortSpeciesMembers

```ts
_sortSpeciesMembers(
  sp: SpeciesLike,
): void
```

Sort members within a species according to fitness and lineage rules.

Parameters:
- `sp` - Species whose members should be sorted.

Returns: Sorted species members.

#### _speciate

```ts
_speciate(): void
```

Partition population into species using configured compatibility metrics.

Returns: Updated species assignments.

#### _speciesHistory

Time-series history of species stats (for exports/telemetry).

#### _structuralEntropy

```ts
_structuralEntropy(
  genome: default,
): number
```

Compatibility wrapper retained for tests that reach `_structuralEntropy` through loose controller casts.

Parameters:
- `genome` - Genome whose structural entropy is calculated.

Returns: Structural entropy score for the genome.

#### _telemetry

Telemetry buffer storing diagnostic snapshots per generation.

#### _updateSpeciesStagnation

```ts
_updateSpeciesStagnation(): void
```

Update stagnation metrics per species to inform pruning and selection.

Returns: Updated stagnation state.

#### _warnIfNoBestGenome

```ts
_warnIfNoBestGenome(): void
```

Emit a standardized warning when evolution loop finds no valid best genome (test hook).

#### addGenome

```ts
addGenome(
  genome: default,
  parents: number[] | undefined,
): void
```

Register an externally-created genome into the `Neat` population.

Parameters:
- `genome` - Genome to append into the population.
- `parents` - Optional lineage metadata recorded for teaching and telemetry.

#### applyAdaptivePruning

```ts
applyAdaptivePruning(): Promise<void>
```

Run the adaptive pruning controller once using the controller's latest signals.

Unlike scheduled pruning, this path reacts to the current search state,
such as stagnation or complexity pressure, instead of only looking at the
generation index.

Returns: Promise resolving after adaptive pruning has completed.

#### applyEvolutionPruning

```ts
applyEvolutionPruning(): Promise<void>
```

Manually apply the configured generation-based pruning policy once.

This is mainly useful when you are experimenting with pruning behavior and
want to trigger the controller's scheduled pruning logic outside the normal
evolve loop.

Returns: Promise resolving after the pruning policy has been evaluated.

#### clearObjectives

```ts
clearObjectives(): void
```

Remove all custom objective registrations.

Use this when a run is changing from one multi-objective regime to another
and you want the controller to forget the previous objective schema.

#### clearParetoArchive

```ts
clearParetoArchive(): void
```

Clear the stored Pareto archive.

Use this when a new phase of a run should stop comparing itself against the
previous archive history.

#### clearTelemetry

```ts
clearTelemetry(): void
```

Clear the recorded telemetry history.

This does not reset the population or controller options. It only removes
the accumulated diagnostic snapshots so a new experiment phase can start
with a clean telemetry timeline.

#### createPool

```ts
createPool(
  network: default | null,
): void
```

Create the initial population pool, optionally cloning from a seed network.

This is the explicit population bootstrap surface. Call it when you want to
start from a known architecture template instead of relying on whatever setup
a surrounding example or harness applies for you.

Parameters:
- `network` - Optional template network copied into the initial pool.

#### ensureMinHiddenNodes

```ts
ensureMinHiddenNodes(
  network: default,
  multiplierOverride: number | undefined,
): Promise<void>
```

Ensure a network has the minimum number of hidden nodes according to configured policy.

#### ensureNoDeadEnds

```ts
ensureNoDeadEnds(
  network: default,
): void
```

Repair dead-end connectivity through the focused maintenance facade.

#### evaluate

```ts
evaluate(): Promise<void>
```

Evaluate the current population using the configured fitness function.
Delegates to the migrated evaluation helper to keep this class thin.

In practice, this is the scoring half of the controller loop. It transforms a
population of candidate networks into evidence the rest of the algorithm can use:
fitness scores, objective values, telemetry, diversity statistics, and derived
signals needed by selection or pruning.

Returns: Aggregated evaluation result (implementation specific).

#### evolve

```ts
evolve(): Promise<default>
```

Advance the evolutionary loop by one generation.

Conceptually, `evolve()` is the reproduction half of NEAT. It selects parents,
preserves elites and provenance when configured, applies structural and parametric
mutation, updates search bookkeeping, and returns the best genome observed for the step.
The heavy mechanics live in `src/neat/evolve/evolve.ts`; this method stays as the
readable front door to that orchestration.

Returns: Best genome selected by the evolution step.

Example:

// Score the current population first, then breed the next generation.
await neat.evaluate();
await neat.evolve();

#### export

```ts
export(): GenomeJSON[]
```

Export the current population as plain JSON objects.

Choose this lighter snapshot when you only need the genomes themselves and
do not need generation counters, innovation maps, or other controller-level
state.

Returns: JSON-safe population snapshot.

#### exportParetoFrontJSONL

```ts
exportParetoFrontJSONL(
  maxEntries: number,
): string
```

Export recent Pareto archive entries as JSON Lines.

This is the easiest way to persist frontier history for offline analysis or
later replay in notebooks and visualization tools.

Parameters:
- `maxEntries` - Maximum number of recent archive entries to export.

Returns: JSONL payload for the requested Pareto archive window.

#### exportRNGState

```ts
exportRNGState(): number | undefined
```

Export the current RNG state for external persistence or tests.

Returns: Opaque RNG snapshot suitable for later replay.

#### exportSpeciesHistoryCSV

```ts
exportSpeciesHistoryCSV(
  maxEntries: number,
): string
```

Export recent species history as CSV.

This is useful when you want to chart species growth, collapse, or
stagnation in a spreadsheet or notebook without writing a custom parser.

Parameters:
- `maxEntries` - Maximum number of recent history entries to export.

Returns: CSV payload representing recent species history snapshots.

#### exportSpeciesHistoryJSONL

```ts
exportSpeciesHistoryJSONL(
  maxEntries: number,
): string
```

Export recent species history as JSON Lines.

Choose this when you want machine-friendly archival output instead of the
flatter spreadsheet-oriented CSV export.

Parameters:
- `maxEntries` - Maximum number of recent history entries to export.

Returns: JSONL payload describing recent species-history entries.

#### exportState

```ts
exportState(): NeatStateJSON
```

Export the full controller state, including metadata and population.

This is the pause-and-resume snapshot. It is the best choice when you want
to continue the same run later with the same innovation history, generation
counter, and serialized genomes.

Returns: Full controller snapshot including metadata and population.

#### exportTelemetryCSV

```ts
exportTelemetryCSV(
  maxEntries: number,
): string
```

Export recent telemetry entries as CSV.

Parameters:
- `maxEntries` - Maximum number of recent telemetry entries to export.

Returns: CSV string for quick spreadsheet or notebook analysis.

#### exportTelemetryJSONL

```ts
exportTelemetryJSONL(): string
```

Export the telemetry buffer as JSON Lines.

JSONL is the easiest format to append to files, stream into data tools, or
inspect generation-by-generation without loading a giant array into memory.

Returns: JSONL payload with one telemetry entry per line.

#### fromJSON

```ts
fromJSON(
  json: NeatMetaJSON,
  fitness: NeatFitnessFunction,
): default
```

Rebuild a `Neat` controller from serialized metadata without importing a population bundle.

This is the lighter-weight sibling of `importState()`. It is useful when you
want controller defaults, innovation bookkeeping, or archive metadata back,
but you are handling genome population state separately.

Parameters:
- `json` - Serialized controller metadata produced by `toJSON()`.
- `fitness` - Fitness function to attach to the reconstructed controller.

Returns: Reconstructed `Neat` controller instance.

#### getAverage

```ts
getAverage(): number
```

Calculates the average fitness score of the population.

#### getDiversityStats

```ts
getDiversityStats(): DiversityStats
```

Return the latest cached diversity statistics.

Diversity summaries answer a different question than raw fitness: whether
the population still explores varied structures or is collapsing toward a
narrower family of genomes.

Returns: Diversity metrics for the current population snapshot.

#### getFittest

```ts
getFittest(): default
```

Retrieves the fittest genome from the population.

#### getLineageSnapshot

```ts
getLineageSnapshot(
  limit: number,
): { id: number; parents: number[]; }[]
```

Return an array of {id, parents} for the first `limit` genomes in population.

Parameters:
- `limit` - Maximum number of lineage records to return.

Returns: Compact lineage snapshot for debugging and teaching inheritance flow.

#### getMinimumHiddenSize

```ts
getMinimumHiddenSize(
  multiplierOverride: number | undefined,
): number
```

Minimum hidden size considering explicit minHidden or multiplier policy.

#### getMultiObjectiveMetrics

```ts
getMultiObjectiveMetrics(): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Return compact multi-objective metrics for each genome in the current population.

Use this when you want a flattened view of Pareto rank, crowding, raw score,
and structural size without reconstructing the full fronts yourself.

Returns: One compact metric record per genome.

#### getNoveltyArchiveSize

```ts
getNoveltyArchiveSize(): number
```

Return the current novelty-archive size.

This is a small diagnostic hook that tells you whether novelty search is
actively accumulating behavior representatives or staying mostly unused.

Returns: Number of archived novelty descriptors.

#### getObjectiveEvents

```ts
getObjectiveEvents(): { gen: number; type: "add" | "remove"; key: string; }[]
```

Get recent objective add/remove events for telemetry exports and teaching.

#### getObjectiveKeys

```ts
getObjectiveKeys(): string[]
```

Public helper returning just the objective keys (tests rely on).

#### getObjectives

```ts
getObjectives(): { key: string; direction: "max" | "min"; }[]
```

Return a lightweight list of registered objective keys and their directions.

Returns: Objective descriptors currently active on the controller.

#### getOffspring

```ts
getOffspring(): default
```

Build a child genome from parent selection and crossover.

Use this when you want one reproduction event without running a full
generation step. The method delegates the parent choice to `getParent()` so
it still respects the controller's current breeding policy.

Returns: New network created from selected parent genomes.

#### getOperatorStats

```ts
getOperatorStats(): { name: string; success: number; attempts: number; }[]
```

Return mutation-operator success statistics.

These numbers are useful when operator adaptation is enabled and you want
to inspect which mutation operators are being rewarded or ignored.

Returns: Per-operator attempt and success counters.

#### getParent

```ts
getParent(): default
```

Select a parent genome using the controller's configured selection strategy.

Read this as the "who gets to reproduce" hook. The exact policy depends on
the current selection configuration, but the intent is always the same:
convert the scored population into a plausible breeding candidate.

Returns: The selected parent genome.

#### getParetoArchive

```ts
getParetoArchive(
  maxEntries: number,
): ParetoArchiveEntry[]
```

Return recent Pareto archive entries.

This is the metadata-oriented archive view. Use it when you want to inspect
what front snapshots were retained over time without exporting the full JSONL
payload first.

Parameters:
- `maxEntries` - Maximum number of recent archive entries to return.

Returns: Recent Pareto archive metadata entries.

#### getParetoFronts

```ts
getParetoFronts(
  maxFronts: number,
): default[][]
```

Reconstruct Pareto fronts for the current population snapshot.

Parameters:
- `maxFronts` - Maximum number of fronts to materialize.

Returns: Fronts ordered from most to least dominant under the active objectives.

#### getPerformanceStats

```ts
getPerformanceStats(): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Return timing statistics for the latest evaluation and evolution steps.

This is a lightweight performance probe for experiments and benchmarks that
need to notice when scoring or breeding costs start drifting upward.

Returns: Recent runtime statistics for evaluation and evolution work.

#### getSpeciesHistory

```ts
getSpeciesHistory(): SpeciesHistoryEntry[]
```

Return the recorded species-history timeline.

Unlike `getSpeciesStats()`, which only reflects the current generation,
this method exposes the historical view used for trend analysis.

Returns: Species history entries in recorded order.

#### getSpeciesStats

```ts
getSpeciesStats(): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Return a compact per-species summary for the current population snapshot.

This is the quickest inspection surface when you want to know how many
niches currently exist, how large they are, and whether they have improved
recently.

Returns: One summary record per active species.

#### getTelemetry

```ts
getTelemetry(): TelemetryEntry[]
```

Return the internal telemetry buffer.

Telemetry is the controller's teaching surface for understanding why a run is
behaving a certain way. Instead of watching only the best score, you can inspect
species counts, diversity, objective events, evaluation timing, and other search signals.

Returns: Recorded telemetry entries in chronological order.

#### import

```ts
import(
  json: GenomeJSON[],
): Promise<void>
```

Replace the current population with serialized genomes.

This is the population-only restore path. It keeps the current controller
instance, options, and metadata while swapping in a different genome set.
Use `importState()` when you want to restore controller metadata too.

Parameters:
- `json` - Serialized population to import into the current controller.

Returns: Promise resolving after the population is loaded.

#### importRNGState

```ts
importRNGState(
  state: string | number | undefined,
): void
```

Import an RNG state (alias for restore; kept for compatibility).

Parameters:
- `state` - Numeric RNG state.

Returns: Nothing. This is a compatibility alias for `restoreRNGState()`.

#### importState

```ts
importState(
  bundle: NeatStateJSON,
  fitness: NeatFitnessFunction,
): Promise<default>
```

Restore a full evolutionary snapshot produced by `exportState()`.

Use this when you want a paused experiment to resume with its controller
metadata, population, and archival context intact rather than rebuilding
only the bare genomes.

Parameters:
- `bundle` - Serialized object with the shape `{ neat, population }`.
- `fitness` - Fitness function to attach to the restored controller.

Returns: A `Neat` instance ready to continue evolution from the imported state.

#### mutate

```ts
mutate(): Promise<void>
```

Apply mutation pressure to the current population without advancing generation bookkeeping.

This is the direct "variation" lever. Use it when you want to perturb the
current genomes in place for an experiment, a custom training loop, or a
test harness that separates mutation from the rest of `evolve()`.
In the normal NEAT workflow, `evolve()` is usually the better entry point
because it coordinates parent selection, elitism, offspring creation, and
mutation as one generation step.

Returns: Promise resolving once mutation has been applied to the current population.

#### registerObjective

```ts
registerObjective(
  key: string,
  direction: "max" | "min",
  accessor: (network: default) => number,
): void
```

Register a custom objective for multi-objective optimization.

Register objectives when a single scalar score is too narrow to express the
behavior you want. The controller can then reason about tradeoffs such as raw
score versus simplicity, novelty, or domain-specific constraints.

Parameters:
- `key` - Stable objective identifier used in exports and telemetry.
- `direction` - Whether the objective should be minimized or maximized.
- `accessor` - Function extracting the objective value from a genome.

#### resetNoveltyArchive

```ts
resetNoveltyArchive(): void
```

Reset the novelty archive.

This is useful when you want to restart novelty pressure from a clean slate
without rebuilding the whole controller.

#### restoreRNGState

```ts
restoreRNGState(
  state: string | number | undefined,
): void
```

Restore a previously-snapshotted RNG state. This restores the internal
seed but does not re-create the RNG function until next use.

Parameters:
- `state` - Opaque numeric RNG state produced by `snapshotRNGState()`.

Returns: Nothing. The controller will resume from the restored RNG state on next use.

#### sampleRandom

```ts
sampleRandom(
  sampleCount: number,
): number[]
```

Produce deterministic random samples using the instance RNG.

Parameters:
- `sampleCount` - Number of random values to generate.

Returns: Array of deterministic random samples.

#### selectMutationMethod

```ts
selectMutationMethod(
  genome: default,
  rawReturnForTest: boolean,
): Promise<MutationMethod | MutationMethod[] | null>
```

Selects a mutation method for a given genome based on constraints.

Parameters:
- `genome` - Genome being considered for mutation.
- `rawReturnForTest` - Whether to expose raw selection output for test visibility.

Returns: Selected mutation method or `null` when no valid method can be chosen.

#### snapshotRNGState

```ts
snapshotRNGState(): number | undefined
```

Return the current opaque RNG numeric state used by the instance.
Useful for deterministic test replay and debugging.

Returns: Snapshot of the current controller RNG state.

#### sort

```ts
sort(): void
```

Sorts the population in descending order of fitness scores.

#### spawnFromParent

```ts
spawnFromParent(
  parent: default,
  mutateCount: number,
): default
```

Spawn a new genome derived from a single parent while preserving Neat bookkeeping.

Parameters:
- `parent` - Parent genome to clone and mutate.
- `mutateCount` - Number of mutation passes to apply to the child.

Returns: Child genome registered with the same bookkeeping conventions as normal evolution.

#### toJSON

```ts
toJSON(): NeatMetaJSON
```

Serialize controller metadata without the concrete population.

This is useful when you want to preserve run configuration and innovation
bookkeeping separately from genome payloads, or when the population will be
reconstructed by other means.

Returns: JSON-safe metadata snapshot useful for innovation-history persistence.

### NeatOptions

Public configuration bag for `Neat` evolutionary runs.

`NeatOptions` collects the knobs that shape how search pressure is applied.
In practice, readers can think about the options in four teaching-friendly groups:

- search size and tempo: `popsize`, `elitism`, `provenance`, `mutationRate`, `mutationAmount`
- species formation: compatibility threshold plus the excess, disjoint, and weight-difference coefficients
- observability: telemetry, lineage, diversity sampling, species history, Pareto archive controls
- reproducibility: `seed`, imported RNG state, and exported run state

That organization matters because most experiment tuning questions are really
questions about pressure: how many candidates compete, how disruptive mutation
should feel, how aggressively genomes split into species, and how much evidence
you want to retain while the run is unfolding.

```ts
const options: NeatOptions = {
  popsize: 150,
  elitism: 5,
  mutationRate: 0.6,
  compatibilityThreshold: 3,
  fastMode: true,
  seed: 42,
};

const neat = new Neat(3, 1, fitness, options);
```

This alias stays intentionally permissive for compatibility with legacy callers.
Prefer treating it as the stable front door and the narrower helper-level types in
`src/neat/**` as implementation detail.

## config.ts

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

## neataptic.ts

### neataptic

Node (Neuron)
=============
Fundamental computational unit: aggregates weighted inputs, applies an activation
function (squash) and emits an activation value. Supports:
 - Types: 'input' | 'hidden' | 'output' (affects bias initialization & error handling)
 - Recurrent self‑connections & gated connections (for dynamic / RNN behavior)
 - Dropout mask (`mask`), momentum terms, eligibility & extended traces (for
   a variety of learning rules beyond simple backprop).
 - Optional descriptors via `describe({ label, intent, metadata })` when a
   low-level node should keep a stable human-facing identity inside a larger
   architecture story.

Educational note: Traces (`eligibility` and `xtrace`) illustrate how recurrent credit
assignment works in algorithms like RTRL / policy gradients. They are updated only when
using the traced activation path (`activate`) vs `noTraceActivate` (inference fast path).

Most architecture code should only attach a descriptor when a node boundary
matters outside the current function. That keeps the primitive cheap for raw
graph math while still letting later passes recover names such as
`readoutNode`, `memoryCell`, or `temperatureGate`.

Examples:

```ts
const sensor = new Node('input');
const readout = new Node('output');

readout.describe({
  label: 'readoutNode',
  metadata: { stage: 'policy' },
});
```

```ts
const sensorBlock = new Group(4, 'input');
const readoutBlock = new Group(2, 'output');

sensorBlock.describe({
  label: 'sensorBlock',
  metadata: { stage: 'encoder' },
});
readoutBlock.describe({
  label: 'readoutBlock',
  metadata: { stage: 'readout' },
});

sensorBlock.connect(
  readoutBlock,
  methods.groupConnection.ALL_TO_ALL,
);
```

```ts
const source = new Node('input');
const target = new Node('output');
const edge = new Connection(source, target, 0.42);

edge.gain = 1.5;
edge.enabled = true;
```

```ts
const network = Architect.perceptron(2, 4, 1);
const output = network.activate([0, 1]);
```

### formatConstructSummary

```ts
formatConstructSummary(
  constructResult: ConstructResult,
): string
```

Build a compact human-readable summary for one construct-from-parts result.

The summary is layered on top of the detached `ConstructResult.graph`
snapshot so tooling can log or display one stable explanation of the built
graph without reading mutable `Network` internals.

Parameters:
- `constructResult` - Construct result returned by `Network.construct(...)`.

Returns: Multi-line summary string suitable for logs, diagnostics panels, or snapshots.

Example:

```ts
const construction = Network.construct([sensor, hidden, readout]);
const summary = formatConstructSummary(construction);
```

### Neat

High-level NEAT controller that keeps the public workflow linear while the implementation stays chaptered.

If you are learning the library, this is the class to read first. It owns the
practical experiment loop and answers the first questions most users ask:
how to seed a population, when to call `evaluate()` versus `evolve()`, how to
inspect species and telemetry, and how to export or replay a run deterministically.

Design-wise, `Neat` is intentionally orchestration-first. Mutation operators,
speciation rules, telemetry formatting, archive management, cache invalidation,
and pruning policies all live in dedicated modules so this top-level surface can
stay readable even as the underlying algorithm becomes richer.

Minimal workflow:

```ts
const neat = new Neat(2, 1, fitness, {
  popsize: 50,
  seed: 7,
  fastMode: true,
});

await neat.evaluate();
const bestGenome = await neat.evolve();

console.log(bestGenome.score);
console.log(neat.getTelemetry().at(-1));
```

#### _applyFitnessSharing

```ts
_applyFitnessSharing(): void
```

Apply fitness sharing adjustments within each species.

Returns: Adjusted species fitness data.

#### _compatibilityDistance

```ts
_compatibilityDistance(
  netA: default,
  netB: default,
): number
```

Compute compatibility distance between two networks (delegates to compat module).

Parameters:
- `netA` - First network for comparison.
- `netB` - Second network for comparison.

Returns: Compatibility distance scalar.

#### _computeDiversityStats

```ts
_computeDiversityStats(): DiversityStats
```

Compute and cache diversity statistics used by telemetry and tests.

Returns: Cached diversity statistics snapshot.

#### _diversityStats

Cached diversity metrics (computed lazily).

#### _fallbackInnov

```ts
_fallbackInnov(
  conn: ConnectionLike,
): number
```

Fallback innovation id resolver used when reuse mapping is absent.

Parameters:
- `conn` - Connection metadata used to derive the innovation id.

Returns: Innovation id for the connection.

#### _getObjectives

```ts
_getObjectives(): ObjectiveDescriptor[]
```

Internal: return cached objective descriptors, building if stale.

Returns: Cached or freshly built objective descriptors.

#### _getRNG

```ts
_getRNG(): () => number
```

Provide a memoized RNG function, initializing from internal state if needed.

Returns: RNG function bound to this instance.

#### _innovationTracker

Explicit owner for global innovation ids and generation-local reuse state.

#### _invalidateGenomeCaches

```ts
_invalidateGenomeCaches(
  genome: unknown,
): void
```

Invalidate per-genome caches (compatibility distance, forward pass, etc.).

Parameters:
- `genome` - Genome instance whose caches should be cleared.

#### _lastEvalDuration

Duration of the last evaluation run (ms).

#### _lastEvolveDuration

Duration of the last evolve run (ms).

#### _lastInbreedingCount

Last observed count of inbreeding (used for detecting excessive cloning).

#### _lineageEnabled

Whether lineage metadata should be recorded on genomes.

#### _mutateAddConnReuse

```ts
_mutateAddConnReuse(
  genome: default,
): void
```

Add-connection mutation that reuses global innovation ids when possible.

Parameters:
- `genome` - Genome receiving the mutation.

Returns: Mutated genome with added connection.

#### _mutateAddNodeReuse

```ts
_mutateAddNodeReuse(
  genome: default,
): Promise<void>
```

Add-node mutation that reuses global innovation ids when possible.

Parameters:
- `genome` - Genome receiving the mutation.

Returns: Mutated genome with added node.

#### _nextGenomeId

Counter for assigning unique genome ids.

#### _noveltyArchive

Novelty archive used by novelty search (behavior representatives).

#### _objectiveEvents

Queue of recent objective activation/deactivation events for telemetry.

#### _operatorStats

Operator statistics used by adaptive operator selection.

#### _paretoArchive

Archive of Pareto front metadata for multi-objective tracking.

#### _paretoObjectivesArchive

Archive storing Pareto objectives snapshots.

#### _prepareInnovationTrackerGeneration

```ts
_prepareInnovationTrackerGeneration(
  targetGeneration: number,
): void
```

Prepare the innovation tracker for a specific mutation-generation window.

#### _rng

Cached RNG function; created lazily and seeded from `_rngState` when used.

#### _rngState

Internal numeric state for the deterministic xorshift RNG when no user RNG is provided.

#### _sortSpeciesMembers

```ts
_sortSpeciesMembers(
  sp: SpeciesLike,
): void
```

Sort members within a species according to fitness and lineage rules.

Parameters:
- `sp` - Species whose members should be sorted.

Returns: Sorted species members.

#### _speciate

```ts
_speciate(): void
```

Partition population into species using configured compatibility metrics.

Returns: Updated species assignments.

#### _speciesHistory

Time-series history of species stats (for exports/telemetry).

#### _structuralEntropy

```ts
_structuralEntropy(
  genome: default,
): number
```

Compatibility wrapper retained for tests that reach `_structuralEntropy` through loose controller casts.

Parameters:
- `genome` - Genome whose structural entropy is calculated.

Returns: Structural entropy score for the genome.

#### _telemetry

Telemetry buffer storing diagnostic snapshots per generation.

#### _updateSpeciesStagnation

```ts
_updateSpeciesStagnation(): void
```

Update stagnation metrics per species to inform pruning and selection.

Returns: Updated stagnation state.

#### _warnIfNoBestGenome

```ts
_warnIfNoBestGenome(): void
```

Emit a standardized warning when evolution loop finds no valid best genome (test hook).

#### addGenome

```ts
addGenome(
  genome: default,
  parents: number[] | undefined,
): void
```

Register an externally-created genome into the `Neat` population.

Parameters:
- `genome` - Genome to append into the population.
- `parents` - Optional lineage metadata recorded for teaching and telemetry.

#### applyAdaptivePruning

```ts
applyAdaptivePruning(): Promise<void>
```

Run the adaptive pruning controller once using the controller's latest signals.

Unlike scheduled pruning, this path reacts to the current search state,
such as stagnation or complexity pressure, instead of only looking at the
generation index.

Returns: Promise resolving after adaptive pruning has completed.

#### applyEvolutionPruning

```ts
applyEvolutionPruning(): Promise<void>
```

Manually apply the configured generation-based pruning policy once.

This is mainly useful when you are experimenting with pruning behavior and
want to trigger the controller's scheduled pruning logic outside the normal
evolve loop.

Returns: Promise resolving after the pruning policy has been evaluated.

#### clearObjectives

```ts
clearObjectives(): void
```

Remove all custom objective registrations.

Use this when a run is changing from one multi-objective regime to another
and you want the controller to forget the previous objective schema.

#### clearParetoArchive

```ts
clearParetoArchive(): void
```

Clear the stored Pareto archive.

Use this when a new phase of a run should stop comparing itself against the
previous archive history.

#### clearTelemetry

```ts
clearTelemetry(): void
```

Clear the recorded telemetry history.

This does not reset the population or controller options. It only removes
the accumulated diagnostic snapshots so a new experiment phase can start
with a clean telemetry timeline.

#### createPool

```ts
createPool(
  network: default | null,
): void
```

Create the initial population pool, optionally cloning from a seed network.

This is the explicit population bootstrap surface. Call it when you want to
start from a known architecture template instead of relying on whatever setup
a surrounding example or harness applies for you.

Parameters:
- `network` - Optional template network copied into the initial pool.

#### ensureMinHiddenNodes

```ts
ensureMinHiddenNodes(
  network: default,
  multiplierOverride: number | undefined,
): Promise<void>
```

Ensure a network has the minimum number of hidden nodes according to configured policy.

#### ensureNoDeadEnds

```ts
ensureNoDeadEnds(
  network: default,
): void
```

Repair dead-end connectivity through the focused maintenance facade.

#### evaluate

```ts
evaluate(): Promise<void>
```

Evaluate the current population using the configured fitness function.
Delegates to the migrated evaluation helper to keep this class thin.

In practice, this is the scoring half of the controller loop. It transforms a
population of candidate networks into evidence the rest of the algorithm can use:
fitness scores, objective values, telemetry, diversity statistics, and derived
signals needed by selection or pruning.

Returns: Aggregated evaluation result (implementation specific).

#### evolve

```ts
evolve(): Promise<default>
```

Advance the evolutionary loop by one generation.

Conceptually, `evolve()` is the reproduction half of NEAT. It selects parents,
preserves elites and provenance when configured, applies structural and parametric
mutation, updates search bookkeeping, and returns the best genome observed for the step.
The heavy mechanics live in `src/neat/evolve/evolve.ts`; this method stays as the
readable front door to that orchestration.

Returns: Best genome selected by the evolution step.

Example:

// Score the current population first, then breed the next generation.
await neat.evaluate();
await neat.evolve();

#### export

```ts
export(): GenomeJSON[]
```

Export the current population as plain JSON objects.

Choose this lighter snapshot when you only need the genomes themselves and
do not need generation counters, innovation maps, or other controller-level
state.

Returns: JSON-safe population snapshot.

#### exportParetoFrontJSONL

```ts
exportParetoFrontJSONL(
  maxEntries: number,
): string
```

Export recent Pareto archive entries as JSON Lines.

This is the easiest way to persist frontier history for offline analysis or
later replay in notebooks and visualization tools.

Parameters:
- `maxEntries` - Maximum number of recent archive entries to export.

Returns: JSONL payload for the requested Pareto archive window.

#### exportRNGState

```ts
exportRNGState(): number | undefined
```

Export the current RNG state for external persistence or tests.

Returns: Opaque RNG snapshot suitable for later replay.

#### exportSpeciesHistoryCSV

```ts
exportSpeciesHistoryCSV(
  maxEntries: number,
): string
```

Export recent species history as CSV.

This is useful when you want to chart species growth, collapse, or
stagnation in a spreadsheet or notebook without writing a custom parser.

Parameters:
- `maxEntries` - Maximum number of recent history entries to export.

Returns: CSV payload representing recent species history snapshots.

#### exportSpeciesHistoryJSONL

```ts
exportSpeciesHistoryJSONL(
  maxEntries: number,
): string
```

Export recent species history as JSON Lines.

Choose this when you want machine-friendly archival output instead of the
flatter spreadsheet-oriented CSV export.

Parameters:
- `maxEntries` - Maximum number of recent history entries to export.

Returns: JSONL payload describing recent species-history entries.

#### exportState

```ts
exportState(): NeatStateJSON
```

Export the full controller state, including metadata and population.

This is the pause-and-resume snapshot. It is the best choice when you want
to continue the same run later with the same innovation history, generation
counter, and serialized genomes.

Returns: Full controller snapshot including metadata and population.

#### exportTelemetryCSV

```ts
exportTelemetryCSV(
  maxEntries: number,
): string
```

Export recent telemetry entries as CSV.

Parameters:
- `maxEntries` - Maximum number of recent telemetry entries to export.

Returns: CSV string for quick spreadsheet or notebook analysis.

#### exportTelemetryJSONL

```ts
exportTelemetryJSONL(): string
```

Export the telemetry buffer as JSON Lines.

JSONL is the easiest format to append to files, stream into data tools, or
inspect generation-by-generation without loading a giant array into memory.

Returns: JSONL payload with one telemetry entry per line.

#### fromJSON

```ts
fromJSON(
  json: NeatMetaJSON,
  fitness: NeatFitnessFunction,
): default
```

Rebuild a `Neat` controller from serialized metadata without importing a population bundle.

This is the lighter-weight sibling of `importState()`. It is useful when you
want controller defaults, innovation bookkeeping, or archive metadata back,
but you are handling genome population state separately.

Parameters:
- `json` - Serialized controller metadata produced by `toJSON()`.
- `fitness` - Fitness function to attach to the reconstructed controller.

Returns: Reconstructed `Neat` controller instance.

#### getAverage

```ts
getAverage(): number
```

Calculates the average fitness score of the population.

#### getDiversityStats

```ts
getDiversityStats(): DiversityStats
```

Return the latest cached diversity statistics.

Diversity summaries answer a different question than raw fitness: whether
the population still explores varied structures or is collapsing toward a
narrower family of genomes.

Returns: Diversity metrics for the current population snapshot.

#### getFittest

```ts
getFittest(): default
```

Retrieves the fittest genome from the population.

#### getLineageSnapshot

```ts
getLineageSnapshot(
  limit: number,
): { id: number; parents: number[]; }[]
```

Return an array of {id, parents} for the first `limit` genomes in population.

Parameters:
- `limit` - Maximum number of lineage records to return.

Returns: Compact lineage snapshot for debugging and teaching inheritance flow.

#### getMinimumHiddenSize

```ts
getMinimumHiddenSize(
  multiplierOverride: number | undefined,
): number
```

Minimum hidden size considering explicit minHidden or multiplier policy.

#### getMultiObjectiveMetrics

```ts
getMultiObjectiveMetrics(): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Return compact multi-objective metrics for each genome in the current population.

Use this when you want a flattened view of Pareto rank, crowding, raw score,
and structural size without reconstructing the full fronts yourself.

Returns: One compact metric record per genome.

#### getNoveltyArchiveSize

```ts
getNoveltyArchiveSize(): number
```

Return the current novelty-archive size.

This is a small diagnostic hook that tells you whether novelty search is
actively accumulating behavior representatives or staying mostly unused.

Returns: Number of archived novelty descriptors.

#### getObjectiveEvents

```ts
getObjectiveEvents(): { gen: number; type: "add" | "remove"; key: string; }[]
```

Get recent objective add/remove events for telemetry exports and teaching.

#### getObjectiveKeys

```ts
getObjectiveKeys(): string[]
```

Public helper returning just the objective keys (tests rely on).

#### getObjectives

```ts
getObjectives(): { key: string; direction: "max" | "min"; }[]
```

Return a lightweight list of registered objective keys and their directions.

Returns: Objective descriptors currently active on the controller.

#### getOffspring

```ts
getOffspring(): default
```

Build a child genome from parent selection and crossover.

Use this when you want one reproduction event without running a full
generation step. The method delegates the parent choice to `getParent()` so
it still respects the controller's current breeding policy.

Returns: New network created from selected parent genomes.

#### getOperatorStats

```ts
getOperatorStats(): { name: string; success: number; attempts: number; }[]
```

Return mutation-operator success statistics.

These numbers are useful when operator adaptation is enabled and you want
to inspect which mutation operators are being rewarded or ignored.

Returns: Per-operator attempt and success counters.

#### getParent

```ts
getParent(): default
```

Select a parent genome using the controller's configured selection strategy.

Read this as the "who gets to reproduce" hook. The exact policy depends on
the current selection configuration, but the intent is always the same:
convert the scored population into a plausible breeding candidate.

Returns: The selected parent genome.

#### getParetoArchive

```ts
getParetoArchive(
  maxEntries: number,
): ParetoArchiveEntry[]
```

Return recent Pareto archive entries.

This is the metadata-oriented archive view. Use it when you want to inspect
what front snapshots were retained over time without exporting the full JSONL
payload first.

Parameters:
- `maxEntries` - Maximum number of recent archive entries to return.

Returns: Recent Pareto archive metadata entries.

#### getParetoFronts

```ts
getParetoFronts(
  maxFronts: number,
): default[][]
```

Reconstruct Pareto fronts for the current population snapshot.

Parameters:
- `maxFronts` - Maximum number of fronts to materialize.

Returns: Fronts ordered from most to least dominant under the active objectives.

#### getPerformanceStats

```ts
getPerformanceStats(): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Return timing statistics for the latest evaluation and evolution steps.

This is a lightweight performance probe for experiments and benchmarks that
need to notice when scoring or breeding costs start drifting upward.

Returns: Recent runtime statistics for evaluation and evolution work.

#### getSpeciesHistory

```ts
getSpeciesHistory(): SpeciesHistoryEntry[]
```

Return the recorded species-history timeline.

Unlike `getSpeciesStats()`, which only reflects the current generation,
this method exposes the historical view used for trend analysis.

Returns: Species history entries in recorded order.

#### getSpeciesStats

```ts
getSpeciesStats(): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Return a compact per-species summary for the current population snapshot.

This is the quickest inspection surface when you want to know how many
niches currently exist, how large they are, and whether they have improved
recently.

Returns: One summary record per active species.

#### getTelemetry

```ts
getTelemetry(): TelemetryEntry[]
```

Return the internal telemetry buffer.

Telemetry is the controller's teaching surface for understanding why a run is
behaving a certain way. Instead of watching only the best score, you can inspect
species counts, diversity, objective events, evaluation timing, and other search signals.

Returns: Recorded telemetry entries in chronological order.

#### import

```ts
import(
  json: GenomeJSON[],
): Promise<void>
```

Replace the current population with serialized genomes.

This is the population-only restore path. It keeps the current controller
instance, options, and metadata while swapping in a different genome set.
Use `importState()` when you want to restore controller metadata too.

Parameters:
- `json` - Serialized population to import into the current controller.

Returns: Promise resolving after the population is loaded.

#### importRNGState

```ts
importRNGState(
  state: string | number | undefined,
): void
```

Import an RNG state (alias for restore; kept for compatibility).

Parameters:
- `state` - Numeric RNG state.

Returns: Nothing. This is a compatibility alias for `restoreRNGState()`.

#### importState

```ts
importState(
  bundle: NeatStateJSON,
  fitness: NeatFitnessFunction,
): Promise<default>
```

Restore a full evolutionary snapshot produced by `exportState()`.

Use this when you want a paused experiment to resume with its controller
metadata, population, and archival context intact rather than rebuilding
only the bare genomes.

Parameters:
- `bundle` - Serialized object with the shape `{ neat, population }`.
- `fitness` - Fitness function to attach to the restored controller.

Returns: A `Neat` instance ready to continue evolution from the imported state.

#### mutate

```ts
mutate(): Promise<void>
```

Apply mutation pressure to the current population without advancing generation bookkeeping.

This is the direct "variation" lever. Use it when you want to perturb the
current genomes in place for an experiment, a custom training loop, or a
test harness that separates mutation from the rest of `evolve()`.
In the normal NEAT workflow, `evolve()` is usually the better entry point
because it coordinates parent selection, elitism, offspring creation, and
mutation as one generation step.

Returns: Promise resolving once mutation has been applied to the current population.

#### registerObjective

```ts
registerObjective(
  key: string,
  direction: "max" | "min",
  accessor: (network: default) => number,
): void
```

Register a custom objective for multi-objective optimization.

Register objectives when a single scalar score is too narrow to express the
behavior you want. The controller can then reason about tradeoffs such as raw
score versus simplicity, novelty, or domain-specific constraints.

Parameters:
- `key` - Stable objective identifier used in exports and telemetry.
- `direction` - Whether the objective should be minimized or maximized.
- `accessor` - Function extracting the objective value from a genome.

#### resetNoveltyArchive

```ts
resetNoveltyArchive(): void
```

Reset the novelty archive.

This is useful when you want to restart novelty pressure from a clean slate
without rebuilding the whole controller.

#### restoreRNGState

```ts
restoreRNGState(
  state: string | number | undefined,
): void
```

Restore a previously-snapshotted RNG state. This restores the internal
seed but does not re-create the RNG function until next use.

Parameters:
- `state` - Opaque numeric RNG state produced by `snapshotRNGState()`.

Returns: Nothing. The controller will resume from the restored RNG state on next use.

#### sampleRandom

```ts
sampleRandom(
  sampleCount: number,
): number[]
```

Produce deterministic random samples using the instance RNG.

Parameters:
- `sampleCount` - Number of random values to generate.

Returns: Array of deterministic random samples.

#### selectMutationMethod

```ts
selectMutationMethod(
  genome: default,
  rawReturnForTest: boolean,
): Promise<MutationMethod | MutationMethod[] | null>
```

Selects a mutation method for a given genome based on constraints.

Parameters:
- `genome` - Genome being considered for mutation.
- `rawReturnForTest` - Whether to expose raw selection output for test visibility.

Returns: Selected mutation method or `null` when no valid method can be chosen.

#### snapshotRNGState

```ts
snapshotRNGState(): number | undefined
```

Return the current opaque RNG numeric state used by the instance.
Useful for deterministic test replay and debugging.

Returns: Snapshot of the current controller RNG state.

#### sort

```ts
sort(): void
```

Sorts the population in descending order of fitness scores.

#### spawnFromParent

```ts
spawnFromParent(
  parent: default,
  mutateCount: number,
): default
```

Spawn a new genome derived from a single parent while preserving Neat bookkeeping.

Parameters:
- `parent` - Parent genome to clone and mutate.
- `mutateCount` - Number of mutation passes to apply to the child.

Returns: Child genome registered with the same bookkeeping conventions as normal evolution.

#### toJSON

```ts
toJSON(): NeatMetaJSON
```

Serialize controller metadata without the concrete population.

This is useful when you want to preserve run configuration and innovation
bookkeeping separately from genome payloads, or when the population will be
reconstructed by other means.

Returns: JSON-safe metadata snapshot useful for innovation-history persistence.

### default

#### _activateCore

```ts
_activateCore(
  withTrace: boolean,
  input: number | undefined,
): number
```

Internal shared implementation for activate/noTraceActivate.

Parameters:
- `withTrace` - Whether to update eligibility traces.
- `input` - Optional externally supplied activation (bypasses weighted sum if provided).

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### _globalNodeIndex

Global index counter for assigning unique indices to nodes.

#### _safeUpdateWeight

```ts
_safeUpdateWeight(
  connection: default,
  delta: number,
): void
```

Internal helper to safely update a connection weight with clipping and NaN checks.

#### acquire

```ts
acquire(
  from: default,
  to: default,
  weight: number | undefined,
): default
```

Acquire a connection from the internal pool, or construct a fresh one when the pool is empty.
This is the low-allocation path used by topology mutation and other edge-churn heavy flows.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### activate

```ts
activate(
  input: number[],
  training: boolean,
  _maxActivationDepth: number,
): number[]
```

Standard activation API returning a plain number[] for backward compatibility.
Internally may use pooled typed arrays; if so they are cloned before returning.

#### activate

```ts
activate(
  input: number | undefined,
): number
```

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

```ts
activate(
  value: number[] | undefined,
  training: boolean,
): number[]
```

Activates all nodes within the layer, computing their output values.

If an input `value` array is provided, it's used as the initial activation
for the corresponding nodes in the layer. Otherwise, nodes compute their
activation based on their incoming connections.

During training, layer-level dropout is applied, masking all nodes in the layer together.
During inference, all masks are set to 1.

Parameters:
- `value` - An optional array of activation values to set for the layer's nodes. The length must match the number of nodes.
- `training` - A boolean indicating whether the layer is in training mode. Defaults to false.

Returns: An array containing the activation value of each node in the layer after activation.

#### activate

```ts
activate(
  value: number[] | undefined,
): number[]
```

Activates all nodes in the group.

Parameters:
- `value` - Optional array of input values. Its length must match the number of nodes in the group.

Returns: Activation value of each node in the group, in order.

#### activateBatch

```ts
activateBatch(
  inputs: number[][],
  training: boolean,
): number[][]
```

Activate the network over a batch of input vectors (micro-batching).

Currently iterates sample-by-sample while reusing the network's internal
fast-path allocations. Outputs are cloned number[] arrays for API
compatibility. Future optimizations can vectorize this path.

Parameters:
- `inputs` - Array of input vectors, each length must equal this.input
- `training` - Whether to run with training-time stochastic features

Returns: Array of output vectors, each length equals this.output

#### activateRaw

```ts
activateRaw(
  input: number[],
  training: boolean,
  maxActivationDepth: number,
): ActivationArray
```

Raw activation that can return a typed array when pooling is enabled (zero-copy).
If reuseActivationArrays=false falls back to standard activate().

Parameters:
- `input` - Input vector.
- `training` - Whether to enable training-time stochastic paths.
- `maxActivationDepth` - Maximum graph depth for activation.

Returns: Output activations (typed array when pooling is enabled).

#### activation

The output value of the node after applying the activation function. This is the value transmitted to connected nodes.

#### addNodeBetween

```ts
addNodeBetween(): void
```

Split a random existing connection by inserting one hidden node.

#### adjustRateForAccumulation

```ts
adjustRateForAccumulation(
  rate: number,
  accumulationSteps: number,
  reduction: "average" | "sum",
): number
```

Utility: adjust rate for accumulation mode (use result when switching to 'sum' to mimic 'average').

#### applyBatchUpdates

```ts
applyBatchUpdates(
  momentum: number,
): void
```

Applies accumulated batch updates to incoming and self connections and this node's bias.
Uses momentum in a Nesterov-compatible way: currentDelta = accumulated + momentum * previousDelta.
Resets accumulators after applying. Safe to call on every node type.

Parameters:
- `momentum` - Momentum factor (0 to disable)

#### applyBatchUpdatesWithOptimizer

```ts
applyBatchUpdatesWithOptimizer(
  opts: { type: "sgd" | "rmsprop" | "adagrad" | "adam" | "adamw" | "amsgrad" | "adamax" | "nadam" | "radam" | "lion" | "adabelief" | "lookahead"; momentum?: number | undefined; beta1?: number | undefined; beta2?: number | undefined; eps?: number | undefined; weightDecay?: number | undefined; lrScale?: number | undefined; t?: number | undefined; baseType?: string | undefined; la_k?: number | undefined; la_alpha?: number | undefined; },
): void
```

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

#### attention

```ts
attention(
  size: number,
  heads: number,
): default
```

Creates a multi-head self-attention layer (stub implementation).

Parameters:
- `size` - Number of output nodes.
- `heads` - Number of attention heads (default 1).

Returns: A new Layer instance representing an attention layer.

#### batchNorm

```ts
batchNorm(
  size: number,
): default
```

Creates a batch normalization layer.
Applies batch normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - The number of nodes in this layer.

Returns: A new Layer instance configured as a batch normalization layer.

#### bias

The bias value of the node. Added to the weighted sum of inputs before activation.
Input nodes typically have a bias of 0.

#### clear

```ts
clear(): void
```

Clears the internal state of all nodes in the network.
Resets node activation, state, eligibility traces, and extended traces to their initial values (usually 0).
This is typically done before processing a new input sequence in recurrent networks or between training epochs if desired.

#### clearStochasticDepthSchedule

```ts
clearStochasticDepthSchedule(): void
```

Clear stochastic-depth schedule function.

#### clearWeightNoiseSchedule

```ts
clearWeightNoiseSchedule(): void
```

Clear the dynamic global weight-noise schedule.

#### clone

```ts
clone(): default
```

Creates a deep copy of the network.

Returns: A new Network instance that is a clone of the current network.

#### configurePruning

```ts
configurePruning(
  cfg: { start: number; end: number; targetSparsity: number; regrowFraction?: number | undefined; frequency?: number | undefined; method?: "magnitude" | "snip" | undefined; },
): void
```

Configure scheduled pruning during training.

Parameters:
- `cfg` - Pruning schedule and strategy configuration.

#### connect

```ts
connect(
  from: default,
  to: default,
  weight: number | undefined,
): default[]
```

Creates a connection between two nodes in the network.
Handles both regular connections and self-connections.
Adds the new connection object(s) to the appropriate network list (`connections` or `selfconns`).

Returns: An array containing the newly created connection object(s). Typically contains one connection, but might be empty or contain more in specialized node types.

#### connect

```ts
connect(
  target: default | { nodes: default[]; },
  weight: number | undefined,
): default[]
```

Creates a connection from this node to a target node or all nodes in a group.

Parameters:
- `target` - The target Node or a group object containing a `nodes` array.
- `weight` - The weight for the new connection(s). If undefined, a default or random weight might be assigned by the Connection constructor (currently defaults to 0, consider changing).

Returns: An array containing the newly created Connection object(s).

#### connect

```ts
connect(
  target: default | default | LayerLike,
  method: unknown,
  weight: number | undefined,
): default[]
```

Connects this layer's output to a target component (Layer, Group, or Node).

This method delegates the connection logic primarily to the layer's `output` group
or the target layer's `input` method. It establishes the forward connections
necessary for signal propagation.

Parameters:
- `target` - The destination Layer, Group, or Node to connect to.
- `method` - The connection method (e.g., `ALL_TO_ALL`, `ONE_TO_ONE`) defining the connection pattern. See `methods.groupConnection`.
- `weight` - An optional fixed weight to assign to all created connections.

Returns: An array containing the newly created connection objects.

#### connect

```ts
connect(
  target: default | default | default,
  method: unknown,
  weight: number | undefined,
): default[]
```

Establishes connections from all nodes in this group to a target group, layer, or node.

Parameters:
- `target` - Destination entity to connect to.
- `method` - Connection pattern to use.
- `weight` - Optional fixed weight for all created connections.

Returns: All connection objects created during this wiring step.

#### connections

Connection list.

#### construct

```ts
construct(
  parts: readonly ConstructPart[],
  options: ConstructOptions | undefined,
): ConstructResult
```

Construct a runnable network from mixed `Node`, `Group`, and `Layer` parts.

This builder compiles the provided parts into the ordinary `Network`
runtime, preserving explicit input/output ordering and then rebuilding the
scheduling cache in either acyclic or recurrent mode.

Parameters:
- `parts` - Mixed architecture parts to flatten.
- `options` - Optional construct-time validation, ordering, and runtime flags.

Returns: Materialized runtime plus lightweight diagnostics.

Example:

```ts
const sensor = new Node('input');
const hidden = new Group(2);
const readout = Layer.dense(1, 'output');

sensor.connect(hidden);
hidden.connect(readout);

const { network } = Network.construct([sensor, hidden, readout]);
```

#### construct

```ts
construct(
  list: (default | default | default)[],
): default
```

Constructs a network instance from an array of interconnected layers,
groups, or nodes.

This method is the bridge between manual graph assembly and a runnable
`Network`. It walks the supplied primitives, collects the unique nodes and
connections they reference, infers input/output counts from node types, and
folds the result into one normalized network object.

Parameters:
- `list` - Building blocks that are already interconnected.

Returns: A network representing the supplied architecture.

#### conv1d

```ts
conv1d(
  size: number,
  kernelSize: number,
  stride: number,
  padding: number,
): default
```

Creates a 1D convolutional layer (stub implementation).

Parameters:
- `size` - Number of output nodes (filters).
- `kernelSize` - Size of the convolution kernel.
- `stride` - Stride of the convolution (default 1).
- `padding` - Padding (default 0).

Returns: A new Layer instance representing a 1D convolutional layer.

#### createMLP

```ts
createMLP(
  inputCount: number,
  hiddenCounts: number[],
  outputCount: number,
): default
```

Creates a fully connected, strictly layered MLP network.

Returns: A new, fully connected, layered MLP

#### crossOver

```ts
crossOver(
  network1: default,
  network2: default,
  equal: boolean,
): default
```

NEAT-style crossover delegate.

#### dcMask

DropConnect active mask: `1` means active for this stochastic pass, `0` means dropped.

#### dense

```ts
dense(
  size: number,
  nodeType: PrimitiveNodeType,
): default
```

Creates a standard fully connected (dense) layer.

All nodes in the source layer/group will connect to all nodes in this layer
when using the default `ALL_TO_ALL` connection method via `layer.input()`.
Dense layers also stamp default descriptor metadata (`family: 'dense'`) so
later tooling can recognize the block even when the caller never names it.

Parameters:
- `size` - The number of nodes (neurons) in this layer.
- `nodeType` - Optional primitive role assigned to the dense block.

Returns: A new Layer instance configured as a dense layer.

Example:

```ts
const output = Layer.dense(2, 'output');

output.describe({ label: 'policyHead' });
```

#### derivative

The derivative of the activation function evaluated at the node's current state. Used in backpropagation.

#### describe

```ts
describe(
  descriptor: PrimitiveDescriptor,
): void
```

Attaches optional descriptor metadata to the primitive boundary.

This descriptor is advisory only. It does not change runtime activation,
mutation, or serialization behavior, but it gives later architecture
assembly, diagnostics, and visualization passes a stable place to read
human-facing labels and intent.

Reach for this when the node is still the right abstraction but a later
reader should not have to infer its purpose from connection order alone.

Parameters:
- `descriptor` - Optional label, intent, and scalar metadata to merge.

Returns: Nothing.

Examples:

```ts
const readout = new Node('output');
readout.describe({
  label: 'readoutNode',
  metadata: { stage: 'readout' },
});
```

```ts
const readout = Layer.dense(2, 'output');

readout.describe({
  label: 'readoutHead',
  metadata: { stage: 'policy' },
});
```

```ts
const forgetGate = new Group(8);

forgetGate.describe({
  label: 'forgetGate',
  intent: 'gate',
  metadata: { family: 'lstm' },
});
```

#### describeArchitecture

```ts
describeArchitecture(): NetworkArchitectureDescriptor
```

Resolves a stable architecture descriptor for telemetry/UI consumers.

Prefers live graph analysis and only falls back to hydrated serialization
metadata when graph-based resolution is purely inferred.

Returns: Architecture descriptor with hidden-layer widths and provenance.

#### describeTemporalStructure

```ts
describeTemporalStructure(): NetworkTemporalStructureDescriptor
```

Resolves the validated temporal-module structure for diagnostics and visualization.

Call this when the coarse hidden-layer descriptor is not enough and you
need the explicit recurrent-module and gated-block ownership story that the
runtime builders preserve for LSTM, GRU, and NARX networks.

Returns: Temporal-structure descriptor synchronized against the live graph.

#### deserialize

```ts
deserialize(
  data: unknown[] | CompactSerializedNetworkTuple,
  inputSize: number | undefined,
  outputSize: number | undefined,
): default
```

Static lightweight tuple deserializer delegate

#### disableDropConnect

```ts
disableDropConnect(): void
```

Disable DropConnect.

#### disableStochasticDepth

```ts
disableStochasticDepth(): void
```

Disable stochastic depth.

#### disableWeightNoise

```ts
disableWeightNoise(): void
```

Disable all weight-noise settings.

#### disconnect

```ts
disconnect(
  from: default,
  to: default,
): void
```

Disconnects two nodes, removing the connection between them.
Handles both regular connections and self-connections.
If the connection being removed was gated, it is also ungated.

#### disconnect

```ts
disconnect(
  target: default,
  twosided: boolean,
): void
```

Removes the connection from this node to the target node.

Parameters:
- `target` - The target node to disconnect from.
- `twosided` - If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.

#### disconnect

```ts
disconnect(
  target: default | default,
  twosided: boolean | undefined,
): void
```

Removes connections between this layer's nodes and a target Group or Node.

Parameters:
- `target` - The Group or Node to disconnect from.
- `twosided` - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.

#### disconnect

```ts
disconnect(
  target: default | default,
  twosided: boolean,
): void
```

Removes connections between nodes in this group and a target group or node.

Parameters:
- `target` - Group or node to disconnect from.
- `twosided` - Whether to also remove reciprocal connections.

Returns: Nothing.

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### dropout

Dropout probability.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene is currently expressed and participates in the forward pass.

#### enableDropConnect

```ts
enableDropConnect(
  p: number,
): void
```

Enable DropConnect with a probability in $[0,1)$.

Parameters:
- `p` - DropConnect probability.

#### enableWeightNoise

```ts
enableWeightNoise(
  stdDev: number | { perHiddenLayer: number[]; },
): void
```

Enable weight noise using either a global standard deviation or per-hidden-layer values.

Parameters:
- `stdDev` - Global standard deviation or hidden-layer schedule.

#### enforceMinimumHiddenLayerSizes

```ts
enforceMinimumHiddenLayerSizes(
  network: default,
): default
```

Enforces the minimum hidden layer size rule on a network.

Parameters:
- `network` - The network to normalize.

Returns: The same network with hidden layers grown to the minimum size when needed.

#### error

Stores error values calculated during backpropagation.

#### evolve

```ts
evolve(
  set: { input: number[]; output: number[]; }[],
  options: Record<string, unknown> | undefined,
): Promise<{ error: number; iterations: number; time: number; }>
```

Evolve the network against a dataset using the neuroevolution chapter.

The implementation lives outside this class so the public surface stays
orchestration-first while population search, mutation policy, and stopping
criteria remain chapter-owned.

Parameters:
- `set` - Evaluation samples with `input` and `output` vectors.
- `options` - Evolution options controlling population search and stopping criteria.

Returns: Promise resolving to the final error, iteration count, and elapsed time.

#### fastSlabActivate

```ts
fastSlabActivate(
  input: number[],
): number[]
```

Public wrapper for fast slab forward pass.

Parameters:
- `input` - Input vector.

Returns: Activation output.

#### firstMoment

First moment estimate used by Adam-family optimizers.

#### from

The source (pre-synaptic) node supplying activation.

#### fromJSON

```ts
fromJSON(
  json: Record<string, unknown>,
): default
```

Verbose JSON static deserializer

#### fromJSON

```ts
fromJSON(
  json: { bias: number; response?: number | undefined; type: string; squash: string; mask: number; },
): default
```

Creates a Node instance from a JSON object.

Parameters:
- `json` - The JSON object containing node configuration.

Returns: A new Node instance configured according to the JSON object.

#### gain

Multiplicative modulation applied after weight. Neutral gain `1` is omitted from storage.

#### gate

```ts
gate(
  node: default,
  connection: default,
): void
```

Gates a connection with a specified node.
The activation of the `node` (gater) will modulate the weight of the `connection`.
Adds the connection to the network's `gates` list.

#### gate

```ts
gate(
  connections: default | default[],
): void
```

Makes this node gate the provided connection(s).
The connection's gain will be controlled by this node's activation value.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to be gated.

#### gate

```ts
gate(
  connections: default[],
  method: unknown,
): void
```

Applies gating to a set of connections originating from this layer's output group.

Gating allows the activity of nodes in this layer (specifically, the output group)
to modulate the flow of information through the specified `connections`.

Parameters:
- `connections` - An array of connection objects to be gated.
- `method` - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.

#### gate

```ts
gate(
  connections: default | default[],
  method: unknown,
): void
```

Configures nodes within this group to act as gates for the specified connection set.

Parameters:
- `connections` - Single connection or list of connections to gate.
- `method` - Gating mechanism to use.

Returns: Nothing.

#### gater

Optional gating node whose activation modulates effective weight.

#### gates

Network gates collection.

#### geneId

Stable per-node gene identifier for NEAT innovation reuse

#### getActivationSchedulingDiagnostics

```ts
getActivationSchedulingDiagnostics(): ActivationSchedulingDiagnostics
```

Read a human-friendly snapshot of the current activation-ordering contract.

Use this after activation or structural edits to see whether the runtime is
using a compiled schedule, a cycle fallback, or a raw-node-order fallback,
and what to do next if that result is not the one you expected.

Returns: Activation scheduling diagnostics snapshot.

#### getConnectionSlab

```ts
getConnectionSlab(): ConnectionSlabView
```

Read slab structures for fast activation.

Returns: Slab connection structures.

#### getCurrentSparsity

```ts
getCurrentSparsity(): number
```

Compute the current connection sparsity ratio.

Returns: Current sparsity in $[0,1]$.

#### getLastGradClipGroupCount

```ts
getLastGradClipGroupCount(): number
```

Returns last gradient clipping group count (0 if no clipping yet).

#### getLossScale

```ts
getLossScale(): number
```

Returns current mixed precision loss scale (1 if disabled).

#### getRandomFn

```ts
getRandomFn(): (() => number) | undefined
```

Read the active deterministic RNG function.

Returns: RNG function when deterministic state is initialized.

#### getRawGradientNorm

```ts
getRawGradientNorm(): number
```

Returns last recorded raw (pre-update) gradient L2 norm.

#### getRegularizationStats

```ts
getRegularizationStats(): Record<string, unknown> | null
```

Read regularization statistics collected during training.

Returns: Regularization stats payload.

#### getRNGState

```ts
getRNGState(): number | undefined
```

Read the raw deterministic RNG state word.

Returns: RNG state value when present.

#### getTopologyIntent

```ts
getTopologyIntent(): NetworkTopologyIntent
```

Returns the public topology intent for this network.

Returns: Current topology intent.

#### getTrainingStats

```ts
getTrainingStats(): TrainingStatsSnapshot
```

Consolidated training stats snapshot.

#### gradientAccumulator

Generic gradient accumulator used by RMSProp and AdaGrad.

#### gru

```ts
gru(
  size: number,
): default
```

Creates a Gated Recurrent Unit (GRU) layer.

GRUs are another type of recurrent neural network cell, often considered
simpler than LSTMs but achieving similar performance on many tasks.
They use an update gate and a reset gate to manage information flow.

Parameters:
- `size` - The number of GRU units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as a GRU layer.

#### gru

```ts
gru(
  layerArgs: (number | ArchitectRecurrentShortcutOptions)[],
): default
```

Creates a Gated Recurrent Unit network.

This builder keeps the GRU graph explicit: update, inverse-update, reset,
memory, output, and previous-output groups are all wired from the public
primitive surface rather than hidden behind a fused recurrent runtime.

The optional `inputToOutput` shortcut is disabled by default so existing
GRU topologies keep their historical shape. Enable it when you want a
direct input-to-readout path in addition to the recurrent block stack.

Recommended starting knobs:

- start with one small recurrent block before stacking multiple GRU stages,
- supervised fine-tuning: keep `rate` low, usually around `0.001` to
  `0.01`, and preserve sequence order,
- evolution: compare runs with a fixed `seed`, moderate `mutationRate`,
  and only enable `inputToOutput` when the task clearly benefits from a
  direct readout shortcut.

Common pitfalls:

- forgetting to call `clear()` between independent sequences or episodes,
- enabling the shortcut before checking whether the recurrent block alone
  already solves the task,
- shuffling timesteps rather than whole sequences.

Parameters:
- `layerArgs` - Layer sizes plus an optional trailing options object.

Returns: The constructed GRU network.

Example:

```ts
const network = Architect.gru(1, 3, 1, { inputToOutput: true });

const outputs = [0.1, 0.4, 0.2].map(
  (value) => network.activate([value])[0],
);

network.clear();

console.log(outputs);
```

#### hasGater

Whether a gater node is assigned to modulate this connection's effective weight.

#### hopfield

```ts
hopfield(
  size: number,
): default
```

Creates a Hopfield network.

Parameters:
- `size` - The number of nodes in the network.

Returns: The constructed Hopfield network.

#### index

Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.

#### infinityNorm

Adamax infinity norm accumulator.

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

```ts
innovationID(
  sourceNodeId: number,
  targetNodeId: number,
): number
```

Deterministic Cantor pairing function for a `(sourceNodeId, targetNodeId)` pair.
Use it when you need a stable edge identifier without relying on the mutable
auto-increment counter.

Parameters:
- `sourceNodeId` - Source node integer id or index.
- `targetNodeId` - Target node integer id or index.

Returns: Unique non-negative integer derived from the ordered pair.

Example:

```ts
const id = Connection.innovationID(2, 5);
```

#### input

Input node count.

#### input

```ts
input(
  from: default | LayerLike,
  method: unknown,
  weight: number | undefined,
): default[]
```

Handles the connection logic when this layer is the *target* of a connection.

It connects the output of the `from` layer or group to this layer's primary
input mechanism (which is often the `output` group itself, but depends on the layer type).
This method is usually called by the `connect` method of the source layer/group.

Parameters:
- `from` - The source Layer or Group connecting *to* this layer.
- `method` - The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
- `weight` - An optional fixed weight for the connections.

Returns: An array containing the newly created connection objects.

#### inputNodeIds

Ordered stable gene ids that define the network input vector contract.

Returns a cloned array so callers can inspect role metadata without
mutating runtime state.

Returns: Ordered input node gene ids.

#### intent

Optional semantic intent for architecture tooling and diagnostics.

#### isActivating

Internal flag to detect cycles during activation

#### isConnectedTo

```ts
isConnectedTo(
  target: default,
): boolean
```

Checks if this node is connected to another node.

Parameters:
- `target` - The target node to check the connection with.

Returns: True if connected, otherwise false.

#### isProjectedBy

```ts
isProjectedBy(
  node: default,
): boolean
```

Checks if the given node has a direct outgoing connection to this node.
Considers both regular incoming connections and the self-connection.

Parameters:
- `node` - The potential source node.

Returns: True if the given node projects to this node, false otherwise.

#### isProjectingTo

```ts
isProjectingTo(
  node: default,
): boolean
```

Checks if this node has a direct outgoing connection to the given node.
Considers both regular outgoing connections and the self-connection.

Parameters:
- `node` - The potential target node.

Returns: True if this node projects to the target node, false otherwise.

#### label

Optional human-readable descriptor label for architecture tooling.

#### lastSkippedLayers

Last skipped stochastic-depth layers from activation runtime state.

#### layerNorm

```ts
layerNorm(
  size: number,
): default
```

Creates a layer normalization layer.
Applies layer normalization to the activations of the nodes in this layer during activation.

Parameters:
- `size` - The number of nodes in this layer.

Returns: A new Layer instance configured as a layer normalization layer.

#### layers

Optional layered view cache.

#### lookaheadShadowWeight

Lookahead slow-weight snapshot.

#### lstm

```ts
lstm(
  size: number,
): default
```

Creates a Long Short-Term Memory (LSTM) layer.

LSTMs are a type of recurrent neural network (RNN) cell capable of learning
long-range dependencies. This implementation uses standard LSTM architecture
with input, forget, and output gates, and a memory cell.

Parameters:
- `size` - The number of LSTM units (and nodes in each gate/cell group).

Returns: A new Layer instance configured as an LSTM layer.

#### lstm

```ts
lstm(
  layerArgs: (number | ArchitectRecurrentShortcutOptions)[],
): default
```

Creates a Long Short-Term Memory network.

This builder keeps the LSTM graph explicit: each recurrent block is
assembled from gate groups, a memory-cell group, and an output block using
the same primitive wiring surface used elsewhere in the architecture layer.

The optional `inputToOutput` shortcut preserves the historical builder
behavior by default. Disable it when you want the public preset to route
information strictly through the recurrent block stack.

Recommended starting knobs:

- start with one small block such as `Architect.lstm(input, 4, output)`
  before stacking multiple recurrent stages,
- supervised fine-tuning: keep `rate` low, usually around `0.001` to
  `0.01`, and feed one ordered sequence at a time,
- evolution: keep `seed` fixed for comparisons and prefer `NARX` first
  when a short explicit window already explains the task.

Common pitfalls:

- forgetting that `inputToOutput` defaults to `true`,
- flattening unrelated episodes into one activation stream without calling
  `clear()`,
- shuffling timesteps inside a sequence and destroying the temporal story
  the recurrent block is supposed to learn.

Parameters:
- `layerArgs` - Layer sizes plus an optional trailing options object.

Returns: The constructed LSTM network.

Example:

```ts
const network = Architect.lstm(1, 4, 1, { inputToOutput: false });

const firstPass = [0.1, 0.4, 0.2].map(
  (value) => network.activate([value])[0],
);

network.clear();

const secondPass = [0.1, 0.4, 0.2].map(
  (value) => network.activate([value])[0],
);

console.log(firstPass, secondPass);
```

#### mask

A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.

#### maxSecondMoment

AMSGrad maximum of past second-moment estimates.

#### memory

```ts
memory(
  size: number,
  memory: number,
): default
```

Creates a Memory layer, designed to hold state over a fixed number of time steps.

This layer consists of multiple groups (memory blocks), each holding the state
from a previous time step. The input connects to the most recent block, and
information propagates backward through the blocks. The layer's output
concatenates the states of all memory blocks.

Parameters:
- `size` - The number of nodes in each memory block (must match the input size).
- `memory` - The number of time steps to remember (number of memory blocks).

Returns: A new Layer instance configured as a Memory layer.

#### metadata

Optional scalar metadata retained on the primitive boundary.

#### mutate

```ts
mutate(
  method: MutationMethod,
): void
```

Mutates the network's structure or parameters according to the specified method.
This is a core operation for neuro-evolutionary algorithms (like NEAT).
The method argument should be one of the mutation types defined in `methods.mutation`.

Parameters:
- `method` - The mutation method to apply (e.g., `mutation.ADD_NODE`, `mutation.MOD_WEIGHT`).
 Some methods might have associated parameters (e.g., `MOD_WEIGHT` uses `min`, `max`).

#### mutate

```ts
mutate(
  method: unknown,
): void
```

Applies a mutation method to the node. Used in neuro-evolution.

This allows modifying the node's properties, such as its activation function or bias,
based on predefined mutation methods.

Parameters:
- `method` - A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).

#### narx

```ts
narx(
  inputSize: number,
  hiddenLayers: number | number[],
  outputSize: number,
  previousInput: number,
  previousOutput: number,
): default
```

Creates a Nonlinear AutoRegressive network with eXogenous inputs.

This is the smallest stateful preset in the public builder surface. The
main processing path receives the current exogenous input plus two explicit
delay lines: one for recent inputs and one for recent outputs. That keeps
the temporal story readable for debugging, visualization, and evolution
work without immediately jumping to gated cells.

Both delay lines are built from `Layer.memory(...)` blocks. Those memory
blocks use identity activation, zero bias, and one-to-one unit carry links,
so remembered values behave like a deterministic rolling window rather than
a learned recurrent cell.

Recommended starting knobs:

- start with short shelves such as `previousInput` and `previousOutput`
  in the `1` to `3` range,
- prefer `narx()` before gated builders when the task is really "predict
  from a small rolling window",
- keep sequence order intact during fine-tuning or evaluation and reset the
  carried state between independent runs.

Common pitfalls:

- forgetting that the delay lines intentionally carry state until
  `clear()` is called,
- shuffling timesteps from different sequences together,
- using long memory shelves when a shorter explicit window would be easier
  to debug, visualize, and evolve.

Clear-state guidance: call `network.clear()` before starting a new
independent sequence, episode, or evaluation run. If you keep activating
the same runtime without clearing it, the delay lines intentionally carry
their terminal state into the next activation stream.

Parameters:
- `inputSize` - The exogenous input size at each time step.
- `hiddenLayers` - Hidden layer sizes, or zero / empty for none.
- `outputSize` - The prediction output size.
- `previousInput` - The number of delayed input steps.
- `previousOutput` - The number of delayed output steps.

Returns: The constructed NARX network.

Example:

```ts
const network = Architect.narx(1, [4], 1, 2, 1);
const firstSequence = [[0.2], [0.7], [0.4]];
const secondSequence = [[1], [0], [0]];

const firstOutputs = firstSequence.map((input) => network.activate(input)[0]);

network.clear();

const secondOutputs = secondSequence.map((input) => network.activate(input)[0]);

console.log(firstOutputs, secondOutputs);
```

#### nodes

Network node collection.

#### noTraceActivate

```ts
noTraceActivate(
  input: number[],
): number[]
```

Activates the network without calculating eligibility traces.
This is a performance optimization for scenarios where backpropagation is not needed,
such as during testing, evaluation, or deployment (inference).

Returns: An array of numerical values representing the activations of the network's output nodes.

#### noTraceActivate

```ts
noTraceActivate(
  input: number | undefined,
): number
```

Activates the node without calculating eligibility traces (`xtrace`).
This is a performance optimization used during inference (when the network
is just making predictions, not learning) as trace calculations are only needed for training.

Parameters:
- `input` - Optional input value. If provided, sets the node's activation directly (used for input nodes).

Returns: The calculated activation value of the node.

#### old

The node's state from the previous activation cycle. Used for recurrent self-connections.

#### output

Output node count.

#### outputNodeIds

Ordered stable gene ids that define the network output vector contract.

Returns a cloned array so callers can inspect role metadata without
mutating runtime state.

Returns: Ordered output node gene ids.

#### perceptron

```ts
perceptron(
  layers: number[],
): default
```

Creates a standard multi-layer perceptron network.

The returned network is marked with the public `feed-forward` topology
intent so acyclic enforcement and slab fast-path eligibility stay aligned
with the builder users already chose.

Recommended starting knobs:

- gradient training: start with `iterations` in the low thousands, `rate`
  around `0.1` to `0.3`, and `momentum` around `0.9` for small normalized tasks,
- evolution: use this as the deterministic baseline profile with a fixed
  `seed`, `popsize` around `50` to `150`, and moderate `mutationRate`.

Common pitfalls:

- expecting a feed-forward graph to remember prior timesteps,
- keeping the exact same sample order every epoch on small i.i.d. datasets
  when some order randomization would reduce training bias.

Parameters:
- `layers` - Layer sizes starting with input, followed by hidden layers,
and ending with output.

Returns: The constructed MLP network.

Example:

```ts
const network = Architect.perceptron(2, 4, 1);

network.train(
  [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] },
  ],
  {
    iterations: 3_000,
    rate: 0.3,
    momentum: 0.9,
  },
);

const output = network.activate([1, 0])[0];
```

#### plastic

Whether this connection participates in plastic adaptation.

#### plasticityRate

Per-connection plasticity rate. `0` means the connection is not plastic.

#### previousDeltaBias

The change in bias applied in the previous training iteration. Used for calculating momentum.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  update: boolean,
  target: number[],
  regularization: number,
  costDerivative: ((target: number, output: number) => number) | undefined,
): void
```

Propagates the error backward through the network (backpropagation).
Calculates the error gradient for each node and connection.
If `update` is true, it adjusts the weights and biases based on the calculated gradients,
learning rate, momentum, and optional L2 regularization.

The process starts from the output nodes and moves backward layer by layer (or topologically for recurrent nets).

#### propagate

```ts
propagate(
  rate: number,
  momentum: number,
  update: boolean,
  regularization: number | { type: "L1" | "L2"; lambda: number; } | ((weight: number) => number),
  target: number | undefined,
): void
```

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

```ts
propagate(
  rate: number,
  momentum: number,
  target: number[] | undefined,
): void
```

Propagates the error backward through all nodes in the layer.

This is a core step in the backpropagation algorithm used for training.
If a `target` array is provided (typically for the output layer), it's used
to calculate the initial error for each node. Otherwise, nodes calculate
their error based on the error propagated from subsequent layers.

Parameters:
- `rate` - The learning rate, controlling the step size of weight adjustments.
- `momentum` - The momentum factor, used to smooth weight updates and escape local minima.
- `target` - An optional array of target values (expected outputs) for the layer's nodes. The length must match the number of nodes.

#### pruneToSparsity

```ts
pruneToSparsity(
  targetSparsity: number,
  method: "magnitude" | "snip",
): void
```

Immediately prune connections to reach (or approach) a target sparsity fraction.
Used by evolutionary pruning (generation-based) independent of training iteration schedule.

Parameters:
- `targetSparsity` - fraction in (0,1). 0.8 means keep 20% of original (if first call sets baseline)
- `method` - 'magnitude' | 'snip'

#### random

```ts
random(
  input: number,
  hidden: number,
  output: number,
  options: ArchitectLegacyRandomOptions,
): default
```

Creates a randomly structured network based on node counts and connection
options.

This compatibility wrapper preserves the historical `random()` surface
while forwarding to the stricter `randomSparse()` builder that uses the
Phase 2 sparse-profile vocabulary.

Parameters:
- `input` - The number of input nodes.
- `hidden` - The number of hidden nodes to add.
- `output` - The number of output nodes.
- `options` - Optional legacy configuration using lowercase back/self keys.

Returns: The constructed randomized network.

#### randomSparse

```ts
randomSparse(
  input: number,
  hidden: number,
  output: number,
  options: ArchitectRandomSparseOptions,
): default
```

Creates a sparse random starting graph for topology-oriented search.

This builder is the explicit sparse-profile entrypoint for the public
architecture set. It accepts camelCase option names, validates the request
up front, and throws a clear error as soon as one requested structural edit
cannot be satisfied instead of silently leaving the graph underspecified.

Recommended starting knobs:

- topology search: keep `connections` only a little above the hidden-node
  count, then add `backConnections`, `selfConnections`, and `gates`
  gradually as the task proves it benefits from richer recurrence,
- deterministic replay: set `seed` whenever you want to compare topology
  changes instead of random initializers,
- NEAT seeding: pair this preset with `popsize` around `100` and moderate
  `mutationRate` so evolution still has structural headroom left.

Common pitfalls:

- requesting more structural edits than the graph can satisfy,
- starting from a graph that is already so dense that topology search has
  little left to discover.

Parameters:
- `input` - The number of input nodes.
- `hidden` - The number of hidden nodes to add.
- `output` - The number of output nodes.
- `options` - Optional sparse-structure counts for forward connections,
back connections, self connections, gates, and an optional deterministic seed.

Returns: The constructed sparse random network.

Example:

```ts
const network = Architect.randomSparse(3, 6, 1, {
  connections: 10,
  backConnections: 1,
  selfConnections: 1,
  seed: 7,
});

const output = network.activate([0.2, -0.1, 0.8])[0];
```

#### rebuildConnections

```ts
rebuildConnections(
  net: default,
): void
```

Rebuilds the network's connections array from all per-node connections.
This ensures that the network.connections array is consistent with the actual
outgoing connections of all nodes. Useful after manual wiring or node manipulation.

Returns: Example usage:
  Network.rebuildConnections(net);

#### rebuildConnectionSlab

```ts
rebuildConnectionSlab(
  force: boolean,
): void
```

Rebuild slab structures for fast activation.

Parameters:
- `force` - Whether to force a rebuild.

Returns: Slab rebuild result.

#### refreshExplicitIORoles

```ts
refreshExplicitIORoles(): void
```

Refresh explicit ordered input and output role ids from the current graph.

Builder, restore, and evolutionary materialization paths use this after
replacing `nodes` wholesale so the role contract stays explicit even while
activation semantics still rely on legacy ordering rules.

Returns: Nothing.

#### release

```ts
release(
  conn: default,
): void
```

Return a connection instance to the internal pool for later reuse.
Treat the instance as surrendered after calling this method.

Parameters:
- `conn` - The connection instance to recycle.

Returns: Nothing.

#### remove

```ts
remove(
  node: default,
): void
```

Removes a node from the network.
This involves:
1. Disconnecting all incoming and outgoing connections associated with the node.
2. Removing self-connections.
3. Removing the node from the `nodes` array.
4. Attempting to reconnect the node's direct predecessors to its direct successors
   to maintain network flow, if possible and configured.
5. Handling gates involving the removed node (ungating connections gated *by* this node,
   and potentially re-gating connections that were gated *by other nodes* onto the removed node's connections).

#### resetDropoutMasks

```ts
resetDropoutMasks(): void
```

Resets all masks in the network to 1 (no dropout). Applies to both node-level and layer-level dropout.
Should be called after training to ensure inference is unaffected by previous dropout.

#### resetInnovationCounter

```ts
resetInnovationCounter(
  value: number,
): void
```

Reset the monotonic innovation counter used for newly constructed or pooled connections.
You usually call this at the start of an experiment or before rebuilding a whole population.

Parameters:
- `value` - New starting value.

Returns: Nothing.

#### response

Response multiplier applied to the node state before the squash function.

A neutral response of `1` preserves the historical runtime behavior. Values
above or below `1` steepen or flatten the node's effective transfer curve
without changing the chosen activation family.

#### restoreRNG

```ts
restoreRNG(
  fn: () => number,
): void
```

Restore deterministic RNG function from a snapshot source.

Parameters:
- `fn` - RNG function to restore.

#### score

Optional fitness score.

#### secondMoment

Second raw moment estimate used by Adam-family optimizers.

#### secondMomentum

Secondary momentum buffer used by Lion-style updates.

#### selfconns

Self-connection list.

#### serialize

```ts
serialize(): CompactSerializedNetworkTuple
```

Lightweight tuple serializer delegating to network.serialize.ts

#### set

```ts
set(
  values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; },
): void
```

Sets specified properties (e.g., bias, squash function) for all nodes in the network.
Useful for initializing or resetting node properties uniformly.

#### set

```ts
set(
  values: { bias?: number | undefined; squash?: ((x: number, derivate?: boolean | undefined) => number) | undefined; type?: string | undefined; },
): void
```

Configures properties for all nodes within the layer.

Allows batch setting of common node properties like bias, activation function (`squash`),
or node type. If a node within the `nodes` array is actually a `Group` (e.g., in memory layers),
the configuration is applied recursively to the nodes within that group.

Parameters:
- `values` - An object containing the properties and their values to set.
   Example: `{ bias: 0.5, squash: methods.Activation.ReLU }`

#### setActivation

```ts
setActivation(
  fn: (x: number, derivate?: boolean | undefined) => number,
): void
```

Sets a custom activation function for this node at runtime.

Parameters:
- `fn` - The activation function (should handle derivative if needed).

#### setEnforceAcyclic

```ts
setEnforceAcyclic(
  flag: boolean,
): void
```

Enable or disable acyclic topology enforcement.

Parameters:
- `flag` - Whether to enforce acyclic connectivity.

#### setRandom

```ts
setRandom(
  fn: () => number,
): void
```

Replace the network random number generator.

Parameters:
- `fn` - RNG function returning values in $[0,1)$.

#### setRNGState

```ts
setRNGState(
  state: number,
): void
```

Set the raw deterministic RNG state word.

Parameters:
- `state` - RNG state value.

#### setSeed

```ts
setSeed(
  seed: number,
): void
```

Seed the internal deterministic RNG.

Parameters:
- `seed` - Seed value.

#### setStochasticDepth

```ts
setStochasticDepth(
  survival: number[],
): void
```

Configure stochastic depth with survival probabilities per hidden layer.

Parameters:
- `survival` - Survival probabilities for hidden layers.

#### setStochasticDepthSchedule

```ts
setStochasticDepthSchedule(
  fn: (step: number, current: number[]) => number[],
): void
```

Set stochastic-depth schedule function.

Parameters:
- `fn` - Function mapping step and current schedule to next schedule.

#### setTopologyIntent

```ts
setTopologyIntent(
  topologyIntent: NetworkTopologyIntent,
): void
```

Sets the public topology intent and keeps acyclic enforcement aligned.

Parameters:
- `topologyIntent` - Desired topology intent.

Returns: Nothing.

#### setWeightNoiseSchedule

```ts
setWeightNoiseSchedule(
  fn: (step: number) => number,
): void
```

Set a dynamic scheduler for global weight noise.

Parameters:
- `fn` - Function mapping training step to noise standard deviation.

#### snapshotRNG

```ts
snapshotRNG(): RNGSnapshot
```

Snapshot deterministic RNG runtime state.

Returns: Current RNG snapshot.

#### squash

```ts
squash(
  x: number,
  derivate: boolean | undefined,
): number
```

The activation function (squashing function) applied to the node's state.
Maps the internal state to the node's output (activation).

Parameters:
- `x` - The node's internal state (sum of weighted inputs + bias).
- `derivate` - If true, returns the derivative of the function instead of the function value.

Returns: The activation value or its derivative.

#### standalone

```ts
standalone(): string
```

Generate a dependency-light standalone inference function for this network.

Use this when you want to snapshot the current topology and weights into a
self-contained JavaScript function for deployment, offline benchmarking,
or browser embedding without the full training runtime.

Returns: Standalone JavaScript source for inference.

#### state

The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.

#### syncGeneIdCounter

```ts
syncGeneIdCounter(
  maxObservedGeneId: number,
): void
```

Advances the global gene-id cursor past a restored maximum.

Restore flows use this after hydrating persisted genomes so the next freshly
created node cannot collide with an older serialized `geneId`.

Parameters:
- `maxObservedGeneId` - Highest restored node gene id currently in memory.

Returns: Nothing.

#### syncInnovationCounter

```ts
syncInnovationCounter(
  maxObservedInnovation: number,
): void
```

Advances the innovation cursor past a restored maximum.

This keeps import and clone paths monotonic: once a payload brings in a high
innovation id, newly created edges continue from above that value.

Parameters:
- `maxObservedInnovation` - Highest restored innovation id currently in memory.

Returns: Nothing.

#### test

```ts
test(
  set: { input: number[]; output: number[]; }[],
  cost: ((target: number[], output: number[]) => number) | undefined,
): { error: number; time: number; }
```

Tests the network's performance on a given dataset.
Calculates the average error over the dataset using a specified cost function.
Uses `noTraceActivate` for efficiency as gradients are not needed.
Handles dropout scaling if dropout was used during training.

Returns: An object containing the calculated average error over the dataset and the time taken for the test in milliseconds.

#### testForceOverflow

```ts
testForceOverflow(): void
```

Force the next mixed-precision overflow path (test utility).

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

```ts
toJSON(): Record<string, unknown>
```

Verbose JSON serializer delegate

#### toJSON

```ts
toJSON(): { index: number | undefined; bias: number; response: number; type: string; squash: string | null; mask: number; }
```

Converts the node's essential properties to a JSON object for serialization.
Does not include state, activation, error, or connection information, as these
are typically transient or reconstructed separately.

Returns: A JSON representation of the node's configuration.

#### toJSON

```ts
toJSON(): { size: number; nodeIndices: (number | undefined)[]; connections: { in: number; out: number; self: number; }; }
```

Serializes the group into a JSON-compatible format, avoiding circular references.

Returns: JSON-friendly representation with node indices and connection counts.

#### toJSON

```ts
toJSON(): { from: number | undefined; to: number | undefined; weight: number; gain: number; innovation: number; enabled: boolean; gater?: number | undefined; }
```

Serialize to a minimal JSON-friendly shape used by genome and network save flows.
Undefined node indices are preserved so callers can resolve or remap them later.

Returns: Object with node indices, weight, gain, innovation id, enabled flag, and gater index when one exists.

Example:

```ts
const json = connection.toJSON();
// => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
```

#### toONNX

```ts
toONNX(): OnnxModel
```

Exports the network to ONNX format (JSON object, minimal MLP support).
Only standard feedforward architectures and standard activations are supported.
Gating, custom activations, and evolutionary features are ignored or replaced with Identity.

Returns: ONNX model as a JSON object.

#### totalDeltaBias

Accumulates changes in bias over a mini-batch during batch training. Reset after each weight update.

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### train

```ts
train(
  set: { input: number[]; output: number[]; }[],
  options: unknown,
): { error: number; iterations: number; time: number; }
```

Train the network against a supervised dataset using the gradient-based
training chapter.

This wrapper keeps the public `Network` API stable while the training
helpers own batching, optimizer steps, regularization, and mixed-precision
runtime behavior.

Parameters:
- `set` - Supervised samples with `input` and `output` vectors.
- `options` - Training options such as learning rate, iteration limits, batching, and optimizer settings.

Returns: Aggregate training result with final error, iteration count, and elapsed time.

#### trainingStep

Current training step counter.

#### type

The type of the node: 'input', 'hidden', or 'output'.
Determines behavior (e.g., input nodes don't have biases modified typically, output nodes calculate error differently).

#### ungate

```ts
ungate(
  connection: default,
): void
```

Removes the gate from a specified connection.
The connection will no longer be modulated by its gater node.
Removes the connection from the network's `gates` list.

#### ungate

```ts
ungate(
  connections: default | default[],
): void
```

Removes this node's gating control over the specified connection(s).
Resets the connection's gain to 1 and removes it from the `connections.gated` list.

Parameters:
- `connections` - A single Connection object or an array of Connection objects to ungate.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.
