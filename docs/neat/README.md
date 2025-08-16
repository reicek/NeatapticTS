# neat

## neat/neat.adaptive.ts

### applyAdaptiveMutation

`() => void`

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

### applyAncestorUniqAdaptive

`() => void`

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

Decay operator adaptation statistics (success/attempt counters).

Many adaptive operator-selection schemes keep running tallies of how
successful each operator has been. This helper applies an exponential
moving-average style decay to those counters so older outcomes
progressively matter less.

The `_operatorStats` map on `this` is expected to contain values of
the shape `{ success: number, attempts: number }` keyed by operator
id/name.

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

## neat/neat.compat.ts

### _compatibilityDistance

`(genomeA: any, genomeB: any) => number`

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

`(connection: any) => number`

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

## neat/neat.constants.ts

### EPSILON

### EXTRA_CONNECTION_PROBABILITY

### NORM_EPSILON

### PROB_EPSILON

## neat/neat.diversity.ts

### arrayMean

`(values: number[]) => number`

Compute the arithmetic mean of a numeric array. Returns 0 for empty arrays.
Extracted as a helper so it can be documented/tested independently.

### arrayVariance

`(values: number[]) => number`

Compute the variance (population variance) of a numeric array.
Returns 0 for empty arrays. Uses arrayMean internally.

### CompatComputer

Minimal interface that provides a compatibility distance function.
Implementors should expose a compatible signature with legacy NEAT code.

### computeDiversityStats

`(population: any[], compatibilityComputer: CompatComputer) => import("D:/code-practice/NeatapticTS/src/neat/neat.diversity").DiversityStats | undefined`

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

### structuralEntropy

`(graph: import("D:/code-practice/NeatapticTS/src/architecture/network").default) => number`

const JSDoc short descriptions above each constant

## neat/neat.evaluate.ts

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

## neat/neat.evolve.ts

### evolve

`() => Promise<import("D:/code-practice/NeatapticTS/src/architecture/network").default>`

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

## neat/neat.export.ts

### exportPopulation

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.export").GenomeJSON[]`

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

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.export").NeatStateJSON`

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

### fromJSONImpl

`(neatJSON: import("D:/code-practice/NeatapticTS/src/neat/neat.export").NeatMetaJSON, fitnessFunction: (network: any) => number) => any`

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

### importPopulation

`(populationJSON: import("D:/code-practice/NeatapticTS/src/neat/neat.export").GenomeJSON[]) => void`

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

`(stateBundle: import("D:/code-practice/NeatapticTS/src/neat/neat.export").NeatStateJSON, fitnessFunction: (network: any) => number) => any`

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

### NeatMetaJSON

Serialized meta information describing a NEAT run, excluding the concrete
population genomes. This allows you to persist & resume experiment context
(innovation history, current generation, IO sizes, hyper‑parameters) without
committing to a particular population snapshot.

### NeatStateJSON

Top‑level bundle containing both NEAT meta information and the full array of
serialized genomes (population). This is what you get from `exportState()` and
feed into `importStateImpl()` to resume exactly where you left off.

### toJSONImpl

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.export").NeatMetaJSON`

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

`(genome: any, parents: number[] | undefined) => void`

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

`(seedNetwork: any) => void`

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

### spawnFromParent

`(parentGenome: any, mutateCount: number) => any`

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

### buildAnc

`(genome: import("D:/code-practice/NeatapticTS/src/neat/neat.lineage").GenomeLike) => Set<number>`

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

Lineage / ancestry analysis helpers for NEAT populations.

These utilities were migrated from the historical implementation inside `src/neat.ts`
to keep core NEAT logic lean while still exposing educational metrics for users who
want to introspect evolutionary diversity.

Glossary:
 - Genome: An individual network encoding (has a unique `_id` and optional `_parents`).
 - Ancestor Window: A shallow breadth‑first window (default depth = 4) over the lineage graph.
 - Jaccard Distance: 1 - |A ∩ B| / |A ∪ B|, measuring dissimilarity between two sets.

### NeatLineageContext

Expected `this` context for lineage helpers (a subset of the NEAT instance).

## neat/neat.multiobjective.ts

### fastNonDominated

`(pop: import("D:/code-practice/NeatapticTS/src/architecture/network").default[]) => import("D:/code-practice/NeatapticTS/src/architecture/network").default[][]`

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

### ObjectiveDescriptor

Shape of an objective descriptor used by the Neat instance.
- `accessor` extracts a numeric objective from a genome
- `direction` optionally indicates whether the objective is maximized or
  minimized (defaults to 'max')

## neat/neat.mutation.ts

### ensureMinHiddenNodes

`(network: any, multiplierOverride: number | undefined) => void`

Ensure the network has a minimum number of hidden nodes and connectivity.

### ensureNoDeadEnds

`(network: any) => void`

Ensure there are no dead-end nodes (input/output isolation) in the network.

### mutate

`() => void`

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

`(genome: any) => void`

Add a connection between two unconnected nodes reusing a stable innovation id per pair.

### mutateAddNodeReuse

`(genome: any) => void`

Split a random enabled connection inserting a hidden node while reusing historical
innovations for identical (from,to) pairs across genomes. Extracted from Neat class.

### selectMutationMethod

`(genome: any, rawReturnForTest: boolean) => any`

Select a mutation method respecting structural constraints and adaptive controllers.
Mirrors legacy implementation from `neat.ts` to preserve test expectations.
`rawReturnForTest` retains historical behavior where the full FFW array is
returned for identity checks in tests.

## neat/neat.objectives.ts

### _getObjectives

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.types").ObjectiveDescriptor[]`

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

`(key: string, direction: "max" | "min", accessor: (genome: import("D:/code-practice/NeatapticTS/src/neat/neat.types").GenomeLike) => number) => void`

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

Parameters:
- `` - Unique name for the objective (used for sorting/lookup)
- `` - Whether the objective should be minimized or maximized
- `` - Function to extract a numeric value from a genome

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

## neat/neat.selection.ts

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

`() => any`

Return the fittest genome in the population.

This will trigger an `evaluate()` if genomes have not been scored yet, and
will ensure the population is sorted so index 0 contains the fittest.

Example:
const best = neat.getFittest();
console.log(best.score);

Returns: The genome object judged to be the fittest (highest score).

### getParent

`() => any`

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

## neat/neat.speciation.ts

### _applyFitnessSharing

`() => void`

Apply fitness sharing within each species.

Fitness sharing reduces the effective fitness of genomes that are clustered
tightly together (close compatibility distance), promoting diversity by
penalizing dense species. Two modes are supported:
 - Kernel sharing with bandwidth `sharingSigma` (quadratic kernel)
 - Equal sharing based on species size when `sharingSigma` is 0

Example:
neat.options.sharingSigma = 3;
neat._applyFitnessSharing();

### _sortSpeciesMembers

`(sp: any) => void`

Sort members of a species in descending order by score.

Simple utility used by stagnation checks and selection routines to ensure
the top-performing genomes are at index 0.

Parameters:
- `sp` - species-like object with a `members` array and member `.score`

### _speciate

`() => void`

Assign genomes into species based on compatibility distance and maintain species structures.
This function creates new species for unassigned genomes, prunes empty species, updates
dynamic compatibility threshold controllers, performs optional auto coefficient tuning, and
records per‑species history statistics used by telemetry and adaptive controllers.

Implementation notes:
- Uses existing representatives; any unassigned genome that doesn't fit an existing species
  creates a new species with itself as representative.
- Representatives are refreshed each generation (first member heuristic) to reduce drift cost.
- Includes optional age penalty for very old species to gently reduce their reproductive share.
- PID‑style controller adjusts the global compatibility threshold toward `targetSpecies`.
- Auto compatibility coefficient tuning slightly nudges excess/disjoint coefficients to influence
  clustering granularity when enabled.
- Extended history snapshot captures structural and innovation statistics for richer telemetry.

### _updateSpeciesStagnation

`() => void`

Update species stagnation statistics and prune species that have not
improved within the configured stagnation window.

This updates each species' `bestScore` and `lastImproved` fields and then
removes species whose age since last improvement exceeds `stagnationGenerations`.

## neat/neat.species.ts

### getSpeciesHistory

`() => import("D:/code-practice/NeatapticTS/src/neat/neat.types").SpeciesHistoryEntry[]`

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

## neat/neat.telemetry.exports.ts

### buildSpeciesHistoryCsv

`(recentHistory: { generation: number; stats: any[]; }[], headers: string[]) => string`

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

`(entries: any[]) => TelemetryHeaderInfo`

Collect header metadata from the raw telemetry entries.
- Discovers base (top‑level) keys excluding grouped objects.
- Discovers nested keys inside complexity, perf, lineage, diversity groups.
- Tracks presence of optional multi-value structures (ops, objectives, etc.).

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

`(entry: any, headers: string[]) => string`

Serialize one telemetry entry into a CSV row using previously computed headers.
Uses a `switch(true)` pattern instead of a long if/else chain to reduce
cognitive complexity while preserving readability of each scenario.

### TelemetryHeaderInfo

Shape describing collected telemetry header discovery info.

## neat/neat.telemetry.ts

### applyTelemetrySelect

`(entry: any) => any`

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

`(fittest: any) => import("D:/code-practice/NeatapticTS/src/neat/neat.types").TelemetryEntry`

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

This helper is intentionally conservative in runtime: when `fastMode` is
enabled it will automatically tune a few sampling defaults to keep the
computation cheap. The computed statistics are written to
`this._diversityStats` as an object with keys like `meanCompat` and
`graphletEntropy`.

The method mutates instance-level temporary fields and reads a number of
runtime options from `this.options`.

### recordTelemetryEntry

`(entry: import("D:/code-practice/NeatapticTS/src/neat/neat.types").TelemetryEntry) => void`

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

`(graph: any) => number`

Lightweight proxy for structural entropy based on degree-distribution.

This function computes an approximate entropy of a graph topology by
counting node degrees and computing the entropy of the degree histogram.
The result is cached on the graph object for the current generation in
`_entropyVal` to avoid repeated expensive recomputation.

Example:

Parameters:
- `graph` - - A genome-like object with `nodes` and `connections` arrays.

Returns: A non-negative number approximating structural entropy.

## neat/neat.types.ts

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

### DiversityStats

Diversity statistics captured each generation. Individual fields may be
omitted in telemetry output if diversity tracking is partially disabled to
reduce runtime cost.

### GenomeLike

Minimal genome structural surface used by several helpers (incrementally expanded).

NOTE: `nodes` & `connections` intentionally remain `any[]` until a stable
`NodeLike` / `ConnectionLike` abstraction is finalised.

### LineageSnapshot

Snapshot of lineage & ancestry statistics for the current generation.

### NeatLike

Minimal surface every helper currently expects from a NEAT instance while
extraction continues. At present it carries no guaranteed properties besides
an index signature. As helpers converge, promote concrete, documented fields.

### ObjAges

Map of objective key to age in generations since introduction.

### ObjectiveDescriptor

Descriptor for a single optimisation objective (single or multi‑objective runs).

### ObjEvent

Dynamic objective lifecycle event (addition or removal).

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

### PerformanceMetrics

Timing metrics for coarse evolutionary phases (milliseconds).

### SpeciesAlloc

Offspring allocation for a species during reproduction.

### SpeciesHistoryEntry

Species statistics captured for a particular generation.

### SpeciesHistoryStat

Species statistics at a single historical snapshot (generation boundary).

### TelemetryEntry

Telemetry summary for one generation.

Optional properties are feature‑dependent; consumers MUST test for presence.
