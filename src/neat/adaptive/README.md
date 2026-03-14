# neat/adaptive

Adaptive NEAT controllers for complexity, acceptance, lineage diversity,
and mutation-rate tuning.

This root entrypoint stays intentionally small: it explains the high-level
adaptive story, while the detailed heuristics now live in focused teaching
folders for complexity, mutation, acceptance, lineage, and shared runtime
vocabulary.

## neat/adaptive/adaptive.ts

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

### NeatLikeWithAdaptive

Minimal NEAT controller shape required by the adaptive helper boundary.

The adaptive folders share this vocabulary so each teaching-oriented README
can focus on its local heuristics without redefining the controller surface.
