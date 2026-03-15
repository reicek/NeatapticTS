# neat/evolve

## neat/evolve/evolve.types.ts

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

## neat/evolve/evolve.ts

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
