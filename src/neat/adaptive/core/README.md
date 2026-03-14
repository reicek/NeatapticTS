# neat/adaptive/core

Minimal NEAT controller shape required by the adaptive helper boundary.

The adaptive folders share this vocabulary so each teaching-oriented README
can focus on its local heuristics without redefining the controller surface.

## neat/adaptive/core/adaptive.core.types.ts

### AdaptiveMutationConfig

### AncestorUniqAdaptiveConfig

### ComplexityBudgetConfig

### Genome

### MinimalCriterionAdaptiveConfig

### MutationOutcome

### MutationPartitions

### MutationSettings

### NeatLikeWithAdaptive

Minimal NEAT controller shape required by the adaptive helper boundary.

The adaptive folders share this vocabulary so each teaching-oriented README
can focus on its local heuristics without redefining the controller surface.

### OperatorAdaptationConfig

### PhasedComplexityConfig

## neat/adaptive/core/adaptive.core.ts

Shared vocabulary for the adaptive controllers.

Read this folder when you need the common constants, config shapes, and
runtime contracts that the other adaptive categories build on.

### AdaptiveMutationConfig

### AncestorUniqAdaptiveConfig

### ComplexityBudgetConfig

### Genome

### MinimalCriterionAdaptiveConfig

### MutationOutcome

### MutationPartitions

### MutationSettings

### NeatLikeWithAdaptive

Minimal NEAT controller shape required by the adaptive helper boundary.

The adaptive folders share this vocabulary so each teaching-oriented README
can focus on its local heuristics without redefining the controller surface.

### OperatorAdaptationConfig

### PhasedComplexityConfig

## neat/adaptive/core/adaptive.core.constants.ts

Constant: zero value.

### ACCEPTANCE_LOWER_MULTIPLIER

Lower acceptance multiplier.

### ACCEPTANCE_UPPER_MULTIPLIER

Upper acceptance multiplier.

### ADJUST_RATE_DEFAULT

Default adjustment rate in minimal criterion.

### ANCESTOR_UNIQ_MODE_EPSILON

Ancestor uniqueness epsilon mode.

### ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE

Ancestor uniqueness lineage pressure mode.

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

### MUTATION_SIGMA_SCALE

Scale applied to mutation sigma for perturbations.

### MUTATION_STRATEGY_ANNEAL

Strategy identifier for annealed mutation.

### MUTATION_STRATEGY_EXPLORE_LOW

Strategy identifier for explore-low mutation.

### MUTATION_STRATEGY_TWO_TIER

Strategy identifier for two-tier mutation.

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

### PHASE_COMPLEXIFY

Phase label for complexify.

### PHASE_LENGTH_DEFAULT

Default phase length in generations.

### PHASE_SIMPLIFY

Phase label for simplify.

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
