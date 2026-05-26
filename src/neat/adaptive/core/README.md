# neat/adaptive/core

Reading map and stable index for the adaptive shared-vocabulary chapter.

The other adaptive folders explain concrete feedback loops: acceptance
rewrites scores, complexity rewrites structural budgets, mutation rewrites
per-genome pressure, and lineage rewrites future diversity pressure. This
folder exists for the question underneath all of them: what shared language
lets those controllers stay narrow without each one inventing its own host
contract, option slice, scratch-field names, and fallback constants?

That is why `adaptive/core/` should be read as a glossary with ownership,
not as a bag of types. `adaptive.core.types.ts` defines what adaptive
helpers are allowed to read and rewrite. `adaptive.core.constants.ts`
defines the shared defaults, mode names, and numeric guard rails that keep
those helpers speaking the same policy dialect. This file is the stable
import surface that points later helpers at both shelves without making them
depend on file-by-file internals.

A useful mental model is to treat the chapter as the protocol layer between
adaptive controllers. Complexity, acceptance, mutation, and lineage each own
a different control loop, but they still have to agree on the shape of the
host, the option slices they may inspect, and the small amount of adaptive
state they may persist across generations.

```mermaid
flowchart TD
  Core[adaptive core index] --> Types[types<br/>host contract and working shapes]
  Core --> Constants[constants<br/>defaults modes and guard rails]
  Types --> Acceptance[acceptance helpers]
  Types --> Complexity[complexity helpers]
  Types --> Mutation[mutation helpers]
  Types --> Lineage[lineage helpers]
  Constants --> Acceptance
  Constants --> Complexity
  Constants --> Mutation
  Constants --> Lineage
```

Read the chapter in three passes:

1. start with `adaptive.core.types.ts` when the missing question is what an
   adaptive helper may legally read or rewrite,
2. continue through the re-exported aliases in this file when you want the
   shortest stable import path for that vocabulary,
3. finish with `adaptive.core.constants.ts` when the missing question is
   which defaults, phase labels, thresholds, and clamps keep the adaptive
   loops aligned.

Example: type a new adaptive helper against the shared host contract instead
of a broader `Neat` runtime surface.

```ts
import type {
  NeatLikeWithAdaptive,
  ComplexityBudgetConfig,
} from './adaptive.core';

function previewBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
) {
  return { generation: engine.generation, enabled: config.enabled ?? false };
}
```

Example: keep constants and contracts on one import path when an extracted
helper needs both a typed option slice and a shared mode label.

```ts
import type { PhasedComplexityConfig } from './adaptive.core';
import { PHASE_COMPLEXIFY } from './adaptive.core.constants';

function isComplexifyPhase(config: PhasedComplexityConfig) {
  return (config.initialPhase ?? PHASE_COMPLEXIFY) === PHASE_COMPLEXIFY;
}
```

## neat/adaptive/core/adaptive.core.ts

### AdaptiveMutationConfig

Shared config view for per-genome adaptive mutation helpers.

The mutation adaptation loop reads this slice to clamp rates, normalize
perturbation scales, and decide how often genome-local parameters are
refreshed. It is the mutation-side policy vocabulary before defaults are
resolved into `MutationSettings`.

### AncestorUniqAdaptiveConfig

Shared config view for ancestor-uniqueness feedback helpers.

This captures the thresholds, cooldowns, and mode switches used when the
controller nudges diversity pressure in response to lineage concentration.
Read it as the lineage-feedback slice of the broader adaptive options object.

### ComplexityBudgetConfig

Shared config view for complexity-budget helpers.

Use this alias when a helper only cares about node and connection caps,
schedule shape, and improvement-window tuning for the adaptive budget loop.
It is the smallest view needed for the controller that grows or shrinks the
allowed topology budget over time.

### Genome

Shared genome view used by the adaptive helpers.

This is intentionally opaque beyond the adaptive scratch fields already
exposed through `NeatLikeWithAdaptive['population']`. The adaptive core cares
about per-genome scores and adaptive overrides, not full network structure.

### MinimalCriterionAdaptiveConfig

Shared config view for adaptive minimal-criterion helpers.

Helpers use this slice when they are only adjusting the acceptance
threshold, not inspecting the rest of the controller policy surface. This is
the acceptance-side tuning vocabulary, not a whole-population runtime view.

### MutationOutcome

Outcome flags used to detect whether mutation pressure stayed balanced.

Mutation adaptation uses this tiny result object to summarize whether recent
adjustments produced both upward and downward movement rather than collapsing
into one-sided pressure.

### MutationPartitions

Score-ranked population halves used by two-tier and explore-low strategies.

This keeps the mutation strategies' working partitions explicit so helpers
can talk about "top half" and "bottom half" without recomputing or loosely
describing that split.

### MutationSettings

Normalized adaptive-mutation settings after config fallback resolution.

This is the working form used after helpers merge user options with shared
defaults, so downstream logic does not need to repeatedly re-interpret
optional config fields. Read it as the mutation chapter's resolved call
frame: one object with every clamp, sigma, strategy, and baseline already in
concrete form.

### NeatLikeWithAdaptive

Contract map for the adaptive helper boundary.

The adaptive subtree works because each policy chapter can stay focused on a
single feedback loop while still sharing one precise agreement about what it
may read, what it may rewrite, and which option family owns each tuning
decision. This file is that agreement.

Read the contracts in three passes:

- start with `NeatLikeWithAdaptive` to see the runtime host surface and the
  scratch fields adaptive controllers are allowed to maintain,
- continue with the exported `*Config` aliases to see how complexity,
  acceptance, mutation, operator adaptation, and lineage feedback each slice
  the broader options object,
- finish with `Genome`, `MutationSettings`, `MutationPartitions`, and
  `MutationOutcome` when you want the normalized working shapes used inside
  adaptive mutation helpers.

The matching defaults and mode labels live in `adaptive.core.constants.ts`.
This file stays focused on contracts so the generated chapter reads as a
bounded vocabulary map rather than a second controller implementation.

```mermaid
flowchart TD
  Host[NeatLikeWithAdaptive host] --> Options[Adaptive option families]
  Host --> Population[Population runtime state]
  Host --> Scratch[Adaptive scratch fields and telemetry]
  Options --> Complexity[Complexity and phased schedules]
  Options --> Acceptance[Acceptance and minimal criterion]
  Options --> Mutation[Mutation and operator adaptation]
  Options --> Lineage[Ancestor uniqueness and lineage pressure]
```

### OperatorAdaptationConfig

Shared config view for operator-stat adaptation helpers.

This is the policy surface for helpers that bias mutation-operator choice
using historical success and attempt statistics. It stays separate from the
broader adaptive mutation config because operator-choice decay is a different
feedback loop from per-genome rate tuning.

### PhasedComplexityConfig

Shared config view for phased-complexity helpers.

This isolates the alternating complexify/simplify schedule from the broader
adaptive options object so phase-oriented helpers can stay narrow and think
in terms of mode transitions instead of the entire adaptive policy surface.

## neat/adaptive/core/adaptive.core.types.ts

Contract map for the adaptive helper boundary.

The adaptive subtree works because each policy chapter can stay focused on a
single feedback loop while still sharing one precise agreement about what it
may read, what it may rewrite, and which option family owns each tuning
decision. This file is that agreement.

Read the contracts in three passes:

- start with `NeatLikeWithAdaptive` to see the runtime host surface and the
  scratch fields adaptive controllers are allowed to maintain,
- continue with the exported `*Config` aliases to see how complexity,
  acceptance, mutation, operator adaptation, and lineage feedback each slice
  the broader options object,
- finish with `Genome`, `MutationSettings`, `MutationPartitions`, and
  `MutationOutcome` when you want the normalized working shapes used inside
  adaptive mutation helpers.

The matching defaults and mode labels live in `adaptive.core.constants.ts`.
This file stays focused on contracts so the generated chapter reads as a
bounded vocabulary map rather than a second controller implementation.

```mermaid
flowchart TD
  Host[NeatLikeWithAdaptive host] --> Options[Adaptive option families]
  Host --> Population[Population runtime state]
  Host --> Scratch[Adaptive scratch fields and telemetry]
  Options --> Complexity[Complexity and phased schedules]
  Options --> Acceptance[Acceptance and minimal criterion]
  Options --> Mutation[Mutation and operator adaptation]
  Options --> Lineage[Ancestor uniqueness and lineage pressure]
```

### AdaptiveMutationConfig

Shared config view for per-genome adaptive mutation helpers.

The mutation adaptation loop reads this slice to clamp rates, normalize
perturbation scales, and decide how often genome-local parameters are
refreshed. It is the mutation-side policy vocabulary before defaults are
resolved into `MutationSettings`.

### AncestorUniqAdaptiveConfig

Shared config view for ancestor-uniqueness feedback helpers.

This captures the thresholds, cooldowns, and mode switches used when the
controller nudges diversity pressure in response to lineage concentration.
Read it as the lineage-feedback slice of the broader adaptive options object.

### ComplexityBudgetConfig

Shared config view for complexity-budget helpers.

Use this alias when a helper only cares about node and connection caps,
schedule shape, and improvement-window tuning for the adaptive budget loop.
It is the smallest view needed for the controller that grows or shrinks the
allowed topology budget over time.

### Genome

Shared genome view used by the adaptive helpers.

This is intentionally opaque beyond the adaptive scratch fields already
exposed through `NeatLikeWithAdaptive['population']`. The adaptive core cares
about per-genome scores and adaptive overrides, not full network structure.

### MinimalCriterionAdaptiveConfig

Shared config view for adaptive minimal-criterion helpers.

Helpers use this slice when they are only adjusting the acceptance
threshold, not inspecting the rest of the controller policy surface. This is
the acceptance-side tuning vocabulary, not a whole-population runtime view.

### MutationOutcome

Outcome flags used to detect whether mutation pressure stayed balanced.

Mutation adaptation uses this tiny result object to summarize whether recent
adjustments produced both upward and downward movement rather than collapsing
into one-sided pressure.

### MutationPartitions

Score-ranked population halves used by two-tier and explore-low strategies.

This keeps the mutation strategies' working partitions explicit so helpers
can talk about "top half" and "bottom half" without recomputing or loosely
describing that split.

### MutationSettings

Normalized adaptive-mutation settings after config fallback resolution.

This is the working form used after helpers merge user options with shared
defaults, so downstream logic does not need to repeatedly re-interpret
optional config fields. Read it as the mutation chapter's resolved call
frame: one object with every clamp, sigma, strategy, and baseline already in
concrete form.

### NeatLikeWithAdaptive

Contract map for the adaptive helper boundary.

The adaptive subtree works because each policy chapter can stay focused on a
single feedback loop while still sharing one precise agreement about what it
may read, what it may rewrite, and which option family owns each tuning
decision. This file is that agreement.

Read the contracts in three passes:

- start with `NeatLikeWithAdaptive` to see the runtime host surface and the
  scratch fields adaptive controllers are allowed to maintain,
- continue with the exported `*Config` aliases to see how complexity,
  acceptance, mutation, operator adaptation, and lineage feedback each slice
  the broader options object,
- finish with `Genome`, `MutationSettings`, `MutationPartitions`, and
  `MutationOutcome` when you want the normalized working shapes used inside
  adaptive mutation helpers.

The matching defaults and mode labels live in `adaptive.core.constants.ts`.
This file stays focused on contracts so the generated chapter reads as a
bounded vocabulary map rather than a second controller implementation.

```mermaid
flowchart TD
  Host[NeatLikeWithAdaptive host] --> Options[Adaptive option families]
  Host --> Population[Population runtime state]
  Host --> Scratch[Adaptive scratch fields and telemetry]
  Options --> Complexity[Complexity and phased schedules]
  Options --> Acceptance[Acceptance and minimal criterion]
  Options --> Mutation[Mutation and operator adaptation]
  Options --> Lineage[Ancestor uniqueness and lineage pressure]
```

### OperatorAdaptationConfig

Shared config view for operator-stat adaptation helpers.

This is the policy surface for helpers that bias mutation-operator choice
using historical success and attempt statistics. It stays separate from the
broader adaptive mutation config because operator-choice decay is a different
feedback loop from per-genome rate tuning.

### PhasedComplexityConfig

Shared config view for phased-complexity helpers.

This isolates the alternating complexify/simplify schedule from the broader
adaptive options object so phase-oriented helpers can stay narrow and think
in terms of mode transitions instead of the entire adaptive policy surface.

## neat/adaptive/core/adaptive.core.constants.ts

Shared defaults, labels, and numeric guard rails for the adaptive subtree.

The adaptive chapters reuse this file so they can talk about policy with one
stable vocabulary for thresholds, strategy names, phase labels, and fallback
numbers instead of redefining those values inline.

Read this after the contract map in `adaptive.core.types.ts`. The types file
answers "which knobs and scratch fields exist?" while this file answers
"what should those knobs and scratch fields default to when the caller does
not override them?"

Read the exports as four families rather than one long constant shelf:

- tiny numeric helpers such as `ZERO`, `ONE`, and `NEGATIVE_ONE` keep the
  utility math explicit without scattering magic numbers,
- schedule defaults such as `DEFAULT_IMPROVEMENT_WINDOW`,
  `DEFAULT_CB_INCREASE_FACTOR`, and `PHASE_LENGTH_DEFAULT` shape how quickly
  adaptive controllers react,
- mode labels such as `COMPLEXITY_MODE_ADAPTIVE`, `PHASE_COMPLEXIFY`, and
  `MUTATION_STRATEGY_ANNEAL` give the helper files one shared vocabulary,
- clamp and multiplier values define the safe operating envelope for
  acceptance tuning, lineage pressure, and mutation adaptation.

The goal is not to memorize every export. The goal is to see that the
adaptive subtree reuses one glossary for tiny math helpers, schedule timing,
mode names, and safety clamps instead of scattering unrelated literals across
each control loop.

### ACCEPTANCE_LOWER_MULTIPLIER

Lower multiplier applied when the adaptive controller relaxes acceptance threshold pressure.

### ACCEPTANCE_UPPER_MULTIPLIER

Upper multiplier used when the controller nudges acceptance pressure upward.

### ADJUST_RATE_DEFAULT

Default adjustment step size for adaptive minimal-criterion acceptance threshold updates.

### ANCESTOR_UNIQ_MODE_EPSILON

Mode label for epsilon-style ancestor-uniqueness feedback, adjusting the pressure threshold directly.

### ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE

Mode label for ancestor-uniqueness control achieved through lineage-pressure strength tuning.

### ANNEAL_BASELINE_GENERATIONS

Baseline generation count used when computing annealing schedule progress towards maximum.

### ANNEAL_PROGRESS_MAX

Maximum progress ratio clamp applied in annealing schedule pressure calculations.

### BUDGET_GROWTH_MULTIPLIER

Default complexity budget growth multiplier applied when the controller detects sustained fitness improvement.

### COMPLEXITY_MODE_ADAPTIVE

Mode label for feedback-driven adaptive complexity-budget scheduling with slope detection.

### COMPLEXITY_MODE_LINEAR

Mode label for pre-planned linear complexity-budget scheduling with a fixed generation horizon.

### DEFAULT_ADAPT_EVERY

Default generation cadence for refreshing per-genome adaptive mutation rate settings.

### DEFAULT_ANCESTOR_UNIQ_ADJUST

Default adjustment magnitude applied when nudging ancestor-uniqueness pressure thresholds upward or downward.

### DEFAULT_ANCESTOR_UNIQ_COOLDOWN

Default generation cooldown between consecutive ancestor-uniqueness acceptance threshold adjustment steps.

### DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD

Default upper bound for acceptable ancestor uniqueness, above which pressure is relaxed.

### DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD

Default lower bound for acceptable ancestor uniqueness, below which pressure is increased.

### DEFAULT_CB_INCREASE_FACTOR

Default multiplier applied when adaptive complexity budgeting detects a fitness improvement.

### DEFAULT_CB_STAGNATION_FACTOR

Default multiplier used when adaptive complexity budgeting responds to stagnation.

### DEFAULT_IMPROVEMENT_WINDOW

Default score-history window length used by trend-aware adaptive complexity budgeting.

### DEFAULT_INITIAL_MUTATION_RATE

Default initial mutation rate used before adaptive balancing specializes genomes.

### DEFAULT_LINEAGE_PRESSURE_STRENGTH

Default lineage pressure strength applied when initializing the ancestor-uniqueness option.

### DEFAULT_MAX_MUTATION_AMOUNT

Default upper bound for the per-genome adaptive mutation amount value.

### DEFAULT_MAX_MUTATION_RATE

Default upper clamp applied to per-genome adaptive mutation rate evolution.

### DEFAULT_MIN_MUTATION_AMOUNT

Default lower bound for the per-genome adaptive mutation amount value.

### DEFAULT_MIN_MUTATION_RATE

Default lower clamp applied to per-genome adaptive mutation rate evolution.

### DEFAULT_MUTATION_AMOUNT

Default mutation amount applied when a genome's target value is missing or not set.

### DEFAULT_MUTATION_AMOUNT_SIGMA

Default Gaussian perturbation spread used for adaptive mutation-amount self-adaptation rate evolution.

### DEFAULT_MUTATION_SIGMA

Default Gaussian perturbation spread used for adaptive mutation-rate self-adaptation rate evolution.

### DENOMINATOR_FALLBACK

Fallback denominator substituted when the real denominator is zero or missing.

### EXPLORE_LOW_DECREASE_MULTIPLIER

Multiplicative decay applied to top-half genomes under the explore-low adaptive strategy.

### EXPLORE_LOW_INCREASE_MULTIPLIER

Multiplicative boost applied to bottom-half genomes under the explore-low adaptive strategy.

### FIVE

Small count baseline reused by archive-size, cooldown, and neighbor-count defaults.

### FOUR

Small numeric constant reused by adaptive-budget growth defaults and initialisation helpers.

### HALF_INDEX_DIVISOR

Divisor applied when splitting ranked populations into top and bottom halves.

### HISTORY_MIN_IMPROVEMENT_COUNT

Minimum score-history length required to compute an improvement signal reliably.

### HISTORY_MIN_SLOPE_COUNT

Minimum score-history length required to compute a trend slope value.

### LINEAGE_PRESSURE_DECREASE_MULTIPLIER

Multiplier applied when the controller decreases lineage pressure strength after over-pressure.

### LINEAGE_PRESSURE_INCREASE_MULTIPLIER

Multiplier applied when the controller increases lineage pressure strength to drive diversity.

### LINEAGE_PRESSURE_MODE_SPREAD

Lineage pressure spread mode label for population-wide diversity pressure distribution.

### LINEAR_HORIZON_DEFAULT

Default generation horizon used by the pre-planned linear complexity-budget schedule.

### MINIMAL_TOPOLOGY_OFFSET

Offset added to input and output width for minimal feed-forward topology sizing.

### MUTATION_SIGMA_SCALE

Scale factor applied to mutation sigma when computing self-adaptation perturbation bounds.

### MUTATION_STRATEGY_ANNEAL

Strategy label for annealed mutation pressure that decreases across run progress.

### MUTATION_STRATEGY_EXPLORE_LOW

Strategy label for boosting structural risk on the lower-ranked population half.

### MUTATION_STRATEGY_TWO_TIER

Strategy label for ranking-sensitive two-tier adaptive mutation with separate top and bottom behavior.

### NEGATIVE_ONE

Last-index sentinel reused when helpers need the final recorded item.

### NOVELTY_ARCHIVE_MIN_SIZE

Minimum number of entries required in the novelty archive for reliable scoring.

### NOVELTY_FACTOR_DEFAULT

Novelty scaling factor applied when the archive holds enough neighbor candidates.

### NOVELTY_FACTOR_SMALL

Novelty scaling factor applied when the archive holds too few neighbor candidates.

### ONE

Unit baseline reused by clamp, ratio, fallback, and normalisation calculations.

### ONE_HUNDRED

Large round-number constant used as a default for long-horizon adaptive scheduling.

### OPERATOR_DECAY_DEFAULT

Default decay factor applied per generation to unused adaptive mutation operator weights.

### PHASE_COMPLEXIFY

Phase label for the structure-growth side of phased complexity scheduling.

### PHASE_LENGTH_DEFAULT

Default phase length in generations for phased complexify and simplify schedule cycling.

### PHASE_SIMPLIFY

Phase label for the structure-pruning side of phased complexity scheduling.

### PROGRESS_RATIO_MAX

Maximum progress ratio clamp for linear and annealing complexity schedule calculations.

### RNG_CENTER_OFFSET

Random center offset subtracted when mapping uniform samples to signed perturbation deltas.

### RNG_SPREAD_MULTIPLIER

Random range multiplier used when computing signed self-adaptation perturbation deltas.

### SLOPE_BOOST_MULTIPLIER

Slope boost multiplier applied to the adaptive increase factor under improving fitness trends.

### SLOPE_NORMALIZE_CLAMP

Clamp magnitude applied when normalizing fitness trend slopes to bounded values.

### SLOPE_PENALTY_MULTIPLIER

Slope penalty multiplier applied to the stagnation factor under declining fitness trends.

### TARGET_ACCEPTANCE_DEFAULT

Default acceptance probability target used by adaptive minimal-criterion acceptance control.

### TEN

Round-number constant reused by improvement windows and default phase lengths.

### THREE

Small threshold used by helpers that need a minimal multi-sample floor.

### TWO

Small divisor and offset used by split and normalization helpers.

### ZERO

Zero baseline reused by tiny adaptive arithmetic helpers.
