# neat/speciation/shared

Shared vocabulary and defaults for NEAT speciation.

This chapter is the shared language layer beneath the speciation subtree.
The adjacent chapters split responsibilities on purpose: assignment decides
membership, threshold tuning adjusts the future compatibility boundary,
sharing and stagnation reshape pressure after assignment, and history records
what happened. This file supplies the common words and defaults they all need
in order to cooperate without re-defining the same contracts in each folder.

Read the exports in three families:

1. narrow runtime contexts such as {@link FitnessSharingContext} and
   {@link StagnationContext},
2. compact summary types such as {@link CompatAdjust} and
   {@link InnovationAccumulator},
3. default constants that define the baseline threshold, sharing, age,
   history, and score semantics across the subtree.

## neat/speciation/shared/speciation.shared.ts

### CompatAdjust

Resolved compatibility-threshold adjustment settings.

This is the non-nullable form of {@link SpeciationOptions.compatAdjust} used
by the speciation PID controller.

### DEFAULT_COMPAT_INTEGRAL

Neutral starting value for the compatibility-threshold integral accumulator.

### DEFAULT_COMPATIBILITY_INTEGRAL_GAIN

Default integral gain for accumulated threshold response across generations.

### DEFAULT_COMPATIBILITY_PROPORTIONAL_GAIN

Default proportional gain for immediate threshold response to species-count error.

### DEFAULT_COMPATIBILITY_THRESHOLD

Baseline compatibility boundary used before adaptive tuning moves it.

### DEFAULT_LAST_IMPROVED_GENERATION

Default last improved generation when missing.

### DEFAULT_MAX_COMPATIBILITY_THRESHOLD

Upper bound that keeps adaptive threshold control from merging too aggressively.

### DEFAULT_MEMBER_COUNT_FALLBACK

Fallback divisor when member count is zero.

### DEFAULT_MIN_COMPATIBILITY_THRESHOLD

Lower bound that keeps adaptive threshold control from collapsing to zero.

### DEFAULT_SCORE_FALLBACK

Shared numeric fallback used when a speciation summary needs a missing score.

### DEFAULT_SHARING_SIGMA

Default sharing radius; zero selects the simpler uniform sharing fallback.

### DEFAULT_SPECIES_AGE_GRACE

Default grace window before older-species penalties are allowed to apply.

### DEFAULT_SPECIES_OLD_PENALTY

Default score multiplier applied when an old species is no longer protected.

### DEFAULT_STAGNATION_WINDOW

Default number of generations a species may stagnate before pruning is allowed.

### DEFAULT_TARGET_SPECIES

Default species-count target that the threshold controller tries to maintain.

### FitnessSharingContext

Minimal context required to apply fitness sharing.

Fitness sharing normalizes per-genome fitness within each species to reduce
selection pressure toward dense clusters of very similar genomes.

The contract stays deliberately small: one current species registry and one
compatibility-distance reader. Sharing does not need assignment state,
history buffers, or threshold integrals, so those concerns stay outside this
context.

### HISTORY_BUFFER_MAX_ENTRIES

Maximum number of recent species-history rows retained in memory.

### InnovationAccumulator

Accumulator for innovation-id statistics across a set of connections.

Used for extended history telemetry (mean innovation, innovation range, and
enabled/disabled ratios).

Read this as the folded evidence bag for one species-history snapshot. The
history chapter gathers raw connection-level signals here first and only then
converts them into reader-friendly summary numbers.

### NEGATIVE_INFINITY

Shared negative infinity constant for score initialization.

### PENALTY_NO_EFFECT_THRESHOLD

Penalty cutoff where no reduction should occur.

### SHARING_MAX_CONTRIBUTION

Maximum sharing contribution per peer.

### SHARING_SELF_DISTANCE

Distance used when comparing a member with itself.

### SHARING_SUM_FLOOR

Fallback divisor when sharing sum is zero.

### SPECIES_AGE_GRACE_MULTIPLIER

Multiplier that turns the coarse grace setting into the runtime age threshold.

### StagnationContext

Minimal context required to update species stagnation.

Stagnation pruning removes species that have not improved their best score
within a configured number of generations.

This context is intentionally narrower than the full speciation harness
because stagnation only needs a live registry and a generation counter to
decide whether a species is still earning its place.
