# neat/speciation/shared

Shared vocabulary and defaults for NEAT speciation.

This chapter defines the common runtime contracts used across assignment,
threshold control, history capture, and fitness sharing so each speciation
category can stay focused on one responsibility.

## neat/speciation/shared/speciation.shared.ts

### CompatAdjust

Resolved compatibility-threshold adjustment settings.

This is the non-nullable form of {@link SpeciationOptions.compatAdjust} used
by the speciation PID controller.

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

### SHARING_MAX_CONTRIBUTION

Maximum sharing contribution per peer.

### SHARING_SELF_DISTANCE

Distance used when comparing a member with itself.

### SHARING_SUM_FLOOR

Fallback divisor when sharing sum is zero.

### SPECIES_AGE_GRACE_MULTIPLIER

Multiplier used to convert grace generations to age threshold.

### StagnationContext

Minimal context required to update species stagnation.

Stagnation pruning removes species that have not improved their best score
within a configured number of generations.
