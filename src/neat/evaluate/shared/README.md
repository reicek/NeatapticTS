# neat/evaluate/shared

Genome with score, novelty, and clearing capabilities.

This interface describes the minimal genome shape required by evaluation
helpers. It intentionally stays permissive for compatibility with legacy
genome variants while documenting the expected properties.

## neat/evaluate/shared/evaluate.types.ts

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

## neat/evaluate/shared/evaluate.constants.ts

Default number of nearest neighbors used when computing novelty.

Small values keep novelty sensitive to local behavioral differences without requiring a large
archive or population.

### AUTO_COEFF_ADJUST_DEFAULT

Default rate used when auto distance-coefficient tuning rebalances structural distance weights.

### AUTO_COEFF_MAX_DEFAULT

Maximum structural-distance coefficient allowed during automatic tuning.

### AUTO_COEFF_MIN_DEFAULT

Minimum structural-distance coefficient allowed during automatic tuning.

### COMPAT_MAX_THRESHOLD_DEFAULT

Maximum compatibility threshold allowed during automatic compatibility tuning.

### COMPAT_MIN_THRESHOLD_DEFAULT

Minimum compatibility threshold allowed during automatic compatibility tuning.

### COMPAT_THRESHOLD_DEFAULT

Baseline compatibility threshold used when no explicit value is configured.

### DISTANCE_COEFF_DEFAULT

Baseline structural-distance coefficient used before any automatic tuning occurs.

### ENTROPY_ADJUST_DEFAULT

Default rate used when compatibility tuning nudges the threshold upward or downward.

### ENTROPY_DEADBAND_DEFAULT

Deadband around the entropy target where compatibility tuning intentionally does nothing.

### ENTROPY_TARGET_DEFAULT

Target mean entropy used when tuning the compatibility threshold.

The goal is to keep speciation pressure near a stable diversity level instead of drifting toward
either species collapse or fragmentation.

### ENTROPY_VAR_ADJUST_DEFAULT

Default step size used when entropy-sharing tuning increases or decreases sharing sigma.

### ENTROPY_VAR_HIGH_BAND

Upper tolerance band for deciding that observed entropy variance is meaningfully high.

### ENTROPY_VAR_LOW_BAND

Lower tolerance band for deciding that observed entropy variance is meaningfully low.

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

Upper bound for the sharing sigma used by entropy-sharing adaptation.

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

Lower bound for the sharing sigma used by entropy-sharing adaptation.

### ENTROPY_VAR_TARGET_DEFAULT

Target variance used by entropy-sharing tuning.

The controller nudges sharing sigma toward a population whose entropy spread is neither too flat
nor too unstable.

### NOVELTY_ARCHIVE_CAP

Maximum number of descriptors retained in the novelty archive.

The cap keeps novelty history useful for exploration while preventing unbounded memory growth.

### NOVELTY_DEFAULT_BLEND

Default blend factor used when mixing novelty into an existing fitness score.

A mid-range value keeps novelty influential without letting exploratory behavior completely drown
out task performance.

### NOVELTY_DEFAULT_NEIGHBORS

Default number of nearest neighbors used when computing novelty.

Small values keep novelty sensitive to local behavioral differences without requiring a large
archive or population.

### VARIANCE_DECREASE_THRESHOLD

Multiplier below which observed variance is treated as a meaningful decrease.

### VARIANCE_INCREASE_THRESHOLD

Multiplier above which observed variance is treated as a meaningful increase.
