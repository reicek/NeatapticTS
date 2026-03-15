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

Default neighbor count for novelty calculation.

### AUTO_COEFF_ADJUST_DEFAULT

Default adjustment rate for auto distance coefficient tuning.

### AUTO_COEFF_MAX_DEFAULT

Default maximum coefficient for auto distance coefficient tuning.

### AUTO_COEFF_MIN_DEFAULT

Default minimum coefficient for auto distance coefficient tuning.

### COMPAT_MAX_THRESHOLD_DEFAULT

Default maximum compatibility threshold.

### COMPAT_MIN_THRESHOLD_DEFAULT

Default minimum compatibility threshold.

### COMPAT_THRESHOLD_DEFAULT

Default compatibility threshold when not provided.

### DISTANCE_COEFF_DEFAULT

Default coefficient value when not provided.

### ENTROPY_ADJUST_DEFAULT

Default adjustment rate for compatibility tuning.

### ENTROPY_DEADBAND_DEFAULT

Default deadband for compatibility tuning.

### ENTROPY_TARGET_DEFAULT

Default target entropy for compatibility tuning.

### ENTROPY_VAR_ADJUST_DEFAULT

Default adjustment rate for entropy sharing.

### ENTROPY_VAR_HIGH_BAND

Upper band multiplier for entropy variance tuning.

### ENTROPY_VAR_LOW_BAND

Lower band multiplier for entropy variance tuning.

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

Default maximum sigma for entropy sharing.

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

Default minimum sigma for entropy sharing.

### ENTROPY_VAR_TARGET_DEFAULT

Default target variance for entropy sharing.

### NOVELTY_ARCHIVE_CAP

Maximum number of entries stored in the novelty archive.

### NOVELTY_DEFAULT_BLEND

Default blend factor for novelty vs. fitness.

### NOVELTY_DEFAULT_NEIGHBORS

Default neighbor count for novelty calculation.

### VARIANCE_DECREASE_THRESHOLD

Variance decrease threshold multiplier.

### VARIANCE_INCREASE_THRESHOLD

Variance increase threshold multiplier.
