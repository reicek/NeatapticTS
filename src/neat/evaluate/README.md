# neat/evaluate

Root orchestration for NEAT population evaluation.

This chapter keeps the evaluation pipeline readable at the top level: run
fitness, optionally blend novelty into scores, then apply the small adaptive
tuning passes that depend on freshly computed population statistics.

The neighboring chapters own the narrower mechanics:
- `fitness/` drives per-genome or population-level scoring.
- `novelty/` computes behavioral novelty, blending, and archive writes.
- `entropy-sharing/`, `entropy-compat/`, and `auto-distance/` tune the
  controller's diversity-related parameters.
- `speciation/` keeps lightweight post-evaluation species maintenance small.
- `objectives/` handles opt-in automatic entropy-objective registration.

## neat/evaluate/evaluate.ts

### evaluate

```ts
evaluate(): Promise<void>
```

Evaluate the population or population-wide fitness delegate.

This preserves the long-standing `Neat.evaluate()` flow while moving the
implementation into a chaptered boundary that is easier to discover in the
generated docs.

Top-level responsibilities:
1. Run fitness either on each genome or once for the whole population.
2. Blend novelty into scores when novelty search is enabled.
3. Ensure diversity statistics storage exists before adaptive tuning.
4. Apply entropy-sharing, entropy-compatibility, speciation, and automatic
   distance-coefficient adjustments.
5. Register the entropy objective when multi-objective mode asks for it.

Returns: Promise that resolves after evaluation and adaptive follow-up steps.

Example:

```ts
await evaluate.call(controller);
```

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

### NeatControllerForEval

NEAT controller interface for evaluation.

This interface models the subset of a NEAT controller used by the evaluation
helpers. It includes options, population data, and optional adaptive tuning
hooks.

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
