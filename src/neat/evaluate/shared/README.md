# neat/evaluate/shared

Shared contracts for the NEAT evaluate chapter.

This file is the common vocabulary layer beneath the evaluate subtree. The
surrounding chapters each own one narrow stage of the evaluation pipeline,
but they all need to agree on the same small set of contracts for genomes,
controller state, diversity evidence, novelty archive rows, and dynamic
objectives.

Read this chapter when you want to answer questions such as:
- Which genome shape is considered sufficient for evaluation helpers?
- What controller fields are shared across fitness, novelty, and tuning
  stages?
- Why do the shared contracts stay permissive instead of mirroring the whole
  runtime controller type exactly?
- Which pieces of evidence are expected to survive long enough for later
  tuning, selection, or multi-objective reads?

The exports below fall into five small families:
- genome evidence contracts,
- novelty archive memory,
- diversity-stat summaries,
- objective registration vocabulary,
- the controller host surface used across the evaluate subtree.

This boundary stays intentionally small so the evaluate helpers can share one
stable language without widening into a second full controller facade.

## neat/evaluate/shared/evaluate.types.ts

### DiversityStats

Diversity statistics tracked during evaluation.

The values are optional because different evaluation passes may compute only
a subset of metrics. Tuning chapters read this structure opportunistically,
which lets the evaluation pipeline add evidence incrementally instead of
requiring every metric to exist on every pass.

### GenomeForEvaluation

Genome with score, novelty, and clearing capabilities.

This interface describes the smallest practical genome shape that the
evaluate subtree can reason about. It is intentionally permissive so legacy
genome variants and downstream extensions can still participate in
evaluation, while the documented fields mark the pieces of evidence that the
shared helpers actually rely on.

### NeatControllerForEval

NEAT controller interface for evaluation.

This interface models the subset of a NEAT controller used by the evaluation
helpers. It includes the runtime options, population data, lightweight
evidence caches, and optional hooks that the evaluate subtree needs in order
to enrich one generation without taking ownership of the whole controller.

The contract is intentionally broader than an individual helper needs but
still much smaller than the full `Neat` surface. That tradeoff keeps the
evaluate chapters interoperable while preserving a clear boundary between
evaluation and the rest of the runtime.

### NoveltyArchiveEntry

Novelty archive entry with descriptor and novelty score.

Entries store one behavior descriptor alongside its novelty score so later
evaluation passes can keep some memory of previously unusual behavior without
retaining the entire population history.

### ObjectiveDef

Objective definition for multi-objective optimization.

Objectives are registered dynamically so evaluation can surface additional
evidence, such as entropy, without forcing the entire objective stack to be
hard-coded at controller construction time.

## neat/evaluate/shared/evaluate.constants.ts

Shared default constants for the NEAT evaluate chapter.

These constants anchor the small policy loops used across evaluation. The
root evaluate chapter explains the full scoring-and-adaptation flow, while
the helper chapters consume these values as stable defaults for novelty,
entropy-sharing, entropy-compatibility, and auto-distance tuning.

Read this chapter when you want to answer questions such as:
- Which defaults shape novelty exploration before the user configures
  anything?
- What target values and bands anchor the entropy-based tuning loops?
- Which bounds keep compatibility and distance tuning from drifting too far?
- How does the evaluate subtree keep its small policy helpers aligned with
  one another?

The exports below group into four families:
- novelty defaults,
- entropy-sharing defaults,
- entropy-compatibility defaults,
- auto-distance and variance-baseline defaults.

### AUTO_COEFF_ADJUST_DEFAULT

Default rate used when auto distance-coefficient tuning rebalances structural distance weights.

This shared step size controls how quickly excess and disjoint coefficients react when topology
variance drifts away from the recent baseline.

### AUTO_COEFF_MAX_DEFAULT

Maximum structural-distance coefficient allowed during automatic tuning.

The upper bound prevents excess and disjoint penalties from dominating every later compatibility
comparison.

### AUTO_COEFF_MIN_DEFAULT

Minimum structural-distance coefficient allowed during automatic tuning.

The lower bound keeps structural differences meaningful even when the auto-distance policy is
softening species pressure.

### COMPAT_MAX_THRESHOLD_DEFAULT

Maximum compatibility threshold allowed during automatic compatibility tuning.

The upper clamp prevents compatibility from becoming so permissive that species boundaries lose
practical meaning.

### COMPAT_MIN_THRESHOLD_DEFAULT

Minimum compatibility threshold allowed during automatic compatibility tuning.

The lower clamp prevents the threshold from collapsing until even small structural differences
force unnecessary species fragmentation.

### COMPAT_THRESHOLD_DEFAULT

Baseline compatibility threshold used when no explicit value is configured.

This serves as the neutral starting point before entropy-based tuning or explicit user policy
begins to reshape species pressure.

### DISTANCE_COEFF_DEFAULT

Baseline structural-distance coefficient used before automatic tuning occurs.

This is the neutral starting point for the structural-distance fold before variance-driven
updates begin to reshape it.

### ENTROPY_ADJUST_DEFAULT

Default rate used when compatibility tuning nudges the threshold upward or downward.

A conservative step size keeps the compatibility-threshold loop gradual enough that later
speciation reads remain interpretable from one generation to the next.

### ENTROPY_DEADBAND_DEFAULT

Deadband around the entropy target where compatibility tuning intentionally does nothing.

The deadband reduces threshold jitter by treating small entropy deviations as normal noise
rather than signals that demand a policy change.

### ENTROPY_TARGET_DEFAULT

Target mean entropy used when tuning the compatibility threshold.

The goal is to keep speciation pressure near a stable diversity level instead of drifting toward
either species collapse or fragmentation.

### ENTROPY_VAR_ADJUST_DEFAULT

Default step size used when entropy-sharing tuning increases or decreases sharing sigma.

A modest default keeps the sigma controller responsive without letting one noisy variance read
swing the sharing radius too aggressively.

### ENTROPY_VAR_HIGH_BAND

Upper tolerance band for deciding that observed entropy variance is meaningfully high.

Values above this multiplier mark a population whose entropy spread is noisier than the tuning
target expects.

### ENTROPY_VAR_LOW_BAND

Lower tolerance band for deciding that observed entropy variance is meaningfully low.

Values below this multiplier mark a population whose entropy spread is flatter than the tuning
target expects.

### ENTROPY_VAR_MAX_SIGMA_DEFAULT

Upper bound for the sharing sigma used by entropy-sharing adaptation.

The upper clamp prevents the radius from widening until entropy sharing loses practical
contrast across the population.

### ENTROPY_VAR_MIN_SIGMA_DEFAULT

Lower bound for the sharing sigma used by entropy-sharing adaptation.

The lower clamp prevents the sharing radius from shrinking so far that later sharing pressure
becomes hypersensitive to tiny entropy fluctuations.

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

Drops below this band tell the auto-distance loop that topology sizes are converging relative to
the recent baseline.

### VARIANCE_INCREASE_THRESHOLD

Multiplier above which observed variance is treated as a meaningful increase.

Values above this band tell the auto-distance loop that topology sizes are spreading relative to
the recent baseline.
