# neat

Shared numerical and heuristic constants reused across the NEAT controller.

This file intentionally stays flat at the `src/neat` root even after the
controller folderization work. Unlike the chaptered controller helpers, these
values are consumed both by NEAT internals and by non-NEAT architecture code,
so keeping one dependency-free constants surface avoids inventing a fake
chapter boundary just to move a few numbers around.

The constants here fall into two teaching-friendly groups:

- numerical safety values that keep logs, divisions, and normalization stable,
- mutation heuristics that communicate a default controller policy.

Example:

```ts
import { EPSILON, EXTRA_CONNECTION_PROBABILITY } from './neat/neat.constants';

const safeRatio = value / (total + EPSILON);
const shouldTryExtraConnection = rng() < EXTRA_CONNECTION_PROBABILITY;
```

## neat/neat.constants.ts

### EPSILON

Numerical stability offset used inside division and logarithmic expressions.

Use this when a denominator or logarithm input can drift toward zero during
fitness shaping, telemetry aggregation, or probability-style calculations.

### EXTRA_CONNECTION_PROBABILITY

Default probability of attempting one opportunistic extra add-connection mutation.

This is a heuristic rather than a numerical safety constant. It slightly
increases the chance that a genome gains new connectivity during mutation
without making extra-connection attempts mandatory on every pass.

### NORM_EPSILON

Variance-smoothing epsilon used by normalization-oriented helpers.

The value matches the larger scale commonly used in normalization math where
the goal is stable variance handling rather than near-exact probability work.

### PROB_EPSILON

Very small epsilon reserved for probability-loss style ratios and logs.

This is intentionally smaller than {@link EPSILON} because probability terms
often need protection without materially changing the magnitude of tiny
values.
