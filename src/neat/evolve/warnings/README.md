# neat/evolve/warnings

The evolve-time warnings boundary centralizes the small amount of best-effort
failure signaling that should survive even when the rest of evolution keeps
running.

Most evolve bookkeeping lives in `runtime/`, `telemetry/`, or the main
orchestration chapter. This file stays separate on purpose because warning
emission is not part of generation construction or ranking logic. It only
explains the narrow final edge case where evolve finishes without a usable
champion snapshot and wants to surface that fact without turning logging into
a broader error-handling subsystem.

Read this chapter when you want to understand:

- what "no valid best genome" means in practice,
- why the controller emits a warning instead of throwing,
- how warning emission remains best-effort in restricted runtimes.

## neat/evolve/warnings/evolve.warnings.utils.ts

### EVOLVE_NO_BEST_GENOME_WARNING

Warning emitted when evolution finishes without a best genome.

This message signals that the generation loop completed without producing a
champion snapshot that the controller considers safe to return. That usually
points to an upstream evaluation or population-state problem rather than to a
normal low-fitness generation, so the wording intentionally distinguishes
"no valid best genome" from merely "no improvement."

### warnIfNoBestGenome

```ts
warnIfNoBestGenome(): void
```

Emit the standard warning for runs that end without a valid best genome.

The helper is deliberately conservative: it tries to warn, but it does not
let missing or restricted console support destabilize the evolve loop further.
That keeps warning emission informative for normal runtimes while preserving a
best-effort contract for test harnesses and embedded environments.
