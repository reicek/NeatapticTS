# neat/adaptive/acceptance

Acceptance-gating heuristics for adaptive NEAT.

This category focuses on the minimal-criterion threshold that decides when a
genome is good enough to stay in play, making it easier to study selection
pressure separately from topology growth or mutation schedules.

## neat/adaptive/acceptance/adaptive.acceptance.ts

### applyMinimalCriterionAdaptive

```ts
applyMinimalCriterionAdaptive(): void
```

Apply adaptive minimal criterion (MC) acceptance.

This method maintains an MC threshold used to decide whether an
individual genome is considered acceptable. It adapts the threshold
based on the proportion of the population that meets the current
threshold, trying to converge to a target acceptance rate.

Behavior summary:
- Initializes `_mcThreshold` from configuration if undefined.
- Computes the proportion of genomes with score >= threshold.
- Adjusts threshold multiplicatively by `adjustRate` to move the
  observed proportion towards `targetAcceptance`.
- Sets `g.score = 0` for genomes that fall below the final threshold
  — effectively rejecting them from selection.

Example:

// Example config snippet used by the engine
// options.minimalCriterionAdaptive = { enabled: true, initialThreshold: 0.1, targetAcceptance: 0.5, adjustRate: 0.1 }
engine.applyMinimalCriterionAdaptive();

### applyRejection

```ts
applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void
```

Zero scores below the final threshold.

Parameters:
- `engine` - - NEAT engine instance.
- `threshold` - - Final MC threshold.

### collectScores

```ts
collectScores(
  engine: NeatLikeWithAdaptive,
): number[]
```

Collect population scores into a snapshot array.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Array of scores (missing scores treated as 0).

### computeAcceptance

```ts
computeAcceptance(
  scores: number[],
  threshold: number,
): number
```

Compute acceptance metrics for the current threshold.

Parameters:
- `scores` - - Population score snapshot.
- `threshold` - - Current MC threshold.

Returns: Acceptance proportion.

### initializeThreshold

```ts
initializeThreshold(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): void
```

Initialize MC threshold if missing.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Minimal-criterion adaptive configuration.

### resolveTargetSettings

```ts
resolveTargetSettings(
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): { targetAcceptance: number; adjustRate: number; }
```

Resolve target acceptance and adjust rate settings.

Parameters:
- `config` - - Minimal-criterion adaptive configuration.

Returns: Target settings.

### updateThreshold

```ts
updateThreshold(
  engine: NeatLikeWithAdaptive,
  acceptance: number,
  tuning: { targetAcceptance: number; adjustRate: number; },
): void
```

Update the MC threshold based on acceptance proportion.

Parameters:
- `engine` - - NEAT engine instance.
- `acceptance` - - Observed acceptance proportion.
- `tuning` - - Target acceptance and adjustment settings.

## neat/adaptive/acceptance/adaptive.minimal-criterion.utils.ts

### applyRejection

```ts
applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void
```

Zero scores below the final threshold.

Parameters:
- `engine` - - NEAT engine instance.
- `threshold` - - Final MC threshold.

### collectScores

```ts
collectScores(
  engine: NeatLikeWithAdaptive,
): number[]
```

Collect population scores into a snapshot array.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Array of scores (missing scores treated as 0).

### computeAcceptance

```ts
computeAcceptance(
  scores: number[],
  threshold: number,
): number
```

Compute acceptance metrics for the current threshold.

Parameters:
- `scores` - - Population score snapshot.
- `threshold` - - Current MC threshold.

Returns: Acceptance proportion.

### initializeThreshold

```ts
initializeThreshold(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): void
```

Initialize MC threshold if missing.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Minimal-criterion adaptive configuration.

### resolveTargetSettings

```ts
resolveTargetSettings(
  config: { enabled?: boolean | undefined; initialThreshold?: number | undefined; targetAcceptance?: number | undefined; adjustRate?: number | undefined; },
): { targetAcceptance: number; adjustRate: number; }
```

Resolve target acceptance and adjust rate settings.

Parameters:
- `config` - - Minimal-criterion adaptive configuration.

Returns: Target settings.

### updateThreshold

```ts
updateThreshold(
  engine: NeatLikeWithAdaptive,
  acceptance: number,
  tuning: { targetAcceptance: number; adjustRate: number; },
): void
```

Update the MC threshold based on acceptance proportion.

Parameters:
- `engine` - - NEAT engine instance.
- `acceptance` - - Observed acceptance proportion.
- `tuning` - - Target acceptance and adjustment settings.
