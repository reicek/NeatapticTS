# neat/adaptive/complexity

Complexity-control heuristics for adaptive NEAT.

This category explains how the engine decides when to grow, pause, or shrink
network structure so learners can study budget scheduling separately from
mutation or lineage pressure.

## neat/adaptive/complexity/adaptive.complexity.ts

### applyAdaptiveSchedule

```ts
applyAdaptiveSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply adaptive complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance with adaptive state.
- `config` - - Complexity budget configuration.

### applyComplexityBudgetSchedule

```ts
applyComplexityBudgetSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply the complexity budget schedule for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyLinearSchedule

```ts
applyLinearSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply linear complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializePhaseState

```ts
initializePhaseState(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Ensure phase state is initialized.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### togglePhaseIfNeeded

```ts
togglePhaseIfNeeded(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Toggle phase if the current phase has exceeded its length.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

## neat/adaptive/complexity/adaptive.phases.utils.ts

### initializePhaseState

```ts
initializePhaseState(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Ensure phase state is initialized.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

### resolveNextPhase

```ts
resolveNextPhase(
  currentPhase: string,
): string
```

Resolve next phase name.

Parameters:
- `currentPhase` - - Current phase label.

Returns: Next phase label.

### togglePhaseIfNeeded

```ts
togglePhaseIfNeeded(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; phases?: { generation: number; maxNodes?: number | undefined; maxConns?: number | undefined; }[] | undefined; phaseLength?: number | undefined; initialPhase?: string | undefined; },
): void
```

Toggle phase if the current phase has exceeded its length.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Phased complexity configuration.

## neat/adaptive/complexity/adaptive.complexity.utils.ts

### adjustConnectionBudget

```ts
adjustConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  factors: { increaseFactor: number; stagnationFactor: number; },
  noveltyFactor: number,
  history: number[],
): void
```

Adjust connection budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### adjustNodeBudget

```ts
adjustNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  factors: { increaseFactor: number; stagnationFactor: number; },
  noveltyFactor: number,
  history: number[],
): void
```

Adjust node budget based on trends and factors.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `factors` - - Adjustment factors.
- `noveltyFactor` - - Novelty multiplier.
- `history` - - Rolling history for window checks.

### applyAdaptiveSchedule

```ts
applyAdaptiveSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply adaptive complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance with adaptive state.
- `config` - - Complexity budget configuration.

### applyComplexityBudgetSchedule

```ts
applyComplexityBudgetSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply the complexity budget schedule for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### applyLinearSchedule

```ts
applyLinearSchedule(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Apply linear complexity budget scheduling.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### clampNodeBudget

```ts
clampNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Clamp node budget to configured minimum.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### computeAdjustmentFactors

```ts
computeAdjustmentFactors(
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
  trends: { improvement: number; slope: number; },
  history: number[],
): { increaseFactor: number; stagnationFactor: number; }
```

Compute adjustment factors for budget growth and decay.

Parameters:
- `config` - - Complexity budget configuration.
- `trends` - - Improvement and slope metrics.
- `history` - - Rolling history of best scores.

Returns: Adjustment factors (increase and stagnation multipliers).

### computeNoveltyFactor

```ts
computeNoveltyFactor(
  engine: NeatLikeWithAdaptive,
): number
```

Compute novelty factor based on archive size.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Novelty multiplier (0.9 if archive small, 1.0 otherwise).

### computeSlope

```ts
computeSlope(
  history: number[],
): number
```

Compute linear regression slope using ordinary least squares.

Parameters:
- `history` - - Rolling history of best scores.

Returns: OLS slope estimate.

### computeTrends

```ts
computeTrends(
  history: number[],
): { improvement: number; slope: number; }
```

Compute improvement and slope trends from score history.

Parameters:
- `history` - - Rolling history of best scores.

Returns: Trend metrics (improvement and slope).

### initializeConnectionBudget

```ts
initializeConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Initialize connection budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### initializeNodeBudget

```ts
initializeNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): void
```

Initialize node budget if undefined.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

### normalizeSlope

```ts
normalizeSlope(
  slope: number,
  initialScore: number,
): number
```

Normalize slope magnitude relative to initial score.

Parameters:
- `slope` - - Raw OLS slope.
- `initialScore` - - First score in history window.

Returns: Normalized slope clamped to [-2, 2].

### updateScoreHistory

```ts
updateScoreHistory(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; mode?: string | undefined; improvementWindow?: number | undefined; increaseFactor?: number | undefined; stagnationFactor?: number | undefined; maxNodesStart?: number | undefined; maxNodesEnd?: number | undefined; minNodes?: number | undefined; maxConnsStart?: number | undefined; maxConnsEnd?: number | undefined; horizon?: number | undefined; },
): number[]
```

Update rolling score history with current best score.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Complexity budget configuration.

Returns: Rolling history array after update.
