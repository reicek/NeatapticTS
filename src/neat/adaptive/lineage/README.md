# neat/adaptive/lineage

Lineage-diversity adaptive controllers.

This category explains how ancestor uniqueness telemetry feeds back into the
search so the population can recover when family trees become too uniform.

## neat/adaptive/lineage/adaptive.lineage.ts

### applyAncestorUniqAdaptive

```ts
applyAncestorUniqAdaptive(): void
```

Adaptive adjustments based on ancestor uniqueness telemetry.

This helper inspects the most recent telemetry lineage block (if
available) for an `ancestorUniq` metric indicating how unique
ancestry is across the population. If ancestry uniqueness drifts
outside configured thresholds, the method will adjust either the
multi-objective dominance epsilon (if `mode === 'epsilon'`) or the
lineage pressure strength (if `mode === 'lineagePressure'`).

Typical usage: keep population lineage diversity within a healthy
band. Low ancestor uniqueness means too many genomes share ancestors
(risking premature convergence); high uniqueness might indicate
excessive divergence.

Example:

// Adjusts `options.multiObjective.dominanceEpsilon` when configured
engine.applyAncestorUniqAdaptive();

### applyUniquenessAdjustment

```ts
applyUniquenessAdjustment(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply an adjustment for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### extractAncestorUniqueness

```ts
extractAncestorUniqueness(
  engine: NeatLikeWithAdaptive,
): number | undefined
```

Extract the latest ancestor-uniqueness metric from telemetry.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Ancestor uniqueness value or undefined when missing.

### isCooldownSatisfied

```ts
isCooldownSatisfied(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): boolean
```

Determine whether the cooldown window has elapsed.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: True when adjustment is allowed.

### resolveUniquenessThresholds

```ts
resolveUniquenessThresholds(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): { lowThreshold: number; highThreshold: number; }
```

Resolve thresholds for ancestor-uniqueness decisions.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Threshold bounds.

## neat/adaptive/lineage/adaptive.ancestor-uniqueness.utils.ts

### applyEpsilonAdjustment

```ts
applyEpsilonAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply dominance-epsilon adjustments when configured.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### applyLineagePressureAdjustment

```ts
applyLineagePressureAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
): void
```

Apply lineage pressure strength adjustments.

Parameters:
- `engine` - - NEAT engine instance.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.

### applyUniquenessAdjustment

```ts
applyUniquenessAdjustment(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number; },
  adjustMagnitude: number,
): void
```

Apply an adjustment for the configured mode.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.
- `ancestorUniq` - - Current ancestor uniqueness metric.
- `thresholds` - - Threshold bounds for decisions.
- `adjustMagnitude` - - Adjustment magnitude.

### ensureLineagePressureState

```ts
ensureLineagePressureState(
  engine: NeatLikeWithAdaptive,
): { enabled?: boolean | undefined; mode?: string | undefined; strength?: number | undefined; }
```

Ensure lineage pressure state is available.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Lineage pressure configuration object.

### extractAncestorUniqueness

```ts
extractAncestorUniqueness(
  engine: NeatLikeWithAdaptive,
): number | undefined
```

Extract the latest ancestor-uniqueness metric from telemetry.

Parameters:
- `engine` - - NEAT engine instance.

Returns: Ancestor uniqueness value or undefined when missing.

### isCooldownSatisfied

```ts
isCooldownSatisfied(
  engine: NeatLikeWithAdaptive,
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): boolean
```

Determine whether the cooldown window has elapsed.

Parameters:
- `engine` - - NEAT engine instance.
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: True when adjustment is allowed.

### recordAdjustment

```ts
recordAdjustment(
  engine: NeatLikeWithAdaptive,
): void
```

Record the generation when an adjustment is applied.

Parameters:
- `engine` - - NEAT engine instance.

### resolveAdjustmentMagnitude

```ts
resolveAdjustmentMagnitude(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): number
```

Resolve adjustment magnitude for nudging controlled parameters.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Adjustment magnitude.

### resolveUniquenessThresholds

```ts
resolveUniquenessThresholds(
  config: { enabled?: boolean | undefined; cooldown?: number | undefined; lowThreshold?: number | undefined; highThreshold?: number | undefined; adjust?: number | undefined; mode?: string | undefined; },
): { lowThreshold: number; highThreshold: number; }
```

Resolve thresholds for ancestor-uniqueness decisions.

Parameters:
- `config` - - Ancestor uniqueness adaptive configuration.

Returns: Threshold bounds.
