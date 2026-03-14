# neat/speciation/threshold

Compatibility-threshold tuning mechanics for speciation.

This chapter isolates the PID-like controller that keeps the species count
near a target. It is useful when you want to study threshold adaptation
without also reading the assignment or history code.

## neat/speciation/threshold/speciation.threshold.utils.ts

### adjustCompatibilityThreshold

```ts
adjustCompatibilityThreshold(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: { smoothingWindow?: number | undefined; decay?: number | undefined; kp?: number | undefined; ki?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; },
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void
```

Update the adaptive compatibility threshold and clamp to bounds.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `compatAdjust` - - Compatibility adjustment settings.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Nothing.

### clampCompatibilityThreshold

```ts
clampCompatibilityThreshold(
  options: SpeciationOptions,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): void
```

Clamp the compatibility threshold to configured bounds.

Parameters:
- `options` - - Speciation options.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Nothing.

### computePidThreshold

```ts
computePidThreshold(
  speciationContext: SpeciationHarnessContext<TOptions>,
  options: TOptions,
  compatAdjust: { smoothingWindow?: number | undefined; decay?: number | undefined; kp?: number | undefined; ki?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; },
  currentThreshold: number,
  minCompatibilityThreshold: number,
  maxCompatibilityThreshold: number,
): number
```

Compute a PID-based threshold update and clamp when needed.

Parameters:
- `speciationContext` - - Speciation harness context.
- `options` - - Speciation options.
- `compatAdjust` - - Compatibility adjustment settings.
- `currentThreshold` - - Current compatibility threshold.
- `minCompatibilityThreshold` - - Lower clamp bound.
- `maxCompatibilityThreshold` - - Upper clamp bound.

Returns: Updated threshold.
