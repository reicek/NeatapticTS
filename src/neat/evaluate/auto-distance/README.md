# neat/evaluate/auto-distance

Auto-distance tuning helpers for the NEAT evaluate chapter.

This chapter owns the variance-driven adjustment of compatibility-distance
coefficients. It watches connection-count variance across the evaluated
population and nudges the controller's excess and disjoint coefficients up or
down to keep structural diversity from collapsing.

## neat/evaluate/auto-distance/evaluate.auto-distance.ts

### applyAutoDistanceCoefficientTuning

```ts
applyAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  autoDistanceCoeffOptions: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; },
  connectionVariance: number,
): void
```

Apply the coefficient-tuning policy using the observed connection variance.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `autoDistanceCoeffOptions` - - Tuning options.
- `connectionVariance` - - Variance of connection counts.

### applyDistanceCoefficientDecrease

```ts
applyDistanceCoefficientDecrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number; },
  adjustRate: number,
): void
```

Decrease distance coefficients within the configured bounds.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `bounds` - - Min and max coefficient bounds.
- `adjustRate` - - Adjustment rate.

### applyDistanceCoefficientIncrease

```ts
applyDistanceCoefficientIncrease(
  controller: NeatControllerForEval,
  bounds: { minCoeff: number; maxCoeff: number; },
  adjustRate: number,
): void
```

Increase distance coefficients within the configured bounds.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `bounds` - - Min and max coefficient bounds.
- `adjustRate` - - Adjustment rate.

### computeMean

```ts
computeMean(
  values: number[],
): number
```

Compute the arithmetic mean of a number list.

Parameters:
- `values` - - Input values.

Returns: Mean of the values.

### computeVariance

```ts
computeVariance(
  values: number[],
  meanValue: number,
): number
```

Compute the variance of a number list from a precomputed mean.

Parameters:
- `values` - - Input values.
- `meanValue` - - Precomputed mean.

Returns: Variance of the values.

### getDistanceCoefficientBounds

```ts
getDistanceCoefficientBounds(
  autoDistanceCoeffOptions: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; },
): { minCoeff: number; maxCoeff: number; }
```

Resolve the minimum and maximum coefficient bounds for tuning.

Parameters:
- `autoDistanceCoeffOptions` - - Tuning options.

Returns: Min and max coefficient bounds.

### initializeConnectionVarianceBootstrap

```ts
initializeConnectionVarianceBootstrap(
  controller: NeatControllerForEval,
  connectionVariance: number,
  bounds: { minCoeff: number; maxCoeff: number; },
  adjustRate: number,
): void
```

Bootstrap connection-variance tuning the first time the policy runs.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `connectionVariance` - - Current connection variance.
- `bounds` - - Min and max coefficient bounds.
- `adjustRate` - - Adjustment rate.

### runAutoDistanceCoefficientTuning

```ts
runAutoDistanceCoefficientTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Apply variance-driven tuning to the controller's distance coefficients.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.
