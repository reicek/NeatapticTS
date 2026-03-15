# neat/evaluate/entropy-compat

Entropy-compatibility tuning helpers for the NEAT evaluate chapter.

This chapter owns the adaptive compatibility-threshold adjustment that keeps
the controller's speciation threshold moving with the observed mean entropy
of the evaluated population.

## neat/evaluate/entropy-compat/evaluate.entropy-compat.ts

### computeNextCompatibilityThreshold

```ts
computeNextCompatibilityThreshold(
  entropyCompatOptions: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; },
  meanEntropy: number,
  currentThreshold: number,
): number
```

Compute the next compatibility threshold from the observed mean entropy.

Parameters:
- `entropyCompatOptions` - - Tuning options.
- `meanEntropy` - - Current mean entropy.
- `currentThreshold` - - Current compatibility threshold.

Returns: Next compatibility threshold.

### runEntropyCompatibilityTuning

```ts
runEntropyCompatibilityTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Adjust the compatibility threshold when entropy-compatibility tuning is enabled.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.
