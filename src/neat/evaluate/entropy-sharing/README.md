# neat/evaluate/entropy-sharing

Entropy-sharing tuning helpers for the NEAT evaluate chapter.

This chapter owns the small post-evaluation adjustment that nudges
`sharingSigma` based on the observed variance of structural entropy across
the current population.

## neat/evaluate/entropy-sharing/evaluate.entropy-sharing.ts

### computeNextSharingSigma

```ts
computeNextSharingSigma(
  entropySharingOptions: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; },
  currentVarEntropy: number,
  currentSigma: number,
): number
```

Compute the next sharing sigma value from the observed entropy variance.

Parameters:
- `entropySharingOptions` - - Tuning options.
- `currentVarEntropy` - - Current variance of entropy.
- `currentSigma` - - Current sigma value.

Returns: Next sharing sigma value.

### ensureDiversityStatsContainer

```ts
ensureDiversityStatsContainer(
  controller: NeatControllerForEval,
): void
```

Ensure diversity statistics storage exists before tuning writes into it.

Parameters:
- `controller` - - NEAT controller instance for evaluation.

### runEntropySharingTuning

```ts
runEntropySharingTuning(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Adjust the entropy-sharing sigma when entropy sharing tuning is enabled.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.
