# DF16 Racing Curriculum: Fixing stabilization with a positive driving-quality scorer

## Question

[`racing-curriculum-df15-stabilization-scoring-mismatch.md`](./racing-curriculum-df15-stabilization-scoring-mismatch.md) showed that the racing stabilization phase scored weight variants with negative mean-squared-error (MSE) against a scalar target while the baseline was a positive driving-quality score. The commit inequality `bestVariantScore > baselineScore + threshold` could never be satisfied, so every stabilization evaluation returned `committed=false` with `reason="no_weight_mutations"`. How should the score space be aligned so that weight-tuning commits can actually happen?

## Evidence

### 1. Round 1 — aligning the target only

The first attempt preserved the default negative-MSE `VariantScorer` and changed only the stabilization `trainingTarget` so it was still a scalar but better matched the controller's first output dimension. The result was the same score-space mismatch: `baselineScore` stayed positive while `bestVariantScore` stayed negative.

The commit inequality remained unsatisfiable because the units and direction of the two scores were still unrelated.

### 2. Round 2 — injecting a racing-specific `VariantScorer`

The working fix replaced the default scorer with a racing-specific `VariantScorer` that mirrors the positive driving-quality baseline computation used by `evaluateRacingTrendScore`:

```ts
const RACING_VARIANT_SCORER: VariantScorer = (outputs, _target) => {
  // Collapse [throttle, steering] to a scalar, like growth-phase scoring.
  const scalarWindow = outputs.map((outputVector) =>
    reduceOutputToScalar(outputVector),
  );

  // Positive trend/mean/complexity score in the same space as the baseline.
  const scoreTrend = scalarWindow.at(-1)! - scalarWindow[0]!;
  const scoreMean =
    scalarWindow.reduce((accumulated, value) => accumulated + value, 0) /
    scalarWindow.length;
  const behavioralComplexity = resolveBehavioralComplexity(outputs);
  const complexityBonus =
    scoreTrend >= 0 ? behavioralComplexity * RACING_COMPLEXITY_WEIGHT : 0;

  return scoreMean + scoreTrend * 0.5 + complexityBonus;
};
```

The `target` argument is intentionally ignored; the score is derived from the candidate network's own outputs, which is exactly what the live baseline also measures.

The scorer is injected through the `scoreFn` option of `runNgeGrowStabilizeCycle` in `runtime.adaptation.ts`:

```ts
const cycleResult = await runNgeGrowStabilizeCycle({
  // ...
  baselineScore: currentBaseline,
  scoreFn: RACING_VARIANT_SCORER,
  // ...
});
```

Because both the baseline and every variant now live in the same positive driving-quality space, the commit inequality became meaningful.

### 3. Supporting wiring that stayed in place

- The `evaluateNgeWeightVariants` path evaluates patches **sequentially** on the CPU with `useGPU: false`, restoring weights after each patch, so a candidate score truly reflects that patch's weights.
- `runtime.adaptation.ts` caps the stabilization variant count to `NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT` (32) via `stageVariantCounts`, giving the stabilization phase a focused local search budget.
- Growth commits still use the standard NGE lifecycle path; only the stabilization phase needed the task-specific scorer.

## Decision

Align the stabilization score space by injecting a racing-specific `VariantScorer` rather than by redefining the baseline. This keeps the baseline semantics identical to the driving-quality objective the controller already maximizes, and it makes every variant score comparable to that baseline.

The scorer must satisfy three invariants:

1. **Same direction as the baseline**: higher score means better driving quality.
2. **Same units as the baseline**: both are derived from controller outputs, not from an external target vector.
3. **Output-derived, not target-derived**: the `target` argument of the `VariantScorer` contract is ignored because the score is a property of the candidate network's own behavior.

## Implementation

The production change is in `examples/racing_curriculum/controller/runtime.adaptation.ts`:

- `reduceOutputToScalar` collapses the 2-D controller output to a mean-of-absolute-values scalar.
- `RACING_VARIANT_SCORER` combines mean, half-trend, and a behavioral-complexity bonus to match `evaluateRacingTrendScore`.
- `createRuntimeAdaptationEngine` passes `RACING_VARIANT_SCORER` as `scoreFn` whenever it invokes `runNgeGrowStabilizeCycle` for stabilization.

Core documentation and warnings were also added to the grow-stabilize plumbing so future callers do not accidentally mix score spaces:

- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts` now documents the score-space alignment requirement on `runNgeGrowStabilizeCycle`.
- `src/neat/nge-juvenile/neat.nge-juvenile.types.ts` and `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts` warn that a custom `scoreFn` must share the same score space and direction as `baselineScore`.

## DF16.1 cleanup

After the scorer fix landed, temporary diagnostic-logging symbols were removed:

- `DIAGNOSTIC_LOGGING`, `formatNgeDiagnosticLine`, and `diagnosticLogBuffer` were deleted from `runtime.adaptation.ts`.
- The stale P8S22 test that asserted the removed logging behavior was updated to reflect the new telemetry contract.
- Focused validation passed: 4 racing-adaptation suites (85 tests) and 2 NGE juvenile suites (153 tests).

## Metrics

| Variant                                | Stabilization commits | Time (s) | Ticks/s | Nodes | Connections |
| -------------------------------------- | --------------------: | -------: | ------: | ----: | ----------: |
| DF15 (negative-MSE scorer)             |                     0 |     ~198 |   ~51.6 |  N268 |        C774 |
| DF16 (positive driving-quality scorer) |                   268 |   ~111.8 |   ~91.4 |  N277 |        C824 |

The same demo run now commits 268 stabilization weight updates, demonstrating that the stabilization phase is no longer dead.

## Sources

- `examples/racing_curriculum/controller/runtime.adaptation.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.types.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`
- `src/acceleration/acceleration.variants.ts`

## References

- K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," _Evolutionary Computation_, vol. 10, no. 2, pp. 99-127, 2002. [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
- [Wikipedia — Exploration–exploitation dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma)
- [Wikipedia — Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
- [Wikipedia — Hysteresis](https://en.wikipedia.org/wiki/Hysteresis)
- [Wikipedia — Variance](https://en.wikipedia.org/wiki/Variance)
