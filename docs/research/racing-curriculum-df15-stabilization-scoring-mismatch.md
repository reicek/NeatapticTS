# DF15 Racing Curriculum: Why stabilization commits are zero

## Question

`evaluateNgeWeightVariants` evaluates weight patches sequentially on the live network and restores the original weights after each patch, yet the Racing Curriculum demo recorded 10,220 ticks with 509 growth commits and 0 stabilization commits. Every one of the 771 stabilization evaluations returned `committed=false` with `reason="no_weight_mutations"`. The sequential path itself is correct, so what downstream mechanism prevents weight variants from ever beating the adaptive threshold?

## Evidence

### 1. NGE scoring mechanics (`nge-core-scout`)

- `evaluateNgeWeightVariants` (`src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`) evaluates deterministic weight patches sequentially in `evaluatePatch`. Each patch applies perturbations, activates the network on the training batch, restores original weights in a `finally` block, and returns `DEFAULT_VARIANT_SCORER(outputs, target)`.
- `DEFAULT_VARIANT_SCORER` (`src/acceleration/acceleration.variants.ts:66-81`) computes negative mean squared error:
  ```ts
  for (const output of outputs) {
    const limit = Math.min(output.length, target.length);
    for (let index = 0; index < limit; index++) {
      const delta = output[index]! - target[index]!;
      totalError += delta * delta;
    }
  }
  return -totalError / (outputs.length * target.length);
  ```
- `runNgeGrowStabilizeCycle` (`src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:633-669`) commits a variant only when:
  ```ts
  bestScore > baselineScore + threshold;
  ```
  Otherwise it sets `reason = 'no_weight_mutations'`.
- Log parse of `tmp/tmp.log` (1,280 diagnostic rows, 771 stabilization rows):
  | Metric             | Count | Min       | Max      | Mean   |
  | ------------------ | ----- | --------- | -------- | ------ |
  | `bestVariantScore` | 771   | -82.28    | -0.00056 | -16.02 |
  | `threshold`        | 771   | 0.0000084 | 1.22     | 0.213  |
  | `baselineScore`    | 771   | 0.095     | 1.10     | 0.683  |
  - Commit inequality `bestVariantScore > baselineScore + threshold` was satisfied **0 / 771** times.

### 2. Integration wiring (`implementation-pattern-scout`)

- In `examples/racing_curriculum/controller/runtime.adaptation.ts:424`, the stabilization training target is built as:
  ```ts
  const trainingTarget = evidenceWindow.map(toDrivingQuality);
  ```
  `toDrivingQuality` (around lines 1168-1180) collapses the composite `RacingQualitySignal` into a single scalar: `progress*0.35 + speed*0.25 + alignment*0.3 - offTrack*0.3 + physicsReward*0.5`.
- The racing controller network outputs a 2-D control vector (`CONTROLLER_OUTPUT_COUNT = 2` in `nge.controller.ts:12`); `controllerOutputs[0]` drives throttle, `controllerOutputs[1]` drives steering.
- Because `target.length === 1`, `DEFAULT_VARIANT_SCORER` only compares the first output dimension (throttle) against the scalar driving-quality value. Steering is ignored.
- The growth-phase path (`buildCandidateScoreWindow`, `runtime.adaptation.ts:992-1007`) reduces the multi-dimensional network output to a scalar via mean-of-absolute-activations before scoring. The stabilization path does **not** perform any analogous reduction.

### 3. Code-path reachability (`performance-trace-specialist`)

- The dev server on port 8080 was already running; the demo loaded with HTTP 200.
- Chrome DevTools performance traces during live stabilization showed sampled CPU time in minified bundle functions that source-map back to `evaluateNgeWeightVariants` and `evaluateNetworkScore`.
- Console `NGE_DIAGNOSTIC` markers consistently showed `phase=stabilization`, `actualVariantCount=32`, a finite `bestVariantScore`, and then `reason="no_weight_mutations"`.
- Example marker: tick 9868, `bestVariantScore=-87.017`, `baselineScore=0.538`, `threshold=1.292`, `reason=no_weight_mutations`.

## Decision

The sequential apply/score/undo path is **running** and is **not** short-circuited. The reason zero stabilization commits occur is a **score-space mismatch** inside `runNgeGrowStabilizeCycle`:

- `baselineScore` is the positive racing-trend quality score (the objective the controller actually maximizes, ~0.095–1.10).
- `bestScore` (from the variant evaluator) is negative MSE against the scalar driving-quality target (~-82 to ~0), which is neither in the same units nor optimizing the same objective.
- The commit test `-0.14 > 0.98 + 0.002` is therefore impossible to satisfy.

Additionally, there is a **dimensionality/semantic mismatch**: the network emits a 2-D control vector, but the scorer compares only the first dimension (throttle) to a scalar driving-quality signal. The stabilization phase is effectively asking "which weight perturbation makes throttle predict driving quality better?" rather than "which weight perturbation makes the car drive better?"

## Resolution

The fix documented in [`racing-curriculum-df16-stabilization-scorer-fix.md`](./racing-curriculum-df16-stabilization-scorer-fix.md) injects a racing-specific `VariantScorer` into the grow-stabilize cycle as `scoreFn`. The scorer collapses the 2-D controller output to a scalar and returns a positive driving-quality score, so `bestVariantScore` and `baselineScore` share the same units and direction and the commit inequality can be satisfied.

## Risks

1. **Fix direction ambiguity.** The seam can be fixed in multiple places, each with different consequences:
   - Make the stabilization baseline use the same negative-MSE units as the variant scorer.
   - Inject a racing-trend scorer into `evaluateNgeWeightVariants` so `bestScore` and `baselineScore` share the racing objective.
   - Reduce the network's control-vector output to a scalar quality score before stabilization scoring, analogous to the growth-phase `buildCandidateScoreWindow` reduction.
     A wrong choice may restore commits without actually improving driving quality.
2. **Growth/stabilization score coupling.** If the same `previousScore` value is reused for plateau detection, quality history, or other cycle contracts, changing its semantics may have side effects beyond the commit check.
3. **Threshold inflation.** The threshold formula uses `Math.max(|baseline|, |bestScore|)` when no `scoreCeiling` is supplied. Large negative `bestVariantScore` values can inflate the threshold, compounding the problem even after units are aligned.
4. **Demo bundle drift.** The performance trace used the shipped bundle (`docs/assets/racing-curriculum.bundle.js`) which was built before the trace was captured. This is low-risk but should be verified before treating trace evidence as final.

## Sources

- `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts`
- `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`
- `src/acceleration/acceleration.variants.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts`
- `examples/racing_curriculum/controller/nge.controller.ts`
- `tmp/tmp.log` (1,280 NGE_DIAGNOSTIC lines)
- Chrome DevTools performance traces in `tmp/traces/racing-stabilization-noreload*.json`

## References

- K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through Augmenting Topologies," _Evolutionary Computation_, vol. 10, no. 2, pp. 99-127, 2002. [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
- [Wikipedia — Exploration–exploitation dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma)
- [Wikipedia — Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
- [Wikipedia — Hysteresis](https://en.wikipedia.org/wiki/Hysteresis)
- [Wikipedia — Variance](https://en.wikipedia.org/wiki/Variance)
