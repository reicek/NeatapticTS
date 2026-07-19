/**
 * Generic async weight-variant evaluator for the network acceleration layer.
 *
 * This module evaluates a set of candidate weight perturbations against a fixed
 * input batch and target vector. It is intentionally decoupled from
 * `src/architecture/network`; any object that satisfies
 * {@link VariantEvaluationNetwork} can be evaluated. The active backend is resolved
 * through the existing acceleration orchestrator (GPU first, then WebWorker, then
 * CPU), so callers receive backend metadata that matches the actual execution mode.
 *
 * The evaluator restores each connection's original weight after scoring the
 * corresponding variant, so the network is left in the same state it was in when
 * the function was called (minus possible mutation from the network's own
 * activation side effects, which are outside this module's scope).
 */

import { resolveAccelerationConfig } from './acceleration.config';
import { DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD } from './acceleration.constants';
import { evaluateWeightVariantsOnGpu } from './acceleration.gpu';
import type { AccelerationObserver } from './acceleration.observer';
import { autoEnableAcceleration } from './acceleration.orchestrator';
import type {
  AccelerationConfig,
  AccelerationMode,
  AccelerationStatus,
  VariantEvaluationNetwork,
  VariantScorer,
  WeightVariant,
  WeightVariantInputs,
  WeightVariantResult,
  WeightVariantTarget,
} from './acceleration.types';

export type {
  VariantEvaluationNetwork,
  VariantScorer,
  WeightVariant,
  WeightVariantInputs,
  WeightVariantResult,
  WeightVariantTarget,
} from './acceleration.types';

/**
 * Default variant scorer: negative mean squared error against the target vector.
 *
 * The scorer averages the squared error across all inputs and outputs, then
 * negates the result so that higher scores are better. A perfect prediction
 * produces a score of `0`; worse predictions produce increasingly negative
 * scores.
 *
 * Background reading:
 * - Mean squared error:
 *   [Wikipedia — Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error).
 *
 * @param outputs - Stack of network output vectors, one per input sample.
 * @param target - Target output vector. Only the first `target.length` entries
 *   of each output are compared.
 * @returns Negative mean squared error.
 *
 * @example
 * ```ts
 * const score = DEFAULT_VARIANT_SCORER([[0.5, 0.5]], [1.0]);
 * console.log(score); // -0.25
 * ```
 */
export const DEFAULT_VARIANT_SCORER: VariantScorer = (outputs, target) => {
  if (outputs.length === 0 || target.length === 0) {
    return 0;
  }

  let totalError = 0;
  for (const output of outputs) {
    const limit = Math.min(output.length, target.length);
    for (let index = 0; index < limit; index++) {
      const delta = output[index]! - target[index]!;
      totalError += delta * delta;
    }
  }

  return -totalError / (outputs.length * target.length);
};

/**
 * Evaluate a list of weight variants asynchronously.
 *
 * For each variant the evaluator:
 * 1. applies the signed delta to the connection at `weightIndex`,
 * 2. runs the network on every input in `inputs`,
 * 3. scores the stacked outputs,
 * 4. restores the original connection weight.
 *
 * The function always returns a `Promise` so callers can `await` it whether the
 * network's `activate` method is synchronous or asynchronous. The returned
 * metadata reports the backend used, the number of variants, the largest
 * absolute delta in the set, and whether the default or a custom scorer was
 * used.
 *
 * Weight perturbation followed by a forward pass and score is a local-search
 * pattern for exploring candidate weights. Background reading:
 * - Local search (optimization):
 *   [Wikipedia contributors — Local search (optimization)](https://en.wikipedia.org/wiki/Local_search_(optimization)).
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *   classDef pool fill:#0a1f2e,stroke:#00d4aa,color:#e0fff7,stroke-width:1.5px;
 *
 *   Start([variants + network]) --> ResolveConfig[resolveAccelerationConfig]:::accent
 *   ResolveConfig --> Probe{parallelVariantCount > 1?}
 *   Probe -->|no| Sequential[Evaluate variants one by one]:::base
 *   Probe -->|yes| Batch[Slice into batches of size parallelVariantCount]:::accent
 *   Batch --> Parallel[Evaluate batch in parallel]:::pool
 *   Parallel --> Restore[Restore original weights]:::accent
 *   Restore --> More{More batches?}
 *   More -->|yes| Batch
 *   More -->|no| Score[Return scores + backend metadata]:::base
 *   Sequential --> Score
 * ```
 *
 * @param network - Network surface to evaluate.
 * @param variants - Candidate weight perturbations.
 * @param inputs - Input batch, one vector per sample.
 * @param target - Target output vector for the default scorer.
 * @param scoreFn - Optional scorer; defaults to {@link DEFAULT_VARIANT_SCORER}.
 * @param seed - Optional determinism seed reserved for future backend selection.
 * @param config - Optional acceleration configuration. The resolved backend
 *   is selected through `autoEnableAcceleration` (GPU → worker → CPU). When
 *   `parallelVariantCount` is greater than `1`, variants are dispatched
 *   concurrently in batches of that size and original weights are restored
 *   between batches. A value of `1` evaluates variants sequentially. The
 *   default `backend` is `auto` and the default `parallelVariantCount` is `16`;
 *   `stageVariantCounts` is forwarded to NGE lifecycle consumers such as
 *   {@link evaluateNgeWeightVariants}.
 * @param observer - Optional acceleration observer. When supplied, backend-change
 *   and fallback events are emitted during backend selection, and a telemetry
 *   event is emitted after the evaluation finishes.
 * @returns Promise resolving to per-variant scores and backend metadata.
 *
 * @example
 * ```ts
 * const result = await evaluateWeightVariantsAsync(
 *   network,
 *   [{ weightIndex: 0, delta: 0.05 }],
 *   [[0.5, 0.5]],
 *   [1.0],
 * );
 * console.log(result.bestIndex, result.metadata.backend);
 * ```
 */
export async function evaluateWeightVariantsAsync(
  network: VariantEvaluationNetwork,
  variants: readonly WeightVariant[],
  inputs: WeightVariantInputs,
  target: WeightVariantTarget,
  scoreFn?: VariantScorer,
  seed?: number,
  config?: AccelerationConfig,
  observer?: AccelerationObserver,
): Promise<WeightVariantResult> {
  const scorer = scoreFn ?? DEFAULT_VARIANT_SCORER;
  const scorerName = scoreFn === undefined ? 'default' : 'custom';
  const resolvedConfig = resolveAccelerationConfig(config);
  // `resolveAccelerationConfig` always defaults `backend` to 'auto', so the
  // value is guaranteed to be defined here. Avoid `?? 'auto'` because it
  // creates an unreachable branch that shows up as uncovered code.
  const requestedBackend = resolvedConfig.backend!;

  let status: AccelerationStatus;
  if (requestedBackend === 'cpu') {
    status = {
      mode: 'cpu',
      gpu: { available: false, reason: 'CPU requested' },
      worker: { available: false, count: 0, reason: 'CPU requested' },
      cpu: { available: true },
    };
    console.log(
      '[NeatapticTS Acceleration] Explicit CPU backend requested, skipping GPU/worker',
    );
    if (observer?.onBackendChange) {
      observer.onBackendChange({
        previous: null,
        current: 'cpu',
        reason: 'Acceleration backend selected: cpu',
        timestamp: Date.now(),
      });
    }
  } else {
    status = await autoEnableAcceleration({
      nodeCount: network.nodes.length,
      batchParallelCount: variants.length,
      config,
    });
  }

  const backend: AccelerationMode = status.mode;
  console.log(
    `[NeatapticTS Acceleration] Variant evaluator selected backend: ${backend}`,
  );

  if (observer?.onBackendChange) {
    observer.onBackendChange({
      previous: null,
      current: backend,
      reason: `Acceleration backend selected: ${backend}`,
      timestamp: Date.now(),
    });
  }

  if (
    observer?.onFallback &&
    requestedBackend !== 'auto' &&
    requestedBackend !== backend
  ) {
    observer.onFallback({
      requested: requestedBackend,
      chosen: backend,
      reason: `Requested ${requestedBackend} backend unavailable; falling back to ${backend}`,
      timestamp: Date.now(),
    });
  }

  const scaleDivisor = resolveScaleDivisor(variants);
  const parallelVariantCount = resolvedConfig.parallelVariantCount!;
  const gpuDevice =
    backend === 'gpu' &&
    network.nodes.length >= DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD
      ? (status.gpu.device ?? undefined)
      : undefined;
  const useGpu = gpuDevice !== undefined;

  const scores: number[] = [];
  if (parallelVariantCount > 1) {
    for (
      let index = 0;
      index < variants.length;
      index += parallelVariantCount
    ) {
      const batch = variants.slice(index, index + parallelVariantCount);
      const batchScores = useGpu
        ? await evaluateWeightVariantsOnGpu(
            network,
            batch,
            inputs,
            target,
            scorer,
            gpuDevice,
          )
        : await Promise.all(
            batch.map((variant) =>
              evaluateVariant(network, variant, inputs, target, scorer),
            ),
          );
      scores.push(...batchScores);
    }
  } else {
    for (const variant of variants) {
      const batchScores = useGpu
        ? await evaluateWeightVariantsOnGpu(
            network,
            [variant],
            inputs,
            target,
            scorer,
            gpuDevice,
          )
        : [await evaluateVariant(network, variant, inputs, target, scorer)];
      scores.push(...batchScores);
    }
  }

  const bestIndex = findBestIndex(scores);
  const bestScore = scores.length > 0 ? scores[bestIndex]! : 0;

  if (observer?.onTelemetry) {
    observer.onTelemetry({
      backend,
      inferenceMs: 0,
    });
  }

  return {
    bestIndex,
    bestScore,
    scores,
    metadata: {
      backend,
      variantCount: variants.length,
      scaleDivisor,
      scorer: scorerName,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Resolve the scale divisor used for telemetry and normalization.
 *
 * Returns the largest absolute delta in the variant set, floored to `1` so
 * callers always have a meaningful nonzero divisor.
 *
 * @param variants - Candidate weight perturbations.
 * @returns Largest absolute delta, or `1` when no variants are given.
 * @internal
 */
function resolveScaleDivisor(variants: readonly WeightVariant[]): number {
  let maxDelta = 0;
  for (const variant of variants) {
    const absoluteDelta = Math.abs(variant.delta);
    if (absoluteDelta > maxDelta) {
      maxDelta = absoluteDelta;
    }
  }
  return Math.max(1, maxDelta);
}

/**
 * Find the index of the highest score, preferring the first in a tie.
 *
 * @param scores - Per-variant scores.
 * @returns Index of the best variant, or `0` when the array is empty.
 * @internal
 */
function findBestIndex(scores: readonly number[]): number {
  if (scores.length === 0) {
    return 0;
  }

  let bestIndex = 0;
  let bestScore = scores[0]!;
  for (let index = 1; index < scores.length; index++) {
    const score = scores[index]!;
    if (score > bestScore) {
      bestScore = score;
      bestIndex = index;
    }
  }
  return bestIndex;
}

/**
 * Evaluate a single variant and restore the original connection weight.
 *
 * If the variant's `weightIndex` points past the end of the connection list,
 * the delta is silently skipped; the network is still evaluated so the variant
 * produces a valid score. This makes the evaluator tolerant of test doubles
 * that do not materialize every connection.
 *
 * @param network - Network surface to evaluate.
 * @param variant - Weight perturbation to apply.
 * @param inputs - Input batch.
 * @param target - Target output vector.
 * @param scorer - Scoring function.
 * @returns Score for this variant.
 * @internal
 */
async function evaluateVariant(
  network: VariantEvaluationNetwork,
  variant: WeightVariant,
  inputs: WeightVariantInputs,
  target: WeightVariantTarget,
  scorer: VariantScorer,
): Promise<number> {
  const connection = network.connections[variant.weightIndex];
  const originalWeight = connection?.weight;

  if (connection !== undefined) {
    connection.weight += variant.delta;
  }

  try {
    const outputs: number[][] = [];
    for (const input of inputs) {
      const output = await Promise.resolve(network.activate(input));
      outputs.push([...output]);
    }
    return scorer(outputs, target);
  } finally {
    if (connection !== undefined && originalWeight !== undefined) {
      connection.weight = originalWeight;
    }
  }
}
