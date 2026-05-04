import { NORM_EPSILON } from '../../neat/neat.constants';
import { buildDenseLayer } from './layer.factory.core.utils';
import type {
  LayerFactoryContext,
  LayerFactoryLayer,
} from './layer.utils.types';

const BATCH_NORM_ENABLED = true;
const LAYER_NORM_ENABLED = true;
const SUM_INITIAL_VALUE = 0;
const DEFAULT_TRAINING = false;

/**
 * Builds a dense layer decorated with batch-style normalization.
 *
 * This helper keeps dense connectivity but post-processes activations so
 * each activation vector is centered and scaled using mean/variance.
 *
 * Educational intuition:
 * - Centering subtracts the mean so the vector has average ~0.
 * - Scaling divides by standard deviation so typical magnitude is ~1.
 *
 * In this implementation, the normalization is applied to the activation vector
 * produced *per call*.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of nodes in the normalization layer.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * const normalized = buildBatchNormLayer(factoryContext, 16);
 * ```
 */
export function buildBatchNormLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  const layer = buildDenseLayer(context, size);
  (layer as unknown as { batchNorm: boolean }).batchNorm = BATCH_NORM_ENABLED;
  layer.describe?.({
    intent: 'normalization',
    metadata: { family: 'batchNorm', size },
  });
  layer.output?.describe({
    intent: 'normalization',
    metadata: { family: 'batchNorm', size },
  });
  applyNormalizationActivation(layer);

  return layer;
}

/**
 * Builds a dense layer decorated with layer-style normalization.
 *
 * This helper mirrors the batch variant in this implementation, applying
 * normalization to the activation vector produced for the current call.
 *
 * Note: in many frameworks "batch norm" and "layer norm" differ in how they
 * compute statistics. Here they share the same post-processing to keep the code
 * simple and educational.
 *
 * @param context Factory helpers for constructing the layer instance.
 * @param size Number of nodes in the normalization layer.
 * @returns The configured layer instance.
 *
 * Example:
 *
 * ```ts
 * const normalized = buildLayerNormLayer(factoryContext, 16);
 * ```
 */
export function buildLayerNormLayer<TLayer extends LayerFactoryLayer>(
  context: LayerFactoryContext<TLayer>,
  size: number,
): TLayer {
  const layer = buildDenseLayer(context, size);
  (layer as unknown as { layerNorm: boolean }).layerNorm = LAYER_NORM_ENABLED;
  layer.describe?.({
    intent: 'normalization',
    metadata: { family: 'layerNorm', size },
  });
  layer.output?.describe({
    intent: 'normalization',
    metadata: { family: 'layerNorm', size },
  });
  applyNormalizationActivation(layer);

  return layer;
}

/**
 * Wraps a layer activation function with normalization post-processing.
 *
 * The wrapper preserves existing activation semantics, then applies
 * `normalizeActivations(...)` to produce zero-centered, variance-scaled output.
 *
 * This is implemented as a function wrapper rather than modifying node math.
 * That makes it easy to layer normalization behavior onto each dense layer.
 *
 * @param layer Dense layer to decorate with normalization behavior.
 * @returns No return value.
 *
 * Example (conceptual flow):
 *
 * ```ts
 * // 1) baseActivate(...) computes raw activations
 * // 2) wrapper normalizes the vector before returning
 * ```
 */
function applyNormalizationActivation<TLayer extends LayerFactoryLayer>(
  layer: TLayer,
): void {
  const baseActivate = layer.activate.bind(layer);
  layer.activate = (
    values?: number[],
    training: boolean = DEFAULT_TRAINING,
  ): number[] => {
    const activations = baseActivate(values, training);
    const mean = computeMean(activations);
    const variance = computeVariance(activations, mean);
    return normalizeActivations(activations, mean, variance);
  };
}

/**
 * Computes the arithmetic mean for a vector of activations.
 *
 * @param activations Activation values to summarize.
 * @returns Mean activation value.
 *
 * Example:
 *
 * ```ts
 * computeMean([1, 2, 3]); // -> 2
 * ```
 */
function computeMean(activations: number[]): number {
  return (
    activations.reduce((sum, value) => sum + value, SUM_INITIAL_VALUE) /
    activations.length
  );
}

/**
 * Computes activation variance relative to a known mean.
 *
 * Variance is computed as the average squared deviation from the mean.
 *
 * Educational note: the standard deviation is `Math.sqrt(variance)`.
 *
 * @param activations Activation values to summarize.
 * @param mean Mean value used for centering.
 * @returns Variance of the activation values.
 *
 * Example:
 *
 * ```ts
 * const mean = computeMean([1, 2, 3]);
 * computeVariance([1, 2, 3], mean); // -> 2/3
 * ```
 */
function computeVariance(activations: number[], mean: number): number {
  return (
    activations.reduce(
      (sum, value) => sum + (value - mean) ** 2,
      SUM_INITIAL_VALUE,
    ) / activations.length
  );
}

/**
 * Normalizes activation values using mean and variance.
 *
 * A small epsilon (`NORM_EPSILON`) is added for numerical stability so
 * division remains well-defined when variance is very small.
 *
 * The transformation is applied per element:
 * $(x - \mu) / \sqrt{\sigma^2 + \epsilon}$
 *
 * @param activations Activation values to normalize.
 * @param mean Mean activation value.
 * @param variance Variance activation value.
 * @returns Normalized activation values.
 *
 * Example:
 *
 * ```ts
 * const mean = computeMean([1, 2, 3]);
 * const variance = computeVariance([1, 2, 3], mean);
 * const normalized = normalizeActivations([1, 2, 3], mean, variance);
 * ```
 */
function normalizeActivations(
  activations: number[],
  mean: number,
  variance: number,
): number[] {
  return activations.map(
    (activation) => (activation - mean) / Math.sqrt(variance + NORM_EPSILON),
  );
}
