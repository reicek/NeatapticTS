/**
 * Backpropagation executors for the Neatenstein warm-start module.
 *
 * Splits the bounded backprop algorithm into pure leaf functions (forward
 * pass, output delta, hidden backward, gradient accumulation, weight update,
 * shuffle) plus a thin orchestrator (`trainMlpBackprop`).
 *
 * @module enemy-warmstart.backprop.utils
 */

import { NEATENSTEIN_MLP_TOPOLOGY } from './constants';
import { sigmoid } from './enemy-warmstart.mlp-math.utils';
import {
  BCE_EPSILON,
  BCE_BATCH_SIZE,
  EARLY_STOP_LOSS,
} from '../warmstart.constants';
import {
  SHUFFLE_LCG_SEED_MULTIPLIER,
  SHUFFLE_LCG_STEP_MULTIPLIER,
  SHUFFLE_LCG_STEP_OFFSET,
  SHUFFLE_LCG_MASK,
} from './enemy-mlp.constants';

// ---------------------------------------------------------------------------
// Layer offset computation
// ---------------------------------------------------------------------------

/**
 * Precompute layer offsets into the flat weight vector.
 *
 * Each layer `l` has `topology[l] * topology[l+1]` connection weights followed
 * by `topology[l+1]` biases. This function returns the starting index for each
 * layer's weight block.
 *
 * @param topology - Ordered layer sizes.
 * @returns Array of offsets, one per layer transition.
 */
export function computeLayerOffsets(topology: readonly number[]): number[] {
  const offsets: number[] = [];
  let offset = 0;
  for (let l = 1; l < topology.length; l++) {
    offsets.push(offset);
    offset += topology[l - 1] * topology[l] + topology[l];
  }
  return offsets;
}

// ---------------------------------------------------------------------------
// Deterministic shuffle
// ---------------------------------------------------------------------------

/**
 * Deterministic Fisher-Yates shuffle of case indices for a given iteration.
 *
 * Uses a seeded LCG so the same iteration always produces the same shuffle
 * order, ensuring reproducibility.
 *
 * @param numCases - Number of training cases.
 * @param iteration - Current iteration index (seeds the shuffle).
 * @returns Shuffled array of case indices.
 */
export function shuffleCaseOrder(
  numCases: number,
  iteration: number,
): number[] {
  const order = Array.from({ length: numCases }, (_, i) => i);
  let s = (iteration + 1) * SHUFFLE_LCG_SEED_MULTIPLIER;
  for (let i = numCases - 1; i > 0; i--) {
    s = (s * SHUFFLE_LCG_STEP_MULTIPLIER + SHUFFLE_LCG_STEP_OFFSET) & SHUFFLE_LCG_MASK;
    const j = s % (i + 1);
    const tmp = order[i];
    order[i] = order[j];
    order[j] = tmp;
  }
  return order;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

/**
 * Forward pass through the MLP for a single training case.
 *
 * Computes layer activations and pre-activations (zs) for use by the backward
 * pass. Hidden layers use `tanh`; the output layer uses `sigmoid`.
 *
 * @param weights - Flat weight vector (connections + biases per layer).
 * @param input - Input vector for this case.
 * @param topology - Layer sizes.
 * @param offsets - Precomputed layer offsets.
 * @param numLayers - Number of layer transitions (topology.length - 1).
 * @returns Object with `activations` (per-layer, including input) and `zs` (pre-activations).
 */
export function forwardPass(
  weights: Float32Array,
  input: Float32Array,
  topology: readonly number[],
  offsets: readonly number[],
  numLayers: number,
): { activations: number[][]; zs: number[][] } {
  const activations: number[][] = [Array.from(input)];
  const zs: number[][] = [];

  for (let l = 0; l < numLayers; l++) {
    const inSize = topology[l];
    const outSize = topology[l + 1];
    const off = offsets[l];
    const z = new Array<number>(outSize);
    const a = new Array<number>(outSize);
    const isOutput = l === numLayers - 1;

    for (let o = 0; o < outSize; o++) {
      let sum = 0;
      for (let i = 0; i < inSize; i++) {
        sum += activations[l][i] * weights[off + o * inSize + i];
      }
      sum += weights[off + inSize * outSize + o];
      z[o] = sum;
      a[o] = isOutput ? sigmoid(sum) : Math.tanh(sum);
    }
    zs.push(z);
    activations.push(a);
  }

  return { activations, zs };
}

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

/**
 * Compute binary cross-entropy (BCE) loss for one case.
 *
 * @param outputs - Sigmoid outputs from the forward pass.
 * @param target - Target vector for this case.
 * @returns Sum of per-output BCE loss.
 */
export function computeBceLoss(
  outputs: readonly number[],
  target: Float32Array,
): number {
  let loss = 0;
  const eps = BCE_EPSILON;
  for (let o = 0; o < outputs.length; o++) {
    const p = Math.min(1 - eps, Math.max(eps, outputs[o]));
    loss += -(target[o] * Math.log(p) + (1 - target[o]) * Math.log(1 - p));
  }
  return loss;
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

/**
 * Compute output layer delta (BCE + sigmoid simplification: delta = output - target).
 *
 * @param outputs - Sigmoid outputs from the forward pass.
 * @param target - Target vector for this case.
 * @returns Delta array for the output layer.
 */
export function computeOutputDelta(
  outputs: readonly number[],
  target: Float32Array,
): number[] {
  const delta = new Array<number>(outputs.length);
  for (let o = 0; o < delta.length; o++) {
    delta[o] = outputs[o] - target[o];
  }
  return delta;
}

/**
 * Backward pass through hidden layers, filling the deltas array in-place.
 *
 * For each hidden layer `l`, computes: `delta[l] = (1 - tanh²(z)) * W_next^T * delta[l+1]`.
 *
 * @param deltas - Deltas array (output layer delta must already be set at `deltas[numLayers - 1]`).
 * @param weights - Flat weight vector.
 * @param topology - Layer sizes.
 * @param offsets - Precomputed layer offsets.
 * @param numLayers - Number of layer transitions.
 * @param zs - Pre-activations from the forward pass.
 */
export function backwardHidden(
  deltas: number[][],
  weights: Float32Array,
  topology: readonly number[],
  offsets: readonly number[],
  numLayers: number,
  zs: number[][],
): void {
  for (let l = numLayers - 2; l >= 0; l--) {
    const outSize = topology[l + 1];
    const nextSize = topology[l + 2];
    const nextOff = offsets[l + 1];
    const delta = new Array<number>(outSize);

    for (let o = 0; o < outSize; o++) {
      let weightedDelta = 0;
      for (let n = 0; n < nextSize; n++) {
        weightedDelta += weights[nextOff + n * outSize + o] * deltas[l + 1][n];
      }
      const tanhZ = Math.tanh(zs[l][o]);
      delta[o] = (1 - tanhZ * tanhZ) * weightedDelta;
    }
    deltas[l] = delta;
  }
}

// ---------------------------------------------------------------------------
// Gradient accumulation
// ---------------------------------------------------------------------------

/**
 * Accumulate gradients for a single case into the gradient accumulator.
 *
 * @param gradAcc - Gradient accumulator (modified in-place).
 * @param deltas - Deltas from the backward pass.
 * @param activations - Activations from the forward pass.
 * @param topology - Layer sizes.
 * @param offsets - Precomputed layer offsets.
 * @param numLayers - Number of layer transitions.
 * @param caseWeight - Per-case loss weight (1.0 default, higher for combat cases).
 */
export function accumulateGradients(
  gradAcc: Float64Array,
  deltas: readonly number[][],
  activations: readonly number[][],
  topology: readonly number[],
  offsets: readonly number[],
  numLayers: number,
  caseWeight: number,
): void {
  for (let l = 0; l < numLayers; l++) {
    const inSize = topology[l];
    const outSize = topology[l + 1];
    const off = offsets[l];

    for (let o = 0; o < outSize; o++) {
      for (let i = 0; i < inSize; i++) {
        gradAcc[off + o * inSize + i] +=
          caseWeight * deltas[l][o] * activations[l][i];
      }
      gradAcc[off + inSize * outSize + o] += caseWeight * deltas[l][o];
    }
  }
}

// ---------------------------------------------------------------------------
// Weight update
// ---------------------------------------------------------------------------

/**
 * Apply mini-batch gradient descent update to the weights.
 *
 * @param weights - Flat weight vector (modified in-place).
 * @param gradAcc - Accumulated gradients for this mini-batch.
 * @param learningRate - SGD learning rate.
 * @param batchSize - Number of cases in this mini-batch.
 */
export function applyMiniBatchUpdate(
  weights: Float32Array,
  gradAcc: Float64Array,
  learningRate: number,
  batchSize: number,
): void {
  const avgScale = learningRate / batchSize;
  for (let w = 0; w < weights.length; w++) {
    weights[w] -= avgScale * gradAcc[w];
  }
}

// ---------------------------------------------------------------------------
// Thin orchestrator
// ---------------------------------------------------------------------------

/**
 * Bounded backpropagation for the fixed [6,6,4,4] tanh MLP.
 *
 * Hidden layers use `tanh` activation with gradient `1 − tanh²(z)`. The output
 * layer uses `sigmoid` activation with Binary Cross-Entropy (BCE) cost. Because
 * the BCE + sigmoid derivative simplifies to `output − target`, the output
 * delta is computed directly without explicit sigmoid-derivative
 * multiplication.
 *
 * Training uses stochastic gradient descent (one weight update per mini-batch)
 * for faster convergence on small datasets. This is a thin orchestrator that
 * delegates to {@link forwardPass}, {@link computeOutputDelta},
 * {@link backwardHidden}, {@link accumulateGradients},
 * {@link applyMiniBatchUpdate}, and {@link shuffleCaseOrder}.
 *
 * @param weights - Flat weight vector (connections + biases per layer). Modified in place.
 * @param inputs - Array of input vectors (one per training case).
 * @param targets - Array of target vectors (one per training case).
 * @param learningRate - SGD learning rate.
 * @param iterations - Maximum number of full passes over the training set.
 * @param topology - Layer sizes; defaults to the fixed enemy topology.
 * @param caseWeights - Optional per-case loss weights.
 * @returns Final average loss across all cases (BCE). Returns 0 for empty inputs or zero iterations.
 *
 * @example
 * ```ts
 * const weights = new Float32Array(90);
 * const curriculum = buildNeatensteinCurriculum();
 * const inputs = curriculum.map(c => new Float32Array(c.input));
 * const targets = curriculum.map(c => new Float32Array(c.target));
 * const loss = trainMlpBackprop(weights, inputs, targets, 0.5, 60);
 * console.log(loss); // final average BCE loss
 * ```
 */
export function trainMlpBackprop(
  weights: Float32Array,
  inputs: readonly Float32Array[],
  targets: readonly Float32Array[],
  learningRate: number,
  iterations: number,
  topology: readonly number[] = NEATENSTEIN_MLP_TOPOLOGY,
  caseWeights?: readonly number[],
): number {
  if (inputs.length === 0 || iterations <= 0) return 0;

  const numLayers = topology.length - 1;
  const numCases = inputs.length;
  const offsets = computeLayerOffsets(topology);
  const gradAcc = new Float64Array(weights.length);
  // Mini-batch size: smaller batches give more updates per iteration (faster
  // learning) but are noisier. We use a batch size of 3 to balance speed and
  // stability — this yields ~8 updates per iteration on a 23-case curriculum.
  const batchSize = Math.min(BCE_BATCH_SIZE, numCases);

  let lastLoss = 0;

  for (let iter = 0; iter < iterations; iter++) {
    let totalLoss = 0;
    const order = shuffleCaseOrder(numCases, iter);

    let batchStart = 0;
    while (batchStart < numCases) {
      const batchEnd = Math.min(batchStart + batchSize, numCases);
      gradAcc.fill(0);

      for (let c = batchStart; c < batchEnd; c++) {
        const caseIdx = order[c];

        // --- Forward pass ---
        const { activations, zs } = forwardPass(
          weights,
          inputs[caseIdx],
          topology,
          offsets,
          numLayers,
        );
        const outputs = activations[numLayers];

        // --- Loss (BCE) ---
        totalLoss += computeBceLoss(outputs, targets[caseIdx]);

        // --- Backward pass ---
        const deltas: number[][] = new Array(numLayers);
        deltas[numLayers - 1] = computeOutputDelta(outputs, targets[caseIdx]);
        backwardHidden(deltas, weights, topology, offsets, numLayers, zs);

        // --- Accumulate gradients ---
        const cw = caseWeights ? caseWeights[caseIdx] : 1;
        accumulateGradients(
          gradAcc,
          deltas,
          activations,
          topology,
          offsets,
          numLayers,
          cw,
        );
      }

      // --- Weight update (mini-batch gradient descent) ---
      applyMiniBatchUpdate(
        weights,
        gradAcc,
        learningRate,
        batchEnd - batchStart,
      );

      batchStart = batchEnd;
    }

    lastLoss = totalLoss / numCases;

    // Early stop when loss is negligible.
    if (lastLoss < EARLY_STOP_LOSS) break;
  }

  return lastLoss;
}
