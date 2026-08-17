/**
 * Pure MLP math executors for the Neatenstein warm-start module.
 *
 * Contains numerically-stable activation functions, parameter counting, noise
 * sampling, and inference forward pass. These are pure leaf functions with no
 * module-level state.
 *
 * @module enemy-warmstart.mlp-math.utils
 */

import { NEATENSTEIN_MLP_TOPOLOGY } from './constants';
import { BOX_MULLER_FLOOR } from './enemy-mlp.constants';

// ---------------------------------------------------------------------------
// Activation functions
// ---------------------------------------------------------------------------

/**
 * Numerically stable sigmoid activation for the output layer.
 *
 * @param x - Pre-activation value.
 * @returns sigmoid(x) in (0, 1).
 */
export function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

// ---------------------------------------------------------------------------
// Topology helpers
// ---------------------------------------------------------------------------

/**
 * Count total parameters (connection weights + biases) for a fully-connected
 * feed-forward topology with per-layer bias.
 *
 * This is a local copy to avoid a circular import from `enemy-mlp.ts`.
 *
 * @param topology - Ordered layer sizes.
 * @returns Total parameter count.
 */
export function countParametersLocal(topology: readonly number[]): number {
  let total = 0;
  for (let i = 0; i < topology.length - 1; i++) {
    total += topology[i] * topology[i + 1] + topology[i + 1];
  }
  return total;
}

// ---------------------------------------------------------------------------
// Noise sampling
// ---------------------------------------------------------------------------

/**
 * Sample a standard-normal value via the Box–Muller transform.
 *
 * @param rng - Seeded PRNG function returning [0, 1).
 * @returns A single Gaussian N(0, 1) sample.
 */
export function gaussianNoise(rng: () => number): number {
  const u1 = Math.max(BOX_MULLER_FLOOR, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Inference forward pass
// ---------------------------------------------------------------------------

/**
 * Sigmoid-based forward pass through the MLP.
 *
 * Hidden layers use `tanh`; the output layer uses `sigmoid`. This is the
 * inference companion to `trainMlpBackprop` and is used to verify
 * convergence after warm-start training.
 *
 * @param weights - Flat weight vector (connections + biases per layer).
 * @param inputs - Input vector matching the first layer size.
 * @param topology - Layer sizes; defaults to the fixed enemy topology.
 * @returns Float32Array of sigmoid outputs for the final layer.
 *
 * @example
 * ```ts
 * const out = predictMlp(weights, new Float32Array([0, 1, 0, 0, 0, 0.5]));
 * console.log(out[0]); // move probability in (0, 1)
 * ```
 */
export function predictMlp(
  weights: Float32Array,
  inputs: Float32Array,
  topology: readonly number[] = NEATENSTEIN_MLP_TOPOLOGY,
): Float32Array {
  const numLayers = topology.length - 1;
  let activations = Array.from(inputs);
  let offset = 0;

  for (let l = 0; l < numLayers; l++) {
    const inSize = topology[l];
    const outSize = topology[l + 1];
    const next = new Array<number>(outSize);
    const isOutput = l === numLayers - 1;

    for (let o = 0; o < outSize; o++) {
      let sum = 0;
      for (let i = 0; i < inSize; i++) {
        sum += activations[i] * weights[offset + o * inSize + i];
      }
      sum += weights[offset + inSize * outSize + o];
      next[o] = isOutput ? sigmoid(sum) : Math.tanh(sum);
    }

    offset += inSize * outSize + outSize;
    activations = next;
  }

  return new Float32Array(activations);
}
