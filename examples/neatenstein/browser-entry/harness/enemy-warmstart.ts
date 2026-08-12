/**
 * Bounded backprop warm-start for the Neatenstein MLP enemy population.
 *
 * This module implements a lightweight Lamarckian warm-start that briefly
 * steps outside pure evolution to give the enemy population a better starting
 * shape before the main search pressure takes over. It mirrors the
 * asciiMaze `trainingWarmStart.ts` pattern but targets the fixed
 * [6,6,4,4] weight-only MLP.
 *
 * Responsibilities:
 * - `trainMlpBackprop` — bounded backprop with tanh hidden activations and
 *   per-output sigmoid (BCE) cost
 * - `buildNeatensteinCurriculum` — ~23 curated combat+maze training cases
 *   with deterministic jitter
 * - `warmStartTemplate` / `warmStartWeights` — pretrain one template, then
 *   copy with per-variant noise to produce 32 variants
 *
 * @module enemy-warmstart
 */

import seedrandom from 'seedrandom';

import { NEATENSTEIN_MLP_TOPOLOGY } from './constants';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * One supervised training case used by the Neatenstein warm-start curriculum.
 */
export interface CurriculumCase {
  /** Six-dimensional vision input [compassScalar, openN, openE, openS, openW, progressDelta]. */
  input: number[];
  /** Four-dimensional soft action target [move, strafe, turn, fire]. */
  target: number[];
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Numerically stable sigmoid activation for the output layer.
 *
 * @param x - Pre-activation value.
 * @returns sigmoid(x) in (0, 1).
 */
function sigmoid(x: number): number {
  if (x >= 0) {
    return 1 / (1 + Math.exp(-x));
  }
  const e = Math.exp(x);
  return e / (1 + e);
}

/**
 * Count total parameters (connection weights + biases) for a fully-connected
 * feed-forward topology with per-layer bias.
 *
 * This is a local copy to avoid a circular import from `enemy-mlp.ts`.
 *
 * @param topology - Ordered layer sizes.
 * @returns Total parameter count.
 */
function countParametersLocal(topology: readonly number[]): number {
  let total = 0;
  for (let i = 0; i < topology.length - 1; i++) {
    total += topology[i] * topology[i + 1] + topology[i + 1];
  }
  return total;
}

/**
 * Sample a standard-normal value via the Box–Muller transform.
 *
 * @param rng - Seeded PRNG function returning [0, 1).
 * @returns A single Gaussian N(0, 1) sample.
 */
function gaussianNoise(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ---------------------------------------------------------------------------
// Bounded backprop
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
 * Training uses stochastic gradient descent (one weight update per case) for
 * faster convergence on small datasets.
 *
 * @param weights - Flat weight vector (connections + biases per layer). Modified in place.
 * @param inputs - Array of input vectors (one per training case).
 * @param targets - Array of target vectors (one per training case).
 * @param learningRate - SGD learning rate.
 * @param iterations - Maximum number of full passes over the training set.
 * @param topology - Layer sizes; defaults to the fixed enemy topology.
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

  // Precompute layer offsets into the flat weight vector.
  const offsets: number[] = [];
  let offset = 0;
  for (let l = 1; l < topology.length; l++) {
    offsets.push(offset);
    offset += topology[l - 1] * topology[l] + topology[l];
  }

  // Gradient accumulator for mini-batch updates.
  const gradAcc = new Float64Array(weights.length);
  // Mini-batch size: smaller batches give more updates per iteration (faster
  // learning) but are noisier. We use a batch size of 3 to balance speed and
  // stability — this yields ~8 updates per iteration on a 23-case curriculum.
  const batchSize = Math.min(3, numCases);

  let lastLoss = 0;

  for (let iter = 0; iter < iterations; iter++) {
    let totalLoss = 0;

    // Shuffle case order deterministically (Fisher-Yates with a per-iteration
    // seed so the same iteration always sees the same shuffle).
    const order = Array.from({ length: numCases }, (_, i) => i);
    let s = (iter + 1) * 2654435761;
    for (let i = numCases - 1; i > 0; i--) {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      const j = s % (i + 1);
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }

    let batchStart = 0;
    while (batchStart < numCases) {
      const batchEnd = Math.min(batchStart + batchSize, numCases);
      gradAcc.fill(0);

      for (let c = batchStart; c < batchEnd; c++) {
        const caseIdx = order[c];

        // ---- Forward pass ----
        const activations: number[][] = [Array.from(inputs[caseIdx])];
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

        // ---- Loss (BCE) ----
        const outputs = activations[numLayers];
        const target = targets[caseIdx];
        for (let o = 0; o < outputs.length; o++) {
          const eps = 1e-7;
          const p = Math.min(1 - eps, Math.max(eps, outputs[o]));
          totalLoss += -(
            target[o] * Math.log(p) +
            (1 - target[o]) * Math.log(1 - p)
          );
        }

        // ---- Backward pass ----
        const deltas: number[][] = new Array(numLayers);

        // Output layer: delta = output - target (BCE + sigmoid simplification)
        const outputDelta = new Array<number>(topology[numLayers]);
        for (let o = 0; o < outputDelta.length; o++) {
          outputDelta[o] = outputs[o] - target[o];
        }
        deltas[numLayers - 1] = outputDelta;

        // Hidden layers: delta = (1 - tanh²(z)) * W_next^T * delta_next
        for (let l = numLayers - 2; l >= 0; l--) {
          const outSize = topology[l + 1];
          const nextSize = topology[l + 2];
          const nextOff = offsets[l + 1];
          const delta = new Array<number>(outSize);

          for (let o = 0; o < outSize; o++) {
            let weightedDelta = 0;
            for (let n = 0; n < nextSize; n++) {
              weightedDelta +=
                weights[nextOff + n * outSize + o] * deltas[l + 1][n];
            }
            const tanhZ = Math.tanh(zs[l][o]);
            delta[o] = (1 - tanhZ * tanhZ) * weightedDelta;
          }
          deltas[l] = delta;
        }

        // ---- Accumulate gradients for this mini-batch ----
        const cw = caseWeights ? caseWeights[caseIdx] : 1;
        for (let l = 0; l < numLayers; l++) {
          const inSize = topology[l];
          const outSize = topology[l + 1];
          const off = offsets[l];

          for (let o = 0; o < outSize; o++) {
            for (let i = 0; i < inSize; i++) {
              gradAcc[off + o * inSize + i] +=
                cw * deltas[l][o] * activations[l][i];
            }
            gradAcc[off + inSize * outSize + o] += cw * deltas[l][o];
          }
        }
      }

      // ---- Weight update (mini-batch gradient descent) ----
      const avgScale = learningRate / (batchEnd - batchStart);
      for (let w = 0; w < weights.length; w++) {
        weights[w] -= avgScale * gradAcc[w];
      }

      batchStart = batchEnd;
    }

    lastLoss = totalLoss / numCases;

    // Early stop when loss is negligible.
    if (lastLoss < 0.001) break;
  }

  return lastLoss;
}

/**
 * Sigmoid-based forward pass through the MLP.
 *
 * Hidden layers use `tanh`; the output layer uses `sigmoid`. This is the
 * inference companion to {@link trainMlpBackprop} and is used to verify
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

// ---------------------------------------------------------------------------
// Neatenstein curriculum
// ---------------------------------------------------------------------------

/** Soft-target probability for the primary action. */
const TARGET_HIGH = 0.8;
/** Soft-target probability for inactive actions. */
const TARGET_LOW = 0.05;
/** Soft-target probability for fire during movement (encourages opportunistic firing). */
const TARGET_FIRE_MOVE = 0.3;
/** Soft-target probability for fire during aggressive pursuit. */
const TARGET_FIRE_PURSUE = 0.3;
/** Soft-target probability for fire during stalled/turning states. */
const TARGET_FIRE_STALL = 0.3;
/** Soft-target for moderate movement (stalled/reorienting). */
const TARGET_MOVE_STALLED = 0.45;
/** Soft-target for moderate turning (stalled/reorienting). */
const TARGET_TURN_STALLED = 0.3;
/** Soft-target for strafe primary action. */
const TARGET_STRAFE = 0.4;
/** Soft-target for moderate strafe context actions. */
const TARGET_STRAFE_CONTEXT = 0.3;
/** Soft-target for turn primary action. */
const TARGET_TURN = 0.7;
/** Soft-target for fire during turn (looking for player). */
const TARGET_FIRE_TURN = 0.35;
/** Soft-target for very low movement (fire dominant). */
const TARGET_VERY_LOW = 0.05;

/**
 * Build the Neatenstein warm-start curriculum.
 *
 * Generates ~23 base cases mapping combat+maze vision inputs to soft action
 * targets, with deterministic jitter (±0.1 on inputs) for robustness. The
 * structure mirrors the asciiMaze `buildLamarckianTrainingSet` pattern:
 *
 * - Single-path corridors (4 cases, one per cardinal direction)
 * - Strong-progress corridors (2 cases)
 * - Two-way junctions with directional bias (8 cases)
 * - Regressing/stalled movement (3 cases)
 * - Combat scenarios: fire (2), strafe (2), turn (1), pursue (1)
 *
 * Each case maps a 6-dimensional input `[compassScalar, openN, openE, openS,
 * openW, progressDelta]` to a 4-dimensional soft target
 * `[move, strafe, turn, fire]` with values in [0, 1].
 *
 * Combat cases use input signatures that are clearly distinguishable from
 * movement cases (e.g., all directions open + very high progress for fire,
 * two opposite directions open for strafe, all closed for turn) so the
 * network can learn non-linear decision boundaries in ≤60 iterations.
 *
 * @returns Array of curriculum cases. The output is deterministic — the same
 *   array is produced on every call.
 *
 * @example
 * ```ts
 * const curriculum = buildNeatensteinCurriculum();
 * console.log(curriculum.length); // ~23
 * console.log(curriculum[0].input.length); // 6
 * console.log(curriculum[0].target.length); // 4
 * ```
 */
export function buildNeatensteinCurriculum(): CurriculumCase[] {
  const cases: CurriculumCase[] = [];

  const pushCase = (input: number[], target: number[]): void => {
    cases.push({ input, target });
  };

  // === Movement cases (17 cases) ===

  // Single open path — one direction open, moderate progress (4 cases)
  pushCase(
    [0, 1, 0, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.25, 0, 1, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.5, 0, 0, 1, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.75, 0, 0, 0, 1, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );

  // Strong progress — closing on target (2 cases)
  pushCase(
    [0, 1, 0, 0, 0, 0.9],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_PURSUE],
  );
  pushCase(
    [0.25, 0, 1, 0, 0, 0.9],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_PURSUE],
  );

  // Two-way junctions — compass guides primary direction (8 cases)
  pushCase(
    [0, 1, 0.6, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0, 1, 0, 0.6, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.25, 0.6, 1, 0, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.25, 0, 1, 0.6, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.5, 0, 0.6, 1, 0, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.5, 0, 0, 1, 0.6, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.75, 0, 0, 0.6, 1, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );
  pushCase(
    [0.75, 0.6, 0, 0, 1, 0.5],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );

  // Regressing/stalled — low progress, need to reorient (3 cases)
  pushCase(
    [0, 1, 0.3, 0, 0, 0.1],
    [TARGET_MOVE_STALLED, TARGET_LOW, TARGET_TURN_STALLED, TARGET_FIRE_STALL],
  );
  pushCase(
    [0.25, 0.5, 1, 0.4, 0, 0.1],
    [TARGET_MOVE_STALLED, TARGET_LOW, TARGET_TURN_STALLED, TARGET_FIRE_STALL],
  );
  pushCase(
    [0.5, 0, 0.3, 1, 0.2, 0.1],
    [TARGET_MOVE_STALLED, TARGET_LOW, TARGET_TURN_STALLED, TARGET_FIRE_STALL],
  );

  // === Combat cases (6 cases) ===

  // Player very close, all directions open — fire (2 cases)
  // Input signature: all-open + very high progress is unique to fire cases.
  pushCase(
    [0, 1, 1, 1, 1, 0.95],
    [TARGET_VERY_LOW, TARGET_LOW, TARGET_LOW, TARGET_HIGH],
  );
  pushCase(
    [0.25, 1, 1, 1, 1, 0.95],
    [TARGET_VERY_LOW, TARGET_LOW, TARGET_LOW, TARGET_HIGH],
  );

  // Player flanking — two opposite directions open, strafe (2 cases)
  // Input signature: opposite-pair-open is unique to strafe cases.
  pushCase(
    [0.5, 1, 0, 1, 0, 0.6],
    [
      TARGET_STRAFE_CONTEXT,
      TARGET_STRAFE,
      TARGET_STRAFE_CONTEXT,
      TARGET_FIRE_MOVE,
    ],
  );
  pushCase(
    [0.75, 0, 1, 0, 1, 0.6],
    [
      TARGET_STRAFE_CONTEXT,
      TARGET_STRAFE,
      TARGET_STRAFE_CONTEXT,
      TARGET_FIRE_MOVE,
    ],
  );

  // Stalled, all closed — turn to find player (1 case)
  // Input signature: all-closed + very low progress is unique to turn case.
  pushCase(
    [0.5, 0, 0, 0, 0, 0.05],
    [TARGET_VERY_LOW, TARGET_LOW, TARGET_TURN, TARGET_FIRE_TURN],
  );

  // Player retreating — moderate progress, pursue (1 case)
  pushCase(
    [0, 1, 0, 0, 0, 0.3],
    [TARGET_HIGH, TARGET_LOW, TARGET_LOW, TARGET_FIRE_MOVE],
  );

  // === Deterministic jitter (±0.1 on inputs) ===
  const rng = seedrandom('neatenstein:curriculum:jitter:42');
  const jitterAmp = 0.1;
  for (const c of cases) {
    for (let i = 0; i < c.input.length; i++) {
      c.input[i] = Math.max(
        0,
        Math.min(1, c.input[i] + (rng() * 2 - 1) * jitterAmp),
      );
    }
  }

  return cases;
}

/**
 * Per-case loss weighting for the Neatenstein curriculum.
 *
 * Combat cases (fire, strafe, turn, pursue) are a minority of the curriculum
 * but represent critical behavioural modes. Without weighting, the majority
 * movement cases dominate the gradient and the network fails to learn combat
 * behaviours. This function returns a weight array giving combat cases 3×
 * influence so the network can learn both movement and combat within ≤60
 * iterations.
 *
 * @param numCases - Number of curriculum cases (must match the curriculum).
 * @returns Array of per-case weights (1.0 for movement, 3.0 for combat).
 */
export function getCurriculumCaseWeights(numCases: number): number[] {
  // The curriculum layout is: 17 movement/stalled + (numCases - 17) combat.
  // The curriculum layout is: 14 movement (0-13), 3 stalled (14-16),
  // and combat cases (17+). Combat cases represent distinct behavioural
  // modes that are minority cases; weighting them 2× ensures the network
  // learns both movement and combat within ≤60 iterations.
  const weights = new Array<number>(numCases);
  for (let i = 0; i < numCases; i++) {
    weights[i] = i >= 17 ? 2.0 : 1.0;
  }
  return weights;
}

// ---------------------------------------------------------------------------
// Warm-start weight generation
// ---------------------------------------------------------------------------

/** Weight initialization scale for the template (avoids tanh saturation). */
const TEMPLATE_INIT_SCALE = 0.3;
/** Per-variant Gaussian noise standard deviation for weight copying. */
const VARIANT_NOISE_STDDEV = 0.08;
/** Default learning rate for warm-start backprop. */
const WARMSTART_LEARNING_RATE = 0.7;
/** Default iteration count for warm-start backprop. */
const WARMSTART_ITERATIONS = 60;

/**
 * Pretrain one template weight vector via bounded backprop on the curriculum.
 *
 * The template is initialized with small deterministic random values (scaled
 * by `seed`) and trained on the Neatenstein curriculum using
 * {@link trainMlpBackprop}. The same seed always produces the same template.
 *
 * @param seed - Deterministic seed for weight initialization.
 * @returns A trained weight vector sized for the fixed MLP topology.
 *
 * @example
 * ```ts
 * const template = warmStartTemplate(42);
 * console.log(template.length); // 90
 * ```
 */
export function warmStartTemplate(seed: number): Float32Array {
  const topology = NEATENSTEIN_MLP_TOPOLOGY;
  const weightCount = countParametersLocal(topology);

  const rng = seedrandom(`${seed}:warmstart:template`);
  const weights = new Float32Array(weightCount);
  for (let i = 0; i < weightCount; i++) {
    weights[i] = (rng() * 2 - 1) * TEMPLATE_INIT_SCALE;
  }

  const curriculum = buildNeatensteinCurriculum();
  const inputs = curriculum.map((c) => new Float32Array(c.input));
  const targets = curriculum.map((c) => new Float32Array(c.target));
  const caseWeights = getCurriculumCaseWeights(curriculum.length);

  trainMlpBackprop(
    weights,
    inputs,
    targets,
    WARMSTART_LEARNING_RATE,
    WARMSTART_ITERATIONS,
    NEATENSTEIN_MLP_TOPOLOGY,
    caseWeights,
  );

  return weights;
}

/**
 * Generate one warm-started variant weight vector.
 *
 * Trains a deterministic template (same for every `variantId` given the same
 * `seed`) and adds per-variant Gaussian noise so the 32 variants explore
 * around the trained starting point rather than being identical clones.
 *
 * @param seed - Population seed (determines the template).
 * @param variantId - Stable variant index (determines the noise pattern).
 * @returns A noisy copy of the trained template, sized for the fixed topology.
 *
 * @example
 * ```ts
 * const w0 = warmStartWeights(7, 0);
 * const w1 = warmStartWeights(7, 1);
 * console.log(w0.length); // 90
 * ```
 */
export function warmStartWeights(
  seed: number,
  variantId: number,
): Float32Array {
  const template = warmStartTemplate(seed);
  const weights = new Float32Array(template);

  if (variantId < 0) return weights;

  const rng = seedrandom(`${seed}:variant:${variantId}`);
  for (let i = 0; i < weights.length; i++) {
    weights[i] += gaussianNoise(rng) * VARIANT_NOISE_STDDEV;
  }

  return weights;
}
