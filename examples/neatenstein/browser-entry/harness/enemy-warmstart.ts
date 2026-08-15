/**
 * Bounded backprop warm-start for the Neatenstein MLP enemy population.
 *
 * This module is the public facade for the warm-start system. It delegates to
 * three sibling util files:
 * - `enemy-warmstart.mlp-math.utils.ts` — pure math (sigmoid, noise, predictMlp)
 * - `enemy-warmstart.backprop.utils.ts` — backprop executors + thin orchestrator
 * - `enemy-warmstart.curriculum.utils.ts` — curriculum cases + weights + consts
 *
 * The main module retains only the `warmStartTemplate` and `warmStartWeights`
 * orchestrators.
 *
 * @module enemy-warmstart
 */

import seedrandom from 'seedrandom';

import { NEATENSTEIN_MLP_TOPOLOGY } from './constants';
import {
  TEMPLATE_INIT_SCALE,
  VARIANT_NOISE_STDDEV,
  WARMSTART_LEARNING_RATE,
  WARMSTART_ITERATIONS,
} from '../warmstart.constants';
import {
  countParametersLocal,
  gaussianNoise,
} from './enemy-warmstart.mlp-math.utils';
import { trainMlpBackprop } from './enemy-warmstart.backprop.utils';
import {
  buildNeatensteinCurriculum,
  getCurriculumCaseWeights,
} from './enemy-warmstart.curriculum.utils';

// Re-export previously-public symbols so consumers keep working unchanged.
export { predictMlp } from './enemy-warmstart.mlp-math.utils';
export { trainMlpBackprop } from './enemy-warmstart.backprop.utils';
export {
  buildNeatensteinCurriculum,
  getCurriculumCaseWeights,
  type CurriculumCase,
} from './enemy-warmstart.curriculum.utils';

// ---------------------------------------------------------------------------
// Warm-start weight generation
// ---------------------------------------------------------------------------

// Re-export warm-start hyperparameter constants for consumer convenience.
export {
  TEMPLATE_INIT_SCALE,
  VARIANT_NOISE_STDDEV,
  WARMSTART_LEARNING_RATE,
  WARMSTART_ITERATIONS,
} from '../warmstart.constants';

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
