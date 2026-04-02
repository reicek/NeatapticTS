/**
 * Flappy Bird example exports.
 *
 * This folder contains a Flappy Bird control environment, a deterministic
 * evaluation pipeline, and a browser/worker playback stack built on top of this
 * repository's NeatapticTS implementation.
 *
 * Educational note:
 * Treat this file as the example's "public shelf." It re-exports the pieces a
 * learner is most likely to reach for first: constants, random utilities,
 * environment stepping, and evaluation helpers.
 *
 * @example
 * ```ts
 * import {
 *   createInitialFlappyState,
 *   createXorshift32,
 *   FLAPPY_DEFAULT_RNG_SEED,
 *   getFlappyObservation,
 * } from './index';
 *
 * const rng = createXorshift32(FLAPPY_DEFAULT_RNG_SEED);
 * const state = createInitialFlappyState(rng);
 * const observation = getFlappyObservation(state);
 * console.log(observation.length);
 * ```
 */

export * from './constants/constants';
export * from './rng';
export * from './flappyEnvironment';
export * from './flappyEvaluation';
