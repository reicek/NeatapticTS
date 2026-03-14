/**
 * Deterministic RNG support for the NEAT controller.
 *
 * The root RNG entrypoint stays small on purpose: it tells readers that random
 * state handling has two educational layers underneath it.
 *
 * - `core/` explains the xorshift-based random stream and seed lifecycle.
 * - `facade/` explains the stable `Neat` methods used by tests, replay, and
 *   diagnostics.
 */
export * from './core/rng.constants';
export * from './core/rng.types';
export * from './core/rng.utils';