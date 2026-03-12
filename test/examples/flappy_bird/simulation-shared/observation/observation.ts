/**
 * Shared observation public entry.
 *
 * This focused boundary keeps observation feature assembly separate from
 * observation-vector projection so browser, environment, and worker callers can
 * depend on a smaller, clearer public surface.
 *
 * Conceptually, this folder is the "state representation" layer for the
 * Flappy Bird example. Instead of learning directly from pixels, the agent sees
 * a compact set of geometric and kinematic signals derived from the current
 * world state. That makes the example easier to study, faster to evaluate, and
 * closer to classic feature-engineering pipelines used in small control tasks.
 *
 * If you want background reading, the Wikipedia articles on "feature
 * engineering" and "state space representation" give useful intuition for why
 * this boundary exists at all.
 */
export * from './observation.features.utils';
export * from './observation.vector.utils';
