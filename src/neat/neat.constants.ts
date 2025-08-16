/**
 * Shared numerical / heuristic constants for NEAT modules.
 *
 * Keeping these in a single dependency‑free module avoids scattering magic
 * numbers and simplifies tuning while refactoring.
 */

/** Numerical stability offset used inside log / division expressions. */
export const EPSILON = 1e-9; // generic stability epsilon (moderate scale)

/** Extremely small epsilon for log/ratio protections in probability losses. */
export const PROB_EPSILON = 1e-15;

/** Epsilon used in normalization layers (variance smoothing). */
export const NORM_EPSILON = 1e-5;

/** Probability of performing an opportunistic extra ADD_CONN mutation. */
export const EXTRA_CONNECTION_PROBABILITY = 0.5;

// Add new constants above; keep file import‑free for minimal load overhead.
