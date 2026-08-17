/**
 * MLP warm-start hyperparameter constants for the Neatenstein enemy population.
 *
 * Extracts the learning rate, iteration count, weight initialization scale,
 * per-variant noise standard deviation, BCE epsilon, mini-batch size, and
 * early-stop loss threshold so that the warm-start modules contain only
 * algorithm logic with no inline magic numbers.
 *
 * @module
 */

// ---------------------------------------------------------------------------
// Template & variant generation constants
// ---------------------------------------------------------------------------

/** Weight initialization scale for the template (avoids tanh saturation). */
export const TEMPLATE_INIT_SCALE = 0.3;

/** Per-variant Gaussian noise standard deviation for weight copying. */
export const VARIANT_NOISE_STDDEV = 0.08;

// ---------------------------------------------------------------------------
// Backprop hyperparameters
// ---------------------------------------------------------------------------

/** Default learning rate for warm-start backprop. */
export const WARMSTART_LEARNING_RATE = 0.7;

/** Default iteration count for warm-start backprop. */
export const WARMSTART_ITERATIONS = 60;

// ---------------------------------------------------------------------------
// Loss & training constants (used in backprop.utils)
// ---------------------------------------------------------------------------

/** Epsilon for binary cross-entropy numerical stability. */
export const BCE_EPSILON = 1e-7;

/** Mini-batch size for stochastic gradient descent. */
export const BCE_BATCH_SIZE = 3;

/** Early-stop loss threshold — training halts when average loss drops below this. */
export const EARLY_STOP_LOSS = 0.001;
