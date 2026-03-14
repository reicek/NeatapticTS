/**
 * Stable tuning values consumed by the ASCII maze evolution facade.
 *
 * The public EvolutionEngine class should read as orchestration-first code.
 * These constants live in a dedicated module so warm-start tuning, loop
 * thresholds, and shared fallback arrays do not crowd the facade itself.
 */

/**
 * Shared empty vector reused when engine helpers need a stable array fallback.
 *
 * @example
 * const outgoing = node.connections?.out ?? EVOLUTION_ENGINE_EMPTY_VECTOR;
 */
export const EVOLUTION_ENGINE_EMPTY_VECTOR: unknown[] = [];

/**
 * Number of action outputs emitted by the ASCII maze policy network.
 *
 * @example
 * console.log(EVOLUTION_ENGINE_ACTION_DIMENSION); // 4
 */
export const EVOLUTION_ENGINE_ACTION_DIMENSION = 4;

/**
 * Initial ring-buffer capacity used for logits telemetry.
 *
 * The ring may grow at runtime, but the facade starts from this size so the
 * first generations stay allocation-light.
 */
export const EVOLUTION_ENGINE_INITIAL_LOGITS_RING_CAPACITY = 512;

/**
 * Hard safety limit for the logits telemetry ring-buffer capacity.
 */
export const EVOLUTION_ENGINE_MAX_LOGITS_RING_CAPACITY = 8_192;

/**
 * Warm-start curriculum samples used to bias early supervised guidance.
 *
 * These values shape the synthetic targets used before the main NEAT loop
 * takes over, so they are grouped here instead of being scattered across the
 * facade method body.
 */
export const EVOLUTION_ENGINE_WARM_START_CONSTANTS = {
  TRAIN_OUT_PROB_HIGH: 0.92,
  TRAIN_OUT_PROB_LOW: 0.02,
  PROGRESS_MEDIUM: 0.7,
  PROGRESS_STRONG: 0.9,
  PROGRESS_JUNCTION: 0.6,
  PROGRESS_FOURWAY: 0.55,
  PROGRESS_REGRESS: 0.4,
  PROGRESS_MILD_REGRESS: 0.45,
  PROGRESS_MIN_SIGNAL: 0.001,
  DEFAULT_JITTER_PROB: 0.25,
  AUGMENT_JITTER_BASE: 0.95,
  AUGMENT_JITTER_RANGE: 0.05,
  AUGMENT_PROGRESS_JITTER_PROB: 0.35,
  AUGMENT_PROGRESS_DELTA_RANGE: 0.1,
  AUGMENT_PROGRESS_DELTA_HALF: 0.05,
} as const;

/**
 * Pretraining controls used by the Lamarckian warm-start helpers.
 *
 * @example
 * const iterations = EVOLUTION_ENGINE_PRETRAIN_CONSTANTS.PRETRAIN_MAX_ITER;
 */
export const EVOLUTION_ENGINE_PRETRAIN_CONSTANTS = {
  PRETRAIN_MAX_ITER: 60,
  PRETRAIN_BASE_ITER: 8,
  DEFAULT_TRAIN_ERROR: 0.01,
  DEFAULT_PRETRAIN_RATE: 0.002,
  DEFAULT_PRETRAIN_MOMENTUM: 0.1,
  DEFAULT_TRAIN_BATCH_SMALL: 2,
} as const;

/**
 * Main-loop tuning values passed into the extracted evolution-loop helpers.
 *
 * This table keeps the public facade declarative while preserving the same
 * runtime thresholds and training behaviour.
 */
export const EVOLUTION_ENGINE_LOOP_CONSTANTS = {
  DEFAULT_TRAIN_ERROR: 0.01,
  DEFAULT_TRAIN_RATE: 0.001,
  DEFAULT_TRAIN_MOMENTUM: 0.2,
  DEFAULT_TRAIN_BATCH_SMALL: 2,
  DEFAULT_TRAIN_BATCH_LARGE: 20,
  FITTEST_TRAIN_ITERATIONS: 1_000,
  SATURATION_PRUNE_THRESHOLD: 0.5,
  RECENT_WINDOW: 40,
  DEFAULT_STD_SMALL: 0.25,
  DEFAULT_STD_ADJUST_MULT: 0.7,
} as const;