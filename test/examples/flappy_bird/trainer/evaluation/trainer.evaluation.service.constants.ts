/**
 * Fallback score assigned to genomes that have not yet been evaluated.
 *
 * Using negative infinity guarantees unevaluated genomes lose any ranking tie
 * against genomes that already have real aggregate results.
 */
export const FLAPPY_TRAINER_NEGATIVE_INFINITY_SCORE = Number.NEGATIVE_INFINITY;

/**
 * Minimum pipe-progress baseline used when no aggregates are available.
 *
 * This keeps early-stage aggregate scoring well-defined even before any genome
 * has established meaningful pipe progress.
 */
export const FLAPPY_TRAINER_MIN_PIPE_PROGRESS = 0;
