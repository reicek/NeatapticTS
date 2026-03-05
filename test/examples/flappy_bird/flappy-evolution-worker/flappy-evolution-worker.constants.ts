/** Synthetic sample count used for generation-0 warm-start pretraining. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT = 512;

/** Visible world width used when generating synthetic warm-start samples. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX = 720;

/** Optimizer iteration budget for generation-0 warm-start pretraining. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS = 60;

/** Batch size for generation-0 warm-start pretraining. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE = 32;

/** Learning rate for generation-0 warm-start pretraining. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_RATE = 0.02;

/** Gaussian stddev used for post-pretrain connection-weight diversification. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV = 0.08;

/** Gaussian stddev used for post-pretrain node-bias diversification. */
export const FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV = 0.03;
