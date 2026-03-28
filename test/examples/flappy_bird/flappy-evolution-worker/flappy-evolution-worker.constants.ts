/**
 * Synthetic sample count used for generation-0 warm-start pretraining.
 *
 * Educational note:
 * The warm-start service briefly trains a template network on a heuristic
 * teacher before the first NEAT generation is evolved. This value controls how
 * many synthetic state/action examples are generated for that bootstrap pass.
 * Larger values usually make the teacher signal more stable, but they also
 * increase startup latency inside the worker.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT = 512;

/**
 * Visible world width used when generating synthetic warm-start samples.
 *
 * This should roughly match the browser playback framing so the generated
 * observation vectors look like the states the policy will later see during
 * real worker playback.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX = 720;

/**
 * Optimizer iteration budget for generation-0 warm-start pretraining.
 *
 * The goal is not to fully solve Flappy Bird with supervised learning. The
 * worker only needs a short nudge away from completely random action logits so
 * the first browser-visible generation looks less chaotic.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS = 60;

/**
 * Batch size for generation-0 warm-start pretraining.
 *
 * Smaller batches inject a bit more stochasticity into the bootstrap fit,
 * while still keeping the pass cheap enough for a browser worker.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE = 32;

/**
 * Learning rate for generation-0 warm-start pretraining.
 *
 * This is intentionally moderate: the template network should learn a simple
 * corridor-following prior without overfitting the heuristic teacher.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_RATE = 0.02;

/**
 * Shared-seed batch size used by rollout-guided generation-0 template refinement.
 *
 * After the heuristic teacher fit, the worker evaluates the fixed topology on a
 * few real Flappy rollouts so the warm-start prior is pushed toward trajectories
 * that actually survive the environment instead of only matching the synthetic
 * teacher labels.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_SEED_COUNT = 5;

/**
 * Hill-climb step budget used by rollout-guided generation-0 template refinement.
 *
 * Each step perturbs the current best template, evaluates it on the shared
 * rollout seed batch, and keeps the candidate only when it improves robust
 * fitness. The budget stays intentionally small so worker startup remains fast.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_OPTIMIZATION_STEPS = 12;

/**
 * Initial connection-weight noise scale for rollout-guided template refinement.
 *
 * Early optimization steps search broadly, then later steps cool toward the
 * smaller end scale below for finer local refinement.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_START = 0.12;

/**
 * Final connection-weight noise scale for rollout-guided template refinement.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_WEIGHT_STDDEV_END = 0.015;

/**
 * Initial node-bias noise scale for rollout-guided template refinement.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_START = 0.05;

/**
 * Final node-bias noise scale for rollout-guided template refinement.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_ROLLOUT_BIAS_STDDEV_END = 0.008;

/**
 * Gaussian standard deviation used for post-pretrain connection-weight diversification.
 *
 * After the template network is trained once, each genome receives a noisy copy
 * of its weights. That keeps generation 0 visually coherent while preserving
 * enough diversity for NEAT to search meaningfully.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV = 0.08;

/**
 * Gaussian standard deviation used for post-pretrain node-bias diversification.
 *
 * Bias noise is slightly smaller than weight noise so the warm-start remains a
 * prior, not a rigid clone of the teacher-fitted template.
 */
export const FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV = 0.03;
