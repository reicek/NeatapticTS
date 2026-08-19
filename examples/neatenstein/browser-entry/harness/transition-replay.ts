/**
 * Transition-level experience replay for Lamarckian weight updates in the
 * Neatenstein co-evolution harness.
 *
 * Provides per-enemy transition buffers, prioritized death replay by
 * "surprise" score, and CERL-style shared replay for the main agent.
 *
 * Stub module — implementation lands in Step B4 (04-implementing).
 * Tests in `transition-replay.test.ts` define the RED contracts.
 *
 * @module
 */

/**
 * Transition-level experience replay, prioritized death replay, and
 * CERL-style shared replay for the main agent.
 *
 * Provides bounded FIFO transition buffers, Lamarckian backprop-style weight
 * updates from replay samples, death-surprise scoring for prioritized replay,
 * and a shared replay buffer for warm-starting hero variants.
 *
 * @module
 */

/** Fixed MLP weight count for the 6→6→4→4 topology (90 parameters). */
const MLP_WEIGHT_COUNT = 6 * 6 + 6 * 4 + 4 * 4 + 6 + 4 + 4;

/** Input dimension for the MLP. */
const MLP_INPUT_SIZE = 6;

/** Output dimension for the MLP. */
const MLP_OUTPUT_SIZE = 4;

/** Hidden layer 1 size. */
const MLP_HIDDEN1_SIZE = 6;

/** Hidden layer 2 size. */
const MLP_HIDDEN2_SIZE = 4;

/** Weight offset for layer 1 (input → hidden1). */
const W1_OFFSET = 0;
/** Weight offset for layer 1 biases. */
const B1_OFFSET = MLP_HIDDEN1_SIZE * MLP_INPUT_SIZE;
/** Weight offset for layer 2 (hidden1 → hidden2). */
const W2_OFFSET = B1_OFFSET + MLP_HIDDEN1_SIZE;
/** Weight offset for layer 2 biases. */
const B2_OFFSET = W2_OFFSET + MLP_HIDDEN2_SIZE * MLP_HIDDEN1_SIZE;
/** Weight offset for layer 3 (hidden2 → output). */
const W3_OFFSET = B2_OFFSET + MLP_HIDDEN2_SIZE;
/** Weight offset for layer 3 biases. */
const B3_OFFSET = W3_OFFSET + MLP_OUTPUT_SIZE * MLP_HIDDEN2_SIZE;

/**
 * Interface for a transition stored in the replay buffer.
 */
export interface Transition {
  /** Tick at which the transition occurred. */
  tick: number;
  /** Input observation. */
  input: Float32Array;
  /** Output action. */
  output: Float32Array;
  /** Reward received. */
  reward: number;
  /** Whether the transition was terminal. */
  done: boolean;
  /** Optional variant ID for shared replay. */
  variantId?: number;
}

/**
 * Bounded FIFO transition buffer.
 */
export interface TransitionBuffer {
  /** Pushes a transition into the buffer, evicting the oldest if full. */
  push: (transition: Transition) => void;
  /** Returns the current number of stored transitions. */
  size: () => number;
  /**
   * Samples up to `count` random transitions from the buffer.
   *
   * @param count The number of transitions to sample.
   * @param rng Optional seeded RNG returning [0, 1). When omitted, uses
   *   `Math.random` (non-deterministic).
   */
  sample: (count: number, rng?: () => number) => Transition[];
}

/**
 * Creates a bounded FIFO transition buffer with the given capacity.
 *
 * @param capacity Maximum number of transitions to store.
 * @returns A transition buffer with push, size, and sample methods.
 */
export function createTransitionBuffer(capacity: number): TransitionBuffer {
  const data: Transition[] = [];
  return {
    push(transition: Transition): void {
      data.push(transition);
      while (data.length > capacity) {
        data.shift();
      }
    },
    size(): number {
      return data.length;
    },
    sample(count: number, rng?: () => number): Transition[] {
      if (data.length === 0) return [];
      const rand = rng ?? Math.random;
      const result: Transition[] = [];
      const maxSamples = Math.min(count, data.length);
      for (let i = 0; i < maxSamples; i++) {
        const idx = Math.floor(rand() * data.length);
        result.push(data[idx]);
      }
      return result;
    },
  };
}

/**
 * Creates a mulberry32 deterministic RNG from a seed.
 *
 * @param seed The random seed.
 * @returns A function returning deterministic floats in [0, 1).
 */
function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Performs a simple forward pass through the MLP and computes gradients
 * with respect to the weights using a basic backpropagation step.
 *
 * This is a simplified Lamarckian update: it computes the MSE loss between
 * the network output and the target output, then applies gradient descent.
 *
 * @param weights The current weight vector (modified in place).
 * @param input The input observation.
 * @param target The target output.
 * @param learningRate The learning rate for the update.
 */
function backpropStep(
  weights: Float32Array,
  input: Float32Array,
  target: Float32Array,
  learningRate: number,
): void {
  // Forward pass
  const h1 = new Float32Array(MLP_HIDDEN1_SIZE);
  const h2 = new Float32Array(MLP_HIDDEN2_SIZE);
  const out = new Float32Array(MLP_OUTPUT_SIZE);

  // Layer 1: input → hidden1
  for (let j = 0; j < MLP_HIDDEN1_SIZE; j++) {
    let sum = weights[B1_OFFSET + j];
    for (let i = 0; i < MLP_INPUT_SIZE; i++) {
      sum += input[i] * weights[W1_OFFSET + j * MLP_INPUT_SIZE + i];
    }
    h1[j] = Math.tanh(sum);
  }

  // Layer 2: hidden1 → hidden2
  for (let j = 0; j < MLP_HIDDEN2_SIZE; j++) {
    let sum = weights[B2_OFFSET + j];
    for (let i = 0; i < MLP_HIDDEN1_SIZE; i++) {
      sum += h1[i] * weights[W2_OFFSET + j * MLP_HIDDEN1_SIZE + i];
    }
    h2[j] = Math.tanh(sum);
  }

  // Layer 3: hidden2 → output
  for (let j = 0; j < MLP_OUTPUT_SIZE; j++) {
    let sum = weights[B3_OFFSET + j];
    for (let i = 0; i < MLP_HIDDEN2_SIZE; i++) {
      sum += h2[i] * weights[W3_OFFSET + j * MLP_HIDDEN2_SIZE + i];
    }
    out[j] = sum;
  }

  // Backward pass (gradient descent on MSE loss)
  const dOut = new Float32Array(MLP_OUTPUT_SIZE);
  for (let j = 0; j < MLP_OUTPUT_SIZE; j++) {
    dOut[j] = (out[j] - target[j]) * 2.0;
  }

  // Layer 3 gradients
  const dH2 = new Float32Array(MLP_HIDDEN2_SIZE);
  for (let j = 0; j < MLP_OUTPUT_SIZE; j++) {
    weights[B3_OFFSET + j] -= learningRate * dOut[j];
    for (let i = 0; i < MLP_HIDDEN2_SIZE; i++) {
      const grad = dOut[j] * h2[i];
      dH2[i] += dOut[j] * weights[W3_OFFSET + j * MLP_HIDDEN2_SIZE + i];
      weights[W3_OFFSET + j * MLP_HIDDEN2_SIZE + i] -= learningRate * grad;
    }
  }

  // Layer 2 gradients
  const dH1 = new Float32Array(MLP_HIDDEN1_SIZE);
  for (let j = 0; j < MLP_HIDDEN2_SIZE; j++) {
    const dh2 = dH2[j] * (1 - h2[j] * h2[j]);
    weights[B2_OFFSET + j] -= learningRate * dh2;
    for (let i = 0; i < MLP_HIDDEN1_SIZE; i++) {
      const grad = dh2 * h1[i];
      dH1[i] += dh2 * weights[W2_OFFSET + j * MLP_HIDDEN1_SIZE + i];
      weights[W2_OFFSET + j * MLP_HIDDEN1_SIZE + i] -= learningRate * grad;
    }
  }

  // Layer 1 gradients
  for (let j = 0; j < MLP_HIDDEN1_SIZE; j++) {
    const dh1 = dH1[j] * (1 - h1[j] * h1[j]);
    weights[B1_OFFSET + j] -= learningRate * dh1;
    for (let i = 0; i < MLP_INPUT_SIZE; i++) {
      const grad = dh1 * input[i];
      weights[W1_OFFSET + j * MLP_INPUT_SIZE + i] -= learningRate * grad;
    }
  }
}

/**
 * Configuration for runReplayUpdates.
 */
export interface ReplayUpdateConfig {
  /** Initial weight vector. */
  weights: Float32Array;
  /** Transition buffer to sample from. */
  buffer: TransitionBuffer;
  /** Number of backprop steps to run. */
  steps: number;
  /** Learning rate. */
  learningRate: number;
  /** Random seed for deterministic sampling. */
  seed: number;
}

/**
 * Result of runReplayUpdates.
 */
export interface ReplayUpdateResult {
  /** Updated weight vector after replay. */
  weights: Float32Array;
}

/**
 * Runs N backprop steps on transitions sampled from the buffer and returns
 * the updated weights.
 *
 * @param config Configuration with weights, buffer, steps, learningRate, and seed.
 * @returns Result containing the updated weights.
 */
export function runReplayUpdates(
  config: ReplayUpdateConfig,
): ReplayUpdateResult {
  const { weights, buffer, steps, learningRate, seed } = config;
  const result = new Float32Array(weights);
  const rng = createRng(seed);
  const bufferSize = buffer.size();
  for (let s = 0; s < steps; s++) {
    if (bufferSize === 0) break;
    const samples = buffer.sample(1, rng);
    const transition = samples[0];
    if (transition) {
      backpropStep(result, transition.input, transition.output, learningRate);
    }
  }
  return { weights: result };
}

/**
 * Configuration for computeDeathSurprise.
 */
export interface DeathSurpriseConfig {
  /** Number of ticks the agent survived before death. */
  survivalTicks: number;
  /** Health remaining at the time of death (0–100). */
  healthAtDeath: number;
  /** Expected movement direction (unused in basic formulation). */
  expectedDirection: number;
}

/** Scale factor for survival ticks in the surprise formula. */
const SURVIVAL_TAU = 100;

/** Maximum health value for normalization. */
const MAX_HEALTH = 100;

/**
 * Computes a death surprise score in (0, 1]. High surprise results from quick
 * death with high remaining health; low surprise from long survival with low
 * health.
 *
 * Formula: `surprise = clamp(healthAtDeath / 100 * exp(-survivalTicks / 100), 0, 1)`
 *
 * @param config Configuration with survivalTicks, healthAtDeath, and expectedDirection.
 * @returns Surprise score in (0, 1].
 */
export function computeDeathSurprise(config: DeathSurpriseConfig): number {
  const healthRatio = Math.max(
    0,
    Math.min(1, config.healthAtDeath / MAX_HEALTH),
  );
  const survivalFactor = Math.exp(-config.survivalTicks / SURVIVAL_TAU);
  const surprise = healthRatio * survivalFactor;
  return Math.max(0, Math.min(1, surprise));
}

/**
 * Configuration for computeReplayPressureFromSurprise.
 */
export interface ReplayPressureConfig {
  /** The death surprise score. */
  surpriseScore: number;
  /** The maximum surprise observed (for normalization). */
  maxSurprise: number;
}

/**
 * Computes replay pressure from a surprise score, normalized by the maximum
 * observed surprise and clamped to [0, 1].
 *
 * @param config Configuration with surpriseScore and maxSurprise.
 * @returns Replay pressure in [0, 1].
 */
export function computeReplayPressureFromSurprise(
  config: ReplayPressureConfig,
): number {
  if (config.maxSurprise <= 0) return 0;
  const pressure = config.surpriseScore / config.maxSurprise;
  return Math.max(0, Math.min(1, pressure));
}

/**
 * Shared replay buffer interface (same shape as TransitionBuffer but with
 * variant tagging support).
 */
export interface SharedReplayBuffer {
  /** Pushes a transition (optionally tagged with variantId) into the buffer. */
  push: (transition: Transition) => void;
  /** Returns the current number of stored transitions. */
  size: () => number;
  /**
   * Samples up to `count` random transitions from the buffer.
   *
   * @param count The number of transitions to sample.
   * @param rng Optional seeded RNG returning [0, 1). When omitted, uses
   *   `Math.random` (non-deterministic).
   */
  sample: (count: number, rng?: () => number) => Transition[];
}

/**
 * Creates a CERL-style shared replay buffer that all hero variants contribute
 * to. Transitions can be tagged with `variantId` to distinguish contributors.
 *
 * @param capacity Maximum number of transitions to store.
 * @returns A shared replay buffer.
 */
export function createSharedReplayBuffer(capacity: number): SharedReplayBuffer {
  const data: Transition[] = [];
  return {
    push(transition: Transition): void {
      data.push(transition);
      while (data.length > capacity) {
        data.shift();
      }
    },
    size(): number {
      return data.length;
    },
    sample(count: number, rng?: () => number): Transition[] {
      if (data.length === 0) return [];
      const rand = rng ?? Math.random;
      const result: Transition[] = [];
      const maxSamples = Math.min(count, data.length);
      for (let i = 0; i < maxSamples; i++) {
        const idx = Math.floor(rand() * data.length);
        result.push(data[idx]);
      }
      return result;
    },
  };
}

/**
 * Configuration for warmStartFromSharedReplay.
 */
export interface WarmStartConfig {
  /** The shared replay buffer to sample from. */
  sharedBuffer: SharedReplayBuffer;
  /** Number of backprop steps to run. */
  steps: number;
  /** Learning rate. */
  learningRate: number;
  /** Random seed for deterministic sampling. */
  seed: number;
}

/**
 * Produces warm-started weights from shared replay samples by running
 * backprop steps on transitions drawn from the shared buffer.
 *
 * @param config Configuration with sharedBuffer, steps, learningRate, and seed.
 * @returns Warm-started weight vector.
 */
export function warmStartFromSharedReplay(
  config: WarmStartConfig,
): Float32Array {
  const { sharedBuffer, steps, learningRate, seed } = config;
  const weights = new Float32Array(MLP_WEIGHT_COUNT);
  const rng = createRng(seed);
  const bufferSize = sharedBuffer.size();
  for (let s = 0; s < steps; s++) {
    if (bufferSize === 0) break;
    const samples = sharedBuffer.sample(1, rng);
    const transition = samples[0];
    if (transition) {
      backpropStep(weights, transition.input, transition.output, learningRate);
    }
  }
  return weights;
}
