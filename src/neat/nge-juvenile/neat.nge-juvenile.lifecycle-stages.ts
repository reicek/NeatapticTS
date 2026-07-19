/**
 * NGE lifecycle stage resolution for baby/juvenile/adult phases.
 *
 * The NGE lifecycle progresses through discrete stages: embryo → baby →
 * juvenile → adult → equilibrium. Each stage carries different defaults for
 * growth cadence, stabilization intensity, and mutation magnitude, reflecting
 * the explore–exploit tradeoff that governs brain-like neuro-evolution.
 *
 * - **Baby** (≤1k nodes): aggressive growth, high variant count (16), high
 *   mutation magnitude, low stabilization. The network is small enough that
 *   many weight variants can be evaluated in parallel.
 * - **Juvenile** (1k–4k nodes): transitioning. Growth cadence and mutation
 *   magnitude ramp down from baby to adult levels. Variant count ramps from
 *   16 to 2.
 * - **Adult** (>4k nodes): stability-focused. Low growth cadence, low mutation
 *   magnitude, high stabilization intensity, and only 2 variants evaluated
 *   to preserve real-time performance.
 *
 * All numeric thresholds and magnitudes are **config-overridable defaults**.
 * Core code reads values from the config first, falling back to documented
 * constants in `neat.nge-juvenile.constants.ts`. No policy value is ever
 * hard-coded.
 *
 * ## Determinism
 *
 * All resolve functions are pure: same inputs produce the same outputs with
 * no side effects or randomness. The lifecycle stage resolution is deterministic
 * given a fixed node count and config.
 *
 * ## Background
 *
 * The explore–exploit tradeoff that drives these stage-based parameters is
 * inspired by developmental neuroscience and reinforcement learning:
 * - [Wikipedia — Exploration vs exploitation](https://en.wikipedia.org/wiki/Exploration_exploitation_dilemma)
 * - The lifecycle stage model mirrors developmental phases in biological
 *   neural networks where early plasticity decreases and stability increases
 *   with maturation.
 *
 * @example
 * Resolve the lifecycle stage and parameters for a 2,500-node network.
 * ```ts
 * import { resolveLifecycleStage, resolveGrowthCadence } from 'neataptic';
 *
 * const stage = resolveLifecycleStage(2_500, {});
 * // stage === 'juvenile'
 *
 * const cadence = resolveGrowthCadence(stage, {});
 * // cadence ≈ 0.5 (midpoint of baby 0.8 and adult 0.2)
 * ```
 */

import type {
  NgeLifecycleStage,
  NgeLifecycleStageConfig,
} from './neat.nge-juvenile.types';

export type {
  NgeLifecycleStage,
  NgeLifecycleStageConfig,
} from './neat.nge-juvenile.types';
import {
  NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE,
  NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY,
  NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE,
  NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD,
  NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY,
  NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD,
  NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT,
} from './neat.nge-juvenile.constants';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Resolve the effective baby node threshold from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective baby node threshold.
 */
function resolveBabyNodeThreshold(config: NgeLifecycleStageConfig): number {
  return config.babyNodeThreshold ?? NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD;
}

/**
 * Resolve the effective juvenile node threshold from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective juvenile node threshold.
 */
function resolveJuvenileNodeThreshold(config: NgeLifecycleStageConfig): number {
  return (
    config.juvenileNodeThreshold ??
    NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD
  );
}

/**
 * Resolve the effective baby variant count from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective baby variant count.
 */
function resolveBabyVariantCount(config: NgeLifecycleStageConfig): number {
  return config.babyVariantCount ?? NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT;
}

/**
 * Resolve the effective adult variant count from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective adult variant count.
 */
function resolveAdultVariantCount(config: NgeLifecycleStageConfig): number {
  return config.adultVariantCount ?? NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT;
}

/**
 * Resolve the effective juvenile variant count from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective juvenile variant count.
 */
function resolveJuvenileVariantCount(config: NgeLifecycleStageConfig): number {
  return (
    config.juvenileVariantCount ?? NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT
  );
}

/**
 * Resolve the effective baby growth cadence from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective baby growth cadence.
 */
function resolveBabyGrowthCadence(config: NgeLifecycleStageConfig): number {
  return config.babyGrowthCadence ?? NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE;
}

/**
 * Resolve the effective adult growth cadence from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective adult growth cadence.
 */
function resolveAdultGrowthCadence(config: NgeLifecycleStageConfig): number {
  return (
    config.adultGrowthCadence ?? NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE
  );
}

/**
 * Resolve the effective baby stabilization intensity from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective baby stabilization intensity.
 */
function resolveBabyStabilizationIntensity(
  config: NgeLifecycleStageConfig,
): number {
  return (
    config.babyStabilizationIntensity ??
    NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY
  );
}

/**
 * Resolve the effective adult stabilization intensity from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective adult stabilization intensity.
 */
function resolveAdultStabilizationIntensity(
  config: NgeLifecycleStageConfig,
): number {
  return (
    config.adultStabilizationIntensity ??
    NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY
  );
}

/**
 * Resolve the effective baby mutation magnitude from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective baby mutation magnitude.
 */
function resolveBabyMutationMagnitude(config: NgeLifecycleStageConfig): number {
  return (
    config.babyMutationMagnitude ??
    NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE
  );
}

/**
 * Resolve the effective adult mutation magnitude from config or default.
 *
 * @param config - Optional config overrides.
 * @returns Effective adult mutation magnitude.
 */
function resolveAdultMutationMagnitude(
  config: NgeLifecycleStageConfig,
): number {
  return (
    config.adultMutationMagnitude ??
    NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE
  );
}

/**
 * Compute the midpoint between two numbers.
 *
 * @param low - The lower value.
 * @param high - The higher value.
 * @returns The arithmetic mean of `low` and `high`.
 */
function midpoint(low: number, high: number): number {
  return (low + high) / 2;
}

/**
 * Linearly interpolate between two values at a given ratio.
 *
 * @param from - The start value (at ratio 0).
 * @param to - The end value (at ratio 1).
 * @param ratio - Interpolation ratio in [0, 1].
 * @returns The interpolated value, rounded to the nearest integer.
 */
function lerp(from: number, to: number, ratio: number): number {
  return Math.round(from - ratio * (from - to));
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Resolve the number of weight variants to evaluate for a given network size.
 *
 * Returns the baby-stage variant count (default 16) for networks at or below
 * the baby node threshold (default 1,000), ramps linearly to the adult-stage
 * variant count (default 2) between the baby and juvenile thresholds
 * (1k–4k), and returns the adult-stage variant count (default 2) for
 * networks above the juvenile node threshold (default 4,000).
 *
 * All thresholds and counts are config-overridable via
 * {@link NgeLifecycleStageConfig}.
 *
 * @param nodeCount - Current network node count.
 * @param config - Optional config overrides; all fields are optional.
 * @returns Number of weight variants to evaluate.
 *
 * @example
 * ```ts
 * const count = resolveVariantCount(500, {}); // 16 (baby stage)
 * const ramped = resolveVariantCount(2_500, {}); // 8 (juvenile ramp midpoint)
 * const adult = resolveVariantCount(5_000, {}); // 2 (adult stage)
 * const custom = resolveVariantCount(500, { babyVariantCount: 24 }); // 24
 * ```
 */
export function resolveVariantCount(
  nodeCount: number,
  config: NgeLifecycleStageConfig,
): number {
  // Step 1: Resolve effective thresholds from config overrides or defaults
  const babyThreshold = resolveBabyNodeThreshold(config);
  const juvenileThreshold = resolveJuvenileNodeThreshold(config);
  const babyVariants = resolveBabyVariantCount(config);
  const adultVariants = resolveAdultVariantCount(config);

  // Step 2: Baby-phase threshold — small networks get maximum variants
  if (nodeCount <= babyThreshold) {
    return babyVariants;
  }

  // Step 3: Adult-phase threshold — large networks get minimum variants
  if (nodeCount > juvenileThreshold) {
    return adultVariants;
  }

  // Step 4: Piecewise-linear ramp with juvenileVariantCount as the midpoint
  const juvenileVariants = resolveJuvenileVariantCount(config);
  const rampMidpoint = midpoint(babyThreshold, juvenileThreshold);

  if (nodeCount <= rampMidpoint) {
    // First half: babyVariants → juvenileVariants
    const ratio = (nodeCount - babyThreshold) / (rampMidpoint - babyThreshold);
    return lerp(babyVariants, juvenileVariants, ratio);
  }

  // Second half: juvenileVariants → adultVariants
  const ratio = (nodeCount - rampMidpoint) / (juvenileThreshold - rampMidpoint);
  return lerp(juvenileVariants, adultVariants, ratio);
}

/**
 * Resolve the lifecycle stage for a given network node count.
 *
 * Returns `'baby'` for networks at or below `babyNodeThreshold` (default 1,000),
 * `'juvenile'` for networks between the baby and juvenile thresholds
 * (1k–4k), and `'adult'` for networks above `juvenileNodeThreshold`
 * (default 4,000).
 *
 * The thresholds are config fields, not hard-coded constants, so callers can
 * shift the stage boundaries to match their runtime or memory constraints.
 *
 * @param nodeCount - Current network node count.
 * @param config - Optional config overrides; all fields are optional.
 * @returns The resolved lifecycle stage.
 *
 * @example
 * ```ts
 * const stage = resolveLifecycleStage(500, {}); // 'baby'
 * const stage = resolveLifecycleStage(2_000, {}); // 'juvenile'
 * const stage = resolveLifecycleStage(5_000, {}); // 'adult'
 * ```
 */
export function resolveLifecycleStage(
  nodeCount: number,
  config: NgeLifecycleStageConfig,
): NgeLifecycleStage {
  const babyThreshold = resolveBabyNodeThreshold(config);
  const juvenileThreshold = resolveJuvenileNodeThreshold(config);

  if (nodeCount <= babyThreshold) {
    return 'baby';
  }

  if (nodeCount <= juvenileThreshold) {
    return 'juvenile';
  }

  return 'adult';
}

/**
 * Resolve the growth cadence for a given lifecycle stage.
 *
 * Baby-stage networks get aggressive growth cadence (default 0.8) to expand
 * structural capacity rapidly. Adult-stage networks get stability-focused
 * cadence (default 0.2). Juvenile-stage networks get the midpoint between
 * baby and adult. Embryo uses baby defaults; equilibrium uses adult defaults.
 *
 * @param stage - The lifecycle stage to resolve cadence for.
 * @param config - Optional config overrides; all fields are optional.
 * @returns The growth cadence value for the stage.
 *
 * @example
 * ```ts
 * const babyCadence = resolveGrowthCadence('baby', {}); // 0.8
 * const adultCadence = resolveGrowthCadence('adult', {}); // 0.2
 * ```
 */
export function resolveGrowthCadence(
  stage: NgeLifecycleStage,
  config: NgeLifecycleStageConfig,
): number {
  const babyCadence = resolveBabyGrowthCadence(config);
  const adultCadence = resolveAdultGrowthCadence(config);

  switch (stage) {
    case 'baby':
    case 'embryo':
      return babyCadence;
    case 'adult':
    case 'equilibrium':
      return adultCadence;
    case 'juvenile':
      return midpoint(adultCadence, babyCadence);
  }
}

/**
 * Resolve the stabilization intensity for a given lifecycle stage.
 *
 * Adult-stage networks get higher stabilization intensity (default 0.7) to
 * focus on extracting maximum performance from current structural capacity.
 * Baby-stage networks get lower intensity (default 0.3) since they prioritize
 * growth over stabilization. Juvenile-stage networks get the midpoint. Embryo
 * uses baby defaults; equilibrium uses adult defaults.
 *
 * @param stage - The lifecycle stage to resolve intensity for.
 * @param config - Optional config overrides; all fields are optional.
 * @returns The stabilization intensity value for the stage.
 *
 * @example
 * ```ts
 * const babyIntensity = resolveStabilizationIntensity('baby', {}); // 0.3
 * const adultIntensity = resolveStabilizationIntensity('adult', {}); // 0.7
 * ```
 */
export function resolveStabilizationIntensity(
  stage: NgeLifecycleStage,
  config: NgeLifecycleStageConfig,
): number {
  const babyIntensity = resolveBabyStabilizationIntensity(config);
  const adultIntensity = resolveAdultStabilizationIntensity(config);

  switch (stage) {
    case 'baby':
    case 'embryo':
      return babyIntensity;
    case 'adult':
    case 'equilibrium':
      return adultIntensity;
    case 'juvenile':
      return midpoint(babyIntensity, adultIntensity);
  }
}

/**
 * Resolve the mutation magnitude for a given lifecycle stage.
 *
 * Baby-stage networks use higher mutation magnitude (default 0.5) for broad
 * exploration of the solution space. Adult-stage networks use lower magnitude
 * (default 0.05) for fine-tuning. Juvenile-stage networks get the midpoint.
 * Embryo uses baby defaults; equilibrium uses adult defaults.
 *
 * @param stage - The lifecycle stage to resolve magnitude for.
 * @param config - Optional config overrides; all fields are optional.
 * @returns The mutation magnitude value for the stage.
 *
 * @example
 * ```ts
 * const babyMag = resolveMutationMagnitude('baby', {}); // 0.5
 * const adultMag = resolveMutationMagnitude('adult', {}); // 0.05
 * ```
 */
export function resolveMutationMagnitude(
  stage: NgeLifecycleStage,
  config: NgeLifecycleStageConfig,
): number {
  const babyMagnitude = resolveBabyMutationMagnitude(config);
  const adultMagnitude = resolveAdultMutationMagnitude(config);

  switch (stage) {
    case 'baby':
    case 'embryo':
      return babyMagnitude;
    case 'adult':
    case 'equilibrium':
      return adultMagnitude;
    case 'juvenile':
      return midpoint(adultMagnitude, babyMagnitude);
  }
}
