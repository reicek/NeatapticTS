import type { NeatControllerForEvolution } from '../evolve.types';

/**
 * The evolve-time adaptive bridge coordinates optional controllers and
 * post-mutation maintenance inside one finished generation step.
 *
 * The root `adaptive/` chapter explains the full catalog of adaptive policies:
 * complexity budgeting, phased growth, acceptance pressure, lineage feedback,
 * mutation tuning, and operator-stat decay. This file owns the narrower evolve
 * question: when one generation is already in flight, which of those policies
 * should be invoked now, which ones must remain best-effort, and which cleanup
 * steps need to happen after mutation so the next generation starts from a
 * coherent runtime state?
 *
 * Read this chapter when you want to understand:
 *
 * - why evolve keeps optional adaptive imports behind safe bridge helpers,
 * - which adaptive controllers run before or after population mutation,
 * - how auto-compatibility tuning stays close to evolve instead of the broader
 *   adaptive root chapter,
 * - why cache invalidation and re-enable adaptation belong beside mutation-time
 *   maintenance rather than inside the mutation chapter itself.
 *
 * The helper flow is easiest to retain as four responsibilities:
 *
 * 1. invoke optional controller-level adaptation without breaking narrower runtimes,
 * 2. retune compatibility pressure when species counts drift,
 * 3. run pruning and mutation as one mutation-phase maintenance block,
 * 4. repair mutation-side runtime state such as compatibility caches and
 *    re-enable probabilities before the next generation is evaluated.
 *
 * ```mermaid
 * flowchart TD
 *   Ranked[Ranked current generation] --> Controllers[Run optional adaptive controllers]
 *   Controllers --> Compat[Retune compatibility coefficients when enabled]
 *   Compat --> Mutate[Apply pruning and mutation]
 *   Mutate --> Repair[Adapt re-enable probability and clear compat caches]
 *   Repair --> Ready[Next generation runtime state ready]
 * ```
 */

/* Module introduction boundary for generated README output. */

/**
 * Apply adaptive complexity controllers if available.
 *
 * This helper is the evolve bridge into the root adaptive complexity policies.
 * It keeps both optional calls together because they rewrite controller-level
 * structure policy rather than one single genome: budget scheduling can change
 * allowed network size, and phased complexity can flip the controller between
 * growth and simplification modes.
 *
 * @param internal - NEAT controller instance.
 * @returns A promise that resolves after optional complexity controllers have run.
 */
export async function applyAdaptiveComplexityControllers(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Apply complexity budget controller (optional).
  try {
    const { applyComplexityBudget } = await import('../../adaptive/adaptive');
    applyComplexityBudget.call(internal as never);
  } catch {
    // Empty catch: adaptive complexity budget may not be configured.
  }
  // Step 2: Apply phased complexity controller (optional).
  try {
    const { applyPhasedComplexity } = await import('../../adaptive/adaptive');
    applyPhasedComplexity.call(internal as never);
  } catch {
    // Empty catch: phased complexity may not be configured.
  }
}

/**
 * Apply minimal criterion adaptive controller if available.
 *
 * Minimal criterion adaptation is one of the few adaptive policies that can
 * rewrite the current generation's score landscape immediately. Keeping it in a
 * safe wrapper lets evolve apply that pressure when configured without forcing
 * every runtime surface to include the full adaptive subtree.
 *
 * @param internal - NEAT controller instance.
 * @returns A promise that resolves after the optional acceptance controller runs.
 */
export async function applyMinimalCriterionAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Delegate to optional module.
  try {
    const { applyMinimalCriterionAdaptive } =
      await import('../../adaptive/adaptive');
    applyMinimalCriterionAdaptive.call(internal as never);
  } catch {
    // Empty catch: minimal criterion adaptation may not be configured.
  }
}

/**
 * Apply ancestor uniqueness adaptation if available.
 *
 * This helper is the evolve-side bridge from freshly recorded telemetry lineage
 * evidence back into future controller policy. The underlying adaptive rule may
 * change dominance epsilon or lineage-pressure settings, but the bridge itself
 * stays narrow: it only attempts the optional handoff and tolerates runtimes
 * where lineage adaptation is absent.
 *
 * @param internal - NEAT controller instance.
 * @returns A promise that resolves after the optional lineage controller runs.
 */
export async function applyAncestorUniqAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Delegate to adaptive module when present.
  try {
    const adaptiveModule = await import('../../adaptive/adaptive');
    adaptiveModule.applyAncestorUniqAdaptive.call(internal as never);
  } catch {
    // Empty catch: ancestor uniqueness adaptation is optional.
  }
}

/**
 * Apply auto-compatibility tuning if enabled.
 *
 * This helper keeps one small adaptive loop close to evolve because it depends
 * directly on the just-observed species count from the ranked generation. When
 * the observed registry drifts from the target, the helper nudges excess and
 * disjoint coefficients together so the next speciation pass can tighten or
 * loosen the compatibility boundary.
 *
 * The fallback behavior when `error === 0` is also deliberate: instead of doing
 * nothing forever at equilibrium, the helper allows a bounded random nudge so
 * the controller can keep exploring nearby coefficient space.
 *
 * @param internal - NEAT controller instance.
 * @param config - Tuning constants.
 * @returns Nothing.
 */
export function applyAutoCompatibilityTuning(
  internal: NeatControllerForEvolution,
  config: {
    targetMin: number;
    adjustRate: number;
    minCoeff: number;
    maxCoeff: number;
    randomScale: number;
  },
): void {
  // Step 1: Guard for missing config.
  try {
    const tuningInputs = resolveAutoCompatibilityTuningInputs(internal, config);
    if (!tuningInputs) return;

    // Step 2: Compute the next shared compatibility multiplier.
    const factor = resolveAutoCompatibilityAdjustmentFactor(
      internal,
      tuningInputs.error,
      tuningInputs.rate,
      config.randomScale,
    );

    // Step 3: Apply the coefficient updates.
    applyAutoCompatibilityCoefficientUpdate(tuningInputs, factor);
  } catch {
    // Empty catch: auto-compatibility tuning is optional.
  }
}

function resolveAutoCompatibilityTuningInputs(
  internal: NeatControllerForEvolution,
  config: {
    targetMin: number;
    adjustRate: number;
    minCoeff: number;
    maxCoeff: number;
    randomScale: number;
  },
):
  | {
      options: NeatControllerForEvolution['options'];
      error: number;
      rate: number;
      minCoeff: number;
      maxCoeff: number;
      excessCoeff: number;
      disjointCoeff: number;
    }
  | undefined {
  const { options } = internal;
  if (!options.autoCompatTuning?.enabled) return undefined;

  const target = resolveAutoCompatibilityTarget(internal, config.targetMin);
  const observedSpeciesCount = (internal._species?.length ?? 0) || 1;
  const minCoeff = options.autoCompatTuning.minCoeff ?? config.minCoeff;

  return {
    options,
    error: target - observedSpeciesCount,
    rate: options.autoCompatTuning.adjustRate ?? config.adjustRate,
    minCoeff,
    maxCoeff: options.autoCompatTuning.maxCoeff ?? config.maxCoeff,
    excessCoeff: options.excessCoeff ?? minCoeff,
    disjointCoeff: options.disjointCoeff ?? minCoeff,
  };
}

function resolveAutoCompatibilityTarget(
  internal: NeatControllerForEvolution,
  targetMin: number,
): number {
  const { options } = internal;
  return (
    options.autoCompatTuning?.target ??
    options.targetSpecies ??
    Math.max(targetMin, Math.round(Math.sqrt(internal.population.length)))
  );
}

function resolveAutoCompatibilityAdjustmentFactor(
  internal: NeatControllerForEvolution,
  error: number,
  rate: number,
  randomScale: number,
): number {
  if (error !== 0) {
    return 1 - rate * Math.sign(error);
  }

  return 1 + (internal._getRNG()() - 0.5) * rate * randomScale;
}

function applyAutoCompatibilityCoefficientUpdate(
  tuningInputs: {
    options: NeatControllerForEvolution['options'];
    minCoeff: number;
    maxCoeff: number;
    excessCoeff: number;
    disjointCoeff: number;
  },
  factor: number,
): void {
  tuningInputs.options.excessCoeff = clampAutoCompatibilityCoefficient(
    tuningInputs.excessCoeff * factor,
    tuningInputs.minCoeff,
    tuningInputs.maxCoeff,
  );
  tuningInputs.options.disjointCoeff = clampAutoCompatibilityCoefficient(
    tuningInputs.disjointCoeff * factor,
    tuningInputs.minCoeff,
    tuningInputs.maxCoeff,
  );
}

function clampAutoCompatibilityCoefficient(
  value: number,
  minCoeff: number,
  maxCoeff: number,
): number {
  return Math.min(maxCoeff, Math.max(minCoeff, value));
}

/**
 * Apply pruning and mutation phases.
 *
 * Mutation-time maintenance is grouped here because evolve needs one compact
 * place where structural removal, adaptive pruning, baseline mutation, and
 * optional adaptive mutation all happen in the intended order. The helper does
 * not decide whether the population should be rebuilt; it assumes the next
 * generation already exists and now needs its structural edits.
 *
 * Example:
 *
 * ```ts
 * await applyPruningAndMutation(internal);
 * invalidateCompatibilityCaches(internal);
 * ```
 *
 * @param internal - NEAT controller instance.
 * @returns A promise that resolves after mutation-phase maintenance finishes.
 */
export async function applyPruningAndMutation(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Apply evolution-time pruning.
  try {
    const pruningModule = await import('../../pruning/pruning');
    pruningModule.applyEvolutionPruning.call(internal as never);
  } catch {
    // Empty catch: evolution-time pruning is optional.
  }
  // Step 2: Apply adaptive pruning.
  try {
    const pruningModule = await import('../../pruning/pruning');
    pruningModule.applyAdaptivePruning.call(internal as never);
  } catch {
    // Empty catch: adaptive pruning is optional.
  }
  // Step 3: Prepare generation-scoped innovation tracking for the new population.
  internal._prepareInnovationTrackerGeneration?.(internal.generation + 1);
  // Step 4: Mutate population.
  await internal.mutate?.();
  // Step 5: Apply adaptive mutation if available.
  try {
    const adaptiveModule = await import('../../adaptive/adaptive');
    adaptiveModule.applyAdaptiveMutation.call(internal as never);
  } catch {
    // Empty catch: genome-level adaptive mutation is optional.
  }
}

/**
 * Invalidate compatibility caches after mutations.
 *
 * Structural mutation can make cached compatibility comparisons stale.
 * Clearing those caches here ensures later speciation and distance reads are
 * recomputed from the post-mutation topology instead of reusing scores from the
 * previous generation.
 *
 * @param internal - NEAT controller instance.
 * @returns Nothing.
 */
export function invalidateCompatibilityCaches(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Remove cached compatibility on each genome.
  internal.population.forEach((genome) => {
    if (genome._compatCache) delete genome._compatCache;
  });
}

/**
 * Adapt the re-enable probability based on recent success ratios.
 *
 * Re-enable adaptation turns the last generation's connection-revival outcomes
 * into one controller-level probability update for the next generation. It
 * aggregates success and attempt counters across the whole population, resets
 * those per-genome counters once consumed, and only adjusts the shared
 * probability when the sample size is large enough to be meaningful.
 *
 * @param internal - NEAT controller instance.
 * @param config - Adaptation thresholds and bounds for re-enable probability.
 * @returns Nothing.
 */
export function adaptReenableProbability(
  internal: NeatControllerForEvolution,
  config: {
    minSamples: number;
    target: number;
    min: number;
    max: number;
    deltaScale: number;
  },
): void {
  // Step 1: Guard for disabled configuration.
  if (internal.options.reenableProb === undefined) return;
  // Step 2: Aggregate success and attempt counters.
  let reenableSuccessTotal = 0;
  let reenableAttemptsTotal = 0;
  for (const genome of internal.population) {
    reenableSuccessTotal += genome._reenableSuccess ?? 0;
    reenableAttemptsTotal += genome._reenableAttempts ?? 0;
    genome._reenableSuccess = 0;
    genome._reenableAttempts = 0;
  }
  // Step 3: Adjust only with sufficient sample size.
  if (reenableAttemptsTotal > config.minSamples) {
    const ratio = reenableSuccessTotal / reenableAttemptsTotal;
    const delta = ratio - config.target;
    internal.options.reenableProb = Math.min(
      config.max,
      Math.max(
        config.min,
        (internal.options.reenableProb ?? config.target) -
          delta * config.deltaScale,
      ),
    );
  }
}

/**
 * Apply operator adaptation if available.
 *
 * Operator adaptation is another best-effort policy-maintenance bridge. It
 * decays long-running operator statistics so later mutation choices weight more
 * recent evidence without forcing evolve to know the details of the adaptive
 * operator-selection subsystem.
 *
 * @param internal - NEAT controller instance.
 * @returns A promise that resolves after optional operator-stat decay runs.
 */
export async function applyOperatorAdaptationSafe(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Delegate to adaptive module when present.
  try {
    const adaptiveModule = await import('../../adaptive/adaptive');
    adaptiveModule.applyOperatorAdaptation.call(internal as never);
  } catch {
    // Empty catch: operator adaptation is optional.
  }
}
