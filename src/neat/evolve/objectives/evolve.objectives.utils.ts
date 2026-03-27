import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  ObjectiveDescriptor,
} from '../evolve.types';

/**
 * Objective-scheduling and maintenance helpers for NEAT evolution.
 *
 * The public `objectives/` chapter explains how the controller defines the
 * active objective set. This `evolve/objectives/` chapter answers the next
 * question: while generations advance, when should that objective set change,
 * and which controller-owned traces must stay synchronized when it does?
 *
 * This boundary owns the evolve-time policy layer around objectives:
 * - clear the cached resolved objective list when policy changes,
 * - schedule new objectives such as complexity or entropy,
 * - drop and later re-add entropy when stagnation rules say it should leave,
 * - track objective ages and compact importance snapshots for later telemetry,
 * - preserve a small test-only escape hatch for targeted harness scenarios.
 *
 * It stays separate from `objectives/` and `evaluate/objectives/` on purpose.
 * Those chapters define or register objectives. This file reacts to generation
 * progress, stagnation, and runtime policy so the evolve loop can maintain a
 * changing objective set without collapsing that policy into the main
 * orchestration spine.
 *
 * ```mermaid
 * flowchart TD
 *   A[Read current objective keys] --> B[Apply dynamic schedule or auto-entropy fallback]
 *   B --> C[Handle entropy drop or re-add rules]
 *   C --> D[Invalidate objective cache when set changes]
 *   D --> E[Update objective ages and pending add or remove state]
 *   E --> F[Capture optional importance snapshot]
 * ```
 */

/**
 * Clear cached objectives so dynamic schedules can rebuild them.
 *
 * Objective reads are intentionally cached elsewhere because the resolved list
 * is reused across several helpers. Whenever evolve-time policy adds, removes,
 * or suppresses an objective, this cache must be cleared so the next
 * `_getObjectives()` read reflects the updated configuration rather than the
 * earlier generation's list.
 *
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function resetObjectivesCache(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Invalidate the cached list.
  internal._objectivesList = undefined;
}

/**
 * Update objective schedule, pending adds/removes, and objective ages.
 *
 * This is the small orchestration entrypoint for evolve-time objective
 * maintenance. It reads the currently resolved objective keys, applies the
 * caller-provided scheduling policy, then updates the age bookkeeping that
 * later telemetry or policy helpers may inspect.
 *
 * The whole pass is wrapped in a best-effort guard because dynamic objective
 * tracking is useful metadata, not a requirement for the rest of evolution to
 * continue.
 *
 * @param internal - NEAT controller instance.
 * @param helpers - Helper callbacks used by scheduling logic.
 * @param helpers.applyDynamicObjectiveSchedule - Dynamic objective scheduler.
 * @returns void.
 */
export async function updateObjectiveScheduleAndAges(
  internal: NeatControllerForEvolution,
  helpers: {
    applyDynamicObjectiveSchedule: (currentObjectiveKeys: string[]) => void;
  },
): Promise<void> {
  // Step 1: Guard for optional features.
  try {
    const currentObjectiveKeys = (internal._getObjectives?.() ?? []).map(
      (objective) => objective.key,
    );
    // Step 2: Apply dynamic objective schedule.
    helpers.applyDynamicObjectiveSchedule(currentObjectiveKeys);
    // Step 3: Update objective ages.
    for (const key of currentObjectiveKeys) {
      internal._objectiveAges.set(
        key,
        (internal._objectiveAges.get(key) || 0) + 1,
      );
    }
    for (const addedKey of internal._pendingObjectiveAdds) {
      internal._objectiveAges.set(addedKey, 0);
    }
  } catch {
    // Empty catch: objective age tracking is optional telemetry enhancement.
  }
}

/**
 * Suppress fitness objective for specific test scenarios.
 *
 * This helper is intentionally narrow and harness-oriented. It exists for test
 * scenarios that need to force a non-fitness objective mix without turning the
 * rest of the dynamic-objective machinery into test-specific code. Production
 * runs should normally ignore this path entirely.
 *
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function applyFitnessSuppressionForTests(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Guard for optional multi-objective config.
  try {
    const multiObjective = internal.options.multiObjective;
    if (
      multiObjective?.enabled &&
      multiObjective.pruneInactive &&
      multiObjective.pruneInactive.enabled === false
    ) {
      const keys = (internal._getObjectives?.() ?? []).map(
        (objective) => objective.key,
      );
      if (
        keys.includes('fitness') &&
        keys.length > 1 &&
        !internal._fitnessSuppressedOnce
      ) {
        internal._suppressFitnessObjective = true;
        internal._fitnessSuppressedOnce = true;
        internal._objectivesList = undefined;
      }
    }
  } catch {
    // Empty catch: fitness suppression is a test-only helper.
  }
}

/**
 * Capture objective importance stats for telemetry.
 *
 * The snapshot is a compact read-side hint about which objectives are still
 * separating the population meaningfully. By recording per-objective range and
 * variance after objective maintenance runs, later telemetry or diagnostics can
 * explain why an objective may be a good candidate for pruning or why a newly
 * added objective is not yet influencing search strongly.
 *
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function captureObjectiveImportanceSnapshot(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Compute range/variance for objective values.
  try {
    const objectivesList = internal._getObjectives?.() ?? [];
    if (!objectivesList.length) return;
    const importance: Record<string, { range: number; var: number }> = {};
    const population = internal.population;
    for (const objective of objectivesList) {
      const values = population.map((genome) => objective.accessor(genome));
      const minValue = Math.min(...values);
      const maxValue = Math.max(...(values as number[]));
      const meanValue =
        values.reduce((sum: number, value: number) => sum + value, 0) /
        values.length;
      const variance =
        values.reduce(
          (sum: number, value: number) =>
            sum + (value - meanValue) * (value - meanValue),
          0,
        ) / (values.length || 1);
      importance[objective.key] = { range: maxValue - minValue, var: variance };
    }
    internal._lastObjImportance = importance;
  } catch {
    // Empty catch: objective importance calculation is optional telemetry enhancement.
  }
}

/**
 * Apply dynamic objective scheduling and entropy rules.
 *
 * This helper is the main generation-time policy switchboard for objectives.
 * When dynamic scheduling is enabled, it decides when to add scheduled
 * objectives such as complexity or entropy and then hands off to the entropy
 * drop or re-add rules. When dynamic scheduling is disabled but auto-entropy is
 * enabled, it applies the simpler fallback rule that adds entropy after a
 * configured generation threshold.
 *
 * The helper mutates controller policy state in place by registering
 * objectives, updating pending-add queues, and delegating entropy maintenance.
 * Callers should therefore treat it as a policy update step rather than a pure
 * read.
 *
 * @param internal - NEAT controller instance.
 * @param currentObjectiveKeys - Keys of active objectives.
 * @param config - Scheduling constants.
 * @returns void.
 */
export function applyDynamicObjectiveSchedule(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  config: {
    autoEntropyAddAt: number;
  },
): void {
  // Step 1: Guard for disabled multi-objective.
  const multiObjective = internal.options.multiObjective;
  if (!multiObjective?.enabled) return;
  const dynamicConfig = multiObjective.dynamic;
  if (dynamicConfig?.enabled) {
    // Step 2: Add scheduled objectives.
    const addComplexityAt = dynamicConfig.addComplexityAt ?? Infinity;
    const addEntropyAt = dynamicConfig.addEntropyAt ?? Infinity;
    if (
      internal.generation + 1 >= addComplexityAt &&
      !currentObjectiveKeys.includes('complexity')
    ) {
      internal.registerObjective(
        'complexity',
        'min',
        (genome) => genome.connections.length,
      );
      internal._pendingObjectiveAdds.push('complexity');
    }
    if (
      internal.generation + 1 >= addEntropyAt &&
      !currentObjectiveKeys.includes('entropy')
    ) {
      internal.registerObjective(
        'entropy',
        'max',
        createEntropyAccessor(internal),
      );
      internal._pendingObjectiveAdds.push('entropy');
    }
    // Step 3: Handle entropy drop/re-add.
    handleEntropyDropAndReadd(internal, currentObjectiveKeys, dynamicConfig);
  } else if (multiObjective.autoEntropy) {
    // Step 4: Auto-entropy fallback.
    if (
      internal.generation >= config.autoEntropyAddAt &&
      !currentObjectiveKeys.includes('entropy')
    ) {
      internal.registerObjective(
        'entropy',
        'max',
        createEntropyAccessor(internal),
      );
      internal._pendingObjectiveAdds.push('entropy');
    }
  }
}

/**
 * Handle entropy removal and re-addition rules.
 *
 * Entropy is special because some runs benefit from introducing it, dropping
 * it during stagnation, then restoring it after a cooldown. This helper keeps
 * that lifecycle in one place so the evolve loop can reason about entropy as a
 * deliberate scheduled objective rather than a one-way configuration toggle.
 *
 * The important state transitions are:
 * - remove `entropy` from the configured objective set when stagnation crosses
 *   the configured drop generation,
 * - mark the removal generation in `_entropyDropped`,
 * - queue the removal for downstream bookkeeping,
 * - re-register entropy after the configured cooldown, then clear the dropped
 *   marker.
 *
 * @param internal - NEAT controller instance.
 * @param currentObjectiveKeys - Active objective keys.
 * @param dynamicConfig - Dynamic objective config.
 * @returns void.
 */
function handleEntropyDropAndReadd(
  internal: NeatControllerForEvolution,
  currentObjectiveKeys: string[],
  dynamicConfig: NonNullable<
    NeatControllerForEvolution['options']['multiObjective']
  >['dynamic'],
): void {
  if (!dynamicConfig) return;
  // Step 1: Drop entropy when stagnation threshold hit.
  if (
    currentObjectiveKeys.includes('entropy') &&
    dynamicConfig.dropEntropyOnStagnation != null
  ) {
    const stagnationGeneration = dynamicConfig.dropEntropyOnStagnation;
    if (
      internal.generation >= stagnationGeneration &&
      !internal._entropyDropped
    ) {
      if (internal.options.multiObjective?.objectives) {
        internal.options.multiObjective.objectives =
          internal.options.multiObjective.objectives.filter(
            (objective) => objective.key !== 'entropy',
          );
        internal._objectivesList = undefined;
        internal._pendingObjectiveRemoves.push('entropy');
        internal._entropyDropped = internal.generation;
      }
    }
    return;
  }
  // Step 2: Re-add entropy after cooldown.
  if (
    !currentObjectiveKeys.includes('entropy') &&
    internal._entropyDropped &&
    dynamicConfig.readdEntropyAfter != null
  ) {
    if (
      internal.generation - internal._entropyDropped >=
      dynamicConfig.readdEntropyAfter
    ) {
      internal.registerObjective(
        'entropy',
        'max',
        createEntropyAccessor(internal),
      );
      internal._pendingObjectiveAdds.push('entropy');
      internal._entropyDropped = undefined;
    }
  }
}

/**
 * Build the entropy accessor used by dynamic objective scheduling.
 *
 * The evolve controller already advertises `_structuralEntropy` as an optional
 * hook. This helper centralizes the non-null assertion so the scheduling logic
 * can stay declarative while preserving the existing expectation that entropy
 * scheduling only makes sense on hosts exposing that hook.
 *
 * @param internal - NEAT controller instance.
 * @returns Accessor that reads structural entropy from one genome.
 */
function createEntropyAccessor(
  internal: NeatControllerForEvolution,
): ObjectiveDescriptor['accessor'] {
  return (genome: GenomeWithMetadata) => internal._structuralEntropy!(genome);
}
