/*
 * ESLint configuration for intentional `any` usage in NEAT evolution objectives utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import type { NeatControllerForEvolution } from './neat.evolve.types';

/**
 * Clear cached objectives so dynamic schedules can rebuild them.
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
        internal._objectivesList = undefined as any;
      }
    }
  } catch {
    // Empty catch: fitness suppression is a test-only helper.
  }
}

/**
 * Capture objective importance stats for telemetry.
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
    const population = internal.population as any[];
    for (const objective of objectivesList as any[]) {
      const values = population.map((genome: any) =>
        objective.accessor(genome),
      );
      const minValue = Math.min(...(values as number[]));
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
    internal._lastObjImportance = importance as any;
  } catch {
    // Empty catch: objective importance calculation is optional telemetry enhancement.
  }
}

/**
 * Apply dynamic objective scheduling and entropy rules.
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
        (genome: any) => genome.connections.length,
      );
      internal._pendingObjectiveAdds.push('complexity');
    }
    if (
      internal.generation + 1 >= addEntropyAt &&
      !currentObjectiveKeys.includes('entropy')
    ) {
      internal.registerObjective('entropy', 'max', (genome: any) =>
        (internal as any)._structuralEntropy(genome),
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
      internal.registerObjective('entropy', 'max', (genome: any) =>
        (internal as any)._structuralEntropy(genome),
      );
      internal._pendingObjectiveAdds.push('entropy');
    }
  }
}

/**
 * Handle entropy removal and re-addition rules.
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
            (objective: any) => objective.key !== 'entropy',
          );
        internal._objectivesList = undefined as any;
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
      internal.registerObjective('entropy', 'max', (genome: any) =>
        (internal as any)._structuralEntropy(genome),
      );
      internal._pendingObjectiveAdds.push('entropy');
      internal._entropyDropped = undefined;
    }
  }
}
