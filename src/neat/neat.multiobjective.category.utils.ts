/*
 * ESLint configuration for intentional `any` usage in NEAT multi-objective category utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import { fastNonDominated } from './neat.multiobjective';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  ObjectiveDescriptor,
} from './neat.evolve.types';

/**
 * Run multi-objective ranking, crowding distance, and archives.
 * @param internal - NEAT controller instance.
 * @param config - Multi-objective tuning constants.
 * @returns void.
 */
export function processMultiObjective(
  internal: NeatControllerForEvolution,
  config: {
    paretoArchiveMax: number;
    targetFrontMin: number;
    targetFrontUpperRatio: number;
    targetFrontLowerRatio: number;
    defaultEpsilonAdjust: number;
    defaultEpsilonMin: number;
    defaultEpsilonMax: number;
    defaultEpsilonCooldown: number;
    pruneWindowDefault: number;
    pruneRangeEpsDefault: number;
  },
): void {
  // Step 1: Prepare the population snapshot.
  const populationSnapshot = internal.population;
  const paretoFronts = fastNonDominated.call(
    internal as never,
    populationSnapshot as never,
  ) as unknown as GenomeWithMetadata[][];
  const objectives = internal._getObjectives?.() ?? [];
  const crowdingDistances = computeCrowdingDistances(
    internal,
    populationSnapshot,
    paretoFronts,
    objectives,
  );
  // Step 2: Apply stable sorting using ranks + crowding.
  sortPopulationByPareto(internal, populationSnapshot, crowdingDistances);
  // Step 3: Persist multi-objective archives.
  recordParetoArchives(
    internal,
    paretoFronts,
    objectives,
    config.paretoArchiveMax,
  );
  // Step 4: Adapt dominance epsilon if enabled.
  adaptDominanceEpsilon(internal, paretoFronts, {
    targetFrontMin: config.targetFrontMin,
    targetFrontUpperRatio: config.targetFrontUpperRatio,
    targetFrontLowerRatio: config.targetFrontLowerRatio,
    defaultEpsilonAdjust: config.defaultEpsilonAdjust,
    defaultEpsilonMin: config.defaultEpsilonMin,
    defaultEpsilonMax: config.defaultEpsilonMax,
    defaultEpsilonCooldown: config.defaultEpsilonCooldown,
  });
  // Step 5: Prune inactive objectives if enabled.
  pruneInactiveObjectives(internal, {
    pruneWindowDefault: config.pruneWindowDefault,
    pruneRangeEpsDefault: config.pruneRangeEpsDefault,
  });
}

/**
 * Compute crowding distances for multi-objective fronts.
 * @param internal - NEAT controller instance.
 * @param populationSnapshot - Current population reference.
 * @param paretoFronts - Non-dominated fronts.
 * @param objectives - Active objectives.
 * @returns crowding distances aligned with population order.
 */
function computeCrowdingDistances(
  internal: NeatControllerForEvolution,
  populationSnapshot: GenomeWithMetadata[],
  paretoFronts: GenomeWithMetadata[][],
  objectives: ObjectiveDescriptor[],
): number[] {
  // Step 1: Initialize crowding distances to zero.
  const crowdingDistances: number[] = new Array(populationSnapshot.length).fill(
    0,
  );
  // Step 2: Precompute objective values.
  const objectiveValues = objectives.map((objective) =>
    populationSnapshot.map((genome) => objective.accessor(genome)),
  );
  // Step 3: Compute distances per front.
  for (const front of paretoFronts) {
    const frontIndices = front.map((genome) =>
      internal.population.indexOf(genome as never),
    );
    if (frontIndices.length < 3) {
      frontIndices.forEach(
        (populationIndex) => (crowdingDistances[populationIndex] = Infinity),
      );
      continue;
    }
    for (
      let objectiveIndex = 0;
      objectiveIndex < objectives.length;
      objectiveIndex++
    ) {
      const sortedIndices = frontIndices.toSorted(
        (leftIndex, rightIndex) =>
          objectiveValues[objectiveIndex][leftIndex] -
          objectiveValues[objectiveIndex][rightIndex],
      );
      crowdingDistances[sortedIndices[0]] = Infinity;
      crowdingDistances[sortedIndices.at(-1)!] = Infinity;
      const minValue = objectiveValues[objectiveIndex][sortedIndices[0]];
      const maxValue = objectiveValues[objectiveIndex][sortedIndices.at(-1)!];
      const denom = maxValue - minValue || 1;
      for (
        let sortedIndex = 1;
        sortedIndex < sortedIndices.length - 1;
        sortedIndex++
      ) {
        const prevValue =
          objectiveValues[objectiveIndex][sortedIndices[sortedIndex - 1]];
        const nextValue =
          objectiveValues[objectiveIndex][sortedIndices[sortedIndex + 1]];
        crowdingDistances[sortedIndices[sortedIndex]] +=
          (nextValue - prevValue) / denom;
      }
    }
  }
  return crowdingDistances;
}

/**
 * Sort population by Pareto rank and crowding distance.
 * @param internal - NEAT controller instance.
 * @param populationSnapshot - Current population reference.
 * @param crowdingDistances - Crowding distances aligned with population order.
 * @returns void.
 */
function sortPopulationByPareto(
  internal: NeatControllerForEvolution,
  populationSnapshot: GenomeWithMetadata[],
  crowdingDistances: number[],
): void {
  // Step 1: Build stable index map.
  const indexMap = new Map<GenomeWithMetadata, number>();
  for (
    let populationIndex = 0;
    populationIndex < populationSnapshot.length;
    populationIndex++
  ) {
    indexMap.set(populationSnapshot[populationIndex], populationIndex);
  }
  // Step 2: Sort by (rank, crowding distance).
  internal.population.sort((genomeA, genomeB) => {
    const rankA = genomeA._moRank ?? 0;
    const rankB = genomeB._moRank ?? 0;
    if (rankA !== rankB) return rankA - rankB;
    const indexA = indexMap.get(genomeA as never)!;
    const indexB = indexMap.get(genomeB as never)!;
    return crowdingDistances[indexB] - crowdingDistances[indexA];
  });
  // Step 3: Persist crowding distances.
  for (
    let populationIndex = 0;
    populationIndex < populationSnapshot.length;
    populationIndex++
  ) {
    populationSnapshot[populationIndex]._moCrowd =
      crowdingDistances[populationIndex];
  }
}

/**
 * Record Pareto front archives for telemetry.
 * @param internal - NEAT controller instance.
 * @param paretoFronts - Non-dominated fronts.
 * @param objectives - Active objectives.
 * @param archiveMax - Maximum archive size.
 * @returns void.
 */
function recordParetoArchives(
  internal: NeatControllerForEvolution,
  paretoFronts: GenomeWithMetadata[][],
  objectives: ObjectiveDescriptor[],
  archiveMax: number,
): void {
  // Step 1: Exit if no fronts exist.
  if (!paretoFronts.length) return;
  const firstFront = paretoFronts[0];
  const snapshot = firstFront.map((genome: any) => ({
    id: (genome as any)._id ?? -1,
    score: genome.score || 0,
    nodes: genome.nodes.length,
    connections: genome.connections.length,
  }));
  internal._paretoArchive.push({
    gen: internal.generation,
    size: firstFront.length,
    genomes: snapshot,
  });
  if (internal._paretoArchive.length > archiveMax)
    internal._paretoArchive.shift();
  // Step 2: Record objective vectors if requested.
  if (objectives.length) {
    const vectors = firstFront.map((genome: any) => ({
      id: (genome as any)._id ?? -1,
      values: (objectives as any[]).map((objective: any) =>
        objective.accessor(genome),
      ),
    }));
    internal._paretoObjectivesArchive.push({
      gen: internal.generation,
      vectors,
    });
    if (internal._paretoObjectivesArchive.length > archiveMax)
      internal._paretoObjectivesArchive.shift();
  }
}

/**
 * Adapt dominance epsilon based on Pareto front size.
 * @param internal - NEAT controller instance.
 * @param paretoFronts - Non-dominated fronts.
 * @param config - Epsilon tuning constants.
 * @returns void.
 */
function adaptDominanceEpsilon(
  internal: NeatControllerForEvolution,
  paretoFronts: GenomeWithMetadata[][],
  config: {
    targetFrontMin: number;
    targetFrontUpperRatio: number;
    targetFrontLowerRatio: number;
    defaultEpsilonAdjust: number;
    defaultEpsilonMin: number;
    defaultEpsilonMax: number;
    defaultEpsilonCooldown: number;
  },
): void {
  // Step 1: Guard on config and availability.
  if (!internal.options.multiObjective?.adaptiveEpsilon?.enabled) return;
  if (!paretoFronts.length) return;
  // Step 2: Resolve tuning configuration.
  const adaptiveConfig = internal.options.multiObjective.adaptiveEpsilon;
  const target =
    adaptiveConfig.targetFront ??
    Math.max(
      config.targetFrontMin,
      Math.floor(Math.sqrt(internal.population.length)),
    );
  const adjust = adaptiveConfig.adjust ?? config.defaultEpsilonAdjust;
  const minE = adaptiveConfig.min ?? config.defaultEpsilonMin;
  const maxE = adaptiveConfig.max ?? config.defaultEpsilonMax;
  const cooldown = adaptiveConfig.cooldown ?? config.defaultEpsilonCooldown;
  if (internal.generation - internal._lastEpsilonAdjustGen < cooldown) return;
  // Step 3: Adjust epsilon based on front size.
  const currentSize = paretoFronts[0].length;
  let epsilon = internal.options.multiObjective.dominanceEpsilon || 0;
  if (currentSize > target * config.targetFrontUpperRatio) {
    epsilon = Math.min(maxE, epsilon + adjust);
  } else if (currentSize < target * config.targetFrontLowerRatio) {
    epsilon = Math.max(minE, epsilon - adjust);
  }
  internal.options.multiObjective.dominanceEpsilon = epsilon;
  internal._lastEpsilonAdjustGen = internal.generation;
}

/**
 * Prune objectives that have collapsed ranges over a window.
 * @param internal - NEAT controller instance.
 * @param config - Pruning constants.
 * @returns void.
 */
function pruneInactiveObjectives(
  internal: NeatControllerForEvolution,
  config: {
    pruneWindowDefault: number;
    pruneRangeEpsDefault: number;
  },
): void {
  // Step 1: Guard on configuration.
  if (!internal.options.multiObjective?.pruneInactive?.enabled) return;
  const pruneConfig = internal.options.multiObjective.pruneInactive;
  const window = pruneConfig.window ?? config.pruneWindowDefault;
  const rangeEps = pruneConfig.rangeEps ?? config.pruneRangeEpsDefault;
  const protect = new Set([
    'fitness',
    'complexity',
    ...(pruneConfig.protect || []),
  ]);
  const objectivesList = internal._getObjectives?.() ?? [];
  // Step 2: Compute per-objective ranges.
  const ranges: Record<string, { min: number; max: number }> = {};
  for (const objective of objectivesList) {
    let minValue = Infinity;
    let maxValue = -Infinity;
    for (const genome of internal.population) {
      const value = objective.accessor(genome);
      if (value < minValue) minValue = value;
      if (value > maxValue) maxValue = value;
    }
    ranges[objective.key] = { min: minValue, max: maxValue };
  }
  // Step 3: Track stale objectives.
  const toRemove: string[] = [];
  for (const objective of objectivesList) {
    if (protect.has(objective.key)) continue;
    const objRange = ranges[objective.key];
    const span = objRange.max - objRange.min;
    if (span < rangeEps) {
      const count = (internal._objectiveStale.get(objective.key) || 0) + 1;
      internal._objectiveStale.set(objective.key, count);
      if (count >= window) toRemove.push(objective.key);
    } else {
      internal._objectiveStale.set(objective.key, 0);
    }
  }
  // Step 4: Apply removals and invalidate cache.
  if (toRemove.length && internal.options.multiObjective?.objectives) {
    internal.options.multiObjective.objectives =
      internal.options.multiObjective.objectives.filter(
        (objective: any) => !toRemove.includes(objective.key),
      );
    internal._objectivesList = undefined as any;
  }
}
