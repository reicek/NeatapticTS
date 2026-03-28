/**
 * Evolution-policy helpers for multi-objective runs.
 *
 * The root `multiobjective/` chapter answers one question: how are Pareto
 * fronts and crowding distances computed? This `category/` chapter answers the
 * next one: once those ranks exist, how should the evolve loop react to them?
 *
 * It keeps the controller-facing policy in one place: stable rank+crowding
 * sorting, archive persistence, adaptive dominance-epsilon tuning, and
 * pruning of objectives that have gone structurally inactive.
 *
 * That separation matters because this file does not decide Pareto ranks from
 * scratch. It treats the ranking helpers as an evidence-producing pipeline,
 * then applies the evolve-loop reactions that depend on that evidence:
 * reorder the live population, persist telemetry-friendly snapshots, tune the
 * dominance threshold when the leading front grows too wide or too narrow, and
 * remove objectives that have stopped contributing useful variation.
 *
 * Read this chapter when the missing question is "what does the controller do
 * with Pareto ranks after ranking finishes?" Read `multiobjective/` first if
 * the missing context is how fronts and crowding were computed in the first
 * place.
 *
 * ```mermaid
 * flowchart TD
 *   A[Run fastNonDominated] --> B[Compute per-front crowding distances]
 *   B --> C[Sort live population by rank then crowding]
 *   C --> D[Record Pareto archive snapshots]
 *   D --> E[Adapt dominance epsilon when enabled]
 *   E --> F[Prune inactive objectives when enabled]
 * ```
 */

// Keep the chapter introduction separate from the first exported symbol JSDoc.

import { fastNonDominated } from '../multiobjective';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  ObjectiveDescriptor,
} from '../../evolve/evolve.types';

/**
 * Apply the multi-objective evolution policy for the current generation.
 *
 * This is the bridge between the generic Pareto-ranking helpers and the larger
 * evolve loop. It runs the ranking pass, reorders the live population by
 * `(rank, crowding)`, snapshots the best fronts for telemetry, optionally
 * adjusts dominance epsilon to keep the frontier size useful, and prunes
 * objectives that have gone flat for long enough to stop influencing search.
 *
 * Conceptually, this helper owns the post-ranking reaction layer:
 * 1. compute fresh fronts and crowding evidence,
 * 2. convert that evidence into stable population order,
 * 3. persist compact history for later reads,
 * 4. tune or prune long-lived multi-objective policy state.
 *
 * The function updates the controller in place because evolve needs the new
 * ordering, archive state, and adaptive settings immediately for the rest of
 * the generation loop.
 *
 * @param internal - NEAT controller instance.
 * @param config - Multi-objective tuning constants.
 * @returns Nothing. The controller is updated in place.
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
 *
 * This helper replays the within-front spacing calculation in controller-local
 * coordinates so `processMultiObjective()` can sort the live population by
 * `(rank, crowding)` after fast non-dominated sorting finishes. The returned
 * array is aligned with the current population order, which is why the helper
 * works with front members and population indices together.
 *
 * Small fronts receive `Infinity` immediately because every member is an edge
 * solution in that degenerate case. Larger fronts accumulate normalized
 * neighbor distance objective by objective.
 *
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
 *
 * The ordering rule is lexicographic: lower `_moRank` wins first, then higher
 * crowding distance wins within the same front. This keeps the live population
 * aligned with NSGA-II style selection pressure while preserving one stable
 * index map from the pre-sort snapshot to the later crowding write-back.
 *
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
 *
 * This helper writes two compact history streams when the controller has the
 * corresponding archive arrays available: a lightweight first-front snapshot
 * for quick inspection and, when objectives exist, a parallel objective-vector
 * snapshot that preserves the frontier's raw tradeoff coordinates.
 *
 * The stored data is intentionally smaller than the live population. It keeps
 * just enough evidence for telemetry and retrospective inspection without
 * retaining every dominated genome in every generation.
 *
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
  const snapshot = firstFront.map((genome) => ({
    id: genome._id ?? -1,
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
    const vectors = firstFront.map((genome) => ({
      id: genome._id ?? -1,
      values: objectives.map((objective) => objective.accessor(genome)),
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
 *
 * This is the controller's feedback loop for keeping the leading front in a
 * useful size band. If too many genomes land on the first front, epsilon grows
 * so future dominance becomes stricter. If too few survive, epsilon shrinks so
 * the controller relaxes back toward a broader competitive set.
 *
 * The cooldown gate matters because the frontier can oscillate from one
 * generation to the next. Waiting a few generations between adjustments keeps
 * the threshold from chattering.
 *
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
 *
 * This helper removes objectives that are no longer contributing meaningful
 * discrimination across the current population. An objective is considered
 * structurally inactive when its observed range stays below the configured
 * epsilon for enough consecutive generations.
 *
 * The pruning pass is intentionally conservative:
 * - protected objectives such as `fitness` and `complexity` are never removed,
 * - stale counters must persist for a full window before removal,
 * - objective-cache invalidation happens only after an actual removal so later
 *   reads rebuild the descriptor list from the surviving objective set.
 *
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
        (objective) => !toRemove.includes(objective.key),
      );
    internal._objectivesList = undefined;
  }
}
