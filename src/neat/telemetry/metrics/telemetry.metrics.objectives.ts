import type {
  GenomeDetailed,
  NeatOptions,
  ObjImportance,
  ObjectiveEvent,
  SpeciesAlloc,
} from '../../shared/neat.shared.types';
import type {
  TelemetryDiversityOptions,
  TelemetryEntryRecord,
} from '../types/telemetry.types';

/**
 * Compute a hypervolume-like proxy for the active Pareto frontier so telemetry captures objective tradeoff quality with a stable, generation-comparable scalar.
 *
 * @param telemetryOptions - Options controlling complexity metric.
 * @param population - Population snapshot.
 * @returns Hypervolume proxy value.
 */
export function computeHyperVolumeProxy(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
): number {
  // Step 1: Resolve complexity metric.
  const complexityMetric: 'nodes' | 'connections' =
    telemetryOptions.multiObjective?.complexityMetric || 'connections';

  // Step 2: Compute normalization bounds for primary objective scores.
  const primaryObjectiveScores = population.map(
    (genome) => (genome.score as number) || 0,
  );
  const minPrimaryScore = Math.min(...primaryObjectiveScores);
  const maxPrimaryScore = Math.max(...primaryObjectiveScores);

  // Step 3: Fold Pareto-front genomes into the hypervolume proxy.
  let hyperVolumeProxy = 0;
  for (const genome of population) {
    const rank = genome._moRank ?? 0;
    if (rank !== 0) continue;
    const normalizedScore =
      maxPrimaryScore > minPrimaryScore
        ? ((genome.score || 0) - minPrimaryScore) /
          (maxPrimaryScore - minPrimaryScore)
        : 0;
    const genomeComplexity =
      complexityMetric === 'nodes'
        ? genome.nodes.length
        : genome.connections.length;
    hyperVolumeProxy += normalizedScore * (1 / (genomeComplexity + 1));
  }

  // Step 4: Return the computed proxy value.
  return hyperVolumeProxy;
}

/**
 * Compute sizes of the earliest Pareto fronts so recorder outputs can show frontier stratification pressure and rank distribution at this generation.
 *
 * @param population - Population snapshot.
 * @returns Array of front sizes (rank 0..4).
 */
export function computeParetoFrontSizes(
  population: GenomeDetailed[],
): number[] {
  // Step 1: Collect sizes for the first few ranks.
  const paretoFrontSizes: number[] = [];
  for (let rankIndex = 0; rankIndex < 5; rankIndex++) {
    const frontSize = population.filter(
      (genome) => ((genome as GenomeDetailed)._moRank ?? 0) === rankIndex,
    ).length;
    if (!frontSize) break;
    paretoFrontSizes.push(frontSize);
  }

  // Step 2: Return the collected sizes.
  return paretoFrontSizes;
}

/**
 * Apply the most recent objective-importance snapshot so telemetry entries preserve objective spread evidence computed earlier in the evolutionary pass history.
 *
 * @param telemetryContext - Neat-like context with objective importance.
 * @param entry - Telemetry entry to update.
 */
export function applyObjectiveImportance(
  telemetryContext: { _lastObjImportance?: ObjImportance },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Default to an empty importance map.
  if (!entry.objImportance) entry.objImportance = {};

  // Step 2: Apply the latest importance snapshot when available.
  if (telemetryContext._lastObjImportance) {
    const lastImportance = telemetryContext._lastObjImportance;
    if (typeof lastImportance === 'object' && lastImportance !== null)
      entry.objImportance = lastImportance;
  }
}

/**
 * Apply objective age snapshots to the entry so analysts can distinguish mature objectives from newly introduced optimization signals over long runs.
 *
 * @param telemetryContext - Neat-like context with objective ages.
 * @param entry - Telemetry entry to update.
 */
export function applyObjectiveAges(
  telemetryContext: { _objectiveAges?: Map<string, number> },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach objective ages when available.
  if (telemetryContext._objectiveAges?.size) {
    entry.objAges = Object.fromEntries(
      telemetryContext._objectiveAges.entries(),
    );
  }
}

/**
 * Apply and flush objective lifecycle events so each telemetry entry records adds and removals exactly once at the generation boundary.
 *
 * @param telemetryContext - Neat-like context holding objective events.
 * @param entry - Telemetry entry to update.
 * @param generation - Generation index for event records.
 */
export function applyObjectiveEvents(
  telemetryContext: {
    _pendingObjectiveAdds?: string[];
    _pendingObjectiveRemoves?: string[];
    _objectiveEvents?: ObjectiveEvent[];
  },
  entry: TelemetryEntryRecord,
  generation: number,
): void {
  // Step 1: Skip if there are no pending events.
  if (
    !telemetryContext._pendingObjectiveAdds?.length &&
    !telemetryContext._pendingObjectiveRemoves?.length
  )
    return;

  // Step 2: Build event list from pending changes.
  entry.objEvents = [];
  for (const objectiveKey of telemetryContext._pendingObjectiveAdds ?? []) {
    entry.objEvents.push({ gen: generation, type: 'add', key: objectiveKey });
  }
  for (const objectiveKey of telemetryContext._pendingObjectiveRemoves ?? []) {
    entry.objEvents.push({
      gen: generation,
      type: 'remove',
      key: objectiveKey,
    });
  }

  // Step 3: Persist events and clear pending arrays.
  telemetryContext._objectiveEvents = telemetryContext._objectiveEvents ?? [];
  telemetryContext._objectiveEvents.push(...entry.objEvents);
  telemetryContext._pendingObjectiveAdds = [];
  telemetryContext._pendingObjectiveRemoves = [];
}

/**
 * Apply the per-species offspring allocation snapshot so downstream dashboards can correlate selection pressure with later diversity and fitness changes reliably.
 *
 * @param telemetryContext - Neat-like context with allocation snapshot.
 * @param entry - Telemetry entry to update.
 */
export function applySpeciesAllocation(
  telemetryContext: { _lastOffspringAlloc?: SpeciesAlloc[] },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach last allocation snapshot when present.
  if (telemetryContext._lastOffspringAlloc) {
    const lastAlloc = telemetryContext._lastOffspringAlloc;
    if (Array.isArray(lastAlloc)) entry.speciesAlloc = lastAlloc.slice();
  }
}

/**
 * Apply the active objectives list snapshot using objective keys only so entries remain compact while still exposing current optimization scope.
 *
 * @param telemetryContext - Neat-like context with objective provider.
 * @param entry - Telemetry entry to update.
 */
export function applyObjectivesSnapshot(
  telemetryContext: { _getObjectives?: () => { key: string }[] },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attempt to read objective keys; ignore absence.
  try {
    entry.objectives =
      telemetryContext._getObjectives?.().map((objective) => objective.key) ||
      [];
  } catch {
    // ignored: objective provider not present - skip objectives snapshot
  }
}

/**
 * Attach a rounded hypervolume scalar when requested so telemetry consumers can track Pareto quality trends without recalculating expensive frontier aggregates.
 *
 * @param telemetryOptions - Options controlling telemetry fields.
 * @param hyperVolumeProxy - Hypervolume proxy value.
 * @param entry - Telemetry entry to update.
 */
export function applyHypervolumeTelemetry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  hyperVolumeProxy: number,
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach rounded hypervolume proxy when enabled.
  if (
    telemetryOptions.telemetry?.hypervolume &&
    telemetryOptions.multiObjective?.enabled
  )
    entry.hv = +hyperVolumeProxy.toFixed(4);
}
