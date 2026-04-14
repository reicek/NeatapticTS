/**
 * Telemetry-export boundary for the ASCII Maze dashboard.
 *
 * This file is where the dashboard stops being only a terminal or browser view
 * and becomes a source of structured data. It builds the bounded histories,
 * rich detail snapshots, and export payloads that let other hosts inspect the
 * same run through charts, hooks, or serialized snapshots instead of only live
 * text output.
 *
 * The telemetry boundary exists so "what should be observable" stays separate
 * from "how should the dashboard be painted." That distinction keeps browser
 * integrations, debug tooling, and offline inspection from leaking presentation
 * assumptions back into the redraw path.
 *
 * A practical reading order is:
 *
 * 1. `getDashboardLastTelemetry()` for the public snapshot surface,
 * 2. `updateTelemetryHistory()` for bounded-history upkeep,
 * 3. the detail builders below when you want the richer analytical shelf.
 */
import { MazeUtils } from '../../mazeUtils';
import type { IActivationSchedulingDiagnostics, INetwork } from '../../interfaces';
import { DASHBOARD_MANAGER_CONSTANTS as C } from '../dashboardManager.constants';
import type {
  AsciiMazeActivationSchedulingStats,
  AsciiMazeDetailedStats,
  AsciiMazeTelemetrySnapshot,
  DashboardManagerState,
  DashboardTelemetry,
  DashboardTelemetryPayload,
  MutationStatsMap,
  NeatInstance,
  OperatorStatsEntry,
} from '../dashboardManager.types';
import {
  buildDashboardSparkline,
  sliceDashboardHistoryForExport,
} from '../dashboardManager.utils';

/**
 * Produce the latest public telemetry snapshot from current dashboard state.
 *
 * @param state - Mutable dashboard state.
 * @returns Public telemetry snapshot used by browser hosts.
 */
export function getDashboardLastTelemetry(
  state: DashboardManagerState,
): AsciiMazeTelemetrySnapshot {
  const elapsedMs =
    state.perfStart != null && typeof performance !== 'undefined'
      ? performance.now() - state.perfStart
      : state.runStartTs
        ? Date.now() - state.runStartTs
        : 0;
  const generation = state.lastGeneration ?? 0;
  const gensPerSec = elapsedMs > 0 ? generation / (elapsedMs / 1000) : 0;

  return {
    generation,
    bestFitness: state.lastBestFitness,
    progress: state.currentBest?.result?.progress ?? null,
    speciesCount: MazeUtils.safeLast(state.histories.speciesCount) ?? null,
    gensPerSec: +gensPerSec.toFixed(3),
    timestamp: resolveLastUpdateWallMs(state),
    details: state.lastDetailedStats,
  };
}

/**
 * Pull the latest NEAT telemetry snapshot and update bounded dashboard histories.
 *
 * @param state - Mutable dashboard state that owns bounded histories.
 * @param neatInstance - Optional NEAT-like runtime exposing `getTelemetry()`.
 */
export function updateTelemetryHistory(
  state: DashboardManagerState,
  neatInstance?: { getTelemetry?: () => unknown[] } | undefined,
): void {
  const telemetrySeriesCandidate = neatInstance?.getTelemetry?.();
  if (
    !Array.isArray(telemetrySeriesCandidate) ||
    !telemetrySeriesCandidate.length
  ) {
    return;
  }

  const telemetrySeries = telemetrySeriesCandidate as DashboardTelemetry[];
  state.lastTelemetry =
    MazeUtils.safeLast<DashboardTelemetry>(telemetrySeries) ?? null;

  const latestFitness = state.currentBest?.result?.fitness;
  if (typeof latestFitness === 'number') {
    state.lastBestFitness = latestFitness;
    state.histories.bestFitness = MazeUtils.pushHistory(
      state.histories.bestFitness,
      latestFitness,
      C.HISTORY_MAX,
    );
  }

  const complexitySnapshot = state.lastTelemetry?.complexity;
  if (typeof complexitySnapshot?.meanNodes === 'number') {
    state.histories.complexityNodes = MazeUtils.pushHistory(
      state.histories.complexityNodes,
      complexitySnapshot.meanNodes,
      C.HISTORY_MAX,
    );
  }
  if (typeof complexitySnapshot?.meanConns === 'number') {
    state.histories.complexityConns = MazeUtils.pushHistory(
      state.histories.complexityConns,
      complexitySnapshot.meanConns,
      C.HISTORY_MAX,
    );
  }

  if (typeof state.lastTelemetry?.hyper === 'number') {
    state.histories.hypervolume = MazeUtils.pushHistory(
      state.histories.hypervolume,
      state.lastTelemetry.hyper,
      C.HISTORY_MAX,
    );
  }

  const currentProgress = state.currentBest?.result?.progress;
  if (typeof currentProgress === 'number') {
    state.histories.progress = MazeUtils.pushHistory(
      state.histories.progress,
      currentProgress,
      C.HISTORY_MAX,
    );
  }

  if (typeof state.lastTelemetry?.species === 'number') {
    state.histories.speciesCount = MazeUtils.pushHistory(
      state.histories.speciesCount,
      state.lastTelemetry.species,
      C.HISTORY_MAX,
    );
  }
}

/**
 * Build the rich telemetry detail snapshot shown in browser hooks and exported snapshots.
 *
 * @param state - Mutable dashboard state with histories and current best candidate.
 * @param neat - Optional NEAT instance used for population-level telemetry.
 * @returns Detailed telemetry snapshot or `null` when no data is available.
 */
export function createDetailedStatsSnapshot(
  state: DashboardManagerState,
  neat?: unknown,
): AsciiMazeDetailedStats | null {
  const telemetry = state.lastTelemetry;
  if (!telemetry && !state.currentBest) return null;

  try {
    const complexitySnapshot = telemetry?.complexity;
    const populationStats = computePopulationStats(state, neat);
    const bestFitnessValue = state.currentBest?.result?.fitness;
    if (populationStats.mean == null && typeof bestFitnessValue === 'number') {
      populationStats.mean = +bestFitnessValue.toFixed(2);
    }
    if (
      populationStats.median == null &&
      typeof bestFitnessValue === 'number'
    ) {
      populationStats.median = +bestFitnessValue.toFixed(2);
    }
    if (
      populationStats.speciesCount == null &&
      typeof telemetry?.species === 'number'
    ) {
      populationStats.speciesCount = telemetry.species;
    }

    const sparklines = {
      fitness:
        buildDashboardSparkline(
          state.histories.bestFitness,
          C.GENERAL_SPARK_WIDTH,
        ) || null,
      nodes:
        buildDashboardSparkline(
          state.histories.complexityNodes,
          C.GENERAL_SPARK_WIDTH,
        ) || null,
      conns:
        buildDashboardSparkline(
          state.histories.complexityConns,
          C.GENERAL_SPARK_WIDTH,
        ) || null,
      hyper:
        buildDashboardSparkline(
          state.histories.hypervolume,
          C.GENERAL_SPARK_WIDTH,
        ) || null,
      progress:
        buildDashboardSparkline(
          state.histories.progress,
          C.GENERAL_SPARK_WIDTH,
        ) || null,
      species:
        buildDashboardSparkline(
          state.histories.speciesCount,
          C.GENERAL_SPARK_WIDTH,
        ) || null,
    } as const;

    const rawFrontsArray = Array.isArray(telemetry?.fronts)
      ? telemetry.fronts
      : null;
    const neatRuntime = neat as NeatInstance;
    const mutationStatsObj: MutationStatsMap | null =
      telemetry?.mutationStats ?? telemetry?.mutation?.stats ?? null;

    return {
      generation: state.currentBest?.generation ?? 0,
      bestFitness:
        typeof bestFitnessValue === 'number' ? bestFitnessValue : null,
      bestFitnessDelta: computeBestFitnessDelta(state.histories.bestFitness),
      saturationFraction:
        typeof state.currentBest?.result?.saturationFraction === 'number'
          ? state.currentBest.result.saturationFraction
          : null,
      actionEntropy:
        typeof state.currentBest?.result?.actionEntropy === 'number'
          ? state.currentBest.result.actionEntropy
          : null,
      activationScheduling: resolveActivationSchedulingDetails(
        state.currentBest?.network,
      ),
      populationMean: populationStats.mean,
      populationMedian: populationStats.median,
      enabledConnRatio: populationStats.enabledRatio,
      complexity: complexitySnapshot || null,
      simplifyPhaseActive: Boolean(
        (typeof complexitySnapshot?.growthNodes === 'number' &&
          complexitySnapshot.growthNodes < 0) ||
        (typeof complexitySnapshot?.growthConns === 'number' &&
          complexitySnapshot.growthConns < 0),
      ),
      perf: telemetry?.perf || null,
      lineage: telemetry?.lineage || null,
      diversity: telemetry?.diversity || null,
      speciesCount: populationStats.speciesCount,
      topSpeciesSizes: computeTopSpeciesSizes(state, neat),
      objectives: telemetry?.objectives || null,
      paretoFrontSizes: rawFrontsArray
        ? rawFrontsArray.map((frontValue) => frontValue?.length ?? 0)
        : null,
      firstFrontSize: rawFrontsArray?.[0]?.length || 0,
      hypervolume:
        typeof telemetry?.hyper === 'number' ? telemetry.hyper : null,
      noveltyArchiveSize: safeInvoke(
        () =>
          neatRuntime?.getNoveltyArchive
            ? (neatRuntime.getNoveltyArchive()?.length ?? null)
            : null,
        null,
      ),
      operatorAcceptance: computeOperatorAcceptance(state, neat),
      topMutations: computeTopMutations(state, mutationStatsObj),
      mutationStats: mutationStatsObj || null,
      trends: sparklines,
      histories: {
        bestFitness: sliceDashboardHistoryForExport(
          state.histories.bestFitness,
        ),
        nodes: sliceDashboardHistoryForExport(state.histories.complexityNodes),
        conns: sliceDashboardHistoryForExport(state.histories.complexityConns),
        hyper: sliceDashboardHistoryForExport(state.histories.hypervolume),
        progress: sliceDashboardHistoryForExport(state.histories.progress),
        species: sliceDashboardHistoryForExport(state.histories.speciesCount),
      },
      timestamp: Date.now(),
    };
  } catch {
    return null;
  }
}

/**
 * Resolve compact activation-scheduling details for telemetry export.
 *
 * @param network - Current best network instance.
 * @returns Compact scheduling detail snapshot or null when unavailable.
 */
function resolveActivationSchedulingDetails(
  network: INetwork | null | undefined,
): AsciiMazeActivationSchedulingStats | null {
  if (!network || typeof network.getActivationSchedulingDiagnostics !== 'function') {
    return null;
  }

  try {
    const schedulingDiagnostics =
      network.getActivationSchedulingDiagnostics() as IActivationSchedulingDiagnostics;

    return {
      requestedMode: schedulingDiagnostics.requestedMode ?? null,
      executionPath: schedulingDiagnostics.executionPath ?? null,
      issue: schedulingDiagnostics.issue ?? null,
      stepCount:
        typeof schedulingDiagnostics.stepCount === 'number'
          ? schedulingDiagnostics.stepCount
          : 0,
      recurrentComponentCount:
        typeof schedulingDiagnostics.recurrentComponentCount === 'number'
          ? schedulingDiagnostics.recurrentComponentCount
          : 0,
    };
  } catch {
    return null;
  }
}

/**
 * Emit the structured telemetry payload used by browser hosts and runtime hooks.
 *
 * @param state - Mutable dashboard state used to assemble the payload.
 * @param generation - Current generation number.
 * @param telemetryHook - Optional runtime hook installed by the browser host.
 */
export function emitTelemetryPayload(
  state: DashboardManagerState,
  generation: number,
  telemetryHook?: (payload: DashboardTelemetryPayload) => void,
): void {
  try {
    const elapsedMs =
      state.perfStart != null && globalThis.performance?.now
        ? globalThis.performance.now() - state.perfStart
        : state.runStartTs
          ? Date.now() - state.runStartTs
          : 0;
    const generationsPerSecond =
      elapsedMs > 0 ? generation / (elapsedMs / 1000) : 0;
    const payload: DashboardTelemetryPayload = {
      type: 'asciiMaze:telemetry',
      generation,
      bestFitness: state.lastBestFitness,
      progress: state.currentBest?.result?.progress ?? null,
      speciesCount: state.histories.speciesCount.at(-1) ?? null,
      gensPerSec: +generationsPerSecond.toFixed(3),
      timestamp: Date.now(),
      details: state.lastDetailedStats,
    };

    if (typeof window !== 'undefined') {
      try {
        window.dispatchEvent(
          new CustomEvent('asciiMazeTelemetry', { detail: payload }),
        );
      } catch {
        // Ignore browser event dispatch failures.
      }
      try {
        if (window.parent && window.parent !== window) {
          window.parent.postMessage(payload, '*');
        }
      } catch {
        // Ignore parent-frame postMessage failures.
      }
      (
        window as Window & {
          asciiMazeLastTelemetry?: DashboardTelemetryPayload;
        }
      ).asciiMazeLastTelemetry = payload;
    }

    try {
      telemetryHook?.(payload);
    } catch {
      // Ignore runtime telemetry hook failures.
    }
  } catch {
    // Ignore telemetry emission failures.
  }
}

function resolveLastUpdateWallMs(state: DashboardManagerState): number {
  if (state.lastUpdateTs == null) return Date.now();
  if (
    state.perfStart != null &&
    typeof globalThis.performance?.now === 'function' &&
    state.runStartTs != null
  ) {
    return state.runStartTs + (state.lastUpdateTs - state.perfStart);
  }
  return state.lastUpdateTs;
}

function computePopulationStats(
  state: DashboardManagerState,
  neat?: unknown,
): {
  mean: number | null;
  median: number | null;
  speciesCount: number | null;
  enabledRatio: number | null;
} {
  const neatRuntime = neat as NeatInstance;
  if (
    !neatRuntime ||
    !Array.isArray(neatRuntime.population) ||
    neatRuntime.population.length === 0
  ) {
    return {
      mean: null,
      median: null,
      speciesCount: null,
      enabledRatio: null,
    };
  }

  const { scores } = state.scratch;
  scores.length = 0;
  let enabledConnectionsCount = 0;
  let totalConnectionsCount = 0;
  for (const genome of neatRuntime.population) {
    if (typeof genome?.score === 'number') {
      scores.push(genome.score);
    }

    if (!Array.isArray(genome?.connections)) continue;
    for (const connectionValue of genome.connections) {
      totalConnectionsCount++;
      if (connectionValue?.enabled !== false) {
        enabledConnectionsCount++;
      }
    }
  }

  let mean: number | null = null;
  let median: number | null = null;
  if (scores.length) {
    const sum = scores.reduce(
      (runningTotal, scoreValue) => runningTotal + scoreValue,
      0,
    );
    mean = +(sum / scores.length).toFixed(2);
    const sortedScores = scores.toSorted(
      (leftScore, rightScore) => leftScore - rightScore,
    );
    const middleIndex = Math.floor(sortedScores.length / 2);
    const medianRaw =
      sortedScores.length % 2 === 0
        ? (sortedScores[middleIndex - 1] + sortedScores[middleIndex]) / 2
        : sortedScores[middleIndex];
    median = +medianRaw.toFixed(2);
  }

  const speciesCount = Array.isArray(neatRuntime.species)
    ? neatRuntime.species.length
    : null;
  const enabledRatio = totalConnectionsCount
    ? +(enabledConnectionsCount / totalConnectionsCount).toFixed(2)
    : null;

  return { mean, median, speciesCount, enabledRatio };
}

function computeOperatorAcceptance(
  state: DashboardManagerState,
  neat?: unknown,
): Array<{ name: string; acceptancePct: number }> | null {
  const neatRuntime = neat as NeatInstance;
  if (typeof neatRuntime?.getOperatorStats !== 'function') return null;

  let rawOperatorStats: unknown;
  try {
    rawOperatorStats = neatRuntime.getOperatorStats();
  } catch {
    return null;
  }
  if (!Array.isArray(rawOperatorStats) || !rawOperatorStats.length) return null;

  const scratchBuffer = state.scratch.operatorStats;
  scratchBuffer.length = 0;
  for (const operatorValue of rawOperatorStats) {
    const operatorRecord = operatorValue as Record<string, unknown>;
    if (
      typeof operatorRecord.name === 'string' &&
      typeof operatorRecord.success === 'number' &&
      typeof operatorRecord.attempts === 'number'
    ) {
      scratchBuffer.push(operatorRecord as unknown as OperatorStatsEntry);
    }
  }
  if (!scratchBuffer.length) return null;

  return scratchBuffer
    .toSorted((leftStat, rightStat) => {
      const leftAcceptance = leftStat.success / Math.max(1, leftStat.attempts);
      const rightAcceptance =
        rightStat.success / Math.max(1, rightStat.attempts);
      return rightAcceptance - leftAcceptance;
    })
    .slice(0, C.TOP_OPERATOR_LIMIT)
    .map((rankedStat) => ({
      name: rankedStat.name,
      acceptancePct: +(
        (100 * rankedStat.success) /
        Math.max(1, rankedStat.attempts)
      ).toFixed(2),
    }));
}

function computeTopMutations(
  state: DashboardManagerState,
  mutationStats: MutationStatsMap | null,
): Array<{ name: string; count: number }> | null {
  if (!mutationStats) return null;

  const mutationEntries = state.scratch.mutationEntries;
  mutationEntries.length = 0;
  for (const [mutationName, mutationCount] of Object.entries(mutationStats)) {
    if (typeof mutationCount === 'number' && Number.isFinite(mutationCount)) {
      mutationEntries.push([mutationName, mutationCount]);
    }
  }
  if (!mutationEntries.length) return null;

  return mutationEntries
    .toSorted((leftEntry, rightEntry) => rightEntry[1] - leftEntry[1])
    .slice(0, C.TOP_MUTATION_LIMIT)
    .map(([mutationName, mutationCount]) => ({
      name: mutationName,
      count: mutationCount,
    }));
}

function computeTopSpeciesSizes(
  state: DashboardManagerState,
  neat?: unknown,
): number[] | null {
  const neatRuntime = neat as NeatInstance;
  if (!Array.isArray(neatRuntime?.species) || !neatRuntime.species.length) {
    return null;
  }

  const speciesSizes = state.scratch.speciesSizes;
  speciesSizes.length = 0;
  for (const speciesEntry of neatRuntime.species) {
    const memberCount = Array.isArray(speciesEntry?.members)
      ? speciesEntry.members.length
      : 0;
    speciesSizes.push(memberCount);
  }
  if (!speciesSizes.length) return null;

  return speciesSizes
    .toSorted((leftSize, rightSize) => rightSize - leftSize)
    .slice(0, C.TOP_SPECIES_LIMIT);
}

function computeBestFitnessDelta(history: number[]): number | null {
  const previousSample = history.at(-2);
  const latestSample = history.at(-1);
  if (typeof previousSample !== 'number' || typeof latestSample !== 'number') {
    return null;
  }
  return +(latestSample - previousSample).toFixed(3);
}

function safeInvoke<T>(operation: () => T, fallback: T): T {
  try {
    return operation();
  } catch {
    return fallback;
  }
}
