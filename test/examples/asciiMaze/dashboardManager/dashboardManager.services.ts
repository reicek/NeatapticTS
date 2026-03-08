import type {
  AsciiMazeTelemetrySnapshot,
  DashboardManagerContext,
  DashboardManagerState,
  DashboardManagerUpdateArgs,
} from './dashboardManager.types';
import { recordSolvedMaze } from './archive/dashboardManager.archive.services';
import { redrawDashboard as redrawDashboardLive } from './live/dashboardManager.live.services';
import {
  emitTelemetryPayload,
  getDashboardLastTelemetry as getDashboardLastTelemetrySnapshot,
  updateTelemetryHistory,
} from './telemetry/dashboardManager.telemetry.services';

/**
 * Repaint the live dashboard from current state and refresh the detailed snapshot.
 *
 * @param context - Dashboard runtime context for state and output callbacks.
 * @param currentMaze - Maze currently being evolved.
 * @param neat - Optional NEAT runtime instance used for detailed stats.
 */
export function redrawDashboard(
  context: DashboardManagerContext,
  currentMaze: string[],
  neat?: unknown,
): void {
  redrawDashboardLive(context, currentMaze, neat);
}

/**
 * Ingest one engine update, refresh the live view, and emit external telemetry.
 *
 * @param context - Dashboard runtime context for state and output callbacks.
 * @param args - Latest update payload from the evolution engine.
 */
export function applyDashboardUpdate(
  context: DashboardManagerContext,
  args: DashboardManagerUpdateArgs,
): void {
  const { state } = context;
  const { generation, maze, result, network, neatInstance, telemetryHook } =
    args;

  if (state.runStartTs == null) {
    state.runStartTs = Date.now();
    state.perfStart = globalThis.performance?.now?.() ?? state.runStartTs;
  }

  state.lastUpdateTs = globalThis.performance?.now?.() ?? Date.now();
  state.lastGeneration = generation;

  if (result) {
    state.currentBest = { result, network, generation };
  }

  if (result?.success && network) {
    recordSolvedMaze(context, maze, result, network, generation);
  }

  updateTelemetryHistory(state, neatInstance);
  redrawDashboardLive(context, maze, neatInstance);
  emitTelemetryPayload(state, generation, telemetryHook);
}

/**
 * Produce the latest public telemetry snapshot from current dashboard state.
 *
 * @param state - Mutable dashboard state.
 * @returns Public telemetry snapshot used by browser hosts.
 */
export function getDashboardLastTelemetry(
  state: DashboardManagerState,
): AsciiMazeTelemetrySnapshot {
  return getDashboardLastTelemetrySnapshot(state);
}

/**
 * Clear retained archive, best-candidate, and history state for a fresh run.
 *
 * @param state - Mutable dashboard state to clear.
 */
export function resetDashboardState(state: DashboardManagerState): void {
  state.solvedMazes = [];
  state.solvedMazeKeys.clear();
  state.currentBest = null;
  state.lastTelemetry = null;
  state.lastBestFitness = null;
  state.lastDetailedStats = null;
  state.runStartTs = null;
  state.perfStart = null;
  state.lastGeneration = null;
  state.lastUpdateTs = null;
  state.histories.bestFitness = [];
  state.histories.complexityNodes = [];
  state.histories.complexityConns = [];
  state.histories.hypervolume = [];
  state.histories.progress = [];
  state.histories.speciesCount = [];
  state.scratch.scores.length = 0;
  state.scratch.speciesSizes.length = 0;
  state.scratch.operatorStats.length = 0;
  state.scratch.mutationEntries.length = 0;
}
