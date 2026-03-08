/**
 * Public DashboardManager facade for the dedicated dashboardManager module boundary.
 *
 * The folder now owns local contracts, constants, pure formatting helpers, and
 * stateful rendering and telemetry services. This facade keeps the established
 * class-based API stable while delegating the heavy work to focused helpers.
 */

import type Neat from '../../../../src/neat';
import { colors } from '../colors';
import type {
  IDashboardManager,
  IMazeRunResult,
  INetwork,
} from '../interfaces';
import { NetworkVisualization } from '../networkVisualization';
import { DASHBOARD_MANAGER_CONSTANTS as C } from './dashboardManager.constants';
import {
  applyDashboardUpdate,
  getDashboardLastTelemetry,
  redrawDashboard,
  resetDashboardState,
} from './dashboardManager.services';
import type {
  AsciiMazeTelemetrySnapshot,
  DashboardArchiveFunction,
  DashboardClearFunction,
  DashboardLogFunction,
  DashboardManagerContext,
  DashboardManagerState,
  DashboardTelemetryHook,
  RuntimeDashboardManager,
} from './dashboardManager.types';
import { formatDashboardStat } from './dashboardManager.utils';

/**
 * Rich ASCII maze dashboard used by browser and terminal example hosts.
 *
 * @remarks
 * Create one instance per evolution run. The dashboard stores bounded history
 * buffers and solved-archive output so a fresh instance gives the cleanest
 * example lifecycle.
 */
export class DashboardManager
  implements IDashboardManager, RuntimeDashboardManager
{
  /** Allow runtime host hooks such as `_telemetryHook`. */
  [key: string]: unknown;

  #state: DashboardManagerState = {
    solvedMazes: [],
    solvedMazeKeys: new Set(),
    currentBest: null,
    lastTelemetry: null,
    lastBestFitness: null,
    histories: {
      bestFitness: [],
      complexityNodes: [],
      complexityConns: [],
      hypervolume: [],
      progress: [],
      speciesCount: [],
    },
    lastDetailedStats: null,
    runStartTs: null,
    perfStart: null,
    lastGeneration: null,
    lastUpdateTs: null,
    scratch: {
      scores: [],
      speciesSizes: [],
      operatorStats: [],
      mutationEntries: [],
    },
  };

  #clearFn: DashboardClearFunction;
  #logFn: DashboardLogFunction;
  #archiveFn?: DashboardArchiveFunction;

  /**
   * Create a dashboard instance bound to the host's clear, live-log, and archive outputs.
   *
   * @param clearFn - Function that clears the live dashboard region.
   * @param logFn - Function used for live framed dashboard lines.
   * @param archiveFn - Optional function used for solved-maze archive output.
   */
  constructor(
    clearFn: () => void,
    logFn: (...args: unknown[]) => void,
    archiveFn?: (...args: unknown[]) => void,
  ) {
    const noop = () => {};
    this.#clearFn = typeof clearFn === 'function' ? clearFn : noop;
    this.#logFn = typeof logFn === 'function' ? logFn : noop;
    this.#archiveFn = typeof archiveFn === 'function' ? archiveFn : undefined;
  }

  /** Optional log function exposed for engine-side safe-writer fallbacks. */
  get logFunction(): ((msg: string) => void) | undefined {
    return this.#logFn as (msg: string) => void;
  }

  /**
   * Clear and repaint the live dashboard using the current best candidate and histories.
   *
   * @param currentMaze - Maze currently shown in the live panel.
   * @param neat - Optional NEAT instance used to enrich stats.
   */
  redraw(currentMaze: string[], neat?: unknown): void {
    redrawDashboard(this.#createContext(), currentMaze, neat);
  }

  /**
   * Ingest one evolution update, refresh the live dashboard, and emit telemetry.
   *
   * @param maze - Current maze layout.
   * @param result - Latest run result for the tracked candidate.
   * @param network - Candidate network used for the run.
   * @param generation - Current generation number.
   * @param neatInstance - Optional NEAT runtime used for advanced telemetry.
   */
  update(
    maze: string[],
    result: IMazeRunResult | undefined,
    network: INetwork | null,
    generation: number,
    neatInstance?: Neat,
  ): void {
    applyDashboardUpdate(this.#createContext(), {
      maze,
      result,
      network,
      generation,
      neatInstance,
      telemetryHook: this._telemetryHook as DashboardTelemetryHook | undefined,
    });
  }

  /**
   * Return the latest public telemetry snapshot, including rich detail history when available.
   *
   * @returns Latest dashboard telemetry snapshot.
   */
  getLastTelemetry(): AsciiMazeTelemetrySnapshot {
    return getDashboardLastTelemetry(this.#state);
  }

  /** Clear archive, current best, and telemetry state so the instance can be reused. */
  reset(): void {
    resetDashboardState(this.#state);
  }

  #createContext(): DashboardManagerContext {
    return {
      state: this.#state,
      clearFn: this.#clearFn,
      logFn: this.#logFn,
      archiveFn: this.#archiveFn,
      logBlank: () => this.#logBlank(),
      formatStat: (
        label: string,
        value: string | number,
        colorLabel?: string,
        colorValue?: string,
        labelWidth?: number,
      ) =>
        formatDashboardStat(label, value, colorLabel, colorValue, labelWidth),
    };
  }

  #logBlank(): void {
    this.#logFn(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        C.FRAME_INNER_WIDTH,
        ' ',
      )}${colors.blueCore}║${colors.reset}`,
    );
  }
}
