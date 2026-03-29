/**
 * Search-visibility boundary for ASCII Maze browser and terminal hosts.
 *
 * Evolution is difficult to trust when all you see is a final score. The
 * dashboard boundary turns per-generation updates into a live board, a solved
 * archive, and a reusable telemetry snapshot so long runs stay legible while
 * they are still happening.
 *
 * That is why this folder exists as a separate boundary instead of staying
 * mixed into the engine. The engine should answer whether the search is making
 * progress. The dashboard should answer how a human can see that progress:
 * current best candidate, recent trends, species counts, complexity drift, and
 * solved artifacts worth preserving.
 *
 * Read the folder as three cooperating shelves. The public `DashboardManager`
 * class keeps the stable host-facing API. `dashboardManager.services.ts`
 * handles update ingestion, redraws, and telemetry emission. The archive, live,
 * and telemetry subfolders keep presentation concerns decomposed so browser and
 * terminal hosts can reuse the same core state with different outputs.
 *
 * The chapter matters because telemetry without presentation quickly becomes a
 * pile of numbers, while presentation without a stable telemetry contract
 * becomes fragile host-specific logic. This boundary keeps those concerns close
 * enough to cooperate and separate enough to evolve independently.
 *
 * Read this chapter in three passes. Start with the class surface for the
 * public update, redraw, and snapshot methods. Continue to the service module
 * for the runtime dataflow. Finish in `archive/`, `live/`, and `telemetry/`
 * when you want the host-specific rendering details rather than the stable
 * dashboard contract.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Engine["Evolution engine update"]:::base --> Dashboard["DashboardManager.update(...)\npublic ingest surface"]:::accent
 *   Dashboard --> Live["live board\ncurrent best and trends"]:::base
 *   Dashboard --> Archive["solved archive\nretained artifacts"]:::base
 *   Dashboard --> Telemetry["public telemetry snapshot\nhost subscriptions"]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Hosts["Browser or terminal host"]:::base --> Facade["DashboardManager\nclass facade"]:::accent
 *   Facade --> Services["dashboardManager.services.ts\nstateful orchestration"]:::base
 *   Services --> State["history state\ncurrent best\nlast telemetry"]:::base
 *   Services --> Views["archive/ live/ telemetry/\nrendering helpers"]:::base
 * ```
 *
 * Example: create one dashboard instance for a run and feed it updates.
 *
 * ```ts
 * const dashboard = new DashboardManager(clearOutput, logOutput, archiveOutput);
 *
 * dashboard.update(maze, result, network, generation, neatInstance);
 * dashboard.redraw(maze, neatInstance);
 * ```
 *
 * Example: read the latest public telemetry snapshot after several updates.
 *
 * ```ts
 * const telemetry = dashboard.getLastTelemetry();
 *
 * console.log(telemetry.generation);
 * console.log(telemetry.bestFitness);
 * ```
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
