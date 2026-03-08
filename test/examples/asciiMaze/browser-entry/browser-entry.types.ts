import type { DashboardManager } from '../dashboardManager';
import type { EvolutionHostAdapter } from '../evolutionEngine/evolutionEngine.types';
import type { INetwork } from '../interfaces';

/**
 * Public lifecycle handle returned by the browser demo entrypoint.
 *
 * @example
 * ```ts
 * const handle = await start('ascii-maze-output');
 * const unsubscribe = handle.onTelemetry((telemetry) => {
 *   console.log('generation', telemetry.generation);
 * });
 *
 * await handle.done;
 * unsubscribe();
 * ```
 */
export interface AsciiMazeRunHandle {
  /** Stop the running curriculum. This also aborts the internal signal. */
  stop: () => void;
  /** Whether the curriculum is still active. */
  isRunning: () => boolean;
  /** Promise that resolves when the curriculum finishes or is stopped. */
  done: Promise<void>;
  /** Subscribe to lightweight per-generation telemetry snapshots. */
  onTelemetry: (
    listener: (telemetry: Record<string, unknown>) => void,
  ) => () => void;
  /** Pull the most recently emitted telemetry snapshot, if one exists. */
  getTelemetry: () => unknown;
}

/** Options accepted by the browser-hosted ASCII Maze entrypoint. */
export interface BrowserEntryStartOptions {
  /** Optional external abort signal that cooperatively stops the curriculum. */
  signal?: AbortSignal;
}

/** Stable callable shape used by globals compatibility wiring. */
export type BrowserEntryStartFunction = (
  container?: string | HTMLElement,
  opts?: BrowserEntryStartOptions,
) => Promise<AsciiMazeRunHandle>;

/** Runtime dashboard surface used by the browser entry host adapter. */
export interface RuntimeDashboard {
  _telemetryHook?: (telemetry: Record<string, unknown>) => void;
  redraw?: (data: unknown[], state: unknown) => void;
  getLastTelemetry?: () => Record<string, unknown>;
  [key: string]: unknown;
}

/** Runtime AbortSignal shape used for older or polyfilled environments. */
export interface RuntimeAbortSignal {
  aborted?: boolean;
  onabort?: (() => void) | null;
  addEventListener: AbortSignal['addEventListener'];
  removeEventListener: AbortSignal['removeEventListener'];
  dispatchEvent: AbortSignal['dispatchEvent'];
  reason?: unknown;
  throwIfAborted?: () => void;
}

/** AbortSignal constructor shape with optional static composition helpers. */
export interface RuntimeAbortSignalConstructor {
  prototype: AbortSignal;
  new (): AbortSignal;
  any?: (signals: AbortSignal[]) => AbortSignal;
}

/** Runtime evolution result shape used by the browser curriculum adapter. */
export interface RuntimeEvolutionResult {
  bestResult?: {
    progress?: number;
    [key: string]: unknown;
  };
  bestNetwork?: INetwork;
  [key: string]: unknown;
}

/** Global namespace exposed for direct browser-script loading compatibility. */
export interface RuntimeWindow extends Window {
  asciiMaze?: {
    start?: BrowserEntryStartFunction;
    _autoStarted?: boolean;
    [key: string]: unknown;
  };
  asciiMazePaused?: boolean;
  asciiMazeStart?: (containerElement?: unknown) => unknown;
  [key: string]: unknown;
}

/** Resolved host elements used by logger, dashboard, and resize services. */
export interface BrowserEntryHostElements {
  hostElement: HTMLElement | null;
  archiveElement: HTMLElement | null;
  liveElement: HTMLElement | null;
  observeTarget: HTMLElement | null;
}

/** Evolution settings used for a single procedural maze phase. */
export interface BrowserEntryEvolutionSettings {
  agentMaxSteps: number;
  popSize: number;
  maxStagnantGenerations: number;
  maxGenerations: number;
  lamarckianIterations: number;
  lamarckianSampleSize: number;
  mazeFactory: () => string[];
}

/** Lightweight telemetry hub contract shared between host and public handle. */
export interface BrowserEntryTelemetryHub<
  TTelemetry extends Record<string, unknown>,
> {
  add(listener: (payload: TTelemetry) => void): () => void;
  dispatch(payload: TTelemetry): void;
}

/** Browser host services assembled for one running demo instance. */
export interface BrowserEntryHostServices {
  dashboard: DashboardManager;
  runtimeDashboard: RuntimeDashboard;
  telemetryHub: BrowserEntryTelemetryHub<Record<string, unknown>>;
  disposeResizeHandling: () => void;
}

/** State and callbacks used by the curriculum runtime service. */
export interface BrowserEntryCurriculumContext {
  dashboard: DashboardManager;
  combinedSignal: AbortSignal;
  isCancelled: () => boolean;
  finish: () => void;
  hostAdapter: EvolutionHostAdapter;
}
