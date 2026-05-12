import type { DashboardManager } from '../dashboardManager';
import type {
  AsciiMazeTelemetrySnapshot,
  DashboardPresentationAdapter,
  DashboardTelemetryPayload,
} from '../dashboardManager/dashboardManager.types';
import type {
  EvolutionAdaptiveMutationConfig,
  EvolutionHostAdapter,
} from '../evolutionEngine/evolutionEngine.types';
import type { ExampleArchitectureProfileId } from '../../architectureProfiles';

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
  /** Subscribe to per-generation telemetry payloads emitted by the dashboard boundary. */
  onTelemetry: (
    listener: (telemetry: DashboardTelemetryPayload) => void,
  ) => () => void;
  /** Pull the most recently emitted telemetry snapshot, if one exists. */
  getTelemetry: () => AsciiMazeTelemetrySnapshot | undefined;
}

/** Options accepted by the browser-hosted ASCII Maze entrypoint. */
export interface BrowserEntryStartOptions {
  /** Optional external abort signal that cooperatively stops the curriculum. */
  signal?: AbortSignal;
  /** Architecture profile to use for this run. Defaults to the maze default when omitted. */
  architectureProfileId?: ExampleArchitectureProfileId;
}

/** Stable callable shape used by globals compatibility wiring. */
export type BrowserEntryStartFunction = (
  container?: string | HTMLElement,
  opts?: BrowserEntryStartOptions,
) => Promise<AsciiMazeRunHandle>;

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
  networkCanvasElement: HTMLCanvasElement | null;
  observeTarget: HTMLElement | null;
  /** Container for architecture selector buttons, populated by JS after mount. */
  archButtonsElement: HTMLElement | null;
}

/** Evolution settings used for a single procedural maze phase. */
export interface BrowserEntryEvolutionSettings {
  agentMaxSteps: number;
  allowRecurrent: boolean;
  adaptiveMutation?: EvolutionAdaptiveMutationConfig;
  popSize: number;
  maxStagnantGenerations: number;
  maxGenerations: number;
  lamarckianIterations: number;
  lamarckianSampleSize: number;
  mazeFactory: () => string[];
}

/** Lightweight telemetry hub contract shared between host and public handle. */
export interface BrowserEntryTelemetryHub<TTelemetry extends object> {
  add(listener: (payload: TTelemetry) => void): () => void;
  dispatch(payload: TTelemetry): void;
}

/** Browser host services assembled for one running demo instance. */
export interface BrowserEntryHostServices {
  dashboard: DashboardManager;
  runtimeDashboard: DashboardPresentationAdapter;
  telemetryHub: BrowserEntryTelemetryHub<DashboardTelemetryPayload>;
  disposeResizeHandling: () => void;
}

/** State and callbacks used by the curriculum runtime service. */
export interface BrowserEntryCurriculumContext {
  dashboard: DashboardManager;
  combinedSignal: AbortSignal;
  isCancelled: () => boolean;
  finish: () => void;
  hostAdapter: EvolutionHostAdapter;
  /** Shared architecture profile id used to seed every phase of this curriculum run. */
  architectureProfileId?: ExampleArchitectureProfileId;
}
