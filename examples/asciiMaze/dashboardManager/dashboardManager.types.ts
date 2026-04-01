import type Neat from '../../../src/neat';
import type {
  IDashboardManager,
  IMazeRunResult,
  INetwork,
} from '../interfaces';

/** NEAT genome/network with runtime properties used by dashboard telemetry. */
export interface NeatGenome {
  score?: number;
  nodes?: Array<{ type?: string; [key: string]: unknown }>;
  connections?: Array<{ enabled?: boolean; [key: string]: unknown }>;
  [key: string]: unknown;
}

/** NEAT species collection entry surface used by dashboard telemetry. */
export interface NeatSpecies {
  members?: unknown[];
  length?: number;
  [key: string]: unknown;
}

/** NEAT instance shape needed by dashboard helpers. */
export interface NeatInstance {
  population?: NeatGenome[];
  species?: NeatSpecies[];
  getTelemetry?: () => unknown[];
  getOperatorStats?: () => unknown[];
  getMutationStats?: () => Record<string, number>;
  getNoveltyArchive?: () => unknown[];
  [key: string]: unknown;
}

/** Operator stats entry from NEAT. */
export interface OperatorStatsEntry {
  name: string;
  success: number;
  attempts: number;
  accepted?: number;
  total?: number;
  [key: string]: unknown;
}

export type NumericTelemetryMap = Record<string, number | null | undefined>;

export interface AsciiMazeComplexityStats extends NumericTelemetryMap {
  meanNodes?: number | null;
  meanConns?: number | null;
  growthNodes?: number | null;
  growthConns?: number | null;
}

export type MutationStatsMap = Record<string, unknown>;

/** Raw telemetry shape received from NEAT dashboard integrations. */
export interface DashboardTelemetry {
  complexity?: AsciiMazeComplexityStats | null;
  perf?: NumericTelemetryMap | null;
  lineage?: NumericTelemetryMap | null;
  diversity?: NumericTelemetryMap | null;
  fronts?: ReadonlyArray<ReadonlyArray<unknown>> | null;
  objectives?: NumericTelemetryMap | null;
  hyper?: number | null;
  mutationStats?: MutationStatsMap | null;
  mutation?: { stats?: MutationStatsMap | null } | null;
  species?: number | null;
  saturationFraction?: number | null;
  actionEntropy?: number | null;
  populationMean?: number | null;
  populationMedian?: number | null;
  enabledConnRatio?: number | null;
  bestFitness?: number | null;
  bestFitnessDelta?: number | null;
  topSpeciesSizes?: number[] | null;
  noveltyArchiveSize?: number | null;
  operatorAcceptance?: Array<{ name: string; acceptancePct: number }> | null;
  topMutations?: Array<{ name: string; count: number }> | null;
  trends?: {
    fitness?: string | null;
    nodes?: string | null;
    conns?: string | null;
    hyper?: string | null;
    progress?: string | null;
    species?: string | null;
  } | null;
  histories?: {
    bestFitness?: number[];
    nodes?: number[];
    conns?: number[];
    hyper?: number[];
    progress?: number[];
    species?: number[];
  } | null;
  timestamp?: number;
  generation?: number;
}

/** Expanded telemetry details retained by the dashboard between redraws. */
export interface AsciiMazeDetailedStats {
  generation: number;
  bestFitness: number | null;
  bestFitnessDelta: number | null;
  saturationFraction: number | null;
  actionEntropy: number | null;
  populationMean: number | null;
  populationMedian: number | null;
  enabledConnRatio: number | null;
  complexity: AsciiMazeComplexityStats | null;
  simplifyPhaseActive: boolean;
  perf: NumericTelemetryMap | null;
  lineage: NumericTelemetryMap | null;
  diversity: NumericTelemetryMap | null;
  speciesCount: number | null;
  topSpeciesSizes: number[] | null;
  objectives: NumericTelemetryMap | null;
  paretoFrontSizes: number[] | null;
  firstFrontSize: number;
  hypervolume: number | null;
  noveltyArchiveSize: number | null;
  operatorAcceptance: Array<{ name: string; acceptancePct: number }> | null;
  topMutations: Array<{ name: string; count: number }> | null;
  mutationStats: MutationStatsMap | null;
  trends: {
    fitness: string | null;
    nodes: string | null;
    conns: string | null;
    hyper: string | null;
    progress: string | null;
    species: string | null;
  };
  histories: {
    bestFitness: number[];
    nodes: number[];
    conns: number[];
    hyper: number[];
    progress: number[];
    species: number[];
  };
  timestamp: number;
}

/** Public telemetry snapshot surfaced to browser consumers. */
export interface AsciiMazeTelemetrySnapshot {
  generation: number;
  bestFitness: number | null;
  progress: number | null;
  speciesCount: number | null;
  gensPerSec: number;
  timestamp: number;
  details: AsciiMazeDetailedStats | null;
}

/** Telemetry payload emitted through events, postMessage, and runtime hooks. */
export interface DashboardTelemetryPayload {
  type: 'asciiMaze:telemetry';
  generation: number;
  bestFitness: number | null;
  progress: number | null;
  speciesCount: number | null;
  gensPerSec: number;
  timestamp: number;
  details: AsciiMazeDetailedStats | null;
}

/** Stored solved-maze archive entry. */
export interface SolvedMazeRecord {
  maze: string[];
  result: IMazeRunResult;
  network: INetwork;
  generation: number;
}

/** Latest best candidate used by live rendering and telemetry. */
export interface CurrentBestRecord {
  result: IMazeRunResult;
  network: INetwork | null;
  generation: number;
}

/** Bounded numeric histories used for trends and exports. */
export interface DashboardHistoryState {
  bestFitness: number[];
  complexityNodes: number[];
  complexityConns: number[];
  hypervolume: number[];
  progress: number[];
  speciesCount: number[];
}

/** Reused scratch arrays to keep redraw allocations predictable. */
export interface DashboardScratchState {
  scores: number[];
  speciesSizes: number[];
  operatorStats: OperatorStatsEntry[];
  mutationEntries: [string, number][];
}

/** Mutable runtime state owned by one dashboard instance. */
export interface DashboardManagerState {
  solvedMazes: SolvedMazeRecord[];
  solvedMazeKeys: Set<string>;
  currentBest: CurrentBestRecord | null;
  lastTelemetry: DashboardTelemetry | null;
  lastBestFitness: number | null;
  histories: DashboardHistoryState;
  lastDetailedStats: AsciiMazeDetailedStats | null;
  runStartTs: number | null;
  perfStart: number | null;
  lastGeneration: number | null;
  lastUpdateTs: number | null;
  scratch: DashboardScratchState;
}

export type DashboardClearFunction = () => void;
export type DashboardLogFunction = (...args: unknown[]) => void;
export type DashboardArchiveFunction = (...args: unknown[]) => void;
export type DashboardTelemetryHook = (
  payload: DashboardTelemetryPayload,
) => void;

/** Shared runtime context passed into dashboard services. */
export interface DashboardManagerContext {
  state: DashboardManagerState;
  clearFn: DashboardClearFunction;
  logFn: DashboardLogFunction;
  archiveFn?: DashboardArchiveFunction;
  logBlank: () => void;
  formatStat: (
    label: string,
    value: string | number,
    colorLabel?: string,
    colorValue?: string,
    labelWidth?: number,
  ) => string;
}

/** Input accepted by the update orchestration service. */
export interface DashboardManagerUpdateArgs {
  maze: string[];
  result: IMazeRunResult | undefined;
  network: INetwork | null;
  generation: number;
  neatInstance?: Neat;
  telemetryHook?: DashboardTelemetryHook;
}

/**
 * Shared presentation adapter used by browser and non-browser hosts.
 *
 * @remarks
 * Host wiring should prefer this interface when it needs redraw and telemetry
 * access without depending on the concrete `DashboardManager` implementation.
 */
export interface DashboardPresentationAdapter extends IDashboardManager {
  _telemetryHook?: DashboardTelemetryHook;
  redraw?: (currentMaze: string[], neat?: unknown) => void;
  getLastTelemetry?: () => AsciiMazeTelemetrySnapshot;
}

/**
 * Compatibility alias for older runtime-facing imports.
 *
 * @remarks
 * Step 7 promotes `DashboardPresentationAdapter` as the primary owner of this
 * presentation seam while preserving the existing export name.
 */
export type RuntimeDashboardManager = DashboardPresentationAdapter;
