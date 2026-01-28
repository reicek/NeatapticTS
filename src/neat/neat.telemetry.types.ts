import type { ConnectionLike, NodeLike, TelemetryEntry } from './neat.types';

/** Minimal genome shape used by telemetry helpers. */
export type TelemetryGenome = {
  nodes: NodeLike[];
  connections: ConnectionLike[];
  _depth?: number;
};

/** Diversity telemetry options for sampling and novelty defaults. */
export type TelemetryDiversityOptions = {
  diversityMetrics?: {
    enabled?: boolean;
    pairSample?: number;
    graphletSample?: number;
  };
  fastMode?: boolean;
  novelty?: { enabled?: boolean; k?: number };
};

/** Minimal telemetry stream options for streaming helpers. */
export type TelemetryStreamOptions = {
  telemetryStream?: {
    enabled?: boolean;
    onEntry?: (entry: TelemetryEntry) => void;
  };
};

/** Telemetry entry shape used for constructing snapshots. */
export type TelemetryEntryRecord = TelemetryEntry & Record<string, unknown>;

/** Operator stats map shape for telemetry extraction. */
export type OperatorStatsMap = Map<
  string,
  { success: number; attempts: number }
>;

/** Minimal telemetry buffer context shape. */
export type TelemetryBufferContext = {
  _telemetry?: TelemetryEntry[];
};

/** Minimal telemetry selection context shape. */
export type TelemetrySelectContext = {
  _telemetrySelect?: Set<string>;
};

/**
 * Core telemetry field keys used by selection helpers.
 */
export type TelemetryCoreFields = readonly string[];
