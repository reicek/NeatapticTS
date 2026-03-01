/** Runtime stats table key union used across browser-entry helpers. */
export type FlappyStatsKey =
  | 'currentHeader'
  | 'currentFrames'
  | 'currentPipes'
  | 'currentMaxFrames'
  | 'currentMaxPipes'
  | 'currentArchitecture'
  | 'telemetryHeader'
  | 'telemetryActivationsPerFrame'
  | 'telemetrySimulationStepsPerRaf'
  | 'telemetryHudUpdatesPerSecond'
  | 'telemetryMinorGcPerMinute'
  | 'bestHeader'
  | 'bestFrames'
  | 'bestPipes'
  | 'bestMaxFrames'
  | 'bestMaxPipes'
  | 'bestArchitecture'
  | 'status';

/** Declarative row descriptor for the runtime stats table. */
export interface FlappyStatsRowDescriptor {
  key: FlappyStatsKey;
  label: string;
}

/** Runtime lookup map of stat keys to writable value cells. */
export type FlappyStatsTableCells = Partial<
  Record<FlappyStatsKey, HTMLTableCellElement>
>;

/** Color pair used for stats category key/value styling. */
export interface FlappyStatsCategoryColors {
  keyColor: string;
  valueColor: string;
}

/** Input contract for declarative runtime stats table row builder. */
export interface CreateFlappyStatsTableRowsInput {
  statsTable: HTMLTableElement;
  enableRuntimeInstrumentation: boolean;
  resolveCategoryColor: (statsKey: FlappyStatsKey) => FlappyStatsCategoryColors;
}
