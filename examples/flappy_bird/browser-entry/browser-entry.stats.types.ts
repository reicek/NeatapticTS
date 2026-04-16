/**
 * HUD and runtime stats contracts for the Flappy Bird browser demo.
 *
 * The browser HUD is intentionally declarative: keys describe what should be
 * shown, and helper utilities map those keys to DOM rows and live values. The
 * panel is split into a live current-run section, a generation-summary section,
 * and optional instrumentation rows so the browser can show both immediate
 * playback state and worker-computed population context without recomputing
 * aggregates on the main thread.
 */

/** Runtime stats table key union used across browser-entry helpers. */
export type FlappyStatsKey =
  | 'currentHeader'
  | 'currentFrames'
  | 'currentPipes'
  | 'currentArchitecture'
  | 'summaryHeader'
  | 'summaryFitness'
  | 'summaryWinnerFrames'
  | 'summaryWinnerPipes'
  | 'summaryAveragePipes'
  | 'summaryP90Frames'
  | 'summaryArchitecture'
  | 'telemetryHeader'
  | 'telemetryActivationsPerFrame'
  | 'telemetrySimulationStepsPerRaf'
  | 'telemetryHudUpdatesPerSecond'
  | 'telemetryMinorGcPerMinute'
  | 'status'
  | 'birds';

/**
 * Declarative row descriptor for the runtime stats table.
 *
 * Each row is described as data first so the HUD can be assembled in a stable,
 * testable order instead of being hand-written imperatively.
 */
export interface FlappyStatsRowDescriptor {
  key: FlappyStatsKey;
  label: string;
}

/**
 * Runtime lookup map of stat keys to writable value cells.
 *
 * This acts like a small DOM index so the update loop can mutate the correct
 * cells directly without repeatedly querying the document.
 */
export type FlappyStatsTableCells = Partial<
  Record<FlappyStatsKey, HTMLTableCellElement>
>;

/**
 * Color pair used for stats category key/value styling.
 *
 * The HUD uses paired colors so labels and values stay visually grouped while
 * still separating categories such as current run, generation summary,
 * telemetry, and status.
 */
export interface FlappyStatsCategoryColors {
  keyColor: string;
  valueColor: string;
}

/**
 * Input contract for declarative runtime stats table row builder.
 *
 * The builder needs both the target table and a small policy surface that says
 * whether instrumentation rows should appear and how rows should be colored.
 */
export interface CreateFlappyStatsTableRowsInput {
  statsTable: HTMLTableElement;
  enableRuntimeInstrumentation: boolean;
  resolveCategoryColor: (statsKey: FlappyStatsKey) => FlappyStatsCategoryColors;
}
