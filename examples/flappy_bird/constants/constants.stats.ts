import type {
  FlappyStatsKey,
  FlappyStatsRowDescriptor,
} from '../browser-entry/browser-entry.types';
import { FLAPPY_NEON_PALETTE } from './constants.palette';

/**
 * Browser HUD stats table constants.
 *
 * This module defines the teaching layout for the browser HUD: a live
 * current-run block, a generation-summary block fed by worker-side aggregates,
 * and an optional instrumentation block for runtime diagnostics.
 */

/** Ordered keys for the runtime stats table in rendered top-to-bottom order. */
export const FLAPPY_STATS_KEYS = [
  'currentHeader',
  'currentFrames',
  'currentPipes',
  'currentArchitecture',
  'summaryHeader',
  'summaryFitness',
  'summaryWinnerFrames',
  'summaryWinnerPipes',
  'summaryAveragePipes',
  'summaryP90Frames',
  'summaryArchitecture',
  'telemetryHeader',
  'telemetryActivationsPerFrame',
  'telemetrySimulationStepsPerRaf',
  'telemetryHudUpdatesPerSecond',
  'telemetryMinorGcPerMinute',
  'status',
  'birds',
] as const;

/** Ordered row descriptors rendered into the runtime stats table. */
export const FLAPPY_STATS_ROWS: readonly FlappyStatsRowDescriptor[] = [
  { key: 'currentHeader', label: 'Current run' },
  { key: 'currentFrames', label: 'Frames' },
  { key: 'currentPipes', label: 'Pipes' },
  { key: 'currentArchitecture', label: 'NN architecture' },
  { key: 'summaryHeader', label: 'Generation summary' },
  { key: 'summaryFitness', label: 'Fitness' },
  { key: 'summaryWinnerFrames', label: 'Winner frames' },
  { key: 'summaryWinnerPipes', label: 'Winner pipes' },
  { key: 'summaryAveragePipes', label: 'Avg pipes' },
  { key: 'summaryP90Frames', label: 'P90 frames' },
  { key: 'summaryArchitecture', label: 'NN architecture' },
  { key: 'telemetryHeader', label: 'Instrumentation' },
  { key: 'telemetryActivationsPerFrame', label: 'Act/frame' },
  { key: 'telemetrySimulationStepsPerRaf', label: 'Steps/RAF' },
  { key: 'telemetryHudUpdatesPerSecond', label: 'HUD upd/s' },
  { key: 'telemetryMinorGcPerMinute', label: 'Minor GC/min' },
  { key: 'status', label: 'Status' },
  { key: 'birds', label: 'Birds' },
] as const;

/** Stats keys that are hidden when runtime instrumentation is disabled. */
export const FLAPPY_INSTRUMENTATION_STATS_KEYS: readonly FlappyStatsKey[] = [
  'telemetryHeader',
  'telemetryActivationsPerFrame',
  'telemetrySimulationStepsPerRaf',
  'telemetryHudUpdatesPerSecond',
  'telemetryMinorGcPerMinute',
] as const;

/** Stats keys rendered as section headers rather than key/value rows. */
export const FLAPPY_STATS_SECTION_KEYS: readonly FlappyStatsKey[] = [
  'currentHeader',
  'summaryHeader',
  'telemetryHeader',
] as const;

/** Stats keys whose values should render multi-line architecture content. */
export const FLAPPY_STATS_ARCHITECTURE_KEYS: readonly FlappyStatsKey[] = [
  'currentArchitecture',
  'summaryArchitecture',
] as const;

/** Border style for regular stats rows. */
export const FLAPPY_UI_STATS_ROW_BORDER = '1px double rgba(15,181,255,0.3)';

/** Border style for stats section headers. */
export const FLAPPY_UI_STATS_SECTION_BORDER = '2px double rgba(15,181,255,0.4)';

/** Legend header color. */
export const FLAPPY_NETWORK_LEGEND_HEADER_COLOR =
  FLAPPY_NEON_PALETTE.statusText;
