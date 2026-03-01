import type {
  FlappyStatsKey,
  FlappyStatsRowDescriptor,
} from '../browser-entry/browser-entry.types';
import { FLAPPY_NEON_PALETTE } from './constants.palette';

/**
 * Browser HUD stats table constants.
 *
 * This module defines table ordering, hidden groups, and row metadata for the
 * runtime status panel shown during playback.
 */

/** Ordered keys for the runtime stats table. */
export const FLAPPY_STATS_KEYS = [
  'currentHeader',
  'currentFrames',
  'currentPipes',
  'currentMaxFrames',
  'currentMaxPipes',
  'currentArchitecture',
  'telemetryHeader',
  'telemetryActivationsPerFrame',
  'telemetrySimulationStepsPerRaf',
  'telemetryHudUpdatesPerSecond',
  'telemetryMinorGcPerMinute',
  'bestHeader',
  'bestFrames',
  'bestPipes',
  'bestMaxFrames',
  'bestMaxPipes',
  'bestArchitecture',
  'status',
] as const;

/** Ordered row descriptors rendered into the runtime stats table. */
export const FLAPPY_STATS_ROWS: readonly FlappyStatsRowDescriptor[] = [
  { key: 'currentHeader', label: 'Current run' },
  { key: 'currentFrames', label: 'Frames' },
  { key: 'currentPipes', label: 'Pipes' },
  { key: 'currentMaxFrames', label: 'Max frames' },
  { key: 'currentMaxPipes', label: 'Max pipes' },
  { key: 'currentArchitecture', label: 'NN architecture' },
  { key: 'telemetryHeader', label: 'Instrumentation' },
  { key: 'telemetryActivationsPerFrame', label: 'Act/frame' },
  { key: 'telemetrySimulationStepsPerRaf', label: 'Steps/RAF' },
  { key: 'telemetryHudUpdatesPerSecond', label: 'HUD upd/s' },
  { key: 'telemetryMinorGcPerMinute', label: 'Minor GC/min' },
  { key: 'bestHeader', label: 'Best run' },
  { key: 'bestFrames', label: 'Frames' },
  { key: 'bestPipes', label: 'Pipes' },
  { key: 'bestMaxFrames', label: 'Max frames' },
  { key: 'bestMaxPipes', label: 'Max pipes' },
  { key: 'bestArchitecture', label: 'NN architecture' },
  { key: 'status', label: 'Status' },
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
  'telemetryHeader',
  'bestHeader',
] as const;

/** Stats keys whose values should render multi-line architecture content. */
export const FLAPPY_STATS_ARCHITECTURE_KEYS: readonly FlappyStatsKey[] = [
  'currentArchitecture',
  'bestArchitecture',
] as const;

/** Border style for regular stats rows. */
export const FLAPPY_UI_STATS_ROW_BORDER = '1px double rgba(15,181,255,0.3)';

/** Border style for stats section headers. */
export const FLAPPY_UI_STATS_SECTION_BORDER = '2px double rgba(15,181,255,0.4)';

/** Legend header color. */
export const FLAPPY_NETWORK_LEGEND_HEADER_COLOR =
  FLAPPY_NEON_PALETTE.statusText;
