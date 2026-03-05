import {
  createFlappyStatsTableRows,
  formatArchitectureStatsValue,
} from '../browser-entry.stats.utils';
import {
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_HUD_INITIALIZING_TEXT,
  FLAPPY_HUD_OFF_TEXT,
  FLAPPY_HUD_ZERO_DECIMAL_TEXT,
  FLAPPY_HUD_ZERO_TEXT,
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_STATS_KEYS,
  FLAPPY_UI_STATS_ROW_BORDER,
} from '../../constants/constants';
import { FLAPPY_HOST_TABLE_FONT_SIZE } from './host.constants';
import type {
  FlappyStatsKey,
  FlappyStatsTableCells,
} from '../browser-entry.types';
import type { HostStatsPartialValues } from './host.types';

/**
 * Creates the host stats table, appends it into the provided host element, and
 * initializes all HUD values to their baseline placeholders.
 *
 * @param statsTableHost - DOM host that receives the table.
 * @returns Lookup map for future incremental stat updates.
 */
export function createAndAttachHostStatsTable(
  statsTableHost: HTMLElement,
): FlappyStatsTableCells {
  // Step 1: Create table element with shared visual styling.
  const statsTable = document.createElement('table');
  statsTable.style.borderCollapse = 'collapse';
  statsTable.style.width = '100%';
  statsTable.style.maxWidth = '100%';
  statsTable.style.tableLayout = 'fixed';
  statsTable.style.fontFamily = FLAPPY_MONOSPACE_FONT_FAMILY;
  statsTable.style.fontSize = FLAPPY_HOST_TABLE_FONT_SIZE;
  statsTable.style.borderBottom = FLAPPY_UI_STATS_ROW_BORDER;

  // Step 2: Build row value-cell lookup using category color policy.
  const statsValueByKey = createFlappyStatsTableRows({
    statsTable,
    enableRuntimeInstrumentation: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
    resolveCategoryColor: (
      statsKey,
    ): { keyColor: string; valueColor: string } => {
      if (statsKey === 'status') {
        return {
          keyColor: FLAPPY_NEON_PALETTE.statusText,
          valueColor: FLAPPY_NEON_PALETTE.statusText,
        };
      }
      if (statsKey === 'birds') {
        return {
          keyColor: FLAPPY_NEON_PALETTE.statusText,
          valueColor: FLAPPY_NEON_PALETTE.statusText,
        };
      }
      if (statsKey.startsWith('current')) {
        return {
          keyColor: FLAPPY_NEON_PALETTE.currentRunText,
          valueColor: FLAPPY_NEON_PALETTE.currentRunText,
        };
      }
      if (statsKey.startsWith('best')) {
        return {
          keyColor: FLAPPY_NEON_PALETTE.bestRunText,
          valueColor: FLAPPY_NEON_PALETTE.bestRunText,
        };
      }
      return {
        keyColor: FLAPPY_NEON_PALETTE.hudText,
        valueColor: FLAPPY_NEON_PALETTE.hudAccent,
      };
    },
  });

  // Step 3: Seed default table values before first simulation tick.
  updateStatsTableValues(statsValueByKey, {
    currentHeader: 'Current run · Gen -',
    currentFrames: FLAPPY_HUD_ZERO_TEXT,
    currentPipes: FLAPPY_HUD_ZERO_TEXT,
    currentMaxFrames: FLAPPY_HUD_ZERO_TEXT,
    currentMaxPipes: FLAPPY_HUD_ZERO_TEXT,
    currentArchitecture: '-',
    telemetryHeader: 'Instrumentation',
    telemetryActivationsPerFrame: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetrySimulationStepsPerRaf: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetryHudUpdatesPerSecond: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetryMinorGcPerMinute: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    bestHeader: 'Best run',
    bestFrames: FLAPPY_HUD_ZERO_TEXT,
    bestPipes: FLAPPY_HUD_ZERO_TEXT,
    bestMaxFrames: FLAPPY_HUD_ZERO_TEXT,
    bestMaxPipes: FLAPPY_HUD_ZERO_TEXT,
    bestArchitecture: '-',
    status: FLAPPY_HUD_INITIALIZING_TEXT,
  });

  // Step 4: Attach completed table to the host panel.
  statsTableHost.appendChild(statsTable);
  return statsValueByKey;
}

/**
 * Applies partial stat updates to the rendered stats table.
 *
 * @param statsValueByKey - Lookup of stat keys to value cells.
 * @param partialValues - Subset of values to write this tick.
 * @returns Nothing.
 */
export function updateStatsTableValues(
  statsValueByKey: FlappyStatsTableCells,
  partialValues: HostStatsPartialValues,
): void {
  FLAPPY_STATS_KEYS.forEach((statsKey: FlappyStatsKey) => {
    const nextValue = partialValues[statsKey];
    if (nextValue == null) {
      return;
    }
    const statsCell = statsValueByKey[statsKey];
    if (!statsCell) {
      return;
    }
    const formattedValue =
      statsKey === 'currentArchitecture' || statsKey === 'bestArchitecture'
        ? formatArchitectureStatsValue(nextValue)
        : nextValue;
    statsCell.textContent = formattedValue;
  });
}
