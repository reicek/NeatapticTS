import {
  FLAPPY_INSTRUMENTATION_STATS_KEYS,
  FLAPPY_STATS_ARCHITECTURE_KEYS,
  FLAPPY_STATS_ROWS,
  FLAPPY_STATS_SECTION_KEYS,
  FLAPPY_UI_STATS_ROW_BORDER,
  FLAPPY_UI_STATS_SECTION_BORDER,
} from './constants';
import type {
  CreateFlappyStatsTableRowsInput,
  FlappyStatsKey,
} from './browser-entry.types';

/**
 * Builds stats table rows and returns value-cell lookup by key.
 *
 * @param input - Table construction inputs.
 * @returns Mapping from stat key to value cell.
 */
export function createFlappyStatsTableRows(
  input: CreateFlappyStatsTableRowsInput,
): Partial<Record<FlappyStatsKey, HTMLTableCellElement>> {
  // Step 1: Prepare row-key lookup sets used by rendering branches.
  const statsValueByKey: Partial<Record<FlappyStatsKey, HTMLTableCellElement>> =
    {};
  const instrumentationStatsKeys = new Set<FlappyStatsKey>(
    FLAPPY_INSTRUMENTATION_STATS_KEYS,
  );
  const sectionKeys = new Set<FlappyStatsKey>(FLAPPY_STATS_SECTION_KEYS);
  const architectureKeys = new Set<FlappyStatsKey>(
    FLAPPY_STATS_ARCHITECTURE_KEYS,
  );

  // Step 2: Build rows declaratively from configured row descriptors.
  FLAPPY_STATS_ROWS.forEach((statsRow) => {
    // Step 2.1: Skip instrumentation rows when runtime instrumentation is disabled.
    if (
      !input.enableRuntimeInstrumentation &&
      instrumentationStatsKeys.has(statsRow.key)
    ) {
      return;
    }

    const rowElement = document.createElement('tr');
    const { keyColor, valueColor } = input.resolveCategoryColor(statsRow.key);

    // Step 2.2: Render section-header rows as full-width `<th>` cells.
    if (sectionKeys.has(statsRow.key)) {
      const sectionCell = document.createElement('th');
      sectionCell.colSpan = 2;
      sectionCell.textContent = statsRow.label;
      sectionCell.style.color = keyColor;
      sectionCell.style.textAlign = 'left';
      sectionCell.style.padding = '6px 0 4px 0';
      sectionCell.style.fontWeight = '700';
      sectionCell.style.textTransform = 'uppercase';
      sectionCell.style.letterSpacing = '0.06em';
      sectionCell.style.borderBottom = FLAPPY_UI_STATS_SECTION_BORDER;
      rowElement.appendChild(sectionCell);
      input.statsTable.appendChild(rowElement);
      statsValueByKey[statsRow.key] = sectionCell as HTMLTableCellElement;
      return;
    }

    // Step 2.3: Render standard key/value stat rows.
    const keyCell = document.createElement('th');
    keyCell.textContent = statsRow.label;
    keyCell.style.color = keyColor;
    keyCell.style.textAlign = 'left';
    keyCell.style.padding = '2px 8px 2px 0';
    keyCell.style.fontWeight = '600';
    keyCell.style.textTransform = 'uppercase';
    keyCell.style.letterSpacing = '0.04em';
    keyCell.style.width = '20%';
    keyCell.style.borderBottom = FLAPPY_UI_STATS_ROW_BORDER;

    const valueCell = document.createElement('td');
    valueCell.textContent = '-';
    valueCell.style.color = valueColor;
    valueCell.style.textAlign = 'left';
    valueCell.style.padding = '2px 0';
    valueCell.style.borderBottom = FLAPPY_UI_STATS_ROW_BORDER;
    valueCell.style.whiteSpace = 'nowrap';
    valueCell.style.overflow = 'hidden';
    valueCell.style.textOverflow = 'ellipsis';

    // Step 2.4: Expand architecture rows to multi-line wrapping mode.
    if (architectureKeys.has(statsRow.key)) {
      valueCell.style.whiteSpace = 'pre-line';
      valueCell.style.overflow = 'visible';
      valueCell.style.textOverflow = 'clip';
      valueCell.style.wordBreak = 'break-word';
      valueCell.style.lineHeight = '1.2';
    }

    rowElement.appendChild(keyCell);
    rowElement.appendChild(valueCell);
    input.statsTable.appendChild(rowElement);
    statsValueByKey[statsRow.key] = valueCell;
  });

  // Step 3: Return key-to-cell lookup for incremental HUD updates.
  return statsValueByKey;
}

/**
 * Splits architecture suffix onto a second line for readability in the stats table.
 *
 * @param architectureValue - Full architecture label.
 * @returns Line-broken label value.
 */
export function formatArchitectureStatsValue(
  architectureValue: string,
): string {
  return architectureValue.replace(
    /\s*(\(\d+\s+nodes,\s*\d+\s+connections\))$/,
    '\n$1',
  );
}
