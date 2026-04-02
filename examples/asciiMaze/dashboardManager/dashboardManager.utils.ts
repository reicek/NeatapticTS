import { colors } from '../colors';
import type { IMazeRunResult, INetwork } from '../interfaces';
import { MazeUtils } from '../mazeUtils';
import { NetworkVisualization } from '../networkVisualization';
import { DASHBOARD_MANAGER_CONSTANTS as C } from './dashboardManager.constants';

/**
 * Format a single framed dashboard stat line with aligned label and value columns.
 *
 * @param label - Descriptive stat label.
 * @param value - String or number value displayed after the label.
 * @param colorLabel - Color token applied to the label segment.
 * @param colorValue - Color token applied to the value segment.
 * @param labelWidth - Fixed width used for the label column.
 * @returns Ready-to-log framed stat line.
 */
export function formatDashboardStat(
  label: string,
  value: string | number,
  colorLabel = colors.neonSilver,
  colorValue = colors.cyanNeon,
  labelWidth: number = C.STAT_LABEL_WIDTH,
): string {
  const canonicalLabel = label.endsWith(':') ? label : `${label}:`;
  const paddedLabel = canonicalLabel.padEnd(labelWidth, ' ');
  const valueString = typeof value === 'number' ? `${value}` : String(value);
  const coloredContent = `${colorLabel}${paddedLabel}${colorValue} ${valueString}${colors.reset}`;
  return `${colors.blueCore}║${' '.repeat(C.LEFT_PADDING)}${NetworkVisualization.pad(
    coloredContent,
    C.CONTENT_WIDTH,
    ' ',
    'left',
  )}${' '.repeat(C.RIGHT_PADDING)}${colors.blueCore}║${colors.reset}`;
}

/**
 * Convert the recent tail of a numeric series into a compact sparkline.
 *
 * @param series - Numeric history in chronological order.
 * @param width - Maximum sample count included in the sparkline.
 * @returns Unicode sparkline string.
 */
export function buildDashboardSparkline(series: number[], width = 32): string {
  if (!Array.isArray(series) || !series.length || width <= 0) return '';

  const tailSlice = MazeUtils.tail<number>(series, width);
  if (!tailSlice.length) return '';

  const numericSamples = tailSlice.filter((sampleValue) =>
    Number.isFinite(sampleValue),
  );
  if (!numericSamples.length) return '';

  const minValue = Math.min(...numericSamples);
  const maxValue = Math.max(...numericSamples);
  const safeRange =
    Math.abs(maxValue - minValue) < C.DELTA_EPSILON
      ? C.DELTA_EPSILON
      : maxValue - minValue;
  const rampTopIndex = C.SPARK_BLOCKS.length - 1;

  return numericSamples
    .map((sampleValue) => {
      const normalized = (sampleValue - minValue) / safeRange;
      const blockIndex = Math.min(
        rampTopIndex,
        Math.max(0, Math.floor(normalized * rampTopIndex)),
      );
      return C.SPARK_BLOCKS[blockIndex];
    })
    .join('');
}

/**
 * Build a lightweight dedupe key for a maze layout.
 *
 * @param maze - Maze rows in display order.
 * @returns Joined maze key used by the solved archive.
 */
export function getDashboardMazeKey(maze: string[]): string {
  return maze.join('');
}

/**
 * Compute solved-path efficiency and visitation metrics for archive output.
 *
 * @param maze - Maze layout containing start and exit markers.
 * @param result - Run result with path, steps, and fitness.
 * @returns Derived path metrics used by solved archive formatting.
 */
export function computeDashboardPathMetrics(
  maze: string[],
  result: Pick<IMazeRunResult, 'path' | 'steps' | 'fitness'>,
): {
  optimalLength: number;
  pathLength: number;
  efficiencyPct: string;
  overheadPct: string;
  uniqueCellsVisited: number;
  revisitedCells: number;
  totalSteps: number;
  fitnessValue: number;
} {
  const startPosition = MazeUtils.findPosition(maze, 'S');
  const exitPosition = MazeUtils.findPosition(maze, 'E');
  const bfsLength = MazeUtils.bfsDistance(
    MazeUtils.encodeMaze(maze),
    startPosition,
    exitPosition,
  );
  const optimalLength = typeof bfsLength === 'number' ? bfsLength : 0;
  const pathLength = Math.max(0, result.path.length - 1);

  let efficiencyPct = '0.0';
  let overheadPct = '0.0';
  if (pathLength > 0 && optimalLength > 0) {
    const efficiency = Math.min(1, optimalLength / pathLength) * 100;
    efficiencyPct = efficiency.toFixed(1);
    overheadPct = ((pathLength / optimalLength) * 100 - 100).toFixed(1);
  }

  const uniqueCells = new Set<string>();
  let revisitedCells = 0;
  for (const [cellX, cellY] of result.path) {
    const cellKey = `${cellX},${cellY}`;
    if (uniqueCells.has(cellKey)) {
      revisitedCells++;
      continue;
    }
    uniqueCells.add(cellKey);
  }

  return {
    optimalLength,
    pathLength,
    efficiencyPct,
    overheadPct,
    uniqueCellsVisited: uniqueCells.size,
    revisitedCells,
    totalSteps: result.steps,
    fitnessValue: result.fitness,
  };
}

/**
 * Infer a compact architecture string from a network-like runtime object.
 *
 * @param networkInstance - Network instance from the maze example runtime.
 * @returns Architecture string such as `6 - 8 - 4`, or `n/a` when unavailable.
 */
export function deriveDashboardArchitecture(
  networkInstance?: INetwork | null,
): string {
  if (!networkInstance) return 'n/a';

  const networkRecord = networkInstance as unknown as Record<string, unknown>;
  const layerArray = networkRecord.layers;
  if (Array.isArray(layerArray) && layerArray.length >= 2) {
    const layerSizes = layerArray.map((layerValue) => {
      const layerRecord = layerValue as Record<string, unknown>;
      if (Array.isArray(layerRecord.nodes)) return layerRecord.nodes.length;
      if (Array.isArray(layerValue)) return layerValue.length;
      return 0;
    });
    return layerSizes.join(' - ');
  }

  const flatNodes = networkRecord.nodes;
  if (Array.isArray(flatNodes)) {
    type NodeWithType = {
      type?: string;
      connections?: { in?: Array<{ from?: unknown }> };
      [key: string]: unknown;
    };

    const inputNodes = flatNodes.filter(
      (nodeValue) => (nodeValue as NodeWithType).type === 'input',
    );
    const outputNodes = flatNodes.filter(
      (nodeValue) => (nodeValue as NodeWithType).type === 'output',
    );
    const hiddenNodes = flatNodes.filter(
      (nodeValue) => (nodeValue as NodeWithType).type === 'hidden',
    );

    if (!hiddenNodes.length) {
      if (
        typeof networkInstance.input === 'number' &&
        typeof networkInstance.output === 'number'
      ) {
        return `${networkInstance.input} - ${networkInstance.output}`;
      }
      return `${inputNodes.length} - ${outputNodes.length}`;
    }

    const assignedNodes = new Set<unknown>(inputNodes);
    let remainingHidden = hiddenNodes.slice();
    const inferredHiddenSizes: number[] = [];
    const safetyLimit = hiddenNodes.length * C.LAYER_INFER_LOOP_MULTIPLIER;
    let iterationCount = 0;

    while (remainingHidden.length && iterationCount < safetyLimit) {
      iterationCount++;
      const currentLayer = remainingHidden.filter((hiddenNode) => {
        const hiddenNodeRecord = hiddenNode as NodeWithType;
        return hiddenNodeRecord.connections?.in?.every((connectionValue) =>
          assignedNodes.has(connectionValue.from),
        );
      });

      if (!currentLayer.length) {
        inferredHiddenSizes.push(remainingHidden.length);
        break;
      }

      inferredHiddenSizes.push(currentLayer.length);
      for (const currentNode of currentLayer) {
        assignedNodes.add(currentNode);
      }
      remainingHidden = remainingHidden.filter(
        (hiddenNode) => !assignedNodes.has(hiddenNode),
      );
    }

    return [
      `${inputNodes.length}`,
      ...inferredHiddenSizes.map((hiddenSize) => `${hiddenSize}`),
      `${outputNodes.length}`,
    ].join(' - ');
  }

  if (
    typeof networkInstance.input === 'number' &&
    typeof networkInstance.output === 'number'
  ) {
    return `${networkInstance.input} - ${networkInstance.output}`;
  }

  return 'n/a';
}

/**
 * Return the recent export window of a bounded numeric history buffer.
 *
 * @param history - History buffer in chronological order.
 * @returns Independent tail slice suitable for telemetry export.
 */
export function sliceDashboardHistoryForExport(
  history: number[] | undefined | null,
): number[] {
  if (!Array.isArray(history) || !history.length) return [];
  const startIndex = Math.max(0, history.length - C.HISTORY_EXPORT_WINDOW);
  return history.slice(startIndex);
}
