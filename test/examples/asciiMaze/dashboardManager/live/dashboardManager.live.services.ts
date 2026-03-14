/**
 * Live redraw boundary for the ASCII Maze dashboard.
 *
 * This file is the runtime-facing chapter opening for the dashboard's live
 * surface. It owns the part of the dashboard that needs to feel immediate to a
 * human watching evolution unfold: the frame, the current champion snapshot,
 * and the refreshed detailed stats that make each redraw more than a raw text
 * repaint.
 *
 * The live boundary exists so rendering cadence can stay separate from archive
 * persistence and telemetry export. That split matters because redraw work is
 * frequent and user-facing, while archive and telemetry helpers answer slower,
 * more analytical questions.
 *
 * A useful reading order is:
 *
 * 1. start here for the visible dashboard refresh path,
 * 2. continue with `dashboardManager.telemetry.services.ts` to see how redraws
 *    pick up richer detail snapshots,
 * 3. finish with `dashboardManager.archive.services.ts` for the long-lived
 *    record of especially important runs.
 */
import { colors } from '../../colors';
import type { IMazeRunResult, INetwork } from '../../interfaces';
import { MazeVisualization } from '../../mazeVisualization';
import { NetworkVisualization } from '../../networkVisualization';
import { DASHBOARD_MANAGER_CONSTANTS as C } from '../dashboardManager.constants';
import type { DashboardManagerContext } from '../dashboardManager.types';
import { createDetailedStatsSnapshot } from '../telemetry/dashboardManager.telemetry.services';

const int32ScratchPool: Int32Array[] = [];

/**
 * Repaint the live dashboard from current state and refresh the detailed snapshot.
 *
 * @param context - Dashboard runtime context for state and output callbacks.
 * @param currentMaze - Maze currently being evolved.
 * @param neat - Optional NEAT runtime instance used for detailed stats.
 */
export function redrawDashboard(
  context: DashboardManagerContext,
  currentMaze: string[],
  neat?: unknown,
): void {
  context.state.lastUpdateTs = globalThis.performance?.now?.() ?? Date.now();
  context.clearFn();
  printTopFrame(context);
  if (context.state.currentBest) {
    printCurrentBestSection(context, currentMaze);
  }
  context.state.lastDetailedStats = createDetailedStatsSnapshot(
    context.state,
    neat,
  );
  context.logBlank();
}

function printTopFrame(context: DashboardManagerContext): void {
  context.logFn(
    `${colors.blueCore}╔${NetworkVisualization.pad(
      C.FRAME_SINGLE_LINE_CHAR,
      C.FRAME_INNER_WIDTH,
      C.FRAME_SINGLE_LINE_CHAR,
    )}╗${colors.reset}`,
  );
  context.logFn(
    `${colors.blueCore}╚${NetworkVisualization.pad(
      C.FRAME_BRIDGE_TOP,
      C.FRAME_INNER_WIDTH,
      C.FRAME_SINGLE_LINE_CHAR,
    )}╝${colors.reset}`,
  );

  const uncoloredTemplate = '║ ASCII maze ║';
  const remainingSpace = C.FRAME_INNER_WIDTH - uncoloredTemplate.length;
  const leftPaddingCount = Math.max(0, Math.ceil(remainingSpace / 2)) + 1;
  const rightPaddingCount = Math.max(0, remainingSpace - leftPaddingCount);
  context.logFn(
    `${colors.blueCore}${' '.repeat(leftPaddingCount)}║ ${colors.neonYellow}ASCII maze${colors.blueCore} ║${' '.repeat(rightPaddingCount)}${colors.reset}`,
  );
  context.logFn(
    `${colors.blueCore}╔${NetworkVisualization.pad(
      C.FRAME_BRIDGE_BOTTOM,
      C.FRAME_INNER_WIDTH,
      C.FRAME_SINGLE_LINE_CHAR,
    )}╗${colors.reset}`,
  );
}

function printCurrentBestSection(
  context: DashboardManagerContext,
  currentMaze: string[],
): void {
  const generation = context.state.currentBest?.generation ?? 0;
  context.logFn(
    `${colors.blueCore}╠${NetworkVisualization.pad(
      C.EVOLVING_SECTION_LINE,
      C.FRAME_INNER_WIDTH,
      '═',
    )}${colors.blueCore}╣${colors.reset}`,
  );
  context.logFn(
    `${colors.blueCore}║${NetworkVisualization.pad(
      `${colors.orangeNeon}EVOLVING (GEN ${generation})`,
      C.FRAME_INNER_WIDTH,
      ' ',
    )}${colors.blueCore}║${colors.reset}`,
  );
  context.logFn(
    `${colors.blueCore}╠${NetworkVisualization.pad(
      C.EVOLVING_SECTION_LINE,
      C.FRAME_INNER_WIDTH,
      '═',
    )}${colors.blueCore}╣${colors.reset}`,
  );
  context.logBlank();
  printNetworkSummary(context);
  printLiveMaze(context, currentMaze);
  printLiveStats(context, currentMaze);
  printProgressBar(context);
}

function printNetworkSummary(context: DashboardManagerContext): void {
  context.logBlank();
  const currentBest = context.state.currentBest;
  if (currentBest?.network) {
    context.logFn(
      NetworkVisualization.visualizeNetworkSummary(currentBest.network),
    );
  } else {
    context.logFn(
      context.formatStat(
        'Architecture',
        'n/a',
        colors.neonSilver,
        colors.cyanNeon,
      ),
    );
  }
  context.logBlank();
}

function printLiveMaze(
  context: DashboardManagerContext,
  currentMaze: string[],
): void {
  const currentBest = context.state.currentBest;
  if (!currentBest) {
    context.logBlank();
    return;
  }

  const endOfPathPosition = currentBest.result.path?.at(-1) ?? [0, 0];
  const visualization = MazeVisualization.visualizeMaze(
    currentMaze,
    endOfPathPosition as readonly [number, number],
    currentBest.result.path,
  );
  const visualizationLines = Array.isArray(visualization)
    ? visualization
    : visualization.split('\n');

  context.logBlank();
  for (const unpaddedRow of visualizationLines) {
    const paddedRow = NetworkVisualization.pad(
      unpaddedRow,
      C.FRAME_INNER_WIDTH,
      ' ',
    );
    context.logFn(
      `${colors.blueCore}║${paddedRow}${colors.blueCore}║${colors.reset}`,
    );
  }
  context.logBlank();
}

function printLiveStats(
  context: DashboardManagerContext,
  currentMaze: string[],
): void {
  context.logBlank();
  const currentBest = context.state.currentBest;
  if (!currentBest) {
    context.logBlank();
    return;
  }

  const scratch = rentInt32(3);
  scratch[0] =
    typeof currentBest.result?.fitness === 'number' &&
    Number.isFinite(currentBest.result.fitness)
      ? Math.round(currentBest.result.fitness * 100)
      : 0;
  scratch[1] = Number.isFinite(Number(currentBest.result?.steps ?? 0))
    ? Number(currentBest.result.steps)
    : 0;
  scratch[2] = Number.isFinite(Number(currentBest.result?.progress ?? 0))
    ? Math.round(Number(currentBest.result.progress) * 100)
    : 0;

  context.logFn(
    context.formatStat(
      'Fitness',
      (scratch[0] / 100).toFixed(2),
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  context.logFn(
    context.formatStat(
      'Steps',
      `${scratch[1]}`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  context.logFn(
    context.formatStat(
      'Progress',
      `${scratch[2]}%`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  if (currentBest.network) {
    MazeVisualization.printMazeStats(
      currentBest as {
        result: IMazeRunResult;
        network: INetwork;
        generation: number;
      },
      currentMaze,
      context.logFn,
    );
  }
  releaseInt32(scratch);
  context.logBlank();
}

function printProgressBar(context: DashboardManagerContext): void {
  const emitFrameBlank = () =>
    context.logFn(
      `${colors.blueCore}║${NetworkVisualization.pad(
        ' ',
        C.FRAME_INNER_WIDTH,
        ' ',
      )}${colors.blueCore}║${colors.reset}`,
    );

  emitFrameBlank();
  const rawProgressValue = context.state.currentBest?.result?.progress ?? 0;
  const safeProgressFraction = Number.isFinite(Number(rawProgressValue))
    ? Number(rawProgressValue)
    : 0;
  const progressLabel = `Progress to exit: ${MazeVisualization.displayProgressBar(
    safeProgressFraction,
  )}`;
  context.logFn(
    `${colors.blueCore}║${NetworkVisualization.pad(
      ` ${colors.neonSilver}${progressLabel}${colors.reset}`,
      C.FRAME_INNER_WIDTH,
      ' ',
    )}${colors.blueCore}║${colors.reset}`,
  );
  emitFrameBlank();
}

function rentInt32(requestedLength: number): Int32Array {
  const pooled = int32ScratchPool.pop();
  if (pooled && pooled.length >= requestedLength) {
    return pooled.subarray(0, requestedLength) as Int32Array;
  }
  return new Int32Array(requestedLength);
}

function releaseInt32(buffer: Int32Array): void {
  if (int32ScratchPool.length < 8) {
    const isFullView =
      buffer.byteOffset === 0 && buffer.byteLength === buffer.buffer.byteLength;
    const pooledBuffer = isFullView ? buffer : new Int32Array(buffer.buffer);
    int32ScratchPool.push(pooledBuffer);
  }
}
