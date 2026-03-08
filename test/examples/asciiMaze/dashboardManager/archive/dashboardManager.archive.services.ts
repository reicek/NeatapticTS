import { colors } from '../../colors';
import type { IMazeRunResult, INetwork } from '../../interfaces';
import { MazeVisualization } from '../../mazeVisualization';
import { NetworkVisualization } from '../../networkVisualization';
import { DASHBOARD_MANAGER_CONSTANTS as C } from '../dashboardManager.constants';
import type {
  DashboardManagerContext,
  SolvedMazeRecord,
} from '../dashboardManager.types';
import {
  buildDashboardSparkline,
  computeDashboardPathMetrics,
  deriveDashboardArchitecture,
  getDashboardMazeKey,
} from '../dashboardManager.utils';

let cachedSolvedFooterBorder: string | null = null;

/**
 * Record and emit a newly solved maze archive block when the layout has not been seen before.
 *
 * @param context - Dashboard runtime context containing archive state and callbacks.
 * @param maze - Solved maze layout.
 * @param result - Successful run result used for archive stats.
 * @param network - Network that solved the maze.
 * @param generation - Generation number at solve time.
 */
export function recordSolvedMaze(
  context: DashboardManagerContext,
  maze: string[],
  result: IMazeRunResult,
  network: INetwork,
  generation: number,
): void {
  const { state } = context;
  const solvedMazeKey = getDashboardMazeKey(maze);
  if (state.solvedMazeKeys.has(solvedMazeKey)) return;

  const solvedRecord: SolvedMazeRecord = { maze, result, network, generation };
  state.solvedMazes.push(solvedRecord);
  state.solvedMazeKeys.add(solvedMazeKey);
  appendSolvedToArchive(context, solvedRecord, state.solvedMazes.length);
}

function appendSolvedToArchive(
  context: DashboardManagerContext,
  solved: SolvedMazeRecord,
  displayNumber: number,
): void {
  if (!context.archiveFn) return;

  const blockLines: string[] = [];
  appendSolvedHeader(blockLines, solved, displayNumber);
  appendSolvedSparklines(context, blockLines, solved.network);
  appendSolvedMaze(blockLines, solved);
  appendSolvedPathStats(context, blockLines, solved);
  appendSolvedFooterAndEmit(context.archiveFn, blockLines);
}

function appendSolvedHeader(
  blockLines: string[],
  solved: SolvedMazeRecord,
  displayNumber: number,
): void {
  const rawFitness = solved.result?.fitness;
  const formattedFitness =
    typeof rawFitness === 'number' && Number.isFinite(rawFitness)
      ? rawFitness.toFixed(2)
      : 'n/a';
  const title = ` SOLVED #${Math.max(1, displayNumber)} (GEN ${solved.generation})  FITNESS ${formattedFitness} `;
  const leftPaddingSize = Math.max(
    0,
    Math.floor((C.FRAME_INNER_WIDTH - title.length) / 2),
  );
  const rightPaddingSize = Math.max(
    0,
    C.FRAME_INNER_WIDTH - title.length - leftPaddingSize,
  );

  blockLines.push(
    `${colors.blueCore}╔${NetworkVisualization.pad(
      '═'.repeat(C.FRAME_INNER_WIDTH),
      C.FRAME_INNER_WIDTH,
      '═',
    )}╗${colors.reset}`,
  );
  blockLines.push(
    `${colors.blueCore}║${' '.repeat(leftPaddingSize)}${colors.orangeNeon}${title}${colors.blueCore}${' '.repeat(rightPaddingSize)}║${colors.reset}`,
  );
  blockLines.push(
    `${colors.blueCore}║${NetworkVisualization.pad(' ', C.FRAME_INNER_WIDTH, ' ')}║${colors.reset}`,
  );
}

function appendSolvedSparklines(
  context: DashboardManagerContext,
  blockLines: string[],
  network?: INetwork | null,
): void {
  const pushIfPresent = (label: string, value: string | null | undefined) => {
    if (!value) return;
    blockLines.push(
      context.formatStat(
        label,
        value,
        colors.neonSilver,
        colors.cyanNeon,
        C.SOLVED_LABEL_WIDTH,
      ),
    );
  };

  const architecture = deriveDashboardArchitecture(network);
  if (architecture !== 'n/a') {
    pushIfPresent(C.LABEL_ARCH, architecture.split(/\s*-\s*/).join(' <=> '));
  }

  const { histories } = context.state;
  pushIfPresent(
    'Fitness trend',
    buildDashboardSparkline(histories.bestFitness, C.ARCHIVE_SPARK_WIDTH),
  );
  pushIfPresent(
    'Nodes trend',
    buildDashboardSparkline(histories.complexityNodes, C.ARCHIVE_SPARK_WIDTH),
  );
  pushIfPresent(
    'Conns trend',
    buildDashboardSparkline(histories.complexityConns, C.ARCHIVE_SPARK_WIDTH),
  );
  pushIfPresent(
    'Hypervol trend',
    buildDashboardSparkline(histories.hypervolume, C.ARCHIVE_SPARK_WIDTH),
  );
  pushIfPresent(
    'Progress trend',
    buildDashboardSparkline(histories.progress, C.ARCHIVE_SPARK_WIDTH),
  );
  pushIfPresent(
    'Species trend',
    buildDashboardSparkline(histories.speciesCount, C.ARCHIVE_SPARK_WIDTH),
  );

  blockLines.push(
    `${colors.blueCore}║${NetworkVisualization.pad(' ', C.FRAME_INNER_WIDTH, ' ')}${colors.blueCore}║${colors.reset}`,
  );
}

function appendSolvedMaze(
  blockLines: string[],
  solved: Pick<SolvedMazeRecord, 'maze' | 'result'>,
): void {
  const pathCoordinates = solved.result.path;
  const endPosition = pathCoordinates?.at(-1) ?? [0, 0];
  const visualization = MazeVisualization.visualizeMaze(
    solved.maze,
    endPosition as [number, number],
    pathCoordinates,
  );
  const rawLines = Array.isArray(visualization)
    ? visualization
    : visualization.split('\n');

  for (const rawLine of rawLines) {
    const paddedRow = NetworkVisualization.pad(
      rawLine,
      C.FRAME_INNER_WIDTH,
      ' ',
    );
    blockLines.push(
      `${colors.blueCore}║${NetworkVisualization.pad(
        paddedRow,
        C.FRAME_INNER_WIDTH,
        ' ',
      )}${colors.blueCore}║${colors.reset}`,
    );
  }
}

function appendSolvedPathStats(
  context: DashboardManagerContext,
  blockLines: string[],
  solved: Pick<SolvedMazeRecord, 'maze' | 'result'>,
): void {
  const metrics = computeDashboardPathMetrics(solved.maze, solved.result);
  blockLines.push(
    context.formatStat(
      C.LABEL_PATH_EFF,
      `${metrics.optimalLength}/${metrics.pathLength} (${metrics.efficiencyPct}%)`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  blockLines.push(
    context.formatStat(
      C.LABEL_PATH_OVER,
      `${metrics.overheadPct}% longer than optimal`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  blockLines.push(
    context.formatStat(
      C.LABEL_UNIQUE,
      `${metrics.uniqueCellsVisited}`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  blockLines.push(
    context.formatStat(
      C.LABEL_REVISITS,
      `${metrics.revisitedCells} times`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  blockLines.push(
    context.formatStat(
      C.LABEL_STEPS,
      `${metrics.totalSteps}`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
  blockLines.push(
    context.formatStat(
      C.LABEL_FITNESS,
      `${metrics.fitnessValue.toFixed(2)}`,
      colors.neonSilver,
      colors.cyanNeon,
      C.SOLVED_LABEL_WIDTH,
    ),
  );
}

function appendSolvedFooterAndEmit(
  archiveFn: (...args: unknown[]) => void,
  blockLines: string[],
): void {
  if (cachedSolvedFooterBorder === null) {
    cachedSolvedFooterBorder = `${colors.blueCore}╚${NetworkVisualization.pad(
      '═'.repeat(C.FRAME_INNER_WIDTH),
      C.FRAME_INNER_WIDTH,
      '═',
    )}╝${colors.reset}`;
  }

  blockLines.push(cachedSolvedFooterBorder);
  try {
    const archiveEmitter = archiveFn as (
      payload: string,
      options?: { prepend?: boolean },
    ) => void;
    archiveEmitter(blockLines.join('\n'), { prepend: true });
  } catch {
    for (const lineValue of blockLines) {
      archiveFn(lineValue);
    }
  }
  blockLines.length = 0;
}
