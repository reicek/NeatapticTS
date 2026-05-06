/**
 * Browser-hosted solved-maze reveal helpers.
 *
 * The evolution engine knows when a maze is solved, but it should not own how
 * the browser celebrates or explains that moment. This small boundary turns a
 * solved path into a short teaching animation in the live maze panel: show the
 * start position first, then reveal one additional route step at a fixed host
 * cadence until the path reaches the exit, then hold for one final tick before
 * the curriculum advances.
 *
 * That choice keeps the browser demo readable. A fully solved maze is useful as
 * a record, but a progressive reveal is better at teaching which corridor the
 * policy actually discovered.
 */
import { createBrowserLogger } from '../browserLogger';
import { BrowserTerminalUtility } from '../browserTerminalUtility';
import { MazeVisualization } from '../mazeVisualization';

/**
 * Delay between solved-path animation frames in the browser host.
 *
 * The same delay is also used for the final hold after the exit is reached, so
 * the viewer gets one extra beat before the next maze replaces the solved one.
 */
export const SOLVED_MAZE_FRAME_DELAY_MS = 2;

type MazePathStep = [number, number];

/**
 * Animate the solved path inside the browser live-output host.
 *
 * The reveal sequence is deliberately simple:
 *
 * 1. render the maze with only the start cell active,
 * 2. wait one host tick,
 * 3. add one more solved-path step,
 * 4. repeat until the exit is reached,
 * 5. wait one final host tick before resolving.
 *
 * @example
 * ```ts
 * await animateSolvedMazeInBrowserHost({
 *   liveElement,
 *   maze: ['S..E'],
 *   path: [
 *     [0, 0],
 *     [1, 0],
 *     [2, 0],
 *     [3, 0],
 *   ],
 * });
 * ```
 *
 * @param input - Live host element, maze layout, and solved path to reveal.
 * @returns Promise that resolves after the full path and final pause complete.
 */
export const animateSolvedMazeInBrowserHost = async (input: {
  liveElement: HTMLElement | null;
  maze: string[];
  path?: readonly MazePathStep[];
}): Promise<void> => {
  if (!input.liveElement || !input.path || input.path.length === 0) {
    return;
  }

  const clearLiveOutput = BrowserTerminalUtility.createTerminalClearer(
    input.liveElement,
  );
  const writeLiveLine = createBrowserLogger(input.liveElement);

  renderSolvedMazeFrame(
    input.maze,
    input.path.slice(0, 1),
    clearLiveOutput,
    writeLiveLine,
  );

  for (let stepIndex = 1; stepIndex < input.path.length; stepIndex++) {
    await waitForSolvedMazeAnimationDelay();
    renderSolvedMazeFrame(
      input.maze,
      input.path.slice(0, stepIndex + 1),
      clearLiveOutput,
      writeLiveLine,
    );
  }

  await waitForSolvedMazeAnimationDelay();
};

function renderSolvedMazeFrame(
  maze: string[],
  pathPrefix: readonly MazePathStep[],
  clearLiveOutput: () => void,
  writeLiveLine: (...args: unknown[]) => void,
): void {
  const currentPosition = pathPrefix.at(-1)!;

  clearLiveOutput();

  const mazeLines = MazeVisualization.visualizeMaze(
    maze,
    currentPosition,
    pathPrefix,
  ).split('\n');

  mazeLines.forEach((mazeLine) => {
    writeLiveLine(mazeLine);
  });
}

function waitForSolvedMazeAnimationDelay(): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, SOLVED_MAZE_FRAME_DELAY_MS);
  });
}
