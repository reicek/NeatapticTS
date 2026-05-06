/** @jest-environment jsdom */

import {
  animateSolvedMazeInBrowserHost,
  SOLVED_MAZE_FRAME_DELAY_MS,
} from './browser-entry.solved-maze-animation.services';

describe('animateSolvedMazeInBrowserHost', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    document.body.innerHTML = '<div id="live-output"></div>';
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('returns immediately when no live host or solved path is available', async () => {
    const liveElement = document.getElementById('live-output');

    const results = await Promise.all([
      animateSolvedMazeInBrowserHost({
        liveElement: null,
        maze: ['S.E'],
        path: [
          [0, 0],
          [1, 0],
        ],
      }),
      animateSolvedMazeInBrowserHost({
        liveElement,
        maze: ['S.E'],
        path: [],
      }),
    ]);

    expect({
      liveText: liveElement?.textContent ?? '',
      results,
    }).toEqual({
      liveText: '',
      results: [undefined, undefined],
    });
  });

  it('animates one additional solved-path step every 41 ms and waits one final tick before resolving', async () => {
    const liveElement = document.getElementById('live-output');
    if (!liveElement) {
      throw new Error('Expected a live output host element');
    }

    let settled = false;
    const animationPromise = animateSolvedMazeInBrowserHost({
      liveElement,
      maze: ['S..E'],
      path: [
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
      ],
    }).then(() => {
      settled = true;
    });

    const initialFrame = normalizeFrameText(liveElement.textContent ?? '');

    jest.advanceTimersByTime(SOLVED_MAZE_FRAME_DELAY_MS);
    await Promise.resolve();
    const secondFrame = normalizeFrameText(liveElement.textContent ?? '');

    jest.advanceTimersByTime(SOLVED_MAZE_FRAME_DELAY_MS);
    await Promise.resolve();
    const thirdFrame = normalizeFrameText(liveElement.textContent ?? '');

    jest.advanceTimersByTime(SOLVED_MAZE_FRAME_DELAY_MS);
    await Promise.resolve();
    const solvedFrame = normalizeFrameText(liveElement.textContent ?? '');
    const settledBeforeFinalDelay = settled;

    jest.advanceTimersByTime(SOLVED_MAZE_FRAME_DELAY_MS);
    await animationPromise;

    expect({
      initialFrame,
      secondFrame,
      settled,
      settledBeforeFinalDelay,
      solvedFrame,
      thirdFrame,
    }).toEqual({
      initialFrame: 'S..E',
      secondFrame: 'SA.E',
      settled: true,
      settledBeforeFinalDelay: false,
      solvedFrame: 'S••E',
      thirdFrame: 'S•AE',
    });
  });
});

function normalizeFrameText(value: string): string {
  return value.replace(/\s+/g, '');
}
