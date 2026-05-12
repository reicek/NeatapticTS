/** @jest-environment jsdom */

import { resolveAsciiMazeEvaluationWorkerBundleUrl } from './browser-entry.worker-url.service';

describe('resolveAsciiMazeEvaluationWorkerBundleUrl', () => {
  it('resolves the ASCII Maze evaluation worker next to the browser bundle asset', () => {
    document.body.innerHTML =
      '<script src="https://example.com/docs/assets/ascii-maze.bundle.js?v=20260511-1"></script>';

    expect(resolveAsciiMazeEvaluationWorkerBundleUrl()).toBe(
      'https://example.com/docs/assets/ascii-maze-evaluation.worker.bundle.js?v=20260511-1',
    );
  });
});
