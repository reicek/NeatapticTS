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

  it('prefers an explicit worker URL override for embedded browser hosts', () => {
    document.body.innerHTML =
      '<script src="https://example.com/docs/assets/ascii-maze.bundle.js?v=20260511-1"></script>';

    expect(
      resolveAsciiMazeEvaluationWorkerBundleUrl(
        'https://cdn.example.com/workers/ascii-maze.worker.js?v=abc123',
      ),
    ).toBe('https://cdn.example.com/workers/ascii-maze.worker.js?v=abc123');
  });
});
