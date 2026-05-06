import { resolvePeriodicDashboardSnapshot } from './evolutionLoop';
import type { IMazeRunResult } from '../interfaces';

describe('resolvePeriodicDashboardSnapshot', () => {
  it('prefers the current generation champion when both current result and network exist', () => {
    const bestResult = { progress: 40 } as IMazeRunResult;
    const currentResult = { progress: 35 } as IMazeRunResult;
    const bestNetwork = { label: 'best' } as never;
    const currentNetwork = { label: 'current' } as never;

    expect(
      resolvePeriodicDashboardSnapshot(
        bestResult,
        bestNetwork,
        currentResult,
        currentNetwork,
      ),
    ).toEqual({
      result: currentResult,
      network: currentNetwork,
    });
  });

  it('falls back to the global best snapshot when the current generation data is incomplete', () => {
    const bestResult = { progress: 40 } as IMazeRunResult;
    const bestNetwork = { label: 'best' } as never;

    expect(
      resolvePeriodicDashboardSnapshot(
        bestResult,
        bestNetwork,
        undefined,
        null,
      ),
    ).toEqual({
      result: bestResult,
      network: bestNetwork,
    });
  });
});
