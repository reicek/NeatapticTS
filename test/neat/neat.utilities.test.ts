import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('utility exports & resets', () => {
  test('exportSpeciesHistoryJSONL returns lines', async () => {
    const neat = new Neat(3, 1, (n: Network) => (n as any).connections.length, {
      popsize: 12,
      seed: 600,
      speciation: true,
      speciesAllocation: { extendedHistory: true },
    });
    await neat.evaluate();
    await neat.evolve();
    const jsonl = neat.exportSpeciesHistoryJSONL();
    expect(jsonl.split('\n').length).toBeGreaterThan(0);
  });
  test('resetNoveltyArchive clears archive', async () => {
    const neat = new Neat(3, 1, (n: Network) => (n as any).connections.length, {
      popsize: 14,
      seed: 601,
      novelty: {
        enabled: true,
        descriptor: (g: Network) => [g.nodes.length],
        archiveAddThreshold: 0.0,
      },
    });
    await neat.evaluate();
    const before = neat.getNoveltyArchiveSize();
    neat.resetNoveltyArchive();
    expect(neat.getNoveltyArchiveSize()).toBe(0);
    expect(before).toBeGreaterThanOrEqual(0);
  });
  test('clearParetoArchive empties archive', async () => {
    const neat = new Neat(3, 2, (n: Network) => (n as any).connections.length, {
      popsize: 16,
      seed: 602,
      multiObjective: { enabled: true },
    });
    await neat.evolve();
    expect(neat.getParetoArchive().length).toBeGreaterThan(0);
    neat.clearParetoArchive();
    expect(neat.getParetoArchive().length).toBe(0);
  });
});
