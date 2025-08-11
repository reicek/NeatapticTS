import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Species history CSV export', () => {
  const fit = (n: Network) => n.nodes.length;
  it('exports non-empty CSV with dynamic headers', async () => {
    const neat = new Neat(3, 2, fit, {
      popsize: 30,
      seed: 5,
      speciation: true,
      speciesAllocation: { extendedHistory: true, minOffspring: 1 },
    });
    for (let g = 0; g < 6; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    const csv = neat.exportSpeciesHistoryCSV();
    expect(csv.length).toBeGreaterThan(0);
    const header = csv.split(/\r?\n/)[0];
    expect(header).toMatch(/generation/);
    expect(header).toMatch(/id/);
    expect(header).toMatch(/size/);
  });
});
