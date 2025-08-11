import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Species allocation telemetry', () => {
  const fitness = (net: Network) => net.nodes.length;
  it('emits speciesAlloc and objAges columns in CSV', async () => {
    const neat = new Neat(3, 2, fitness, {
      popsize: 25,
      seed: 9,
      telemetry: { enabled: true },
      multiObjective: { enabled: true },
      speciation: true,
    });
    for (let g = 0; g < 5; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    const telem = neat.getTelemetry();
    const last = telem[telem.length - 1];
    expect(Array.isArray(last.speciesAlloc)).toBe(true);
    const csv = neat.exportTelemetryCSV();
    const header = csv.split(/\r?\n/)[0];
    expect(header).toMatch(/speciesAlloc/);
    expect(header).toMatch(/objAges/);
  });
});
