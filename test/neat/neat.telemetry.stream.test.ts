import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Telemetry stream callback', () => {
  test('invokes callback with entries', async () => {
    const entries: any[] = [];
    const neat = new Neat(3, 1, (n: Network) => n.connections.length, {
      popsize: 10,
      seed: 91,
      speciation: false,
      telemetry: { enabled: true, logEvery: 1 },
      telemetryStream: { enabled: true, onEntry: (e: any) => entries.push(e) },
    });
    await neat.evaluate();
    await neat.evolve();
    expect(entries.length).toBeGreaterThan(0);
    expect(entries[0].gen).toBeDefined();
  });
});
