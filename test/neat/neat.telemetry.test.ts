import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('telemetry snapshot', () => {
  const fitness = (net: Network) => (net as any).connections.length;
  test('telemetry records hyper proxy', async () => {
    const neat = new Neat(3, 2, fitness, {
      popsize: 30,
      multiObjective: { enabled: true, complexityMetric: 'connections' },
      telemetry: { enabled: true },
    });
    for (let i = 0; i < 3; i++) await neat.evolve();
    const log = neat.getTelemetry();
    expect(log.length).toBeGreaterThan(0);
  });
});
