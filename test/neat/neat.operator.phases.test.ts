import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

// Single expectation per test.

describe('Phased complexity switching', () => {
  test('phase toggles after phaseLength generations', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 6,
      seed: 400,
      speciation: false,
      phasedComplexity: { enabled: true, phaseLength: 2 },
    });
    await neat.evaluate();
    await neat.evolve(); // gen1
    const neatWithPhase = (neat as unknown) as { _phase?: number };
    const initialPhase = neatWithPhase._phase;
    await neat.evolve(); // gen2 toggles
    await neat.evolve(); // gen3 (second phase)
    const toggledPhase = neatWithPhase._phase;
    expect(initialPhase).not.toBe(toggledPhase);
  });
});

describe('Operator adaptation tracking', () => {
  test('records operator stats', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 6,
      seed: 410,
      speciation: false,
      mutation: [
        mutation.MOD_WEIGHT,
        mutation.MOD_BIAS,
        mutation.ADD_NODE,
        mutation.SUB_NODE,
      ],
      mutationRate: 1, // Ensure mutations always happen
      mutationAmount: 1,
      operatorAdaptation: { enabled: true, boost: 2 },
    });
    await neat.evaluate();
    await neat.evolve();
    const neatWithStats = (neat as unknown) as {
      _operatorStats: Map<string, unknown>;
    };
    const hasAnyOperatorStat = neatWithStats._operatorStats.size > 0;
    expect(hasAnyOperatorStat).toBe(true);
  });
});
