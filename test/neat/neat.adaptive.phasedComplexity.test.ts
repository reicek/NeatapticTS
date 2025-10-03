import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/**
 * Phased complexity should alternate _phase between 'complexify' and 'simplify'
 * after each phaseLength window. We set phaseLength=1 for a rapid toggle.
 */

describe('Phased Complexity Controller', () => {
  it('toggles phase after configured phaseLength generations', async () => {
  const fitness = (network: Network) => network.connections.length;
  const neat = new Neat(2, 1, fitness, {
      popsize: 8,
      phasedComplexity: { enabled: true, phaseLength: 1 },
      mutationRate: 1,
      mutationAmount: 1,
    });
    // First evolve initializes phase to 'complexify'
    await neat.evolve();
  const firstPhase = Reflect.get(neat, '_phase') as string | undefined;
    expect(firstPhase).toBe('complexify');
    // Second evolve should toggle to 'simplify'
    await neat.evolve();
  const secondPhase = Reflect.get(neat, '_phase') as string | undefined;
    expect(secondPhase).toBe('simplify');
  });
});
