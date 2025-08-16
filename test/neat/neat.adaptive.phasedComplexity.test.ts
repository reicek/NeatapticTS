import Neat from '../../src/neat';

/**
 * Phased complexity should alternate _phase between 'complexify' and 'simplify'
 * after each phaseLength window. We set phaseLength=1 for a rapid toggle.
 */

describe('Phased Complexity Controller', () => {
  it('toggles phase after configured phaseLength generations', async () => {
    const neat = new Neat(2, 1, (n: any) => (n as any).connections.length, {
      popsize: 8,
      phasedComplexity: { enabled: true, phaseLength: 1 },
      mutationRate: 1,
      mutationAmount: 1,
    });
    // First evolve initializes phase to 'complexify'
    await neat.evolve();
    const firstPhase = (neat as any)._phase;
    expect(firstPhase).toBe('complexify');
    // Second evolve should toggle to 'simplify'
    await neat.evolve();
    const secondPhase = (neat as any)._phase;
    expect(secondPhase).toBe('simplify');
  });
});
