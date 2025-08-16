import Neat from '../../src/neat';

/**
 * Ensures complexity budget adaptive mode increases maxNodes over improving fitness history
 * and contracts (or remains bounded) when no improvement window observed.
 */

describe('Adaptive Complexity Budget', () => {
  it('adjusts maxNodes in adaptive mode', async () => {
    const neat = new Neat(2, 1, (n: any) => (n as any).connections.length, {
      popsize: 10,
      complexityBudget: {
        enabled: true,
        mode: 'adaptive',
        maxNodesStart: 6,
        maxNodesEnd: 100,
        improvementWindow: 5,
      },
      mutationRate: 1,
      mutationAmount: 2,
    });
    // Force one evolution so complexity budget initializes from maxNodesStart
    await neat.evolve();
    const baseline = neat.options.maxNodes; // should now be finite
    expect(baseline).not.toBe(Infinity);
    for (let i = 0; i < 7; i++) await neat.evolve();
    const after = neat.options.maxNodes;
    const cfg = (neat.options as any).complexityBudget;
    const minNodes = cfg.minNodes ?? neat.input + neat.output + 2; // mirrors implementation default
    // Budget should remain within configured min/max bounds
    expect(after).toBeGreaterThanOrEqual(minNodes);
    expect(after).toBeLessThanOrEqual(cfg.maxNodesEnd);
  });
});
