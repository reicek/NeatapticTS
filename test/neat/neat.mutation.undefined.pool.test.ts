import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

describe('Mutation selection robustness with undefined entries', () => {
  test('selectMutationMethod handles undefined without throwing', async () => {
    const fitness = (n: Network) => n.nodes.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 6,
      seed: 999,
      phasedComplexity: { enabled: true, phaseLength: 1 },
      mutation: [mutation.ADD_NODE, undefined as any, mutation.SUB_CONN] as any,
    });
    await neat.evaluate();
    // Force phase for both branches
    for (let i = 0; i < 3; i++) {
      await neat.evolve();
      const g = neat.population[0];
      // Call internal method indirectly by triggering mutate on a clone
      (neat as any).selectMutationMethod(g, false);
    }
    expect(true).toBe(true); // No throw implies pass
  });
});
