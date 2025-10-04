import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

describe('Mutation selection robustness with undefined entries', () => {
  test('selectMutationMethod handles undefined without throwing', async () => {
    const fitness = (n: Network) => n.nodes.length;
    const mutationPool: Array<
      (typeof mutation)[keyof typeof mutation] | undefined
    > = [mutation.ADD_NODE, undefined, mutation.SUB_CONN];
    const neat = new Neat(3, 1, fitness, {
      popsize: 6,
      seed: 999,
      phasedComplexity: { enabled: true, phaseLength: 1 },
      mutation: mutationPool,
    });
    await neat.evaluate();
    // Force phase for both branches
    const neatWithInternals = neat as unknown as {
      selectMutationMethod: (genome: Network, forceExploit: boolean) => void;
    };
    for (let generationIndex = 0; generationIndex < 3; generationIndex += 1) {
      await neat.evolve();
      const genome = neat.population[0];
      // Call internal method indirectly by triggering mutate on a clone
      neatWithInternals.selectMutationMethod(genome, false);
    }
    expect(true).toBe(true); // No throw implies pass
  });
});
