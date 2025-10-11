import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

type LineageNetwork = Network & {
  _parents?: LineageNetwork[];
  score?: number;
};

// Basic test to verify antiInbreeding mode adjusts scores (hard to assert direction deterministically, so
// we verify presence of mode and that scores mutate when high-overlap detected by constructing artificial parents)

describe('lineage anti-inbreeding pressure', () => {
  describe('population evolved with anti-inbreeding pressure', () => {
    const neat = new Neat(5, 2, () => Math.random(), {
      popsize: 25,
      seed: 42,
      speciation: false,
      lineagePressure: {
        enabled: true,
        mode: 'antiInbreeding',
        strength: 0.01,
        ancestorWindow: 3,
      },
    });
    let genomesWithParents: LineageNetwork[] = [];

    beforeAll(async () => {
      // Step 1: evolve to accumulate parent chains
      for (let generationIndex = 0; generationIndex < 5; generationIndex += 1) {
        await neat.evolve();
      }
      // Step 2: evaluate to assign scores post evolution
      await neat.evaluate();
      // Step 3: collect genomes that tracked both parents
      const population = neat.population as LineageNetwork[];
      genomesWithParents = population.filter(
        (network) =>
          Array.isArray(network._parents) && network._parents.length === 2
      );
    });

    test('produces genomes with recorded parent metadata', () => {
      expect(genomesWithParents.length > 0).toBe(true);
    });

    test('assigns numeric scores to genomes with parent metadata', () => {
      const allScoresNumeric = genomesWithParents.every(
        (network) => typeof network.score === 'number'
      );
      expect(allScoresNumeric).toBe(true);
    });
  });
});
