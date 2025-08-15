import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('diversityPressure & autoCompatTuning', () => {
  const fitness = (net: Network) => {
    // Simple fitness proportional to hidden node count to create motif similarity pressure
    const hidden = (net as any).nodes.filter((n: any) => n.type === 'H').length;
    return hidden;
  };
  test('diversity pressure adjusts scores (no crash)', async () => {
    const neat = new Neat(2, 1, fitness, {
      popsize: 30,
      speciation: true,
      diversityPressure: {
        enabled: true,
        motifSample: 10,
        penaltyStrength: 0.05,
      },
      targetSpecies: 5,
      autoCompatTuning: { enabled: false },
    });
    await neat.evolve();
    // Evaluate the newly created generation so all genomes have scores
    await neat.evaluate();
    const pop = neat.population;
    expect(pop.every((g) => typeof g.score === 'number')).toBe(true);
  });
  test('auto compatibility tuning nudges coefficients', async () => {
    const neat = new Neat(2, 1, fitness, {
      popsize: 40,
      speciation: true,
      targetSpecies: 6,
      autoCompatTuning: {
        enabled: true,
        adjustRate: 0.05,
        minCoeff: 0.2,
        maxCoeff: 3,
      },
    });
    const startExcess = neat.options.excessCoeff!;
    // run a few generations to allow adjustment
    for (let i = 0; i < 5; i++) await neat.evolve();
    const endExcess = neat.options.excessCoeff!;
    // Coefficient should have moved (unless perfectly matched already which is unlikely)
    expect(endExcess).not.toBe(startExcess);
  });
});
