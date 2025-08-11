import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Adaptive target species & distance coeff tuning', () => {
  const fit = (n: Network) => n.nodes.length;
  it('adjusts targetSpecies over generations based on entropy', async () => {
    const neat = new Neat(4, 2, fit, {
      popsize: 25,
      seed: 101,
      speciation: true,
      targetSpecies: 6,
      adaptiveTargetSpecies: {
        enabled: true,
        entropyRange: [0, 2],
        speciesRange: [4, 10],
        smooth: 0.5,
      },
      telemetry: { enabled: true },
    });
    for (let g = 0; g < 5; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    const ts = neat.options.targetSpecies as number;
    expect(ts).toBeGreaterThanOrEqual(4);
  });
  it('autoDistanceCoeffTuning nudges coefficients', async () => {
    const neat = new Neat(3, 1, fit, {
      popsize: 20,
      seed: 55,
      speciation: true,
      autoDistanceCoeffTuning: { enabled: true, adjustRate: 0.05 },
    });
    const startEx = neat.options.excessCoeff!;
    const startDis = neat.options.disjointCoeff!;
    for (let g = 0; g < 4; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    expect(neat.options.excessCoeff).not.toBe(startEx);
    expect(neat.options.disjointCoeff).not.toBe(startDis);
  });
});
