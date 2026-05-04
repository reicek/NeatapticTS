import { evaluateFlappyFitnessAcrossSeeds } from './evaluation.fitness.utils';

jest.mock('./evaluation.rollout.service', () => ({
  rolloutEpisode: jest.fn(() => ({
    fitness: 1,
    pipesPassed: 0,
    framesSurvived: 1,
    done: true,
    fitnessBreakdown: {
      survival: 1,
      pipeProgress: 0,
      denseShaping: 0,
      terminalShaping: 0,
    },
  })),
}));

describe('evaluateFlappyFitnessAcrossSeeds', () => {
  it('clears carried network state before each seeded rollout when the network exposes a reset hook', () => {
    const clear = jest.fn();
    const clearableNetwork = {
      activate: () => [0, 1],
      clear,
    };

    evaluateFlappyFitnessAcrossSeeds(clearableNetwork, [11, 22, 33]);

    expect(clear.mock.calls).toHaveLength(3);
  });
});
