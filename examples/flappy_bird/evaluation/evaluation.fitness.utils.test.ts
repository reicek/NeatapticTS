import {
  evaluateFlappyFitnessAcrossSeeds,
  evaluateFlappyFitnessAcrossSeedsWithInferenceChannel,
} from './evaluation.fitness.utils';
import {
  exportTransferableInferencePayload,
  openInferenceChannel,
} from '../../../src/neataptic';
import { rolloutEpisodeWithPredictor } from './evaluation.rollout.service';

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
  rolloutEpisodeWithPredictor: jest.fn(async () => ({
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

jest.mock('../../../src/neataptic', () => ({
  exportTransferableInferencePayload: jest.fn(() => ({
    strategy: 'transferable',
  })),
  openInferenceChannel: jest.fn(() => ({
    close: jest.fn(async () => undefined),
    predict: jest.fn(async () => new Float64Array([0.25, 0.75])),
    reset: jest.fn(async () => undefined),
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

describe('evaluateFlappyFitnessAcrossSeedsWithInferenceChannel', () => {
  it('reuses one persistent channel across seeded rollouts while resetting and closing it explicitly', async () => {
    const inferenceChannel = {
      close: jest.fn(async () => undefined),
      predict: jest.fn(async () => new Float64Array([0.25, 0.75])),
      reset: jest.fn(async () => undefined),
    };

    (openInferenceChannel as jest.Mock).mockReturnValueOnce(inferenceChannel);

    await evaluateFlappyFitnessAcrossSeedsWithInferenceChannel(
      {
        _id: 42,
      } as never,
      [11, 22, 33],
      {
        workerUrl: 'flappy-inference-channel.worker.bundle.js',
      },
    );

    expect({
      closeCalls: inferenceChannel.close.mock.calls.length,
      exportCalls: (exportTransferableInferencePayload as jest.Mock).mock.calls
        .length,
      resetCalls: inferenceChannel.reset.mock.calls.length,
      rolloutCalls: (rolloutEpisodeWithPredictor as jest.Mock).mock.calls
        .length,
    }).toEqual({
      closeCalls: 1,
      exportCalls: 1,
      resetCalls: 3,
      rolloutCalls: 3,
    });
  });
});
