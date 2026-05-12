import { rolloutEpisode } from './evaluation.rollout.service';
import type { FlappyNetworkLike } from '../evaluation.types';

describe('rolloutEpisode', () => {
  it('stops on the next frame boundary when a caller abort hook fires', () => {
    let activationCallCount = 0;
    const network: FlappyNetworkLike = {
      activate: () => {
        activationCallCount += 1;
        return [1, 0];
      },
    };

    const rolloutResult = rolloutEpisode(network, {
      maxFrames: 80,
      seed: 123,
      shouldStop: () => activationCallCount > 0,
    });

    expect({
      done: rolloutResult.done,
      doneReason: rolloutResult.doneReason,
      stoppedBeforeFrameBudget: rolloutResult.framesSurvived < 80,
    }).toEqual({
      done: true,
      doneReason: 'timeout',
      stoppedBeforeFrameBudget: true,
    });
  });
});
