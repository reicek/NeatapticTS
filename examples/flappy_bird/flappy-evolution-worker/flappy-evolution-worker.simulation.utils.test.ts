import { createXorshift32 } from '../rng';
import { createWorkerPopulationRenderState } from './flappy-evolution-worker.simulation.utils';

describe('createWorkerPopulationRenderState', () => {
  it('clears carried network state before a fresh playback session starts', () => {
    const firstClear = jest.fn();
    const secondClear = jest.fn();

    createWorkerPopulationRenderState(
      [
        { clear: firstClear },
        { clear: secondClear },
      ] as never[],
      createXorshift32(12345),
      1280,
      720,
    );

    expect({
      first: firstClear.mock.calls.length,
      second: secondClear.mock.calls.length,
    }).toEqual({
      first: 1,
      second: 1,
    });
  });
});