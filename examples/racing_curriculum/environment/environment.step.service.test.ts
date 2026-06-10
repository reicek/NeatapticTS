import {
  createInitialState,
  stepEnvironmentBatch,
} from './environment.step.service';

describe('environment step service sibling seam', () => {
  describe('stepEnvironmentBatch', () => {
    it('advances the tick count by the requested batch length', () => {
      const nextState = stepEnvironmentBatch(
        createInitialState(),
        { throttle: 1, steer: 0 },
        3,
      );

      expect(nextState.tick).toBe(3);
    });
  });
});
