import {
  createInitialState,
  stepEnvironment,
  stepEnvironmentBatch,
} from './environment.step.service';

/**
 * Red-phase contracts for the fixed-timestep environment stepping service.
 *
 * These tests target the Tier-0 acceptance criteria:
 * - tick counter increments by exactly 1 per `stepEnvironment` call
 * - fixed-timestep replay is deterministic (same inputs → same physics state)
 *
 * All tests intentionally fail until `environment.step.service.ts` is
 * implemented.
 */
describe('environment.step.service', () => {
  describe('createInitialState', () => {
    it('returns a state with tick equal to 0', () => {
      // Arrange + Act
      const state = createInitialState();

      // Assert — verifies the baseline so step-increment tests have a known starting value
      expect(state.tick).toBe(0);
    });
  });

  describe('stepEnvironment — tick counter', () => {
    it('returns a new state with tick equal to 1 after a single step from tick 0', () => {
      // Arrange
      const initial = createInitialState();

      // Act
      const next = stepEnvironment(initial, { throttle: 0, steer: 0 });

      // Assert — stub returns same state (tick=0), expected 1 → red
      expect(next.tick).toBe(1);
    });

    it('returns tick equal to 5 after 5 sequential stepEnvironment calls', () => {
      // Arrange
      let state = createInitialState();

      // Act
      for (let stepIndex = 0; stepIndex < 5; stepIndex++) {
        state = stepEnvironment(state, { throttle: 0, steer: 0 });
      }

      // Assert — stub never increments tick, ends at 0, expected 5 → red
      expect(state.tick).toBe(5);
    });
  });

  describe('stepEnvironment — physics integration', () => {
    it('produces carX greater than 0 after 10 steps with throttle=1 steer=0', () => {
      // Arrange
      let state = createInitialState();

      // Act
      for (let stepIndex = 0; stepIndex < 10; stepIndex++) {
        state = stepEnvironment(state, { throttle: 1.0, steer: 0.0 });
      }

      // Assert — stub returns carX=0 unchanged, expected > 0 → red
      expect(state.carX).toBeGreaterThan(0);
    });
  });

  describe('stepEnvironmentBatch — replay determinism', () => {
    it('matches 10 single steps with one 10-step batch replay from the same initial state', () => {
      // Arrange
      const control = { throttle: 1.0, steer: 0.0 };
      const initial = createInitialState();
      let singleReplay = initial;

      // Act
      for (let stepIndex = 0; stepIndex < 10; stepIndex++) {
        singleReplay = stepEnvironment(singleReplay, control);
      }
      const batchReplay = stepEnvironmentBatch(initial, control, 10);

      // Assert — both stubs return the unchanged initial state, so the
      // equivalence holds but the advancement-to-tick-10 contract fails → red
      expect({
        singleReplay,
        batchReplay,
        didAdvanceToTick10:
          singleReplay.tick === 10 &&
          batchReplay.tick === 10 &&
          singleReplay.carX > initial.carX,
      }).toEqual({
        singleReplay: batchReplay,
        batchReplay,
        didAdvanceToTick10: true,
      });
    });
  });
});
