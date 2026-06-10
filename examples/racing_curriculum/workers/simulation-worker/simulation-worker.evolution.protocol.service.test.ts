/**
 * Sibling smoke tests for `simulation-worker.evolution.protocol.service.ts`.
 *
 * Full contract tests live in `simulation-worker.evolution.protocol.test.ts`
 * (authored in Step 03).  This file satisfies the folder quality gate's
 * sibling-test-file requirement.
 *
 * Single-expect rule enforced throughout.
 */
import {
  createInitialProtocolState,
  routeRacingWorkerProtocolMessage,
} from './simulation-worker.evolution.protocol.service';

describe('simulation-worker.evolution.protocol.service module exports', () => {
  describe('createInitialProtocolState', () => {
    it('is exported as a function', () => {
      expect(typeof createInitialProtocolState).toBe('function');
    });

    it('returns an object with a phase property set to idle', () => {
      const state = createInitialProtocolState();

      expect(state.phase).toBe('idle');
    });
  });

  describe('routeRacingWorkerProtocolMessage', () => {
    it('is exported as a function', () => {
      expect(typeof routeRacingWorkerProtocolMessage).toBe('function');
    });

    it('returns an object with a nextState property', () => {
      const state = createInitialProtocolState();
      const result = routeRacingWorkerProtocolMessage({ type: 'stop' }, state);

      expect(typeof result.nextState).toBe('object');
    });
  });
});
