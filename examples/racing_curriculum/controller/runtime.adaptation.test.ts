/**
 * Sibling smoke tests for `controller/runtime.adaptation.ts`.
 *
 * The runtime adaptation engine is a demo-local within-episode mutation tool.
 * These tests verify the exported function surface is loadable and correctly
 * typed.  Detailed behaviour tests can be added in a dedicated coverage pass.
 *
 * Single-expect rule enforced throughout.
 */
import {
  createRuntimeAdaptationEngine,
  evaluateRollingScoreWindow,
} from './runtime.adaptation';

describe('runtime.adaptation module exports', () => {
  describe('createRuntimeAdaptationEngine', () => {
    it('is exported as a function', () => {
      expect(typeof createRuntimeAdaptationEngine).toBe('function');
    });
  });

  describe('evaluateRollingScoreWindow', () => {
    it('is exported as a function', () => {
      expect(typeof evaluateRollingScoreWindow).toBe('function');
    });
  });
});
