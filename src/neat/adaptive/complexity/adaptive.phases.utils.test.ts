import {
  initializePhaseState,
  resolveNextPhase,
  togglePhaseIfNeeded,
} from './adaptive.phases.utils';
import {
  PHASE_COMPLEXIFY,
  PHASE_SIMPLIFY,
} from '../core/adaptive.core.constants';
import type {
  NeatLikeWithAdaptive,
  PhasedComplexityConfig,
} from '../core/adaptive.core.types';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function buildEngine(
  overrides: Partial<NeatLikeWithAdaptive> = {},
): NeatLikeWithAdaptive {
  return {
    options: {},
    population: [],
    input: 2,
    output: 1,
    generation: 0,
    ...overrides,
  } as NeatLikeWithAdaptive;
}

describe('adaptive phases utils chapter', () => {
  describe('initializePhaseState()', () => {
    describe('given an engine with no prior phase state', () => {
      it('sets the phase to the config initialPhase', () => {
        // Arrange
        const engine = buildEngine({ generation: 3 });
        const config: PhasedComplexityConfig = { initialPhase: PHASE_SIMPLIFY };

        // Act
        initializePhaseState(engine, config);

        // Assert
        expect(engine._phase).toBe(PHASE_SIMPLIFY);
      });
    });

    describe('given an engine that already has a phase', () => {
      it('skips reinitialisation and leaves _phase unchanged (early-return arm)', () => {
        // Arrange
        const engine = buildEngine({
          _phase: PHASE_SIMPLIFY,
          _phaseStartGeneration: 2,
        });
        const config: PhasedComplexityConfig = {
          initialPhase: PHASE_COMPLEXIFY,
        };

        // Act
        initializePhaseState(engine, config);

        // Assert: original phase untouched
        expect(engine._phase).toBe(PHASE_SIMPLIFY);
      });
    });

    describe('given a config with no initialPhase', () => {
      it('defaults to PHASE_COMPLEXIFY (line 40 ?? fallback)', () => {
        // Arrange: no initialPhase → ?? PHASE_COMPLEXIFY fires
        const engine = buildEngine({ generation: 0 });
        const config: PhasedComplexityConfig = {};

        // Act
        initializePhaseState(engine, config);

        // Assert
        expect(engine._phase).toBe(PHASE_COMPLEXIFY);
      });
    });
  });

  describe('togglePhaseIfNeeded()', () => {
    describe('given a config with no phaseLength', () => {
      it('uses PHASE_LENGTH_DEFAULT as the window (line 61 ?? fallback)', () => {
        // Arrange: phaseLength omitted → ?? PHASE_LENGTH_DEFAULT fires (= 10)
        // Set generation = 11 and _phaseStartGeneration = 0 so elapsed (11) >= 10
        const engine = buildEngine({
          generation: 11,
          _phase: PHASE_COMPLEXIFY,
          _phaseStartGeneration: 0,
        });
        const config: PhasedComplexityConfig = {};

        // Act
        togglePhaseIfNeeded(engine, config);

        // Assert: phase toggled because DEFAULT window elapsed
        expect(engine._phase).toBe(PHASE_SIMPLIFY);
      });
    });

    describe('given an engine with undefined _phaseStartGeneration', () => {
      it('treats undefined _phaseStartGeneration as ZERO (line 62 ?? fallback)', () => {
        // Arrange: _phaseStartGeneration = undefined → ?? ZERO fires
        // generation=5, phaseLength=3, so elapsed = 5 - 0 = 5 >= 3 → toggles
        const engine = buildEngine({
          generation: 5,
          _phase: PHASE_COMPLEXIFY,
          _phaseStartGeneration: undefined,
        });
        const config: PhasedComplexityConfig = { phaseLength: 3 };

        // Act
        togglePhaseIfNeeded(engine, config);

        // Assert: phase toggled using ZERO as the start baseline
        expect(engine._phase).toBe(PHASE_SIMPLIFY);
      });
    });

    describe('given an engine with undefined _phase when elapsed window expires', () => {
      it('treats undefined _phase as PHASE_COMPLEXIFY and toggles to simplify (line 66 ?? fallback)', () => {
        // Arrange: _phase = undefined → ?? PHASE_COMPLEXIFY fires inside resolveNextPhase call
        const engine = buildEngine({
          generation: 5,
          _phase: undefined,
          _phaseStartGeneration: 0,
        });
        const config: PhasedComplexityConfig = { phaseLength: 3 };

        // Act
        togglePhaseIfNeeded(engine, config);

        // Assert: resolveNextPhase('complexify') returns 'simplify'
        expect(engine._phase).toBe(PHASE_SIMPLIFY);
      });
    });

    describe('given elapsed time less than phaseLength', () => {
      it('skips the toggle and leaves _phase unchanged (early-return arm)', () => {
        // Arrange: elapsed < phaseLength → return fires
        const engine = buildEngine({
          generation: 1,
          _phase: PHASE_COMPLEXIFY,
          _phaseStartGeneration: 0,
        });
        const config: PhasedComplexityConfig = { phaseLength: 5 };

        // Act
        togglePhaseIfNeeded(engine, config);

        // Assert: no toggle
        expect(engine._phase).toBe(PHASE_COMPLEXIFY);
      });
    });
  });

  describe('resolveNextPhase()', () => {
    describe('given PHASE_COMPLEXIFY', () => {
      it('returns PHASE_SIMPLIFY', () => {
        // Act + Assert
        expect(resolveNextPhase(PHASE_COMPLEXIFY)).toBe(PHASE_SIMPLIFY);
      });
    });

    describe('given PHASE_SIMPLIFY', () => {
      it('returns PHASE_COMPLEXIFY', () => {
        // Act + Assert
        expect(resolveNextPhase(PHASE_SIMPLIFY)).toBe(PHASE_COMPLEXIFY);
      });
    });
  });
});
