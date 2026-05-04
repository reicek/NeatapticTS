import {
  adjustConnectionBudget,
  adjustNodeBudget,
  applyLinearSchedule,
  clampNodeBudget,
  computeAdjustmentFactors,
  computeNoveltyFactor,
  computeSlope,
  computeTrends,
  initializeConnectionBudget,
  initializeNodeBudget,
  updateScoreHistory,
} from './adaptive.complexity.utils';
import type {
  ComplexityBudgetConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function buildEngine(
  overrides: Partial<NeatLikeWithAdaptive> = {},
): NeatLikeWithAdaptive {
  return {
    options: {},
    population: [{ score: 5 }],
    input: 2,
    output: 1,
    generation: 0,
    ...overrides,
  } as NeatLikeWithAdaptive;
}

describe('adaptive complexity utils chapter', () => {
  describe('applyLinearSchedule()', () => {
    describe('given no maxNodesStart in config', () => {
      it('uses minimalTopology as start budget (line 126 ?? right arm)', () => {
        // Arrange: maxNodesStart omitted → ?? minimalTopology fires
        const engine = buildEngine({ input: 2, output: 1 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'linear',
          maxNodesEnd: 10,
          horizon: 5,
        };

        // Act
        applyLinearSchedule(engine, config);

        // Assert: maxNodes is at least minimalTopology (2+1+2=5) at generation 0
        expect(engine.options.maxNodes).toBe(5);
      });
    });

    describe('given no maxNodesEnd in config', () => {
      it('uses startBudget * BUDGET_GROWTH_MULTIPLIER as end budget (line 128 ?? right arm)', () => {
        // Arrange: maxNodesEnd omitted → ?? startBudget * 4 fires
        const engine = buildEngine({ input: 2, output: 1, generation: 10 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'linear',
          maxNodesStart: 6,
          horizon: 10,
        };

        // Act
        applyLinearSchedule(engine, config);

        // Assert: maxNodes computed using default end = 6 * 4 = 24
        expect(engine.options.maxNodes).toBeGreaterThan(6);
      });
    });

    describe('given no horizon in config', () => {
      it('uses LINEAR_HORIZON_DEFAULT as the horizon (line 129 ?? right arm)', () => {
        // Arrange: horizon omitted → ?? LINEAR_HORIZON_DEFAULT (= 100) fires
        const engine = buildEngine({ input: 2, output: 1, generation: 0 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'linear',
          maxNodesStart: 6,
          maxNodesEnd: 12,
        };

        // Act
        applyLinearSchedule(engine, config);

        // Assert: at generation 0 with 100-gen horizon, stays at start
        expect(engine.options.maxNodes).toBe(6);
      });
    });
  });

  describe('updateScoreHistory()', () => {
    describe('given first genome has no score', () => {
      it('uses ZERO as currentBestScore via ?? fallback (line 158 ?? right arm)', () => {
        // Arrange: genome.score = undefined → ?. returns undefined → ?? ZERO fires
        const engine = buildEngine({
          population: [
            { score: undefined },
          ] as NeatLikeWithAdaptive['population'],
        });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 3,
        };

        // Act
        const history = updateScoreHistory(engine, config);

        // Assert: zero pushed into history
        expect(history).toEqual([0]);
      });
    });

    describe('given no improvementWindow in config', () => {
      it('uses DEFAULT_IMPROVEMENT_WINDOW (10) as window (line 161 ?? right arm)', () => {
        // Arrange: improvementWindow omitted → ?? DEFAULT_IMPROVEMENT_WINDOW fires
        const engine = buildEngine();
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
        };

        // Act — push 11 scores to see if the window trims to 10
        for (let i = 0; i < 11; i++) updateScoreHistory(engine, config);

        // Assert: capped at DEFAULT_IMPROVEMENT_WINDOW = 10
        expect(engine._cbHistory?.length).toBe(10);
      });
    });

    describe('given population is empty', () => {
      it('uses ZERO via ?. fallback (line 158 ?. short-circuit)', () => {
        // Arrange: empty population → population[0] is undefined → ?. returns undefined → ?? ZERO
        const engine = buildEngine({ population: [] });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 5,
        };

        // Act
        const history = updateScoreHistory(engine, config);

        // Assert: zero pushed into history
        expect(history).toEqual([0]);
      });
    });
  });

  describe('computeTrends()', () => {
    describe('given history shorter than min improvement count', () => {
      it('returns zero improvement and zero slope (both false arms)', () => {
        // Arrange: history length < 2 → improvement = 0; < 3 → slope = 0
        const result = computeTrends([5]);

        // Assert
        expect(result).toEqual({ improvement: 0, slope: 0 });
      });
    });

    describe('given history longer than min slope count', () => {
      it('computes non-zero slope from trend data', () => {
        // Arrange: length >= 3 → computeSlope is called
        const result = computeTrends([1, 2, 3]);

        // Assert: positive slope
        expect(result.slope).toBeGreaterThan(0);
      });
    });
  });

  describe('computeSlope()', () => {
    describe('given history with all same values', () => {
      it('returns slope of zero (flat trend)', () => {
        // Arrange: no change across history → OLS slope = 0
        const result = computeSlope([5, 5, 5]);

        // Assert
        expect(result).toBe(0);
      });
    });
  });

  describe('computeAdjustmentFactors()', () => {
    describe('given no increaseFactor or stagnationFactor in config', () => {
      it('uses default factors (both ?? right arms)', () => {
        // Arrange: both omitted → defaults fire
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
        };
        const trends = { improvement: 0, slope: 0 };
        const history = [5, 5, 5];

        // Act
        const factors = computeAdjustmentFactors(config, trends, history);

        // Assert: factors based on defaults
        expect(factors.increaseFactor).toBeGreaterThan(1);
        expect(factors.stagnationFactor).toBeLessThan(1);
      });
    });
  });

  describe('computeNoveltyFactor()', () => {
    describe('given no novelty archive on engine', () => {
      it('returns NOVELTY_FACTOR_SMALL (0.9) via ?? fallback', () => {
        // Arrange: _noveltyArchive = undefined → ?. returns undefined → ?? ZERO → not > min
        const engine = buildEngine();

        // Act
        const factor = computeNoveltyFactor(engine);

        // Assert
        expect(factor).toBe(0.9);
      });
    });
  });

  describe('initializeNodeBudget()', () => {
    describe('given no maxNodesStart in config', () => {
      it('uses minimalTopology as default (line 296 ?? right arm)', () => {
        // Arrange: maxNodesStart omitted → ?? minimalTopology fires
        const engine = buildEngine({ _cbMaxNodes: undefined });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
        };

        // Act
        initializeNodeBudget(engine, config);

        // Assert: _cbMaxNodes initialized to minimalTopology (2+1+2=5)
        expect(engine._cbMaxNodes).toBe(5);
      });
    });
  });

  describe('adjustNodeBudget()', () => {
    describe('given improving trend with no maxNodesEnd', () => {
      it('uses _cbMaxNodes * BUDGET_GROWTH_MULTIPLIER as cap (line 330 ?? right arm)', () => {
        // Arrange: improving + no maxNodesEnd → ?? _cbMaxNodes * 4
        const engine = buildEngine({ _cbMaxNodes: 10 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
          improvementWindow: 3,
        };
        const trends = { improvement: 1, slope: 0 };
        const factors = { increaseFactor: 1.5, stagnationFactor: 0.95 };
        const history = [1, 2, 3];

        // Act
        adjustNodeBudget(engine, config, trends, factors, 1, history);

        // Assert: maxNodes grew (improvement path), capped at 10*4=40
        expect(engine._cbMaxNodes).toBeGreaterThan(10);
        expect(engine._cbMaxNodes).toBeLessThanOrEqual(40);
      });
    });

    describe('given no improvementWindow in config', () => {
      it('uses DEFAULT_IMPROVEMENT_WINDOW as window size (line 324 ?? right arm)', () => {
        // Arrange: improvementWindow omitted → ?? DEFAULT_IMPROVEMENT_WINDOW (10) fires
        const engine = buildEngine({ _cbMaxNodes: 6 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
        };
        const trends = { improvement: 0, slope: 0 };
        const factors = { increaseFactor: 1.1, stagnationFactor: 0.95 };

        // Act — history shorter than DEFAULT_IMPROVEMENT_WINDOW (10) → not full → no stagnation
        adjustNodeBudget(engine, config, trends, factors, 1, [1, 2]);

        // Assert: budget unchanged (neither improving nor window full)
        expect(engine._cbMaxNodes).toBe(6);
      });
    });
  });

  describe('clampNodeBudget()', () => {
    describe('given no minNodes in config', () => {
      it('uses minimalTopology as min (line 360 ?? right arm)', () => {
        // Arrange: minNodes omitted → ?? minimalTopology fires; _cbMaxNodes < minimalTopology
        const engine = buildEngine({ _cbMaxNodes: 1 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
        };

        // Act
        clampNodeBudget(engine, config);

        // Assert: _cbMaxNodes clamped to minimalTopology (2+1+2=5)
        expect(engine._cbMaxNodes).toBe(5);
      });
    });
  });

  describe('initializeConnectionBudget()', () => {
    describe('given _cbMaxConns is undefined', () => {
      it('seeds _cbMaxConns from config.maxConnsStart', () => {
        // Arrange
        const engine = buildEngine({ _cbMaxConns: undefined });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
          maxConnsStart: 12,
        };

        // Act
        initializeConnectionBudget(engine, config);

        // Assert
        expect(engine._cbMaxConns).toBe(12);
      });
    });
  });

  describe('adjustConnectionBudget()', () => {
    describe('given improving trend with no maxConnsEnd', () => {
      it('uses _cbMaxConns * BUDGET_GROWTH_MULTIPLIER as cap (line 411 ?? right arm)', () => {
        // Arrange: improving + no maxConnsEnd → ?? _cbMaxConns * 4
        const engine = buildEngine({ _cbMaxConns: 10 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
          maxConnsStart: 10,
          improvementWindow: 3,
        };
        const trends = { improvement: 1, slope: 0 };
        const factors = { increaseFactor: 1.5, stagnationFactor: 0.95 };
        const history = [1, 2, 3];

        // Act
        adjustConnectionBudget(engine, config, trends, factors, 1, history);

        // Assert: maxConns grew, capped at 10*4=40
        expect(engine._cbMaxConns).toBeGreaterThan(10);
        expect(engine._cbMaxConns).toBeLessThanOrEqual(40);
      });
    });

    describe('given no improvementWindow in config', () => {
      it('uses DEFAULT_IMPROVEMENT_WINDOW as window size (line 405 ?? right arm)', () => {
        // Arrange: improvementWindow omitted → ?? DEFAULT_IMPROVEMENT_WINDOW fires
        const engine = buildEngine({ _cbMaxConns: 10 });
        const config: ComplexityBudgetConfig = {
          enabled: true,
          mode: 'adaptive',
          maxConnsStart: 10,
        };
        const trends = { improvement: 0, slope: 0 };
        const factors = { increaseFactor: 1.1, stagnationFactor: 0.95 };

        // Act — history shorter than 10 → not window full → no stagnation
        adjustConnectionBudget(engine, config, trends, factors, 1, [1, 2]);

        // Assert: budget unchanged
        expect(engine._cbMaxConns).toBe(10);
      });
    });
  });
});
