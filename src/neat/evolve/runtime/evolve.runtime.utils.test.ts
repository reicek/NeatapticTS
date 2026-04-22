import Network from '../../../architecture/network/network';
import {
  buildFittestSnapshot,
  clearPopulationScores,
  computeElapsedTime,
  ensurePopulationEvaluated,
  resolveStartTime,
  trackGlobalImprovement,
  updateGlobalBestTracking,
} from './evolve.runtime.utils';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
} from '../evolve.types';

type PerformanceLike = {
  now?: () => number;
};

function createEvolutionHost(
  overrides: Partial<NeatControllerForEvolution> = {},
): NeatControllerForEvolution {
  const defaultGenome: GenomeWithMetadata = {
    nodes: [],
    connections: [],
    score: 0,
  };

  const defaultHost = {
    input: 1,
    output: 1,
    population: [defaultGenome],
    generation: 0,
    options: {
      speciation: { enabled: true },
      speciesAllocation: { extendedHistory: false },
      elitism: 0,
      globalStagnationGenerations: 0,
      reenableProb: 0.25,
      minHidden: 0,
    },
    _getRNG: () => Math.random,
    _nextGenomeId: 1,
    _bestGlobalScore: 0,
    _paretoArchive: [],
    _paretoObjectivesArchive: [],
    _lastEpsilonAdjustGen: 0,
    _objectiveStale: new Map(),
    _pendingObjectiveAdds: [],
    _pendingObjectiveRemoves: [],
    _objectiveAges: new Map(),
    _lastOffspringAlloc: [],
    _prevInbreedingCount: 0,
    _lastInbreedingCount: 0,
    _sortSpeciesMembers: jest.fn(),
    _updateSpeciesStagnation: jest.fn(),
    _lastEvolveDuration: 0,
    evaluate: async () => undefined,
    sort: jest.fn(),
    mutate: async () => undefined,
    getOffspring: async () => defaultGenome,
    selectParent: () => defaultGenome,
    registerObjective: jest.fn(),
    ensureMinHiddenNodes: async () => undefined,
    ensureNoDeadEnds: jest.fn(),
  } satisfies Partial<NeatControllerForEvolution>;

  return {
    ...defaultHost,
    ...overrides,
    options: {
      ...defaultHost.options,
      ...overrides.options,
    },
  } as unknown as NeatControllerForEvolution;
}

function withMockedPerformance<T>(
  performanceValue: PerformanceLike | undefined,
  callback: () => T,
): T {
  const originalPerformanceDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    'performance',
  );

  if (!originalPerformanceDescriptor) {
    throw new Error('Expected a global performance descriptor.');
  }

  Object.defineProperty(globalThis, 'performance', {
    configurable: true,
    value: performanceValue,
  });

  try {
    return callback();
  } finally {
    Object.defineProperty(
      globalThis,
      'performance',
      originalPerformanceDescriptor,
    );
  }
}

describe('neat evolve runtime chapter', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('resolveStartTime', () => {
    describe('given the runtime exposes a high-resolution timer', () => {
      it('returns the timer callback result', () => {
        // Arrange
        const highResolutionNow = jest.fn(() => 12.5);

        // Act
        const startTime = withMockedPerformance(
          { now: highResolutionNow },
          () => resolveStartTime(),
        );

        // Assert
        expect({
          highResolutionNowCalls: highResolutionNow.mock.calls.length,
          startTime,
        }).toStrictEqual({
          highResolutionNowCalls: 1,
          startTime: 12.5,
        });
      });
    });

    describe('given the runtime exposes no high-resolution timer', () => {
      it('falls back to Date.now()', () => {
        // Arrange
        const dateNowSpy = jest.spyOn(Date, 'now').mockReturnValue(4321);

        // Act
        const startTime = withMockedPerformance(undefined, () =>
          resolveStartTime(),
        );

        // Assert
        expect({
          dateNowCalls: dateNowSpy.mock.calls.length,
          startTime,
        }).toStrictEqual({
          dateNowCalls: 1,
          startTime: 4321,
        });
      });
    });
  });

  describe('ensurePopulationEvaluated', () => {
    describe('given the current population is empty', () => {
      it('returns without requesting evaluation', async () => {
        // Arrange
        const evaluate = jest.fn(async () => undefined);
        const evolutionHost = createEvolutionHost({
          population: [],
          evaluate,
        });

        // Act
        await ensurePopulationEvaluated(evolutionHost);

        // Assert
        expect(evaluate).not.toHaveBeenCalled();
      });
    });

    describe('given the current tail genome still lacks a score', () => {
      it('forces evaluation before the generation continues', async () => {
        // Arrange
        const evaluate = jest.fn(async () => undefined);
        const evolutionHost = createEvolutionHost({
          population: [{ nodes: [], connections: [], score: undefined }],
          evaluate,
        });

        // Act
        await ensurePopulationEvaluated(evolutionHost);

        // Assert
        expect(evaluate).toHaveBeenCalledTimes(1);
      });
    });

    describe('given the current tail genome is already scored', () => {
      it('skips the evaluation hook', async () => {
        // Arrange
        const evaluate = jest.fn(async () => undefined);
        const evolutionHost = createEvolutionHost({
          population: [{ nodes: [], connections: [], score: 4 }],
          evaluate,
        });

        // Act
        await ensurePopulationEvaluated(evolutionHost);

        // Assert
        expect(evaluate).not.toHaveBeenCalled();
      });
    });
  });

  describe('updateGlobalBestTracking', () => {
    describe('given the current leader improves on an undefined prior best', () => {
      it('stores the leader score and the current generation', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 6,
          _bestScoreLastGen: undefined,
          population: [{ nodes: [], connections: [], score: 11 }],
        });

        // Act
        updateGlobalBestTracking(evolutionHost);

        // Assert
        expect({
          bestScoreLastGen: evolutionHost._bestScoreLastGen,
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
        }).toStrictEqual({
          bestScoreLastGen: 11,
          lastGlobalImproveGeneration: 6,
        });
      });
    });

    describe('given the current leader does not beat the recorded best', () => {
      it('leaves the existing tracking values unchanged', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 7,
          _bestScoreLastGen: 12,
          _lastGlobalImproveGeneration: 3,
          population: [{ nodes: [], connections: [], score: 11 }],
        });

        // Act
        updateGlobalBestTracking(evolutionHost);

        // Assert
        expect({
          bestScoreLastGen: evolutionHost._bestScoreLastGen,
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
        }).toStrictEqual({
          bestScoreLastGen: 12,
          lastGlobalImproveGeneration: 3,
        });
      });
    });

    describe('given the current leader has no numeric score yet', () => {
      it('leaves generation-level best tracking untouched', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _bestScoreLastGen: 9,
          _lastGlobalImproveGeneration: 2,
          population: [{ nodes: [], connections: [], score: undefined }],
        });

        // Act
        updateGlobalBestTracking(evolutionHost);

        // Assert
        expect({
          bestScoreLastGen: evolutionHost._bestScoreLastGen,
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
        }).toStrictEqual({
          bestScoreLastGen: 9,
          lastGlobalImproveGeneration: 2,
        });
      });
    });
  });

  describe('trackGlobalImprovement', () => {
    describe('given the cloned snapshot beats the recorded global best', () => {
      it('updates the best score and improvement generation', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _bestGlobalScore: 3,
          generation: 8,
        });
        const snapshot = new Network(1, 1);
        snapshot.score = 7;

        // Act
        trackGlobalImprovement(evolutionHost, snapshot);

        // Assert
        expect({
          bestGlobalScore: evolutionHost._bestGlobalScore,
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
        }).toStrictEqual({
          bestGlobalScore: 7,
          lastGlobalImproveGeneration: 8,
        });
      });
    });

    describe('given the cloned snapshot has no usable score improvement', () => {
      it('leaves the recorded global best state unchanged', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _bestGlobalScore: 5,
          generation: 4,
          _lastGlobalImproveGeneration: 1,
        });
        const snapshot = new Network(1, 1);

        // Act
        trackGlobalImprovement(evolutionHost, snapshot);

        // Assert
        expect({
          bestGlobalScore: evolutionHost._bestGlobalScore,
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
        }).toStrictEqual({
          bestGlobalScore: 5,
          lastGlobalImproveGeneration: 1,
        });
      });
    });
  });

  describe('computeElapsedTime', () => {
    describe('given the runtime exposes a high-resolution timer', () => {
      it('uses that timer to compute the elapsed duration', () => {
        // Arrange
        const highResolutionNow = jest.fn(() => 22.25);

        // Act
        const elapsedTime = withMockedPerformance(
          { now: highResolutionNow },
          () => computeElapsedTime(20),
        );

        // Assert
        expect({
          elapsedTime,
          highResolutionNowCalls: highResolutionNow.mock.calls.length,
        }).toStrictEqual({
          elapsedTime: 2.25,
          highResolutionNowCalls: 1,
        });
      });
    });

    describe('given the runtime exposes no high-resolution timer', () => {
      it('uses Date.now() to compute the elapsed duration', () => {
        // Arrange
        const dateNowSpy = jest.spyOn(Date, 'now').mockReturnValue(2650);

        // Act
        const elapsedTime = withMockedPerformance(undefined, () =>
          computeElapsedTime(2600),
        );

        // Assert
        expect({
          dateNowCalls: dateNowSpy.mock.calls.length,
          elapsedTime,
        }).toStrictEqual({
          dateNowCalls: 1,
          elapsedTime: 50,
        });
      });
    });
  });

  describe('clearPopulationScores', () => {
    describe('given the current population still carries stale scores', () => {
      it('clears every stored score before the next evaluation cycle', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          population: [
            { nodes: [], connections: [], score: 3 },
            { nodes: [], connections: [], score: 8 },
          ],
        });

        // Act
        clearPopulationScores(evolutionHost);

        // Assert
        expect(evolutionHost.population.map((genome) => genome.score)).toStrictEqual([
          undefined,
          undefined,
        ]);
      });
    });
  });

  describe('buildFittestSnapshot', () => {
    describe('given the leading genome does not expose a toJSON helper', () => {
      it('falls back to an empty JSON payload before preserving the score', () => {
        // Arrange
        const leadingGenome: GenomeWithMetadata = {
          nodes: [],
          connections: [],
          score: 9,
        };
        const evolutionHost = createEvolutionHost({
          population: [leadingGenome],
        });
        const clonedNetwork = new Network(1, 1);
        const fromJsonSpy = jest
          .spyOn(Network, 'fromJSON')
          .mockReturnValue(clonedNetwork);

        // Act
        const snapshot = buildFittestSnapshot(evolutionHost);

        // Assert
        expect({
          fromJsonArgs: fromJsonSpy.mock.calls[0],
          score: snapshot.score,
        }).toStrictEqual({
          fromJsonArgs: [{}],
          score: 9,
        });
      });
    });

    describe('given the current population is empty', () => {
      it('creates a fresh fallback network using the controller dimensions', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          input: 3,
          output: 2,
          population: [],
        });

        // Act
        const snapshot = buildFittestSnapshot(evolutionHost);

        // Assert
        expect({
          input: snapshot.input,
          output: snapshot.output,
          score: snapshot.score,
        }).toStrictEqual({
          input: 3,
          output: 2,
          score: undefined,
        });
      });
    });
  });
});