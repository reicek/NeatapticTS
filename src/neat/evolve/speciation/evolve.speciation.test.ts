import Node from '../../../architecture/node';
import Network from '../../../architecture/network/network';
import {
  applyGlobalStagnationInjectionIfNeeded,
  applySpeciationAndSharingIfEnabled,
  buildFreshGenomeForStagnation,
  ensureSpeciesHistorySnapshot,
  recordSpeciesHistorySnapshot,
  updateSpeciesStagnationIfEnabled,
} from './evolve.speciation.utils';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  SpeciesHistoryRecord,
  SpeciesWithMetadata,
} from '../evolve.types';

function createGenome(genomeId: number): GenomeWithMetadata {
  return {
    _id: genomeId,
    connections: [],
    nodes: [],
  } as GenomeWithMetadata;
}

function createSpecies(options: {
  speciesId: number;
  memberCount: number;
  avgSharedFitness?: number;
  bestScore?: number;
  lastImproved?: number;
}): SpeciesWithMetadata {
  return {
    members: Array.from(
      { length: options.memberCount },
      (_unusedValue, memberIndex) =>
        createGenome(options.speciesId * 100 + memberIndex),
    ),
    id: options.speciesId,
    generation: 0,
    avgSharedFitness: options.avgSharedFitness,
    bestScore: options.bestScore,
    lastImproved: options.lastImproved ?? 0,
  };
}

function createSpeciesHistoryRecord(
  generation: number,
  speciesId: number,
): SpeciesHistoryRecord {
  return {
    generation,
    stats: [
      {
        id: speciesId,
        size: 1,
        avgSharedFitness: speciesId,
        bestScore: speciesId * 10,
        lastImproved: generation,
      },
    ],
  };
}

function createEvolutionHost(
  overrides: Partial<NeatControllerForEvolution> = {},
): NeatControllerForEvolution {
  const defaultHost = {
    input: 1,
    output: 1,
    population: [],
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
    getOffspring: async () => createGenome(999),
    selectParent: () => createGenome(1),
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

describe('neat evolve speciation chapter', () => {
  describe('applySpeciationAndSharingIfEnabled', () => {
    describe('when speciation is disabled for the current controller', () => {
      it('skips speciation, sharing, sorting, and history callbacks', async () => {
        // Arrange
        const speciate = jest.fn();
        const applyFitnessSharing = jest.fn();
        const sort = jest.fn();
        const applyAutoCompatibilityTuning = jest.fn();
        const recordSnapshot = jest.fn();
        const evolutionHost = createEvolutionHost({
          options: {
            speciation: undefined,
          },
          _speciate: speciate,
          _applyFitnessSharing: applyFitnessSharing,
          sort,
        });

        // Act
        await applySpeciationAndSharingIfEnabled(evolutionHost, {
          applyAutoCompatibilityTuning,
          recordSpeciesHistorySnapshot: recordSnapshot,
        });

        // Assert
        expect({
          applyAutoCompatibilityTuningCalls:
            applyAutoCompatibilityTuning.mock.calls.length,
          applyFitnessSharingCalls: applyFitnessSharing.mock.calls.length,
          historySnapshotCalls: recordSnapshot.mock.calls.length,
          sortCalls: sort.mock.calls.length,
          speciateCalls: speciate.mock.calls.length,
        }).toEqual({
          applyAutoCompatibilityTuningCalls: 0,
          applyFitnessSharingCalls: 0,
          historySnapshotCalls: 0,
          sortCalls: 0,
          speciateCalls: 0,
        });
      });
    });

    describe('when speciation stays enabled but the optional hooks throw', () => {
      it('still applies tuning, sorting, and snapshot recording for the generation', async () => {
        // Arrange
        const sort = jest.fn();
        const applyAutoCompatibilityTuning = jest.fn();
        const recordSnapshot = jest.fn();
        const evolutionHost = createEvolutionHost({
          _speciate: jest.fn(() => {
            throw new Error('speciation failed');
          }),
          _applyFitnessSharing: jest.fn(() => {
            throw new Error('sharing failed');
          }),
          sort,
        });

        // Act
        await applySpeciationAndSharingIfEnabled(evolutionHost, {
          applyAutoCompatibilityTuning,
          recordSpeciesHistorySnapshot: recordSnapshot,
        });

        // Assert
        expect({
          applyAutoCompatibilityTuningCalls:
            applyAutoCompatibilityTuning.mock.calls.length,
          historySnapshotCalls: recordSnapshot.mock.calls.length,
          sortCalls: sort.mock.calls.length,
        }).toEqual({
          applyAutoCompatibilityTuningCalls: 1,
          historySnapshotCalls: 1,
          sortCalls: 1,
        });
      });
    });
  });

  describe('buildFreshGenomeForStagnation', () => {
    describe('when the fresh network starts without hidden nodes', () => {
      it('injects one hidden bridge and assigns fresh runtime metadata', async () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _nextGenomeId: 10,
          options: {
            reenableProb: 0.3,
          },
        });

        // Act
        const freshGenome = (await buildFreshGenomeForStagnation(
          evolutionHost,
        )) as unknown as Network & GenomeWithMetadata;

        // Assert
        expect({
          assignedGenomeId: freshGenome._id,
          hiddenNodeCount: freshGenome.nodes.filter(
            (node) => node.type === 'hidden',
          ).length,
          nextGenomeId: evolutionHost._nextGenomeId,
          reenableProbability: freshGenome._reenableProb,
        }).toEqual({
          assignedGenomeId: 10,
          hiddenNodeCount: 1,
          nextGenomeId: 11,
          reenableProbability: 0.3,
        });
      });
    });

    describe('when lineage tracking is enabled and minimum-hidden setup already added one node', () => {
      it('preserves lineage metadata without injecting another hidden node', async () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _lineageEnabled: true,
          ensureMinHiddenNodes: async (genome) => {
            (genome as unknown as Network).nodes.splice(
              1,
              0,
              new Node('hidden'),
            );
          },
        });

        // Act
        const freshGenome = (await buildFreshGenomeForStagnation(
          evolutionHost,
        )) as unknown as Network & GenomeWithMetadata;

        // Assert
        expect({
          hiddenNodeCount: freshGenome.nodes.filter(
            (node) => node.type === 'hidden',
          ).length,
          lineageDepth: freshGenome._depth,
          lineageParents: freshGenome._parents,
        }).toEqual({
          hiddenNodeCount: 1,
          lineageDepth: 0,
          lineageParents: [],
        });
      });
    });

    describe('when the best-effort structure hooks throw during rescue setup', () => {
      it('still returns a fresh genome instead of aborting the rescue pass', async () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _nextGenomeId: 20,
          ensureMinHiddenNodes: async () => {
            throw new Error('inject hidden setup failed');
          },
        });

        // Act
        const freshGenome = (await buildFreshGenomeForStagnation(
          evolutionHost,
        )) as unknown as Network & GenomeWithMetadata;

        // Assert
        expect({
          assignedGenomeId: freshGenome._id,
          hiddenNodeCount: freshGenome.nodes.filter(
            (node) => node.type === 'hidden',
          ).length,
          nextGenomeId: evolutionHost._nextGenomeId,
        }).toEqual({
          assignedGenomeId: 20,
          hiddenNodeCount: 0,
          nextGenomeId: 21,
        });
      });
    });

    describe('when rescue setup leaves the fresh network without an input lane', () => {
      it('returns the partially prepared genome without trying to connect the injected hidden node', async () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          _nextGenomeId: 30,
          ensureMinHiddenNodes: async (genome) => {
            (genome as unknown as Network).nodes.shift();
          },
        });

        // Act
        const freshGenome = (await buildFreshGenomeForStagnation(
          evolutionHost,
        )) as unknown as Network & GenomeWithMetadata;

        // Assert
        expect({
          assignedGenomeId: freshGenome._id,
          connectionCount: freshGenome.connections.length,
          inputNodeCount: freshGenome.nodes.filter(
            (node) => node.type === 'input',
          ).length,
        }).toEqual({
          assignedGenomeId: 30,
          connectionCount: 1,
          inputNodeCount: 0,
        });
      });
    });
  });

  describe('recordSpeciesHistorySnapshot', () => {
    describe('when extended history is already enabled', () => {
      it('keeps the fallback snapshot shelf untouched', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 5,
          options: {
            speciesAllocation: { extendedHistory: true },
          },
        });

        // Act
        recordSpeciesHistorySnapshot(evolutionHost, 2);

        // Assert
        expect(evolutionHost._speciesHistory).toBeUndefined();
      });
    });

    describe('when the fallback shelf does not exist yet', () => {
      it('creates the first summary row for the current generation', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 1,
          _species: [
            createSpecies({
              speciesId: 5,
              memberCount: 2,
              avgSharedFitness: 1.25,
              bestScore: 7,
              lastImproved: 1,
            }),
          ],
        });

        // Act
        recordSpeciesHistorySnapshot(evolutionHost, 3);

        // Assert
        expect(evolutionHost._speciesHistory).toEqual([
          {
            generation: 1,
            stats: [
              {
                id: 5,
                size: 2,
                avgSharedFitness: 1.25,
                bestScore: 7,
                lastImproved: 1,
              },
            ],
          },
        ]);
      });
    });

    describe('when the current generation does not yet have a fallback snapshot', () => {
      it('appends one trimmed summary row for the live species registry', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 3,
          _species: [
            createSpecies({
              speciesId: 7,
              memberCount: 2,
              avgSharedFitness: 1.5,
              bestScore: 9,
              lastImproved: 2,
            }),
          ],
          _speciesHistory: [
            createSpeciesHistoryRecord(1, 1),
            createSpeciesHistoryRecord(2, 2),
          ],
        });

        // Act
        recordSpeciesHistorySnapshot(evolutionHost, 2);

        // Assert
        expect(evolutionHost._speciesHistory).toEqual([
          createSpeciesHistoryRecord(2, 2),
          {
            generation: 3,
            stats: [
              {
                id: 7,
                size: 2,
                avgSharedFitness: 1.5,
                bestScore: 9,
                lastImproved: 2,
              },
            ],
          },
        ]);
      });
    });

    describe('when the live species registry is unavailable for the generation', () => {
      it('records an empty summary row instead of failing history capture', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 6,
        });

        // Act
        recordSpeciesHistorySnapshot(evolutionHost, 3);

        // Assert
        expect(evolutionHost._speciesHistory).toEqual([
          {
            generation: 6,
            stats: [],
          },
        ]);
      });
    });

    describe('when the current generation already has a fallback snapshot', () => {
      it('keeps one row for that generation instead of appending a duplicate', () => {
        // Arrange
        const existingHistory = [createSpeciesHistoryRecord(3, 3)];
        const evolutionHost = createEvolutionHost({
          generation: 3,
          _species: [createSpecies({ speciesId: 8, memberCount: 4 })],
          _speciesHistory: existingHistory,
        });

        // Act
        recordSpeciesHistorySnapshot(evolutionHost, 4);

        // Assert
        expect(evolutionHost._speciesHistory).toBe(existingHistory);
      });
    });
  });

  describe('updateSpeciesStagnationIfEnabled', () => {
    describe('when speciation is disabled', () => {
      it('skips the species stagnation hook', () => {
        // Arrange
        const updateSpeciesStagnation = jest.fn();
        const evolutionHost = createEvolutionHost({
          options: {
            speciation: undefined,
          },
          _updateSpeciesStagnation: updateSpeciesStagnation,
        });

        // Act
        updateSpeciesStagnationIfEnabled(evolutionHost);

        // Assert
        expect(updateSpeciesStagnation.mock.calls.length).toBe(0);
      });
    });

    describe('when speciation is enabled', () => {
      it('advances the species stagnation hook once for the generation', () => {
        // Arrange
        const updateSpeciesStagnation = jest.fn();
        const evolutionHost = createEvolutionHost({
          _updateSpeciesStagnation: updateSpeciesStagnation,
        });

        // Act
        updateSpeciesStagnationIfEnabled(evolutionHost);

        // Assert
        expect(updateSpeciesStagnation.mock.calls.length).toBe(1);
      });
    });
  });

  describe('applyGlobalStagnationInjectionIfNeeded', () => {
    describe('when global stagnation injection is disabled', () => {
      it('keeps the ranked population unchanged', async () => {
        // Arrange
        const buildFreshGenomeForInjection = jest.fn(async () =>
          createGenome(100),
        );
        const evolutionHost = createEvolutionHost({
          generation: 5,
          population: [createGenome(1), createGenome(2)],
          options: {
            elitism: 1,
            globalStagnationGenerations: 0,
          },
          _lastGlobalImproveGeneration: 1,
        });

        // Act
        await applyGlobalStagnationInjectionIfNeeded(evolutionHost, {
          buildFreshGenomeForStagnation: buildFreshGenomeForInjection,
          replaceFraction: 0.5,
        });

        // Assert
        expect({
          buildCalls: buildFreshGenomeForInjection.mock.calls.length,
          populationIds: evolutionHost.population.map((genome) => genome._id),
        }).toEqual({
          buildCalls: 0,
          populationIds: [1, 2],
        });
      });
    });

    describe('when the controller is still inside the allowed stagnation window', () => {
      it('preserves the current non-elite tail', async () => {
        // Arrange
        const buildFreshGenomeForInjection = jest.fn(async () =>
          createGenome(200),
        );
        const evolutionHost = createEvolutionHost({
          generation: 4,
          population: [createGenome(1), createGenome(2), createGenome(3)],
          options: {
            elitism: 1,
            globalStagnationGenerations: 2,
          },
          _lastGlobalImproveGeneration: 3,
        });

        // Act
        await applyGlobalStagnationInjectionIfNeeded(evolutionHost, {
          buildFreshGenomeForStagnation: buildFreshGenomeForInjection,
          replaceFraction: 0.5,
        });

        // Assert
        expect({
          buildCalls: buildFreshGenomeForInjection.mock.calls.length,
          populationIds: evolutionHost.population.map((genome) => genome._id),
        }).toEqual({
          buildCalls: 0,
          populationIds: [1, 2, 3],
        });
      });
    });

    describe('when the run exceeded the configured global stagnation window', () => {
      it('replaces only the bounded non-elite tail and resets the improvement generation', async () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 6,
          population: [
            createGenome(1),
            createGenome(2),
            createGenome(3),
            createGenome(4),
          ],
          options: {
            elitism: 1,
            globalStagnationGenerations: 2,
          },
          _lastGlobalImproveGeneration: 3,
        });
        let nextFreshGenomeId = 100;

        // Act
        await applyGlobalStagnationInjectionIfNeeded(evolutionHost, {
          buildFreshGenomeForStagnation: async () =>
            createGenome(nextFreshGenomeId++),
          replaceFraction: 0.5,
        });

        // Assert
        expect({
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
          populationIds: evolutionHost.population.map((genome) => genome._id),
        }).toEqual({
          lastGlobalImproveGeneration: 6,
          populationIds: [1, 2, 100, 101],
        });
      });

      it('falls back to generation zero improvement tracking and zero elitism when those values are absent', async () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 3,
          population: [
            createGenome(1),
            createGenome(2),
            createGenome(3),
            createGenome(4),
          ],
          options: {
            elitism: 0,
            globalStagnationGenerations: 1,
          },
          _lastGlobalImproveGeneration: undefined,
        });

        // Act
        await applyGlobalStagnationInjectionIfNeeded(evolutionHost, {
          buildFreshGenomeForStagnation: async () => createGenome(300),
          replaceFraction: 0.25,
        });

        // Assert
        expect({
          lastGlobalImproveGeneration:
            evolutionHost._lastGlobalImproveGeneration,
          populationIds: evolutionHost.population.map((genome) => genome._id),
        }).toEqual({
          lastGlobalImproveGeneration: 3,
          populationIds: [1, 2, 3, 300],
        });
      });
    });
  });

  describe('ensureSpeciesHistorySnapshot', () => {
    describe('when extended history is already enabled', () => {
      it('skips fallback export snapshot creation', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 9,
          options: {
            speciesAllocation: { extendedHistory: true },
          },
        });

        // Act
        ensureSpeciesHistorySnapshot(evolutionHost, 2);

        // Assert
        expect(evolutionHost._speciesHistory).toBeUndefined();
      });
    });

    describe('when the fallback export shelf does not exist yet', () => {
      it('creates the first export-ready row for the current generation', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 2,
          _species: [
            createSpecies({
              speciesId: 11,
              memberCount: 3,
              avgSharedFitness: 2.1,
              bestScore: 13,
              lastImproved: 2,
            }),
          ],
        });

        // Act
        ensureSpeciesHistorySnapshot(evolutionHost, 2);

        // Assert
        expect(evolutionHost._speciesHistory).toEqual([
          {
            generation: 2,
            stats: [
              {
                id: 11,
                size: 3,
                avgSharedFitness: 2.1,
                bestScore: 13,
                lastImproved: 2,
              },
            ],
          },
        ]);
      });
    });

    describe('when the current generation is missing from the fallback export shelf', () => {
      it('backfills one trimmed export-ready summary row', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 4,
          _species: [
            createSpecies({
              speciesId: 12,
              memberCount: 3,
              avgSharedFitness: 2.25,
              bestScore: 14,
              lastImproved: 4,
            }),
          ],
          _speciesHistory: [
            createSpeciesHistoryRecord(2, 2),
            createSpeciesHistoryRecord(3, 3),
          ],
        });

        // Act
        ensureSpeciesHistorySnapshot(evolutionHost, 2);

        // Assert
        expect(evolutionHost._speciesHistory).toEqual([
          createSpeciesHistoryRecord(3, 3),
          {
            generation: 4,
            stats: [
              {
                id: 12,
                size: 3,
                avgSharedFitness: 2.25,
                bestScore: 14,
                lastImproved: 4,
              },
            ],
          },
        ]);
      });
    });

    describe('when the live species registry is unavailable during export fallback', () => {
      it('creates an empty export-ready summary row', () => {
        // Arrange
        const evolutionHost = createEvolutionHost({
          generation: 8,
        });

        // Act
        ensureSpeciesHistorySnapshot(evolutionHost, 2);

        // Assert
        expect(evolutionHost._speciesHistory).toEqual([
          {
            generation: 8,
            stats: [],
          },
        ]);
      });
    });

    describe('when the current generation already has a fallback export row', () => {
      it('keeps one row for the generation', () => {
        // Arrange
        const existingHistory = [createSpeciesHistoryRecord(7, 7)];
        const evolutionHost = createEvolutionHost({
          generation: 7,
          _species: [createSpecies({ speciesId: 14, memberCount: 2 })],
          _speciesHistory: existingHistory,
        });

        // Act
        ensureSpeciesHistorySnapshot(evolutionHost, 3);

        // Assert
        expect(evolutionHost._speciesHistory).toBe(existingHistory);
      });
    });
  });
});
