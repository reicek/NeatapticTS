import {
  adaptReenableProbability,
  applyAutoCompatibilityTuning,
} from './evolve.adaptive.utils';
import type {
  GenomeWithMetadata,
  NeatControllerForEvolution,
  SpeciesWithMetadata,
} from '../evolve.types';

function createAutoCompatController(input?: {
  targetSpecies?: number;
  observedSpeciesCount?: number;
  adjustRate?: number;
  excessCoeff?: number;
  disjointCoeff?: number;
  randomValue?: number;
}): NeatControllerForEvolution {
  const genome: GenomeWithMetadata = {
    nodes: [],
    connections: [],
    _id: 1,
    score: 1,
  };

  const speciesRegistry: SpeciesWithMetadata[] = Array.from(
    { length: input?.observedSpeciesCount ?? 1 },
    (_, speciesIndex) => ({
      id: speciesIndex + 1,
      members: [genome],
      generation: 0,
      lastImproved: 0,
    }),
  );

  return {
    input: 1,
    output: 1,
    population: [genome],
    generation: 0,
    options: {
      speciation: { enabled: true },
      targetSpecies: input?.targetSpecies ?? 1,
      autoCompatTuning: {
        enabled: true,
        target: input?.targetSpecies ?? 1,
        adjustRate: input?.adjustRate ?? 0.01,
        minCoeff: 0.1,
        maxCoeff: 5,
      },
      excessCoeff: input?.excessCoeff ?? 1,
      disjointCoeff: input?.disjointCoeff ?? 1,
    },
    _bestGlobalScore: 1,
    _species: speciesRegistry,
    _speciesHistory: [],
    _nextGenomeId: 2,
    _paretoArchive: [],
    _paretoObjectivesArchive: [],
    _lastEpsilonAdjustGen: 0,
    _objectiveStale: new Map<string, number>(),
    _pendingObjectiveAdds: [],
    _pendingObjectiveRemoves: [],
    _objectiveAges: new Map<string, number>(),
    _lastOffspringAlloc: [],
    _prevInbreedingCount: 0,
    _lastInbreedingCount: 0,
    _getRNG: () => () => input?.randomValue ?? 0.5,
    _sortSpeciesMembers: (speciesWithMetadata: SpeciesWithMetadata) => {
      void speciesWithMetadata;
    },
    _updateSpeciesStagnation: () => {},
    _lastEvolveDuration: 0,
    evaluate: async () => {},
    sort: () => {},
    mutate: async () => {},
    getOffspring: async () => genome,
    selectParent: () => genome,
    registerObjective: (key, direction, accessor) => {
      void key;
      void direction;
      void accessor;
    },
    ensureMinHiddenNodes: async (genomeWithMetadata: GenomeWithMetadata) => {
      void genomeWithMetadata;
    },
    ensureNoDeadEnds: (genomeWithMetadata: GenomeWithMetadata) => {
      void genomeWithMetadata;
    },
  };
}

function createReenableController(input?: {
  reenableProb?: number;
  attemptsByGenome?: number[];
  successesByGenome?: number[];
}): NeatControllerForEvolution {
  const attemptsByGenome = input?.attemptsByGenome ?? [0];
  const successesByGenome = input?.successesByGenome ?? [0];

  const population: GenomeWithMetadata[] = attemptsByGenome.map(
    (attemptCount, genomeIndex) => ({
      nodes: [],
      connections: [],
      _id: genomeIndex + 1,
      score: 1,
      _reenableAttempts: attemptCount,
      _reenableSuccess: successesByGenome[genomeIndex] ?? 0,
    }),
  );

  return {
    input: 1,
    output: 1,
    population,
    generation: 0,
    options: {
      reenableProb: input?.reenableProb ?? 0.3,
    },
    _bestGlobalScore: 1,
    _species: [],
    _speciesHistory: [],
    _nextGenomeId: population.length + 1,
    _paretoArchive: [],
    _paretoObjectivesArchive: [],
    _lastEpsilonAdjustGen: 0,
    _objectiveStale: new Map<string, number>(),
    _pendingObjectiveAdds: [],
    _pendingObjectiveRemoves: [],
    _objectiveAges: new Map<string, number>(),
    _lastOffspringAlloc: [],
    _prevInbreedingCount: 0,
    _lastInbreedingCount: 0,
    _getRNG: () => () => 0.5,
    _sortSpeciesMembers: (speciesWithMetadata: SpeciesWithMetadata) => {
      void speciesWithMetadata;
    },
    _updateSpeciesStagnation: () => {},
    _lastEvolveDuration: 0,
    evaluate: async () => {},
    sort: () => {},
    mutate: async () => {},
    getOffspring: async () => population[0],
    selectParent: () => population[0],
    registerObjective: (key, direction, accessor) => {
      void key;
      void direction;
      void accessor;
    },
    ensureMinHiddenNodes: async (genomeWithMetadata: GenomeWithMetadata) => {
      void genomeWithMetadata;
    },
    ensureNoDeadEnds: (genomeWithMetadata: GenomeWithMetadata) => {
      void genomeWithMetadata;
    },
  };
}

describe('neat evolve adaptive chapter', () => {
  describe('applyAutoCompatibilityTuning', () => {
    describe('given zero target error and a neutral random sample', () => {
      it('keeps the compatibility coefficients inside the current equilibrium point', () => {
        // Arrange
        const evolutionController = createAutoCompatController();

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.01,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert
        expect([
          evolutionController.options.excessCoeff,
          evolutionController.options.disjointCoeff,
        ]).toEqual([1, 1]);
      });
    });

    describe('given fewer observed species than the configured target', () => {
      it('nudges both compatibility coefficients downward by the configured adjustment rate', () => {
        // Arrange
        const evolutionController = createAutoCompatController({
          targetSpecies: 3,
          observedSpeciesCount: 1,
          adjustRate: 0.1,
        });

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.1,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert
        expect([
          evolutionController.options.excessCoeff,
          evolutionController.options.disjointCoeff,
        ]).toEqual([0.9, 0.9]);
      });
    });
  });

  describe('adaptReenableProbability', () => {
    describe('given recent re-enable success stays below the configured target', () => {
      it('increases the shared re-enable probability for the next generation', () => {
        // Arrange
        const evolutionController = createReenableController({
          reenableProb: 0.3,
          attemptsByGenome: [12, 12],
          successesByGenome: [0, 0],
        });

        // Act
        adaptReenableProbability(evolutionController, {
          minSamples: 20,
          target: 0.3,
          min: 0.05,
          max: 0.9,
          deltaScale: 0.1,
        });

        // Assert
        expect(evolutionController.options.reenableProb).toBeCloseTo(0.33);
      });
    });

    describe('given the accumulated sample count does not clear the minimum gate', () => {
      it('keeps the shared re-enable probability unchanged', () => {
        // Arrange
        const evolutionController = createReenableController({
          reenableProb: 0.3,
          attemptsByGenome: [10, 10],
          successesByGenome: [0, 0],
        });

        // Act
        adaptReenableProbability(evolutionController, {
          minSamples: 20,
          target: 0.3,
          min: 0.05,
          max: 0.9,
          deltaScale: 0.1,
        });

        // Assert
        expect(evolutionController.options.reenableProb).toBe(0.3);
      });
    });

    describe('given re-enable attempts were consumed during adaptation', () => {
      it('resets the per-genome success and attempt counters', () => {
        // Arrange
        const evolutionController = createReenableController({
          attemptsByGenome: [12, 12],
          successesByGenome: [4, 5],
        });

        // Act
        adaptReenableProbability(evolutionController, {
          minSamples: 20,
          target: 0.3,
          min: 0.05,
          max: 0.9,
          deltaScale: 0.1,
        });

        // Assert
        expect(
          evolutionController.population.map((genome) => ({
            attempts: genome._reenableAttempts,
            success: genome._reenableSuccess,
          })),
        ).toEqual([
          { attempts: 0, success: 0 },
          { attempts: 0, success: 0 },
        ]);
      });
    });
  });
});
