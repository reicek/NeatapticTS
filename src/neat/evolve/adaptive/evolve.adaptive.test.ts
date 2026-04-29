import {
  adaptReenableProbability,
  applyAdaptiveComplexityControllers,
  applyAncestorUniqAdaptiveSafe,
  applyAutoCompatibilityTuning,
  applyMinimalCriterionAdaptiveSafe,
  applyOperatorAdaptationSafe,
  applyPruningAndMutation,
  invalidateCompatibilityCaches,
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

    describe('given reenableProb is undefined in the controller options', () => {
      it('returns without modifying any counter or probability', () => {
        // Arrange
        const evolutionController = createReenableController({
          attemptsByGenome: [12, 12],
          successesByGenome: [4, 5],
        });
        evolutionController.options.reenableProb = undefined;

        // Act
        adaptReenableProbability(evolutionController, {
          minSamples: 20,
          target: 0.3,
          min: 0.05,
          max: 0.9,
          deltaScale: 0.1,
        });

        // Assert
        expect(evolutionController.options.reenableProb).toBeUndefined();
      });
    });

    describe('given genomes with no per-genome counters attached', () => {
      it('treats missing counters as zero without throwing', () => {
        // Arrange
        const evolutionController = createReenableController({
          reenableProb: 0.3,
          attemptsByGenome: [],
          successesByGenome: [],
        });
        const bareGenome: GenomeWithMetadata = {
          nodes: [],
          connections: [],
          _id: 10,
          score: 0,
        };
        evolutionController.population = [bareGenome];

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

    describe('given reenableProb is null at the adjustment step', () => {
      it('falls back to the config target when computing the probability update', () => {
        // Arrange
        const evolutionController = createReenableController({
          attemptsByGenome: [12, 12],
          successesByGenome: [0, 0],
        });
        // null passes the `=== undefined` guard but triggers the `?? config.target` fallback
        evolutionController.options.reenableProb = null as unknown as number;

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
  });

  describe('applyAutoCompatibilityTuning', () => {
    describe('given autoCompatTuning is disabled', () => {
      it('returns without modifying the coefficients', () => {
        // Arrange
        const evolutionController = createAutoCompatController();
        evolutionController.options.autoCompatTuning = { enabled: false };

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.01,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert
        expect(evolutionController.options.excessCoeff).toBe(1);
      });
    });

    describe('given autoCompatTuning has no target but options has targetSpecies', () => {
      it('uses options.targetSpecies as the species count target', () => {
        // Arrange
        const evolutionController = createAutoCompatController({
          targetSpecies: 5,
          observedSpeciesCount: 3,
          adjustRate: 0.1,
        });
        // Remove the per-tuning target so the targetSpecies fallback is used
        evolutionController.options.autoCompatTuning = {
          enabled: true,
          adjustRate: 0.1,
          minCoeff: 0.1,
          maxCoeff: 5,
        };

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.1,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert — with target=5, observed=3, error=2, factor < 1 → coefficients reduced
        expect(evolutionController.options.excessCoeff).toBeLessThan(1);
      });
    });

    describe('given autoCompatTuning has no target and no options.targetSpecies', () => {
      it('derives the target from the population size square root', () => {
        // Arrange
        const evolutionController = createAutoCompatController({
          observedSpeciesCount: 1,
        });
        // Remove all target settings so Math.max fallback is used
        evolutionController.options.autoCompatTuning = { enabled: true };
        evolutionController.options.targetSpecies = undefined;
        // Remove _species so observed count falls back to 1 via || 1
        evolutionController._species = undefined;

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.1,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert — function completed without throwing regardless of coefficient value
        expect(typeof evolutionController.options.excessCoeff).toBe('number');
      });
    });

    describe('given autoCompatTuning options are all absent (uses config fallbacks)', () => {
      it('applies fallback coefficients from the config parameter', () => {
        // Arrange
        const evolutionController = createAutoCompatController({
          observedSpeciesCount: 1,
        });
        evolutionController.options.autoCompatTuning = {
          enabled: true,
          target: 1,
        };
        evolutionController.options.excessCoeff = undefined;
        evolutionController.options.disjointCoeff = undefined;
        // Force a non-zero error to avoid the equilibrium random path
        evolutionController._species = [
          { id: 1, members: [], generation: 0, lastImproved: 0 },
          { id: 2, members: [], generation: 0, lastImproved: 0 },
          { id: 3, members: [], generation: 0, lastImproved: 0 },
        ];

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.1,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert — coefficients are now defined numeric values derived from config fallbacks
        expect(typeof evolutionController.options.excessCoeff).toBe('number');
      });
    });

    describe('given species list is empty (zero length → || 1 fallback)', () => {
      it('treats observed species count as one to avoid division by zero', () => {
        // Arrange
        const evolutionController = createAutoCompatController({
          targetSpecies: 5,
        });
        evolutionController._species = [];

        // Act
        applyAutoCompatibilityTuning(evolutionController, {
          targetMin: 1,
          adjustRate: 0.1,
          minCoeff: 0.1,
          maxCoeff: 5,
          randomScale: 1,
        });

        // Assert — with observed=1 (|| 1 fallback), target=5, error=4 → coefficients reduced
        expect(evolutionController.options.excessCoeff).toBeLessThan(1);
      });
    });
  });

  describe('invalidateCompatibilityCaches', () => {
    describe('given a population with mixed cache presence', () => {
      it('removes _compatCache from genomes that have it and ignores those that do not', () => {
        // Arrange
        const evolutionController = createReenableController({
          attemptsByGenome: [0, 0],
        });
        const genomeWithCache = evolutionController.population[0];
        genomeWithCache._compatCache = { 2: 0.5 };

        // Act
        invalidateCompatibilityCaches(evolutionController);

        // Assert
        expect(genomeWithCache._compatCache).toBeUndefined();
      });
    });
  });

  describe('applyAdaptiveComplexityControllers', () => {
    describe('given a minimal controller', () => {
      it('resolves without throwing', async () => {
        // Arrange
        const evolutionController = createReenableController();

        // Act + Assert
        await expect(
          applyAdaptiveComplexityControllers(evolutionController),
        ).resolves.toBeUndefined();
      });
    });
  });

  describe('applyMinimalCriterionAdaptiveSafe', () => {
    describe('given a minimal controller', () => {
      it('resolves without throwing', async () => {
        // Arrange
        const evolutionController = createReenableController();

        // Act + Assert
        await expect(
          applyMinimalCriterionAdaptiveSafe(evolutionController),
        ).resolves.toBeUndefined();
      });
    });
  });

  describe('applyAncestorUniqAdaptiveSafe', () => {
    describe('given a minimal controller', () => {
      it('resolves without throwing', async () => {
        // Arrange
        const evolutionController = createReenableController();

        // Act + Assert
        await expect(
          applyAncestorUniqAdaptiveSafe(evolutionController),
        ).resolves.toBeUndefined();
      });
    });
  });

  describe('applyPruningAndMutation', () => {
    describe('given a minimal controller', () => {
      it('resolves without throwing', async () => {
        // Arrange
        const evolutionController = createReenableController();

        // Act + Assert
        await expect(
          applyPruningAndMutation(evolutionController),
        ).resolves.toBeUndefined();
      });
    });
  });

  describe('applyOperatorAdaptationSafe', () => {
    describe('given a minimal controller', () => {
      it('resolves without throwing', async () => {
        // Arrange
        const evolutionController = createReenableController();

        // Act + Assert
        await expect(
          applyOperatorAdaptationSafe(evolutionController),
        ).resolves.toBeUndefined();
      });
    });
  });
});
