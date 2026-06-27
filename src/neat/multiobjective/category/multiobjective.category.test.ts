import Network from '../../../architecture/network';
import Neat from '../../../neat';
import * as multiobjectiveModule from '../multiobjective';
import { processMultiObjective } from './multiobjective.category';

type PrunableObjectiveDescriptor = {
  [key: string]: unknown;
  key: string;
  direction?: 'max' | 'min';
  accessor?: (genome: Network) => number;
};

type DelegatedRankingNetwork = Network & {
  _moRank?: number;
  _moCrowd?: number;
  tradeoffScore?: number;
  tradeoffComplexity?: number;
};

type AdaptiveDominanceEpsilonNeat = Neat & {
  _lastEpsilonAdjustGen: number;
};

type ArchiveTrackingSurface = {
  generation: number;
  population: DelegatedRankingNetwork[];
  _paretoArchive: Array<{ gen: number; size: number; genomes: unknown[] }>;
  _paretoObjectivesArchive: Array<{ gen: number; vectors: unknown[] }>;
};

type ObjectiveStaleTrackingSurface = {
  population: SignalObjectiveNetwork[];
  _objectiveStale: Map<string, number>;
};

type SignalObjectiveNetwork = Network & {
  objectiveSignal?: number;
};

function scoreByWeightMagnitude(network: Network): number {
  return network.connections.reduce(
    (weightMagnitude, connectionEntry) =>
      weightMagnitude + Math.abs(connectionEntry.weight),
    0,
  );
}

function createObjectivePruningNeat(input: {
  popsize: number;
  seed: number;
  objectives: PrunableObjectiveDescriptor[];
  pruneInactive: {
    enabled: boolean;
    window: number;
    rangeEps: number;
    protect?: string[];
  };
}): Neat {
  return new Neat(6, 2, scoreByWeightMagnitude, {
    popsize: input.popsize,
    seed: input.seed,
    multiObjective: {
      enabled: true,
      objectives: input.objectives,
      pruneInactive: input.pruneInactive,
    },
  });
}

function createDelegatedObjectiveDescriptors(): PrunableObjectiveDescriptor[] {
  return [
    {
      key: 'tradeoffScore',
      direction: 'max',
      accessor: (genome) =>
        (genome as DelegatedRankingNetwork).tradeoffScore ?? 0,
    },
    {
      key: 'tradeoffComplexity',
      direction: 'min',
      accessor: (genome) =>
        (genome as DelegatedRankingNetwork).tradeoffComplexity ?? 0,
    },
  ];
}

async function evolveGenerations(
  neat: Neat,
  generationCount: number,
): Promise<void> {
  for (
    let generationIndex = 0;
    generationIndex < generationCount;
    generationIndex++
  ) {
    await neat.evolve();
  }
}

function readObjectiveKeys(neat: Neat): string[] {
  return neat
    .getObjectives()
    .map((objective) => objective.key)
    .toSorted();
}

function createDelegatedRankingNeat(): Neat {
  return new Neat(3, 2, (network) => network.connections.length, {
    popsize: 4,
    seed: 123,
    multiObjective: {
      enabled: true,
      objectives: createDelegatedObjectiveDescriptors(),
    },
  });
}

function createAdaptiveDominanceEpsilonNeat(): AdaptiveDominanceEpsilonNeat {
  return new Neat(3, 2, (network) => network.connections.length, {
    popsize: 4,
    seed: 123,
    multiObjective: {
      enabled: true,
      dominanceEpsilon: 0.1,
      adaptiveEpsilon: {
        enabled: true,
        targetFront: 2,
        adjust: 0.05,
        cooldown: 0,
      },
      objectives: createDelegatedObjectiveDescriptors(),
    },
  }) as AdaptiveDominanceEpsilonNeat;
}

function assignDelegatedTradeoffValues(
  population: DelegatedRankingNetwork[],
): void {
  const tradeoffValues = [
    { tradeoffScore: 9, tradeoffComplexity: 4 },
    { tradeoffScore: 7, tradeoffComplexity: 2 },
    { tradeoffScore: 4, tradeoffComplexity: 1 },
    { tradeoffScore: 3, tradeoffComplexity: 5 },
  ];

  population.forEach((genome, genomeIndex) => {
    const tradeoffValue = tradeoffValues[genomeIndex];
    genome.tradeoffScore = tradeoffValue.tradeoffScore;
    genome.tradeoffComplexity = tradeoffValue.tradeoffComplexity;
  });
}

function assignDominatedTradeoffValues(
  population: DelegatedRankingNetwork[],
): void {
  const tradeoffValues = [
    { tradeoffScore: 9, tradeoffComplexity: 1 },
    { tradeoffScore: 8, tradeoffComplexity: 2 },
    { tradeoffScore: 7, tradeoffComplexity: 3 },
    { tradeoffScore: 6, tradeoffComplexity: 4 },
  ];

  population.forEach((genome, genomeIndex) => {
    const tradeoffValue = tradeoffValues[genomeIndex];
    genome.tradeoffScore = tradeoffValue.tradeoffScore;
    genome.tradeoffComplexity = tradeoffValue.tradeoffComplexity;
  });
}

function createCategoryProcessConfig(input?: {
  paretoArchiveMax?: number;
  targetFrontMin?: number;
  targetFrontUpperRatio?: number;
  targetFrontLowerRatio?: number;
  defaultEpsilonAdjust?: number;
  defaultEpsilonMin?: number;
  defaultEpsilonMax?: number;
  defaultEpsilonCooldown?: number;
  pruneWindowDefault?: number;
  pruneRangeEpsDefault?: number;
}): Parameters<typeof processMultiObjective>[1] {
  return {
    paretoArchiveMax: input?.paretoArchiveMax ?? 5,
    targetFrontMin: input?.targetFrontMin ?? 2,
    targetFrontUpperRatio: input?.targetFrontUpperRatio ?? 1.5,
    targetFrontLowerRatio: input?.targetFrontLowerRatio ?? 0.75,
    defaultEpsilonAdjust: input?.defaultEpsilonAdjust ?? 0.05,
    defaultEpsilonMin: input?.defaultEpsilonMin ?? 0.01,
    defaultEpsilonMax: input?.defaultEpsilonMax ?? 1,
    defaultEpsilonCooldown: input?.defaultEpsilonCooldown ?? 0,
    pruneWindowDefault: input?.pruneWindowDefault ?? 3,
    pruneRangeEpsDefault: input?.pruneRangeEpsDefault ?? 1e-9,
  };
}

describe('neat multiobjective category chapter', () => {
  describe('delegated Pareto ranking during evolve', () => {
    describe('given evolve delegates to the multi-objective ranking pipeline for a controlled tradeoff surface', () => {
      const neat = createDelegatedRankingNeat();
      const population = neat.population as DelegatedRankingNetwork[];

      beforeAll(async () => {
        // Arrange
        assignDelegatedTradeoffValues(population);

        // Act
        await neat.evolve();
      });

      it('assigns the expected Pareto rank layers onto the evolved population snapshot', () => {
        // Assert
        expect(
          population
            .map((genome) => genome._moRank ?? -1)
            .toSorted((leftRank, rightRank) => leftRank - rightRank),
        ).toEqual([0, 0, 0, 1]);
      });

      it('keeps the frontier boundary genomes at infinite crowding distance after delegation', () => {
        // Assert
        expect(
          population.filter(
            (genome) => genome._moRank === 0 && genome._moCrowd === Infinity,
          ).length,
        ).toBe(2);
      });
    });
  });

  describe('adaptive dominance epsilon', () => {
    describe('given the leading Pareto front grows wider than the configured target band', () => {
      let epsilonAdjustmentSummary = {
        dominanceEpsilon: 0,
        lastAdjustGeneration: -1,
      };

      beforeAll(async () => {
        // Arrange
        const neat = createAdaptiveDominanceEpsilonNeat();
        assignDelegatedTradeoffValues(
          neat.population as DelegatedRankingNetwork[],
        );

        // Act
        await neat.evolve();

        epsilonAdjustmentSummary = {
          dominanceEpsilon: Number(
            (neat.options.multiObjective?.dominanceEpsilon ?? 0).toFixed(2),
          ),
          lastAdjustGeneration: neat._lastEpsilonAdjustGen,
        };
      });

      it('raises the dominance epsilon and records the generation that triggered the adjustment', () => {
        // Assert
        expect(epsilonAdjustmentSummary).toEqual({
          dominanceEpsilon: 0.15,
          lastAdjustGeneration: 0,
        });
      });
    });

    describe('given the leading Pareto front shrinks below the configured target band', () => {
      it('reduces the dominance epsilon and records the triggering generation', () => {
        // Arrange
        const neat = createAdaptiveDominanceEpsilonNeat();
        neat.options.multiObjective!.adaptiveEpsilon!.targetFront = 3;
        assignDominatedTradeoffValues(
          neat.population as DelegatedRankingNetwork[],
        );

        // Act
        processMultiObjective(neat as never, createCategoryProcessConfig());

        // Assert
        expect({
          dominanceEpsilon: Number(
            (neat.options.multiObjective?.dominanceEpsilon ?? 0).toFixed(2),
          ),
          lastAdjustGeneration: neat._lastEpsilonAdjustGen,
        }).toEqual({
          dominanceEpsilon: 0.05,
          lastAdjustGeneration: 0,
        });
      });
    });
  });

  describe('Pareto archive persistence', () => {
    describe('given archive recording exceeds the configured cap', () => {
      it('trims both archive streams down to the most recent generation', () => {
        // Arrange
        const neat = createDelegatedRankingNeat();
        const archiveTracking = neat as unknown as ArchiveTrackingSurface;
        assignDelegatedTradeoffValues(archiveTracking.population);
        const config = createCategoryProcessConfig({ paretoArchiveMax: 1 });

        // Act
        archiveTracking.generation = 0;
        processMultiObjective(neat as never, config);
        archiveTracking.generation = 1;
        processMultiObjective(neat as never, config);

        // Assert
        expect({
          categoryArchiveGenerations: archiveTracking._paretoArchive
            .filter((entry: { gen?: number }) => typeof entry.gen === 'number')
            .map((entry: { gen?: number }) => entry.gen),
          latestObjectiveArchiveGeneration:
            archiveTracking._paretoObjectivesArchive.at(-1)?.gen ?? null,
          objectiveArchiveLength:
            archiveTracking._paretoObjectivesArchive.length,
        }).toEqual({
          categoryArchiveGenerations: [1],
          latestObjectiveArchiveGeneration: 1,
          objectiveArchiveLength: 1,
        });
      });
    });

    describe('given category snapshots see genomes without exported ids or scores', () => {
      it('writes fallback archive identifiers and objective vectors', () => {
        // Arrange
        const neat =
          createDelegatedRankingNeat() as unknown as ArchiveTrackingSurface;
        const firstGenome = neat.population[0];
        firstGenome._id = undefined;
        firstGenome.score = undefined;
        const latestCategoryArchive = () =>
          neat._paretoArchive.at(-1) as
            { genomes?: Array<{ id?: number; score?: number }> } | undefined;
        const latestObjectiveArchive = () =>
          neat._paretoObjectivesArchive.at(-1) as
            { vectors?: Array<{ id?: number }> } | undefined;
        const fastNonDominatedSpy = jest
          .spyOn(multiobjectiveModule, 'fastNonDominated')
          .mockReturnValue([[firstGenome]] as unknown as ReturnType<
            typeof multiobjectiveModule.fastNonDominated
          >);

        try {
          // Act
          processMultiObjective(
            neat as never,
            createCategoryProcessConfig({ paretoArchiveMax: 1 }),
          );

          // Assert
          expect({
            archiveId: latestCategoryArchive()?.genomes?.[0]?.id ?? null,
            archiveScore: latestCategoryArchive()?.genomes?.[0]?.score ?? null,
            vectorId: latestObjectiveArchive()?.vectors?.[0]?.id ?? null,
          }).toEqual({ archiveId: -1, archiveScore: 0, vectorId: -1 });
        } finally {
          fastNonDominatedSpy.mockRestore();
        }
      });
    });
  });

  describe('inactive objective pruning', () => {
    describe('given inactive-objective pruning is disabled', () => {
      it('keeps the configured objective keys active across several generations', async () => {
        // Arrange
        const neat = createObjectivePruningNeat({
          popsize: 15,
          seed: 77,
          objectives: [
            { key: 'constA', direction: 'max', accessor: () => 1 },
            {
              key: 'varB',
              direction: 'max',
              accessor: scoreByWeightMagnitude,
            },
          ],
          pruneInactive: {
            enabled: false,
            window: 2,
            rangeEps: 1e-9,
          },
        });

        // Act
        await evolveGenerations(neat, 4);

        // Assert
        expect(readObjectiveKeys(neat)).toEqual(['constA', 'varB'].toSorted());
      });
    });

    describe('given the stagnant window has not fully elapsed', () => {
      it('keeps the constant objectives available before the removal threshold', async () => {
        // Arrange
        const neat = createObjectivePruningNeat({
          popsize: 18,
          seed: 88,
          objectives: [
            { key: 'constA', direction: 'max', accessor: () => 1 },
            { key: 'constB', direction: 'max', accessor: () => 2 },
            {
              key: 'varB',
              direction: 'max',
              accessor: scoreByWeightMagnitude,
            },
          ],
          pruneInactive: {
            enabled: true,
            window: 3,
            rangeEps: 1e-9,
            protect: ['varB'],
          },
        });

        // Act
        await evolveGenerations(neat, 2);
        const objectiveKeys = readObjectiveKeys(neat);

        // Assert
        expect(
          objectiveKeys.includes('constA') &&
            objectiveKeys.includes('constB') &&
            objectiveKeys.includes('varB'),
        ).toBe(true);
      });
    });

    describe('given the stagnant window has been exceeded', () => {
      it('removes the stagnant unprotected objectives and keeps the protected variable objective', async () => {
        // Arrange
        const neat = createObjectivePruningNeat({
          popsize: 18,
          seed: 88,
          objectives: [
            { key: 'constA', direction: 'max', accessor: () => 1 },
            { key: 'constB', direction: 'max', accessor: () => 2 },
            {
              key: 'varB',
              direction: 'max',
              accessor: scoreByWeightMagnitude,
            },
          ],
          pruneInactive: {
            enabled: true,
            window: 3,
            rangeEps: 1e-9,
            protect: ['varB'],
          },
        });

        // Act
        await evolveGenerations(neat, 4);
        const objectivePresence = readObjectiveKeys(neat).reduce(
          (
            presence: { constA: boolean; constB: boolean; varB: boolean },
            objectiveKey,
          ) => {
            if (objectiveKey === 'constA') presence.constA = true;
            if (objectiveKey === 'constB') presence.constB = true;
            if (objectiveKey === 'varB') presence.varB = true;
            return presence;
          },
          { constA: false, constB: false, varB: false },
        );

        // Assert
        expect(objectivePresence).toEqual({
          constA: false,
          constB: false,
          varB: true,
        });
      });
    });

    describe('given a prunable objective becomes variable again', () => {
      it('resets the stale counter instead of carrying forward the old count', () => {
        // Arrange
        const neat = createObjectivePruningNeat({
          popsize: 4,
          seed: 89,
          objectives: [
            {
              key: 'signal',
              direction: 'max',
              accessor: (genome) =>
                (genome as SignalObjectiveNetwork).objectiveSignal ?? 0,
            },
          ],
          pruneInactive: {
            enabled: true,
            window: 3,
            rangeEps: 1e-9,
          },
        });
        const objectiveStaleTracking =
          neat as unknown as ObjectiveStaleTrackingSurface;
        objectiveStaleTracking.population.forEach((genome, genomeIndex) => {
          genome.objectiveSignal = genomeIndex;
        });
        objectiveStaleTracking._objectiveStale.set('signal', 2);

        // Act
        processMultiObjective(neat as never, createCategoryProcessConfig());

        // Assert
        expect(objectiveStaleTracking._objectiveStale.get('signal')).toBe(0);
      });
    });

    describe('given prune-inactive uses default thresholds and objectives are unavailable', () => {
      it('leaves the stale registry untouched', () => {
        // Arrange
        const neat = new Neat(3, 2, scoreByWeightMagnitude, {
          popsize: 4,
          seed: 90,
          multiObjective: {
            enabled: true,
            objectives: createDelegatedObjectiveDescriptors(),
            pruneInactive: {
              enabled: true,
            },
          },
        }) as unknown as ObjectiveStaleTrackingSurface & {
          _getObjectives?: () => ReturnType<Neat['getObjectives']>;
        };
        neat._objectiveStale.set('signal', 2);
        neat._getObjectives = () => [];

        // Act
        processMultiObjective(neat as never, createCategoryProcessConfig());

        // Assert
        expect(Object.fromEntries(neat._objectiveStale)).toEqual({ signal: 2 });
      });
    });

    describe('given no Pareto fronts are produced and objectives are unavailable', () => {
      it('keeps archives and stale counters unchanged', () => {
        // Arrange
        const neat =
          createDelegatedRankingNeat() as unknown as ArchiveTrackingSurface &
            ObjectiveStaleTrackingSurface & {
              _getObjectives?: () => ReturnType<Neat['getObjectives']>;
              options: Neat['options'];
              _lastEpsilonAdjustGen: number;
            };
        neat._objectiveStale.set('signal', 2);
        neat._getObjectives = undefined;
        neat.options.multiObjective!.adaptiveEpsilon = { enabled: true };
        neat.options.multiObjective!.pruneInactive = { enabled: true };
        const fastNonDominatedSpy = jest
          .spyOn(multiobjectiveModule, 'fastNonDominated')
          .mockReturnValue(
            [] as unknown as ReturnType<
              typeof multiobjectiveModule.fastNonDominated
            >,
          );

        try {
          // Act
          processMultiObjective(neat as never, createCategoryProcessConfig());

          // Assert
          expect({
            archiveLength: neat._paretoArchive.length,
            objectiveArchiveLength: neat._paretoObjectivesArchive.length,
            stale: Object.fromEntries(neat._objectiveStale),
          }).toEqual({
            archiveLength: 0,
            objectiveArchiveLength: 0,
            stale: { signal: 2 },
          });
        } finally {
          fastNonDominatedSpy.mockRestore();
        }
      });
    });
  });

  describe('direct process fallbacks', () => {
    describe('given adaptive epsilon uses default tuning and the frontier stays inside the target band', () => {
      it('keeps epsilon at the zero fallback and records the generation', () => {
        // Arrange
        const neat = createAdaptiveDominanceEpsilonNeat();
        neat.options.multiObjective!.dominanceEpsilon = undefined;
        neat.options.multiObjective!.adaptiveEpsilon = {
          enabled: true,
          min: 0.01,
          max: 1,
        };
        const inBandFront = (
          neat.population as DelegatedRankingNetwork[]
        ).slice(0, 2);
        const fastNonDominatedSpy = jest
          .spyOn(multiobjectiveModule, 'fastNonDominated')
          .mockReturnValue([inBandFront] as unknown as ReturnType<
            typeof multiobjectiveModule.fastNonDominated
          >);

        try {
          // Act
          processMultiObjective(neat as never, createCategoryProcessConfig());

          // Assert
          expect({
            dominanceEpsilon:
              neat.options.multiObjective?.dominanceEpsilon ?? 0,
            lastAdjustGeneration: neat._lastEpsilonAdjustGen,
          }).toEqual({ dominanceEpsilon: 0, lastAdjustGeneration: 0 });
        } finally {
          fastNonDominatedSpy.mockRestore();
        }
      });
    });

    describe('given adaptive epsilon is still cooling down under the default cooldown', () => {
      it('returns before changing the stored dominance epsilon', () => {
        // Arrange
        const neat = createAdaptiveDominanceEpsilonNeat();
        neat._lastEpsilonAdjustGen = 0;
        neat.options.multiObjective!.adaptiveEpsilon = {
          enabled: true,
        };
        const wideFront = neat.population as DelegatedRankingNetwork[];
        const fastNonDominatedSpy = jest
          .spyOn(multiobjectiveModule, 'fastNonDominated')
          .mockReturnValue([wideFront] as unknown as ReturnType<
            typeof multiobjectiveModule.fastNonDominated
          >);

        try {
          // Act
          processMultiObjective(
            neat as never,
            createCategoryProcessConfig({ defaultEpsilonCooldown: 2 }),
          );

          // Assert
          expect({
            dominanceEpsilon:
              neat.options.multiObjective?.dominanceEpsilon ?? 0,
            lastAdjustGeneration: neat._lastEpsilonAdjustGen,
          }).toEqual({ dominanceEpsilon: 0.1, lastAdjustGeneration: 0 });
        } finally {
          fastNonDominatedSpy.mockRestore();
        }
      });
    });
  });
});
