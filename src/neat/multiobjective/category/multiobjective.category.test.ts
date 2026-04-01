import Network from '../../../architecture/network';
import Neat from '../../../neat';

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
  });
});
