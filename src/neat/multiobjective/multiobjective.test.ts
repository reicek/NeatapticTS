import Network from '../../architecture/network';
import { fastNonDominated } from './multiobjective';
import type {
  NeatLikeWithMultiObjective,
  NetworkWithMOAnnotations,
  ObjectiveDescriptor,
} from './shared/multiobjective.types';

type RankedNetwork = NetworkWithMOAnnotations & {
  fitnessValue: number;
  complexityValue: number;
};

type MultiObjectiveTestHost = NeatLikeWithMultiObjective & {
  _getObjectives: () => ObjectiveDescriptor[];
};

function createObjectiveRankingHost(): MultiObjectiveTestHost {
  const objectives: ObjectiveDescriptor[] = [
    {
      accessor: (genome) => (genome as RankedNetwork).fitnessValue,
      direction: 'max',
    },
    {
      accessor: (genome) => (genome as RankedNetwork).complexityValue,
      direction: 'min',
    },
  ];

  return {
    _getObjectives: () => objectives,
    options: { multiObjective: { enabled: true, archiveParetoFronts: true } },
    _paretoArchive: [],
    generation: 12,
  };
}

function createObjectiveGenome(input: {
  id: number;
  fitnessValue: number;
  complexityValue: number;
}): RankedNetwork {
  const genome = new Network(2, 1, { seed: input.id }) as RankedNetwork;

  genome._id = input.id;
  genome.fitnessValue = input.fitnessValue;
  genome.complexityValue = input.complexityValue;

  return genome;
}

function normalizeFrontIds(fronts: RankedNetwork[][]): number[][] {
  return fronts.map((front) =>
    front
      .map((genome) => genome._id ?? 0)
      .toSorted((leftId, rightId) => leftId - rightId),
  );
}

describe('neat multiobjective chapter', () => {
  describe('fastNonDominated', () => {
    describe('given a population with one dominated genome below three tradeoff survivors', () => {
      const rankingHost = createObjectiveRankingHost();
      const population = [
        createObjectiveGenome({ id: 101, fitnessValue: 9, complexityValue: 4 }),
        createObjectiveGenome({ id: 102, fitnessValue: 7, complexityValue: 2 }),
        createObjectiveGenome({ id: 103, fitnessValue: 4, complexityValue: 1 }),
        createObjectiveGenome({ id: 104, fitnessValue: 3, complexityValue: 5 }),
      ];

      let paretoFronts: RankedNetwork[][] = [];

      beforeAll(() => {
        // Arrange
        rankingHost._paretoArchive = [];

        // Act
        paretoFronts = fastNonDominated.call(
          rankingHost,
          population,
        ) as RankedNetwork[][];
      });

      it('returns the three tradeoff genomes on the first front and the dominated genome on the second', () => {
        // Assert
        expect(normalizeFrontIds(paretoFronts)).toEqual([
          [101, 102, 103],
          [104],
        ]);
      });

      it('annotates the genomes with Pareto ranks that match the returned fronts', () => {
        // Assert
        expect(
          population.map((genome) => ({
            id: genome._id ?? 0,
            rank: genome._moRank,
          })),
        ).toEqual([
          { id: 101, rank: 0 },
          { id: 102, rank: 0 },
          { id: 103, rank: 0 },
          { id: 104, rank: 1 },
        ]);
      });

      it('assigns infinity crowding to the frontier boundaries and the singleton dominated front', () => {
        // Assert
        expect(
          population.map((genome) => ({
            id: genome._id ?? 0,
            isInfiniteCrowding: genome._moCrowd === Infinity,
          })),
        ).toEqual([
          { id: 101, isInfiniteCrowding: true },
          { id: 102, isInfiniteCrowding: false },
          { id: 103, isInfiniteCrowding: true },
          { id: 104, isInfiniteCrowding: true },
        ]);
      });

      it('archives a compact snapshot of the ranked fronts with the current generation label', () => {
        // Assert
        expect(
          rankingHost._paretoArchive.map((snapshot) => ({
            generation: snapshot.generation,
            fronts: snapshot.fronts.map((front) =>
              front.toSorted((leftId, rightId) => leftId - rightId),
            ),
          })),
        ).toEqual([
          {
            generation: 12,
            fronts: [[101, 102, 103], [104]],
          },
        ]);
      });
    });
  });
});
