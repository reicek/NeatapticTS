import Network from '../../../architecture/network';
import { crossOverWithRandomGenerator } from '../../../architecture/network/genetic/network.genetic.utils';
import {
  createOffspring,
  type OffspringContext,
} from './evolve.offspring.utils';

jest.mock('../../../architecture/network/genetic/network.genetic.utils', () => ({
  crossOverWithRandomGenerator: jest.fn(),
}));

type OffspringMetadataNetwork = Network & {
  _depth?: number;
  _id?: number;
  _parents?: Array<number | undefined>;
  _reenableProb?: number;
};

const mockedCrossOverWithRandomGenerator =
  crossOverWithRandomGenerator as jest.MockedFunction<
    typeof crossOverWithRandomGenerator
  >;

function createParentNetwork(input: {
  depth?: number;
  genomeId: number;
}): OffspringMetadataNetwork {
  const parentNetwork = new Network(2, 1) as OffspringMetadataNetwork;

  parentNetwork._id = input.genomeId;

  if (typeof input.depth === 'number') {
    parentNetwork._depth = input.depth;
  }

  return parentNetwork;
}

describe('neat evolve offspring coverage chapter', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('given parent selection fails after one successful read', () => {
    describe('when createOffspring recovers through the population fallback', () => {
      it('uses the first population genome as the deterministic fallback parent', () => {
        // Arrange
        const fallbackParent = createParentNetwork({ genomeId: 5, depth: 1 });
        const selectedParent = createParentNetwork({ genomeId: 8, depth: 2 });
        const mockedOffspring = new Network(2, 1) as OffspringMetadataNetwork;
        const offspringContext: OffspringContext = {
          population: [fallbackParent, selectedParent],
          options: { reenableProb: 0.25 },
          _getRNG: () => () => 0.5,
          _nextGenomeId: 30,
          _lineageEnabled: false,
        };
        let selectionAttemptCount = 0;

        mockedCrossOverWithRandomGenerator.mockReturnValue(mockedOffspring);

        const selectParent = (): Network => {
          selectionAttemptCount += 1;

          if (selectionAttemptCount === 1) {
            return selectedParent;
          }

          throw new Error('selection failed');
        };

        // Act
        const offspring = createOffspring(
          offspringContext,
          selectParent,
        ) as OffspringMetadataNetwork;
        const [parentOne, parentTwo] =
          mockedCrossOverWithRandomGenerator.mock.calls[0] ?? [];

        // Assert
        expect({
          nextGenomeId: offspringContext._nextGenomeId,
          offspringId: offspring._id,
          parentOneId: (parentOne as OffspringMetadataNetwork | undefined)?._id,
          parentTwoId: (parentTwo as OffspringMetadataNetwork | undefined)?._id,
        }).toEqual({
          nextGenomeId: 31,
          offspringId: 30,
          parentOneId: 8,
          parentTwoId: 5,
        });
      });
    });
  });

  describe('given parent selection fails while the fallback population is empty', () => {
    describe('when createOffspring reaches the random recovery path', () => {
      it('forwards undefined parents into crossover and still annotates the offspring id', () => {
        // Arrange
        const mockedOffspring = new Network(2, 1) as OffspringMetadataNetwork;
        const offspringContext: OffspringContext = {
          population: [],
          options: {},
          _getRNG: () => () => 0.5,
          _nextGenomeId: 40,
          _lineageEnabled: false,
        };

        mockedCrossOverWithRandomGenerator.mockReturnValue(mockedOffspring);

        const selectParent = (): Network => {
          throw new Error('selection failed');
        };

        // Act
        const offspring = createOffspring(
          offspringContext,
          selectParent,
        ) as OffspringMetadataNetwork;
        const [parentOne, parentTwo, equalFlag, randomGenerator] =
          mockedCrossOverWithRandomGenerator.mock.calls[0] ?? [];

        // Assert
        expect({
          equalFlag,
          offspringId: offspring._id,
          parentOneIsUndefined: parentOne === undefined,
          parentTwoIsUndefined: parentTwo === undefined,
          randomGeneratorType: typeof randomGenerator,
          reenableProb: offspring._reenableProb,
        }).toEqual({
          equalFlag: false,
          offspringId: 40,
          parentOneIsUndefined: true,
          parentTwoIsUndefined: true,
          randomGeneratorType: 'function',
          reenableProb: undefined,
        });
      });
    });
  });

  describe('given lineage-enabled crossover reuses the same parent genome id', () => {
    describe('when createOffspring annotates the child metadata', () => {
      it('increments the inbreeding counter and falls back to the base lineage depth', () => {
        // Arrange
        const firstParent = createParentNetwork({ genomeId: 9 });
        const secondParent = createParentNetwork({ genomeId: 9 });
        const mockedOffspring = new Network(2, 1) as OffspringMetadataNetwork;
        const offspringContext: OffspringContext = {
          population: [firstParent, secondParent],
          options: { reenableProb: 0.3 },
          _getRNG: () => () => 0.5,
          _nextGenomeId: 50,
          _lineageEnabled: true,
        };
        let selectionAttemptCount = 0;

        mockedCrossOverWithRandomGenerator.mockReturnValue(mockedOffspring);

        const selectParent = (): Network => {
          selectionAttemptCount += 1;
          return selectionAttemptCount === 1 ? firstParent : secondParent;
        };

        // Act
        const offspring = createOffspring(
          offspringContext,
          selectParent,
        ) as OffspringMetadataNetwork;

        // Assert
        expect({
          depth: offspring._depth,
          inbreedingCount: offspringContext._lastInbreedingCount,
          parents: offspring._parents,
        }).toEqual({
          depth: 1,
          inbreedingCount: 1,
          parents: [9, 9],
        });
      });
    });
  });
});