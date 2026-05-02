import Network from '../../../architecture/network';
import * as methods from '../../../methods/methods';
import Neat from '../../../neat';
import { createGenomeFromNetwork } from '../../genome/genome';
import {
  createOffspring,
  type OffspringContext,
} from './evolve.offspring.utils';

type OffspringMetadataNetwork = Network & {
  _id?: number;
  _depth?: number;
  _parents?: Array<number | undefined>;
  _reenableProb?: number;
};

type OffspringRandomCarrier = {
  _rand?: () => number;
};

function createDeterministicRandomSequence(values: number[]): () => number {
  let randomIndex = 0;

  return () => {
    const nextValue = values[randomIndex] ?? values.at(-1) ?? 0;

    randomIndex += 1;
    return nextValue;
  };
}

function setNetworkRandomSource(
  network: Network,
  randomGenerator: () => number,
): void {
  (network as unknown as OffspringRandomCarrier)._rand = randomGenerator;
}

function createParentNetwork(input: {
  genomeId: number;
  depth: number;
  score: number;
}): OffspringMetadataNetwork {
  const parentNetwork = new Network(2, 1) as OffspringMetadataNetwork;

  parentNetwork._id = input.genomeId;
  parentNetwork._depth = input.depth;
  parentNetwork.score = input.score;

  return parentNetwork;
}

function createOffspringContext(
  parents: OffspringMetadataNetwork[],
): OffspringContext {
  return {
    population: parents,
    options: { equal: false, reenableProb: 0.2 },
    _getRNG: () => () => 0.5,
    _nextGenomeId: 30,
    _lineageEnabled: true,
    ensureMinHiddenNodes: () => {},
    ensureNoDeadEnds: () => {},
  };
}

function createSequentialParentSelector(
  parents: OffspringMetadataNetwork[],
): () => Network {
  let parentIndex = 0;

  return () => {
    const nextParent = parents[parentIndex] ?? parents.at(-1);
    parentIndex += 1;
    return nextParent as Network;
  };
}

describe('neat evolve offspring chapter', () => {
  describe('createOffspring', () => {
    describe('given lineage-enabled crossover between two scored parents', () => {
      it('records both parent ids and derives depth from the deeper parent', () => {
        // Arrange
        const firstParent = createParentNetwork({
          genomeId: 7,
          depth: 1,
          score: 2,
        });
        const secondParent = createParentNetwork({
          genomeId: 11,
          depth: 4,
          score: 3,
        });
        const offspringContext = createOffspringContext([
          firstParent,
          secondParent,
        ]);

        // Act
        const offspring = createOffspring(
          offspringContext,
          createSequentialParentSelector([firstParent, secondParent]),
        ) as OffspringMetadataNetwork;

        // Assert
        expect({
          offspringId: offspring._id,
          parents: offspring._parents,
          depth: offspring._depth,
        }).toEqual({
          offspringId: 30,
          parents: [7, 11],
          depth: 5,
        });
      });

      it('uses the controller rng instead of parent-owned runtime rng during crossover', () => {
        // Arrange
        const firstParent = createParentNetwork({
          genomeId: 13,
          depth: 1,
          score: 2,
        });
        const secondParent = firstParent.clone() as OffspringMetadataNetwork;
        secondParent._id = 17;
        secondParent._depth = 2;
        secondParent.score = 1;
        firstParent.connections[0].enabled = false;
        secondParent.connections[0].enabled = false;
        firstParent._reenableProb = 0.75;
        secondParent._reenableProb = 0.75;
        setNetworkRandomSource(firstParent, () => 0.99);
        setNetworkRandomSource(secondParent, () => 0.99);
        const randomSource = createDeterministicRandomSequence([0.8, 0.8, 0.5]);
        const offspringContext = {
          ...createOffspringContext([firstParent, secondParent]),
          _getRNG: () => randomSource,
        };
        const mathRandomSpy = jest.spyOn(Math, 'random').mockReturnValue(0.99);

        try {
          // Act
          const offspring = createOffspring(
            offspringContext,
            createSequentialParentSelector([firstParent, secondParent]),
          ) as OffspringMetadataNetwork;

          // Assert
          expect(offspring.connections[0]?.enabled).toBe(true);
        } finally {
          mathRandomSpy.mockRestore();
        }
      });

      it('keeps crossover offspring strict-genome clean for feed-forward XOR parents before population mutation runs', () => {
        // Arrange
        const evolutionController = new Neat(2, 1, () => 1, {
          elitism: 5,
          fastMode: true,
          mutation: methods.mutation.FFW,
          mutationAmount: 2,
          mutationRate: 0.8,
          popsize: 10,
          seed: 42,
        });
        const [firstParent, secondParent] = evolutionController.population;

        if (!firstParent || !secondParent) {
          throw new Error(
            'Expected at least two starter genomes for offspring testing.',
          );
        }

        const offspringContext: OffspringContext = {
          population: evolutionController.population,
          options: {
            equal:
              typeof evolutionController.options.equal === 'boolean'
                ? evolutionController.options.equal
                : undefined,
            reenableProb:
              typeof evolutionController.options.reenableProb === 'number'
                ? evolutionController.options.reenableProb
                : undefined,
          },
          _getRNG: (
            evolutionController as unknown as {
              _getRNG: () => () => number;
            }
          )._getRNG.bind(evolutionController),
          _nextGenomeId: 1,
          _lineageEnabled: false,
          ensureMinHiddenNodes: (genome) => {
            void evolutionController.ensureMinHiddenNodes(genome);
          },
          ensureNoDeadEnds: (genome) => {
            evolutionController.ensureNoDeadEnds(genome);
          },
        };
        let parentReadCount = 0;

        const selectParent = (): Network => {
          parentReadCount += 1;
          return parentReadCount === 1 ? firstParent : secondParent;
        };

        // Act
        const offspring = createOffspring(offspringContext, selectParent);

        // Assert
        expect(() => createGenomeFromNetwork(offspring)).not.toThrow();
      });
    });
  });
});
