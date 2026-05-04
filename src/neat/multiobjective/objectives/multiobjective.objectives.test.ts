import Network from '../../../architecture/network';
import {
  buildGenomeValues,
  buildValuesMatrix,
  readObjectiveValue,
} from './multiobjective.objectives';
import type { ObjectiveDescriptor } from '../shared/multiobjective.types';

type ObjectiveTestGenome = Network & {
  primaryValue: number;
  secondaryValue: number;
};

function createObjectiveTestGenome(input: {
  seed: number;
  primaryValue: number;
  secondaryValue: number;
}): ObjectiveTestGenome {
  const genome = new Network(2, 1, { seed: input.seed }) as ObjectiveTestGenome;

  genome.primaryValue = input.primaryValue;
  genome.secondaryValue = input.secondaryValue;

  return genome;
}

describe('neat multiobjective objectives chapter', () => {
  describe('readObjectiveValue', () => {
    describe('given the objective accessor throws while reading one genome', () => {
      it('returns zero instead of letting the ranking pass fail', () => {
        // Arrange
        const genome = createObjectiveTestGenome({
          seed: 901,
          primaryValue: 3,
          secondaryValue: 7,
        });
        const descriptor: ObjectiveDescriptor = {
          accessor: () => {
            throw new Error('objective failure');
          },
        };

        // Act
        const objectiveValue = readObjectiveValue(genome, descriptor);

        // Assert
        expect(objectiveValue).toBe(0);
      });
    });
  });

  describe('buildGenomeValues', () => {
    describe('given two descriptors define one stable objective schema', () => {
      it('returns the objective vector in descriptor order', () => {
        // Arrange
        const genome = createObjectiveTestGenome({
          seed: 902,
          primaryValue: 5,
          secondaryValue: 11,
        });
        const descriptors: ObjectiveDescriptor[] = [
          {
            accessor: (candidate) =>
              (candidate as ObjectiveTestGenome).secondaryValue,
          },
          {
            accessor: (candidate) =>
              (candidate as ObjectiveTestGenome).primaryValue,
          },
        ];

        // Act
        const objectiveVector = buildGenomeValues(genome, descriptors);

        // Assert
        expect(objectiveVector).toEqual([11, 5]);
      });
    });
  });

  describe('buildValuesMatrix', () => {
    describe('given the population contains two genomes with the same descriptor schema', () => {
      it('returns one objective row per genome while preserving population order', () => {
        // Arrange
        const population = [
          createObjectiveTestGenome({
            seed: 903,
            primaryValue: 2,
            secondaryValue: 13,
          }),
          createObjectiveTestGenome({
            seed: 904,
            primaryValue: 4,
            secondaryValue: 17,
          }),
        ];
        const descriptors: ObjectiveDescriptor[] = [
          {
            accessor: (candidate) =>
              (candidate as ObjectiveTestGenome).primaryValue,
          },
          {
            accessor: (candidate) =>
              (candidate as ObjectiveTestGenome).secondaryValue,
          },
        ];

        // Act
        const valuesMatrix = buildValuesMatrix(population, descriptors);

        // Assert
        expect(valuesMatrix).toEqual([
          [2, 13],
          [4, 17],
        ]);
      });
    });
  });
});
