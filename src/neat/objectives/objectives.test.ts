import {
  _getObjectives,
  clearObjectives,
  registerObjective,
} from './objectives';
import Network from '../../architecture/network/network';
import { structuralEntropy } from '../diversity/diversity';
import type { ObjectiveDescriptor } from '../shared/neat.shared.types';
import type { NeatLikeWithObjectives } from './core/objectives.types';

type ObjectivesTestHost = NeatLikeWithObjectives & {
  _getObjectives: () => ObjectiveDescriptor[];
};

function createObjectivesHost(input?: {
  multiObjectiveEnabled?: boolean;
  objectives?: ObjectiveDescriptor[];
  suppressFitnessObjective?: boolean;
}): ObjectivesTestHost {
  const host: ObjectivesTestHost = {
    options: input?.multiObjectiveEnabled
      ? {
          multiObjective: {
            enabled: true,
            objectives: input.objectives,
          },
        }
      : {},
    _objectivesList: undefined,
    _suppressFitnessObjective: input?.suppressFitnessObjective,
    _getObjectives: () => [],
  };

  host._getObjectives = () => _getObjectives.call(host);

  return host;
}

describe('neat objectives chapter', () => {
  describe('_getObjectives', () => {
    describe('given a controller with no custom objective layer', () => {
      it('returns the default fitness objective', () => {
        // Arrange
        const objectivesHost = createObjectivesHost();

        // Act
        const objectiveKeys = objectivesHost
          ._getObjectives()
          .map((objective) => objective.key);

        // Assert
        expect(objectiveKeys).toEqual(['fitness']);
      });
    });
  });

  describe('registerObjective', () => {
    describe('given multi-objective mode with an empty registered list', () => {
      it('adds the new descriptor after the default fitness objective', () => {
        // Arrange
        const objectivesHost = createObjectivesHost({
          multiObjectiveEnabled: true,
          objectives: [],
        });

        // Act
        registerObjective.call(objectivesHost, 'sparsity', 'min', () => 0);

        // Assert
        expect(
          objectivesHost._getObjectives().map((objective) => objective.key),
        ).toEqual(['fitness', 'sparsity']);
      });
    });

    describe('given a controller registering a custom entropy objective', () => {
      it('rebuilds the resolved objective list with the entropy descriptor appended', () => {
        // Arrange
        const objectivesHost = createObjectivesHost({
          multiObjectiveEnabled: true,
          objectives: [],
        });

        // Act
        registerObjective.call(objectivesHost, 'entropy', 'max', (genome) =>
          structuralEntropy(genome as Network),
        );

        // Assert
        expect(
          objectivesHost._getObjectives().map((objective) => objective.key),
        ).toEqual(['fitness', 'entropy']);
      });
    });
  });

  describe('clearObjectives', () => {
    describe('given a controller with one registered custom objective', () => {
      it('removes the custom layer and falls back to the default fitness objective', () => {
        // Arrange
        const objectivesHost = createObjectivesHost({
          multiObjectiveEnabled: true,
          objectives: [],
        });
        registerObjective.call(objectivesHost, 'temp', 'max', () => 0);

        // Act
        clearObjectives.call(objectivesHost);

        // Assert
        expect(
          objectivesHost._getObjectives().map((objective) => objective.key),
        ).toEqual(['fitness']);
      });
    });
  });
});
