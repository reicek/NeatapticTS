import Network from '../../../architecture/network/network';
import type { ObjectiveDescriptor } from '../../shared/neat.shared.types';
import {
  buildDefaultFitnessObjective,
  collectDefaultObjectives,
  collectUserObjectives,
  ensureMultiObjectiveOptions,
  ensureObjectivesList,
  getObjectiveCandidates,
  isMultiObjectiveEnabled,
  isValidObjective,
  replaceObjectiveByKey,
} from './objectives.core';
import type { NeatLikeWithObjectives } from './objectives.types';

function createObjectiveDescriptor(
  input?: Partial<ObjectiveDescriptor>,
): ObjectiveDescriptor {
  return {
    key: input?.key ?? 'fitness',
    direction: input?.direction ?? 'max',
    accessor: input?.accessor ?? (() => 0),
  };
}

function createObjectivesHost(input?: {
  multiObjective?: NeatLikeWithObjectives['options']['multiObjective'];
}): NeatLikeWithObjectives {
  return {
    options: input?.multiObjective
      ? { multiObjective: input.multiObjective }
      : {},
    _objectivesList: undefined,
    _suppressFitnessObjective: undefined,
  };
}

describe('neat objectives core chapter', () => {
  describe('collectDefaultObjectives', () => {
    describe('given the default fitness objective is suppressed', () => {
      it('returns no default objectives', () => {
        // Arrange
        const objectivesHost: NeatLikeWithObjectives = {
          ...createObjectivesHost(),
          _suppressFitnessObjective: true,
        };

        // Act
        const defaultObjectives = collectDefaultObjectives(objectivesHost);

        // Assert
        expect(defaultObjectives).toEqual([]);
      });
    });

    describe('given the default fitness objective is active', () => {
      it('returns the fitness objective key', () => {
        // Arrange
        const objectivesHost = createObjectivesHost();

        // Act
        const defaultObjectiveKeys = collectDefaultObjectives(
          objectivesHost,
        ).map((objective) => objective.key);

        // Assert
        expect(defaultObjectiveKeys).toEqual(['fitness']);
      });
    });
  });

  describe('collectUserObjectives', () => {
    describe('given multi-objective mode is disabled', () => {
      it('returns no user objectives', () => {
        // Arrange
        const objectivesHost = createObjectivesHost();

        // Act
        const userObjectives = collectUserObjectives(objectivesHost);

        // Assert
        expect(userObjectives).toEqual([]);
      });
    });

    describe('given multi-objective mode contains valid and invalid candidates', () => {
      it('keeps only the valid objective key', () => {
        // Arrange
        const objectivesHost = createObjectivesHost({
          multiObjective: {
            enabled: true,
            objectives: [
              createObjectiveDescriptor({ key: 'novelty', direction: 'max' }),
              createObjectiveDescriptor({ key: '' }),
            ],
          },
        });

        // Act
        const userObjectiveKeys = collectUserObjectives(objectivesHost).map(
          (objective) => objective.key,
        );

        // Assert
        expect(userObjectiveKeys).toEqual(['novelty']);
      });
    });
  });

  describe('buildDefaultFitnessObjective', () => {
    describe('given the genome exposes a numeric score', () => {
      it('returns that score from the accessor', () => {
        // Arrange
        const defaultFitnessObjective = buildDefaultFitnessObjective();
        const scoredGenome = new Network(1, 1);
        Reflect.set(scoredGenome, 'score', 7);

        // Act
        const fitnessScore = defaultFitnessObjective.accessor(scoredGenome);

        // Assert
        expect(fitnessScore).toBe(7);
      });
    });

    describe('given the genome score is missing', () => {
      it('falls back to zero', () => {
        // Arrange
        const defaultFitnessObjective = buildDefaultFitnessObjective();
        const unscoredGenome = new Network(1, 1);

        // Act
        const fitnessScore = defaultFitnessObjective.accessor(unscoredGenome);

        // Assert
        expect(fitnessScore).toBe(0);
      });
    });
  });

  describe('isMultiObjectiveEnabled', () => {
    describe('given the controller has an enabled objective array', () => {
      it('returns true', () => {
        // Arrange
        const objectivesHost = createObjectivesHost({
          multiObjective: {
            enabled: true,
            objectives: [createObjectiveDescriptor({ key: 'novelty' })],
          },
        });

        // Act
        const isEnabled = isMultiObjectiveEnabled(objectivesHost);

        // Assert
        expect(isEnabled).toBe(true);
      });
    });
  });

  describe('getObjectiveCandidates', () => {
    describe('given the multi-objective container is missing', () => {
      it('returns an empty objective list', () => {
        // Arrange
        const objectivesHost = createObjectivesHost();

        // Act
        const objectiveCandidates = getObjectiveCandidates(objectivesHost);

        // Assert
        expect(objectiveCandidates).toEqual([]);
      });
    });

    describe('given the multi-objective container already has objectives', () => {
      it('returns the configured objective list', () => {
        // Arrange
        const configuredObjectives = [
          createObjectiveDescriptor({ key: 'novelty' }),
        ];
        const objectivesHost = createObjectivesHost({
          multiObjective: {
            enabled: true,
            objectives: configuredObjectives,
          },
        });

        // Act
        const objectiveCandidates = getObjectiveCandidates(objectivesHost);

        // Assert
        expect(objectiveCandidates).toBe(configuredObjectives);
      });
    });
  });

  describe('isValidObjective', () => {
    describe('given the candidate objective is undefined', () => {
      it('returns false', () => {
        // Arrange
        const candidateObjective = undefined;

        // Act
        const isCandidateValid = isValidObjective(candidateObjective);

        // Assert
        expect(isCandidateValid).toBe(false);
      });
    });

    describe('given the candidate objective has an empty key', () => {
      it('returns false', () => {
        // Arrange
        const candidateObjective = createObjectiveDescriptor({ key: '' });

        // Act
        const isCandidateValid = isValidObjective(candidateObjective);

        // Assert
        expect(isCandidateValid).toBe(false);
      });
    });

    describe('given the candidate objective has a key and accessor', () => {
      it('returns true', () => {
        // Arrange
        const candidateObjective = createObjectiveDescriptor({ key: 'novelty' });

        // Act
        const isCandidateValid = isValidObjective(candidateObjective);

        // Assert
        expect(isCandidateValid).toBe(true);
      });
    });
  });

  describe('ensureMultiObjectiveOptions', () => {
    describe('given the multi-objective container is missing', () => {
      it('creates an enabled multi-objective container', () => {
        // Arrange
        const objectivesHost = createObjectivesHost();

        // Act
        const multiObjectiveOptions = ensureMultiObjectiveOptions(
          objectivesHost,
        );

        // Assert
        expect(multiObjectiveOptions.enabled).toBe(true);
      });
    });

    describe('given the multi-objective container already exists', () => {
      it('returns the existing container reference', () => {
        // Arrange
        const existingOptions = { enabled: false, objectives: [] };
        const objectivesHost = createObjectivesHost({
          multiObjective: existingOptions,
        });

        // Act
        const multiObjectiveOptions = ensureMultiObjectiveOptions(
          objectivesHost,
        );

        // Assert
        expect(multiObjectiveOptions).toBe(existingOptions);
      });
    });
  });

  describe('ensureObjectivesList', () => {
    describe('given the objectives list is missing', () => {
      it('creates an empty objectives list', () => {
        // Arrange
        const multiObjectiveOptions: NonNullable<
          NeatLikeWithObjectives['options']['multiObjective']
        > = { enabled: true };

        // Act
        const objectivesList = ensureObjectivesList(multiObjectiveOptions);

        // Assert
        expect(objectivesList).toEqual([]);
      });
    });

    describe('given the objectives list already exists', () => {
      it('returns the existing list reference', () => {
        // Arrange
        const existingObjectives = [createObjectiveDescriptor({ key: 'energy' })];
        const multiObjectiveOptions: NonNullable<
          NeatLikeWithObjectives['options']['multiObjective']
        > = {
          enabled: true,
          objectives: existingObjectives,
        };

        // Act
        const objectivesList = ensureObjectivesList(multiObjectiveOptions);

        // Assert
        expect(objectivesList).toBe(existingObjectives);
      });
    });
  });

  describe('replaceObjectiveByKey', () => {
    describe('given one objective already uses the incoming key', () => {
      it('replaces that objective and preserves the other keys', () => {
        // Arrange
        const originalObjectives = [
          createObjectiveDescriptor({ key: 'fitness' }),
          createObjectiveDescriptor({ key: 'novelty' }),
        ];

        // Act
        const updatedObjectiveKeys = replaceObjectiveByKey(
          originalObjectives,
          'fitness',
          'min',
          () => 1,
        ).map((objective) => objective.key);

        // Assert
        expect(updatedObjectiveKeys).toEqual(['novelty', 'fitness']);
      });
    });
  });
});