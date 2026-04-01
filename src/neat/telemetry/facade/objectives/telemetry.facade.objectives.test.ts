import {
  getObjectiveEvents,
  getObjectiveKeys,
  getObjectives,
} from './telemetry.facade.objectives';
import type { ObjectiveDescriptor } from '../../../shared/neat.shared.types';
import type { TelemetryFacadeObjectivesHost } from './telemetry.facade.objectives';

function createTelemetryObjectivesHost(
  objectivesList: ObjectiveDescriptor[],
  objectiveEvents?: Array<{ gen: number; type: 'add' | 'remove'; key: string }>,
): TelemetryFacadeObjectivesHost {
  return {
    options: {},
    _getObjectives: () => objectivesList,
    _objectiveEvents: objectiveEvents,
  } as TelemetryFacadeObjectivesHost;
}

describe('neat telemetry facade objectives chapter', () => {
  describe('getObjectiveKeys', () => {
    describe('given a resolved objective list with the default and one custom objective', () => {
      it('returns the stable key order without exposing descriptor internals', () => {
        // Arrange
        const objectivesHost = createTelemetryObjectivesHost([
          { key: 'fitness', direction: 'max', accessor: () => 0 },
          { key: 'complexity', direction: 'min', accessor: () => 0 },
        ]);

        // Act
        const objectiveKeys = getObjectiveKeys(objectivesHost);

        // Assert
        expect(objectiveKeys).toEqual(['fitness', 'complexity']);
      });
    });
  });

  describe('getObjectiveEvents', () => {
    describe('given the controller already persisted delayed objective additions', () => {
      it('returns the recorded add-event history in stable order', () => {
        // Arrange
        const objectivesHost = createTelemetryObjectivesHost(
          [],
          [
            { gen: 2, type: 'add', key: 'complexity' },
            { gen: 4, type: 'add', key: 'entropy' },
          ],
        );

        // Act
        const objectiveEvents = getObjectiveEvents(objectivesHost);

        // Assert
        expect(objectiveEvents).toEqual([
          { gen: 2, type: 'add', key: 'complexity' },
          { gen: 4, type: 'add', key: 'entropy' },
        ]);
      });
    });
  });

  describe('getObjectives', () => {
    describe('given the resolved objective registry includes a custom entropy descriptor', () => {
      it('returns only the compact key and direction summary in evaluation order', () => {
        // Arrange
        const objectivesHost = createTelemetryObjectivesHost([
          { key: 'fitness', direction: 'max', accessor: () => 0 },
          { key: 'entropy', direction: 'max', accessor: () => 1 },
        ]);

        // Act
        const objectiveSummaries = getObjectives(objectivesHost);

        // Assert
        expect(objectiveSummaries).toEqual([
          { key: 'fitness', direction: 'max' },
          { key: 'entropy', direction: 'max' },
        ]);
      });
    });
  });
});
