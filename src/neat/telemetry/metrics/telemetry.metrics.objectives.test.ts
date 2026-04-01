import {
  applyObjectiveAges,
  applyObjectiveEvents,
  applyObjectiveImportance,
  applyObjectivesSnapshot,
} from './telemetry.metrics.objectives';
import type { TelemetryEntryRecord } from '../types/telemetry.types';

function createTelemetryEntryRecord(): TelemetryEntryRecord {
  return {
    gen: 4,
    best: 9,
    species: 2,
    hyper: 0,
    ops: [],
    objImportance: {},
  };
}

describe('neat telemetry metrics objectives chapter', () => {
  describe('applyObjectiveImportance', () => {
    describe('given an evolve-side importance snapshot already exists', () => {
      it('copies the latest objective importance map onto the telemetry entry', () => {
        // Arrange
        const telemetryContext = {
          _lastObjImportance: {
            fitness: { range: 2, var: 1 },
          },
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectiveImportance(telemetryContext, telemetryEntry);

        // Assert
        expect(telemetryEntry.objImportance).toEqual({
          fitness: { range: 2, var: 1 },
        });
      });
    });
  });

  describe('applyObjectiveAges', () => {
    describe('given objective ages are tracked in controller state', () => {
      it('serializes the age map into the telemetry entry payload', () => {
        // Arrange
        const telemetryContext = {
          _objectiveAges: new Map<string, number>([
            ['fitness', 7],
            ['complexity', 3],
            ['entropy', 1],
          ]),
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectiveAges(telemetryContext, telemetryEntry);

        // Assert
        expect(telemetryEntry.objAges).toEqual({
          fitness: 7,
          complexity: 3,
          entropy: 1,
        });
      });
    });
  });

  describe('applyObjectiveEvents', () => {
    describe('given delayed objective additions are still pending at telemetry record time', () => {
      it('persists the add events and clears the pending queues', () => {
        // Arrange
        const telemetryContext = {
          _pendingObjectiveAdds: ['complexity', 'entropy'],
          _pendingObjectiveRemoves: [],
          _objectiveEvents: [],
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectiveEvents(telemetryContext, telemetryEntry, 4);

        // Assert
        expect({
          entryEvents: telemetryEntry.objEvents,
          persistedEvents: telemetryContext._objectiveEvents,
          pendingAdds: telemetryContext._pendingObjectiveAdds,
          pendingRemoves: telemetryContext._pendingObjectiveRemoves,
        }).toEqual({
          entryEvents: [
            { gen: 4, type: 'add', key: 'complexity' },
            { gen: 4, type: 'add', key: 'entropy' },
          ],
          persistedEvents: [
            { gen: 4, type: 'add', key: 'complexity' },
            { gen: 4, type: 'add', key: 'entropy' },
          ],
          pendingAdds: [],
          pendingRemoves: [],
        });
      });
    });
  });

  describe('applyObjectivesSnapshot', () => {
    describe('given an objective provider exposes the active objective list', () => {
      it('records the active objective keys onto the telemetry entry', () => {
        // Arrange
        const telemetryContext = {
          _getObjectives: () => [{ key: 'fitness' }, { key: 'entropy' }],
        };
        const telemetryEntry = createTelemetryEntryRecord();

        // Act
        applyObjectivesSnapshot(telemetryContext, telemetryEntry);

        // Assert
        expect(telemetryEntry.objectives).toEqual(['fitness', 'entropy']);
      });
    });
  });
});
