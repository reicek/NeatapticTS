import {
  computeOperatorStatsSnapshot,
  readOperatorStats,
} from './telemetry.metrics.operator';

describe('neat telemetry operator metrics chapter', () => {
  describe('computeOperatorStatsSnapshot', () => {
    describe('given the recorder stores operator outcomes in a map', () => {
      it('projects the map into telemetry-friendly operator rows', () => {
        // Arrange
        const operatorStats = new Map([
          [
            'ADD_NODE',
            {
              success: 2,
              attempts: 5,
            },
          ],
        ]);

        // Act
        const operatorSnapshot = computeOperatorStatsSnapshot(operatorStats);

        // Assert
        expect(operatorSnapshot).toEqual([
          {
            op: 'ADD_NODE',
            succ: 2,
            att: 5,
          },
        ]);
      });
    });

    describe('given the recorder has not stored operator outcomes yet', () => {
      it('returns an empty telemetry snapshot array', () => {
        // Arrange
        const operatorStats = undefined;

        // Act
        const operatorSnapshot = computeOperatorStatsSnapshot(operatorStats);

        // Assert
        expect(operatorSnapshot).toEqual([]);
      });
    });
  });

  describe('readOperatorStats', () => {
    describe('given the host has recorded operator outcomes', () => {
      it('projects them into the public accessor shape', () => {
        // Arrange
        const operatorStats = new Map([
          [
            'ADD_CONN',
            {
              success: 3,
              attempts: 4,
            },
          ],
        ]);

        // Act
        const operatorSnapshot = readOperatorStats(operatorStats);

        // Assert
        expect(operatorSnapshot).toEqual([
          {
            name: 'ADD_CONN',
            success: 3,
            attempts: 4,
          },
        ]);
      });
    });

    describe('given the host has not recorded operator stats yet', () => {
      it('returns an empty public accessor snapshot', () => {
        // Arrange
        const operatorStats = undefined;

        // Act
        const operatorSnapshot = readOperatorStats(operatorStats);

        // Assert
        expect(operatorSnapshot).toEqual([]);
      });
    });
  });
});
