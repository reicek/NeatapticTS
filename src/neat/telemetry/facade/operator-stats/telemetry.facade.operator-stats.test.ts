import { getOperatorStats } from './telemetry.facade.operator-stats';
import type { TelemetryFacadeOperatorStatsHost } from './telemetry.facade.operator-stats';

function createOperatorStatsHost(): TelemetryFacadeOperatorStatsHost {
  return {
    _operatorStats: new Map([
      ['ADD_NODE', { success: 2, attempts: 5 }],
      ['SUB_CONN', { success: 1, attempts: 4 }],
    ]),
  };
}

describe('neat telemetry facade operator-stats chapter', () => {
  describe('getOperatorStats', () => {
    describe('given recorded operator counters on the host', () => {
      it('returns public summaries that preserve operator names and counts', () => {
        // Arrange
        const operatorStatsHost = createOperatorStatsHost();

        // Act
        const operatorStats = getOperatorStats(operatorStatsHost);

        // Assert
        expect(operatorStats).toEqual([
          { name: 'ADD_NODE', success: 2, attempts: 5 },
          { name: 'SUB_CONN', success: 1, attempts: 4 },
        ]);
      });
    });
  });
});
