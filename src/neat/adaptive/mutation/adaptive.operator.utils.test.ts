import { OPERATOR_DECAY_DEFAULT } from '../core/adaptive.core.constants';
import {
  applyOperatorDecay,
  collectOperatorStatsEntries,
  decayOperatorStat,
  resolveOperatorDecay,
} from './adaptive.operator.utils';

describe('neat adaptive operator utilities chapter', () => {
  describe('resolveOperatorDecay', () => {
    describe('given the operator adaptation config omits a decay override', () => {
      it('returns the shared default decay factor', () => {
        // Arrange
        const config = {};

        // Act
        const decay = resolveOperatorDecay(config);

        // Assert
        expect(decay).toBe(OPERATOR_DECAY_DEFAULT);
      });
    });

    describe('given the operator adaptation config provides an explicit decay override', () => {
      it('returns the configured decay factor', () => {
        // Arrange
        const config = { decay: 0.4 };

        // Act
        const decay = resolveOperatorDecay(config);

        // Assert
        expect(decay).toBe(0.4);
      });
    });
  });

  describe('collectOperatorStatsEntries', () => {
    describe('given the operator stats map contains two operators', () => {
      it('returns a stable snapshot of the current entries', () => {
        // Arrange
        const stats = new Map<string, { success: number; attempts: number }>([
          ['add-node', { attempts: 8, success: 4 }],
          ['add-connection', { attempts: 6, success: 3 }],
        ]);

        // Act
        const entries = collectOperatorStatsEntries(stats);

        // Assert
        expect(entries).toEqual([
          ['add-node', { attempts: 8, success: 4 }],
          ['add-connection', { attempts: 6, success: 3 }],
        ]);
      });
    });
  });

  describe('decayOperatorStat', () => {
    describe('given one operator statistic record is decayed by half', () => {
      it('returns both success and attempt counts scaled by the same factor', () => {
        // Arrange
        const operatorStat = { attempts: 12, success: 5 };

        // Act
        const nextStat = decayOperatorStat(operatorStat, 0.5);

        // Assert
        expect(nextStat).toEqual({ attempts: 6, success: 2.5 });
      });
    });
  });

  describe('applyOperatorDecay', () => {
    describe('given the stats map receives one pre-collected operator snapshot', () => {
      it('writes the decayed stats back to the map by operator id', () => {
        // Arrange
        const stats = new Map<string, { success: number; attempts: number }>([
          ['rewire', { attempts: 8, success: 6 }],
        ]);
        const entries = [['rewire', { attempts: 8, success: 6 }]] satisfies Array<
          [string, { success: number; attempts: number }]
        >;

        // Act
        applyOperatorDecay(stats, entries, 0.25);

        // Assert
        expect(stats.get('rewire')).toEqual({ attempts: 2, success: 1.5 });
      });
    });
  });
});