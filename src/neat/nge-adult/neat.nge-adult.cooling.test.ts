import { resolveGrowthCoolingDecision } from './neat.nge-adult.cooling';

describe('nge adult cooling', () => {
  describe('resolveGrowthCoolingDecision', () => {
    describe('given a growth cooling factor of zero', () => {
      it('returns growthCoolingActive false because no residual growth budget remains', () => {
        // Act
        const coolingDecision = resolveGrowthCoolingDecision(
          0.82,
          /* growthCoolingFactor= */ 0,
          /* focusFloor= */ 0.6,
        );

        // Assert
        expect(coolingDecision.growthCoolingActive).toBe(false);
      });
    });

    describe('given an adult focus score below the focus floor', () => {
      it('returns growthCoolingActive false because growth is fully suppressed', () => {
        // Act
        const coolingDecision = resolveGrowthCoolingDecision(
          0.59,
          /* growthCoolingFactor= */ 0.1,
          /* focusFloor= */ 0.6,
        );

        // Assert
        expect(coolingDecision.growthCoolingActive).toBe(false);
      });
    });
  });
});
