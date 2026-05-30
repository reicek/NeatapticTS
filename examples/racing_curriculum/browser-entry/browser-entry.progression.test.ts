import {
  resolveTierPromotionFromLapCount,
  runDeterministicControllerProbe,
} from './browser-entry';
  describe('racing curriculum steering and tier progression', () => {
    describe('runDeterministicControllerProbe', () => {
      it('produces non-trivial steering activity and heading change over a deterministic horizon', () => {
        const probeResult = runDeterministicControllerProbe(240, 1);

        expect({
          hasSteeringActivity: probeResult.nonTrivialSteerSamples > 10,
          hasHeadingChange: probeResult.headingDeltaRadians > 0.2,
          meanAbsoluteSteer: Number(probeResult.meanAbsoluteSteer.toFixed(3)),
        }).toEqual({
          hasSteeringActivity: true,
          hasHeadingChange: true,
          meanAbsoluteSteer: expect.any(Number),
        });
      });
    });

    describe('resolveTierPromotionFromLapCount', () => {
      it('advances to the next tier after two laps and carries no extra lap count', () => {
        const promotionResult = resolveTierPromotionFromLapCount(1, 2);

        expect(promotionResult).toEqual({
          nextTier: 2,
          didAdvance: true,
          remainingLaps: 0,
        });
      });

      it('caps promotion at tier six when the curriculum is already at the top tier', () => {
        const promotionResult = resolveTierPromotionFromLapCount(6, 4);

        expect(promotionResult).toEqual({
          nextTier: 6,
          didAdvance: false,
          remainingLaps: 4,
        });
      });
    });
  });
