import {
  createCurriculumEpisodeState,
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
    it('advances to the next tier after three laps and carries no extra lap count', () => {
      const promotionResult = resolveTierPromotionFromLapCount(1, 3);

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

    it('holds promotion at Tier 5 when cross-team fairness has not been confirmed yet', () => {
      const promotionResult = resolveTierPromotionFromLapCount(5, 3);

      expect(promotionResult.didAdvance).toBe(false);
    });
  });

  describe('createCurriculumEpisodeState', () => {
    it('starts Tier 1 with a two-car 1v1 pack so both team guiding lines render', () => {
      const episodeState = createCurriculumEpisodeState(1);

      expect(episodeState.envState.cars?.length ?? 0).toBe(2);
    });

    it('expands Tier 4 to the four-car grid on the large track bucket', () => {
      const episodeState = createCurriculumEpisodeState(4);

      expect({
        carCount: episodeState.envState.cars?.length ?? 0,
        sizeBucket: episodeState.trackSpec.sizeBucket,
      }).toEqual({
        carCount: 4,
        sizeBucket: 'large',
      });
    });

    it('expands Tier 5 to the six-car grid for the future co-evolution pack', () => {
      const episodeState = createCurriculumEpisodeState(5);

      expect(episodeState.envState.cars?.length ?? 0).toBe(6);
    });
  });
});
