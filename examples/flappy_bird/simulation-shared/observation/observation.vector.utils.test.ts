import {
  resolveCoreObservationVectorFromFeatures,
  resolveObservationVectorFromFeatures,
} from '../../flappy.simulation.shared.utils';
import type { SharedObservationFeatures } from '../simulation-shared.types';

const SAMPLE_OBSERVATION_FEATURES: SharedObservationFeatures = {
  normalizedBirdY: 0.1,
  normalizedVelocity: -0.2,
  normalizedDistanceToNextPipe: 0.3,
  normalizedDeltaToNextGap: -0.4,
  normalizedNextGapTop: 0.5,
  normalizedNextGapBottom: 0.6,
  normalizedDistanceToPipeEntrance: 0.25,
  normalizedNextGapClearance: -0.1,
  normalizedDeltaToSecondGap: -0.3,
  normalizedTimeToNextPipe: 0.9,
  normalizedRequiredVerticalVelocityToNextGap: 0.2,
  normalizedFramesToGapEntry: 0.4,
  normalizedFramesToGapExit: 0.5,
  normalizedRequiredVerticalVelocityAtGapEntry: -0.6,
  normalizedRequiredVerticalVelocityAtGapExit: 0.7,
  normalizedEntryUrgency: 0.8,
  normalizedOneFlapReachabilityAtGapEntry: 0.9,
};

describe('resolveObservationVectorFromFeatures', () => {
  it('keeps bird-state, next-gap, and look-ahead channels in the public controller input', () => {
    expect(
      resolveObservationVectorFromFeatures(SAMPLE_OBSERVATION_FEATURES),
    ).toEqual([0.1, -0.2, 0.3, -0.4, 0.5, 0.6, 0.25, -0.1, -0.3]);
  });
});

describe('resolveCoreObservationVectorFromFeatures', () => {
  it('matches the live controller input ordering so compatibility bookkeeping cannot drift', () => {
    expect(
      resolveCoreObservationVectorFromFeatures(SAMPLE_OBSERVATION_FEATURES),
    ).toEqual(
      resolveObservationVectorFromFeatures(SAMPLE_OBSERVATION_FEATURES),
    );
  });
});
