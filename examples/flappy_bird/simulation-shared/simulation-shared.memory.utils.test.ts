import {
  commitSharedObservationMemoryStep,
  createSharedObservationMemoryState,
  resolveCoreObservationVectorFromFeatures,
  resolveTemporalObservationVector,
} from '../flappy.simulation.shared.utils';
import type { SharedObservationFeatures } from './simulation-shared.types';

const SAMPLE_OBSERVATION_FEATURES: SharedObservationFeatures = {
  normalizedBirdY: 0.1,
  normalizedVelocity: -0.2,
  normalizedDistanceToNextPipe: 0.3,
  normalizedDeltaToNextGap: -0.4,
  normalizedNextGapTop: 0.5,
  normalizedNextGapBottom: 0.6,
  normalizedDistanceToSecondPipe: 0.7,
  normalizedDeltaToSecondGap: -0.8,
  normalizedTimeToNextPipe: 0.9,
  normalizedNextGapClearance: -0.1,
  normalizedRequiredVerticalVelocityToNextGap: 0.2,
  normalizedNextToSecondGapTransition: -0.3,
  normalizedFramesToGapEntry: 0.4,
  normalizedFramesToGapExit: 0.5,
  normalizedRequiredVerticalVelocityAtGapEntry: -0.6,
  normalizedRequiredVerticalVelocityAtGapExit: 0.7,
  normalizedEntryUrgency: 0.8,
  normalizedOneFlapReachabilityAtGapEntry: 0.9,
};

describe('resolveTemporalObservationVector', () => {
  it('returns only the current core observation frame for every architecture', () => {
    const observationMemoryState = createSharedObservationMemoryState();

    observationMemoryState.previousCoreObservationFrames = [
      Array.from({ length: 12 }, () => 1),
      Array.from({ length: 12 }, () => 2),
    ];
    observationMemoryState.recentFlapActions = [1, 0, 1, 1];

    expect(
      resolveTemporalObservationVector(
        SAMPLE_OBSERVATION_FEATURES,
        observationMemoryState,
      ),
    ).toEqual(
      resolveCoreObservationVectorFromFeatures(SAMPLE_OBSERVATION_FEATURES),
    );
  });
});

describe('commitSharedObservationMemoryStep', () => {
  it('leaves external observation memory empty when hard-coded temporal inputs are disabled', () => {
    const observationMemoryState = createSharedObservationMemoryState();

    commitSharedObservationMemoryStep(
      observationMemoryState,
      SAMPLE_OBSERVATION_FEATURES,
      true,
    );

    expect(observationMemoryState).toEqual({
      previousCoreObservationFrames: [],
      recentFlapActions: [],
    });
  });
});