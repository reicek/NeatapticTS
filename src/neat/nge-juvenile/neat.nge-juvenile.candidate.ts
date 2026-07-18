import type Network from '../../architecture/network';
import {
  NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES,
  NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE,
} from './neat.nge-juvenile.constants';
import type {
  NgeCandidateScoringConfig,
  NgeObservationEncoder,
} from './neat.nge-juvenile.types';

export type { NgeCandidateScoringConfig, NgeObservationEncoder };

/**
 * Collect forward-pass output vectors from a network for a set of observations.
 *
 * Each observation is encoded through the caller-supplied {@link NgeObservationEncoder}
 * to match the network's input size, then activated. The returned array preserves
 * the observation order so downstream scorers can compare behavioral variance or
 * reduce each output to a scalar quality score.
 *
 * @typeParam T - Application-specific observation type accepted by the encoder.
 * @param network - Live network to activate. Must expose `input` and `activate`.
 * @param observations - Observations drawn from the evidence window.
 * @param encoder - Domain-agnostic encoder that produces an input vector per observation.
 * @returns Array of output vectors, one per observation.
 */
export function collectForwardPassOutputs<T>(
  network: Network,
  observations: readonly T[],
  encoder: NgeObservationEncoder<T>,
): number[][] {
  const inputSize = network.input;
  if (inputSize <= 0) {
    return [];
  }

  return observations.map((observation) => {
    const inputVector = encoder.encode(observation, inputSize);
    return [...network.activate(inputVector)];
  });
}

/**
 * Build a sliding window of the most recent scores from a candidate score history.
 *
 * The window size is config-driven and defaults to
 * {@link NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE}. When the history is shorter
 * than the requested window, the full history is returned without padding.
 *
 * @param scoreHistory - Ordered numeric score history.
 * @param config - Optional window size override.
 * @returns The last `windowSize` entries, or the full history if shorter.
 */
export function buildCandidateScoreWindow(
  scoreHistory: readonly number[],
  config: Partial<NgeCandidateScoringConfig> = {},
): number[] {
  const windowSize =
    config.windowSize ?? NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE;
  const size = Math.min(windowSize, scoreHistory.length);

  return scoreHistory.slice(-size);
}

/**
 * Resolve a deterministic, evenly-spaced subset of sample indices.
 *
 * When `totalSamples` fits within the configured maximum, every index is
 * returned in order. Otherwise the indices are spread across the full range
 * using integer floor division so repeated calls with the same parameters
 * produce the same sample set. The `random` argument is part of the public
 * surface for future randomized sampling strategies but is intentionally unused
 * by the current deterministic spread.
 *
 * @param totalSamples - Number of observations available.
 * @param config - Optional maximum sample count override.
 * @param _random - Deterministic random source reserved for future strategies.
 * @returns Array of selected indices in ascending order.
 */
export function resolveSampleIndices(
  totalSamples: number,
  config: Partial<NgeCandidateScoringConfig> = {},
  _random?: () => number,
): number[] {
  void _random;

  const maxSamples =
    config.maxSamples ?? NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES;
  const cappedSamples = Math.min(maxSamples, totalSamples);

  if (totalSamples <= maxSamples) {
    return Array.from({ length: totalSamples }, (_, index) => index);
  }

  const indices: number[] = [];
  for (let sampleIndex = 0; sampleIndex < cappedSamples; sampleIndex++) {
    indices.push(Math.floor((sampleIndex * totalSamples) / cappedSamples));
  }

  return indices;
}
