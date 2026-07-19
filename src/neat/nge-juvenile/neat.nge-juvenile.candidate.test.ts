/**
 * Red-phase test contracts for the NGE candidate forward-pass scoring primitives.
 *
 * These tests define the expected domain-agnostic contract for
 * `src/neat/nge-juvenile/neat.nge-juvenile.candidate.ts` before the implementation
 * exists. The primitives extracted from the racing demo are:
 *
 * - `collectForwardPassOutputs` — runs forward passes on sample inputs and collects
 *   output vectors, parameterized by an `NgeObservationEncoder` so the core stays
 *   free of domain-specific observation shapes.
 * - `buildCandidateScoreWindow` — builds a sliding window of recent scores for
 *   plateau detection and improvement comparison, with a config-driven window size.
 * - `resolveSampleIndices` — selects a deterministic or random subset of sample
 *   indices for forward-pass evaluation, with a config-driven maximum sample count.
 *
 * All tests fail because `./neat.nge-juvenile.candidate` does not exist yet. The
 * failure reason is "missing implementation" (import error), not syntax error or
 * bad fixture.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */

import { readFileSync } from 'node:fs';
import Network from '../../architecture/network';
import {
  buildCandidateScoreWindow,
  collectForwardPassOutputs,
  resolveSampleIndices,
  type NgeObservationEncoder,
} from './neat.nge-juvenile.candidate';

/**
 * Simple domain-specific observation used to prove the encoder boundary.
 * It is intentionally not a racing type — any domain object works.
 */
interface FakeObservation {
  readonly value: number;
}

/**
 * Domain-agnostic mock encoder: converts the fake observation into a repeated
 * scalar vector of the requested input size.
 */
const fakeEncoder: NgeObservationEncoder<FakeObservation> = {
  encode(observation: FakeObservation, inputSize: number) {
    return Array.from({ length: inputSize }, () => observation.value);
  },
};

/**
 * Deterministic seeded random that always returns the same value so sample
 * selection tests are stable and reproducible.
 */
const constantRandom =
  (value = 0.5) =>
  () =>
    value;

describe('NGE candidate forward-pass scoring primitives', () => {
  describe('module exports', () => {
    it('exports collectForwardPassOutputs as a function', () => {
      // Assert
      expect(typeof collectForwardPassOutputs).toBe('function');
    });

    it('exports buildCandidateScoreWindow as a function', () => {
      // Assert
      expect(typeof buildCandidateScoreWindow).toBe('function');
    });

    it('exports resolveSampleIndices as a function', () => {
      // Assert
      expect(typeof resolveSampleIndices).toBe('function');
    });

    it('does not import racing-specific types in the source file', () => {
      // Arrange — tests run from the repo root, so resolve the sibling source file
      const sourcePath = 'src/neat/nge-juvenile/neat.nge-juvenile.candidate.ts';
      const source = readFileSync(sourcePath, 'utf-8');

      // Assert
      expect({
        hasRacingSignal: source.includes('RacingQualitySignal'),
        hasRacingImport: source.includes('examples/racing'),
      }).toEqual({
        hasRacingSignal: false,
        hasRacingImport: false,
      });
    });
  });

  describe('collectForwardPassOutputs', () => {
    it('returns one output vector per input observation', () => {
      // Arrange — 2-input, 1-output network activated on three samples
      const network = new Network(2, 1, { seed: 42 });
      const observations: readonly FakeObservation[] = [
        { value: 0.1 },
        { value: 0.5 },
        { value: 0.9 },
      ];

      // Act
      const outputs = collectForwardPassOutputs(
        network,
        observations,
        fakeEncoder,
      );

      // Assert
      expect(outputs.length).toBe(observations.length);
    });

    it('uses the encoder to turn each observation into the network input size', () => {
      // Arrange — 3-input network and a capturing encoder
      const network = new Network(3, 1, { seed: 42 });
      const observations: readonly FakeObservation[] = [{ value: 2.0 }];
      const capturedSizes: number[] = [];
      const sizingEncoder: NgeObservationEncoder<FakeObservation> = {
        encode(observation: FakeObservation, inputSize: number) {
          capturedSizes.push(inputSize);
          return Array.from({ length: inputSize }, () => observation.value);
        },
      };

      // Act
      collectForwardPassOutputs(network, observations, sizingEncoder);

      // Assert
      expect(capturedSizes).toEqual([3]);
    });

    it('returns empty outputs when the network has no input nodes', () => {
      // Arrange — mock network with zero input nodes
      const network = {
        input: 0,
        activate: () => [0],
      } as unknown as Network;
      const observations: readonly FakeObservation[] = [{ value: 1 }];

      // Act
      const outputs = collectForwardPassOutputs(
        network,
        observations,
        fakeEncoder,
      );

      // Assert
      expect(outputs).toEqual([]);
    });
  });

  describe('buildCandidateScoreWindow', () => {
    it('returns the last windowSize entries from the score history', () => {
      // Arrange
      const scoreHistory = [1, 2, 3, 4, 5, 6, 7];
      const config = { windowSize: 3 };

      // Act
      const window = buildCandidateScoreWindow(scoreHistory, config);

      // Assert
      expect(window).toEqual([5, 6, 7]);
    });

    it('returns the full history when it is shorter than the window size', () => {
      // Arrange
      const scoreHistory = [10, 20];
      const config = { windowSize: 5 };

      // Act
      const window = buildCandidateScoreWindow(scoreHistory, config);

      // Assert
      expect(window).toEqual([10, 20]);
    });

    it('uses the default window size when config omits windowSize', () => {
      // Arrange
      const scoreHistory = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const config = {};

      // Act
      const window = buildCandidateScoreWindow(scoreHistory, config);

      // Assert — default contract from the grow-stabilize constants is 5
      expect(window.length).toBe(5);
    });

    it('uses default config when config argument is omitted', () => {
      // Arrange — 10-entry history; default window size is 5
      const scoreHistory = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

      // Act — call with only the scoreHistory argument to exercise the
      // `config = {}` default parameter branch
      const window = buildCandidateScoreWindow(scoreHistory);

      // Assert — default contract from the grow-stabilize constants is 5
      expect(window.length).toBe(5);
    });
  });

  describe('resolveSampleIndices', () => {
    it('returns all indices when history length is less than or equal to maxSamples', () => {
      // Arrange
      const totalSamples = 3;
      const config = { maxSamples: 5 };

      // Act
      const indices = resolveSampleIndices(
        totalSamples,
        config,
        constantRandom(),
      );

      // Assert
      expect(indices).toEqual([0, 1, 2]);
    });

    it('returns exactly maxSamples indices when history exceeds maxSamples', () => {
      // Arrange
      const totalSamples = 20;
      const config = { maxSamples: 4 };

      // Act
      const indices = resolveSampleIndices(
        totalSamples,
        config,
        constantRandom(),
      );

      // Assert
      expect(indices.length).toBe(4);
    });

    it('uses the default maxSamples when config omits maxSamples', () => {
      // Arrange
      const totalSamples = 10;
      const config = {};

      // Act
      const indices = resolveSampleIndices(
        totalSamples,
        config,
        constantRandom(),
      );

      // Assert — default contract is NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES (5)
      expect(indices.length).toBe(5);
    });

    it('deterministically spreads indices across the full history', () => {
      // Arrange — 8 entries, max 3 samples
      const totalSamples = 8;
      const config = { maxSamples: 3 };

      // Act
      const indices = resolveSampleIndices(
        totalSamples,
        config,
        constantRandom(),
      );

      // Assert — evenly spaced deterministic indices
      expect(indices).toEqual([0, 2, 5]);
    });

    it('works without a random argument', () => {
      // Arrange
      const totalSamples = 3;
      const config = { maxSamples: 5 };

      // Act — call with only 2 arguments to exercise the optional
      // `_random` parameter's undefined branch
      const indices = resolveSampleIndices(totalSamples, config);

      // Assert — all 3 indices returned since totalSamples ≤ maxSamples
      expect(indices).toEqual([0, 1, 2]);
    });

    it('uses default config when config argument is omitted entirely', () => {
      // Arrange — 10 entries; default maxSamples is 5
      const totalSamples = 10;

      // Act — call with only 1 argument to exercise the
      // `config = {}` default parameter branch on resolveSampleIndices
      const indices = resolveSampleIndices(totalSamples);

      // Assert — default contract is NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES (5)
      expect(indices.length).toBe(5);
    });
  });
});
