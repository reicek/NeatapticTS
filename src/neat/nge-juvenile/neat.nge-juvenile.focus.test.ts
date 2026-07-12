/**
 * Focused tests for the NGE juvenile focus scoring module.
 *
 * These tests exercise `resolveFocusConfig` and `computeFocusScores` from
 * `neat.nge-juvenile.focus.ts` in isolation, covering default resolution,
 * partial overrides, weighted scoring, softmax normalization, and edge
 * cases such as empty inputs and single-module slices.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */

import {
  NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
  NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD,
  NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
  NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE,
  NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW,
  NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
  NGE_JUVENILE_DEFAULT_NODE_ADDITION_COUNT,
  NGE_JUVENILE_DEFAULT_NODE_GROWTH_SIGNAL_FLOOR,
  NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR,
  NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
} from './neat.nge-juvenile.constants';
import {
  computeFocusScores,
  resolveFocusConfig,
} from './neat.nge-juvenile.focus';
import type {
  NgeFocusScore,
  NgeJuvenileFocusWeights,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
} from './neat.nge-juvenile.types';

/**
 * Build a metrics snapshot with all fields explicitly set.
 *
 * @param moduleId - Stable module identifier.
 * @param utilization - Relative module usage.
 * @param rewardDelta - Reward improvement.
 * @param novelty - Novel behavior contribution.
 * @param stabilityAge - Stable window count.
 * @param wiringCost - Structural wiring cost.
 * @returns A fully populated metrics snapshot.
 */
function createSnapshot(
  moduleId: string,
  utilization: number,
  rewardDelta: number,
  novelty: number,
  stabilityAge: number,
  wiringCost: number,
): NgeModuleMetricsSnapshot {
  return {
    moduleId,
    utilization,
    rewardDelta,
    novelty,
    stabilityAge,
    wiringCost,
  };
}

describe('nge juvenile focus module', () => {
  describe('resolveFocusConfig', () => {
    it('fills every default field when given an empty partial', () => {
      // Arrange & Act
      const config = resolveFocusConfig({});

      // Assert
      expect(config).toEqual({
        focusWeights: NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
        episodicHitRateThreshold:
          NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD,
        recurrentRefreshFloor: NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR,
        hysteresisWindowCount: NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
        cooldownWindowCount: NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
        gainStabilityWindow: NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW,
        gainStabilityTolerance: NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE,
        windowIndex: 0,
        nodeGrowthSignalFloor: NGE_JUVENILE_DEFAULT_NODE_GROWTH_SIGNAL_FLOOR,
        nodeAdditionCount: NGE_JUVENILE_DEFAULT_NODE_ADDITION_COUNT,
        edgeDensificationCount: NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
        maxStructuralEditsPerStep:
          NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
      });
    });

    it('overrides only the supplied hysteresisWindowCount and keeps defaults for the rest', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ hysteresisWindowCount: 7 });

      // Assert
      expect(config.hysteresisWindowCount).toBe(7);
      expect(config.cooldownWindowCount).toBe(
        NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
      );
    });

    it('overrides only the supplied cooldownWindowCount independently from hysteresisWindowCount', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ cooldownWindowCount: 9 });

      // Assert
      expect(config.cooldownWindowCount).toBe(9);
      expect(config.hysteresisWindowCount).toBe(
        NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
      );
    });

    it('merges partial focus weights with the defaults rather than replacing them', () => {
      // Arrange
      const customWeights: Partial<NgeJuvenileFocusWeights> = {
        w_u: 0.5,
        w_n: 0.01,
      };

      // Act
      const config = resolveFocusConfig({
        focusWeights: customWeights as NgeJuvenileFocusWeights,
      });

      // Assert
      expect(config.focusWeights).toEqual({
        ...NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
        ...customWeights,
      });
    });

    it('overrides windowIndex from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ windowIndex: 42 });

      // Assert
      expect(config.windowIndex).toBe(42);
    });

    it('overrides nodeAdditionCount from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ nodeAdditionCount: 3 });

      // Assert
      expect(config.nodeAdditionCount).toBe(3);
    });

    it('overrides edgeDensificationCount from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ edgeDensificationCount: 8 });

      // Assert
      expect(config.edgeDensificationCount).toBe(8);
    });

    it('overrides maxStructuralEditsPerStep from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ maxStructuralEditsPerStep: 12 });

      // Assert
      expect(config.maxStructuralEditsPerStep).toBe(12);
    });

    it('overrides gainStabilityWindow from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ gainStabilityWindow: 7 });

      // Assert
      expect(config.gainStabilityWindow).toBe(7);
    });

    it('overrides gainStabilityTolerance from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ gainStabilityTolerance: 0.1 });

      // Assert
      expect(config.gainStabilityTolerance).toBe(0.1);
    });

    it('overrides nodeGrowthSignalFloor from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ nodeGrowthSignalFloor: 0.25 });

      // Assert
      expect(config.nodeGrowthSignalFloor).toBe(0.25);
    });

    it('overrides recurrentRefreshFloor from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ recurrentRefreshFloor: 0.5 });

      // Assert
      expect(config.recurrentRefreshFloor).toBe(0.5);
    });

    it('overrides episodicHitRateThreshold from the partial config', () => {
      // Arrange & Act
      const config = resolveFocusConfig({ episodicHitRateThreshold: 0.8 });

      // Assert
      expect(config.episodicHitRateThreshold).toBe(0.8);
    });
  });

  describe('computeFocusScores', () => {
    it('returns an empty scores array for zero snapshots', () => {
      // Arrange & Act
      const vector = computeFocusScores([], {});

      // Assert
      expect(vector.scores).toEqual([]);
    });

    it('returns a single score of one for a single-module slice', () => {
      // Arrange
      const snapshots = [createSnapshot('solo', 2, 4, 8, 6, 1)];

      // Act
      const vector = computeFocusScores(snapshots, {});

      // Assert
      expect(vector.scores[0]?.normalizedScore).toBe(1);
    });

    it('produces softmax-normalized scores that sum to one', () => {
      // Arrange
      const snapshots = [
        createSnapshot('alpha', 0.8, 0.2, 0.1, 0.5, 0.3),
        createSnapshot('beta', 0.4, 0.1, 0.0, 0.9, 0.1),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});
      const sum = vector.scores.reduce(
        (total, { normalizedScore }) => total + normalizedScore,
        0,
      );

      // Assert
      expect(sum).toBeCloseTo(1, 10);
    });

    it('produces equal normalized scores for uniform snapshots', () => {
      // Arrange
      const snapshots = [
        createSnapshot('a', 5, 5, 5, 5, 5),
        createSnapshot('b', 5, 5, 5, 5, 5),
        createSnapshot('c', 5, 5, 5, 5, 5),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});

      // Assert
      expect(
        vector.scores.map(({ normalizedScore }) => normalizedScore),
      ).toEqual([1 / 3, 1 / 3, 1 / 3]);
    });

    it('ranks a higher-reward module above a higher-utilization module with default weights', () => {
      // Arrange
      const snapshots = [
        createSnapshot('reward', 0, 10, 0, 0, 0),
        createSnapshot('util', 10, 0, 0, 0, 0),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});

      // Assert
      expect(vector.scores[0]!.rawScore > vector.scores[1]!.rawScore).toBe(
        true,
      );
    });

    it('sets supportsGrowth to true when the raw score is positive', () => {
      // Arrange — high reward module with zero wiring cost yields a positive score
      const snapshots = [
        createSnapshot('grow', 1, 1, 1, 1, 0),
        createSnapshot('suppress', 0, 0, 0, 0, 1),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});

      // Assert
      expect(vector.scores[0]!.supportsGrowth).toBe(true);
    });

    it('sets supportsGrowth to false when the raw score is zero or negative', () => {
      // Arrange — costly module has all-zero positive metrics so it normalizes to 0
      // across every positive dimension, yielding rawScore = -w_c * 1.0 < 0.
      const snapshots = [
        createSnapshot('good', 10, 10, 10, 10, 0),
        createSnapshot('costly', 0, 0, 0, 0, 10),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});

      // Assert — the second module has normalized wiring cost 1.0, so its
      // raw score is negative (w_c * 1.0 subtracted), making supportsGrowth false.
      expect(vector.scores[1]!.supportsGrowth).toBe(false);
    });

    it('carries the windowIndex from the resolved config into the focus vector', () => {
      // Arrange
      const snapshots = [createSnapshot('a', 1, 1, 1, 1, 1)];
      const config: Partial<NgeJuvenilePhaseConfig> = { windowIndex: 7 };

      // Act
      const vector = computeFocusScores(snapshots, config);

      // Assert
      expect(vector.windowIndex).toBe(7);
    });

    it('produces deterministic score math across two calls with identical inputs', () => {
      // Arrange
      const snapshots = [
        createSnapshot('x', 0.8, 0.2, 0.1, 0.5, 0.3),
        createSnapshot('y', 0.4, 0.1, 0.0, 0.9, 0.1),
      ];

      // Act
      const first = computeFocusScores(snapshots, {});
      const second = computeFocusScores(snapshots, {});

      // Assert — raw and normalized scores must be identical across calls
      expect(
        second.scores.map(({ rawScore, normalizedScore }: NgeFocusScore) => ({
          rawScore,
          normalizedScore,
        })),
      ).toEqual(
        first.scores.map(({ rawScore, normalizedScore }: NgeFocusScore) => ({
          rawScore,
          normalizedScore,
        })),
      );
    });

    it('populates all five normalized metric fields in each focus score', () => {
      // Arrange
      const snapshots = [
        createSnapshot('full', 10, 8, 6, 4, 2),
        createSnapshot('empty', 0, 0, 0, 0, 0),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});
      const firstScore = vector.scores[0]!;

      // Assert — all five normalized metric fields are populated
      expect({
        hasUtilization: typeof firstScore.normalizedUtilization === 'number',
        hasRewardDelta: typeof firstScore.normalizedRewardDelta === 'number',
        hasNovelty: typeof firstScore.normalizedNovelty === 'number',
        hasStabilityAge: typeof firstScore.normalizedStabilityAge === 'number',
        hasWiringCost: typeof firstScore.normalizedWiringCost === 'number',
      }).toEqual({
        hasUtilization: true,
        hasRewardDelta: true,
        hasNovelty: true,
        hasStabilityAge: true,
        hasWiringCost: true,
      });
    });

    it('respects custom focus weights that emphasize utilization over reward', () => {
      // Arrange — emphasize utilization, suppress reward
      const snapshots = [
        createSnapshot('util-focused', 10, 0, 0, 0, 0),
        createSnapshot('reward-focused', 0, 10, 0, 0, 0),
      ];
      const config: Partial<NgeJuvenilePhaseConfig> = {
        focusWeights: { w_u: 0.5, w_r: 0.01, w_n: 0, w_s: 0, w_c: 0 },
      };

      // Act
      const vector = computeFocusScores(snapshots, config);

      // Assert — utilization module now scores higher than reward module
      expect(vector.scores[0]!.rawScore > vector.scores[1]!.rawScore).toBe(
        true,
      );
    });

    it('keeps the moduleId from each snapshot in the output score', () => {
      // Arrange
      const snapshots = [
        createSnapshot('module:alpha', 1, 2, 3, 4, 5),
        createSnapshot('module:beta', 5, 4, 3, 2, 1),
      ];

      // Act
      const vector = computeFocusScores(snapshots, {});

      // Assert
      expect(vector.scores.map(({ moduleId }) => moduleId)).toEqual([
        'module:alpha',
        'module:beta',
      ]);
    });
  });
});
