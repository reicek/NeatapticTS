import {
  NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD,
  NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
  NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE,
  NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW,
  NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
  NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR,
} from './neat.nge-juvenile.constants';
import { minMaxNormalize } from './neat.nge-juvenile.utils';
import type {
  NgeFocusScore,
  NgeFocusVector,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
} from './neat.nge-juvenile.types';

/**
 * Resolve a partial juvenile focus config against the seeded plan defaults.
 *
 * @param partial - Partial config whose omitted fields should resolve conservatively.
 * @returns A fully resolved config packet ready for deterministic focus scoring.
 */
export function resolveFocusConfig(
  partial: Partial<NgeJuvenilePhaseConfig>,
): NgeJuvenilePhaseConfig {
  return {
    focusWeights: {
      ...NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
      ...partial.focusWeights,
    },
    episodicHitRateThreshold:
      partial.episodicHitRateThreshold ??
      NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD,
    recurrentRefreshFloor:
      partial.recurrentRefreshFloor ??
      NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR,
    hysteresisWindowCount:
      partial.hysteresisWindowCount ??
      NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
    cooldownWindowCount:
      partial.cooldownWindowCount ??
      NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
    gainStabilityWindow:
      partial.gainStabilityWindow ?? NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW,
    gainStabilityTolerance:
      partial.gainStabilityTolerance ??
      NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE,
    windowIndex: partial.windowIndex ?? 0,
  };
}

/**
 * Compute the weighted juvenile focus vector for one evaluation window.
 *
 * The score math is deterministic for a fixed snapshot and config. The `computedAt`
 * field is metadata only and must not participate in any deterministic fingerprint.
 *
 * @param snapshots - Module metrics observed in the active evaluation slice.
 * @param config - Partial or fully resolved juvenile focus configuration.
 * @returns A focus vector carrying raw and normalized module scores.
 */
export function computeFocusScores(
  snapshots: readonly NgeModuleMetricsSnapshot[],
  config: Partial<NgeJuvenilePhaseConfig>,
): NgeFocusVector {
  const resolvedConfig = resolveFocusConfig(config);

  // Step 1: Normalize each metric column independently so mixed units stay comparable.
  const normalizedMetricColumns = buildNormalizedMetricColumns(snapshots);

  // Step 2: Fold the plan weights into raw scalar focus scores per module.
  const rawScores = snapshots.map((snapshot, snapshotIndex) =>
    buildFocusScore(
      snapshot,
      snapshotIndex,
      normalizedMetricColumns,
      resolvedConfig,
    ),
  );

  // Step 3: Normalize the raw focus vector into a probability-like allocation shelf.
  const normalizedScores = normalizeRawScores(
    rawScores.map(({ rawScore }) => rawScore),
  );

  return {
    scores: rawScores.map((rawScore, scoreIndex) => ({
      ...rawScore,
      normalizedScore: normalizedScores[scoreIndex],
    })),
    windowIndex: resolvedConfig.windowIndex,
    computedAt: Date.now(),
  };

  /**
   * @param metricsSnapshots - Raw module metrics collected for one evaluation window.
   * @returns Independently normalized metric columns aligned by snapshot index.
   */
  function buildNormalizedMetricColumns(
    metricsSnapshots: readonly NgeModuleMetricsSnapshot[],
  ): {
    utilization: number[];
    rewardDelta: number[];
    novelty: number[];
    stabilityAge: number[];
    wiringCost: number[];
  } {
    return {
      utilization: minMaxNormalize(
        metricsSnapshots.map(({ utilization }) => utilization),
      ),
      rewardDelta: minMaxNormalize(
        metricsSnapshots.map(({ rewardDelta }) => rewardDelta),
      ),
      novelty: minMaxNormalize(metricsSnapshots.map(({ novelty }) => novelty)),
      stabilityAge: minMaxNormalize(
        metricsSnapshots.map(({ stabilityAge }) => stabilityAge),
      ),
      wiringCost: minMaxNormalize(
        metricsSnapshots.map(({ wiringCost }) => wiringCost),
      ),
    };
  }

  /**
   * @param snapshot - One module metrics snapshot.
   * @param snapshotIndex - Input-order index shared by every normalized metric column.
   * @param normalizedColumns - Independently normalized metrics aligned by index.
   * @param phaseConfig - Fully resolved juvenile focus config.
   * @returns The raw weighted focus score for one module.
   */
  function buildFocusScore(
    snapshot: NgeModuleMetricsSnapshot,
    snapshotIndex: number,
    normalizedColumns: {
      utilization: number[];
      rewardDelta: number[];
      novelty: number[];
      stabilityAge: number[];
      wiringCost: number[];
    },
    phaseConfig: NgeJuvenilePhaseConfig,
  ): NgeFocusScore {
    const { focusWeights } = phaseConfig;
    const rawScore =
      focusWeights.w_u * normalizedColumns.utilization[snapshotIndex] +
      focusWeights.w_r * normalizedColumns.rewardDelta[snapshotIndex] +
      focusWeights.w_n * normalizedColumns.novelty[snapshotIndex] +
      focusWeights.w_s * normalizedColumns.stabilityAge[snapshotIndex] -
      focusWeights.w_c * normalizedColumns.wiringCost[snapshotIndex];

    return {
      moduleId: snapshot.moduleId,
      rawScore,
      normalizedScore: 0,
    };
  }

  /**
   * @param rawScores - Weighted scalar focus scores.
   * @returns Softmax-normalized scores that sum to one when the vector is non-empty.
   */
  function normalizeRawScores(rawScores: readonly number[]): number[] {
    const highestRawScore = rawScores.reduce(
      (currentHighestScore, rawScore) =>
        Math.max(currentHighestScore, rawScore),
      Number.NEGATIVE_INFINITY,
    );
    const exponentiatedScores = rawScores.map((rawScore) =>
      Math.exp(rawScore - highestRawScore),
    );
    const totalExponentiatedScore = exponentiatedScores.reduce(
      (currentTotal, exponentiatedScore) => currentTotal + exponentiatedScore,
      0,
    );

    return exponentiatedScores.map(
      (exponentiatedScore) => exponentiatedScore / totalExponentiatedScore,
    );
  }
}
