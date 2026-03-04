import Network from '../../../../../src/architecture/network';
import {
  FLAPPY_CENTER_BLUE_RAMP,
  FLAPPY_LIGHT_NEON_RAMP,
  FLAPPY_NEON_PALETTE,
  FLAPPY_REGULAR_NEON_RAMP,
} from '../../constants/constants';
import { clamp } from '../browser-entry.math.utils';
import type { ColorTier } from '../browser-entry.types';
import type {
  DynamicColorScale,
  NetworkVisualizationColorScales,
} from './visualization.types';

/**
 * Builds logarithmic diverging color tiers with a center band and edge extension.
 *
 * @param input - Tier creation options.
 * @returns Ordered tier list.
 */
export function createLogDivergingColorTiers(input: {
  maxAbsValue: number;
  centerBlueThreshold: number;
  negativePalette: readonly string[];
  centerBluePalette: readonly string[];
  positivePalette: readonly string[];
  logarithmicSteepness: number;
  edgeStartAbsValue?: number;
  edgeTierCount?: number;
}): ColorTier[] {
  const negativeMagnitudes = resolveTwoZoneMagnitudes({
    minimumMagnitude: input.centerBlueThreshold,
    maximumMagnitude: input.maxAbsValue,
    tierCount: input.negativePalette.length,
    logarithmicSteepness: input.logarithmicSteepness,
    edgeStartAbsValue: input.edgeStartAbsValue,
    edgeTierCount: input.edgeTierCount,
  });

  const positiveMagnitudes = resolveTwoZoneMagnitudes({
    minimumMagnitude: input.centerBlueThreshold,
    maximumMagnitude: input.maxAbsValue,
    tierCount: input.positivePalette.length,
    logarithmicSteepness: input.logarithmicSteepness,
    edgeStartAbsValue: input.edgeStartAbsValue,
    edgeTierCount: input.edgeTierCount,
  });

  const negativeTiers = input.negativePalette.map((color, colorIndex) => {
    const magnitude = negativeMagnitudes[colorIndex];
    return { upperBound: -magnitude, color };
  });

  const centerTiers = input.centerBluePalette.map((color, colorIndex) => {
    const linearProgress = (colorIndex + 1) / input.centerBluePalette.length;
    return {
      upperBound:
        -input.centerBlueThreshold +
        linearProgress * input.centerBlueThreshold * 2,
      color,
    };
  });

  const positiveTiers = input.positivePalette.map((color, colorIndex) => {
    const magnitude = positiveMagnitudes[colorIndex];
    return { upperBound: magnitude, color };
  });

  return [...negativeTiers, ...centerTiers, ...positiveTiers].toSorted(
    (leftTier, rightTier) => leftTier.upperBound - rightTier.upperBound,
  );
}

/**
 * Resolves a color from ordered tier definitions.
 *
 * @param value - Numeric value to classify.
 * @param tiers - Ordered tier list.
 * @param aboveTierColor - Fallback color for values above the last tier.
 * @returns Resolved color string.
 */
export function resolveTierColor(
  value: number,
  tiers: ColorTier[],
  aboveTierColor: string,
): string {
  const resolvedTier = tiers.find((tier) => value <= tier.upperBound);
  return resolvedTier?.color ?? aboveTierColor;
}

/**
 * Resolves connection color for a raw weight.
 *
 * @param connectionWeight - Connection weight.
 * @returns Tier color.
 */
export function resolveConnectionRangeColor(connectionWeight: number): string {
  const connectionScale = createDynamicColorScale([connectionWeight], {
    minimumValue: -1,
    maximumValue: 1,
  });
  return resolveTierColor(
    connectionWeight,
    connectionScale.tiers,
    connectionScale.aboveTierColor,
  );
}

/**
 * Resolves bias color for a raw node bias.
 *
 * @param nodeBias - Node bias.
 * @returns Tier color.
 */
export function resolveBiasRangeColor(nodeBias: number): string {
  const biasScale = createDynamicColorScale([nodeBias], {
    minimumValue: -1,
    maximumValue: 1,
  });
  return resolveTierColor(nodeBias, biasScale.tiers, biasScale.aboveTierColor);
}

/**
 * Resolves dynamic connection/bias color scales from the active network range.
 *
 * @param network - Active network.
 * @returns Dynamic scales used by graph drawing and legend rows.
 */
export function resolveNetworkVisualizationColorScales(
  network: Network | undefined,
): NetworkVisualizationColorScales {
  const connectionValues = ((network?.connections ?? []) as Array<{
    weight?: number;
  }>)
    .map((connection) => Number(connection.weight ?? 0))
    .filter((weight) => Number.isFinite(weight));

  const biasValues = ((network?.nodes ?? []) as Array<{
    type?: string;
    bias?: number;
  }>)
    .filter((node) => node.type !== 'output')
    .map((node) => Number(node.bias ?? 0))
    .filter((bias) => Number.isFinite(bias));

  return {
    connectionScale: createDynamicColorScale(connectionValues, {
      minimumValue: -1,
      maximumValue: 1,
    }),
    biasScale: createDynamicColorScale(biasValues, {
      minimumValue: -1,
      maximumValue: 1,
    }),
  };
}

function createDynamicColorScale(
  values: number[],
  fallbackRange: { minimumValue: number; maximumValue: number },
): DynamicColorScale {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  const observedMinimumValue =
    finiteValues.length > 0
      ? Math.min(...finiteValues)
      : fallbackRange.minimumValue;
  const observedMaximumValue =
    finiteValues.length > 0
      ? Math.max(...finiteValues)
      : fallbackRange.maximumValue;
  const hasRange = observedMaximumValue > observedMinimumValue;

  const minimumValue = hasRange
    ? observedMinimumValue
    : observedMinimumValue - Math.max(1e-6, Math.abs(observedMinimumValue) * 0.01);
  const maximumValue = hasRange
    ? observedMaximumValue
    : observedMaximumValue + Math.max(1e-6, Math.abs(observedMaximumValue) * 0.01);

  const dynamicTiers = resolveSignedDynamicTiers(minimumValue, maximumValue);
  const aboveTierColor =
    dynamicTiers.at(-1)?.color ?? FLAPPY_NEON_PALETTE.currentRunText;

  return {
    minimumValue,
    maximumValue,
    tiers: dynamicTiers,
    aboveTierColor,
  };
}

function resolveSignedDynamicTiers(
  minimumValue: number,
  maximumValue: number,
): ColorTier[] {
  const negativePaletteLowToHigh = [
    ...FLAPPY_LIGHT_NEON_RAMP.toReversed(),
    ...FLAPPY_CENTER_BLUE_RAMP,
  ] as const;
  const positivePaletteLowToHigh = [
    ...FLAPPY_CENTER_BLUE_RAMP,
    ...FLAPPY_REGULAR_NEON_RAMP,
  ] as const;

  if (minimumValue >= 0) {
    return createLinearColorTiers({
      minimumValue,
      maximumValue,
      palette: positivePaletteLowToHigh,
    });
  }

  if (maximumValue <= 0) {
    return createLinearColorTiers({
      minimumValue,
      maximumValue,
      palette: negativePaletteLowToHigh,
    });
  }

  const negativeTiers = createLinearColorTiers({
    minimumValue,
    maximumValue: 0,
    palette: negativePaletteLowToHigh,
  });
  const positiveTiers = createLinearColorTiers({
    minimumValue: 0,
    maximumValue,
    palette: positivePaletteLowToHigh,
  });
  return [...negativeTiers, ...positiveTiers];
}

function createLinearColorTiers(input: {
  minimumValue: number;
  maximumValue: number;
  palette: readonly string[];
}): ColorTier[] {
  const paletteSize = Math.max(1, input.palette.length);
  const range = Math.max(1e-12, input.maximumValue - input.minimumValue);
  const step = range / paletteSize;

  return input.palette.map((color, colorIndex) => ({
    upperBound:
      colorIndex === paletteSize - 1
        ? input.maximumValue
        : input.minimumValue + step * (colorIndex + 1),
    color,
  }));
}

function resolveTwoZoneMagnitudes(input: {
  minimumMagnitude: number;
  maximumMagnitude: number;
  tierCount: number;
  logarithmicSteepness: number;
  edgeStartAbsValue?: number;
  edgeTierCount?: number;
}): number[] {
  const safeTierCount = Math.max(1, input.tierCount);
  const targetEdgeStart = clamp(
    input.edgeStartAbsValue ?? input.maximumMagnitude,
    input.minimumMagnitude,
    input.maximumMagnitude,
  );
  const requestedEdgeTierCount = clamp(
    input.edgeTierCount ?? 0,
    0,
    safeTierCount,
  );
  const edgeTierCount =
    targetEdgeStart >= input.maximumMagnitude ? 0 : requestedEdgeTierCount;
  const nearTierCount = Math.max(1, safeTierCount - edgeTierCount);

  const nearMagnitudes = Array.from(
    { length: nearTierCount },
    (_unusedValue, tierIndex) => {
      const logarithmicProgress = mapLogarithmicProgress(
        tierIndex + 1,
        nearTierCount,
        input.logarithmicSteepness,
      );
      return (
        input.minimumMagnitude +
        (targetEdgeStart - input.minimumMagnitude) * logarithmicProgress
      );
    },
  );

  if (edgeTierCount === 0) {
    return nearMagnitudes;
  }

  const edgeMagnitudes = Array.from(
    { length: edgeTierCount },
    (_unusedValue, edgeIndex) => {
      const linearProgress = (edgeIndex + 1) / edgeTierCount;
      return (
        targetEdgeStart +
        (input.maximumMagnitude - targetEdgeStart) * linearProgress
      );
    },
  );

  return [...nearMagnitudes, ...edgeMagnitudes];
}

function mapLogarithmicProgress(
  position: number,
  totalPositions: number,
  logarithmicSteepness: number,
): number {
  const normalizedPosition = clamp(position / Math.max(1, totalPositions), 0, 1);
  return (
    Math.log1p(logarithmicSteepness * normalizedPosition) /
    Math.log1p(logarithmicSteepness)
  );
}
