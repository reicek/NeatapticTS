import type { NeatGenomeSubstrateCoordinate } from '../genome/genome.types';

import { NGE_DNA_SubstrateError } from './neat.nge-dna.errors';
import { canonicalSerialize, computeFingerprint } from './neat.nge-dna.utils';
import type {
  NgeSubstrateConfig,
  NgeSubstrateZone,
  NgeZonePartitionConfig,
} from './neat.nge-dna.types';

const SUBSTRATE_AXES = ['x', 'y', 'z'] as const;

type SubstrateAxis = (typeof SUBSTRATE_AXES)[number];

/**
 * Clamp one raw substrate coordinate into the unit cube range [0, 1].
 *
 * @param raw - Raw three-axis coordinate to normalize.
 * @returns Clamped coordinate whose axes stay within `[0, 1]`.
 * @throws NGE_DNA_SubstrateError When any axis is `NaN` or infinite.
 */
export function normalizeCoordinate(
  raw: NeatGenomeSubstrateCoordinate,
): NeatGenomeSubstrateCoordinate {
  return raw.map((coordinateValue, axisIndex) => {
    const axisLabel = SUBSTRATE_AXES[axisIndex];

    return normalizeAxisCoordinate(coordinateValue, axisLabel);
  }) as NeatGenomeSubstrateCoordinate;
}

/**
 * Resolve one deterministic zone id for a normalized substrate coordinate.
 *
 * @param coord - Raw or normalized three-axis coordinate.
 * @param partition - Per-axis partition configuration for the substrate grid.
 * @returns Deterministic zone identifier of the form `z:x:y:z`.
 */
export function assignZone(
  coord: NeatGenomeSubstrateCoordinate,
  partition: NgeZonePartitionConfig,
): string {
  validateZonePartition(partition);

  const [normalizedX, normalizedY, normalizedZ] = normalizeCoordinate(coord);

  return [
    resolveZoneIndex(normalizedX, partition.x.count),
    resolveZoneIndex(normalizedY, partition.y.count),
    resolveZoneIndex(normalizedZ, partition.z.count),
  ]
    .join(':')
    .replace(/^/, 'z:');
}

/**
 * Build the full deterministic zone map for one unit-cube substrate partition.
 *
 * @param partition - Per-axis partition configuration for the substrate grid.
 * @returns Map from zone id to resolved zone descriptor.
 * @throws NGE_DNA_SubstrateError When any partition count is not a positive integer.
 */
export function buildZoneMap(
  partition: NgeZonePartitionConfig,
): Map<string, NgeSubstrateZone> {
  validateZonePartition(partition);

  const zoneMap = new Map<string, NgeSubstrateZone>();

  for (let xIndex = 0; xIndex < partition.x.count; xIndex += 1) {
    for (let yIndex = 0; yIndex < partition.y.count; yIndex += 1) {
      for (let zIndex = 0; zIndex < partition.z.count; zIndex += 1) {
        const zoneId = `z:${xIndex}:${yIndex}:${zIndex}`;

        zoneMap.set(zoneId, {
          zoneId,
          bounds: {
            xMin: xIndex / partition.x.count,
            xMax: (xIndex + 1) / partition.x.count,
            yMin: yIndex / partition.y.count,
            yMax: (yIndex + 1) / partition.y.count,
            zMin: zIndex / partition.z.count,
            zMax: (zIndex + 1) / partition.z.count,
          },
        });
      }
    }
  }

  return zoneMap;
}

/**
 * Compute the deterministic SHA-256 fingerprint of one canonical substrate configuration.
 *
 * @param config - Canonical substrate configuration to fingerprint.
 * @returns SHA-256 fingerprint of the canonical substrate JSON.
 */
export function buildSubstrateFingerprint(config: NgeSubstrateConfig): string {
  validateZonePartition(config.zonePartition);
  return computeFingerprint(canonicalSerialize(config));
}

function normalizeAxisCoordinate(
  coordinateValue: number,
  axisLabel: SubstrateAxis,
): number {
  if (!Number.isFinite(coordinateValue)) {
    throw new NGE_DNA_SubstrateError(
      `NGE_DNA substrate coordinate on axis ${axisLabel} must be finite.`,
    );
  }

  return Math.min(1, Math.max(0, coordinateValue));
}

function validateZonePartition(partition: NgeZonePartitionConfig): void {
  SUBSTRATE_AXES.forEach((axisLabel) => {
    validateAxisPartitionCount(
      axisLabel,
      partition[axisLabel]?.count as number | undefined,
    );
  });
}

function validateAxisPartitionCount(
  axisLabel: SubstrateAxis,
  partitionCount: number | undefined,
): void {
  if (!Number.isInteger(partitionCount) || Number(partitionCount) <= 0) {
    throw new NGE_DNA_SubstrateError(
      `NGE_DNA substrate axis ${axisLabel} requires one positive integer partition count.`,
    );
  }
}

function resolveZoneIndex(
  coordinateValue: number,
  partitionCount: number,
): number {
  return Math.min(
    Math.floor(coordinateValue * partitionCount),
    partitionCount - 1,
  );
}
