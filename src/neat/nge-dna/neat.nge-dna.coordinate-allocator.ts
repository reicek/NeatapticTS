import type { NeatGenomeSubstrateCoordinate } from '../genome/genome.types';

import { assignZone } from './neat.nge-dna.substrate';
import type { NgeZonePartitionConfig } from './neat.nge-dna.types';
import { canonicalSerialize, computeFingerprint } from './neat.nge-dna.utils';

/**
 * Input contract for the deterministic per-enemy substrate coordinate allocator.
 *
 * The allocator produces a stable unit-cube coordinate from the tuple
 * `(swarmSize, enemyIndex, seed)` so that repeated builds with identical inputs
 * always resolve to the same coordinate and zone assignment. No runtime allocation
 * order is consulted, which keeps enemy placement reproducible across workers and
 * across sessions.
 */
export interface EnemyCoordinateAllocatorInput {
  /** Number of enemies in the swarm. Must be a positive integer. */
  swarmSize: number;
  /** Zero-based enemy index. Must be an integer in `[0, swarmSize)`. */
  enemyIndex: number;
  /** Deterministic seed folded into the coordinate hash. */
  seed: number;
  /** Zone partition used to derive the deterministic zone id. */
  zonePartition: NgeZonePartitionConfig;
}

/**
 * Result of allocating one deterministic enemy substrate coordinate.
 */
export interface EnemyCoordinateAllocatorResult {
  /** Three-axis unit-cube coordinate assigned to the enemy. */
  coordinate: NeatGenomeSubstrateCoordinate;
  /** Deterministic zone id of the form `z:x:y:z` for the coordinate. */
  zoneId: string;
}

const SUBSTRATE_DIMENSIONS = 3;
const NORMALIZATION_MODE = 'unit-cube';
const HEX_DIGITS_PER_AXIS = 8;
const MAX_UINT32 = 0xffff_ffff;

/**
 * Allocate one deterministic unit-cube substrate coordinate for a single enemy.
 *
 * The returned coordinate is a function of `(swarmSize, enemyIndex, seed)` only,
 * making the placement reproducible across processes and independent of the
 * order in which enemies are materialized. The zone id is derived from the
 * existing zone-partition logic so that coordinate and zone remain consistent.
 *
 * @param input - Allocation request including swarm size, enemy index, seed, and zone partition.
 * @returns Deterministic coordinate and matching zone id.
 * @throws Error When `swarmSize`, `enemyIndex`, or `seed` violate their contracts.
 *
 * @example
 * ```ts
 * const result = allocateEnemySubstrateCoordinates({
 *   swarmSize: 8,
 *   enemyIndex: 3,
 *   seed: 42,
 *   zonePartition: { x: { count: 4 }, y: { count: 4 }, z: { count: 4 } },
 * });
 * console.log(result.coordinate, result.zoneId);
 * ```
 */
export function allocateEnemySubstrateCoordinates(
  input: EnemyCoordinateAllocatorInput,
): EnemyCoordinateAllocatorResult {
  validateAllocatorInput(input);

  const coordinate = deriveDeterministicCoordinate(input);
  const zoneId = assignZone(coordinate, input.zonePartition);

  return { coordinate, zoneId };
}

function validateAllocatorInput(input: EnemyCoordinateAllocatorInput): void {
  if (!Number.isInteger(input.swarmSize) || input.swarmSize <= 0) {
    throw new Error(
      `swarmSize must be a positive integer, got ${input.swarmSize}.`,
    );
  }

  if (
    !Number.isInteger(input.enemyIndex) ||
    input.enemyIndex < 0 ||
    input.enemyIndex >= input.swarmSize
  ) {
    throw new Error(
      `enemyIndex must be an integer in [0, ${input.swarmSize}), got ${input.enemyIndex}.`,
    );
  }

  if (!Number.isFinite(input.seed)) {
    throw new Error(`seed must be finite, got ${input.seed}.`);
  }
}

function deriveDeterministicCoordinate(
  input: EnemyCoordinateAllocatorInput,
): NeatGenomeSubstrateCoordinate {
  const hashInput = {
    dimensions: SUBSTRATE_DIMENSIONS,
    enemyIndex: input.enemyIndex,
    normalization: NORMALIZATION_MODE,
    seed: input.seed,
    swarmSize: input.swarmSize,
  };
  const hash = computeFingerprint(canonicalSerialize(hashInput));

  return [
    hexSliceToUnitInterval(hash, 0),
    hexSliceToUnitInterval(hash, HEX_DIGITS_PER_AXIS),
    hexSliceToUnitInterval(hash, HEX_DIGITS_PER_AXIS * 2),
  ];
}

function hexSliceToUnitInterval(hash: string, offset: number): number {
  const hexChunk = hash.slice(offset, offset + HEX_DIGITS_PER_AXIS);
  const integerValue = Number.parseInt(hexChunk, 16);

  return integerValue / MAX_UINT32;
}
