/**
 * Red-phase test contracts for Phase 4 Step 01 — deterministic per-enemy
 * substrate coordinate allocator.
 *
 * Covers AC-407: repeated builds with identical (swarmSize, enemyIndex, seed)
 * produce identical NeatGenomeSubstrateCoordinate set and stable zoneId ordering.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { NeatGenomeSubstrateCoordinate } from '../genome/genome.types';

import { assignZone } from './neat.nge-dna.substrate';
import {
  allocateEnemySubstrateCoordinates,
  type EnemyCoordinateAllocatorInput,
} from './neat.nge-dna.coordinate-allocator';

const defaultZonePartition = {
  x: { count: 4 },
  y: { count: 4 },
  z: { count: 4 },
} as const;

function createAllocatorInput(
  overrides?: Partial<EnemyCoordinateAllocatorInput>,
): EnemyCoordinateAllocatorInput {
  return {
    swarmSize: 8,
    enemyIndex: 3,
    seed: 42,
    zonePartition: defaultZonePartition,
    ...overrides,
  };
}

describe('allocateEnemySubstrateCoordinates', () => {
  describe('determinism contract', () => {
    it('returns the same coordinate for identical inputs', () => {
      // Arrange
      const input = createAllocatorInput();

      // Act
      const first = allocateEnemySubstrateCoordinates(input);
      const second = allocateEnemySubstrateCoordinates(input);

      // Assert
      expect(first.coordinate).toEqual(second.coordinate);
    });

    it('returns the same zoneId for identical inputs', () => {
      // Arrange
      const input = createAllocatorInput();

      // Act
      const first = allocateEnemySubstrateCoordinates(input);
      const second = allocateEnemySubstrateCoordinates(input);

      // Assert
      expect(first.zoneId).toBe(second.zoneId);
    });

    it('produces different coordinates for different seeds', () => {
      // Arrange
      const inputA = createAllocatorInput({ seed: 42 });
      const inputB = createAllocatorInput({ seed: 43 });

      // Act
      const resultA = allocateEnemySubstrateCoordinates(inputA);
      const resultB = allocateEnemySubstrateCoordinates(inputB);

      // Assert
      expect(resultA.coordinate).not.toEqual(resultB.coordinate);
    });

    it('produces different coordinates for different enemy indices', () => {
      // Arrange
      const inputA = createAllocatorInput({ enemyIndex: 0 });
      const inputB = createAllocatorInput({ enemyIndex: 1 });

      // Act
      const resultA = allocateEnemySubstrateCoordinates(inputA);
      const resultB = allocateEnemySubstrateCoordinates(inputB);

      // Assert
      expect(resultA.coordinate).not.toEqual(resultB.coordinate);
    });
  });

  describe('unit-cube conformance', () => {
    it('returns coordinates within the unit interval on every axis', () => {
      // Arrange
      const input = createAllocatorInput();

      // Act
      const result = allocateEnemySubstrateCoordinates(input);

      // Assert
      expect(
        result.coordinate.every(
          (axisValue: number) => axisValue >= 0 && axisValue <= 1,
        ),
      ).toBe(true);
    });

    it('produces a 3-dimensional coordinate', () => {
      // Arrange
      const input = createAllocatorInput();

      // Act
      const result = allocateEnemySubstrateCoordinates(input);

      // Assert
      expect(result.coordinate).toHaveLength(3);
    });
  });

  describe('zone stability', () => {
    it('assigns a zoneId matching the coordinate and partition', () => {
      // Arrange
      const input = createAllocatorInput();

      // Act
      const result = allocateEnemySubstrateCoordinates(input);
      const expectedZoneId = assignZone(
        result.coordinate,
        defaultZonePartition,
      );

      // Assert
      expect(result.zoneId).toBe(expectedZoneId);
    });

    it('returns stable integer-formatted zoneId strings', () => {
      // Arrange
      const input = createAllocatorInput();

      // Act
      const result = allocateEnemySubstrateCoordinates(input);

      // Assert
      expect(result.zoneId).toMatch(/^z:\d+:\d+:\d+$/);
    });
  });

  describe('batch set contract', () => {
    it('can build a deterministic set of length equal to swarmSize', () => {
      // Arrange
      const swarmSize = 8;
      const seed = 42;

      // Act
      const coordinates: NeatGenomeSubstrateCoordinate[] = [];
      for (let enemyIndex = 0; enemyIndex < swarmSize; enemyIndex += 1) {
        const result = allocateEnemySubstrateCoordinates({
          swarmSize,
          enemyIndex,
          seed,
          zonePartition: defaultZonePartition,
        });
        coordinates.push(result.coordinate);
      }

      // Assert
      expect(coordinates).toHaveLength(swarmSize);
    });

    it('rebuilds the identical coordinate set from the same seed and swarm size', () => {
      // Arrange
      const swarmSize = 8;
      const seed = 42;

      // Act
      const buildSet = () => {
        const coordinates: NeatGenomeSubstrateCoordinate[] = [];
        for (let enemyIndex = 0; enemyIndex < swarmSize; enemyIndex += 1) {
          const result = allocateEnemySubstrateCoordinates({
            swarmSize,
            enemyIndex,
            seed,
            zonePartition: defaultZonePartition,
          });
          coordinates.push(result.coordinate);
        }
        return coordinates;
      };

      // Assert
      expect(buildSet()).toEqual(buildSet());
    });
  });

  describe('input validation', () => {
    it('throws when swarmSize is not positive', () => {
      // Arrange
      const input = createAllocatorInput({ swarmSize: 0 });

      // Act + Assert
      expect(() => allocateEnemySubstrateCoordinates(input)).toThrow(
        'swarmSize must be a positive integer',
      );
    });

    it('throws when enemyIndex is greater than or equal to swarmSize', () => {
      // Arrange
      const input = createAllocatorInput({ enemyIndex: 8 });

      // Act + Assert
      expect(() => allocateEnemySubstrateCoordinates(input)).toThrow(
        'enemyIndex must be an integer in [0, 8)',
      );
    });

    it('throws when enemyIndex is negative', () => {
      // Arrange
      const input = createAllocatorInput({ enemyIndex: -1 });

      // Act + Assert
      expect(() => allocateEnemySubstrateCoordinates(input)).toThrow(
        'enemyIndex must be an integer in [0, 8)',
      );
    });

    it('throws when enemyIndex is not an integer', () => {
      // Arrange
      const input = createAllocatorInput({ enemyIndex: 1.5 });

      // Act + Assert
      expect(() => allocateEnemySubstrateCoordinates(input)).toThrow(
        'enemyIndex must be an integer in [0, 8)',
      );
    });

    it('throws when seed is not finite', () => {
      // Arrange
      const input = createAllocatorInput({ seed: Infinity });

      // Act + Assert
      expect(() => allocateEnemySubstrateCoordinates(input)).toThrow(
        'seed must be finite',
      );
    });
  });
});
