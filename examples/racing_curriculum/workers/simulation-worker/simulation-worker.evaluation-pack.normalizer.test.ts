/**
 * Red tests for the racing evaluation-pack normalizer (Layer 3 wrapper).
 *
 * Phase 4 Step 04 — red contracts targeting the seam where the generic Layer 2
 * `DeterministicEvaluationPack` is consumed to produce a benchmark-owned
 * `RacingRenderFrame`.
 *
 * These tests assert the desired future behavior defined by the Step 01 frozen
 * determinism contract and Step 02 seam mapping.  They fail because the
 * normalizer module throws `Error('Not implemented …')`.
 *
 * Single-expect rule enforced throughout.
 */

import {
  createDeterministicEvaluationPack,
  type EvaluationPackInputs,
} from '../../../../src/architecture/network/evaluation-pack/network.evaluation-pack';
import {
  populateRacingFrame,
  resolveRacingTransferList,
  assertRacingPackSchemaVersion,
  type RacingEvaluationConfig,
} from './simulation-worker.evaluation-pack.normalizer';

// ---------------------------------------------------------------------------
// Deterministic fixtures
// ---------------------------------------------------------------------------

/** Fixed seed for all deterministic normalizer tests. */
const SEED = 42;

/** Minimal transport-neutral inputs for the generic Layer 2 pack. */
const CORE_INPUTS: EvaluationPackInputs = {
  agentCount: 4,
  schemaVersion: 'eval-pack-v1',
};

/** Racing-specific configuration paired with the generic pack. */
const RACING_CONFIG: RacingEvaluationConfig = {
  opponentSnapshot: {
    snapshotId: 'snap-test-0',
    generation: 1,
    networkPayloads: [],
  },
  trackId: 0,
  featureFlags: 0,
  agentCount: 4,
  packSchemaVersion: 'eval-pack-v1',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Compares two typed arrays byte-for-byte.
 * Pure helper — does not call `expect`.
 */
function typedArraysEqual(a: ArrayBufferView, b: ArrayBufferView): boolean {
  if (a.byteLength !== b.byteLength) {
    return false;
  }
  const aBytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
  const bBytes = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
  for (let i = 0; i < aBytes.length; i++) {
    if (aBytes[i] !== bBytes[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Compares two racing render frames for equality of their typed-array fields.
 * Pure helper — does not call `expect`.
 */
function racingFramesEqual(a: unknown, b: unknown): boolean {
  if (
    typeof a !== 'object' ||
    a === null ||
    typeof b !== 'object' ||
    b === null
  ) {
    return false;
  }
  const frameA = a as Record<string, unknown>;
  const frameB = b as Record<string, unknown>;
  const fields: string[] = [
    'carX',
    'carY',
    'carHeading',
    'carActive',
    'carTeam',
    'carMode',
    'tireState',
    'radioField',
    'lap',
    'place',
  ];
  return fields.every((field) => {
    const arrA = frameA[field] as ArrayBufferView | undefined;
    const arrB = frameB[field] as ArrayBufferView | undefined;
    if (arrA === undefined || arrB === undefined) {
      return arrA === arrB;
    }
    return typedArraysEqual(arrA, arrB);
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('simulation-worker.evaluation-pack.normalizer', () => {
  describe('populateRacingFrame', () => {
    it('returns a RacingRenderFrame with the racing schema version sentinel', () => {
      // Arrange
      const pack = createDeterministicEvaluationPack(SEED, CORE_INPUTS);

      // Act
      const frame = populateRacingFrame(pack, RACING_CONFIG);

      // Assert
      expect(frame.schemaVersion).toBe('racing-packed-v1');
    });

    it('produces identical frames from the same core pack and racing config', () => {
      // Arrange
      const pack = createDeterministicEvaluationPack(SEED, CORE_INPUTS);

      // Act
      const frameA = populateRacingFrame(pack, RACING_CONFIG);
      const frameB = populateRacingFrame(pack, RACING_CONFIG);

      // Assert
      expect(racingFramesEqual(frameA, frameB)).toBe(true);
    });

    it('throws RangeError when the core pack schema version does not match', () => {
      // Arrange
      const pack = createDeterministicEvaluationPack(SEED, {
        ...CORE_INPUTS,
        schemaVersion: 'wrong-version',
      });

      // Act + Assert
      expect(() => populateRacingFrame(pack, RACING_CONFIG)).toThrow(
        RangeError,
      );
    });

    it('throws RangeError when the racing agent count mismatches the pack arrays', () => {
      // Arrange
      const pack = createDeterministicEvaluationPack(SEED, CORE_INPUTS);
      const mismatchedConfig = { ...RACING_CONFIG, agentCount: 99 };

      // Act + Assert
      expect(() => populateRacingFrame(pack, mismatchedConfig)).toThrow(
        RangeError,
      );
    });
  });

  describe('resolveRacingTransferList', () => {
    it('collects every distinct ArrayBuffer from a populated racing frame', () => {
      // Arrange
      const pack = createDeterministicEvaluationPack(SEED, CORE_INPUTS);
      const frame = populateRacingFrame(pack, RACING_CONFIG);

      // Act
      const transferList = resolveRacingTransferList(frame);

      // Assert
      expect(transferList.length).toBeGreaterThan(0);
    });
  });

  describe('assertRacingPackSchemaVersion', () => {
    it('accepts a pack whose schema version matches the expected version', () => {
      // Arrange
      const pack = { schemaVersion: 'eval-pack-v1' };

      // Act + Assert
      expect(() =>
        assertRacingPackSchemaVersion(pack, 'eval-pack-v1'),
      ).not.toThrow();
    });

    it('throws RangeError when the pack schema version does not match', () => {
      // Arrange
      const pack = { schemaVersion: 'eval-pack-v1' };

      // Act + Assert
      expect(() =>
        assertRacingPackSchemaVersion(pack, 'wrong-version'),
      ).toThrow(RangeError);
    });
  });
});
