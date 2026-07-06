/**
 * Tests for the deterministic evaluation-pack normalization seam (Layer 2).
 *
 * Covers:
 * 1. Deterministic pack creation (same reproducibility tuple → identical pack).
 * 2. Zero-copy transfer-list resolution (deduplication and coverage).
 * 3. Schema-version boundary assertion with a clear `RangeError`.
 *
 * Single-expect rule enforced throughout.
 */
import {
  createDeterministicEvaluationPack,
  resolveTransferList,
  assertSchemaVersion,
  type DeterministicEvaluationPack,
  type EvaluationPackInputs,
} from './network.evaluation-pack';

// ---------------------------------------------------------------------------
// Deterministic fixtures
// ---------------------------------------------------------------------------

/** Fixed seed for all deterministic pack tests. */
const SEED = 42;

/** Minimal transport-neutral inputs (no domain-specific fields). */
const INPUTS: EvaluationPackInputs = {
  agentCount: 4,
  schemaVersion: 'test-eval-pack-v1',
};

/** Alternative seed to test pack distinctness. */
const ALT_SEED = 99;

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
 * Compares two evaluation packs for byte-level equality across all arrays.
 * Pure helper — does not call `expect`.
 */
function packsEqual(
  a: DeterministicEvaluationPack,
  b: DeterministicEvaluationPack,
): boolean {
  if (a.seed !== b.seed) {
    return false;
  }
  if (a.schemaVersion !== b.schemaVersion) {
    return false;
  }
  if (a.arrays.length !== b.arrays.length) {
    return false;
  }
  return a.arrays.every((arr, i) => typedArraysEqual(arr, b.arrays[i]));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('network.evaluation-pack', () => {
  describe('createDeterministicEvaluationPack — pack normalization', () => {
    it('produces identical packs from the same seed and inputs', () => {
      // Arrange — two calls with identical (seed, inputs)
      const pack1 = createDeterministicEvaluationPack(SEED, INPUTS);
      const pack2 = createDeterministicEvaluationPack(SEED, INPUTS);

      // Assert — byte-level equality
      expect(packsEqual(pack1, pack2)).toBe(true);
    });

    it('produces different packs from different seeds', () => {
      // Arrange — same inputs, different seeds
      const pack1 = createDeterministicEvaluationPack(SEED, INPUTS);
      const pack2 = createDeterministicEvaluationPack(ALT_SEED, INPUTS);

      // Assert — packs are NOT byte-identical
      expect(packsEqual(pack1, pack2)).toBe(false);
    });

    it('falls back to a non-zero PRNG state when the seed is zero', () => {
      // Arrange — deterministic inputs with a zero seed
      const zeroSeedInputs = INPUTS;

      // Act — create a pack from the zero seed
      const pack = createDeterministicEvaluationPack(0, zeroSeedInputs);

      // Assert — the pack is produced without error and contains the expected arrays
      expect(pack.arrays).toHaveLength(3);
    });
  });

  describe('replay stability', () => {
    it('replays produce identical typed array contents from the same seed', () => {
      // Arrange — two independent packs from the same (seed, inputs)
      const pack1 = createDeterministicEvaluationPack(SEED, INPUTS);
      const pack2 = createDeterministicEvaluationPack(SEED, INPUTS);

      // Act — compare every array element-by-element
      const allArraysMatch = pack1.arrays.every((arr, i) =>
        typedArraysEqual(arr, pack2.arrays[i]),
      );

      // Assert — every typed array is byte-identical on replay
      expect(allArraysMatch).toBe(true);
    });
  });

  describe('resolveTransferList — episode-step transport', () => {
    it('collects all distinct buffer entries from a pack', () => {
      // Arrange — a pack with three distinct buffers (no sharing)
      const bufA = new ArrayBuffer(16);
      const bufB = new ArrayBuffer(8);
      const bufC = new ArrayBuffer(32);
      const pack: DeterministicEvaluationPack = {
        schemaVersion: 'test-v1',
        seed: SEED,
        arrays: [
          new Float32Array(bufA, 0, 4),
          new Uint8Array(bufB, 0, 8),
          new Float32Array(bufC, 0, 8),
        ],
      };

      // Act
      const transferList = resolveTransferList(pack);

      // Assert — one entry per distinct buffer (3)
      expect(transferList).toHaveLength(3);
    });

    it('deduplicates buffers shared across multiple typed arrays', () => {
      // Arrange — two typed arrays sharing one underlying buffer
      const sharedBuffer = new ArrayBuffer(32);
      const pack: DeterministicEvaluationPack = {
        schemaVersion: 'test-v1',
        seed: SEED,
        arrays: [
          new Float32Array(sharedBuffer, 0, 4),
          new Float32Array(sharedBuffer, 16, 4),
        ],
      };

      // Act
      const transferList = resolveTransferList(pack);

      // Assert — shared buffer listed only once
      expect(transferList).toHaveLength(1);
    });
  });

  describe('assertSchemaVersion — schema versioning', () => {
    it('accepts a pack whose schema version matches the expected version', () => {
      // Arrange — pack with matching schema version
      const pack = { schemaVersion: 'test-eval-pack-v1' };

      // Act + Assert — must NOT throw
      expect(() =>
        assertSchemaVersion(pack, 'test-eval-pack-v1'),
      ).not.toThrow();
    });

    it('throws RangeError when the schema version does not match', () => {
      // Arrange — pack with mismatched schema version
      const pack = { schemaVersion: 'test-eval-pack-v1' };

      // Act + Assert — must throw RangeError (not generic Error)
      expect(() => assertSchemaVersion(pack, 'wrong-version')).toThrow(
        RangeError,
      );
    });
  });
});
