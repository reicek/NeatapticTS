import {
  resolveRacingRenderFrameTransferList,
  assertRacingSchemaVersion,
} from './simulation-worker.snapshot.utils';
import type { RacingRenderFrame } from './simulation-worker.types';

/**
 * Red-phase contracts for the `RacingRenderFrame` packing helpers.
 *
 * These tests target three Tier-0 acceptance criteria:
 * 1. Every typed-array buffer in a `RacingRenderFrame` appears in the transfer
 *    list (zero-copy transport).
 * 2. Unknown `schemaVersion` values are rejected before any field is read.
 * 3. Buffers are actually detached after a simulated postMessage transfer.
 *
 * All tests intentionally fail until `simulation-worker.snapshot.utils.ts` is
 * implemented.
 */
describe('simulation-worker.snapshot.utils', () => {
  describe('resolveRacingRenderFrameTransferList', () => {
    it('returns exactly 10 ArrayBuffer entries for a 1-car Tier-0 frame', () => {
      // Arrange
      const frame = createMinimalFrame(1);

      // Act
      const transferList = resolveRacingRenderFrameTransferList(frame);

      // Assert — stub returns [], length 0, expected 10 → red
      expect(transferList.length).toBe(10);
    });

    it('includes the carX buffer in the transfer list', () => {
      // Arrange
      const frame = createMinimalFrame(1);

      // Act
      const transferList = resolveRacingRenderFrameTransferList(frame);

      // Assert — stub returns [], carX.buffer not present → red
      expect(transferList.includes(frame.carX.buffer as ArrayBuffer)).toBe(
        true,
      );
    });

    it('includes the carY buffer in the transfer list', () => {
      // Arrange
      const frame = createMinimalFrame(1);

      // Act
      const transferList = resolveRacingRenderFrameTransferList(frame);

      // Assert — stub returns [] → red
      expect(transferList.includes(frame.carY.buffer as ArrayBuffer)).toBe(
        true,
      );
    });

    it('includes the tireState buffer in the transfer list', () => {
      // Arrange
      const frame = createMinimalFrame(1);

      // Act
      const transferList = resolveRacingRenderFrameTransferList(frame);

      // Assert — stub returns [] → red
      expect(transferList.includes(frame.tireState.buffer as ArrayBuffer)).toBe(
        true,
      );
    });

    it('includes the place buffer in the transfer list', () => {
      // Arrange
      const frame = createMinimalFrame(1);

      // Act
      const transferList = resolveRacingRenderFrameTransferList(frame);

      // Assert — stub returns [] → red
      expect(transferList.includes(frame.place.buffer as ArrayBuffer)).toBe(
        true,
      );
    });

    it('contains no duplicate ArrayBuffer entries for a 1-car frame', () => {
      // Arrange
      const frame = createAliasedFrame(1);
      const sharedBuffer = frame.carX.buffer as ArrayBuffer;

      // Act
      const transferList = resolveRacingRenderFrameTransferList(frame);
      const uniqueCount = new Set(transferList).size;

      // Assert — carX and carY intentionally alias one buffer, so a correct
      // transfer list contains 9 unique buffers and still includes the shared
      // backing store exactly once. Stub returns [] → red
      expect({
        totalCount: transferList.length,
        uniqueCount,
        containsSharedBuffer: transferList.includes(sharedBuffer),
      }).toEqual({
        totalCount: 9,
        uniqueCount: 9,
        containsSharedBuffer: true,
      });
    });

    it('detaches carX buffer after a simulated postMessage transfer using the resolved list', () => {
      // Arrange
      const frame = createMinimalFrame(1);
      const transferList = resolveRacingRenderFrameTransferList(frame);

      // Act — structuredClone with transfer detaches listed buffers
      structuredClone(frame, { transfer: transferList });

      // Assert — stub returns [], carX.buffer is NOT transferred, byteLength
      // stays at 4 (1 Float32 = 4 bytes), expected 0 → red
      expect(frame.carX.byteLength).toBe(0);
    });
  });

  describe('assertRacingSchemaVersion', () => {
    it('throws RangeError when schemaVersion is an unknown string', () => {
      // Arrange
      const staleFrame = { schemaVersion: 'outdated-schema-v0' };

      // Act + Assert — stub does nothing, no throw → expected throw never
      // arrives → red
      expect(() => assertRacingSchemaVersion(staleFrame)).toThrow(RangeError);
    });

    it('throws RangeError when schemaVersion is undefined', () => {
      // Arrange
      const missingSchemaFrame = { schemaVersion: undefined };

      // Act + Assert — stub does nothing → red
      expect(() => assertRacingSchemaVersion(missingSchemaFrame)).toThrow(
        RangeError,
      );
    });

    it('throws RangeError when schemaVersion is a numeric version tag', () => {
      // Arrange
      const numericVersionFrame = { schemaVersion: 1 };

      // Act + Assert — stub does nothing → red
      expect(() => assertRacingSchemaVersion(numericVersionFrame)).toThrow(
        RangeError,
      );
    });
  });
});

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/**
 * Builds a minimal valid `RacingRenderFrame` for snapshot transport tests.
 *
 * All typed arrays are freshly allocated (no shared buffers), which allows
 * transfer-list uniqueness checks to be unambiguous.
 *
 * @param agentCount - Number of agent slots to allocate.
 * @returns A structurally valid frame with zeroed typed-array fields.
 */
function createMinimalFrame(agentCount: number): RacingRenderFrame {
  return {
    schemaVersion: 'racing-packed-v1',
    tick: 0,
    seed: 0,
    trackId: 0,
    agentCount,
    featureFlags: 0,
    carX: new Float32Array(agentCount),
    carY: new Float32Array(agentCount),
    carHeading: new Float32Array(agentCount),
    carActive: new Uint8Array(agentCount),
    carTeam: new Uint8Array(agentCount),
    carMode: new Uint8Array(agentCount),
    tireState: new Float32Array(agentCount * 4),
    radioField: new Float32Array(0),
    lap: new Uint16Array(agentCount),
    place: new Uint8Array(agentCount),
    raceTimeMs: 0,
    done: false,
  };
}

/**
 * Builds a frame whose `carX` and `carY` arrays intentionally share one
 * backing buffer so the transfer-list de-duplication contract can be asserted.
 *
 * @param agentCount - Number of agent slots to allocate.
 * @returns A frame with aliased position buffers.
 */
function createAliasedFrame(agentCount: number): RacingRenderFrame {
  const sharedPositionBuffer = new ArrayBuffer(
    Float32Array.BYTES_PER_ELEMENT * agentCount,
  );

  return {
    ...createMinimalFrame(agentCount),
    carX: new Float32Array(sharedPositionBuffer),
    carY: new Float32Array(sharedPositionBuffer),
  };
}
