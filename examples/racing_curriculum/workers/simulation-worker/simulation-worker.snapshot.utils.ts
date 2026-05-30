import type { RacingRenderFrame } from './simulation-worker.types';

/**
 * Collects every `ArrayBuffer` backing a typed-array field in `frame` into a
 * transfer list for zero-copy `postMessage` transfer.
 *
 * Rules enforced by the real implementation:
 * - Every typed-array field must contribute exactly one buffer entry.
 * - If two typed arrays share the same underlying buffer they must be listed
 *   only once.
 * - The returned list must have the same length as the number of distinct
 *   typed-array buffers in the frame (10 for a standard Tier-0 solo frame,
 *   11 when Tier 4 adds the optional `pitStatus` array).
 *
 * @param frame - The packed render frame whose buffers will be transferred.
 * @returns Ordered list of `ArrayBuffer` references for postMessage transfer.
 *
 * @example
 * ```ts
 * const transferList = resolveRacingRenderFrameTransferList(frame);
 * worker.postMessage({ type: 'frame', frame }, transferList);
 * ```
 */
export function resolveRacingRenderFrameTransferList(
  frame: RacingRenderFrame,
): ArrayBuffer[] {
  const transferList: ArrayBuffer[] = [];
  const seenBuffers = new Set<ArrayBuffer>();
  const typedArrayFields = [
    frame.carX,
    frame.carY,
    frame.carHeading,
    frame.carActive,
    frame.carTeam,
    frame.carMode,
    frame.tireState,
    frame.radioField,
    ...(frame.pitStatus === undefined ? [] : [frame.pitStatus]),
    frame.lap,
    frame.place,
  ];

  for (const typedArrayField of typedArrayFields) {
    const buffer = typedArrayField.buffer as ArrayBuffer;

    if (seenBuffers.has(buffer)) {
      continue;
    }

    seenBuffers.add(buffer);
    transferList.push(buffer);
  }

  return transferList;
}

/**
 * Asserts that the `schemaVersion` field of an incoming frame matches the
 * expected `'racing-packed-v1'` sentinel.
 *
 * Consumers must call this before reading any typed-array field so that a
 * version mismatch is caught at the boundary rather than silently
 * misinterpreting the packed bytes.
 *
 * @param frame - Incoming message payload (or any object with `schemaVersion`).
 * @throws {RangeError} When `schemaVersion` is not `'racing-packed-v1'`.
 *
 * @example
 * ```ts
 * assertRacingSchemaVersion(receivedMessage);
 * // now safe to read typed-array fields
 * ```
 */
export function assertRacingSchemaVersion(frame: {
  schemaVersion: unknown;
}): void {
  if (frame.schemaVersion !== 'racing-packed-v1') {
    throw new RangeError(
      `Unsupported RacingRenderFrame schemaVersion: ${String(frame.schemaVersion)}`,
    );
  }
}
