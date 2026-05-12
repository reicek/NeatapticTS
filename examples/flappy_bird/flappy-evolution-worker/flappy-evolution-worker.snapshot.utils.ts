import type {
  WorkerPlaybackFrameSnapshot,
  WorkerPlaybackState,
} from './flappy-evolution-worker.types';

type WorkerPlaybackSnapshotBuffers = {
  birdDoneFlags: Uint8Array;
  birdFramesSurvived: Uint32Array;
  birdPipesPassed: Uint32Array;
  birdYPositionsPx: Float32Array;
  pipeGapCenterYPositionsPx: Float32Array;
  pipeGapSizesPx: Float32Array;
  pipeXPositionsPx: Float32Array;
};

const sharedSnapshotBuffersByPlaybackState = new WeakMap<
  WorkerPlaybackState,
  WorkerPlaybackSnapshotBuffers
>();

/**
 * Creates a serializable snapshot of current playback state.
 *
 * Workers should send only structured-clone-safe payloads. This helper strips
 * runtime-only references (e.g., network instances, sets) and keeps only
 * renderer-relevant fields.
 *
 * Educational note:
 * The snapshot is intentionally column-oriented. By packing values into typed
 * arrays, the worker can transfer large bird populations to the host with much
 * lower overhead than a per-frame array of nested objects.
 *
 * This is a small example of a structure-of-arrays transport layout. If that
 * pattern is unfamiliar, the Wikipedia article on "AoS and SoA" is a good short
 * reference for why packed columns are often friendlier to hot-path data
 * movement than arrays of rich objects.
 *
 * @param playbackState - Current mutable playback state.
 * @returns Immutable frame snapshot for the host.
 */
export function createWorkerPlaybackSnapshot(
  playbackState: WorkerPlaybackState,
): WorkerPlaybackFrameSnapshot {
  const pipeCount = playbackState.pipes.length;
  const birdCount = playbackState.birds.length;
  const snapshotBuffers = resolveWorkerPlaybackSnapshotBuffers(
    playbackState,
    pipeCount,
    birdCount,
  );

  for (let pipeIndex = 0; pipeIndex < pipeCount; pipeIndex += 1) {
    const pipe = playbackState.pipes[pipeIndex];
    snapshotBuffers.pipeXPositionsPx[pipeIndex] = pipe.xPx;
    snapshotBuffers.pipeGapCenterYPositionsPx[pipeIndex] = pipe.gapCenterYPx;
    snapshotBuffers.pipeGapSizesPx[pipeIndex] = pipe.gapSizePx;
  }

  for (let birdIndex = 0; birdIndex < birdCount; birdIndex += 1) {
    const bird = playbackState.birds[birdIndex];
    snapshotBuffers.birdYPositionsPx[birdIndex] = bird.yPx;
    snapshotBuffers.birdPipesPassed[birdIndex] = bird.pipesPassed;
    snapshotBuffers.birdFramesSurvived[birdIndex] = bird.framesSurvived;
    snapshotBuffers.birdDoneFlags[birdIndex] = bird.done ? 1 : 0;
  }

  return {
    format: 'packed-v1',
    frameIndex: playbackState.frameIndex,
    cumulativePipeTravelPx: playbackState.cumulativePipeTravelPx,
    visibleWorldWidthPx: playbackState.visibleWorldWidthPx,
    visibleWorldHeightPx: playbackState.visibleWorldHeightPx,
    pipeCount,
    birdCount,
    pipes: {
      xPositionsPx: snapshotBuffers.pipeXPositionsPx,
      gapCenterYPositionsPx: snapshotBuffers.pipeGapCenterYPositionsPx,
      gapSizesPx: snapshotBuffers.pipeGapSizesPx,
    },
    birds: {
      yPositionsPx: snapshotBuffers.birdYPositionsPx,
      pipesPassed: snapshotBuffers.birdPipesPassed,
      framesSurvived: snapshotBuffers.birdFramesSurvived,
      doneFlags: snapshotBuffers.birdDoneFlags,
    },
  };
}

/**
 * Resolves transferable buffers for one packed playback snapshot.
 *
 * The returned buffers should be passed as the second argument to
 * `postMessage(...)` so ownership moves to the host thread instead of copying
 * the typed-array contents.
 *
 * That ownership transfer is a large part of why the worker can stream full
 * population snapshots without forcing the main thread to pay unnecessary copy
 * costs every frame.
 *
 * @param snapshot - Packed playback snapshot posted back to the browser host.
 * @returns Transfer list used to move typed-array buffers without copying.
 */
export function resolveWorkerPlaybackSnapshotTransferList(
  snapshot: WorkerPlaybackFrameSnapshot,
): Transferable[] {
  return [
    snapshot.pipes.xPositionsPx.buffer,
    snapshot.pipes.gapCenterYPositionsPx.buffer,
    snapshot.pipes.gapSizesPx.buffer,
    snapshot.birds.yPositionsPx.buffer,
    snapshot.birds.pipesPassed.buffer,
    snapshot.birds.framesSurvived.buffer,
    snapshot.birds.doneFlags.buffer,
  ].filter(isTransferableArrayBuffer);
}

/**
 * Resolves snapshot column storage for one playback state.
 *
 * @param playbackState - Current mutable playback state.
 * @param pipeCount - Number of visible pipes to pack.
 * @param birdCount - Number of playback birds to pack.
 * @returns Snapshot buffers sized for the current frame.
 */
function resolveWorkerPlaybackSnapshotBuffers(
  playbackState: WorkerPlaybackState,
  pipeCount: number,
  birdCount: number,
): WorkerPlaybackSnapshotBuffers {
  if (!canReuseSharedSnapshotBuffers()) {
    return createWorkerPlaybackSnapshotBuffers(pipeCount, birdCount, false);
  }

  const cachedSnapshotBuffers =
    sharedSnapshotBuffersByPlaybackState.get(playbackState);
  if (
    cachedSnapshotBuffers &&
    cachedSnapshotBuffers.pipeXPositionsPx.length === pipeCount &&
    cachedSnapshotBuffers.birdYPositionsPx.length === birdCount
  ) {
    return cachedSnapshotBuffers;
  }

  const nextSnapshotBuffers = createWorkerPlaybackSnapshotBuffers(
    pipeCount,
    birdCount,
    true,
  );
  sharedSnapshotBuffersByPlaybackState.set(playbackState, nextSnapshotBuffers);
  return nextSnapshotBuffers;
}

/**
 * Creates typed-array storage for one packed snapshot.
 *
 * @param pipeCount - Number of visible pipes to pack.
 * @param birdCount - Number of playback birds to pack.
 * @param useSharedBuffers - Whether buffers should be reusable shared memory.
 * @returns Snapshot buffer shelf.
 */
function createWorkerPlaybackSnapshotBuffers(
  pipeCount: number,
  birdCount: number,
  useSharedBuffers: boolean,
): WorkerPlaybackSnapshotBuffers {
  return {
    pipeXPositionsPx: createFloat32SnapshotArray(pipeCount, useSharedBuffers),
    pipeGapCenterYPositionsPx: createFloat32SnapshotArray(
      pipeCount,
      useSharedBuffers,
    ),
    pipeGapSizesPx: createFloat32SnapshotArray(pipeCount, useSharedBuffers),
    birdYPositionsPx: createFloat32SnapshotArray(birdCount, useSharedBuffers),
    birdPipesPassed: createUint32SnapshotArray(birdCount, useSharedBuffers),
    birdFramesSurvived: createUint32SnapshotArray(birdCount, useSharedBuffers),
    birdDoneFlags: createUint8SnapshotArray(birdCount, useSharedBuffers),
  };
}

/**
 * Creates one packed float column for snapshot transport.
 *
 * @param elementCount - Number of elements in the column.
 * @param useSharedBuffer - Whether to allocate reusable shared memory.
 * @returns Float32 snapshot column.
 */
function createFloat32SnapshotArray(
  elementCount: number,
  useSharedBuffer: boolean,
): Float32Array {
  return useSharedBuffer
    ? new Float32Array(
        new SharedArrayBuffer(elementCount * Float32Array.BYTES_PER_ELEMENT),
      )
    : new Float32Array(elementCount);
}

/**
 * Creates one packed uint32 column for snapshot transport.
 *
 * @param elementCount - Number of elements in the column.
 * @param useSharedBuffer - Whether to allocate reusable shared memory.
 * @returns Uint32 snapshot column.
 */
function createUint32SnapshotArray(
  elementCount: number,
  useSharedBuffer: boolean,
): Uint32Array {
  return useSharedBuffer
    ? new Uint32Array(
        new SharedArrayBuffer(elementCount * Uint32Array.BYTES_PER_ELEMENT),
      )
    : new Uint32Array(elementCount);
}

/**
 * Creates one packed uint8 column for snapshot transport.
 *
 * @param elementCount - Number of elements in the column.
 * @param useSharedBuffer - Whether to allocate reusable shared memory.
 * @returns Uint8 snapshot column.
 */
function createUint8SnapshotArray(
  elementCount: number,
  useSharedBuffer: boolean,
): Uint8Array {
  return useSharedBuffer
    ? new Uint8Array(
        new SharedArrayBuffer(elementCount * Uint8Array.BYTES_PER_ELEMENT),
      )
    : new Uint8Array(elementCount);
}

/**
 * Resolves whether this host can reuse shared snapshot buffers safely.
 *
 * @returns True when SharedArrayBuffer snapshot storage is available.
 */
function canReuseSharedSnapshotBuffers(): boolean {
  return (
    typeof globalThis.importScripts === 'function' &&
    typeof globalThis.Worker === 'function' &&
    typeof SharedArrayBuffer === 'function' &&
    globalThis.crossOriginIsolated === true
  );
}

/**
 * Narrows transfer-list candidates to transferable ArrayBuffers.
 *
 * @param buffer - Typed-array backing buffer.
 * @returns True when the buffer can be passed through postMessage transfer list.
 */
function isTransferableArrayBuffer(
  buffer: ArrayBufferLike,
): buffer is ArrayBuffer {
  return Object.prototype.toString.call(buffer) === '[object ArrayBuffer]';
}
