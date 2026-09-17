import type {
  WorkerPlaybackFrameSnapshot,
  WorkerPlaybackState,
} from './flappy-evolution-worker.types';
import {
  resolveFramePrimaryWinnerIndex,
  resolveLeaderPipesPassed,
  resolveObservationVector,
} from '../browser-entry/browser-entry.observation.utils';
import { resolveAdaptiveDifficultyProfile } from '../flappy.simulation.shared.utils';

type WorkerPlaybackSnapshotBuffers = {
  birdDoneFlags: Uint8Array;
  birdFramesSurvived: Uint32Array;
  birdPipesPassed: Uint32Array;
  birdYPositionsPx: Float32Array;
  pipeGapCenterYPositionsPx: Float32Array;
  pipeGapSizesPx: Float32Array;
  pipeXPositionsPx: Float32Array;
  winnerNodeActivations: Float32Array;
};

const sharedSnapshotBuffersByPlaybackState = new WeakMap<
  WorkerPlaybackState,
  WorkerPlaybackSnapshotBuffers
>();

/**
 * Resolves the frame's winner bird index for the activation stream.
 *
 * Live frames prefer the leading alive bird so the visualization highlights
 * the current leader while frames stream. When every bird is done, the
 * resolution falls back to the overall population winner so the final frame's
 * activation stream stays consistent with the end-of-run winner summary.
 *
 * @param birds - Current mutable playback birds.
 * @returns Winner bird index, or -1 when no bird can be resolved.
 *
 * @example
 * ```ts
 * const birds = [
 *   { done: false, pipesPassed: 2, framesSurvived: 100 },
 *   { done: true, pipesPassed: 1, framesSurvived: 50 },
 * ] as unknown as WorkerPlaybackState['birds'];
 * const winnerIndex = resolveWorkerPlaybackWinnerBirdIndex(birds);
 * ```
 */
export function resolveWorkerPlaybackWinnerBirdIndex(
  birds: WorkerPlaybackState['birds'],
): number {
  const aliveWinnerIndex = resolveFramePrimaryWinnerIndex(birds, true);
  if (aliveWinnerIndex >= 0) {
    return aliveWinnerIndex;
  }
  return resolveFramePrimaryWinnerIndex(birds, false);
}

/**
 * Populates the winner bird's network node activations before snapshot packing.
 *
 * The transferable inference-channel fast path predicts outputs without writing
 * activations back into the live `bird.network.nodes` shelf. Visualization
 * needs the full node activation stream, so for the frame winner that is using
 * an inference channel we run one dedicated CPU activation pass on the
 * underlying network. This is winner-only so the cost stays bounded to one
 * extra forward pass per frame.
 *
 * @param playbackState - Current mutable playback state.
 * @param winnerBirdIndex - Index of the bird whose activations should stream.
 * @returns Nothing.
 */
function populateWinnerNodeActivationsBeforePacking(
  playbackState: WorkerPlaybackState,
  winnerBirdIndex: number,
): void {
  const winnerBird = playbackState.birds[winnerBirdIndex];
  if (
    !winnerBird?.inferenceChannel ||
    typeof winnerBird.network.activate !== 'function'
  ) {
    return;
  }

  const leaderPipesPassed = resolveLeaderPipesPassed(playbackState.birds);
  const difficultyProfile = resolveAdaptiveDifficultyProfile(leaderPipesPassed);
  const observation = resolveObservationVector(
    winnerBird.yPx,
    winnerBird.velocityYPxPerFrame,
    playbackState.pipes,
    playbackState.visibleWorldWidthPx,
    playbackState.visibleWorldHeightPx,
    difficultyProfile,
    playbackState.lastSpawnedPipeSpawnIntervalFrames,
    winnerBird.observationMemoryState,
  );

  winnerBird.network.activate(observation.observationVector);
}

/**
 * Creates a serializable snapshot of current playback state.
 *
 * Workers should send only structured-clone-safe payloads. This helper strips
 * runtime-only references (e.g., network instances, sets) and keeps only
 * renderer-relevant fields.
 *
 * The snapshot is intentionally column-oriented. By packing values into typed
 * arrays, the worker can transfer large bird populations to the host with much
 * lower overhead than a per-frame array of nested objects.
 *
 * This is a small example of a structure-of-arrays transport layout. If that
 * pattern is unfamiliar, the Wikipedia article on "AoS and SoA" is a good short
 * reference for why packed columns are often friendlier to hot-path data
 * movement than arrays of rich objects.
 *
 * For the frame winner, the snapshot also includes a `winnerNodeActivations`
 * stream. When the winner uses a transferable inference channel its live
 * `network.nodes` activations are not populated by the fast path, so this
 * helper runs one dedicated CPU activation pass on the winner's network before
 * packing. That keeps the network visualizer labels live without forcing every
 * bird through a redundant activation pass.
 *
 * @param playbackState - Current mutable playback state.
 * @returns Fresh frame snapshot. The returned object is a new reference, but
 *   its typed-array columns are mutable and may reuse shared backing memory
 *   across frames when cross-origin isolation allows.
 *
 * @example
 * ```ts
 * import { createSharedObservationMemoryState } from '../flappy.simulation.shared.utils';
 *
 * const playbackState = {
 *   frameIndex: 12,
 *   cumulativePipeTravelPx: 120,
 *   visibleWorldWidthPx: 640,
 *   visibleWorldHeightPx: 480,
 *   nextPipeId: 0,
 *   lastSpawnedPipeGapPx: 0,
 *   lastSpawnedPipeGapCenterYPx: 0,
 *   lastSpawnedPipeSpawnIntervalFrames: 0,
 *   framesUntilNextPipeSpawn: 0,
 *   pipes: [{ id: 0, xPx: 100, gapCenterYPx: 200, gapSizePx: 120 }],
 *   birds: [
 *     {
 *       done: false,
 *       yPx: 240,
 *       velocityYPxPerFrame: 0,
 *       pipesPassed: 2,
 *       framesSurvived: 100,
 *       passedPipeIds: new Set<number>(),
 *       observationMemoryState: createSharedObservationMemoryState(),
 *       network: {
 *         activate: () => [0.5, -0.2],
 *         nodes: [{ activation: 0.5 }, { activation: -0.2 }],
 *       } as unknown as import('../../../src/architecture/network').default,
 *     },
 *   ],
 * } as unknown as WorkerPlaybackState;
 *
 * const snapshot = createWorkerPlaybackSnapshot(playbackState);
 * console.log(snapshot.winnerNodeActivations);
 * ```
 */
export function createWorkerPlaybackSnapshot(
  playbackState: WorkerPlaybackState,
): WorkerPlaybackFrameSnapshot {
  const pipeCount = playbackState.pipes.length;
  const birdCount = playbackState.birds.length;
  const winnerBirdIndex = resolveWorkerPlaybackWinnerBirdIndex(
    playbackState.birds,
  );

  if (winnerBirdIndex >= 0) {
    populateWinnerNodeActivationsBeforePacking(playbackState, winnerBirdIndex);
  }

  const winnerNodeCount =
    winnerBirdIndex >= 0
      ? (playbackState.birds[winnerBirdIndex]?.network.nodes.length ?? 0)
      : 0;

  const snapshotBuffers = resolveWorkerPlaybackSnapshotBuffers(
    playbackState,
    pipeCount,
    birdCount,
    winnerNodeCount,
  );

  packPipeColumnsIntoSnapshot(playbackState.pipes, snapshotBuffers);
  packBirdColumnsIntoSnapshot(playbackState.birds, snapshotBuffers);
  packWinnerNodeActivationsIntoSnapshot(
    winnerBirdIndex >= 0 ? playbackState.birds[winnerBirdIndex] : undefined,
    winnerNodeCount,
    snapshotBuffers.winnerNodeActivations,
  );

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
    winnerNodeActivations: snapshotBuffers.winnerNodeActivations,
  };
}

/**
 * Writes pipe geometry into the snapshot's packed pipe columns.
 *
 * @param pipes - Visible pipes from the current playback state.
 * @param buffers - Mutable snapshot buffer shelf.
 */
function packPipeColumnsIntoSnapshot(
  pipes: WorkerPlaybackState['pipes'],
  buffers: WorkerPlaybackSnapshotBuffers,
): void {
  for (let pipeIndex = 0; pipeIndex < pipes.length; pipeIndex += 1) {
    const pipe = pipes[pipeIndex];
    buffers.pipeXPositionsPx[pipeIndex] = pipe.xPx;
    buffers.pipeGapCenterYPositionsPx[pipeIndex] = pipe.gapCenterYPx;
    buffers.pipeGapSizesPx[pipeIndex] = pipe.gapSizePx;
  }
}

/**
 * Writes bird state into the snapshot's packed bird columns.
 *
 * @param birds - Playback birds from the current playback state.
 * @param buffers - Mutable snapshot buffer shelf.
 */
function packBirdColumnsIntoSnapshot(
  birds: WorkerPlaybackState['birds'],
  buffers: WorkerPlaybackSnapshotBuffers,
): void {
  for (let birdIndex = 0; birdIndex < birds.length; birdIndex += 1) {
    const bird = birds[birdIndex];
    buffers.birdYPositionsPx[birdIndex] = bird.yPx;
    buffers.birdPipesPassed[birdIndex] = bird.pipesPassed;
    buffers.birdFramesSurvived[birdIndex] = bird.framesSurvived;
    buffers.birdDoneFlags[birdIndex] = bird.done ? 1 : 0;
  }
}

/**
 * Writes the winner bird's node activations into the snapshot column.
 *
 * @param winnerBird - Frame winner bird, or undefined when there is none.
 * @param winnerNodeCount - Number of winner network nodes to pack.
 * @param winnerNodeActivations - Mutable activation column to populate.
 */
function packWinnerNodeActivationsIntoSnapshot(
  winnerBird: WorkerPlaybackState['birds'][number] | undefined,
  winnerNodeCount: number,
  winnerNodeActivations: Float32Array,
): void {
  const winnerBirdNodes = winnerBird?.network.nodes ?? [];
  for (let nodeIndex = 0; nodeIndex < winnerNodeCount; nodeIndex += 1) {
    winnerNodeActivations[nodeIndex] = winnerBirdNodes[nodeIndex]?.activation ?? 0;
  }
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
 * When cross-origin isolation is active, the helper uses `SharedArrayBuffer`
 * backing stores. Those cannot be transferred with `postMessage`, so the transfer
 * list is intentionally empty in that mode; the host already reads the same
 * shared memory.
 *
 * @param snapshot - Packed playback snapshot posted back to the browser host.
 * @returns Transfer list used to move plain `ArrayBuffer` typed-array buffers
 *   without copying. Empty when the snapshot uses `SharedArrayBuffer` backing.
 *
 * @example
 * ```ts
 * const snapshot = createWorkerPlaybackSnapshot(playbackState);
 * const transferList = resolveWorkerPlaybackSnapshotTransferList(snapshot);
 * if (transferList.length > 0) {
 *   postMessage({ type: 'playback-step', payload: snapshot }, transferList);
 * } else {
 *   postMessage({ type: 'playback-step', payload: snapshot });
 * }
 * ```
 */

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
 *
 * @example
 * ```ts
 * const snapshot = createWorkerPlaybackSnapshot(playbackState);
 * const transferList = resolveWorkerPlaybackSnapshotTransferList(snapshot);
 * postMessage({ type: 'playback-step', payload: snapshot }, transferList);
 * ```
 */
export function resolveWorkerPlaybackSnapshotTransferList(
  snapshot: WorkerPlaybackFrameSnapshot,
): Transferable[] {
  const transferCandidates: ArrayBufferLike[] = [
    snapshot.pipes.xPositionsPx.buffer,
    snapshot.pipes.gapCenterYPositionsPx.buffer,
    snapshot.pipes.gapSizesPx.buffer,
    snapshot.birds.yPositionsPx.buffer,
    snapshot.birds.pipesPassed.buffer,
    snapshot.birds.framesSurvived.buffer,
    snapshot.birds.doneFlags.buffer,
    ...(snapshot.winnerNodeActivations
      ? [snapshot.winnerNodeActivations.buffer]
      : []),
  ];
  return transferCandidates.filter(isTransferableArrayBuffer);
}

/**
 * Resolves snapshot column storage for one playback state.
 *
 * @param playbackState - Current mutable playback state.
 * @param pipeCount - Number of visible pipes to pack.
 * @param birdCount - Number of playback birds to pack.
 * @param winnerNodeCount - Number of winner network nodes to pack.
 * @returns Snapshot buffers sized for the current frame.
 */
function resolveWorkerPlaybackSnapshotBuffers(
  playbackState: WorkerPlaybackState,
  pipeCount: number,
  birdCount: number,
  winnerNodeCount: number,
): WorkerPlaybackSnapshotBuffers {
  if (!canReuseSharedSnapshotBuffers()) {
    return createWorkerPlaybackSnapshotBuffers(
      pipeCount,
      birdCount,
      winnerNodeCount,
      false,
    );
  }

  const cachedSnapshotBuffers =
    sharedSnapshotBuffersByPlaybackState.get(playbackState);
  if (
    cachedSnapshotBuffers &&
    cachedSnapshotBuffers.pipeXPositionsPx.length === pipeCount &&
    cachedSnapshotBuffers.birdYPositionsPx.length === birdCount &&
    cachedSnapshotBuffers.winnerNodeActivations.length === winnerNodeCount
  ) {
    return cachedSnapshotBuffers;
  }

  const nextSnapshotBuffers = createWorkerPlaybackSnapshotBuffers(
    pipeCount,
    birdCount,
    winnerNodeCount,
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
 * @param winnerNodeCount - Number of winner network nodes to pack.
 * @param useSharedBuffers - Whether buffers should be reusable shared memory.
 * @returns Snapshot buffer shelf.
 */
function createWorkerPlaybackSnapshotBuffers(
  pipeCount: number,
  birdCount: number,
  winnerNodeCount: number,
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
    winnerNodeActivations: createFloat32SnapshotArray(
      winnerNodeCount,
      useSharedBuffers,
    ),
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
