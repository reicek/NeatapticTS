/**
 * RED contract for slice P1-editor-preview (Step 14; AC-019 preview lane): a RAF-driven editor
 * model preview render loop reusing decodeVoxelModel + meshVoxelModel + a bins draw under the
 * Step 08 editor camera.
 *
 * RED PREMISE: while ./editor-preview.render is intentionally absent, this suite must fail to
 * run with exactly one compile error: `error TS2307: Cannot find module
 * './editor-preview.render' or its corresponding type declarations.` (batch 90/93 precedent;
 * evidence batch 106). Any fixture/import-typo failure is NOT a valid red.
 *
 * GREEN TARGET — 04-implementing must make these pass:
 * 1. startEditorPreviewRenderLoop(config) -> { stop() }; config = canvas (structural { width,
 *    height } use only), camera, getSelectedModelDocument, binsDraw (a draw-only seam over
 *    { viewProjection = P*V, cameraWorld = orbit pose eye, bins = VoxelBinModel }), and
 *    requestFrame/cancelFrame seams defaulting to the window RAF pair.
 * 2. Per-RAF: dt = 0 first frame else (now - previous)/1000 -> camera.tick(dt) ->
 *    buildViewMatrix() + buildProjectionMatrix(canvas.width / canvas.height) -> column-major
 *    P*V -> draw; cameraWorld = center + distance * [cosPitch*sinYaw, sinPitch, cosPitch*cosYaw].
 * 3. bins = decodeVoxelModel(document) -> meshVoxelModel(decoded) bridged as
 *    { dims: mesh.dims, palette: decoded.palette, quads: mesh.quads } (VoxelBinQuad is a
 *    structural superset of the mesher's VoxelQuad).
 * 4. Null document: tick + RAF chain keep running, no draws; 5. stop() cancels the pending
 *    frame and disposes exactly once, repeats are no-ops; 6. preview-only lane, injected seams
 *    only (production wiring belongs in editor-entry).
 *
 * Fixture discipline: deterministic pure JS (no RNG, timers, DOM); fresh harness per `it` plus
 * JS GC; matrices column-major Float32Array(16), Y-up.
 */

import { describe, expect, it } from '@jest/globals';
import {
  startEditorPreviewRenderLoop,
  type EditorPreviewBinsDrawState,
} from './editor-preview.render';
import { createEditorCamera, type EditorCameraPose } from './editor.camera';
import { decodeVoxelModel } from '../voxel/model.decode.utils';
import { meshVoxelModel } from '../voxel/greedy-mesher';
import type {
  VoxelModelDocument,
  VoxelPaletteEntry,
} from '../voxel/model.schema';
import type { VoxelBinModel } from '../voxel/bin.codec';

const VIEW_PROJECTION_CLOSENESS_DIGITS = 6;

/** RLE run: [[count, paletteIndex], ...] packed as zero-padded pairs per the voxel tests. */
function rleRow(runs: number[][]): string {
  return runs
    .map(
      ([count, value]) =>
        String(count).padStart(2, '0') + String(value).padStart(2, '0'),
    )
    .join('');
}

/** 2x2x1 red slab: palette index 0 = EMPTY sentinel, index 1 = opaque red body voxel. */
const FIXTURE_DOC: VoxelModelDocument = {
  meta: { name: 'preview-fixture', dims: [2, 2, 2], symmetric: false },
  palette: [
    [0, 0, 0, 0, 0],
    [255, 0, 0, 255, 0],
  ],
  slices: [
    [rleRow([[2, 1]]), rleRow([[2, 1]])],
    [rleRow([[2, 0]]), rleRow([[2, 0]])],
  ],
};

/** Reference decode -> greedy-mesh chain: the exact bridge the production loop dogfoods. */
function referenceBins(document: VoxelModelDocument): VoxelBinModel {
  const decoded = decodeVoxelModel(document);
  const mesh = meshVoxelModel(decoded);
  return {
    dims: mesh.dims,
    palette: decoded.palette,
    quads: mesh.quads as unknown as VoxelBinModel['quads'],
  };
}

/** Column-major product P*V: out[column * 4 + row] = sum_k projection[k*4+row] * view[column*4+k]. */
function multiplyViewProjection(
  view: Float32Array,
  projection: Float32Array,
): Float32Array {
  const product = new Float32Array(16);
  for (let columnIndex = 0; columnIndex < 4; columnIndex += 1) {
    for (let rowIndex = 0; rowIndex < 4; rowIndex += 1) {
      let accumulator = 0;
      for (let termIndex = 0; termIndex < 4; termIndex += 1) {
        accumulator +=
          projection[termIndex * 4 + rowIndex] *
          view[columnIndex * 4 + termIndex];
      }
      product[columnIndex * 4 + rowIndex] = accumulator;
    }
  }
  return product;
}

/** Documented buildViewMatrix eye derivation for the current orbit pose. */
function referenceCameraEye(pose: EditorCameraPose): [number, number, number] {
  return [
    pose.center[0] + pose.distance * Math.cos(pose.pitch) * Math.sin(pose.yaw),
    pose.center[1] + pose.distance * Math.sin(pose.pitch),
    pose.center[2] + pose.distance * Math.cos(pose.pitch) * Math.cos(pose.yaw),
  ];
}

/** Deep snapshot of a bins draw state, captured synchronously at draw time. */
function cloneDrawState(state: EditorPreviewBinsDrawState) {
  return {
    viewProjection: new Float32Array(state.viewProjection),
    cameraWorld: [
      state.cameraWorld[0],
      state.cameraWorld[1],
      state.cameraWorld[2],
    ],
    bins: {
      dims: state.bins.dims,
      palette: state.bins.palette as unknown as VoxelPaletteEntry[],
      quads: state.bins.quads.map((quad: VoxelBinModel['quads'][number]) => ({
        position: [quad.position[0], quad.position[1], quad.position[2]],
        normalAxis: quad.normalAxis,
        width: quad.width,
        height: quad.height,
        paletteIndex: quad.paletteIndex,
      })),
    },
  };
}

/** Harness: hand-rolled RAF queue (schedule ids from 1; no jest.fn); pump() drains then invokes FIFO. */
function makeHarness(
  options: {
    document: VoxelModelDocument | null;
    getDocument?: () => VoxelModelDocument | null;
  } = { document: FIXTURE_DOC },
) {
  const canvas = { width: 1280, height: 720 } as unknown as HTMLCanvasElement;
  const camera = createEditorCamera();
  const referenceCamera = createEditorCamera();
  const pendingFrames: {
    frameId: number;
    callback: (timeMs: number) => void;
  }[] = [];
  const cancelRecords: number[] = [];
  const drawRecords: ReturnType<typeof cloneDrawState>[] = [];
  const disposeCount = { value: 0 };
  let nextFrameId = 1;
  const requestFrame = (callback: (timeMs: number) => void): number => {
    const frameId = nextFrameId;
    nextFrameId += 1;
    pendingFrames.push({ frameId, callback });
    return frameId;
  };
  const cancelFrame = (frameId: number): void => {
    cancelRecords.push(frameId);
    const pendingIndex = pendingFrames.findIndex(
      (pending) => pending.frameId === frameId,
    );
    if (pendingIndex >= 0) {
      pendingFrames.splice(pendingIndex, 1);
    }
  };
  const binsDraw = {
    draw: (state: EditorPreviewBinsDrawState) =>
      drawRecords.push(cloneDrawState(state)),
    dispose: () => {
      disposeCount.value += 1;
    },
  };
  return {
    camera,
    referenceCamera,
    cancelRecords,
    drawRecords,
    disposeCount,
    pendingCount: (): number => pendingFrames.length,
    runLoop: (): ReturnType<typeof startEditorPreviewRenderLoop> =>
      startEditorPreviewRenderLoop({
        canvas,
        camera,
        getSelectedModelDocument: (): VoxelModelDocument | null =>
          options.getDocument !== undefined
            ? options.getDocument()
            : options.document,
        binsDraw,
        requestFrame,
        cancelFrame,
      }),
    pump: (timeMs: number): void => {
      pendingFrames
        .splice(0, pendingFrames.length)
        .forEach((frame) => frame.callback(timeMs));
    },
  };
}

function runMirroredFirstFrame(): ReturnType<typeof makeHarness> {
  const harness = makeHarness();
  harness.camera.orbit(0.5, 0);
  harness.referenceCamera.orbit(0.5, 0);
  harness.runLoop();
  harness.pump(16);
  harness.referenceCamera.tick(0);
  return harness;
}

/** Per-element close match, mirroring editor.camera.test.ts expectMatrixCloseTo. */
function expectViewProjectionCloseTo(
  actual: Float32Array,
  expected: Float32Array,
): void {
  for (let entryIndex = 0; entryIndex < 16; entryIndex += 1) {
    expect(actual[entryIndex]).toBeCloseTo(
      expected[entryIndex],
      VIEW_PROJECTION_CLOSENESS_DIGITS,
    );
  }
}

function expectCameraWorldCloseTo(
  actual: readonly [number, number, number],
  expected: readonly [number, number, number],
): void {
  expect(actual[0]).toBeCloseTo(expected[0], VIEW_PROJECTION_CLOSENESS_DIGITS);
  expect(actual[1]).toBeCloseTo(expected[1], VIEW_PROJECTION_CLOSENESS_DIGITS);
  expect(actual[2]).toBeCloseTo(expected[2], VIEW_PROJECTION_CLOSENESS_DIGITS);
}

/** A document the shared validator rejects (zero extent) — decode path throws. */
const REJECTED_DOC: VoxelModelDocument = {
  meta: {
    name: 'broken-preview-model',
    dims: [0, 2, 1],
    symmetric: false,
  },
  palette: [[0, 0, 0, 0, 0]],
  slices: [['0101']],
} as unknown as VoxelModelDocument;

/**
 * Records console.error calls for one test (hand-rolled spy; no jest.fn in
 * this fixture). Call `restore()` in a `finally` so a failing assertion
 * still cleans up after the global console.
 *
 * @returns The recorded call lists plus the restore thunk.
 */
function captureConsoleError(): {
  records: unknown[][];
  restore: () => void;
} {
  const records: unknown[][] = [];
  const original = console.error;
  console.error = (...args: unknown[]): void => {
    records.push(args);
  };
  return {
    records,
    restore: (): void => {
      console.error = original;
    },
  };
}

describe('editor-preview.render (P1-editor-preview; spec AC-019)', () => {
  describe('RAF-driven decode -> greedy mesher -> bins draw chain', () => {
    it('renders the real decoded and greedy-meshed bins of the selected model on the first RAF frame', () => {
      const harness = runMirroredFirstFrame();
      expect(harness.drawRecords).toHaveLength(1);
      expect(harness.drawRecords[0].bins).toEqual(referenceBins(FIXTURE_DOC));
    });
    it('feeds the first draw the editor camera view-projection product and the orbit eye point', () => {
      const harness = runMirroredFirstFrame();
      const expectedProduct = multiplyViewProjection(
        harness.referenceCamera.buildViewMatrix(),
        harness.referenceCamera.buildProjectionMatrix(1280 / 720),
      );
      const expectedEye = referenceCameraEye(
        harness.referenceCamera.snapshot().current,
      );
      expectViewProjectionCloseTo(
        harness.drawRecords[0].viewProjection,
        expectedProduct,
      );
      expectCameraWorldCloseTo(
        harness.drawRecords[0].cameraWorld as [number, number, number],
        expectedEye,
      );
    });
    it('eases the Step 08 camera transform into the second RAF frame draw', () => {
      const harness = runMirroredFirstFrame();
      harness.pump(32);
      harness.referenceCamera.tick(0.016);
      const expectedProduct = multiplyViewProjection(
        harness.referenceCamera.buildViewMatrix(),
        harness.referenceCamera.buildProjectionMatrix(1280 / 720),
      );
      expect(harness.drawRecords).toHaveLength(2);
      expectViewProjectionCloseTo(
        harness.drawRecords[1].viewProjection,
        expectedProduct,
      );
      expect(harness.referenceCamera.snapshot().current.yaw).toBeGreaterThan(0);
    });
  });
  describe('preview-only lane policy', () => {
    it('keeps the RAF chain ticking with zero draws while no model document is selected', () => {
      // Arrange / Act — nothing selected; two frames through the idle lane.
      const harness = makeHarness({ document: null });
      harness.runLoop();
      harness.pump(16);
      harness.pump(32);
      expect(harness.drawRecords).toHaveLength(0);
      expect(harness.pendingCount()).toBe(1);
    });
  });
  describe('stop contract', () => {
    it('cancels the pending frame and disposes the bins draw handle exactly once on stop', () => {
      const harness = makeHarness();
      const handle = harness.runLoop();
      handle.stop();
      expect(harness.cancelRecords).toEqual([1]);
      expect(harness.disposeCount.value).toBe(1);
    });
    it('stops drawing after stop and treats repeated stop calls as no-ops', () => {
      const harness = makeHarness();
      const handle = harness.runLoop();
      handle.stop();
      handle.stop();
      harness.pump(48);
      expect(harness.disposeCount.value).toBe(1);
      expect(harness.drawRecords).toHaveLength(0);
      expect(harness.cancelRecords).toEqual([1]);
    });
  });
  describe('observable decode/mesh failure surface (fix-packet-P1-composition-green)', () => {
    it('surfaces a rejected document on the preview label exactly once and keeps the RAF alive', () => {
      // Arrange — a document the shared validator rejects; the failure must
      // be loud (previously a byte-silent degrade), fire exactly once per
      // document identity (memoized null bins), and keep the RAF chain alive.
      const consoleSurface = captureConsoleError();
      try {
        const harness = makeHarness({ document: REJECTED_DOC });
        // Act — two frames; only the first may resolve (and fail) the bins.
        harness.runLoop();
        harness.pump(16);
        harness.pump(32);
        // Assert — labeled surface, once; lane still ticking; zero draws.
        expect(consoleSurface.records).toHaveLength(1);
        expect(String(consoleSurface.records[0][0])).toContain(
          'neatRacing editor preview',
        );
        expect(String(consoleSurface.records[0][0])).toContain(
          'selected model decode/mesh failed; preview renders nothing',
        );
        expect(harness.drawRecords).toHaveLength(0);
        expect(harness.pendingCount()).toBe(1);
      } finally {
        consoleSurface.restore();
      }
    });
    it('emits no failure surface while the selection decodes and draws', () => {
      // Arrange / Act — healthy document through two frames.
      const consoleSurface = captureConsoleError();
      try {
        const harness = makeHarness();
        harness.runLoop();
        harness.pump(16);
        harness.pump(32);
        // Assert — draws happen, RAF chain alive, console untouched.
        expect(harness.drawRecords).toHaveLength(2);
        expect(harness.pendingCount()).toBe(1);
        expect(consoleSurface.records).toHaveLength(0);
      } finally {
        consoleSurface.restore();
      }
    });
    it('recovers drawing after switching from a rejected document to a healthy one', () => {
      // Arrange — provider starts rejected, then flips to a healthy doc.
      const consoleSurface = captureConsoleError();
      let current: VoxelModelDocument | null = REJECTED_DOC;
      try {
        const harness = makeHarness({
          document: FIXTURE_DOC,
          getDocument: () => current,
        });
        // Act — frame one fails; the switch lands; frame two draws.
        harness.runLoop();
        harness.pump(16);
        current = FIXTURE_DOC;
        harness.pump(32);
        // Assert — exactly the earlier failure, then a healthy draw, and no
        // new error after recovery.
        expect(consoleSurface.records).toHaveLength(1);
        expect(harness.drawRecords).toHaveLength(1);
        expect(harness.drawRecords[0].bins).toEqual(referenceBins(FIXTURE_DOC));
        expect(harness.pendingCount()).toBe(1);
      } finally {
        consoleSurface.restore();
      }
    });
  });
});
