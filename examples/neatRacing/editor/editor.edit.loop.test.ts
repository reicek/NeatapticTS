/**
 * @jest-environment jsdom
 */

/**
 * RED -> IMPLEMENT -> GREEN contract tests for the Neat Racing editor
 * pointer-pick edit loop (plans/neatracing-voxel-racer.plans.md - phase 1,
 * step 15, slice P1-editor-editloop; spec section 8.3, AC-021 partial,
 * LD-28).
 *
 * The suite pins the slice's acceptance points:
 *
 * - the edit chain: a canvas pointerdown/pointermove picks the voxel under
 *   the pointer, captures the stroke's undo snapshot, paints the
 *   symmetry-canonicalized targets (mirror-X ON by default per
 *   NEATRACING_EDITOR_DEFAULT_MIRROR_X), and routes the edit through the
 *   rebuild pipeline's edit(sliceZ) dirty window;
 * - the pointer-up flush: a stroke that painted ends with the pipeline's
 *   pointerUp() full rebuild, while a stroke that painted nothing never
 *   flushes (no spurious full rebuild);
 * - the pointer pick: the ray cast returns the first OCCUPIED voxel along
 *   the camera ray, falls back to the first cell the ray entered when the
 *   ray crosses no occupied voxel, and reports a miss when the ray leaves
 *   the editor volume;
 * - the working-volume projection: the edited volume projects into a
 *   full-width section 7.2 model document that decodes back to the same
 *   volume, with the document palette index-aligned to the editor palette;
 * - the boot wiring: the editor session publishes an edited document after
 *   a canvas pointerdown/pointerup and detaches the edit loop on stop.
 *
 * The undo snapshot is captured BEFORE the paint of the stroke it covers:
 * the undo stack restores the NEWEST retained snapshot, so a post-paint push
 * would make the stroke's undo a no-op. The plan's chain prose lists the
 * push after the paint; this suite pins the ordering that keeps the undo
 * meaningful (the deviation is documented in the module JSDoc and in the
 * slice's PlanUpdate).
 *
 * RED (pre-implementation): resolving './editor.edit.loop' fails with TS2307
 * ("Cannot find module './editor.edit.loop'") while the module is absent
 * (the batch 90/93 right-reason precedent).
 */

import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import { createEditorCamera } from './editor.camera';
import { createEditorState } from './editor.state';
import type { EditorState } from './editor.state';
import { createEditorSymmetry } from './editor.symmetry';
import { createEditorUndo } from './editor.undo';
import type { EditorUndo } from './editor.undo';
import type { EditorRebuildPipeline } from './editor.rebuild.pipeline';
import type { VoxelPaletteEntry } from '../voxel/model.schema';
import { decodeVoxelModel, decodeVoxelRow } from '../voxel/model.decode.utils';
import { NEATRACING_VOXEL_WORLD_SIZE } from '../voxel/lod.distance.utils';
import {
  NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY,
  buildEditorVolumeDocument,
  createEditorEditLoop,
  createEditorVoxelPicker,
} from './editor.edit.loop';
import type {
  EditorEditLoop,
  EditorPointerPosition,
  EditorVoxelHit,
  EditorVoxelPicker,
} from './editor.edit.loop';

// Side-effect import: evaluating the entry module registers
// `window.neatRacingEditorStart` once on the jsdom host.
import './editor-entry';
import type {
  NeatRacingEditorSession,
  NeatRacingEditorStart,
} from './editor-entry';

/** The preview host id the editor page and the boot contract share. */
const PREVIEW_HOST_ID = 'neatracing-editor-preview';

/** The editor tool frame container id pinned by the shell markup. */
const EDITOR_TOOLS_ID = 'neatracing-editor-tools';

/** Editor volume edge length in cells (the working volume is 128-cubed). */
const VOLUME_EDGE = 128;

/** Elements to skip per step along the volume's y (row) axis. */
const VOLUME_ROW_STRIDE = VOLUME_EDGE;

/** Elements to skip per step along the volume's z (slice) axis. */
const VOLUME_SLICE_STRIDE = VOLUME_EDGE * VOLUME_EDGE;

/** The voxel the picker doubles answer with in the chain lanes. */
const PICKED_VOXEL: EditorVoxelHit = { x: 100, y: 20, z: 30 };

/** The mirror-X canonical half of {@linkcode PICKED_VOXEL} (127 - 100 = 27). */
const CANONICAL_VOXEL: EditorVoxelHit = { x: 27, y: 20, z: 30 };

/** The slice the second pick of the queued-picker lane answers with. */
const SECOND_PICKED_VOXEL: EditorVoxelHit = { x: 100, y: 20, z: 44 };

/** Opaque red paint entry used by the chain and projection lanes. */
const RED_PAINT_ENTRY: VoxelPaletteEntry = [255, 0, 0, 255, 0];

/** The single occupied voxel the picker lanes seed into the volume. */
const OCCUPIED_VOXEL: EditorVoxelHit = { x: 64, y: 64, z: 64 };

/** World size of one editor voxel cell (the 4.5 m car length over 128 cells). */
const VOXEL_WORLD_SIZE = NEATRACING_VOXEL_WORLD_SIZE;

/** The world focus the picker lanes aim at: the center cell of the volume. */
const PICKER_FOCUS: [number, number, number] = [
  0.5 * VOXEL_WORLD_SIZE,
  64.5 * VOXEL_WORLD_SIZE,
  0.5 * VOXEL_WORLD_SIZE,
];

/** The orbit distance the picker lanes place the camera at. */
const PICKER_CAMERA_DISTANCE = 100;

/** The vertical field of view the picker lanes place the camera at (90 degrees). */
const PICKER_CAMERA_FOV_RADIANS = Math.PI / 2;

/** The near clip plane the picker lanes place the camera at. */
const PICKER_CAMERA_NEAR = 0.1;

/** The far clip plane the picker lanes place the camera at. */
const PICKER_CAMERA_FAR = 1000;

/** The square viewport the picker lanes report (CSS pixels and aspect). */
const PICKER_VIEWPORT_SIZE = 400;

/** Pointer event types the edit loop listens to. */
type EditorPointerEventType =
  | 'pointerdown'
  | 'pointermove'
  | 'pointerup'
  | 'pointercancel'
  | 'pointerleave';

/**
 * Creates a picker double answering with one fixed hit (or a miss).
 *
 * @param hit - Voxel the double answers with, or null for a miss.
 * @returns The picker double and its mock.
 */
function createStubPicker(hit: EditorVoxelHit | null): {
  readonly picker: EditorVoxelPicker;
  readonly pick: jest.Mock<
    (position: EditorPointerPosition) => EditorVoxelHit | null
  >;
} {
  const pick = jest.fn<
    (position: EditorPointerPosition) => EditorVoxelHit | null
  >(() => hit);
  return { picker: { pick }, pick };
}

/**
 * Creates a picker double answering the queued hits in order (the last hit
 * repeats once the queue is exhausted).
 *
 * @param hits - Voxels the double answers with, in call order.
 * @returns The picker double and its mock.
 */
function createQueuedPicker(hits: readonly (EditorVoxelHit | null)[]): {
  readonly picker: EditorVoxelPicker;
  readonly pick: jest.Mock<
    (position: EditorPointerPosition) => EditorVoxelHit | null
  >;
} {
  const pick =
    jest.fn<(position: EditorPointerPosition) => EditorVoxelHit | null>();
  for (const hit of hits) {
    pick.mockReturnValueOnce(hit);
  }
  pick.mockReturnValue(hits.at(-1) ?? null);
  return { picker: { pick }, pick };
}

/**
 * Creates an undo double recording the pushed snapshots.
 *
 * @returns The undo double and its push mock.
 */
function createStubUndo(): {
  readonly undo: EditorUndo;
  readonly push: jest.Mock<() => void>;
} {
  const push = jest.fn<() => void>();
  const undo: EditorUndo = {
    push,
    undo: () => false,
    size: () => 0,
    canUndo: () => false,
    bytesInUse: () => 0,
  };
  return { undo, push };
}

/**
 * Creates a rebuild-pipeline double recording the dispatched work.
 *
 * @returns The pipeline double and its edit/pointerUp mocks.
 */
function createStubPipeline(): {
  readonly pipeline: EditorRebuildPipeline;
  readonly edit: jest.Mock<(sliceZ: number) => void>;
  readonly pointerUp: jest.Mock<() => void>;
} {
  const edit = jest.fn<(sliceZ: number) => void>();
  const pointerUp = jest.fn<() => void>();
  const pipeline = {
    edit,
    pointerUp,
    snapshot: () => ({
      inFlight: false,
      queued: false,
      supersededCount: 0,
      durationsMs: [] as number[],
      p95Ms: 0,
    }),
  } as unknown as EditorRebuildPipeline;
  return { pipeline, edit, pointerUp };
}

/**
 * Assembles the edit loop over a fresh canvas, editor state and stub seams.
 *
 * @param picker - Picker the loop resolves the pointer hit through.
 * @returns The harness the chain lanes drive.
 */
function createEditLoopHarness(picker: EditorVoxelPicker): {
  readonly canvas: HTMLCanvasElement;
  readonly state: EditorState;
  readonly loop: EditorEditLoop;
  readonly undo: EditorUndo;
  readonly push: jest.Mock<() => void>;
  readonly edit: jest.Mock<(sliceZ: number) => void>;
  readonly pointerUp: jest.Mock<() => void>;
} {
  const canvas = document.createElement('canvas');
  const state = createEditorState();
  const { undo, push } = createStubUndo();
  const { pipeline, edit, pointerUp } = createStubPipeline();
  const loop = createEditorEditLoop({
    canvas,
    state,
    symmetry: createEditorSymmetry(),
    undo,
    pipeline,
    picker,
  });
  return { canvas, state, loop, undo, push, edit, pointerUp };
}

/**
 * Dispatches a pointer event of the given type at the given client point.
 *
 * @param canvas - Canvas the event is dispatched on.
 * @param type - Pointer event type.
 * @param clientX - Client-space x coordinate of the pointer.
 * @param clientY - Client-space y coordinate of the pointer.
 */
function dispatchPointer(
  canvas: HTMLCanvasElement,
  type: EditorPointerEventType,
  clientX: number,
  clientY: number,
): void {
  canvas.dispatchEvent(
    new MouseEvent(type, { clientX, clientY, bubbles: true }),
  );
}

/**
 * Returns the window global after the entry module side effect ran.
 *
 * @returns The registered editor boot function.
 */
function getEditorStart(): NeatRacingEditorStart {
  const start = window.neatRacingEditorStart;
  if (typeof start !== 'function') {
    throw new TypeError(
      'window.neatRacingEditorStart was not registered by editor-entry.ts',
    );
  }
  return start;
}

/**
 * Returns the preview canvas the booted editor session mounted.
 *
 * @returns The mounted preview canvas.
 */
function getPreviewCanvas(): HTMLCanvasElement {
  const canvas = document.querySelector('canvas');
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new TypeError(
      'the booted editor session did not mount a preview canvas',
    );
  }
  return canvas;
}

/**
 * Collects the painted cells of a working volume in scan order.
 *
 * @param volume - Dense working volume to scan.
 * @returns The occupied cells, ordered by slice, then row, then column.
 */
function collectPaintedCells(
  volume: Uint8Array,
): readonly { x: number; y: number; z: number }[] {
  const cells: { x: number; y: number; z: number }[] = [];
  for (let z = 0; z < VOLUME_EDGE; z += 1) {
    for (let y = 0; y < VOLUME_EDGE; y += 1) {
      for (let x = 0; x < VOLUME_EDGE; x += 1) {
        if (volume[x + VOLUME_ROW_STRIDE * y + VOLUME_SLICE_STRIDE * z] !== 0) {
          cells.push({ x, y, z });
        }
      }
    }
  }
  return cells;
}

/**
 * Builds the camera the picker lanes aim through the volume center.
 *
 * @returns The picker-lane camera.
 */
function createPickerLaneCamera(): ReturnType<typeof createEditorCamera> {
  return createEditorCamera({
    center: PICKER_FOCUS,
    yaw: 0,
    pitch: 0,
    distance: PICKER_CAMERA_DISTANCE,
    fovYRadians: PICKER_CAMERA_FOV_RADIANS,
    near: PICKER_CAMERA_NEAR,
    far: PICKER_CAMERA_FAR,
  });
}

/**
 * Builds the volume the picker lanes seed the occupied voxel into.
 *
 * @returns The seeded working volume.
 */
function createSeededVolume(): Uint8Array {
  const volume = new Uint8Array(VOLUME_EDGE ** 3);
  volume[
    OCCUPIED_VOXEL.x +
      VOLUME_ROW_STRIDE * OCCUPIED_VOXEL.y +
      VOLUME_SLICE_STRIDE * OCCUPIED_VOXEL.z
  ] = 1;
  return volume;
}

describe('Neat Racing editor pointer-pick edit loop (slice P1-editor-editloop)', () => {
  describe('the pick -> symmetry-canonicalized paint -> undo push -> pipeline edit chain', () => {
    it('paints the symmetry-canonicalized mirror-X targets of the picked voxel', () => {
      // Arrange - a picked voxel above the mirror line (100 > 63).
      const { picker } = createStubPicker(PICKED_VOXEL);
      const harness = createEditLoopHarness(picker);
      const paintSpy = jest.spyOn(harness.state, 'paint');
      // Act - one pointerdown paints the stroke.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      // Assert - the canonical half is painted first, the mirror second.
      expect(
        paintSpy.mock.calls.map((call) => [call[0], call[1], call[2]]),
      ).toEqual([
        [CANONICAL_VOXEL.x, CANONICAL_VOXEL.y, CANONICAL_VOXEL.z],
        [PICKED_VOXEL.x, PICKED_VOXEL.y, PICKED_VOXEL.z],
      ]);
    });

    it('captures the undo snapshot before the paint of the stroke', () => {
      // Arrange - a shared call-order recorder over the undo push and paint.
      const callOrder: string[] = [];
      const undo: EditorUndo = {
        push: () => {
          callOrder.push('undo.push');
        },
        undo: () => false,
        size: () => 0,
        canUndo: () => false,
        bytesInUse: () => 0,
      };
      const canvas = document.createElement('canvas');
      const state = createEditorState();
      jest.spyOn(state, 'paint').mockImplementation(() => {
        callOrder.push('state.paint');
      });
      const { picker } = createStubPicker(PICKED_VOXEL);
      const { pipeline } = createStubPipeline();
      const loop = createEditorEditLoop({
        canvas,
        state,
        symmetry: createEditorSymmetry(),
        undo,
        pipeline,
        picker,
      });
      // Act - one pointerdown paints the stroke.
      loop.attach();
      dispatchPointer(canvas, 'pointerdown', 12, 34);
      // Assert - the snapshot precedes the first paint.
      expect(callOrder.slice(0, 2)).toEqual(['undo.push', 'state.paint']);
    });

    it('pushes exactly one undo snapshot across a whole pointer stroke', () => {
      // Arrange - a stroke of one pointerdown and two pointermoves.
      const { picker } = createStubPicker(PICKED_VOXEL);
      const harness = createEditLoopHarness(picker);
      // Act - the stroke runs to its pointermove end.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      dispatchPointer(harness.canvas, 'pointermove', 20, 40);
      dispatchPointer(harness.canvas, 'pointermove', 28, 46);
      // Assert - one snapshot covers the whole stroke.
      expect(harness.push).toHaveBeenCalledTimes(1);
    });

    it('routes every painted edit through the rebuild pipeline edit(sliceZ)', () => {
      // Arrange - the second pick lands on another slice.
      const { picker } = createQueuedPicker([
        PICKED_VOXEL,
        SECOND_PICKED_VOXEL,
      ]);
      const harness = createEditLoopHarness(picker);
      // Act - a pointerdown and one pointermove.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      dispatchPointer(harness.canvas, 'pointermove', 20, 40);
      // Assert - each edit carries the picked slice.
      expect(harness.edit.mock.calls).toEqual([
        [PICKED_VOXEL.z],
        [SECOND_PICKED_VOXEL.z],
      ]);
    });

    it('picks with the pointer position mapped from the client point into the canvas box', () => {
      // Arrange - a pointer down away from the canvas origin.
      const { picker, pick } = createStubPicker(PICKED_VOXEL);
      const harness = createEditLoopHarness(picker);
      // Act - one pointerdown.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      // Assert - the client point is the canvas-local position.
      expect(pick).toHaveBeenCalledWith({ x: 12, y: 34 });
    });

    it('flushes the rebuild pipeline with a full rebuild when the stroke ends', () => {
      // Arrange - a stroke that paints.
      const { picker } = createStubPicker(PICKED_VOXEL);
      const harness = createEditLoopHarness(picker);
      // Act - the stroke ends with a pointerup.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      dispatchPointer(harness.canvas, 'pointerup', 12, 34);
      // Assert - the pointer-up flush ran exactly once.
      expect(harness.pointerUp).toHaveBeenCalledTimes(1);
    });

    it('does not flush the rebuild pipeline when the stroke painted nothing', () => {
      // Arrange - a picker that never hits a voxel.
      const { picker } = createStubPicker(null);
      const harness = createEditLoopHarness(picker);
      // Act - a stroke over empty space.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      dispatchPointer(harness.canvas, 'pointerup', 12, 34);
      // Assert - no full rebuild was flushed.
      expect(harness.pointerUp).not.toHaveBeenCalled();
    });

    it('paints the default white entry when the caller names no paint color', () => {
      // Arrange - a picked voxel and the default paint entry.
      const { picker } = createStubPicker(PICKED_VOXEL);
      const harness = createEditLoopHarness(picker);
      const paintSpy = jest.spyOn(harness.state, 'paint');
      // Act - one pointerdown.
      harness.loop.attach();
      dispatchPointer(harness.canvas, 'pointerdown', 12, 34);
      // Assert - the default entry reached the volume paint path.
      expect(paintSpy.mock.calls[0]?.[3]).toEqual(
        NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY,
      );
    });
  });

  describe('the editor voxel picker (camera ray cast)', () => {
    it('returns the first occupied voxel along the camera ray', () => {
      // Arrange - one occupied voxel on the camera axis.
      const picker = createEditorVoxelPicker({
        volume: createSeededVolume(),
        camera: createPickerLaneCamera(),
        readViewport: () => ({
          cssWidth: PICKER_VIEWPORT_SIZE,
          cssHeight: PICKER_VIEWPORT_SIZE,
          aspect: 1,
        }),
      });
      // Act - the pointer sits in the middle of the viewport.
      const hit = picker.pick({
        x: PICKER_VIEWPORT_SIZE / 2,
        y: PICKER_VIEWPORT_SIZE / 2,
      });
      // Assert - the occupied voxel is the hit.
      expect(hit).toEqual({
        x: OCCUPIED_VOXEL.x,
        y: OCCUPIED_VOXEL.y,
        z: OCCUPIED_VOXEL.z,
      });
    });

    it('falls back to the first cell the ray entered when it crosses no occupied voxel', () => {
      // Arrange - the same camera ray over an empty volume.
      const picker = createEditorVoxelPicker({
        volume: new Uint8Array(VOLUME_EDGE ** 3),
        camera: createPickerLaneCamera(),
        readViewport: () => ({
          cssWidth: PICKER_VIEWPORT_SIZE,
          cssHeight: PICKER_VIEWPORT_SIZE,
          aspect: 1,
        }),
      });
      // Act - the pointer sits in the middle of the viewport.
      const hit = picker.pick({
        x: PICKER_VIEWPORT_SIZE / 2,
        y: PICKER_VIEWPORT_SIZE / 2,
      });
      // Assert - the front-most cell of the volume along the ray.
      expect(hit).toEqual({ x: 64, y: 64, z: 127 });
    });

    it('reports a miss when the camera ray leaves the editor volume', () => {
      // Arrange - a camera aimed away from the volume.
      const camera = createEditorCamera({
        center: [1000, 1000, 1000],
        yaw: 0,
        pitch: 0,
        distance: PICKER_CAMERA_DISTANCE,
        fovYRadians: PICKER_CAMERA_FOV_RADIANS,
        near: PICKER_CAMERA_NEAR,
        far: PICKER_CAMERA_FAR,
      });
      const picker = createEditorVoxelPicker({
        volume: new Uint8Array(VOLUME_EDGE ** 3),
        camera,
        readViewport: () => ({
          cssWidth: PICKER_VIEWPORT_SIZE,
          cssHeight: PICKER_VIEWPORT_SIZE,
          aspect: 1,
        }),
      });
      // Act - the pointer sits in the middle of the viewport.
      const hit = picker.pick({
        x: PICKER_VIEWPORT_SIZE / 2,
        y: PICKER_VIEWPORT_SIZE / 2,
      });
      // Assert - the ray never enters the volume.
      expect(hit).toBeNull();
    });
  });

  describe('the working-volume projection into a section 7.2 model document', () => {
    it('projects the working volume into a full-width document that decodes back to it', () => {
      // Arrange - two painted cells in one row.
      const state = createEditorState();
      state.paint(3, 4, 5, RED_PAINT_ENTRY);
      state.paint(120, 4, 5, RED_PAINT_ENTRY);
      // Act - the volume projects and decodes.
      const document = buildEditorVolumeDocument({
        volume: state.volume,
        palette: state.palette,
        name: 'car-f1',
      });
      const decoded = decodeVoxelModel(document);
      const row = decoded.slices[5].slice(4 * VOLUME_EDGE, 5 * VOLUME_EDGE);
      const occupiedColumns = Array.from(row).flatMap((value, column) =>
        value === 0 ? [] : [column],
      );
      // Assert - the painted columns survived the round trip.
      expect(occupiedColumns).toEqual([3, 120]);
    });

    it('keeps the document palette index-aligned with the editor palette', () => {
      // Arrange - one painted color in the editor palette.
      const state = createEditorState();
      state.paint(1, 1, 1, RED_PAINT_ENTRY);
      // Act - the volume projects.
      const document = buildEditorVolumeDocument({
        volume: state.volume,
        palette: state.palette,
      });
      // Assert - index 0 is the EMPTY sentinel, index 1 the painted color.
      expect(document.palette).toEqual([
        [0, 0, 0, 0, 0],
        [255, 0, 0, 255, 0],
      ]);
    });

    it('declares the full 128-cubed working volume as a non-symmetric document', () => {
      // Arrange - an untouched working volume.
      const state = createEditorState();
      // Act - the volume projects.
      const document = buildEditorVolumeDocument({
        volume: state.volume,
        palette: state.palette,
      });
      // Assert - the editor stores the full width, so the rows are not halved.
      expect(document.meta.dims).toEqual([
        VOLUME_EDGE,
        VOLUME_EDGE,
        VOLUME_EDGE,
      ]);
      expect(document.meta.symmetric).toBe(false);
    });
  });

  describe('the editor boot wiring', () => {
    beforeEach(() => {
      document.body.innerHTML = `
        <main id="neatracing-editor">
          <section id="${PREVIEW_HOST_ID}" aria-label="3D model preview"></section>
          <aside id="${EDITOR_TOOLS_ID}" aria-label="Editor tools"></aside>
        </main>
      `;
      // jsdom does not ship a WebGL implementation: stub the canvas context
      // so the boot path exercises the context service's fallback branch.
      jest
        .spyOn(HTMLCanvasElement.prototype, 'getContext')
        .mockReturnValue(null);
    });

    it('publishes an edited document after a canvas pointerdown and pointerup', async () => {
      // Arrange - a booted editor session with its mounted canvas.
      const session: NeatRacingEditorSession =
        getEditorStart()(PREVIEW_HOST_ID);
      const canvas = getPreviewCanvas();
      // Act - a pointer stroke in the middle of the canvas, then let the
      // queued full rebuild settle before reading the published document.
      dispatchPointer(canvas, 'pointerdown', 0.5, 0.5);
      dispatchPointer(canvas, 'pointerup', 0.5, 0.5);
      await new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      });
      // Assert - the flushed full rebuild published the projected document
      // and the painted cells survived the projection.
      const document = session.editedDocument;
      if (document === null) {
        throw new Error(
          'the flushed full rebuild published no edited document',
        );
      }
      expect(document.meta.dims).toEqual([
        VOLUME_EDGE,
        VOLUME_EDGE,
        VOLUME_EDGE,
      ]);
      const paintedCells = collectPaintedCells(session.toolframe.state.volume);
      const paintedColumns = paintedCells.map(
        (cell) =>
          decodeVoxelRow(
            document.slices[cell.z][cell.y],
            VOLUME_EDGE,
            document.palette.length,
          )[cell.x],
      );
      expect(paintedColumns.length).toBeGreaterThan(0);
      expect(paintedColumns).not.toContain(0);
    });

    it('detaches the edit loop on stop so a later pointer event paints nothing', () => {
      // Arrange - a booted editor session that is stopped.
      const session: NeatRacingEditorSession =
        getEditorStart()(PREVIEW_HOST_ID);
      const canvas = getPreviewCanvas();
      session.stop();
      // Act - a pointer stroke after the stop.
      dispatchPointer(canvas, 'pointerdown', 0.5, 0.5);
      let paintedCells = 0;
      for (const value of session.toolframe.state.volume) {
        if (value !== 0) {
          paintedCells += 1;
        }
      }
      // Assert - the stopped session painted nothing.
      expect(paintedCells).toBe(0);
    });
  });

  describe('the undo stack integration', () => {
    it('restores the pre-stroke volume when the stroke is undone', () => {
      // Arrange - the real undo stack over the real working volume.
      const canvas = document.createElement('canvas');
      const state = createEditorState();
      const undo = createEditorUndo(state.volume);
      const { picker } = createStubPicker(PICKED_VOXEL);
      const { pipeline } = createStubPipeline();
      const loop = createEditorEditLoop({
        canvas,
        state,
        symmetry: createEditorSymmetry(),
        undo,
        pipeline,
        picker,
      });
      // Act - a stroke paints, then the undo restores the pre-stroke volume.
      loop.attach();
      dispatchPointer(canvas, 'pointerdown', 12, 34);
      undo.undo();
      // Assert - the mirror-X pair is gone again.
      expect(
        state.volume[
          CANONICAL_VOXEL.x +
            VOLUME_ROW_STRIDE * CANONICAL_VOXEL.y +
            VOLUME_SLICE_STRIDE * CANONICAL_VOXEL.z
        ],
      ).toBe(0);
      expect(
        state.volume[
          PICKED_VOXEL.x +
            VOLUME_ROW_STRIDE * PICKED_VOXEL.y +
            VOLUME_SLICE_STRIDE * PICKED_VOXEL.z
        ],
      ).toBe(0);
    });
  });
});
