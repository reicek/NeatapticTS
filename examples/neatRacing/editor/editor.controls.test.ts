/**
 * @jest-environment jsdom
 */

/**
 * RED → IMPLEMENT → GREEN contract tests for the Neat Racing editor
 * controls (plans/neatracing-voxel-racer.plans.md — phase 1, step 15,
 * slice P1-editor-controls; spec section 8.3, AC-020 completion + AC-019
 * CAD views, LD-28).
 *
 * The suite pins the slice's acceptance points:
 *
 * - the palette add/select handlers: a new color resolves through
 *   `createEditorPalette` under `NEATRACING_EDITOR_PALETTE_MAX_ENTRIES`
 *   (a full palette resolves the nearest existing entry instead of
 *   allocating a slot), the added entry is rendered into the mounted
 *   palette list, and the selection drives the shared paint entry;
 * - the undo/redo buttons: the undo button restores the previous working
 *   volume through the bounded `createEditorUndo` stack (the oldest
 *   snapshot is evicted beyond the 3-step cap), and the redo button
 *   re-applies the undone state through the SAME snapshot mechanism;
 * - the model dropdown: a `change` loads the selected model from the
 *   model store, decodes it into the working volume, and selects it in the
 *   editor state;
 * - the CAD-view buttons: the front/side/top buttons snap the editor
 *   camera to the CAD orthographic views and the orbit button returns to
 *   the perspective view.
 *
 * The boot lane pins the editor-entry wiring: the session exposes the
 * wired controls and the framed camera, the boot publishes the
 * WORKING-VOLUME projection the pointer picker inverts (review obs 1),
 * and the selected palette entry is the color the pointer paint writes.
 *
 * RED (pre-implementation): resolving './editor.controls' fails with
 * TS2307 ("Cannot find module './editor.controls'") while the module is
 * absent (batch 90/93 right-reason precedent).
 */

import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import {
  NEATRACING_EDITOR_CONTROL_IDS,
  createEditorControls,
} from './editor.controls';
import type { EditorControls } from './editor.controls';
import { createEditorCamera } from './editor.camera';
import type { EditorCamera } from './editor.camera';
import {
  NEATRACING_EDITOR_TOOLS_ID,
  createEditorToolframe,
} from './editor.toolframe';
import type {
  EditorToolframeFetch,
  EditorToolframeSession,
} from './editor.toolframe';
import { NEATRACING_EDITOR_PALETTE_MAX_ENTRIES } from './editor.palette';
import { NEATRACING_EDITOR_UNDO_MAX_STEPS } from './editor.undo';
import { NEATRACING_EDITOR_VOLUME_EDGE } from './editor.state';

// Side-effect import: evaluating the entry module registers
// `window.neatRacingEditorStart` once on the jsdom host (the boot lane
// below drives it).
import './editor-entry';
import type {
  NeatRacingEditorBootDeps,
  NeatRacingEditorSession,
} from './editor-entry';

/** The reserved tool-frame container id pinned by the editor shell markup. */
const EDITOR_TOOLS_ID = NEATRACING_EDITOR_TOOLS_ID;

/** The preview host id the editor boot resolves before the tool frame. */
const PREVIEW_HOST_ID = 'neatracing-editor-preview';

/** Dev-server model list route the tool frame reads the model names from. */
const MODEL_LIST_PATH = '/api/models';

/** First model name the model-store double serves. */
const FIRST_MODEL_NAME = 'car-f1';

/** Second model name the model-store double serves (the dropdown reload). */
const SECOND_MODEL_NAME = 'car-f2';

/**
 * A minimal valid section 7.2 model document: two columns of palette
 * index 1 in a single 2x1x1 slice (row hex `0201` = run length 2 at
 * palette index 1).
 */
const FIRST_MODEL_DOCUMENT = {
  meta: { name: FIRST_MODEL_NAME, dims: [2, 1, 1], symmetric: false },
  palette: [
    [0, 0, 0, 0, 0],
    [255, 0, 0, 255, 0],
  ],
  slices: [['0201']],
};

/**
 * The second served document: ONE column of palette index 1 in a single
 * 1x1x1 slice (row hex `0101`), so the reloaded working volume is
 * distinguishable from the first load (the second column is cleared).
 */
const SECOND_MODEL_DOCUMENT = {
  meta: { name: SECOND_MODEL_NAME, dims: [1, 1, 1], symmetric: false },
  palette: [
    [0, 0, 0, 0, 0],
    [0, 0, 255, 255, 0],
  ],
  slices: [['0101']],
};

/** Paint entry used by the palette lanes (opaque red). */
const RED_ENTRY = [255, 0, 0, 255, 0] as const;

/** Second paint entry used by the undo lanes (opaque green). */
const GREEN_ENTRY = [0, 255, 0, 255, 0] as const;

/**
 * Builds a model-store fetch double: the list route answers with the
 * model names, the named route answers with that model's document.
 *
 * @param models - Model names the list route reports.
 * @returns A fetch seam the tool frame can drive.
 */
function createModelStoreFetch(
  models: readonly string[] = [FIRST_MODEL_NAME, SECOND_MODEL_NAME],
): EditorToolframeFetch {
  return async (url: string) => {
    if (url.endsWith(MODEL_LIST_PATH)) {
      return { ok: true, json: async () => ({ models }) };
    }
    const modelName = decodeURIComponent(url.slice(`${MODEL_LIST_PATH}/`.length));
    const document =
      modelName === SECOND_MODEL_NAME
        ? SECOND_MODEL_DOCUMENT
        : FIRST_MODEL_DOCUMENT;
    return { ok: true, json: async () => ({ document }) };
  };
}

/**
 * Builds a model-store fetch double whose named read FAILS, so the
 * dropdown lane can observe the degraded load.
 *
 * @returns A fetch seam whose model read always fails.
 */
function createFailingModelStoreFetch(): EditorToolframeFetch {
  return async (url: string) => {
    if (url.endsWith(MODEL_LIST_PATH)) {
      return { ok: true, json: async () => ({ models: [FIRST_MODEL_NAME] }) };
    }
    return { ok: false, status: 500, json: async () => ({}) };
  };
}

/** Mounts the editor shell markup the tool frame and the boot resolve. */
function mountEditorShell(): void {
  document.body.innerHTML = `
    <main id="neatracing-editor">
      <section id="${PREVIEW_HOST_ID}" aria-label="3D model preview"></section>
      <aside id="${EDITOR_TOOLS_ID}" aria-label="Editor tools"></aside>
    </main>
  `;
}

/** The wired controls plus the tool frame and camera they drive. */
interface ControlsHarness {
  /** The wired control surface under test. */
  controls: EditorControls;
  /** The tool frame whose controls the module binds. */
  toolframe: EditorToolframeSession;
  /** The camera the CAD-view buttons drive. */
  camera: EditorCamera;
}

/**
 * Boots a tool frame, a camera, and the wired controls over them.
 *
 * The initial model load is awaited so the palette and the working volume
 * are deterministic when a lane asserts on palette indices.
 *
 * @returns The wired controls plus the tool frame and camera they drive.
 */
async function bootControls(): Promise<ControlsHarness> {
  const toolframe = createEditorToolframe({
    fetchImpl: createModelStoreFetch(),
  });
  const camera = createEditorCamera();
  const controls = createEditorControls({ toolframe, camera });
  await toolframe.ready;
  return { controls, toolframe, camera };
}

/**
 * Boots the real editor entry against the mounted shell.
 *
 * @param deps - Optional boot dependencies (the model-store fetch seam).
 * @returns The booted editor session.
 */
function bootEditorSession(
  deps?: NeatRacingEditorBootDeps,
): NeatRacingEditorSession {
  const start = window.neatRacingEditorStart;
  if (typeof start !== 'function') {
    throw new TypeError(
      'window.neatRacingEditorStart was not registered by editor-entry.ts',
    );
  }
  return start(PREVIEW_HOST_ID, deps);
}

/** Counts the occupied cells of the working volume. */
function countOccupiedCells(toolframe: EditorToolframeSession): number {
  let occupied = 0;
  for (const value of toolframe.state.volume) {
    if (value !== 0) {
      occupied += 1;
    }
  }
  return occupied;
}

/**
 * Builds a model-store fetch double that lists NO model, so the working
 * volume stays empty and a paint lane observes only its own cells.
 *
 * @returns A fetch seam whose model list is empty.
 */
function createEmptyModelStoreFetch(): EditorToolframeFetch {
  return async () => ({
    ok: true,
    json: async () => ({ models: [] }),
  });
}

describe('Neat Racing editor controls (slice P1-editor-controls)', () => {
  beforeEach(() => {
    mountEditorShell();
    // jsdom ships no WebGL implementation: stub the canvas context so the
    // boot lane exercises the context service's fallback branch the same
    // way a browser without WebGL2 would (LD-1), without "Not implemented"
    // console noise.
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  });

  describe('AC-020 (completion, browser wiring): palette add/select handlers', () => {
    it('adds a palette entry and selects it as the paint entry', async () => {
      const { controls, toolframe } = await bootControls();
      const paletteIndex = controls.addPaletteEntry([...GREEN_ENTRY]);
      expect({
        paletteIndex,
        usedCount: toolframe.palette.snapshot().usedCount,
        paintEntry: [...controls.paintEntry],
      }).toEqual({
        paletteIndex: toolframe.palette.snapshot().usedCount,
        usedCount: toolframe.palette.snapshot().usedCount,
        paintEntry: [...GREEN_ENTRY],
      });
    });

    it('keeps the palette at the cap when a new color arrives at the cap', async () => {
      const { controls, toolframe } = await bootControls();
      for (
        let colorIndex = 0;
        colorIndex < NEATRACING_EDITOR_PALETTE_MAX_ENTRIES;
        colorIndex += 1
      ) {
        toolframe.palette.addColor([colorIndex, 0, 0, 255, 0]);
      }
      const paletteIndex = controls.addPaletteEntry([255, 255, 255, 255, 0]);
      expect(toolframe.palette.snapshot().usedCount).toBe(
        NEATRACING_EDITOR_PALETTE_MAX_ENTRIES,
      );
      expect(paletteIndex).toBeGreaterThan(0);
      expect(paletteIndex).toBeLessThanOrEqual(
        NEATRACING_EDITOR_PALETTE_MAX_ENTRIES,
      );
    });

    it('renders the added entry into the mounted palette list', async () => {
      const { controls, toolframe } = await bootControls();
      controls.addPaletteEntry([...RED_ENTRY]);
      controls.addPaletteEntry([...GREEN_ENTRY]);
      const swatches = toolframe.controls.paletteList.querySelectorAll('li');
      expect(swatches.length).toBe(toolframe.palette.snapshot().usedCount);
    });

    it('drives the shared paint entry when a palette swatch is selected', async () => {
      const { controls } = await bootControls();
      const paletteIndex = controls.addPaletteEntry([...GREEN_ENTRY]);
      controls.selectPaletteEntry(paletteIndex);
      expect([...controls.paintEntry]).toEqual([...GREEN_ENTRY]);
    });

    it('mounts the palette add affordance into the palette control group', async () => {
      const { toolframe } = await bootControls();
      const colorInput = document.getElementById(
        NEATRACING_EDITOR_CONTROL_IDS.paletteColorInput,
      );
      const addButton = document.getElementById(
        NEATRACING_EDITOR_CONTROL_IDS.paletteAddButton,
      );
      expect({
        colorInputMounted: colorInput !== null,
        addButtonMounted: addButton !== null,
        insidePaletteGroup:
          colorInput?.closest('section') ===
          toolframe.controls.paletteList.closest('section'),
      }).toEqual({
        colorInputMounted: true,
        addButtonMounted: true,
        insidePaletteGroup: true,
      });
    });
  });

  describe('AC-020 (completion, browser wiring): undo/redo buttons', () => {
    it('restores the previous working volume when the undo button is clicked', async () => {
      const { controls, toolframe } = await bootControls();
      const redIndex = toolframe.palette.addColor([...RED_ENTRY]);
      toolframe.state.paint(0, 0, 0, [...RED_ENTRY]);
      controls.undo.push();
      toolframe.state.paint(0, 0, 0, [...GREEN_ENTRY]);
      toolframe.controls.undoButton.click();
      expect(toolframe.state.volume[0]).toBe(redIndex);
    });

    it('evicts the oldest snapshot beyond the 3-step undo cap', async () => {
      const { controls, toolframe } = await bootControls();
      const firstIndex = toolframe.palette.addColor([...RED_ENTRY]);
      const secondIndex = toolframe.palette.addColor([...GREEN_ENTRY]);
      toolframe.state.paint(0, 0, 0, [...RED_ENTRY]);
      controls.undo.push();
      toolframe.state.paint(0, 0, 0, [...GREEN_ENTRY]);
      controls.undo.push();
      toolframe.state.paint(0, 0, 0, [0, 0, 255, 255, 0]);
      controls.undo.push();
      toolframe.state.paint(0, 0, 0, [255, 255, 255, 255, 0]);
      // The fourth push evicts the OLDEST snapshot, so the fourth undo has
      // nothing left to restore.
      controls.undo.push();
      toolframe.controls.undoButton.click();
      toolframe.controls.undoButton.click();
      toolframe.controls.undoButton.click();
      toolframe.controls.undoButton.click();
      const restoredEntry = toolframe.state.volume[0];
      expect({
        retainedSteps: toolframe.undo.size(),
        capSteps: NEATRACING_EDITOR_UNDO_MAX_STEPS,
        restoredEntry,
      }).toEqual({
        retainedSteps: 0,
        capSteps: NEATRACING_EDITOR_UNDO_MAX_STEPS,
        restoredEntry: secondIndex,
      });
      expect(restoredEntry).not.toBe(firstIndex);
    });

    it('re-applies the undone state through the same snapshot mechanism', async () => {
      const { controls, toolframe } = await bootControls();
      const greenIndex = toolframe.palette.addColor([...GREEN_ENTRY]);
      toolframe.state.paint(0, 0, 0, [...RED_ENTRY]);
      controls.undo.push();
      toolframe.state.paint(0, 0, 0, [...GREEN_ENTRY]);
      toolframe.controls.undoButton.click();
      toolframe.controls.redoButton.click();
      expect(toolframe.state.volume[0]).toBe(greenIndex);
    });

    it('drops the redo history when a new edit is pushed', async () => {
      const { controls, toolframe } = await bootControls();
      const redIndex = toolframe.palette.addColor([...RED_ENTRY]);
      toolframe.state.paint(0, 0, 0, [...RED_ENTRY]);
      controls.undo.push();
      toolframe.state.paint(0, 0, 0, [...GREEN_ENTRY]);
      toolframe.controls.undoButton.click();
      controls.undo.push();
      toolframe.controls.redoButton.click();
      expect(toolframe.state.volume[0]).toBe(redIndex);
    });
  });

  describe('AC-020 (completion, browser wiring): model dropdown', () => {
    it('loads the selected model into the working volume and selects it', async () => {
      const { controls, toolframe } = await bootControls();
      toolframe.controls.modelSelect.value = SECOND_MODEL_NAME;
      toolframe.controls.modelSelect.dispatchEvent(new Event('change'));
      await controls.whenIdle();
      expect({
        selectedModel: toolframe.state.snapshot().model,
        firstCell: toolframe.state.volume[0],
        secondCell: toolframe.state.volume[1],
      }).toEqual({
        selectedModel: SECOND_MODEL_NAME,
        // The palette ACCUMULATES across loads: car-f1 claimed index 1
        // (red), so car-f2's blue resolves to the first free slot 2. The
        // second cell is cleared because car-f2 is a 1x1x1 document.
        firstCell: 2,
        secondCell: 0,
      });
    });

    it('keeps the previous selection when the model load fails', async () => {
      const consoleError = jest
        .spyOn(console, 'error')
        .mockImplementation(() => undefined);
      const toolframe = createEditorToolframe({
        fetchImpl: createFailingModelStoreFetch(),
      });
      const camera = createEditorCamera();
      const controls = createEditorControls({ toolframe, camera });
      await toolframe.ready;
      const dropdownBefore = toolframe.controls.modelSelect.value;
      toolframe.controls.modelSelect.dispatchEvent(new Event('change'));
      await controls.whenIdle();
      expect({
        selectedModel: toolframe.state.snapshot().model,
        dropdownUnchanged: toolframe.controls.modelSelect.value === dropdownBefore,
      }).toEqual({
        selectedModel: null,
        dropdownUnchanged: true,
      });
      consoleError.mockRestore();
    });
  });

  describe('AC-019 (extension, CAD views)', () => {
    it('snaps the camera to the front/side/top CAD orthographic views', async () => {
      const { toolframe, camera } = await bootControls();
      toolframe.controls.cadFrontButton.click();
      const front = camera.snapshot();
      toolframe.controls.cadSideButton.click();
      const side = camera.snapshot();
      toolframe.controls.cadTopButton.click();
      const top = camera.snapshot();
      expect({
        front: [front.mode, front.orthoView],
        side: [side.mode, side.orthoView],
        top: [top.mode, top.orthoView],
      }).toEqual({
        front: ['orthographic', 'front'],
        side: ['orthographic', 'side'],
        top: ['orthographic', 'top'],
      });
    });

    it('returns the camera to the perspective orbit view', async () => {
      const { toolframe, camera } = await bootControls();
      toolframe.controls.cadTopButton.click();
      toolframe.controls.perspectiveButton.click();
      const snapshot = camera.snapshot();
      expect([snapshot.mode, snapshot.orthoView]).toEqual([
        'perspective',
        null,
      ]);
    });
  });

  describe('editor boot wiring (slice P1-editor-controls)', () => {
    it('exposes the wired controls and the framed camera on the session', () => {
      const session = bootEditorSession();
      const snapshot = session.camera.snapshot();
      expect({
        controlsBound: session.controls !== undefined,
        distance: snapshot.target.distance,
        center: snapshot.target.center.map((value) => Number(value.toFixed(4))),
      }).toEqual({
        controlsBound: true,
        distance: 6.75,
        center: [0, 2.25, 0],
      });
    });

    it('publishes the working-volume projection the picker inverts (obs 1)', async () => {
      const session = bootEditorSession({ fetch: createModelStoreFetch() });
      await session.toolframe.ready;
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(session.editedDocument?.meta.dims).toEqual([
        NEATRACING_EDITOR_VOLUME_EDGE,
        NEATRACING_EDITOR_VOLUME_EDGE,
        NEATRACING_EDITOR_VOLUME_EDGE,
      ]);
    });

    it('paints the working volume with the selected palette entry', () => {
      const session = bootEditorSession({ fetch: createEmptyModelStoreFetch() });
      const paletteIndex = session.controls.addPaletteEntry([...RED_ENTRY]);
      session.controls.selectPaletteEntry(paletteIndex);
      const canvas = document.getElementById('neatracing-editor-canvas');
      canvas?.dispatchEvent(
        new MouseEvent('pointerdown', { clientX: 0.5, clientY: 0.5 }),
      );
      const occupied = Array.from(session.toolframe.state.volume).filter(
        (value) => value !== 0,
      );
      expect(occupied.length).toBeGreaterThan(0);
      expect(new Set(occupied)).toEqual(new Set([paletteIndex]));
    });

    it('paints the mirrored voxel pair the symmetry surface expands', () => {
      const session = bootEditorSession({ fetch: createEmptyModelStoreFetch() });
      session.controls.addPaletteEntry([...RED_ENTRY]);
      const canvas = document.getElementById('neatracing-editor-canvas');
      canvas?.dispatchEvent(
        new MouseEvent('pointerdown', { clientX: 0.5, clientY: 0.5 }),
      );
      expect(countOccupiedCells(session.toolframe)).toBe(2);
    });
  });
});
