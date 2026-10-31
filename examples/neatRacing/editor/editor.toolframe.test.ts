/**
 * @jest-environment jsdom
 */

/**
 * RED → IMPLEMENT → GREEN contract tests for the Neat Racing editor tool
 * frame (plans/neatracing-voxel-racer.plans.md — phase 1, step 15, slice
 * P1-editor-toolframe; spec section 8.3, AC-020 partial, LD-28).
 *
 * The suite pins the slice's three acceptance points:
 *
 * - the editor-state binding: the tool frame owns the editor state
 *   container (the dense working volume plus its palette), the bounded
 *   undo stack, and the mirror-symmetry surface;
 * - the control mounts: the palette list, the undo/redo buttons, the model
 *   dropdown, the CAD-view buttons, and the symmetry toggle are mounted
 *   into the reserved `neatracing-editor-tools` container, which is
 *   resolved or thrown (batch-94 handoff contract);
 * - the initial model-volume load: the first listed model is read from the
 *   dev-server model store and decoded into the working volume.
 *
 * The boot lane pins the slice's session extension: the tool frame extends
 * the `window.neatRacingEditorStart` session so the later editing-loop
 * slices bind onto one state container instead of re-plumbing the shell.
 *
 * The fix-packet lanes (fix-packet-P1-editor-toolframe-iteration-1) pin the
 * live projections and the boot fetch seam: the session's `state` and
 * `currentDocument` containers READ the tool frame on every access instead
 * of snapshotting the boot-time load, and the boot accepts an optional
 * fetch seam so the load chain is exercisable without a dev server.
 *
 * RED (pre-implementation): resolving './editor.toolframe' fails with
 * TS2307 ("Cannot find module './editor.toolframe'") while the module is
 * absent (batch 90/93 right-reason precedent).
 */

import { beforeEach, describe, expect, it, jest } from '@jest/globals';

import {
  NEATRACING_EDITOR_TOOLS_ID,
  NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS,
  createEditorToolframe,
} from './editor.toolframe';
import type {
  EditorToolframeFetch,
  EditorToolframeSession,
} from './editor.toolframe';
import { NEATRACING_EDITOR_VOLUME_LENGTH } from './editor.state';
import { NEATRACING_EDITOR_DEFAULT_MIRROR_X } from './editor.symmetry';

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

/** Model name served by the model-store fetch double. */
const MODEL_STORE_NAME = 'car-f1';

/** Model name the name-echoing fetch double serves on the later load. */
const RELOAD_MODEL_NAME = 'car-f2';

/**
 * A minimal valid section 7.2 model document: two columns of palette
 * index 1 in a single 2x1x1 slice (row hex `0201` = run length 2 at
 * palette index 1).
 */
const MODEL_DOCUMENT = {
  meta: { name: MODEL_STORE_NAME, dims: [2, 1, 1], symmetric: false },
  palette: [
    [0, 0, 0, 0, 0],
    [255, 0, 0, 255, 0],
  ],
  slices: [['0201']],
};

/**
 * Builds a model-store fetch double: the list route answers with the model
 * names, the named route answers with the model document.
 *
 * @param models - Model names the list route reports.
 * @returns A fetch seam the tool frame can drive.
 */
function createModelStoreFetch(
  models: readonly string[] = [MODEL_STORE_NAME],
): EditorToolframeFetch {
  return async (url: string) => {
    if (url.endsWith(MODEL_LIST_PATH)) {
      return { ok: true, json: async () => ({ models }) };
    }
    return { ok: true, json: async () => ({ document: MODEL_DOCUMENT }) };
  };
}

/**
 * Builds a model-store fetch double that echoes the requested model name
 * back in the served document, so a later load is observable through the
 * live document projection.
 *
 * @param models - Model names the list route reports.
 * @returns A fetch seam the tool frame can drive.
 */
function createEchoingModelStoreFetch(
  models: readonly string[] = [MODEL_STORE_NAME, RELOAD_MODEL_NAME],
): EditorToolframeFetch {
  return async (url: string) => {
    if (url.endsWith(MODEL_LIST_PATH)) {
      return { ok: true, json: async () => ({ models }) };
    }
    const modelName = decodeURIComponent(
      url.slice(`${MODEL_LIST_PATH}/`.length),
    );
    return {
      ok: true,
      json: async () => ({
        document: {
          ...MODEL_DOCUMENT,
          meta: { ...MODEL_DOCUMENT.meta, name: modelName },
        },
      }),
    };
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

/** Boots a tool frame against the mounted shell and the model-store double. */
function bootToolframe(): EditorToolframeSession {
  return createEditorToolframe({ fetchImpl: createModelStoreFetch() });
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

/** Reads the tool-frame container the controls must be mounted into. */
function getToolframeContainer(): HTMLElement | null {
  return document.getElementById(EDITOR_TOOLS_ID);
}

describe('Neat Racing editor tool frame (slice P1-editor-toolframe)', () => {
  beforeEach(() => {
    mountEditorShell();
    // jsdom ships no WebGL implementation: stub the canvas context so the
    // boot lane exercises the context service's fallback branch the same
    // way a browser without WebGL2 would (LD-1), without "Not implemented"
    // console noise.
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  });

  describe('AC-020 (partial, tool-frame boot): control mounts', () => {
    it('mounts every tool-frame control into the reserved neatracing-editor-tools container', () => {
      const session = bootToolframe();
      const container = getToolframeContainer();
      const missing = Object.values(
        NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS,
      ).filter((controlId) => container?.querySelector(`#${controlId}`) === null);
      expect(missing).toEqual([]);
      expect(session.controls.paletteList.isConnected).toBe(true);
    });

    it('resolves the tool-frame container or throws when it is missing', () => {
      document.body.innerHTML = '';
      expect(() => createEditorToolframe()).toThrow(
        /Editor tool frame host #neatracing-editor-tools not found/u,
      );
    });

    it('mounts the symmetry toggle checked when mirror-X defaults on', () => {
      const session = bootToolframe();
      expect(session.controls.symmetryToggle.checked).toBe(
        NEATRACING_EDITOR_DEFAULT_MIRROR_X,
      );
    });

    it('unmounts the mounted controls when the tool frame stops', () => {
      const session = bootToolframe();
      session.stop();
      const remaining = document
        .getElementById(EDITOR_TOOLS_ID)
        ?.querySelectorAll(
          `#${NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS.paletteList}, #${NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS.modelSelect}`,
        );
      expect(remaining?.length ?? 0).toBe(0);
    });
  });

  describe('AC-020 (partial, tool-frame boot): editor-state binding', () => {
    it('binds the editor state container with its palette and the dense working volume', () => {
      const session = bootToolframe();
      expect({
        volumeLength: session.state.volume.length,
        paletteShared: session.state.palette === session.palette,
        selectedModel: session.state.snapshot().model,
      }).toEqual({
        volumeLength: NEATRACING_EDITOR_VOLUME_LENGTH,
        paletteShared: true,
        selectedModel: null,
      });
    });

    it('exposes the bounded undo stack and the mirror-symmetry surface on the tool frame', () => {
      const session = bootToolframe();
      expect({
        undoVolumeBound: session.undo.canUndo(),
        mirrorX: session.symmetry.mirrorX,
      }).toEqual({
        undoVolumeBound: false,
        mirrorX: NEATRACING_EDITOR_DEFAULT_MIRROR_X,
      });
    });
  });

  describe('AC-020 (partial, tool-frame boot): initial model-volume load', () => {
    it('decodes the listed model into the working volume', async () => {
      const session = bootToolframe();
      await session.ready;
      const volume = session.state.volume;
      expect([volume[0], volume[1], volume[2]]).toEqual([1, 1, 0]);
    });

    it('selects the loaded model in the editor state and the dropdown', async () => {
      const session = bootToolframe();
      await session.ready;
      expect({
        selectedModel: session.state.snapshot().model,
        dropdownValue: session.controls.modelSelect.value,
        dropdownOptions: session.controls.modelSelect.options.length,
      }).toEqual({
        selectedModel: MODEL_STORE_NAME,
        dropdownValue: MODEL_STORE_NAME,
        dropdownOptions: 1,
      });
    });

    it('adds the loaded model palette to the editor palette', async () => {
      const session = bootToolframe();
      await session.ready;
      expect(session.palette.snapshot().usedCount).toBe(1);
    });
  });

  describe('editor boot session extension', () => {
    it('extends the window.neatRacingEditorStart session with the tool frame', () => {
      const session = bootEditorSession();
      const container = getToolframeContainer();
      expect({
        paletteMounted:
          container?.querySelector(
            `#${NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS.paletteList}`,
          ) !== null,
        volumeLength: session.toolframe.state.volume.length,
      }).toEqual({
        paletteMounted: true,
        volumeLength: NEATRACING_EDITOR_VOLUME_LENGTH,
      });
    });

    it('mirrors the tool-frame model selection on the session state container', async () => {
      const session = bootEditorSession();
      await session.toolframe.ready;
      expect(session.state.model).toBe(session.toolframe.state.snapshot().model);
    });
  });

  describe('live session projections and the boot fetch seam (fix-packet-P1-editor-toolframe-iteration-1)', () => {
    it('projects the tool-frame selection live onto the session state container', () => {
      // Floor lane (F1): the projection must READ the tool-frame state on
      // every access. A snapshot captured at boot still reports null here,
      // so this lane fails for a snapshot implementation.
      const session = bootEditorSession();
      session.toolframe.state.selectModel('probe-model');
      expect(session.state.model).toBe('probe-model');
    });

    it('loads the model through the injected fetch seam and projects it live', async () => {
      // Discriminating lane (F1 + F2): the injected seam makes the initial
      // load land a real model, so both projections are compared against a
      // non-null value instead of null === null.
      const session = bootEditorSession({ fetch: createModelStoreFetch() });
      await session.toolframe.ready;
      expect({
        stateModel: session.state.model,
        documentName: session.currentDocument?.meta.name ?? null,
      }).toEqual({
        stateModel: MODEL_STORE_NAME,
        documentName: MODEL_STORE_NAME,
      });
    });

    it('follows a later model load through the live document projection', async () => {
      // F3 lane: the one-shot snapshot of the initial load keeps the first
      // document, so this lane fails for a snapshot implementation.
      const session = bootEditorSession({
        fetch: createEchoingModelStoreFetch(),
      });
      await session.toolframe.ready;
      const firstDocumentName = session.currentDocument?.meta.name ?? null;
      await session.toolframe.loadModel(RELOAD_MODEL_NAME);
      expect({
        firstDocumentName,
        reloadedDocumentName: session.currentDocument?.meta.name ?? null,
        reloadedStateModel: session.state.model,
      }).toEqual({
        firstDocumentName: MODEL_STORE_NAME,
        reloadedDocumentName: RELOAD_MODEL_NAME,
        reloadedStateModel: RELOAD_MODEL_NAME,
      });
    });
  });
});
