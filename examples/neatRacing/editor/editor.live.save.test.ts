/**
 * @jest-environment jsdom
 */

/**
 * RED → IMPLEMENT → GREEN contract tests for the Neat Racing editor live
 * save and the live mirror-X symmetry toggle (plans/neatracing-voxel-racer.plans.md
 * — phase 1, step 15, slice P1-editor-symmetry-save; spec section 8.3,
 * AC-021 completion, LD-28).
 *
 * The suite pins the slice's acceptance points:
 *
 * - the live mirror-X toggle: the toggle is read at PAINT time (the
 *   `editor.symmetry.ts` policy freezes its flags at construction, so the
 *   live symmetry wraps it), a checked toggle expands one paint into the
 *   mirrored pair, an unchecked toggle paints only the primary voxel, and
 *   the default is ON (`NEATRACING_EDITOR_DEFAULT_MIRROR_X`);
 * - the immediate live save: every enqueued edit issues ONE PUT to the
 *   model-store route with the `application/json` content type the store
 *   requires, a successful save triggers the rebuild flight, a failed save
 *   is reported with a labeled reason and issues NO rebuild flight, an
 *   over-budget payload is refused BEFORE any request leaves the editor,
 *   and the saves enqueued during an in-flight save coalesce (latest
 *   wins);
 * - the on-disk payload: a mirror-symmetric volume is stored in the
 *   mirror-X HALF-WIDTH symmetric form (`meta.symmetric` true, rows
 *   covering 64 stored columns), an asymmetric volume is stored in the
 *   full width, and the payload passes the model store's gates (the S5
 *   top-level key gate plus the section 7.2 schema and row gates), so the
 *   restored volume after a reload is unchanged;
 * - the boot wiring: the editor boot drives the live symmetry into the
 *   pointer edit loop and enqueues one save per edit flight.
 *
 * RED (pre-implementation): resolving './editor.live.save' fails with
 * TS2307 ("Cannot find module './editor.live.save'") while the module is
 * absent (batch 90/93 right-reason precedent).
 */

import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  NEATRACING_EDITOR_MAX_SAVE_BODY_BYTES,
  buildEditorLiveSaveDocument,
  createEditorLiveSave,
  createEditorLiveSymmetry,
  isEditorVolumeMirrorSymmetric,
} from './editor.live.save';
import type {
  EditorLiveSave,
  EditorLiveSaveFetch,
  EditorLiveSaveResult,
} from './editor.live.save';
import { createEditorState } from './editor.state';
import type { EditorState } from './editor.state';
import {
  NEATRACING_EDITOR_VOLUME_EDGE,
  NEATRACING_EDITOR_VOLUME_ROW_STRIDE,
  NEATRACING_EDITOR_VOLUME_SLICE_STRIDE,
} from './editor.state';
import {
  NEATRACING_EDITOR_DEFAULT_MIRROR_X,
  createEditorSymmetry,
} from './editor.symmetry';
import { decodeVoxelModel } from '../voxel/model.decode.utils';
import type { DecodedVoxelModel } from '../voxel/model.decode.utils';
import {
  readVoxelModelDocument,
  writeVoxelModelDocument,
} from '../server/model-store.utils';
import { NEATRACING_MAX_SAVE_BODY_BYTES } from '../server/dev-server';
import type { VoxelPaletteEntry } from '../voxel/model.schema';

// Side-effect import: evaluating the entry module registers
// `window.neatRacingEditorStart` once on the jsdom host (the boot lane
// below drives it).
import './editor-entry';
import type {
  NeatRacingEditorBootDeps,
  NeatRacingEditorSession,
} from './editor-entry';

/** The reserved tool-frame container id pinned by the editor shell markup. */
const EDITOR_TOOLS_ID = 'neatracing-editor-tools';

/** The preview host id the editor boot resolves before the tool frame. */
const PREVIEW_HOST_ID = 'neatracing-editor-preview';

/** Dev-server model list route the tool frame and the live save share. */
const MODEL_LIST_PATH = '/api/models';

/** The model name the store double serves and the live save writes. */
const MODEL_NAME = 'car-f1';

/** Paint entry used by the volume lanes (opaque red). */
const RED_ENTRY: VoxelPaletteEntry = [255, 0, 0, 255, 0];

/**
 * A minimal valid section 7.2 model document: two columns of palette
 * index 1 in a single 2x1x1 slice (row hex `0201` = run length 2 at
 * palette index 1).
 */
const FIRST_MODEL_DOCUMENT = {
  meta: { name: MODEL_NAME, dims: [2, 1, 1], symmetric: false },
  palette: [
    [0, 0, 0, 0, 0],
    [255, 0, 0, 255, 0],
  ],
  slices: [['0201']],
};

/** One recorded live-save request. */
interface RecordedRequest {
  /** The requested URL. */
  readonly url: string;
  /** The request method (`PUT` for the save, `POST` for the rebuild). */
  readonly method: string;
  /** The request headers. */
  readonly headers: Readonly<Record<string, string>>;
  /** The request body, when one was sent. */
  readonly body: string | undefined;
}

/** A live-save fetch double plus the requests it recorded. */
interface SaveFetchHarness {
  /** The fetch seam the live save drives. */
  readonly fetchImpl: EditorLiveSaveFetch;
  /** Every request the seam recorded, in call order. */
  readonly requests: RecordedRequest[];
}

/**
 * Builds a live-save fetch double that records every request and answers
 * with the dev server's real envelopes: `dev-server/saved` for the save
 * and `dev-server/rebuild-started` for the rebuild flight.
 *
 * @param respond - Optional per-request response override.
 * @returns The fetch seam plus the requests it recorded.
 */
function createRecordingSaveFetch(
  respond?: (request: RecordedRequest) => {
    readonly ok: boolean;
    readonly status?: number;
    readonly code?: string;
  },
): SaveFetchHarness {
  const requests: RecordedRequest[] = [];
  const fetchImpl: EditorLiveSaveFetch = async (url, init) => {
    const request: RecordedRequest = {
      url,
      method: init?.method ?? 'GET',
      headers: init?.headers ?? {},
      body: init?.body,
    };
    requests.push(request);
    const override = respond?.(request);
    if (override !== undefined) {
      return {
        ok: override.ok,
        status: override.status ?? 500,
        json: async () => ({
          status: override.status ?? 500,
          code: override.code ?? 'dev-server/error',
        }),
      };
    }
    const isRebuild = request.method === 'POST';
    return {
      ok: true,
      status: isRebuild ? 202 : 200,
      json: async () => ({
        status: isRebuild ? 202 : 200,
        code: isRebuild ? 'dev-server/rebuild-started' : 'dev-server/saved',
      }),
    };
  };
  return { fetchImpl, requests };
}

/**
 * Builds a model-store fetch double: the list route answers with the
 * model names, the named route answers with that model's document.
 *
 * @param models - Model names the list route reports.
 * @returns A fetch seam the tool frame can drive.
 */
function createModelStoreFetch(
  models: readonly string[] = [MODEL_NAME],
): (url: string) => Promise<{ ok: boolean; json: () => Promise<unknown> }> {
  return async (url: string) => {
    if (url.endsWith(MODEL_LIST_PATH)) {
      return { ok: true, json: async () => ({ models }) };
    }
    return { ok: true, json: async () => ({ document: FIRST_MODEL_DOCUMENT }) };
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

/**
 * Boots the real editor entry against the mounted shell.
 *
 * @param deps - Optional boot dependencies (the fetch seams).
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

/** A promise plus the resolver that settles it. */
interface Deferred {
  /** The promise the test awaits. */
  readonly promise: Promise<void>;
  /** Settles the promise. */
  resolve: () => void;
}

/**
 * Builds a deferred a test settles by hand.
 *
 * @returns The deferred promise and its resolver.
 */
function createDeferred(): Deferred {
  let resolvePromise: () => void = () => {};
  const promise = new Promise<void>((resolve) => {
    resolvePromise = resolve;
  });
  return { promise, resolve: () => resolvePromise() };
}

/**
 * Builds an editor state holding one mirror-symmetric painted pair (the
 * pair the mirror-X toggle expands a paint into).
 *
 * @returns The editor state with the painted pair.
 */
function createSymmetricState(): EditorState {
  const state = createEditorState();
  const symmetry = createEditorSymmetry();
  for (const [x, y, z] of symmetry.map(10, 20, 30)) {
    state.paint(x, y, z, RED_ENTRY);
  }
  return state;
}

/**
 * Builds an editor state holding ONE painted voxel (an asymmetric volume:
 * the mirror partner is missing).
 *
 * @returns The editor state with the single painted voxel.
 */
function createAsymmetricState(): EditorState {
  const state = createEditorState();
  state.paint(10, 20, 30, RED_ENTRY);
  return state;
}

/**
 * Counts the cells where the decoded model grid differs from the editor
 * working volume.
 *
 * @param decoded - The decoded on-disk model.
 * @param state - The editor working volume to compare against.
 * @returns The number of differing cells.
 */
function countVolumeMismatches(
  decoded: DecodedVoxelModel,
  state: EditorState,
): number {
  let mismatches = 0;
  for (let sliceIndex = 0; sliceIndex < NEATRACING_EDITOR_VOLUME_EDGE; sliceIndex += 1) {
    const decodedSlice = decoded.slices[sliceIndex];
    for (let rowIndex = 0; rowIndex < NEATRACING_EDITOR_VOLUME_EDGE; rowIndex += 1) {
      const rowStart =
        NEATRACING_EDITOR_VOLUME_ROW_STRIDE * rowIndex +
        NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * sliceIndex;
      for (
        let columnIndex = 0;
        columnIndex < NEATRACING_EDITOR_VOLUME_EDGE;
        columnIndex += 1
      ) {
        const decodedValue =
          decodedSlice[rowIndex * NEATRACING_EDITOR_VOLUME_EDGE + columnIndex];
        if (decodedValue !== state.volume[rowStart + columnIndex]) {
          mismatches += 1;
        }
      }
    }
  }
  return mismatches;
}

/** Counts the occupied (non-EMPTY) cells of a working volume. */
function countOccupiedCells(state: EditorState): number {
  let occupied = 0;
  for (const value of state.volume) {
    if (value !== 0) {
      occupied += 1;
    }
  }
  return occupied;
}

/** The live save plus the state it serializes. */
interface LiveSaveHarness {
  /** The live save under test. */
  readonly liveSave: EditorLiveSave;
  /** The working volume the live save serializes. */
  readonly state: EditorState;
  /** Every request the live save issued. */
  readonly requests: RecordedRequest[];
}

/**
 * Builds a live save over an editor state, with the model name the save
 * writes.
 *
 * @param options - State, model name, and optional fetch seam override.
 * @returns The live save plus the requests it recorded.
 */
function createLiveSaveHarness(
  options: {
    readonly state?: EditorState;
    readonly modelName?: string | null;
    readonly fetchImpl?: EditorLiveSaveFetch | null;
    readonly maxBodyBytes?: number;
  } = {},
): LiveSaveHarness {
  const state = options.state ?? createSymmetricState();
  const harness = createRecordingSaveFetch();
  const liveSave = createEditorLiveSave({
    modelName: () =>
      options.modelName === undefined ? MODEL_NAME : options.modelName,
    readSource: () => ({ volume: state.volume, palette: state.palette }),
    fetchImpl:
      options.fetchImpl === undefined ? harness.fetchImpl : options.fetchImpl,
    ...(options.maxBodyBytes === undefined
      ? {}
      : { maxBodyBytes: options.maxBodyBytes }),
  });
  return { liveSave, state, requests: harness.requests };
}

/** Temporary model-store directories created by the round-trip lanes. */
const temporaryModelDirs: string[] = [];

/**
 * Creates a temporary model-store directory for a round-trip lane.
 *
 * @returns The temporary directory path.
 */
function createTemporaryModelDir(): string {
  const directory = mkdtempSync(join(tmpdir(), 'neatracing-live-save-'));
  temporaryModelDirs.push(directory);
  return directory;
}

describe('Neat Racing editor live save (slice P1-editor-symmetry-save)', () => {
  beforeEach(() => {
    mountEditorShell();
    // jsdom ships no WebGL implementation: stub the canvas context so the
    // boot lanes exercise the context service's fallback branch the same
    // way a browser without WebGL2 would (LD-1), without "Not implemented"
    // console noise.
    jest.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  });

  afterEach(() => {
    while (temporaryModelDirs.length > 0) {
      const directory = temporaryModelDirs.pop();
      if (directory !== undefined) {
        rmSync(directory, { recursive: true, force: true });
      }
    }
    jest.restoreAllMocks();
  });

  describe('AC-021 (completion): the live mirror-X symmetry toggle', () => {
    it('defaults the mirror-X toggle to ON', () => {
      const symmetry = createEditorLiveSymmetry({
        toggle: { checked: NEATRACING_EDITOR_DEFAULT_MIRROR_X },
      });
      expect({
        defaultFlag: NEATRACING_EDITOR_DEFAULT_MIRROR_X,
        mirrorX: symmetry.mirrorX,
        targets: symmetry.map(10, 20, 30),
      }).toEqual({
        defaultFlag: true,
        mirrorX: true,
        targets: [
          [10, 20, 30],
          [117, 20, 30],
        ],
      });
    });

    it('expands one paint into the mirrored pair while the toggle is checked', () => {
      const symmetry = createEditorLiveSymmetry({ toggle: { checked: true } });
      expect(symmetry.map(10, 20, 30)).toEqual([
        [10, 20, 30],
        [117, 20, 30],
      ]);
    });

    it('paints only the primary voxel while the toggle is unchecked', () => {
      const symmetry = createEditorLiveSymmetry({ toggle: { checked: false } });
      expect(symmetry.map(10, 20, 30)).toEqual([[10, 20, 30]]);
    });

    it('reads the toggle at paint time so a mid-session flip takes effect', () => {
      const toggle = { checked: true };
      const symmetry = createEditorLiveSymmetry({ toggle });
      const beforeFlip = symmetry.map(10, 20, 30);
      toggle.checked = false;
      const afterFlip = symmetry.map(10, 20, 30);
      expect({ beforeFlip, afterFlip, mirrorX: symmetry.mirrorX }).toEqual({
        beforeFlip: [
          [10, 20, 30],
          [117, 20, 30],
        ],
        afterFlip: [[10, 20, 30]],
        mirrorX: false,
      });
    });
  });

  describe('AC-021 (completion): the immediate live save', () => {
    it('issues one PUT to the model-store route per enqueued edit', async () => {
      const { liveSave, requests } = createLiveSaveHarness();
      liveSave.enqueueSave();
      await liveSave.idle();
      const saveRequests = requests.filter(
        (request) => request.method === 'PUT',
      );
      expect({
        saveCount: saveRequests.length,
        url: saveRequests[0]?.url,
        method: saveRequests[0]?.method,
      }).toEqual({
        saveCount: 1,
        url: `${MODEL_LIST_PATH}/${MODEL_NAME}`,
        method: 'PUT',
      });
    });

    it('sends the application/json content type the model store requires', async () => {
      const { liveSave, requests } = createLiveSaveHarness();
      liveSave.enqueueSave();
      await liveSave.idle();
      expect(requests[0]?.headers['content-type']).toBe('application/json');
    });

    it('triggers the rebuild flight after a successful save', async () => {
      const { liveSave, requests } = createLiveSaveHarness();
      liveSave.enqueueSave();
      await liveSave.idle();
      expect(
        requests.map((request) => `${request.method} ${request.url}`),
      ).toEqual([
        `PUT ${MODEL_LIST_PATH}/${MODEL_NAME}`,
        `POST ${MODEL_LIST_PATH}/${MODEL_NAME}/rebuild`,
      ]);
    });

    it('reports a failed save and issues no rebuild flight', async () => {
      const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      const harness = createRecordingSaveFetch(() => ({ ok: false, status: 500 }));
      const { liveSave } = createLiveSaveHarness({
        fetchImpl: harness.fetchImpl,
      });
      liveSave.enqueueSave();
      await liveSave.idle();
      const result = liveSave.snapshot().lastResult as EditorLiveSaveResult;
      expect({
        status: result.status,
        model: result.model,
        error: result.error,
        requests: harness.requests.map((request) => request.method),
        errorReported: errorSpy.mock.calls.length,
      }).toEqual({
        status: 'failed',
        model: MODEL_NAME,
        error: expect.stringContaining('neatracing-editor-live-save'),
        requests: ['PUT'],
        errorReported: 1,
      });
    });

    it('refuses an over-budget payload before any request leaves the editor', async () => {
      const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      const { liveSave, requests } = createLiveSaveHarness({
        maxBodyBytes: 64,
      });
      liveSave.enqueueSave();
      await liveSave.idle();
      const result = liveSave.snapshot().lastResult as EditorLiveSaveResult;
      expect({
        status: result.status,
        error: result.error,
        requestCount: requests.length,
        errorReported: errorSpy.mock.calls.length,
      }).toEqual({
        status: 'failed',
        error: expect.stringContaining('exceeds'),
        requestCount: 0,
        errorReported: 1,
      });
    });

    it('skips the save when no model is selected', async () => {
      const { liveSave, requests } = createLiveSaveHarness({ modelName: null });
      liveSave.enqueueSave();
      await liveSave.idle();
      const result = liveSave.snapshot().lastResult as EditorLiveSaveResult;
      expect({ status: result.status, requestCount: requests.length }).toEqual({
        status: 'skipped-no-model',
        requestCount: 0,
      });
    });

    it('skips the save when no fetch seam is available', async () => {
      const { liveSave, requests } = createLiveSaveHarness({
        fetchImpl: null,
      });
      liveSave.enqueueSave();
      await liveSave.idle();
      const result = liveSave.snapshot().lastResult as EditorLiveSaveResult;
      expect({ status: result.status, requestCount: requests.length }).toEqual({
        status: 'skipped-no-seam',
        requestCount: 0,
      });
    });

    it('coalesces the saves enqueued during an in-flight save (latest wins)', async () => {
      const gate = createDeferred();
      const requests: RecordedRequest[] = [];
      let callIndex = 0;
      const fetchImpl: EditorLiveSaveFetch = async (url, init) => {
        callIndex += 1;
        requests.push({
          url,
          method: init?.method ?? 'GET',
          headers: init?.headers ?? {},
          body: init?.body,
        });
        if (callIndex === 1) {
          await gate.promise;
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ status: 200, code: 'dev-server/saved' }),
        };
      };
      const { liveSave, state } = createLiveSaveHarness({ fetchImpl });
      liveSave.enqueueSave();
      state.paint(30, 40, 50, RED_ENTRY);
      liveSave.enqueueSave();
      state.paint(31, 40, 50, RED_ENTRY);
      liveSave.enqueueSave();
      gate.resolve();
      await liveSave.idle();
      const saveRequests = requests.filter(
        (request) => request.method === 'PUT',
      );
      expect(saveRequests.length).toBe(2);
    });

    it('mirrors the dev-server save body budget', () => {
      expect(NEATRACING_EDITOR_MAX_SAVE_BODY_BYTES).toBe(
        NEATRACING_MAX_SAVE_BODY_BYTES,
      );
    });
  });

  describe('AC-021 (completion): the on-disk save payload', () => {
    it('stores the mirror-X half-width symmetric form for a symmetric volume', () => {
      const state = createSymmetricState();
      const document = buildEditorLiveSaveDocument({
        volume: state.volume,
        palette: state.palette,
        name: MODEL_NAME,
        symmetric: true,
      });
      expect({
        symmetric: document.meta.symmetric,
        dims: document.meta.dims,
        storedRowWidth: document.slices[30][20],
      }).toEqual({
        symmetric: true,
        dims: [128, 128, 128],
        // 10 empty cells, one painted cell, 53 empty cells (64 stored cols).
        storedRowWidth: '0a0001013500',
      });
    });

    it('stores the full width for a volume that is not mirror-symmetric', () => {
      const state = createAsymmetricState();
      const document = buildEditorLiveSaveDocument({
        volume: state.volume,
        palette: state.palette,
        name: MODEL_NAME,
        symmetric: false,
      });
      expect({
        symmetric: document.meta.symmetric,
        dims: document.meta.dims,
        storedRowWidth: document.slices[30][20],
      }).toEqual({
        symmetric: false,
        dims: [128, 128, 128],
        // 10 empty cells, one painted cell, 117 empty cells (128 cols).
        storedRowWidth: '0a0001017500',
      });
    });

    it('detects the mirror-symmetric volume', () => {
      expect({
        symmetric: isEditorVolumeMirrorSymmetric(createSymmetricState().volume),
        asymmetric: isEditorVolumeMirrorSymmetric(
          createAsymmetricState().volume,
        ),
      }).toEqual({ symmetric: true, asymmetric: false });
    });

    it('round-trips the edited volume through the model store', () => {
      const modelsDir = createTemporaryModelDir();
      const state = createSymmetricState();
      const document = buildEditorLiveSaveDocument({
        volume: state.volume,
        palette: state.palette,
        name: MODEL_NAME,
        symmetric: true,
      });
      writeVoxelModelDocument(modelsDir, MODEL_NAME, document);
      const stored = readVoxelModelDocument(modelsDir, MODEL_NAME);
      const decoded = decodeVoxelModel(stored);
      expect({
        name: decoded.name,
        symmetric: decoded.symmetric,
        mismatches: countVolumeMismatches(decoded, state),
      }).toEqual({ name: MODEL_NAME, symmetric: true, mismatches: 0 });
    });

    it('restores the edited volume after a reload', () => {
      const modelsDir = createTemporaryModelDir();
      const state = createSymmetricState();
      const document = buildEditorLiveSaveDocument({
        volume: state.volume,
        palette: state.palette,
        name: MODEL_NAME,
        symmetric: true,
      });
      writeVoxelModelDocument(modelsDir, MODEL_NAME, document);
      // The reload reads the STORED document back (the same read the model
      // store route serves) and decodes it into the working lattice.
      const reloaded = decodeVoxelModel(
        readVoxelModelDocument(modelsDir, MODEL_NAME),
      );
      const paintedRow = reloaded.slices[30];
      expect({
        painted: paintedRow[20 * 128 + 10],
        mirror: paintedRow[20 * 128 + 117],
        empty: paintedRow[20 * 128 + 11],
      }).toEqual({ painted: 1, mirror: 1, empty: 0 });
    });
  });

  describe('editor boot wiring (slice P1-editor-symmetry-save)', () => {
    it('saves the edited volume after every edit flight', async () => {
      const saveFetch = createRecordingSaveFetch();
      const session = bootEditorSession({
        fetch: createModelStoreFetch(),
        saveFetch: saveFetch.fetchImpl,
      });
      await session.toolframe.ready;
      const canvas = document.getElementById('neatracing-editor-canvas');
      canvas?.dispatchEvent(
        new MouseEvent('pointerdown', { clientX: 0.5, clientY: 0.5 }),
      );
      canvas?.dispatchEvent(
        new MouseEvent('pointerup', { clientX: 0.5, clientY: 0.5 }),
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
      await session.liveSave.idle();
      const saveRequests = saveFetch.requests.filter(
        (request) => request.method === 'PUT',
      );
      expect({
        saveCount: saveRequests.length,
        url: saveRequests[0]?.url,
        contentType: saveRequests[0]?.headers['content-type'],
      }).toEqual({
        saveCount: 1,
        url: `${MODEL_LIST_PATH}/${MODEL_NAME}`,
        contentType: 'application/json',
      });
    });

    it('paints the mirrored pair through the live symmetry', async () => {
      const session = bootEditorSession({ fetch: createModelStoreFetch([]) });
      await session.toolframe.ready;
      const canvas = document.getElementById('neatracing-editor-canvas');
      canvas?.dispatchEvent(
        new MouseEvent('pointerdown', { clientX: 0.5, clientY: 0.5 }),
      );
      expect(countOccupiedCells(session.toolframe.state)).toBe(2);
    });

    it('paints only the primary voxel when the toggle is unchecked', async () => {
      const session = bootEditorSession({ fetch: createModelStoreFetch([]) });
      await session.toolframe.ready;
      session.toolframe.controls.symmetryToggle.checked = false;
      const canvas = document.getElementById('neatracing-editor-canvas');
      canvas?.dispatchEvent(
        new MouseEvent('pointerdown', { clientX: 0.5, clientY: 0.5 }),
      );
      expect(countOccupiedCells(session.toolframe.state)).toBe(1);
    });
  });
});
