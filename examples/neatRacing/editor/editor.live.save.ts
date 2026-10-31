/// <reference lib="dom" />

/**
 * Live mirror-X symmetry and the immediate live save for the Neat Racing
 * voxel model editor (slice P1-editor-symmetry-save; spec section 8.3,
 * AC-021 completion, LD-28 of `plans/neatracing-voxel-racer.spec.md`).
 *
 * The module owns two surfaces:
 *
 * 1. {@linkcode createEditorLiveSymmetry} - the LIVE mirror-X symmetry
 *    policy. `editor.symmetry.ts` freezes its axis flags at construction,
 *    so a mid-session toggle flip could never reach the paint path; the
 *    live wrapper reads the toggle's `checked` state at PAINT time and
 *    resolves one of two cached frozen policies (the toggle is only ever
 *    ON or OFF, so the policy set is built once, not per paint). The
 *    mirror-Y and mirror-Z flags stay frozen on the base policy.
 * 2. {@linkcode createEditorLiveSave} - the immediate live save. Every
 *    enqueued edit serializes the working volume into a section 7.2 model
 *    document and PUTs it to the dev-server model store
 *    (`PUT /api/models/:name`), then triggers the rebuild flight
 *    (`POST /api/models/:name/rebuild`). The flight is latest-wins: the
 *    saves enqueued while a request is in flight coalesce into ONE
 *    follow-up request carrying the CURRENT volume, so a drag burst never
 *    piles up requests.
 *
 * STORAGE FORM: the stored models are mirror-X symmetric documents
 * (`meta.symmetric` true, rows covering half the columns), so the live
 * save stores the HALF-WIDTH form whenever the working volume is verified
 * mirror-symmetric (`volume[x] === volume[127 - x]` on every row) and the
 * mirror-X toggle is on. Any other volume (an asymmetric edit, or a model
 * that was never symmetric) is stored in the FULL width
 * (`meta.symmetric` false). The verification is what keeps the round trip
 * lossless: the loader mirrors the stored half back onto the full width,
 * so folding an asymmetric volume would silently drop the right half.
 *
 * BODY BUDGET: the dev server rejects an over-budget save body with
 * `413 dev-server/body-too-large`
 * (`NEATRACING_MAX_SAVE_BODY_BYTES`, 8 MiB). The editor mirrors that
 * budget in {@linkcode NEATRACING_EDITOR_MAX_SAVE_BODY_BYTES} (the server
 * module cannot be imported into the browser bundle) and refuses an
 * over-budget payload BEFORE any request leaves the editor, so a doomed
 * request never reaches the server.
 *
 * FAILURE HANDLING: a missing model or a missing fetch seam is not a
 * failure but a missing precondition - the save is skipped with a status
 * and no noise (the same "no dev server is not a failure worth surfacing"
 * precedent the tool frame follows). A rejected request, an over-budget
 * payload, or a transport error is reported once through a labeled
 * `console.error` and a labeled status. The save never rejects: every
 * failure path resolves into a result.
 *
 * The module stays free of WebGL and of any ambient timer: the symmetry
 * wrapper is pure data over the injected toggle, and the save drives the
 * injected fetch seam only.
 *
 * @module
 */

import { createEditorSymmetry } from './editor.symmetry';
import type { EditorSymmetry } from './editor.symmetry';
import { NEATRACING_EDITOR_MODEL_LIST_PATH } from './editor.toolframe';
import {
  NEATRACING_EDITOR_VOLUME_EDGE,
  NEATRACING_EDITOR_VOLUME_ROW_STRIDE,
  NEATRACING_EDITOR_VOLUME_SLICE_STRIDE,
} from './editor.state';
import type { EditorPalette } from './editor.palette';
import { VOXEL_EMPTY_PALETTE_ENTRY } from '../voxel/model.schema';
import type {
  VoxelModelDocument,
  VoxelPaletteEntry,
} from '../voxel/model.schema';

/**
 * Save body budget the editor enforces before it issues a save request.
 *
 * Mirrors the dev server's `NEATRACING_MAX_SAVE_BODY_BYTES` (8 MiB): the
 * server rejects an over-budget body with `413 dev-server/body-too-large`,
 * and the editor refuses the same payload locally so no doomed request
 * leaves the browser. The server constant cannot be imported here (the
 * editor bundle is a browser build and the server module pulls in
 * `node:http`), so the contract test pins the two values together.
 */
export const NEATRACING_EDITOR_MAX_SAVE_BODY_BYTES = 8 * 1024 * 1024;

/** Pass label prefixing every observable live-save failure surface. */
export const NEATRACING_EDITOR_LIVE_SAVE_LABEL = 'neatracing-editor-live-save';

/** Stored width of a mirror-X symmetric document (half of the 128 columns). */
const SYMMETRIC_STORED_WIDTH = NEATRACING_EDITOR_VOLUME_EDGE / 2;

/** Largest number of cells one RLE run may cover. */
const MAX_RUN_LENGTH = 255;

/** Hex digits per RLE field (the run length and the palette index). */
const RLE_FIELD_HEX_CHARS = 2;

/** Content type the model store's save route requires. */
const SAVE_CONTENT_TYPE = 'application/json';

/** Dev-server envelope code of a coalesced rebuild flight. */
const REBUILD_COALESCED_CODE = 'dev-server/rebuild-coalesced';

/**
 * Minimal checkbox surface the live mirror-X flag reads.
 *
 * The DOM checkbox (`HTMLInputElement`) satisfies it structurally; the
 * contract tests drive a plain object instead.
 */
export interface EditorMirrorXToggle {
  /** Whether the mirror-X toggle is currently checked. */
  readonly checked: boolean;
}

/** Factory configuration for the live mirror-X symmetry policy. */
export interface EditorLiveSymmetryConfig {
  /** The mirror-X toggle the live flag reads at paint time. */
  readonly toggle: EditorMirrorXToggle;
  /**
   * The frozen base policy the live flag extends; defaults to the editor
   * default (mirror-X on, mirror-Y and mirror-Z off).
   */
  readonly base?: EditorSymmetry;
}

/**
 * Public live symmetry surface the editor paint path drives.
 *
 * The `mirrorX` flag and the expansion follow the toggle at call time, so
 * flipping the toggle mid-session takes effect on the NEXT paint without
 * rebuilding the policy.
 */
export interface EditorLiveSymmetry extends EditorSymmetry {
  /** Whether the mirror-X toggle is currently checked. */
  readonly mirrorX: boolean;
}

/**
 * Minimal response shape the live save reads from its fetch seam. The
 * browser `Response` satisfies it structurally; the test doubles only
 * implement the members the live save actually consumes.
 */
export interface EditorLiveSaveFetchResponse {
  /** Whether the request succeeded. */
  readonly ok: boolean;
  /** HTTP status code, when the seam reports one. */
  readonly status?: number;
  /** Reads the response body as JSON. */
  json(): Promise<unknown>;
}

/** Request shape the live save issues through its fetch seam. */
export interface EditorLiveSaveRequestInit {
  /** HTTP method (`PUT` for the save, `POST` for the rebuild flight). */
  readonly method?: string;
  /** Request headers. */
  readonly headers?: Readonly<Record<string, string>>;
  /** Request body. */
  readonly body?: string;
}

/**
 * Fetch seam the live save drives.
 *
 * Unlike the tool frame's GET-only seam, this one issues the write
 * methods the model store's save route requires.
 */
export type EditorLiveSaveFetch = (
  url: string,
  init?: EditorLiveSaveRequestInit,
) => Promise<EditorLiveSaveFetchResponse>;

/** The working volume the live save serializes. */
export interface EditorLiveSaveSource {
  /** Dense palette-index working volume. */
  readonly volume: Uint8Array;
  /** Palette the volume's indices address. */
  readonly palette: EditorPalette;
}

/** Outcome kind of one live-save attempt. */
export type EditorLiveSaveStatus =
  /** The document was stored by the model store. */
  | 'saved'
  /** No model was selected, so there was nothing to save. */
  | 'skipped-no-model'
  /** No fetch seam was available (no dev server). */
  | 'skipped-no-seam'
  /** The document was not stored (rejected, over budget, or transport error). */
  | 'failed';

/** Outcome kind of the rebuild flight that follows a stored document. */
export type EditorLiveSaveRebuild =
  /** The server started a rebuild flight. */
  | 'started'
  /** The server coalesced the flight into the running one. */
  | 'coalesced'
  /** The rebuild flight request failed. */
  | 'failed';

/** Result of one live-save attempt. */
export interface EditorLiveSaveResult {
  /** Outcome kind of the attempt. */
  readonly status: EditorLiveSaveStatus;
  /** The model the attempt targeted, or null when no model was selected. */
  readonly model: string | null;
  /** Serialized payload size in bytes (0 when nothing was serialized). */
  readonly bytes: number;
  /** Whether the payload used the mirror-X half-width storage form. */
  readonly symmetric: boolean;
  /** The labeled failure reason, or null when the attempt succeeded. */
  readonly error: string | null;
  /** The rebuild flight outcome, or null when no flight was triggered. */
  readonly rebuild: EditorLiveSaveRebuild | null;
}

/** Read-only view of the live save's state. */
export interface EditorLiveSaveSnapshot {
  /** True while a save is queued or in flight. */
  readonly pending: boolean;
  /** Number of completed save attempts. */
  readonly attempts: number;
  /** Number of attempts that stored the document. */
  readonly saved: number;
  /** The last attempt's result, or null before the first attempt. */
  readonly lastResult: EditorLiveSaveResult | null;
}

/** Factory configuration for the live save. */
export interface EditorLiveSaveConfig {
  /** Resolves the model name to save, or null while no model is selected. */
  readonly modelName: () => string | null;
  /** Reads the working volume the save serializes. */
  readonly readSource: () => EditorLiveSaveSource;
  /**
   * Fetch seam the save drives. `undefined` resolves the ambient
   * `globalThis.fetch` at save time (the production default), `null`
   * declares that no seam exists (the save is skipped), and a function is
   * used as-is.
   */
  readonly fetchImpl?: EditorLiveSaveFetch | null;
  /** Save body budget in bytes; defaults to the mirrored server budget. */
  readonly maxBodyBytes?: number;
  /**
   * Whether the mirror-X storage form is preferred right now; defaults to
   * `true` (mirror-X is on by default).
   */
  readonly mirrorXEnabled?: () => boolean;
  /** Model-store route prefix; defaults to `/api/models`. */
  readonly modelPath?: string;
}

/** Public live-save surface the editor boot drives. */
export interface EditorLiveSave {
  /**
   * Enqueues one immediate save of the current working volume.
   *
   * The enqueue coalesces: the saves enqueued while a request is in
   * flight collapse into ONE follow-up request carrying the current
   * volume.
   *
   * @returns Nothing; the save runs through the injected fetch seam.
   */
  enqueueSave(): void;
  /**
   * Resolves once the queued save flight settled.
   *
   * @returns A promise resolved after the in-flight save finished.
   */
  idle(): Promise<void>;
  /**
   * Read-only snapshot of the save state.
   *
   * @returns The snapshot at call time.
   */
  snapshot(): EditorLiveSaveSnapshot;
  /**
   * Stops the live save: the queued save is dropped and further enqueues
   * are refused.
   *
   * @returns Nothing.
   */
  stop(): void;
}

/** Factory configuration for the live-save document projection. */
export interface EditorLiveSaveDocumentConfig {
  /** Dense palette-index working volume. */
  readonly volume: Uint8Array;
  /** Palette the volume's indices address. */
  readonly palette: EditorPalette;
  /** Model name stamped into the document metadata. */
  readonly name: string;
  /**
   * Whether the document uses the mirror-X half-width storage form. The
   * caller decides (the live save folds only a VERIFIED symmetric
   * volume), so the projection never loses the right half on its own.
   */
  readonly symmetric: boolean;
}

/**
 * Creates the live mirror-X symmetry policy.
 *
 * The toggle is read at PAINT time, so flipping it mid-session changes
 * the next paint without rebuilding the policy. The mirror-Y and
 * mirror-Z flags stay frozen on the base policy (the slice owns the
 * mirror-X toggle only).
 *
 * @param config - The mirror-X toggle plus the optional frozen base
 *   policy.
 * @returns The live symmetry surface the edit loop drives.
 * @example
 * ```ts
 * const toggle = { checked: true };
 * const symmetry = createEditorLiveSymmetry({ toggle });
 * symmetry.map(10, 20, 30); // [[10, 20, 30], [117, 20, 30]]
 * toggle.checked = false;
 * symmetry.map(10, 20, 30); // [[10, 20, 30]]
 * ```
 */
export function createEditorLiveSymmetry(
  config: EditorLiveSymmetryConfig,
): EditorLiveSymmetry {
  const base = config.base ?? createEditorSymmetry();
  // Step 1: Resolve the two frozen policies the toggle selects between.
  // The base policy already carries the editor's mirror-Y/mirror-Z flags.
  const enabled =
    base.mirrorX
      ? base
      : createEditorSymmetry({
          mirrorX: true,
          mirrorY: base.mirrorY,
          mirrorZ: base.mirrorZ,
        });
  const disabled = base.mirrorX
    ? createEditorSymmetry({
        mirrorX: false,
        mirrorY: base.mirrorY,
        mirrorZ: base.mirrorZ,
      })
    : base;

  return {
    // Step 2: The live flag follows the toggle at call time.
    get mirrorX(): boolean {
      return config.toggle.checked;
    },
    mirrorY: base.mirrorY,
    mirrorZ: base.mirrorZ,

    // Step 3: The expansion resolves the matching frozen policy per
    // paint, so the emitted target list stays exactly deterministic.
    map(x, y, z) {
      return (config.toggle.checked ? enabled : disabled).map(x, y, z);
    },
  };
}

/**
 * Builds the index-aligned document palette: the EMPTY sentinel at index 0
 * plus every real color entry, so the volume's palette indices address
 * the document palette directly.
 *
 * @param palette - Editor palette the volume's indices address.
 * @returns The document palette (index 0 is the EMPTY sentinel).
 */
function buildDocumentPalette(palette: EditorPalette): VoxelPaletteEntry[] {
  const entries = palette.snapshot().entries;
  const documentPalette: VoxelPaletteEntry[] = [
    [
      VOXEL_EMPTY_PALETTE_ENTRY[0],
      VOXEL_EMPTY_PALETTE_ENTRY[1],
      VOXEL_EMPTY_PALETTE_ENTRY[2],
      VOXEL_EMPTY_PALETTE_ENTRY[3],
      VOXEL_EMPTY_PALETTE_ENTRY[4],
    ],
  ];
  for (const entry of entries.slice(1)) {
    if (entry === undefined) {
      continue;
    }
    documentPalette.push([entry[0], entry[1], entry[2], entry[3], entry[4]]);
  }
  return documentPalette;
}

/**
 * Clamps a volume palette index into the document palette's range.
 *
 * @param value - Raw palette index stored in the volume.
 * @param paletteSize - Number of entries the document palette holds.
 * @returns The index to write, or 0 (EMPTY) when it is out of range.
 */
function resolvePaletteIndex(value: number, paletteSize: number): number {
  return value < paletteSize ? value : 0;
}

/**
 * Encodes one RLE run into its hex pair.
 *
 * @param runLength - Run length in cells (1..255).
 * @param paletteIndex - Palette index the run paints.
 * @returns The `<runLen:2hex><paletteIdx:2hex>` hex pair.
 */
function encodeRun(runLength: number, paletteIndex: number): string {
  return (
    runLength.toString(16).padStart(RLE_FIELD_HEX_CHARS, '0') +
    paletteIndex.toString(16).padStart(RLE_FIELD_HEX_CHARS, '0')
  );
}

/**
 * Encodes the stored part of one volume row into a section 7.2 RLE hex
 * row string.
 *
 * The stored part is the FULL row for a full-width document and the LEFT
 * half (cols 0..63) for a mirror-X symmetric document - the half the
 * loader mirrors back onto the full width (LD-28).
 *
 * @param volume - Dense working volume.
 * @param z - Voxel slice the row belongs to.
 * @param y - Voxel row the row belongs to.
 * @param storedWidth - Number of columns the row must cover.
 * @param paletteSize - Number of entries the document palette holds.
 * @returns The RLE hex row string.
 */
function encodeRow(
  volume: Uint8Array,
  z: number,
  y: number,
  storedWidth: number,
  paletteSize: number,
): string {
  const rowStart =
    NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * z +
    NEATRACING_EDITOR_VOLUME_ROW_STRIDE * y;
  let row = '';
  let runIndex = resolvePaletteIndex(volume[rowStart], paletteSize);
  let runLength = 1;
  for (let x = 1; x < storedWidth; x += 1) {
    const value = resolvePaletteIndex(volume[rowStart + x], paletteSize);
    if (value === runIndex && runLength < MAX_RUN_LENGTH) {
      runLength += 1;
      continue;
    }
    row += encodeRun(runLength, runIndex);
    runIndex = value;
    runLength = 1;
  }
  return row + encodeRun(runLength, runIndex);
}

/**
 * Projects the editor working volume into a section 7.2 model document
 * for the model store.
 *
 * The document palette is index-aligned with the editor palette (index 0
 * is the EMPTY sentinel), so the volume's palette indices address the
 * document palette directly. The stored width follows the requested
 * storage form: the full 128 columns, or the LEFT 64 columns the loader
 * mirrors back onto the full width.
 *
 * @param config - Volume, palette, model name, and storage form.
 * @returns The projected model document.
 * @example
 * ```ts
 * const document = buildEditorLiveSaveDocument({
 *   volume: state.volume,
 *   palette: state.palette,
 *   name: 'car-f1',
 *   symmetric: true,
 * });
 * console.log(document.meta.symmetric); // true
 * ```
 */
export function buildEditorLiveSaveDocument(
  config: EditorLiveSaveDocumentConfig,
): VoxelModelDocument {
  const { volume, palette, name } = config;

  // Step 1: Build the index-aligned document palette.
  const documentPalette = buildDocumentPalette(palette);

  // Step 2: Encode every stored row of every slice into RLE hex rows.
  const storedWidth = config.symmetric
    ? SYMMETRIC_STORED_WIDTH
    : NEATRACING_EDITOR_VOLUME_EDGE;
  const slices: string[][] = [];
  for (let z = 0; z < NEATRACING_EDITOR_VOLUME_EDGE; z += 1) {
    const rows: string[] = [];
    for (let y = 0; y < NEATRACING_EDITOR_VOLUME_EDGE; y += 1) {
      rows.push(encodeRow(volume, z, y, storedWidth, documentPalette.length));
    }
    slices.push(rows);
  }

  // Step 3: Stamp the section 7.2 metadata header.
  return {
    meta: {
      name,
      dims: [
        NEATRACING_EDITOR_VOLUME_EDGE,
        NEATRACING_EDITOR_VOLUME_EDGE,
        NEATRACING_EDITOR_VOLUME_EDGE,
      ],
      symmetric: config.symmetric,
    },
    palette: documentPalette,
    slices,
  };
}

/**
 * Reports whether the working volume is mirror-X symmetric.
 *
 * The volume is symmetric when every row mirrors onto itself across the
 * x axis (`volume[x] === volume[127 - x]` for cols 0..63). The live save
 * folds only a verified symmetric volume into the half-width storage
 * form, because the loader mirrors the stored half back onto the full
 * width - folding an asymmetric volume would silently drop the right
 * half.
 *
 * @param volume - Dense working volume.
 * @returns True when every row mirrors onto itself.
 * @example
 * ```ts
 * if (isEditorVolumeMirrorSymmetric(state.volume)) {
 *   // the half-width storage form is lossless for this volume
 * }
 * ```
 */
export function isEditorVolumeMirrorSymmetric(volume: Uint8Array): boolean {
  for (let z = 0; z < NEATRACING_EDITOR_VOLUME_EDGE; z += 1) {
    for (let y = 0; y < NEATRACING_EDITOR_VOLUME_EDGE; y += 1) {
      const rowStart =
        NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * z +
        NEATRACING_EDITOR_VOLUME_ROW_STRIDE * y;
      for (let x = 0; x < SYMMETRIC_STORED_WIDTH; x += 1) {
        if (volume[rowStart + x] !== volume[rowStart + NEATRACING_EDITOR_VOLUME_EDGE - 1 - x]) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * Describes a thrown value for a labeled failure message.
 *
 * @param error - The thrown value.
 * @returns A short human-readable description.
 */
function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Reads the dev-server envelope code of one response.
 *
 * @param response - Response the seam returned.
 * @returns The envelope code, or null when the body carries none.
 */
async function readResponseCode(
  response: EditorLiveSaveFetchResponse,
): Promise<string | null> {
  try {
    const payload = await response.json();
    if (typeof payload === 'object' && payload !== null) {
      const code = (payload as { code?: unknown }).code;
      return typeof code === 'string' ? code : null;
    }
  } catch {
    // A body-less or non-JSON envelope carries no code; the caller
    // treats the absence as the default outcome.
  }
  return null;
}

/**
 * Creates the immediate live save.
 *
 * The save serializes the working volume into a section 7.2 document,
 * PUTs it to the model store, and triggers the rebuild flight. The flight
 * is latest-wins: the saves enqueued while a request is in flight
 * coalesce into ONE follow-up request carrying the current volume.
 *
 * @param config - Model name, volume source, and the optional fetch seam.
 * @returns The live-save surface the editor boot drives.
 * @example
 * ```ts
 * const liveSave = createEditorLiveSave({
 *   modelName: () => toolframe.state.snapshot().model,
 *   readSource: () => ({
 *     volume: toolframe.state.volume,
 *     palette: toolframe.palette,
 *   }),
 *   fetchImpl: deps?.saveFetch,
 * });
 * liveSave.enqueueSave();
 * ```
 */
export function createEditorLiveSave(
  config: EditorLiveSaveConfig,
): EditorLiveSave {
  const modelPath = config.modelPath ?? NEATRACING_EDITOR_MODEL_LIST_PATH;
  const maxBodyBytes =
    config.maxBodyBytes ?? NEATRACING_EDITOR_MAX_SAVE_BODY_BYTES;

  /** Whether a save is queued for the next flight. */
  let pending = false;
  /** Whether the live save was stopped (no further saves are accepted). */
  let stopped = false;
  /** The in-flight drain promise, or null while no save is running. */
  let flight: Promise<void> | null = null;
  /** Number of completed save attempts. */
  let attempts = 0;
  /** Number of attempts that stored the document. */
  let saved = 0;
  /** The last attempt's result. */
  let lastResult: EditorLiveSaveResult | null = null;

  /**
   * Resolves the fetch seam the save drives.
   *
   * @returns The seam, or null when no seam is available.
   */
  function resolveFetch(): EditorLiveSaveFetch | null {
    if (config.fetchImpl === null) {
      return null;
    }
    if (config.fetchImpl !== undefined) {
      return config.fetchImpl;
    }
    const ambient = (globalThis as { fetch?: unknown }).fetch;
    return typeof ambient === 'function'
      ? (ambient as unknown as EditorLiveSaveFetch)
      : null;
  }

  /**
   * Issues the save request and the rebuild flight.
   *
   * @returns The result of the attempt.
   */
  async function performSave(): Promise<EditorLiveSaveResult> {
    // Step 1: Resolve the target model. No model means nothing to save.
    const modelName = config.modelName();
    if (modelName === null || modelName.length === 0) {
      return {
        status: 'skipped-no-model',
        model: null,
        bytes: 0,
        symmetric: false,
        error: null,
        rebuild: null,
      };
    }

    // Step 2: Resolve the fetch seam. No dev server is not a failure.
    const fetchImpl = resolveFetch();
    if (fetchImpl === null) {
      return {
        status: 'skipped-no-seam',
        model: modelName,
        bytes: 0,
        symmetric: false,
        error: null,
        rebuild: null,
      };
    }

    // Step 3: Project the working volume into the storage form. The
    // half-width form is used only for a VERIFIED symmetric volume while
    // the mirror-X toggle is on; every other volume keeps the full width.
    const source = config.readSource();
    const mirrorXEnabled = config.mirrorXEnabled?.() ?? true;
    const symmetric =
      mirrorXEnabled && isEditorVolumeMirrorSymmetric(source.volume);
    const document = buildEditorLiveSaveDocument({
      volume: source.volume,
      palette: source.palette,
      name: modelName,
      symmetric,
    });
    const body = JSON.stringify(document);
    // The payload is ASCII by construction (the model name matches the
    // store's name pattern, the palette holds byte values, and the rows
    // are hex), so the string length IS the UTF-8 byte length.
    const bytes = body.length;

    // Step 4: Refuse an over-budget payload BEFORE any request leaves
    // the editor (the server would reject it with 413 anyway).
    if (bytes > maxBodyBytes) {
      return {
        status: 'failed',
        model: modelName,
        bytes,
        symmetric,
        error: `${NEATRACING_EDITOR_LIVE_SAVE_LABEL}: the save body is ${bytes} bytes and exceeds the ${maxBodyBytes}-byte budget`,
        rebuild: null,
      };
    }

    // Step 5: Store the document through the model store's save route.
    const saveUrl = `${modelPath}/${encodeURIComponent(modelName)}`;
    let saveResponse: EditorLiveSaveFetchResponse;
    try {
      saveResponse = await fetchImpl(saveUrl, {
        method: 'PUT',
        headers: { 'content-type': SAVE_CONTENT_TYPE },
        body,
      });
    } catch (error) {
      return {
        status: 'failed',
        model: modelName,
        bytes,
        symmetric,
        error: `${NEATRACING_EDITOR_LIVE_SAVE_LABEL}: the model-store save request failed: ${describeError(error)}`,
        rebuild: null,
      };
    }
    if (!saveResponse.ok) {
      return {
        status: 'failed',
        model: modelName,
        bytes,
        symmetric,
        error: `${NEATRACING_EDITOR_LIVE_SAVE_LABEL}: the model-store save request failed with status ${saveResponse.status ?? 'unknown'}`,
        rebuild: null,
      };
    }

    // Step 6: Trigger the rebuild flight. The server's rebuild
    // coordinator coalesces the flights, so one request per stored
    // document is enough.
    const rebuildUrl = `${saveUrl}/rebuild`;
    let rebuildResponse: EditorLiveSaveFetchResponse;
    try {
      rebuildResponse = await fetchImpl(rebuildUrl, { method: 'POST' });
    } catch (error) {
      return {
        status: 'saved',
        model: modelName,
        bytes,
        symmetric,
        error: `${NEATRACING_EDITOR_LIVE_SAVE_LABEL}: the rebuild flight request failed: ${describeError(error)}`,
        rebuild: 'failed',
      };
    }
    if (!rebuildResponse.ok) {
      return {
        status: 'saved',
        model: modelName,
        bytes,
        symmetric,
        error: `${NEATRACING_EDITOR_LIVE_SAVE_LABEL}: the rebuild flight request failed with status ${rebuildResponse.status ?? 'unknown'}`,
        rebuild: 'failed',
      };
    }
    const rebuildCode = await readResponseCode(rebuildResponse);
    return {
      status: 'saved',
      model: modelName,
      bytes,
      symmetric,
      error: null,
      rebuild:
        rebuildCode === REBUILD_COALESCED_CODE ? 'coalesced' : 'started',
    };
  }

  /**
   * Runs the save attempts until no save is queued any more.
   *
   * The loop re-runs while a save was enqueued during the in-flight
   * attempt, so the coalesced enqueues collapse into ONE follow-up
   * request carrying the current volume.
   */
  async function drain(): Promise<void> {
    while (pending && !stopped) {
      pending = false;
      attempts += 1;
      let result: EditorLiveSaveResult;
      try {
        result = await performSave();
      } catch (error) {
        // The save never rejects: an unexpected failure still resolves
        // into a labeled result.
        result = {
          status: 'failed',
          model: null,
          bytes: 0,
          symmetric: false,
          error: `${NEATRACING_EDITOR_LIVE_SAVE_LABEL}: the live save failed: ${describeError(error)}`,
          rebuild: null,
        };
      }
      lastResult = result;
      if (result.status === 'saved') {
        saved += 1;
      }
      if (result.error !== null) {
        console.error(result.error);
      }
    }
  }

  return {
    enqueueSave(): void {
      if (stopped) {
        return;
      }
      pending = true;
      if (flight === null) {
        flight = drain().finally(() => {
          flight = null;
        });
      }
    },

    async idle(): Promise<void> {
      while (flight !== null) {
        await flight;
      }
    },

    snapshot(): EditorLiveSaveSnapshot {
      return { pending, attempts, saved, lastResult };
    },

    stop(): void {
      stopped = true;
      pending = false;
    },
  };
}
