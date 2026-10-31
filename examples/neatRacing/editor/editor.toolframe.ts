/// <reference lib="dom" />

/**
 * Editor tool frame for Neat Racing (slice P1-editor-toolframe; spec
 * section 8.3, AC-020 partial, LD-28).
 *
 * The tool frame is the editor's control surface: it owns the editor state
 * container (the dense 128-cubed working volume plus its palette), the
 * bounded undo stack, and the mirror-symmetry surface, and it mounts the
 * palette list, the undo/redo buttons, the model dropdown, the CAD-view
 * buttons, and the symmetry toggle into the reserved
 * `neatracing-editor-tools` container of the editor shell.
 *
 * The reserved container is resolved or thrown (batch-94 handoff contract):
 * a shell that lost the container must fail loudly at boot instead of
 * silently rendering an editor without controls.
 *
 * Boot also performs the initial model load: the model store is listed
 * (`GET /api/models`), the first listed model is read
 * (`GET /api/models/:name`), and the document is decoded into the working
 * volume — the document's palette is mapped onto the editor palette (the
 * EMPTY sentinel stays index 0) and the decoded cells are written at the
 * volume's `x + row * y + slice * z` addresses. Every load failure degrades
 * silently (LD-1) to an empty working volume, except for the absence of a
 * fetch seam, which is not a failure but a missing dev server.
 *
 * The loaded document is a LIVE surface: `currentDocument` reads the
 * document currently decoded into the working volume on every access, so a
 * later control slice that drives `loadModel()` (the model dropdown) reaches
 * the preview and the session without re-plumbing the boot.
 *
 * The control *behaviors* (palette add/select, undo/redo clicks, the
 * dropdown change handler, the CAD-view toggles, the symmetry toggle) are
 * owned by the later editing-loop slices; this module owns the mounts, the
 * state binding, and the initial model-volume load.
 *
 * @module
 */

import { createEditorPalette } from './editor.palette';
import type { EditorPalette } from './editor.palette';
import {
  NEATRACING_EDITOR_VOLUME_EDGE,
  NEATRACING_EDITOR_VOLUME_ROW_STRIDE,
  NEATRACING_EDITOR_VOLUME_SLICE_STRIDE,
  createEditorState,
} from './editor.state';
import type { EditorState } from './editor.state';
import { createEditorUndo } from './editor.undo';
import type { EditorUndo } from './editor.undo';
import {
  NEATRACING_EDITOR_DEFAULT_MIRROR_X,
  createEditorSymmetry,
} from './editor.symmetry';
import type { EditorSymmetry } from './editor.symmetry';
import { decodeVoxelModel } from '../voxel/model.decode.utils';
import { validateVoxelModelDocument } from '../voxel/model.schema';
import type { VoxelModelDocument } from '../voxel/model.schema';

/**
 * Element id of the reserved editor tool-frame container (the editor
 * shell's `neatracing-editor-tools` aside).
 */
export const NEATRACING_EDITOR_TOOLS_ID = 'neatracing-editor-tools';

/**
 * Element ids of every control the tool frame mounts into the reserved
 * container. The ids are the tool frame's public DOM contract: the later
 * editing-loop slices bind their handlers to these elements.
 */
export const NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS = {
  paletteList: 'neatracing-editor-palette',
  undoButton: 'neatracing-editor-undo',
  redoButton: 'neatracing-editor-redo',
  modelSelect: 'neatracing-editor-model',
  cadFrontButton: 'neatracing-editor-view-front',
  cadSideButton: 'neatracing-editor-view-side',
  cadTopButton: 'neatracing-editor-view-top',
  perspectiveButton: 'neatracing-editor-view-perspective',
  symmetryToggle: 'neatracing-editor-symmetry-x',
} as const;

/** Dev-server model-store list route the tool frame reads model names from. */
export const NEATRACING_EDITOR_MODEL_LIST_PATH = '/api/models';

/** Pass label prefixing every observable tool-frame failure surface. */
const NEATRACING_EDITOR_TOOLFRAME_LABEL = 'neatRacing editor tool frame';

/** Root element class of the mounted tool frame. */
const TOOLFRAME_ROOT_CLASS = 'neatracing-editor-toolframe';

/** Section element class of every mounted tool-frame control group. */
const TOOLFRAME_SECTION_CLASS = 'neatracing-editor-toolframe-section';

/** Palette swatch element class of one rendered palette entry. */
const PALETTE_SWATCH_CLASS = 'neatracing-editor-palette-swatch';

/** Data attribute carrying the palette index of a rendered swatch. */
const PALETTE_INDEX_ATTRIBUTE = 'data-palette-index';

/** Data attribute naming the control group a mounted section belongs to. */
const TOOLFRAME_GROUP_ATTRIBUTE = 'data-neatracing-toolframe';

/**
 * Minimal response shape the tool frame reads from its fetch seam. The
 * browser `Response` satisfies it structurally; the test doubles only
 * implement the two members the tool frame actually consumes.
 */
export interface EditorToolframeFetchResponse {
  /** Whether the request succeeded. */
  readonly ok: boolean;
  /** HTTP status code, when the seam exposes one. */
  readonly status?: number;
  /** Parses the response body as JSON. */
  json(): Promise<unknown>;
}

/**
 * Fetch seam the tool frame drives: it resolves a URL to a JSON response.
 */
export type EditorToolframeFetch = (
  url: string,
) => Promise<EditorToolframeFetchResponse>;

/** Factory configuration for the editor tool frame. */
export interface EditorToolframeConfig {
  /** Reserved container id; defaults to {@linkcode NEATRACING_EDITOR_TOOLS_ID}. */
  readonly containerId?: string;
  /** Document the container is resolved in; defaults to the global document. */
  readonly documentRef?: Document;
  /** Fetch seam; defaults to the browser `fetch` when one is available. */
  readonly fetchImpl?: EditorToolframeFetch;
}

/** The controls the tool frame mounts into the reserved container. */
export interface EditorToolframeControls {
  /** List element rendering one swatch per used palette entry. */
  readonly paletteList: HTMLUListElement;
  /** Undo button (handler owned by the editing-loop slices). */
  readonly undoButton: HTMLButtonElement;
  /** Redo button (handler owned by the editing-loop slices). */
  readonly redoButton: HTMLButtonElement;
  /** Model dropdown listing the model-store names. */
  readonly modelSelect: HTMLSelectElement;
  /** CAD front-view button. */
  readonly cadFrontButton: HTMLButtonElement;
  /** CAD side-view button. */
  readonly cadSideButton: HTMLButtonElement;
  /** CAD top-view button. */
  readonly cadTopButton: HTMLButtonElement;
  /** Orbit (perspective) view button. */
  readonly perspectiveButton: HTMLButtonElement;
  /** Mirror-X symmetry toggle, checked when mirror-X defaults on. */
  readonly symmetryToggle: HTMLInputElement;
}

/** Outcome of a model load into the working volume. */
export interface EditorToolframeLoadResult {
  /** Loaded model name, or null when no model could be loaded. */
  readonly model: string | null;
  /** Loaded model document, or null when no model could be loaded. */
  readonly document: VoxelModelDocument | null;
}

/** Public tool-frame surface the editor boot and the later slices drive. */
export interface EditorToolframeSession {
  /** Editor state container (working volume + palette) the frame owns. */
  readonly state: EditorState;
  /** Palette the editor resolves paint colors through. */
  readonly palette: EditorPalette;
  /** Bounded 3-step undo stack over the working volume. */
  readonly undo: EditorUndo;
  /** Mirror-symmetry surface (mirror-X on by default). */
  readonly symmetry: EditorSymmetry;
  /** The mounted control elements. */
  readonly controls: EditorToolframeControls;
  /** Model names the model store listed. */
  readonly models: readonly string[];
  /** Currently selected model name, or null. */
  readonly selectedModel: string | null;
  /**
   * Live projection of the loaded model document: the document currently
   * decoded into the working volume, or null while no model is loaded.
   *
   * This is a getter, not a one-shot snapshot, so a later `loadModel()` call
   * (the model dropdown) is visible through it without re-plumbing the boot.
   */
  readonly currentDocument: VoxelModelDocument | null;
  /** Settles when the boot-time initial model load finished. */
  readonly ready: Promise<EditorToolframeLoadResult>;
  /**
   * Reads a model from the model store and decodes it into the working
   * volume.
   *
   * @param modelName - Name of the model to load.
   * @returns The load outcome (the model name and its document).
   */
  loadModel(modelName: string): Promise<EditorToolframeLoadResult>;
  /**
   * Re-renders the palette list from the current palette snapshot.
   */
  refreshPaletteList(): void;
  /** Unmounts the tool frame from the reserved container. */
  stop(): void;
}

/**
 * Reports whether a value is a plain object record.
 *
 * @param value - Value to inspect.
 * @returns True when the value is a non-null object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

/**
 * Browser fetch seam: resolves a URL through the global `fetch`.
 *
 * @param url - Absolute or root-relative URL to request.
 * @returns The JSON response.
 */
async function defaultToolframeFetch(
  url: string,
): Promise<EditorToolframeFetchResponse> {
  return window.fetch(url);
}

/**
 * Resolves the fetch seam the tool frame drives.
 *
 * @param config - Optional tool-frame configuration.
 * @returns The injected seam, the browser `fetch`, or null when no fetch
 *   is available (a missing dev server, not a failure).
 */
function resolveToolframeFetch(
  config?: EditorToolframeConfig,
): EditorToolframeFetch | null {
  if (config?.fetchImpl !== undefined) {
    return config.fetchImpl;
  }
  if (typeof window === 'undefined' || typeof window.fetch !== 'function') {
    return null;
  }
  return defaultToolframeFetch;
}

/**
 * Lists the model names the dev-server model store holds.
 *
 * @param fetchImpl - Fetch seam to drive.
 * @returns The listed model names (empty when the store lists none).
 * @throws Error when the list request fails or the body is not an object.
 */
async function listModelNames(
  fetchImpl: EditorToolframeFetch,
): Promise<readonly string[]> {
  const response = await fetchImpl(NEATRACING_EDITOR_MODEL_LIST_PATH);
  if (!response.ok) {
    throw new Error(
      `model list request failed (status ${response.status ?? 'unknown'})`,
    );
  }
  const body: unknown = await response.json();
  if (!isRecord(body)) {
    throw new Error('model list response body must be a JSON object');
  }
  const rawModels: unknown = body['models'];
  if (!Array.isArray(rawModels)) {
    return [];
  }
  return rawModels.filter(
    (name): name is string => typeof name === 'string' && name.length > 0,
  );
}

/**
 * Reads one model document from the dev-server model store.
 *
 * @param fetchImpl - Fetch seam to drive.
 * @param modelName - Name of the model to read.
 * @returns The validated model document.
 * @throws Error when the request fails or the body carries no document.
 */
async function readModelDocument(
  fetchImpl: EditorToolframeFetch,
  modelName: string,
): Promise<VoxelModelDocument> {
  const response = await fetchImpl(
    `${NEATRACING_EDITOR_MODEL_LIST_PATH}/${encodeURIComponent(modelName)}`,
  );
  if (!response.ok) {
    throw new Error(
      `model read request failed (status ${response.status ?? 'unknown'})`,
    );
  }
  const body: unknown = await response.json();
  if (!isRecord(body)) {
    throw new Error('model response body must be a JSON object');
  }
  // The schema validator is the normative section 7.2 gate: an invalid
  // document is rejected here instead of reaching the working volume.
  return validateVoxelModelDocument(body['document']);
}

/**
 * Decodes a model document into the editor working volume.
 *
 * The document palette is mapped onto the editor palette first (the EMPTY
 * sentinel keeps index 0), then every decoded cell is written at the
 * volume's `x + row * y + slice * z` address. The volume is cleared first
 * so a reload never leaves stale voxels behind.
 *
 * @param state - Editor state container owning the working volume.
 * @param palette - Editor palette the document palette is mapped onto.
 * @param document - Validated model document to decode.
 * @throws Error when the document dims exceed the editor working volume.
 */
function applyDocumentToVolume(
  state: EditorState,
  palette: EditorPalette,
  document: VoxelModelDocument,
): void {
  const decoded = decodeVoxelModel(document);
  const [width, height, depth] = decoded.dims;
  if (
    width > NEATRACING_EDITOR_VOLUME_EDGE ||
    height > NEATRACING_EDITOR_VOLUME_EDGE ||
    depth > NEATRACING_EDITOR_VOLUME_EDGE
  ) {
    throw new Error(
      `model "${decoded.name}" dims [${width}, ${height}, ${depth}] exceed the ${NEATRACING_EDITOR_VOLUME_EDGE}-cubed editor working volume`,
    );
  }

  // Step 1: map the document palette onto the editor palette.
  const paletteIndexMap = decoded.palette.map((entry) => palette.addColor(entry));

  // Step 2: clear the working volume, then write the decoded cells.
  state.volume.fill(0);
  for (let sliceIndex = 0; sliceIndex < decoded.slices.length; sliceIndex += 1) {
    const slice = decoded.slices[sliceIndex];
    for (let row = 0; row < height; row += 1) {
      for (let column = 0; column < width; column += 1) {
        const documentIndex = slice[row * width + column];
        state.volume[
          column +
            NEATRACING_EDITOR_VOLUME_ROW_STRIDE * row +
            NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * sliceIndex
        ] = paletteIndexMap[documentIndex];
      }
    }
  }
}

/**
 * Renders one swatch per used palette entry into the palette list.
 *
 * @param paletteList - List element the swatches are rendered into.
 * @param palette - Palette whose snapshot is rendered.
 */
function renderPaletteList(
  paletteList: HTMLUListElement,
  palette: EditorPalette,
): void {
  const snapshot = palette.snapshot();
  const swatches: HTMLLIElement[] = [];
  // Palette index 0 stays the reserved EMPTY sentinel, so the rendered
  // swatches start at index 1 and run to the used color count.
  for (let paletteIndex = 1; paletteIndex <= snapshot.usedCount; paletteIndex += 1) {
    const entry = snapshot.entries[paletteIndex];
    const swatch = paletteList.ownerDocument.createElement('li');
    swatch.className = PALETTE_SWATCH_CLASS;
    swatch.setAttribute(PALETTE_INDEX_ATTRIBUTE, String(paletteIndex));
    swatch.style.backgroundColor = `rgb(${entry[0]}, ${entry[1]}, ${entry[2]})`;
    swatches.push(swatch);
  }
  paletteList.replaceChildren(...swatches);
}

/**
 * Renders the model-store names as dropdown options.
 *
 * @param modelSelect - Select element the options are rendered into.
 * @param modelNames - Model names to render.
 */
function renderModelOptions(
  modelSelect: HTMLSelectElement,
  modelNames: readonly string[],
): void {
  const options = modelNames.map((modelName) => {
    const option = modelSelect.ownerDocument.createElement('option');
    option.value = modelName;
    option.textContent = modelName;
    return option;
  });
  modelSelect.replaceChildren(...options);
}

/**
 * Creates one titled control-group section inside the tool frame.
 *
 * @param documentRef - Document the section is created in.
 * @param root - Tool-frame root the section is appended to.
 * @param group - Control-group name recorded on the section.
 * @param title - Human-readable section title.
 * @returns The created section element.
 */
function appendToolframeSection(
  documentRef: Document,
  root: HTMLElement,
  group: string,
  title: string,
): HTMLElement {
  const section = documentRef.createElement('section');
  section.className = TOOLFRAME_SECTION_CLASS;
  section.setAttribute(TOOLFRAME_GROUP_ATTRIBUTE, group);
  const heading = documentRef.createElement('h2');
  heading.textContent = title;
  section.appendChild(heading);
  root.appendChild(section);
  return section;
}

/**
 * Creates one labeled button inside a control-group section.
 *
 * @param documentRef - Document the button is created in.
 * @param buttonId - Element id of the button.
 * @param label - Human-readable button label.
 * @returns The created button element.
 */
function createToolframeButton(
  documentRef: Document,
  buttonId: string,
  label: string,
): HTMLButtonElement {
  const button = documentRef.createElement('button');
  button.id = buttonId;
  button.type = 'button';
  button.textContent = label;
  return button;
}

/**
 * Mounts the tool-frame controls into the tool-frame root element.
 *
 * @param documentRef - Document the controls are created in.
 * @param root - Tool-frame root the control groups are appended to.
 * @returns The mounted control elements.
 */
function mountToolframeControls(
  documentRef: Document,
  root: HTMLElement,
): EditorToolframeControls {
  const controlIds = NEATRACING_EDITOR_TOOLFRAME_CONTROL_IDS;

  // Step 1: the palette list.
  const paletteSection = appendToolframeSection(
    documentRef,
    root,
    'palette',
    'Palette',
  );
  const paletteList = documentRef.createElement('ul');
  paletteList.id = controlIds.paletteList;
  paletteList.className = 'neatracing-editor-palette-list';
  paletteSection.appendChild(paletteList);

  // Step 2: the undo/redo buttons.
  const historySection = appendToolframeSection(
    documentRef,
    root,
    'history',
    'History',
  );
  const undoButton = createToolframeButton(
    documentRef,
    controlIds.undoButton,
    'Undo',
  );
  const redoButton = createToolframeButton(
    documentRef,
    controlIds.redoButton,
    'Redo',
  );
  historySection.appendChild(undoButton);
  historySection.appendChild(redoButton);

  // Step 3: the model dropdown.
  const modelSection = appendToolframeSection(
    documentRef,
    root,
    'model',
    'Model',
  );
  const modelSelect = documentRef.createElement('select');
  modelSelect.id = controlIds.modelSelect;
  modelSection.appendChild(modelSelect);

  // Step 4: the CAD-view buttons.
  const viewSection = appendToolframeSection(
    documentRef,
    root,
    'view',
    'CAD view',
  );
  const cadFrontButton = createToolframeButton(
    documentRef,
    controlIds.cadFrontButton,
    'Front',
  );
  const cadSideButton = createToolframeButton(
    documentRef,
    controlIds.cadSideButton,
    'Side',
  );
  const cadTopButton = createToolframeButton(
    documentRef,
    controlIds.cadTopButton,
    'Top',
  );
  const perspectiveButton = createToolframeButton(
    documentRef,
    controlIds.perspectiveButton,
    'Perspective',
  );
  viewSection.appendChild(cadFrontButton);
  viewSection.appendChild(cadSideButton);
  viewSection.appendChild(cadTopButton);
  viewSection.appendChild(perspectiveButton);

  // Step 5: the mirror-X symmetry toggle.
  const symmetrySection = appendToolframeSection(
    documentRef,
    root,
    'symmetry',
    'Symmetry',
  );
  const symmetryLabel = documentRef.createElement('label');
  const symmetryToggle = documentRef.createElement('input');
  symmetryToggle.id = controlIds.symmetryToggle;
  symmetryToggle.type = 'checkbox';
  symmetryToggle.checked = NEATRACING_EDITOR_DEFAULT_MIRROR_X;
  symmetryLabel.appendChild(symmetryToggle);
  symmetryLabel.appendChild(
    documentRef.createTextNode(' Mirror X'),
  );
  symmetrySection.appendChild(symmetryLabel);

  return {
    paletteList,
    undoButton,
    redoButton,
    modelSelect,
    cadFrontButton,
    cadSideButton,
    cadTopButton,
    perspectiveButton,
    symmetryToggle,
  };
}

/**
 * Creates the editor tool frame and mounts it into the reserved container.
 *
 * The frame owns the editor state container (working volume plus palette),
 * the bounded undo stack, and the mirror-symmetry surface; it mounts the
 * palette list, the undo/redo buttons, the model dropdown, the CAD-view
 * buttons, and the symmetry toggle into the reserved container; and it
 * starts the initial model load (model-store list, read, and decode into
 * the working volume).
 *
 * @param config - Optional configuration (container id, document, fetch
 *   seam).
 * @returns The tool-frame session.
 * @throws Error when the reserved container cannot be resolved.
 *
 * @example
 * ```ts
 * const toolframe = createEditorToolframe();
 * await toolframe.ready;
 * console.log(toolframe.state.snapshot().model);
 * ```
 */
export function createEditorToolframe(
  config?: EditorToolframeConfig,
): EditorToolframeSession {
  // Step 1: resolve the reserved tool-frame container (resolve-or-throw).
  const documentRef: Document | undefined =
    config?.documentRef ?? globalThis.document;
  if (documentRef === undefined) {
    throw new Error('Editor tool frame requires a DOM document');
  }
  const containerId = config?.containerId ?? NEATRACING_EDITOR_TOOLS_ID;
  const container = documentRef.getElementById(containerId);
  if (container === null) {
    throw new Error(`Editor tool frame host #${containerId} not found`);
  }

  // Step 2: create the editor state container with its own palette, the
  // bounded undo stack over the working volume, and the symmetry surface.
  const palette = createEditorPalette();
  const state = createEditorState({ palette });
  const undo = createEditorUndo(state.volume);
  const symmetry = createEditorSymmetry();

  // Step 3: mount the controls into the reserved container.
  const root = documentRef.createElement('div');
  root.className = TOOLFRAME_ROOT_CLASS;
  const controls = mountToolframeControls(documentRef, root);
  container.appendChild(root);

  // Step 4: run the initial model load (list, read, decode into the
  // working volume). The load is asynchronous, so the session exposes it
  // through `ready`.
  const fetchImpl = resolveToolframeFetch(config);
  let modelNames: readonly string[] = [];
  // Live projection state: the document currently decoded into the working
  // volume. It is read through the `currentDocument` getter on every access,
  // so a later `loadModel()` call is visible without re-plumbing the boot.
  let loadedDocument: VoxelModelDocument | null = null;

  /**
   * Loads one model into the working volume and selects it.
   *
   * @param modelName - Name of the model to load.
   * @returns The load outcome.
   */
  const loadModel = async (
    modelName: string,
  ): Promise<EditorToolframeLoadResult> => {
    if (fetchImpl === null) {
      loadedDocument = null;
      return { model: null, document: null };
    }
    const modelDocument = await readModelDocument(fetchImpl, modelName);
    applyDocumentToVolume(state, palette, modelDocument);
    // Publish the document only after it decoded into the working volume, so
    // the live projection never names a document the volume does not hold.
    loadedDocument = modelDocument;
    state.selectModel(modelName);
    controls.modelSelect.value = modelName;
    renderPaletteList(controls.paletteList, palette);
    return { model: modelName, document: modelDocument };
  };

  const ready = (async (): Promise<EditorToolframeLoadResult> => {
    if (fetchImpl === null) {
      // No dev server behind the editor: the working volume simply stays
      // empty (LD-1), which is not a failure worth surfacing.
      return { model: null, document: null };
    }
    try {
      modelNames = await listModelNames(fetchImpl);
      renderModelOptions(controls.modelSelect, modelNames);
      const firstModelName = modelNames.at(0);
      if (firstModelName === undefined) {
        return { model: null, document: null };
      }
      return await loadModel(firstModelName);
    } catch (error) {
      console.error(
        `${NEATRACING_EDITOR_TOOLFRAME_LABEL}: initial model load failed; the editor starts with an empty working volume`,
        error,
      );
      return { model: null, document: null };
    }
  })();

  return {
    state,
    palette,
    undo,
    symmetry,
    controls,
    get models(): readonly string[] {
      return modelNames;
    },
    get selectedModel(): string | null {
      return state.snapshot().model;
    },
    get currentDocument(): VoxelModelDocument | null {
      return loadedDocument;
    },
    ready,
    loadModel,
    refreshPaletteList(): void {
      renderPaletteList(controls.paletteList, palette);
    },
    stop(): void {
      root.remove();
    },
  };
}
