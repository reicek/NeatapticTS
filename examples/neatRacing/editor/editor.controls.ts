/// <reference lib="dom" />

/**
 * Editor control behaviors for Neat Racing (slice P1-editor-controls; spec
 * section 8.3, AC-020 completion, LD-28).
 *
 * The tool frame (`./editor.toolframe`) owns the control MOUNTS — the
 * palette list, the undo/redo buttons, the model dropdown, and the CAD-view
 * buttons — but it deliberately owns no control BEHAVIOR. This module binds
 * those mounts to the editor state:
 *
 * - **Palette add/select.** The tool frame renders one swatch per used
 *   palette entry; clicking a swatch selects it, and the module mounts its
 *   own color input plus an "Add color" button into the palette control
 *   group. A new color resolves through `createEditorPalette`, which caps
 *   the palette at `NEATRACING_EDITOR_PALETTE_MAX_ENTRIES` and resolves an
 *   unknown color against a full palette to the nearest existing entry
 *   (exact-else-nearest, AC-020).
 * - **Undo/redo.** The undo button restores the previous working volume
 *   through the bounded 3-step `createEditorUndo` stack (the oldest
 *   snapshot is evicted beyond the cap). The redo button re-applies the
 *   undone state through the SAME snapshot mechanism: the module keeps a
 *   second bounded undo stack as the redo history, and the redo path
 *   restores a dense snapshot into the same working volume. A new edit
 *   invalidates the redo history (the standard editor contract).
 * - **Model dropdown.** A dropdown `change` loads the selected model from
 *   the model store, decodes it into the working volume, and selects it in
 *   the editor state. A failed load degrades to the previous model (LD-1).
 * - **CAD views.** The front/side/top buttons snap the editor camera to
 *   the CAD orthographic views and the orbit button returns it to the
 *   perspective view (AC-019).
 *
 * The module also owns the shared PAINT ENTRY: the pointer edit loop
 * (`./editor.edit.loop`) captures its paint entry once at construction, so
 * the module hands out one mutable palette-entry tuple and mutates it IN
 * PLACE whenever the selection changes. Painting therefore follows the
 * selected color without re-plumbing the edit loop.
 *
 * Paint semantics (obs 6): the editor REPLACES a voxel with the resolved
 * palette index of the selected color — there is no additive mode — so
 * painting over an occupied voxel recolors it, and painting air (the EMPTY
 * sentinel, palette index 0) clears it.
 *
 * The module keeps no DOM state beyond the two affordances it mounts and
 * the listeners it registers: `stop()` removes both.
 *
 * @module
 */

import { VOXEL_EMPTY_PALETTE_ENTRY } from '../voxel/model.schema';
import type { VoxelPaletteEntry } from '../voxel/model.schema';
import { NEATRACING_EDITOR_PALETTE_EMPTY_INDEX } from './editor.palette';
import { createEditorUndo } from './editor.undo';
import type { EditorUndo } from './editor.undo';
import { NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY } from './editor.edit.loop';
import type { EditorCamera } from './editor.camera';
import type { EditorToolframeSession } from './editor.toolframe';

/** Ids of the affordances this module mounts next to the palette list. */
export const NEATRACING_EDITOR_CONTROL_IDS = {
  /** Color input the "Add color" button reads the new palette entry from. */
  paletteColorInput: 'neatracing-editor-palette-color',
  /** Button that adds the color input's color to the editor palette. */
  paletteAddButton: 'neatracing-editor-palette-add',
} as const;

/**
 * Attribute the tool frame stamps on every rendered palette swatch.
 *
 * The attribute name is owned by the tool frame's renderer; this module
 * mirrors it as a local constant because the swatch selection is read
 * through event delegation on the palette list (the tool frame replaces the
 * list's children on every palette change, so a per-swatch listener would
 * not survive a re-render).
 */
const PALETTE_INDEX_ATTRIBUTE = 'data-palette-index';

/**
 * Attribute the module stamps on the selected palette swatch.
 */
const PALETTE_SELECTED_ATTRIBUTE = 'data-selected';

/** Alpha component written for colors picked in the color input. */
const PICKED_COLOR_ALPHA = 255;

/** Glow component written for colors picked in the color input. */
const PICKED_COLOR_GLOW = 0;

/** Label prefixing every observable failure surface of this module. */
const NEATRACING_EDITOR_CONTROLS_LABEL = 'neatRacing editor controls';

/** Factory configuration for the editor control behaviors. */
export interface EditorControlsConfig {
  /** Tool frame whose control mounts the behaviors bind to. */
  toolframe: EditorToolframeSession;
  /** Editor camera the CAD-view buttons drive. */
  camera: EditorCamera;
  /**
   * Called after the working volume changed underneath the controls (a model
   * load, an undo, or a redo) so the host can republish the volume's
   * section 7.2 projection.
   */
  onWorkingVolumeChanged?: () => void;
}

/**
 * Public editor control surface the editor shell drives.
 */
export interface EditorControls {
  /**
   * Shared paint entry the pointer edit loop paints with. The tuple is
   * mutated IN PLACE when the palette selection changes, so a paint always
   * uses the currently selected color.
   */
  readonly paintEntry: VoxelPaletteEntry;
  /**
   * Undo seam the pointer edit loop captures its per-stroke snapshot
   * through. A push invalidates the redo history, so a new edit can never
   * be redone into a state that no longer exists.
   */
  readonly undo: EditorUndo;
  /**
   * Palette index of the selected swatch, or
   * {@linkcode NEATRACING_EDITOR_PALETTE_EMPTY_INDEX} while no color is
   * selected (painting then clears the voxel).
   */
  readonly selectedPaletteIndex: number;
  /**
   * Adds a color to the editor palette, selects it, and re-renders the
   * palette list. A full palette resolves the color to the nearest existing
   * entry instead of allocating a slot (AC-020).
   *
   * @param entry - Palette entry to add.
   * @returns The palette index the entry resolves to.
   */
  addPaletteEntry(entry: VoxelPaletteEntry): number;
  /**
   * Selects a palette entry and drives the shared paint entry with it.
   *
   * @param index - Palette index to select.
   * @throws RangeError when the index is outside the used palette range.
   */
  selectPaletteEntry(index: number): void;
  /**
   * Loads a model from the model store into the working volume and selects
   * it in the editor state. A failed load degrades to the previous model.
   *
   * @param modelName - Name of the model to load.
   * @returns A promise that settles when the load finished.
   */
  loadModel(modelName: string): Promise<void>;
  /**
   * Restores the previous working volume through the bounded undo stack and
   * captures the undone state for the redo history.
   *
   * @returns true when a snapshot was restored.
   */
  undoStep(): boolean;
  /**
   * Re-applies the last undone state through the redo history.
   *
   * @returns true when a snapshot was re-applied.
   */
  redoStep(): boolean;
  /**
   * Reports whether an undo restore is available right now.
   *
   * @returns true when at least one undo snapshot is retained.
   */
  canUndo(): boolean;
  /**
   * Reports whether a redo is available right now.
   *
   * @returns true when the redo history holds a snapshot.
   */
  canRedo(): boolean;
  /**
   * Resolves once the pending model load settled.
   *
   * @returns A promise that settles after the pending load.
   */
  whenIdle(): Promise<void>;
  /** Removes the mounted affordances and every registered listener. */
  stop(): void;
}

/**
 * Converts a color input's `#rrggbb` value into a palette entry.
 *
 * @param hexValue - Color input value (a six-digit hex triple).
 * @returns The `[r, g, b, a, glow]` palette entry the value names.
 */
function readPickedColorEntry(hexValue: string): VoxelPaletteEntry {
  const normalized = hexValue.replace(/^#/, '');
  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  const safe = (value: number): number =>
    Number.isFinite(value) ? Math.min(255, Math.max(0, value)) : 0;
  return [
    safe(red),
    safe(green),
    safe(blue),
    PICKED_COLOR_ALPHA,
    PICKED_COLOR_GLOW,
  ];
}

/**
 * Creates the editor control behaviors over a mounted tool frame.
 *
 * @param config - Tool frame, camera, and the working-volume change hook.
 * @returns Ready-to-drive controls implementing {@linkcode EditorControls}.
 * @example
 * ```ts
 * const controls = createEditorControls({ toolframe, camera });
 * controls.selectPaletteEntry(controls.addPaletteEntry([255, 0, 0, 255, 0]));
 * ```
 */
export function createEditorControls(
  config: EditorControlsConfig,
): EditorControls {
  const { toolframe, camera } = config;
  const palette = toolframe.palette;
  const volume = toolframe.state.volume;
  const controls = toolframe.controls;

  // The shared paint entry: the pointer edit loop captures the tuple by
  // reference, so mutating it in place retargets every later paint.
  const paintEntry: VoxelPaletteEntry = [
    ...NEATRACING_EDITOR_DEFAULT_PAINT_ENTRY,
  ];
  /** Palette index of the selected swatch (0 = air / no color selected). */
  let selectedPaletteIndex: number = NEATRACING_EDITOR_PALETTE_EMPTY_INDEX;
  /**
   * Redo history: a second bounded undo stack over the SAME working
   * volume, so the redo path restores dense snapshots through the same
   * mechanism the undo path uses. `null` means "no redo history" (the
   * snapshots are released).
   */
  let redoStack: EditorUndo | null = null;
  /** The pending model load, tracked so `whenIdle` can await it. */
  let pendingLoad: Promise<void> = Promise.resolve();

  /**
   * Returns the redo history, creating it on first use.
   *
   * @returns The bounded redo stack over the working volume.
   */
  function getRedoStack(): EditorUndo {
    redoStack ??= createEditorUndo(volume);
    return redoStack;
  }

  /**
   * Marks the selected swatch in the rendered palette list.
   */
  function updateSelectionMarkers(): void {
    const swatches = controls.paletteList.querySelectorAll('li');
    swatches.forEach((swatch) => {
      const index = Number.parseInt(
        swatch.getAttribute(PALETTE_INDEX_ATTRIBUTE) ?? '',
        10,
      );
      if (index === selectedPaletteIndex) {
        swatch.setAttribute(PALETTE_SELECTED_ATTRIBUTE, 'true');
      } else {
        swatch.removeAttribute(PALETTE_SELECTED_ATTRIBUTE);
      }
    });
  }

  /**
   * Copies a palette entry into the shared paint entry in place.
   *
   * @param entry - Palette entry the paint should use.
   */
  function applyPaintEntry(entry: VoxelPaletteEntry): void {
    paintEntry[0] = entry[0];
    paintEntry[1] = entry[1];
    paintEntry[2] = entry[2];
    paintEntry[3] = entry[3];
    paintEntry[4] = entry[4];
  }

  /**
   * Notifies the host that the working volume changed underneath.
   */
  function notifyWorkingVolumeChanged(): void {
    config.onWorkingVolumeChanged?.();
  }

  /**
   * Selects a palette entry and drives the shared paint entry with it.
   *
   * @param index - Palette index to select.
   * @throws RangeError when the index is outside the used palette range.
   */
  function selectPaletteEntry(index: number): void {
    const snapshot = palette.snapshot();
    if (
      !Number.isInteger(index) ||
      index < NEATRACING_EDITOR_PALETTE_EMPTY_INDEX ||
      index > snapshot.usedCount
    ) {
      throw new RangeError(
        `palette index ${index} is outside the used range ${NEATRACING_EDITOR_PALETTE_EMPTY_INDEX}..${snapshot.usedCount}`,
      );
    }
    selectedPaletteIndex = index;
    // Palette index 0 is the reserved EMPTY slot: on a palette holding
    // colors the snapshot leaves it unfilled (a hole), and selecting it
    // means "paint air" (the voxel is cleared).
    const entry = snapshot.entries[index] as VoxelPaletteEntry | undefined;
    applyPaintEntry(
      entry === undefined
        ? ([...VOXEL_EMPTY_PALETTE_ENTRY] as VoxelPaletteEntry)
        : entry,
    );
    updateSelectionMarkers();
  }

  /**
   * Adds a color to the editor palette and selects it.
   *
   * @param entry - Palette entry to add.
   * @returns The palette index the entry resolves to.
   */
  function addPaletteEntry(entry: VoxelPaletteEntry): number {
    const index = palette.addColor(entry);
    toolframe.refreshPaletteList();
    selectPaletteEntry(index);
    return index;
  }

  /**
   * Restores the previous working volume and captures the undone state for
   * the redo history.
   *
   * @returns true when a snapshot was restored.
   */
  function undoStep(): boolean {
    const restored = undoProxy.undo();
    if (restored) {
      notifyWorkingVolumeChanged();
    }
    return restored;
  }

  /**
   * Re-applies the last undone state through the redo history.
   *
   * @returns true when a snapshot was re-applied.
   */
  function redoStep(): boolean {
    const stack = redoStack;
    if (stack === null || !stack.canUndo()) {
      return false;
    }
    // The redo itself stays undoable: the current state is captured on the
    // undo stack BEFORE the redo restores the undone one.
    toolframe.undo.push();
    const restored = stack.undo();
    if (restored) {
      notifyWorkingVolumeChanged();
    }
    return restored;
  }

  /**
   * Undo seam the pointer edit loop captures its per-stroke snapshot
   * through: a push invalidates the redo history, and an undo captures the
   * state being left so the redo path can re-apply it.
   */
  const undoProxy: EditorUndo = {
    push(): void {
      // A new edit invalidates the redo history (standard editor contract).
      redoStack = null;
      toolframe.undo.push();
    },
    undo(): boolean {
      if (!toolframe.undo.canUndo()) {
        return false;
      }
      // Capture the state being left BEFORE the restore, so the redo path
      // re-applies it through the same dense-snapshot mechanism.
      getRedoStack().push();
      return toolframe.undo.undo();
    },
    size(): number {
      return toolframe.undo.size();
    },
    canUndo(): boolean {
      return toolframe.undo.canUndo();
    },
    bytesInUse(): number {
      return toolframe.undo.bytesInUse();
    },
  };

  /**
   * Loads a model into the working volume and selects it.
   *
   * @param modelName - Name of the model to load.
   * @returns A promise that settles when the load finished.
   */
  async function loadModel(modelName: string): Promise<void> {
    try {
      await toolframe.loadModel(modelName);
      // The load may have grown the palette; keep the selection and the
      // shared paint entry consistent with the re-rendered swatches.
      const snapshot = palette.snapshot();
      if (selectedPaletteIndex > snapshot.usedCount) {
        selectedPaletteIndex = NEATRACING_EDITOR_PALETTE_EMPTY_INDEX;
      }
      const entry = snapshot.entries[selectedPaletteIndex] as
        VoxelPaletteEntry | undefined;
      if (entry !== undefined) {
        applyPaintEntry(entry);
      }
      updateSelectionMarkers();
      notifyWorkingVolumeChanged();
    } catch (error) {
      console.error(
        `${NEATRACING_EDITOR_CONTROLS_LABEL}: model load failed; the editor keeps the previous model`,
        error,
      );
    }
  }

  /**
   * Tracks a model load so `whenIdle` can await it.
   *
   * @param modelName - Name of the model to load.
   * @returns A promise that settles when the load finished.
   */
  function trackLoad(modelName: string): Promise<void> {
    const load = loadModel(modelName);
    pendingLoad = load;
    return load;
  }

  /**
   * Handles a palette-list click: the clicked swatch becomes the selected
   * palette entry.
   *
   * @param event - Click event raised inside the palette list.
   */
  function handlePaletteListClick(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const indexAttribute = target.getAttribute(PALETTE_INDEX_ATTRIBUTE);
    if (indexAttribute === null) {
      return;
    }
    const index = Number.parseInt(indexAttribute, 10);
    if (!Number.isInteger(index)) {
      return;
    }
    selectPaletteEntry(index);
  }

  /**
   * Handles the "Add color" click: the color input's color joins the
   * palette and becomes the selected entry.
   */
  function handleAddColor(): void {
    addPaletteEntry(readPickedColorEntry(colorInput.value));
  }

  /**
   * Handles the model dropdown change: the selected model is loaded into
   * the working volume.
   *
   * @returns A promise that settles when the load finished.
   */
  function handleModelChange(): Promise<void> {
    return trackLoad(controls.modelSelect.value);
  }

  // The color affordance lives next to the palette list: the tool frame
  // mounts no color input of its own.
  const documentRef = controls.paletteList.ownerDocument;
  const colorInput = documentRef.createElement('input');
  colorInput.id = NEATRACING_EDITOR_CONTROL_IDS.paletteColorInput;
  colorInput.type = 'color';
  colorInput.value = '#ffffff';
  const addButton = documentRef.createElement('button');
  addButton.id = NEATRACING_EDITOR_CONTROL_IDS.paletteAddButton;
  addButton.type = 'button';
  addButton.textContent = 'Add color';
  const paletteSection = controls.paletteList.parentElement;
  paletteSection?.append(colorInput, addButton);

  // Step 1: bind the palette selection and the add affordance.
  controls.paletteList.addEventListener('click', handlePaletteListClick);
  addButton.addEventListener('click', handleAddColor);

  // Step 2: bind the undo/redo buttons.
  controls.undoButton.addEventListener('click', handleUndoClick);
  controls.redoButton.addEventListener('click', handleRedoClick);

  // Step 3: bind the model dropdown.
  controls.modelSelect.addEventListener('change', handleModelChangeEvent);

  // Step 4: bind the CAD-view buttons (AC-019).
  controls.cadFrontButton.addEventListener('click', handleCadFrontClick);
  controls.cadSideButton.addEventListener('click', handleCadSideClick);
  controls.cadTopButton.addEventListener('click', handleCadTopClick);
  controls.perspectiveButton.addEventListener('click', handlePerspectiveClick);

  /**
   * Handles the undo button click.
   */
  function handleUndoClick(): void {
    undoStep();
  }

  /**
   * Handles the redo button click.
   */
  function handleRedoClick(): void {
    redoStep();
  }

  /**
   * Handles the model dropdown change event.
   */
  function handleModelChangeEvent(): void {
    void handleModelChange();
  }

  /**
   * Handles the CAD front-view button click.
   */
  function handleCadFrontClick(): void {
    camera.setOrthoView('front');
  }

  /**
   * Handles the CAD side-view button click.
   */
  function handleCadSideClick(): void {
    camera.setOrthoView('side');
  }

  /**
   * Handles the CAD top-view button click.
   */
  function handleCadTopClick(): void {
    camera.setOrthoView('top');
  }

  /**
   * Handles the perspective-view button click.
   */
  function handlePerspectiveClick(): void {
    camera.setPerspectiveView();
  }

  return {
    paintEntry,
    undo: undoProxy,
    get selectedPaletteIndex(): number {
      return selectedPaletteIndex;
    },
    addPaletteEntry,
    selectPaletteEntry,
    loadModel: (modelName: string) => trackLoad(modelName),
    undoStep,
    redoStep,
    canUndo: (): boolean => toolframe.undo.canUndo(),
    canRedo: (): boolean => redoStack?.canUndo() ?? false,
    whenIdle: (): Promise<void> => pendingLoad,
    stop(): void {
      controls.paletteList.removeEventListener('click', handlePaletteListClick);
      addButton.removeEventListener('click', handleAddColor);
      controls.undoButton.removeEventListener('click', handleUndoClick);
      controls.redoButton.removeEventListener('click', handleRedoClick);
      controls.modelSelect.removeEventListener(
        'change',
        handleModelChangeEvent,
      );
      controls.cadFrontButton.removeEventListener('click', handleCadFrontClick);
      controls.cadSideButton.removeEventListener('click', handleCadSideClick);
      controls.cadTopButton.removeEventListener('click', handleCadTopClick);
      controls.perspectiveButton.removeEventListener(
        'click',
        handlePerspectiveClick,
      );
      colorInput.remove();
      addButton.remove();
      redoStack = null;
    },
  };
}
