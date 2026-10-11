/**
 * Editor volume state for NeatRacing (slice P1-S6-A; spec section 8.3 /
 * AC-019 and AC-020 of `plans/neatracing-voxel-racer.spec.md`): the dense
 * 128-cubed working volume the editor paints into before export.
 *
 * The editor is an explicitly permitted exception to the sparse-storage
 * rule (R-07, appendix P7): it owns a single dense
 * {@linkcode NEATRACING_EDITOR_VOLUME_LENGTH} = 128³ = 2,097,152 byte
 * `Uint8Array` (~2.1 MB) where each voxel stores a palette index, and 0
 * stays reserved as the EMPTY sentinel entry (air; never rendered,
 * section 7.2). Voxels address as
 * `x + {@linkcode NEATRACING_EDITOR_VOLUME_ROW_STRIDE} * y +
 * {@linkcode NEATRACING_EDITOR_VOLUME_SLICE_STRIDE} * z`:
 * x = column, y = row (row 0 = down), z = slice (slice 0 = front).
 *
 * {@linkcode EditorState.paint} delegates color resolution to the editor
 * palette (exact-else-nearest, AC-020) and writes the resolved palette
 * index, so painting the same color twice reuses its entry instead of
 * duplicating it, and {@linkcode EditorState.selectModel} records the
 * currently selected model name for the editor shell (null until the
 * model dropdown first resolves a selection).
 *
 * The module stays pure data — no DOM and no WebGL context — so the
 * editor volume is testable in the plain Node Jest environment while the
 * shell and renderer consume its public reads.
 */

import { createEditorPalette } from './editor.palette';
import type { EditorPalette } from './editor.palette';
import type { VoxelPaletteEntry } from '../voxel/model.schema';

/**
 * Edge length, in voxels, of the editor's cubic working volume.
 */
export const NEATRACING_EDITOR_VOLUME_EDGE = 128;

/**
 * Total element count of the editor volume: 128 × 128 × 128 = 2,097,152
 * palette-index bytes (the editor's dense-storage exception; appendix P7).
 */
export const NEATRACING_EDITOR_VOLUME_LENGTH =
  NEATRACING_EDITOR_VOLUME_EDGE *
  NEATRACING_EDITOR_VOLUME_EDGE *
  NEATRACING_EDITOR_VOLUME_EDGE;

/** Elements to skip per step along the y (row) axis. */
export const NEATRACING_EDITOR_VOLUME_ROW_STRIDE =
  NEATRACING_EDITOR_VOLUME_EDGE;

/** Elements to skip per step along the z (slice) axis. */
export const NEATRACING_EDITOR_VOLUME_SLICE_STRIDE =
  NEATRACING_EDITOR_VOLUME_EDGE * NEATRACING_EDITOR_VOLUME_EDGE;

/** Factory configuration for an editor state. */
export interface EditorStateConfig {
  /**
   * Palette the editor paints through; defaults to a fresh
   * `createEditorPalette()` volume-backed palette.
   */
  palette?: EditorPalette;
}

/**
 * Immutable point-in-time view of the editor's document-level state.
 *
 * `model` is the currently selected model name, or `null` while no
 * model selection has been made.
 */
export interface EditorStateSnapshot {
  /** Currently selected model name, or null before any selection. */
  model: string | null;
}

/** Public editor state surface the editor shell drives. */
export interface EditorState {
  /** Dense palette-index working volume, length {@linkcode NEATRACING_EDITOR_VOLUME_LENGTH}. */
  readonly volume: Uint8Array;
  /** Palette the editor resolves paint colors through (AC-020). */
  readonly palette: EditorPalette;

  /**
   * Paints a voxel by writing the palette index the entry resolves to.
   *
   * @param x - Voxel column, inclusive in 0..127.
   * @param y - Voxel row, inclusive in 0..127.
   * @param z - Voxel slice, inclusive in 0..127.
   * @param entry - Palette entry to resolve through the palette.
   * @throws RangeError when a coordinate falls outside the volume.
   * @example
   * ```ts
   * const state = createEditorState();
   * state.paint(10, 20, 30, [255, 0, 0, 255, 0]);
   * ```
   */
  paint(x: number, y: number, z: number, entry: VoxelPaletteEntry): void;

  /**
   * Records the currently selected model name for the editor shell.
   *
   * @param modelName - Model name to select.
   * @example
   * ```ts
   * const state = createEditorState();
   * state.selectModel('car-f1');
   * ```
   */
  selectModel(modelName: string): void;

  /**
   * Returns an immutable point-in-time view of the document-level state.
   *
   * @returns Snapshot with the selected model name (or null).
   * @example
   * ```ts
   * const state = createEditorState();
   * state.selectModel('car-f1');
   * const model = state.snapshot().model;
   * ```
   */
  snapshot(): EditorStateSnapshot;
}

/**
 * Validates one voxel coordinate against the volume edge, reporting the
 * failing axis name in the thrown range error.
 *
 * @param axisName - Axis label for error messages ('x', 'y', or 'z').
 * @param value - Coordinate value to validate.
 * @throws RangeError when `value` falls outside `0..edge-1`.
 */
function validateVoxelCoordinate(axisName: string, value: number): void {
  if (
    !Number.isInteger(value) ||
    value < 0 ||
    value >= NEATRACING_EDITOR_VOLUME_EDGE
  ) {
    throw new RangeError(
      `voxel ${axisName} ${value} is outside the volume range 0..${NEATRACING_EDITOR_VOLUME_EDGE - 1}`,
    );
  }
}

/**
 * Creates the editor state with defaults plus partial overrides.
 *
 * @param config - Partial state configuration to spread over defaults.
 * @returns Ready-to-use editor state implementing {@linkcode EditorState}.
 * @example
 * ```ts
 * import { createEditorState } from './editor.state';
 * const state = createEditorState();
 * state.paint(15, 0, 15, [255, 0, 0, 255, 0]);
 * state.selectModel('car-f1');
 * ```
 */
export function createEditorState(config?: EditorStateConfig): EditorState {
  const palette = config?.palette ?? createEditorPalette();
  const volume = new Uint8Array(NEATRACING_EDITOR_VOLUME_LENGTH);
  let selectedModel: string | null = null;

  return {
    volume,
    palette,

    paint(x, y, z, entry): void {
      validateVoxelCoordinate('x', x);
      validateVoxelCoordinate('y', y);
      validateVoxelCoordinate('z', z);
      volume[
        x +
          NEATRACING_EDITOR_VOLUME_ROW_STRIDE * y +
          NEATRACING_EDITOR_VOLUME_SLICE_STRIDE * z
      ] = palette.addColor(entry);
    },

    selectModel(modelName): void {
      selectedModel = modelName;
    },

    snapshot(): EditorStateSnapshot {
      return { model: selectedModel };
    },
  };
}
