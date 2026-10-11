/**
 * Editor palette state for NeatRacing (slice P1-S6-A; spec section 8.3 /
 * AC-020 of `plans/neatracing-voxel-racer.spec.md`): the pure palette
 * object that manages the on-disk `[r, g, b, a, glow]` entries behind the
 * editor's color picker.
 *
 * Palette index 0 is reserved as the EMPTY sentinel entry
 * (`VOXEL_EMPTY_PALETTE_ENTRY` — air; never rendered, never counted,
 * section 7.2), so real color entries occupy palette indices
 * 1..maxEntries over a default of
 * {@linkcode NEATRACING_EDITOR_PALETTE_MAX_ENTRIES} = 128 slots. Color
 * resolution is exact-else-nearest (AC-020):
 *
 * 1. An entry that already exists exactly (all five components) returns
 *    its existing index without allocating a slot; the all-zero entry
 *    resolves to the EMPTY sentinel index 0 itself.
 * 2. A new entry claims the first free index.
 * 3. A full palette resolves an unknown entry to the nearest existing
 *    color entry by Manhattan RGB distance, lowest palette index winning
 *    ties. Palette index 0 is never a nearest candidate because it is
 *    reserved air, not a paintable color.
 *
 * The module stays pure data — no DOM, no WebGL context, and no RNG — so
 * the palette is testable in the plain Node Jest environment, and the
 * editor shell feeds it color picks while later phases serialize palette
 * entries into on-disk model JSON.
 */

import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  VOXEL_EMPTY_PALETTE_INDEX,
  VOXEL_PALETTE_COMPONENT_MAX,
} from '../voxel/model.schema';
import type { VoxelPaletteEntry } from '../voxel/model.schema';

/**
 * Maximum number of real color entries the editor palette holds (AC-020:
 * the 128-color palette). Palette index 0 is the reserved EMPTY sentinel
 * and does not count against this budget.
 */
export const NEATRACING_EDITOR_PALETTE_MAX_ENTRIES = 128;

/** Palette index reserved as EMPTY (air; never rendered, section 7.2). */
export const NEATRACING_EDITOR_PALETTE_EMPTY_INDEX = VOXEL_EMPTY_PALETTE_INDEX;

/** Inclusive lower bound of the palette byte range (section 7.2). */
const PALETTE_BYTE_MIN = 0;

/** Smallest number of real color slots a palette keeps open. */
const PALETTE_MIN_COLOR_ENTRIES = 1;

/** Alpha component position within a `[r, g, b, a, glow]` entry. */
const PALETTE_ALPHA_COMPONENT = 3;

/** Glow component position within a `[r, g, b, a, glow]` entry. */
const PALETTE_GLOW_COMPONENT = 4;

/** Factory configuration for an editor palette. */
export interface EditorPaletteConfig {
  /**
   * Maximum number of real color entries (palette indices 1..maxEntries).
   * Defaults to {@linkcode NEATRACING_EDITOR_PALETTE_MAX_ENTRIES}.
   */
  maxEntries?: number;
}

/**
 * Immutable point-in-time view of the palette entries and occupancy.
 *
 * `entries` is indexed by palette index. An empty palette exposes exactly
 * the EMPTY sentinel entry at index 0. A palette holding colors leaves
 * index 0 as a hole (runtime `undefined`, despite the declared element
 * type) because that slot is reserved EMPTY air and is never a color
 * entry — `Array.prototype.filter` skips the hole, so
 * `snapshot.entries.filter(Boolean).length` always equals
 * {@linkcode EditorPaletteSnapshot.usedCount}.
 */
export interface EditorPaletteSnapshot {
  /**
   * Palette entries by palette index (detached copies, safe to hold);
   * index 0 is left unfilled (a hole) on non-empty palettes because the
   * slot is reserved EMPTY air.
   */
  entries: VoxelPaletteEntry[];
  /** Number of real color entries (palette indices 1..entries.length - 1). */
  usedCount: number;
}

/** Public editor palette surface the color picker and paint resolver drive. */
export interface EditorPalette {
  /**
   * Resolves an `[r, g, b, a, glow]` entry by exact match first: the
   * all-zero entry resolves to the EMPTY sentinel index 0, a matching
   * existing color returns its index without allocating a slot, a new
   * color claims the first free index, and a full palette returns the
   * nearest existing color entry by Manhattan RGB distance.
   *
   * @param entry - Palette entry to resolve into the palette.
   * @returns Palette index the entry resolves to (never out of range).
   * @example
   * ```ts
   * const palette = createEditorPalette();
   * const index = palette.addColor([255, 0, 0, 255, 0]);
   * ```
   */
  addColor(entry: VoxelPaletteEntry): number;

  /**
   * Tunes a color entry's alpha, clamped into the 0-255 byte range.
   *
   * @param index - Palette index to tune (never the EMPTY sentinel).
   * @param value - Requested alpha component, clamped to 0-255.
   * @throws Error when `index` is the reserved EMPTY sentinel.
   * @throws RangeError when `index` is not a used color index.
   * @example
   * ```ts
   * const palette = createEditorPalette();
   * const index = palette.addColor([255, 0, 0, 255, 0]);
   * palette.setAlpha(index, 128);
   * ```
   */
  setAlpha(index: number, value: number): void;

  /**
   * Tunes a color entry's glow, clamped into the 0-255 byte range.
   *
   * @param index - Palette index to tune (never the EMPTY sentinel).
   * @param value - Requested glow component, clamped to 0-255.
   * @throws Error when `index` is the reserved EMPTY sentinel.
   * @throws RangeError when `index` is not a used color index.
   * @example
   * ```ts
   * const palette = createEditorPalette();
   * const index = palette.addColor([255, 255, 0, 255, 0]);
   * palette.setGlow(index, 96);
   * ```
   */
  setGlow(index: number, value: number): void;

  /**
   * Returns an immutable point-in-time view of the palette entries.
   *
   * @returns Snapshot with detached entry copies safe to hold.
   * @example
   * ```ts
   * const palette = createEditorPalette();
   * palette.addColor([0, 255, 0, 255, 0]);
   * const entries = palette.snapshot().entries;
   * ```
   */
  snapshot(): EditorPaletteSnapshot;
}

interface EditorPaletteState {
  /** Maximum real color capacity (palette indices 1..maxColorEntries). */
  maxColorEntries: number;
  /** Real color entries; palette index = array position + 1. */
  colors: VoxelPaletteEntry[];
}

/**
 * Any palette entry view, including the immutable
 * `VOXEL_EMPTY_PALETTE_ENTRY` sentinel constant from the schema.
 */
type PaletteEntryView = readonly [number, number, number, number, number];

/**
 * Copies a palette entry so stored state never aliases caller input and
 * snapshots never alias living entries.
 *
 * @param entry - Palette entry to copy.
 * @returns Detached `[r, g, b, a, glow]` copy.
 */
function cloneEntry(entry: PaletteEntryView): VoxelPaletteEntry {
  return [entry[0], entry[1], entry[2], entry[3], entry[4]];
}

/**
 * Clamps a numeric component into the 0-255 palette byte range.
 *
 * @param value - Requested component value.
 * @returns Value clamped into the byte range.
 */
function clampToByteRange(value: number): number {
  return Math.min(
    VOXEL_PALETTE_COMPONENT_MAX,
    Math.max(PALETTE_BYTE_MIN, value),
  );
}

/**
 * Compares two palette entries component by component (exact match rule).
 *
 * @param a - First entry.
 * @param b - Second entry.
 * @returns True when every one of the five components matches exactly.
 */
function entriesEqual(a: PaletteEntryView, b: PaletteEntryView): boolean {
  return (
    a[0] === b[0] &&
    a[1] === b[1] &&
    a[2] === b[2] &&
    a[3] === b[3] &&
    a[4] === b[4]
  );
}

/**
 * Measures the Manhattan RGB distance from an entry to target channels.
 *
 * @param entry - Existing color entry to measure from.
 * @param r - Target red channel.
 * @param g - Target green channel.
 * @param b - Target blue channel.
 * @returns Manhattan distance over the RGB channels (0 by construction).
 */
function rgbDistance(
  entry: VoxelPaletteEntry,
  r: number,
  g: number,
  b: number,
): number {
  return (
    Math.abs(entry[0] - r) + Math.abs(entry[1] - g) + Math.abs(entry[2] - b)
  );
}

/**
 * Finds the color entry nearest an entry by Manhattan RGB distance. Only
 * real color entries are candidates — palette index 0 is reserved EMPTY
 * air (section 7.2) and must never resolve a paint. Ties resolve to the
 * lowest palette index for determinism.
 *
 * @param entry - Entry to locate the nearest color for.
 * @param colors - Occupied color entries (palette index = position + 1).
 * @returns Palette index of the nearest existing color entry.
 */
function nearestColorIndex(
  entry: VoxelPaletteEntry,
  colors: readonly VoxelPaletteEntry[],
): number {
  let nearestPosition = -1;
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (let position = 0; position < colors.length; position += 1) {
    const distance = rgbDistance(
      colors[position],
      entry[0],
      entry[1],
      entry[2],
    );
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestPosition = position;
    }
  }
  return nearestPosition + 1;
}

/**
 * Resolves a palette index to its stored color entry, rejecting the
 * reserved EMPTY sentinel and out-of-range indices.
 *
 * @param index - Palette index to resolve.
 * @param state - Palette state the index resolves against.
 * @returns The stored color entry (mutable for tuning).
 * @throws Error when `index` is the reserved EMPTY sentinel.
 * @throws RangeError when `index` is not a used color index.
 */
function requireColorEntry(
  index: number,
  state: EditorPaletteState,
): VoxelPaletteEntry {
  if (index === NEATRACING_EDITOR_PALETTE_EMPTY_INDEX) {
    throw new Error(
      'palette index 0 is reserved as the EMPTY sentinel (air, section 7.2) and cannot be mutated',
    );
  }
  if (
    !Number.isInteger(index) ||
    index < NEATRACING_EDITOR_PALETTE_EMPTY_INDEX + 1 ||
    index > state.colors.length
  ) {
    throw new RangeError(
      `palette index ${index} is outside the used color range 1..${state.colors.length}`,
    );
  }
  return state.colors[index - 1];
}

/**
 * Builds the snapshot entries array: palette index 0 carries the EMPTY
 * sentinel entry on an empty palette, while on any palette holding colors
 * index 0 stays unfilled (a hole) because that slot is reserved EMPTY air
 * and is never a color entry — `Array.prototype.filter` skips the hole,
 * keeping `entries.filter(Boolean).length` equal to the color count.
 * Copies keep snapshots detached from live palette state.
 *
 * @param colors - Occupied color entries (palette index = position + 1).
 * @returns Snapshot entries array indexed by palette index.
 */
function buildSnapshotEntries(
  colors: readonly VoxelPaletteEntry[],
): VoxelPaletteEntry[] {
  const entries: VoxelPaletteEntry[] = new Array<VoxelPaletteEntry>(
    colors.length + 1,
  );
  if (colors.length === 0) {
    entries[0] = cloneEntry(VOXEL_EMPTY_PALETTE_ENTRY);
  } else {
    // Palette index 0 is the reserved EMPTY slot (a hole here, section 7.2).
    for (let position = 0; position < colors.length; position += 1) {
      entries[position + 1] = cloneEntry(colors[position]);
    }
  }
  return entries;
}

/**
 * Creates the editor palette with defaults plus partial overrides.
 *
 * @param config - Partial palette configuration to spread over defaults.
 * @returns Ready-to-use palette implementing {@linkcode EditorPalette}.
 * @example
 * ```ts
 * import { createEditorPalette } from './editor.palette';
 * const palette = createEditorPalette({ maxEntries: 4 });
 * const redIndex = palette.addColor([255, 0, 0, 255, 0]);
 * ```
 */
export function createEditorPalette(
  config?: EditorPaletteConfig,
): EditorPalette {
  const state: EditorPaletteState = {
    maxColorEntries:
      config?.maxEntries ?? NEATRACING_EDITOR_PALETTE_MAX_ENTRIES,
    colors: [],
  };
  if (state.maxColorEntries < PALETTE_MIN_COLOR_ENTRIES) {
    state.maxColorEntries = PALETTE_MIN_COLOR_ENTRIES;
  }
  if (state.maxColorEntries > NEATRACING_EDITOR_PALETTE_MAX_ENTRIES) {
    state.maxColorEntries = NEATRACING_EDITOR_PALETTE_MAX_ENTRIES;
  }

  return {
    addColor(entry): number {
      if (entriesEqual(entry, VOXEL_EMPTY_PALETTE_ENTRY)) {
        return NEATRACING_EDITOR_PALETTE_EMPTY_INDEX;
      }
      for (let position = 0; position < state.colors.length; position += 1) {
        if (entriesEqual(state.colors[position], entry)) {
          return position + 1;
        }
      }
      if (state.colors.length < state.maxColorEntries) {
        state.colors.push(cloneEntry(entry));
        return state.colors.length;
      }
      return nearestColorIndex(entry, state.colors);
    },

    setAlpha(index, value): void {
      requireColorEntry(index, state)[PALETTE_ALPHA_COMPONENT] =
        clampToByteRange(value);
    },

    setGlow(index, value): void {
      requireColorEntry(index, state)[PALETTE_GLOW_COMPONENT] =
        clampToByteRange(value);
    },

    snapshot(): EditorPaletteSnapshot {
      return {
        entries: buildSnapshotEntries(state.colors),
        usedCount: state.colors.length,
      };
    },
  };
}
