/**
 * Red-first contract tests for the NeatRacing editor palette and dropdown
 * state (slice P1-S6-A; spec section 8.3 / AC-020).
 *
 * Locks the palette contract: the on-disk `[r, g, b, a, glow]` entries are
 * managed through a pure palette object; index 0 stays the EMPTY sentinel
 * `[0, 0, 0, 0, 0]`, adding a color that already exists returns the existing
 * index (exact match), adding a color to a full palette returns the nearest
 * existing entry by RGB distance, and alpha/glow can be tuned with 0-255
 * clamping. The empty sentinel is immutable by design.
 *
 * The state volume half of the slice contract: `createEditorState` owns one
 * dense 128^3 `Uint8Array` palette-index working volume (the R-07 P7 editor
 * exception, exactly 2,097,152 bytes ≈ 2.1 MB) plus the palette it resolves
 * against (injected via `createEditorState({ palette })`, defaulted
 * otherwise). `state.paint(x, y, z, entry)` resolves the entry exact-else-
 * nearest against that palette and writes the resolved index at layout index
 * `x + 128 * y + 16384 * z` (x col, y row with row 0 = down, z front slice).
 *
 * The editor state slice is also covered here because the P1-S6-A slice
 * couples palette add/tune with the model dropdown state; the state container
 * starts with no selected model and can switch to a named model.
 *
 * Both modules are pure data/state — no DOM, no WebGL context, and no RNG —
 * so every contract is exercised through the `createEditorPalette` and
 * `createEditorState` factories in the plain Node Jest environment.
 */
import { describe, expect, it } from '@jest/globals';
import {
  createEditorPalette,
  NEATRACING_EDITOR_PALETTE_EMPTY_INDEX,
  NEATRACING_EDITOR_PALETTE_MAX_ENTRIES,
} from './editor.palette';
import { createEditorState } from './editor.state';
import type { VoxelPaletteEntry } from '../voxel/model.schema';

/** A solid red palette entry with no glow. */
const redEntry: VoxelPaletteEntry = [255, 0, 0, 255, 0];

/** A solid green palette entry with no glow. */
const greenEntry: VoxelPaletteEntry = [0, 255, 0, 255, 0];

/** A solid blue palette entry with no glow. */
const blueEntry: VoxelPaletteEntry = [0, 0, 255, 255, 0];

/** A solid white palette entry with no glow. */
const whiteEntry: VoxelPaletteEntry = [255, 255, 255, 255, 0];

describe('editor.palette (P1-S6-A; spec AC-020)', () => {
  describe('palette defaults', () => {
    it('exposes the 128-color max entry constant', () => {
      expect(NEATRACING_EDITOR_PALETTE_MAX_ENTRIES).toBe(128);
    });

    it('reserves index 0 as the empty sentinel', () => {
      const palette = createEditorPalette();
      const snapshot = palette.snapshot();
      expect(NEATRACING_EDITOR_PALETTE_EMPTY_INDEX).toBe(0);
      expect(snapshot.entries[0]).toEqual([0, 0, 0, 0, 0]);
    });
  });

  describe('exact + nearest color matching', () => {
    it('returns the first free index when adding a new color', () => {
      const palette = createEditorPalette();
      const index = palette.addColor(redEntry);
      expect(index).toBe(1);
      expect(palette.snapshot().entries[index]).toEqual(redEntry);
    });

    it('returns the existing index for an exact color match', () => {
      const palette = createEditorPalette();
      const first = palette.addColor(redEntry);
      const second = palette.addColor([...redEntry] as VoxelPaletteEntry);
      expect(first).toBe(second);
      expect(palette.snapshot().entries.filter(Boolean).length).toBe(1);
    });

    it('returns the nearest existing index when the palette is full', () => {
      const palette = createEditorPalette({ maxEntries: 4 });
      palette.addColor(redEntry);
      palette.addColor(greenEntry);
      palette.addColor(blueEntry);
      palette.addColor(whiteEntry);
      const nearest = palette.addColor([
        250, 5, 5, 255, 0,
      ] as VoxelPaletteEntry);
      expect(nearest).toBe(1);
    });
  });

  describe('alpha and glow tuning', () => {
    it('updates alpha and clamps it to the 0-255 byte range', () => {
      const palette = createEditorPalette();
      const index = palette.addColor(redEntry);
      palette.setAlpha(index, 128);
      expect(palette.snapshot().entries[index][3]).toBe(128);
      palette.setAlpha(index, 300);
      expect(palette.snapshot().entries[index][3]).toBe(255);
      palette.setAlpha(index, -10);
      expect(palette.snapshot().entries[index][3]).toBe(0);
    });

    it('updates glow and clamps it to the 0-255 byte range', () => {
      const palette = createEditorPalette();
      const index = palette.addColor(redEntry);
      palette.setGlow(index, 64);
      expect(palette.snapshot().entries[index][4]).toBe(64);
      palette.setGlow(index, 400);
      expect(palette.snapshot().entries[index][4]).toBe(255);
      palette.setGlow(index, -3);
      expect(palette.snapshot().entries[index][4]).toBe(0);
    });

    it('rejects mutations of the empty sentinel at index 0', () => {
      const palette = createEditorPalette();
      expect(() => palette.setAlpha(0, 255)).toThrow(/sentinel|empty|index 0/);
      expect(() => palette.setGlow(0, 255)).toThrow(/sentinel|empty|index 0/);
    });
  });

  describe('painting into the editor state volume', () => {
    it('exposes the dense editor working volume within the 2.1 MB editor budget', () => {
      const state = createEditorState();
      expect(state.volume).toBeInstanceOf(Uint8Array);
      expect(state.volume.length).toBe(128 * 128 * 128);
    });

    it('paints a new color resolved by its palette into a voxel of the volume', () => {
      const state = createEditorState();
      state.paint(10, 20, 30, redEntry);
      const paintedIndex = state.volume[10 + 128 * 20 + 16384 * 30];
      expect(paintedIndex).toBe(1);
      expect(state.palette.snapshot().entries[paintedIndex]).toEqual(redEntry);
    });

    it('falls back to the nearest palette entry when painting with a full palette', () => {
      const palette = createEditorPalette({ maxEntries: 4 });
      const state = createEditorState({ palette });
      palette.addColor(redEntry);
      palette.addColor(greenEntry);
      palette.addColor(blueEntry);
      palette.addColor(whiteEntry);
      state.paint(0, 0, 0, [250, 5, 5, 255, 0] as VoxelPaletteEntry);
      expect(state.volume[0]).toBe(1);
      expect(state.palette.snapshot().entries.filter(Boolean).length).toBe(4);
    });
  });

  describe('editor state dropdown', () => {
    it('initializes with no selected model', () => {
      const state = createEditorState();
      expect(state.snapshot().model).toBeNull();
    });

    it('selects a model by name', () => {
      const state = createEditorState();
      state.selectModel('car-f1');
      expect(state.snapshot().model).toBe('car-f1');
    });
  });
});
