/**
 * Contract tests for the Neat Racing team-color remap (slice P1-S4-B;
 * AC-014, spec sections 6.5 / LD-24).
 *
 * These tests pin the Neatenstein agent-parity remap contract: the
 * designated team-livery palette slot remaps its RGB channels to one of the
 * four named team colors (cyan, red, green, blue) deterministically at draw
 * time while alpha and glow bytes stay verbatim, every non-designated
 * palette entry — EMPTY sentinel, body, glass, black — is untouched by every
 * team's remap (no cross-team bleed), input palettes are never mutated, and
 * palettes without the designated slot pass through unchanged.
 */
import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_TEAM_ID_BLUE,
  NEATRACING_TEAM_ID_CYAN,
  NEATRACING_TEAM_ID_GREEN,
  NEATRACING_TEAM_ID_RED,
  NEATRACING_TEAM_REMAP_SLOT,
  NEATRACING_TEAM_COLOR_COUNT,
  NEATRACING_TEAM_COLORS,
  VoxelTeamRemapError,
  remapVoxelPaletteToTeam,
  resolveVoxelTeamColor,
  type VoxelTeamColorId,
} from './team-remap';
import type { VoxelPaletteEntry } from './model.schema';

/** Authoring-time team-livery placeholder occupying the designated slot (opaque, glow 24). */
const TEAM_SLOT_LIVERY: VoxelPaletteEntry = [230, 230, 230, 255, 24];

/** Opaque body color shared by every team (untouched by every team remap). */
const BODY_INDIGO: VoxelPaletteEntry = [10, 12, 40, 255, 0];

/** Light-blue shining-glass tint shared by every team (untouched by every team remap). */
const GLASS_BLUE_GLOWING: VoxelPaletteEntry = [190, 225, 255, 0, 160];

/** Black solid trim with glow shared by every team (untouched by every team remap). */
const BLACK_SOLID_GLOWING: VoxelPaletteEntry = [0, 0, 0, 255, 200];

/** Glowing decal color shared by every team (untouched by every team remap). */
const BODY_BLUE_GLOWING: VoxelPaletteEntry = [40, 90, 200, 255, 90];

/** Fixture car palette covering every entry class plus the team-livery slot at index 1. */
const FIXTURE_PALETTE: readonly VoxelPaletteEntry[] = [
  [0, 0, 0, 0, 0], // index 0: EMPTY sentinel
  TEAM_SLOT_LIVERY, // index 1: designated team-livery slot
  BODY_INDIGO, // index 2: opaque body
  GLASS_BLUE_GLOWING, // index 3: shining glass
  BLACK_SOLID_GLOWING, // index 4: black trim
  BODY_BLUE_GLOWING, // index 5: glowing decal
];

/** Team color ids in the documented table order (cyan, red, green, blue). */
const ALL_TEAM_COLOR_IDS: readonly VoxelTeamColorId[] = [
  NEATRACING_TEAM_ID_CYAN,
  NEATRACING_TEAM_ID_RED,
  NEATRACING_TEAM_ID_GREEN,
  NEATRACING_TEAM_ID_BLUE,
];

describe('team-remap (slice P1-S4-B: AC-014, spec 6.5, LD-24)', () => {
  describe('remapVoxelPaletteToTeam (designated team-livery slot)', () => {
    it('remaps the designated slot rgb to each team color while preserving alpha and glow', () => {
      for (const team of ALL_TEAM_COLOR_IDS) {
        const remapped = remapVoxelPaletteToTeam(FIXTURE_PALETTE, team);
        const slotEntry = remapped[NEATRACING_TEAM_REMAP_SLOT];
        expect(slotEntry.slice(0, 3)).toEqual(resolveVoxelTeamColor(team));
        expect(slotEntry[3]).toBe(255);
        expect(slotEntry[4]).toBe(24);
      }
    });

    it('alters only the designated slot index across every team remap', () => {
      for (const team of ALL_TEAM_COLOR_IDS) {
        const remapped = remapVoxelPaletteToTeam(FIXTURE_PALETTE, team);
        for (let index = 0; index < FIXTURE_PALETTE.length; index += 1) {
          if (index === NEATRACING_TEAM_REMAP_SLOT) {
            continue;
          }
          expect(remapped[index]).toEqual(FIXTURE_PALETTE[index]);
        }
      }
    });

    it('exposes exactly four named team colors in cyan, red, green, blue order', () => {
      expect(NEATRACING_TEAM_COLOR_COUNT).toBe(4);
      expect(NEATRACING_TEAM_COLORS.map((teamColor) => teamColor.name)).toEqual(
        ['cyan', 'red', 'green', 'blue'],
      );
    });

    it('pins the neon rgb bytes resolved for every team id', () => {
      const expectedTeamRgbs: readonly number[][] = [
        [95, 255, 255], // cyan
        [255, 77, 77], // red
        [77, 255, 77], // green
        [77, 153, 255], // blue
      ];
      for (const [team, expected] of expectedTeamRgbs.entries()) {
        expect([...resolveVoxelTeamColor(team as VoxelTeamColorId)]).toEqual(
          expected,
        );
      }
    });

    it('leaves the EMPTY sentinel untouched when it occupies the designated slot', () => {
      const paletteWithEmptySlot: readonly VoxelPaletteEntry[] = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        BODY_INDIGO,
      ];
      const remapped = remapVoxelPaletteToTeam(
        paletteWithEmptySlot,
        NEATRACING_TEAM_ID_GREEN,
      );
      expect(remapped[NEATRACING_TEAM_REMAP_SLOT]).toEqual([0, 0, 0, 0, 0]);
    });

    it('returns palettes shorter than the designated slot unchanged', () => {
      const singleEntryPalette: readonly VoxelPaletteEntry[] = [
        [0, 0, 0, 0, 0],
      ];
      const remapped = remapVoxelPaletteToTeam(
        singleEntryPalette,
        NEATRACING_TEAM_ID_CYAN,
      );
      expect(remapped).toEqual(singleEntryPalette);
    });

    it('returns a new palette and never mutates the input (draw-time remap, no bin rebuild)', () => {
      const before: readonly VoxelPaletteEntry[] =
        structuredClone(FIXTURE_PALETTE);
      const remapped = remapVoxelPaletteToTeam(
        FIXTURE_PALETTE,
        NEATRACING_TEAM_ID_RED,
      );
      expect(remapped).not.toBe(FIXTURE_PALETTE);
      expect(FIXTURE_PALETTE).toEqual(before);
    });

    it('is deterministic for repeated identical remaps', () => {
      const first = remapVoxelPaletteToTeam(
        FIXTURE_PALETTE,
        NEATRACING_TEAM_ID_BLUE,
      );
      const second = remapVoxelPaletteToTeam(
        FIXTURE_PALETTE,
        NEATRACING_TEAM_ID_BLUE,
      );
      expect(second).toEqual(first);
    });

    it('rejects out-of-range team ids with the named remap error', () => {
      expect(() => resolveVoxelTeamColor(4 as VoxelTeamColorId)).toThrow(
        VoxelTeamRemapError,
      );
      expect(() =>
        remapVoxelPaletteToTeam(FIXTURE_PALETTE, 4 as VoxelTeamColorId),
      ).toThrow(VoxelTeamRemapError);
      expect(() => resolveVoxelTeamColor(99 as VoxelTeamColorId)).toThrow(
        /team color id/,
      );
    });

    it('rejects negative and non-integer team ids with the named remap error', () => {
      expect(() => resolveVoxelTeamColor(-1 as VoxelTeamColorId)).toThrow(
        VoxelTeamRemapError,
      );
      expect(() =>
        remapVoxelPaletteToTeam(FIXTURE_PALETTE, 2.5 as VoxelTeamColorId),
      ).toThrow(VoxelTeamRemapError);
    });
  });
});
