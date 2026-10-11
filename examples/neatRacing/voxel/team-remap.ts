/**
 * Team-color remap for the Neat Racing voxel racer (slice P1-S4-B, AC-014;
 * spec sections 6.5 / LD-24 in `plans/neatracing-voxel-racer.spec.md`).
 *
 * Neatenstein agent parity: every car model carries ONE designated
 * team-livery palette slot, and at draw time that slot's RGB channels are
 * swapped to the current team's color while its alpha and glow bytes stay
 * verbatim (spec section 6.5). The swap is a pure palette-level operation —
 * the remapped palette is a new array and the input is never mutated — so
 * cars of every team draw and shine from the same model files and bins
 * without rebuilding them (AC-014, including the shining-glass pass, which
 * consumes the remapped palette like any other consumer).
 *
 * The module owns the four named team colors in a fixed deterministic table
 * with the LD-24 "few colors overall" neon look, in documented table order
 * id 0 = cyan, id 1 = red, id 2 = green, id 3 = blue. Applying one team's
 * remap touches only the designated slot: body colors, glass, black trim,
 * glowing decals, and the EMPTY sentinel are untouched by every team's remap
 * (no cross-team bleed), and a remap of the same palette to the same team
 * id is deterministic and repeatable.
 *
 * Inputs are trusted palette entries as validated by `model.schema.ts`, the
 * same way `materials.ts` trusts decoded models; this module adds no input
 * or output and performs one pure pass per palette.
 *
 * @module
 */

import {
  VOXEL_EMPTY_PALETTE_ENTRY,
  VoxelModelError,
  type VoxelPaletteEntry,
} from './model.schema';

/**
 * Designated team-livery palette slot (spec section 6.5: "a designated
 * palette slot remaps to team colors at draw time").
 *
 * Model palettes reserve index 0 for the EMPTY sentinel and paint index 1
 * with the authoring-time livery placeholder that every team remap swaps;
 * all other indices carry shared non-team colors. Car models must therefore
 * author their livery placeholder at this index, while team-less voxel
 * models (for example wheel bodies) simply reuse the palette entries the
 * table leaves untouched.
 */
export const NEATRACING_TEAM_REMAP_SLOT = 1;

/** RGB byte triple of one team color (LD-24 neon look). */
export type VoxelTeamColorRgb = readonly [number, number, number];

/**
 * One named team color in the deterministic remap table.
 *
 * The `name` is exposed for HUD and telemetry surfaces that label the four
 * team liveries; the `rgb` bytes are what the draw-time palette remap swaps
 * into the designated slot.
 */
export interface VoxelTeamColor {
  /** Stable team color name in table order: cyan, red, green, blue. */
  readonly name: 'cyan' | 'red' | 'green' | 'blue';
  /** Neon RGB bytes of the team livery. */
  readonly rgb: VoxelTeamColorRgb;
}

/** Team color id: a NEATRACING_TEAM_COLORS table index between 0 and 3. */
export type VoxelTeamColorId = 0 | 1 | 2 | 3;

/** Team id 0 livery: neon cyan (matches the star-sky cyan `rgba(95, 255, 255, 1)`). */
export const NEATRACING_TEAM_COLOR_CYAN: VoxelTeamColorRgb = [95, 255, 255];

/** Team id 1 livery: neon red. */
export const NEATRACING_TEAM_COLOR_RED: VoxelTeamColorRgb = [255, 77, 77];

/** Team id 2 livery: neon green. */
export const NEATRACING_TEAM_COLOR_GREEN: VoxelTeamColorRgb = [77, 255, 77];

/** Team id 3 livery: neon blue. */
export const NEATRACING_TEAM_COLOR_BLUE: VoxelTeamColorRgb = [77, 153, 255];

/**
 * Deterministic team-color table in documented table order: index 0 = cyan,
 * 1 = red, 2 = green, 3 = blue. The order is normative for team color ids,
 * so consumers resolve liveries by NEATRACING_TEAM_ID_* constants instead of
 * hard-coding positions.
 */
export const NEATRACING_TEAM_COLORS: readonly VoxelTeamColor[] = [
  { name: 'cyan', rgb: NEATRACING_TEAM_COLOR_CYAN },
  { name: 'red', rgb: NEATRACING_TEAM_COLOR_RED },
  { name: 'green', rgb: NEATRACING_TEAM_COLOR_GREEN },
  { name: 'blue', rgb: NEATRACING_TEAM_COLOR_BLUE },
];

/** Number of team colors the remap table carries (the four Neat Racing teams). */
export const NEATRACING_TEAM_COLOR_COUNT = NEATRACING_TEAM_COLORS.length;

/** Team color id for the cyan livery (table index 0). */
export const NEATRACING_TEAM_ID_CYAN: VoxelTeamColorId = 0;

/** Team color id for the red livery (table index 1). */
export const NEATRACING_TEAM_ID_RED: VoxelTeamColorId = 1;

/** Team color id for the green livery (table index 2). */
export const NEATRACING_TEAM_ID_GREEN: VoxelTeamColorId = 2;

/** Team color id for the blue livery (table index 3). */
export const NEATRACING_TEAM_ID_BLUE: VoxelTeamColorId = 3;

/**
 * Named rejection for team color ids outside the four-entry table (or
 * non-integer ids), so invalid draw-time ids cannot silently resolve to an
 * undefined team color.
 */
export class VoxelTeamRemapError extends VoxelModelError {
  /**
   * @param code - Stable machine-readable rejection code
   *   (`voxel-team-remap/team-id` for team id rejections).
   * @param message - Human-readable rejection reason.
   */
  constructor(code: string, message: string) {
    super(code, message);
  }
}

/**
 * Resolve one team color id to its neon RGB byte triple.
 *
 * Deterministic table lookup: the same team id always resolves to the same
 * named entry from NEATRACING_TEAM_COLORS on every call, and out-of-range or
 * non-integer ids reject with a named `VoxelTeamRemapError` instead of
 * silently yielding an undefined color.
 *
 * @param team - Team color id within 0 to NEATRACING_TEAM_COLOR_COUNT - 1.
 * @returns The team color's RGB byte triple from the team color table.
 * @throws {VoxelTeamRemapError} When the team id is not an integer within
 *   the table range.
 *
 * @example
 * ```ts
 * const teamRgb = resolveVoxelTeamColor(NEATRACING_TEAM_ID_RED);
 * // -> [255, 77, 77]
 * ```
 */
export function resolveVoxelTeamColor(
  team: VoxelTeamColorId,
): VoxelTeamColorRgb {
  assertValidTeamColorId(team);
  return NEATRACING_TEAM_COLORS[team].rgb;
}

/**
 * Remap a model palette's designated team-livery slot to one team's color.
 *
 * Returns a fresh palette where the NEATRACING_TEAM_REMAP_SLOT entry keeps
 * its resolved alpha and glow bytes but paints the team color's RGB bytes,
 * exactly like the Neatenstein agent palette replacement. Every other entry
 * — body colors, glass, black trim, decals, the EMPTY sentinel — is copied
 * through unchanged, so the same model draw works for every team without
 * rebuilding bins (AC-014). The input palette is never mutated, and repeated
 * remaps of the same palette to the same team id produce identical output.
 *
 * Palettes shorter than the designated slot (for example a single-entry
 * wheel palette) carry no team livery and pass through unchanged, and an
 * EMPTY sentinel occupying the designated slot is left untouched rather
 * than being remapped into a renderable color.
 *
 * @param palette - Trusted model palette as validated by `model.schema.ts`.
 * @param team - Team color id within 0 to NEATRACING_TEAM_COLOR_COUNT - 1.
 * @returns The per-team remapped palette (a new array).
 * @throws {VoxelTeamRemapError} When the team id is not an integer within
 *   the table range.
 *
 * @example
 * ```ts
 * const teamPalette = remapVoxelPaletteToTeam(decodedPalette, NEATRACING_TEAM_ID_RED);
 * // teamPalette[NEATRACING_TEAM_REMAP_SLOT] now paints neon red;
 * // every other entry is byte-identical to decodedPalette.
 * ```
 */
export function remapVoxelPaletteToTeam(
  palette: readonly VoxelPaletteEntry[],
  team: VoxelTeamColorId,
): VoxelPaletteEntry[] {
  const teamRgb = resolveVoxelTeamColor(team);
  return palette.map((entry, index) =>
    isTeamRemapSlotIndex(index) && !isEmptyPaletteEntry(entry)
      ? remapPaletteEntryToTeam(entry, teamRgb)
      : clonePaletteEntry(entry),
  );
}

/** Lookup of the designated slot position, shared by the remap pass. */
function isTeamRemapSlotIndex(index: number): boolean {
  return index === NEATRACING_TEAM_REMAP_SLOT;
}

/**
 * Detect the EMPTY sentinel so it can never be remapped: a sentinel slot
 * carries no geometry, and forging `[teamR, teamG, teamB, 0, 0]` there would
 * fabricate a renderable shining-glass voxel out of air.
 */
function isEmptyPaletteEntry(entry: VoxelPaletteEntry): boolean {
  return VOXEL_EMPTY_PALETTE_ENTRY.every(
    (component, index) => component === entry[index],
  );
}

/**
 * Swap one palette entry's RGB channels to a team color while keeping its
 * alpha and glow bytes verbatim, so shine and transparency behavior is
 * preserved for every team livery.
 */
function remapPaletteEntryToTeam(
  entry: VoxelPaletteEntry,
  teamRgb: VoxelTeamColorRgb,
): VoxelPaletteEntry {
  const [, , , alpha, glow] = entry;
  return [teamRgb[0], teamRgb[1], teamRgb[2], alpha, glow];
}

/**
 * Copy a palette entry byte-for-byte so the remapped palette owns its
 * entries and draw-time consumers cannot mutate the interpreted input
 * through the returned array.
 */
function clonePaletteEntry(entry: VoxelPaletteEntry): VoxelPaletteEntry {
  const [r, g, b, alpha, glow] = entry;
  return [r, g, b, alpha, glow];
}

/**
 * Reject team color ids outside the four-entry table (or non-integer ids)
 * with the named `VoxelTeamRemapError`, so a corrupt id fails loudly at
 * draw time instead of painting an undefined color.
 */
function assertValidTeamColorId(team: VoxelTeamColorId): void {
  if (
    !Number.isInteger(team) ||
    team < 0 ||
    team >= NEATRACING_TEAM_COLOR_COUNT
  ) {
    throw new VoxelTeamRemapError(
      `voxel-team-remap/team-id`,
      `team color id must be an integer in 0-${NEATRACING_TEAM_COLOR_COUNT - 1}, received ${String(team)}`,
    );
  }
}
