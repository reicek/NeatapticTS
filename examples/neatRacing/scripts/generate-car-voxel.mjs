/* global Buffer:readonly, console:readonly, process:readonly */

/**
 * Deterministic initial car + wheel voxel generator for Neat Racing
 * (spec sections 7.2 / 7.4 / 8.4, LD-29; slice P1-S6-D, AC-022).
 *
 * Procedurally authors the initial F1-style voxel car document
 * (`car-f1.model.json`, normative 128-cubed grid) and the wheel document
 * (`wheel.model.json`, TP-14 32-cubed grid) that pass the section 7.2
 * schema (`voxel/model.schema.ts`), decode with the shared decoder, and
 * produce non-empty solid bodies. The geometry is a pure function of named
 * constants — no RNG, no clocks, no timestamps — so two runs emit
 * byte-identical JSON (AC-022 determinism contract).
 *
 * Palette conventions honored here (spec 7.2 / AC-014 / LD-23):
 *
 * - Palette index 0 is the all-zero EMPTY sentinel.
 * - The car paints its team-livery placeholder at index 1, the
 *   designated team-remap slot (`NEATRACING_TEAM_REMAP_SLOT` in
 *   `voxel/team-remap`), so the draw-time remap recolors the body only.
 * - The wheel palette parks a duplicated EMPTY sentinel at index 1 too:
 *   the remap leaves EMPTY entries untouched, so team-less wheels never
 *   bleed a team color into their tire/hub palette.
 * - Glass entries are fully transparent (`a === 0`) colored entries with
 *   `glow > 0` (shining cockpit canopy, glow-only pass downstream).
 * - Tire black `r === g === b === 0` is a legal solid entry (LD-23 black
 *   exception is enforced downstream, never here).
 *
 * CLI seam: `node examples/neatRacing/scripts/generate-car-voxel.mjs
 * [--out <dir>]`. With no arguments the script writes the committed
 * `examples/neatRacing/assets/models` directory (registered as
 * `npm run voxel:generate:car`); `--out <dir>` redirects both documents to
 * any directory (the red determinism contract spawns exactly this form and
 * never touches the committed assets, LD-29). The committed assets are
 * hand-edited in the editor afterwards and become authoritative — this
 * generator is never re-run over tuned models (LD-29).
 *
 * @author 04-implementing (P1-S6-D implementation executor)
 * @see     plans/neatracing-voxel-racer.spec.md — sections 7.2, 7.4, 8.4
 * @see     examples/neatRacing/voxel/generated-assets.test.ts — red contract
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// ── Constants ────────────────────────────────────────────────────────────

/** Directory holding this script (`examples/neatRacing/scripts`). */
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

/** Default output directory: the committed spec 10.2 model assets. */
const DEFAULT_MODELS_DIR = resolve(SCRIPT_DIR, '..', 'assets', 'models');

/** CLI flag selecting the output directory (red-contract seam). */
const OUT_FLAG = '--out';

/** Model document file suffix (spec 10.2 asset naming). */
const MODEL_FILE_SUFFIX = '.model.json';

/** Prefix of every progress line from this script. */
const LOG_PREFIX = 'voxel:generate:car:';

/** Prefix of the failure message when generation fails. */
const FAILURE_MESSAGE_PREFIX = 'voxel-generate-car failed:';

/** Normative car grid side (spec 7.2/LD-19; parity with VOXEL_GRID_DIMS). */
const CAR_GRID_DIMS = 128;

/** Normative wheel grid side (spec 7.4/TP-14). */
const WHEEL_GRID_DIMS = 32;

/** Model names; bins derive `<name>-lod128.bin` from these (spec 10.2). */
const CAR_MODEL_NAME = 'car-f1';
const WHEEL_MODEL_NAME = 'wheel';

// ── Palette entries ([[r, g, b, a, glow]]) ───────────────────────────────

/** The all-zero EMPTY sentinel; palette index 0 must hold it (spec 7.2). */
const VOXEL_EMPTY_PALETTE_ENTRY = [0, 0, 0, 0, 0];

/** Car team-livery placeholder at the designated remap slot 1 (AC-014). */
const CAR_TEAM_LIVERY_PLACEHOLDER = [214, 34, 64, 255, 0];

/** Shared dark charcoal trim (monocoque spine, floor, wing endplates). */
const CHARCOAL_TRIM = [26, 28, 34, 255, 0];

/** Cockpit canopy: shining glass (alpha 0, light-blue color, glow 90). */
const CANOPY_GLASS = [170, 220, 255, 0, 90];

/** Amber glowing decal stripe (sidepods + nose flare), opaque + glow. */
const GLOW_DECAL = [255, 170, 40, 255, 120];

/**
 * Car palette: 0 EMPTY sentinel, 1 team-livery placeholder (designated
 * slot), 2 charcoal trim, 3 canopy glass, 4 glow decal.
 */
const CAR_PALETTE = [
  VOXEL_EMPTY_PALETTE_ENTRY,
  CAR_TEAM_LIVERY_PLACEHOLDER,
  CHARCOAL_TRIM,
  CANOPY_GLASS,
  GLOW_DECAL,
];

/** Palette index of the all-zero EMPTY sentinel (openings / air cells). */
const VOXEL_EMPTY_PALETTE_INDEX = 0;

/** Car palette index of the team-livery placeholder (designated slot 1). */
const CAR_LIVERY_INDEX = 1;

/** Car charcoal trim palette index (floor, spines, endplates, skirts). */
const CHARCOAL_TRIM_INDEX = 2;

/** Car canopy glass palette index (shining, alpha-0). */
const CANOPY_GLASS_INDEX = 3;

/** Car glowing decal stripe palette index. */
const GLOW_DECAL_INDEX = 4;

// Wheel palette indices (duplicated EMPTY at slot 1 opts out of remaps).
const WHEEL_TIRE_INDEX = 2;

/** Wheel hub palette index (dark metal). */
const WHEEL_HUB_INDEX = 3;

/** Wheel glowing rim-ring palette index (amber accent). */
const WHEEL_RIM_GLOW_INDEX = 4;

/**
 * Wheel palette: 0 EMPTY sentinel, 1 duplicated EMPTY sentinel parked in
 * the designated remap slot (opt-out trick), 2 tire, 3 hub, 4 rim glow.
 */
const WHEEL_PALETTE = [
  VOXEL_EMPTY_PALETTE_ENTRY,
  VOXEL_EMPTY_PALETTE_ENTRY,
  [0, 0, 0, 255, 0], // tire black (LD-23: legal solid black)
  CHARCOAL_TRIM, // hub metal
  GLOW_DECAL, // glowing rim ring
];

// ── Car geometry constants (voxel grid space; slice 0 = rear, 127 = nose) ─

// Floor plank: full-length underbody, inboard of the sidepods.
const CAR_FLOOR_Z0 = 8;
const CAR_FLOOR_Z1 = 117;
const CAR_FLOOR_HALF_WIDTH = 15;

/** Sidepod floor extension carrying the pods outboard of the plank. */
const CAR_SIDEPOD_FLOOR_Z0 = 56;
const CAR_SIDEPOD_FLOOR_Z1 = 96;
const CAR_SIDEPOD_FLOOR_HALF_WIDTH = 23;

// Rear wing: main plane + flap + endplates + central pillar.
const CAR_REAR_WING_Z0 = 2;
const CAR_REAR_WING_Z1 = 10;
const CAR_REAR_WING_HALF_WIDTH = 25;
const CAR_REAR_WING_Y0 = 33;
const CAR_REAR_WING_Y1 = 35;
const CAR_REAR_WING_FLAP_Y0 = 37;
const CAR_REAR_WING_FLAP_Y1 = 39;
const CAR_REAR_WING_ENDPLATE_DX_MIN = 18;
const CAR_REAR_WING_ENDPLATE_DX_MAX = 21;
const CAR_REAR_WING_ENDPLATE_Y0 = 28;
const CAR_REAR_WING_ENDPLATE_Y1 = 42;
const CAR_REAR_WING_PILLAR_HALF_WIDTH = 2;
const CAR_REAR_WING_PILLAR_Y0 = 10;
const CAR_REAR_WING_PILLAR_Y1 = 32;

// Tail (gearbox section): widens from the diffuser toward the engine cover.
const CAR_TAIL_Z0 = 22;
const CAR_TAIL_Z1 = 54;
const CAR_TAIL_MIN_HALF_WIDTH = 7;
const CAR_TAIL_MAX_HALF_WIDTH = 9;
const CAR_TAIL_Y0 = 2;
const CAR_TAIL_Y1 = 9;

// Engine cover: constant-width central body between cockpit and tail.
const CAR_ENGINE_COVER_Z0 = 46;
const CAR_ENGINE_COVER_Z1 = 100;
const CAR_ENGINE_COVER_HALF_WIDTH = 9;
const CAR_ENGINE_COVER_Y0 = 2;
const CAR_ENGINE_COVER_Y1 = 13;

// Airbox spine above the engine cover, ahead of the canopy glass.
const CAR_AIRBOX_Z0 = 46;
const CAR_AIRBOX_Z1 = 68;
const CAR_AIRBOX_HALF_WIDTH = 5;
const CAR_AIRBOX_Y0 = 14;
const CAR_AIRBOX_Y1 = 21;

// Sidepods: radiator bodies flanking the cockpit.
const CAR_SIDEPODS_Z0 = 58;
const CAR_SIDEPODS_Z1 = 96;
const CAR_SIDEPODS_HALF_WIDTH = 23;
const CAR_SIDEPODS_BODY_Y0 = 2;
const CAR_SIDEPODS_BODY_Y1 = 6;
const CAR_SIDEPODS_SKIRT_Y0 = 7;
const CAR_SIDEPODS_SKIRT_Y1 = 9;

// Cockpit tub and canopy: opening cut + glass hump above the driver.
const CAR_TUB_Z0 = 60;
const CAR_TUB_Z1 = 102;
const CAR_TUB_HALF_WIDTH = 10;
const CAR_TUB_Y0 = 2;
const CAR_TUB_Y1 = 13;
const CAR_COCKPIT_OPENING_Z0 = 68;
const CAR_COCKPIT_OPENING_Z1 = 84;
const CAR_COCKPIT_OPENING_HALF_WIDTH = 6;
const CAR_COCKPIT_OPENING_Y0 = 11;
const CAR_COCKPIT_OPENING_Y1 = 13;
const CAR_CANOPY_Z0 = 70;
const CAR_CANOPY_Z1 = 82;
const CAR_CANOPY_HALF_WIDTH = 5;
const CAR_CANOPY_Y0 = 13;
const CAR_CANOPY_Y1 = 16;

// Nose cone: tapers from the cockpit toward the front wing.
const CAR_NOSE_Z0 = 94;
const CAR_NOSE_Z1 = 124;
const CAR_NOSE_MAX_HALF_WIDTH = 9;
const CAR_NOSE_MIN_HALF_WIDTH = 3;
const CAR_NOSE_Y0 = 2;
const CAR_NOSE_Y1 = 11;

// Front wing: full-span main plane with raised charcoal endplates.
const CAR_FRONT_WING_Z0 = 118;
const CAR_FRONT_WING_Z1 = 126;
const CAR_FRONT_WING_HALF_WIDTH = 26;
const CAR_FRONT_WING_Y0 = 0;
const CAR_FRONT_WING_Y1 = 1;
const CAR_FRONT_WING_ENDPLATE_DX_MIN = 22;
const CAR_FRONT_WING_ENDPLATE_DX_MAX = 25;
const CAR_FRONT_WING_ENDPLATE_Y0 = 0;
const CAR_FRONT_WING_ENDPLATE_Y1 = 5;

// Glow decals: two amber stripes per sidepod + one nose flare stripe.
const CAR_SIDEPOD_GLOW_Z0 = 64;
const CAR_SIDEPOD_GLOW_Z1 = 71;
const CAR_SIDEPOD_GLOW_DX_MIN = 14;
const CAR_SIDEPOD_GLOW_DX_MAX = 17;
const CAR_SIDEPOD_GLOW_Y0 = 5;
const CAR_SIDEPOD_GLOW_Y1 = 6;
const CAR_SIDEPOD_GLOW_2_Z0 = 80;
const CAR_SIDEPOD_GLOW_2_Z1 = 87;
const CAR_NOSE_GLOW_Z0 = 104;
const CAR_NOSE_GLOW_Z1 = 111;
const CAR_NOSE_GLOW_DX_MIN = 4;
const CAR_NOSE_GLOW_DX_MAX = 6;
const CAR_NOSE_GLOW_Y0 = 5;
const CAR_NOSE_GLOW_Y1 = 6;

// ── Wheel geometry constants (32-cubed disc; disc plane contains y) ──────

// Tread zone along the depth axis (pre-mount spin axis).
const WHEEL_TREAD_Z0 = 6;
const WHEEL_TREAD_Z1 = 25;

// Hub and glow rim live in the narrower central depth zone.
const WHEEL_HUB_Z0 = 10;
const WHEEL_HUB_Z1 = 21;
const WHEEL_HUB_RADIUS = 5;
const WHEEL_RIM_GLOW_RADIUS = 7;

/** Disc radius from the wheel grid center (`|16 - (cell + 0.5)| ≤ 15.5`). */
const WHEEL_DISC_RADIUS = 15.9;

// ── Grid painting helpers ────────────────────────────────────────────────

/**
 * Creates an all-air grid for a symmetric model: one stored half column
 * range (`dims[0]` stored cols, e.g. 64 of 128), full height and depth.
 *
 * @param {number} storedWidth - Stored column count (`full width / 2`).
 * @param {number} height - Grid height (rows, row 0 = ground).
 * @param {number} depth - Grid depth (slices).
 * @returns {Uint8Array} Row-major `[x + y * storedWidth + z * storedWidth
 *   * height]` palette-index grid, all EMPTY (0).
 */
function createHalfGrid(storedWidth, height, depth) {
  return new Uint8Array(storedWidth * height * depth);
}

/**
 * Paints a symmetric band into the stored half grid: every slice in
 * `[z0, z1]`, every row in `[y0, y1]`, and the stored columns centered at
 * the grid center covering half-width `halfWidth` (cols
 * `storedWidth - halfWidth .. storedWidth - 1`).
 *
 * @param {Uint8Array} grid - Half grid to paint onto.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Grid extents (stored width, height, depth).
 * @param {number} z0 - First depth slice (inclusive).
 * @param {number} z1 - Last depth slice (inclusive).
 * @param {number} y0 - First height row (inclusive).
 * @param {number} y1 - Last height row (inclusive).
 * @param {number} halfWidth - Symmetric half width in columns.
 * @param {number} paletteIndex - Palette index painted into the band.
 * @returns {void}
 */
function paintHalfWidthBand(grid, dims, z0, z1, y0, y1, halfWidth, paletteIndex) {
  const [storedWidth] = dims;
  const xFrom = Math.max(0, storedWidth - halfWidth);
  paintColumnRange(grid, dims, xFrom, storedWidth - 1, y0, y1, z0, z1, paletteIndex);
}

/**
 * Paints one inclusive dx band (center-distance window) into the half grid.
 *
 * @param {Uint8Array} grid - Half grid to paint onto.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Grid extents.
 * @param {number} z0 - First depth slice (inclusive).
 * @param {number} z1 - Last depth slice (inclusive).
 * @param {number} y0 - First height row (inclusive).
 * @param {number} y1 - Last height row (inclusive).
 * @param {number} dxMin - Min center distance (inclusive).
 * @param {number} dxMax - Max center distance (inclusive).
 * @param {number} paletteIndex - Palette index painted into the band.
 * @returns {void}
 */
function paintDxRange(
  grid,
  dims,
  z0,
  z1,
  y0,
  y1,
  dxMin,
  dxMax,
  paletteIndex,
) {
  const [storedWidth] = dims;
  const xFrom = Math.max(0, storedWidth - 1 - dxMax);
  const xTo = storedWidth - 1 - dxMin;
  paintColumnRange(grid, dims, xFrom, xTo, y0, y1, z0, z1, paletteIndex);
}

/**
 * Paints one inclusive axis-aligned box into the half grid.
 *
 * @param {Uint8Array} grid - Half grid to paint onto.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Grid extents.
 * @param {number} x0 - First stored column (inclusive).
 * @param {number} x1 - Last stored column (inclusive).
 * @param {number} y0 - First height row (inclusive).
 * @param {number} y1 - Last height row (inclusive).
 * @param {number} z0 - First depth slice (inclusive).
 * @param {number} z1 - Last depth slice (inclusive).
 * @param {number} paletteIndex - Palette index painted into the box.
 * @returns {void}
 */
function paintColumnRange(
  grid,
  dims,
  x0,
  x1,
  y0,
  y1,
  z0,
  z1,
  paletteIndex,
) {
  const [storedWidth, height] = dims;
  const xFrom = Math.max(0, x0);
  const xTo = Math.min(x1, storedWidth - 1);
  for (let z = z0; z <= z1; z++) {
    for (let y = y0; y <= y1; y++) {
      for (let x = xFrom; x <= xTo; x++) {
        grid[y * storedWidth + x + z * storedWidth * height] = paletteIndex;
      }
    }
  }
}

// ── Car geometry ─────────────────────────────────────────────────────────

/**
 * Builds the deterministic initial 128-cubed F1-style car document.
 *
 * The body is painted rear (slice 0) toward the nose (slice 127) as a
 * declarative band pipeline; every band is a pure function of named
 * constants, so rebuilds replay byte-identically.
 *
 * @returns {object} The section 7.2 car model document object.
 */
function buildCarModel() {
  const dims = [CAR_GRID_DIMS / 2, CAR_GRID_DIMS, CAR_GRID_DIMS];
  const grid = createHalfGrid(dims[0], dims[1], dims[2]);

  paintCarFloor(grid, dims);
  paintCarRearWing(grid, dims);
  paintCarTail(grid, dims);
  paintCarEngineCoverAndAirbox(grid, dims);
  paintCarSidepods(grid, dims);
  paintCarCockpit(grid, dims);
  paintCarNose(grid, dims);
  paintCarFrontWing(grid, dims);
  paintCarGlowDecals(grid, dims);

  return {
    meta: { name: CAR_MODEL_NAME, dims: [CAR_GRID_DIMS, CAR_GRID_DIMS, CAR_GRID_DIMS], symmetric: true },
    palette: CAR_PALETTE.map((entry) => [...entry]),
    slices: encodeHalfGridSlices(grid, dims),
  };
}

/**
 * Paints the floor plank plus the sidepod floor extension.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarFloor(grid, dims) {
  paintHalfWidthBand(
    grid, dims,
    CAR_FLOOR_Z0, CAR_FLOOR_Z1, 0, 0,
    CAR_FLOOR_HALF_WIDTH, CHARCOAL_TRIM_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_SIDEPOD_FLOOR_Z0, CAR_SIDEPOD_FLOOR_Z1, 0, 0,
    CAR_SIDEPOD_FLOOR_HALF_WIDTH, CHARCOAL_TRIM_INDEX,
  );
}

/**
 * Paints the rear wing: main plane, flap, endplates, and central pillar.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarRearWing(grid, dims) {
  paintHalfWidthBand(
    grid, dims,
    CAR_REAR_WING_Z0, CAR_REAR_WING_Z1,
    CAR_REAR_WING_Y0, CAR_REAR_WING_Y1,
    CAR_REAR_WING_HALF_WIDTH, CAR_LIVERY_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_REAR_WING_Z0 + 2, CAR_REAR_WING_Z1,
    CAR_REAR_WING_FLAP_Y0, CAR_REAR_WING_FLAP_Y1,
    CAR_REAR_WING_HALF_WIDTH, CAR_LIVERY_INDEX,
  );
  paintDxRange(
    grid, dims,
    CAR_REAR_WING_Z0, CAR_REAR_WING_Z1,
    CAR_REAR_WING_ENDPLATE_Y0, CAR_REAR_WING_ENDPLATE_Y1,
    CAR_REAR_WING_ENDPLATE_DX_MIN, CAR_REAR_WING_ENDPLATE_DX_MAX,
    CHARCOAL_TRIM_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_REAR_WING_Z0 + 2, CAR_REAR_WING_Z1 - 2,
    CAR_REAR_WING_PILLAR_Y0, CAR_REAR_WING_PILLAR_Y1,
    CAR_REAR_WING_PILLAR_HALF_WIDTH, CHARCOAL_TRIM_INDEX,
  );
}

/**
 * Paints the tapering tail section between the diffuser and engine cover.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarTail(grid, dims) {
  for (let z = CAR_TAIL_Z0; z <= CAR_TAIL_Z1; z++) {
    const halfWidth = taperHalfWidth(
      CAR_TAIL_Z0, CAR_TAIL_Z1,
      CAR_TAIL_MIN_HALF_WIDTH, CAR_TAIL_MAX_HALF_WIDTH,
      z,
    );
    paintHalfWidthBand(grid, dims, z, z, CAR_TAIL_Y0, CAR_TAIL_Y1, halfWidth, CAR_LIVERY_INDEX);
  }
}

/**
 * Computes an integer linear taper value for one slice: the half width at
 * `z` interpolated deterministically between the section endpoints.
 *
 * @param {number} z0 - Section start depth (value = min there).
 * @param {number} z1 - Section end depth (value = max there).
 * @param {number} minWidth - Half width at the start of the section.
 * @param {number} maxWidth - Half width at the end of the section.
 * @param {number} z - Query depth.
 * @returns {number} The interpolated half width, clamped to the endpoints.
 */
function taperHalfWidth(z0, z1, minWidth, maxWidth, z) {
  const span = z1 - z0;
  const clamped = Math.min(Math.max(z, z0), z1);
  const grown = minWidth + Math.floor(((maxWidth - minWidth) * (clamped - z0)) / span);
  return Math.min(maxWidth, grown);
}

/**
 * Paints the engine cover body and the charcoal airbox spine above it.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarEngineCoverAndAirbox(grid, dims) {
  paintHalfWidthBand(
    grid, dims,
    CAR_ENGINE_COVER_Z0, CAR_ENGINE_COVER_Z1,
    CAR_ENGINE_COVER_Y0, CAR_ENGINE_COVER_Y1,
    CAR_ENGINE_COVER_HALF_WIDTH, CAR_LIVERY_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_AIRBOX_Z0, CAR_AIRBOX_Z1,
    CAR_AIRBOX_Y0, CAR_AIRBOX_Y1,
    CAR_AIRBOX_HALF_WIDTH, CHARCOAL_TRIM_INDEX,
  );
}

/**
 * Paints the sidepods: livery radiator bodies with charcoal lower skirts.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarSidepods(grid, dims) {
  paintHalfWidthBand(
    grid, dims,
    CAR_SIDEPODS_Z0, CAR_SIDEPODS_Z1,
    CAR_SIDEPODS_BODY_Y0, CAR_SIDEPODS_BODY_Y1,
    CAR_SIDEPODS_HALF_WIDTH, CAR_LIVERY_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_SIDEPODS_Z0, CAR_SIDEPODS_Z1,
    CAR_SIDEPODS_SKIRT_Y0, CAR_SIDEPODS_SKIRT_Y1,
    CAR_SIDEPODS_HALF_WIDTH, CHARCOAL_TRIM_INDEX,
  );
}

/**
 * Paints the cockpit tub, cuts the driver opening, and raises the glass
 * canopy hump.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarCockpit(grid, dims) {
  paintHalfWidthBand(
    grid, dims,
    CAR_TUB_Z0, CAR_TUB_Z1, CAR_TUB_Y0, CAR_TUB_Y1,
    CAR_TUB_HALF_WIDTH, CAR_LIVERY_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_COCKPIT_OPENING_Z0, CAR_COCKPIT_OPENING_Z1,
    CAR_COCKPIT_OPENING_Y0, CAR_COCKPIT_OPENING_Y1,
    CAR_COCKPIT_OPENING_HALF_WIDTH, VOXEL_EMPTY_PALETTE_INDEX,
  );
  paintHalfWidthBand(
    grid, dims,
    CAR_CANOPY_Z0, CAR_CANOPY_Z1, CAR_CANOPY_Y0, CAR_CANOPY_Y1,
    CAR_CANOPY_HALF_WIDTH, CANOPY_GLASS_INDEX,
  );
}

/**
 * Paints the tapering nose cone toward the front wing.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarNose(grid, dims) {
  for (let z = CAR_NOSE_Z0; z <= CAR_NOSE_Z1; z++) {
    const halfWidth = reverseTaperHalfWidth(
      CAR_NOSE_Z0, CAR_NOSE_Z1,
      CAR_NOSE_MAX_HALF_WIDTH, CAR_NOSE_MIN_HALF_WIDTH,
      z,
    );
    paintHalfWidthBand(grid, dims, z, z, CAR_NOSE_Y0, CAR_NOSE_Y1, halfWidth, CAR_LIVERY_INDEX);
  }
}

/**
 * Computes the forward taper value used by the nose: the width shrinks
 * linearly (integer floor) from `startValue` at `z0` to `endValue` at `z1`.
 *
 * @param {number} z0 - Section start depth (value = startValue there).
 * @param {number} z1 - Section end depth (value = endValue there).
 * @param {number} startValue - Half width at the start of the section.
 * @param {number} endValue - Half width at the end of the section.
 * @param {number} z - Query depth.
 * @returns {number} The tapered half width, clamped to the endpoints.
 */
function reverseTaperHalfWidth(z0, z1, startValue, endValue, z) {
  const span = z1 - z0;
  const clamped = Math.min(Math.max(z, z0), z1);
  const shrunk = startValue - Math.floor(((startValue - endValue) * (clamped - z0)) / span);
  return Math.max(endValue, shrunk);
}

/**
 * Paints the front wing main plane and raised charcoal endplates.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarFrontWing(grid, dims) {
  paintHalfWidthBand(
    grid, dims,
    CAR_FRONT_WING_Z0, CAR_FRONT_WING_Z1,
    CAR_FRONT_WING_Y0, CAR_FRONT_WING_Y1,
    CAR_FRONT_WING_HALF_WIDTH, CAR_LIVERY_INDEX,
  );
  paintDxRange(
    grid, dims,
    CAR_FRONT_WING_Z0, CAR_FRONT_WING_Z1,
    CAR_FRONT_WING_ENDPLATE_Y0, CAR_FRONT_WING_ENDPLATE_Y1,
    CAR_FRONT_WING_ENDPLATE_DX_MIN, CAR_FRONT_WING_ENDPLATE_DX_MAX,
    CHARCOAL_TRIM_INDEX,
  );
}

/**
 * Paints the amber glowing decal stripes: sidepod pairs + nose flare.
 *
 * @param {Uint8Array} grid - Car half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Car grid extents.
 * @returns {void}
 */
function paintCarGlowDecals(grid, dims) {
  paintDxRange(
    grid, dims,
    CAR_SIDEPOD_GLOW_Z0, CAR_SIDEPOD_GLOW_Z1,
    CAR_SIDEPOD_GLOW_Y0, CAR_SIDEPOD_GLOW_Y1,
    CAR_SIDEPOD_GLOW_DX_MIN, CAR_SIDEPOD_GLOW_DX_MAX,
    GLOW_DECAL_INDEX,
  );
  paintDxRange(
    grid, dims,
    CAR_SIDEPOD_GLOW_2_Z0, CAR_SIDEPOD_GLOW_2_Z1,
    CAR_SIDEPOD_GLOW_Y0, CAR_SIDEPOD_GLOW_Y1,
    CAR_SIDEPOD_GLOW_DX_MIN, CAR_SIDEPOD_GLOW_DX_MAX,
    GLOW_DECAL_INDEX,
  );
  paintDxRange(
    grid, dims,
    CAR_NOSE_GLOW_Z0, CAR_NOSE_GLOW_Z1,
    CAR_NOSE_GLOW_Y0, CAR_NOSE_GLOW_Y1,
    CAR_NOSE_GLOW_DX_MIN, CAR_NOSE_GLOW_DX_MAX,
    GLOW_DECAL_INDEX,
  );
}

// ── Wheel geometry ───────────────────────────────────────────────────────

/**
 * Builds the deterministic TP-14 32-cubed wheel document: a disc whose
 * plane contains the y axis (tire ring, glowing rim, metal hub) whose
 * pre-mount spin axis is the depth axis.
 *
 * @returns {object} The section 7.2 wheel model document object.
 */
function buildWheelModel() {
  const dims = [WHEEL_GRID_DIMS / 2, WHEEL_GRID_DIMS, WHEEL_GRID_DIMS];
  const grid = createHalfGrid(dims[0], dims[1], dims[2]);

  paintWheelDisc(grid, dims);

  return {
    meta: { name: WHEEL_MODEL_NAME, dims: [WHEEL_GRID_DIMS, WHEEL_GRID_DIMS, WHEEL_GRID_DIMS], symmetric: true },
    palette: WHEEL_PALETTE.map((entry) => [...entry]),
    slices: encodeHalfGridSlices(grid, dims),
  };
}

/**
 * Paints the wheel disc bands by radial distance from the grid center:
 * tire ring across the tread zone, sidewalls across the full tread depth,
 * glowing rim ring + metal hub in the narrower central zone.
 *
 * @param {Uint8Array} grid - Wheel half grid.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Wheel grid extents.
 * @returns {void}
 */
function paintWheelDisc(grid, dims) {
  const [storedWidth, height, depth] = dims;
  const centerCol = storedWidth; // continuous center of the full grid
  for (let z = 0; z < depth; z++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < storedWidth; x++) {
        const dx = centerCol - (x + 0.5);
        const dy = Math.abs(y + 0.5 - height / 2);
        const radius = Math.hypot(dx, dy);
        const index = classifyWheelVoxel(z, radius);
        grid[y * storedWidth + x + z * storedWidth * height] = index;
      }
    }
  }
}

/**
 * Classifies one wheel voxel's palette index from its depth and radius.
 *
 * The two exposed tread planes (z === WHEEL_TREAD_Z0 / WHEEL_TREAD_Z1) are
 * the only disc faces the greedy mesher ever exposes, so they carry the
 * readable radial pattern — charcoal hub disc, glowing rim ring, tire
 * annulus. Interior slices keep the structural bands (tire sidewall
 * sheathing the hub cylinder) so no interior face ever leaks.
 *
 * @param {number} z - Depth slice (pre-mount spin axis position).
 * @param {number} radius - Distance from the wheel disc center.
 * @returns {number} The palette index for this cell.
 */
function classifyWheelVoxel(z, radius) {
  if (z < WHEEL_TREAD_Z0 || z > WHEEL_TREAD_Z1 || radius > WHEEL_DISC_RADIUS) {
    return VOXEL_EMPTY_PALETTE_INDEX; // air outside the tread zone / disc
  }
  if (z === WHEEL_TREAD_Z0 || z === WHEEL_TREAD_Z1) {
    // Exposed disc face: show the radial hub + glow-ring pattern.
    if (radius <= WHEEL_HUB_RADIUS) {
      return WHEEL_HUB_INDEX;
    }
    if (radius <= WHEEL_RIM_GLOW_RADIUS) {
      return WHEEL_RIM_GLOW_INDEX;
    }
    return WHEEL_TIRE_INDEX;
  }
  if (z < WHEEL_HUB_Z0 || z > WHEEL_HUB_Z1) {
    return WHEEL_TIRE_INDEX; // sidewall
  }
  if (radius <= WHEEL_HUB_RADIUS) {
    return WHEEL_HUB_INDEX;
  }
  if (radius <= WHEEL_RIM_GLOW_RADIUS) {
    return WHEEL_RIM_GLOW_INDEX;
  }
  return WHEEL_TIRE_INDEX;
}

// ── RLE encoding ─────────────────────────────────────────────────────────

/** Hex digits per RLE field (`<runLen>` and `<paletteIdx>`). */
const RLE_FIELD_HEX_CHARS = 2;

/**
 * Encodes one stored half row into its `<runLen:2hex><paletteIdx:2hex>`
 * RLE hex string (spec 7.2). Runs are greedily merged; the trailing
 * remainder (if any) is padded with one EMPTY run so every stored row sums
 * to the stored width exactly.
 *
 * @param {Uint8Array} rowCells - The stored half row palette indices.
 * @returns {string} The encoded RLE hex row string.
 */
function encodeHalfRow(rowCells) {
  let row = '';
  for (let cursor = 0; cursor < rowCells.length; ) {
    let runEnd = cursor + 1;
    while (runEnd < rowCells.length && rowCells[runEnd] === rowCells[cursor]) {
      runEnd++;
    }
    row += encodeRun(runEnd - cursor, rowCells[cursor]);
    cursor = runEnd;
  }
  return row;
}

/**
 * Formats one `<runLen><paletteIdx>` RLE pair, zero-padded to two hex
 * digits per field.
 *
 * @param {number} runLength - Run length (1-255).
 * @param {number} paletteIndex - Palette index (0-255).
 * @returns {string} The 4-hex RLE pair.
 */
function encodeRun(runLength, paletteIndex) {
  return (
    runLength.toString(16).padStart(RLE_FIELD_HEX_CHARS, '0') +
    paletteIndex.toString(16).padStart(RLE_FIELD_HEX_CHARS, '0')
  );
}

/**
 * Encodes every slice of the half grid into the document's rows matrix:
 * `slices[z][y]` is the RLE hex string of stored half row `y`.
 *
 * @param {Uint8Array} grid - Half grid palette indices.
 * @param {[storedWidth: number, height: number, depth: number]} dims -
 *   Grid extents.
 * @returns {string[][]} One row string array per depth slice.
 */
function encodeHalfGridSlices(grid, dims) {
  const [storedWidth, height, depth] = dims;
  const slices = [];
  for (let z = 0; z < depth; z++) {
    const rows = [];
    for (let y = 0; y < height; y++) {
      const rowCells = grid.subarray(
        y * storedWidth + z * storedWidth * height,
        (y + 1) * storedWidth + z * storedWidth * height,
      );
      rows.push(encodeHalfRow(rowCells));
    }
    slices.push(rows);
  }
  return slices;
}

// ── Deterministic self-check ─────────────────────────────────────────────

/** Hex digit count of one full RLE pair (`<runLen><paletteIdx>`). */
const RLE_PAIR_HEX_CHARS = RLE_FIELD_HEX_CHARS * 2;

/**
 * Asserts a generated document is self-consistent before it is written:
 * every row holds well-formed RLE pairs summing exactly to the stored
 * width, every palette index stays inside the palette, and at least one
 * run paints a non-EMPTY voxel (AC-022 non-empty body).
 *
 * @param {object} document - The generated section 7.2 candidate document.
 * @param {string} label - Model label for diagnostics.
 * @returns {void} Throws when any invariant fails.
 */
function assertDocumentSelfConsistent(document, label) {
  const [fullWidth] = document.meta.dims;
  const storedWidth = document.meta.symmetric
    ? fullWidth / 2
    : fullWidth;
  let sawSolidRun = false;

  for (const rows of document.slices) {
    for (const row of rows) {
      let cursor = 0;
      for (let offset = 0; offset < row.length; offset += RLE_PAIR_HEX_CHARS) {
        const runLength = Number.parseInt(row.slice(offset, offset + 2), 16);
        const paletteIndex = Number.parseInt(row.slice(offset + 2, offset + RLE_PAIR_HEX_CHARS), 16);
        if (runLength < 1) {
          throw new Error(`${label}: zero-length RLE run at offset ${offset}`);
        }
        if (paletteIndex >= document.palette.length) {
          throw new Error(
            `${label}: palette index ${paletteIndex} out of range at offset ${offset}`,
          );
        }
        if (paletteIndex !== 0) {
          sawSolidRun = true;
        }
        cursor += runLength;
      }
      if (cursor !== storedWidth) {
        throw new Error(
          `${label}: stored row must sum to ${storedWidth}, received ${cursor}`,
        );
      }
    }
  }
  if (!sawSolidRun) {
    throw new Error(`${label}: document holds no non-EMPTY voxel (AC-022)`);
  }
}

// ── CLI entry point ──────────────────────────────────────────────────────

/**
 * Parses the `--out <dir>` CLI flag, defaulting to the committed models
 * directory. Unknown arguments fail loudly so mistakes never silently
 * write the wrong directory.
 *
 * @param {string[]} argv - Process argv tail (after the script path).
 * @returns {string} The resolved output directory path.
 */
function parseOutputDirectory(argv) {
  const outFlagIndex = argv.indexOf(OUT_FLAG);
  if (outFlagIndex === -1) {
    if (argv.length > 0) {
      throw new Error(
        `unknown argument "${argv[0]}"; usage: node generate-car-voxel.mjs [${OUT_FLAG} <dir>]`,
      );
    }
    return DEFAULT_MODELS_DIR;
  }
  const directory = argv[outFlagIndex + 1];
  if (directory === undefined || directory.startsWith(OUT_FLAG)) {
    throw new Error(`usage: ${OUT_FLAG} <dir> requires a directory value`);
  }
  if (outFlagIndex !== 0 || argv.length !== 2) {
    throw new Error(
      `usage: node generate-car-voxel.mjs [${OUT_FLAG} <dir>] (unknown extra arguments)`,
    );
  }
  return resolve(directory);
}

/**
 * Generates the committed initial car + wheel model documents (AC-022).
 *
 * @param {string} outputDirectory - Directory to write both model files.
 * @returns {void} Writes the documents and logs one progress line each.
 */
function generateCommittedModels(outputDirectory) {
  const models = [
    { label: 'car-f1', document: buildCarModel() },
    { label: 'wheel', document: buildWheelModel() },
  ];

  mkdirSync(outputDirectory, { recursive: true });
  for (const model of models) {
    assertDocumentSelfConsistent(model.document, model.label);
    writeModelDocument(outputDirectory, model.label, model.document);
  }
}

/**
 * Writes one model document to `<outputDirectory>/<name>.model.json`.
 *
 * @param {string} outputDirectory - Target directory.
 * @param {string} name - Model name (file stem).
 * @param {object} document - The section 7.2 model document.
 * @returns {void}
 */
function writeModelDocument(outputDirectory, name, document) {
  const filePath = join(outputDirectory, `${name}${MODEL_FILE_SUFFIX}`);
  const contents = `${JSON.stringify(document, null, 2)}\n`;
  const bytes = Buffer.from(contents, 'utf8');
  writeFileSync(filePath, bytes);
  console.log(`${LOG_PREFIX} ${displayPath(filePath)} (${bytes.length} bytes)`);
}

/**
 * Formats an absolute path relative to the current working directory for
 * readable logs (build-voxel-bins precedent).
 *
 * @param {string} absolutePath - Absolute filesystem path.
 * @returns {string} The path relative to the process working directory.
 */
function displayPath(absolutePath) {
  return relative(process.cwd(), absolutePath);
}

try {
  const outputDirectory = parseOutputDirectory(process.argv.slice(2));
  generateCommittedModels(outputDirectory);
} catch (error) {
  console.error(
    `${FAILURE_MESSAGE_PREFIX} ${error instanceof Error ? error.message : String(error)}`,
  );
  process.exitCode = 1;
}