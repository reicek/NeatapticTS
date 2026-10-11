/**
 * @module neatRacing/browser-entry/constants.starfield
 *
 * Starfield background constants for the neatRacing browser host entry
 * (slice P1-S2-A; spec §6.4.1 and AC-004). The module is a direct port of
 * the flappy bird starfield tuning table so the neon sky renders with the
 * same three deterministic cyan parallax layers the Radiant-style background
 * already ships with; only the tile height differs, because neatRacing
 * reserves a fixed 240-pixel above-horizon sky band (TP-19) instead of the
 * full world height.
 *
 * Everything here is visual-only: the starfield sits outside the frame
 * contract, its seeds derive from the visual-only seed base
 * {@link NEATRACING_VISUAL_SEED} (TP-20), and the layer is exempt from
 * replay byte parity (spec §9.7.2).
 */

/**
 * Layer recipe describing one deterministic parallax starfield band.
 *
 * The fields mirror the flappy layer spec so the placement algorithm can be
 * ported one-to-one; `id` names the layer for diagnostics and draw ordering.
 */
export interface StarfieldLayerSpec {
  /** Layer identifier: farthest band first, nearest band last. */
  readonly id: 'far' | 'mid' | 'near';
  /** Unsigned xorshift32 seed producing the layer's deterministic layout. */
  readonly seed: number;
  /** Fraction of a full camera revolution that scrolls the layer one tile. */
  readonly scrollRatio: number;
  /** Number of stars generated inside one tile. */
  readonly starCount: number;
  /** Smallest rendered star edge length in pixels. */
  readonly minSizePx: number;
  /** Largest rendered star edge length in pixels (inclusive). */
  readonly maxSizePx: number;
  /** Lowest star fill alpha. */
  readonly minAlpha: number;
  /** Highest star fill alpha. */
  readonly maxAlpha: number;
  /** Glow blur radius in pixels kept for parity with the 2D-canvas port. */
  readonly blurPx: number;
}

/**
 * Width of one repeated starfield tile in pixels (flappy parity).
 *
 * The tile wraps horizontally across the backing store, so this width fixes
 * both the generated point layout and the horizontal wrap period.
 */
export const NEATRACING_STARFIELD_TILE_WIDTH_PX = 512;

/**
 * Height of the fixed above-horizon sky band in pixels (TP-19).
 *
 * The voxel racer renders its sky as a 240-pixel-tall band at the top of the
 * 720-pixel backing store; the horizon and everything below it are drawn by
 * the later track/car passes.
 */
export const NEATRACING_SKY_BAND_HEIGHT_PX = 240;

/** Height of one repeated starfield tile: exactly the sky band. */
export const NEATRACING_STARFIELD_TILE_HEIGHT_PX =
  NEATRACING_SKY_BAND_HEIGHT_PX;

/**
 * Visual-only seed base for every deterministic visual effect (TP-20).
 *
 * Starfield layers derive their xorshift32 seeds from this base (the far
 * layer uses it directly). Because the value only shifts which pseudo-random
 * stars appear, it is exempt from replay byte parity per spec §9.7.2.
 */
export const NEATRACING_VISUAL_SEED = 1_337;

/**
 * Fill color used for every star in the flappy-parity cyan tint.
 *
 * The string keeps source-text parity with the flappy
 * `FLAPPY_STARFIELD_CYAN_FILL_STYLE` constant; the WebGL pass converts the
 * same RGB triple into a shader uniform.
 */
export const NEATRACING_STARFIELD_CYAN_FILL_STYLE = 'rgba(95, 255, 255, 1)';

/**
 * Horizontal scroll ratio for the farthest parallax layer (flappy parity).
 *
 * One full camera revolution scrolls the far layer by 3% of a tile width.
 */
export const NEATRACING_STARFIELD_FAR_SCROLL_RATIO = 0.03;

/**
 * Horizontal scroll ratio for the middle parallax layer (flappy parity).
 */
export const NEATRACING_STARFIELD_MID_SCROLL_RATIO = 0.04;

/**
 * Horizontal scroll ratio for the nearest parallax layer (flappy parity).
 */
export const NEATRACING_STARFIELD_NEAR_SCROLL_RATIO = 0.05;

/**
 * Inclusive range offset used when converting size bounds into random spans.
 *
 * Adding 1 before `Math.floor` makes the upper size bound reachable, matching
 * the flappy placement algorithm exactly.
 */
export const NEATRACING_STARFIELD_INCLUSIVE_RANGE_OFFSET = 1;

/** Divisor that normalizes unsigned 32-bit RNG state into the unit interval. */
export const NEATRACING_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR = 0x1_0000_0000;

/** Primary left-shift used by the xorshift32 random transition. */
export const NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY = 13;

/** Middle right-shift used by the xorshift32 random transition. */
export const NEATRACING_STARFIELD_XORSHIFT_RIGHT_SHIFT = 17;

/** Final left-shift used by the xorshift32 random transition. */
export const NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL = 5;

/**
 * Declarative layer specs for the three flappy-parity parallax bands.
 *
 * Ordered far → mid → near; the pass draws them back-to-front. The far layer
 * seed is the visual seed base itself (TP-20), and no cloud layers exist in
 * the voxel racer sky (AC-004 pins exactly these three star layers).
 */
export const NEATRACING_STARFIELD_LAYER_SPECS = [
  {
    id: 'far',
    seed: NEATRACING_VISUAL_SEED,
    scrollRatio: NEATRACING_STARFIELD_FAR_SCROLL_RATIO,
    starCount: 35,
    minSizePx: 1,
    maxSizePx: 2,
    minAlpha: 0.08,
    maxAlpha: 0.22,
    blurPx: 4,
  },
  {
    id: 'mid',
    seed: 2_777,
    scrollRatio: NEATRACING_STARFIELD_MID_SCROLL_RATIO,
    starCount: 28,
    minSizePx: 1,
    maxSizePx: 3,
    minAlpha: 0.1,
    maxAlpha: 0.28,
    blurPx: 6,
  },
  {
    id: 'near',
    seed: 4_242,
    scrollRatio: NEATRACING_STARFIELD_NEAR_SCROLL_RATIO,
    starCount: 23,
    minSizePx: 2,
    maxSizePx: 4,
    minAlpha: 0.12,
    maxAlpha: 0.34,
    blurPx: 8,
  },
] as const satisfies readonly StarfieldLayerSpec[];
