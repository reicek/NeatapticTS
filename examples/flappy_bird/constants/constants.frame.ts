/**
 * Glyph frame and text-layout constants for browser HUD rendering.
 *
 * These values keep the box-drawing title/header and text frame deterministic
 * across browsers with slightly different font metrics.
 */

/** Reusable monospaced HUD font for glyph-based frame rendering. */
export const FLAPPY_FRAME_MONOSPACE_FONT =
  '16px Consolas, Menlo, Monaco, monospace';

/** Shared monospace font stack used by all HUD and visualization text. */
export const FLAPPY_MONOSPACE_FONT_FAMILY =
  'Consolas, Menlo, Monaco, monospace';

/** Title rendered inside the standalone header box. */
export const FLAPPY_HEADER_TITLE_TEXT = ' Astro Bird (NeatapticTS) ';

/** Header canvas fixed height (pixels). */
export const FLAPPY_HEADER_CANVAS_HEIGHT_PX = 68;

/** Glyph-row height used for box-drawing rows (pixels). */
export const FLAPPY_FRAME_GLYPH_ROW_HEIGHT_PX = 16;

/** Minimum measured glyph width fallback (pixels). */
export const FLAPPY_FRAME_MIN_GLYPH_WIDTH_PX = 6;

/** Minimum glyph columns to keep frame readable on narrow widths. */
export const FLAPPY_FRAME_MIN_COLUMNS = 34;

/** Reserved columns to keep right edge in bounds during metrics fit. */
export const FLAPPY_FRAME_RESERVED_COLUMNS = 1;

/** Minimum box width when building standalone title frame. */
export const FLAPPY_TITLE_BOX_MIN_WIDTH = 16;

/** Total side margin columns reserved around centered title box. */
export const FLAPPY_TITLE_BOX_MARGIN_COLUMNS = 4;

/** Preferred title width relative to available frame width. */
export const FLAPPY_TITLE_BOX_WIDTH_RATIO = 0.68;

/** Minimum half-span when clamping centered title extents. */
export const FLAPPY_TITLE_BOX_MIN_HALF_SPAN = 2;

/** Minimum columns between title box and outer rails. */
export const FLAPPY_TITLE_BOX_MIN_OUTER_GAP_COLUMNS = 2;

/** Minimum safe columns for box-drawing helper output. */
export const FLAPPY_BOX_MIN_COLUMNS = 3;

/** Minimum safe rows for box-drawing helper output. */
export const FLAPPY_BOX_MIN_ROWS = 2;

/** Glyph character for top-left box corner. */
export const FLAPPY_GLYPH_TOP_LEFT = '╔';

/** Glyph character for top-right box corner. */
export const FLAPPY_GLYPH_TOP_RIGHT = '╗';

/** Glyph character for bottom-left box corner. */
export const FLAPPY_GLYPH_BOTTOM_LEFT = '╚';

/** Glyph character for bottom-right box corner. */
export const FLAPPY_GLYPH_BOTTOM_RIGHT = '╝';

/** Glyph character for horizontal box segment. */
export const FLAPPY_GLYPH_HORIZONTAL = '═';

/** Glyph character for vertical box segment. */
export const FLAPPY_GLYPH_VERTICAL = '║';

/** Glyph character for interior spacing. */
export const FLAPPY_GLYPH_SPACE = ' ';
