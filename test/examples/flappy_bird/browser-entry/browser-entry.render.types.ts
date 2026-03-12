/**
 * Canvas frame-layout contracts for the Flappy Bird browser demo.
 *
 * These types support the demo's deliberately stylized text-frame chrome: title
 * boxes, outer borders, and viewport transforms that make the example feel more
 * like an instrument panel than a plain canvas game.
 */

/** Input contract for standalone title frame rendering helper. */
export interface RenderStandaloneTitleBoxInput {
  context: CanvasRenderingContext2D;
  widthPx: number;
  heightPx: number;
  titleText: string;
  glyphColor: string;
  font: string;
}

/**
 * Input contract for outer frame rendering helper.
 *
 * The outer frame is the decorative shell that visually separates the playable
 * world and telemetry panels from the rest of the page.
 */
export interface RenderClosedOuterBoxInput {
  context: CanvasRenderingContext2D;
  widthPx: number;
  heightPx: number;
  glyphColor: string;
  font: string;
}

/**
 * Viewport transform values for world-to-canvas rendering.
 *
 * These numbers answer the classic graphics question: how does one unit in the
 * simulated world map into the current canvas rectangle?
 */
export interface ViewportInfo {
  offsetXPx: number;
  offsetYPx: number;
  scale: number;
}

/**
 * Canvas text-grid metrics used for frame rendering layout helpers.
 *
 * The frame renderer measures glyph and row geometry once, then uses that grid
 * to place ASCII-style UI elements consistently.
 */
export interface TextFrameMetrics {
  centeredColumns: number;
  offsetXPx: number;
  glyphWidthPx: number;
  rowHeightPx: number;
  totalRows: number;
}
