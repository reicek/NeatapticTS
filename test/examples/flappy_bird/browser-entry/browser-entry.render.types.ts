/** Input contract for standalone title frame rendering helper. */
export interface RenderStandaloneTitleBoxInput {
  context: CanvasRenderingContext2D;
  widthPx: number;
  heightPx: number;
  titleText: string;
  glyphColor: string;
  font: string;
}

/** Input contract for outer frame rendering helper. */
export interface RenderClosedOuterBoxInput {
  context: CanvasRenderingContext2D;
  widthPx: number;
  heightPx: number;
  glyphColor: string;
  font: string;
}

/** Viewport transform values for world-to-canvas rendering. */
export interface ViewportInfo {
  offsetXPx: number;
  offsetYPx: number;
  scale: number;
}

/** Canvas text-grid metrics used for frame rendering layout helpers. */
export interface TextFrameMetrics {
  centeredColumns: number;
  offsetXPx: number;
  glyphWidthPx: number;
  rowHeightPx: number;
  totalRows: number;
}
