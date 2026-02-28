import {
  FLAPPY_BOX_MIN_COLUMNS,
  FLAPPY_BOX_MIN_ROWS,
  FLAPPY_FRAME_GLYPH_ROW_HEIGHT_PX,
  FLAPPY_FRAME_MIN_COLUMNS,
  FLAPPY_FRAME_MIN_GLYPH_WIDTH_PX,
  FLAPPY_FRAME_RESERVED_COLUMNS,
  FLAPPY_GLYPH_BOTTOM_LEFT,
  FLAPPY_GLYPH_BOTTOM_RIGHT,
  FLAPPY_GLYPH_HORIZONTAL,
  FLAPPY_GLYPH_SPACE,
  FLAPPY_GLYPH_TOP_LEFT,
  FLAPPY_GLYPH_TOP_RIGHT,
  FLAPPY_GLYPH_VERTICAL,
  FLAPPY_HALF,
  FLAPPY_TITLE_BOX_MARGIN_COLUMNS,
  FLAPPY_TITLE_BOX_MIN_HALF_SPAN,
  FLAPPY_TITLE_BOX_MIN_OUTER_GAP_COLUMNS,
  FLAPPY_TITLE_BOX_MIN_WIDTH,
  FLAPPY_TITLE_BOX_WIDTH_RATIO,
} from './browser-entry.constants';
import type {
  RenderClosedOuterBoxInput,
  RenderStandaloneTitleBoxInput,
  TextFrameMetrics,
} from './browser-entry.types';

/**
 * Resolves core text-frame metrics for glyph box rendering.
 *
 * @param frameWidthPx - Frame width.
 * @param frameHeightPx - Frame height.
 * @param glyphWidthPx - Measured glyph width.
 * @param rowHeightPx - Glyph row height.
 * @param minimumColumns - Minimum column count.
 * @returns Text frame metrics.
 */
export function resolveTextFrameMetrics(
  frameWidthPx: number,
  frameHeightPx: number,
  glyphWidthPx: number,
  rowHeightPx = FLAPPY_FRAME_GLYPH_ROW_HEIGHT_PX,
  minimumColumns = FLAPPY_FRAME_MIN_COLUMNS,
): TextFrameMetrics {
  const maximumColumns = Math.max(
    minimumColumns,
    Math.floor(frameWidthPx / Math.max(1, glyphWidthPx)) -
      FLAPPY_FRAME_RESERVED_COLUMNS,
  );
  const centeredColumns =
    maximumColumns % 2 === 0
      ? maximumColumns - FLAPPY_FRAME_RESERVED_COLUMNS
      : maximumColumns;
  const offsetXPx = Math.max(
    0,
    Math.floor((frameWidthPx - centeredColumns * glyphWidthPx) * FLAPPY_HALF),
  );

  return {
    centeredColumns,
    offsetXPx,
    glyphWidthPx,
    rowHeightPx,
    totalRows: Math.max(3, Math.floor((frameHeightPx - 2) / rowHeightPx)),
  };
}

/**
 * Builds an ASCII outer frame with closed borders.
 *
 * @param centeredColumns - Centered column count.
 * @param totalRows - Total row count.
 * @returns Frame lines.
 */
export function buildOuterBoxLines(
  centeredColumns: number,
  totalRows: number,
): string[] {
  const safeColumns = Math.max(FLAPPY_BOX_MIN_COLUMNS, centeredColumns);
  const safeRows = Math.max(FLAPPY_BOX_MIN_ROWS, totalRows);
  const topRow = `${FLAPPY_GLYPH_TOP_LEFT}${FLAPPY_GLYPH_HORIZONTAL.repeat(
    safeColumns - FLAPPY_BOX_MIN_ROWS,
  )}${FLAPPY_GLYPH_TOP_RIGHT}`;
  const middleRow = `${FLAPPY_GLYPH_VERTICAL}${FLAPPY_GLYPH_SPACE.repeat(
    safeColumns - FLAPPY_BOX_MIN_ROWS,
  )}${FLAPPY_GLYPH_VERTICAL}`;
  const bottomRow = `${FLAPPY_GLYPH_BOTTOM_LEFT}${FLAPPY_GLYPH_HORIZONTAL.repeat(
    safeColumns - FLAPPY_BOX_MIN_ROWS,
  )}${FLAPPY_GLYPH_BOTTOM_RIGHT}`;

  return Array.from({ length: safeRows }, (_unusedValue, rowIndex) => {
    if (rowIndex === 0) return topRow;
    if (rowIndex === safeRows - 1) return bottomRow;
    return middleRow;
  });
}

/**
 * Builds an ASCII centered title box.
 *
 * @param centeredColumns - Available centered column count.
 * @param titleText - Title text.
 * @returns Three-row title box.
 */
export function buildCenteredTitleBoxLines(
  centeredColumns: number,
  titleText: string,
): string[] {
  const safeColumns = Math.max(FLAPPY_BOX_MIN_COLUMNS, centeredColumns);
  const centerIndex = Math.floor(safeColumns / FLAPPY_BOX_MIN_ROWS);
  const maxTitleBoxWidth = Math.max(
    FLAPPY_TITLE_BOX_MIN_WIDTH,
    safeColumns - FLAPPY_TITLE_BOX_MARGIN_COLUMNS,
  );
  const requestedTitleBoxWidth = Math.min(
    maxTitleBoxWidth,
    Math.max(
      titleText.length + FLAPPY_BOX_MIN_ROWS,
      Math.floor(safeColumns * FLAPPY_TITLE_BOX_WIDTH_RATIO),
    ),
  );
  const halfSpan = Math.max(
    FLAPPY_TITLE_BOX_MIN_HALF_SPAN,
    Math.floor((requestedTitleBoxWidth - FLAPPY_FRAME_RESERVED_COLUMNS) / 2),
  );
  const minimumOuterGapColumns = FLAPPY_TITLE_BOX_MIN_OUTER_GAP_COLUMNS;
  const maximumHalfSpan = Math.max(
    FLAPPY_TITLE_BOX_MIN_HALF_SPAN,
    Math.min(
      centerIndex - minimumOuterGapColumns,
      safeColumns -
        FLAPPY_FRAME_RESERVED_COLUMNS -
        minimumOuterGapColumns -
        centerIndex,
    ),
  );
  const resolvedHalfSpan = Math.min(halfSpan, maximumHalfSpan);
  const boxLeftIndex = centerIndex - resolvedHalfSpan;
  const boxRightIndex = centerIndex + resolvedHalfSpan;
  const textInnerStartIndex = boxLeftIndex + FLAPPY_FRAME_RESERVED_COLUMNS;
  const textInnerEndIndex = boxRightIndex - FLAPPY_FRAME_RESERVED_COLUMNS;
  const availableTextColumns = Math.max(
    0,
    textInnerEndIndex - textInnerStartIndex + FLAPPY_FRAME_RESERVED_COLUMNS,
  );
  const normalizedTitleText = titleText.trim();
  const clippedTitleText = normalizedTitleText.slice(0, availableTextColumns);
  const centeredTitleOffset = Math.max(
    0,
    Math.floor((availableTextColumns - clippedTitleText.length) * FLAPPY_HALF),
  );
  const centeredTitleStartIndex = textInnerStartIndex + centeredTitleOffset;

  const topRow = Array.from({ length: safeColumns }, () => FLAPPY_GLYPH_SPACE);
  const textRow = Array.from({ length: safeColumns }, () => FLAPPY_GLYPH_SPACE);
  const bottomRow = Array.from(
    { length: safeColumns },
    () => FLAPPY_GLYPH_SPACE,
  );

  for (
    let columnIndex = boxLeftIndex;
    columnIndex <= boxRightIndex;
    columnIndex++
  ) {
    if (columnIndex === boxLeftIndex) {
      topRow[columnIndex] = FLAPPY_GLYPH_TOP_LEFT;
      textRow[columnIndex] = FLAPPY_GLYPH_VERTICAL;
      bottomRow[columnIndex] = FLAPPY_GLYPH_BOTTOM_LEFT;
      continue;
    }
    if (columnIndex === boxRightIndex) {
      topRow[columnIndex] = FLAPPY_GLYPH_TOP_RIGHT;
      textRow[columnIndex] = FLAPPY_GLYPH_VERTICAL;
      bottomRow[columnIndex] = FLAPPY_GLYPH_BOTTOM_RIGHT;
      continue;
    }

    topRow[columnIndex] = FLAPPY_GLYPH_HORIZONTAL;
    bottomRow[columnIndex] = FLAPPY_GLYPH_HORIZONTAL;
    if (
      columnIndex >= centeredTitleStartIndex &&
      columnIndex < centeredTitleStartIndex + clippedTitleText.length
    ) {
      const centeredTitleIndex = columnIndex - centeredTitleStartIndex;
      textRow[columnIndex] =
        clippedTitleText[centeredTitleIndex] ?? FLAPPY_GLYPH_SPACE;
    }
  }

  return [topRow.join(''), textRow.join(''), bottomRow.join('')];
}

/**
 * Renders only the centered title box.
 *
 * @param input - Rendering input object.
 * @returns Nothing.
 */
export function renderStandaloneTitleBox(
  input: RenderStandaloneTitleBoxInput,
): void {
  input.context.clearRect(0, 0, input.widthPx, input.heightPx);
  input.context.font = input.font;
  input.context.textBaseline = 'top';
  input.context.fillStyle = input.glyphColor;
  input.context.shadowColor = 'transparent';
  input.context.shadowBlur = 0;

  const glyphWidthPx = Math.max(
    FLAPPY_FRAME_MIN_GLYPH_WIDTH_PX,
    Math.floor(input.context.measureText(FLAPPY_GLYPH_HORIZONTAL).width),
  );
  const frameMetrics = resolveTextFrameMetrics(
    input.widthPx,
    input.heightPx,
    glyphWidthPx,
    FLAPPY_FRAME_GLYPH_ROW_HEIGHT_PX,
  );
  const titleLines = buildCenteredTitleBoxLines(
    frameMetrics.centeredColumns,
    input.titleText,
  );
  const titleBlockHeightPx = titleLines.length * frameMetrics.rowHeightPx;
  const offsetYPx = Math.max(
    0,
    Math.floor((input.heightPx - titleBlockHeightPx) * FLAPPY_HALF),
  );

  titleLines.forEach((lineText, lineIndex) => {
    input.context.fillText(
      lineText,
      frameMetrics.offsetXPx,
      offsetYPx + lineIndex * frameMetrics.rowHeightPx,
    );
  });
}

/**
 * Renders a complete closed outer glyph box.
 *
 * @param input - Rendering input object.
 * @returns Nothing.
 */
export function renderClosedOuterBox(input: RenderClosedOuterBoxInput): void {
  input.context.clearRect(0, 0, input.widthPx, input.heightPx);
  input.context.font = input.font;
  input.context.textBaseline = 'top';
  input.context.textAlign = 'left';
  input.context.fillStyle = input.glyphColor;

  const glyphWidthPx = Math.max(
    FLAPPY_FRAME_MIN_GLYPH_WIDTH_PX,
    Math.floor(input.context.measureText(FLAPPY_GLYPH_HORIZONTAL).width),
  );
  const frameMetrics = resolveTextFrameMetrics(
    input.widthPx,
    input.heightPx,
    glyphWidthPx,
    FLAPPY_FRAME_GLYPH_ROW_HEIGHT_PX,
  );
  const outerLines = buildOuterBoxLines(
    frameMetrics.centeredColumns,
    frameMetrics.totalRows,
  );
  const outerBoxHeightPx = outerLines.length * frameMetrics.rowHeightPx;
  const offsetYPx = Math.max(
    0,
    Math.floor((input.heightPx - outerBoxHeightPx) * FLAPPY_HALF),
  );

  outerLines.forEach((lineText, rowIndex) => {
    input.context.fillText(
      lineText,
      frameMetrics.offsetXPx,
      offsetYPx + rowIndex * frameMetrics.rowHeightPx,
    );
  });
}
