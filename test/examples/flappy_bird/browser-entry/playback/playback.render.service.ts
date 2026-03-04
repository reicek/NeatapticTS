import {
  FLAPPY_NEON_PALETTE,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX,
} from '../../constants/constants';

/**
 * Draws a simplified neon outline around a pipe rectangle.
 *
 * @param context - Canvas 2D context.
 * @param rectangleLeftPx - Rectangle left position.
 * @param rectangleTopPx - Rectangle top position.
 * @param rectangleWidthPx - Rectangle width.
 * @param rectangleHeightPx - Rectangle height.
 * @returns Nothing.
 */
export function drawPipeNeonOutline(
  context: CanvasRenderingContext2D,
  rectangleLeftPx: number,
  rectangleTopPx: number,
  rectangleWidthPx: number,
  rectangleHeightPx: number,
): void {
  context.save();
  // Step 1: Guard degenerate rectangles.
  if (rectangleWidthPx <= 0 || rectangleHeightPx <= 0) {
    context.restore();
    return;
  }

  const alignedLeftPx = Math.round(rectangleLeftPx);
  const alignedTopPx = Math.round(rectangleTopPx);
  const alignedWidthPx = Math.max(1, Math.round(rectangleWidthPx));
  const alignedHeightPx = Math.max(1, Math.round(rectangleHeightPx));

  // Step 2: Compute outline geometry.
  // We want a small side gap and a larger "entrance" gap (top for bottom pipe,
  // bottom for top pipe) to suggest a pipe rim.
  const outlineStrokeWidthPx = FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;
  const outlineStrokeHalfPx = outlineStrokeWidthPx / 2;
  const sideExpandPx = Math.round(
    FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX + outlineStrokeHalfPx,
  );
  const entranceExpandPx = Math.round(
    FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX + outlineStrokeHalfPx,
  );

  const isTopPipeSegment = alignedTopPx === 0;
  const outlineTopExpandPx = isTopPipeSegment ? sideExpandPx : entranceExpandPx;
  const outlineBottomExpandPx = isTopPipeSegment
    ? entranceExpandPx
    : sideExpandPx;

  const outlineLeftPx = alignedLeftPx - sideExpandPx;
  const outlineTopPx = alignedTopPx - outlineTopExpandPx;
  const outlineWidthPx = alignedWidthPx + sideExpandPx * 2;
  const outlineHeightPx =
    alignedHeightPx + outlineTopExpandPx + outlineBottomExpandPx;

  // Step 3: Define pixel-aligned stroke helper to avoid blurry edges.
  const strokeAlignedRect = (
    leftPx: number,
    topPx: number,
    widthPx: number,
    heightPx: number,
    lineWidthPx: number,
  ): void => {
    const oddLineAlignmentOffsetPx = lineWidthPx % 2 === 1 ? 0.5 : 0;
    context.lineWidth = lineWidthPx;
    context.strokeRect(
      Math.round(leftPx) + oddLineAlignmentOffsetPx,
      Math.round(topPx) + oddLineAlignmentOffsetPx,
      Math.max(1, Math.round(widthPx)),
      Math.max(1, Math.round(heightPx)),
    );
  };

  // Step 4: Draw neon-green outline with cyan neon glow.
  const previousCompositeOperation = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';
  context.strokeStyle = FLAPPY_NEON_PALETTE.pipeFill;

  // Step 4.1: Soft glow pass (thicker stroke + cyan shadow).
  context.globalAlpha = FLAPPY_PIPE_OUTLINE_GLOW_ALPHA;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX;
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  strokeAlignedRect(
    outlineLeftPx,
    outlineTopPx,
    outlineWidthPx,
    outlineHeightPx,
    FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
  );

  // Step 4.2: Crisp outline pass (no shadow).
  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  strokeAlignedRect(
    outlineLeftPx,
    outlineTopPx,
    outlineWidthPx,
    outlineHeightPx,
    outlineStrokeWidthPx,
  );

  context.globalCompositeOperation = previousCompositeOperation;
  context.restore();
}
