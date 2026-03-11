import {
  FLAPPY_PIPE_ENTRY_RIM_INSET_PX,
  FLAPPY_NEON_PALETTE,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX,
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

  // Step 2: Resolve the gap-facing rim line inset for this segment.
  const isTopPipeSegment = alignedTopPx === 0;
  const entranceLineYPx = isTopPipeSegment
    ? alignedTopPx + alignedHeightPx - FLAPPY_PIPE_ENTRY_RIM_INSET_PX
    : alignedTopPx + FLAPPY_PIPE_ENTRY_RIM_INSET_PX;

  // Step 3: Create one shared path for the outer outline plus the entrance rim.
  const pipeOutlinePath = new Path2D();
  pipeOutlinePath.rect(
    alignedLeftPx,
    alignedTopPx,
    alignedWidthPx,
    alignedHeightPx,
  );
  if (
    entranceLineYPx > alignedTopPx &&
    entranceLineYPx < alignedTopPx + alignedHeightPx
  ) {
    pipeOutlinePath.moveTo(alignedLeftPx, entranceLineYPx);
    pipeOutlinePath.lineTo(alignedLeftPx + alignedWidthPx, entranceLineYPx);
  }

  // Step 4: Draw one green glow pass followed by one crisp outline pass.
  const previousCompositeOperation = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';
  context.strokeStyle = FLAPPY_NEON_PALETTE.pipeFill;

  // Step 4.1: Soft glow pass for the outline and entrance rim.
  context.globalAlpha = FLAPPY_PIPE_OUTLINE_GLOW_ALPHA;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX;
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.lineWidth = FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX;
  context.stroke(pipeOutlinePath);

  // Step 4.2: Crisp outline pass with no extra shadow.
  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.lineWidth = FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;
  context.stroke(pipeOutlinePath);

  context.globalCompositeOperation = previousCompositeOperation;
  context.restore();
}
