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
  if (rectangleWidthPx <= 0 || rectangleHeightPx <= 0) {
    context.restore();
    return;
  }

  const alignedRectangle = resolveAlignedPipeOutlineRectangle({
    rectangleLeftPx,
    rectangleTopPx,
    rectangleWidthPx,
    rectangleHeightPx,
  });
  const pipeOutlinePath = resolvePipeOutlinePath(alignedRectangle);

  const previousCompositeOperation = context.globalCompositeOperation;
  context.globalCompositeOperation = 'lighter';
  context.strokeStyle = FLAPPY_NEON_PALETTE.pipeFill;

  context.globalAlpha = FLAPPY_PIPE_OUTLINE_GLOW_ALPHA;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX;
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.lineWidth = FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX;
  context.stroke(pipeOutlinePath);

  context.globalAlpha = 1;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.lineWidth = FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;
  context.stroke(pipeOutlinePath);

  context.globalCompositeOperation = previousCompositeOperation;
  context.restore();
}

/**
 * Resolves a pixel-aligned rectangle used by the pipe outline renderer.
 *
 * @param input - Raw pipe rectangle values.
 * @returns Aligned rectangle ready for outline rendering.
 */
function resolveAlignedPipeOutlineRectangle(input: {
  rectangleLeftPx: number;
  rectangleTopPx: number;
  rectangleWidthPx: number;
  rectangleHeightPx: number;
}): {
  alignedLeftPx: number;
  alignedTopPx: number;
  alignedWidthPx: number;
  alignedHeightPx: number;
} {
  return {
    alignedLeftPx: Math.round(input.rectangleLeftPx),
    alignedTopPx: Math.round(input.rectangleTopPx),
    alignedWidthPx: Math.max(1, Math.round(input.rectangleWidthPx)),
    alignedHeightPx: Math.max(1, Math.round(input.rectangleHeightPx)),
  };
}

/**
 * Resolves the reusable outline path for one pipe body and its entrance rim.
 *
 * @param alignedRectangle - Pixel-aligned rectangle used by the outline renderer.
 * @returns Path containing the outer pipe outline and optional entrance rim.
 */
function resolvePipeOutlinePath(alignedRectangle: {
  alignedLeftPx: number;
  alignedTopPx: number;
  alignedWidthPx: number;
  alignedHeightPx: number;
}): Path2D {
  const isTopPipeSegment = alignedRectangle.alignedTopPx === 0;
  const entranceLineYPx = isTopPipeSegment
    ? alignedRectangle.alignedTopPx +
      alignedRectangle.alignedHeightPx -
      FLAPPY_PIPE_ENTRY_RIM_INSET_PX
    : alignedRectangle.alignedTopPx + FLAPPY_PIPE_ENTRY_RIM_INSET_PX;
  const pipeOutlinePath = new Path2D();

  pipeOutlinePath.rect(
    alignedRectangle.alignedLeftPx,
    alignedRectangle.alignedTopPx,
    alignedRectangle.alignedWidthPx,
    alignedRectangle.alignedHeightPx,
  );
  if (
    entranceLineYPx > alignedRectangle.alignedTopPx &&
    entranceLineYPx <
      alignedRectangle.alignedTopPx + alignedRectangle.alignedHeightPx
  ) {
    pipeOutlinePath.moveTo(alignedRectangle.alignedLeftPx, entranceLineYPx);
    pipeOutlinePath.lineTo(
      alignedRectangle.alignedLeftPx + alignedRectangle.alignedWidthPx,
      entranceLineYPx,
    );
  }

  return pipeOutlinePath;
}
