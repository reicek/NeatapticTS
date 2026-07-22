import { describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';
import type { NeatensteinRenderState } from '../renderer/frame';

const loadModule = (path: string): Promise<any> => import(path);

function createRenderState(): NeatensteinRenderState {
  return {
    canvasWidth: 640,
    canvasHeight: 360,
    simTick: 1,
  };
}

describe('Neatenstein host canvas resize', () => {
  describe('handleNeatensteinResize', () => {
    it('re-derives column stride and reallocates SoA frame buffers for the tier', async () => {
      const { handleNeatensteinResize } = await loadModule('./resize.ts');
      const state = createRenderState();
      const result = handleNeatensteinResize(state, 'cpu');
      expect({
        columnStride: result.columnStride,
        frameFormat: result.frame.format,
        frameColumnCount: result.frame.columnCount,
        frameCanvasWidth: result.frame.canvasWidth,
        frameCanvasHeight: result.frame.canvasHeight,
        wallDistancesLength: result.frame.wallDistances.length,
        wallSidesLength: result.frame.wallSides.length,
        zBufferLength: result.frame.zBuffer.length,
      }).toEqual({
        columnStride: 4,
        frameFormat: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
        frameColumnCount: NEATENSTEIN_CPU_COLUMN_COUNT,
        frameCanvasWidth: 640,
        frameCanvasHeight: 360,
        wallDistancesLength: NEATENSTEIN_CPU_COLUMN_COUNT,
        wallSidesLength: NEATENSTEIN_CPU_COLUMN_COUNT,
        zBufferLength: NEATENSTEIN_CPU_COLUMN_COUNT,
      });
    });
  });
});
