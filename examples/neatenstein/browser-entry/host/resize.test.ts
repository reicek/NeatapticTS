import { describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION } from '../constants';
import type { NeatensteinRenderState } from '../renderer/frame';

const loadModule = <T>(path: string): Promise<T> => import(path) as Promise<T>;

function createRenderState(): NeatensteinRenderState {
  return {
    canvasWidth: 640,
    canvasHeight: 360,
    simTick: 1,
    cameraX: 12.5,
    cameraY: 12.5,
    cameraYaw: 0.25,
    mapSeed: 42,
  };
}

describe('Neatenstein host canvas resize', () => {
  describe('handleNeatensteinResize', () => {
    it('re-derives column stride and reallocates SoA frame buffers for the tier', async () => {
      const { handleNeatensteinResize } =
        await loadModule<typeof import('./resize.ts')>('./resize.ts');
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
        columnStride: 2,
        frameFormat: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
        frameColumnCount: 320,
        frameCanvasWidth: 640,
        frameCanvasHeight: 360,
        wallDistancesLength: 320,
        wallSidesLength: 320,
        zBufferLength: 320,
      });
    });

    it('uses the canvas width as the worker tier column count', async () => {
      const { handleNeatensteinResize } =
        await loadModule<typeof import('./resize.ts')>('./resize.ts');
      const state = createRenderState();
      const result = handleNeatensteinResize(state, 'worker');
      expect({
        columnStride: result.columnStride,
        frameColumnCount: result.frame.columnCount,
      }).toEqual({
        columnStride: 1,
        frameColumnCount: 640,
      });
    });

    it('throws when canvas dimensions are not finite', async () => {
      const { handleNeatensteinResize } =
        await loadModule<typeof import('./resize.ts')>('./resize.ts');
      const state = createRenderState();
      state.canvasWidth = NaN;
      state.canvasHeight = Infinity;
      expect(() => handleNeatensteinResize(state, 'cpu')).toThrow(
        'Canvas dimensions must be finite numbers',
      );
    });

    it('throws when canvas dimensions are not positive', async () => {
      const { handleNeatensteinResize } =
        await loadModule<typeof import('./resize.ts')>('./resize.ts');
      const state = createRenderState();
      state.canvasWidth = 0;
      state.canvasHeight = -1;
      expect(() => handleNeatensteinResize(state, 'cpu')).toThrow(
        'Canvas dimensions must be positive',
      );
    });

    it('throws when the render tier is unknown', async () => {
      const { handleNeatensteinResize } =
        await loadModule<typeof import('./resize.ts')>('./resize.ts');
      const state = createRenderState();
      expect(() =>
        handleNeatensteinResize(state, 'unknown' as unknown as 'cpu'),
      ).toThrow('Unknown render tier');
    });
  });
});
