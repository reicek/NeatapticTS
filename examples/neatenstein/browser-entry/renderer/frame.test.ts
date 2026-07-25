import { describe, expect, it } from '@jest/globals';
const expectedColumnCount = 320;

const loadModule = (path: string): Promise<any> => import(path);

function createMinimalRenderState() {
  return {
    canvasWidth: 640,
    canvasHeight: 360,
    simTick: 7,
    cameraX: 12.5,
    cameraY: 12.5,
    cameraYaw: 0.25,
    mapSeed: 42,
  };
}

describe('Neatenstein render frame helpers', () => {
  it('returns a frame with the versioned format identifier', async () => {
    const { buildNeatensteinRenderFrame } = await loadModule('./frame.ts');
    const frame = buildNeatensteinRenderFrame(
      createMinimalRenderState(),
      expectedColumnCount,
    );
    expect({ format: frame.format, version: frame.version }).toEqual({
      format: 'neatenstein-frame-v1',
      version: 'neatenstein-frame-v1',
    });
  });

  it('allocates SoA typed arrays sized to columnCount', async () => {
    const { buildNeatensteinRenderFrame } = await loadModule('./frame.ts');
    const columnCount = expectedColumnCount;
    const frame = buildNeatensteinRenderFrame(
      createMinimalRenderState(),
      columnCount,
    );
    expect({
      wallDistances: {
        length: frame.wallDistances.length,
        type: frame.wallDistances.constructor,
      },
      wallSides: {
        length: frame.wallSides.length,
        type: frame.wallSides.constructor,
      },
      zBuffer: {
        length: frame.zBuffer.length,
        type: frame.zBuffer.constructor,
      },
      enemyScreenX: {
        length: frame.enemyScreenX.length,
        type: frame.enemyScreenX.constructor,
      },
      enemyScale: {
        length: frame.enemyScale.length,
        type: frame.enemyScale.constructor,
      },
      projectileScreenX: {
        length: frame.projectileScreenX.length,
        type: frame.projectileScreenX.constructor,
      },
    }).toEqual({
      wallDistances: { length: columnCount, type: Float32Array },
      wallSides: { length: columnCount, type: Uint8Array },
      zBuffer: { length: columnCount, type: Float32Array },
      enemyScreenX: { length: columnCount, type: Float32Array },
      enemyScale: { length: columnCount, type: Float32Array },
      projectileScreenX: { length: columnCount, type: Float32Array },
    });
  });

  it('returns every typed-array buffer in the transfer list', async () => {
    const {
      buildNeatensteinRenderFrame,
      resolveNeatensteinRenderFrameTransferList,
    } = await loadModule('./frame.ts');
    const frame = buildNeatensteinRenderFrame(
      createMinimalRenderState(),
      expectedColumnCount,
    );
    const transferList = resolveNeatensteinRenderFrameTransferList(frame);
    expect({
      length: transferList.length,
      hasWallDistances: transferList.includes(frame.wallDistances.buffer),
      hasWallSides: transferList.includes(frame.wallSides.buffer),
      hasZBuffer: transferList.includes(frame.zBuffer.buffer),
      hasEnemyScreenX: transferList.includes(frame.enemyScreenX.buffer),
      hasEnemyScale: transferList.includes(frame.enemyScale.buffer),
      hasProjectileScreenX: transferList.includes(
        frame.projectileScreenX.buffer,
      ),
    }).toEqual({
      length: 6,
      hasWallDistances: true,
      hasWallSides: true,
      hasZBuffer: true,
      hasEnemyScreenX: true,
      hasEnemyScale: true,
      hasProjectileScreenX: true,
    });
  });

  it('increments requestId across frame builds', async () => {
    const { buildNeatensteinRenderFrame } = await loadModule('./frame.ts');
    const first = buildNeatensteinRenderFrame(
      createMinimalRenderState(),
      expectedColumnCount,
    );
    const second = buildNeatensteinRenderFrame(
      createMinimalRenderState(),
      expectedColumnCount,
    );
    expect(second.requestId).toBe(first.requestId + 1);
  });

  it('includes canvas and simulation metadata in the frame', async () => {
    const { buildNeatensteinRenderFrame } = await loadModule('./frame.ts');
    const state = createMinimalRenderState();
    const frame = buildNeatensteinRenderFrame(state, expectedColumnCount);
    expect({
      canvasWidth: frame.canvasWidth,
      canvasHeight: frame.canvasHeight,
      columnCount: frame.columnCount,
      simTick: frame.simTick,
    }).toEqual({
      canvasWidth: state.canvasWidth,
      canvasHeight: state.canvasHeight,
      columnCount: expectedColumnCount,
      simTick: state.simTick,
    });
  });
});
