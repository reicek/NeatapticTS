/**
 * Green-phase contract tests for examples/neatenstein/scripts/enemy-sprite.ts.
 *
 * Covers AC-703I and AC-704:
 * - billboard voxel sprite projection
 * - per-column z-buffer clipping
 * - directional diffuse lighting
 * - teal/orange bolt lighting for the 3-second spawn force-field and 4-second
 *   death de-rez
 */

import { describe, expect, it, jest } from '@jest/globals';
import type { ControlledEnemy } from './enemy-controller';
import {
  applyBoltLight,
  buildEnemyBillboard,
  clipEnemyBillboardSprite,
  computeDirectionalLightIntensity,
  computeEnemySpriteDirection,
  ENEMY_BOLT_LIGHT_ORANGE,
  ENEMY_BOLT_LIGHT_TEAL,
  ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS,
  ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS,
  projectEnemyBillboardSprite,
  renderEnemyBillboardSprite,
  sampleAtlasFramePixel,
  worldYawToSpriteDirection,
  type NeatensteinDirectionalLight,
  type NeatensteinEnemyCamera,
  type NeatensteinSpriteAtlas,
  type NeatensteinSpriteRenderContext,
} from './enemy-sprite';
import { buildNeatensteinZBuffer } from '../browser-entry/renderer/zbuffer';

/** Default camera used by most projection tests. */
function makeCamera(
  overrides?: Partial<NeatensteinEnemyCamera>,
): NeatensteinEnemyCamera {
  return {
    posX: 0,
    posY: 0,
    dirX: 1,
    dirY: 0,
    planeX: 0,
    planeY: 0.66,
    ...overrides,
  };
}

/** Build a minimal controlled enemy for renderer tests. */
function makeEnemy(
  overrides?: Partial<ControlledEnemy> & {
    position?: { x: number; y: number };
  },
): ControlledEnemy {
  return {
    index: 0,
    position: { x: 3, y: 0 },
    health: 100,
    yawRad: 0,
    animationState: 'idle',
    ammo: 3,
    fireCooldownMs: 0,
    deRezElapsedMs: 0,
    active: true,
    ...overrides,
  } as ControlledEnemy;
}

/** Build a tiny synthetic sprite atlas for deterministic tests. */
function makeAtlas(
  cellSize = 2,
  directions = 2,
  states = 2,
  frames = 2,
): NeatensteinSpriteAtlas {
  const width = frames * cellSize;
  const height = directions * states * cellSize;
  const data = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const cellX = Math.floor(x / cellSize);
      const cellY = Math.floor(y / cellSize);
      const idx = (y * width + x) * 4;
      // Encode direction in red, state in green, frame in blue so tests can
      // verify atlas addressing.
      data[idx] = cellY * 17;
      data[idx + 1] = cellX * 23;
      data[idx + 2] = 200;
      data[idx + 3] = 255;
    }
  }

  return {
    width,
    height,
    cellSize,
    directions,
    states,
    data,
  };
}

/** Build a solid-color atlas (every frame identical). */
function makeSolidAtlas(
  color: { r: number; g: number; b: number },
  cellSize = 2,
): NeatensteinSpriteAtlas {
  const directions = 8;
  const states = 4;
  const frames = 1;
  const width = frames * cellSize;
  const height = directions * states * cellSize;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = color.r;
    data[i + 1] = color.g;
    data[i + 2] = color.b;
    data[i + 3] = 255;
  }
  return { width, height, cellSize, directions, states, data };
}

/** Build a mock render context that records putImageData calls. */
function makeRenderContext(): NeatensteinSpriteRenderContext & {
  calls: Array<{ data: Uint8ClampedArray; width: number; height: number }>;
} {
  const calls: Array<{
    data: Uint8ClampedArray;
    width: number;
    height: number;
  }> = [];
  return {
    calls,
    putImageData: jest.fn(
      (imageData: {
        data: Uint8ClampedArray;
        width: number;
        height: number;
      }) => {
        calls.push({
          data: imageData.data,
          width: imageData.width,
          height: imageData.height,
        });
      },
    ),
  };
}

describe('enemy-sprite timing constants (AC-704)', () => {
  it('exports a 3000 ms spawn force-field duration', () => {
    expect(ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS).toBe(3000);
  });

  it('exports a 4000 ms death de-rez duration', () => {
    expect(ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS).toBe(4000);
  });

  it('buildEnemyBillboard clamps negative elapsed times to zero', () => {
    const enemy = makeEnemy();
    const billboard = buildEnemyBillboard(enemy, -100, -50);
    expect(billboard.spawnElapsedMs).toBe(0);
    expect(billboard.animationElapsedMs).toBe(0);
  });
});

describe('enemy-sprite direction mapping', () => {
  it('worldYawToSpriteDirection wraps angles into 8 directions', () => {
    expect(worldYawToSpriteDirection(0)).toBe(0);
    expect(worldYawToSpriteDirection(Math.PI / 4)).toBe(1);
    expect(worldYawToSpriteDirection(2 * Math.PI)).toBe(0);
    expect(worldYawToSpriteDirection(-Math.PI / 4)).toBe(7);
    expect(worldYawToSpriteDirection(Number.NaN)).toBe(0);
  });

  it('worldYawToSpriteDirection handles custom direction counts', () => {
    expect(worldYawToSpriteDirection(0, 4)).toBe(0);
    expect(worldYawToSpriteDirection(Math.PI / 2, 4)).toBe(1);
    expect(worldYawToSpriteDirection(0, 0)).toBe(0);
  });

  it('computeEnemySpriteDirection returns a valid atlas direction', () => {
    const billboard = buildEnemyBillboard(makeEnemy({ yawRad: 0 }));
    const camera = makeCamera();
    const direction = computeEnemySpriteDirection(billboard, camera);
    expect(direction).toBeGreaterThanOrEqual(0);
    expect(direction).toBeLessThan(8);
  });

  it('computeEnemySpriteDirection changes when the camera rotates', () => {
    const billboard = buildEnemyBillboard(makeEnemy({ yawRad: Math.PI / 2 }));
    const forward = computeEnemySpriteDirection(billboard, makeCamera());
    const rotated = computeEnemySpriteDirection(
      billboard,
      makeCamera({ dirX: 0, dirY: 1, planeX: -0.66, planeY: 0 }),
    );
    expect(rotated).not.toBe(forward);
  });

  it('computeEnemySpriteDirection falls back to enemy yaw for a degenerate camera direction', () => {
    const billboard = buildEnemyBillboard(makeEnemy({ yawRad: Math.PI / 4 }));
    const fallback = computeEnemySpriteDirection(
      billboard,
      makeCamera({ dirX: Number.NaN, dirY: Number.NaN }),
    );
    expect(fallback).toBe(worldYawToSpriteDirection(billboard.enemy.yawRad));
  });
});

describe('enemy-sprite billboard projection', () => {
  it('projectEnemyBillboardSprite returns a visible projection for an enemy in front', () => {
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 } }),
    );
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera(),
      320,
      200,
    );

    expect(projection.visible).toBe(true);
    expect(projection.perpDist).toBeGreaterThan(0);
    expect(projection.scale).toBeGreaterThan(0);
    expect(projection.screenX).toBeGreaterThan(0);
    expect(projection.screenX).toBeLessThan(320);
    expect(projection.left).toBeLessThan(projection.right);
  });

  it('projectEnemyBillboardSprite returns invisible for an enemy behind the camera', () => {
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: -3, y: 0 } }),
    );
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera(),
      320,
      200,
    );

    expect(projection.visible).toBe(false);
  });

  it('projectEnemyBillboardSprite returns invisible for invalid dimensions', () => {
    const billboard = buildEnemyBillboard(makeEnemy());
    expect(
      projectEnemyBillboardSprite(billboard, makeCamera(), -1, 200).visible,
    ).toBe(false);
    expect(
      projectEnemyBillboardSprite(billboard, makeCamera(), 320, 0).visible,
    ).toBe(false);
  });

  it('projectEnemyBillboardSprite returns invisible for a degenerate camera', () => {
    const billboard = buildEnemyBillboard(makeEnemy());
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera({ dirX: 0, dirY: 0, planeX: 0, planeY: 0 }),
      320,
      200,
    );
    expect(projection.visible).toBe(false);
  });

  it('projectEnemyBillboardSprite returns invisible for non-finite positions', () => {
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: Number.NaN, y: 0 } }),
    );
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera(),
      320,
      200,
    );
    expect(projection.visible).toBe(false);
  });
});

describe('enemy-sprite z-buffer clipping', () => {
  it('clipEnemyBillboardSprite exposes all columns when the z-buffer is empty', () => {
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 } }),
    );
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera(),
      32,
      16,
    );
    const zBuffer = buildNeatensteinZBuffer(32);
    const clipped = clipEnemyBillboardSprite(projection, zBuffer);

    expect(clipped.visibleColumns.length).toBeGreaterThan(0);
  });

  it('clipEnemyBillboardSprite occludes columns where a wall is closer', () => {
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 } }),
    );
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera(),
      32,
      16,
    );
    const zBuffer = buildNeatensteinZBuffer(32);
    for (let i = 0; i < zBuffer.length; i += 1) {
      zBuffer[i] = 1; // wall closer than enemy
    }
    const clipped = clipEnemyBillboardSprite(projection, zBuffer);

    expect(clipped.visibleColumns.length).toBe(0);
  });

  it('clipEnemyBillboardSprite returns empty for an invisible projection', () => {
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: -3, y: 0 } }),
    );
    const projection = projectEnemyBillboardSprite(
      billboard,
      makeCamera(),
      32,
      16,
    );
    const clipped = clipEnemyBillboardSprite(
      projection,
      buildNeatensteinZBuffer(32),
    );
    expect(clipped.visibleColumns.length).toBe(0);
  });
});

describe('enemy-sprite directional light', () => {
  it('computeDirectionalLightIntensity returns ambient for light from behind', () => {
    const light: NeatensteinDirectionalLight = {
      dirX: -1,
      dirY: 0,
      intensity: 1,
    };
    expect(computeDirectionalLightIntensity(0, light)).toBe(0.25);
  });

  it('computeDirectionalLightIntensity returns ambient plus intensity for aligned light', () => {
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 0.5,
    };
    expect(computeDirectionalLightIntensity(0, light)).toBe(0.75);
  });

  it('computeDirectionalLightIntensity clamps total light to 1.0', () => {
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 2,
    };
    expect(computeDirectionalLightIntensity(0, light)).toBe(1);
  });

  it('computeDirectionalLightIntensity uses custom ambient when provided', () => {
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 0,
      ambient: 0.4,
    };
    expect(computeDirectionalLightIntensity(0, light)).toBe(0.4);
  });

  it('computeDirectionalLightIntensity falls back to ambient for invalid light', () => {
    expect(
      computeDirectionalLightIntensity(0, {
        dirX: Number.NaN,
        dirY: 0,
        intensity: 1,
      }),
    ).toBe(0.25);
  });

  it('computeDirectionalLightIntensity clamps ambient below 0 up to 0', () => {
    expect(
      computeDirectionalLightIntensity(0, {
        dirX: 1,
        dirY: 0,
        intensity: 0,
        ambient: -0.5,
      }),
    ).toBe(0);
  });

  it('computeDirectionalLightIntensity clamps ambient above 1 down to 1', () => {
    expect(
      computeDirectionalLightIntensity(0, {
        dirX: 1,
        dirY: 0,
        intensity: 0,
        ambient: 1.5,
      }),
    ).toBe(1);
  });
});

describe('enemy-sprite bolt lighting', () => {
  it('applyBoltLight returns the base color when elapsed equals or exceeds duration', () => {
    expect(
      applyBoltLight(100, 50, 25, 3000, 3000, ENEMY_BOLT_LIGHT_TEAL),
    ).toEqual({
      r: 100,
      g: 50,
      b: 25,
    });
    expect(
      applyBoltLight(100, 50, 25, 4000, 3000, ENEMY_BOLT_LIGHT_TEAL),
    ).toEqual({
      r: 100,
      g: 50,
      b: 25,
    });
  });

  it('applyBoltLight tints toward the bolt color during the interval', () => {
    const tinted = applyBoltLight(255, 0, 0, 0, 3000, ENEMY_BOLT_LIGHT_TEAL);
    expect(tinted.g).toBeGreaterThan(0);
    expect(tinted.b).toBeGreaterThan(0);
    expect(tinted.r).toBeLessThan(255);
  });

  it('applyBoltLight produces a deterministic flicker over time', () => {
    const first = applyBoltLight(255, 0, 0, 0, 3000, ENEMY_BOLT_LIGHT_TEAL);
    const later = applyBoltLight(255, 0, 0, 100, 3000, ENEMY_BOLT_LIGHT_TEAL);
    expect(first).not.toEqual(later);
  });

  it('applyBoltLight clamps a negative elapsed value to zero', () => {
    const tinted = applyBoltLight(255, 0, 0, -1, 3000, ENEMY_BOLT_LIGHT_ORANGE);
    expect(tinted.g).toBeGreaterThan(0);
  });

  it('applyBoltLight returns the base color for a non-positive or non-finite duration', () => {
    expect(applyBoltLight(100, 100, 100, 0, 0, ENEMY_BOLT_LIGHT_TEAL)).toEqual({
      r: 100,
      g: 100,
      b: 100,
    });
    expect(
      applyBoltLight(100, 100, 100, 0, Number.NaN, ENEMY_BOLT_LIGHT_TEAL),
    ).toEqual({
      r: 100,
      g: 100,
      b: 100,
    });
  });
});

describe('enemy-sprite atlas sampling', () => {
  it('sampleAtlasFramePixel reads the correct cell for direction/state/frame', () => {
    const atlas = makeAtlas(2, 2, 2, 2);
    const pixel = sampleAtlasFramePixel(atlas, 1, 1, 1, 0.5, 0.5);
    // cellY for direction=1, state=1 is row 1*2+1 = 3; red channel = 3*17 = 51
    expect(pixel.r).toBe(51);
    // cellX for frame=1 is 1; green channel = 1*23 = 23
    expect(pixel.g).toBe(23);
  });

  it('sampleAtlasFramePixel returns transparent black for invalid atlas data', () => {
    const atlas = makeAtlas(2, 2, 2, 2);
    atlas.data = new Uint8ClampedArray(8); // wrong size
    expect(sampleAtlasFramePixel(atlas, 0, 0, 0, 0.5, 0.5).a).toBe(0);
  });

  it('sampleAtlasFramePixel returns transparent black for out-of-range frames', () => {
    const atlas = makeAtlas(2, 2, 2, 2);
    expect(sampleAtlasFramePixel(atlas, 2, 0, 0, 0.5, 0.5).a).toBe(0);
    expect(sampleAtlasFramePixel(atlas, 0, 3, 0, 0.5, 0.5).a).toBe(0);
    expect(sampleAtlasFramePixel(atlas, 0, 0, 2, 0.5, 0.5).a).toBe(0);
  });
});

describe('enemy-sprite rendering', () => {
  it('renderEnemyBillboardSprite draws visible pixels and flushes the framebuffer', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 }, yawRad: 0 }),
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(1);
    const nonTransparentPixels = framebuffer.filter(
      (v, i) => i % 4 === 3 && v > 0,
    ).length;
    expect(nonTransparentPixels).toBeGreaterThan(0);
  });

  it('renderEnemyBillboardSprite does not flush when the sprite is fully occluded', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    for (let i = 0; i < zBuffer.length; i += 1) {
      zBuffer[i] = 1;
    }
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 }, yawRad: 0 }),
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(0);
  });

  it('renderEnemyBillboardSprite applies teal spawn force-field bolt lighting', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 }, yawRad: 0 }),
      0,
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(1);
    // Find a non-transparent pixel and assert it shifted toward teal.
    let found = false;
    for (let i = 0; i < framebuffer.length; i += 4) {
      if (framebuffer[i + 3] > 0) {
        expect(framebuffer[i + 2]).toBeGreaterThan(50); // blue channel boosted
        found = true;
        break;
      }
    }
    expect(found).toBe(true);
  });

  it('renderEnemyBillboardSprite applies orange death de-rez bolt lighting', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({
        position: { x: 3, y: 0 },
        yawRad: 0,
        animationState: 'death',
        deRezElapsedMs: 0,
      }),
      ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS,
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(1);
    let found = false;
    for (let i = 0; i < framebuffer.length; i += 4) {
      if (framebuffer[i + 3] > 0) {
        expect(framebuffer[i + 1]).toBeGreaterThan(50); // green channel boosted
        found = true;
        break;
      }
    }
    expect(found).toBe(true);
  });

  it('renderEnemyBillboardSprite skips transparent atlas pixels', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    for (let i = 3; i < atlas.data.length; i += 4) {
      atlas.data[i] = 0;
    }
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 }, yawRad: 0 }),
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(1);
    expect(framebuffer.filter((v, i) => i % 4 === 3 && v > 0).length).toBe(0);
  });

  it('renderEnemyBillboardSprite renders without bolt tint when force-field and de-rez are inactive', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({
        position: { x: 3, y: 0 },
        yawRad: 0,
        animationState: 'move',
      }),
      ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS,
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(1);
    const nonTransparentPixels = framebuffer.filter(
      (v, i) => i % 4 === 3 && v > 0,
    ).length;
    expect(nonTransparentPixels).toBeGreaterThan(0);
  });

  it('renderEnemyBillboardSprite returns early for an invalid framebuffer', () => {
    const framebuffer = new Uint8ClampedArray(8);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 } }),
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(0);
  });

  it('renderEnemyBillboardSprite returns early for an empty z-buffer', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const billboard = buildEnemyBillboard(
      makeEnemy({ position: { x: 3, y: 0 } }),
    );
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      new Float32Array(0),
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(0);
  });

  it('renderEnemyBillboardSprite returns early for an unsupported animation state', () => {
    const framebuffer = new Uint8ClampedArray(32 * 16 * 4);
    const zBuffer = buildNeatensteinZBuffer(32);
    const atlas = makeSolidAtlas({ r: 200, g: 50, b: 50 }, 2);
    const enemy = {
      ...makeEnemy({ position: { x: 3, y: 0 } }),
      animationState: 'damage',
    } as unknown as ControlledEnemy;
    const billboard = buildEnemyBillboard(enemy);
    const ctx = makeRenderContext();
    const light: NeatensteinDirectionalLight = {
      dirX: 1,
      dirY: 0,
      intensity: 1,
    };

    renderEnemyBillboardSprite(
      framebuffer,
      32,
      16,
      zBuffer,
      billboard,
      makeCamera(),
      atlas,
      light,
      ctx,
    );

    expect(ctx.calls.length).toBe(0);
  });
});
