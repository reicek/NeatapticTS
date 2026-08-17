/**
 * @module enemy-sprite
 *
 * Billboard voxel enemy-sprite rendering for the Neatenstein NGE demo.
 *
 * Consumes the 8-direction × 4-state sprite atlas produced by the build-time
 * pipeline (`generate-enemy-sprites.ts`) and renders enemies as camera-facing
 * billboards with:
 *
 * - inverse-camera projection
 * - per-column z-buffer clipping
 * - directional diffuse light keyed to enemy yaw
 * - teal/orange bolt lighting for the 3-second spawn force-field and
 *   700 ms death de-rez
 *
 * The module is intentionally environment-agnostic: it only operates on
 * typed arrays and simple math, so it can be unit-tested in Node and later
 * dropped into the browser CPU renderer without change.
 *
 * The camera-plane billboard projection follows Lode Vandevenne's classic
 * raycasting sprite transform.
 * @see {@link https://lodev.org/cgtutor/raycasting2.html}
 */

// Re-export extracted types and constants so existing imports stay valid.
export type {
  NeatensteinEnemyCamera,
  NeatensteinEnemyProjection,
  NeatensteinDirectionalLight,
  NeatensteinEnemyBillboard,
  NeatensteinSpriteRenderContext,
  NeatensteinSpriteAtlas,
} from './enemy-sprite.types';
export {
  ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS,
  ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS,
  ENEMY_SPRITE_WORLD_SIZE,
  ENEMY_SPRITE_NEAR_CLIP,
  ENEMY_SPRITE_RGBA_CHANNELS,
  ENEMY_BOLT_LIGHT_TEAL,
  ENEMY_BOLT_LIGHT_ORANGE,
  ENEMY_SPRITE_DETERMINANT_EPSILON,
  ENEMY_SPRITE_AMBIENT_LIGHT,
  ENEMY_SPRITE_MAX_LIGHT,
  ENEMY_BOLT_PULSE_PERIOD_MS,
  ENEMY_SPRITE_STATES,
  ENEMY_SPRITE_DIRECTIONS,
  ENEMY_SPRITE_ATLAS_FILENAME,
  ENEMY_SPRITE_MANIFEST_FILENAME,
} from './enemy-sprite.constants';

// Re-export public symbols that moved to the sibling utils file.
export {
  applyBoltLight,
  buildEnemyBillboard,
  clipEnemyBillboardSprite,
  computeDirectionalLightIntensity,
  computeEnemySpriteDirection,
  projectEnemyBillboardSprite,
  sampleAtlasFramePixel,
  worldYawToSpriteDirection,
} from './enemy-sprite.utils';

// Import executors used by the orchestrator (not re-exported).
import {
  applyBoltLight,
  clipEnemyBillboardSprite,
  computeDirectionalLightIntensity,
  computeEnemySpriteDirection,
  projectEnemyBillboardSprite,
  resolveEnemyAnimationFrame,
  resolveFramebufferDimensions,
  sampleAtlasFramePixel,
} from './enemy-sprite.utils';

// Import types and constants needed by the orchestrator.
import type {
  NeatensteinEnemyBillboard,
  NeatensteinEnemyCamera,
  NeatensteinSpriteAtlas,
  NeatensteinDirectionalLight,
  NeatensteinSpriteRenderContext,
} from './enemy-sprite.types';
import {
  ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS,
  ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS,
  ENEMY_BOLT_LIGHT_TEAL,
  ENEMY_BOLT_LIGHT_ORANGE,
  ENEMY_SPRITE_RGBA_CHANNELS,
} from './enemy-sprite.constants';

/**
 * Render an enemy billboard into a flat RGBA framebuffer with z-buffer
 * occlusion, directional light, and bolt lighting.
 *
 * Only sprite columns closer than the wall at the same screen column are
 * drawn. The framebuffer is flushed once with `putImageData` after all visible
 * columns are processed.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param framebufferWidth - Framebuffer width in pixels.
 * @param framebufferHeight - Framebuffer height in pixels.
 * @param zBuffer - Per-column wall-depth buffer.
 * @param billboard - Enemy billboard descriptor.
 * @param camera - Current camera transform.
 * @param atlas - Runtime sprite atlas.
 * @param light - Directional light source.
 * @param ctx - Canvas-like context with `putImageData`.
 */
export function renderEnemyBillboardSprite(
  framebuffer: Uint8ClampedArray,
  framebufferWidth: number,
  framebufferHeight: number,
  zBuffer: Readonly<Float32Array>,
  billboard: NeatensteinEnemyBillboard,
  camera: NeatensteinEnemyCamera,
  atlas: NeatensteinSpriteAtlas,
  light: NeatensteinDirectionalLight,
  ctx: NeatensteinSpriteRenderContext,
): void {
  const dimensions = resolveFramebufferDimensions(
    framebuffer,
    framebufferWidth,
    framebufferHeight,
  );
  if (dimensions === null || zBuffer.length === 0) {
    return;
  }

  const { width, height } = dimensions;

  const projection = projectEnemyBillboardSprite(
    billboard,
    camera,
    width,
    height,
  );
  const clipped = clipEnemyBillboardSprite(projection, zBuffer);

  if (clipped.visibleColumns.length === 0) {
    return;
  }

  const animation = resolveEnemyAnimationFrame(billboard);
  if (animation === null) {
    return;
  }

  const direction = computeEnemySpriteDirection(billboard, camera);
  const lightIntensity = computeDirectionalLightIntensity(
    billboard.enemy.yawRad,
    light,
  );

  const spawnElapsedMs = billboard.spawnElapsedMs;
  const spawnBoltActive =
    spawnElapsedMs < ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS;

  const deathElapsedMs = billboard.enemy.deRezElapsedMs;
  const deathBoltActive =
    billboard.enemy.animationState === 'death' &&
    deathElapsedMs < ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS;

  const centerY = height / 2;
  const halfScale = clipped.scale / 2;
  const drawStart = Math.floor(centerY - halfScale);
  const drawEnd = Math.floor(centerY + halfScale);
  const spanWidth = clipped.right - clipped.left;

  for (const column of clipped.visibleColumns) {
    const u = (column - clipped.left) / spanWidth;

    for (let row = drawStart; row < drawEnd; row += 1) {
      const v = (row - drawStart) / (drawEnd - drawStart);
      const sample = sampleAtlasFramePixel(
        atlas,
        direction,
        animation.stateIndex,
        animation.frameIndex,
        u,
        v,
      );

      if (sample.a === 0) {
        continue;
      }

      let r = sample.r;
      let g = sample.g;
      let b = sample.b;

      // Directional diffuse lighting.
      r = Math.round(r * lightIntensity);
      g = Math.round(g * lightIntensity);
      b = Math.round(b * lightIntensity);

      // Spawn/death bolt lighting (spawn takes precedence over death).
      if (spawnBoltActive) {
        const bolt = applyBoltLight(
          r,
          g,
          b,
          spawnElapsedMs,
          ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS,
          ENEMY_BOLT_LIGHT_TEAL,
        );
        r = bolt.r;
        g = bolt.g;
        b = bolt.b;
      } else if (deathBoltActive) {
        const bolt = applyBoltLight(
          r,
          g,
          b,
          deathElapsedMs,
          ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS,
          ENEMY_BOLT_LIGHT_ORANGE,
        );
        r = bolt.r;
        g = bolt.g;
        b = bolt.b;
      }

      const offset = (row * width + column) * ENEMY_SPRITE_RGBA_CHANNELS;
      framebuffer[offset] = r;
      framebuffer[offset + 1] = g;
      framebuffer[offset + 2] = b;
      framebuffer[offset + 3] = sample.a;
    }
  }

  ctx.putImageData({ data: framebuffer, width, height }, 0, 0);
}
