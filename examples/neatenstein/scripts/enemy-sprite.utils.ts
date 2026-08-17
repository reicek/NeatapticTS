/**
 * @module enemy-sprite.utils
 *
 * Pure executor functions for enemy billboard sprite rendering.
 *
 * Extracted from {@link module:enemy-sprite} as part of the SOLID-split
 * refactoring. All functions are pure (no module-level mutable state) and
 * operate on typed arrays and simple math. The main module retains
 * {@link renderEnemyBillboardSprite} as the declarative orchestrator and
 * re-exports every previously-public symbol from this file.
 */

import type { ControlledEnemy } from './enemy-controller.types';
import {
  getEnemyAnimationFrame,
  type EnemyAnimationState,
} from './enemy-animator';
import {
  clipNeatensteinSpriteSpan,
  type NeatensteinSpriteClip,
} from '../browser-entry/renderer/zbuffer';
import type {
  NeatensteinEnemyBillboard,
  NeatensteinEnemyCamera,
  NeatensteinEnemyProjection,
  NeatensteinDirectionalLight,
  NeatensteinSpriteAtlas,
} from './enemy-sprite.types';
import {
  ENEMY_SPRITE_NEAR_CLIP,
  ENEMY_SPRITE_RGBA_CHANNELS,
  ENEMY_SPRITE_WORLD_SIZE,
  ENEMY_SPRITE_DETERMINANT_EPSILON,
  ENEMY_SPRITE_AMBIENT_LIGHT,
  ENEMY_SPRITE_MAX_LIGHT,
  ENEMY_BOLT_PULSE_PERIOD_MS,
  ENEMY_SPRITE_STATES,
  ENEMY_SPRITE_DIRECTIONS,
} from './enemy-sprite.constants';

/**
 * Return whether a number is finite and greater than zero.
 *
 * @param value - Candidate scalar.
 * @returns Whether the value is positive and finite.
 */
function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Return whether a number is a positive finite integer dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is usable as a framebuffer/canvas dimension.
 */
function isPositiveIntegerDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

/**
 * Clamp a value to the inclusive `[0, 1]` range.
 *
 * @param value - Value to clamp.
 * @returns Clamped value.
 */
function clamp01(value: number): number {
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

/**
 * Wrap a signed integer into the `[0, directions - 1]` range.
 *
 * @param value - Signed direction index.
 * @param directions - Number of directions in the atlas.
 * @returns Wrapped direction index.
 */
function wrapDirection(value: number, directions: number): number {
  let wrapped = value % directions;
  if (wrapped < 0) {
    wrapped += directions;
  }
  return wrapped;
}

/**
 * Extract 8-bit RGB channels from a packed 0xRRGGBB color.
 *
 * @param packed - Packed RGB color.
 * @returns RGB triplet in `[0, 255]`.
 */
function unpackBoltRgb(packed: number): { r: number; g: number; b: number } {
  return {
    r: (packed >> 16) & 0xff,
    g: (packed >> 8) & 0xff,
    b: packed & 0xff,
  };
}

/**
 * Return a reusable invisible projection result.
 *
 * @param perpDist - Perpendicular distance associated with the rejected sprite.
 * @returns Invisible sprite projection.
 */
function createInvisibleEnemyProjection(
  perpDist = Number.POSITIVE_INFINITY,
): NeatensteinEnemyProjection {
  return {
    screenX: -1,
    scale: 0,
    perpDist,
    left: -1,
    right: -1,
    visible: false,
  };
}

/**
 * Build a billboard wrapper around a controlled enemy, adding spawn timing and
 * animation timing fields that the sprite renderer needs for effects.
 *
 * @param enemy - Controlled enemy descriptor.
 * @param spawnElapsedMs - Milliseconds since the enemy spawned.
 * @param animationElapsedMs - Milliseconds to use for non-death frame selection.
 * @returns Billboard descriptor ready for projection/rendering.
 */
export function buildEnemyBillboard(
  enemy: ControlledEnemy,
  spawnElapsedMs = 0,
  animationElapsedMs = 0,
): NeatensteinEnemyBillboard {
  return {
    enemy,
    spawnElapsedMs: Math.max(0, spawnElapsedMs),
    animationElapsedMs: Math.max(0, animationElapsedMs),
  };
}

/**
 * Convert a world yaw angle into an atlas direction index.
 *
 * Directions are spaced evenly around the full circle. The result is wrapped to
 * `[0, directions - 1]`.
 *
 * @param yawRad - World yaw in radians; 0 = +X axis.
 * @param directions - Number of directions in the atlas.
 * @returns Direction index.
 */
export function worldYawToSpriteDirection(
  yawRad: number,
  directions = ENEMY_SPRITE_DIRECTIONS,
): number {
  if (!Number.isFinite(yawRad) || directions <= 0) {
    return 0;
  }

  const turns = yawRad / (2 * Math.PI);
  return wrapDirection(Math.round(turns * directions), directions);
}

/**
 * Select the atlas yaw direction that faces the camera for a given enemy
 * billboard.
 *
 * The billboard always turns to face the camera. The chosen direction is the
 * one where the enemy's front points toward the camera, computed from the
 * relative angle between the camera view direction and the enemy yaw.
 *
 * @param billboard - Enemy billboard descriptor.
 * @param camera - Current camera transform.
 * @returns Atlas direction index in `[0, directions - 1]`.
 */
export function computeEnemySpriteDirection(
  billboard: NeatensteinEnemyBillboard,
  camera: NeatensteinEnemyCamera,
): number {
  if (!Number.isFinite(camera.dirX) || !Number.isFinite(camera.dirY)) {
    return worldYawToSpriteDirection(billboard.enemy.yawRad);
  }

  const cameraAngle = Math.atan2(camera.dirY, camera.dirX);
  const relativeAngle = cameraAngle - billboard.enemy.yawRad;
  return worldYawToSpriteDirection(relativeAngle);
}

/**
 * Map a controlled enemy animation state to its atlas row index.
 *
 * @param state - Animation state.
 * @returns Row index in the generated atlas, or -1 for unknown states.
 */
function animationStateToAtlasIndex(state: EnemyAnimationState): number {
  const index = ENEMY_SPRITE_STATES.indexOf(state);
  return index >= 0 ? index : -1;
}

/**
 * Project an enemy billboard from world space into screen coordinates.
 *
 * Uses the same inverse-camera-matrix transform as the existing wall/sprite
 * renderer. Sprites behind the camera plane, too close to the camera, or
 * produced by an invalid camera matrix are marked invisible.
 *
 * @see {@link https://lodev.org/cgtutor/raycasting2.html | Lode Vandevenne, Raycasting Tutorial Part 2}
 *
 * @param billboard - Enemy billboard descriptor.
 * @param camera - Current camera transform.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @returns The screen-space projection, including the span `[left, right]`.
 */
export function projectEnemyBillboardSprite(
  billboard: NeatensteinEnemyBillboard,
  camera: NeatensteinEnemyCamera,
  canvasWidth: number,
  canvasHeight: number,
): NeatensteinEnemyProjection {
  if (!isPositiveFinite(canvasWidth) || !isPositiveFinite(canvasHeight)) {
    return createInvisibleEnemyProjection();
  }

  const enemy = billboard.enemy;
  if (
    !Number.isFinite(enemy.position.x) ||
    !Number.isFinite(enemy.position.y) ||
    !Number.isFinite(camera.posX) ||
    !Number.isFinite(camera.posY) ||
    !Number.isFinite(camera.dirX) ||
    !Number.isFinite(camera.dirY) ||
    !Number.isFinite(camera.planeX) ||
    !Number.isFinite(camera.planeY)
  ) {
    return createInvisibleEnemyProjection();
  }

  const relativeX = enemy.position.x - camera.posX;
  const relativeY = enemy.position.y - camera.posY;

  const determinant = camera.planeX * camera.dirY - camera.dirX * camera.planeY;
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) < ENEMY_SPRITE_DETERMINANT_EPSILON
  ) {
    return createInvisibleEnemyProjection();
  }

  const invDet = 1 / determinant;

  const transformX =
    invDet * (camera.dirY * relativeX - camera.dirX * relativeY);
  const transformY =
    invDet * (-camera.planeY * relativeX + camera.planeX * relativeY);

  const perpDist = transformY;
  if (!isPositiveFinite(perpDist) || perpDist <= ENEMY_SPRITE_NEAR_CLIP) {
    return createInvisibleEnemyProjection(perpDist);
  }

  const screenX = (canvasWidth / 2) * (1 + transformX / transformY);
  const scale = Math.abs(canvasHeight / transformY) * ENEMY_SPRITE_WORLD_SIZE;
  const halfWidth = scale / 2;
  const left = screenX - halfWidth;
  const right = screenX + halfWidth;

  return {
    screenX,
    scale,
    perpDist,
    left,
    right,
    visible: true,
  };
}

/**
 * Clip a projected enemy billboard against the per-column z-buffer so only
 * sprite columns closer than the wall at the same screen column are drawn.
 *
 * Only sprite columns closer than the wall at the same screen column are kept.
 *
 * @param projection - Screen-space projection from
 *   {@link projectEnemyBillboardSprite}.
 * @param zBuffer - Per-column wall-depth buffer.
 * @returns The projection plus clamped span and visible column indices.
 */
export function clipEnemyBillboardSprite(
  projection: NeatensteinEnemyProjection,
  zBuffer: Readonly<Float32Array>,
): NeatensteinEnemyProjection & NeatensteinSpriteClip {
  if (!projection.visible || !Number.isFinite(projection.left)) {
    return {
      ...projection,
      left: 0,
      right: -1,
      visibleColumns: [],
    };
  }

  const clip = clipNeatensteinSpriteSpan(
    zBuffer,
    projection.left,
    projection.right,
    projection.perpDist,
  );

  return { ...projection, ...clip };
}

/**
 * Compute a directional light multiplier for an enemy billboard based on its
 * facing angle and the supplied directional light source.
 *
 * The enemy's facing vector acts as the surface normal. Light that aligns with
 * the normal is brightest; light from behind only receives ambient.
 *
 * @param yawRad - Enemy facing angle in radians.
 * @param light - Directional light source.
 * @returns Light multiplier in `[0, 1]`.
 */
export function computeDirectionalLightIntensity(
  yawRad: number,
  light: NeatensteinDirectionalLight,
): number {
  if (
    !Number.isFinite(yawRad) ||
    !Number.isFinite(light.dirX) ||
    !Number.isFinite(light.dirY) ||
    !Number.isFinite(light.intensity)
  ) {
    return ENEMY_SPRITE_AMBIENT_LIGHT;
  }

  const normalX = Math.cos(yawRad);
  const normalY = Math.sin(yawRad);
  const dot = normalX * light.dirX + normalY * light.dirY;
  const diffuse = Math.max(0, dot);
  const ambient =
    light.ambient !== undefined
      ? clamp01(light.ambient)
      : ENEMY_SPRITE_AMBIENT_LIGHT;

  return Math.min(ENEMY_SPRITE_MAX_LIGHT, ambient + diffuse * light.intensity);
}

/**
 * Apply a teal/orange bolt light tint to an RGB color.
 *
 * The effect is strongest at the start of the interval and fades linearly. A
 * high-frequency sine pulse is overlaid on top to give the energy shell a
 * "bolt" flicker.
 *
 * @param baseR - Base red channel in `[0, 255]`.
 * @param baseG - Base green channel in `[0, 255]`.
 * @param baseB - Base blue channel in `[0, 255]`.
 * @param elapsedMs - Milliseconds into the bolt interval.
 * @param durationMs - Total bolt interval in milliseconds.
 * @param boltColor - Packed 0xRRGGBB bolt color.
 * @returns Tinted RGB triplet.
 */
export function applyBoltLight(
  baseR: number,
  baseG: number,
  baseB: number,
  elapsedMs: number,
  durationMs: number,
  boltColor: number,
): { r: number; g: number; b: number } {
  const elapsed = Math.max(0, elapsedMs);
  if (
    !Number.isFinite(durationMs) ||
    durationMs <= 0 ||
    elapsed >= durationMs
  ) {
    return { r: baseR, g: baseG, b: baseB };
  }

  const fade = 1 - elapsed / durationMs;
  const pulse =
    0.5 + 0.5 * Math.sin((2 * Math.PI * elapsed) / ENEMY_BOLT_PULSE_PERIOD_MS);
  const mix = clamp01(fade * (0.4 + 0.6 * pulse));

  const bolt = unpackBoltRgb(boltColor);
  return {
    r: Math.round(baseR + (bolt.r - baseR) * mix),
    g: Math.round(baseG + (bolt.g - baseG) * mix),
    b: Math.round(baseB + (bolt.b - baseB) * mix),
  };
}

/**
 * Sample a single RGBA pixel from the sprite atlas, wrapping coordinates to the
 * current animation frame cell.
 *
 * The sample coordinate is wrapped to the frame cell; coordinates outside the
 * atlas return transparent black.
 *
 * @param atlas - Runtime sprite atlas.
 * @param direction - Yaw direction index.
 * @param stateIndex - Animation-state row index.
 * @param frameIndex - Animation frame index within the state.
 * @param u - Normalized horizontal sample coordinate in `[0, 1]`.
 * @param v - Normalized vertical sample coordinate in `[0, 1]`.
 * @returns RGBA sample in `[0, 255]` with alpha.
 */
export function sampleAtlasFramePixel(
  atlas: NeatensteinSpriteAtlas,
  direction: number,
  stateIndex: number,
  frameIndex: number,
  u: number,
  v: number,
): { r: number; g: number; b: number; a: number } {
  if (
    !isPositiveIntegerDimension(atlas.width) ||
    !isPositiveIntegerDimension(atlas.height) ||
    !isPositiveIntegerDimension(atlas.cellSize) ||
    atlas.data.length !==
      atlas.width * atlas.height * ENEMY_SPRITE_RGBA_CHANNELS
  ) {
    return { r: 0, g: 0, b: 0, a: 0 };
  }

  if (
    direction < 0 ||
    direction >= atlas.directions ||
    stateIndex < 0 ||
    stateIndex >= atlas.states ||
    frameIndex < 0 ||
    frameIndex * atlas.cellSize >= atlas.width
  ) {
    return { r: 0, g: 0, b: 0, a: 0 };
  }

  const safeU = clamp01(u);
  const safeV = clamp01(v);
  const cellSize = atlas.cellSize;

  const x = frameIndex * cellSize + Math.floor(safeU * (cellSize - 1));
  const y =
    (direction * atlas.states + stateIndex) * cellSize +
    Math.floor(safeV * (cellSize - 1));

  const offset = (y * atlas.width + x) * ENEMY_SPRITE_RGBA_CHANNELS;
  return {
    r: atlas.data[offset],
    g: atlas.data[offset + 1],
    b: atlas.data[offset + 2],
    a: atlas.data[offset + 3],
  };
}

/**
 * Resolve the framebuffer dimensions and verify they can be drawn into.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param framebufferWidth - Explicit framebuffer width.
 * @param framebufferHeight - Explicit framebuffer height.
 * @returns Verified width/height, or null when invalid.
 */
export function resolveFramebufferDimensions(
  framebuffer: Uint8ClampedArray,
  framebufferWidth: number,
  framebufferHeight: number,
): { width: number; height: number } | null {
  if (
    !isPositiveIntegerDimension(framebufferWidth) ||
    !isPositiveIntegerDimension(framebufferHeight) ||
    framebuffer.length <
      framebufferWidth * framebufferHeight * ENEMY_SPRITE_RGBA_CHANNELS
  ) {
    return null;
  }

  return { width: framebufferWidth, height: framebufferHeight };
}

/**
 * Pick the animation state index and frame for an enemy billboard.
 *
 * @param billboard - Enemy billboard descriptor.
 * @returns State row index and frame index, or null for unsupported states.
 */
export function resolveEnemyAnimationFrame(
  billboard: NeatensteinEnemyBillboard,
): { stateIndex: number; frameIndex: number } | null {
  const state = billboard.enemy.animationState;
  const stateIndex = animationStateToAtlasIndex(state);
  if (stateIndex < 0) {
    return null;
  }

  const elapsedMs =
    state === 'death'
      ? billboard.enemy.deRezElapsedMs
      : billboard.animationElapsedMs;

  const frame = getEnemyAnimationFrame(state, elapsedMs);

  return { stateIndex, frameIndex: frame.frameIndex };
}
