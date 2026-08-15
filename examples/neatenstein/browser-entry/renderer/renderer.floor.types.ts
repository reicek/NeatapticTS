/**
 * Floor rendering type definitions extracted from the floor renderer modules.
 *
 * Centralises the interfaces and type aliases consumed by
 * {@link module:./floor}, {@link module:./floor.band.utils},
 * {@link module:./floor.projection.utils}, and
 * {@link module:./floor.shade.utils}.
 *
 * @module
 */

/**
 * Camera state consumed by the floor and ceiling renderer.
 */
export interface NeatensteinFloorCamera {
  /** Camera yaw in radians. `0` looks down the world +X axis. */
  yaw: number;
  /** Camera world X position. */
  x: number;
  /** Camera world Y position. */
  y: number;
}

/**
 * Minimal canvas-like rendering context consumed by the grid renderer.
 *
 * The interface is intentionally narrow so tests can provide small mocks
 * instead of implementing the full `CanvasRenderingContext2D` API.
 */
export interface NeatensteinFloorRenderContext {
  /** Start a new path. */
  beginPath(): void;
  /** Move the path cursor to `(x, y)`. */
  moveTo(x: number, y: number): void;
  /** Add a line segment to `(x, y)`. */
  lineTo(x: number, y: number): void;
  /** Stroke the current path. */
  stroke(): void;
  /** Snapshot current context state. */
  save(): void;
  /** Restore the most recently saved context state. */
  restore(): void;
  /** Optional backing canvas dimensions. */
  canvas?: { width: number; height: number };
  /** Optional stroke color. */
  strokeStyle?: string | CanvasGradient | CanvasPattern;
  /** Optional line width in pixels. */
  lineWidth?: number;
  /** Optional glow blur radius in pixels. */
  shadowBlur?: number;
  /** Optional glow color. */
  shadowColor?: string | CanvasGradient | CanvasPattern;
}

/**
 * Flat screen-space segment buffer.
 *
 * Values are stored in groups of four:
 *
 * ```ts
 * [x1, y1, x2, y2, x1, y1, x2, y2, ...]
 * ```
 *
 * This avoids allocating one tuple/object per projected line segment.
 */
export type NeatensteinFloorSegmentBuffer = number[];

/**
 * Sanitized camera values used by the projection hot path.
 */
export interface SafeNeatensteinFloorCamera {
  /** Finite camera yaw in radians. */
  yaw: number;
  /** Finite camera world X coordinate. */
  x: number;
  /** Finite camera world Y coordinate. */
  y: number;
}

/**
 * Shared projection constants for a single grid draw call.
 */
export interface NeatensteinGridProjectionContext {
  /** Canvas width in backing-store pixels. */
  width: number;
  /** Canvas height in backing-store pixels. */
  height: number;
  /** Camera world X coordinate. */
  cameraX: number;
  /** Camera world Y coordinate. */
  cameraY: number;
  /** Cosine of camera yaw. */
  cosYaw: number;
  /** Sine of camera yaw. */
  sinYaw: number;
  /** Perspective focal length in pixels. */
  focalLength: number;
  /** Half canvas width in pixels. */
  halfWidth: number;
  /** Horizon Y coordinate in pixels. */
  horizonY: number;
  /** Camera height above the floor in world units. */
  cameraHeight: number;
}

/**
 * Projected screen-space point used while building line segments.
 */
export interface ProjectedNeatensteinGridPoint {
  /** Screen-space X coordinate. */
  x: number;
  /** Screen-space Y coordinate. */
  y: number;
  /** Normalized depth ratio, where `0` is far and `1` is near. */
  depthRatio: number;
  /** Positive camera-space forward distance. */
  distance: number;
}
