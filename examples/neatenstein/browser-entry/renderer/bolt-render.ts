/**
 * Bolt and impact-spot rendering for the Neatenstein worker tier.
 *
 * This module is the public facade for bolt rendering. The per-item render
 * executors, projection-context resolvers, and shared drawing helpers live in
 * {@link module:./bolt.utils}, while this file holds the thin declarative
 * orchestrators that set up additive blending, iterate items, and restore
 * context state.
 *
 * @module
 */

import type {
  AmmoPickupState,
  BoltState,
  EnemyBoltState,
  EnemyImpactSpot,
  ImpactSpot,
} from '../host/game/types';
import type { NeatensteinFloorCamera } from './renderer.floor.types';
import { COMPOSITE_OP_LIGHTER } from './renderer.bolt.constants';
import {
  renderAmmoPickup,
  renderEnemyBolt,
  renderEnemyImpactSpot,
  renderImpactSpot,
  renderPlayerBolt,
  resolveFloorProjectionContext,
  resolveLateralProjectionContext,
  resolveMuzzleScreenPosition,
  restoreRenderContext,
} from './bolt.utils';

// Re-export constants and types for external consumers.
export { COMPOSITE_OP_LIGHTER } from './renderer.bolt.constants';
export type {
  LateralProjectionContext,
  FloorProjectionContext,
  MuzzleScreenPosition,
} from './renderer.bolt.types';

/**
 * Draw active wall-impact neon spots in the worker tier.
 *
 * Each spot is rendered only after the plasma bolt that created it has reached
 * the wall (`travelRatio >= 1`). Until then the spot stays invisible so the
 * impact does not appear to happen before the projectile arrives.
 *
 * Declarative pipeline: resolve lateral projection context → set additive
 * blend → render each spot via {@link renderImpactSpot} → restore context.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impacts - Active wall-impact list.
 * @param zBuffer - Per-column depth buffer.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawImpactSpots(
  context: OffscreenCanvasRenderingContext2D,
  impacts: readonly ImpactSpot[],
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
): void {
  if (impacts.length === 0) {
    return;
  }

  const latCtx = resolveLateralProjectionContext(
    camera,
    canvasWidth,
    canvasHeight,
  );
  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = COMPOSITE_OP_LIGHTER;

  for (const impact of impacts) {
    renderImpactSpot(context, impact, zBuffer, latCtx, simTimeMs);
  }

  restoreRenderContext(context, savedComposite);
}

/**
 * Draw active ammo pickups as shiny squares with additive blending.
 *
 * Each pickup is rendered as a white-core square with a cool-white halo,
 * projected to screen space using the same floor-plane projection as
 * {@link drawImpactSpots}. Only active pickups within the camera frustum
 * and passing the z-buffer depth test are drawn.
 *
 * Paint order: after enemy impacts, before bolts.
 *
 * Declarative pipeline: filter active pickups → resolve lateral projection
 * context → set additive blend → render each pickup via
 * {@link renderAmmoPickup} → restore context.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param pickups - Active ammo pickup list.
 * @param zBuffer - Per-column depth buffer.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawAmmoPickups(
  context: OffscreenCanvasRenderingContext2D,
  pickups: readonly AmmoPickupState[],
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
): void {
  void simTimeMs;
  const activePickups = pickups.filter((pickup) => pickup.active);
  if (activePickups.length === 0) {
    return;
  }

  const latCtx = resolveLateralProjectionContext(
    camera,
    canvasWidth,
    canvasHeight,
  );
  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = COMPOSITE_OP_LIGHTER;

  for (const pickup of activePickups) {
    renderAmmoPickup(context, pickup, zBuffer, latCtx);
  }

  restoreRenderContext(context, savedComposite);
}

/**
 * Draw active enemy-impact neon spots in the worker tier.
 *
 * Each spot is rendered only after the plasma bolt that created it has reached
 * the enemy (`travelRatio >= 1`). The persistent mark uses additive-blend neon
 * glow with distance-scaling, and a brief expanding burst effect plays during
 * the first {@link NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS} after arrival.
 *
 * Paint order: after sprites, before bolts.
 *
 * Declarative pipeline: resolve floor projection context → set additive blend
 * → render each spot via {@link renderEnemyImpactSpot} → restore context.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param impacts - Active enemy-impact list.
 * @param zBuffer - Per-column depth buffer.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawEnemyImpactSpots(
  context: OffscreenCanvasRenderingContext2D,
  impacts: readonly EnemyImpactSpot[],
  zBuffer: Float32Array,
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
): void {
  if (impacts.length === 0) {
    return;
  }

  const floorCtx = resolveFloorProjectionContext(
    camera,
    canvasWidth,
    canvasHeight,
  );
  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = COMPOSITE_OP_LIGHTER;

  for (const impact of impacts) {
    renderEnemyImpactSpot(context, impact, zBuffer, floorCtx, simTimeMs);
  }

  restoreRenderContext(context, savedComposite);
}

/**
 * Draw active plasma bolts as traveling glowing projectiles.
 *
 * Each bolt is interpolated on screen from the gun muzzle toward its projected
 * impact point. The visual travel duration is fixed so every bolt moves at the
 * same screen-space speed regardless of target distance.
 *
 * Declarative pipeline: resolve floor projection context → resolve muzzle
 * screen position → set additive blend → render each bolt via
 * {@link renderPlayerBolt} → restore context.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param bolts - Active bolt list.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawBolts(
  context: OffscreenCanvasRenderingContext2D,
  bolts: readonly BoltState[],
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
): void {
  if (bolts.length === 0) {
    return;
  }

  const floorCtx = resolveFloorProjectionContext(
    camera,
    canvasWidth,
    canvasHeight,
  );
  const muzzle = resolveMuzzleScreenPosition(canvasWidth, canvasHeight);
  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = COMPOSITE_OP_LIGHTER;

  for (const bolt of bolts) {
    renderPlayerBolt(context, bolt, floorCtx, muzzle, simTimeMs);
  }

  restoreRenderContext(context, savedComposite);
}

/**
 * Draw active enemy plasma bolts as traveling glowing projectiles.
 *
 * Enemy bolts are rendered with a red/orange color palette to distinguish them
 * visually from the player's teal plasma bolts. Each bolt is projected from
 * its world-space position to screen space, interpolated from its origin
 * toward its current position, and faded based on travel distance.
 *
 * Declarative pipeline: resolve floor projection context → set additive blend
 * → render each bolt via {@link renderEnemyBolt} → restore context.
 *
 * @param context - Worker-tier 2D canvas context.
 * @param bolts - Active enemy bolt list.
 * @param camera - Camera position and yaw.
 * @param canvasWidth - Canvas width.
 * @param canvasHeight - Canvas height.
 * @param simTimeMs - Current simulation time in milliseconds.
 */
export function drawEnemyBolts(
  context: OffscreenCanvasRenderingContext2D,
  bolts: readonly EnemyBoltState[],
  camera: NeatensteinFloorCamera,
  canvasWidth: number,
  canvasHeight: number,
  simTimeMs: number,
): void {
  if (bolts.length === 0) {
    return;
  }

  const floorCtx = resolveFloorProjectionContext(
    camera,
    canvasWidth,
    canvasHeight,
  );
  const savedComposite = context.globalCompositeOperation;
  context.globalCompositeOperation = COMPOSITE_OP_LIGHTER;

  for (const bolt of bolts) {
    renderEnemyBolt(context, bolt, floorCtx, simTimeMs);
  }

  restoreRenderContext(context, savedComposite);
}
