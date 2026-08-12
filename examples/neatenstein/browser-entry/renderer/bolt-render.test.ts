import { describe, expect, it, jest } from '@jest/globals';

import {
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
} from '../host/game/constants';
import type {
  BoltState,
  EnemyBoltState,
  EnemyImpactSpot,
  ImpactSpot,
} from '../host/game/types';
import { NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS } from '../host/game/constants';
import {
  NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS,
  NEATENSTEIN_ENEMY_IMPACT_COLOR,
  NEATENSTEIN_ENEMY_IMPACT_GLOW_COLOR,
  NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
} from '../constants';
import {
  drawBolts,
  drawEnemyBolts,
  drawEnemyImpactSpots,
  drawImpactSpots,
} from './bolt-render';
import type { NeatensteinFloorCamera } from './floor';

/**
 * Build a lightweight mock 2D canvas context for the bolt renderer tests.
 */
function createMockContext(): OffscreenCanvasRenderingContext2D & {
  canvas: { width: number; height: number };
  globalAlphaHistory: number[];
} {
  const canvas = { width: 640, height: 360 };
  const target: Record<string, unknown> = { canvas };
  const propertyValues: Record<string, unknown> = {};
  const globalAlphaHistory: number[] = [];

  for (const property of [
    'fillStyle',
    'globalCompositeOperation',
    'shadowColor',
    'shadowBlur',
    'strokeStyle',
    'lineWidth',
  ]) {
    Object.defineProperty(target, property, {
      get: () => propertyValues[property],
      set: (value) => {
        propertyValues[property] = value;
      },
      configurable: true,
      enumerable: true,
    });
  }

  Object.defineProperty(target, 'globalAlpha', {
    get: () => propertyValues.globalAlpha,
    set: (value) => {
      propertyValues.globalAlpha = value;
      if (typeof value === 'number') {
        globalAlphaHistory.push(value);
      }
    },
    configurable: true,
    enumerable: true,
  });

  for (const method of [
    'fillRect',
    'beginPath',
    'arc',
    'moveTo',
    'lineTo',
    'stroke',
    'fill',
    'save',
    'restore',
    'translate',
    'closePath',
    'ellipse',
  ]) {
    target[method] = jest.fn();
  }

  target.createRadialGradient = jest.fn(() => ({ addColorStop: jest.fn() }));
  target.createLinearGradient = jest.fn(() => ({ addColorStop: jest.fn() }));

  target.globalAlphaHistory = globalAlphaHistory;

  return target as unknown as OffscreenCanvasRenderingContext2D & {
    canvas: { width: number; height: number };
    globalAlphaHistory: number[];
  };
}

function createBolt(overrides?: Partial<BoltState>): BoltState {
  return {
    active: true,
    position: { x: 10, y: 0 },
    direction: { x: 1, y: 0 },
    speedCellsPerSecond: 36,
    createdAtMs: 0,
    origin: { x: 0, y: 0 },
    targetDistance: 10,
    ...overrides,
  };
}

function createCamera(): NeatensteinFloorCamera {
  return { x: 0, y: 0, yaw: 0 };
}

function createImpact(overrides?: Partial<ImpactSpot>): ImpactSpot {
  return {
    wallHit: { mapX: 10, mapY: 10, side: 0, wallX: 0.5 },
    position: { x: 11, y: 11 },
    createdAtMs: 0,
    lifetimeMs: 1000,
    perpWallDist: 10,
    boltTravelTimeMs: 100,
    ...overrides,
  };
}

function createEnemyBolt(overrides?: Partial<EnemyBoltState>): EnemyBoltState {
  return {
    position: { x: 10, y: 0 },
    direction: { x: 1, y: 0 },
    speedCellsPerSecond: 36,
    active: true,
    createdAtMs: 0,
    origin: { x: 0, y: 0 },
    damage: 10,
    ...overrides,
  };
}

/**
 * Build a lightweight enemy-impact spot for the drawEnemyImpactSpots tests.
 */
function createEnemyImpact(
  overrides?: Partial<EnemyImpactSpot>,
): EnemyImpactSpot {
  return {
    position: { x: 10, y: 0 },
    createdAtMs: 0,
    lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
    boltTravelTimeMs: 0,
    ...overrides,
  };
}

/**
 * Build a lightweight ammo pickup for the drawAmmoPickups tests.
 */
function createAmmoPickup(
  overrides?: Partial<{
    position: { x: number; y: number };
    amount: number;
    active: boolean;
    createdAtMs: number;
  }>,
): {
  position: { x: number; y: number };
  amount: number;
  active: boolean;
  createdAtMs: number;
} {
  return {
    position: { x: 10, y: 0 },
    amount: 5,
    active: true,
    createdAtMs: 0,
    ...overrides,
  };
}

describe('bolt-render', () => {
  describe('drawBolts', () => {
    it('returns early for an empty bolts array without drawing', () => {
      const context = createMockContext();

      drawBolts(context, [], createCamera(), 640, 360, 0);

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.stroke).not.toHaveBeenCalled();
    });

    it('does not stroke the bolt path', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt()],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.stroke).not.toHaveBeenCalled();
    });

    it('does not draw line segments for the bolt', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt()],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.lineTo).not.toHaveBeenCalled();
    });

    it('draws a bolt circle', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt()],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.arc).toHaveBeenCalled();
    });

    it('fills the bolt circle', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt()],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.fill).toHaveBeenCalled();
    });

    it('starts the bolt at the muzzle screen point at createdAtMs', () => {
      const context = createMockContext();

      drawBolts(context, [createBolt()], createCamera(), 640, 360, 0);

      const calls = (context.arc as jest.Mock).mock.calls as number[][];
      expect({ x: calls[0][0], y: calls[0][1] }).toEqual({
        x: 640 * 0.5,
        y: 360 * 0.82,
      });
    });

    it('reaches the projected target at the end of the travel duration', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt()],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
      );

      const calls = (context.arc as jest.Mock).mock.calls as number[][];
      expect(calls[0][1]).toBe(360 * 0.5);
    });

    it('is halfway to the projected target at the midpoint of the travel duration', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt()],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      const calls = (context.arc as jest.Mock).mock.calls as number[][];
      const expectedY = (360 * 0.82 + 360 * 0.5) / 2;
      expect(calls[0][1]).toBe(expectedY);
    });

    it('draws a wider bolt at the muzzle than the previous 1.5 px radius', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt({ position: { x: 10, y: 0 }, origin: { x: 10, y: 0 } })],
        createCamera(),
        640,
        360,
        0,
      );

      const calls = (context.arc as jest.Mock).mock.calls as number[][];
      expect(calls[0][2]).toBeGreaterThan(1.5);
    });

    it('shrinks the bolt radius as it travels away from the muzzle', () => {
      const nearContext = createMockContext();
      drawBolts(
        nearContext,
        [createBolt({ position: { x: 1, y: 0 }, targetDistance: 1 })],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      const farContext = createMockContext();
      drawBolts(
        farContext,
        [createBolt({ position: { x: 15, y: 0 }, targetDistance: 15 })],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      const nearCalls = (nearContext.arc as jest.Mock).mock.calls as number[][];
      const farCalls = (farContext.arc as jest.Mock).mock.calls as number[][];
      expect(nearCalls[0][2]).toBeGreaterThan(farCalls[0][2]);
    });

    it('fades the bolt alpha as it travels away from the muzzle', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt({ position: { x: 5, y: 0 }, targetDistance: 5 })],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      const minAlpha = Math.min(...context.globalAlphaHistory);
      expect(minAlpha).toBeLessThan(0.95);
    });

    it('does not draw a bolt that has reached max range', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            position: { x: NEATENSTEIN_BOLT_MAX_RANGE_CELLS, y: 0 },
            targetDistance: NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips a bolt whose elapsed time is outside the travel window', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt({ createdAtMs: 0 })],
        createCamera(),
        640,
        360,
        -1,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips a bolt that projects behind the camera', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [createBolt({ position: { x: -1, y: 0 }, origin: { x: -1, y: 0 } })],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('draws a bolt that lacks origin, direction and targetDistance', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            origin: undefined,
            direction: undefined,
            targetDistance: undefined,
            position: { x: 5, y: 0 },
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.arc).toHaveBeenCalled();
    });

    it('falls back to distanceTraveled when origin and targetDistance are missing', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            origin: undefined,
            targetDistance: undefined,
            position: { x: 5, y: 0 },
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.arc).toHaveBeenCalled();
    });

    it('falls back to the current position when the projected target is behind the camera', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            origin: { x: 1, y: 0 },
            direction: { x: -1, y: 0 },
            targetDistance: 2,
            position: { x: 0.5, y: 0 },
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      expect(context.arc).toHaveBeenCalled();
    });
  });

  describe('drawImpactSpots', () => {
    it('returns early for an empty impacts array without drawing', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawImpactSpots(context, [], zBuffer, createCamera(), 640, 360, 0);

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips rendering before travelRatio reaches 1.0', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        640,
        360,
        50, // elapsed 50ms < 100ms travel time
      );

      expect(context.fill).not.toHaveBeenCalled();
    });

    it('renders once travelRatio reaches 1.0', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        640,
        360,
        150, // elapsed 150ms >= 100ms travel time
      );

      expect(context.fill).toHaveBeenCalled();
    });

    it('does not draw an impact spot exactly one millisecond before arrival', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        640,
        360,
        99,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('draws an impact spot arc exactly at arrival', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        640,
        360,
        100,
      );

      expect(context.arc).toHaveBeenCalled();
    });

    it('skips an impact spot behind the camera', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 12, y: 12, yaw: Math.PI / 4 },
        640,
        360,
        100,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips an impact spot occluded by a closer wall', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(1);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        640,
        360,
        100,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips an impact spot with a non-finite screen X', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact();

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        Number.POSITIVE_INFINITY,
        360,
        100,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('renders an impact spot immediately when bolt travel time is zero', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      const impact = createImpact({ boltTravelTimeMs: 0 });

      drawImpactSpots(
        context,
        [impact],
        zBuffer,
        { x: 9, y: 9, yaw: Math.PI / 4 },
        640,
        360,
        0,
      );

      expect(context.arc).toHaveBeenCalled();
    });
  });

  describe('AC-404R: bolt stays visible for the full travel duration on close walls', () => {
    it('draws a close-wall plasma bolt that was deactivated by a wall hit before the visual travel duration expires', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            active: false,
            position: { x: 0.5, y: 0 },
            origin: { x: 0, y: 0 },
            direction: { x: 1, y: 0 },
            targetDistance: 0.5,
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS - 1,
      );

      expect(context.arc).toHaveBeenCalled();
    });
  });

  describe('drawEnemyBolts', () => {
    it('returns early for an empty bolt array without drawing', () => {
      const context = createMockContext();

      drawEnemyBolts(context, [], createCamera(), 640, 360, 0);

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips a bolt whose lifetime has expired (elapsedMs > lifetime)', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS + 100,
      );

      // Bolt is past lifetime → continue (no drawing)
      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips a bolt with negative elapsedMs', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            createdAtMs: 5000,
          }),
        ],
        createCamera(),
        640,
        360,
        1000,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips a bolt whose projection returns null (behind camera)', () => {
      const context = createMockContext();

      // Camera at {0,0,0}; bolt at negative X is behind camera (yaw=0 → forward is +X)
      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: -10, y: 0 },
            origin: { x: -10, y: 0 },
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        100,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips a bolt whose fadeRatio >= 1 (fully faded by range)', () => {
      const context = createMockContext();

      // origin far from current position → distanceTraveled > 30 → rangeRatio = 1 → fadeRatio >= 1
      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 50, y: 0 },
            origin: { x: 0, y: 0 },
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        100, // elapsedMs = 100, lifetimeRatio = 0.05, rangeRatio = 50/30 > 1 → fadeRatio >= 1
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips a bolt whose fadeRatio >= 1 (fully faded by lifetime)', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 10, y: 0 },
            origin: { x: 10, y: 0 },
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_ENEMY_BOLT_LIFETIME_MS, // elapsedMs = 2000, lifetimeRatio = 1 → fadeRatio >= 1
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('draws a normal enemy bolt with arc and fill', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 10, y: 0 },
            origin: { x: 0, y: 0 },
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        100, // elapsedMs = 100, lifetimeRatio = 0.05, rangeRatio = 10/30 ≈ 0.33 → fadeRatio < 1
      );

      expect(context.arc).toHaveBeenCalled();
      expect(context.fill).toHaveBeenCalled();
    });

    it('does not stroke the enemy bolt path', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [createEnemyBolt()],
        createCamera(),
        640,
        360,
        100,
      );

      expect(context.stroke).not.toHaveBeenCalled();
    });

    it('draws an explosion flash when bolt.hitPlayer is true', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 10, y: 0 },
            origin: { x: 0, y: 0 },
            createdAtMs: 0,
            hitPlayer: true,
          }),
        ],
        createCamera(),
        640,
        360,
        100,
      );

      // Explosion flash adds extra arc+fill calls (outer glow + core + flash = 3 arcs)
      const arcCalls = (context.arc as jest.Mock).mock.calls;
      expect(arcCalls.length).toBeGreaterThanOrEqual(3);
      const fillCalls = (context.fill as jest.Mock).mock.calls;
      expect(fillCalls.length).toBeGreaterThanOrEqual(3);
    });

    it('does not draw explosion flash when bolt.hitPlayer is false', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 10, y: 0 },
            origin: { x: 0, y: 0 },
            createdAtMs: 0,
            hitPlayer: false,
          }),
        ],
        createCamera(),
        640,
        360,
        100,
      );

      // Only outer glow + inner core = 2 arcs (no flash)
      const arcCalls = (context.arc as jest.Mock).mock.calls;
      expect(arcCalls.length).toBe(2);
    });

    it('uses the projected current position when bolt has no origin', () => {
      const context = createMockContext();

      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 10, y: 0 },
            origin: undefined,
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        100,
      );

      // Should still draw — origin defaults to projectedCurrent
      expect(context.arc).toHaveBeenCalled();
    });

    it('falls back to projectedCurrent when projectedOrigin is null', () => {
      const context = createMockContext();

      // Camera at {0,0,0}; origin behind camera (negative X) but current position in front
      drawEnemyBolts(
        context,
        [
          createEnemyBolt({
            position: { x: 10, y: 0 },
            origin: { x: -10, y: 0 }, // behind camera → projectedOrigin = null
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        100,
      );

      // Should still draw since projectedCurrent is valid and origin falls back
      expect(context.arc).toHaveBeenCalled();
    });

    it('covers hitEnemyIndex undefined branch (line 261 false path)', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            hitEnemyIndex: undefined,
            origin: { x: 0, y: 0 },
            direction: { x: 1, y: 0 },
            targetDistance: 10,
            position: { x: 5, y: 0 },
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      // hitEnemyIndex is not a number → falls to bolt.targetDistance
      expect(context.arc).toHaveBeenCalled();
    });

    it('covers hitEnemyIndex numeric branch (line 261 true path)', () => {
      const context = createMockContext();

      drawBolts(
        context,
        [
          createBolt({
            hitEnemyIndex: 0,
            radius: 0.5,
            origin: { x: 0, y: 0 },
            direction: { x: 1, y: 0 },
            targetDistance: 10,
            position: { x: 5, y: 0 },
            createdAtMs: 0,
          }),
        ],
        createCamera(),
        640,
        360,
        NEATENSTEIN_BOLT_TRAVEL_DURATION_MS / 2,
      );

      // hitEnemyIndex is a number ≥ 0, radius > 0 → uses Math.hypot
      expect(context.arc).toHaveBeenCalled();
    });
  });

  describe('drawEnemyImpactSpots', () => {
    it('returns early for an empty impact array without drawing', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(context, [], zBuffer, createCamera(), 640, 360, 0);

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('sets additive blend (globalCompositeOperation to lighter)', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Track composite operation values set during the call
      const compositeValues: unknown[] = [];
      let _composite: unknown = 'source-over';
      Object.defineProperty(context, 'globalCompositeOperation', {
        get: () => _composite,
        set: (v: unknown) => {
          compositeValues.push(v);
          _composite = v;
        },
        configurable: true,
        enumerable: true,
      });

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(compositeValues).toContain('lighter');
    });

    it('skips rendering when travelRatio < 1 (bolt has not arrived yet)', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            createdAtMs: 0,
            boltTravelTimeMs: 300,
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        50, // elapsed 50ms < 300ms travel time → travelRatio < 1
      );

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips an impact spot whose projection returns null (behind camera)', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            position: { x: -10, y: 0 }, // behind camera at yaw 0
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips an impact spot with perpDist <= 0', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Camera at {0, 0, yaw: π} faces -X; impact at +X → perpDist < 0
      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            position: { x: 10, y: 0 },
          }),
        ],
        zBuffer,
        { x: 0, y: 0, yaw: Math.PI },
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips an impact spot whose perpDist is NaN (Number.isFinite guard)', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Position with NaN coordinate → perpDist = NaN → !Number.isFinite triggers
      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            position: { x: Number.NaN, y: 0 },
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('skips an impact spot occluded by a closer wall (depth test fail)', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(1); // wall at distance 1

      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            position: { x: 10, y: 0 }, // perpDist = 10 > 1 → occluded
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
    });

    it('renders the main mark with arc and fill when the bolt has arrived', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).toHaveBeenCalled();
      expect(context.fill).toHaveBeenCalled();
    });

    it('sets shadowColor to the enemy impact glow color', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.shadowColor).toBe(NEATENSTEIN_ENEMY_IMPACT_GLOW_COLOR);
    });

    it('sets fillStyle to the enemy impact color', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.fillStyle).toBe(NEATENSTEIN_ENEMY_IMPACT_COLOR);
    });

    it('fades alpha based on remaining lifetime', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Full lifetime → alpha = 1.0
      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      const fullAlpha = context.globalAlpha;
      expect(fullAlpha).toBe(1);

      // alpha for full lifetime should be 1.0 (it's in the history before burst/cleanup)
      expect(context.globalAlphaHistory).toContain(1);

      // Half lifetime → alpha = 0.5
      const context2 = createMockContext();
      drawEnemyImpactSpots(
        context2,
        [
          createEnemyImpact({
            lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS / 2,
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      // The main mark alpha = 0.5 should be in the history
      expect(context2.globalAlphaHistory).toContain(0.5);
    });

    it('draws the burst effect within burst duration after arrival', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // boltTravelTimeMs=0 → visible immediately; simTimeMs=50 < burstDuration 200
      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            createdAtMs: 0,
            boltTravelTimeMs: 0,
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        50, // visibleElapsedMs = 50 < BURST_DURATION_MS (200)
      );

      // Main mark + burst = at least 2 arc calls
      const arcCalls = (context.arc as jest.Mock).mock.calls;
      expect(arcCalls.length).toBeGreaterThanOrEqual(2);
      const fillCalls = (context.fill as jest.Mock).mock.calls;
      expect(fillCalls.length).toBeGreaterThanOrEqual(2);
    });

    it('does not draw the burst effect after burst duration has elapsed', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [
          createEnemyImpact({
            createdAtMs: 0,
            boltTravelTimeMs: 0,
            lifetimeMs: NEATENSTEIN_ENEMY_IMPACT_LIFETIME_MS,
          }),
        ],
        zBuffer,
        createCamera(),
        640,
        360,
        NEATENSTEIN_ENEMY_IMPACT_BURST_DURATION_MS + 100, // past burst
      );

      // Only main mark = 1 arc, 1 fill (no burst)
      const arcCalls = (context.arc as jest.Mock).mock.calls;
      expect(arcCalls.length).toBe(1);
      const fillCalls = (context.fill as jest.Mock).mock.calls;
      expect(fillCalls.length).toBe(1);
    });

    it('restores globalAlpha to 1 after rendering', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.globalAlpha).toBe(1);
    });

    it('restores shadowBlur to 0 after rendering', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.shadowBlur).toBe(0);
    });

    it('restores globalCompositeOperation to the saved value after rendering', () => {
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Set initial composite to source-over (default)
      context.globalCompositeOperation = 'source-over';

      drawEnemyImpactSpots(
        context,
        [createEnemyImpact()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.globalCompositeOperation).toBe('source-over');
    });
  });

  describe('drawAmmoPickups', () => {
    it('exports drawAmmoPickups', async () => {
      const mod = (await import('./bolt-render')) as Record<string, unknown>;
      expect(typeof mod.drawAmmoPickups).toBe('function');
    });

    it('sets globalCompositeOperation to lighter for additive blending', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Track composite operation values set during the call (same approach
      // as the drawEnemyImpactSpots test — the function uses save/restore so
      // the final value is 'source-over', not 'lighter').
      const compositeValues: unknown[] = [];
      let _composite: unknown = 'source-over';
      Object.defineProperty(context, 'globalCompositeOperation', {
        get: () => _composite,
        set: (v: unknown) => {
          compositeValues.push(v);
          _composite = v;
        },
        configurable: true,
        enumerable: true,
      });

      drawAmmoPickups(
        context,
        [createAmmoPickup()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(compositeValues).toContain('lighter');
    });

    it('draws an arc for an active ammo pickup', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawAmmoPickups(
        context,
        [createAmmoPickup()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.beginPath).toHaveBeenCalled();
      expect(context.arc).toHaveBeenCalled();
      expect(context.fill).toHaveBeenCalled();
    });

    it('restores globalCompositeOperation after rendering', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);
      context.globalCompositeOperation = 'source-over';

      drawAmmoPickups(
        context,
        [createAmmoPickup()],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.globalCompositeOperation).toBe('source-over');
    });

    it('does not draw inactive ammo pickups', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      drawAmmoPickups(
        context,
        [createAmmoPickup({ active: false })],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips pickups behind the camera (perpDist <= 0)', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Camera at origin facing yaw=0 (dirX=1, dirY=0).  A pickup at x=-10
      // produces perpDist = -10 <= 0, hitting the `continue` at line 246.
      drawAmmoPickups(
        context,
        [createAmmoPickup({ position: { x: -10, y: 0 } })],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips pickups that project to a non-finite screen position', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Camera at origin facing yaw=0.  A pickup at x=1, y=Number.MAX_VALUE
      // produces a finite positive perpDist (=1) but an extremely large
      // lateral (=Number.MAX_VALUE) that overflows screenX to Infinity,
      // hitting the `continue` at line 254.
      drawAmmoPickups(
        context,
        [createAmmoPickup({ position: { x: 1, y: Number.MAX_VALUE } })],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });

    it('skips pickups that fail the depth test (hidden behind wall)', async () => {
      const { drawAmmoPickups } = (await import('./bolt-render')) as Record<
        string,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- dynamic import test helper
        any
      >;
      const context = createMockContext();
      const zBuffer = new Float32Array(640).fill(Number.POSITIVE_INFINITY);

      // Camera at origin facing yaw=0.  Pickup at (10, 0) produces:
      //   perpDist = 10, lateral = 0, screenX = 320, screenColumn = 320.
      // Set zBuffer[320] = 5 (< 10) so depthTestPulse returns false,
      // hitting the `continue` at line 260.
      zBuffer[320] = 5;

      drawAmmoPickups(
        context,
        [createAmmoPickup({ position: { x: 10, y: 0 } })],
        zBuffer,
        createCamera(),
        640,
        360,
        0,
      );

      expect(context.arc).not.toHaveBeenCalled();
      expect(context.fill).not.toHaveBeenCalled();
    });
  });
});
