import { describe, expect, it, jest } from '@jest/globals';

import {
  NEATENSTEIN_BOLT_MAX_RANGE_CELLS,
  NEATENSTEIN_BOLT_TRAVEL_DURATION_MS,
} from '../host/game/constants';
import type { BoltState, ImpactSpot } from '../host/game/types';
import { drawBolts, drawImpactSpots } from './bolt-render';
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

describe('bolt-render', () => {
  describe('drawBolts', () => {
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
});
