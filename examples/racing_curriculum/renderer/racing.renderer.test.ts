import {
  computeWorldTransform,
  createRacingRenderState,
  renderRacingFrame,
} from './racing.renderer';
import { generateTrack } from '../track/track.generator';
import type { EnvironmentState } from '../environment/environment.types';
import type { TrackSpec } from '../track/track.generator.types';

describe('racing renderer sibling seam', () => {
  describe('computeWorldTransform', () => {
    it('returns a positive scale for a generated track', () => {
      const canvasElement = {
        width: 960,
        height: 540,
      } as HTMLCanvasElement;
      const trackSpec = generateTrack({
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'medium',
      });
      const worldTransform = computeWorldTransform(canvasElement, trackSpec);

      expect(worldTransform.scale).toBeGreaterThan(0);
    });

    it('applies larger edge padding as a smaller world scale', () => {
      const canvasElement = {
        width: 960,
        height: 540,
      } as HTMLCanvasElement;
      const trackSpec = generateTrack({
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'medium',
      });
      const compactPaddingTransform = computeWorldTransform(
        canvasElement,
        trackSpec,
        { edgePaddingPx: 24 },
      );
      const widePaddingTransform = computeWorldTransform(
        canvasElement,
        trackSpec,
        {
          edgePaddingPx: 72,
        },
      );

      expect(widePaddingTransform.scale).toBeLessThan(
        compactPaddingTransform.scale,
      );
    });
  });

  describe('renderRacingFrame', () => {
    it('uses the Team B pit color for car outlines when pit visuals are enabled', () => {
      const trackSpec = createTrackSpecWithoutPits();
      const { canvasElement, strokeStyleAssignments } =
        createMockCanvasAndStrokeRecorder();
      const envState = createRendererEnvironmentState(trackSpec);

      renderRacingFrame(
        canvasElement,
        trackSpec,
        envState,
        createRacingRenderState(),
        computeWorldTransform(canvasElement, trackSpec),
        {
          frame: {
            featureFlags: 0b100,
            carTeam: Uint8Array.from([1]),
            tireState: new Float32Array([1, 1, 1, 1]),
          },
        },
      );

      expect(strokeStyleAssignments.includes('rgba(255, 122, 69, 0.38)')).toBe(
        true,
      );
    });

    it('keeps the default car outline color when pit visuals are disabled', () => {
      const trackSpec = createTrackSpecWithoutPits();
      const { canvasElement, strokeStyleAssignments } =
        createMockCanvasAndStrokeRecorder();
      const envState = createRendererEnvironmentState(trackSpec);

      renderRacingFrame(
        canvasElement,
        trackSpec,
        envState,
        createRacingRenderState(),
        computeWorldTransform(canvasElement, trackSpec),
        {
          frame: {
            featureFlags: 0,
            carTeam: Uint8Array.from([1]),
            tireState: new Float32Array([1, 1, 1, 1]),
          },
        },
      );

      expect(strokeStyleAssignments.includes('#00e5ff')).toBe(true);
    });
  });
});

function createTrackSpecWithoutPits(): TrackSpec {
  const generatedTrackSpec = generateTrack({
    seed: 42,
    layoutVersion: 1,
    sizeBucket: 'medium',
  });

  return {
    ...generatedTrackSpec,
    pitBoxes: undefined,
  };
}

function createRendererEnvironmentState(
  trackSpec: TrackSpec,
): EnvironmentState {
  return {
    tick: 0,
    carX: 0,
    carY: 0,
    carHeading: 0,
    teamIndex: 0,
    tireState: [1, 1, 1, 1],
    cars: [
      {
        carX: 0,
        carY: 0,
        carHeading: 0,
        teamIndex: 0,
        tireState: [1, 1, 1, 1],
      },
    ],
    trackSpec,
  };
}

function createMockCanvasAndStrokeRecorder(): {
  canvasElement: HTMLCanvasElement;
  strokeStyleAssignments: string[];
} {
  const strokeStyleAssignments: string[] = [];
  const mockContextState: Record<string, unknown> = {
    createRadialGradient: () => ({ addColorStop: () => undefined }),
    restore: () => undefined,
    save: () => undefined,
    setLineDash: () => undefined,
  };
  const mockCanvasContext = new Proxy(mockContextState, {
    get(target, propertyKey: string) {
      if (propertyKey in target) {
        return target[propertyKey];
      }

      return () => undefined;
    },
    set(target, propertyKey: string, value) {
      if (propertyKey === 'strokeStyle' && typeof value === 'string') {
        strokeStyleAssignments.push(value);
      }

      target[propertyKey] = value;
      return true;
    },
  }) as unknown as CanvasRenderingContext2D;
  const canvasElement = {
    width: 960,
    height: 540,
    getContext: () => mockCanvasContext,
  } as unknown as HTMLCanvasElement;

  return {
    canvasElement,
    strokeStyleAssignments,
  };
}
