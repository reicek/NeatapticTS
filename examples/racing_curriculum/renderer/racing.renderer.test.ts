import {
  computeWorldTransform,
  createRacingRenderState,
  renderRacingFrame,
} from './racing.renderer';
import { generateTrack } from '../track/track.generator';
import {
  resolveInnerLaneCenterlinePoint,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';
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

    it('draws the optimal-line guidance along the inner-lane centerline', () => {
      const trackSpec = createTrackSpecWithoutPits();
      const { canvasElement, recordedPaths } =
        createMockCanvasAndPathRecorder();
      const envState = createRendererEnvironmentState(trackSpec);
      const transform = computeWorldTransform(canvasElement, trackSpec);

      renderRacingFrame(
        canvasElement,
        trackSpec,
        envState,
        createRacingRenderState(),
        transform,
        { guidanceAlpha: 0.8 },
      );

      const expectedInnerLanePoints = trackSpec.splineSamples.map((sample) => {
        const frame = resolveSplineSampleFrame(
          trackSpec.splineSamples,
          sample.globalIndex,
        );
        const laneCount = 2;
        const laneWidthWorld = sample.width / laneCount;
        const innerOffsetWorld = sample.width / 2 - laneWidthWorld / 2;

        return {
          x: sample.x + frame.normalX * innerOffsetWorld,
          y: sample.y + frame.normalY * innerOffsetWorld,
        };
      });
      const expectedCanvasPoints = expectedInnerLanePoints.map((point) => ({
        x: point.x * transform.scale + transform.offsetX,
        y: point.y * transform.scale + transform.offsetY,
      }));
      const guidancePath = recordedPaths.find((recordedPath) =>
        recordedPath.strokeStyle.includes('255,209,102'),
      );
      const actualCanvasPoints =
        guidancePath?.operations.map((operation) => ({
          x: operation.x,
          y: operation.y,
        })) ?? [];

      expect(actualCanvasPoints).toEqual(expectedCanvasPoints);
    });
  });
});

describe('per-agent guiding line geometry', () => {
  it('exports a buildGuidingLineForTeam helper from the renderer module', async () => {
    const rendererModule = (await import('./racing.renderer')) as unknown as {
      buildGuidingLineForTeam?: unknown;
    };

    expect(typeof rendererModule.buildGuidingLineForTeam).toBe('function');
  });

  it('returns distinct point arrays for Team A and Team B', async () => {
    const rendererModule = (await import('./racing.renderer')) as unknown as {
      buildGuidingLineForTeam?: (
        trackSpec: TrackSpec,
        teamIndex: number,
      ) => Array<{ readonly x: number; readonly y: number }>;
    };
    const trackSpec = createTrackSpecWithoutPits();

    const teamAGuidingLine = rendererModule.buildGuidingLineForTeam?.(
      trackSpec,
      0,
    ) ?? [];
    const teamBGuidingLine = rendererModule.buildGuidingLineForTeam?.(
      trackSpec,
      1,
    ) ?? [];
    const bothAreNonEmptyArrays =
      Array.isArray(teamAGuidingLine) &&
      teamAGuidingLine.length > 0 &&
      Array.isArray(teamBGuidingLine) &&
      teamBGuidingLine.length > 0;

    expect(
      bothAreNonEmptyArrays && teamAGuidingLine !== teamBGuidingLine,
    ).toBe(true);
  });

  it('places each guiding line parallel to the inner-lane centerline and within the inner lane', async () => {
    const rendererModule = (await import('./racing.renderer')) as unknown as {
      buildGuidingLineForTeam?: (
        trackSpec: TrackSpec,
        teamIndex: number,
      ) => Array<{ readonly x: number; readonly y: number }>;
    };
    const trackSpec = createTrackSpecWithoutPits();
    const buildGuidingLine = rendererModule.buildGuidingLineForTeam;

    let allOffsetsValid = true;
    for (const teamIndex of [0, 1]) {
      const guidingLine = buildGuidingLine?.(trackSpec, teamIndex) ?? [];
      if (!Array.isArray(guidingLine) || guidingLine.length === 0) {
        allOffsetsValid = false;
        break;
      }

      const offsets: number[] = [];
      for (
        let sampleIndex = 0;
        sampleIndex < guidingLine.length;
        sampleIndex++
      ) {
        const guidingPoint = guidingLine[sampleIndex];
        if (guidingPoint === undefined) {
          allOffsetsValid = false;
          break;
        }
        const splineSample = trackSpec.splineSamples[
          sampleIndex % trackSpec.splineSamples.length
        ];
        if (splineSample === undefined) {
          allOffsetsValid = false;
          break;
        }
        const sampleFrame = resolveSplineSampleFrame(
          trackSpec.splineSamples,
          sampleIndex,
        );
        const innerLanePoint = resolveInnerLaneCenterlinePoint(
          splineSample,
          sampleFrame,
        );
        const deltaX = guidingPoint.x - innerLanePoint.x;
        const deltaY = guidingPoint.y - innerLanePoint.y;
        const signedOffset =
          deltaX * sampleFrame.normalX + deltaY * sampleFrame.normalY;
        offsets.push(signedOffset);
      }

      if (offsets.length === 0) {
        allOffsetsValid = false;
        break;
      }
      const firstOffset = offsets[0];
      const laneWidthWorld = trackSpec.splineSamples[0]?.width ?? 0;
      const halfLaneWidthWorld = laneWidthWorld / 4;
      const offsetConstant = offsets.every(
        (offset) => Math.abs(offset - firstOffset) < 1e-6,
      );
      const withinInnerLane = offsets.every(
        (offset) => Math.abs(offset) < halfLaneWidthWorld,
      );
      if (!offsetConstant || !withinInnerLane) {
        allOffsetsValid = false;
        break;
      }
    }

    expect(allOffsetsValid).toBe(true);
  });

  it('starts the Team A guiding line at the inner-lane centerline start point', async () => {
    const rendererModule = (await import('./racing.renderer')) as unknown as {
      buildGuidingLineForTeam?: (
        trackSpec: TrackSpec,
        teamIndex: number,
      ) => Array<{ readonly x: number; readonly y: number }>;
    };
    const trackSpec = createTrackSpecWithoutPits();
    const teamAGuidingLine =
      rendererModule.buildGuidingLineForTeam?.(trackSpec, 0) ?? [];
    const startSample = trackSpec.splineSamples[0];
    if (startSample === undefined) {
      throw new Error('Track spec has no spline samples');
    }
    const startFrame = resolveSplineSampleFrame(trackSpec.splineSamples, 0);
    const expectedStart = resolveInnerLaneCenterlinePoint(
      startSample,
      startFrame,
    );
    const actualStart = teamAGuidingLine[0];
    const distance =
      actualStart !== undefined
        ? Math.hypot(
            actualStart.x - expectedStart.x,
            actualStart.y - expectedStart.y,
          )
        : Infinity;

    expect(distance).toBeLessThan(1e-6);
  });
});

describe('per-agent guiding line draw calls', () => {
  it('draws a guiding line for each team when guidance overlay is enabled', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const { canvasElement, recordedPaths } = createMockCanvasAndPathRecorder();
    const envState = createTwoCarEnvironmentState(trackSpec);
    const transform = computeWorldTransform(canvasElement, trackSpec);

    renderRacingFrame(
      canvasElement,
      trackSpec,
      envState,
      createRacingRenderState(),
      transform,
      { guidanceAlpha: 0.8 },
    );

    const teamAPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(0,229,255,'),
    );
    const teamBPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,255,'),
    );

    expect(teamAPath !== undefined && teamBPath !== undefined).toBe(true);
  });

  it('draws Team A and Team B guiding lines in different colors', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const { canvasElement, recordedPaths } = createMockCanvasAndPathRecorder();
    const envState = createTwoCarEnvironmentState(trackSpec);
    const transform = computeWorldTransform(canvasElement, trackSpec);

    renderRacingFrame(
      canvasElement,
      trackSpec,
      envState,
      createRacingRenderState(),
      transform,
      { guidanceAlpha: 0.8 },
    );

    const teamAPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(0,229,255,'),
    );
    const teamBPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,255,'),
    );
    const bothFound =
      teamAPath !== undefined &&
      teamBPath !== undefined &&
      teamAPath.strokeStyle !== teamBPath.strokeStyle;

    expect(bothFound).toBe(true);
  });

  it('draws guiding lines before car bodies in the render order', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const { canvasElement, recordedPaths } = createMockCanvasAndPathRecorder();
    const envState = createTwoCarEnvironmentState(trackSpec);
    const transform = computeWorldTransform(canvasElement, trackSpec);

    renderRacingFrame(
      canvasElement,
      trackSpec,
      envState,
      createRacingRenderState(),
      transform,
      { guidanceAlpha: 0.8 },
    );

    const teamAIndex = recordedPaths.findIndex((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(0,229,255,'),
    );
    const teamBIndex = recordedPaths.findIndex((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,255,'),
    );
    const lastGuidingLineIndex = Math.max(teamAIndex, teamBIndex);
    const firstCarBodyIndex = recordedPaths.findIndex(
      (recordedPath) => recordedPath.strokeStyle === '#00e5ff',
    );
    const bothGuidingLinesFound = teamAIndex >= 0 && teamBIndex >= 0;
    const carBodyFound = firstCarBodyIndex >= 0;
    const orderCorrect =
      bothGuidingLinesFound && carBodyFound && lastGuidingLineIndex < firstCarBodyIndex;

    expect(orderCorrect).toBe(true);
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

function createTwoCarEnvironmentState(trackSpec: TrackSpec): EnvironmentState {
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
      {
        carX: 0,
        carY: 0,
        carHeading: 0,
        teamIndex: 1,
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

type PathOperation =
  | { readonly type: 'moveTo'; readonly x: number; readonly y: number }
  | { readonly type: 'lineTo'; readonly x: number; readonly y: number };

type RecordedPath = {
  readonly strokeStyle: string;
  readonly operations: PathOperation[];
};

function createMockCanvasAndPathRecorder(): {
  canvasElement: HTMLCanvasElement;
  recordedPaths: RecordedPath[];
} {
  const recordedPaths: RecordedPath[] = [];
  let currentPath: RecordedPath | undefined;
  let currentStrokeStyle = '';

  const mockContextState: Record<string, unknown> = {
    createRadialGradient: () => ({ addColorStop: () => undefined }),
    restore: () => undefined,
    save: () => undefined,
    setLineDash: () => undefined,
    beginPath: () => {
      currentPath = { strokeStyle: currentStrokeStyle, operations: [] };
      recordedPaths.push(currentPath);
    },
    closePath: () => {
      // Closing a polyline does not emit an explicit lineTo in the API, so no
      // operation is recorded here.
    },
    moveTo: (x: number, y: number) => {
      currentPath?.operations.push({ type: 'moveTo', x, y });
    },
    lineTo: (x: number, y: number) => {
      currentPath?.operations.push({ type: 'lineTo', x, y });
    },
    stroke: () => undefined,
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
        currentStrokeStyle = value;
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
    recordedPaths,
  };
}
