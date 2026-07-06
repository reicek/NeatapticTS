import {
  buildGuidingLineForTeam,
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

      expect(strokeStyleAssignments.includes('rgba(255, 0, 0, 0.38)')).toBe(
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

      expect(strokeStyleAssignments.includes('#ff0000')).toBe(true);
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

    const teamAGuidingLine =
      rendererModule.buildGuidingLineForTeam?.(trackSpec, 0) ?? [];
    const teamBGuidingLine =
      rendererModule.buildGuidingLineForTeam?.(trackSpec, 1) ?? [];
    const bothAreNonEmptyArrays =
      Array.isArray(teamAGuidingLine) &&
      teamAGuidingLine.length > 0 &&
      Array.isArray(teamBGuidingLine) &&
      teamBGuidingLine.length > 0;

    expect(bothAreNonEmptyArrays && teamAGuidingLine !== teamBGuidingLine).toBe(
      true,
    );
  });

  it('places each guiding line parallel to the inner-lane centerline at the team baseline offset', async () => {
    const rendererModule = (await import('./racing.renderer')) as unknown as {
      buildGuidingLineForTeam?: (
        trackSpec: TrackSpec,
        teamIndex: number,
        lateralOffsetWorld?: number,
      ) => Array<{ readonly x: number; readonly y: number }>;
    };
    const trackSpec = createTrackSpecWithoutPits();
    const buildGuidingLine = rendererModule.buildGuidingLineForTeam;

    const firstSample = trackSpec.splineSamples[0];
    const laneWidthWorld = (firstSample?.width ?? 0) / 2;
    const innerOffsetWorld =
      firstSample?.innerOffsetWorld ??
      (firstSample?.width ?? 0) / 2 - laneWidthWorld / 2;
    const expectedOffsets = [innerOffsetWorld, -innerOffsetWorld];
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
        const splineSample =
          trackSpec.splineSamples[sampleIndex % trackSpec.splineSamples.length];
        if (splineSample === undefined) {
          allOffsetsValid = false;
          break;
        }
        const sampleFrame = resolveSplineSampleFrame(
          trackSpec.splineSamples,
          sampleIndex,
        );
        const deltaX = guidingPoint.x - splineSample.x;
        const deltaY = guidingPoint.y - splineSample.y;
        const signedOffset =
          deltaX * sampleFrame.normalX + deltaY * sampleFrame.normalY;
        offsets.push(signedOffset);
      }

      if (offsets.length === 0) {
        allOffsetsValid = false;
        break;
      }
      const expectedOffset = expectedOffsets[teamIndex] ?? 0;
      const offsetConstant = offsets.every(
        (offset) => Math.abs(offset - (offsets[0] ?? 0)) < 1e-6,
      );
      const offsetMatchesBaseline = offsets.every(
        (offset) => Math.abs(offset - expectedOffset) < 1e-6,
      );
      if (!offsetConstant || !offsetMatchesBaseline) {
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

  it('defaults Team 0 lateral offset to the inner-lane centerline distance', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const firstSample = trackSpec.splineSamples[0];
    if (firstSample === undefined) {
      throw new Error('Track spec has no spline samples');
    }
    const laneWidthWorld = firstSample.width / 2;
    const innerOffsetWorld =
      firstSample.innerOffsetWorld ??
      firstSample.width / 2 - laneWidthWorld / 2;

    const defaultTeam0Line = buildGuidingLineForTeam(trackSpec, 0);
    const explicitTeam0Line = buildGuidingLineForTeam(
      trackSpec,
      0,
      innerOffsetWorld,
    );

    expect(defaultTeam0Line).toEqual(explicitTeam0Line);
  });

  it('places Team 1 guiding line on the outer-lane centerline', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const firstSample = trackSpec.splineSamples[0];
    if (firstSample === undefined) {
      throw new Error('Track spec has no spline samples');
    }
    const laneWidthWorld = firstSample.width / 2;
    const innerOffsetWorld =
      firstSample.innerOffsetWorld ??
      firstSample.width / 2 - laneWidthWorld / 2;

    const team1GuidingLine = buildGuidingLineForTeam(trackSpec, 1);
    const firstFrame = resolveSplineSampleFrame(trackSpec.splineSamples, 0);
    const actualStart = team1GuidingLine[0];
    const signedOffset =
      actualStart === undefined
        ? Infinity
        : (actualStart.x - firstSample.x) * firstFrame.normalX +
          (actualStart.y - firstSample.y) * firstFrame.normalY;

    expect(signedOffset).toBeCloseTo(-innerOffsetWorld, 6);
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
      recordedPath.strokeStyle.includes('rgba(0,0,255,'),
    );
    const teamBPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,0,'),
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
      recordedPath.strokeStyle.includes('rgba(0,0,255,'),
    );
    const teamBPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,0,'),
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
      recordedPath.strokeStyle.includes('rgba(0,0,255,'),
    );
    const teamBIndex = recordedPaths.findIndex((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,0,'),
    );
    const lastGuidingLineIndex = Math.max(teamAIndex, teamBIndex);
    const firstCarBodyIndex = recordedPaths.findIndex(
      (recordedPath) =>
        recordedPath.strokeStyle === '#0000ff' ||
        recordedPath.strokeStyle === '#ff0000',
    );
    const bothGuidingLinesFound = teamAIndex >= 0 && teamBIndex >= 0;
    const carBodyFound = firstCarBodyIndex >= 0;
    const orderCorrect =
      bothGuidingLinesFound &&
      carBodyFound &&
      lastGuidingLineIndex < firstCarBodyIndex;

    expect(orderCorrect).toBe(true);
  });

  it('draws blue and red guide lines at inner and outer lane offsets for a 2-car pack', () => {
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

    const firstSample = trackSpec.splineSamples[0];
    const laneWidthWorld = (firstSample?.width ?? 0) / 2;
    const innerOffsetWorld =
      firstSample?.innerOffsetWorld ??
      (firstSample?.width ?? 0) / 2 - laneWidthWorld / 2;

    const bluePath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(0,0,255,'),
    );
    const redPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,0,'),
    );
    const blueOffset =
      bluePath === undefined
        ? Infinity
        : computeAverageSignedOffsetFromCenterline(
            bluePath,
            trackSpec,
            transform,
          );
    const redOffset =
      redPath === undefined
        ? Infinity
        : computeAverageSignedOffsetFromCenterline(
            redPath,
            trackSpec,
            transform,
          );
    const guidePathCount = recordedPaths.filter(
      (recordedPath) =>
        recordedPath.strokeStyle.includes('rgba(0,0,255,') ||
        recordedPath.strokeStyle.includes('rgba(255,0,0,'),
    ).length;

    const contractMet =
      bluePath !== undefined &&
      redPath !== undefined &&
      guidePathCount === 2 &&
      Math.abs(blueOffset - innerOffsetWorld) < 1e-6 &&
      Math.abs(redOffset - -innerOffsetWorld) < 1e-6;

    expect(contractMet).toBe(true);
  });
});

describe('Tier 1/Tier 2 baseline color and guide-track contracts', () => {
  it('draws Team 0 car bodies in blue when pit visuals are disabled', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const { canvasElement, strokeStyleAssignments } =
      createMockCanvasAndStrokeRecorder();
    const envState = createTwoCarEnvironmentState(trackSpec);

    renderRacingFrame(
      canvasElement,
      trackSpec,
      envState,
      createRacingRenderState(),
      computeWorldTransform(canvasElement, trackSpec),
      {
        frame: {
          featureFlags: 0,
          carTeam: Uint8Array.from([0, 1]),
          tireState: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]),
        },
      },
    );

    const hasBlueStroke = strokeStyleAssignments.some(
      (strokeStyle) =>
        strokeStyle === '#0000ff' || strokeStyle.includes('rgba(0,0,255,'),
    );

    expect(hasBlueStroke).toBe(true);
  });

  it('draws Team 1 car bodies in red when pit visuals are disabled', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const { canvasElement, strokeStyleAssignments } =
      createMockCanvasAndStrokeRecorder();
    const envState = createTwoCarEnvironmentState(trackSpec);

    renderRacingFrame(
      canvasElement,
      trackSpec,
      envState,
      createRacingRenderState(),
      computeWorldTransform(canvasElement, trackSpec),
      {
        frame: {
          featureFlags: 0,
          carTeam: Uint8Array.from([0, 1]),
          tireState: new Float32Array([1, 1, 1, 1, 1, 1, 1, 1]),
        },
      },
    );

    const hasRedStroke = strokeStyleAssignments.some(
      (strokeStyle) =>
        strokeStyle === '#ff0000' || strokeStyle.includes('rgba(255,0,0,'),
    );

    expect(hasRedStroke).toBe(true);
  });

  it('draws Team 0 guiding line in blue on the inner-lane centerline', () => {
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

    const teamZeroPath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(0,0,255,'),
    );
    const firstSample = trackSpec.splineSamples[0];
    const laneWidthWorld =
      firstSample === undefined ? 0 : (firstSample.width ?? 0) / 2;
    const innerOffsetWorld =
      firstSample === undefined
        ? 0
        : firstSample.width / 2 - laneWidthWorld / 2;
    const averageOffset =
      teamZeroPath === undefined
        ? Infinity
        : computeAverageSignedOffsetFromCenterline(
            teamZeroPath,
            trackSpec,
            transform,
          );

    expect(averageOffset).toBeCloseTo(innerOffsetWorld, 6);
  });

  it('draws Team 1 guiding line in red on the outer-lane centerline', () => {
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

    const teamOnePath = recordedPaths.find((recordedPath) =>
      recordedPath.strokeStyle.includes('rgba(255,0,0,'),
    );
    const firstSample = trackSpec.splineSamples[0];
    const laneWidthWorld =
      firstSample === undefined ? 0 : (firstSample.width ?? 0) / 2;
    const innerOffsetWorld =
      firstSample === undefined
        ? 0
        : firstSample.width / 2 - laneWidthWorld / 2;
    const expectedOuterOffset = -innerOffsetWorld;
    const averageOffset =
      teamOnePath === undefined
        ? Infinity
        : computeAverageSignedOffsetFromCenterline(
            teamOnePath,
            trackSpec,
            transform,
          );

    expect(averageOffset).toBeCloseTo(expectedOuterOffset, 6);
  });

  it('draws one guiding line per car when the state contains multiple cars per team', () => {
    const trackSpec = createTrackSpecWithoutPits();
    const { canvasElement, recordedPaths } = createMockCanvasAndPathRecorder();
    const envState = createFourCarEnvironmentState(trackSpec);
    const transform = computeWorldTransform(canvasElement, trackSpec);

    renderRacingFrame(
      canvasElement,
      trackSpec,
      envState,
      createRacingRenderState(),
      transform,
      { guidanceAlpha: 0.8 },
    );

    const guidePathCount = recordedPaths.filter(
      (recordedPath) =>
        recordedPath.strokeStyle.includes('rgba(0,0,255,') ||
        recordedPath.strokeStyle.includes('rgba(255,0,0,'),
    ).length;

    expect(guidePathCount).toBe(4);
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

function createFourCarEnvironmentState(trackSpec: TrackSpec): EnvironmentState {
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

function computeAverageSignedOffsetFromCenterline(
  recordedPath: RecordedPath,
  trackSpec: TrackSpec,
  transform: ReturnType<typeof computeWorldTransform>,
): number {
  const { splineSamples } = trackSpec;
  const offsets: number[] = [];

  for (
    let operationIndex = 0;
    operationIndex < recordedPath.operations.length;
    operationIndex++
  ) {
    const operation = recordedPath.operations[operationIndex];
    if (operation === undefined) {
      continue;
    }
    const sampleIndex = operationIndex % splineSamples.length;
    const sample = splineSamples[sampleIndex];
    if (sample === undefined) {
      continue;
    }
    const sampleFrame = resolveSplineSampleFrame(splineSamples, sampleIndex);
    const worldX = (operation.x - transform.offsetX) / transform.scale;
    const worldY = (operation.y - transform.offsetY) / transform.scale;
    const deltaX = worldX - sample.x;
    const deltaY = worldY - sample.y;
    const signedOffset =
      deltaX * sampleFrame.normalX + deltaY * sampleFrame.normalY;

    offsets.push(signedOffset);
  }

  if (offsets.length === 0) {
    return NaN;
  }

  const sum = offsets.reduce((accumulator, offset) => accumulator + offset, 0);
  return sum / offsets.length;
}

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
    stroke: () => {
      currentPath = undefined;
    },
    fill: () => {
      currentPath = undefined;
    },
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
        if (currentPath !== undefined) {
          (currentPath as { strokeStyle: string }).strokeStyle = value;
        }
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
