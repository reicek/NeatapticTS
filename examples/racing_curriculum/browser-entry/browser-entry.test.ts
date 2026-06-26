/** @jest-environment jsdom */

import fs from 'fs';
import path from 'path';

import {
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_UI_NETWORK_HOST_BACKGROUND,
} from '../../flappy_bird/constants/constants.layout';
import { createNgeController } from '../controller/nge.controller';
import * as actualNgeControllerModule from '../controller/nge.controller';
import { derivePerCarObservationState } from '../controller/observation.assembler';
import type { CarControlOutput, EnvironmentState } from '../environment/environment.types';
import * as actualEnvironmentStepModule from '../environment/environment.step.service';
import { generateTrack } from '../track/track.generator';
import { resolveSplineSampleFrame } from '../track/track.spline.utils';
import {
  createCurriculumEnvironmentState,
  createCurriculumEpisodeState,
  createDeterministicRacingControllerNetwork,
  resolveNetworkHudStatus,
  stabilizeCurriculumTierTireGrip,
  start,
} from './browser-entry';
import {
  drawRacingNetworkVisualizationFromFrame,
  resolveRacingNetworkVisualizationFrame,
} from './network-view/network-view';

jest.mock('./network-view/network-view', () => ({
  drawRacingNetworkVisualizationFromFrame: jest.fn(() => ({
    positionedNodes: [],
    nodeDimensions: { widthPx: 0, heightPx: 0 },
    inputDescriptionScenes: [],
    inputGroupLabelBandScenes: [],
    hiddenColumnLabelScenes: [],
  })),
  resolveRacingNetworkVisualizationFrame: jest.fn(() => ({
    canvasWidthPx: 320,
    canvasHeightPx: 240,
    positionedScene: {
      positionedNodes: [],
      nodeDimensions: { widthPx: 0, heightPx: 0 },
      inputDescriptionScenes: [],
      inputGroupLabelBandScenes: [],
      hiddenColumnLabelScenes: [],
    },
    staticLayerCanvas: document.createElement('canvas'),
  })),
}));

const mockedDrawRacingNetworkVisualizationFromFrame = jest.mocked(
  drawRacingNetworkVisualizationFromFrame,
);
const mockedResolveRacingNetworkVisualizationFrame = jest.mocked(
  resolveRacingNetworkVisualizationFrame,
);

function hexToRgb(hexColor: string): string {
  const hexDigits = hexColor.replace('#', '');
  const red = parseInt(hexDigits.slice(0, 2), 16);
  const green = parseInt(hexDigits.slice(2, 4), 16);
  const blue = parseInt(hexDigits.slice(4, 6), 16);
  return `rgb(${red}, ${green}, ${blue})`;
}

describe('racing curriculum browser entry start()', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('rejects when the requested host container is missing', async () => {
    await expect(start('missing-racing-curriculum-host')).rejects.toThrow(
      'Racing curriculum container "missing-racing-curriculum-host" was not found.',
    );
  });

  it('keeps active panel readiness copy free of future-facing and deferred placeholders', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.textContent?.includes('Future-facing') ||
        hostElement.textContent?.includes('Deferred'),
    ).toBe(false);
  });

  it('renders a live network canvas in the focused network panel', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.querySelector(
        '.racing-host__region--network canvas.racing-network-canvas',
      ),
    ).not.toBeNull();
  });

  it('does not render a visualizer-bottom region after start', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(
      hostElement.querySelector('.racing-host__region--visualizer'),
    ).toBeNull();
  });

  it('does not show the Race Pack Slot placeholder', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(hostElement.textContent?.includes('Race Pack Slot')).toBe(false);
  });

  it('does not show the Controller Status card in the network region', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    const networkRegion = hostElement.querySelector(
      '.racing-host__region--network',
    );

    expect(
      Array.from(networkRegion?.querySelectorAll('*') ?? []).some(
        (descendant) => descendant.textContent?.includes('Controller Status'),
      ),
    ).toBe(false);
  });

  it('places runtime controls below the track instead of in the visualizer region', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    const controlsElement = hostElement.querySelector('.racing-controls');
    const canvasRegion = hostElement.querySelector(
      '.racing-host__region--canvas',
    );
    const visualizerRegion = hostElement.querySelector(
      '.racing-host__region--visualizer',
    );

    expect(
      (canvasRegion?.contains(controlsElement) ?? false) &&
        !(visualizerRegion?.contains(controlsElement) ?? false),
    ).toBe(true);
  });

  it('paints the network host with the Flappy Bird network-host background', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    const networkCanvasHost =
      hostElement.querySelector('.racing-network-canvas-host') ??
      document.createElement('div');

    expect(getComputedStyle(networkCanvasHost).backgroundColor).toBe(
      hexToRgb(FLAPPY_UI_NETWORK_HOST_BACKGROUND),
    );
  });

  it('paints the network canvas with the Flappy Bird network-canvas background', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    const networkCanvas = hostElement.querySelector('.racing-network-canvas');

    expect(getComputedStyle(networkCanvas as Element).backgroundColor).toBe(
      hexToRgb(FLAPPY_UI_NETWORK_CANVAS_BACKGROUND),
    );
  });

  it('shows a network tooltip while hovering the network canvas', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    const networkCanvas = hostElement.querySelector('.racing-network-canvas');
    networkCanvas?.dispatchEvent(
      new MouseEvent('pointermove', { clientX: 20, clientY: 20 }),
    );

    runHandle.stop();

    expect(hostElement.querySelector('.racing-network-tooltip')).not.toBeNull();
  });

  it('resizes the network canvas backing store when the window resizes', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    const networkCanvas = hostElement.querySelector(
      '.racing-network-canvas',
    ) as HTMLCanvasElement;
    const initialWidth = networkCanvas.width;
    const initialHeight = networkCanvas.height;

    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: window.innerWidth + 200,
    });
    window.dispatchEvent(new Event('resize'));

    runHandle.stop();

    expect(
      networkCanvas.width !== initialWidth ||
        networkCanvas.height !== initialHeight,
    ).toBe(true);
  });

  it('keeps tire wear disabled before the wear tier and preserves it at the wear tier', () => {
    const wornEnvironmentState = {
      tick: 1,
      carX: 0,
      carY: 0,
      carHeading: 0,
      tireState: [0.4, 0.3, 0.2, 0.1] as const,
      cars: [
        {
          carX: 0,
          carY: 0,
          carHeading: 0,
          teamIndex: 0,
          tireState: [0.4, 0.3, 0.2, 0.1] as const,
        },
      ],
    } satisfies EnvironmentState;

    expect({
      tier1: stabilizeCurriculumTierTireGrip(wornEnvironmentState, 1),
      tier4: stabilizeCurriculumTierTireGrip(wornEnvironmentState, 4),
    }).toMatchObject({
      tier1: {
        tireState: [1, 1, 1, 1],
        cars: [{ tireState: [1, 1, 1, 1] }],
      },
      tier4: {
        tireState: [0.4, 0.3, 0.2, 0.1],
        cars: [{ tireState: [0.4, 0.3, 0.2, 0.1] }],
      },
    });
  });

  it('delegates network rendering to the shared racing network visualizer', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(mockedResolveRacingNetworkVisualizationFrame).toHaveBeenCalled();
    expect(mockedDrawRacingNetworkVisualizationFromFrame).toHaveBeenCalled();
  });

  it('does not render an architecture selector', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(hostElement.textContent?.includes('Architectures')).toBe(false);
  });

  it('does not use the docs alias as the page title', async () => {
    document.title = 'Racing Curriculum (NeatapticTS)';
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    expect(document.title).not.toBe('docs:folders:racing-curriculum');
  });

  it('does not use the docs alias in any heading element', async () => {
    document.title = 'Racing Curriculum (NeatapticTS)';
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    const headings = Array.from(
      document.querySelectorAll('h1, h2, h3, h4, h5, h6'),
    );
    const hasAliasHeading = headings.some(
      (heading) => heading.textContent === 'docs:folders:racing-curriculum',
    );

    expect(hasAliasHeading).toBe(false);
  });

  it('sets a non-empty human-readable page title or heading', async () => {
    document.title = 'Racing Curriculum (NeatapticTS)';
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);

    runHandle.stop();

    const headings = Array.from(
      document.querySelectorAll('h1, h2, h3, h4, h5, h6'),
    );
    const readableHeading = headings.find((heading) => {
      const text = heading.textContent?.trim() ?? '';
      return text.length > 0 && text !== 'docs:folders:racing-curriculum';
    });
    const titleIsReadable =
      document.title.trim().length > 0 &&
      document.title !== 'docs:folders:racing-curriculum';

    expect(titleIsReadable || readableHeading !== undefined).toBe(true);
  });
});

describe('racing curriculum network HUD status resolver', () => {
  it('reports HOLD when adaptation is disabled', () => {
    expect(resolveNetworkHudStatus(false, 'improving')).toBe('HOLD');
  });

  it('reports ADAPTING when adaptation is enabled and trend is improving', () => {
    expect(resolveNetworkHudStatus(true, 'improving')).toBe('ADAPTING');
  });

  it('reports CAUTION when adaptation is enabled and trend is regressing', () => {
    expect(resolveNetworkHudStatus(true, 'regressing')).toBe('CAUTION');
  });

  it('reports STABLE for any non-improving, non-regressing trend', () => {
    expect(resolveNetworkHudStatus(true, 'flat')).toBe('STABLE');
  });
});

describe('racing curriculum browser entry evolution worker protocol', () => {
  const originalWorker = globalThis.Worker;
  const postMessageSpy = jest.fn();
  const addEventListenerStub = jest.fn();
  const fakeWorker = {
    postMessage: postMessageSpy,
    addEventListener: addEventListenerStub,
    removeEventListener: jest.fn(),
    terminate: jest.fn(),
  } as unknown as Worker;

  beforeEach(() => {
    globalThis.Worker = jest.fn(
      () => fakeWorker,
    ) as unknown as typeof globalThis.Worker;

    const currentScriptElement = document.createElement('script');
    currentScriptElement.src = 'http://localhost/racing-bundle.js';
    Object.defineProperty(document, 'currentScript', {
      value: currentScriptElement,
      configurable: true,
    });
  });

  afterEach(() => {
    globalThis.Worker = originalWorker;
    postMessageSpy.mockClear();
    addEventListenerStub.mockClear();
  });

  it('posts an init message to the simulation worker when starting', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);
    runHandle.stop();

    expect(postMessageSpy.mock.calls[0]?.[0]).toMatchObject({
      type: 'init',
      populationSize: expect.any(Number),
      rngSeed: expect.any(Number),
      tier: expect.any(Number),
    });
  });

  it('posts an init message with curriculum tier 1', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);
    runHandle.stop();

    expect(postMessageSpy.mock.calls[0]?.[0]).toMatchObject({
      type: 'init',
      tier: 1,
    });
  });
});

describe('racing curriculum browser entry source contracts', () => {
  it('does not import network rendering from the old shared visualization path', () => {
    const sourcePath = path.join(__dirname, 'browser-entry.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(
      sourceText.includes(
        "from '../../../src/visualization/network-view/network-view'",
      ),
    ).toBe(false);
  });
});

describe('createCurriculumEnvironmentState', () => {
  it('places the primary car on the inner-lane centerline of the first spline sample', () => {
    const trackSpec = generateTrack({
      seed: 42,
      layoutVersion: 1,
      sizeBucket: 'medium',
    });
    const envState = createCurriculumEnvironmentState(trackSpec, 1);
    const firstSplineSample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      firstSplineSample.globalIndex,
    );
    const laneCount = 2;
    const laneWidthWorld = firstSplineSample.width / laneCount;
    const innerOffsetWorld = firstSplineSample.width / 2 - laneWidthWorld / 2;
    const expectedCarX = firstSplineSample.x + frame.normalX * innerOffsetWorld;
    const expectedCarY = firstSplineSample.y + frame.normalY * innerOffsetWorld;

    expect({
      carX: envState.carX,
      carY: envState.carY,
    }).toEqual({
      carX: expectedCarX,
      carY: expectedCarY,
    });
  });
});

describe('Tier 2 1v1 race pack layout', () => {
  it('resolves the Tier 2 race-pack layout as one car per team', () => {
    const episodeState = createCurriculumEpisodeState(2);
    const cars = episodeState.envState.cars ?? [];
    const teamIndices = cars.map((car) => car.teamIndex);

    expect({
      carCount: cars.length,
      teamIndices,
    }).toEqual({
      carCount: 2,
      teamIndices: [0, 1],
    });
  });
});

describe('Tier 3 2v2 fallback race pack layout', () => {
  it('returns four cars for Tier 3', () => {
    const episodeState = createCurriculumEpisodeState(3);
    const cars = episodeState.envState.cars ?? [];

    expect(cars.length).toBe(4);
  });

  it('assigns team indices [0, 0, 1, 1] for Tier 3', () => {
    const episodeState = createCurriculumEpisodeState(3);
    const cars = episodeState.envState.cars ?? [];
    const teamIndices = cars.map((car) => car.teamIndex);

    expect(teamIndices).toEqual([0, 0, 1, 1]);
  });

  it('places Team 0 on the inner lane and Team 1 on the outer lane for Tier 3', () => {
    const episodeState = createCurriculumEpisodeState(3);
    const trackSpec = episodeState.trackSpec;
    const firstSplineSample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      firstSplineSample.globalIndex,
    );
    const cars = episodeState.envState.cars ?? [];
    const team0Cars = cars.filter((car) => car.teamIndex === 0);
    const team1Cars = cars.filter((car) => car.teamIndex === 1);
    const team0Inner = team0Cars.every(
      (car) =>
        (car.carX - firstSplineSample.x) * frame.normalX +
          (car.carY - firstSplineSample.y) * frame.normalY >
        0,
    );
    const team1Outer = team1Cars.every(
      (car) =>
        (car.carX - firstSplineSample.x) * frame.normalX +
          (car.carY - firstSplineSample.y) * frame.normalY <
        0,
    );

    expect(team0Inner && team1Outer && team1Cars.length === 2).toBe(true);
  });
});

describe('Tier 1 and Tier 2 regression guards', () => {
  it('keeps Tier 1 as a two-car pack', () => {
    const episodeState = createCurriculumEpisodeState(1);
    const cars = episodeState.envState.cars ?? [];
    const teamIndices = cars.map((car) => car.teamIndex);

    expect({ carCount: cars.length, teamIndices }).toEqual({
      carCount: 2,
      teamIndices: [0, 1],
    });
  });

  it('keeps Tier 2 as a two-car pack', () => {
    const episodeState = createCurriculumEpisodeState(2);
    const cars = episodeState.envState.cars ?? [];
    const teamIndices = cars.map((car) => car.teamIndex);

    expect({ carCount: cars.length, teamIndices }).toEqual({
      carCount: 2,
      teamIndices: [0, 1],
    });
  });
});

describe('Tier 2 deterministic controller network', () => {
  it('builds a 77-input, 9-output action head for control plus radio write', () => {
    const network = createDeterministicRacingControllerNetwork(2);

    expect({
      inputCount: network.input,
      outputCount: network.output,
    }).toEqual({
      inputCount: 77,
      outputCount: 9,
    });
  });
});

describe('racing network visualizer live refresh', () => {
  let rafSpy: jest.SpyInstance | undefined;

  beforeEach(() => {
    jest.useFakeTimers();
    rafSpy = jest
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation(() => 0);
    Object.defineProperty(document, 'currentScript', {
      value: null,
      configurable: true,
    });
  });

  afterEach(() => {
    jest.useRealTimers();
    rafSpy?.mockRestore();
  });

  it('refreshes the focused network render on a 5000 ms interval', async () => {
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);
    const callsBeforeInterval =
      mockedResolveRacingNetworkVisualizationFrame.mock.calls.length;

    jest.advanceTimersByTime(5000);

    runHandle.stop();

    expect(
      mockedResolveRacingNetworkVisualizationFrame.mock.calls.length,
    ).toBeGreaterThan(callsBeforeInterval);
  });

  it('clears the focused-network refresh interval on stop', async () => {
    const setIntervalSpy = jest.spyOn(window, 'setInterval');
    const clearIntervalSpy = jest.spyOn(window, 'clearInterval');

    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    const runHandle = await start(hostElement);
    const intervalId = setIntervalSpy.mock.results[0]?.value;

    runHandle.stop();

    const cleanup = (): void => {
      setIntervalSpy.mockRestore();
      clearIntervalSpy.mockRestore();
      hostElement.remove();
    };

    expect(clearIntervalSpy).toHaveBeenCalledWith(intervalId);
    cleanup();
  });
});

describe('Tier 1/Tier 2 lane and color assignment baseline', () => {
  it('places Team 0 on the inner lane and Team 1 on the outer lane', () => {
    const episodeState = createCurriculumEpisodeState(2);
    const trackSpec = episodeState.trackSpec;
    const firstSplineSample = trackSpec.splineSamples[0]!;
    const frame = resolveSplineSampleFrame(
      trackSpec.splineSamples,
      firstSplineSample.globalIndex,
    );
    const cars = episodeState.envState.cars ?? [];
    const team0Car = cars.find((car) => car.teamIndex === 0)!;
    const team1Car = cars.find((car) => car.teamIndex === 1)!;
    const team0Offset =
      (team0Car.carX - firstSplineSample.x) * frame.normalX +
      (team0Car.carY - firstSplineSample.y) * frame.normalY;
    const team1Offset =
      (team1Car.carX - firstSplineSample.x) * frame.normalX +
      (team1Car.carY - firstSplineSample.y) * frame.normalY;

    expect(team0Offset > 0 && team1Offset < 0).toBe(true);
  });

  it('exports team color index constants for blue and red mapping', async () => {
    const mod = (await import('./browser-entry')) as unknown as {
      TEAM_BLUE_INDEX?: number;
      TEAM_RED_INDEX?: number;
    };

    expect([mod.TEAM_BLUE_INDEX, mod.TEAM_RED_INDEX]).toEqual([0, 1]);
  });
});

describe('Phase 3 per-car controller red contracts', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
    Object.defineProperty(document, 'currentScript', {
      value: null,
      configurable: true,
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('does not contain the shared fan-out helper in browser-entry source', () => {
    const sourcePath = path.join(__dirname, 'browser-entry.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(sourceText.includes('resolveControlFanOut')).toBe(false);
  });

  it('creates one independent controller per car at startup', async () => {
    const { runHandle, createNgeControllerSpy } =
      await startTier1WithControllerSpy();

    runHandle.stop();

    expect(createNgeControllerSpy).toHaveBeenCalledTimes(2);
  });

  it('passes a distinct network instance to each car controller', async () => {
    const { runHandle, createNgeControllerSpy } =
      await startTier1WithControllerSpy();

    runHandle.stop();

    expect({
      callCount: createNgeControllerSpy.mock.calls.length,
      sameNetwork:
        createNgeControllerSpy.mock.calls[0]?.[0] ===
        createNgeControllerSpy.mock.calls[1]?.[0],
    }).toEqual({
      callCount: 2,
      sameNetwork: false,
    });
  });

  it('passes distinct controls to each car during a fixed-timestep tick', async () => {
    const { runHandle, stepEnvironmentSpy, rafCallback } =
      await startTier1WithStepEnvironmentSpy();

    await rafCallback(0);
    await rafCallback(17);

    runHandle.stop();

    const lastControls = stepEnvironmentSpy.mock.calls.at(-1)?.[1] as unknown as readonly CarControlOutput[];

    expect(lastControls[0].steer).not.toBe(lastControls[1].steer);
  });

  it('produces different steering from blue and red per-car observations', () => {
    const episodeState = createCurriculumEpisodeState(1);
    const blueState = derivePerCarObservationState(episodeState.envState, 0);
    const redState = derivePerCarObservationState(episodeState.envState, 1);
    const controller = createNgeController(
      createDeterministicRacingControllerNetwork(1),
      { tier: 1 },
    );

    const blueSteer = controller.computeControl(blueState, episodeState.trackSpec).steer;
    const redSteer = controller.computeControl(redState, episodeState.trackSpec).steer;

    expect(blueSteer).not.toBe(redSteer);
  });
});

async function startTier1WithControllerSpy(): Promise<{
  runHandle: { stop: () => void };
  createNgeControllerSpy: jest.Mock;
}> {
  let capturedResult:
    | { runHandle: { stop: () => void }; createNgeControllerSpy: jest.Mock }
    | undefined;

  await jest.isolateModulesAsync(async () => {
    const createNgeControllerSpy = jest.fn(
      actualNgeControllerModule.createNgeController,
    );

    jest.doMock('../controller/nge.controller', () => ({
      ...actualNgeControllerModule,
      createNgeController: createNgeControllerSpy,
    }));

    const { start: isolatedStart } = await import('./browser-entry');
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);
    const runHandle = await isolatedStart(hostElement);

    capturedResult = { runHandle, createNgeControllerSpy };
  });

  if (capturedResult === undefined) {
    throw new Error('startTier1WithControllerSpy did not capture a result');
  }

  return capturedResult;
}

async function startTier1WithStepEnvironmentSpy(): Promise<{
  runHandle: { stop: () => void };
  stepEnvironmentSpy: jest.Mock;
  rafCallback: (timestamp: number) => Promise<void>;
}> {
  let capturedResult:
    | {
        runHandle: { stop: () => void };
        stepEnvironmentSpy: jest.Mock;
        rafCallback: (timestamp: number) => Promise<void>;
      }
    | undefined;

  await jest.isolateModulesAsync(async () => {
    const stepEnvironmentSpy = jest.fn(
      actualEnvironmentStepModule.stepEnvironment,
    );

    jest.doMock('../environment/environment.step.service', () => ({
      ...actualEnvironmentStepModule,
      stepEnvironment: stepEnvironmentSpy,
    }));

    const { start: isolatedStart } = await import('./browser-entry');
    const hostElement = document.createElement('div');
    hostElement.id = 'racing-curriculum-output';
    document.body.append(hostElement);

    let capturedRafCallback:
      | ((timestamp: number) => Promise<void>)
      | undefined;
    jest
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((callback) => {
        capturedRafCallback = callback as (timestamp: number) => Promise<void>;
        return 0;
      });

    const runHandle = await isolatedStart(hostElement);
    if (capturedRafCallback === undefined) {
      throw new Error('requestAnimationFrame callback was not captured');
    }

    capturedResult = { runHandle, stepEnvironmentSpy, rafCallback: capturedRafCallback };
  });

  if (capturedResult === undefined) {
    throw new Error('startTier1WithStepEnvironmentSpy did not capture a result');
  }

  return capturedResult;
}
