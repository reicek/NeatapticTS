/**
 * Red-phase contracts for the missing deterministic race-pack factory in
 * `simulation-worker.race-pack.service.ts`.
 *
 * Contracts verified here:
 * - Same seed + same opponent snapshot → identical initial conditions
 * - Packed `race-step` frames use the shared `'racing-packed-v1'` schema
 * - Transfer-list ownership includes the expected typed-array buffers
 * - Detached buffers cannot be reused after transfer
 * - Car starting positions are not all zero (cars spread on the grid)
 *
 * All tests stay red until Step 04 implements the service boundary.
 * Single-expect rule is enforced throughout.
 *
 * TODO: NGE_TODO — When NGE EpisodicSlot and GatingRouter primitives become
 * available (upstream Phase G/E), the opponent snapshot payload format should
 * be extended to include episodic context and hard task-switch state.
 */
import type { RacingRenderFrame } from './simulation-worker.types';
import type { TrackSpec } from '../../track/track.generator.types';
import { generateTrack } from '../../track/track.generator';
import {
  resolveSplineSampleFrame,
  resolveInnerLaneCenterlinePoint,
} from '../../track/track.spline.utils';

// ---------------------------------------------------------------------------
// Locally-defined interface
// ---------------------------------------------------------------------------

type OpponentSnapshot = {
  /** Stable identifier frozen at snapshot capture time. */
  readonly snapshotId: string;
  /** Generation index at which this snapshot was captured. */
  readonly generation: number;
  /** Serialised network payloads for opponent controllers. */
  readonly networkPayloads: readonly unknown[];
};

interface RacePackService {
  /**
   * Constructs an initial race frame deterministically from a seed and a
   * frozen opponent snapshot.  Identical inputs must return identical frames.
   */
  createDeterministicRacePack(
    seed: number,
    opponentSnapshot: OpponentSnapshot,
  ): RacingRenderFrame;

  /**
   * Collects every ArrayBuffer backing a typed-array field in the frame into a
   * transfer list suitable for zero-copy postMessage transfer.
   * Mirrors the existing `resolveRacingRenderFrameTransferList` contract but is
   * owned by this service boundary.
   */
  resolveRaceStepTransferList(frame: RacingRenderFrame): ArrayBuffer[];

  /**
   * Builds a runnable episode whose `tick()` advances physics and runs one
   * controller inference per car per tick.
   */
  createRaceEpisodeRunner(
    seed: number,
    opponentSnapshot: OpponentSnapshot,
    networks: readonly RaceControllerNetwork[],
  ): RaceEpisodeRunner;
}

/**
 * Minimal controller handle used inside a race episode runner.
 * The runner only needs an `activate(inputs)` function; real networks are
 * wrapped by the caller.
 */
type RaceControllerNetwork = {
  /** Runs inference and returns the controller's output vector. */
  activate(inputs: number[]): number[];
};

type RaceControllerNetworkSpy = RaceControllerNetwork & {
  /** Jest mock used to assert inference calls in red-phase contracts. */
  activate: jest.Mock<number[], [number[]]>;
};

/**
 * Mutable episode state returned by `createRaceEpisodeRunner`.
 * `frame` exposes the current packed render frame; `tick()` advances it.
 */
interface RaceEpisodeRunner {
  /** Current packed render frame. */
  readonly frame: RacingRenderFrame;
  /** Advances the physics state by one fixed timestep and runs inference. */
  tick(): void;
}

// ---------------------------------------------------------------------------
// Module loader
// ---------------------------------------------------------------------------

async function loadRacePackService(): Promise<RacePackService> {
  const modulePath = './simulation-worker.race-pack.service';
  return (await import(modulePath)) as RacePackService;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeMinimalOpponentSnapshot(): OpponentSnapshot {
  return {
    snapshotId: 'snap-test-0',
    generation: 1,
    networkPayloads: [],
  };
}

function makeSpyRaceNetworks(count: number): RaceControllerNetworkSpy[] {
  return Array.from({ length: count }, () => ({
    activate: jest.fn<number[], [number[]]>().mockReturnValue([0, 0]),
  }));
}

// ---------------------------------------------------------------------------
// Red tests
// ---------------------------------------------------------------------------

describe('simulation worker deterministic race-pack service', () => {
  describe('createDeterministicRacePack determinism', () => {
    it('returns identical car-X positions for identical seed and opponent snapshot', async () => {
      // Arrange
      const service = await loadRacePackService();
      const snapshot = makeMinimalOpponentSnapshot();

      // Act
      const packA = service.createDeterministicRacePack(42, snapshot);
      const packB = service.createDeterministicRacePack(42, snapshot);

      // Assert — same seed + same snapshot must produce identical initial car positions
      expect(Array.from(packA.carX)).toEqual(Array.from(packB.carX));
    });

    it('places at least one car at a non-zero X position to confirm grid spread', async () => {
      // Arrange
      const service = await loadRacePackService();
      const snapshot = makeMinimalOpponentSnapshot();

      // Act
      const pack = service.createDeterministicRacePack(99, snapshot);

      // Assert — cars must not all start at the origin
      expect(pack.carX.some((position) => position !== 0)).toBe(true);
    });

    it('emits the shared racing-packed-v1 schema sentinel for packed race-step frames', async () => {
      // Arrange
      const service = await loadRacePackService();
      const snapshot = makeMinimalOpponentSnapshot();

      // Act
      const pack = service.createDeterministicRacePack(11, snapshot);

      // Assert
      expect(pack.schemaVersion).toBe('racing-packed-v1');
    });
  });

  describe('resolveRaceStepTransferList buffer ownership', () => {
    it('includes the expected typed-array buffers exactly once for a packed race-step frame', async () => {
      // Arrange
      const service = await loadRacePackService();
      const pack = service.createDeterministicRacePack(
        7,
        makeMinimalOpponentSnapshot(),
      );

      // Act
      const transferList = service.resolveRaceStepTransferList(pack);
      const uniqueBufferCount = new Set(transferList).size;

      // Assert — ownership contract requires every typed-array buffer exactly once
      expect({
        containsCarXBuffer: transferList.includes(
          pack.carX.buffer as ArrayBuffer,
        ),
        entryCount: transferList.length,
        uniqueBufferCount,
      }).toEqual({
        containsCarXBuffer: true,
        entryCount: 10,
        uniqueBufferCount: 10,
      });
    });

    it('detaches the carX buffer after a simulated transfer using the resolved list', async () => {
      // Arrange
      const service = await loadRacePackService();
      const pack = service.createDeterministicRacePack(
        7,
        makeMinimalOpponentSnapshot(),
      );
      const transferList = service.resolveRaceStepTransferList(pack);

      // Act
      structuredClone(pack, { transfer: transferList });

      // Assert
      expect(pack.carX.byteLength).toBe(0);
    });
  });

  describe('createRaceEpisodeRunner episode tick contract', () => {
    it('exists as a function exported from the race-pack service', async () => {
      // Arrange
      const service = await loadRacePackService();

      // Assert
      expect(typeof service.createRaceEpisodeRunner).toBe('function');
    });

    it('advances the episode tick counter when tick() is called', async () => {
      // Arrange
      const service = await loadRacePackService();
      const networks = makeSpyRaceNetworks(4);
      const runner = service.createRaceEpisodeRunner(
        42,
        makeMinimalOpponentSnapshot(),
        networks,
      );
      const tickBefore = runner.frame.tick;

      // Act
      runner.tick();

      // Assert
      expect(runner.frame.tick).toBeGreaterThan(tickBefore);
    });

    it('activates a network for every car during tick()', async () => {
      // Arrange
      const service = await loadRacePackService();
      const networks = makeSpyRaceNetworks(4);
      const runner = service.createRaceEpisodeRunner(
        42,
        makeMinimalOpponentSnapshot(),
        networks,
      );

      // Act
      runner.tick();

      // Assert
      expect(
        networks.every((network) => network.activate.mock.calls.length === 1),
      ).toBe(true);
    });
  });
});

describe('per-agent guiding line state', () => {
  it('attaches a guidingLines array with one entry per car to the runner', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );
    const runnerWithGuidingLines = runner as unknown as {
      guidingLines?: unknown[];
    };
    const guidingLines = runnerWithGuidingLines.guidingLines;
    const isValidArray =
      Array.isArray(guidingLines) &&
      guidingLines.length === runner.frame.agentCount;

    expect(isValidArray).toBe(true);
  });

  it('positions each guiding line start point at the matching car start position', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );
    const runnerWithGuidingLines = runner as unknown as {
      guidingLines?: Array<Array<{ readonly x: number; readonly y: number }>>;
    };
    const guidingLines = runnerWithGuidingLines.guidingLines ?? [];

    let allStartPositionsValid = true;
    for (let carIndex = 0; carIndex < runner.frame.agentCount; carIndex++) {
      const carGuidingLine = guidingLines[carIndex];
      if (!Array.isArray(carGuidingLine) || carGuidingLine.length === 0) {
        allStartPositionsValid = false;
        break;
      }
      const startPoint = carGuidingLine[0];
      if (startPoint === undefined) {
        allStartPositionsValid = false;
        break;
      }
      const expectedX = runner.frame.carX[carIndex];
      const expectedY = runner.frame.carY[carIndex];
      const distance = Math.hypot(
        startPoint.x - expectedX,
        startPoint.y - expectedY,
      );
      if (distance >= 1e-6) {
        allStartPositionsValid = false;
        break;
      }
    }

    expect(allStartPositionsValid).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tier 1 deterministic constants (pinned by Step 03 red tests)
// ---------------------------------------------------------------------------

const MAX_EPISODE_TICKS = 1800 as const;
const OFF_TRACK_GRACE_TICKS = 60 as const;
const COMPLETION_BONUS = 2000 as const;
const PROGRESS_WEIGHT = 0.5 as const;
const OFF_TRACK_PENALTY = 500 as const;

// ---------------------------------------------------------------------------
// Tier 1 deterministic fixtures and helpers
// ---------------------------------------------------------------------------

function makeTier1Track(seed: number): TrackSpec {
  return generateTrack({ seed, layoutVersion: 1, sizeBucket: 'medium' });
}

function resolveTier1InnerLaneStart(track: TrackSpec): {
  x: number;
  y: number;
  heading: number;
} {
  const startSample = track.splineSamples[0];
  const sampleFrame = resolveSplineSampleFrame(track.splineSamples, 0);
  const point = resolveInnerLaneCenterlinePoint(startSample, sampleFrame);

  return {
    x: point.x,
    y: point.y,
    heading: sampleFrame.tangentHeadingRadians,
  };
}

function makeFullThrottleNetworks(count: number): RaceControllerNetwork[] {
  return Array.from({ length: count }, () => ({
    activate: () => [1, 0],
  }));
}

function getRunnerFitness(
  runner: unknown,
  carIndex: number,
): number | undefined {
  const runnerWithFitness = runner as {
    computeFitness?: (index: number) => number;
  };
  if (typeof runnerWithFitness.computeFitness === 'function') {
    return runnerWithFitness.computeFitness(carIndex);
  }
  return undefined;
}

function createRaceStepMessageOrUndefined(runner: unknown): unknown {
  const runnerWithMessage = runner as {
    createRaceStepMessage?: () => unknown;
  };
  if (typeof runnerWithMessage.createRaceStepMessage === 'function') {
    return runnerWithMessage.createRaceStepMessage();
  }
  return undefined;
}

type FitnessState = {
  progress01: number[];
  lapCompleted: number[];
  lapTimeTicks?: number[];
  endedOffTrack: boolean;
};

function setFitnessState(runner: unknown, state: FitnessState): void {
  const runnerRecord = runner as Record<string, unknown>;
  const frameRecord = (runner as { frame: Record<string, unknown> }).frame;

  frameRecord.progress01 = new Float32Array(state.progress01);
  runnerRecord.lapCompleted = new Uint8Array(state.lapCompleted);
  runnerRecord.lapTimeTicks =
    state.lapTimeTicks === undefined
      ? new Uint32Array(state.progress01.length)
      : new Uint32Array(state.lapTimeTicks);
  runnerRecord.endedOffTrack = state.endedOffTrack;
}

// ---------------------------------------------------------------------------
// Tier 1 red tests — race-pack factory / setup
// ---------------------------------------------------------------------------

describe('Tier 1 race-pack factory / setup', () => {
  it('creates a runner with exactly two active cars', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    expect(runner.frame.agentCount).toBe(2);
  });

  it('marks both cars as active in the initial frame', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    expect(Array.from(runner.frame.carActive)).toEqual([1, 1]);
  });

  it('assigns Team A to car 0 and Team B to car 1', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    expect(Array.from(runner.frame.carTeam)).toEqual([0, 1]);
  });

  it('places both cars on the inner-lane centerline of the generated simple track', async () => {
    const service = await loadRacePackService();
    const seed = 42;
    const track = makeTier1Track(seed);
    const startPoint = resolveTier1InnerLaneStart(track);
    const runner = service.createRaceEpisodeRunner(
      seed,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    let maxDistance = 0;
    for (let carIndex = 0; carIndex < runner.frame.agentCount; carIndex++) {
      const positionDistance = Math.hypot(
        runner.frame.carX[carIndex] - startPoint.x,
        runner.frame.carY[carIndex] - startPoint.y,
      );
      const headingDelta = Math.abs(
        runner.frame.carHeading[carIndex] - startPoint.heading,
      );
      const wrappedHeadingDelta = Math.min(
        headingDelta,
        2 * Math.PI - headingDelta,
      );
      maxDistance = Math.max(
        maxDistance,
        positionDistance + wrappedHeadingDelta,
      );
    }

    expect(maxDistance).toBeLessThan(0.01);
  });

  it('sets frame.trackId to the deterministic track seed', async () => {
    const service = await loadRacePackService();
    const seed = 42;
    const runner = service.createRaceEpisodeRunner(
      seed,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    expect(runner.frame.trackId).toBe(seed);
  });
});

// ---------------------------------------------------------------------------
// Tier 1 red tests — episode runner tick loop
// ---------------------------------------------------------------------------

describe('Tier 1 episode runner tick loop', () => {
  it('advances car positions along the track after a single tick', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );
    const carXBefore = Array.from(runner.frame.carX);
    const carYBefore = Array.from(runner.frame.carY);

    runner.tick();

    let maxDistance = 0;
    for (let carIndex = 0; carIndex < runner.frame.agentCount; carIndex++) {
      const distance = Math.hypot(
        runner.frame.carX[carIndex] - carXBefore[carIndex],
        runner.frame.carY[carIndex] - carYBefore[carIndex],
      );
      maxDistance = Math.max(maxDistance, distance);
    }

    expect(maxDistance).toBeGreaterThan(0);
  });

  it('terminates the episode after consecutive off-track ticks exceed the grace budget', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    for (let tickIndex = 0; tickIndex <= OFF_TRACK_GRACE_TICKS; tickIndex++) {
      (runner.frame.carX as Float32Array)[0] = 9999;
      (runner.frame.carY as Float32Array)[0] = 9999;
      runner.tick();
    }

    expect(runner.frame.done).toBe(true);
  });

  it('terminates the episode when the max tick budget is reached', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    for (let tickIndex = 0; tickIndex < MAX_EPISODE_TICKS; tickIndex++) {
      runner.tick();
    }

    expect(runner.frame.done).toBe(true);
  });

  it('reports per-car progress01 inside the closed unit interval after a tick', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    runner.tick();

    const frameWithProgress = runner.frame as unknown as {
      progress01?: Float32Array;
    };
    const progress01 = frameWithProgress.progress01;
    const valuesAreInRange =
      progress01 !== undefined &&
      progress01.length === runner.frame.agentCount &&
      Array.from(progress01).every((value) => value >= 0 && value <= 1);

    expect(valuesAreInRange).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tier 1 red tests — lap detection
// ---------------------------------------------------------------------------

describe('Tier 1 lap detection', () => {
  it('increments frame.lap when the car crosses the start/finish line', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    for (let tickIndex = 0; tickIndex < MAX_EPISODE_TICKS; tickIndex++) {
      runner.tick();
      if (runner.frame.lap[0] > 0) {
        break;
      }
    }

    expect(runner.frame.lap[0]).toBeGreaterThan(0);
  });

  it('sets a per-car lapCompleted flag after a lap is finished', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    for (let tickIndex = 0; tickIndex < MAX_EPISODE_TICKS; tickIndex++) {
      runner.tick();
      const lapCompleted = (runner as unknown as { lapCompleted?: Uint8Array })
        .lapCompleted;
      if (lapCompleted !== undefined && lapCompleted[0] === 1) {
        break;
      }
    }

    const lapCompleted = (runner as unknown as { lapCompleted?: Uint8Array })
      .lapCompleted;
    expect(lapCompleted !== undefined && lapCompleted[0] === 1).toBe(true);
  });

  it('records lapTimeTicks when a lap is completed', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    for (let tickIndex = 0; tickIndex < MAX_EPISODE_TICKS; tickIndex++) {
      runner.tick();
      const lapTimeTicks = (runner as unknown as { lapTimeTicks?: Uint32Array })
        .lapTimeTicks;
      if (lapTimeTicks !== undefined && lapTimeTicks[0] > 0) {
        break;
      }
    }

    const lapTimeTicks = (runner as unknown as { lapTimeTicks?: Uint32Array })
      .lapTimeTicks;
    expect(lapTimeTicks !== undefined && lapTimeTicks[0] > 0).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tier 1 red tests — lap-time fitness
// ---------------------------------------------------------------------------

describe('Tier 1 lap-time fitness', () => {
  it('assigns higher fitness to a completed lap than to an incomplete episode', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    setFitnessState(runner, {
      progress01: [0.5, 0.25],
      lapCompleted: [1, 0],
      lapTimeTicks: [600, 0],
      endedOffTrack: false,
    });

    const completedFitness = getRunnerFitness(runner, 0);
    const incompleteFitness = getRunnerFitness(runner, 1);

    expect(
      (completedFitness ?? Number.NEGATIVE_INFINITY) >
        (incompleteFitness ?? Number.NEGATIVE_INFINITY),
    ).toBe(true);
  });

  it('adds the completion bonus and rewards lower lap times', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    setFitnessState(runner, {
      progress01: [0.5, 0.5],
      lapCompleted: [1, 1],
      lapTimeTicks: [600, 1200],
      endedOffTrack: false,
    });

    const fasterFitness = getRunnerFitness(runner, 0);
    const slowerFitness = getRunnerFitness(runner, 1);
    const expectedDifference =
      COMPLETION_BONUS +
      (MAX_EPISODE_TICKS - 600) -
      (COMPLETION_BONUS + (MAX_EPISODE_TICKS - 1200));

    expect((fasterFitness ?? NaN) - (slowerFitness ?? NaN)).toBe(
      expectedDifference,
    );
  });

  it('applies the off-track penalty to incomplete fitness when the episode ends off-track', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    setFitnessState(runner, {
      progress01: [0.5, 0.5],
      lapCompleted: [0, 0],
      endedOffTrack: true,
    });

    const penalisedFitness = getRunnerFitness(runner, 0);
    const expectedPenalisedFitness =
      PROGRESS_WEIGHT * 0.5 * MAX_EPISODE_TICKS - OFF_TRACK_PENALTY;

    expect(penalisedFitness).toBe(expectedPenalisedFitness);
  });

  it('returns deterministic fitness for identical genomes and track seeds', async () => {
    const service = await loadRacePackService();
    const snapshot = makeMinimalOpponentSnapshot();
    const networks = makeFullThrottleNetworks(2);
    const runnerA = service.createRaceEpisodeRunner(7, snapshot, networks);
    const runnerB = service.createRaceEpisodeRunner(7, snapshot, networks);

    setFitnessState(runnerA, {
      progress01: [0.4, 0],
      lapCompleted: [0, 0],
      endedOffTrack: true,
    });
    setFitnessState(runnerB, {
      progress01: [0.4, 0],
      lapCompleted: [0, 0],
      endedOffTrack: true,
    });

    const fitnessA = getRunnerFitness(runnerA, 0);
    const fitnessB = getRunnerFitness(runnerB, 0);

    expect(fitnessA !== undefined && fitnessA === fitnessB).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Tier 1 red tests — physics hardening (worker fitness and car separation)
// ---------------------------------------------------------------------------

describe('Tier 1 physics hardening — worker fitness and car separation', () => {
  it('does not penalize the on-track car when only the other car leaves the track', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    // Place car 0 far off track and leave car 1 on the centerline.
    (runner.frame.carX as Float32Array)[0] = 9999;
    (runner.frame.carY as Float32Array)[0] = 9999;

    for (let tickIndex = 0; tickIndex <= OFF_TRACK_GRACE_TICKS; tickIndex++) {
      runner.tick();
    }

    const onTrackFitness = getRunnerFitness(runner, 1);

    expect(onTrackFitness).toBeGreaterThan(0);
  });

  it('penalizes the car that leaves the track', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    (runner.frame.carX as Float32Array)[0] = 9999;
    (runner.frame.carY as Float32Array)[0] = 9999;

    for (let tickIndex = 0; tickIndex <= OFF_TRACK_GRACE_TICKS; tickIndex++) {
      runner.tick();
    }

    const offTrackFitness = getRunnerFitness(runner, 0);

    expect(offTrackFitness).toBe(-500);
  });

  it('pushes overlapping cars apart during a tick', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );

    // Place both cars at the same starting position.
    (runner.frame.carX as Float32Array)[1] = runner.frame.carX[0];
    (runner.frame.carY as Float32Array)[1] = runner.frame.carY[0];

    runner.tick();

    const separation = Math.hypot(
      runner.frame.carX[0] - runner.frame.carX[1],
      runner.frame.carY[0] - runner.frame.carY[1],
    );

    expect(separation).toBeGreaterThan(1e-6);
  });
});

describe('Tier 1 worker race-step message', () => {
  it('produces a message whose type is race-step', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );
    const message = createRaceStepMessageOrUndefined(runner) as
      { type?: string } | undefined;

    expect(message?.type).toBe('race-step');
  });

  it('includes the packed frame with car positions, progress, lap counts, and done flag', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );
    const message = createRaceStepMessageOrUndefined(runner) as
      { frame?: RacingRenderFrame & { progress01?: Float32Array } } | undefined;
    const frame = message?.frame;
    const hasRequiredFields =
      frame !== undefined &&
      frame.carX !== undefined &&
      frame.carY !== undefined &&
      frame.lap !== undefined &&
      frame.done !== undefined &&
      frame.progress01 !== undefined;

    expect(hasRequiredFields).toBe(true);
  });

  it('includes a transfer list with the car position buffers', async () => {
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(2),
    );
    const message = createRaceStepMessageOrUndefined(runner) as
      { frame?: RacingRenderFrame; transferList?: ArrayBuffer[] } | undefined;
    const transferList = message?.transferList ?? [];
    const carXBuffer = message?.frame?.carX.buffer as ArrayBuffer | undefined;

    expect(carXBuffer !== undefined && transferList.includes(carXBuffer)).toBe(
      true,
    );
  });
});

// ---------------------------------------------------------------------------
// Tier 3 red tests — shared-equal team-fitness and four-car team layout
// ---------------------------------------------------------------------------

describe('Tier 3 shared-equal team-fitness aggregation', () => {
  it('resolves team fitness as the average of member finishing positions for a two-car team', async () => {
    // Arrange
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );

    // Act — car 0 finishes at position 3, car 1 finishes at position 7
    // Shared-equal: (3 + 7) / 2 = 5
    // Best-position (current): min(3, 7) = 3
    const runnerWithTeamFitness = runner as unknown as {
      resolveTeamFitness?: (
        teamId: 0 | 1,
        carFinishPositions: readonly number[],
      ) => number;
    };

    // Assert — must be 5 (average), not 3 (min)
    expect(runnerWithTeamFitness.resolveTeamFitness?.(0, [3, 7])).toBe(5);
  });

  it('does not use best (minimum) finishing position for team fitness', async () => {
    // Arrange
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );

    // Act — positions 1 and 9: average = 5, best = 1
    const runnerWithTeamFitness = runner as unknown as {
      resolveTeamFitness?: (
        teamId: 0 | 1,
        carFinishPositions: readonly number[],
      ) => number;
    };

    // Assert — must be 5 (average), not 1 (best/min)
    expect(runnerWithTeamFitness.resolveTeamFitness?.(0, [1, 9])).toBe(5);
  });
});

describe('Tier 3 four-car team layout', () => {
  it('assigns team layout [0, 0, 1, 1] for four cars', async () => {
    // Arrange
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );

    // Act
    const teamLayout = Array.from(runner.frame.carTeam);

    // Assert — cars 0-1 = Team A, cars 2-3 = Team B
    expect(teamLayout).toEqual([0, 0, 1, 1]);
  });

  it('creates a runner with four active cars when four networks are provided', async () => {
    // Arrange
    const service = await loadRacePackService();

    // Act
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );

    // Assert
    expect(runner.frame.agentCount).toBe(4);
  });
});

// ---------------------------------------------------------------------------
// Tier 4 red tests — tire and pit coevolution contracts
// ---------------------------------------------------------------------------

/**
 * Resolves the world-space center of a Team B pit entrance corridor from the
 * track spec.  Used to position a Team A car at the opposing team's pit for
 * the exclusion contract.
 */
function resolveTeamBPitEntranceCenter(track: TrackSpec): {
  x: number;
  y: number;
} {
  const teamBPitBox = track.pitBoxes?.find((box) => box.teamIndex === 1);
  if (!teamBPitBox) {
    throw new Error('No Team B pit box found in track spec');
  }
  const corridor = teamBPitBox.entranceCorridor;
  return {
    x: corridor.x + corridor.width / 2,
    y: corridor.y + corridor.height / 2,
  };
}

describe('Tier 4 tire and pit coevolution contracts', () => {
  it('feeds a 103-channel observation vector to each car controller during tick', async () => {
    // Arrange
    const service = await loadRacePackService();
    const networks = makeSpyRaceNetworks(4);
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      networks,
    );

    // Act
    runner.tick();

    // Assert — Tier 4 must feed 103 channels (91 Tier 3 + 4 tire + 8 pit/strategy), not 5
    const firstCallInput = networks[0].activate.mock.calls[0]?.[0];
    expect(firstCallInput?.length).toBe(103);
  });

  it('includes own-car tire health in the observation tail channels 91 through 94', async () => {
    // Arrange
    const service = await loadRacePackService();
    const networks = makeSpyRaceNetworks(4);
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      networks,
    );
    // Set car 0 tire state to known distinct values
    const tireState = runner.frame.tireState as Float32Array;
    tireState[0] = 0.8; // FL
    tireState[1] = 0.7; // FR
    tireState[2] = 0.6; // RL
    tireState[3] = 0.5; // RR

    // Act
    runner.tick();

    // Assert — channels 91–94 must match the car's tire health [FL, FR, RL, RR]
    const input = networks[0].activate.mock.calls[0]?.[0];
    expect(
      [input?.[91], input?.[92], input?.[93], input?.[94]].map((v) =>
        Number((v ?? 0).toFixed(6)),
      ),
    ).toEqual([0.8, 0.7, 0.6, 0.5]);
  });

  it('decays tire state over a full-throttle race episode', async () => {
    // Arrange
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );
    // Initialize car 0 tires to fresh (1.0)
    const tireState = runner.frame.tireState as Float32Array;
    tireState[0] = 1.0;
    tireState[1] = 1.0;
    tireState[2] = 1.0;
    tireState[3] = 1.0;

    // Act — run 120 ticks at full throttle
    for (let tickIndex = 0; tickIndex < 120; tickIndex++) {
      runner.tick();
    }

    // Assert — tire state must decrease from 1.0 due to driving forces
    expect(runner.frame.tireState[0]).toBeLessThan(1.0);
  });

  it('reduces forward progress for worn tires versus fresh tires at the same throttle', async () => {
    // Arrange — two identical runners, same seed and networks
    const service = await loadRacePackService();
    const snapshot = makeMinimalOpponentSnapshot();
    const freshRunner = service.createRaceEpisodeRunner(
      42,
      snapshot,
      makeFullThrottleNetworks(4),
    );
    const wornRunner = service.createRaceEpisodeRunner(
      42,
      snapshot,
      makeFullThrottleNetworks(4),
    );
    // Fresh tires = 1.0, worn tires = 0.3
    const freshTires = freshRunner.frame.tireState as Float32Array;
    freshTires[0] = 1.0;
    freshTires[1] = 1.0;
    freshTires[2] = 1.0;
    freshTires[3] = 1.0;
    const wornTires = wornRunner.frame.tireState as Float32Array;
    wornTires[0] = 0.3;
    wornTires[1] = 0.3;
    wornTires[2] = 0.3;
    wornTires[3] = 0.3;

    // Act — run 60 ticks on each
    for (let tickIndex = 0; tickIndex < 60; tickIndex++) {
      freshRunner.tick();
      wornRunner.tick();
    }
    const freshProgress =
      (
        freshRunner.frame as unknown as {
          progress01?: Float32Array;
        }
      ).progress01?.[0] ?? 0;
    const wornProgress =
      (
        wornRunner.frame as unknown as {
          progress01?: Float32Array;
        }
      ).progress01?.[0] ?? 0;

    // Assert — fresh tires must achieve more progress than worn tires
    expect(freshProgress).toBeGreaterThan(wornProgress);
  });

  it('exposes a pitStatus field on the race episode frame for a four-car pack', async () => {
    // Arrange
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );

    // Assert — Tier 4 four-car frame must include a pitStatus typed array
    expect(runner.frame.pitStatus).toBeDefined();
  });

  it('initializes all pit car slots to the no-car sentinel at episode start', async () => {
    // Arrange
    const service = await loadRacePackService();
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );
    const pitStatus = runner.frame.pitStatus;

    // Assert — both team car slots (index 0 = Team A, index 2 = Team B) must be 255
    expect(
      pitStatus !== undefined && pitStatus[0] === 255 && pitStatus[2] === 255,
    ).toBe(true);
  });

  it('does not assign a Team A car to a Team B pit slot when the Team A car is at the Team B pit entrance', async () => {
    // Arrange
    const service = await loadRacePackService();
    const track = makeTier1Track(42);
    const teamBPitCenter = resolveTeamBPitEntranceCenter(track);
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      makeFullThrottleNetworks(4),
    );
    // Position car 0 (Team A) at the Team B pit entrance corridor center
    (runner.frame.carX as Float32Array)[0] = teamBPitCenter.x;
    (runner.frame.carY as Float32Array)[0] = teamBPitCenter.y;

    // Act
    runner.tick();

    // Assert — Team B pit car slot (index 2) must remain unoccupied (255)
    expect(runner.frame.pitStatus?.[2]).toBe(255);
  });
});

describe('pit/strategy observation channels', () => {
  it('feeds a 103-channel observation vector to each car controller during tick', async () => {
    // Arrange
    const service = await loadRacePackService();
    const networks = makeSpyRaceNetworks(2);
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      networks,
    );

    // Act
    runner.tick();

    // Assert
    const firstCallInput = networks[0].activate.mock.calls[0]?.[0];
    expect(firstCallInput?.length).toBe(103);
  });

  it('exposes defined pit/strategy values at observation offsets 95 through 102', async () => {
    // Arrange
    const service = await loadRacePackService();
    const networks = makeSpyRaceNetworks(2);
    const runner = service.createRaceEpisodeRunner(
      42,
      makeMinimalOpponentSnapshot(),
      networks,
    );

    // Act
    runner.tick();

    // Assert
    const input = networks[0].activate.mock.calls[0]?.[0];
    const values = [95, 96, 97, 98, 99, 100, 101, 102].map(
      (offset) => input?.[offset],
    );
    expect(values).toEqual([
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
      expect.any(Number),
    ]);
  });
});
