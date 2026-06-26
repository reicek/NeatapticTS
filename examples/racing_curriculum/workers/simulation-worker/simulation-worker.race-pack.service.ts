import type { RacingRenderFrame } from './simulation-worker.types';
import type { TrackSpec } from '../../track/track.generator.types';
import type { SplineSample } from '../../track/track.generator.types';
import { generateTrack } from '../../track/track.generator';
import {
  resolveSplineSampleFrame,
  resolveInnerLaneCenterlinePoint,
} from '../../track/track.spline.utils';
import { buildGuidingLineForTeam } from '../../renderer/racing.renderer';

/** Schema sentinel for all packed race-step frames. */
const RACING_SCHEMA_VERSION = 'racing-packed-v1' as const;

/** Default agent count for a deterministic pack without controller networks. */
const DEFAULT_AGENT_COUNT = 2 as const;

/** Maximum number of fixed-timestep ticks in a single episode. */
const MAX_EPISODE_TICKS = 1800 as const;

/** Consecutive off-track ticks allowed before the episode is terminated. */
const OFF_TRACK_GRACE_TICKS = 60 as const;

/** Fitness bonus awarded for completing at least one lap. */
const COMPLETION_BONUS = 2000 as const;

/** Weight applied to incomplete-episode progress fitness. */
const PROGRESS_WEIGHT = 0.5 as const;

/** Fitness penalty applied when the episode ends off-track. */
const OFF_TRACK_PENALTY = 500 as const;

/** Minimum center-to-center distance between two cars after separation. */
const CAR_MIN_CENTER_SEPARATION = 2.5 as const;

/** Fixed physics timestep in seconds (60 Hz). */
const FIXED_TIMESTEP_SECONDS = 1 / 60;

/** Maximum forward speed in logical world units per second. */
const MAX_FORWARD_SPEED_UNITS_PER_SECOND = 108 as const;

/** Expected number of distinct ArrayBuffer entries in a Tier-0 transfer list. */
const EXPECTED_TRANSFER_BUFFER_COUNT = 10 as const;

/**
 * Frozen opponent snapshot used as deterministic race-pack input.
 *
 * Identical seed + identical snapshot produce identical starting frames, which
 * makes race episodes replayable and comparative fitness claims fair.  The
 * zero-copy transfer path uses `ArrayBuffer` transfer lists supported by Web
 * Workers; see [Transferable objects (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferring_objects)
 * for details.
 *
 * Extension point:
 * - Extend the snapshot payload format to include episodic context and hard
 *   task-switch state when `EpisodicSlot` and `GatingRouter` primitives are
 *   available.
 */
export type OpponentSnapshot = {
  /** Stable identifier frozen at snapshot capture time. */
  readonly snapshotId: string;
  /** Generation index at which this snapshot was captured. */
  readonly generation: number;
  /** Serialised network payloads for opponent controllers. */
  readonly networkPayloads: readonly unknown[];
};

/** Minimal controller handle used inside a race episode runner. */
export type RaceControllerNetwork = {
  /** Runs inference and returns the controller's output vector. */
  activate(inputs: number[]): number[];
};

/**
 * Packed render frame augmented with the per-car unit-progress field used by the
 * race-pack runner. The `progress01` buffer is intentionally not part of the
 * zero-copy transfer list; it is computed locally and kept attached to the
 * runner frame.
 */
type RaceEpisodeRunnerFrame = RacingRenderFrame & {
  /** Unit progress [0, 1] along the current lap for each car. */
  progress01: Float32Array;
};

/**
 * Mutable episode state returned by `createRaceEpisodeRunner`.
 *
 * `frame` exposes the current packed render frame; `tick()` advances it by one
 * fixed timestep and runs one controller inference per car.
 */
export type RaceEpisodeRunner = {
  /** Current packed render frame including local-only `progress01`. */
  readonly frame: RaceEpisodeRunnerFrame;
  /** Advances the physics state by one fixed timestep and runs inference. */
  tick(): void;
  /** Computes the lap-time fitness for the requested car. */
  computeFitness(carIndex: number): number;
  /**
   * Builds a `race-step` worker message for the current frame.
   *
   * @returns Message object and the transfer list that owns its typed-array buffers.
   */
  createRaceStepMessage(): {
    type: 'race-step';
    frame: RacingRenderFrame;
    transferList: ArrayBuffer[];
  };
  /** Per-car lap-completion flag (1 = completed, 0 = not yet). */
  lapCompleted: Uint8Array;
  /** Per-car tick index at which the first lap was completed. */
  lapTimeTicks: Uint32Array;
  /** True when the episode was terminated because a car left the track. */
  endedOffTrack: boolean;
  /**
   * Per-car off-track flags; `1` for the car that triggered the episode end.
   * Used by computeFitness so only the offending car pays the penalty.
   */
  endedOffTrackPerCar: Uint8Array;
  /**
   * Per-car guiding line points used by the host renderer to draw a dedicated
   * lane marker for each agent.
   */
  guidingLines: Array<readonly { readonly x: number; readonly y: number }[]>;
};

type BuildRaceFrameOptions = {
  seed: number;
  trackId: number;
  featureFlags: number;
  agentCount: number;
  track: TrackSpec;
  centerline: TrackCenterline;
};

/**
 * Constructs an initial race frame deterministically from a seed and a frozen
 * opponent snapshot.  Identical seed + identical snapshot → identical frame.
 *
 * All cars start on the inner-lane centerline of the deterministic medium
 * simple track so that full-throttle episodes produce comparable lap times.
 *
 * @param seed - Deterministic race-pack seed.
 * @param _opponentSnapshot - Frozen opponent snapshot for the episode (reserved
 *   for future episodic context wiring; currently unused for determinism).
 * @returns Packed `RacingRenderFrame` with `schemaVersion: 'racing-packed-v1'`.
 *
 * @example
 * ```ts
 * const packA = createDeterministicRacePack(42, snapshot);
 * const packB = createDeterministicRacePack(42, snapshot);
 * // Array.from(packA.carX) deepEquals Array.from(packB.carX)
 * ```
 */
export function createDeterministicRacePack(
  seed: number,
  _opponentSnapshot: OpponentSnapshot,
): RacingRenderFrame {
  void _opponentSnapshot;
  const track = generateTrack({ seed, layoutVersion: 1, sizeBucket: 'medium' });
  const centerline = buildTrackCenterline(track.splineSamples);

  return buildRaceFrame({
    seed,
    trackId: seed,
    featureFlags: 0,
    agentCount: DEFAULT_AGENT_COUNT,
    track,
    centerline,
  });
}
/**
 * Collects every `ArrayBuffer` backing a typed-array field in the frame into a
 * transfer list for zero-copy `postMessage` transfer.
 *
 * Mirrors `resolveRacingRenderFrameTransferList` from the snapshot utils but is
 * owned by this service boundary so race-step streaming follows the same
 * zero-copy ownership contract.
 *
 * Rules:
 * - Every typed-array field contributes exactly one buffer entry.
 * - Shared buffers are deduplicated (listed only once).
 * - A standard pack without `pitStatus` produces exactly
 *   {@link EXPECTED_TRANSFER_BUFFER_COUNT} entries.
 * - The local-only `progress01` field is never transferred.
 *
 * @param frame - Packed render frame whose buffers will be transferred.
 * @returns Ordered list of `ArrayBuffer` references for postMessage transfer.
 *
 * @example
 * ```ts
 * const transferList = resolveRaceStepTransferList(pack);
 * worker.postMessage({ type: 'race-step', pack }, transferList);
 * ```
 */
export function resolveRaceStepTransferList(
  frame: RacingRenderFrame,
): ArrayBuffer[] {
  const transferList: ArrayBuffer[] = [];
  const seenBuffers = new Set<ArrayBuffer>();

  const typedArrayFields = [
    frame.carX,
    frame.carY,
    frame.carHeading,
    frame.carActive,
    frame.carTeam,
    frame.carMode,
    frame.tireState,
    frame.radioField,
    frame.lap,
    frame.place,
    ...(frame.pitStatus !== undefined ? [frame.pitStatus] : []),
  ];

  for (const typedArray of typedArrayFields) {
    const buffer = typedArray.buffer as ArrayBuffer;

    if (seenBuffers.has(buffer)) {
      continue;
    }

    seenBuffers.add(buffer);
    transferList.push(buffer);
  }

  return transferList;
}

/**
 * Builds a runnable race episode whose `tick()` advances physics and runs one
 * controller inference per car per tick.
 *
 * The runner owns a deterministic initial frame produced from the same medium
 * simple-track generator used by {@link createDeterministicRacePack}. Each call
 * to `tick()` increments the frame tick counter, advances each car along the
 * inner-lane centerline, detects lap completion, and invokes every provided
 * network exactly once.
 *
 * @param seed - Deterministic race-pack seed.
 * @param _opponentSnapshot - Frozen opponent snapshot for the episode (reserved
 *   for future episodic context wiring; currently unused for determinism).
 * @param networks - One controller network per car slot.
 * @returns Runnable race episode with an initial packed frame.
 *
 * @example
 * ```ts
 * const runner = createRaceEpisodeRunner(42, snapshot, [netA, netB]);
 * runner.tick();
 * console.log(runner.frame.tick); // 1
 * ```
 */
export function createRaceEpisodeRunner(
  seed: number,
  _opponentSnapshot: OpponentSnapshot,
  networks: readonly RaceControllerNetwork[],
): RaceEpisodeRunner {
  void _opponentSnapshot;
  const agentCount = networks.length;
  const track = generateTrack({ seed, layoutVersion: 1, sizeBucket: 'medium' });
  const centerline = buildTrackCenterline(track.splineSamples);
  const trackLength = centerline.trackLength;
  const frame = buildRaceFrame({
    seed,
    trackId: seed,
    featureFlags: 0,
    agentCount,
    track,
    centerline,
  });

  const distanceAlongTrack = new Float32Array(agentCount);
  const offTrackCounter = new Int16Array(agentCount);
  const lapCompleted = new Uint8Array(agentCount);
  const lapTimeTicks = new Uint32Array(agentCount);
  const endedOffTrackPerCar = new Uint8Array(agentCount);

  const guidingLines = Array.from({ length: agentCount }, (_, carIndex) => {
    const carGuidingLine = buildGuidingLineForTeam(
      track,
      frame.carTeam[carIndex] ?? 0,
    );
    // Snap the first point to the car's actual start position so the host
    // renderer can anchor the per-agent line exactly where the car appears.
    return [
      { x: frame.carX[carIndex], y: frame.carY[carIndex] },
      ...carGuidingLine.slice(1),
    ];
  });

  const runnerState: RaceEpisodeRunner = {
    frame,
    tick,
    computeFitness,
    createRaceStepMessage,
    lapCompleted,
    lapTimeTicks,
    endedOffTrack: false,
    endedOffTrackPerCar,
    guidingLines,
  };

  return runnerState;

  function tick(): void {
    if (runnerState.frame.done) {
      return;
    }

    runnerState.frame.tick += 1;
    runnerState.frame.raceTimeMs =
      runnerState.frame.tick * FIXED_TIMESTEP_SECONDS * 1000;

    for (let carIndex = 0; carIndex < agentCount; carIndex++) {
      const network = networks[carIndex];
      if (network === undefined) {
        continue;
      }

      const controllerOutput = network.activate([
        runnerState.frame.carX[carIndex],
        runnerState.frame.carY[carIndex],
        runnerState.frame.carHeading[carIndex],
        runnerState.frame.progress01[carIndex],
        runnerState.frame.tick,
      ]);
      const throttle = clamp01(controllerOutput[0] ?? 0);

      const currentCenterline = resolveTrackPointAtDistance(
        distanceAlongTrack[carIndex],
        centerline,
      );
      const distanceFromCenterline = Math.hypot(
        runnerState.frame.carX[carIndex] - currentCenterline.x,
        runnerState.frame.carY[carIndex] - currentCenterline.y,
      );

      const currentSampleIndex = resolveProgressSampleIndex(
        runnerState.frame.progress01[carIndex],
        track.splineSamples.length,
      );
      const currentSample = track.splineSamples[currentSampleIndex]!;

      if (distanceFromCenterline > currentSample.width / 2) {
        offTrackCounter[carIndex] += 1;

        if (offTrackCounter[carIndex] >= OFF_TRACK_GRACE_TICKS) {
          runnerState.endedOffTrackPerCar[carIndex] = 1;
          runnerState.endedOffTrack = true;
          runnerState.frame.done = true;
        }

        continue;
      }

      offTrackCounter[carIndex] = 0;

      const forwardStep =
        throttle * MAX_FORWARD_SPEED_UNITS_PER_SECOND * FIXED_TIMESTEP_SECONDS;
      distanceAlongTrack[carIndex] += forwardStep;

      const newCenterline = resolveTrackPointAtDistance(
        distanceAlongTrack[carIndex],
        centerline,
      );

      runnerState.frame.carX[carIndex] = newCenterline.x;
      runnerState.frame.carY[carIndex] = newCenterline.y;
      runnerState.frame.carHeading[carIndex] = newCenterline.heading;

      const completedLaps = Math.floor(
        distanceAlongTrack[carIndex] / trackLength,
      );
      if (completedLaps > runnerState.frame.lap[carIndex]) {
        runnerState.frame.lap[carIndex] = completedLaps;

        if (completedLaps === 1 && runnerState.lapCompleted[carIndex] === 0) {
          runnerState.lapCompleted[carIndex] = 1;
          runnerState.lapTimeTicks[carIndex] = runnerState.frame.tick;
        }
      }

      runnerState.frame.progress01[carIndex] =
        (distanceAlongTrack[carIndex] % trackLength) / trackLength;
    }

    separateCarsInFrame(runnerState.frame, agentCount);

    recomputePlaces();

    if (runnerState.frame.tick >= MAX_EPISODE_TICKS) {
      runnerState.frame.done = true;
    }
  }

  function computeFitness(carIndex: number): number {
    const progress = runnerState.frame.progress01[carIndex];

    if (runnerState.lapCompleted[carIndex] === 1) {
      return (
        COMPLETION_BONUS +
        (MAX_EPISODE_TICKS - runnerState.lapTimeTicks[carIndex])
      );
    }

    const baseFitness = PROGRESS_WEIGHT * progress * MAX_EPISODE_TICKS;
    if (runnerState.endedOffTrackPerCar[carIndex] === 1) {
      return baseFitness - OFF_TRACK_PENALTY;
    }

    // Backward-compatible fallback for tests and callers that set the legacy
    // global flag directly instead of the per-car array.
    const hasAnyPerCarFlag = Array.from(runnerState.endedOffTrackPerCar).some(
      (value) => value !== 0,
    );
    if (!hasAnyPerCarFlag && runnerState.endedOffTrack) {
      return baseFitness - OFF_TRACK_PENALTY;
    }

    return baseFitness;
  }

  function createRaceStepMessage(): {
    type: 'race-step';
    frame: RacingRenderFrame;
    transferList: ArrayBuffer[];
  } {
    return {
      type: 'race-step',
      frame: runnerState.frame,
      transferList: resolveRaceStepTransferList(runnerState.frame),
    };
  }

  function recomputePlaces(): void {
    const indices = Array.from({ length: agentCount }, (_, index) => index);
    indices.sort((a, b) => {
      const progressDelta =
        runnerState.frame.progress01[b] - runnerState.frame.progress01[a];
      if (progressDelta !== 0) {
        return progressDelta;
      }
      return a - b;
    });

    for (let rank = 0; rank < agentCount; rank++) {
      runnerState.frame.place[indices[rank]] = rank + 1;
    }
  }

  function separateCarsInFrame(
    targetFrame: RaceEpisodeRunnerFrame,
    targetAgentCount: number,
  ): void {
    for (let firstIndex = 0; firstIndex < targetAgentCount; firstIndex++) {
      for (
        let secondIndex = firstIndex + 1;
        secondIndex < targetAgentCount;
        secondIndex++
      ) {
        const deltaX =
          targetFrame.carX[secondIndex] - targetFrame.carX[firstIndex];
        const deltaY =
          targetFrame.carY[secondIndex] - targetFrame.carY[firstIndex];
        const distance = Math.hypot(deltaX, deltaY);

        if (distance >= CAR_MIN_CENTER_SEPARATION) {
          continue;
        }

        let unitX: number;
        let unitY: number;
        if (distance < 1e-9) {
          unitX = 1;
          unitY = 0;
        } else {
          unitX = deltaX / distance;
          unitY = deltaY / distance;
        }

        const push = (CAR_MIN_CENTER_SEPARATION - distance) / 2;
        targetFrame.carX[firstIndex] -= unitX * push;
        targetFrame.carY[firstIndex] -= unitY * push;
        targetFrame.carX[secondIndex] += unitX * push;
        targetFrame.carY[secondIndex] += unitY * push;
      }
    }
  }
}

function buildRaceFrame(
  options: BuildRaceFrameOptions,
): RaceEpisodeRunnerFrame {
  const { seed, trackId, featureFlags, agentCount, centerline } = options;

  const carX = new Float32Array(agentCount);
  const carY = new Float32Array(agentCount);
  const carHeading = new Float32Array(agentCount);
  const carActive = new Uint8Array(agentCount).fill(1);
  const carTeam = new Uint8Array(agentCount);
  const carMode = new Uint8Array(agentCount);
  const tireState = new Float32Array(agentCount * 4);
  const radioField = new Float32Array(0);
  const lap = new Uint16Array(agentCount);
  const place = new Uint8Array(agentCount);
  const progress01 = new Float32Array(agentCount);

  for (let carIndex = 0; carIndex < agentCount; carIndex++) {
    carTeam[carIndex] = carIndex % 2;
    place[carIndex] = carIndex + 1;
  }

  const startPoint = resolveTrackPointAtDistance(0, centerline);

  for (let carIndex = 0; carIndex < agentCount; carIndex++) {
    carX[carIndex] = startPoint.x;
    carY[carIndex] = startPoint.y;
    carHeading[carIndex] = startPoint.heading;
  }

  return {
    schemaVersion: RACING_SCHEMA_VERSION,
    tick: 0,
    seed,
    trackId,
    agentCount,
    featureFlags,
    carX,
    carY,
    carHeading,
    carActive,
    carTeam,
    carMode,
    tireState,
    radioField,
    lap,
    place,
    raceTimeMs: 0,
    done: false,
    progress01,
  };
}

type TrackCenterline = {
  /** Cumulative chordal distance from sample 0 to the start of each segment. */
  readonly cumulativeDistances: Float32Array;
  /** Inner-lane centerline point and heading for each spline sample. */
  readonly points: readonly {
    readonly x: number;
    readonly y: number;
    readonly heading: number;
  }[];
  /** Total closed-loop length in world units. */
  readonly trackLength: number;
};

function buildTrackCenterline(
  splineSamples: readonly SplineSample[],
): TrackCenterline {
  const sampleCount = splineSamples.length;

  if (sampleCount === 0) {
    return {
      cumulativeDistances: new Float32Array(1),
      points: [],
      trackLength: 0,
    };
  }

  const cumulativeDistances = new Float32Array(sampleCount + 1);
  const points: { x: number; y: number; heading: number }[] = [];

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const sample = splineSamples[sampleIndex]!;
    const splineFrame = resolveSplineSampleFrame(splineSamples, sampleIndex);
    const centerlinePoint = resolveInnerLaneCenterlinePoint(
      sample,
      splineFrame,
    );

    points.push({
      x: centerlinePoint.x,
      y: centerlinePoint.y,
      heading: splineFrame.tangentHeadingRadians,
    });

    if (sampleIndex < sampleCount - 1) {
      const nextSample = splineSamples[sampleIndex + 1]!;
      const nextFrame = resolveSplineSampleFrame(
        splineSamples,
        sampleIndex + 1,
      );
      const nextPoint = resolveInnerLaneCenterlinePoint(nextSample, nextFrame);
      cumulativeDistances[sampleIndex + 1] =
        cumulativeDistances[sampleIndex] +
        Math.hypot(
          nextPoint.x - centerlinePoint.x,
          nextPoint.y - centerlinePoint.y,
        );
    }
  }

  const firstPoint = points[0]!;
  const lastPoint = points[sampleCount - 1]!;
  cumulativeDistances[sampleCount] =
    cumulativeDistances[sampleCount - 1] +
    Math.hypot(firstPoint.x - lastPoint.x, firstPoint.y - lastPoint.y);

  return {
    cumulativeDistances,
    points,
    trackLength: cumulativeDistances[sampleCount],
  };
}

function resolveTrackPointAtDistance(
  distance: number,
  centerline: TrackCenterline,
): { x: number; y: number; heading: number } {
  const { trackLength, cumulativeDistances, points } = centerline;

  if (points.length === 0) {
    return { x: 0, y: 0, heading: 0 };
  }

  if (trackLength <= 0) {
    return points[0]!;
  }

  let normalizedDistance = distance % trackLength;
  if (normalizedDistance < 0) {
    normalizedDistance += trackLength;
  }

  const segmentCount = points.length;
  let segmentIndex = 0;
  while (
    segmentIndex < segmentCount &&
    cumulativeDistances[segmentIndex + 1] < normalizedDistance
  ) {
    segmentIndex++;
  }
  if (segmentIndex >= segmentCount) {
    segmentIndex = segmentCount - 1;
  }

  const segmentStart = cumulativeDistances[segmentIndex]!;
  const segmentEnd = cumulativeDistances[segmentIndex + 1]!;
  const segmentLength = segmentEnd - segmentStart || Number.MIN_VALUE;
  const interpolationFactor =
    (normalizedDistance - segmentStart) / segmentLength;

  const startPoint = points[segmentIndex]!;
  const endPoint = points[(segmentIndex + 1) % segmentCount]!;

  return {
    x: startPoint.x + (endPoint.x - startPoint.x) * interpolationFactor,
    y: startPoint.y + (endPoint.y - startPoint.y) * interpolationFactor,
    heading: lerpAngle(
      startPoint.heading,
      endPoint.heading,
      interpolationFactor,
    ),
  };
}

function lerpAngle(
  startRadians: number,
  endRadians: number,
  factor: number,
): number {
  const wrappedDelta =
    ((endRadians - startRadians + Math.PI) % (2 * Math.PI)) - Math.PI;
  const delta =
    wrappedDelta < -Math.PI ? wrappedDelta + 2 * Math.PI : wrappedDelta;
  return startRadians + delta * factor;
}

function resolveProgressSampleIndex(
  progress01: number,
  sampleCount: number,
): number {
  const floatIndex = progress01 * (sampleCount - 1);
  const index = Math.floor(floatIndex);
  return Math.max(0, Math.min(sampleCount - 1, index));
}

function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

export { EXPECTED_TRANSFER_BUFFER_COUNT };
