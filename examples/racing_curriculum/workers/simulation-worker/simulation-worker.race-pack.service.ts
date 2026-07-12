/**
 * Race episode runner and packed snapshot producer for the racing curriculum.
 *
 * This module owns the deterministic race episode lifecycle: building a race
 * pack from a seed and frozen opponent snapshot, ticking physics + controller
 * inference per car per tick, tracking lap completion and pit stops, computing
 * per-car fitness from finish positions, and resolving the zero-copy transfer
 * list for streaming packed `race-step` frames to the host.
 *
 * Key concepts:
 * - **Deterministic race pack**: identical seed + identical opponent snapshot
 *   produce identical starting frames, making race episodes replayable and
 *   comparative fitness claims fair.
 * - **Fixed-timestep simulation**: each `tick()` advances physics by
 *   `FIXED_TIMESTEP_SECONDS` (1/60 s) and runs one controller inference per
 *   car. The episode ends when all cars finish or `MAX_EPISODE_TICKS` is
 *   reached.
 * - **Pit lifecycle**: cars enter the pit when tire health drops below a
 *   threshold, remain for `PIT_STOP_TICKS`, then exit with fresh tires. The
 *   lap number at which each car pitted is recorded in `pitLapPerCar` for
 *   strategy-divergence observables.
 * - **Pit-lap distribution**: `extractPitLapDistribution` returns per-car lap
 *   numbers at which each car on a team pitted. A value of 0 means that car
 *   never pitted. These distributions feed the strategy-divergence tracker so
 *   the host can observe whether teams are converging on similar pit strategies
 *   or diverging.
 * - **Fitness from finish positions**: lap finishers are ranked by lap time
 *   ascending (fewer ticks = better finish); non-finishers are ranked by track
 *   progress descending. Each car receives a base fitness scaled by finish
 *   position plus a lap-completion bonus.
 * - **Shared-equal team fitness**: each team's fitness is the arithmetic mean
 *   of all member fitness scores, computed by `computeSharedEqualTeamFitness`
 *   from the evolution protocol service.
 *
 * ## Race episode tick lifecycle
 *
 * The diagram below shows one tick of the race episode runner. Each tick
 * advances physics, runs controller inference, updates pit and tire state,
 * checks lap completion, and optionally runs per-car adaptation.
 *
 * ```mermaid
 * flowchart TD
 *     A["tick()"] --> B["Advance physics<br/>(position, speed, heading)"]
 *     B --> C["Detect off-track<br/>+ wrong-direction"]
 *     C --> D["Decay tire state"]
 *     D --> E["Check pit entry/exit<br/>+ pit stop countdown"]
 *     E --> F["Detect lap completion"]
 *     F --> G["Run controller inference<br/>per car"]
 *     G --> H["Run per-car adaptation<br/>(adaptOnTick)"]
 *     H --> I["Check episode end<br/>(all done or max ticks)"]
 *     I -- "not done" --> J["Emit packed race-step frame"]
 *     I -- "done" --> K["Episode complete<br/>→ compute fitness"]
 * ```
 *
 * ## Pit-lap distribution observables
 *
 * After a race episode completes, `extractPitLapDistribution` reads the
 * `pitLapPerCar` array and filters by team to produce per-team pit-lap
 * distributions. These feed the strategy-divergence tracker so the host can
 * observe whether teams are converging on similar pit strategies or diverging.
 * The distributions are observability-only — they do NOT change fitness or
 * reproduction.
 *
 * See [Coevolution (Wikipedia)](https://en.wikipedia.org/wiki/Coevolution)
 * for background on why observing strategy divergence helps assess whether a
 * competitive coevolution arms race is producing diverse team strategies.
 */

import type { RacingRenderFrame } from './simulation-worker.types';
import type { TrackSpec, TrackAabb } from '../../track/track.generator.types';
import type { SplineSample } from '../../track/track.generator.types';
import { generateTrack } from '../../track/track.generator';
import {
  resolveSplineSampleFrame,
  resolveInnerLaneCenterlinePoint,
} from '../../track/track.spline.utils';
import { buildGuidingLineForTeam } from '../../renderer/racing.renderer';
import type { Network } from '../../../../src/browser-entry.ts';
import type { RuntimeAdaptationEngine } from '../../controller/runtime.adaptation';
import type { OpponentSnapshot as CoreOpponentSnapshot } from '../../../../src/neat/nge-collective/neat.nge-collective.types';
import { computeSharedEqualTeamFitness } from './simulation-worker.evolution.protocol.service';
import { decayTireState } from '../../environment/environment.step.service';
import {
  assembleTier4Observation,
  assembleTier5Observation,
  derivePerCarObservationState,
  type RacingObservationState,
} from '../../controller/observation.assembler';
import type {
  PitStrategyState,
  RacingCarState,
  TireStateTuple,
} from '../../environment/environment.types';

/** Schema sentinel for all packed race-step frames. */
const RACING_SCHEMA_VERSION = 'racing-packed-v1' as const;

/** Default agent count for a deterministic pack without controller networks. */
const DEFAULT_AGENT_COUNT = 2 as const;

/** Maximum number of fixed-timestep ticks in a single episode. */
const MAX_EPISODE_TICKS = 1800 as const;

/** Rolling score history cap for per-car adaptation decisions. */
const ADAPTATION_SCORE_HISTORY_CAP = 48 as const;

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

/** Sentinel value indicating no car is currently occupying a pit slot. */
const NO_CAR_INDEX = 255 as const;

/** Fixed number of ticks a car must remain in the pit before tire reset. */
const PIT_STOP_TICKS = 4 as const;

/** Normalization scale for pit-entrance distance channels in world units. */
const PIT_DISTANCE_WORLD_SCALE = 256;

/** Normalization scale for `lapsSincePit01` before clamping to `[0, 1]`. */
const LAPS_SINCE_PIT_SCALE = 10;

/** Normalization scale for estimated laps before tire failure. */
const ESTIMATED_LAPS_BEFORE_FAILURE_SCALE = 10;

/** Fresh tire health value at episode start. */
const FRESH_TIRE_HEALTH = 1.0;

/** Team layout for Tier 1–2: one car per team (2-car 1v1). */
const TIER_ONE_TWO_TEAM_LAYOUT = [0, 1] as const;

/** Team layout for Tier 3–4: two cars per team (4-car 2v2). */
const TIER_THREE_TEAM_LAYOUT = [0, 0, 1, 1] as const;

/** Team layout for Tier 5: three cars per team (6-car 3v3). */
const TIER_FIVE_TEAM_LAYOUT = [0, 0, 0, 1, 1, 1] as const;

/** Car count threshold above which the Tier 3 four-car 2v2 layout applies. */
const TIER_THREE_CAR_COUNT = 4 as const;

/** Car count threshold above which the Tier 5 six-car 3v3 layout applies. */
const TIER_FIVE_CAR_COUNT = 6 as const;

/** Radio channels per car (7 channels: posX, posY, headingSin, speed, relX, relY, relHeadingSin). */
const RADIO_CHANNELS_PER_CAR = 7;

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

/**
 * Converts a core collective opponent snapshot into the race-pack shape.
 *
 * The core `OpponentSnapshot` ({agentId, snapshot, frozenAt}) uses a generic
 * payload record, while the race-pack variant ({snapshotId, generation,
 * networkPayloads}) expects a flat serialised network payload array. This
 * adapter bridges the two so snapshots accumulated in the core
 * `OpponentSnapshotPool` can be consumed by the race-pack racing pipeline.
 *
 * @param core - Core collective snapshot to convert.
 * @returns Race-pack-shaped opponent snapshot.
 *
 * @example
 * ```ts
 * const racePackSnapshot = convertCoreToRacePackSnapshot(coreSnapshot);
 * console.log(racePackSnapshot.snapshotId); // core.agentId
 * ```
 */
export function convertCoreToRacePackSnapshot(
  core: CoreOpponentSnapshot,
): OpponentSnapshot {
  return {
    snapshotId: core.agentId,
    generation: core.frozenAt,
    networkPayloads: extractNetworkPayloadsFromSnapshot(core.snapshot),
  };
}

/**
 * Extracts a network-payloads array from a core snapshot payload record.
 *
 * @param snapshot - Generic payload record from a core opponent snapshot.
 * @returns The `networkPayloads` array when present, otherwise an empty array.
 */
function extractNetworkPayloadsFromSnapshot(
  snapshot: Readonly<Record<string, unknown>>,
): readonly unknown[] {
  const candidate = snapshot['networkPayloads'];
  return Array.isArray(candidate) ? candidate : [];
}

/** Minimal controller handle used inside a race episode runner. */
export type RaceControllerNetwork = {
  /** Runs inference and returns the controller's output vector. */
  activate(inputs: number[]): number[];
};

/**
 * Per-car adaptation context passed to the race episode runner.
 *
 * Each entry pairs a {@link RuntimeAdaptationEngine} with the live
 * {@link Network} it mutates, so the runner can call `adaptOnTick` after
 * physics + inference for each car on every tick.
 */
export type RaceAdaptationContext = {
  /** Per-car adaptation engines, keyed by car index. */
  readonly engines: ReadonlyMap<number, RuntimeAdaptationEngine>;
  /** Per-car live Network instances, keyed by car index. */
  readonly networks: ReadonlyMap<number, Network>;
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
   * Resolves shared-equal team fitness as the average of member finishing
   * positions for the requested team.
   *
   * Reuses the evolution protocol's {@link computeSharedEqualTeamFitness}
   * aggregator so the race-pack service honours the cooperative team-fitness
   * policy used across the racing curriculum. A team is only as strong as its
   * average member, not its single best performer.
   *
   * @param teamId - 0 for Team A (blue), 1 for Team B (red).
   * @param carFinishPositions - Finish positions for that team's cars only.
   * @returns Average finish position, or 0 when the team has no members.
   */
  resolveTeamFitness(
    teamId: 0 | 1,
    carFinishPositions: readonly number[],
  ): number;
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
  /**
   * Per-car lap number at which the car entered the pit lane.
   * A value of 0 means the car never pitted during this episode.
   * Used by strategy-divergence analytics to compute pit-lap distributions.
   */
  pitLapPerCar: Uint16Array;
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
  /**
   * Serializes the blue team #1 car (car 0) network for browser visualization.
   *
   * Only car 0's network is copied back to the browser for the network-panel
   * visualization. All other car networks remain worker-side only.
   *
   * @returns Float32Array of connection weights, or undefined when no
   *   adaptation network is registered for car 0.
   */
  serializeVisualizationNetwork(): Float32Array | undefined;
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
  adaptationContext?: RaceAdaptationContext,
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
  const pitLapPerCar = new Uint16Array(agentCount);
  const endedOffTrackPerCar = new Uint8Array(agentCount);

  // Per-car rolling score history for adaptation decisions.
  const perCarScoreHistory: number[][] = Array.from(
    { length: agentCount },
    () => [],
  );

  const guidingLines = Array.from({ length: agentCount }, (_, carIndex) => {
    const carGuidingLine = buildGuidingLineForTeam(
      track,
      frame.carTeam[carIndex]!,
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
    resolveTeamFitness,
    createRaceStepMessage,
    lapCompleted,
    lapTimeTicks,
    pitLapPerCar,
    endedOffTrack: false,
    endedOffTrackPerCar,
    guidingLines,
    serializeVisualizationNetwork,
  };

  return runnerState;

  function tick(): void {
    if (runnerState.frame.done) {
      return;
    }

    runnerState.frame.tick += 1;
    runnerState.frame.raceTimeMs =
      runnerState.frame.tick * FIXED_TIMESTEP_SECONDS * 1000;

    // Step 1: Tick pit lifecycle (decrement active stops, release finished cars).
    tickPitLifecycle();

    // Step 2: Build the cars roster once from the current frame state.
    const cars = buildCarsFromFrame();

    for (let carIndex = 0; carIndex < agentCount; carIndex++) {
      const network = networks[carIndex];

      // Step 3: Build 103-channel Tier 4/5 observation and run controller inference.
      const observationInput = resolvePerCarObservation(carIndex, cars);
      const controllerOutput = network.activate(observationInput);
      const throttle = clamp01(controllerOutput[0]!);
      const steer = Math.max(-1, Math.min(1, controllerOutput[1]!));

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

      const stoppedInPit = isCarStoppedInPit(carIndex);

      // Step 4: Apply grip multiplier from tire health to forward progress.
      if (!stoppedInPit) {
        const tireOffset = carIndex * 4;
        const gripMultiplier = Math.sqrt(resolveMeanTireHealth(tireOffset));
        const forwardStep =
          throttle *
          gripMultiplier *
          MAX_FORWARD_SPEED_UNITS_PER_SECOND *
          FIXED_TIMESTEP_SECONDS;
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
          runnerState.lapCompleted[carIndex] = 1;
          runnerState.lapTimeTicks[carIndex] = runnerState.frame.tick;
        }

        runnerState.frame.progress01[carIndex] =
          (distanceAlongTrack[carIndex] % trackLength) / trackLength;

        // Step 5: Apply tire decay based on driving forces.
        const currentTireState = [
          runnerState.frame.tireState[tireOffset],
          runnerState.frame.tireState[tireOffset + 1],
          runnerState.frame.tireState[tireOffset + 2],
          runnerState.frame.tireState[tireOffset + 3],
        ] as TireStateTuple;
        const effectiveSpeed =
          Math.abs(throttle * gripMultiplier) *
          MAX_FORWARD_SPEED_UNITS_PER_SECOND;
        const decayedTires = decayTireState(
          currentTireState,
          Math.abs(steer),
          Math.abs(throttle),
          effectiveSpeed,
        );
        runnerState.frame.tireState[tireOffset] = decayedTires[0];
        runnerState.frame.tireState[tireOffset + 1] = decayedTires[1];
        runnerState.frame.tireState[tireOffset + 2] = decayedTires[2];
        runnerState.frame.tireState[tireOffset + 3] = decayedTires[3];
      }

      // Step 6: Run continuous adaptation for this car after physics + inference.
      runCarAdaptation(carIndex);
    }

    // Step 7: Resolve pit entries after all car updates.
    resolvePitEntries();

    separateCarsInFrame(runnerState.frame, agentCount);

    recomputePlaces();

    if (runnerState.frame.tick >= MAX_EPISODE_TICKS) {
      runnerState.frame.done = true;
    }
  }

  /**
   * Runs per-car continuous adaptation (adaptOnTick) when an adaptation
   * context is provided. Uses the car's progress as the score sample.
   */
  function runCarAdaptation(carIndex: number): void {
    if (!adaptationContext) {
      return;
    }

    const engine = adaptationContext.engines.get(carIndex);
    const adaptationNetwork = adaptationContext.networks.get(carIndex);
    if (!engine || !adaptationNetwork) {
      return;
    }

    // Track per-car progress as the rolling score for adaptation decisions.
    const scoreSample = runnerState.frame.progress01[carIndex];
    perCarScoreHistory[carIndex].push(scoreSample);
    if (perCarScoreHistory[carIndex].length > ADAPTATION_SCORE_HISTORY_CAP) {
      perCarScoreHistory[carIndex].shift();
    }

    engine.adaptOnTick({
      tick: runnerState.frame.tick,
      network: adaptationNetwork,
      scoreHistory: perCarScoreHistory[carIndex],
      completedLaps: runnerState.frame.lap[carIndex],
    });
  }

  /**
   * Builds the ordered car roster from the current frame state.
   * Used as input to observation assembly for all cars in the same tick.
   */
  function buildCarsFromFrame(): RacingCarState[] {
    const cars: RacingCarState[] = [];
    for (let i = 0; i < agentCount; i++) {
      const tireOffset = i * 4;
      cars.push({
        carX: runnerState.frame.carX[i],
        carY: runnerState.frame.carY[i],
        carHeading: runnerState.frame.carHeading[i],
        teamIndex: runnerState.frame.carTeam[i]! as 0 | 1,
        tireState: [
          runnerState.frame.tireState[tireOffset],
          runnerState.frame.tireState[tireOffset + 1],
          runnerState.frame.tireState[tireOffset + 2],
          runnerState.frame.tireState[tireOffset + 3],
        ] as TireStateTuple,
      });
    }
    return cars;
  }

  /**
   * Resolves the 103-channel Tier 4/5 observation vector for one car.
   *
   * The observation is assembled from the pre-physics, pre-decay state so
   * the tire channels at `[91..94]` capture the car's current health
   * *before* this tick's degradation is applied, and channels `[95..102]` carry
   * pit/strategy context. This ordering lets the controller observe tire wear
   * and pit state before deciding whether to push, lift off, or pit.
   *
   * @param carIndex - Index of the car to observe.
   * @param cars - Ordered car roster from the current frame state.
   * @returns 103-element observation array (91 Tier 3 + 4 tire health + 8 pit/strategy).
   */
  function resolvePerCarObservation(
    carIndex: number,
    cars: RacingCarState[],
  ): number[] {
    const tireOffset = carIndex * 4;
    const teamIndex = runnerState.frame.carTeam[carIndex]! as 0 | 1;
    const pitStrategyState = resolvePitStrategyState(
      carIndex,
      teamIndex,
      tireOffset,
    );
    const envState: RacingObservationState = {
      tick: runnerState.frame.tick,
      carX: runnerState.frame.carX[carIndex],
      carY: runnerState.frame.carY[carIndex],
      carHeading: runnerState.frame.carHeading[carIndex],
      teamIndex,
      tireState: [
        runnerState.frame.tireState[tireOffset],
        runnerState.frame.tireState[tireOffset + 1],
        runnerState.frame.tireState[tireOffset + 2],
        runnerState.frame.tireState[tireOffset + 3],
      ] as TireStateTuple,
      ...pitStrategyState,
      cars,
      progress01: runnerState.frame.progress01[carIndex],
    };
    const perCarState = derivePerCarObservationState(envState, carIndex);
    const observation =
      agentCount >= TIER_FIVE_CAR_COUNT
        ? assembleTier5Observation(perCarState, track)
        : assembleTier4Observation(perCarState, track);
    return Array.from(observation);
  }

  /**
   * Builds the optional pit/strategy state for one car from the current frame.
   *
   * The returned fields follow the canonical Tier 4/5 tail order and map to
   * offsets `[95..102]` of the observation vector:
   *   1. `pitDistanceToEntrance01`
   *   2. `pitOccupancyStatus`
   *   3. `lapsSincePit`
   *   4. `teammatePitStatus`
   *   5. `tireDegradationRate`
   *   6. `estimatedLapsBeforeFailure`
   *   7. `reservedPitContext1`
   *   8. `reservedPitContext2`
   *
   * All returned values are normalized to `[0, 1]`. Missing or degenerate data
   * falls back to `0` so the observation tail stays deterministic and safe for
   * controllers that have not seen pit features during training yet.
   */
  function resolvePitStrategyState(
    carIndex: number,
    teamIndex: 0 | 1,
    tireOffset: number,
  ): PitStrategyState {
    const pitDistanceToEntrance01 = resolvePitDistanceToEntrance01(
      carIndex,
      teamIndex,
    );
    const pitBoxStatus = resolvePitBoxStatus(teamIndex);
    const meanTireHealth = resolveMeanTireHealth(tireOffset);
    const tireDegradationRate01 = clamp01(1 - meanTireHealth);
    const estimatedLapsBeforeFailure01 = clamp01(
      meanTireHealth * ESTIMATED_LAPS_BEFORE_FAILURE_SCALE,
    );
    const lapsSincePit01 = resolveLapsSincePit01(carIndex);

    return {
      pitDistanceToEntrance01,
      pitOccupancyStatus: pitBoxStatus.isOccupied ? 1 : 0,
      lapsSincePit: lapsSincePit01,
      teammatePitStatus: pitBoxStatus.isTeammateOccupied ? 1 : 0,
      tireDegradationRate: tireDegradationRate01,
      estimatedLapsBeforeFailure: estimatedLapsBeforeFailure01,
      reservedPitContext1: 0,
      reservedPitContext2: 0,
    };
  }

  /**
   * Computes the normalized distance from a car to its team's pit entrance.
   *
   * If no pit box exists for the team the distance is reported as `1` (far).
   */
  function resolvePitDistanceToEntrance01(
    carIndex: number,
    teamIndex: 0 | 1,
  ): number {
    const teamPitBox = track.pitBoxes!.find(
      (pitBox) => pitBox.teamIndex === teamIndex,
    )!;
    const carX = runnerState.frame.carX[carIndex];
    const carY = runnerState.frame.carY[carIndex];
    const entranceX =
      teamPitBox.entranceCorridor.x + teamPitBox.entranceCorridor.width / 2;
    const entranceY =
      teamPitBox.entranceCorridor.y + teamPitBox.entranceCorridor.height / 2;
    const distanceWorld = Math.hypot(carX - entranceX, carY - entranceY);
    return clamp01(distanceWorld / PIT_DISTANCE_WORLD_SCALE);
  }

  /**
   * Reports whether the team pit box is occupied and whether the occupant is
   * a teammate of the querying car.
   *
   * Because the pit layout is team-scoped (each team has its own box), any
   * occupant of the queried box is definitionally on the querying car's team.
   * `isTeammateOccupied` therefore becomes `true` whenever the box is occupied,
   * including when the querying car itself is the one being serviced. Callers
   * should treat `teammatePitStatus` as "team box busy" rather than "another car
   * is in the box".
   */
  function resolvePitBoxStatus(teamIndex: 0 | 1): {
    readonly isOccupied: boolean;
    readonly isTeammateOccupied: boolean;
  } {
    const pitStatus = runnerState.frame.pitStatus;
    if (pitStatus === undefined) {
      return { isOccupied: false, isTeammateOccupied: false };
    }
    const stride = pitStatus.length >= 6 ? 3 : 2;
    const teamBase = teamIndex * stride;
    const occupyingCarIndex = pitStatus[teamBase];
    const empty = occupyingCarIndex === NO_CAR_INDEX;
    return {
      isOccupied: !empty,
      isTeammateOccupied: !empty,
    };
  }

  /**
   * Computes normalized laps elapsed since the car's last pit stop.
   *
   * A value of `0` means the car has never pitted or pitted on the current lap.
   */
  function resolveLapsSincePit01(carIndex: number): number {
    const pitLap = runnerState.pitLapPerCar[carIndex];
    if (pitLap === 0) {
      return 0;
    }
    const currentLap = runnerState.frame.lap[carIndex];
    return clamp01((currentLap - pitLap) / LAPS_SINCE_PIT_SCALE);
  }

  /**
   * Computes the mean tire health for one car from the frame's tire buffer.
   */
  function resolveMeanTireHealth(tireOffset: number): number {
    return (
      (runnerState.frame.tireState[tireOffset] +
        runnerState.frame.tireState[tireOffset + 1] +
        runnerState.frame.tireState[tireOffset + 2] +
        runnerState.frame.tireState[tireOffset + 3]) /
      4
    );
  }

  /**
   * Returns true when the point lies inside the axis-aligned bounding box.
   */
  function isPointInsideAabb(x: number, y: number, aabb: TrackAabb): boolean {
    return (
      x >= aabb.x &&
      x <= aabb.x + aabb.width &&
      y >= aabb.y &&
      y <= aabb.y + aabb.height
    );
  }

  /**
   * Returns true when the car is currently occupying its team's pit slot and
   * the stop timer is still active. Such cars must not move or decay tires.
   */
  function isCarStoppedInPit(carIndex: number): boolean {
    const pitStatus = runnerState.frame.pitStatus;
    if (pitStatus === undefined) {
      return false;
    }
    const carTeam = runnerState.frame.carTeam[carIndex]!;
    const stride = pitStatus.length >= 6 ? 3 : 2;
    const teamBase = carTeam * stride;
    return pitStatus[teamBase] === carIndex && pitStatus[teamBase + 1] > 0;
  }

  /**
   * Ticks the pit lifecycle: decrements active stop counters and releases
   * cars whose stop has completed, resetting their tires to fresh health.
   *
   * The compact `pitStatus` layout depends on the car count:
   * - 4-car packs (Tier 3–4): `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`
   *   with stride 2 per team.
   * - 6-car packs (Tier 5): `[teamA_car, teamA_ticks, teamA_waiting,
   *   teamB_car, teamB_ticks, teamB_waiting]` with stride 3 per team.
   *
   * For each team, if the car slot is not `255`, the tick counter is
   * decremented. When the counter reaches zero, the car's four tire channels
   * are restored to `1.0` (fresh tires) and the slot is released back to `255`.
   */
  function tickPitLifecycle(): void {
    const pitStatus = runnerState.frame.pitStatus;
    if (pitStatus === undefined) {
      return;
    }

    const stride = pitStatus.length >= 6 ? 3 : 2;
    for (let teamBase = 0; teamBase < pitStatus.length; teamBase += stride) {
      const carSlot = pitStatus[teamBase];
      if (carSlot === NO_CAR_INDEX) {
        continue;
      }
      if (pitStatus[teamBase + 1] > 0) {
        pitStatus[teamBase + 1] -= 1;
      }
      if (pitStatus[teamBase + 1] === 0) {
        const tireOffset = carSlot * 4;
        runnerState.frame.tireState[tireOffset] = FRESH_TIRE_HEALTH;
        runnerState.frame.tireState[tireOffset + 1] = FRESH_TIRE_HEALTH;
        runnerState.frame.tireState[tireOffset + 2] = FRESH_TIRE_HEALTH;
        runnerState.frame.tireState[tireOffset + 3] = FRESH_TIRE_HEALTH;
        pitStatus[teamBase] = NO_CAR_INDEX;
      }
    }
  }

  /**
   * Resolves pit entries: checks whether any car has entered its own team's
   * pit entrance corridor and claims the team's compact pit slot if available.
   *
   * A car can only enter its own team's pit, never the opposing team's. The
   * check is `candidatePitBox.teamIndex === car.teamIndex`, so Team A cars
   * are excluded from Team B pit boxes and vice versa. Each team's compact
   * slot in `pitStatus` holds at most one car; if the slot is already
   * occupied (`pitStatus[teamBase] !== 255`), the entry is skipped.
   *
   * The compact `pitStatus` layout depends on car count:
   * - 4-car packs (Tier 3–4): `[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`
   *   with stride 2 per team.
   * - 6-car packs (Tier 5): `[teamA_car, teamA_ticks, teamA_waiting,
   *   teamB_car, teamB_ticks, teamB_waiting]` with stride 3 per team.
   */
  function resolvePitEntries(): void {
    const pitStatus = runnerState.frame.pitStatus;
    if (pitStatus === undefined) {
      return;
    }

    const stride = pitStatus.length >= 6 ? 3 : 2;
    for (let carIndex = 0; carIndex < agentCount; carIndex++) {
      const carTeam = runnerState.frame.carTeam[carIndex]!;
      const ownPitBox = track.pitBoxes!.find(
        (box) => box.teamIndex === carTeam,
      )!;

      const teamBase = carTeam * stride;
      if (pitStatus[teamBase] !== NO_CAR_INDEX) {
        continue;
      }

      if (
        isPointInsideAabb(
          runnerState.frame.carX[carIndex],
          runnerState.frame.carY[carIndex],
          ownPitBox.entranceCorridor,
        )
      ) {
        pitStatus[teamBase] = carIndex;
        pitStatus[teamBase + 1] = PIT_STOP_TICKS;
        // Record the lap on which this car entered the pit lane.
        runnerState.pitLapPerCar[carIndex] = runnerState.frame.lap[carIndex];
      }
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

  /**
   * Resolves shared-equal team fitness by delegating to the evolution
   * protocol's cooperative aggregator.
   *
   * The caller supplies the finish positions for one team's cars only, so the
   * shared-equal average is computed over the provided positions directly.
   *
   * @param teamId - 0 for Team A (blue), 1 for Team B (red).
   * @param carFinishPositions - Finish positions for that team's cars only.
   * @returns Average finish position, or 0 when the team has no members.
   */
  function resolveTeamFitness(
    teamId: 0 | 1,
    carFinishPositions: readonly number[],
  ): number {
    // The caller already filtered positions to this team's cars, so every
    // entry belongs to `teamId`. Reuse the shared-equal aggregator by tagging
    // each provided position with the requested team id.
    const teamLayout = carFinishPositions.map(() => teamId) as (0 | 1)[];
    return computeSharedEqualTeamFitness(
      carFinishPositions,
      teamLayout,
      teamId,
    );
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

  /**
   * Serializes car 0 (blue team #1) network for visualization.
   *
   * Only car 0's network is copied back to the browser for network-panel
   * visualization. All other car networks remain worker-side only.
   */
  function serializeVisualizationNetwork(): Float32Array | undefined {
    if (!adaptationContext) {
      return undefined;
    }

    const car0Network = adaptationContext.networks.get(0);
    if (!car0Network) {
      return undefined;
    }

    return Float32Array.from(
      car0Network.connections.map((connection) => connection.weight),
    );
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
        // Cars stopped in the pit box must not be pushed by moving teammates.
        if (isCarStoppedInPit(firstIndex) || isCarStoppedInPit(secondIndex)) {
          continue;
        }

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
  const tireState = new Float32Array(agentCount * 4).fill(FRESH_TIRE_HEALTH);
  const radioField = new Float32Array(agentCount * RADIO_CHANNELS_PER_CAR);
  const lap = new Uint16Array(agentCount);
  const place = new Uint8Array(agentCount);
  const progress01 = new Float32Array(agentCount);
  const pitStatus =
    agentCount >= TIER_FIVE_CAR_COUNT
      ? new Uint8Array([NO_CAR_INDEX, 0, 0, NO_CAR_INDEX, 0, 0])
      : agentCount >= TIER_THREE_CAR_COUNT
        ? new Uint8Array([NO_CAR_INDEX, 0, NO_CAR_INDEX, 0])
        : undefined;

  const teamLayout =
    agentCount >= TIER_FIVE_CAR_COUNT
      ? TIER_FIVE_TEAM_LAYOUT
      : agentCount >= TIER_THREE_CAR_COUNT
        ? TIER_THREE_TEAM_LAYOUT
        : TIER_ONE_TWO_TEAM_LAYOUT;

  for (let carIndex = 0; carIndex < agentCount; carIndex++) {
    carTeam[carIndex] = teamLayout[carIndex]!;
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
    ...(pitStatus !== undefined ? { pitStatus } : {}),
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

  const normalizedDistance = distance % trackLength;

  const segmentCount = points.length;
  let segmentIndex = 0;
  while (
    segmentIndex < segmentCount &&
    cumulativeDistances[segmentIndex + 1] < normalizedDistance
  ) {
    segmentIndex++;
  }

  const segmentStart = cumulativeDistances[segmentIndex]!;
  const segmentEnd = cumulativeDistances[segmentIndex + 1]!;
  const segmentLength = segmentEnd - segmentStart;
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

/**
 * Clamps a value to the closed `[0, 1]` interval.
 */
function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.max(0, Math.min(1, value));
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

/**
 * Extract the per-car pit-lap distribution for one team from a race episode
 * runner.
 *
 * Returns an array of lap numbers, one entry per car on the requested team.
 * A value of 0 means that car never pitted during the episode. The array
 * length equals the number of cars on the team.
 *
 * Handles mock runners gracefully: when the runner does not expose
 * `pitLapPerCar` or `frame.carTeam` (e.g. in unit tests with minimal mocks),
 * an empty array is returned.
 *
 * @param runner - Race episode runner after the episode has completed
 * @param teamId - 0 for Team A, 1 for Team B
 * @returns Per-car pit-lap distribution for the requested team
 *
 * @example
 * ```ts
 * const teamADistribution = extractPitLapDistribution(runner, 0);
 * console.log(teamADistribution); // [2, 0, 4] — car 0 pitted on lap 2, etc.
 * ```
 */
export function extractPitLapDistribution(
  runner: {
    readonly pitLapPerCar?: Uint16Array;
    readonly frame?: { readonly carTeam?: Uint8Array | readonly number[] };
  },
  teamId: 0 | 1,
): number[] {
  const pitLapPerCar = runner.pitLapPerCar;
  const carTeam = runner.frame?.carTeam;
  if (pitLapPerCar === undefined || carTeam === undefined) {
    return [];
  }
  const distribution: number[] = [];
  for (let carIndex = 0; carIndex < pitLapPerCar.length; carIndex++) {
    if (carTeam[carIndex] === teamId) {
      distribution.push(pitLapPerCar[carIndex]);
    }
  }
  return distribution;
}

export { EXPECTED_TRANSFER_BUFFER_COUNT };
