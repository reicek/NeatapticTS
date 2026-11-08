/**
 * In-band live simulation worker for the neatRacing browser demo (Phase 2,
 * Step 01, slice P2-S1-A).
 *
 * Educational note — why this file exists:
 * The phase-1 race loop lived on the browser main thread (`host/race-session.ts`):
 * every rAF frame advanced physics, race rules, and controllers inline, which
 * couples the frame budget of the render pipeline to the cost of the simulation.
 * The Phase-2 learning-document contract (LD P2-S1) moves that authority to a
 * single in-band live-worker file: the worker owns ALL simulation state, and
 * the host only paces frames and forwards the once-per-frame human control
 * sample (decision D5) as a message. Every piece of racing truth now crosses a
 * typed message boundary, so the ownership question ("which side owns this
 * state?") has one visible answer for every field.
 *
 * What this worker owns per slice P2-S1-A:
 * - the 60 Hz fixed-step accumulator (`host/render-loop.utils.ts` reuse),
 * - the car fleet physics (`stepPhysics` per car through `stepPhysicsFleet`,
 *   whose single live-car path stays `stepPhysics`),
 * - the race-rules state machine (`createRaceRulesState` / `stepRaceRules`),
 * - the autopilot controllers (`stepAutopilotInput`) for the `ai` mode and
 *   every agent car beyond car 0,
 * - the per-tick forward lap-line crossing detector (ported phase-1 host
 *   boot logic, now owned where the car state it reads lives).
 *
 * What the host keeps until later phases: rendering, HUD, selector, and the
 * keyboard-sampling seam — the host samples driving input once per drawn
 * frame (D5) and ships it to the worker with the frame timestamp.
 *
 * Message protocol contract (slice A):
 * - host → worker `{ type: 'init', payload }` boots the session. The payload
 *   carries the AC-026 reproducibility tuple fields (`seed`, `agentCount`,
 *   `packSchemaVersion`, `opponentSnapshot`, `trackId`, `featureFlags`) plus
 *   the `trackSpec` geometry the sim needs to own its state. Replies exactly
 *   one `{ type: 'ready', payload }`.
 * - host → worker `{ type: 'frame', payload }` paces one drawn frame
 *   (`timestampMs` drives the accumulator; `sample` is the D5 human control
 *   sample for car 0 in `manual` mode). Replies exactly one
 *   `{ type: 'frames', payload }` whose payload snapshots the released ticks
 *   (`fromTick`..`toTick`), the post-frame fleet, and the post-frame race
 *   rules. Slice B replaces this payload with the packed
 *   `neatracing-packed-v1` budget; the message shape is the stable seam.
 * - host → worker `{ type: 'stop' }` halts the session. Replies nothing
 *   (fire-and-forget lifecycle message, matching the flappy stop precedent).
 * - anything else (unknown type, malformed payload, bad ordering) replies one
 *   `{ type: 'error', payload: { message } }`.
 *
 * Determinism discipline (LD D1 / D5, AC-026):
 * - no `Math.random` anywhere in this file; every spawn is derived from the
 *   init seed and car index, and every controller is a pure step function.
 * - no clock reads inside the tick pipeline: wall-clock timestamps enter ONLY
 *   through `frame` messages and feed the release accumulator, never tick math.
 * - identical tuple + identical frame pacing + identical sample streams
 *   produce identical tick content (Level-2 ordered determinism), which the
 *   byte-parity slice (P2-S1-C) formalizes.
 */

/// <reference lib="webworker" />

import { stepAutopilotInput } from '../../physics/ai.autopilot';
import { createTrackSurfaceResolver } from '../../physics/collision.utils';
import {
  createInitialPhysicsCarState,
  stepPhysicsFleet,
  type PhysicsCarSpawnPose,
  type PhysicsCarState,
  type PhysicsStepInput,
} from '../../physics/physics.step.service';
import {
  createRaceRulesState,
  stepRaceRules,
  type RaceRulesState,
} from '../../physics/race-rules';
import {
  advanceFrameSteps,
  createNeatRacingRenderLoopState,
  NEATRACING_FIXED_TIMESTEP_SECONDS,
  type NeatRacingRenderLoopState,
} from '../../host/render-loop.utils';
import type {
  SplineSample,
  TrackSegment,
  TrackSpec,
} from '../../../racing_curriculum/track/track.generator.types';

/** Smallest legal fleet size for a live sim-worker session (just the car). */
export const NEATRACING_SIM_WORKER_MIN_AGENTS = 1;

/**
 * Largest legal fleet size for a live sim-worker session.
 *
 * Educational note: the AC-033 demo fleet is three cars (human car 0 plus
 * two evolving agents), so this cap is deliberately generous rather than
 * load-bearing. The NGE eval-worker slice revisits the bound when real
 * populations flow through instead of hard-coding a tuning guess here.
 */
export const NEATRACING_SIM_WORKER_MAX_AGENTS = 32;

/**
 * Along-track lead distance (meters past segment 0's START point) where the
 * first fleet car spawns, matching the phase-1 host spawn recipe exactly
 * (segment-0 chord heading, 2 m along it) so the AC-026 baseline run keeps
 * the same opening pose the host used.
 */
export const NEATRACING_SIM_WORKER_SPAWN_ALONG_LEAD_M = 2;

/** Line-astern grid pitch (meters) between consecutive fleet spawn slots. */
export const NEATRACING_SIM_WORKER_SPAWN_GRID_PITCH_M = 12;

/**
 * Lateral gate (meters from segment 0's chord) inside which a pose counts as
 * crossing the lap line. Carried verbatim from the phase-1 host detector so
 * the worker-owned port stays behaviorally identical.
 */
export const NEATRACING_SIM_WORKER_LAP_LINE_LATERAL_GATE_M = 5;

/**
 * Along-track deadband (meters from the lap-line plane) below which samples
 * do not update the detector baseline. Carried verbatim from the phase-1 host
 * detector.
 */
export const NEATRACING_SIM_WORKER_LAP_LINE_ALONG_EPS_M = 0.5;

/**
 * Neutral control sample replayed for car 0 in `manual` mode before the host
 * has sent any sample. Frozen: it is a constant, never a live accumulator.
 */
export const NEUTRAL_SIM_WORKER_CONTROL_SAMPLE: Readonly<PhysicsStepInput> = Object.freeze(
  { throttle: 0, brake: 0, steer: 0 },
);

/** Race-mode knob the host boots the session with (selector state at init). */
export type NeatRacingSimWorkerRaceMode = 'manual' | 'ai';

/** Lifecycle phases of one sim-worker session. */
export type NeatRacingSimWorkerPhase = 'idle' | 'running' | 'stopped';

/**
 * Init payload for the sim worker.
 *
 * The AC-026 reproducibility tuple components ride alongside the track
 * geometry so one init message carries everything a replay bootstrap needs:
 * `trackId` names the tuple's track while `trackSpec` supplies the actual
 * owned geometry (segments for the surface resolver and lap detector,
 * spline samples for the autopilot controllers).
 */
export interface NeatRacingSimWorkerInitPayload {
  /** AC-026 tuple: deterministic run seed driving per-car spawn states. */
  seed: number;
  /** AC-026 tuple: fleet size (car 0 plus `agentCount - 1` agent cars). */
  agentCount: number;
  /** AC-026 tuple: pack schema version string carried for later slices. */
  packSchemaVersion: string;
  /**
   * AC-026 tuple: opponent snapshot carried verbatim (opaque until the NGE
   * network-controller slice consumes it; not validated here).
   */
  opponentSnapshot: unknown;
  /** AC-026 tuple: reproducibility track identifier. */
  trackId: string;
  /** AC-026 tuple: feature flags carried verbatim (shape pinned by P2-S1-C). */
  featureFlags: Readonly<Record<string, boolean>> | undefined;
  /** Owned track geometry: segments + spline samples + generation seed. */
  trackSpec: TrackSpec;
  /** Boot race mode for car 0 (`'ai'` default; `'manual'` consumes samples). */
  raceMode?: NeatRacingSimWorkerRaceMode;
}

/** Per-frame pace message payload (one drawn frame, one reply). */
export interface NeatRacingSimWorkerFramePayload {
  /** rAF-style monotonic timestamp driving the fixed-step accumulator. */
  timestampMs: number;
  /** D5 once-per-frame human control sample for car 0 (`manual` mode). */
  sample?: PhysicsStepInput;
}

/** Host → worker request message union (the sim-worker protocol inbox). */
export type NeatRacingSimWorkerRequest =
  | { type: 'init'; payload: NeatRacingSimWorkerInitPayload }
  | { type: 'frame'; payload: NeatRacingSimWorkerFramePayload }
  | { type: 'stop' };

/** Boot acknowledgement payload echoing the session identity tuple. */
export interface NeatRacingSimWorkerReadyPayload {
  /** Seed the session booted with. */
  seed: number;
  /** Fleet size the session booted with. */
  agentCount: number;
  /** Pack schema version string the session booted with. */
  packSchemaVersion: string;
  /** Reproducibility track identifier the session booted with. */
  trackId: string;
  /** Effective race mode (`'ai'` when omitted at init). */
  raceMode: NeatRacingSimWorkerRaceMode;
}

/**
 * Per-frame reply payload (slice A structured snapshots).
 *
 * Slice B swaps this payload for the packed `neatracing-packed-v1` budget;
 * the tick-range bookkeeping below is what the pack must preserve.
 */
export interface NeatRacingSimWorkerFramesPayload {
  /** Total-tick counter before this frame's released ticks (car 0 basis). */
  fromTick: number;
  /** Total-tick counter after this frame's released ticks. */
  toTick: number;
  /** Post-frame fleet snapshot (structured clone; plain objects). */
  cars: PhysicsCarState[];
  /** Post-frame race-rules snapshot (structured clone; plain object). */
  rules: RaceRulesState;
}

/** Error payload for any rejected request. */
export interface NeatRacingSimWorkerErrorPayload {
  /** Static human-readable reason (never contains dynamic state). */
  message: string;
}

/** Worker → host response message union (the sim-worker protocol outbox). */
export type NeatRacingSimWorkerResponse =
  | { type: 'ready'; payload: NeatRacingSimWorkerReadyPayload }
  | { type: 'frames'; payload: NeatRacingSimWorkerFramesPayload }
  | { type: 'error'; payload: NeatRacingSimWorkerErrorPayload };

/** Read-model diagnostics snapshot of one sim-worker session. */
export interface NeatRacingSimWorkerSessionSnapshot {
  /** Current lifecycle phase. */
  phase: NeatRacingSimWorkerPhase;
  /** Total ticks released since boot (car 0 tick basis). */
  tick: number;
  /** Live fleet (copy; safe to hold without aliasing tick output). */
  cars: readonly PhysicsCarState[];
  /** Live race-rules state (copy). */
  rules: RaceRulesState;
  /** Lap-line detector baseline (latched along-track side, or null). */
  lapLinePrevSide: number | null;
}

/** Internal mutable session bag: all sim truth owned by the worker lives here. */
interface SimWorkerSessionState {
  phase: NeatRacingSimWorkerPhase;
  renderLoop: NeatRacingRenderLoopState;
  cars: PhysicsCarState[];
  rules: RaceRulesState;
  lapLinePrevSide: number | null;
  raceMode: NeatRacingSimWorkerRaceMode;
  humanSample: PhysicsStepInput | undefined;
  surfaceResolver: ReturnType<typeof createTrackSurfaceResolver> | undefined;
  splineSamples: readonly SplineSample[];
  firstSegment: TrackSegment | undefined;
  readyDetail:
    | {
        seed: number;
        agentCount: number;
        packSchemaVersion: string;
        trackId: string;
      }
    | undefined;
}

/** Public session handle: a pure request/response state machine over the bag. */
export interface NeatRacingSimWorkerSession {
  /**
   * Handles one request and returns the responses it should post.
   *
   * Educational note: the session deliberately RETURNS its replies instead of
   * posting them itself. The pure protocol machine stays testable headless
   * (no worker realm needed), and the worker entrypoint below is the only
   * place that touches `self.postMessage`.
   *
   * @param request - Host request message (validated defensively at runtime).
   * @returns An ordered list of responses to post (0-2 entries).
   */
  handleRequest(
    request: NeatRacingSimWorkerRequest,
  ): NeatRacingSimWorkerResponse[];
  /** Read-model snapshot of the live session (diagnostics + tests). */
  readonly snapshot: NeatRacingSimWorkerSessionSnapshot;
}

/** Validation failure carries a static rejection reason string. */
type SimWorkerValidationError = { error: string };

/**
 * Validates an init payload and returns a rejection reason, or null when the
 * payload may boot. The carried tuple fields (`opponentSnapshot`,
 * `featureFlags`) are intentionally unvalidated: slice A transports them, the
 * byte-parity slice pins their serialized shape.
 *
 * @param payload - Candidate init payload.
 * @returns Rejection reason, or null when valid.
 */
function validateSimWorkerInitPayload(
  payload: unknown,
): SimWorkerValidationError | null {
  // Step 1: Reject non-object payloads wholesale.
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    return { error: 'sim worker init failed: payload must be an object' };
  }
  const candidate = payload as Partial<NeatRacingSimWorkerInitPayload>;

  // Step 2: Validate the numeric identity fields.
  if (typeof candidate.seed !== 'number' || !Number.isFinite(candidate.seed)) {
    return { error: 'sim worker init failed: seed must be a finite number' };
  }
  if (
    typeof candidate.agentCount !== 'number' ||
    !Number.isInteger(candidate.agentCount) ||
    candidate.agentCount < NEATRACING_SIM_WORKER_MIN_AGENTS ||
    candidate.agentCount > NEATRACING_SIM_WORKER_MAX_AGENTS
  ) {
    return {
      error: `sim worker init failed: agentCount must be an integer in [${NEATRACING_SIM_WORKER_MIN_AGENTS}, ${NEATRACING_SIM_WORKER_MAX_AGENTS}]`,
    };
  }

  // Step 3: Validate the two string identity fields.
  if (
    typeof candidate.packSchemaVersion !== 'string' ||
    candidate.packSchemaVersion.length === 0
  ) {
    return {
      error:
        'sim worker init failed: packSchemaVersion must be a non-empty string',
    };
  }
  if (typeof candidate.trackId !== 'string' || candidate.trackId.length === 0) {
    return {
      error: 'sim worker init failed: trackId must be a non-empty string',
    };
  }

  // Step 4: Validate the owned geometry (non-empty closed-loop inputs).
  if (candidate.trackSpec === null || typeof candidate.trackSpec !== 'object') {
    return { error: 'sim worker init failed: trackSpec must be an object' };
  }
  if (
    !Array.isArray(candidate.trackSpec.segments) ||
    candidate.trackSpec.segments.length === 0
  ) {
    return {
      error:
        'sim worker init failed: trackSpec.segments must be a non-empty array',
    };
  }
  if (
    !Array.isArray(candidate.trackSpec.splineSamples) ||
    candidate.trackSpec.splineSamples.length === 0
  ) {
    return {
      error:
        'sim worker init failed: trackSpec.splineSamples must be a non-empty array',
    };
  }

  // Step 5: Validate the optional race-mode knob.
  if (
    candidate.raceMode !== undefined &&
    candidate.raceMode !== 'manual' &&
    candidate.raceMode !== 'ai'
  ) {
    return {
      error:
        "sim worker init failed: raceMode must be 'manual' or 'ai' when present",
    };
  }
  return null;
}

/**
 * Validates a frame payload (object with a finite timestamp; an optional
 * sample carrying finite axes).
 *
 * @param payload - Candidate frame payload.
 * @returns Rejection reason, or null when valid.
 */
function validateSimWorkerFramePayload(
  payload: unknown,
): SimWorkerValidationError | null {
  // Step 1: Reject non-object payloads wholesale.
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    return { error: 'sim worker frame failed: payload must be an object' };
  }
  const candidate = payload as Partial<NeatRacingSimWorkerFramePayload>;

  // Step 2: The pacing timestamp must be a usable number.
  if (
    typeof candidate.timestampMs !== 'number' ||
    !Number.isFinite(candidate.timestampMs)
  ) {
    return {
      error: 'sim worker frame failed: timestampMs must be a finite number',
    };
  }

  // Step 3: When a D5 sample rides along, its axes must be finite numbers.
  if (candidate.sample !== undefined) {
    const sample = candidate.sample;
    if (sample === null || typeof sample !== 'object') {
      return { error: 'sim worker frame failed: sample must be an object' };
    }
    for (const axis of ['throttle', 'brake', 'steer'] as const) {
      const value = sample[axis];
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        return {
          error: `sim worker frame failed: sample.${axis} must be a finite number`,
        };
      }
    }
  }
  return null;
}

/**
 * One step of the worker-owned lap-line detector: decides whether the pose
 * ENTERING this tick crossed the finish plane forward, and carries the
 * detector baseline forward.
 *
 * Educational note: this is the exact port of the phase-1 host detector
 * (browser-entry boot glue) — the finish line is the plane through segment
 * 0's start point perpendicular to its chord; a crossing counts only inside
 * the lateral gate, samples inside the along-track deadband never update the
 * baseline, and leaving the gate resets it (a car that wandered off can never
 * bank a stale side). Phase-1 semantics kept: direction filtering is a
 * carried gap, so ANY sign flip counts.
 *
 * @param car - Fleet car state (the tick's entering pose).
 * @param prevSide - Latched along-track side from the previous qualifying
 *   sample, or null before the first one.
 * @param firstSegment - Segment 0 of the owned track (or undefined when the
 *   geometry is degenerate; the detector then never reports a crossing).
 * @returns The crossing verdict plus the baseline to carry forward.
 */
function resolveSimWorkerLapLineDetectorStep(
  car: PhysicsCarState,
  prevSide: number | null,
  firstSegment: TrackSegment | undefined,
): { crossing: boolean; nextSide: number | null } {
  // Step 1: Degenerate geometry never counts a lap.
  if (firstSegment === undefined) {
    return { crossing: false, nextSide: prevSide };
  }
  const chordX = firstSegment.endX - firstSegment.startX;
  const chordZ = firstSegment.endY - firstSegment.startY;
  const chordLength = Math.hypot(chordX, chordZ);
  if (chordLength === 0) {
    return { crossing: false, nextSide: prevSide };
  }

  // Step 2: Project the pose onto the chord's along/lateral axes.
  const relX = car.x - firstSegment.startX;
  const relY = car.y - firstSegment.startY;
  const alongM = (relX * chordX + relY * chordZ) / chordLength;
  const lateralM = (chordX * relY - chordZ * relX) / chordLength;

  // Step 3: Outside the gate, the baseline resets (stale side dropped).
  if (Math.abs(lateralM) > NEATRACING_SIM_WORKER_LAP_LINE_LATERAL_GATE_M) {
    return { crossing: false, nextSide: null };
  }
  // Step 4: Deadband samples neither count nor update the baseline.
  if (Math.abs(alongM) < NEATRACING_SIM_WORKER_LAP_LINE_ALONG_EPS_M) {
    return { crossing: false, nextSide: prevSide };
  }

  // Step 5: First qualifying sample only latches the side.
  const baseline = prevSide;
  if (baseline === null) {
    return { crossing: false, nextSide: alongM };
  }
  return { crossing: baseline * alongM < 0, nextSide: alongM };
}

/**
 * Resolves the deterministic fleet spawn poses from the owned track.
 *
 * Educational note: car 0 inherits the phase-1 host recipe exactly (segment-0
 * chord heading, a 2 m lead along it), while cars behind it line up astern on
 * the same centerline at fixed grid pitch so the fleet start stays a function
 * of the init config alone (no RNG anywhere). The formal child-seed system is
 * the byte-parity slice's job (P2-S1-C); this slice derives per-car seeds by
 * the documented index offset when it builds the fleet.
 *
 * @param trackSpec - Owned track geometry.
 * @param agentCount - Fleet size (>= 1).
 * @returns One spawn pose per fleet slot, ordered car 0 first.
 */
function resolveSimWorkerFleetSpawnPoses(
  trackSpec: TrackSpec,
  agentCount: number,
): PhysicsCarSpawnPose[] {
  // Step 1: Derive the start straight from segment 0's chord.
  const firstSegment = trackSpec.segments[0];
  const spawnHeading = firstSegment
    ? Math.atan2(
        firstSegment.endY - firstSegment.startY,
        firstSegment.endX - firstSegment.startX,
      )
    : 0;

  // Step 2: Lay out the line-astern grid along the start straight.
  const poses: PhysicsCarSpawnPose[] = [];
  for (let carIndex = 0; carIndex < agentCount; carIndex += 1) {
    const alongM =
      NEATRACING_SIM_WORKER_SPAWN_ALONG_LEAD_M +
      carIndex * NEATRACING_SIM_WORKER_SPAWN_GRID_PITCH_M;
    poses.push({
      x: (firstSegment?.startX ?? 0) + Math.cos(spawnHeading) * alongM,
      y: (firstSegment?.startY ?? 0) + Math.sin(spawnHeading) * alongM,
      heading: spawnHeading,
    });
  }
  return poses;
}

/**
 * Resolves the per-tick control inputs for the whole fleet.
 *
 * Educational note: in `manual` mode car 0 runs on the host's D5 sample (the
 * latest received one is replayed across catch-up ticks), while in `ai` mode
 * car 0 AND every agent car run the worker-owned autopilot step function —
 * a pure function of the entering pose and the frozen spline samples, so the
 * whole control stream stays a deterministic function of sim state.
 *
 * @param cars - Fleet states entering this tick.
 * @param raceMode - Session race mode.
 * @param humanSample - Latest host sample (may be undefined pre-first sample).
 * @param splineSamples - Frozen track spline samples for the autopilot.
 * @returns Exactly one input per fleet car, ordered like the fleet.
 */
function resolveSimWorkerFleetTickInputs(
  cars: readonly PhysicsCarState[],
  raceMode: NeatRacingSimWorkerRaceMode,
  humanSample: PhysicsStepInput | undefined,
  splineSamples: readonly SplineSample[],
): PhysicsStepInput[] {
  // Step 1: Resolve every slot in fleet order (car 0 first).
  return cars.map((car, carIndex) => {
    // Step 2: Human car 0 under manual control takes the replayed sample.
    if (carIndex === 0 && raceMode === 'manual') {
      return humanSample ?? NEUTRAL_SIM_WORKER_CONTROL_SAMPLE;
    }
    // Step 3: Everyone else (and car 0 in ai mode) runs the autopilot.
    return stepAutopilotInput(car, splineSamples);
  });
}

/**
 * Boots the sim state for one session from a validated init payload.
 *
 * @param payload - Validated init payload.
 * @returns The fresh internal session bag in the `running` phase, with the
 *   boot detail guaranteed non-undefined (the running bag always carries it).
 */
function createSimWorkerRunningState(
  payload: NeatRacingSimWorkerInitPayload,
): SimWorkerSessionState & {
  readyDetail: NonNullable<SimWorkerSessionState['readyDetail']>;
} {
  // Step 1: Preallocate the accumulator through the default allocator seam.
  const renderLoop = createNeatRacingRenderLoopState();

  // Step 2: Spawn the fleet deterministically (seed + index offset per car).
  const spawnPoses = resolveSimWorkerFleetSpawnPoses(
    payload.trackSpec,
    payload.agentCount,
  );
  const cars = spawnPoses.map((pose, carIndex) =>
    createInitialPhysicsCarState(payload.seed + carIndex, pose),
  );

  // Step 3: Seed the rules state machine at level 1 (curriculum levels ride
  // through the pack in later steps; the worker starts every race at lvl 1).
  const rules = createRaceRulesState();

  return {
    phase: 'running',
    renderLoop,
    cars,
    rules,
    lapLinePrevSide: null,
    raceMode: payload.raceMode ?? 'ai',
    humanSample: undefined,
    surfaceResolver: createTrackSurfaceResolver(payload.trackSpec.segments),
    splineSamples: payload.trackSpec.splineSamples,
    firstSegment: payload.trackSpec.segments[0],
    readyDetail: {
      seed: payload.seed,
      agentCount: payload.agentCount,
      packSchemaVersion: payload.packSchemaVersion,
      trackId: payload.trackId,
    },
  };
}

/**
 * Releases 0-4 fixed steps for one paced frame and mutates the running session.
 *
 * @param sessionState - Running session bag mutated in place.
 * @param timestampMs - rAF-style frame timestamp driving the accumulator.
 * @param framePayload - The paced frame payload (its optional D5 sample is
 *   latched BEFORE ticking so zero-step frames still refresh the held manual
 *   sample, matching the phase-1 sampling-outside-the-step-guard behavior).
 */
function driveSimWorkerFrame(
  sessionState: SimWorkerSessionState,
  timestampMs: number,
  framePayload: NeatRacingSimWorkerFramePayload,
): void {
  // Step 1: Latch the D5 sample FIRST (manual mode only) so every catch-up
  // tick below replays the SAME once-per-frame sample.
  if (
    sessionState.raceMode === 'manual' &&
    framePayload.sample !== undefined
  ) {
    sessionState.humanSample = framePayload.sample;
  }

  // Step 2: Advance the accumulator and release the frame's tick count.
  const stepCount = advanceFrameSteps(sessionState.renderLoop, timestampMs);

  // Step 3: Run each released tick in the phase-1 order: the detector sees
  // the pose ENTERING the tick, then fleet physics, then race rules.
  for (let stepIndex = 0; stepIndex < stepCount; stepIndex += 1) {
    const detector = resolveSimWorkerLapLineDetectorStep(
      sessionState.cars[0],
      sessionState.lapLinePrevSide,
      sessionState.firstSegment,
    );
    sessionState.lapLinePrevSide = detector.nextSide;
    const inputs = resolveSimWorkerFleetTickInputs(
      sessionState.cars,
      sessionState.raceMode,
      sessionState.humanSample,
      sessionState.splineSamples,
    );
    // One fleet pass per tick: every member steps through `stepPhysics`,
    // then ONE collision-resolution pass keeps overlap bookkeeping in sync.
    sessionState.cars = stepPhysicsFleet(
      sessionState.cars,
      inputs,
      sessionState.surfaceResolver,
    );
    sessionState.rules = stepRaceRules(sessionState.rules, {
      dtS: NEATRACING_FIXED_TIMESTEP_SECONDS,
      forwardLapLineCrossing: detector.crossing,
    });
  }
}

/**
 * Builds the structured `frames` reply for one paced frame.
 *
 * @param fromTick - Total-tick counter before the frame's released ticks.
 * @param sessionState - Running session bag (post-tick state).
 * @returns The reply payload snapshot (fresh copies, no aliasing into later
 *   tick output).
 */
function buildSimWorkerFramesPayload(
  fromTick: number,
  sessionState: SimWorkerSessionState,
): NeatRacingSimWorkerFramesPayload {
  return {
    fromTick,
    toTick: sessionState.cars[0].tick,
    cars: sessionState.cars.map((car) => ({ ...car })),
    rules: { ...sessionState.rules },
  };
}

/**
 * Creates a pure sim-worker session state machine.
 *
 * Educational note: one factory per session keeps every boot test isolated
 * without touching `globalThis.self`. The worker entrypoint below builds the
 * module-level session and is the ONLY place that wires `self.onmessage` /
 * `self.postMessage` — the transport adapter stays three lines.
 *
 * @returns A session whose `handleRequest` folds messages into state and
 *   returns the ordered responses to post.
 *
 * @example
 * ```ts
 * const session = createSimWorkerSession();
 * const [ready] = session.handleRequest({ type: 'init', payload });
 * ```
 */
export function createSimWorkerSession(): NeatRacingSimWorkerSession {
  // Step 1: Start the protocol machine in the idle phase.
  const state: SimWorkerSessionState = {
    phase: 'idle',
    renderLoop: createNeatRacingRenderLoopState(),
    cars: [],
    rules: createRaceRulesState(),
    lapLinePrevSide: null,
    raceMode: 'ai',
    humanSample: undefined,
    surfaceResolver: undefined,
    splineSamples: [],
    firstSegment: undefined,
    readyDetail: undefined,
  };

  return {
    handleRequest(
      request: NeatRacingSimWorkerRequest,
    ): NeatRacingSimWorkerResponse[] {
      // Step 2: Reject non-object messages outright (transport-level junk).
      if (request === null || typeof request !== 'object') {
        return [
          {
            type: 'error',
            payload: { message: 'sim worker: malformed message' },
          },
        ];
      }

      // Step 3: Route by message type; `stop` stays deliberately reply-free.
      switch (request.type) {
        case 'init': {
          if (state.phase !== 'idle') {
            return [
              {
                type: 'error',
                payload: {
                  message: `sim worker init failed: session already ${state.phase}`,
                },
              },
            ];
          }
          const validation = validateSimWorkerInitPayload(request.payload);
          if (validation !== null) {
            return [{ type: 'error', payload: { message: validation.error } }];
          }
          const runningState = createSimWorkerRunningState(request.payload);
          Object.assign(state, runningState);
          return [
            {
              type: 'ready',
              payload: {
                seed: runningState.readyDetail.seed,
                agentCount: runningState.readyDetail.agentCount,
                packSchemaVersion: runningState.readyDetail.packSchemaVersion,
                trackId: runningState.readyDetail.trackId,
                raceMode: runningState.raceMode,
              },
            },
          ];
        }
        case 'frame': {
          if (state.phase === 'idle') {
            return [
              {
                type: 'error',
                payload: {
                  message: 'sim worker frame failed: session not initialized',
                },
              },
            ];
          }
          if (state.phase === 'stopped') {
            return [
              {
                type: 'error',
                payload: {
                  message: 'sim worker frame failed: session stopped',
                },
              },
            ];
          }
          const validation = validateSimWorkerFramePayload(request.payload);
          if (validation !== null) {
            return [{ type: 'error', payload: { message: validation.error } }];
          }
          const fromTick = state.cars[0].tick;
          driveSimWorkerFrame(state, request.payload.timestampMs, request.payload);
          return [
            {
              type: 'frames',
              payload: buildSimWorkerFramesPayload(fromTick, state),
            },
          ];
        }
        case 'stop': {
          // Lifecycle message, reply-free by contract (flappy precedent).
          state.phase = 'stopped';
          return [];
        }
        default: {
          const unknownType = String((request as { type?: unknown }).type);
          return [
            {
              type: 'error',
              payload: {
                message: `sim worker: unknown message type ${unknownType}`,
              },
            },
          ];
        }
      }
    },
    get snapshot(): NeatRacingSimWorkerSessionSnapshot {
      return {
        phase: state.phase,
        tick: state.cars.length > 0 ? state.cars[0].tick : 0,
        cars: state.cars.map((car) => ({ ...car })),
        rules: { ...state.rules },
        lapLinePrevSide: state.lapLinePrevSide,
      };
    },
  };
}

/**
 * Pure-function test seam for the sim worker (P2-S1-A green tests).
 *
 * Educational note: like every established worker module in this repo
 * (flappy evolution, asciiMaze), the entrypoint exposes its internals so
 * Jest can drive the protocol contract headless without a worker realm,
 * while production still boots through the module side effect below.
 */
export const NEATRACING_SIMULATION_WORKER_INTERNALS = {
  createSimWorkerSession,
  resolveSimWorkerLapLineDetectorStep,
  resolveSimWorkerFleetSpawnPoses,
  resolveSimWorkerFleetTickInputs,
  NEUTRAL_SIM_WORKER_CONTROL_SAMPLE,
} as const;

// Boot: one session per worker realm; the message adapter is the only
// impurity in this file. Responses are posted in order, exactly once each.
const simulationWorkerSession = createSimWorkerSession();

self.onmessage = (event: MessageEvent<NeatRacingSimWorkerRequest>): void => {
  for (const response of simulationWorkerSession.handleRequest(event.data)) {
    self.postMessage(response, []);
  }
};
