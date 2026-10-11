/**
 * NeatRacing host live drive loop — plan slice P1-drive-loop-live (plan
 * `neatracing-voxel-racer.plans.md`, phase 1, step 14; spec AC-024,
 * AC-042, AC-033, D5; spec sections 5.1, 13.1, 13.2).
 *
 * The session is the host-owned orchestrator of one live race: every
 * rAF callback feeds the real timestamp to the shared 60 Hz accumulator
 * (`render-loop.utils.advanceFrameSteps`), which releases 0-4 fixed
 * simulation ticks per frame. Each released tick samples the keyboard
 * ONCE per frame (D5 — the same sample object is replayed into every
 * catch-up tick), advances the deterministic physics step, then advances
 * the pure race-rules step with `dtS = NEATRACING_FIXED_TIMESTEP_SECONDS`
 * and the per-tick forward-lap-crossing flag from the injected detector
 * seam. Exactly one `hud.update` repaints the read model after the
 * frame's rules transitions, and the frame composer is invoked exactly
 * once per frame after the simulation — including zero-step frames, so
 * rendering continues while the accumulator is starved or clamped.
 *
 * Restart carrier (AC-033 pre-verify): the mode selector's `onSwitch`
 * hook is subscribed exactly once at creation; each real switch restarts
 * the race — fresh spawn car state and fresh rules at the carried level
 * (`nextRaceLevel` when the previous race completed, the current level
 * otherwise), while dropping the simulated backlog (`accumulatedMs`
 * resets to zero) but keeping the frame clock (`lastTimestampMs` tracks
 * the rAF timeline across the restart, so no timestamp burst appears
 * after the switch). The restart deliberately re-derives both fresh
 * states through the direct module factories instead of the injected
 * steppers — the red contract pins field-observable behavior, not
 * factory identity.
 *
 * Allocation policy (AC-042 vehicle): the render-loop buffers (16/16/3/3
 * float multiset) are allocated ONCE through the injected allocator seam
 * at creation and reused across every frame and every restart; the
 * neutral composer frame is a session-lifetime singleton for the
 * not-yet-injected camera builder, and the only per-frame allocations
 * are the per-tick fresh `RaceRulesStepInput` record (contract: each
 * tick's input is an independently observable object) and the small HUD
 * payload bag that pairs the live speed with the final race state.
 *
 * Determinism: the module never reads a clock (the caller supplies the
 * rAF timestamp), rolls dice, or touches the DOM — every dependency
 * (steppers, sampler, detector, HUD, composer, selector, allocator) is
 * injected through {@link NeatRacingRaceSessionDeps}, so the plain node
 * Jest lane drives the whole cadence through pure doubles.
 *
 * @module
 */

import type { NeatRacingFrameComposerFrame } from './frame-composer';
import type { NeatRacingHudState } from './hud';
import type {
  NeatRacingKeyboardPresses,
  NeatRacingKeyboardSample,
} from './input';
import type {
  NeatRacingRaceMode,
  NeatRacingRaceModeListener,
} from './mode-selector';
import { createInitialPhysicsCarState } from '../physics/physics.step.service';
import type {
  PhysicsCarSpawnPose,
  PhysicsCarState,
  PhysicsStepInput,
} from '../physics/physics.step.service';
import { createRaceRulesState } from '../physics/race-rules';
import type { RaceRulesState, RaceRulesStepInput } from '../physics/race-rules';
import {
  advanceFrameSteps,
  createNeatRacingRenderLoopState,
  NEATRACING_FIXED_TIMESTEP_SECONDS,
  type NeatRacingRenderLoopAllocator,
  type NeatRacingRenderLoopState,
} from './render-loop.utils';

/**
 * Live race state exposed to the boot wiring: the container object is
 * created once and kept for the session lifetime while the `car` and
 * `rules` fields are replaced by every tick, so the camera/crossing glue
 * can read the latest simulation pose without holding a fresh reference
 * per frame.
 */
export interface NeatRacingRaceSessionState {
  /** Latest deterministic car state (replaced by every released tick). */
  car: PhysicsCarState;
  /** Latest pure race-rules state (replaced by every released tick). */
  rules: RaceRulesState;
}

/** Frame-level seams of the live drive loop (structural spy subsets). */
export interface NeatRacingRaceSessionDeps {
  /** Frame composer invoked exactly once per frame, after the simulation. */
  readonly composer: {
    /** Draws the composed frame (composer-owned GL state). */
    drawFrame(frame: NeatRacingFrameComposerFrame): void;
  };
  /** HUD read model repainted once per drawing frame. */
  readonly hud: {
    /** Repaints the speed/lap/time labels from live race state. */
    update(state: NeatRacingHudState): void;
  };
  /** Race-mode selector surface; `onSwitch` is the AC-033 restart hook. */
  readonly modeSelector: {
    /** Currently selected mode (read by boot glue; not by this loop). */
    readonly mode: NeatRacingRaceMode;
    /** Subscribes the restart listener; invoked once per actual switch. */
    onSwitch(listener: NeatRacingRaceModeListener): void;
  };
  /** Live pressed-key map sampled once per drawing frame (D5). */
  readonly presses: NeatRacingKeyboardPresses;
  /** Maps the pressed-key map onto the analog axes sample (D5). */
  readonly sampleKeyboardInput: (
    presses: NeatRacingKeyboardPresses,
  ) => NeatRacingKeyboardSample;
  /**
   * Optional autopilot input seam for the `ai` race-mode boot (user-gap
   * patch batch 108): called at most once per drawn frame with the
   * entering car state; when set and the selector mode is `ai`, the
   * session samples this seam instead of the keyboard sample.
   */
  readonly sampleAutoInput?: (car: PhysicsCarState) => PhysicsStepInput;
  /** Deterministic 60 Hz physics step (one tick per call). */
  readonly stepPhysics: (
    car: PhysicsCarState,
    input: PhysicsStepInput,
  ) => PhysicsCarState;
  /** Pure race-rules step (one tick per call; terminal states freeze). */
  readonly stepRaceRules: (
    rules: RaceRulesState,
    input: RaceRulesStepInput,
  ) => RaceRulesState;
  /** Per-tick forward lap-line crossing detector (zero-arg seam). */
  readonly resolveForwardLapLineCrossing: () => boolean;
  /** Optional starting rules state (defaults to the fresh lvl-1 race). */
  readonly initialRulesState?: RaceRulesState;
  /**
   * Optional spawn-pose factory so the boot wiring can place the car on
   * the generated track centerline. When omitted, the default seeded spawn
   * is used for both the initial start and every mode-switch restart.
   */
  readonly createInitialCarState?: () => PhysicsCarState;
  /** Optional render-loop allocator seam (AC-042; buffers born exactly once). */
  readonly allocator?: NeatRacingRenderLoopAllocator;
  /**
   * Optional composer frame builder owned by the boot wiring (camera,
   * projection, and scene bags are boot concerns; this session never
   * derives them). When omitted, the neutral singleton frame below is
   * sent every frame.
   */
  readonly buildComposerFrame?: () => NeatRacingFrameComposerFrame;
}

/** Drivable live race session handed to the host rAF loop. */
export interface NeatRacingRaceSession {
  /** Drives one rAF frame end to end (accumulate, tick, repaint, draw). */
  handleFrame(timestampMs: number): void;
  /** Live simulation seam for the boot camera/crossing glue. */
  readonly state: NeatRacingRaceSessionState;
}

/** Neutral construction-time camera world placeholder `[x, y, z]` meters. */
const NEUTRAL_CAMERA_WORLD: readonly [number, number, number] = [0, 0, 0];

/**
 * Session-lifetime neutral composer frame used whenever the boot wiring
 * has not injected a camera builder: allocated once (a Float32Array not
 * charged to the AC-042 float64 allocator seam), identity-stable, and
 * sent to the composer on every frame until real composition lands.
 */
const NEUTRAL_COMPOSER_FRAME: NeatRacingFrameComposerFrame = {
  viewProjection: new Float32Array(16),
  cameraWorld: NEUTRAL_CAMERA_WORLD,
  cameraPose: { yawRadians: 0, pitchRadians: 0 },
  pulses: [],
  cars: [],
  enabled: true,
};

/**
 * Creates the live drive loop for one race session.
 *
 * Boot sequence: the render-loop buffers are born through the allocator
 * seam exactly once; the simulated states are seeded (optionally from
 * `initialRulesState`); and the selector's `onSwitch` hook is subscribed
 * exactly once to the restart carrier. Every `handleFrame` call then
 * advances the 60 Hz accumulator and releases 0-4 ticks.
 *
 * @param deps - Frame-level seams (composer, HUD, selector, sampler,
 *   steppers, detector) plus the optional allocator/builder seams.
 * @returns The live session whose `handleFrame` drives one rAF frame.
 *
 * @example
 * ```ts
 * const session = createRaceSession({ composer, hud, modeSelector, ... });
 * window.requestAnimationFrame(session.handleFrame);
 * ```
 */
export function createRaceSession(
  deps: NeatRacingRaceSessionDeps,
): NeatRacingRaceSession {
  // AC-042: allocator-driven buffers born exactly once (16/16/3/3 floats).
  const loop: NeatRacingRenderLoopState = createNeatRacingRenderLoopState(
    deps.allocator,
  );

  // One stable state container the boot glue (camera, crossing detector)
  // reads between frames; ticks replace the car/rules records in place.
  const createCar = deps.createInitialCarState ?? createInitialPhysicsCarState;
  const state: NeatRacingRaceSessionState = {
    car: createCar(),
    rules: deps.initialRulesState ?? createRaceRulesState(),
  };

  // AC-033 restart carrier: subscribe exactly once; each notification
  // seeds a fresh spawn car and a fresh rules state at the carried level
  // (the previous race's nextRaceLevel, else its current level), and
  // drops the simulated backlog while KEEPING the frame clock.
  deps.modeSelector.onSwitch(() => {
    state.car = createCar();
    state.rules = createRaceRulesState(
      state.rules.nextRaceLevel ?? state.rules.level,
    );
    loop.accumulatedMs = 0;
  });

  /** True when the boot wiring owns camera/projection frame building. */
  const hasBootComposerFrame = typeof deps.buildComposerFrame === 'function';

  return {
    handleFrame(timestampMs: number): void {
      const stepCount = advanceFrameSteps(loop, timestampMs);

      if (stepCount > 0) {
        // D5: sample the driving surface ONCE per frame (keyboard in the
        // manual mode, the autopilot seam when the selector is on `ai`);
        // replay the SAME sample object into every catch-up tick.
        const input: PhysicsStepInput =
          deps.modeSelector.mode === 'ai' && deps.sampleAutoInput !== undefined
            ? deps.sampleAutoInput(state.car)
            : deps.sampleKeyboardInput(deps.presses);
        for (let stepIndex = 0; stepIndex < stepCount; stepIndex += 1) {
          // The per-tick crossing flag resolves BEFORE the physics step:
          // the detector answers from the pose entering this tick, and the
          // fresh per-tick rule input keeps each tick's flag an
          // independently observable record (never a shared mutation).
          const crossing = deps.resolveForwardLapLineCrossing();
          state.car = deps.stepPhysics(state.car, input);
          state.rules = deps.stepRaceRules(state.rules, {
            dtS: NEATRACING_FIXED_TIMESTEP_SECONDS,
            forwardLapLineCrossing: crossing,
          });
        }

        // One read-model repaint after all of the frame's rules
        // transitions (zero-step frames never touch the HUD).
        deps.hud.update({
          speedUnitsPerS: state.car.speed,
          raceState: state.rules,
        });
      }

      // Exactly one composer draw per frame, AFTER the simulation
      // (including starved/clamped zero-step frames).
      deps.composer.drawFrame(
        hasBootComposerFrame && deps.buildComposerFrame
          ? deps.buildComposerFrame()
          : NEUTRAL_COMPOSER_FRAME,
      );
    },
    state,
  };
}
