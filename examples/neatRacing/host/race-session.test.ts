/**
 * RED -> GREEN contract tests for the NeatRacing host live drive loop
 * (slice P1-drive-loop-live; plan `neatracing-voxel-racer.plans.md`,
 * phase 1, step 14; spec AC-024, AC-042, AC-033, D5; sections 5.1, 13.1,
 * 13.2). The slice AC: "each RAF tick accumulates elapsed time and drives
 * sampleKeyboardInput -> stepPhysics at the 60 Hz cadence (catch-up ticks
 * included) -> race-rules budget/lap/level/terminal transitions ->
 * hud.update + mode-selector refresh live on the served game page; the
 * onSwitch hook wires the AC-033 restart carrier pre-verify; the frame
 * composer is invoked per frame after the simulation accumulation; zero
 * per-frame allocations".
 *
 * RED PREMISE: `./race-session` does not exist yet. Until
 * `04-implementing` lands it, this suite must fail under ts-jest
 * diagnostics with exactly one compile error — "Cannot find module
 * './race-session' or its corresponding type declarations" (TS2307) —
 * while every sibling host suite stays green. A green suite, or a
 * failure with any other reason (a second absent-module import, a
 * fixture/import typo), is a contract defect: every import besides
 * './race-session' resolves to an existing module today. The source
 * module `race-session.ts` is NOT authored in the red phase.
 *
 * GREEN TARGET (what `04-implementing` must make pass, in order):
 * 1. `createRaceSession(deps)` boots one live race session whose
 *    `handleFrame(timestampMs)` advances a 60 Hz accumulator (the
 *    `advanceFrameSteps` cadence) and releases 0-4 simulation ticks;
 *    the first frame (null previous timestamp) releases 0 steps.
 * 2. Keyboard is sampled ONCE per frame through the injected
 *    `sampleKeyboardInput` and the SAME sample object is replayed into
 *    every catch-up tick of that frame (D5).
 * 3. Every released tick calls the injected `stepPhysics` and then the
 *    injected `stepRaceRules` with `dtS = NEATRACING_FIXED_TIMESTEP_SECONDS`
 *    and a per-tick forward-lap-crossing resolution through the injected
 *    detector seam.
 * 4. The recorded transition inputs/results show the real race-rules
 *    semantics: forward crossings extend the checkpoint budget
 *    (+25 s) and complete laps; a spent budget flips 'outOfTime'; the
 *    lap boundary at `NEATRACING_HUMAN_RACE_LAPS` flips 'raceComplete'
 *    and records `nextRaceLevel` (level ladder).
 * 5. `hud.update` is called only after the frame's race-rules
 *    transitions have run (never before them; a zero-step frame repaints
 *    nothing), receiving the live speed and the final race state of the
 *    frame.
 * 6. The frame composer is invoked exactly once per frame, after all of
 *    that frame's physics/rules steps (including zero-step frames —
 *    rendering continues while the accumulator is starved or clamped).
 * 7. The selector `onSwitch` hook is subscribed exactly once at creation
 *    and each notification restarts the race (AC-033 carrier): fresh
 *    spawn car state, fresh rules state at the carried `nextRaceLevel`
 *    (current level when the previous race did not complete, fresh
 *    budget/elapsed either way). The restart drops simulated backlog but
 *    keeps the frame clock (lastTimestampMs tracks real rAF time across
 *    a restart, so no timestamp burst appears after the switch).
 * 8. The render-loop buffers are allocated exactly once through the
 *    injected allocator seam (16/16/3/3 float multiset) and reused
 *    across every frame and restart (AC-042 vehicle; deeper per-frame
 *    heap auditing is not claimed by this red contract — the seam counts
 *    ONLY `float64Array` calls, so composer-side buffer ownership stays
 *    unpinned).
 *
 * Planned public surface (minimum contract for `04-implementing`;
 * additional optional members are allowed but not required):
 * - `createRaceSession(deps: NeatRacingRaceSessionDeps): NeatRacingRaceSession`
 *   with `NeatRacingRaceSessionDeps = { composer: {drawFrame},
 *   hud: {update}, modeSelector: {mode, onSwitch}, presses,
 *   sampleKeyboardInput, stepPhysics, stepRaceRules,
 *   resolveForwardLapLineCrossing, initialRulesState?, allocator? }`
 *   (structural seam subsets of the existing module types).
 * - `session.handleFrame(timestampMs: number): void` drives one rAF
 *   frame end to end; observation happens through the injected seams.
 *
 * Fixture rules: the real `sampleKeyboardInput` / `stepPhysics` /
 * `stepRaceRules` are wrapped in hand-rolled recording doubles (no
 * `jest.fn` — ESM harness parity with frame-composer.test.ts); the
 * frame-composer, HUD, mode selector, crossing detector, and allocator
 * deps are structural spy doubles. Timestamps are exact multiples of the
 * exported `NEATRACING_FIXED_TIMESTEP_MS` /
 * `NEATRACING_CATCHUP_CEILING_MS` so ulp drift cannot change released
 * step counts (the `NEATRACING_STEP_BOUNDARY_EPSILON` floor covers the
 * quotient; `advanceFrameSteps` guarantees the released counts used
 * here). Crossing detection is keyed to the PHYSICS-step timeline (true
 * for the first `crossingTicks` steps), which is robust to the
 * unpinned detector-vs-physics evaluation order within a tick. No DOM,
 * no timers, no RNG — the plain node Jest project drives everything
 * through pure doubles. Every test builds a fresh harness (fresh
 * session, fresh log arrays, fresh counters) so no state crosses a test
 * boundary; there is no teardown because nothing leaks past the harness.
 *
 * Deliberately NOT pinned (implementation freedom): per-step versus
 * per-frame `hud.update` counts (order only), rules stepping after a
 * terminal state (the pure module freezes itself), the composer frame
 * payload contents (camera/projection math belongs to the boot wiring),
 * and how the restart re-derives its fresh states (field-observable
 * behavior is pinned, factory identity is not).
 */

import { describe, expect, it } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  stepPhysics,
  type PhysicsCarState,
  type PhysicsStepInput,
} from '../physics/physics.step.service';
import {
  createRaceRulesState,
  NEATRACING_CHECKPOINT_INITIAL_BUDGET_S,
  NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S,
  NEATRACING_HUMAN_RACE_LAPS,
  stepRaceRules,
  type RaceRulesState,
  type RaceRulesStepInput,
} from '../physics/race-rules';
import { sampleKeyboardInput, type NeatRacingKeyboardPresses } from './input';
import type { NeatRacingHudState } from './hud';
import {
  createRaceSession,
  type NeatRacingRaceSessionDeps,
} from './race-session';
import {
  NEATRACING_CATCHUP_CEILING_MS,
  NEATRACING_FIXED_TIMESTEP_MS,
  NEATRACING_FIXED_TIMESTEP_SECONDS,
  type NeatRacingRenderLoopAllocator,
} from './render-loop.utils';
import type {
  NeatRacingRaceMode,
  NeatRacingRaceModeListener,
} from './mode-selector';
import type { NeatRacingFrameComposerFrame } from './frame-composer';

/** Recorded physics tick (the pre-step state plus the replayed sample). */
interface PhysicsRecord {
  readonly state: PhysicsCarState;
  readonly input: PhysicsStepInput;
}

/** Live race-session test harness rebuilt fresh for every test. */
interface RaceSessionHarness {
  readonly session: { handleFrame(timestampMs: number): void };
  readonly log: string[];
  readonly physicsRecords: PhysicsRecord[];
  readonly rulesInputs: RaceRulesStepInput[];
  readonly rulesStates: RaceRulesState[];
  readonly hudStates: NeatRacingHudState[];
  readonly composerFrames: NeatRacingFrameComposerFrame[];
  readonly switchListeners: NeatRacingRaceModeListener[];
  readonly allocatorCalls: number[];
  /** Cars passed to the autopilot seam, in call order (ai-mode tests). */
  readonly autoSampleCalls: PhysicsCarState[];
  /** Runs one rAF frame at the given timestamp (first frame may be 0). */
  runFrame(timestampMs: number): void;
  /** Fires the AC-033 carrier notification with the switched-to mode. */
  fireSwitch(mode: NeatRacingRaceMode): void;
}

/** Harness config: all fields optional, all defaults deterministic. */
interface RaceSessionHarnessConfig {
  /** Keyboard press fixture for the whole test (never mutated). */
  readonly presses?: NeatRacingKeyboardPresses;
  /** Forward crossing fed to the first N simulation ticks. */
  readonly crossingTicks?: number;
  /** Optional initial race-rules state override (lap/budget fixtures). */
  readonly initialRulesState?: RaceRulesState;
  /** Selector mode for the whole test (default 'manual'). */
  readonly mode?: NeatRacingRaceMode;
  /**
   * Optional autopilot seam recording fixture; recorded calls land in
   * the harness `autoSampleCalls` list.
   */
  readonly sampleAutoInput?: (car: PhysicsCarState) => PhysicsStepInput;
}

/** Throttle-only press fixture (deterministic, no steer). */
const THROTTLE_PRESSES: NeatRacingKeyboardPresses = {
  up: true,
  down: false,
  left: false,
  right: false,
};

/** Left-steer press fixture (steer axis exactly +1 through the sampler). */
const LEFT_STEER_PRESSES: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: true,
  right: false,
};

/** Neutral (all-keys-up) press fixture. */
const NEUTRAL_PRESSES: NeatRacingKeyboardPresses = {
  up: false,
  down: false,
  left: false,
  right: false,
};

/** Log marker pushed before each frame's call log entries. */
const FRAME_BEGIN = 'frame-begin';

/** Counts one chronological-log tag. */
function countTag(log: readonly string[], tag: string): number {
  let count = 0;
  for (const entry of log) {
    if (entry === tag) count += 1;
  }
  return count;
}

/** Splits the chronological call log into per-frame partitions. */
function framePartitions(log: readonly string[]): string[][] {
  const partitions: string[][] = [];
  let current: string[] = [];
  for (const entry of log) {
    if (entry === FRAME_BEGIN) {
      if (current.length > 0) partitions.push(current);
      current = [];
    } else {
      current.push(entry);
    }
  }
  if (current.length > 0) partitions.push(current);
  return partitions;
}

/**
 * True when every `tag` occurrence in the partition has at least one
 * `afterTags` occurrence strictly earlier (ordering-only assertion).
 */
function isEveryTagAfterAny(
  partition: readonly string[],
  tag: string,
  afterTags: readonly string[],
): boolean {
  return partition.every(
    (entry, index) =>
      entry !== tag ||
      partition.slice(0, index).some((before) => afterTags.includes(before)),
  );
}

/**
 * True when every `tag` occurrence in the partition comes strictly after
 * the last occurrence of every `beforeTags` entry (ordering-only).
 */
function isTagAfterAll(
  partition: readonly string[],
  tag: string,
  beforeTags: readonly string[],
): boolean {
  let lastBefore = -1;
  partition.forEach((entry, index) => {
    if (beforeTags.some((before) => entry === before)) lastBefore = index;
  });
  return partition.every((entry, index) => entry !== tag || index > lastBefore);
}

/**
 * Builds one recording harness around the real pure modules. The deps
 * object is annotated with the planned `NeatRacingRaceSessionDeps` shape
 * so the doubles stay structurally checked against the contract.
 */
function createHarness(
  config: RaceSessionHarnessConfig = {},
): RaceSessionHarness {
  const log: string[] = [];
  const physicsRecords: PhysicsRecord[] = [];
  const rulesInputs: RaceRulesStepInput[] = [];
  const rulesStates: RaceRulesState[] = [];
  const hudStates: NeatRacingHudState[] = [];
  const composerFrames: NeatRacingFrameComposerFrame[] = [];
  const switchListeners: NeatRacingRaceModeListener[] = [];
  const autoSampleCalls: PhysicsCarState[] = [];
  const sampleAutoInput = config.sampleAutoInput;
  const allocatorCalls: number[] = [];
  const crossingTicks = config.crossingTicks ?? 0;
  let crossingStepsLeft = crossingTicks;

  const allocator: NeatRacingRenderLoopAllocator = {
    float64Array: (length: number): Float64Array => {
      allocatorCalls.push(length);
      return new Float64Array(length);
    },
  };

  const deps: NeatRacingRaceSessionDeps = {
    composer: {
      drawFrame: (frame: NeatRacingFrameComposerFrame): void => {
        composerFrames.push(frame);
        log.push('composer');
      },
    },
    hud: {
      update: (state: NeatRacingHudState): void => {
        hudStates.push(state);
        log.push('hud');
      },
    },
    modeSelector: {
      mode: config.mode ?? 'manual',
      onSwitch: (listener: NeatRacingRaceModeListener): void => {
        switchListeners.push(listener);
      },
    },
    presses: config.presses ?? NEUTRAL_PRESSES,
    ...(sampleAutoInput !== undefined
      ? {
          sampleAutoInput: (car: PhysicsCarState): PhysicsStepInput => {
            autoSampleCalls.push(car);
            return sampleAutoInput(car);
          },
        }
      : {}),
    sampleKeyboardInput: (presses: NeatRacingKeyboardPresses) => {
      log.push('sample');
      return sampleKeyboardInput(presses);
    },
    stepPhysics: (state: PhysicsCarState, input: PhysicsStepInput) => {
      physicsRecords.push({ state, input });
      log.push('physics');
      const next = stepPhysics(state, input);
      if (crossingStepsLeft > 0) crossingStepsLeft -= 1;
      return next;
    },
    stepRaceRules: (state: RaceRulesState, input: RaceRulesStepInput) => {
      rulesInputs.push(input);
      log.push('rules');
      const next = stepRaceRules(state, input);
      rulesStates.push(next);
      return next;
    },
    resolveForwardLapLineCrossing: (): boolean => crossingStepsLeft > 0,
    ...(config.initialRulesState
      ? { initialRulesState: config.initialRulesState }
      : {}),
    allocator,
  };

  const session = createRaceSession(deps);

  return {
    session,
    log,
    physicsRecords,
    rulesInputs,
    rulesStates,
    hudStates,
    composerFrames,
    switchListeners,
    allocatorCalls,
    autoSampleCalls,
    runFrame(timestampMs: number): void {
      log.push(FRAME_BEGIN);
      session.handleFrame(timestampMs);
    },
    fireSwitch(mode: NeatRacingRaceMode): void {
      for (const listener of switchListeners) listener(mode);
    },
  };
}

describe('createRaceSession (P1-drive-loop-live red contract)', () => {
  describe('60 Hz cadence across catch-up ticks (AC-024, D5)', () => {
    it('samples the autopilot seam once per frame in the ai mode (keyboard untouched)', () => {
      const autoFixedInput: PhysicsStepInput = {
        throttle: 1,
        brake: 0,
        steer: 0.25,
      };
      const harness = createHarness({
        mode: 'ai',
        sampleAutoInput: () => autoFixedInput,
      });
      harness.runFrame(16);
      harness.runFrame(33);
      expect(harness.autoSampleCalls.length).toBe(1);
      expect(harness.physicsRecords.length).toBe(1);
      expect(
        harness.physicsRecords.every(
          (record) => record.input === autoFixedInput,
        ),
      ).toBe(true);
      expect(harness.log.includes('sample')).toBe(false);
    });

    it('never touches the autopilot seam in the manual mode', () => {
      const sampleAutoInput = (car: PhysicsCarState): PhysicsStepInput => ({
        throttle: 1,
        brake: 0,
        steer: 0,
      });
      const harness = createHarness({ sampleAutoInput });
      harness.runFrame(16);
      harness.runFrame(33);
      expect(harness.autoSampleCalls.length).toBe(0);
    });

    it('releases zero steps on the first frame (no accumulated elapsed time yet)', () => {
      // Arrange
      const harness = createHarness();
      // Act
      harness.runFrame(0);
      // Assert — nothing stepped, sampled, or repainted before real time flows.
      expect({
        sample: countTag(harness.log, 'sample'),
        physics: countTag(harness.log, 'physics'),
        rules: countTag(harness.log, 'rules'),
        hud: countTag(harness.log, 'hud'),
      }).toEqual({ sample: 0, physics: 0, rules: 0, hud: 0 });
    });

    it('replays one keyboard sample into every catch-up tick of a saturated frame (D5)', () => {
      // Arrange — left steer so every replayed input carries steer exactly +1.
      const harness = createHarness({ presses: LEFT_STEER_PRESSES });
      // Act — first frame releases nothing; the ceiling frame releases 4 ticks.
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert
      expect(harness.physicsRecords.length).toBe(4);
      expect(
        harness.physicsRecords.every(
          (record) =>
            record.input === harness.physicsRecords[0]!.input &&
            record.input.steer === 1,
        ),
      ).toBe(true);
      expect(countTag(harness.log, 'sample')).toBe(1);
    });

    it('releases exactly the catch-up cap for a clamped oversized frame delta', () => {
      // Arrange — delta ~1067 ms exceeds NEATRACING_MAX_FRAME_DELTA_MS.
      const harness = createHarness();
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS + 1000);
      // Assert — the clamp plus cap release exactly 4 ticks, never more.
      expect(harness.rulesInputs.length).toBe(4);
      expect(harness.physicsRecords.length).toBe(4);
    });

    it('advances each released tick by exactly one fixed timestep in the race rules', () => {
      // Arrange
      const harness = createHarness();
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — every rules tick consumes one 1/60 s slice.
      expect(harness.rulesInputs.length).toBe(4);
      expect(
        harness.rulesInputs.every(
          (input) =>
            Math.abs(input.dtS - NEATRACING_FIXED_TIMESTEP_SECONDS) < 1e-12,
        ),
      ).toBe(true);
    });
  });

  describe('race-rules transition calls (budget / lap / level / terminal)', () => {
    it('feeds the forward crossing into the transitions so the budget extends and the lap completes', () => {
      // Arrange — one forward crossing on the first simulation tick.
      const harness = createHarness({ crossingTicks: 1 });
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — crossing recorded on tick 1 only; budget 60 - 4/60 + 25.
      expect(
        harness.rulesInputs.map((input) => input.forwardLapLineCrossing),
      ).toEqual([true, false, false, false]);
      const final = harness.rulesStates[harness.rulesStates.length - 1]!;
      expect(final.lapsCompleted).toBe(1);
      expect(final.checkpointBudgetS).toBeCloseTo(
        NEATRACING_CHECKPOINT_INITIAL_BUDGET_S -
          NEATRACING_FIXED_TIMESTEP_SECONDS * 4 +
          NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S,
        9,
      );
    });

    it('flips the race out of time once the checkpoint budget is spent', () => {
      // Arrange — 0.02 s budget, no crossings: tick 1 survives, tick 2 flips.
      const harness = createHarness({
        initialRulesState: {
          ...createRaceRulesState(),
          checkpointBudgetS: 0.02,
        },
      });
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — the terminal flip lands with the clock decayed by 2 ticks
      // (later ticks, if any, stay inert: the pure module freezes itself).
      const final = harness.rulesStates[harness.rulesStates.length - 1]!;
      expect(final.raceStatus).toBe('outOfTime');
      expect(final.checkpointBudgetS).toBe(0);
      expect(final.elapsedS).toBeCloseTo(
        NEATRACING_FIXED_TIMESTEP_SECONDS * 2,
        12,
      );
    });

    it('completes the race and records the next level exactly at the lap boundary', () => {
      // Arrange — one lap left: the crossing tick completes the race.
      const harness = createHarness({
        presses: THROTTLE_PRESSES,
        crossingTicks: 1,
        initialRulesState: {
          ...createRaceRulesState(),
          lapsCompleted: NEATRACING_HUMAN_RACE_LAPS - 1,
        },
      });
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — terminal completion carries the level ladder forward.
      const final = harness.rulesStates[harness.rulesStates.length - 1]!;
      expect(final.raceStatus).toBe('raceComplete');
      expect(final.nextRaceLevel).toBe(2);
      expect(final.lapsCompleted).toBe(NEATRACING_HUMAN_RACE_LAPS);
    });
  });

  describe('hud and frame-composer cadence', () => {
    it('updates the hud only after the frame race-rules transitions have run', () => {
      // Arrange
      const harness = createHarness({ presses: THROTTLE_PRESSES });
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — every hud repaint sits strictly after a rules transition
      // of its frame (order-only: per-step update counting stays unpinned).
      expect(harness.hudStates.length).toBeGreaterThanOrEqual(1);
      expect(
        framePartitions(harness.log)
          .filter((partition) => countTag(partition, 'hud') > 0)
          .every((partition) =>
            isEveryTagAfterAny(partition, 'hud', ['rules']),
          ),
      ).toBe(true);
    });

    it('hands the hud the live speed and the final race state of the frame', () => {
      // Arrange — throttle held: the car moves, so the speed readout is live.
      const harness = createHarness({ presses: THROTTLE_PRESSES });
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — the last hud readout reflects the post-transition state.
      const last = harness.hudStates[harness.hudStates.length - 1]!;
      expect({
        level: last.raceState.level,
        lapsCompleted: last.raceState.lapsCompleted,
        raceStatus: last.raceState.raceStatus,
      }).toEqual({ level: 1, lapsCompleted: 0, raceStatus: 'running' });
      expect(last.raceState.checkpointBudgetS).toBeCloseTo(
        NEATRACING_CHECKPOINT_INITIAL_BUDGET_S -
          NEATRACING_FIXED_TIMESTEP_SECONDS * 4,
        9,
      );
      expect(last.speedUnitsPerS).toBeGreaterThan(0);
    });

    it('invokes the frame composer exactly once per frame after the simulation steps', () => {
      // Arrange — a zero-step frame, a one-step frame, a saturated frame.
      const harness = createHarness({ presses: THROTTLE_PRESSES });
      // Act
      harness.runFrame(0);
      harness.runFrame(NEATRACING_FIXED_TIMESTEP_MS);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Assert — one compose per frame, always after physics and rules.
      expect(harness.composerFrames.length).toBe(3);
      expect(
        framePartitions(harness.log).map((partition) =>
          countTag(partition, 'composer'),
        ),
      ).toEqual([1, 1, 1]);
      expect(
        framePartitions(harness.log).every((partition) =>
          isTagAfterAll(partition, 'composer', ['physics', 'rules']),
        ),
      ).toBe(true);
    });
  });

  describe('AC-033 restart carrier (mode-selector onSwitch)', () => {
    it('subscribes exactly one onSwitch restart listener at creation', () => {
      // Arrange + Act
      const harness = createHarness();
      // Assert
      expect(harness.switchListeners.length).toBe(1);
    });

    it('restarts on switch at the level carried from the completed race', () => {
      // Arrange — finish the race so the ladder records nextRaceLevel 2.
      const harness = createHarness({
        presses: THROTTLE_PRESSES,
        crossingTicks: 1,
        initialRulesState: {
          ...createRaceRulesState(),
          lapsCompleted: NEATRACING_HUMAN_RACE_LAPS - 1,
        },
      });
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      // Act — the AC-033 carrier fires, then one continuation frame runs.
      harness.fireSwitch('ai');
      harness.runFrame(
        NEATRACING_CATCHUP_CEILING_MS + NEATRACING_FIXED_TIMESTEP_MS,
      );
      // Assert — fresh race at the carried level, fresh budget, fresh car.
      const restarted = harness.rulesStates[harness.rulesStates.length - 1]!;
      expect({
        level: restarted.level,
        lapsCompleted: restarted.lapsCompleted,
        raceStatus: restarted.raceStatus,
      }).toEqual({ level: 2, lapsCompleted: 0, raceStatus: 'running' });
      expect(restarted.checkpointBudgetS).toBeCloseTo(
        NEATRACING_CHECKPOINT_INITIAL_BUDGET_S -
          NEATRACING_FIXED_TIMESTEP_SECONDS,
        9,
      );
      expect({
        ...harness.physicsRecords[harness.physicsRecords.length - 1]!.state,
      }).toEqual({ ...createInitialPhysicsCarState() });
    });

    it('restarts on switch at the current level with fresh race clock state', () => {
      // Arrange — run one step of an uncompleted race.
      const harness = createHarness({ presses: THROTTLE_PRESSES });
      harness.runFrame(0);
      harness.runFrame(NEATRACING_FIXED_TIMESTEP_MS);
      // Act — a mid-race switch restarts at the current level (no reset burst).
      harness.fireSwitch('ai');
      harness.runFrame(NEATRACING_FIXED_TIMESTEP_MS * 2);
      // Assert — only the post-restart tick's decay remains on the clock.
      const restarted = harness.rulesStates[harness.rulesStates.length - 1]!;
      expect({
        level: restarted.level,
        raceStatus: restarted.raceStatus,
      }).toEqual({ level: 1, raceStatus: 'running' });
      expect(restarted.elapsedS).toBeCloseTo(
        NEATRACING_FIXED_TIMESTEP_SECONDS,
        12,
      );
      expect(restarted.checkpointBudgetS).toBeCloseTo(
        NEATRACING_CHECKPOINT_INITIAL_BUDGET_S -
          NEATRACING_FIXED_TIMESTEP_SECONDS,
        9,
      );
    });
  });

  describe('zero per-frame allocation seam (AC-042)', () => {
    it('allocates the render-loop buffers once at creation and reuses them across frames', () => {
      // Arrange — counting allocator seam (float64Array calls only).
      const harness = createHarness({ presses: THROTTLE_PRESSES });
      expect([...harness.allocatorCalls].sort((a, b) => a - b)).toEqual([
        3, 3, 16, 16,
      ]);
      // Act — frames with catch-up ticks and a clamped stall reuse the buffers.
      harness.runFrame(0);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS);
      harness.runFrame(NEATRACING_CATCHUP_CEILING_MS + 1000);
      // Assert — no new float64 allocations through the seam after creation.
      expect([...harness.allocatorCalls].sort((a, b) => a - b)).toEqual([
        3, 3, 16, 16,
      ]);
    });
  });
});
