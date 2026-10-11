/**
 * Contract tests for the Neat Racing race rules (slice P1-S7-D; AC-024
 * partial, spec §4.1/§4.2/§20 OQ-001/OQ-002, TP-12/TP-13).
 *
 * Red phase: this suite must fail while './race-rules' does not exist
 * ("Cannot find module './race-rules'" under ts-jest diagnostics); a green
 * suite during the red phase would be a contract defect. The plan red_test
 * wording ("asserts checkpoint budget math and lap counting") was
 * reconciled against the spec source of truth: the OQ-001 checkpoint
 * budget constants (initial 60 s + 25 s per lap-line crossing), the 20-lap
 * human race with lap counting at forward finish-line crossings, and the
 * OQ-002 level rules (first session lvl 1, race-completion boundary
 * advancement, lvl 3 cap). The spec is silent on budget exhaustion, so the
 * arcade-standard semantics were pinned here by design: a step whose
 * remaining budget reaches zero ends the race out-of-time, a crossing
 * granted in the same tick saves the run before that check, and a failed
 * race never advances the level.
 *
 * Determinism rung: same-runtime byte-identical via `JSON.stringify`
 * serialization (reproducibility Level 2, ordered deterministic); every
 * rule is pure — no `Math.random`, no clock reads, the caller supplies
 * `dtS`, so the same `(state, dt stream, crossing stream)` reproduces the
 * identical race state on the same runtime. Not claimed: cross-environment
 * byte equality.
 *
 * Fixture rules: all fixture deltas are binary-exact (0.25 s, 0.5 s, 1 s,
 * 25 s), so budget/lap assertions use bare `toBe` equality with no
 * tolerance; the single sub-second survival fixture uses the exported
 * constants so no magic threshold is introduced. The out-of-time and
 * dying-tick fixtures are hand-built literals of the exported state shape
 * because the stepper contract makes arbitrary partial states reachable
 * legal inputs (a fresh state cannot reach budget 0.005 by whole steps).
 *
 * Green contract for the implementer: './race-rules' exports the named
 * constants NEATRACING_CHECKPOINT_INITIAL_BUDGET_S (60),
 * NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S (25), NEATRACING_HUMAN_RACE_LAPS
 * (20), NEATRACING_HUMAN_START_LEVEL (1), and NEATRACING_HUMAN_MAX_LEVEL
 * (3, the lvl 3 cap), the factory
 * `createRaceRulesState(level = NEATRACING_HUMAN_START_LEVEL)` returning a
 * fresh session state `{ level, lapsCompleted: 0, checkpointBudgetS:
 * NEATRACING_CHECKPOINT_INITIAL_BUDGET_S, elapsedS: 0,
 * raceStatus: 'running', nextRaceLevel: null }`, and the pure stepper
 * `stepRaceRules(state, { dtS, forwardLapLineCrossing })` with
 * `RaceStatus = 'running' | 'raceComplete' | 'outOfTime'` and
 * `nextRaceLevel: number | null`, where every step of a running race first
 * decays `checkpointBudgetS` and advances `elapsedS` by `dtS`, then — only
 * on a forward lap-line crossing — increments `lapsCompleted` and grants
 * the 25 s extension, marks the race `raceComplete` exactly when
 * `lapsCompleted` reaches `NEATRACING_HUMAN_RACE_LAPS` (resolved at that
 * crossing tick, never before), clamps the budget at 0, and switches a
 * zero-budget running race to `outOfTime` unless the same tick granted a
 * crossing (extension applies first; the line saves the second), while
 * steps on either terminal status (`raceComplete` / `outOfTime`) return the
 * state unchanged; `nextRaceLevel` stays `null` until completion and is
 * then `min(level + 1, NEATRACING_HUMAN_MAX_LEVEL)` — the NEXT race's
 * carried level (session-boundary advancement, no mid-race insertion).
 */

import { describe, expect, it } from '@jest/globals';
import {
  createRaceRulesState,
  NEATRACING_CHECKPOINT_INITIAL_BUDGET_S,
  NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S,
  NEATRACING_HUMAN_MAX_LEVEL,
  NEATRACING_HUMAN_RACE_LAPS,
  NEATRACING_HUMAN_START_LEVEL,
  stepRaceRules,
} from './race-rules';
import type { RaceRulesState } from './race-rules';

/** Byte-identity serializer mirroring the sibling physics/track suites. */
const serializeState = (state: RaceRulesState): string => JSON.stringify(state);

/** 0.25 s tick: binary-exact, so budget/elapsed sums stay exact binary. */
const FIXTURE_TICK_S = 0.25;

/**
 * Runs `count` non-crossing ticks of `dtS` seconds from a fresh race,
 * covering the "race keeps running" budget/elapsed accumulation cases.
 */
const runInertTicks = (count: number, dtS = FIXTURE_TICK_S): RaceRulesState => {
  let state = createRaceRulesState();
  for (let tick = 0; tick < count; tick++) {
    state = stepRaceRules(state, { dtS, forwardLapLineCrossing: false });
  }
  return state;
};

/**
 * Runs `crossings` forward lap-line crossings from a race starting at
 * `level`, using dt 0 so only crossing math moves the state (budget grows
 * by the per-lap extension; laps count up) until the race completes.
 */
const runLaps = (
  crossings: number,
  level = NEATRACING_HUMAN_START_LEVEL,
): RaceRulesState => {
  let state = createRaceRulesState(level);
  for (let lap = 0; lap < crossings; lap++) {
    state = stepRaceRules(state, { dtS: 0, forwardLapLineCrossing: true });
  }
  return state;
};

/** Hand-built state one sub-second tick before budget exhaustion. */
const DYING_TICK_STATE: RaceRulesState = {
  level: 1,
  lapsCompleted: 4,
  checkpointBudgetS: 0.005,
  elapsedS: 295,
  raceStatus: 'running',
  nextRaceLevel: null,
};

/** Hand-built terminal state after the budget ran out mid-race. */
const OUT_OF_TIME_STATE: RaceRulesState = {
  level: 1,
  lapsCompleted: 7,
  checkpointBudgetS: 0,
  elapsedS: 300,
  raceStatus: 'outOfTime',
  nextRaceLevel: null,
};

describe('race rules constants (spec §4.1/§4.2, AC-024, OQ-001/OQ-002)', () => {
  it('exposes the OQ-001 checkpoint budget constants', () => {
    expect(NEATRACING_CHECKPOINT_INITIAL_BUDGET_S).toBe(60);
    expect(NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S).toBe(25);
  });

  it('exposes the OQ-002 human race and level constants', () => {
    expect(NEATRACING_HUMAN_RACE_LAPS).toBe(20);
    expect(NEATRACING_HUMAN_START_LEVEL).toBe(1);
    expect(NEATRACING_HUMAN_MAX_LEVEL).toBe(3);
  });
});

describe('race rules initial state', () => {
  it('starts the first session at lvl 1 with no laps and no elapsed time', () => {
    const state = createRaceRulesState();

    expect(state.level).toBe(NEATRACING_HUMAN_START_LEVEL);
    expect(state.lapsCompleted).toBe(0);
    expect(state.elapsedS).toBe(0);
  });

  it('seeds the 60 s checkpoint budget on a running race', () => {
    const state = createRaceRulesState();

    expect(state.checkpointBudgetS).toBe(
      NEATRACING_CHECKPOINT_INITIAL_BUDGET_S,
    );
    expect(state.raceStatus).toBe('running');
    expect(state.nextRaceLevel).toBeNull();
  });

  it('starts a follow-up session at the carried level', () => {
    const state = createRaceRulesState(2);

    expect(state.level).toBe(2);
  });
});

describe('race rules checkpoint budget math (OQ-001, TP-13)', () => {
  it('decays the checkpoint budget by the elapsed step time', () => {
    const state = stepRaceRules(createRaceRulesState(), {
      dtS: 1,
      forwardLapLineCrossing: false,
    });

    expect(state.checkpointBudgetS).toBe(
      NEATRACING_CHECKPOINT_INITIAL_BUDGET_S - 1,
    );
    expect(state.raceStatus).toBe('running');
  });

  it('accumulates many small ticks without float drift', () => {
    const state = runInertTicks(40);

    expect(state.checkpointBudgetS).toBe(
      NEATRACING_CHECKPOINT_INITIAL_BUDGET_S - 10,
    );
    expect(state.elapsedS).toBe(10);
  });

  it('extends the checkpoint budget by 25 s at a lap-line crossing', () => {
    const state = stepRaceRules(createRaceRulesState(), {
      dtS: 0.5,
      forwardLapLineCrossing: true,
    });

    expect(state.checkpointBudgetS).toBe(
      NEATRACING_CHECKPOINT_INITIAL_BUDGET_S -
        0.5 +
        NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S,
    );
  });

  it('clamps the budget at zero and ends the race out of time', () => {
    const state = stepRaceRules(createRaceRulesState(), {
      dtS: 65,
      forwardLapLineCrossing: false,
    });

    expect(state.checkpointBudgetS).toBe(0);
    expect(state.raceStatus).toBe('outOfTime');
    expect(state.nextRaceLevel).toBeNull();
  });

  it('keeps the race running while any positive budget remains', () => {
    const state = runInertTicks(239);

    expect(state.raceStatus).toBe('running');
  });

  it('ends the race out of time exactly when the budget reaches zero', () => {
    const state = runInertTicks(240);

    expect(state.checkpointBudgetS).toBe(0);
    expect(state.raceStatus).toBe('outOfTime');
  });

  it('counts a crossing granted in the same tick as budget exhaustion', () => {
    const state = stepRaceRules(DYING_TICK_STATE, {
      dtS: 1 / 60,
      forwardLapLineCrossing: true,
    });

    expect(state.raceStatus).toBe('running');
    expect(state.checkpointBudgetS).toBeCloseTo(
      DYING_TICK_STATE.checkpointBudgetS +
        NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S -
        1 / 60,
      9,
    );
    expect(state.lapsCompleted).toBe(5);
  });
});

describe('race rules lap counting (AC-024)', () => {
  it('counts one lap per forward lap-line crossing while the race runs', () => {
    const state = runLaps(3);

    expect(state.lapsCompleted).toBe(3);
  });

  it('does not count laps on steps without a forward crossing', () => {
    const state = runInertTicks(100);

    expect(state.lapsCompleted).toBe(0);
  });

  it('completes the race exactly at the 20th lap crossing', () => {
    const state = runLaps(NEATRACING_HUMAN_RACE_LAPS);

    expect(state.lapsCompleted).toBe(NEATRACING_HUMAN_RACE_LAPS);
    expect(state.raceStatus).toBe('raceComplete');
  });

  it('freezes lap counting after the race completes', () => {
    const completed = runLaps(NEATRACING_HUMAN_RACE_LAPS);
    const overCrossed = stepRaceRules(completed, {
      dtS: 0,
      forwardLapLineCrossing: true,
    });

    expect(serializeState(overCrossed)).toBe(serializeState(completed));
  });
});

describe('race rules level advancement (OQ-002, TP-12)', () => {
  it('advances one level at the race-completion boundary', () => {
    const state = runLaps(NEATRACING_HUMAN_RACE_LAPS, 1);

    expect(state.nextRaceLevel).toBe(2);
  });

  it('advances from lvl 2 into the lvl 3 cap', () => {
    const state = runLaps(NEATRACING_HUMAN_RACE_LAPS, 2);

    expect(state.nextRaceLevel).toBe(NEATRACING_HUMAN_MAX_LEVEL);
  });

  it('caps the next level at lvl 3 for an already-maximal race', () => {
    const state = runLaps(
      NEATRACING_HUMAN_RACE_LAPS,
      NEATRACING_HUMAN_MAX_LEVEL,
    );

    expect(state.nextRaceLevel).toBe(NEATRACING_HUMAN_MAX_LEVEL);
  });
});

describe('race rules terminal inertness (no mid-race or post-race drift)', () => {
  it('leaves a completed race state unchanged under further steps', () => {
    const completed = runLaps(NEATRACING_HUMAN_RACE_LAPS);
    const stepped = stepRaceRules(completed, {
      dtS: 1,
      forwardLapLineCrossing: true,
    });

    expect(serializeState(stepped)).toBe(serializeState(completed));
  });

  it('leaves an out-of-time race state unchanged under further steps', () => {
    const stepped = stepRaceRules(OUT_OF_TIME_STATE, {
      dtS: 1,
      forwardLapLineCrossing: false,
    });

    expect(serializeState(stepped)).toBe(serializeState(OUT_OF_TIME_STATE));
  });
});
