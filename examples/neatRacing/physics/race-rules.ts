/**
 * Race rules (laps, checkpoints, level advancement) for the neatRacing
 * example (plan slice P1-S7-D; AC-024 partial; spec sections 4.1, 4.2,
 * 20 OQ-001/OQ-002; TP-12/TP-13).
 *
 * Session semantics (OQ-001 resolved values as named constants, TP-13): a
 * human race lasts NEATRACING_HUMAN_RACE_LAPS forward laps with an
 * arcade-style checkpoint budget that starts at
 * NEATRACING_CHECKPOINT_INITIAL_BUDGET_S and is extended by
 * NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S at every forward lap-line
 * crossing. The spec is silent on budget exhaustion, so the arcade-
 * standard rule is pinned here by design: a step whose remaining budget
 * reaches zero ends the race out-of-time, a crossing granted in the same
 * tick saves the run before that check (the extension applies first; the
 * line saves the second), and the budget is clamped at zero (never
 * negative).
 *
 * Level advancement (OQ-002, TP-12): the first session starts at
 * NEATRACING_HUMAN_START_LEVEL and completing a race is the session
 * boundary — the NEXT race carries
 * min(level + 1, NEATRACING_HUMAN_MAX_LEVEL) in `nextRaceLevel` (capped
 * at lvl 3; a failed race never advances). Advancement only materializes
 * at the completion boundary; no mid-race level change ever happens.
 *
 * Determinism: the module is pure — no `Math.random`, no clock reads,
 * the caller supplies `dtS`. The stepper mutates nothing and returns
 * fresh state records while a race runs, and steps on a terminal status
 * ('raceComplete' / 'outOfTime') return the input record itself
 * unchanged, so the same (state, dt stream, crossing stream) reproduces
 * byte-identical `JSON.stringify` states on the same runtime
 * (same-runtime ordered-deterministic rung; cross-environment byte
 * equality is NOT claimed).
 *
 * Scope notes: HUD display of laps/budget and the mode selector arrive
 * with P1-S7-E; AI-mode lap floors and the promotion predicate are
 * phase-2 rules and never live here. This module is the pure rules core
 * the boot glue (Step 13) wires into the host loop.
 */

/**
 * Lifecycle status of one race session. A terminal status
 * ('raceComplete' / 'outOfTime') freezes the whole state: further steps
 * on a terminal record return the input record unchanged.
 */
export type RaceStatus = 'running' | 'raceComplete' | 'outOfTime';

/**
 * Full race-rules session state for one race. Small enough to
 * JSON-serialize byte-identically (the same-runtime determinism rung),
 * so replay suites can pin rule behavior across repeated runs.
 */
export interface RaceRulesState {
  /** Session level this state belongs to (up to NEATRACING_HUMAN_MAX_LEVEL). */
  level: number;
  /** Forward laps completed so far (0 = none; frozen once the race terminates). */
  lapsCompleted: number;
  /**
   * Remaining checkpoint seconds: decays by `dtS` every step, extended by
   * NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S per forward crossing, never
   * below zero.
   */
  checkpointBudgetS: number;
  /** Total raced seconds; advances by `dtS` every non-terminal step. */
  elapsedS: number;
  /** Lifecycle status; flips to 'raceComplete' exactly at the last lap. */
  raceStatus: RaceStatus;
  /**
   * NEXT race's carried level (session-boundary advancement), set only at
   * the race-completion boundary; null while the race runs and null after
   * a failed ('outOfTime') race.
   */
  nextRaceLevel: number | null;
}

/**
 * Inputs of one race step. The host samples real frame time once per
 * step (D5 analog; replay suites pass binary-exact fixture deltas), and
 * the forward-crossing flag is set by the track-crossing detector only
 * for forward, once-per-circuit line passes.
 */
export interface RaceRulesStepInput {
  /** Elapsed seconds this step advances the race by (0 = crossing-only step). */
  dtS: number;
  /** True only when the car crossed the lap line FORWARD this step. */
  forwardLapLineCrossing: boolean;
}

/**
 * Initial human checkpoint budget in seconds (OQ-001 resolved, TP-13):
 * the arcade per-race time budget that every raced second spends down
 * and every lap-line crossing refills.
 */
export const NEATRACING_CHECKPOINT_INITIAL_BUDGET_S = 60;

/**
 * Checkpoint extension granted at every forward lap-line crossing, in
 * seconds (OQ-001 resolved, TP-13): the arcade "the line extends your
 * clock" rule. The extension lands AFTER the step's decay, so a crossing
 * in the same tick the budget would die saves the run.
 */
export const NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S = 25;

/**
 * Total forward laps of a human-mode race (OQ-002 resolved): the race
 * ends complete exactly at this lap count, never before; AI-mode session
 * floors are phase-2 rules and never read this constant.
 */
export const NEATRACING_HUMAN_RACE_LAPS = 20;

/**
 * Level of the FIRST session (OQ-002 resolved): human sessions start at
 * lvl 1 (one solo car per the level ladder, spec section 4.1).
 */
export const NEATRACING_HUMAN_START_LEVEL = 1;

/**
 * Level cap (OQ-002 resolved): the lvl-3 ceiling of the level ladder —
 * a completed race at the cap keeps the next race at lvl 3.
 */
export const NEATRACING_HUMAN_MAX_LEVEL = 3;

/** Running status label (private; the type is the public surface). */
const RACE_STATUS_RUNNING: RaceStatus = 'running';

/** Completed status label, set exactly at the last forward lap crossing. */
const RACE_STATUS_RACE_COMPLETE: RaceStatus = 'raceComplete';

/** Out-of-time status label, set when the checkpoint budget reaches zero. */
const RACE_STATUS_OUT_OF_TIME: RaceStatus = 'outOfTime';

/**
 * Exact checkpoint-budget floor in seconds: a step that would overdraw
 * the budget clamps the balance at zero instead of going negative.
 */
const CHECKPOINT_BUDGET_FLOOR_S = 0;

/**
 * Creates the fresh race-rules state for one session.
 *
 * The state is the deterministic seed of the pure stepper: a running
 * race with no laps, no raced time, the full initial checkpoint budget,
 * and no carried next-race level yet.
 *
 * @param level - Session level of the race. Defaults to
 * NEATRACING_HUMAN_START_LEVEL; follow-up sessions pass the level
 * carried by the previous race's `nextRaceLevel`.
 * @returns The fresh running race state.
 *
 * @example
 * ```ts
 * const fresh = createRaceRulesState();
 * // { level: 1, lapsCompleted: 0, checkpointBudgetS: 60,
 * //   elapsedS: 0, raceStatus: 'running', nextRaceLevel: null }
 * ```
 */
export function createRaceRulesState(
  level: number = NEATRACING_HUMAN_START_LEVEL,
): RaceRulesState {
  return {
    level,
    lapsCompleted: 0,
    checkpointBudgetS: NEATRACING_CHECKPOINT_INITIAL_BUDGET_S,
    elapsedS: 0,
    raceStatus: RACE_STATUS_RUNNING,
    nextRaceLevel: null,
  };
}

/**
 * Advances the race rules by one step and returns the NEXT state.
 *
 * Step order is contracted: while the race runs, the checkpoint budget
 * first decays and the elapsed clock advances by `dtS`; then a forward
 * lap-line crossing — and only one — counts the lap and grants the
 * extension; the race is marked 'raceComplete' exactly when the lap
 * count reaches NEATRACING_HUMAN_RACE_LAPS (resolved at that crossing
 * tick, never before); the budget is clamped at zero; and a still
 * running race whose budget has reached zero ends 'outOfTime' unless
 * THIS step granted the crossing (the line saves the second). Steps on
 * a terminal status return the input record unchanged.
 *
 * @param state - Current race-rules state; never mutated.
 * @param input - The step inputs (`dtS`, forward crossing flag).
 * @returns The next state: a fresh record while the race runs, otherwise
 * the unchanged input record.
 *
 * @example
 * ```ts
 * const next = stepRaceRules(fresh, {
 *   dtS: 1,
 *   forwardLapLineCrossing: false,
 * });
 * // budget 60 - 1 = 59 s, still running
 * ```
 */
export function stepRaceRules(
  state: RaceRulesState,
  input: RaceRulesStepInput,
): RaceRulesState {
  // Step 1: terminal states are inert — no decay, no laps, no flips.
  if (state.raceStatus !== RACE_STATUS_RUNNING) {
    return state;
  }

  // Step 2: decay the clock — budget AND elapsed advance by `dtS`.
  const decayed = decayRaceClock(state, input.dtS);

  // Step 3: apply the one allowed forward crossing (lap + extension).
  const raced = applyForwardLapCrossing(decayed, input.forwardLapLineCrossing);

  // Step 4: clamp the budget at zero, then resolve the out-of-time flip.
  return resolveBudgetExpiry(raced, input.forwardLapLineCrossing);
}

/**
 * Advances the race clock: spends `dtS` checkpoint seconds and races
 * `dtS` more seconds, returning a fresh record.
 *
 * No clamping happens here — the same step's crossing may still extend
 * the budget, and the dying-tick contract requires the raw (pre-clamp)
 * balance to feed the extension.
 *
 * @param state - Current running race state (never mutated).
 * @param dtS - Elapsed seconds of this step.
 * @returns Fresh record with the decayed budget and the advanced elapsed clock.
 */
function decayRaceClock(state: RaceRulesState, dtS: number): RaceRulesState {
  return {
    ...state,
    checkpointBudgetS: state.checkpointBudgetS - dtS,
    elapsedS: state.elapsedS + dtS,
  };
}

/**
 * Applies the one allowed forward lap-line crossing of this step: counts
 * the lap, grants the extension, and — exactly when the lap count
 * reaches NEATRACING_HUMAN_RACE_LAPS — marks the race complete and
 * records the NEXT race's carried level. No-op when the flag is false.
 *
 * @param state - Clock-decayed race state (never mutated).
 * @param forwardLapLineCrossing - True only for a forward line crossing.
 * @returns Fresh record with the crossing applied (and possibly the
 * completion boundary resolved), or the input record when no crossing.
 */
function applyForwardLapCrossing(
  state: RaceRulesState,
  forwardLapLineCrossing: boolean,
): RaceRulesState {
  if (!forwardLapLineCrossing) {
    return state;
  }

  const crossed: RaceRulesState = {
    ...state,
    lapsCompleted: state.lapsCompleted + 1,
    checkpointBudgetS:
      state.checkpointBudgetS + NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S,
  };

  if (crossed.lapsCompleted >= NEATRACING_HUMAN_RACE_LAPS) {
    return markRaceComplete(crossed);
  }
  return crossed;
}

/**
 * Marks the race complete exactly at the lap-count boundary and records
 * the NEXT race's carried level: min(level + 1, NEATRACING_HUMAN_MAX_LEVEL)
 * (level-capped session-boundary advancement; the completed race record
 * itself is never rewritten after this tick because terminal steps are
 * inert).
 *
 * @param crossed - Fresh record whose lap count just reached the limit.
 * @returns Fresh terminal record with 'raceComplete' and the carried level.
 */
function markRaceComplete(crossed: RaceRulesState): RaceRulesState {
  return {
    ...crossed,
    raceStatus: RACE_STATUS_RACE_COMPLETE,
    nextRaceLevel: Math.min(crossed.level + 1, NEATRACING_HUMAN_MAX_LEVEL),
  };
}

/**
 * Clamps the checkpoint budget at zero and resolves the out-of-time
 * flip: a still running race whose budget has reached zero ends out of
 * time — unless this same step granted a crossing (the line saves the
 * second), and never overriding a race completion marked this tick.
 *
 * @param state - Race state after decay and the optional crossing.
 * @param grantedExtension - True when this step granted the crossing
 * extension (the same-tick escape hatch from the out-of-time flip).
 * @returns Fresh record with the clamped budget and the resolved status.
 */
function resolveBudgetExpiry(
  state: RaceRulesState,
  grantedExtension: boolean,
): RaceRulesState {
  const clamped: RaceRulesState = {
    ...state,
    checkpointBudgetS: Math.max(
      CHECKPOINT_BUDGET_FLOOR_S,
      state.checkpointBudgetS,
    ),
  };

  const budgetExpired =
    clamped.raceStatus === RACE_STATUS_RUNNING &&
    !grantedExtension &&
    clamped.checkpointBudgetS === CHECKPOINT_BUDGET_FLOOR_S;

  if (!budgetExpired) {
    return clamped;
  }
  return {
    ...clamped,
    raceStatus: RACE_STATUS_OUT_OF_TIME,
  };
}
