/**
 * NeatRacing adaptive quality controller (slice P3-S1-A; spec sections 6.3,
 * 13.3 AC-035, G8, TP-15 in `plans/neatracing-voxel-racer.spec.md`).
 *
 * Quality-first degradation: when the frame budget is exceeded the
 * controller drops QUALITY in a fixed ladder — the G8 cheap-bloom
 * pipeline's explicit off-switch first (the `enabled === false` frame state
 * already consumed by `passes/bloom.pass.ts`), then the cheap-tier LOD
 * bias, then full-vs-sparse prop density — BEFORE any frame is dropped.
 * Dropping frames stays the host's last resort; this module never drops
 * frames.
 *
 * The decision function is:
 *
 * - DETERMINISTIC (Level 2 ordered deterministic, reproducibility-contracts
 *   language): the same frameMs sequence drives the same state trajectory
 *   on any runtime — EMA blending plus threshold comparisons only; no
 *   randomness, no wall-clock reads, no cross-tick float reordering.
 * - HYSTERESIS-BOUNDED: drops fire immediately below
 *   {@link NEATRACING_ADAPTIVE_FPS_DROP_FLOOR}; recovery requires smoothed
 *   fps at or above {@link NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING} for a
 *   named {@link NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES} consecutive
 *   frame run, and the mid-band between floor and ceiling holds quality
 *   while burning recovery credit, so boundary noise cannot thrash the
 *   ladder (R-03's boundary-hysteresis philosophy applied to quality).
 * - ALLOCATION-FREE per frame (P9): the state object is created once and
 *   mutated in place — the same in-place executor pattern as
 *   `host/render-loop.utils.ts` (`advanceFrameSteps`).
 *
 * This module owns ORDER and hysteresis only; it renders nothing. The host
 * reads the resolved stage record from
 * {@link NEATRACING_ADAPTIVE_QUALITY_STAGES}`[state.stage]` and applies each
 * knob at composition time (bloom pass `frame.enabled`, LOD tier selection,
 * sparse prop subset). Threshold values are Phase-3 tunables per spec
 * TP-15: retuning is a named-constant change only.
 *
 * @module
 */

/** Milliseconds per second; single-source for the frameMs-to-fps fold. */
const NEATRACING_MS_PER_SECOND = 1000;

/**
 * Smoothed-fps estimate seed and tuning reference (spec 6.3, TP-15).
 *
 * The controller assumes the 60 fps target until smoothed telemetry says
 * otherwise, so a healthy startup never drops quality before real evidence.
 */
export const NEATRACING_ADAPTIVE_FPS_TARGET = 60;

/**
 * Drop floor in smoothed fps (spec TP-15): the first quality knob drops as
 * soon as smoothed fps falls below this floor. Chosen ~9% below the 60 fps
 * target so quality drops happen before visible stutter becomes the norm.
 */
export const NEATRACING_ADAPTIVE_FPS_DROP_FLOOR = 55;

/**
 * Recovery ceiling in smoothed fps (spec TP-15): at or above this value a
 * frame counts toward the sustained recovery streak. Kept strictly below
 * the drop floor's distance from the target so the hold band
 * `[DROP_FLOOR, RECOVER_CEILING)` is non-empty — that band IS the
 * hysteresis gap (no drop, no recover, no boundary thrash).
 */
export const NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING = 59;

/**
 * Consecutive at-or-above-ceiling frames required to recover exactly ONE
 * quality stage (spec TP-15). 120 frames is 2 sustained seconds at 60 Hz:
 * recovery is always deliberately slower than a drop, and each recovered
 * stage restarts the streak, bounding how fast quality unwinds.
 */
export const NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES = 120;

/**
 * Exponential-moving-average weight given to each new fps sample
 * (`smoothed += alpha * (sample - smoothed)`). With 0.1 a sustained breach
 * converges past the drop floor within a handful of frames (~60 ms) while
 * isolated hitches are absorbed, keeping decisions stable.
 */
export const NEATRACING_ADAPTIVE_FPS_EMA_ALPHA = 0.1;

/**
 * Upper clamp on the smoothed-fps estimate (finite-boundary guard, spec 6.3
 * AC-003-adjacent). An unclamped geometric EMA could asymptote toward a
 * non-finite value if a corrupted sub-millisecond delta ever reached the
 * fold; clamping keeps every stored estimate finite and the decision
 * comparisons bounded without affecting normal 0-1000 fps telemetry.
 */
export const NEATRACING_ADAPTIVE_SMOOTHED_FPS_CEILING = 1000;

/**
 * One quality stage in the fixed degradation ladder (AC-035).
 *
 * Stage 0 is full quality; every later entry is the previous entry minus
 * exactly one knob, so the ladder encodes the drop ORDER and never skips a
 * knob or re-enables a dropped one while quality is still dropping.
 */
export interface AdaptiveQualityStage {
  /** Stable label for logs and HUD debugging. */
  readonly label: string;
  /**
   * G8 explicit bloom off-switch: false disables the whole cheap-bloom
   * pipeline for the frame (bloom.pass issues ZERO GL operations for
   * `enabled === false`). Bloom is FIRST on the drop list, before any LOD
   * swap.
   */
  readonly bloomEnabled: boolean;
  /**
   * true once the cheap (64³) LOD tier is preferred for distance swaps.
   * SECOND on the drop list, after bloom.
   */
  readonly lodCheapTier: boolean;
  /**
   * false once the renderer draws only the sparse prop subset (LD-25
   * optional voxel scenery props). THIRD and last on the drop list.
   */
  readonly propsFullDensity: boolean;
}

/**
 * Mutable per-frame controller state (render-loop executor pattern: created
 * once, mutated in place, zero per-frame allocation — P9).
 */
export interface AdaptiveQualityState {
  /** Current stage index into {@link NEATRACING_ADAPTIVE_QUALITY_STAGES}. */
  stage: number;
  /** Consecutive at-or-above-ceiling frames accumulated toward recovery. */
  goodStreak: number;
  /** EMA-smoothed fps estimate driving every decision comparison. */
  smoothedFps: number;
}

/**
 * The fixed quality-first degradation ladder in drop order
 * (AC-035): stage 0 full quality -> 1 bloom off (G8 explicit off-switch)
 * -> 2 cheap LOD tier -> 3 sparse props. Frozen in place, read-only at
 * runtime.
 */
export const NEATRACING_ADAPTIVE_QUALITY_STAGES: readonly AdaptiveQualityStage[] =
  Object.freeze(
    (
      [
        {
          label: 'full',
          bloomEnabled: true,
          lodCheapTier: false,
          propsFullDensity: true,
        },
        {
          label: 'bloom-off',
          bloomEnabled: false,
          lodCheapTier: false,
          propsFullDensity: true,
        },
        {
          label: 'lod-cheap',
          bloomEnabled: false,
          lodCheapTier: true,
          propsFullDensity: true,
        },
        {
          label: 'props-sparse',
          bloomEnabled: false,
          lodCheapTier: true,
          propsFullDensity: false,
        },
      ] as const satisfies readonly AdaptiveQualityStage[]
    ).map((stage) => Object.freeze(stage)),
  );

/**
 * Highest valid stage index (`stage ladder length - 1`), derived from the
 * ladder so the clamp target and the table can never disagree.
 */
export const NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX: number =
  NEATRACING_ADAPTIVE_QUALITY_STAGES.length - 1;

/**
 * Creates the initial controller state: full quality (stage 0), empty
 * recovery streak, and the assume-target smoothing seed so startup frames
 * never drop quality before real telemetry exists.
 *
 * Call once at render-loop boot; every later frame reuses this state object
 * in place.
 *
 * @returns The initial full-quality controller state.
 *
 * @example
 * const state = createInitialAdaptiveQualityState();
 * console.log(state.stage); // 0 (full quality)
 */
export function createInitialAdaptiveQualityState(): AdaptiveQualityState {
  return {
    stage: 0,
    goodStreak: 0,
    smoothedFps: NEATRACING_ADAPTIVE_FPS_TARGET,
  };
}

/**
 * Advances the controller for one frame and returns the resolved stage
 * index after the decision.
 *
 * The orchestrator runs exactly two steps: the smoothing fold of
 * `frameMs` into the state's EMA estimate, then the hysteresis-bounded
 * stage decision over that estimate. The state is mutated in place and
 * the per-frame path performs no heap allocations (P9, executor pattern
 * of `host/render-loop.utils.ts`).
 *
 * Determinism (Level 2, ordered deterministic): identical frameMs
 * sequences produce identical stage/streak/smoothed trajectories on any
 * runtime; the fold ignores non-finite and non-positive deltas instead of
 * emitting machine-dependent values.
 *
 * @param state - Controller state mutated in place.
 * @param frameMs - Latest frame delta in milliseconds (clamped upstream by
 *   the render loop's `NEATRACING_MAX_FRAME_DELTA_MS`).
 * @returns The resolved stage index after this frame's decision (`0..3`).
 *
 * @example
 * const state = createInitialAdaptiveQualityState();
 * const stage = tickAdaptiveQuality(state, 20); // 50 fps sample
 * console.log(NEATRACING_ADAPTIVE_QUALITY_STAGES[stage].label);
 */
export function tickAdaptiveQuality(
  state: AdaptiveQualityState,
  frameMs: number,
): number {
  // Step 1: Fold the frame delta into the smoothed fps estimate.
  updateSmoothedFps(state, frameMs);

  // Step 2: Apply the hysteresis-bounded ladder decision.
  return decideAdaptiveQualityTransition(state);
}

/**
 * Folds one frame delta into the state's EMA fps estimate (helper below the
 * fold). Usable deltas produce a bounded update; unusable deltas (non-finite
 * or non-positive — first-frame zeros, corrupted timers) are ignored so the
 * estimate can never become NaN/Infinity and the decision input stays
 * finite and deterministic.
 *
 * @param state - Controller state mutated in place.
 * @param frameMs - Latest frame delta in milliseconds.
 */
function updateSmoothedFps(
  state: AdaptiveQualityState,
  frameMs: number,
): void {
  if (!Number.isFinite(frameMs) || frameMs <= 0) {
    return;
  }
  const fpsSample = NEATRACING_MS_PER_SECOND / frameMs;
  const blended =
    state.smoothedFps +
    NEATRACING_ADAPTIVE_FPS_EMA_ALPHA * (fpsSample - state.smoothedFps);
  state.smoothedFps = Math.min(
    blended,
    NEATRACING_ADAPTIVE_SMOOTHED_FPS_CEILING,
  );
}

/**
 * Applies the hysteresis-bounded ladder decision to the state (helper below
 * the fold) and returns the resolved stage index.
 *
 * - smoothed fps < drop floor: drop exactly ONE stage immediately (never a
 *   multi-stage jump, even under a severe breach) and reset recovery credit.
 * - smoothed fps >= recover ceiling: accumulate one good-frame credit; at
 *   the named streak, recover exactly ONE stage and restart the streak.
 * - mid-band: hold quality and burn credit (hysteresis gap).
 * - non-finite estimate (defensive): hold quality and burn credit.
 *
 * @param state - Controller state mutated in place.
 * @returns The resolved stage index after the decision.
 */
function decideAdaptiveQualityTransition(
  state: AdaptiveQualityState,
): number {
  if (!Number.isFinite(state.smoothedFps)) {
    state.goodStreak = 0;
    return state.stage;
  }
  if (state.smoothedFps < NEATRACING_ADAPTIVE_FPS_DROP_FLOOR) {
    if (state.stage < NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX) {
      state.stage += 1;
    }
    state.goodStreak = 0;
    return state.stage;
  }
  if (state.smoothedFps >= NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING) {
    state.goodStreak = Math.min(
      state.goodStreak + 1,
      NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES,
    );
    if (
      state.goodStreak >= NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES &&
      state.stage > 0
    ) {
      state.stage -= 1;
      state.goodStreak = 0;
    }
    return state.stage;
  }
  // Mid-band hold (hysteresis gap): no drop, no recovery credit.
  state.goodStreak = 0;
  return state.stage;
}
