/**
 * Green-phase contract tests for the NeatRacing adaptive quality controller
 * (slice P3-S1-A; spec sections 6.3, 13.3 AC-035, G8, TP-15 in
 * `plans/neatracing-voxel-racer.spec.md`).
 *
 * Locks the AC-035 contract: the quality-first drop ORDER (bloom off ->
 * LOD tier -> prop density BEFORE frames), the explicit G8 bloom off-switch
 * as a named frozen boolean knob, and the hysteresis-bounded deterministic
 * decision function — immediate one-stage drops below the floor, sustained
 * reverse-order recovery above the ceiling, mid-band hold, and identical
 * state trajectories for identical frameMs sequences.
 *
 * The module is pure TypeScript (no DOM, no WebGL), so every contract is
 * exercised through the in-place state executors, mirroring the host
 * render-loop executor test pattern from `host/render-loop.utils.test.ts`.
 * Decision-branch tests inject `smoothedFps` directly and tick with NaN
 * frameMs (which the fold ignores by contract) to isolate the decision from
 * the smoothing; smoothing tests drive real frameMs values instead.
 */
import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_ADAPTIVE_FPS_DROP_FLOOR,
  NEATRACING_ADAPTIVE_FPS_EMA_ALPHA,
  NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING,
  NEATRACING_ADAPTIVE_FPS_TARGET,
  NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX,
  NEATRACING_ADAPTIVE_QUALITY_STAGES,
  NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES,
  NEATRACING_ADAPTIVE_SMOOTHED_FPS_CEILING,
  type AdaptiveQualityState,
  createInitialAdaptiveQualityState,
  tickAdaptiveQuality,
} from './adaptive-quality';

/** Builds a state pinned to a stage/telemetry point for decision testing. */
function createControllerAt(
  stage: number,
  smoothedFps: number,
  goodStreak: number = 0,
): AdaptiveQualityState {
  const state = createInitialAdaptiveQualityState();
  state.stage = stage;
  state.smoothedFps = smoothedFps;
  state.goodStreak = goodStreak;
  return state;
}

/** Runs N decision ticks with NaN frameMs (the fold ignores NaN). */
function runDecisionTicks(state: AdaptiveQualityState, tickCount: number): void {
  for (let tickIndex = 0; tickIndex < tickCount; tickIndex++) {
    tickAdaptiveQuality(state, Number.NaN);
  }
}

describe('NEATRACING_ADAPTIVE thresholds (spec TP-15)', () => {
  it('separates the drop floor and recover ceiling inside the fps band', () => {
    expect(NEATRACING_ADAPTIVE_FPS_DROP_FLOOR).toBeLessThan(
      NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING,
    );
    expect(NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING).toBeLessThanOrEqual(
      NEATRACING_ADAPTIVE_FPS_TARGET,
    );
    expect(NEATRACING_ADAPTIVE_FPS_DROP_FLOOR).toBeLessThan(
      NEATRACING_ADAPTIVE_FPS_TARGET,
    );
  });

  it('requires a positive recovery streak and a fractional EMA alpha', () => {
    expect(NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES).toBeGreaterThan(0);
    expect(NEATRACING_ADAPTIVE_FPS_EMA_ALPHA).toBeGreaterThan(0);
    expect(NEATRACING_ADAPTIVE_FPS_EMA_ALPHA).toBeLessThan(1);
  });
});

describe('NEATRACING_ADAPTIVE_QUALITY_STAGES ladder (AC-035)', () => {
  it('exposes exactly four stages with a derived max index', () => {
    expect(NEATRACING_ADAPTIVE_QUALITY_STAGES).toHaveLength(4);
    expect(NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX).toBe(3);
  });

  it('names the ladder stages in drop order', () => {
    const labels = NEATRACING_ADAPTIVE_QUALITY_STAGES.map((stage) => stage.label);
    expect(labels).toEqual(['full', 'bloom-off', 'lod-cheap', 'props-sparse']);
  });

  it('keeps stage 0 at full quality', () => {
    const [fullStage] = NEATRACING_ADAPTIVE_QUALITY_STAGES;
    expect(fullStage.bloomEnabled).toBe(true);
    expect(fullStage.lodCheapTier).toBe(false);
    expect(fullStage.propsFullDensity).toBe(true);
  });

  it('turns the G8 bloom switch off from the first drop onward', () => {
    expect(NEATRACING_ADAPTIVE_QUALITY_STAGES[1].bloomEnabled).toBe(false);
    expect(NEATRACING_ADAPTIVE_QUALITY_STAGES[2].bloomEnabled).toBe(false);
    expect(NEATRACING_ADAPTIVE_QUALITY_STAGES[3].bloomEnabled).toBe(false);
  });

  it('drops the LOD tier second and prop density third', () => {
    const [, bloomStage, lodStage, propsStage] =
      NEATRACING_ADAPTIVE_QUALITY_STAGES;
    expect(bloomStage.lodCheapTier).toBe(false);
    expect(lodStage).toEqual({
      label: 'lod-cheap',
      bloomEnabled: false,
      lodCheapTier: true,
      propsFullDensity: true,
    });
    expect(propsStage.propsFullDensity).toBe(false);
  });

  it('freezes the stage table and every stage record against mutation', () => {
    expect(Object.isFrozen(NEATRACING_ADAPTIVE_QUALITY_STAGES)).toBe(true);
    expect(Object.isFrozen(NEATRACING_ADAPTIVE_QUALITY_STAGES[0])).toBe(true);
    expect(Object.isFrozen(NEATRACING_ADAPTIVE_QUALITY_STAGES[3])).toBe(true);
  });
});

describe('createInitialAdaptiveQualityState', () => {
  it('starts full quality with the assume-target smoothing seed', () => {
    const state = createInitialAdaptiveQualityState();
    expect(state.stage).toBe(0);
    expect(state.goodStreak).toBe(0);
    expect(state.smoothedFps).toBe(NEATRACING_ADAPTIVE_FPS_TARGET);
  });
});

describe('tickAdaptiveQuality smoothing fold', () => {
  it('blends one finite frame delta with the named EMA alpha', () => {
    const state = createInitialAdaptiveQualityState();
    expect(tickAdaptiveQuality(state, 20)).toBe(0);
    expect(state.smoothedFps).toBeCloseTo(59, 12);
  });

  it('ignores non-finite frame deltas without touching the estimate', () => {
    const state = createInitialAdaptiveQualityState();
    tickAdaptiveQuality(state, Number.NaN);
    expect(state.smoothedFps).toBe(NEATRACING_ADAPTIVE_FPS_TARGET);
    tickAdaptiveQuality(state, Number.POSITIVE_INFINITY);
    expect(state.smoothedFps).toBe(NEATRACING_ADAPTIVE_FPS_TARGET);
  });

  it('ignores non-positive frame deltas without touching the estimate', () => {
    const state = createInitialAdaptiveQualityState();
    tickAdaptiveQuality(state, 0);
    expect(state.smoothedFps).toBe(NEATRACING_ADAPTIVE_FPS_TARGET);
    tickAdaptiveQuality(state, -16.67);
    expect(state.smoothedFps).toBe(NEATRACING_ADAPTIVE_FPS_TARGET);
  });

  it('clamps absurd sub-millisecond deltas at the finite ceiling', () => {
    const state = createInitialAdaptiveQualityState();
    tickAdaptiveQuality(state, 0.0005);
    expect(state.smoothedFps).toBe(NEATRACING_ADAPTIVE_SMOOTHED_FPS_CEILING);
    expect(Number.isFinite(state.smoothedFps)).toBe(true);
  });
});

describe('tickAdaptiveQuality drop decisions (AC-035)', () => {
  it('drops bloom first, exactly one stage per breached decision', () => {
    const state = createControllerAt(0, 40);
    expect(tickAdaptiveQuality(state, Number.NaN)).toBe(1);
  });

  it('never skips a stage even under a severe breach', () => {
    const state = createControllerAt(0, 1);
    expect(tickAdaptiveQuality(state, Number.NaN)).toBe(1);
  });

  it('walks the full ladder in order and clamps at the bottom', () => {
    const state = createControllerAt(0, 40);
    const seenStages = [
      tickAdaptiveQuality(state, Number.NaN),
      tickAdaptiveQuality(state, Number.NaN),
      tickAdaptiveQuality(state, Number.NaN),
      tickAdaptiveQuality(state, Number.NaN),
    ];
    expect(seenStages).toEqual([1, 2, 3, 3]);
  });

  it('holds the mid-band band without touching the quality stage', () => {
    const state = createControllerAt(2, 57, 9);
    expect(tickAdaptiveQuality(state, Number.NaN)).toBe(2);
    expect(state.goodStreak).toBe(0);
  });

  it('resets recovery credit when a breach drops a stage', () => {
    const state = createControllerAt(1, 40, 100);
    expect(tickAdaptiveQuality(state, Number.NaN)).toBe(2);
    expect(state.goodStreak).toBe(0);
  });

  it('holds quality and burns credit on a non-finite estimate', () => {
    const state = createControllerAt(1, Number.NaN, 7);
    expect(tickAdaptiveQuality(state, Number.NaN)).toBe(1);
    expect(state.goodStreak).toBe(0);
  });
});

describe('tickAdaptiveQuality hysteresis recovery (AC-035)', () => {
  it('recovers exactly one stage only at the full named streak', () => {
    const state = createControllerAt(3, 61);
    runDecisionTicks(state, NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES - 1);
    expect(state.stage).toBe(3);
    runDecisionTicks(state, 1);
    expect(state.stage).toBe(2);
    expect(state.goodStreak).toBe(0);
  });

  it('recovers in reverse order: props, then LOD, then bloom last', () => {
    const state = createControllerAt(3, 61);
    runDecisionTicks(
      state,
      NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES * 2 +
        NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES,
    );
    expect(state.stage).toBe(0);
    expect(NEATRACING_ADAPTIVE_QUALITY_STAGES[0].bloomEnabled).toBe(true);
    expect(NEATRACING_ADAPTIVE_QUALITY_STAGES[0].propsFullDensity).toBe(true);
  });

  it('caps the credit counter at the streak length while at stage 0', () => {
    const state = createControllerAt(0, 61);
    runDecisionTicks(state, NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES * 5);
    expect(state.stage).toBe(0);
    expect(state.goodStreak).toBe(NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES);
  });
});

describe('tickAdaptiveQuality in-place executor contract (P9)', () => {
  it('reuses the state object across the whole steady-state loop', () => {
    const state = createControllerAt(1, 61);
    const before = state;
    runDecisionTicks(state, 1000);
    expect(state).toBe(before);
    expect(state.stage).toBe(0);
    expect(state.goodStreak).toBe(NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES);
  });
});

describe('deterministic trajectories (Level 2, ordered deterministic)', () => {
  it('replays identical state trajectories for identical frameMs streams', () => {
    const frameMsScript = [
      ...Array.from({ length: 60 }, () => 16.67),
      ...Array.from({ length: 6 }, () => 25),
      ...Array.from({ length: 24 }, () => 16),
      ...Array.from({ length: 2 }, () => 40),
      ...Array.from({ length: 380 }, () => 16.2),
      ...Array.from({ length: 5 }, () => 17.8),
    ];

    const firstTrajectory: Array<[number, number, number]> = [];
    const secondTrajectory: Array<[number, number, number]> = [];

    const firstState = createInitialAdaptiveQualityState();
    const secondState = createInitialAdaptiveQualityState();
    for (let frameIndex = 0; frameIndex < frameMsScript.length; frameIndex++) {
      firstTrajectory.push([
        tickAdaptiveQuality(firstState, frameMsScript[frameIndex]),
        firstState.goodStreak,
        firstState.smoothedFps,
      ]);
      secondTrajectory.push([
        tickAdaptiveQuality(secondState, frameMsScript[frameIndex]),
        secondState.goodStreak,
        secondState.smoothedFps,
      ]);
    }

    expect(firstTrajectory).toEqual(secondTrajectory);
    expect(firstState.stage).toBe(0);
  });
});
