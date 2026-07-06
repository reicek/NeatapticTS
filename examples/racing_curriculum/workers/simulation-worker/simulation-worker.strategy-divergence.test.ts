/**
 * Red-phase contracts for the strategy-divergence analytics module (Phase 7 Step 03).
 *
 * Contracts verified here:
 * - createStrategyDivergenceTracker is exported as a function
 * - Tracker records and returns per-generation team fitness via getTrajectory
 * - Classifier detects alternating advantage from team fitness time series
 * - Classifier computes a finite divergenceScore from team fitness time series
 * - Tracker records pit-lap distribution per team in snapshot trajectory
 *
 * The strategy-divergence module does not exist yet — Step 04 creates:
 *   - simulation-worker.strategy-divergence.service.ts
 *   - simulation-worker.strategy-divergence.types.ts
 *
 * Local type declarations are used because the source types do not exist yet.
 * These use entirely new names that do not conflict with existing types.
 *
 * Single-expect rule is enforced throughout.
 */

// ---------------------------------------------------------------------------
// Local type declarations — strategy-divergence module interface (NEW types)
// ---------------------------------------------------------------------------

/** Per-generation strategy-divergence snapshot. */
interface StrategyDivergenceSnapshot {
  readonly generation: number;
  readonly teamAFitness: number;
  readonly teamBFitness: number;
  readonly teamAPitLapDistribution: readonly number[];
  readonly teamBPitLapDistribution: readonly number[];
  readonly reproductionModeMix: Readonly<Record<string, number>>;
}

/** Classifier output from analysing a team fitness time series. */
interface StrategyDivergenceClassifierResult {
  readonly isAlternating: boolean;
  readonly dominantPeriod: number;
  readonly advantageAmplitude: number;
  readonly divergenceScore: number;
}

/** Tracker that accumulates per-generation snapshots and classifies trajectories. */
interface StrategyDivergenceTracker {
  recordSnapshot(snapshot: StrategyDivergenceSnapshot): void;
  classify(): StrategyDivergenceClassifierResult;
  getTrajectory(): readonly StrategyDivergenceSnapshot[];
}

/** Module surface for the strategy-divergence service. */
interface StrategyDivergenceService {
  createStrategyDivergenceTracker(config: {
    readonly teamSize: number;
    readonly minGenerations: number;
  }): StrategyDivergenceTracker;
}

// ---------------------------------------------------------------------------
// Dynamic module loader (module does not exist yet — Step 04)
// ---------------------------------------------------------------------------

async function loadStrategyDivergenceService(): Promise<StrategyDivergenceService> {
  const modulePath = './simulation-worker.strategy-divergence.service';
  const module = (await import(
    modulePath
  )) as Partial<StrategyDivergenceService>;

  if (typeof module.createStrategyDivergenceTracker !== 'function') {
    throw new Error(
      'Missing strategy-divergence service export: createStrategyDivergenceTracker',
    );
  }

  return module as StrategyDivergenceService;
}

// ---------------------------------------------------------------------------
// Deterministic test fixtures
// ---------------------------------------------------------------------------

/** 3-car team pit-lap distribution: each car pits on a different lap. */
const PIT_LAP_DISTRIBUTION_3: readonly number[] = [2, 3, 4];

/** Empty reproduction mode mix for the analytics-only fallback. */
const EMPTY_REPRODUCTION_MIX: Readonly<Record<string, number>> = {};

// ---------------------------------------------------------------------------
// Red tests — Strategy-divergence analytics module
// ---------------------------------------------------------------------------

describe('Strategy-divergence analytics module', () => {
  it('exports createStrategyDivergenceTracker as a function', async () => {
    // Arrange — attempt to load the strategy-divergence service module
    let service: StrategyDivergenceService | undefined;
    try {
      service = await loadStrategyDivergenceService();
    } catch {
      service = undefined;
    }

    // Assert — the factory function should exist
    // (currently module does not exist at all)
    expect(typeof service?.createStrategyDivergenceTracker).toBe('function');
  });

  it('records and returns per-generation team fitness scores via getTrajectory', async () => {
    // Arrange — load tracker and record a snapshot
    let tracker: StrategyDivergenceTracker | undefined;
    try {
      const service = await loadStrategyDivergenceService();
      tracker = service.createStrategyDivergenceTracker({
        teamSize: 3,
        minGenerations: 1,
      });
    } catch {
      tracker = undefined;
    }

    // Act — record one generation snapshot
    tracker?.recordSnapshot({
      generation: 0,
      teamAFitness: 10,
      teamBFitness: 8,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });

    // Assert — trajectory should contain the recorded snapshot
    // (currently tracker is undefined, so trajectory is undefined)
    expect((tracker?.getTrajectory().length ?? 0) >= 1).toBe(true);
  });

  it('detects alternating advantage from team fitness time series', async () => {
    // Arrange — load tracker and feed an alternating A>B, B>A, A>B pattern
    let tracker: StrategyDivergenceTracker | undefined;
    try {
      const service = await loadStrategyDivergenceService();
      tracker = service.createStrategyDivergenceTracker({
        teamSize: 3,
        minGenerations: 3,
      });
    } catch {
      tracker = undefined;
    }

    // Act — record 3 generations with alternating advantage
    tracker?.recordSnapshot({
      generation: 0,
      teamAFitness: 12,
      teamBFitness: 6,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });
    tracker?.recordSnapshot({
      generation: 1,
      teamAFitness: 5,
      teamBFitness: 11,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });
    tracker?.recordSnapshot({
      generation: 2,
      teamAFitness: 13,
      teamBFitness: 4,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });

    // Assert — classifier should detect alternating advantage
    // (currently tracker is undefined, so classifier result is undefined)
    const result = tracker?.classify();
    expect(result?.isAlternating).toBe(true);
  });

  it('computes a finite divergenceScore from team fitness time series', async () => {
    // Arrange — load tracker and feed a diverging fitness pattern
    let tracker: StrategyDivergenceTracker | undefined;
    try {
      const service = await loadStrategyDivergenceService();
      tracker = service.createStrategyDivergenceTracker({
        teamSize: 3,
        minGenerations: 2,
      });
    } catch {
      tracker = undefined;
    }

    // Act — record 2 generations with increasing divergence
    tracker?.recordSnapshot({
      generation: 0,
      teamAFitness: 10,
      teamBFitness: 9,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });
    tracker?.recordSnapshot({
      generation: 1,
      teamAFitness: 15,
      teamBFitness: 3,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });

    // Assert — divergenceScore should be a finite number
    // (currently tracker is undefined, so divergenceScore is NaN)
    const result = tracker?.classify();
    expect(Number.isFinite(result?.divergenceScore ?? NaN)).toBe(true);
  });

  it('records pit-lap distribution per team in snapshot trajectory', async () => {
    // Arrange — load tracker and record a snapshot with pit-lap data
    let tracker: StrategyDivergenceTracker | undefined;
    try {
      const service = await loadStrategyDivergenceService();
      tracker = service.createStrategyDivergenceTracker({
        teamSize: 3,
        minGenerations: 1,
      });
    } catch {
      tracker = undefined;
    }

    // Act — record one generation with 3-car pit-lap distributions
    tracker?.recordSnapshot({
      generation: 0,
      teamAFitness: 10,
      teamBFitness: 8,
      teamAPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      teamBPitLapDistribution: PIT_LAP_DISTRIBUTION_3,
      reproductionModeMix: EMPTY_REPRODUCTION_MIX,
    });

    // Assert — first trajectory snapshot should have 3-entry pit-lap distribution
    // (currently tracker is undefined, so trajectory is empty)
    const trajectory = tracker?.getTrajectory() ?? [];
    const firstSnapshot = trajectory[0];
    expect(firstSnapshot?.teamAPitLapDistribution.length).toBe(3);
  });
});
