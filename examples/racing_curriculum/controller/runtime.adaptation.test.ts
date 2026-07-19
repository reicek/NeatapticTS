/**
 * Sibling smoke tests for `controller/runtime.adaptation.ts`.
 *
 * The runtime adaptation engine is a demo-local within-episode mutation tool.
 * These tests verify the exported function surface is loadable and correctly
 * typed.  Detailed behaviour tests can be added in a dedicated coverage pass.
 *
 * Single-expect rule enforced throughout.
 */
import fs from 'fs';
import path from 'path';

import { Connection, Network, Node } from '../../../src/browser-entry';
import {
  createRuntimeAdaptationEngine,
  detectScoreWindowOscillation,
  detectSteeringOscillation,
  evaluateRollingScoreWindow,
  evaluateRacingTrendScore,
  RACING_COMPLEXITY_WEIGHT,
  RACING_OSCILLATION_COMMIT_THRESHOLD,
  RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
  RACING_OSCILLATION_MIN_MEAN_STEERING,
  RACING_OSCILLATION_MIN_NEURONS,
  RACING_OSCILLATION_PENALTY_WEIGHT,
  RACING_VARIANT_SCORER,
  resolveOscillationThresholdBoost,
  scoreRacingVariant,
} from './runtime.adaptation';
import * as runtimeAdaptationModule from './runtime.adaptation';

// DF12-5: diagnostic lines are flushed via requestIdleCallback/setTimeout.
// Run pending timers after each test so the async flush cannot fire after
// the suite has torn down and produce a "Can't perform a React state update"
// style console warning.
jest.useFakeTimers();
afterEach(() => {
  jest.runOnlyPendingTimers();
});

describe('runtime.adaptation module exports', () => {
  describe('createRuntimeAdaptationEngine', () => {
    it('is exported as a function', () => {
      expect(typeof createRuntimeAdaptationEngine).toBe('function');
    });
  });

  describe('evaluateRollingScoreWindow', () => {
    it('is exported as a function', () => {
      expect(typeof evaluateRollingScoreWindow).toBe('function');
    });
  });
});

describe('racing-specific trend-only evaluator', () => {
  describe('evaluateRacingTrendScore', () => {
    it('is exported as a function', () => {
      const adaptationExports = runtimeAdaptationModule as Record<
        string,
        unknown
      >;
      expect(typeof adaptationExports['evaluateRacingTrendScore']).toBe(
        'function',
      );
    });

    it('returns different scores for different network topologies with identical history', () => {
      const smallNetwork = new Network(4, 2, { seed: 42 });
      const largeNetwork = new Network(1001, 1, { seed: 42 });
      const scoreHistory = [1, 2, 3, 4] as const;

      const adaptationExports = runtimeAdaptationModule as unknown as Record<
        string,
        (network: Network, scoreHistory: readonly number[]) => number
      >;
      expect(
        adaptationExports['evaluateRacingTrendScore'](
          smallNetwork,
          scoreHistory,
        ),
      ).not.toBe(
        adaptationExports['evaluateRacingTrendScore'](
          largeNetwork,
          scoreHistory,
        ),
      );
    });
  });
});

describe('createRuntimeAdaptationEngine default racing evaluator', () => {
  it('grows a small seeded network after sustained ticks with a permissive threshold', async () => {
    const network = new Network(4, 2, { seed: 42 });
    const initialNodeCount = network.nodes.length;
    const initialConnCount = network.connections.length;
    // The new defaults require a positive improvement threshold and tick
    // cadence.  Use an explicit permissive configuration here so this smoke
    // test keeps exercising the actual growth path.
    const engine = createRuntimeAdaptationEngine({
      cadence: { mode: 'every_tick' },
      improvementThreshold: 0,
      limits: { mutationCooldownTicks: 0, rollbackCooldownTicks: 0 },
    });
    const scoreHistory = [1, 2, 3, 4];

    for (let tick = 0; tick < 60; tick++) {
      await engine.adaptOnTick({ tick, network, scoreHistory });
    }

    const networkGrew =
      network.nodes.length > initialNodeCount ||
      network.connections.length > initialConnCount;
    expect(networkGrew).toBe(true);
  });
});

type RacingQualitySignal = {
  trackProgress: number;
  forwardSpeed: number;
  headingAlignment: number;
  offTrackPenalty: number;
};

describe('composite driving-quality signal evaluator', () => {
  const evaluateRacingTrendScoreWithSignalHistory = (
    runtimeAdaptationModule as unknown as Record<
      string,
      (network: Network, signalHistory: RacingQualitySignal[]) => number
    >
  )['evaluateRacingTrendScore'];

  it('accepts a composite driving-quality signal history and returns a finite score', () => {
    const network = new Network(4, 2, { seed: 42 });
    const signalHistory: RacingQualitySignal[] = [
      {
        trackProgress: 0.8,
        forwardSpeed: 0.7,
        headingAlignment: 0.9,
        offTrackPenalty: 0,
      },
    ];

    const score = evaluateRacingTrendScoreWithSignalHistory(
      network,
      signalHistory,
    );
    expect(Number.isFinite(score)).toBe(true);
  });

  it('scores higher when track progress improves', () => {
    const network = new Network(4, 2, { seed: 42 });
    const lowProgressSignal: RacingQualitySignal = {
      trackProgress: 0.2,
      forwardSpeed: 0.5,
      headingAlignment: 0.8,
      offTrackPenalty: 0,
    };
    const highProgressSignal: RacingQualitySignal = {
      ...lowProgressSignal,
      trackProgress: 0.9,
    };

    expect(
      evaluateRacingTrendScoreWithSignalHistory(network, [highProgressSignal]),
    ).toBeGreaterThan(
      evaluateRacingTrendScoreWithSignalHistory(network, [lowProgressSignal]),
    );
  });

  it('scores lower when off-track penalty grows', () => {
    const network = new Network(4, 2, { seed: 42 });
    const onTrackSignal: RacingQualitySignal = {
      trackProgress: 0.5,
      forwardSpeed: 0.5,
      headingAlignment: 0.8,
      offTrackPenalty: 0,
    };
    const offTrackSignal: RacingQualitySignal = {
      ...onTrackSignal,
      offTrackPenalty: 1,
    };

    expect(
      evaluateRacingTrendScoreWithSignalHistory(network, [offTrackSignal]),
    ).toBeLessThan(
      evaluateRacingTrendScoreWithSignalHistory(network, [onTrackSignal]),
    );
  });
});

describe('default runtime adaptation gating', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('gates the first tick with every_n_ticks cadence', async () => {
    const engine = createRuntimeAdaptationEngine();
    const network = new Network(4, 2, { seed: 42 });
    const telemetry = await engine.adaptOnTick({
      tick: 1,
      network,
      scoreHistory: [1, 2, 3, 4],
    });

    expect(telemetry.reason).toBe('cadence_not_reached');
  });

  it('defaults to a positive improvement threshold constant', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const match = sourceText.match(
      /const\s+DEFAULT_IMPROVEMENT_THRESHOLD\s*=\s*([0-9.eE+-]+)\s*;/,
    );

    expect(match !== null && Number(match[1]) > 0).toBe(true);
  });

  it('defaults to a non-zero mutation cooldown in DEFAULT_LIMITS', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const defaultLimitsMatch = sourceText.match(
      /const\s+DEFAULT_LIMITS[\s\S]*?\n};/,
    );

    expect(
      defaultLimitsMatch !== null &&
        !defaultLimitsMatch[0].includes('mutationCooldownTicks: 0'),
    ).toBe(true);
  });

  it('defaults to a non-zero rollback cooldown in DEFAULT_LIMITS', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const defaultLimitsMatch = sourceText.match(
      /const\s+DEFAULT_LIMITS[\s\S]*?\n};/,
    );

    expect(
      defaultLimitsMatch !== null &&
        !defaultLimitsMatch[0].includes('rollbackCooldownTicks: 0'),
    ).toBe(true);
  });
});

describe('runtime adaptation commit behavior with global innovation counter (DF12-2)', () => {
  afterEach(() => {
    Connection.resetInnovationCounter(1);
  });

  it('commits structural mutations that the grow-stabilize cycle marks committed', async () => {
    // DF12-2 changed the runtime to trust the grow-stabilize cycle's commit
    // decision: when cycleResult.committed is true and structural operations
    // are non-empty, the runtime commits the mutation. The first structural
    // growth still commits unconditionally because isFirstGrowth is true. A
    // subsequent structural mutation that the cycle commits is therefore also
    // accepted by the runtime, even though evaluateScore returns
    // -connections.length and the raw score decreases.
    Connection.resetInnovationCounter(1000);
    const network = new Network(4, 2, { seed: 42 });
    const engine = createRuntimeAdaptationEngine({
      cadence: { mode: 'every_tick' },
      improvementThreshold: 0.01,
      evaluateScore: (n: Network) => -n.connections.length,
    });

    let sawMutation = false;
    let firstGrowthCommitted = false;
    let innovationAfterFirstGrowth = Connection.nextInnovation;
    let sawRollback = false;

    for (let tick = 0; tick < 300; tick++) {
      const telemetry = await engine.adaptOnTick({
        tick,
        network,
        scoreHistory: [1, 2, 3, 4],
      });

      if (telemetry.operations.length > 0) {
        sawMutation = true;
      }

      // Record the innovation counter right after the first growth commits.
      if (
        telemetry.committed &&
        telemetry.operations.length > 0 &&
        !firstGrowthCommitted
      ) {
        firstGrowthCommitted = true;
        innovationAfterFirstGrowth = Connection.nextInnovation;
      }

      // After the first growth, any rollback under DF12-2 would report reason
      // 'improvement_below_threshold'. With the new commit semantics the cycle
      // commits structural growth, so no rollback occurs.
      if (
        firstGrowthCommitted &&
        telemetry.reason === 'improvement_below_threshold'
      ) {
        sawRollback = true;
        break;
      }
    }

    expect({
      counterRestored: Connection.nextInnovation === innovationAfterFirstGrowth,
      sawMutation,
      firstGrowthCommitted,
      sawRollback,
    }).toEqual({
      counterRestored: false,
      sawMutation: true,
      firstGrowthCommitted: true,
      sawRollback: false,
    });
  });
});

describe('network-aware trend scoring (P8S20)', () => {
  it('produces different scores for networks with different topologies', () => {
    const smallNetwork = new Network(4, 2, { seed: 42 });
    const largeNetwork = new Network(8, 4, { seed: 42 });
    const scoreHistory = [1, 2, 3, 4] as const;

    const adaptationExports = runtimeAdaptationModule as unknown as Record<
      string,
      (network: Network, scoreHistory: readonly number[]) => number
    >;

    const scoreA = adaptationExports['evaluateRacingTrendScore'](
      smallNetwork,
      scoreHistory,
    );
    const scoreB = adaptationExports['evaluateRacingTrendScore'](
      largeNetwork,
      scoreHistory,
    );

    expect(scoreA).not.toBe(scoreB);
  });
});

describe('growth budget episodic slot allowance (P8S20)', () => {
  it('reserves non-zero episodic slots in buildGrowthBudget', () => {
    const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const match = sourceText.match(/function buildGrowthBudget[\s\S]*?\n}/);

    expect(match?.[0]?.includes('maxEpisodicSlots: 0')).toBe(false);
  });
});

describe('AC-RC-21-001: forward-pass evaluation rejects behaviorally-neutral mutations (P8S21)', () => {
  it('(P8S21) evaluateRacingTrendScore runs a forward pass on sample observations', () => {
    const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const evaluatorStart = sourceText.indexOf(
      'export function evaluateRacingTrendScore',
    );
    const evaluatorSection = sourceText.slice(
      evaluatorStart,
      evaluatorStart + 2000,
    );

    expect(evaluatorSection.includes('activate')).toBe(true);
  });

  it('(P8S21) rejects behaviorally-neutral mutation with identical forward-pass outputs', () => {
    const baselineNetwork = new Network(4, 2, { seed: 42 });
    const deadWeightNetwork = new Network(4, 2, { seed: 42 });
    const deadNode = new Node('hidden');
    deadWeightNetwork.nodes.push(deadNode);

    const sampleSignal: RacingQualitySignal = {
      trackProgress: 0.5,
      forwardSpeed: 0.5,
      headingAlignment: 0.8,
      offTrackPenalty: 0,
    };

    const baselineScore = evaluateRacingTrendScore(baselineNetwork, [
      sampleSignal,
    ]);
    const candidateScore = evaluateRacingTrendScore(deadWeightNetwork, [
      sampleSignal,
    ]);

    expect(candidateScore).toBe(baselineScore);
  });
});

describe('AC-RC-21-002: performance-gated complexityBonus (P8S21)', () => {
  it('(P8S21) mutation with decreased driving quality is rejected, not just penalized', () => {
    const baselineNetwork = new Network(4, 2, { seed: 42 });
    const largerNetwork = new Network(8, 4, { seed: 42 });

    const decreasedQualityHistory: RacingQualitySignal[] = [
      {
        trackProgress: 0.9,
        forwardSpeed: 0.9,
        headingAlignment: 0.9,
        offTrackPenalty: 0,
      },
      {
        trackProgress: 0.2,
        forwardSpeed: 0.2,
        headingAlignment: 0.2,
        offTrackPenalty: 0.5,
      },
    ];

    const baselineScore = evaluateRacingTrendScore(
      baselineNetwork,
      decreasedQualityHistory,
    );
    const candidateScore = evaluateRacingTrendScore(
      largerNetwork,
      decreasedQualityHistory,
    );

    expect(candidateScore).toBeLessThanOrEqual(baselineScore);
  });

  it('(P8S21) source gates complexityBonus on driving quality improvement', () => {
    const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const evaluatorStart = sourceText.indexOf(
      'export function evaluateRacingTrendScore',
    );
    const evaluatorSection = sourceText.slice(
      evaluatorStart,
      evaluatorStart + 3000,
    );

    const hasComplexityBonus = evaluatorSection.includes('complexityBonus');
    const unconditionalComplexityBonus =
      /complexityBonus\s*=\s*\(\s*network\.nodes\.length\s*\+\s*network\.connections\.length\s*\)\s*\*\s*RACING_COMPLEXITY_WEIGHT/.test(
        evaluatorSection,
      );

    expect(hasComplexityBonus && !unconditionalComplexityBonus).toBe(true);
  });
});

describe('AC-RC-21-003: physics rewards fed to evaluator (P8S21)', () => {
  it('(P8S21) RacingQualitySignal interface includes physicsReward field', () => {
    const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const interfaceStart = sourceText.indexOf(
      'export interface RacingQualitySignal',
    );
    const interfaceSection = sourceText.slice(
      interfaceStart,
      interfaceStart + 500,
    );

    expect(interfaceSection.includes('physicsReward')).toBe(true);
  });

  it('(P8S21) browser-entry reads car.reward from stepEnvironment output', () => {
    const browserEntryPath = path.join(
      __dirname,
      '..',
      'browser-entry',
      'browser-entry.ts',
    );
    const sourceText = fs.readFileSync(browserEntryPath, 'utf8');

    expect(sourceText.includes('.reward')).toBe(true);
  });
});

describe('AC-RC-21-004: per-car trackProgress (P8S21)', () => {
  it('(P8S21) trackProgress is computed per-car, not shared via curriculumProgress.lapProgress', () => {
    const browserEntryPath = path.join(
      __dirname,
      '..',
      'browser-entry',
      'browser-entry.ts',
    );
    const sourceText = fs.readFileSync(browserEntryPath, 'utf8');

    expect(
      sourceText.includes(
        'curriculumProgress.lapProgress.lastClosestSplineSampleIndex',
      ),
    ).toBe(false);
  });
});

describe('AC-RC-21-005: actual physics speed and physics-based signal (P8S21)', () => {
  it('(P8S21) forwardSpeed does not use commanded throttle', () => {
    const browserEntryPath = path.join(
      __dirname,
      '..',
      'browser-entry',
      'browser-entry.ts',
    );
    const sourceText = fs.readFileSync(browserEntryPath, 'utf8');

    expect(sourceText.includes('perCarTickResult.control.throttle')).toBe(
      false,
    );
  });

  it('(P8S21) headingAlignment does not use observation-vector evidence', () => {
    const browserEntryPath = path.join(
      __dirname,
      '..',
      'browser-entry',
      'browser-entry.ts',
    );
    const sourceText = fs.readFileSync(browserEntryPath, 'utf8');

    expect(
      sourceText.includes('perCarTickResult.evidence.headingAlignment01'),
    ).toBe(false);
  });

  it('(P8S21) offTrackPenalty does not use observation-vector lateralError', () => {
    const browserEntryPath = path.join(
      __dirname,
      '..',
      'browser-entry',
      'browser-entry.ts',
    );
    const sourceText = fs.readFileSync(browserEntryPath, 'utf8');

    expect(
      sourceText.includes('perCarTickResult.evidence.lateralErrorNormalized'),
    ).toBe(false);
  });
});

describe('P8S22 — AC-RC-22-001: hysteresisWindowCount uses adaptive resolution >= 2', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S22) lifecycle config uses resolveAdaptiveHysteresis (not hardcoded 0)', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // The lifecycle config should now call resolveAdaptiveHysteresis
    // instead of using a hardcoded literal.
    const hasHysteresisZero = /hysteresisWindowCount:\s*0\b/.test(sourceText);
    const usesAdaptive =
      sourceText.includes('resolveAdaptiveHysteresis') &&
      typeof (
        runtimeAdaptationModule as unknown as Record<
          string,
          (nodeCount: number) => number
        >
      )['resolveAdaptiveHysteresis'] === 'function';

    expect(!hasHysteresisZero && usesAdaptive).toBe(true);
  });
});

describe('P8S22 — AC-RC-22-004: MAX_EPISODIC_SLOTS reduced from 100 to <= 20', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S22) MAX_EPISODIC_SLOTS constant is at most 20 (not 100)', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    const match = sourceText.match(/const\s+MAX_EPISODIC_SLOTS\s*=\s*(\d+)/);
    const slotCount = match ? Number(match[1]) : 100;

    expect(slotCount).toBeLessThanOrEqual(20);
  });
});

describe('P8S22 — AC-RC-22-005: fitness plateau detector before growth', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S22) source contains plateau or variance detection logic', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // After the fix, a plateau detector should exist that checks quality
    // variance before allowing growth. This must be in the adaptOnTick
    // implementation, not in the existing resolveBehavioralComplexity variance
    // computation which is part of the evaluator. Search the full source
    // rather than a brittle substring window so interface declarations do
    // not hide the implementation body.
    const hasPlateauOrVariance =
      sourceText.includes('plateau') ||
      sourceText.includes('Plateau') ||
      sourceText.includes('qualityVariance');

    expect(hasPlateauOrVariance).toBe(true);
  });

  it('(P8S22) telemetry reason includes plateau_not_reached', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // The RuntimeAdaptationTelemetry reason union should include a
    // plateau-related reason after the fix.
    expect(sourceText.includes('plateau_not_reached')).toBe(true);
  });
});

describe('P8S22 — AC-RC-22-008: physicsReward and offTrackPenalty weights >= 0.3', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S22) toDrivingQuality uses physicsReward weight >= 0.3 (not 0.1)', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // Locate the toDrivingQuality function body and extract the physicsReward weight.
    const fnStart = sourceText.indexOf('function toDrivingQuality');
    const fnSection = sourceText.slice(fnStart, fnStart + 500);
    const physicsMatch = fnSection.match(
      /physicsReward\s*\?\?\s*0\)\s*\*\s*([0-9.]+)/,
    );
    const physicsWeight = physicsMatch ? Number(physicsMatch[1]) : 0;

    expect(physicsWeight).toBeGreaterThanOrEqual(0.3);
  });

  it('(P8S22) toDrivingQuality uses offTrackPenalty weight >= 0.3 (not 0.1)', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // Locate the toDrivingQuality function body and extract the offTrackPenalty weight.
    const fnStart = sourceText.indexOf('function toDrivingQuality');
    const fnSection = sourceText.slice(fnStart, fnStart + 500);
    const offTrackMatch = fnSection.match(/offTrackPenalty\s*\*\s*([0-9.]+)/);
    const offTrackWeight = offTrackMatch ? Number(offTrackMatch[1]) : 0;

    expect(offTrackWeight).toBeGreaterThanOrEqual(0.3);
  });
});

describe('P8S22 — AC-RC-22-012: evaluator uses separate baseline/candidate score windows', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S22) adaptOnTick captures pre-mutation baseline score separately from post-mutation candidate', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // After the fix, the evaluator should capture a pre-mutation baseline
    // score and compare against a post-mutation candidate score, rather than
    // using the same scoreHistory for both.
    // Currently both baselineScore and candidateScore call evaluateScore with
    // the same evidenceWindow/scoreHistory. The fix should capture a
    // pre-mutation baseline snapshot.
    const adaptStart = sourceText.indexOf('adaptOnTick');
    const adaptSection = sourceText.slice(adaptStart, adaptStart + 3000);

    // Check that the source distinguishes baseline from candidate — either
    // by capturing a pre-mutation score snapshot or by using separate windows.
    const hasSeparateBaseline =
      adaptSection.includes('preMutation') ||
      adaptSection.includes('baselineSnapshot') ||
      adaptSection.includes('preMutationBaseline') ||
      adaptSection.includes('baselineScoreWindow') ||
      adaptSection.includes('capturedBaseline');

    expect(hasSeparateBaseline).toBe(true);
  });

  it('(P8S22) candidate score is not computed from the same scoreHistory as baseline', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // After the fix, the candidate evaluation should use a post-mutation
    // score window, not the same scoreHistory used for baseline.
    const adaptStart = sourceText.indexOf('adaptOnTick');
    const adaptSection = sourceText.slice(adaptStart, adaptStart + 3000);

    // The current code evaluates both baseline and candidate with
    // evaluateScore(tickInput.network, evidenceWindow). After the fix,
    // the candidate should use a separate post-mutation window or
    // re-evaluation that reflects driving quality changes.
    const hasPostMutationCandidate =
      adaptSection.includes('postMutation') ||
      adaptSection.includes('candidateScoreWindow') ||
      adaptSection.includes('postMutationCandidate') ||
      adaptSection.includes('candidateEvidenceWindow') ||
      adaptSection.includes('candidateWindow');

    expect(hasPostMutationCandidate).toBe(true);
  });
});

describe('P8S23 — AC-023-001: adaptive hysteresis resolution', () => {
  const moduleExports = runtimeAdaptationModule as unknown as Record<
    string,
    (nodeCount: number) => number
  >;

  it('(P8S23) resolveAdaptiveHysteresis is exported as a function', () => {
    expect(typeof moduleExports['resolveAdaptiveHysteresis']).toBe('function');
  });

  it('(P8S23) resolveAdaptiveHysteresis returns 2 for 200 hidden nodes', () => {
    const fn = moduleExports['resolveAdaptiveHysteresis'];
    expect(fn ? fn(200) : undefined).toBe(2);
  });

  it('(P8S23) resolveAdaptiveHysteresis returns 3 for 500 hidden nodes', () => {
    const fn = moduleExports['resolveAdaptiveHysteresis'];
    expect(fn ? fn(500) : undefined).toBe(3);
  });

  it('(P8S23) resolveAdaptiveHysteresis returns 5 for 501 hidden nodes', () => {
    const fn = moduleExports['resolveAdaptiveHysteresis'];
    expect(fn ? fn(501) : undefined).toBe(5);
  });
});

describe('P8S23 — AC-023-002: lifecycle call uses adaptive hysteresis not hardcoded 5', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S23) source does not contain hardcoded hysteresisWindowCount: 5 literal', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    expect(/hysteresisWindowCount:\s*5\b/.test(sourceText)).toBe(false);
  });

  it('(P8S23) lifecycle config area calls resolveAdaptiveHysteresis', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const lifecycleIdx = sourceText.indexOf('runNgeLifecycle({');
    const configSection = sourceText.slice(
      lifecycleIdx - 300,
      lifecycleIdx + 400,
    );
    expect(configSection.includes('resolveAdaptiveHysteresis')).toBe(true);
  });
});

describe('P8S23 — AC-023-003: plateau window reduced to 5 and threshold raised to 0.1', () => {
  const coreSourcePath = path.join(
    __dirname,
    '..',
    '..',
    '..',
    'src',
    'neat',
    'nge-juvenile',
    'neat.nge-juvenile.constants.ts',
  );

  it('(P8S23) NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE is 5 (not 10)', () => {
    const sourceText = fs.readFileSync(coreSourcePath, 'utf8');
    const match = sourceText.match(
      /NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE\s*=\s*(\d+)/,
    );
    const value = match ? Number(match[1]) : 10;
    expect(value).toBe(5);
  });

  it('(P8S23) NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD is 0.1 (not 0.05)', () => {
    const sourceText = fs.readFileSync(coreSourcePath, 'utf8');
    const match = sourceText.match(
      /NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD\s*=\s*([0-9.]+)/,
    );
    const value = match ? Number(match[1]) : 0.05;
    expect(value).toBe(0.1);
  });
});

describe('P8S23 — AC-023-004: time-boxed stabilization with min 5 and max 25 ticks', () => {
  const coreSourcePath = path.join(
    __dirname,
    '..',
    '..',
    '..',
    'src',
    'neat',
    'nge-juvenile',
    'neat.nge-juvenile.grow-stabilize.ts',
  );

  it('(P8S23) isPlateauReached accepts stabilizationTicksSinceGrowth parameter', () => {
    const sourceText = fs.readFileSync(coreSourcePath, 'utf8');
    const fnStart = sourceText.indexOf('function isPlateauReached');
    const fnSection = sourceText.slice(fnStart, fnStart + 600);
    expect(fnSection.includes('stabilizationTicksSinceGrowth')).toBe(true);
  });

  it('(P8S23) isPlateauReached has max stabilization tick cap of 25', () => {
    const sourceText = fs.readFileSync(coreSourcePath, 'utf8');
    const fnStart = sourceText.indexOf('function isPlateauReached');
    const fnSection = sourceText.slice(fnStart, fnStart + 800);
    expect(fnSection.includes('25')).toBe(true);
  });

  it('(P8S23) isPlateauReached has min stabilization tick floor of 5 before plateau', () => {
    const sourceText = fs.readFileSync(coreSourcePath, 'utf8');
    const fnStart = sourceText.indexOf('function isPlateauReached');
    const fnSection = sourceText.slice(fnStart, fnStart + 800);
    const hasMinFloor =
      fnSection.includes('MIN_STABILIZATION') ||
      /stabilizationTicksSinceGrowth\s*<\s*\d+/.test(fnSection);
    expect(hasMinFloor).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Phase 9 Step 03 — Red tests for app-layer thinning (AC-015)
// These tests assert that NGE core functions and constants have been
// EXTRACTED out of runtime.adaptation.ts into src/neat/nge-juvenile/.
// They fail because the extraction has not happened yet.
// ──────────────────────────────────────────────────────────────────────

describe('Phase 9 Step 03 — app-layer thinning: NGE core removed from runtime.adaptation.ts', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P9S03) resolveAdaptiveHysteresis is not defined in runtime.adaptation.ts', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(
      /(?:export\s+)?function\s+resolveAdaptiveHysteresis\s*\(/.test(
        sourceText,
      ),
    ).toBe(false);
  });

  it('(P9S03) isPlateauReached is not defined in runtime.adaptation.ts', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(/function\s+isPlateauReached\s*\(/.test(sourceText)).toBe(false);
  });

  it('(P9S03) applyWeightMutations is not defined in runtime.adaptation.ts', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(/function\s+applyWeightMutations\s*\(/.test(sourceText)).toBe(false);
  });

  it('(P9S03) PLATEAU_WINDOW_SIZE and PLATEAU_VARIANCE_THRESHOLD are not defined in runtime.adaptation.ts', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const hasPlateauWindow = /const\s+PLATEAU_WINDOW_SIZE\s*=/.test(sourceText);
    const hasPlateauVariance = /const\s+PLATEAU_VARIANCE_THRESHOLD\s*=/.test(
      sourceText,
    );

    expect(hasPlateauWindow || hasPlateauVariance).toBe(false);
  });

  it('(P9S03) WEIGHT_MUTATION_RATE and WEIGHT_MUTATION_MAGNITUDE are not defined in runtime.adaptation.ts', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const hasRate = /const\s+WEIGHT_MUTATION_RATE\s*=/.test(sourceText);
    const hasMagnitude = /const\s+WEIGHT_MUTATION_MAGNITUDE\s*=/.test(
      sourceText,
    );

    expect(hasRate || hasMagnitude).toBe(false);
  });

  it('(P9S03) MIN_STABILIZATION_TICKS and MAX_STABILIZATION_TICKS are not defined in runtime.adaptation.ts', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const hasMin = /const\s+MIN_STABILIZATION_TICKS\s*=/.test(sourceText);
    const hasMax = /const\s+MAX_STABILIZATION_TICKS\s*=/.test(sourceText);

    expect(hasMin || hasMax).toBe(false);
  });

  it('(P9S03) runtime.adaptation.ts imports runNgeGrowStabilizeCycle from the core module', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const hasImport = sourceText.includes('runNgeGrowStabilizeCycle');

    expect(hasImport).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Phase 9 Step 03 — Red tests for driving improvement (AC-017)
// These tests assert that driving quality rewards are stronger than
// the current values. They fail because the current values are too low.
// ──────────────────────────────────────────────────────────────────────

describe('Phase 9 Step 03 — driving improvement: stronger reward shaping', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');
  const envSourcePath = path.join(
    __dirname,
    '..',
    'environment',
    'environment.step.service.ts',
  );

  it('(P9S03) toDrivingQuality weights physicsReward at ≥ 0.5', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const fnStart = sourceText.indexOf('function toDrivingQuality');
    const fnSection = sourceText.slice(fnStart, fnStart + 500);
    const match = fnSection.match(
      /physicsReward\s*\?\?\s*0\)\s*\*\s*([0-9.]+)/,
    );

    expect(match !== null && Number(match[1]) >= 0.5).toBe(true);
  });

  it('(P9S03) WRONG_DIRECTION_REWARD uses an escalating penalty, not a flat -5', () => {
    const sourceText = fs.readFileSync(envSourcePath, 'utf8');
    const hasEscalating =
      sourceText.includes('consecutiveWrongDirection') ||
      sourceText.includes('wrongDirectionTicks') ||
      /WRONG_DIRECTION.*\*\s*\w+/.test(sourceText);

    expect(hasEscalating).toBe(true);
  });

  it('(P9S03) DEFAULT_IMPROVEMENT_THRESHOLD is ≥ 0.02', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const match = sourceText.match(
      /const\s+DEFAULT_IMPROVEMENT_THRESHOLD\s*=\s*([0-9.]+)\s*;/,
    );

    expect(match !== null && Number(match[1]) >= 0.02).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Phase 9 Step 03 — Red tests for growth speed (AC-018)
// These tests assert that maxStructuralEditsPerStep is wired into the
// lifecycle call and the default is raised. They fail because the knob
// is currently dead (not passed to runNgeLifecycle) and the default is 1.
// ──────────────────────────────────────────────────────────────────────

describe('Phase 9 Step 03 — growth speed: maxStructuralEditsPerStep wired', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P9S03) maxStructuralEditsPerStep is passed to the runNgeLifecycle call', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const lifecycleStart = sourceText.indexOf('runNgeLifecycle(');
    const lifecycleSection = sourceText.slice(
      lifecycleStart,
      lifecycleStart + 700,
    );

    expect(lifecycleSection.includes('maxStructuralEditsPerStep')).toBe(true);
  });

  it('(P9S03) default maxStructuralEditsPerStep is ≥ 5 for batch growth', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');
    const match = sourceText.match(/maxStructuralEditsPerStep:\s*(\d+)/);

    expect(match !== null && Number(match[1]) >= 5).toBe(true);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Phase 9 Step 03 — deferred: import path extension cleanup (AC-019, AC-043)
// This test asserts that the current test file does not use explicit .ts
// extensions in import paths, which keeps the source compatible with standard
// ESM/Bundler resolution and avoids TS5097 in non-test builds.
// ──────────────────────────────────────────────────────────────────────

describe('Phase 9 Step 03 — deferred: runtime.adaptation.test.ts import path fix', () => {
  it('(P9S03) runtime.adaptation.test.ts does not use .ts extension in import paths', () => {
    const testPath = path.join(__dirname, 'runtime.adaptation.test.ts');
    const testSource = fs.readFileSync(testPath, 'utf8');
    const hasTsExtension = /from\s+['"][^'"]*\.ts['"]/.test(testSource);

    expect(hasTsExtension).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────
// Workstream B — Red tests for racing-side oscillation detection.
// These tests fail because the helpers, variant scorer, and tuning
// constants are not yet exported from runtime.adaptation.ts.
// ──────────────────────────────────────────────────────────────────────

describe('detectSteeringOscillation helper', () => {
  it('returns 0 for smooth steering outputs with the same sign', () => {
    expect(detectSteeringOscillation([0.2, 0.3, 0.4, 0.5])).toBe(0);
  });

  it('returns a high value for alternating steering outputs', () => {
    const metric = detectSteeringOscillation([0.5, -0.5, 0.5, -0.5]);
    expect(metric).toBeGreaterThan(0.5);
  });

  it('returns 0 when fewer than 3 steering outputs are provided', () => {
    expect(detectSteeringOscillation([0.5, -0.5])).toBe(0);
  });

  it('returns 0 for all-zero steering outputs', () => {
    expect(detectSteeringOscillation([0, 0, 0, 0])).toBe(0);
  });

  it('returns 0 for gentle corrections below the magnitude deadband', () => {
    // Small alternating values model a smooth S-curve correction; they must
    // not be treated like aggressive zig-zags.
    expect(detectSteeringOscillation([0.05, -0.05, 0.05, -0.05])).toBe(0);
  });

  it('returns a non-zero metric for alternating outputs above the magnitude deadband', () => {
    const metric = detectSteeringOscillation([0.2, -0.2, 0.2, -0.2]);

    expect(metric).toBeGreaterThan(0);
  });
});

describe('detectScoreWindowOscillation helper', () => {
  it('returns 0 for monotonically increasing scores', () => {
    expect(detectScoreWindowOscillation([1, 2, 3, 4, 5])).toBe(0);
  });

  it('returns a high value for alternating up/down scores', () => {
    const metric = detectScoreWindowOscillation([1, 3, 1, 3, 1]);
    expect(metric).toBeGreaterThan(0.5);
  });

  it('returns 0 when fewer than 4 score values are provided', () => {
    expect(detectScoreWindowOscillation([1, 2, 3])).toBe(0);
  });
});

describe('RACING_VARIANT_SCORER oscillation penalty', () => {
  const zeroTarget = [0, 0, 0, 0];

  it('scores oscillating steering outputs lower than smooth outputs of the same magnitude', () => {
    const smooth = RACING_VARIANT_SCORER(
      [
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
      ],
      zeroTarget,
    );
    const oscillating = RACING_VARIANT_SCORER(
      [
        [0.5, -0.5],
        [-0.5, 0.5],
        [0.5, -0.5],
        [-0.5, 0.5],
      ],
      zeroTarget,
    );

    expect(oscillating).toBeLessThan(smooth);
  });

  it('penalizes proportionally to baseScore times oscillation metric times weight', () => {
    const outputs = [
      [0.5, -0.5],
      [-0.5, 0.5],
      [0.5, -0.5],
      [-0.5, 0.5],
    ];
    const score = RACING_VARIANT_SCORER(outputs, zeroTarget);
    const steering = outputs.map((output) => output[1]);
    const metric = detectSteeringOscillation(steering);

    // Every output has mean absolute magnitude 0.5 and zero trend, so the
    // internal baseScore is exactly 0.5.  The score also receives a tiny
    // behavioral-complexity bonus because the outputs are not uniform,
    // so the expected score is baseScore + complexityBonus - penalty.
    const baseScore = 0.5;
    const behavioralComplexity = 0.5; // total variance across both output dimensions
    const complexityBonus = behavioralComplexity * RACING_COMPLEXITY_WEIGHT;
    const expectedPenalty =
      baseScore * metric * RACING_OSCILLATION_PENALTY_WEIGHT;
    const expectedScore = baseScore + complexityBonus - expectedPenalty;

    expect(score).toBeCloseTo(expectedScore, 10);
  });
});

describe('scoreRacingVariant tier-aware oscillation penalty', () => {
  const zeroTarget = [0, 0, 0, 0];
  const smoothOutputs = [
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
  ];
  const oscillatingOutputs = [
    [0.5, -0.5],
    [-0.5, 0.5],
    [0.5, -0.5],
    [-0.5, 0.5],
  ];

  it('does not penalize oscillation for networks below RACING_OSCILLATION_MIN_NEURONS', () => {
    const smooth = scoreRacingVariant(
      smoothOutputs,
      zeroTarget,
      RACING_OSCILLATION_MIN_NEURONS - 1,
    );
    const oscillating = scoreRacingVariant(
      oscillatingOutputs,
      zeroTarget,
      RACING_OSCILLATION_MIN_NEURONS - 1,
    );

    expect(oscillating).not.toBeLessThan(smooth);
  });

  it('penalizes oscillation for networks at or above RACING_OSCILLATION_MIN_NEURONS', () => {
    const smooth = scoreRacingVariant(
      smoothOutputs,
      zeroTarget,
      RACING_OSCILLATION_MIN_NEURONS,
    );
    const oscillating = scoreRacingVariant(
      oscillatingOutputs,
      zeroTarget,
      RACING_OSCILLATION_MIN_NEURONS,
    );

    expect(oscillating).toBeLessThan(smooth);
  });
});

describe('resolveOscillationThresholdBoost', () => {
  it('returns 0 for networks below RACING_OSCILLATION_MIN_NEURONS', () => {
    expect(
      resolveOscillationThresholdBoost(1.0, RACING_OSCILLATION_MIN_NEURONS - 1),
    ).toBe(0);
  });

  it('returns 0 when the oscillation metric is at or below the commit gate', () => {
    expect(
      resolveOscillationThresholdBoost(
        RACING_OSCILLATION_COMMIT_THRESHOLD,
        RACING_OSCILLATION_MIN_NEURONS,
      ),
    ).toBe(0);
    expect(
      resolveOscillationThresholdBoost(
        RACING_OSCILLATION_COMMIT_THRESHOLD - 0.1,
        RACING_OSCILLATION_MIN_NEURONS,
      ),
    ).toBe(0);
  });

  it('returns the small additive boost for child-tier networks above the commit gate', () => {
    expect(
      resolveOscillationThresholdBoost(
        RACING_OSCILLATION_COMMIT_THRESHOLD + 0.01,
        RACING_OSCILLATION_MIN_NEURONS,
      ),
    ).toBe(RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST);
  });
});

describe('evaluateRacingTrendScore with oscillation penalty', () => {
  it('scores an oscillating score-history lower than a monotonic history', () => {
    const network = new Network(4, 2, { seed: 42 });
    const oscillating = evaluateRacingTrendScore(network, [1, 3, 1, 3, 1]);
    const monotonic = evaluateRacingTrendScore(network, [1, 2, 3, 4, 5]);

    expect(oscillating).toBeLessThan(monotonic);
  });

  it('applies the oscillation penalty only once the network reaches RACING_OSCILLATION_MIN_NEURONS', () => {
    const smallNetwork = new Network(4, 2, { seed: 42 });
    const largeNetwork = new Network(200, 1, { seed: 42 });
    // Same oscillating history for both networks.  The trend is negative so
    // the complexity bonus is gated off, isolating the oscillation-penalty
    // effect to the neuron-count gate.
    const oscillatingHistory = [5, 3, 5, 3, 1];

    const smallScore = evaluateRacingTrendScore(
      smallNetwork,
      oscillatingHistory,
    );
    const largeScore = evaluateRacingTrendScore(
      largeNetwork,
      oscillatingHistory,
    );

    expect(largeScore).toBeLessThan(smallScore);
  });
});

describe('Racing oscillation constants', () => {
  it('exports RACING_OSCILLATION_PENALTY_WEIGHT equal to 0.25', () => {
    expect(RACING_OSCILLATION_PENALTY_WEIGHT).toBe(0.25);
  });

  it('exports RACING_OSCILLATION_COMMIT_THRESHOLD equal to 0.5', () => {
    expect(RACING_OSCILLATION_COMMIT_THRESHOLD).toBe(0.5);
  });

  it('exports RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST equal to 0.03', () => {
    expect(RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST).toBe(0.03);
  });

  it('exports RACING_OSCILLATION_MIN_MEAN_STEERING equal to 0.15', () => {
    expect(RACING_OSCILLATION_MIN_MEAN_STEERING).toBe(0.15);
  });

  it('exports RACING_OSCILLATION_MIN_NEURONS equal to 200', () => {
    expect(RACING_OSCILLATION_MIN_NEURONS).toBe(200);
  });
});

describe('runtime adaptation growth threshold oscillation integration', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('uses the maximum of steering and score-window oscillation for the growth threshold', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(
      sourceText.includes('Math.max(steeringOscillation, scoreOscillation)'),
    ).toBe(true);
  });
});

describe('runtime adaptation scoreFn wiring', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('passes the live candidate neuron count to scoreRacingVariant', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    expect(
      sourceText.includes(
        'scoreRacingVariant(outputs, target, tickInput.network.nodes.length)',
      ),
    ).toBe(true);
  });
});
