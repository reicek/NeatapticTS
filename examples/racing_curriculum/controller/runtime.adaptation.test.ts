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

import { Connection, Network, Node } from '../../../src/browser-entry.ts';
import {
  createRuntimeAdaptationEngine,
  evaluateRollingScoreWindow,
  evaluateRacingTrendScore,
} from './runtime.adaptation';
import * as runtimeAdaptationModule from './runtime.adaptation';

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
  it('grows a small seeded network after sustained ticks with a permissive threshold', () => {
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
      engine.adaptOnTick({ tick, network, scoreHistory });
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

  it('gates the first tick with every_n_ticks cadence', () => {
    const engine = createRuntimeAdaptationEngine();
    const network = new Network(4, 2, { seed: 42 });
    const telemetry = engine.adaptOnTick({
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

describe('runtime adaptation rollback preserves global innovation counter', () => {
  afterEach(() => {
    Connection.resetInnovationCounter(1);
  });

  it('restores Connection.nextInnovation after a rolled-back mutation', () => {
    // With the unconditional first-growth commit (isFirstGrowth → shouldCommit
    // = true), the first structural mutation always commits. To test rollback
    // behavior we must first let the first growth commit, then wait for the
    // plateau + hysteresis gates to open so a SECOND growth is attempted. The
    // evaluateScore returns -connections.length, so any structural growth
    // decreases the score and triggers a rollback on the second attempt.
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
      const telemetry = engine.adaptOnTick({
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

      // After the first growth, a subsequent structural mutation that does not
      // improve the score is rolled back with reason 'improvement_below_threshold'.
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
      counterRestored: true,
      sawMutation: true,
      firstGrowthCommitted: true,
      sawRollback: true,
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

describe('P8S22 — AC-RC-22-001: hysteresisWindowCount raised from 0 to >= 3', () => {
  const sourcePath = path.join(__dirname, 'runtime.adaptation.ts');

  it('(P8S22) lifecycle config passes hysteresisWindowCount >= 3 (not 0)', () => {
    const sourceText = fs.readFileSync(sourcePath, 'utf8');

    // The lifecycle config in adaptOnTick currently passes hysteresisWindowCount: 0.
    // After the fix it should be >= 3.
    const hasHysteresisZero = /hysteresisWindowCount:\s*0\b/.test(sourceText);
    const hysteresisMatch = sourceText.match(/hysteresisWindowCount:\s*(\d+)/);
    const hysteresisValue = hysteresisMatch ? Number(hysteresisMatch[1]) : 0;

    expect(!hasHysteresisZero && hysteresisValue >= 3).toBe(true);
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
    // function, not in the existing resolveBehavioralComplexity variance
    // computation which is part of the evaluator.
    const adaptStart = sourceText.indexOf('adaptOnTick');
    const adaptSection = sourceText.slice(adaptStart, adaptStart + 4000);

    const hasPlateauOrVariance =
      adaptSection.includes('plateau') ||
      adaptSection.includes('Plateau') ||
      adaptSection.includes('qualityVariance');

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
