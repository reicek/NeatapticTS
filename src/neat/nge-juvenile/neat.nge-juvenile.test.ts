import Network from '../../architecture/network';
import { applyMorphDeltas } from './neat.nge-juvenile.apply';
import {
  NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
  NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD,
  NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
  NGE_JUVENILE_DEFAULT_GATING_EDGE_LENGTH_THRESHOLD,
  NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE,
  NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW,
  NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
  NGE_JUVENILE_DEFAULT_LESION_SEVERITY,
  NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR,
  NGE_JUVENILE_DEFAULT_NOISE_SIGMA,
  NGE_JUVENILE_DEFAULT_PRUNE_COST_PRESSURE_THRESHOLD,
  NGE_JUVENILE_DEFAULT_PROBE_CADENCE_EPOCHS,
  NGE_JUVENILE_DEFAULT_PROBE_KINDS,
  NGE_JUVENILE_DEFAULT_PROBE_MAX_LEDGER_ENTRIES,
  NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR,
  NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT,
} from './neat.nge-juvenile.constants';
import {
  NgeJuvenile_BudgetError,
  NgeJuvenile_MorphError,
  NgeJuvenile_ProbeError,
} from './neat.nge-juvenile.errors';
import {
  computeFocusScores,
  resolveFocusConfig,
} from './neat.nge-juvenile.focus';
import {
  advanceGrowthHysteresis,
  canGrowNow,
  commitGrowth,
  planEdgeDensification,
  planGrowthMorphs,
  planNodeAddition,
  planSlotExpansion,
  validateMorphDelta,
} from './neat.nge-juvenile.grow';
import {
  advanceSchedulerState,
  appendProbeLedgerEntry,
  buildProbeLedgerEntry,
  computeProbeRewardDelta,
  decideProbe,
  defaultProbeSchedulerState,
  deserializeLedger,
  resolveProbeSchedulerConfig,
  serializeLedger,
} from './neat.nge-juvenile.probe';
import {
  advancePruneHysteresis,
  canPruneNow,
  commitPrune,
  planCompact,
  planEdgePrune,
  planPruneMorphs,
  selectPruneCandidate,
  validatePruneDelta,
} from './neat.nge-juvenile.prune';
import { minMaxNormalize, softmaxTopK } from './neat.nge-juvenile.utils';
import type {
  NgeFocusScore,
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeMorphDelta,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
  NgePruneCandidate,
  NgeProbeKind,
  NgeProbeLedgerEntry,
  NgeProbeSchedulerConfig,
  NgeProbeSchedulerState,
} from './neat.nge-juvenile.types';

describe('nge juvenile focus metrics', () => {
  describe('computeFocusScores', () => {
    describe('given uniform metrics', () => {
      it('keeps all module scores equal within floating-point tolerance', () => {
        // Arrange
        const snapshots = createUniformSnapshots();

        // Act
        const focusVector = computeFocusScores(
          snapshots,
          createResolvedConfig(),
        );
        const normalizedScoreSpread =
          Math.max(
            ...focusVector.scores.map(({ normalizedScore }) => normalizedScore),
          ) -
          Math.min(
            ...focusVector.scores.map(({ normalizedScore }) => normalizedScore),
          );

        // Assert
        expect(normalizedScoreSpread).toBeLessThanOrEqual(1e-12);
      });
    });

    describe('given competing reward and utilization leaders', () => {
      it('scores reward-delta emphasis above utilization emphasis with the default weights', () => {
        // Arrange
        const focusConfig = createResolvedConfig();
        const snapshots: NgeModuleMetricsSnapshot[] = [
          {
            moduleId: 'module:reward',
            novelty: 0,
            rewardDelta: 10,
            stabilityAge: 0,
            utilization: 0,
            wiringCost: 0,
          },
          {
            moduleId: 'module:utilization',
            novelty: 0,
            rewardDelta: 0,
            stabilityAge: 0,
            utilization: 10,
            wiringCost: 0,
          },
        ];

        // Act
        const focusVector = computeFocusScores(snapshots, focusConfig);

        // Assert
        expect(
          focusVector.scores[0].rawScore > focusVector.scores[1].rawScore,
        ).toBe(true);
      });
    });

    describe('given identical metric values across the whole slice', () => {
      it('produces uniform normalized scores', () => {
        // Arrange
        const focusConfig = createResolvedConfig();
        const snapshots = createUniformSnapshots();

        // Act
        const normalizedScores = computeFocusScores(
          snapshots,
          focusConfig,
        ).scores.map(({ normalizedScore }) => normalizedScore);

        // Assert
        expect(normalizedScores).toEqual([1 / 3, 1 / 3, 1 / 3]);
      });
    });

    describe('given a single-module slice', () => {
      it('returns one normalized score of one', () => {
        // Arrange
        const focusConfig = createResolvedConfig();
        const snapshots: NgeModuleMetricsSnapshot[] = [
          {
            moduleId: 'module:solo',
            novelty: 8,
            rewardDelta: 4,
            stabilityAge: 6,
            utilization: 2,
            wiringCost: 1,
          },
        ];

        // Act
        const focusVector = computeFocusScores(snapshots, focusConfig);

        // Assert
        expect(focusVector.scores[0].normalizedScore).toBe(1);
      });
    });

    describe('given distinct score timestamps', () => {
      it('keeps score math deterministic while leaving computedAt in metadata only', () => {
        // Arrange
        const focusConfig = createResolvedConfig({ windowIndex: 4 });
        const snapshots = createWeightedSnapshots();
        const dateNowSpy = jest
          .spyOn(Date, 'now')
          .mockReturnValueOnce(1_700_000_000_000)
          .mockReturnValueOnce(1_700_000_000_500);

        // Act
        const firstVector = computeFocusScores(snapshots, focusConfig);
        const secondVector = computeFocusScores(snapshots, focusConfig);
        dateNowSpy.mockRestore();

        // Assert
        expect({
          computedAtValues: [firstVector.computedAt, secondVector.computedAt],
          firstScores: firstVector.scores.map(
            ({ rawScore, normalizedScore }) => ({
              normalizedScore,
              rawScore,
            }),
          ),
          secondScores: secondVector.scores.map(
            ({ rawScore, normalizedScore }) => ({
              normalizedScore,
              rawScore,
            }),
          ),
          windowIndices: [firstVector.windowIndex, secondVector.windowIndex],
        }).toEqual({
          computedAtValues: [1_700_000_000_000, 1_700_000_000_500],
          firstScores: secondVector.scores.map(
            ({ rawScore, normalizedScore }) => ({
              normalizedScore,
              rawScore,
            }),
          ),
          secondScores: secondVector.scores.map(
            ({ rawScore, normalizedScore }) => ({
              normalizedScore,
              rawScore,
            }),
          ),
          windowIndices: [4, 4],
        });
      });
    });
  });

  describe('resolveFocusConfig', () => {
    describe('given an empty partial config', () => {
      it('fills every seeded default and keeps the focus weights summing to one', () => {
        // Arrange
        const resolvedConfig = resolveFocusConfig({});

        // Assert
        expect({
          cooldownWindowCount: resolvedConfig.cooldownWindowCount,
          episodicHitRateThreshold: resolvedConfig.episodicHitRateThreshold,
          focusWeightSum: Object.values(resolvedConfig.focusWeights).reduce(
            (currentTotal, weight) => currentTotal + weight,
            0,
          ),
          focusWeights: resolvedConfig.focusWeights,
          gainStabilityTolerance: resolvedConfig.gainStabilityTolerance,
          gainStabilityWindow: resolvedConfig.gainStabilityWindow,
          hysteresisWindowCount: resolvedConfig.hysteresisWindowCount,
          recurrentRefreshFloor: resolvedConfig.recurrentRefreshFloor,
          windowIndex: resolvedConfig.windowIndex,
        }).toEqual({
          cooldownWindowCount: NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
          episodicHitRateThreshold:
            NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD,
          focusWeightSum: 1,
          focusWeights: NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
          gainStabilityTolerance: NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE,
          gainStabilityWindow: NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW,
          hysteresisWindowCount: NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT,
          recurrentRefreshFloor: NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR,
          windowIndex: 0,
        });
      });
    });
  });

  describe('constants', () => {
    describe('given the seeded episodic hit-rate default', () => {
      it('keeps the explicit plan threshold guard at 0.65', () => {
        // Assert
        expect(NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD).toBe(0.65);
      });
    });

    describe('given the seeded prune cost-pressure default', () => {
      it('keeps the explicit prune threshold guard at 0.85', () => {
        // Assert
        expect(NGE_JUVENILE_DEFAULT_PRUNE_COST_PRESSURE_THRESHOLD).toBe(0.85);
      });
    });

    describe('given the seeded minimum edge floor default', () => {
      it('keeps the absolute prune floor at one edge', () => {
        // Assert
        expect(NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR).toBe(1);
      });
    });
  });

  describe('minMaxNormalize', () => {
    describe('given a standard spread', () => {
      it('returns the expected [0, 0.5, 1] normalization', () => {
        // Act
        const normalizedValues = minMaxNormalize([0, 0.5, 1]);

        // Assert
        expect(normalizedValues).toEqual([0, 0.5, 1]);
      });
    });

    describe('given a degenerate equal-valued vector', () => {
      it('returns a uniform vector without NaN values', () => {
        // Act
        const normalizedValues = minMaxNormalize([3, 3, 3]);

        // Assert
        expect(normalizedValues).toEqual([1 / 3, 1 / 3, 1 / 3]);
      });
    });
  });

  describe('softmaxTopK', () => {
    describe('given three differently scored items', () => {
      it('returns the highest two items when k is two', () => {
        // Arrange
        const items = [
          { id: 'low', score: 1 },
          { id: 'high', score: 5 },
          { id: 'mid', score: 3 },
        ];

        // Act
        const selectedItems = softmaxTopK(items, 2);

        // Assert
        expect(selectedItems.map(({ id }) => id)).toEqual(['high', 'mid']);
      });
    });

    describe('given tied scores', () => {
      it('preserves stable input order while selecting the tied leaders', () => {
        // Arrange
        const items = [
          { id: 'first', score: 2 },
          { id: 'second', score: 2 },
          { id: 'third', score: 1 },
        ];

        // Act
        const selectedItems = softmaxTopK(items, 2);

        // Assert
        expect(selectedItems.map(({ id }) => id)).toEqual(['first', 'second']);
      });
    });
  });

  describe('errors', () => {
    describe('given a budget error with a cause', () => {
      it('keeps the cause field and the public error name', () => {
        // Arrange
        const cause = new Error('budget-root-cause');

        // Act
        const budgetError = new NgeJuvenile_BudgetError('budget failed', {
          cause,
        });

        // Assert
        expect({ cause: budgetError.cause, name: budgetError.name }).toEqual({
          cause,
          name: 'NgeJuvenile_BudgetError',
        });
      });
    });

    describe('given a morph error with a cause', () => {
      it('keeps the cause field and the public error name', () => {
        // Arrange
        const cause = new Error('morph-root-cause');

        // Act
        const morphError = new NgeJuvenile_MorphError('morph failed', {
          cause,
        });

        // Assert
        expect({ cause: morphError.cause, name: morphError.name }).toEqual({
          cause,
          name: 'NgeJuvenile_MorphError',
        });
      });
    });
  });

  describe('placeholder JSON contracts', () => {
    describe('given the zero-default hysteresis state and one morph delta', () => {
      it('round-trips both placeholder types through JSON cleanly', () => {
        // Arrange
        const hysteresisState: NgeHysteresisState = {
          cooldownWindowsRemaining: 0,
          growthPositiveWindowCount: 0,
          lastMorphKind: 'none',
          pruneUnderuseWindowCount: 0,
        };
        const morphDelta: NgeMorphDelta = {
          detail: {
            edgesAdded: 1,
          },
          kind: 'edgeDensify',
          targetModuleId: 'module:alpha',
          wiringCostDelta: 0.25,
        };

        // Act
        const roundTrippedContracts = JSON.parse(
          JSON.stringify({ hysteresisState, morphDelta }),
        ) as {
          hysteresisState: NgeHysteresisState;
          morphDelta: NgeMorphDelta;
        };

        // Assert
        expect(roundTrippedContracts).toEqual({ hysteresisState, morphDelta });
      });
    });
  });

  describe('probe scheduler and ledger', () => {
    describe('decideProbe cadence gate', () => {
      describe('given the first epoch and an untouched state', () => {
        it('opens the cadence gate immediately', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const schedulerState = defaultProbeSchedulerState();

          // Act
          const probeDecision = decideProbe(0, schedulerState, schedulerConfig);

          // Assert
          expect(probeDecision).toEqual({
            shouldRun: true,
            probeKind: 'lesion',
          });
        });
      });

      describe('given an epoch inside the cadence window', () => {
        it('keeps the cadence gate closed', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const schedulerState = createProbeSchedulerState({
            lastProbeEpoch: 0,
          });

          // Act
          const probeDecision = decideProbe(5, schedulerState, schedulerConfig);

          // Assert
          expect(probeDecision).toEqual({
            shouldRun: false,
            probeKind: undefined,
          });
        });
      });

      describe('given an epoch at the cadence boundary', () => {
        it('opens the cadence gate and returns the scheduled probe kind', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const schedulerState = createProbeSchedulerState({
            lastProbeEpoch: 0,
            nextProbeKindIndex: 1,
          });

          // Act
          const probeDecision = decideProbe(
            10,
            schedulerState,
            schedulerConfig,
          );

          // Assert
          expect(probeDecision).toEqual({
            shouldRun: true,
            probeKind: 'noise',
          });
        });
      });
    });

    describe('probe kind rotation', () => {
      describe('given three consecutive executed probes', () => {
        it('cycles ledger entries through lesion, noise, and gating in order', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const schedulerState = defaultProbeSchedulerState();
          const firstEntry = buildProbeLedgerEntry(
            'lesion',
            'module:alpha',
            0,
            1,
            0.8,
          );
          const secondState = advanceSchedulerState(
            schedulerState,
            0,
            firstEntry,
            schedulerConfig,
          );
          const secondEntry = buildProbeLedgerEntry(
            'noise',
            'module:alpha',
            10,
            0.8,
            0.6,
          );
          const thirdState = advanceSchedulerState(
            secondState,
            10,
            secondEntry,
            schedulerConfig,
          );
          const thirdEntry = buildProbeLedgerEntry(
            'gating',
            'module:alpha',
            20,
            0.6,
            0.7,
          );

          // Act
          const finalState = advanceSchedulerState(
            thirdState,
            20,
            thirdEntry,
            schedulerConfig,
          );

          // Assert
          expect(finalState.ledger.map(({ probeKind }) => probeKind)).toEqual([
            'lesion',
            'noise',
            'gating',
          ]);
        });
      });

      describe('given a full pass across the configured probe kinds', () => {
        it('wraps the next probe kind index back to zero', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const initialState = createProbeSchedulerState({
            nextProbeKindIndex: 2,
          });
          const ledgerEntry = buildProbeLedgerEntry(
            'gating',
            'module:alpha',
            20,
            0.7,
            0.6,
          );

          // Act
          const nextState = advanceSchedulerState(
            initialState,
            20,
            ledgerEntry,
            schedulerConfig,
          );

          // Assert
          expect(nextState.nextProbeKindIndex).toBe(0);
        });
      });
    });

    describe('buildProbeLedgerEntry', () => {
      describe('given one improved reward reading', () => {
        it('stores a positive signed delta', () => {
          // Arrange
          const rewardBefore = 0.3;
          const rewardAfter = 0.8;

          // Act
          const ledgerEntry = buildProbeLedgerEntry(
            'lesion',
            'module:alpha',
            4,
            rewardBefore,
            rewardAfter,
          );

          // Assert
          expect(ledgerEntry.delta).toBe(0.5);
        });
      });

      describe('given one worsened reward reading', () => {
        it('stores a negative signed delta', () => {
          // Arrange
          const rewardBefore = 0.8;
          const rewardAfter = 0.2;

          // Act
          const ledgerEntry = buildProbeLedgerEntry(
            'noise',
            'module:alpha',
            5,
            rewardBefore,
            rewardAfter,
          );

          // Assert
          expect(ledgerEntry.delta).toBe(-0.6000000000000001);
        });
      });

      describe('given explicit probe metadata inputs', () => {
        it('echoes the caller metadata exactly', () => {
          // Arrange
          const probeKind: NgeProbeKind = 'gating';
          const targetModuleId = 'module:beta';
          const epochIndex = 6;

          // Act
          const ledgerEntry = buildProbeLedgerEntry(
            probeKind,
            targetModuleId,
            epochIndex,
            1.2,
            1.0,
          );

          // Assert
          expect({
            epochIndex: ledgerEntry.epochIndex,
            probeKind: ledgerEntry.probeKind,
            targetModuleId: ledgerEntry.targetModuleId,
          }).toEqual({ epochIndex, probeKind, targetModuleId });
        });
      });
    });

    describe('appendProbeLedgerEntry and computeProbeRewardDelta', () => {
      describe('given one new probe entry', () => {
        it('returns a new ledger whose length increases by one', () => {
          // Arrange
          const originalLedger = createProbeLedger([
            buildProbeLedgerEntry('lesion', 'module:alpha', 0, 1, 0.8),
          ]);
          const newEntry = buildProbeLedgerEntry(
            'noise',
            'module:alpha',
            10,
            0.8,
            0.6,
          );

          // Act
          const nextLedger = appendProbeLedgerEntry(
            originalLedger,
            newEntry,
            5,
          );

          // Assert
          expect(nextLedger.length).toBe(originalLedger.length + 1);
        });
      });

      describe('given one append operation', () => {
        it('does not mutate the original ledger reference', () => {
          // Arrange
          const originalLedger = createProbeLedger([
            buildProbeLedgerEntry('lesion', 'module:alpha', 0, 1, 0.8),
          ]);
          const originalSnapshot = JSON.parse(
            JSON.stringify(originalLedger),
          ) as NgeProbeLedgerEntry[];
          const newEntry = buildProbeLedgerEntry(
            'noise',
            'module:alpha',
            10,
            0.8,
            0.6,
          );

          // Act
          appendProbeLedgerEntry(originalLedger, newEntry, 5);

          // Assert
          expect(originalLedger).toEqual(originalSnapshot);
        });
      });

      describe('given a full ledger at capacity', () => {
        it('keeps the newest entries while evicting the oldest one', () => {
          // Arrange
          const originalLedger = createProbeLedger([
            buildProbeLedgerEntry('lesion', 'module:alpha', 0, 1, 0.8),
            buildProbeLedgerEntry('noise', 'module:alpha', 10, 0.8, 0.6),
          ]);
          const newEntry = buildProbeLedgerEntry(
            'gating',
            'module:alpha',
            20,
            0.6,
            0.7,
          );

          // Act
          const nextLedger = appendProbeLedgerEntry(
            originalLedger,
            newEntry,
            2,
          );

          // Assert
          expect(nextLedger).toEqual([originalLedger[1], newEntry]);
        });
      });

      describe('given an empty ledger', () => {
        it('returns zero reward delta for an unprobed module', () => {
          // Act
          const rewardDelta = computeProbeRewardDelta([], 'module:alpha');

          // Assert
          expect(rewardDelta).toBe(0);
        });
      });

      describe('given two matching module entries', () => {
        it('returns the mean signed reward delta', () => {
          // Arrange
          const probeLedger = createProbeLedger([
            buildProbeLedgerEntry('lesion', 'module:alpha', 0, 0.4, 0.6),
            buildProbeLedgerEntry('noise', 'module:alpha', 10, 0.6, 0.5),
          ]);

          // Act
          const rewardDelta = computeProbeRewardDelta(
            probeLedger,
            'module:alpha',
          );

          // Assert
          expect(rewardDelta).toBeCloseTo(0.05);
        });
      });

      describe('given entries for multiple modules', () => {
        it('ignores deltas that belong to other modules', () => {
          // Arrange
          const probeLedger = createProbeLedger([
            buildProbeLedgerEntry('lesion', 'module:alpha', 0, 0.4, 0.6),
            buildProbeLedgerEntry('noise', 'module:beta', 10, 0.6, 0.1),
          ]);

          // Act
          const rewardDelta = computeProbeRewardDelta(
            probeLedger,
            'module:alpha',
          );

          // Assert
          expect(rewardDelta).toBeCloseTo(0.2);
        });
      });
    });

    describe('serializeLedger / deserializeLedger', () => {
      describe('given one populated ledger', () => {
        it('round-trips every field value through JSON serialization', () => {
          // Arrange
          const probeLedger = createProbeLedger([
            buildProbeLedgerEntry('lesion', 'module:alpha', 0, 1, 0.8),
            buildProbeLedgerEntry('gating', 'module:beta', 10, 0.3, 0.5),
          ]);

          // Act
          const roundTrippedLedger = deserializeLedger(
            serializeLedger(probeLedger),
          );

          // Assert
          expect(roundTrippedLedger).toEqual(probeLedger);
        });
      });

      describe('given one non-array JSON payload', () => {
        it('throws a probe error', () => {
          // Act
          const deserializeNullLedger = () => deserializeLedger('null');

          // Assert
          expect(deserializeNullLedger).toThrow(NgeJuvenile_ProbeError);
        });
      });

      describe('given malformed JSON', () => {
        it('throws a probe error', () => {
          // Act
          const deserializeMalformedLedger = () => deserializeLedger('{');

          // Assert
          expect(deserializeMalformedLedger).toThrow(NgeJuvenile_ProbeError);
        });
      });
    });

    describe('resolveProbeSchedulerConfig / defaultProbeSchedulerState', () => {
      describe('given an empty scheduler config', () => {
        it('fills every probe default from the seeded constants', () => {
          // Act
          const resolvedConfig = resolveProbeSchedulerConfig({});

          // Assert
          expect(resolvedConfig).toEqual({
            cadenceEpochs: NGE_JUVENILE_DEFAULT_PROBE_CADENCE_EPOCHS,
            gatingEdgeLengthThreshold:
              NGE_JUVENILE_DEFAULT_GATING_EDGE_LENGTH_THRESHOLD,
            lesionSeverity: NGE_JUVENILE_DEFAULT_LESION_SEVERITY,
            maxLedgerEntries: NGE_JUVENILE_DEFAULT_PROBE_MAX_LEDGER_ENTRIES,
            noiseSigma: NGE_JUVENILE_DEFAULT_NOISE_SIGMA,
            probeKinds: [...NGE_JUVENILE_DEFAULT_PROBE_KINDS],
          });
        });
      });

      describe('given an empty probe kind list', () => {
        it('throws a probe error', () => {
          // Act
          const resolveEmptyProbeKinds = () =>
            resolveProbeSchedulerConfig({ probeKinds: [] });

          // Assert
          expect(resolveEmptyProbeKinds).toThrow(NgeJuvenile_ProbeError);
        });
      });

      describe('given the default scheduler state', () => {
        it('starts before the first probe with an empty ledger', () => {
          // Act
          const schedulerState = defaultProbeSchedulerState();

          // Assert
          expect(schedulerState).toEqual({
            lastProbeEpoch: -1,
            nextProbeKindIndex: 0,
            ledger: [],
          });
        });
      });

      describe('given one advanced scheduler state', () => {
        it('records the supplied epoch index as the latest probe epoch', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const schedulerState = defaultProbeSchedulerState();
          const ledgerEntry = buildProbeLedgerEntry(
            'lesion',
            'module:alpha',
            7,
            0.5,
            0.4,
          );

          // Act
          const nextState = advanceSchedulerState(
            schedulerState,
            7,
            ledgerEntry,
            schedulerConfig,
          );

          // Assert
          expect(nextState.lastProbeEpoch).toBe(7);
        });
      });

      describe('given one scheduler advance', () => {
        it('does not mutate the input scheduler state object', () => {
          // Arrange
          const schedulerConfig = createProbeSchedulerConfig();
          const initialState = defaultProbeSchedulerState();
          const originalStateSnapshot = JSON.parse(
            JSON.stringify(initialState),
          ) as NgeProbeSchedulerState;
          const ledgerEntry = buildProbeLedgerEntry(
            'lesion',
            'module:alpha',
            7,
            0.5,
            0.4,
          );

          // Act
          advanceSchedulerState(initialState, 7, ledgerEntry, schedulerConfig);

          // Assert
          expect(initialState).toEqual(originalStateSnapshot);
        });
      });

      describe('given one scheduler state payload', () => {
        it('round-trips through JSON without losing scheduler data', () => {
          // Arrange
          const schedulerState = createProbeSchedulerState({
            lastProbeEpoch: 12,
            nextProbeKindIndex: 1,
            ledger: createProbeLedger([
              buildProbeLedgerEntry('noise', 'module:alpha', 12, 0.5, 0.7),
            ]),
          });

          // Act
          const roundTrippedState = JSON.parse(
            JSON.stringify(schedulerState),
          ) as NgeProbeSchedulerState;

          // Assert
          expect(roundTrippedState).toEqual(schedulerState);
        });
      });
    });

    describe('NgeJuvenile_ProbeError', () => {
      describe('given a probe error with a cause', () => {
        it('keeps the cause field and the public error name', () => {
          // Arrange
          const cause = new Error('probe-root-cause');

          // Act
          const probeError = new NgeJuvenile_ProbeError('probe failed', {
            cause,
          });

          // Assert
          expect({ cause: probeError.cause, name: probeError.name }).toEqual({
            cause,
            name: 'NgeJuvenile_ProbeError',
          });
        });
      });
    });

    describe('local growth morphogenesis', () => {
      describe('canGrowNow', () => {
        describe('given a positive-focus streak below the hysteresis threshold', () => {
          it('returns false', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 1,
            });
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });

            // Act
            const shouldGrow = canGrowNow(hysteresisState, phaseConfig);

            // Assert
            expect(shouldGrow).toBe(false);
          });
        });

        describe('given a satisfied streak but an active cooldown', () => {
          it('returns false', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 1,
              growthPositiveWindowCount: 2,
            });
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });

            // Act
            const shouldGrow = canGrowNow(hysteresisState, phaseConfig);

            // Assert
            expect(shouldGrow).toBe(false);
          });
        });

        describe('given a satisfied streak and no cooldown', () => {
          it('returns true', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 0,
              growthPositiveWindowCount: 2,
            });
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });

            // Act
            const shouldGrow = canGrowNow(hysteresisState, phaseConfig);

            // Assert
            expect(shouldGrow).toBe(true);
          });
        });
      });

      describe('advanceGrowthHysteresis', () => {
        describe('given a positive-focus window', () => {
          it('increments the growth-positive streak', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 1,
            });

            // Act
            const advancedState = advanceGrowthHysteresis(
              hysteresisState,
              true,
            );

            // Assert
            expect(advancedState.growthPositiveWindowCount).toBe(2);
          });
        });

        describe('given a non-positive-focus window', () => {
          it('resets the growth-positive streak to zero', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 3,
            });

            // Act
            const advancedState = advanceGrowthHysteresis(
              hysteresisState,
              false,
            );

            // Assert
            expect(advancedState.growthPositiveWindowCount).toBe(0);
          });
        });

        describe('given an active cooldown', () => {
          it('decrements the cooldown by one', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 2,
            });

            // Act
            const advancedState = advanceGrowthHysteresis(
              hysteresisState,
              true,
            );

            // Assert
            expect(advancedState.cooldownWindowsRemaining).toBe(1);
          });
        });

        describe('given a cooldown already at zero', () => {
          it('keeps the cooldown floored at zero', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 0,
            });

            // Act
            const advancedState = advanceGrowthHysteresis(
              hysteresisState,
              true,
            );

            // Assert
            expect(advancedState.cooldownWindowsRemaining).toBe(0);
          });
        });

        describe('given one advanced state', () => {
          it('returns a new object without mutating the input state', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 1,
              growthPositiveWindowCount: 1,
            });
            const originalStateSnapshot = structuredClone(hysteresisState);

            // Act
            const advancedState = advanceGrowthHysteresis(
              hysteresisState,
              true,
            );

            // Assert
            expect({
              inputState: hysteresisState,
              sameReference: advancedState === hysteresisState,
            }).toEqual({
              inputState: originalStateSnapshot,
              sameReference: false,
            });
          });
        });
      });

      describe('commitGrowth', () => {
        describe('given one committed node-add morph', () => {
          it('stores the committed morph kind', () => {
            // Arrange
            const hysteresisState = createHysteresisState();
            const phaseConfig = createResolvedConfig({
              cooldownWindowCount: 3,
            });

            // Act
            const committedState = commitGrowth(
              hysteresisState,
              'nodeAdd',
              phaseConfig,
            );

            // Assert
            expect(committedState.lastMorphKind).toBe('nodeAdd');
          });
        });

        describe('given one committed edge-densification morph', () => {
          it('resets the cooldown from the phase config', () => {
            // Arrange
            const hysteresisState = createHysteresisState();
            const phaseConfig = createResolvedConfig({
              cooldownWindowCount: 4,
            });

            // Act
            const committedState = commitGrowth(
              hysteresisState,
              'edgeDensify',
              phaseConfig,
            );

            // Assert
            expect(committedState.cooldownWindowsRemaining).toBe(4);
          });
        });

        describe('given one committed slot-expansion morph', () => {
          it('resets the growth-positive streak to zero', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 3,
            });
            const phaseConfig = createResolvedConfig();

            // Act
            const committedState = commitGrowth(
              hysteresisState,
              'slotExpand',
              phaseConfig,
            );

            // Assert
            expect(committedState.growthPositiveWindowCount).toBe(0);
          });
        });

        describe('given one committed growth state', () => {
          it('returns a new object without mutating the input state', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });
            const originalStateSnapshot = structuredClone(hysteresisState);
            const phaseConfig = createResolvedConfig({
              cooldownWindowCount: 2,
            });

            // Act
            const committedState = commitGrowth(
              hysteresisState,
              'edgeDensify',
              phaseConfig,
            );

            // Assert
            expect({
              inputState: hysteresisState,
              sameReference: committedState === hysteresisState,
            }).toEqual({
              inputState: originalStateSnapshot,
              sameReference: false,
            });
          });
        });
      });

      describe('planEdgeDensification', () => {
        describe('given a module that remains under the edge budget', () => {
          it('returns an edge-densify morph delta', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore();
            const config = createResolvedConfig();

            // Act
            const morphDelta = planEdgeDensification(
              'module:alpha',
              growthBudget,
              focusScore,
              config,
            );

            // Assert
            expect(morphDelta.kind).toBe('edgeDensify');
          });
        });

        describe('given a valid densification plan', () => {
          it('uses the seeded wiring-cost increment', () => {
            // Arrange
            const growthBudget = createGrowthBudget({ maxEdges: 10 });
            const focusScore = createFocusScore();
            const config = createResolvedConfig({
              edgeDensificationCount:
                NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
            });

            // Act
            const morphDelta = planEdgeDensification(
              'module:alpha',
              growthBudget,
              focusScore,
              config,
            );

            // Assert
            expect(morphDelta.wiringCostDelta).toBe(
              NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
            );
          });
        });

        describe('given a valid densification plan', () => {
          it('echoes the current edge count in the detail packet', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentEdgeCount: 3,
              maxEdges: 10,
            });
            const focusScore = createFocusScore({ normalizedScore: 0.9 });
            const config = createResolvedConfig({
              edgeDensificationCount:
                NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
            });

            // Act
            const morphDelta = planEdgeDensification(
              'module:alpha',
              growthBudget,
              focusScore,
              config,
            );

            // Assert
            expect(morphDelta.detail).toEqual({
              currentEdgeCount: 3,
              proposedAdditions: NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
              normalizedFocusScore: 0.9,
            });
          });
        });

        describe('given a module already at the edge cap', () => {
          it('throws a budget error', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentEdgeCount: 5,
              maxEdges: 5,
            });
            const focusScore = createFocusScore();
            const config = createResolvedConfig();

            // Act
            const planOverBudgetEdgeDensification = () =>
              planEdgeDensification(
                'module:alpha',
                growthBudget,
                focusScore,
                config,
              );

            // Assert
            expect(planOverBudgetEdgeDensification).toThrow(
              NgeJuvenile_BudgetError,
            );
          });
        });
      });

      describe('planSlotExpansion', () => {
        describe('given positive focus, sufficient hit rate, and remaining slot budget', () => {
          it('returns a slot-expand morph delta', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore({ normalizedScore: 0.6 });
            const phaseConfig = createResolvedConfig();

            // Act
            const morphDelta = planSlotExpansion(
              'module:alpha',
              0.8,
              growthBudget,
              focusScore,
              phaseConfig,
            );

            // Assert
            expect(morphDelta.kind).toBe('slotExpand');
          });
        });

        describe('given one valid slot expansion plan', () => {
          it('uses the seeded slot increment as the wiring-cost delta', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore({ normalizedScore: 0.6 });
            const phaseConfig = createResolvedConfig();

            // Act
            const morphDelta = planSlotExpansion(
              'module:alpha',
              0.8,
              growthBudget,
              focusScore,
              phaseConfig,
            );

            // Assert
            expect(morphDelta.wiringCostDelta).toBe(
              NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT,
            );
          });
        });

        describe('given a hit rate below the configured threshold', () => {
          it('throws a morph error', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore({ normalizedScore: 0.6 });
            const phaseConfig = createResolvedConfig({
              episodicHitRateThreshold: 0.65,
            });

            // Act
            const planUnderThresholdSlotExpansion = () =>
              planSlotExpansion(
                'module:alpha',
                0.5,
                growthBudget,
                focusScore,
                phaseConfig,
              );

            // Assert
            expect(planUnderThresholdSlotExpansion).toThrow(
              NgeJuvenile_MorphError,
            );
          });
        });

        describe('given a non-positive normalized focus score', () => {
          it('throws a morph error', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore({ normalizedScore: 0 });
            const phaseConfig = createResolvedConfig();

            // Act
            const planWithoutPositiveFocus = () =>
              planSlotExpansion(
                'module:alpha',
                0.8,
                growthBudget,
                focusScore,
                phaseConfig,
              );

            // Assert
            expect(planWithoutPositiveFocus).toThrow(NgeJuvenile_MorphError);
          });
        });

        describe('given an exhausted slot budget', () => {
          it('throws a budget error after the guards pass', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentEpisodicSlotCount: 5,
              maxEpisodicSlots: 5,
            });
            const focusScore = createFocusScore({ normalizedScore: 0.6 });
            const phaseConfig = createResolvedConfig();

            // Act
            const planOverBudgetSlotExpansion = () =>
              planSlotExpansion(
                'module:alpha',
                0.8,
                growthBudget,
                focusScore,
                phaseConfig,
              );

            // Assert
            expect(planOverBudgetSlotExpansion).toThrow(
              NgeJuvenile_BudgetError,
            );
          });
        });
      });

      describe('planNodeAddition', () => {
        describe('given positive reward evidence and remaining node budget', () => {
          it('returns a node-add morph delta', () => {
            // Arrange
            const growthBudget = createGrowthBudget();

            // Act
            const focusScore = createFocusScore({ supportsGrowth: true });
            const config = createResolvedConfig();

            // Act
            const morphDelta = planNodeAddition(
              'module:alpha',
              growthBudget,
              focusScore,
              config,
            );

            // Assert
            expect(morphDelta.kind).toBe('nodeAdd');
          });
        });

        describe('given one valid node-addition plan', () => {
          it('keeps the wiring-cost delta at zero', () => {
            // Arrange
            const growthBudget = createGrowthBudget();

            // Act
            const focusScore = createFocusScore({ supportsGrowth: true });
            const config = createResolvedConfig();

            // Act
            const morphDelta = planNodeAddition(
              'module:alpha',
              growthBudget,
              focusScore,
              config,
            );

            // Assert
            expect(morphDelta.wiringCostDelta).toBe(0);
          });
        });

        describe('given a non-growth focus score', () => {
          it('throws a morph error', () => {
            // Arrange
            const growthBudget = createGrowthBudget();

            // Act
            const planWithoutPositiveEvidence = () =>
              planNodeAddition(
                'module:alpha',
                growthBudget,
                createFocusScore({ supportsGrowth: false, rawScore: -0.1 }),
                createResolvedConfig(),
              );

            // Assert
            expect(planWithoutPositiveEvidence).toThrow(NgeJuvenile_MorphError);
          });
        });

        describe('given a focus score whose composite growth signal is at the floor', () => {
          it('throws a morph error', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore({
              normalizedNovelty: 0,
              normalizedRewardDelta: 0,
              normalizedStabilityAge: 0,
              normalizedUtilization: 0,
              normalizedWiringCost: 1,
              supportsGrowth: true,
            });
            const config = createResolvedConfig();

            // Act
            const planWithSignalAtFloor = () =>
              planNodeAddition(
                'module:alpha',
                growthBudget,
                focusScore,
                config,
              );

            // Assert
            expect(planWithSignalAtFloor).toThrow(NgeJuvenile_MorphError);
          });
        });

        describe('given a config with no explicit nodeGrowthSignalFloor', () => {
          it('falls back to the default floor and throws for a weak signal', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const focusScore = createFocusScore({
              normalizedNovelty: 0,
              normalizedRewardDelta: 0,
              normalizedStabilityAge: 0,
              normalizedUtilization: 0,
              normalizedWiringCost: 1,
              supportsGrowth: true,
            });
            const configWithoutFloor = {
              ...createResolvedConfig(),
              nodeGrowthSignalFloor: undefined,
            } as unknown as NgeJuvenilePhaseConfig;

            // Act
            const planWithDefaultFloor = () =>
              planNodeAddition(
                'module:alpha',
                growthBudget,
                focusScore,
                configWithoutFloor,
              );

            // Assert
            expect(planWithDefaultFloor).toThrow(NgeJuvenile_MorphError);
          });
        });

        describe('given an exhausted node budget', () => {
          it('throws a budget error', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentNodeCount: 5,
              maxNodes: 5,
            });

            // Act
            const focusScore = createFocusScore({ supportsGrowth: true });
            const config = createResolvedConfig();

            // Act
            const planOverBudgetNodeAddition = () =>
              planNodeAddition(
                'module:alpha',
                growthBudget,
                focusScore,
                config,
              );

            // Assert
            expect(planOverBudgetNodeAddition).toThrow(NgeJuvenile_BudgetError);
          });
        });
      });

      describe('validateMorphDelta', () => {
        describe('given an edge-densify delta that stays within budget', () => {
          it('passes without throwing', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const morphDelta = createMorphDelta({
              kind: 'edgeDensify',
              wiringCostDelta: 1,
            });

            // Act
            const validateUnderBudgetEdgeDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateUnderBudgetEdgeDelta).not.toThrow();
          });
        });

        describe('given an edge-densify delta that exceeds the edge cap', () => {
          it('throws a budget error', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentEdgeCount: 5,
              maxEdges: 5,
            });
            const morphDelta = createMorphDelta({
              kind: 'edgeDensify',
              wiringCostDelta: 1,
            });

            // Act
            const validateOverBudgetEdgeDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateOverBudgetEdgeDelta).toThrow(
              NgeJuvenile_BudgetError,
            );
          });
        });

        describe('given a slot-expand delta that stays within budget', () => {
          it('passes without throwing', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const morphDelta = createMorphDelta({
              kind: 'slotExpand',
              wiringCostDelta: 1,
            });

            // Act
            const validateUnderBudgetSlotDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateUnderBudgetSlotDelta).not.toThrow();
          });
        });

        describe('given a slot-expand delta that exceeds the slot cap', () => {
          it('throws a budget error', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentEpisodicSlotCount: 5,
              maxEpisodicSlots: 5,
            });
            const morphDelta = createMorphDelta({
              kind: 'slotExpand',
              wiringCostDelta: 1,
            });

            // Act
            const validateOverBudgetSlotDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateOverBudgetSlotDelta).toThrow(
              NgeJuvenile_BudgetError,
            );
          });
        });

        describe('given a node-add delta that stays within budget', () => {
          it('passes without throwing', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const morphDelta = createMorphDelta({
              kind: 'nodeAdd',
              wiringCostDelta: 0,
            });

            // Act
            const validateUnderBudgetNodeDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateUnderBudgetNodeDelta).not.toThrow();
          });
        });

        describe('given a node-add delta that exceeds the node cap', () => {
          it('throws a budget error', () => {
            // Arrange
            const growthBudget = createGrowthBudget({
              currentNodeCount: 5,
              maxNodes: 5,
            });
            const morphDelta = createMorphDelta({
              kind: 'nodeAdd',
              wiringCostDelta: 0,
            });

            // Act
            const validateOverBudgetNodeDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateOverBudgetNodeDelta).toThrow(
              NgeJuvenile_BudgetError,
            );
          });
        });

        describe('given an edge-prune delta', () => {
          it('acts as a no-op in the Step 04 validator', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const morphDelta = createMorphDelta({
              kind: 'edgePrune',
              wiringCostDelta: -1,
            });

            // Act
            const validateEdgePruneDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateEdgePruneDelta).not.toThrow();
          });
        });

        describe('given a compact delta', () => {
          it('acts as a no-op in the Step 04 validator', () => {
            // Arrange
            const growthBudget = createGrowthBudget();
            const morphDelta = createMorphDelta({
              kind: 'compact',
              wiringCostDelta: -1,
            });

            // Act
            const validateCompactDelta = () =>
              validateMorphDelta(morphDelta, growthBudget);

            // Assert
            expect(validateCompactDelta).not.toThrow();
          });
        });
      });

      describe('planGrowthMorphs', () => {
        describe('given a hysteresis state that is not yet eligible to grow', () => {
          it('returns an empty delta list', () => {
            // Arrange
            const focusScore = createFocusScore();
            const metricsSnapshot = createGrowthMetrics();
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 1,
            });

            // Act
            const plannedDeltas = planGrowthMorphs(
              'module:alpha',
              focusScore,
              metricsSnapshot,
              growthBudget,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas).toEqual([]);
          });
        });

        describe('given all three growth actions are eligible', () => {
          it('returns edge densify before slot expand before node add', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0.4,
              utilization: 0.8,
            });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });

            // Act
            const plannedDeltas = planGrowthMorphs(
              'module:alpha',
              focusScore,
              metricsSnapshot,
              growthBudget,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual([
              'edgeDensify',
              'slotExpand',
              'nodeAdd',
            ]);
          });
        });

        describe('given the edge budget is exhausted but slot and node budgets remain', () => {
          it('silently omits the edge densification delta', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0.4,
              utilization: 0.8,
            });
            const growthBudget = createGrowthBudget({
              currentEdgeCount: 5,
              maxEdges: 5,
            });
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });

            // Act
            const plannedDeltas = planGrowthMorphs(
              'module:alpha',
              focusScore,
              metricsSnapshot,
              growthBudget,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual([
              'slotExpand',
              'nodeAdd',
            ]);
          });
        });

        describe('given the utilization proxy stays below the hit-rate threshold', () => {
          it('silently omits the slot expansion delta', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0.4,
              utilization: 0.4,
            });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig({
              episodicHitRateThreshold: 0.65,
            });
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });

            // Act
            const plannedDeltas = planGrowthMorphs(
              'module:alpha',
              focusScore,
              metricsSnapshot,
              growthBudget,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual([
              'edgeDensify',
              'nodeAdd',
            ]);
          });
        });

        describe('given reward evidence that is not positive', () => {
          it('silently omits the node-add delta', () => {
            // Arrange
            const focusScore = createFocusScore({
              normalizedScore: 0.8,
              supportsGrowth: false,
            });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0,
              utilization: 0.8,
            });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });

            // Act
            const plannedDeltas = planGrowthMorphs(
              'module:alpha',
              focusScore,
              metricsSnapshot,
              growthBudget,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual([
              'edgeDensify',
              'slotExpand',
            ]);
          });
        });

        describe('given one dry-run planning pass', () => {
          it('does not mutate any of the input planning packets', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0.4,
              utilization: 0.8,
            });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });
            const originalPackets = structuredClone({
              focusScore,
              growthBudget,
              hysteresisState,
              metricsSnapshot,
            });

            // Act
            planGrowthMorphs(
              'module:alpha',
              focusScore,
              metricsSnapshot,
              growthBudget,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect({
              focusScore,
              growthBudget,
              hysteresisState,
              metricsSnapshot,
            }).toEqual(originalPackets);
          });
        });

        describe('given an unexpected edge-planning failure', () => {
          it('rethrows the unexpected error to the caller', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0.4,
              utilization: 0.8,
            });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });
            Object.defineProperty(growthBudget, 'currentEdgeCount', {
              get() {
                throw new Error('unexpected-edge-failure');
              },
            });

            // Act
            const planWithUnexpectedEdgeFailure = () =>
              planGrowthMorphs(
                'module:alpha',
                focusScore,
                metricsSnapshot,
                growthBudget,
                phaseConfig,
                hysteresisState,
              );

            // Assert
            expect(planWithUnexpectedEdgeFailure).toThrow(
              'unexpected-edge-failure',
            );
          });
        });

        describe('given an unexpected slot-planning failure', () => {
          it('rethrows the unexpected error to the caller', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({
              rewardDelta: 0.4,
              utilization: 0.8,
            });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });
            Object.defineProperty(phaseConfig, 'episodicHitRateThreshold', {
              get() {
                throw new Error('unexpected-slot-failure');
              },
            });

            // Act
            const planWithUnexpectedSlotFailure = () =>
              planGrowthMorphs(
                'module:alpha',
                focusScore,
                metricsSnapshot,
                growthBudget,
                phaseConfig,
                hysteresisState,
              );

            // Assert
            expect(planWithUnexpectedSlotFailure).toThrow(
              'unexpected-slot-failure',
            );
          });
        });

        describe('given an unexpected node-planning failure', () => {
          it('rethrows the unexpected error to the caller', () => {
            // Arrange
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            const metricsSnapshot = createGrowthMetrics({ utilization: 0.8 });
            const growthBudget = createGrowthBudget();
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 2,
            });
            Object.defineProperty(focusScore, 'normalizedUtilization', {
              get() {
                throw new Error('unexpected-node-failure');
              },
            });

            // Act
            const planWithUnexpectedNodeFailure = () =>
              planGrowthMorphs(
                'module:alpha',
                focusScore,
                metricsSnapshot,
                growthBudget,
                phaseConfig,
                hysteresisState,
              );

            // Assert
            expect(planWithUnexpectedNodeFailure).toThrow(
              'unexpected-node-failure',
            );
          });
        });
      });

      describe('applyMorphDeltas nodeAdd truthfulness', () => {
        it('reports a node-add delta with zero planned additions as skipped', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 42 });
          const initialNodeCount = network.nodes.length;
          const deltas = [
            {
              kind: 'nodeAdd' as const,
              targetModuleId: 'module:alpha',
              detail: {
                currentNodeCount: network.nodes.length,
                growthSignal: 0.5,
                proposedAdditions: 0,
              },
              wiringCostDelta: 0,
            },
          ];

          // Act
          const outcomes = applyMorphDeltas(network, deltas, {
            growth: {
              currentEdgeCount: network.connections.length,
              currentEpisodicSlotCount: 0,
              currentNodeCount: network.nodes.length,
              maxEdges: 100,
              maxEpisodicSlots: 0,
              maxNodes: 100,
            },
            prune: {
              costExemptEdgeIds: [],
              currentEdgeCount: network.connections.length,
              currentNodeCount: network.nodes.length,
              currentWiringCost:
                network.nodes.length + network.connections.length,
              minEdges: 0,
              minNodes: 1,
            },
          });

          // Assert
          const nodeOutcome = outcomes.find(
            (outcome) => outcome.kind === 'nodeAdd',
          );
          const structuralChange = {
            nodes: network.nodes.length - initialNodeCount,
            reportedStatus: nodeOutcome?.status,
            reportedReason: nodeOutcome?.reason,
          };

          expect(structuralChange).toEqual({
            nodes: 0,
            reportedStatus: 'skipped',
            reportedReason: 'ADD_NODE produced no net hidden nodes.',
          });
        });
      });

      describe('applyMorphDeltas edgeDensify fallback', () => {
        it('defaults to one addition when proposedAdditions is omitted', () => {
          // Arrange
          const network = new Network(2, 1, { seed: 42 });
          const deltas = [
            {
              kind: 'edgeDensify' as const,
              targetModuleId: 'module:alpha',
              detail: {
                currentEdgeCount: network.connections.length,
                normalizedFocusScore: 0.5,
              },
              wiringCostDelta: 1,
            } as NgeMorphDelta,
          ];

          // Act
          const outcomes = applyMorphDeltas(network, deltas, {
            growth: {
              currentEdgeCount: network.connections.length,
              currentEpisodicSlotCount: 0,
              currentNodeCount: network.nodes.length,
              maxEdges: 100,
              maxEpisodicSlots: 0,
              maxNodes: 100,
            },
            prune: {
              costExemptEdgeIds: [],
              currentEdgeCount: network.connections.length,
              currentNodeCount: network.nodes.length,
              currentWiringCost:
                network.nodes.length + network.connections.length,
              minEdges: 0,
              minNodes: 1,
            },
          });

          // Assert
          expect(outcomes.length).toBe(1);
        });
      });
    });

    describe('local prune/compact morphogenesis', () => {
      describe('canPruneNow', () => {
        describe('given an underuse streak below the hysteresis threshold', () => {
          it('returns false', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 1,
            });
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });

            // Act
            const shouldPrune = canPruneNow(hysteresisState, phaseConfig);

            // Assert
            expect(shouldPrune).toBe(false);
          });
        });

        describe('given a satisfied streak but an active cooldown', () => {
          it('returns false', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 1,
              pruneUnderuseWindowCount: 2,
            });
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });

            // Act
            const shouldPrune = canPruneNow(hysteresisState, phaseConfig);

            // Assert
            expect(shouldPrune).toBe(false);
          });
        });

        describe('given a satisfied streak and no cooldown', () => {
          it('returns true', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 0,
              pruneUnderuseWindowCount: 2,
            });
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });

            // Act
            const shouldPrune = canPruneNow(hysteresisState, phaseConfig);

            // Assert
            expect(shouldPrune).toBe(true);
          });
        });
      });

      describe('advancePruneHysteresis', () => {
        describe('given an underuse window', () => {
          it('increments the prune-underuse streak', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 1,
            });

            // Act
            const advancedState = advancePruneHysteresis(hysteresisState, true);

            // Assert
            expect(advancedState.pruneUnderuseWindowCount).toBe(2);
          });
        });

        describe('given a non-underuse window', () => {
          it('resets the prune-underuse streak to zero', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 3,
            });

            // Act
            const advancedState = advancePruneHysteresis(
              hysteresisState,
              false,
            );

            // Assert
            expect(advancedState.pruneUnderuseWindowCount).toBe(0);
          });
        });

        describe('given an active cooldown', () => {
          it('decrements the cooldown by one', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 2,
            });

            // Act
            const advancedState = advancePruneHysteresis(hysteresisState, true);

            // Assert
            expect(advancedState.cooldownWindowsRemaining).toBe(1);
          });
        });

        describe('given a cooldown already at zero', () => {
          it('keeps the cooldown floored at zero', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 0,
            });

            // Act
            const advancedState = advancePruneHysteresis(hysteresisState, true);

            // Assert
            expect(advancedState.cooldownWindowsRemaining).toBe(0);
          });
        });

        describe('given one advanced prune state', () => {
          it('returns a new object without mutating the input state', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              cooldownWindowsRemaining: 1,
              pruneUnderuseWindowCount: 1,
            });
            const originalStateSnapshot = structuredClone(hysteresisState);

            // Act
            const advancedState = advancePruneHysteresis(hysteresisState, true);

            // Assert
            expect({
              inputState: hysteresisState,
              sameReference: advancedState === hysteresisState,
            }).toEqual({
              inputState: originalStateSnapshot,
              sameReference: false,
            });
          });
        });

        describe('given either prune path', () => {
          it('carries the growth-positive streak through unchanged', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 4,
              pruneUnderuseWindowCount: 1,
            });

            // Act
            const advancedState = advancePruneHysteresis(
              hysteresisState,
              false,
            );

            // Assert
            expect(advancedState.growthPositiveWindowCount).toBe(4);
          });
        });
      });

      describe('commitPrune', () => {
        describe('given one committed edge-prune morph', () => {
          it('stores the committed morph kind', () => {
            // Arrange
            const hysteresisState = createHysteresisState();
            const phaseConfig = createResolvedConfig({
              cooldownWindowCount: 3,
            });

            // Act
            const committedState = commitPrune(
              hysteresisState,
              'edgePrune',
              phaseConfig,
            );

            // Assert
            expect(committedState.lastMorphKind).toBe('edgePrune');
          });
        });

        describe('given one committed compact morph', () => {
          it('resets the cooldown from the phase config', () => {
            // Arrange
            const hysteresisState = createHysteresisState();
            const phaseConfig = createResolvedConfig({
              cooldownWindowCount: 4,
            });

            // Act
            const committedState = commitPrune(
              hysteresisState,
              'compact',
              phaseConfig,
            );

            // Assert
            expect(committedState.cooldownWindowsRemaining).toBe(4);
          });
        });

        describe('given one committed prune state', () => {
          it('resets the prune-underuse streak to zero', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 3,
            });
            const phaseConfig = createResolvedConfig();

            // Act
            const committedState = commitPrune(
              hysteresisState,
              'compact',
              phaseConfig,
            );

            // Assert
            expect(committedState.pruneUnderuseWindowCount).toBe(0);
          });
        });

        describe('given one committed prune path', () => {
          it('carries the growth-positive streak through unchanged', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              growthPositiveWindowCount: 5,
            });
            const phaseConfig = createResolvedConfig();

            // Act
            const committedState = commitPrune(
              hysteresisState,
              'compact',
              phaseConfig,
            );

            // Assert
            expect(committedState.growthPositiveWindowCount).toBe(5);
          });
        });

        describe('given one committed prune state', () => {
          it('returns a new object without mutating the input state', () => {
            // Arrange
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });
            const originalStateSnapshot = structuredClone(hysteresisState);
            const phaseConfig = createResolvedConfig({
              cooldownWindowCount: 2,
            });

            // Act
            const committedState = commitPrune(
              hysteresisState,
              'edgePrune',
              phaseConfig,
            );

            // Assert
            expect({
              inputState: hysteresisState,
              sameReference: committedState === hysteresisState,
            }).toEqual({
              inputState: originalStateSnapshot,
              sameReference: false,
            });
          });
        });
      });

      describe('selectPruneCandidate', () => {
        describe('given a mixed candidate list', () => {
          it('returns the highest-wiring-cost non-exempt candidate', () => {
            // Arrange
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({ candidateId: 'edge:low', wiringCost: 2 }),
              createPruneCandidate({ candidateId: 'edge:high', wiringCost: 5 }),
            ]);
            const pruneBudget = createPruneBudget();

            // Act
            const selectedCandidate = selectPruneCandidate(
              pruneCandidates,
              pruneBudget,
            );

            // Assert
            expect(selectedCandidate.candidateId).toBe('edge:high');
          });
        });

        describe('given equal wiring cost candidates', () => {
          it('tiebreaks by descending edge length', () => {
            // Arrange
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:short',
                edgeLength: 1,
              }),
              createPruneCandidate({ candidateId: 'edge:long', edgeLength: 3 }),
            ]);
            const pruneBudget = createPruneBudget();

            // Act
            const selectedCandidate = selectPruneCandidate(
              pruneCandidates,
              pruneBudget,
            );

            // Assert
            expect(selectedCandidate.candidateId).toBe('edge:long');
          });
        });

        describe('given equal wiring cost and equal edge length candidates', () => {
          it('tiebreaks lexicographically by candidate id', () => {
            // Arrange
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({ candidateId: 'edge:beta', edgeLength: 2 }),
              createPruneCandidate({
                candidateId: 'edge:alpha',
                edgeLength: 2,
              }),
            ]);
            const pruneBudget = createPruneBudget();

            // Act
            const selectedCandidate = selectPruneCandidate(
              pruneCandidates,
              pruneBudget,
            );

            // Assert
            expect(selectedCandidate.candidateId).toBe('edge:alpha');
          });
        });

        describe('given an exempt highest-cost candidate', () => {
          it('skips the exempt candidate and selects the next eligible one', () => {
            // Arrange
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:exempt',
                wiringCost: 9,
              }),
              createPruneCandidate({
                candidateId: 'edge:eligible',
                wiringCost: 4,
              }),
            ]);
            const pruneBudget = createPruneBudget({
              costExemptEdgeIds: ['edge:exempt'],
            });

            // Act
            const selectedCandidate = selectPruneCandidate(
              pruneCandidates,
              pruneBudget,
            );

            // Assert
            expect(selectedCandidate.candidateId).toBe('edge:eligible');
          });
        });

        describe('given all candidates are exempt', () => {
          it('throws a morph error', () => {
            // Arrange
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({ candidateId: 'edge:exempt-a' }),
              createPruneCandidate({ candidateId: 'edge:exempt-b' }),
            ]);
            const pruneBudget = createPruneBudget({
              costExemptEdgeIds: ['edge:exempt-a', 'edge:exempt-b'],
            });

            // Act
            const selectFromExemptOnly = () =>
              selectPruneCandidate(pruneCandidates, pruneBudget);

            // Assert
            expect(selectFromExemptOnly).toThrow(NgeJuvenile_MorphError);
          });
        });

        describe('given one selection pass', () => {
          it('does not mutate the input candidate array', () => {
            // Arrange
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:gamma',
                wiringCost: 3,
              }),
              createPruneCandidate({
                candidateId: 'edge:alpha',
                wiringCost: 4,
              }),
            ]);
            const originalCandidates = structuredClone(pruneCandidates);
            const pruneBudget = createPruneBudget();

            // Act
            selectPruneCandidate(pruneCandidates, pruneBudget);

            // Assert
            expect(pruneCandidates).toEqual(originalCandidates);
          });
        });
      });

      describe('planEdgePrune', () => {
        describe('given a valid non-exempt candidate above the edge floor', () => {
          it('returns an edge-prune morph delta', () => {
            // Arrange
            const pruneCandidate = createPruneCandidate();
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 3,
              minEdges: 1,
            });

            // Act
            const morphDelta = planEdgePrune(
              'module:alpha',
              pruneCandidate,
              pruneBudget,
            );

            // Assert
            expect(morphDelta.kind).toBe('edgePrune');
          });
        });

        describe('given one valid edge prune plan', () => {
          it('uses the negated candidate wiring cost as the delta', () => {
            // Arrange
            const pruneCandidate = createPruneCandidate({ wiringCost: 2 });
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 4,
              minEdges: 1,
            });

            // Act
            const morphDelta = planEdgePrune(
              'module:alpha',
              pruneCandidate,
              pruneBudget,
            );

            // Assert
            expect(morphDelta.wiringCostDelta).toBe(-2);
          });
        });

        describe('given one valid edge prune plan', () => {
          it('echoes the candidate id in the detail packet', () => {
            // Arrange
            const pruneCandidate = createPruneCandidate({
              candidateId: 'edge:beta',
            });
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 4,
              minEdges: 1,
            });

            // Act
            const morphDelta = planEdgePrune(
              'module:alpha',
              pruneCandidate,
              pruneBudget,
            );

            // Assert
            expect(morphDelta.detail).toEqual({
              candidateId: 'edge:beta',
              currentEdgeCount: 4,
              edgeLength: 2,
              wiringCost: 1,
            });
          });
        });

        describe('given a cost-exempt candidate', () => {
          it('throws a morph error before the floor guard runs', () => {
            // Arrange
            const pruneCandidate = createPruneCandidate({
              candidateId: 'edge:exempt',
            });
            const pruneBudget = createPruneBudget({
              costExemptEdgeIds: ['edge:exempt'],
              currentEdgeCount: 1,
              minEdges: 1,
            });

            // Act
            const planExemptCandidate = () =>
              planEdgePrune('module:alpha', pruneCandidate, pruneBudget);

            // Assert
            expect(planExemptCandidate).toThrow(NgeJuvenile_MorphError);
          });
        });

        describe('given an edge count already at the floor', () => {
          it('throws a budget error', () => {
            // Arrange
            const pruneCandidate = createPruneCandidate();
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 1,
              minEdges: 1,
            });

            // Act
            const planAtFloorEdgePrune = () =>
              planEdgePrune('module:alpha', pruneCandidate, pruneBudget);

            // Assert
            expect(planAtFloorEdgePrune).toThrow(NgeJuvenile_BudgetError);
          });
        });
      });

      describe('planCompact', () => {
        describe('given a node count above the floor', () => {
          it('returns a compact morph delta', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentNodeCount: 3,
              minNodes: 1,
            });

            // Act
            const morphDelta = planCompact('module:alpha', pruneBudget);

            // Assert
            expect(morphDelta.kind).toBe('compact');
          });
        });

        describe('given one valid compact plan', () => {
          it('keeps the wiring-cost delta at zero', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentNodeCount: 3,
              minNodes: 1,
            });

            // Act
            const morphDelta = planCompact('module:alpha', pruneBudget);

            // Assert
            expect(morphDelta.wiringCostDelta).toBe(0);
          });
        });

        describe('given one valid compact plan', () => {
          it('echoes the current node count in the detail packet', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentNodeCount: 4,
              currentWiringCost: 6,
              minNodes: 1,
            });

            // Act
            const morphDelta = planCompact('module:alpha', pruneBudget);

            // Assert
            expect(morphDelta.detail).toEqual({
              currentNodeCount: 4,
              currentWiringCost: 6,
            });
          });
        });

        describe('given a node count already at the floor', () => {
          it('throws a budget error', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentNodeCount: 1,
              minNodes: 1,
            });

            // Act
            const planAtFloorCompact = () =>
              planCompact('module:alpha', pruneBudget);

            // Assert
            expect(planAtFloorCompact).toThrow(NgeJuvenile_BudgetError);
          });
        });
      });

      describe('validatePruneDelta', () => {
        describe('given an edge-prune delta that stays above the floor', () => {
          it('passes without throwing', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 3,
              minEdges: 1,
            });
            const morphDelta = createMorphDelta({
              kind: 'edgePrune',
              wiringCostDelta: -1,
            });

            // Act
            const validateUnderFloorEdgePrune = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateUnderFloorEdgePrune).not.toThrow();
          });
        });

        describe('given an edge-prune delta that would drop below the floor', () => {
          it('throws a budget error', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 2,
              minEdges: 1,
            });
            const morphDelta = createMorphDelta({
              kind: 'edgePrune',
              wiringCostDelta: -2,
            });

            // Act
            const validateOverFloorEdgePrune = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateOverFloorEdgePrune).toThrow(NgeJuvenile_BudgetError);
          });
        });

        describe('given a compact delta above the node floor', () => {
          it('passes without throwing', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentNodeCount: 3,
              minNodes: 1,
            });
            const morphDelta = createMorphDelta({
              kind: 'compact',
              wiringCostDelta: 0,
            });

            // Act
            const validateUnderFloorCompact = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateUnderFloorCompact).not.toThrow();
          });
        });

        describe('given a compact delta at the node floor', () => {
          it('throws a budget error', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentNodeCount: 1,
              minNodes: 1,
            });
            const morphDelta = createMorphDelta({
              kind: 'compact',
              wiringCostDelta: 0,
            });

            // Act
            const validateAtFloorCompact = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateAtFloorCompact).toThrow(NgeJuvenile_BudgetError);
          });
        });

        describe('given an edge-densify delta', () => {
          it('acts as a no-op in the Step 05 validator', () => {
            // Arrange
            const pruneBudget = createPruneBudget();
            const morphDelta = createMorphDelta({
              kind: 'edgeDensify',
              wiringCostDelta: 1,
            });

            // Act
            const validateGrowthDelta = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateGrowthDelta).not.toThrow();
          });
        });

        describe('given a slot-expand delta', () => {
          it('acts as a no-op in the Step 05 validator', () => {
            // Arrange
            const pruneBudget = createPruneBudget();
            const morphDelta = createMorphDelta({
              kind: 'slotExpand',
              wiringCostDelta: 1,
            });

            // Act
            const validateGrowthDelta = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateGrowthDelta).not.toThrow();
          });
        });

        describe('given a node-add delta', () => {
          it('acts as a no-op in the Step 05 validator', () => {
            // Arrange
            const pruneBudget = createPruneBudget();
            const morphDelta = createMorphDelta({
              kind: 'nodeAdd',
              wiringCostDelta: 0,
            });

            // Act
            const validateGrowthDelta = () =>
              validatePruneDelta(morphDelta, pruneBudget);

            // Assert
            expect(validateGrowthDelta).not.toThrow();
          });
        });
      });

      describe('planPruneMorphs', () => {
        describe('given a hysteresis state that is not yet eligible to prune', () => {
          it('returns an empty delta list', () => {
            // Arrange
            const pruneBudget = createPruneBudget();
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate(),
            ]);
            const phaseConfig = createResolvedConfig({
              hysteresisWindowCount: 2,
            });
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 1,
            });

            // Act
            const plannedDeltas = planPruneMorphs(
              'module:alpha',
              pruneBudget,
              pruneCandidates,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas).toEqual([]);
          });
        });

        describe('given a valid candidate and compact headroom', () => {
          it('returns edge prune before compact', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 4,
              currentNodeCount: 3,
              minEdges: 1,
              minNodes: 1,
            });
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({ candidateId: 'edge:beta', wiringCost: 1 }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });

            // Act
            const plannedDeltas = planPruneMorphs(
              'module:alpha',
              pruneBudget,
              pruneCandidates,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual([
              'edgePrune',
              'compact',
            ]);
          });
        });

        describe('given all candidates are exempt', () => {
          it('silently omits the edge-prune delta', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              costExemptEdgeIds: ['edge:exempt'],
              currentNodeCount: 3,
            });
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({ candidateId: 'edge:exempt' }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });

            // Act
            const plannedDeltas = planPruneMorphs(
              'module:alpha',
              pruneBudget,
              pruneCandidates,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual(['compact']);
          });
        });

        describe('given an edge prune that fails the post-plan floor validation', () => {
          it('silently omits the edge-prune delta', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 2,
              minEdges: 1,
              currentNodeCount: 3,
            });
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:costly',
                wiringCost: 2,
              }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });

            // Act
            const plannedDeltas = planPruneMorphs(
              'module:alpha',
              pruneBudget,
              pruneCandidates,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual(['compact']);
          });
        });

        describe('given a node count at the compact floor', () => {
          it('silently omits the compact delta', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 4,
              currentNodeCount: 1,
              minEdges: 1,
              minNodes: 1,
            });
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:eligible',
                wiringCost: 1,
              }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });

            // Act
            const plannedDeltas = planPruneMorphs(
              'module:alpha',
              pruneBudget,
              pruneCandidates,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect(plannedDeltas.map(({ kind }) => kind)).toEqual([
              'edgePrune',
            ]);
          });
        });

        describe('given one dry-run prune planning pass', () => {
          it('does not mutate the input planning packets', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 4,
              currentNodeCount: 3,
            });
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:alpha',
                wiringCost: 1,
              }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });
            const originalPackets = structuredClone({
              pruneBudget,
              pruneCandidates,
              hysteresisState,
            });

            // Act
            planPruneMorphs(
              'module:alpha',
              pruneBudget,
              pruneCandidates,
              phaseConfig,
              hysteresisState,
            );

            // Assert
            expect({
              pruneBudget,
              pruneCandidates,
              hysteresisState,
            }).toEqual(originalPackets);
          });
        });

        describe('given an unexpected selection failure', () => {
          it('rethrows the unexpected error to the caller', () => {
            // Arrange
            const pruneBudget = createPruneBudget();
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:alpha',
                wiringCost: 1,
              }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });
            Object.defineProperty(pruneBudget, 'costExemptEdgeIds', {
              get() {
                throw new Error('unexpected-prune-selection-failure');
              },
            });

            // Act
            const planWithUnexpectedSelectionFailure = () =>
              planPruneMorphs(
                'module:alpha',
                pruneBudget,
                pruneCandidates,
                phaseConfig,
                hysteresisState,
              );

            // Assert
            expect(planWithUnexpectedSelectionFailure).toThrow(
              'unexpected-prune-selection-failure',
            );
          });
        });

        describe('given an unexpected compact failure', () => {
          it('rethrows the unexpected error to the caller', () => {
            // Arrange
            const pruneBudget = createPruneBudget({
              currentEdgeCount: 4,
              currentNodeCount: 3,
            });
            const pruneCandidates = createPruneCandidates([
              createPruneCandidate({
                candidateId: 'edge:alpha',
                wiringCost: 1,
              }),
            ]);
            const phaseConfig = createResolvedConfig();
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: 2,
            });
            Object.defineProperty(pruneBudget, 'currentNodeCount', {
              get() {
                throw new Error('unexpected-compact-failure');
              },
            });

            // Act
            const planWithUnexpectedCompactFailure = () =>
              planPruneMorphs(
                'module:alpha',
                pruneBudget,
                pruneCandidates,
                phaseConfig,
                hysteresisState,
              );

            // Assert
            expect(planWithUnexpectedCompactFailure).toThrow(
              'unexpected-compact-failure',
            );
          });
        });
      });

      describe('churn safety — alternating grow/prune cycles', () => {
        describe('given five deterministic grow and prune alternations', () => {
          it('keeps the edge-count slope near zero and the high-water mark bounded', () => {
            // Arrange
            const startEdgeCount = 5;
            const phaseConfig = createResolvedConfig();
            const focusScore = createFocusScore({ normalizedScore: 0.8 });
            let growthBudget = createGrowthBudget({
              currentEdgeCount: startEdgeCount,
              maxEdges: 10,
            });
            let pruneBudget = createPruneBudget({
              currentEdgeCount: startEdgeCount,
              currentNodeCount: 3,
              minEdges: 1,
              minNodes: 1,
            });
            const pruneCandidate = createPruneCandidate({
              candidateId: 'edge:stable',
              edgeLength: 1,
              wiringCost: NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
            });
            const hysteresisState = createHysteresisState({
              pruneUnderuseWindowCount: phaseConfig.hysteresisWindowCount,
            });
            let edgeCount = startEdgeCount;
            let maxEdgeCount = startEdgeCount;

            // Act
            for (let cycleIndex = 0; cycleIndex < 5; cycleIndex += 1) {
              const growthDelta = planEdgeDensification(
                'module:alpha',
                growthBudget,
                focusScore,
                phaseConfig,
              );
              edgeCount += growthDelta.wiringCostDelta;
              maxEdgeCount = Math.max(maxEdgeCount, edgeCount);
              growthBudget = createGrowthBudget({
                ...growthBudget,
                currentEdgeCount: edgeCount,
              });
              pruneBudget = createPruneBudget({
                ...pruneBudget,
                currentEdgeCount: edgeCount,
              });

              const plannedPruneDeltas = planPruneMorphs(
                'module:alpha',
                pruneBudget,
                createPruneCandidates([pruneCandidate]),
                phaseConfig,
                hysteresisState,
              );
              const committedPruneDelta = plannedPruneDeltas[0];
              validatePruneDelta(committedPruneDelta, pruneBudget);
              edgeCount += committedPruneDelta.wiringCostDelta;
              growthBudget = createGrowthBudget({
                ...growthBudget,
                currentEdgeCount: edgeCount,
              });
              pruneBudget = createPruneBudget({
                ...pruneBudget,
                currentEdgeCount: edgeCount,
              });
            }

            // Assert
            expect({
              edgeCountWithinTolerance:
                Math.abs(edgeCount - startEdgeCount) <= 1,
              highWaterMarkWithinBound:
                maxEdgeCount <=
                startEdgeCount +
                  5 * NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
            }).toEqual({
              edgeCountWithinTolerance: true,
              highWaterMarkWithinBound: true,
            });
          });
        });
      });
    });
  });
});

/**
 * @param overrides - Partial config overrides for one focused test case.
 * @returns A fully resolved juvenile focus config for test execution.
 */
function createResolvedConfig(overrides: Partial<NgeJuvenilePhaseConfig> = {}) {
  return resolveFocusConfig({
    cooldownWindowCount: 2,
    edgeDensificationCount: 1,
    episodicHitRateThreshold: 0.65,
    focusWeights: NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS,
    gainStabilityTolerance: 0.05,
    gainStabilityWindow: 5,
    hysteresisWindowCount: 2,
    nodeAdditionCount: 1,
    nodeGrowthSignalFloor: 0,
    recurrentRefreshFloor: 0.3,
    windowIndex: 2,
    ...overrides,
  });
}

/**
 * @returns Three modules with identical metric values.
 */
function createUniformSnapshots(): NgeModuleMetricsSnapshot[] {
  return [
    {
      moduleId: 'module:alpha',
      novelty: 2,
      rewardDelta: 2,
      stabilityAge: 2,
      utilization: 2,
      wiringCost: 2,
    },
    {
      moduleId: 'module:beta',
      novelty: 2,
      rewardDelta: 2,
      stabilityAge: 2,
      utilization: 2,
      wiringCost: 2,
    },
    {
      moduleId: 'module:gamma',
      novelty: 2,
      rewardDelta: 2,
      stabilityAge: 2,
      utilization: 2,
      wiringCost: 2,
    },
  ];
}

/**
 * @returns One weighted test slice with distinct values in every metric column.
 */
function createWeightedSnapshots(): NgeModuleMetricsSnapshot[] {
  return [
    {
      moduleId: 'module:alpha',
      novelty: 0.1,
      rewardDelta: 0.2,
      stabilityAge: 2,
      utilization: 4,
      wiringCost: 8,
    },
    {
      moduleId: 'module:beta',
      novelty: 0.4,
      rewardDelta: 0.9,
      stabilityAge: 6,
      utilization: 8,
      wiringCost: 3,
    },
    {
      moduleId: 'module:gamma',
      novelty: 0.8,
      rewardDelta: 0.1,
      stabilityAge: 3,
      utilization: 1,
      wiringCost: 5,
    },
  ];
}

/**
 * @param overrides - Partial scheduler config overrides for one probe test case.
 * @returns A fully resolved probe scheduler config for test execution.
 */
function createProbeSchedulerConfig(
  overrides: Partial<NgeProbeSchedulerConfig> = {},
): NgeProbeSchedulerConfig {
  return resolveProbeSchedulerConfig({
    cadenceEpochs: 10,
    gatingEdgeLengthThreshold: 2,
    lesionSeverity: 1,
    maxLedgerEntries: 500,
    noiseSigma: 0.05,
    probeKinds: ['lesion', 'noise', 'gating'],
    ...overrides,
  });
}

/**
 * @param entries - Probe entries to clone into one owner-local test ledger.
 * @returns A new probe ledger array.
 */
function createProbeLedger(
  entries: NgeProbeLedgerEntry[],
): NgeProbeLedgerEntry[] {
  return [...entries];
}

/**
 * @param overrides - Partial scheduler state overrides for one test case.
 * @returns A scheduler state packet with deterministic defaults.
 */
function createProbeSchedulerState(
  overrides: Partial<NgeProbeSchedulerState> = {},
): NgeProbeSchedulerState {
  return {
    lastProbeEpoch: -1,
    nextProbeKindIndex: 0,
    ledger: [],
    ...overrides,
  };
}

/**
 * @param overrides - Partial focus-score overrides for one local growth test case.
 * @returns A deterministic focus score packet.
 */
function createFocusScore(
  overrides: Partial<NgeFocusScore> = {},
): NgeFocusScore {
  const rawScore = overrides.rawScore ?? 1.25;
  return {
    moduleId: 'module:alpha',
    normalizedScore: 0.75,
    rawScore,
    supportsGrowth: overrides.supportsGrowth ?? rawScore > 0,
    normalizedUtilization: 0.5,
    normalizedRewardDelta: 0.5,
    normalizedNovelty: 0.5,
    normalizedStabilityAge: 0.5,
    normalizedWiringCost: 0.5,
    ...overrides,
  };
}

/**
 * @param overrides - Partial growth-budget overrides for one local growth test case.
 * @returns A deterministic growth budget packet.
 */
function createGrowthBudget(
  overrides: Partial<NgeGrowthBudget> = {},
): NgeGrowthBudget {
  return {
    currentEdgeCount: 2,
    currentEpisodicSlotCount: 2,
    currentNodeCount: 2,
    maxEdges: 5,
    maxEpisodicSlots: 5,
    maxNodes: 5,
    ...overrides,
  };
}

/**
 * @param entries - Prune candidates to clone into one owner-local test list.
 * @returns A new prune candidate array.
 */
function createPruneCandidates(
  entries: NgePruneCandidate[],
): NgePruneCandidate[] {
  return [...entries];
}

/**
 * @param overrides - Partial prune-candidate overrides for one local prune test case.
 * @returns A deterministic prune candidate packet.
 */
function createPruneCandidate(
  overrides: Partial<NgePruneCandidate> = {},
): NgePruneCandidate {
  return {
    candidateId: 'edge:alpha',
    edgeLength: 2,
    wiringCost: 1,
    ...overrides,
  };
}

/**
 * @param overrides - Partial prune-budget overrides for one local prune test case.
 * @returns A deterministic prune budget packet.
 */
function createPruneBudget(
  overrides: Partial<NgePruneBudget> = {},
): NgePruneBudget {
  return {
    costExemptEdgeIds: [],
    currentEdgeCount: 3,
    currentNodeCount: 3,
    currentWiringCost: 4,
    minEdges: NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR,
    minNodes: 1,
    ...overrides,
  };
}

/**
 * @param overrides - Partial hysteresis-state overrides for one local growth test case.
 * @returns A deterministic hysteresis state packet.
 */
function createHysteresisState(
  overrides: Partial<NgeHysteresisState> = {},
): NgeHysteresisState {
  return {
    cooldownWindowsRemaining: 0,
    growthPositiveWindowCount: 0,
    lastMorphKind: 'none',
    pruneUnderuseWindowCount: 0,
    ...overrides,
  };
}

/**
 * @param overrides - Partial module metrics overrides for one local growth test case.
 * @returns A deterministic module metrics snapshot.
 */
function createGrowthMetrics(
  overrides: Partial<NgeModuleMetricsSnapshot> = {},
): NgeModuleMetricsSnapshot {
  return {
    moduleId: 'module:alpha',
    novelty: 0.1,
    rewardDelta: 0.4,
    stabilityAge: 3,
    utilization: 0.8,
    wiringCost: 1,
    ...overrides,
  };
}

/**
 * @param overrides - Partial morph-delta overrides for one validator test case.
 * @returns A deterministic morph delta packet.
 */
function createMorphDelta(
  overrides: Partial<NgeMorphDelta> = {},
): NgeMorphDelta {
  return {
    detail: {},
    kind: 'edgeDensify',
    targetModuleId: 'module:alpha',
    wiringCostDelta: 1,
    ...overrides,
  };
}
