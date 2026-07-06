import {
  NGE_ADULT_DEFAULT_GAIN_STABILITY_TOLERANCE,
  NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
  NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
  NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR,
  NGE_ADULT_DEFAULT_MARGINAL_EPSILON,
  NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
} from './neat.nge-adult.constants';
import {
  NgeAdult_EquilibriumError,
  NgeAdult_PlateauError,
} from './neat.nge-adult.errors';
import {
  advancePlateauRecord,
  computeMarginalReturn,
} from './neat.nge-adult.plateau';
import {
  arbitratePruneCompactDominance,
  resolveGrowthCoolingDecision,
} from './neat.nge-adult.cooling';
import {
  advanceGainStabilityRecord,
  detectEquilibriumCandidate,
} from './neat.nge-adult.equilibrium';
import { advanceAdultState } from './neat.nge-adult';
import type {
  AdultState,
  EquilibriumCandidate,
  GainStabilityRecord,
  PlateauRecord,
} from './neat.nge-adult.types';
import { createAdultState } from './neat.nge-adult.utils';

describe('nge adult contract boundary', () => {
  describe('type contracts', () => {
    describe('given the seeded adult state fixture', () => {
      it('keeps the required adult-state keys in the contract fixture', () => {
        // Arrange
        const plateauRecord: PlateauRecord = {
          windowSize: 6,
          rewardDeltas: [0.01, 0.005, 0.004],
          isStagnant: false,
        };
        const gainStabilityRecord: GainStabilityRecord = {
          windowSize: 5,
          gainHistory: [0.15, 0.13, 0.12],
          isStable: false,
        };
        const equilibriumCandidate: EquilibriumCandidate = {
          zoneId: 'zone:adult:0',
          isGainStable: false,
          isPlateau: false,
        };
        const adultState: AdultState = {
          plateauRecord,
          gainStabilityRecord,
          equilibriumCandidate,
          growthCoolingActive: false,
          cycleCount: 3,
        };

        // Assert
        expect(Object.keys(adultState).toSorted()).toEqual([
          'cycleCount',
          'equilibriumCandidate',
          'gainStabilityRecord',
          'growthCoolingActive',
          'plateauRecord',
        ]);
      });
    });

    describe('given the seeded plateau record fixture', () => {
      it('keeps the plateau-record keys in the contract fixture', () => {
        // Arrange
        const plateauRecord: PlateauRecord = {
          windowSize: 6,
          rewardDeltas: [0.01, 0.005, 0.004],
          isStagnant: true,
        };

        // Assert
        expect(Object.keys(plateauRecord).toSorted()).toEqual([
          'isStagnant',
          'rewardDeltas',
          'windowSize',
        ]);
      });
    });

    describe('given the seeded equilibrium candidate fixture', () => {
      it('keeps the equilibrium-candidate keys in the contract fixture', () => {
        // Arrange
        const equilibriumCandidate: EquilibriumCandidate = {
          zoneId: 'zone:adult:0',
          isGainStable: true,
          isPlateau: true,
        };

        // Assert
        expect(Object.keys(equilibriumCandidate).toSorted()).toEqual([
          'isGainStable',
          'isPlateau',
          'zoneId',
        ]);
      });
    });

    describe('given the seeded gain stability record fixture', () => {
      it('keeps the gain-stability keys in the contract fixture', () => {
        // Arrange
        const gainStabilityRecord: GainStabilityRecord = {
          windowSize: 5,
          gainHistory: [0.15, 0.13, 0.12],
          isStable: true,
        };

        // Assert
        expect(Object.keys(gainStabilityRecord).toSorted()).toEqual([
          'gainHistory',
          'isStable',
          'windowSize',
        ]);
      });
    });
  });

  describe('constants', () => {
    it('keeps the plateau window guard at six evaluation windows', () => {
      expect(NGE_ADULT_DEFAULT_PLATEAU_WINDOW).toBe(6);
    });

    it('keeps the marginal epsilon guard at 0.01', () => {
      expect(NGE_ADULT_DEFAULT_MARGINAL_EPSILON).toBe(0.01);
    });

    it('keeps the gain stability window guard at five evaluation windows', () => {
      expect(NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW).toBe(5);
    });

    it('keeps the gain stability tolerance guard at 0.05', () => {
      expect(NGE_ADULT_DEFAULT_GAIN_STABILITY_TOLERANCE).toBe(0.05);
    });

    it('keeps the growth cooling factor guard at 0.1', () => {
      expect(NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR).toBe(0.1);
    });

    it('keeps the growth focus floor guard at 0.6', () => {
      expect(NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR).toBe(0.6);
    });
  });

  describe('errors', () => {
    it('keeps the plateau error cause and public error name', () => {
      // Arrange
      const cause = new Error('plateau-root-cause');

      // Act
      const plateauError = new NgeAdult_PlateauError('plateau failed', {
        cause,
      });

      // Assert
      expect({ cause: plateauError.cause, name: plateauError.name }).toEqual({
        cause,
        name: 'NgeAdult_PlateauError',
      });
    });

    it('keeps the equilibrium error cause and public error name', () => {
      // Arrange
      const cause = new Error('equilibrium-root-cause');

      // Act
      const equilibriumError = new NgeAdult_EquilibriumError(
        'equilibrium failed',
        { cause },
      );

      // Assert
      expect({
        cause: equilibriumError.cause,
        name: equilibriumError.name,
      }).toEqual({
        cause,
        name: 'NgeAdult_EquilibriumError',
      });
    });
  });

  describe('plateau detection', () => {
    describe('advancePlateauRecord', () => {
      describe('given more reward deltas than the plateau window allows', () => {
        it('keeps the latest six deltas and marks the record stagnant when each one is marginal', () => {
          // Arrange
          const seededPlateauRecord: PlateauRecord = {
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas: [],
            isStagnant: false,
          };
          const rewardDeltas = [0.12, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004];

          // Act
          const plateauRecord = rewardDeltas.reduce(
            (currentRecord, rewardDelta) =>
              advancePlateauRecord(
                currentRecord,
                rewardDelta,
                NGE_ADULT_DEFAULT_MARGINAL_EPSILON,
              ),
            seededPlateauRecord,
          );

          // Assert
          expect(plateauRecord).toEqual({
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas: [0.009, 0.008, 0.007, 0.006, 0.005, 0.004],
            isStagnant: true,
          });
        });
      });

      describe('given one recent reward delta that beats the marginal epsilon', () => {
        it('keeps the record non-stagnant while the positive return remains in the active window', () => {
          // Arrange
          const seededPlateauRecord: PlateauRecord = {
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas: [],
            isStagnant: false,
          };
          const rewardDeltas = [0.009, 0.008, 0.007, 0.006, 0.02, 0.005];

          // Act
          const plateauRecord = rewardDeltas.reduce(
            (currentRecord, rewardDelta) =>
              advancePlateauRecord(
                currentRecord,
                rewardDelta,
                NGE_ADULT_DEFAULT_MARGINAL_EPSILON,
              ),
            seededPlateauRecord,
          );

          // Assert
          expect(plateauRecord).toEqual({
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas,
            isStagnant: false,
          });
        });
      });

      describe('given no custom marginal epsilon override', () => {
        it('uses the seeded adult default epsilon when checking plateau stagnation', () => {
          // Arrange
          const seededPlateauRecord: PlateauRecord = {
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas: [],
            isStagnant: false,
          };
          const rewardDeltas = [0.009, 0.008, 0.007, 0.006, 0.005, 0.004];

          // Act
          const plateauRecord = rewardDeltas.reduce(
            (currentRecord, rewardDelta) =>
              advancePlateauRecord(currentRecord, rewardDelta),
            seededPlateauRecord,
          );

          // Assert
          expect(plateauRecord).toEqual({
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas,
            isStagnant: true,
          });
        });
      });
    });

    describe('computeMarginalReturn', () => {
      describe('given multiple structural edits', () => {
        it('normalizes reward improvement per structural edit', () => {
          // Act
          const marginalReturn = computeMarginalReturn(0.03, 3);

          // Assert
          expect(marginalReturn).toBeCloseTo(0.01, 12);
        });
      });

      describe('given zero structural edits', () => {
        it('rejects the non-normalizable marginal-return input', () => {
          // Assert
          expect(() => computeMarginalReturn(0.03, 0)).toThrow(
            NgeAdult_PlateauError,
          );
        });
      });
    });
  });

  describe('growth cooling', () => {
    describe('resolveGrowthCoolingDecision', () => {
      describe('given a strongly positive adult focus score', () => {
        it('keeps only the residual growth budget and gives the remainder to prune and compact', () => {
          // Act
          const coolingDecision = resolveGrowthCoolingDecision(0.82);

          // Assert
          expect(coolingDecision).toEqual({
            growthCoolingActive: true,
            growthBudgetFraction: NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
            pruneCompactBudgetFraction:
              1 - NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
          });
        });
      });

      describe('given an adult focus score below the seeded floor', () => {
        it('fully suppresses growth so prune and compact receive the whole budget', () => {
          // Act
          const coolingDecision = resolveGrowthCoolingDecision(
            NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR - 0.01,
          );

          // Assert
          expect(coolingDecision).toEqual({
            growthCoolingActive: false,
            growthBudgetFraction: 0,
            pruneCompactBudgetFraction: 1,
          });
        });
      });
    });

    describe('arbitratePruneCompactDominance', () => {
      describe('given one long inter-module edge and compact headroom', () => {
        it('prefers edge prune before compact while protecting reward-critical wiring', () => {
          // Arrange
          const pruneCandidates = [
            {
              candidateId: 'edge:reward-critical',
              edgeLength: 12,
              isInterModule: true,
              isRewardCritical: true,
            },
            {
              candidateId: 'edge:long-inter-module',
              edgeLength: 8,
              isInterModule: true,
              isRewardCritical: false,
            },
            {
              candidateId: 'edge:short-local',
              edgeLength: 3,
              isInterModule: false,
              isRewardCritical: false,
            },
          ];

          // Act
          const arbitration = arbitratePruneCompactDominance(
            pruneCandidates,
            true,
          );

          // Assert
          expect(arbitration).toEqual({
            dominantMorphKinds: ['edgePrune', 'compact'],
            selectedPruneCandidateId: 'edge:long-inter-module',
          });
        });
      });

      describe('given only reward-critical prune candidates', () => {
        it('falls back to compact-only dominance', () => {
          // Arrange
          const pruneCandidates = [
            {
              candidateId: 'edge:reward-critical',
              edgeLength: 12,
              isInterModule: true,
              isRewardCritical: true,
            },
          ];

          // Act
          const arbitration = arbitratePruneCompactDominance(
            pruneCandidates,
            true,
          );

          // Assert
          expect(arbitration).toEqual({
            dominantMorphKinds: ['compact'],
            selectedPruneCandidateId: null,
          });
        });
      });

      describe('given two equally ranked non-critical prune candidates', () => {
        it('uses the stable candidate id tie-breaker before falling back to compact', () => {
          // Arrange
          const pruneCandidates = [
            {
              candidateId: 'edge:zeta',
              edgeLength: 6,
              isInterModule: true,
              isRewardCritical: false,
            },
            {
              candidateId: 'edge:alpha',
              edgeLength: 6,
              isInterModule: true,
              isRewardCritical: false,
            },
          ];

          // Act
          const arbitration = arbitratePruneCompactDominance(
            pruneCandidates,
            true,
          );

          // Assert
          expect(arbitration).toEqual({
            dominantMorphKinds: ['edgePrune', 'compact'],
            selectedPruneCandidateId: 'edge:alpha',
          });
        });
      });

      describe('given two same-scope non-critical prune candidates with different edge lengths', () => {
        it('prefers the longer edge before compact', () => {
          // Arrange
          const pruneCandidates = [
            {
              candidateId: 'edge:short',
              edgeLength: 4,
              isInterModule: false,
              isRewardCritical: false,
            },
            {
              candidateId: 'edge:long',
              edgeLength: 9,
              isInterModule: false,
              isRewardCritical: false,
            },
          ];

          // Act
          const arbitration = arbitratePruneCompactDominance(
            pruneCandidates,
            true,
          );

          // Assert
          expect(arbitration).toEqual({
            dominantMorphKinds: ['edgePrune', 'compact'],
            selectedPruneCandidateId: 'edge:long',
          });
        });
      });

      describe('given a surviving prune candidate without compact headroom', () => {
        it('returns prune-only dominance', () => {
          // Arrange
          const pruneCandidates = [
            {
              candidateId: 'edge:local',
              edgeLength: 4,
              isInterModule: false,
              isRewardCritical: false,
            },
          ];

          // Act
          const arbitration = arbitratePruneCompactDominance(
            pruneCandidates,
            false,
          );

          // Assert
          expect(arbitration).toEqual({
            dominantMorphKinds: ['edgePrune'],
            selectedPruneCandidateId: 'edge:local',
          });
        });
      });

      describe('given no prune candidate and no compact headroom', () => {
        it('returns an empty dominant morph set', () => {
          // Arrange
          const pruneCandidates = [
            {
              candidateId: 'edge:reward-critical',
              edgeLength: 5,
              isInterModule: false,
              isRewardCritical: true,
            },
          ];

          // Act
          const arbitration = arbitratePruneCompactDominance(
            pruneCandidates,
            false,
          );

          // Assert
          expect(arbitration).toEqual({
            dominantMorphKinds: [],
            selectedPruneCandidateId: null,
          });
        });
      });
    });
  });

  describe('equilibrium detection', () => {
    describe('advanceGainStabilityRecord', () => {
      describe('given more gain measurements than the stability window allows', () => {
        it('keeps the latest five gains and marks the record stable when each sample stays within tolerance of the rolling mean', () => {
          // Arrange
          const seededGainStabilityRecord: GainStabilityRecord = {
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: [],
            isStable: false,
          };
          const gainMeasurements = [0.71, 0.46, 0.49, 0.5, 0.47, 0.48];

          // Act
          const gainStabilityRecord = gainMeasurements.reduce(
            (currentRecord, gainMeasurement) =>
              advanceGainStabilityRecord(currentRecord, gainMeasurement),
            seededGainStabilityRecord,
          );

          // Assert
          expect(gainStabilityRecord).toEqual({
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: [0.46, 0.49, 0.5, 0.47, 0.48],
            isStable: true,
          });
        });
      });

      describe('given one outlier that breaks the seeded tolerance', () => {
        it('marks the record unstable while the outlier remains in the active window', () => {
          // Arrange
          const seededGainStabilityRecord: GainStabilityRecord = {
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: [],
            isStable: false,
          };
          const gainMeasurements = [0.46, 0.49, 0.5, 0.47, 0.57];

          // Act
          const gainStabilityRecord = gainMeasurements.reduce(
            (currentRecord, gainMeasurement) =>
              advanceGainStabilityRecord(currentRecord, gainMeasurement),
            seededGainStabilityRecord,
          );

          // Assert
          expect(gainStabilityRecord).toEqual({
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: gainMeasurements,
            isStable: false,
          });
        });
      });

      describe('given a tighter gain tolerance override', () => {
        it('uses the override instead of the seeded default when deciding stability', () => {
          // Arrange
          const seededGainStabilityRecord: GainStabilityRecord = {
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: [],
            isStable: false,
          };
          const gainMeasurements = [0.46, 0.49, 0.5, 0.47, 0.52];

          // Act
          const gainStabilityRecord = gainMeasurements.reduce(
            (currentRecord, gainMeasurement) =>
              advanceGainStabilityRecord(currentRecord, gainMeasurement, 0.02),
            seededGainStabilityRecord,
          );

          // Assert
          expect(gainStabilityRecord).toEqual({
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: gainMeasurements,
            isStable: false,
          });
        });
      });
    });

    describe('detectEquilibriumCandidate', () => {
      describe('given a plateaued zone with stable neuromodulator gains', () => {
        it('emits an equilibrium candidate for downstream assimilation pressure', () => {
          // Arrange
          const plateauRecord: PlateauRecord = {
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas: [0.009, 0.008, 0.007, 0.006, 0.005, 0.004],
            isStagnant: true,
          };
          const gainStabilityRecord: GainStabilityRecord = {
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: [0.46, 0.49, 0.5, 0.47, 0.48],
            isStable: true,
          };

          // Act
          const equilibriumCandidate = detectEquilibriumCandidate(
            'zone:adult:stable',
            plateauRecord,
            gainStabilityRecord,
          );

          // Assert
          expect(equilibriumCandidate).toEqual({
            zoneId: 'zone:adult:stable',
            isGainStable: true,
            isPlateau: true,
          });
        });
      });

      describe('given a non-plateaued zone despite stable gains', () => {
        it('withholds the equilibrium candidate event until both guards hold simultaneously', () => {
          // Arrange
          const plateauRecord: PlateauRecord = {
            windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
            rewardDeltas: [0.02, 0.008, 0.007, 0.006, 0.005, 0.004],
            isStagnant: false,
          };
          const gainStabilityRecord: GainStabilityRecord = {
            windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
            gainHistory: [0.46, 0.49, 0.5, 0.47, 0.48],
            isStable: true,
          };

          // Act
          const equilibriumCandidate = detectEquilibriumCandidate(
            'zone:adult:growing',
            plateauRecord,
            gainStabilityRecord,
          );

          // Assert
          expect(equilibriumCandidate).toBeNull();
        });
      });
    });
  });

  describe('adult orchestration', () => {
    describe('createAdultState', () => {
      describe('given one adult zone id', () => {
        it('seeds the owner-local adult state with the plan defaults and an inactive candidate snapshot', () => {
          // Act
          const adultState = createAdultState('zone:adult:seed');

          // Assert
          expect(adultState).toEqual({
            plateauRecord: {
              windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
              rewardDeltas: [],
              isStagnant: false,
            },
            gainStabilityRecord: {
              windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
              gainHistory: [],
              isStable: false,
            },
            equilibriumCandidate: {
              zoneId: 'zone:adult:seed',
              isGainStable: false,
              isPlateau: false,
            },
            growthCoolingActive: false,
            cycleCount: 0,
          });
        });
      });
    });

    describe('advanceAdultState', () => {
      describe('given one adult cycle that completes the plateau and gain-stability windows', () => {
        it('composes the owner-local plateau, cooling, marginal-return, and equilibrium decisions into one adult transition result', () => {
          // Arrange
          const seededAdultState: AdultState = {
            plateauRecord: {
              windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
              rewardDeltas: [0.009, 0.008, 0.007, 0.006, 0.005],
              isStagnant: false,
            },
            gainStabilityRecord: {
              windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
              gainHistory: [0.46, 0.49, 0.5, 0.47],
              isStable: false,
            },
            equilibriumCandidate: {
              zoneId: 'zone:adult:stable',
              isGainStable: false,
              isPlateau: false,
            },
            growthCoolingActive: false,
            cycleCount: 3,
          };
          const pruneCandidates = [
            {
              candidateId: 'edge:reward-critical',
              edgeLength: 12,
              isInterModule: true,
              isRewardCritical: true,
            },
            {
              candidateId: 'edge:long-inter-module',
              edgeLength: 8,
              isInterModule: true,
              isRewardCritical: false,
            },
          ];

          // Act
          const adultTransition = advanceAdultState({
            adultPhaseEnabled: true,
            adultState: seededAdultState,
            compactEligible: true,
            focusScore: 0.82,
            gainMeasurement: 0.48,
            pruneCandidates,
            rewardDelta: 0.004,
            structuralEditCount: 1,
            zoneId: 'zone:adult:stable',
          });

          // Assert
          expect(adultTransition).toEqual({
            adultState: {
              plateauRecord: {
                windowSize: NGE_ADULT_DEFAULT_PLATEAU_WINDOW,
                rewardDeltas: [0.009, 0.008, 0.007, 0.006, 0.005, 0.004],
                isStagnant: true,
              },
              gainStabilityRecord: {
                windowSize: NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW,
                gainHistory: [0.46, 0.49, 0.5, 0.47, 0.48],
                isStable: true,
              },
              equilibriumCandidate: {
                zoneId: 'zone:adult:stable',
                isGainStable: true,
                isPlateau: true,
              },
              growthCoolingActive: true,
              cycleCount: 4,
            },
            coolingDecision: {
              growthCoolingActive: true,
              growthBudgetFraction: NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
              pruneCompactBudgetFraction:
                1 - NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
            },
            equilibriumCandidate: {
              zoneId: 'zone:adult:stable',
              isGainStable: true,
              isPlateau: true,
            },
            marginalReturn: 0.004,
            pruneCompactDecision: {
              dominantMorphKinds: ['edgePrune', 'compact'],
              selectedPruneCandidateId: 'edge:long-inter-module',
            },
          });
        });
      });
    });
  });
});
