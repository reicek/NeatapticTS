/**
 * Green-phase test contracts for slice 04-main-reproduction.
 *
 * Covers AC-410: the main-agent reproduction stage selects a mode through the
 * hysteresis policy, dispatches to the matching NGE operator, and returns a
 * deterministic offspring with parent fingerprints and the main-agent seed
 * policy enforced.
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import { NGE_DNA } from '../nge-dna/neat.nge-dna';
import type {
  NgeDnaCanonicalEnvelope,
  NgeReproductionPolicyMode,
} from '../nge-dna/neat.nge-dna.types';
import type { ReproductionModePressureSignal } from '../nge-evolution/neat.nge-evolution.reproduction-mode';
import type { NgeMainAgentEquilibriumResult } from './neat.nge-main-agent.adult';
import { runReproductionStage } from './neat.nge-main-agent.reproduction';
import type { NgeMainAgentLifecycleConfig } from './neat.nge-main-agent.types';

const defaultConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 1024,
  maxEdges: 4096,
};

function createConfig(seed = 42): NgeMainAgentLifecycleConfig {
  return { ...defaultConfig, seed };
}

function createEquilibriumResult(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibriumResult {
  const adult = {
    stage: 'adult' as const,
    generation: 1,
    seed: config.seed,
    nodeCount: 4,
    edgeCount: 6,
    archetypes: [],
    schemaVersion: '1.0.0',
  };

  const stableCandidate = {
    genomeState: {
      nodeCount: adult.nodeCount,
      edgeCount: adult.edgeCount,
      archetypes: adult.archetypes,
      schemaVersion: adult.schemaVersion,
    },
    fitnessMetrics: {
      survivalTicks: 0,
      damageDealt: 0,
      kills: 0,
      damageTaken: 0,
      aimMissRate: 0,
      complexityBonus: 0.5,
      parsimonyDensityPenalty: 0.2,
    },
    structuralInfo: {
      withinBudget: true,
      maxNodes: config.maxNodes,
      maxEdges: config.maxEdges,
    },
  };

  return {
    isStable: true,
    adult,
    stableCandidate,
    snapshot: {
      kind: 'main-agent-equilibrium',
      frozenAtGeneration: adult.generation,
      seed: config.seed,
      candidate: stableCandidate,
    },
  };
}

function createParentDna(
  mode: NgeReproductionPolicyMode,
  modeIsEvolvable = false,
): NgeDnaCanonicalEnvelope {
  return new NGE_DNA({
    reproductionPolicy: {
      mode,
      modeIsEvolvable,
      polyandricDroneCount: 2,
      polyandricDroneContributionFraction: 1,
      queenBias: 1,
      assignedRegionStrategy: 'roundRobin',
    },
  }).toCanonical();
}

function createMateDna(
  mode: NgeReproductionPolicyMode = 'sexual',
): NgeDnaCanonicalEnvelope {
  return new NGE_DNA({
    reproductionPolicy: {
      mode,
      modeIsEvolvable: false,
      polyandricDroneCount: 2,
      polyandricDroneContributionFraction: 1,
      queenBias: 1,
      assignedRegionStrategy: 'roundRobin',
    },
  }).toCanonical();
}

function buildPressure(
  kind: 'dominating' | 'struggling' | 'stalemate',
  count = 3,
): ReproductionModePressureSignal[] {
  return Array.from({ length: count }, (_, index) => ({
    generation: index + 1,
    isDominating: kind === 'dominating',
    isStruggling: kind === 'struggling',
    isStalemate: kind === 'stalemate',
  }));
}

describe('runReproductionStage', () => {
  describe('mode dispatch', () => {
    it('produces parthenogenetic offspring when the policy mode is parthenogenesis', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis');

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        buildPressure('dominating'),
        parentDna,
        config,
      );

      // Assert
      expect({
        mode: result.mode,
        seed: result.seed,
        parentFingerprints: result.parentFingerprints,
        seedPolicy: result.offspring.reproductionPolicy.seedPolicy,
      }).toEqual({
        mode: 'parthenogenesis',
        seed: config.seed,
        parentFingerprints: [parentDna.fingerprint],
        seedPolicy: { siblingsDifferBySeed: true, twinsAllowed: false },
      });
    });

    it('produces polyandric offspring when pressure history votes polyandric', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis', true);
      const droneA = createMateDna();
      const droneB = createMateDna();

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        buildPressure('struggling'),
        parentDna,
        config,
        [droneA, droneB],
      );

      // Assert
      expect({
        mode: result.mode,
        seed: result.seed,
        parentFingerprints: result.parentFingerprints,
        seedPolicy: result.offspring.reproductionPolicy.seedPolicy,
      }).toEqual({
        mode: 'polyandric',
        seed: config.seed,
        parentFingerprints: [
          parentDna.fingerprint,
          droneA.fingerprint,
          droneB.fingerprint,
        ],
        seedPolicy: { siblingsDifferBySeed: true, twinsAllowed: false },
      });
    });

    it('produces sexual offspring when pressure history votes sexual', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis', true);
      const secondParent = createMateDna();

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        buildPressure('stalemate'),
        parentDna,
        config,
        [secondParent],
      );

      // Assert
      expect({
        mode: result.mode,
        seed: result.seed,
        parentFingerprints: result.parentFingerprints,
        seedPolicy: result.offspring.reproductionPolicy.seedPolicy,
      }).toEqual({
        mode: 'sexual',
        seed: config.seed,
        parentFingerprints: [parentDna.fingerprint, secondParent.fingerprint],
        seedPolicy: { siblingsDifferBySeed: true, twinsAllowed: false },
      });
    });

    it('falls back to the policy mode when no pressure history is supplied', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis', false);

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        [],
        parentDna,
        config,
      );

      // Assert
      expect({
        mode: result.mode,
        parentFingerprints: result.parentFingerprints,
      }).toEqual({
        mode: 'parthenogenesis',
        parentFingerprints: [parentDna.fingerprint],
      });
    });
  });

  describe('determinism and seed policy', () => {
    it('returns the same offspring fingerprint for identical inputs', () => {
      // Arrange
      const config = createConfig(7);
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis');
      const args: Parameters<typeof runReproductionStage> = [
        equilibriumResult,
        buildPressure('dominating'),
        parentDna,
        config,
      ];

      // Act
      const first = runReproductionStage(...args);
      const second = runReproductionStage(...args);

      // Assert
      expect(first.offspring.fingerprint).toBe(second.offspring.fingerprint);
    });

    it('enforces the main-agent seed policy even when the parent DNA allows twins', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = new NGE_DNA({
        reproductionPolicy: {
          mode: 'parthenogenesis',
          modeIsEvolvable: false,
          seedPolicy: { siblingsDifferBySeed: false, twinsAllowed: true },
        },
      }).toCanonical();

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        buildPressure('dominating'),
        parentDna,
        config,
      );

      // Assert
      expect(result.offspring.reproductionPolicy.seedPolicy).toEqual({
        siblingsDifferBySeed: true,
        twinsAllowed: false,
      });
    });
  });

  describe('mate pool fallbacks', () => {
    it('reuses the primary parent as a fallback mate for polyandric mode', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis', true);

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        buildPressure('struggling'),
        parentDna,
        config,
        [],
      );

      // Assert
      expect(result.parentFingerprints).toEqual([
        parentDna.fingerprint,
        parentDna.fingerprint,
      ]);
    });

    it('reuses the primary parent as a fallback mate for sexual mode', () => {
      // Arrange
      const config = createConfig();
      const equilibriumResult = createEquilibriumResult(config);
      const parentDna = createParentDna('parthenogenesis', true);

      // Act
      const result = runReproductionStage(
        equilibriumResult,
        buildPressure('stalemate'),
        parentDna,
        config,
        [],
      );

      // Assert
      expect(result.parentFingerprints).toEqual([
        parentDna.fingerprint,
        parentDna.fingerprint,
      ]);
    });
  });
});
