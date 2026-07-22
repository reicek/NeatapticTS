/**
 * Red-phase test contracts for Phase 4 Step 01 — internal assimilation prior
 * write-back.
 *
 * Covers AC-405: assimilation writes weak/decaying structural priors derived
 * from the main agent's own equilibrium candidate only; no enemy weights or
 * structure flow into the main agent.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { EquilibriumCandidate } from '../nge-adult/neat.nge-adult.types';

import type { NgeAssimilationCandidate } from './neat.nge-assimilation.types';
import {
  writeInternalAssimilationPriors,
  type InternalAssimilationInput,
} from './neat.nge-assimilation.internal';

function createEquilibriumCandidate(
  overrides?: Partial<EquilibriumCandidate>,
): EquilibriumCandidate {
  return {
    zoneId: 'zone:main-equilibrium',
    isGainStable: true,
    isPlateau: true,
    ...overrides,
  };
}

function createAssimilationCandidate(
  overrides?: Partial<NgeAssimilationCandidate>,
): NgeAssimilationCandidate {
  return {
    equilibriumCandidate: createEquilibriumCandidate(),
    sourceDnaFingerprint: 'fp:main-agent-equilibrium',
    sourceSchemaVersion: 'A.1.0',
    moduleDelta: {
      moduleId: 'module:main-1',
      zoneId: 'zone:main-1',
      ruleParameters: {
        connectionDensity: {
          currentValue: 0.2,
          targetValue: 0.8,
        },
      },
    },
    ...overrides,
  };
}

describe('writeInternalAssimilationPriors', () => {
  describe('single-source constraint', () => {
    it('writes priors derived from the main agent equilibrium candidate', () => {
      // Arrange
      const equilibrium = createEquilibriumCandidate();
      const candidate = createAssimilationCandidate({
        equilibriumCandidate: equilibrium,
      });
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.9,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.sourceDnaFingerprint).toBe('fp:main-agent-equilibrium');
    });

    it('does not accept enemy-derived weights in the input', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.9,
        enemyWeightTensor: new Float32Array([0.1, 0.2, 0.3]),
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert — enemy weights must be rejected or ignored
      expect(result.enemyWeightsIncorporated).toBe(false);
    });

    it('does not accept enemy-derived structure in the input', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.9,
        enemyStructureChecksum: 'enemy:structure:123',
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert — enemy structure must be rejected or ignored
      expect(result.enemyStructureIncorporated).toBe(false);
    });
  });

  describe('weak/decaying prior update', () => {
    it('applies the configured write-back rate to move current toward target', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity
          .currentValue,
      ).toBe(0.26);
    });

    it('decays the applied delta by the configured decay factor', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 0.5,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.appliedDecay).toBe(0.5);
    });

    it('keeps the target value unchanged after write-back', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(
        result.updatedModuleDelta.ruleParameters?.connectionDensity.targetValue,
      ).toBe(0.8);
    });
  });

  describe('result contract', () => {
    it('returns a non-null updated module delta', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.updatedModuleDelta).not.toBeNull();
    });

    it('preserves the module identifier through write-back', () => {
      // Arrange
      const candidate = createAssimilationCandidate();
      const input: InternalAssimilationInput = {
        candidate,
        writeBackRate: 0.1,
        decay: 1,
      };

      // Act
      const result = writeInternalAssimilationPriors(input);

      // Assert
      expect(result.updatedModuleDelta.moduleId).toBe('module:main-1');
    });
  });
});
