/**
 * Red-phase test contracts for Phase 4 Step 05 — MLP enemy selection-pressure
 * policy.
 *
 * Covers AC-401-S05-001: the MLP enemy is fixed-topology weight-only selection
 * pressure with no structural assimilation.
 *
 * All tests fail because the imported source module does not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import {
  createMlpEnemySelectionPolicy,
  guardMlpEnemySelectionPolicy,
  type MlpEnemySelectionPolicy,
} from './neat.nge-evolution.mlp-enemy-policy';

function createDefaultPolicy(): MlpEnemySelectionPolicy {
  return createMlpEnemySelectionPolicy();
}

describe('MlpEnemySelectionPolicy', () => {
  describe('createMlpEnemySelectionPolicy', () => {
    it('returns a policy with structural mutation disabled', () => {
      // Arrange — default MLP enemy policy
      const policy = createDefaultPolicy();

      // Act & Assert
      expect(policy.allowsStructuralMutation).toBe(false);
    });

    it('returns a policy with topological mutation disabled', () => {
      // Arrange — default MLP enemy policy
      const policy = createDefaultPolicy();

      // Act & Assert
      expect(policy.allowsTopologicalMutation).toBe(false);
    });

    it('returns a policy whose allowed mutation kinds are weight and bias only', () => {
      // Arrange — default MLP enemy policy
      const policy = createDefaultPolicy();

      // Act & Assert
      expect(policy.allowedMutationKinds).toEqual(['weight', 'bias']);
    });
  });

  describe('guardMlpEnemySelectionPolicy', () => {
    it('throws RangeError when structural mutation is allowed', () => {
      // Arrange — policy that violates the fixed-topology constraint
      const policy: MlpEnemySelectionPolicy = {
        allowsStructuralMutation: true,
        allowsTopologicalMutation: false,
        allowedMutationKinds: ['weight', 'bias'],
      };

      // Act & Assert
      expect(() => guardMlpEnemySelectionPolicy(policy)).toThrow(RangeError);
    });

    it('throws RangeError when topological mutation is allowed', () => {
      // Arrange — policy that violates the fixed-topology constraint
      const policy: MlpEnemySelectionPolicy = {
        allowsStructuralMutation: false,
        allowsTopologicalMutation: true,
        allowedMutationKinds: ['weight', 'bias'],
      };

      // Act & Assert
      expect(() => guardMlpEnemySelectionPolicy(policy)).toThrow(RangeError);
    });

    it('does not throw for a default MLP enemy policy', () => {
      // Arrange — default fixed-topology weight-only policy
      const policy = createDefaultPolicy();

      // Act & Assert
      expect(() => guardMlpEnemySelectionPolicy(policy)).not.toThrow();
    });
  });
});
