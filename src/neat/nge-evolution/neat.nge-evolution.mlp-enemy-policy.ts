/**
 * MLP enemy selection-pressure policy for the NGE (Neuro-evolutionary Genesis
 * Engine) extension.
 *
 * The MLP enemy is intentionally a fixed-topology, weight-and-bias-only pressure
 * source. It never adds or removes neurons/connections, and it never assimilates
 * structural or topological mutations from other genomes. Keeping the enemy
 * topology frozen makes its behavior easier to reproduce and benchmark, while
 * weight-only drift still supplies meaningful selection pressure to the player
 * swarm.
 */

/** Mutation kinds permitted under the fixed-topology MLP enemy policy. */
export type MlpEnemyMutationKind = 'weight' | 'bias';

/** Fixed-topology selection-pressure policy used for the MLP enemy in evolution. */
export interface MlpEnemySelectionPolicy {
  /** Whether the policy permits structural (add/remove node) mutations. Always false for the MLP enemy. */
  readonly allowsStructuralMutation: boolean;
  /** Whether the policy permits topological (add/remove connection) mutations. Always false for the MLP enemy. */
  readonly allowsTopologicalMutation: boolean;
  /** Mutation kinds the MLP enemy is allowed to apply. Restricted to weight and bias only. */
  readonly allowedMutationKinds: readonly MlpEnemyMutationKind[];
}

const DEFAULT_MLP_ENEMY_ALLOWED_MUTATION_KINDS: readonly MlpEnemyMutationKind[] =
  ['weight', 'bias'];

/**
 * Create the default MLP enemy selection-pressure policy.
 *
 * The returned policy disables structural and topological mutation and limits
 * the enemy to weight and bias mutations only.
 *
 * @returns A fixed-topology, weight-and-bias-only selection-pressure policy.
 */
export function createMlpEnemySelectionPolicy(): MlpEnemySelectionPolicy {
  return {
    allowsStructuralMutation: false,
    allowsTopologicalMutation: false,
    allowedMutationKinds: DEFAULT_MLP_ENEMY_ALLOWED_MUTATION_KINDS,
  };
}

/**
 * Validate that a candidate policy satisfies the MLP enemy fixed-topology
 * contract.
 *
 * @param policy - Candidate policy to validate.
 * @throws RangeError When the policy allows structural or topological mutation.
 */
export function guardMlpEnemySelectionPolicy(
  policy: MlpEnemySelectionPolicy,
): void {
  if (policy.allowsStructuralMutation) {
    throw new RangeError(
      'MLP enemy policy must not allow structural mutation.',
    );
  }

  if (policy.allowsTopologicalMutation) {
    throw new RangeError(
      'MLP enemy policy must not allow topological mutation.',
    );
  }
}
