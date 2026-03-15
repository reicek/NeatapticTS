import * as methods from '../../methods/methods';

/**
 * Shared feed-forward topology-intent policy for NEAT population entry points.
 *
 * This chapter keeps the feed-forward promotion contract in one small shared
 * boundary because both pool bootstrapping and next-population provenance need
 * to answer the same question: when does a genome already satisfy the stricter
 * feed-forward runtime assumptions exposed by the public mutation policy?
 *
 * Example:
 * ```ts
 * const shouldPromote = usesFeedForwardMutationPolicy(neat.options.mutation);
 * promoteGenomeToFeedForwardIntentWhenEligible(genome, shouldPromote);
 * ```
 */

/**
 * Minimal mutation descriptor used by topology-intent helpers.
 */
export interface TopologyIntentMutationMethod {
  /** Optional mutation name used for canonical pool comparison. */
  name?: string;
  /** Additional mutation metadata. */
  [key: string]: unknown;
}

/**
 * Minimal genome surface required to promote feed-forward intent safely.
 */
export interface TopologyIntentGenome {
  /** Runtime node array in current graph order. */
  nodes?: unknown[];
  /** Runtime directed connection list. */
  connections?: Array<{ from: unknown; to: unknown }>;
  /** Runtime gated connection list. */
  gates?: unknown[];
  /** Runtime self-connection list. */
  selfconns?: unknown[];
  /** Public topology intent setter exposed by `Network`. */
  setTopologyIntent?: (
    topologyIntent: 'feed-forward' | 'unconstrained',
  ) => void;
}

/**
 * Determine whether the configured mutation policy communicates feed-forward intent.
 *
 * Accepts the canonical mutation pool reference, the legacy single-item array
 * wrapper, or a flattened pool that exactly matches the canonical feed-forward
 * operator order. The comparison stays strict on purpose so the controller does
 * not silently reinterpret custom mutation pools as feed-forward mode.
 *
 * @param mutationConfig Configured mutation option.
 * @returns True when the option expresses canonical feed-forward intent.
 */
export function usesFeedForwardMutationPolicy(
  mutationConfig:
    | TopologyIntentMutationMethod
    | TopologyIntentMutationMethod[]
    | unknown,
): boolean {
  // Step 1: Accept the canonical direct FFW reference.
  if (mutationConfig === methods.mutation.FFW) {
    return true;
  }

  // Step 2: Accept nested legacy `[methods.mutation.FFW]` wrappers.
  if (
    Array.isArray(mutationConfig) &&
    mutationConfig.length === 1 &&
    mutationConfig[0] === methods.mutation.FFW
  ) {
    return true;
  }

  // Step 3: Accept flattened pools that match the canonical FFW operator names.
  if (!Array.isArray(mutationConfig)) {
    return false;
  }

  return matchesCanonicalFeedForwardPool(
    mutationConfig as TopologyIntentMutationMethod[],
    methods.mutation.FFW as TopologyIntentMutationMethod[],
  );
}

/**
 * Promote a genome to feed-forward topology intent when the structure is eligible.
 *
 * The promotion is intentionally conservative. The helper only switches the
 * runtime contract when the current graph is already ordered like a pure
 * feed-forward network, which avoids changing the meaning of recurrent or
 * gated seed genomes during bootstrapping and provenance insertion.
 *
 * @param genome Genome candidate being inserted into a population.
 * @param shouldPromote Whether the active NEAT options request feed-forward semantics.
 * @returns Nothing.
 */
export function promoteGenomeToFeedForwardIntentWhenEligible(
  genome: TopologyIntentGenome,
  shouldPromote: boolean,
): void {
  // Step 1: Skip work when the configured mutation policy is not FFW.
  if (!shouldPromote) {
    return;
  }

  // Step 2: Skip work when the genome cannot safely adopt feed-forward intent.
  if (!isGenomeEligibleForFeedForwardIntentPromotion(genome)) {
    return;
  }

  // Step 3: Apply the public topology contract through the network API.
  genome.setTopologyIntent?.('feed-forward');
}

/**
 * Check whether a configured mutation pool matches the canonical FFW pool.
 *
 * @param configuredPool Mutation pool configured on the NEAT instance.
 * @param canonicalPool Canonical feed-forward mutation pool.
 * @returns True when both pools align by operator name and order.
 */
function matchesCanonicalFeedForwardPool(
  configuredPool: TopologyIntentMutationMethod[],
  canonicalPool: TopologyIntentMutationMethod[],
): boolean {
  // Step 1: Reject shape mismatches early.
  if (configuredPool.length !== canonicalPool.length) {
    return false;
  }

  // Step 2: Ensure the flattened pool matches the canonical FFW ordering.
  return configuredPool.every((configuredMethod, methodIndex) => {
    return configuredMethod?.name === canonicalPool[methodIndex]?.name;
  });
}

/**
 * Check whether a genome can safely adopt feed-forward topology intent.
 *
 * Eligibility is intentionally conservative: the graph must already be free of
 * gates and self-connections, and every normal connection must follow the
 * current node ordering. This keeps the helper focused on preserving an
 * already-feed-forward structure instead of rewriting arbitrary graphs into a
 * different interpretation.
 *
 * @param genome Genome candidate.
 * @returns True when the genome can safely adopt feed-forward intent.
 */
function isGenomeEligibleForFeedForwardIntentPromotion(
  genome: TopologyIntentGenome,
): boolean {
  // Step 1: Validate that topology collections exist and that no recurrent-only
  // features are currently active.
  if (
    !Array.isArray(genome.nodes) ||
    !Array.isArray(genome.connections) ||
    !Array.isArray(genome.gates) ||
    !Array.isArray(genome.selfconns)
  ) {
    return false;
  }

  if (genome.gates.length > 0 || genome.selfconns.length > 0) {
    return false;
  }

  // Step 2: Require every connection to follow the current node ordering.
  const nodeIndexByReference = new Map<unknown, number>();
  genome.nodes.forEach((node, nodeIndex) => {
    nodeIndexByReference.set(node, nodeIndex);
  });

  return genome.connections.every((connection) => {
    const sourceNodeIndex = nodeIndexByReference.get(connection.from);
    const targetNodeIndex = nodeIndexByReference.get(connection.to);

    return (
      sourceNodeIndex !== undefined &&
      targetNodeIndex !== undefined &&
      sourceNodeIndex < targetNodeIndex
    );
  });
}
