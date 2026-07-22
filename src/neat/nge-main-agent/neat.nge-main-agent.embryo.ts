import type {
  NeatGenomeComputationType,
  NeatGenomeModuleArchetypeDescriptor,
} from '../genome/genome.types';
import type {
  NgeMainAgentEmbryo,
  NgeMainAgentLifecycleConfig,
  NgeMainAgentTopologyBudget,
} from './neat.nge-main-agent.types';

/**
 * Resolve the exact three existing catalogue motifs allowed for the main agent.
 *
 * The main-agent motif set is deliberately minimal and does not introduce any
 * new computation types or schema versions. All returned values are already
 * present in {@link NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE}.
 *
 * @returns A readonly array containing AttentionHead, GatedRecurrentCell, and EpisodicSlot.
 */
export function resolveMainAgentMotifAllowlist(): readonly NeatGenomeComputationType[] {
  return ['AttentionHead', 'GatedRecurrentCell', 'EpisodicSlot'];
}

/**
 * Compute the topology budget for an embryo, capping it at the tier budget.
 *
 * The embryo is intentionally small relative to the tier cap so that juvenile
 * growth and adult pruning have meaningful headroom without risking overflow.
 *
 * @param config - Lifecycle config with the configured tier cap.
 * @returns A positive topology budget bounded by the tier cap.
 */
export function computeEmbryoTopologyBudget(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentTopologyBudget {
  return {
    maxNodes: Math.max(1, Math.min(config.maxNodes, 64)),
    maxEdges: Math.max(1, Math.min(config.maxEdges, 256)),
  };
}

/**
 * Build a deterministic main-agent embryo state.
 *
 * The embryo carries the three allowed motif archetypes, a node and edge count
 * within the tier budget, and the canonical schema version A.1.0. The same
 * config always produces the same embryo, which is required for reproducible
 * generation barriers.
 *
 * @param config - Lifecycle config with seed and tier budget.
 * @returns A deterministic embryo state ready for juvenile growth.
 */
export function buildMainAgentEmbryo(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEmbryo {
  const budget = computeEmbryoTopologyBudget(config);
  const allowlist = resolveMainAgentMotifAllowlist();

  const archetypes: NeatGenomeModuleArchetypeDescriptor[] = allowlist.map(
    (computationType, index) => ({
      archetypeId: `main-agent-${computationType.toLowerCase()}-${index}`,
      computationType,
      receivesCoordinates: true,
    }),
  );

  const nodeCount = Math.min(budget.maxNodes, allowlist.length);
  const edgeCount = Math.min(
    budget.maxEdges,
    Math.max(1, nodeCount * (nodeCount - 1)),
  );

  return {
    stage: 'embryo',
    generation: 0,
    seed: config.seed,
    nodeCount,
    edgeCount,
    archetypes,
    schemaVersion: 'A.1.0',
  };
}
