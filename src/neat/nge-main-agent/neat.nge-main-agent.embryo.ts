import type { NeatGenomeComputationType } from '../genome/genome.types';
import type { NgeZonePartitionConfig } from '../nge-dna/neat.nge-dna.types';
import type { NgeReproductionPolicyMode } from '../nge-dna/neat.nge-dna.types';
import { allocateEnemySubstrateCoordinates } from '../nge-dna/neat.nge-dna.coordinate-allocator';
import type {
  NgeMainAgentEmbryo,
  NgeMainAgentEmbryoArchetypeDescriptor,
  NgeMainAgentLifecycleConfig,
  NgeMainAgentTopologyBudget,
} from './neat.nge-main-agent.types';

/**
 * Default unit-cube zone partition used when the embryo builder allocates
 * substrate coordinates. Four partitions per axis gives each of the three
 * embryo archetypes distinct zone assignments under the deterministic allocator.
 */
const DEFAULT_EMBRYO_ZONE_PARTITION: NgeZonePartitionConfig = {
  x: { count: 4 },
  y: { count: 4 },
  z: { count: 4 },
};

/**
 * Initial reproduction mode for every main-agent lineage.
 *
 * The conservative default is parthenogenesis so that an unproven embryo clones
 * itself until combat-pressure hysteresis later in the lifecycle decides whether
 * to switch modes.
 */
const DEFAULT_EMBRYO_REPRODUCTION_MODE: NgeReproductionPolicyMode =
  'parthenogenesis';

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
 * The embryo carries the three allowed motif archetypes, each allocated a
 * deterministic substrate coordinate and zone, plus an initial parthenogenetic
 * reproduction mode that the hysteresis policy may later update. Node and edge
 * counts stay within the tier budget. The same config always produces the same
 * embryo, which is required for reproducible generation barriers.
 *
 * @param config - Lifecycle config with seed and tier budget.
 * @returns A deterministic embryo state ready for juvenile growth.
 */
export function buildMainAgentEmbryo(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEmbryo {
  const budget = computeEmbryoTopologyBudget(config);
  const allowlist = resolveMainAgentMotifAllowlist();

  const archetypes: NgeMainAgentEmbryoArchetypeDescriptor[] = allowlist.map(
    (computationType, index) => {
      const allocation = allocateEnemySubstrateCoordinates({
        swarmSize: allowlist.length,
        enemyIndex: index,
        seed: config.seed,
        zonePartition: DEFAULT_EMBRYO_ZONE_PARTITION,
      });

      return {
        archetypeId: `main-agent-${computationType.toLowerCase()}-${index}`,
        computationType,
        receivesCoordinates: true,
        coordinate: allocation.coordinate,
        zoneId: allocation.zoneId,
      };
    },
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
    reproductionMode: DEFAULT_EMBRYO_REPRODUCTION_MODE,
    modeIsEvolvable: true,
  };
}
