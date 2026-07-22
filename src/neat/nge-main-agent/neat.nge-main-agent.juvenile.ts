import type {
  NgeMainAgentAdult,
  NgeMainAgentEmbryo,
  NgeMainAgentJuvenile,
  NgeMainAgentLifecycleConfig,
} from './neat.nge-main-agent.types';

/**
 * Grow the embryo topology into a juvenile state while staying within budget.
 *
 * Juvenile growth is deterministic and never shrinks the network below the
 * embryo size. The growth factor is derived from the embryo's seed so that the
 * same embryo always produces the same juvenile.
 *
 * @param embryo - Embryo state to grow.
 * @param config - Lifecycle config with the tier budget cap.
 * @returns Juvenile state with node and edge counts within the tier budget.
 */
export function growJuvenileTopology(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentJuvenile {
  const growthFactor = 1 + Math.min(0.2, embryo.seed / 1_000_000);
  const nodeCount = Math.min(
    config.maxNodes,
    Math.max(embryo.nodeCount, Math.floor(embryo.nodeCount * growthFactor)),
  );
  const edgeCount = Math.min(
    config.maxEdges,
    Math.max(embryo.edgeCount, Math.floor(embryo.edgeCount * growthFactor)),
  );

  return {
    stage: 'juvenile',
    generation: embryo.generation,
    seed: config.seed,
    nodeCount,
    edgeCount,
    archetypes: embryo.archetypes,
    schemaVersion: embryo.schemaVersion,
  };
}

/**
 * Mature a juvenile or embryo state into an adult state.
 *
 * The transition is a deterministic stage change that preserves the input
 * topology. Adult pruning is handled separately by {@link pruneAdultTopology}.
 *
 * @param juvenile - Juvenile or embryo state to mature.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Adult state with the same topology as the input.
 */
export function transitionJuvenileToAdult(
  juvenile: NgeMainAgentJuvenile | NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult {
  return {
    stage: 'adult',
    generation: juvenile.generation,
    seed: config.seed,
    nodeCount: juvenile.nodeCount,
    edgeCount: juvenile.edgeCount,
    archetypes: juvenile.archetypes,
    schemaVersion: juvenile.schemaVersion,
  };
}
