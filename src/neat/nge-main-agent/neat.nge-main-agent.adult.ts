import type {
  NgeMainAgentAdult,
  NgeMainAgentEquilibrium,
  NgeMainAgentJuvenile,
  NgeMainAgentLifecycleConfig,
} from './neat.nge-main-agent.types';

/**
 * Prune a juvenile topology down to adult limits while remaining within budget.
 *
 * The adult stage may reduce or cap node and edge counts, but it never allows
 * them to exceed the configured tier budget.
 *
 * @param juvenile - Juvenile state to prune.
 * @param config - Lifecycle config with the tier budget cap.
 * @returns Adult state whose counts are bounded by the tier budget.
 */
export function pruneAdultTopology(
  juvenile: NgeMainAgentJuvenile,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult {
  return {
    stage: 'adult',
    generation: juvenile.generation,
    seed: config.seed,
    nodeCount: Math.min(juvenile.nodeCount, config.maxNodes),
    edgeCount: Math.min(juvenile.edgeCount, config.maxEdges),
    archetypes: juvenile.archetypes,
    schemaVersion: juvenile.schemaVersion,
  };
}

/**
 * Evaluate whether an adult topology is within the configured tier budget.
 *
 * @param adult - Adult state to evaluate.
 * @param config - Lifecycle config with the tier budget.
 * @returns An evaluation object whose `withinBudget` flag is true when the adult respects both caps.
 */
export function evaluateAdultTopologyBudget(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): { withinBudget: boolean } {
  return {
    withinBudget:
      adult.nodeCount <= config.maxNodes && adult.edgeCount <= config.maxEdges,
  };
}

/**
 * Run adult optimization until an equilibrium candidate is stable.
 *
 * This first-pass implementation treats the pruned adult as already stable so
 * that the lifecycle runner can be tested end-to-end. Later phases will replace
 * this with iterative equilibrium detection.
 *
 * @param adult - Adult state to optimize.
 * @param _config - Lifecycle config (reserved for future equilibrium parameters).
 * @returns Equilibrium candidate wrapping the stable adult.
 */
export function runAdultEquilibrium(
  adult: NgeMainAgentAdult,
  _config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibrium {
  // Reserved for future equilibrium parameters (e.g., stability threshold).
  void _config;

  return {
    isStable: true,
    adult,
  };
}
