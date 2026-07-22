import type {
  NgeMainAgentAdult,
  NgeMainAgentEquilibrium,
  NgeMainAgentLifecycleConfig,
  NgeMainAgentReproducing,
} from './neat.nge-main-agent.types';

/**
 * Transition a stable adult equilibrium into the reproducing stage.
 *
 * The reproducing state inherits its topology from the stable equilibrium
 * adult. It is the final stage before the lifecycle runner loops back to embryo
 * for the next generation.
 *
 * @param adult - Adult state entering reproduction.
 * @param equilibrium - Stable equilibrium candidate produced by adult optimization.
 * @param config - Lifecycle config with the deterministic seed.
 * @returns Reproducing state ready to emit the next generation.
 */
export function transitionAdultToReproducing(
  adult: NgeMainAgentAdult,
  equilibrium: NgeMainAgentEquilibrium,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproducing {
  return {
    stage: 'reproducing',
    generation: adult.generation,
    seed: config.seed,
    nodeCount: equilibrium.adult.nodeCount,
    edgeCount: equilibrium.adult.edgeCount,
    archetypes: adult.archetypes,
    schemaVersion: adult.schemaVersion,
  };
}
