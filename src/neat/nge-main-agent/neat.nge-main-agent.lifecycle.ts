import type {
  NgeMainAgentLifecycleConfig,
  NgeMainAgentLifecycleStage,
  NgeMainAgentLifecycleState,
} from './neat.nge-main-agent.types';

/**
 * Advance one main-agent lifecycle state to the next stage.
 *
 * The stage machine follows the fixed order
 * Embryo → Juvenile → Adult → Reproducing → Embryo. The generation counter
 * increments on every call so that the full cycle is observable and
 * deterministic for the same config.
 *
 * @param state - Current lifecycle state.
 * @param config - Lifecycle config that supplies the deterministic seed.
 * @returns The next lifecycle state with the stage advanced and generation incremented.
 */
export function advanceMainAgentLifecycle(
  state: NgeMainAgentLifecycleState,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentLifecycleState {
  const transitions: Record<
    NgeMainAgentLifecycleStage,
    NgeMainAgentLifecycleStage
  > = {
    embryo: 'juvenile',
    juvenile: 'adult',
    adult: 'reproducing',
    reproducing: 'embryo',
  };

  return {
    stage: transitions[state.stage],
    generation: state.generation + 1,
    seed: config.seed,
  };
}

/**
 * Create a deterministic lifecycle runner bound to the supplied config.
 *
 * The runner is a pure function: the same input state always yields the same
 * output state, which makes generation replay and barrier tests stable.
 *
 * @param config - Lifecycle config to bind to every runner invocation.
 * @returns A function that advances a lifecycle state using the bound config.
 */
export function createMainAgentLifecycleRunner(
  config: NgeMainAgentLifecycleConfig,
): (state: NgeMainAgentLifecycleState) => NgeMainAgentLifecycleState {
  return (state) => advanceMainAgentLifecycle(state, config);
}
