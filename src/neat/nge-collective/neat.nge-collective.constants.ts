/**
 * Named constants for the NGE collective multi-agent system (Phase G).
 *
 * These defaults control pheromone field dynamics and snapshot pool sizing.
 * All values are opt-in; classic NEAT behavior is unaffected when the
 * collective runtime is not active.
 *
 * Consumers may override any constant by passing explicit values to the
 * relevant factory or operator function. The constants exist to document
 * the recommended starting points, not to impose fixed behaviour.
 *
 * ### Tuning guidance
 *
 * | Constant | Increase effect | Decrease effect |
 * |---|---|---|
 * | `NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR` | Signals persist longer, agents rely on older traces | Signals fade quickly, agents must reinforce paths more often |
 * | `NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE` | Signals spread wider, less spatial specificity | Signals stay local, stronger spatial gradients |
 * | `NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY` | More diverse opponent history; higher memory cost | Recency-biased selection; lower memory cost |
 */

/**
 * Default pheromone/signal decay factor applied to all cells each simulation tick.
 * A value of `0.95` means cells retain 95% of their value before the next diffusion step.
 * Lower values make signals fade faster, encouraging agents to reinforce paths more frequently.
 */
export const NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR = 0.95;

/**
 * Default lateral diffusion rate applied to all cells each simulation tick.
 * A value of `0.1` means each cell donates 10% of its value spread evenly across
 * its 4-connected grid neighbors per tick.
 */
export const NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE = 0.1;

/**
 * Default opponent snapshot pool capacity.
 * Controls how many historical opponent snapshots are retained for rolling tournament selection.
 * Older snapshots are evicted in FIFO order when the pool is at capacity.
 */
export const NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY = 5;

/**
 * Initial generation tick value assigned to every new `CollectiveEvaluationContext`.
 * Starts at zero and increments by one each time `resetCollectiveEvaluationState` is called.
 */
export const NGE_COLLECTIVE_INITIAL_GENERATION_TICK = 0;
