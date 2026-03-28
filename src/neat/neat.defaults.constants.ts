import type { NeatConstructorDefaults } from './init/neat.init';

/**
 * Public default knobs for the root `Neat` controller.
 *
 * These constants describe the controller personality a caller gets before a
 * custom option bag starts bending the run toward a different search style.
 * Keeping them in their own root chapter lets `src/neat.ts` stay focused on
 * orchestration while the `init/` chapter consumes one shared defaults packet.
 *
 * The constants fall into four small families:
 *
 * - search volume and tempo,
 * - speciation pressure,
 * - structural ceilings,
 * - observability sampling.
 *
 * Read them as the public defaults shelf, not as hidden implementation trivia.
 * These values are the baseline promises the root controller makes when a user
 * says, "give me an ordinary NEAT run," without specifying every knob.
 */

/**
 * Default population size when caller does not specify `popsize`.
 *
 * This opens the root defaults shelf's search-volume family. It controls how
 * many genomes compete in each generation before elitism, provenance, or
 * mutation pressure begin to reshape the population.
 */
export const DEFAULT_POPULATION_SIZE = 50;

/**
 * Default elitism count applied when unspecified.
 *
 * Read this beside {@link DEFAULT_POPULATION_SIZE} and
 * {@link DEFAULT_PROVENANCE}: the trio defines how much of each generation is
 * reserved for carry-over, how much is freshly injected, and how much capacity
 * remains for ordinary offspring.
 */
export const DEFAULT_ELITISM = 0;

/**
 * Default provenance count applied when unspecified.
 *
 * Provenance is the root controller's small "fresh seed" policy. A value of
 * `0` means the default run does not spend population budget on extra
 * generation-zero style injections unless the caller asks for them.
 */
export const DEFAULT_PROVENANCE = 0;

/**
 * Default mutation rate used by the root controller when no explicit rate is supplied.
 *
 * This belongs to the same search-tempo family as
 * {@link DEFAULT_MUTATION_AMOUNT}. Together they define how often mutation is
 * attempted and how many mutation steps a genome can receive once mutation is
 * active.
 */
export const DEFAULT_MUTATION_RATE = 0.7;

/**
 * Default number of mutation operations applied per genome.
 *
 * The default keeps the baseline search policy conservative: most runs mutate
 * often enough to keep topology moving, but each genome usually pays for only
 * one structural or parametric change per mutation pass.
 */
export const DEFAULT_MUTATION_AMOUNT = 1;

/**
 * Default compatibility threshold controlling speciation distance.
 *
 * This starts the speciation-pressure family of defaults. It is the neutral
 * boundary the controller uses before adaptive tuning or custom settings make
 * species splits stricter or more permissive.
 */
export const DEFAULT_COMPATIBILITY_THRESHOLD = 3;

/**
 * Default maximum allowed nodes where `Infinity` means unbounded growth.
 *
 * Read the three `DEFAULT_MAX_*` exports as one structural-ceiling family.
 * Leaving them unbounded by default tells the root controller to rely on
 * mutation policy, pruning, and adaptive limits instead of an immediate hard
 * cap.
 */
export const DEFAULT_MAX_NODES = Infinity;

/**
 * Default maximum allowed connections where `Infinity` means unbounded growth.
 *
 * This preserves the same baseline policy as {@link DEFAULT_MAX_NODES}: the
 * controller does not impose a fixed connection ceiling unless the caller wants
 * one.
 */
export const DEFAULT_MAX_CONNS = Infinity;

/**
 * Default maximum allowed gates where `Infinity` means unbounded growth.
 *
 * Gate limits stay in the same family as node and connection limits so the
 * whole structural-cap story remains consistent at the root surface.
 */
export const DEFAULT_MAX_GATES = Infinity;

/**
 * Default excess coefficient for NEAT compatibility distance.
 *
 * This begins the root compatibility-weight family. These coefficients explain
 * which kinds of genome disagreement matter most when the controller decides
 * whether two genomes still belong in the same species neighborhood.
 */
export const DEFAULT_EXCESS_COEFF = 1;

/**
 * Default disjoint coefficient for NEAT compatibility distance.
 *
 * Matching the excess coefficient by default gives the root controller a
 * balanced structural view: excess and disjoint innovation gaps both count as
 * first-class evidence during compatibility comparisons.
 */
export const DEFAULT_DISJOINT_COEFF = 1;

/**
 * Default average weight difference coefficient for compatibility distance.
 *
 * This keeps parameter drift relevant without letting weight deltas dominate
 * the whole speciation read. In the default family, topology disagreement still
 * carries more weight than modest edge-weight differences.
 */
export const DEFAULT_WEIGHT_DIFF_COEFF = 0.5;

/**
 * Default pair-sample size used by diversity metrics in fast mode.
 *
 * This starts the observability-sampling family. The root controller uses a
 * bounded sample instead of exhaustive pair checks so diversity reads stay
 * cheap enough for ordinary runs.
 */
export const DEFAULT_DIVERSITY_PAIR_SAMPLE = 20;

/**
 * Default graphlet sample size used by diversity metrics in fast mode.
 *
 * Read this beside {@link DEFAULT_DIVERSITY_PAIR_SAMPLE}: pair samples give the
 * controller quick distance evidence, while graphlet samples provide a small
 * structural texture read without forcing whole-population analysis.
 */
export const DEFAULT_DIVERSITY_GRAPHLET_SAMPLE = 30;

/**
 * Default neighbor count for novelty search when `k` is unspecified.
 *
 * This closes the root observability-and-exploration shelf. It controls how
 * many nearby behaviors contribute to novelty before the caller tunes novelty
 * search more explicitly.
 */
export const DEFAULT_NOVELTY_K = 5;

/**
 * Shared defaults packet consumed by the constructor bootstrap chapter.
 *
 * The root public surface still exports the individual constants for callers
 * and docs, but the constructor now hands one named packet to `init/` instead
 * of rebuilding the same object inline inside `src/neat.ts`.
 */
export const DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS: NeatConstructorDefaults = {
  populationSize: DEFAULT_POPULATION_SIZE,
  elitism: DEFAULT_ELITISM,
  provenance: DEFAULT_PROVENANCE,
  mutationRate: DEFAULT_MUTATION_RATE,
  mutationAmount: DEFAULT_MUTATION_AMOUNT,
  compatibilityThreshold: DEFAULT_COMPATIBILITY_THRESHOLD,
  maxNodes: DEFAULT_MAX_NODES,
  maxConns: DEFAULT_MAX_CONNS,
  maxGates: DEFAULT_MAX_GATES,
  excessCoeff: DEFAULT_EXCESS_COEFF,
  disjointCoeff: DEFAULT_DISJOINT_COEFF,
  weightDiffCoeff: DEFAULT_WEIGHT_DIFF_COEFF,
  diversityPairSample: DEFAULT_DIVERSITY_PAIR_SAMPLE,
  diversityGraphletSample: DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
  noveltyK: DEFAULT_NOVELTY_K,
};
