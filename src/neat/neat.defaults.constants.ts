import type { NeatConstructorDefaults } from './init/neat.init';

/**
 * Root chapter map and public default knobs for the internal `src/neat` controller surface.
 *
 * `src/neat.ts` is the public control desk. `src/neat/` is the machine room
 * behind it. This folder owns the controller chapters that make a run real:
 * initialization, evaluation, evolution, mutation, speciation, telemetry,
 * persistence, and the shared bookkeeping that keeps experiments deterministic
 * instead of magical. Promoting the defaults file to the chapter opening is
 * intentional because default knobs are the quickest way to make the
 * controller's personality legible before a reader dives into implementation
 * detail.
 *
 * The most helpful reading move is to split the folder into four working
 * lanes. `init/`, `evaluate/`, and `evolve/` explain how a population is
 * created, scored, and replaced. `mutation/`, `selection/`, and `speciation/`
 * explain how search pressure and diversity are managed. `telemetry/`,
 * `lineage/`, `diversity/`, and `multiobjective/` explain how the run becomes
 * inspectable. `export/`, `rng/`, `cache/`, `maintenance/`, and `pruning/`
 * explain how the controller stays reproducible and tractable as experiments
 * get larger.
 *
 * The flat root files are small bridges rather than the whole story.
 * `neat.defaults.constants.ts` and `neat.types.ts` keep the public constructor
 * and option bag readable. `neat.lineage.ts` and `neat.constants.ts` keep
 * small shared logic close to the root when multiple chapters need it. The
 * deeper folders own the heavier policy and runtime details.
 *
 * These defaults matter because they are the baseline promises the controller
 * makes when a caller says "give me an ordinary NEAT run." Population size,
 * mutation tempo, compatibility pressure, structural ceilings, and
 * observability sampling are not random numbers. They are the quiet assumptions
 * that decide whether the controller behaves like a conservative search, an
 * exploratory search, or an unstable one.
 *
 * That design follows the original NEAT intuition: protect structural
 * innovation long enough for it to compete, rather than forcing every new
 * topology to beat established species immediately. See Stanley and
 * Miikkulainen,
 * [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02),
 * for the background behind the compatibility and growth vocabulary that keeps
 * surfacing across this folder.
 *
 * Canonical NEAT vs extensions vs experiments:
 *
 * - Canonical NEAT (default mental model): historical markings (innovation ids),
 *   speciation pressure, and crossover alignment by innovation number. This is the
 *   core contract that makes “different topologies can still mate” work.
 * - Repo-specific extensions (opt-in features): additional controller lanes such as
 *   recurrent/gated allowances, multiobjective policy, novelty tracking, pruning,
 *   or richer telemetry. These should preserve the canonical identity rules even
 *   when they add new operators or metrics.
 * - Experimental research features: best-effort lanes that are intentionally marked
 *   as experimental (for example ONNX heuristics or experimental layer builders).
 *   Treat these as evolving prototypes: useful for exploration, but not guaranteed
 *   to match the strict replay or correctness bar of the canonical core.
 *
 * Read this root chapter in three passes. Start with this defaults file and
 * `neat.types.ts` for the public knobs and broad contracts. Continue into
 * `evaluate/`, `evolve/`, and `speciation/` for the live search loop. Finish
 * with `telemetry/`, `lineage/`, `multiobjective/`, `export/`, and `rng/` when
 * you want to inspect, replay, or compare runs rather than only advance them.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Root["src/neat root"]:::accent --> Loop["init / evaluate / evolve"]:::base
 *   Root --> Pressure["mutation / selection / speciation"]:::base
 *   Root --> Observe["telemetry / lineage / diversity / multiobjective"]:::base
 *   Root --> Replay["export / rng / cache / maintenance / pruning"]:::base
 *   Root --> Bridges["root bridges<br/>defaults / types / constants"]:::base
 * ```
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Defaults[Root defaults]:::accent --> Population[popsize elitism provenance]:::base
 *   Defaults --> Variation[mutationRate mutationAmount]:::base
 *   Defaults --> Species[compatibility and weight coefficients]:::base
 *   Defaults --> Observation[diversity and novelty samples]:::base
 *   Population --> Run[Controller behavior]:::base
 *   Variation --> Run
 *   Species --> Run
 *   Observation --> Run
 * ```
 *
 * Example: build one explicit baseline options bag from the documented root
 * defaults.
 *
 * ```ts
 * const baselineOptions = {
 *   popsize: DEFAULT_POPULATION_SIZE,
 *   mutationRate: DEFAULT_MUTATION_RATE,
 *   compatibilityThreshold: DEFAULT_COMPATIBILITY_THRESHOLD,
 * };
 * ```
 *
 * Example: keep the observability defaults visible when teaching or
 * benchmarking runs.
 *
 * ```ts
 * const observabilityDefaults = {
 *   diversityPairSample: DEFAULT_DIVERSITY_PAIR_SAMPLE,
 *   diversityGraphletSample: DEFAULT_DIVERSITY_GRAPHLET_SAMPLE,
 *   noveltyK: DEFAULT_NOVELTY_K,
 * };
 * ```
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
