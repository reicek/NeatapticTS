import type { NeatLike } from './neat.types';
import Network from '../architecture/network';
import {
  promoteGenomeToFeedForwardIntentWhenEligible,
  usesFeedForwardMutationPolicy,
} from './neat.topology-intent.utils';

/**
 * Genome with NEAT-specific metadata and methods.
 */
interface GenomeWithMetadata {
  score?: number;
  _reenableProb?: number;
  _id?: number;
  _parents?: number[];
  _depth?: number;
  clone?: () => GenomeWithMetadata;
  toJSON?: () => Record<string, unknown>;
  mutate?: (method: MutationMethod) => void;
  setTopologyIntent?: (topologyIntent: 'feed-forward' | 'unconstrained') => void;
  [key: string]: unknown;
}

/**
 * Mutation method with optional name.
 */
interface MutationMethod {
  name?: string;
  [key: string]: unknown;
}

/**
 * NEAT controller interface for helper functions.
 */
interface NeatControllerForHelpers {
  input: number;
  output: number;
  population: GenomeWithMetadata[];
  options: {
    reenableProb?: number;
    popsize?: number;
    minHidden?: number;
    [key: string]: unknown;
  };
  _nextGenomeId: number;
  _lineageEnabled?: boolean;
  _getRNG: () => () => number;
  ensureMinHiddenNodes?: (genome: GenomeWithMetadata) => void;
  ensureNoDeadEnds?: (genome: GenomeWithMetadata) => void;
  selectMutationMethod?: (
    genome: GenomeWithMetadata,
    sexual: boolean,
  ) => MutationMethod | MutationMethod[];
  _invalidateGenomeCaches?: (genome: GenomeWithMetadata) => void;
}

/**
 * Helper utilities that augment the core NEAT (NeuroEvolution of Augmenting Topologies)
 * implementation. These functions are kept separate from the main class so they can
 * be tree‑shaken when unused and independently documented for educational purposes.
 *
 * The helpers focus on three core lifecycle operations:
 * 1. Spawning children from an existing parent genome with mutation ("sexual" reproduction not handled here).
 * 2. Registering externally created genomes so lineage & invariants remain consistent.
 * 3. Creating the initial population pool (bootstrapping evolution) either from a seed
 *    network or by synthesizing fresh minimal networks.
 *
 * All helpers expect to be invoked with a `this` context that matches `NeatLike`.
 * They intentionally use defensive try/catch blocks to avoid aborting broader
 * evolutionary runs when an individual genome operation fails; this mirrors the
 * tolerant/robust nature of many historical NEAT library implementations.
 */

/**
 * Spawn (clone & mutate) a child genome from an existing parent genome.
 *
 * The returned child is intentionally NOT auto‑inserted into the population;
 * call {@link addGenome} (or the class method wrapper) once you decide to
 * keep it. This separation allows callers to perform custom validation or
 * scoring heuristics before committing the child genome.
 *
 * Evolutionary rationale:
 * - Cloning preserves the full topology & weights of the parent.
 * - A configurable number of mutation passes are applied sequentially; each
 *   pass may alter structure (add/remove nodes / connections) or weights.
 * - Lineage annotations (`_parents`, `_depth`) enable later analytics (e.g.,
 *   diversity statistics, genealogy visualization, pruning heuristics).
 *
 * Robustness philosophy: individual mutation failures are silently ignored so
 * a single stochastic edge case (e.g., no valid structural mutation) does not
 * derail evolutionary progress.
 *
 * @param this Bound NEAT instance (inferred when used as a method).
 * @param parentGenome Parent genome/network to clone. Must implement either
 * `clone()` OR a pair of `toJSON()` / static `fromJSON()` for deep copying.
 * @param mutateCount Number of sequential mutation operations to attempt; each
 *        iteration chooses a mutation method using the instance's selection logic.
 *        Defaults to 1 for conservative structural drift.
 * @returns A new genome (unregistered) whose score is reset and whose lineage
 *          metadata references the parent.
 * @example
 * ```ts
 * // Assume `neat` is an instance implementing NeatLike and `parent` is a genome in neat.population
 * const child = neat.spawnFromParent(parent, 3); // apply 3 mutation passes
 * // Optionally inspect / filter the child before adding
 * neat.addGenome(child, [parent._id]);
 * ```
 */
export async function spawnFromParent(
  this: NeatLike,
  parentGenome: GenomeWithMetadata,
  mutateCount: number = 1,
): Promise<GenomeWithMetadata> {
  const internal = this as unknown as NeatControllerForHelpers;

  // Step 1: Deep clone the parent (prefer direct clone() for performance).
  let clone: GenomeWithMetadata;
  if (parentGenome.clone) {
    clone = parentGenome.clone();
  } else {
    const { default: NetworkClass } = await import('../architecture/network');
    clone = NetworkClass.fromJSON(
      parentGenome.toJSON?.() ?? {},
    ) as unknown as GenomeWithMetadata;
  }

  // Step 2: Reset evaluation state for the fresh offspring.
  clone.score = undefined;
  clone._reenableProb = internal.options.reenableProb;
  clone._id = internal._nextGenomeId++;

  // Step 3: Record minimal lineage (single direct parent) and generation depth.
  clone._parents = [parentGenome._id ?? 0];
  clone._depth = (parentGenome._depth ?? 0) + 1;

  // Step 4: Enforce structural invariants (minimum hidden nodes, no dead ends).
  internal.ensureMinHiddenNodes?.(clone);
  internal.ensureNoDeadEnds?.(clone);

  // Step 5: Apply the requested number of mutation passes.
  for (let mutationIndex = 0; mutationIndex < mutateCount; mutationIndex++) {
    try {
      // Select a mutation operator; may return a single method or an array of candidates.
      let selectedMutationMethod = await internal.selectMutationMethod?.(
        clone,
        false,
      );
      if (Array.isArray(selectedMutationMethod)) {
        const candidateMutations = selectedMutationMethod;
        selectedMutationMethod =
          candidateMutations[
            Math.floor(internal._getRNG()() * candidateMutations.length)
          ];
      }
      // Execute mutation if a valid operator with a name (convention) is present.
      if (selectedMutationMethod && selectedMutationMethod.name) {
        clone.mutate?.(selectedMutationMethod);
      }
    } catch {
      // Intentionally ignore individual mutation failures to keep evolution moving.
    }
  }

  // Step 6: Invalidate any cached compatibility / distance metrics tied to the genome.
  internal._invalidateGenomeCaches?.(clone);
  return clone;
}

/**
 * Register an externally constructed genome (e.g., deserialized, custom‑built,
 * or imported from another run) into the active population. Ensures lineage
 * metadata and structural invariants are consistent with internally spawned
 * genomes.
 *
 * Defensive design: If invariant enforcement fails, the genome is still added
 * (best effort) so experiments remain reproducible and do not abort mid‑run.
 * Caller can optionally inspect or prune later during evaluation.
 *
 * @param this Bound NEAT instance.
 * @param genome Genome / network object to insert. Mutated in place to add
 *        internal metadata fields (`_id`, `_parents`, `_depth`, `_reenableProb`).
 * @param parents Optional explicit list of parent genome IDs (e.g., 2 parents
 *        for crossover). If omitted, lineage metadata is left empty.
 * @example
 * ```ts
 * const imported = Network.fromJSON(saved);
 * neat.addGenome(imported, [parentA._id, parentB._id]);
 * ```
 */
export function addGenome(
  this: NeatLike,
  genome: GenomeWithMetadata,
  parents?: number[],
): void {
  const internal = this as unknown as NeatControllerForHelpers;

  try {
    // Step 1: Reset score so future evaluations are not biased by stale values.
    genome.score = undefined;
    genome._reenableProb = internal.options.reenableProb;
    genome._id = internal._nextGenomeId++;

    // Step 2: Copy lineage from provided parent IDs (if any).
    genome._parents = Array.isArray(parents) ? parents.slice() : [];
    genome._depth = 0;
    if (genome._parents.length) {
      // Compute depth = (max parent depth) + 1 for genealogical layering.
      const parentDepths = genome._parents
        .map((pid: number) =>
          internal.population.find((g: GenomeWithMetadata) => g._id === pid),
        )
        .filter((g): g is GenomeWithMetadata => g !== undefined)
        .map((g) => g._depth ?? 0);
      genome._depth = parentDepths.length ? Math.max(...parentDepths) + 1 : 1;
    }

    // Step 3: Ensure structural invariants.
    internal.ensureMinHiddenNodes?.(genome);
    internal.ensureNoDeadEnds?.(genome);

    // Step 4: Invalidate caches & persist.
    internal._invalidateGenomeCaches?.(genome);
    internal.population.push(genome);
  } catch {
    // Fallback: still add genome so the evolutionary run can continue.
    internal.population.push(genome);
  }
}

/**
 * Create (or reset) the initial population pool for a NEAT run.
 *
 * If a `seedNetwork` is supplied, every genome is a structural + weight clone
 * of that seed. This is useful for transfer learning or continuing evolution
 * from a known good architecture. When omitted, brand‑new minimal networks are
 * synthesized using the configured input/output sizes (and optional minimum
 * hidden layer size).
 *
 * Design notes:
 * - Population size is derived from `options.popsize` (default 50).
 * - Each genome gets a unique sequential `_id` for reproducible lineage.
 * - When lineage tracking is enabled (`_lineageEnabled`), parent & depth fields
 *   are initialized for later analytics.
 * - Structural invariant checks are best effort. A single failure should not
 *   prevent other genomes from being created, hence broad try/catch blocks.
 *
 * @param this Bound NEAT instance.
 * @param seedNetwork Optional prototype network to clone for every initial genome.
 * @example
 * ```ts
 * // Basic: create 50 fresh minimal networks
 * neat.createPool(null);
 *
 * // Seeded: start with a known topology
 * const seed = new Network(neat.input, neat.output, { minHidden: 4 });
 * neat.createPool(seed);
 * ```
 */
export function createPool(
  this: NeatLike,
  seedNetwork: GenomeWithMetadata | null,
): void {
  const internal = this as unknown as NeatControllerForHelpers;

  try {
    // Step 1: Reset population container.
    internal.population = [];
    const poolSize = internal.options?.popsize ?? 50;
    const shouldPromoteFeedForwardIntent = usesFeedForwardMutationPolicy(
      internal.options?.mutation,
    );

    // Step 2: Generate each initial genome.
    for (let genomeIndex = 0; genomeIndex < poolSize; genomeIndex++) {
      // Clone from seed OR build a fresh network.
      const genomeCopy = seedNetwork
        ? (Network.fromJSON(
            seedNetwork.toJSON?.() ?? {},
          ) as unknown as GenomeWithMetadata)
        : (new Network(internal.input, internal.output, {
            minHidden: internal.options?.minHidden,
          }) as unknown as GenomeWithMetadata);

      // Step 2a: Ensure no stale scoring information.
      genomeCopy.score = undefined;

      // Step 2a.1: Promote feed-forward topology intent when the policy and topology agree.
      promoteGenomeToFeedForwardIntentWhenEligible(
        genomeCopy,
        shouldPromoteFeedForwardIntent,
      );

      // Step 2b: Attempt structural invariant enforcement (best effort).
      try {
        internal.ensureNoDeadEnds?.(genomeCopy);
      } catch {
        // Ignored; genome may still be viable or corrected by later mutations.
      }

      // Step 2c: Annotate runtime metadata.
      genomeCopy._reenableProb = internal.options.reenableProb;
      genomeCopy._id = internal._nextGenomeId++;
      if (internal._lineageEnabled) {
        genomeCopy._parents = [];
        genomeCopy._depth = 0;
      }

      // Step 2d: Insert into population.
      internal.population.push(genomeCopy);
    }
  } catch {
    // Swallow: partial population is acceptable; caller may decide to refill or continue.
  }
}
