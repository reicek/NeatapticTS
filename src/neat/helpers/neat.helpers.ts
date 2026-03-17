import type { NeatLike } from '../shared/neat.shared.types';
import Network from '../../architecture/network';
import {
  promoteGenomeToFeedForwardIntentWhenEligible,
  usesFeedForwardMutationPolicy,
} from '../topology-intent/neat.topology-intent';

/**
 * Helper utilities for the shared NEAT controller lifecycle.
 *
 * This chapter owns the population-entry boundary for the shared NEAT
 * controller. The surrounding chapters decide how genomes should be evaluated,
 * ranked, mutated, or speciated once they are already alive inside the
 * population; this boundary answers the earlier provenance question: how does a
 * genome first become part of that live population, and what metadata must be
 * attached before later controller phases can trust it?
 *
 * The three public helpers cover the full entry story:
 *
 * 1. `createPool()` bootstraps the first generation from either fresh minimal
 *    networks or a supplied seed topology.
 * 2. `spawnFromParent()` creates a provisional child that still needs an
 *    explicit keep-or-discard decision.
 * 3. `addGenome()` registers an externally sourced or newly accepted genome so
 *    lineage, cache, and structural invariants match the rest of the run.
 *
 * Keeping those paths together prevents subtle drift in `_id`, `_parents`,
 * `_depth`, `_reenableProb`, feed-forward intent, and cache invalidation rules.
 * The public `Neat` facade still exposes the same methods, but this file now
 * reads as one small chapter about safe population entry instead of a grab bag
 * of leftover helpers.
 */

/**
 * Minimal genome contract required by the population-entry helpers.
 *
 * This interface deliberately stops short of the full `Network` surface. The
 * helpers only need enough capability to clone or serialize a genome, apply a
 * mutation, and attach the small amount of controller-owned metadata that later
 * chapters rely on for lineage, pruning, telemetry, and deterministic replay.
 *
 * Read this as the runtime envelope around a genome while it is crossing the
 * boundary into the live population. Once the genome is registered, richer
 * controller chapters can treat `_id`, `_parents`, `_depth`, and
 * `_reenableProb` as already normalized.
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
  setTopologyIntent?: (
    topologyIntent: 'feed-forward' | 'unconstrained',
  ) => void;
  [key: string]: unknown;
}

/**
 * Minimal mutation descriptor consumed during parent-derived spawning.
 *
 * The helpers only care about one stable public fact from the mutation system:
 * which operator name should be applied to the cloned child. Keeping this
 * contract narrow avoids importing the full mutation policy layer into the
 * population-entry boundary while still letting `spawnFromParent()` reuse the
 * controller's configured mutation selection flow.
 */
interface MutationMethod {
  name?: string;
  [key: string]: unknown;
}

/**
 * Narrow host seam required by the population-entry helpers.
 *
 * This contract exists so `createPool()`, `spawnFromParent()`, and
 * `addGenome()` can share the same runtime assumptions without depending on the
 * entire public `Neat` facade. The helper chapter needs population storage,
 * identity allocation, structural-repair hooks, RNG-backed mutation selection,
 * and a few option values, but it should not widen into a second controller
 * facade of its own.
 *
 * In practice this seam protects two invariants:
 *
 * - every entering genome receives the same controller-owned metadata shape,
 * - every entry path applies the same best-effort cleanup before later chapters
 *   read the genome.
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
 * Spawn (clone & mutate) a child genome from an existing parent genome.
 *
 * Read this helper as the provisional provenance path. It produces a candidate
 * offspring whose lineage is already meaningful, but whose membership in the
 * active population is still undecided. That split is important when a caller
 * wants to preview, filter, score, or compare several children before allowing
 * one of them to join the population through {@link addGenome}.
 *
 * Evolutionary rationale:
 * - Cloning preserves the full topology and weights of the parent.
 * - A configurable number of mutation passes are applied sequentially; each
 *   pass may alter structure (add/remove nodes or connections) or weights.
 * - Lineage annotations (`_parents`, `_depth`) enable later analytics such as
 *   diversity statistics, genealogy visualization, and pruning heuristics.
 * - Cache invalidation happens before the child is returned so later admission
 *   or evaluation logic never observes stale derived state from the clone.
 *
 * Robustness philosophy: individual mutation failures are silently ignored so a
 * single stochastic edge case does not derail evolutionary progress.
 *
 * @param this Bound NEAT instance (inferred when used as a method).
 * @param parentGenome Parent genome/network to clone. Must implement either
 * `clone()` OR a pair of `toJSON()` / static `fromJSON()` for deep copying.
 * @param mutateCount Number of sequential mutation operations to attempt; each
 *        iteration chooses a mutation method using the instance's selection
 *        logic. Defaults to 1 for conservative structural drift.
 * @returns A new genome whose score and derived caches are reset, whose lineage
 *          metadata references the parent, and whose final admission into the
 *          live population is left to the caller.
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
    const { default: NetworkClass } =
      await import('../../architecture/network');
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
 * Register an externally constructed genome (for example, deserialized,
 * custom-built, or imported from another run) into the active population.
 * This is the provenance-normalization path for genomes that did not originate
 * from `createPool()` or from the controller's normal crossover flow. The
 * helper makes those outside genomes look like first-class population members by
 * assigning the same controller-owned metadata and applying the same structural
 * cleanup that internally created genomes receive.
 *
 * Use this after a deliberate keep decision. `spawnFromParent()` returns a
 * provisional child; deserialization and custom construction create provisional
 * genomes too. `addGenome()` is the moment where those candidates become part
 * of the active run.
 *
 * Defensive design: if invariant enforcement fails, the genome is still added
 * on a best-effort basis so experiments remain reproducible and do not abort
 * mid-run.
 *
 * @param this Bound NEAT instance.
 * @param genome Genome / network object to insert. Mutated in place to add
 *        internal metadata fields (`_id`, `_parents`, `_depth`, `_reenableProb`).
 * @param parents Optional explicit list of parent genome IDs (for example, two
 *        parents for crossover). If omitted, the genome is treated as an
 *        exogenous insertion with empty lineage ancestry.
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
        .map((parentId: number) =>
          internal.population.find(
            (populationGenome: GenomeWithMetadata) =>
              populationGenome._id === parentId,
          ),
        )
        .filter(
          (populationGenome): populationGenome is GenomeWithMetadata =>
            populationGenome !== undefined,
        )
        .map((populationGenome) => populationGenome._depth ?? 0);
      genome._depth = parentDepths.length ? Math.max(...parentDepths) + 1 : 1;
    }

    // Step 3: Ensure structural invariants.
    internal.ensureMinHiddenNodes?.(genome);
    internal.ensureNoDeadEnds?.(genome);

    // Step 4: Invalidate caches and persist.
    internal._invalidateGenomeCaches?.(genome);
    internal.population.push(genome);
  } catch {
    // Fallback: still add genome so the evolutionary run can continue.
    internal.population.push(genome);
  }
}

/**
 * Create or reset the initial population pool for a NEAT run.
 *
 * If a `seedNetwork` is supplied, every genome is a structural and weight clone
 * of that seed. This is useful for transfer learning or continuing evolution
 * from a known good architecture. When omitted, brand-new minimal networks are
 * synthesized using the configured input/output sizes and optional minimum
 * hidden layer size.
 *
 * This is the controller's bootstrap path, not its general-purpose import path.
 * `createPool()` assumes the caller is defining generation zero and therefore
 * assigns clean identity and lineage state from scratch. Later provenance work,
 * such as importing one external genome or admitting a hand-picked offspring,
 * belongs to {@link addGenome} instead.
 *
 * Design notes:
 * - Population size is derived from `options.popsize` (default 50).
 * - Each genome gets a unique sequential `_id` for reproducible lineage.
 * - When lineage tracking is enabled (`_lineageEnabled`), parent and depth
 *   fields are initialized for later analytics.
 * - Feed-forward topology intent is promoted only when the configured mutation
 *   policy requests it and the genome already satisfies the stricter structural
 *   contract.
 * - Structural invariant checks are best effort. A single failure should not
 *   prevent other genomes from being created, hence the broad try/catch.
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
