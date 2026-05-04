import type { NeatLike } from '../shared/neat.shared.types';
import Network from '../../architecture/network/network';
import { createInnovationTracker } from '../innovation-tracker/innovation-tracker';
import type { InnovationTracker } from '../innovation-tracker/innovation-tracker.types';
import {
  promoteGenomeToFeedForwardIntentWhenEligible,
  usesFeedForwardMutationPolicy,
} from '../topology-intent/neat.topology-intent';

const DEFAULT_POOL_SIZE = 50;
const DEFAULT_MAX_INNOVATION = -1;
const RNG_SEED_UPPER_BOUND = 0x1_0000_0000;

interface GenerationZeroConnection {
  innovation?: number;
  from: { index?: number };
  to: { index?: number };
}

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
 * Those three paths are related, but they are not interchangeable. That is the
 * main pedagogical point of this root chapter:
 *
 * - `createPool()` creates generation-zero membership,
 * - `spawnFromParent()` creates a candidate with meaningful lineage but without
 *   guaranteed admission,
 * - `addGenome()` is the commit step that makes a genome part of the live run.
 *
 * Keeping those paths together prevents subtle drift in `_id`, `_parents`,
 * `_depth`, `_reenableProb`, feed-forward intent, and cache invalidation rules.
 * The public `Neat` facade still exposes the same methods, but this file now
 * reads as one small chapter about safe population entry instead of a grab bag
 * of leftover helpers.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   seed[Seed network or empty start]:::base --> pool[createPool<br/>build generation zero]:::accent
 *   parent[Existing parent genome]:::base --> child[spawnFromParent<br/>produce provisional child]:::base
 *   imported[Imported or custom genome]:::base --> admit[addGenome<br/>normalize and admit]:::accent
 *   child --> admit
 *   pool --> population[Live population with normalized metadata]:::base
 *   admit --> population
 * ```
 *
 * Required teaching output: generation-zero alignment.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   intent[Seeded or unseeded start]:::base --> template[Build one normalized template genome]:::accent
 *   template --> ids[Normalize identity\nnode geneIds and connection innovations]:::base
 *   ids --> tracker[Reseed innovation tracker above template maxima]:::base
 *   tracker --> clones[Clone template across popsize]:::accent
 *   clones --> metadata[Assign controller-owned metadata\n_id, lineage, caches]:::base
 *   metadata --> ready[Homologous generation-zero population]:::base
 * ```
 *
 * Read this chapter when you want to answer one practical controller question:
 * before selection, evaluation, and speciation can trust a genome, how does it
 * cross the boundary into the live population in a normalized state?
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
  connections?: GenerationZeroConnection[];
  selfconns?: GenerationZeroConnection[];
  getRNGState?: () => number | undefined;
  getTopologyIntent?: () => 'feed-forward' | 'unconstrained';
  toJSON?: () => Record<string, unknown>;
  mutate?: (method: MutationMethod) => void;
  setRNGState?: (state: number) => void;
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
  generation: number;
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
  _innovationTracker: InnovationTracker;
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
      await import('../../architecture/network/network');
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

  // Step 6: Invalidate cached compatibility / distance metrics tied to the genome.
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

    // Step 2: Copy lineage from provided parent IDs when present.
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
 * of one normalized template derived from that seed. This is useful for
 * transfer learning or continuing evolution from a known good architecture.
 * When omitted, one fresh minimal template is synthesized using the configured
 * input/output sizes and optional minimum hidden layer size, then cloned across
 * the whole starting population so node gene ids and connection innovations are
 * aligned from the first generation.
 *
 * This is the controller's bootstrap path, not its general-purpose import path.
 * `createPool()` assumes the caller is defining generation zero and therefore
 * assigns clean identity and lineage state from scratch. Later provenance work,
 * such as importing one external genome or admitting a hand-picked offspring,
 * belongs to {@link addGenome} instead.
 *
 * Design notes:
 * - Population size is derived from `options.popsize` (default 50).
 * - The controller innovation tracker is reseeded from the normalized
 *   generation-zero template so later structural mutations start above the
 *   starter graph's historical markings.
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
    const poolSize = internal.options?.popsize ?? DEFAULT_POOL_SIZE;
    const shouldPromoteFeedForwardIntent = usesFeedForwardMutationPolicy(
      internal.options?.mutation,
    );
    const generationZeroTemplate = createGenerationZeroTemplate(
      seedNetwork,
      internal,
      shouldPromoteFeedForwardIntent,
    );

    // Step 2: Reseed structural-mutation tracking from the generation-zero template.
    internal._innovationTracker = createInnovationTracker();
    internal._innovationTracker.activeGeneration = internal.generation;
    internal._innovationTracker.nextInnovationId =
      resolveNextInnovationIdFromGenome(generationZeroTemplate);

    // Step 3: Clone one normalized template into the full starting population.
    for (let genomeIndex = 0; genomeIndex < poolSize; genomeIndex++) {
      const genomeCopy = cloneGenerationZeroGenome(generationZeroTemplate);
      normalizeGenerationZeroMetadata(internal, genomeCopy);
      internal._invalidateGenomeCaches?.(genomeCopy);
      internal.population.push(genomeCopy);
    }
  } catch {
    // Swallow: partial population is acceptable; caller may decide to refill or continue.
  }
}

function createGenerationZeroTemplate(
  seedNetwork: GenomeWithMetadata | null,
  internal: NeatControllerForHelpers,
  shouldPromoteFeedForwardIntent: boolean,
): GenomeWithMetadata {
  const generationZeroTemplate = seedNetwork
    ? cloneGenerationZeroGenome(seedNetwork)
    : createFreshGenerationZeroTemplate(internal);

  // Step 1: Apply best-effort structural normalization once on the shared template.
  try {
    internal.ensureNoDeadEnds?.(generationZeroTemplate);
  } catch {
    // Generation-zero repair remains best-effort.
  }

  // Step 2: Replace incidental constructor innovations with explicit generation-zero ordering.
  if (!seedNetwork) {
    canonicalizeFreshGenerationZeroInnovations(generationZeroTemplate);
  }

  // Step 3: Promote the runtime topology contract only after the final template shape is known.
  promoteGenomeToFeedForwardIntentWhenEligible(
    generationZeroTemplate,
    shouldPromoteFeedForwardIntent,
  );

  return generationZeroTemplate;
}

function createFreshGenerationZeroTemplate(
  internal: NeatControllerForHelpers,
): GenomeWithMetadata {
  const controllerRandom = internal._getRNG();
  const generationZeroSeed = Math.floor(
    controllerRandom() * RNG_SEED_UPPER_BOUND,
  );

  return new Network(internal.input, internal.output, {
    minHidden: internal.options?.minHidden,
    seed: generationZeroSeed,
  }) as unknown as GenomeWithMetadata;
}

function cloneGenerationZeroGenome(
  sourceGenome: GenomeWithMetadata,
): GenomeWithMetadata {
  const clonedGenome = sourceGenome.clone
    ? sourceGenome.clone()
    : (Network.fromJSON(
        sourceGenome.toJSON?.() ?? {},
      ) as unknown as GenomeWithMetadata);

  synchronizeGenerationZeroCloneState(sourceGenome, clonedGenome);
  return clonedGenome;
}

function synchronizeGenerationZeroCloneState(
  sourceGenome: GenomeWithMetadata,
  clonedGenome: GenomeWithMetadata,
): void {
  const topologyIntent = sourceGenome.getTopologyIntent?.();
  if (topologyIntent) {
    clonedGenome.setTopologyIntent?.(topologyIntent);
  }

  const sourceRngState = sourceGenome.getRNGState?.();
  if (typeof sourceRngState === 'number') {
    clonedGenome.setRNGState?.(sourceRngState);
  }
}

function canonicalizeFreshGenerationZeroInnovations(
  generationZeroTemplate: GenomeWithMetadata,
): void {
  const generationZeroConnections = collectGenerationZeroConnections(
    generationZeroTemplate,
  ).toSorted(compareGenerationZeroConnections);

  generationZeroConnections.forEach((connection, innovationId) => {
    connection.innovation = innovationId;
  });
}

function collectGenerationZeroConnections(
  genome: GenomeWithMetadata,
): GenerationZeroConnection[] {
  return [...(genome.connections ?? []), ...(genome.selfconns ?? [])];
}

function compareGenerationZeroConnections(
  leftConnection: GenerationZeroConnection,
  rightConnection: GenerationZeroConnection,
): number {
  const sourceIndexDelta =
    leftConnection.from.index! - rightConnection.from.index!;
  if (sourceIndexDelta !== 0) {
    return sourceIndexDelta;
  }

  return leftConnection.to.index! - rightConnection.to.index!;
}

function resolveNextInnovationIdFromGenome(genome: GenomeWithMetadata): number {
  const maxObservedInnovation = collectGenerationZeroConnections(genome).reduce(
    (currentMaxInnovation, connection) =>
      typeof connection.innovation === 'number'
        ? Math.max(currentMaxInnovation, connection.innovation)
        : currentMaxInnovation,
    DEFAULT_MAX_INNOVATION,
  );

  return maxObservedInnovation + 1;
}

function normalizeGenerationZeroMetadata(
  internal: NeatControllerForHelpers,
  genome: GenomeWithMetadata,
): void {
  genome.score = undefined;
  genome._reenableProb = internal.options.reenableProb;
  genome._id = internal._nextGenomeId++;

  if (internal._lineageEnabled) {
    genome._parents = [];
    genome._depth = 0;
    return;
  }

  Reflect.deleteProperty(genome, '_parents');
  Reflect.deleteProperty(genome, '_depth');
}
