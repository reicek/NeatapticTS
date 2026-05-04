import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
} from '../shared/mutation.types';
import Connection from '../../../architecture/connection/connection';
import { EXTRA_CONNECTION_PROBABILITY } from '../../neat.constants';

/** Default mutation rate when not configured. */
const DEFAULT_MUTATION_RATE = 0.7;
/** Default mutation amount when not configured. */
const DEFAULT_MUTATION_AMOUNT = 1;

/**
 * Flow helpers for the mutation root orchestration.
 *
 * This chapter owns the per-genome mutation loop: initialize adaptive state,
 * resolve effective rates and counts, dispatch operators, and keep operator
 * statistics in sync with the actual structural outcome.
 *
 * The root `mutation/` chapter explains the whole-generation view. This file
 * explains what happens once one genome reaches the front of that queue.
 * The lifecycle is intentionally split into four small stages:
 *
 * 1. bootstrap per-genome adaptive state,
 * 2. decide whether this genome should mutate at all and how many attempts it
 *    receives,
 * 3. ask `select/` for one concrete operator at a time and apply it,
 * 4. record the structural outcome so later adaptive policy can learn which
 *    operators are actually producing growth.
 *
 * That separation matters because mutation is not just "roll randomness and
 * edit topology." The controller is trying to preserve several invariants at
 * once: adaptive per-genome settings should remain stable across generations,
 * structural operators should reuse innovation-aware helpers, caches should be
 * invalidated only when topology really changed, and operator statistics
 * should reflect actual outcomes rather than mere selection attempts.
 *
 * Read this chapter in execution order:
 *
 * - `mutateGenome()` for the complete per-genome story,
 * - `initializeAdaptiveMutation()`, `resolveEffectiveRate()`, and
 *   `resolveEffectiveAmount()` for the gating setup,
 * - `selectConcreteMutationMethod()` plus `applyMutationOperator()` for
 *   operator dispatch,
 * - `captureStructuralSizes()` and `updateOperatorStatsIfNeeded()` for the
 *   feedback loop that supports later operator adaptation and bandit policy.
 *
 * ```mermaid
 * flowchart TD
 *   Genome[One genome enters mutation flow] --> Init[Bootstrap adaptive state]
 *   Init --> Gate{Mutate this genome now?}
 *   Gate -->|no| Skip[Leave genome unchanged]
 *   Gate -->|yes| Amount[Resolve mutation attempts]
 *   Amount --> Select[Choose one concrete operator]
 *   Select --> Apply[Apply structural or local mutation]
 *   Apply --> Repair[Optional extra connection and cache upkeep]
 *   Repair --> Stats[Record structural outcome for operator stats]
 * ```
 */

/**
 * Mutate a single genome based on configured mutation policies.
 *
 * This is the orchestration entry point for the `flow/` chapter. It keeps the
 * per-genome lifecycle linear: adaptive settings are prepared first, then one
 * probability gate decides whether work happens at all, and only then does the
 * helper loop ask `select/` for concrete operators.
 *
 * The important design choice is that mutation amount is resolved after the
 * mutate-or-skip gate passes. That keeps genomes with low effective rates from
 * paying the full operator-selection cost every generation while still letting
 * successful genomes perform multiple edits once they have been admitted into
 * the flow.
 *
 * Each loop iteration captures a before-snapshot, applies one operator,
 * performs an extra exploration edge when configured, and finally records whether the genome
 * actually grew. That final feedback is what later allows operator adaptation
 * and bandit-style policies to reward operators that change structure instead
 * of merely consuming attempts.
 *
 * @param genome Genome to mutate.
 * @param internal NEAT controller context.
 * @param methods Mutation methods module.
 * @returns Promise resolving after mutation attempts complete.
 */
export async function mutateGenome(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): Promise<void> {
  // Step 1: initialize adaptive mutation state when enabled.
  initializeAdaptiveMutation(genome, internal);

  // Step 2: resolve effective mutation rate and decide whether to mutate.
  const effectiveRate = resolveEffectiveRate(genome, internal);
  if (!shouldMutateGenome(effectiveRate, internal)) return;

  // Step 3: resolve effective mutation amount and iterate.
  const effectiveAmount = resolveEffectiveAmount(genome, internal);
  for (
    let mutationIndex = 0;
    mutationIndex < effectiveAmount;
    mutationIndex++
  ) {
    const mutationMethod = await selectConcreteMutationMethod(genome, internal);
    if (!mutationMethod?.name) continue;

    const beforeSizes = captureStructuralSizes(genome);
    await applyMutationOperator(genome, mutationMethod, internal, methods);
    maybeAddExtraConnection(genome, internal);
    updateOperatorStatsIfNeeded(genome, mutationMethod, beforeSizes, internal);
  }
}

/**
 * Initialize per-genome adaptive mutation parameters if configured.
 *
 * Adaptive mutation is stateful at the genome level, not just a controller
 * default. This helper assigns the first persistent rate and optional amount so
 * later generations can treat the genome as carrying its own mutation budget.
 *
 * The helper only writes when the genome has not yet been initialized. That is
 * why it runs at the top of `mutateGenome()` on every pass: it behaves like a
 * cheap bootstrap check rather than a repeated reset of evolved mutation
 * behavior.
 *
 * @param genome Genome to initialize.
 * @param internal NEAT controller context.
 * @returns Nothing.
 */
export function initializeAdaptiveMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void {
  // Step 1: skip when adaptive mutation is disabled.
  if (!internal.options.adaptiveMutation?.enabled) return;
  if (genome._mutRate !== undefined) return;

  // Step 2: set rate and amount defaults from configuration.
  genome._mutRate =
    internal.options.mutationRate !== undefined
      ? internal.options.mutationRate
      : (internal.options.adaptiveMutation.initialRate ??
        (internal.options.mutationRate || DEFAULT_MUTATION_RATE));
  if (internal.options.adaptiveMutation.adaptAmount) {
    genome._mutAmount =
      internal.options.mutationAmount || DEFAULT_MUTATION_AMOUNT;
  }
}

/**
 * Resolve the effective mutation rate for a genome.
 *
 * This helper explains the precedence order for rate policy:
 *
 * 1. an explicit controller-level `mutationRate` wins,
 * 2. otherwise an adaptive per-genome `_mutRate` is used when adaptive
 *    mutation is enabled,
 * 3. otherwise the flow falls back to the local default.
 *
 * Keeping that precedence isolated here makes the rest of the mutation flow
 * read as orchestration instead of configuration branching.
 *
 * @param genome Genome to resolve for.
 * @param internal NEAT controller context.
 * @returns Effective mutation rate.
 */
export function resolveEffectiveRate(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number {
  // Step 1: respect explicit mutationRate when provided.
  if (internal.options.mutationRate !== undefined) {
    return internal.options.mutationRate;
  }

  // Step 2: fall back to adaptive rate when enabled.
  if (internal.options.adaptiveMutation?.enabled) {
    return genome._mutRate ?? DEFAULT_MUTATION_RATE;
  }

  // Step 3: final fallback to default constant.
  return internal.options.mutationRate || DEFAULT_MUTATION_RATE;
}

/**
 * Resolve the effective mutation amount for a genome.
 *
 * Mutation amount answers a different question from mutation rate. Rate decides
 * whether the genome enters the flow. Amount decides how many operator draws it
 * receives once admitted. When adaptive mutation amount is enabled, the genome
 * may carry its own evolving attempt budget; otherwise the controller-wide
 * amount stays authoritative.
 *
 * @param genome Genome to resolve for.
 * @param internal NEAT controller context.
 * @returns Effective mutation amount.
 */
export function resolveEffectiveAmount(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number {
  // Step 1: prefer adaptive amount when enabled.
  if (
    internal.options.adaptiveMutation?.enabled &&
    internal.options.adaptiveMutation.adaptAmount
  ) {
    return (
      genome._mutAmount ??
      (internal.options.mutationAmount || DEFAULT_MUTATION_AMOUNT)
    );
  }

  // Step 2: fall back to configured mutation amount.
  return internal.options.mutationAmount || DEFAULT_MUTATION_AMOUNT;
}

/**
 * Decide whether a genome should be mutated based on probability.
 *
 * This is the narrow admission gate for the per-genome flow. Keeping the RNG
 * comparison in one helper makes the orchestration read clearly and gives tests
 * one stable seam for deterministic gating behavior.
 *
 * @param effectiveRate Effective mutation probability.
 * @param internal NEAT controller context.
 * @returns True when the genome should be mutated.
 */
export function shouldMutateGenome(
  effectiveRate: number,
  internal: NeatControllerForMutation,
): boolean {
  // Step 1: compare RNG draw to effective rate.
  return internal._getRNG()() <= effectiveRate;
}

/**
 * Select a concrete mutation method, resolving legacy arrays when present.
 *
 * The selection boundary may already return one final operator, or it may
 * return a legacy array-like pool for backward-compatible paths. This helper is
 * the bridge between that policy layer and the execution layer in `flow/`: it
 * guarantees the rest of the loop sees either one concrete operator or `null`.
 *
 * That normalization keeps `mutateGenome()` focused on lifecycle sequencing
 * rather than on legacy selection-shape quirks.
 *
 * @param genome Genome to select for.
 * @param internal NEAT controller context.
 * @returns Resolved mutation method or null.
 */
export async function selectConcreteMutationMethod(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): Promise<MutationMethod | null> {
  // Step 1: select a candidate method using configured selection logic.
  const selected = await internal.selectMutationMethod(genome, false);
  if (!Array.isArray(selected)) return selected ?? null;

  // Step 2: sample from the legacy array deterministically using RNG.
  const operatorArray = selected as MutationMethod[];
  const selectedIndex = Math.floor(internal._getRNG()() * operatorArray.length);
  return operatorArray[selectedIndex] ?? null;
}

/**
 * Capture structural sizes used to evaluate operator success.
 *
 * Operator adaptation in this subtree is intentionally coarse-grained: it asks
 * whether an attempted mutation increased structural size, not whether the
 * resulting genome later scored better. This helper records the pre-mutation
 * node and connection counts that make that local success signal possible.
 *
 * @param genome Genome to inspect.
 * @returns Structural size snapshot.
 */
export function captureStructuralSizes(genome: GenomeWithMetadata): {
  beforeNodes: number;
  beforeConns: number;
} {
  // Step 1: read counts before applying a mutation.
  return {
    beforeNodes: genome.nodes.length,
    beforeConns: genome.connections.length,
  };
}

/**
 * Apply a mutation operator to a genome and invalidate caches as needed.
 *
 * This helper is the dispatch hinge between policy and topology. Structural
 * growth operators such as `ADD_NODE` and `ADD_CONN` are routed through the
 * innovation-reuse helpers because the controller cares about more than local
 * graph edits; it also needs stable innovation history for later crossover and
 * speciation.
 *
 * Non-structural operators stay delegated to the genome's own `mutate()`
 * implementation. Cache invalidation is then handled separately so the flow can
 * keep the expensive cleanup targeted to methods that plausibly changed the
 * structural view of the genome.
 *
 * @param genome Genome to mutate.
 * @param mutationMethod Mutation operator to apply.
 * @param internal NEAT controller context.
 * @param methods Mutation methods module.
 * @returns Promise resolving after the operator has been applied.
 */
export async function applyMutationOperator(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): Promise<void> {
  const mutationMethods = methods.mutation as Record<string, MutationMethod>;
  const mutationName = mutationMethod?.name;

  // Step 1: handle structural operators that require innovation reuse.
  if (mutationName === mutationMethods.ADD_NODE?.name) {
    await applyAddNodeMutation(genome, internal, methods);
    return;
  }
  if (mutationName === mutationMethods.ADD_CONN?.name) {
    applyAddConnMutation(genome, internal, methods);
    return;
  }

  // Step 2: defer to genome.mutate for other operators, but first sync the
  // static Connection innovation counter above any innovations already in this
  // genome so that Connection.acquire() (used inside genome.mutate) never
  // assigns an innovation ID that is already occupied by an existing edge.
  const allGenomeConnections = [
    ...genome.connections,
    ...((
      genome as GenomeWithMetadata & {
        selfconns?: GenomeWithMetadata['connections'];
      }
    ).selfconns ?? []),
  ];
  const maxExistingInnovation = allGenomeConnections.reduce(
    (currentMax, connectionEntry) => {
      const connectionInnovation = connectionEntry.innovation;
      return typeof connectionInnovation === 'number' &&
        Number.isFinite(connectionInnovation)
        ? Math.max(currentMax, connectionInnovation)
        : currentMax;
    },
    0,
  );
  Connection.syncInnovationCounter(maxExistingInnovation);
  genome.mutate?.(mutationMethod);

  // Step 3: invalidate caches for likely structural changes.
  if (shouldInvalidateCaches(mutationMethod, methods)) {
    internal._invalidateGenomeCaches(genome);
  }
}

/**
 * Apply an ADD_NODE mutation with reuse and weight nudging.
 *
 * The add-node path is special because it needs both innovation-aware
 * structural growth and a post-split weight nudge that makes the topology
 * change observable immediately in downstream behavior and tests. The helper
 * intentionally treats cache invalidation as part of the operation rather than
 * leaving it to callers.
 *
 * @param genome Genome to mutate.
 * @param internal NEAT controller context.
 * @param methods Mutation methods module.
 * @returns Promise resolving after the add-node operation completes.
 */
export async function applyAddNodeMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): Promise<void> {
  // Step 1: perform the structural mutation via reuse helper.
  await internal._mutateAddNodeReuse(genome);

  // Step 2: nudge weights to make the change observable in tests.
  try {
    const mut = methods.mutation as Record<string, MutationMethod>;
    applyDeterministicWeightNudge(
      genome,
      internal,
      mut.MOD_WEIGHT as MutationMethod,
    );
  } catch {
    // Intentionally ignore: mutation may fail if genome structure is invalid.
  }

  // Step 3: invalidate caches after structural change.
  internal._invalidateGenomeCaches(genome);
}

/**
 * Apply an ADD_CONN mutation with reuse and weight nudging.
 *
 * This is the connection-growth companion to `applyAddNodeMutation()`. The
 * structural edit itself is delegated to the reuse-aware connection helper so
 * identical edge discoveries can still share innovation identity across the
 * population.
 *
 * @param genome Genome to mutate.
 * @param internal NEAT controller context.
 * @param methods Mutation methods module.
 * @returns Nothing.
 */
export function applyAddConnMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): void {
  // Step 1: perform the structural mutation via reuse helper.
  internal._mutateAddConnReuse(genome);

  // Step 2: nudge weights to make the change observable in tests.
  try {
    const mut = methods.mutation as Record<string, MutationMethod>;
    applyDeterministicWeightNudge(
      genome,
      internal,
      mut.MOD_WEIGHT as MutationMethod,
    );
  } catch {
    // Intentionally ignore: mutation may fail if genome structure is invalid.
  }

  // Step 3: invalidate caches after structural change.
  internal._invalidateGenomeCaches(genome);
}

/**
 * Determine whether a mutation method invalidates cached structures.
 *
 * Not every mutation should force expensive cache rebuilds. This helper keeps
 * the invalidation policy explicit by listing the operators that can change the
 * graph structure or traversal semantics enough to make cached topology views
 * unsafe.
 *
 * @param mutationMethod Mutation operator to inspect.
 * @param methods Mutation methods module.
 * @returns True when caches should be invalidated.
 */
export function shouldInvalidateCaches(
  mutationMethod: MutationMethod,
  methods: { mutation: unknown },
): boolean {
  // Step 1: check structural mutation types that alter topology.
  const mut = methods.mutation as Record<string, MutationMethod>;
  return (
    mutationMethod === mut.ADD_GATE ||
    mutationMethod === mut.SUB_NODE ||
    mutationMethod === mut.SUB_CONN ||
    mutationMethod === mut.ADD_SELF_CONN ||
    mutationMethod === mut.ADD_BACK_CONN
  );
}

/**
 * Apply the post-structural weight nudge using the controller RNG.
 *
 * The standalone network mutation helpers are allowed to own their own random
 * streams, but NEAT replay needs these follow-up weight changes to come from
 * the controller-owned RNG so the same checkpoint resumes identically.
 *
 * @param genome Genome whose connection weight should be nudged.
 * @param internal NEAT controller owning the deterministic RNG.
 * @param mutationMethod MOD_WEIGHT descriptor providing the delta range.
 * @returns Nothing.
 */
function applyDeterministicWeightNudge(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  mutationMethod: MutationMethod,
): void {
  const randomValue = internal._getRNG();
  const targetConnectionIndex = Math.floor(
    randomValue() * genome.connections.length,
  );
  const targetConnection = genome.connections[targetConnectionIndex];

  if (!targetConnection) {
    return;
  }

  const mutationDescriptor =
    mutationMethod && typeof mutationMethod === 'object'
      ? (mutationMethod as { min?: number; max?: number })
      : {};
  const minDelta = mutationDescriptor.min ?? -1;
  const maxDelta = mutationDescriptor.max ?? 1;
  const sampledDelta = randomValue() * (maxDelta - minDelta) + minDelta;

  targetConnection.weight += sampledDelta;
}

/**
 * Optionally add an extra connection to increase exploration.
 *
 * This small post-operator hook gives the controller one extra chance to add
 * connectivity after the main mutation has landed. It is intentionally
 * probabilistic and lightweight: the flow uses it as a gentle exploration bump,
 * not as a second full operator-selection phase.
 *
 * @param genome Genome to mutate.
 * @param internal NEAT controller context.
 * @returns Nothing.
 */
export function maybeAddExtraConnection(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void {
  // Step 1: probabilistically add a connection using reuse helper.
  if (internal._getRNG()() < EXTRA_CONNECTION_PROBABILITY) {
    internal._mutateAddConnReuse(genome);
  }
}

/**
 * Update operator statistics when adaptation is enabled.
 *
 * The mutation subtree measures operator success using a local structural proxy:
 * did the attempted operator increase nodes or connections compared with the
 * pre-mutation snapshot? That signal is imperfect, but it is cheap enough to
 * collect every generation and concrete enough for later adaptation logic to
 * bias toward operators that are actually creating new structure.
 *
 * @param genome Genome used to compute after-sizes.
 * @param mutationMethod Operator being recorded.
 * @param beforeSizes Structural sizes captured before mutation.
 * @param internal NEAT controller context.
 * @returns Nothing.
 */
export function updateOperatorStatsIfNeeded(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  beforeSizes: { beforeNodes: number; beforeConns: number },
  internal: NeatControllerForMutation,
): void {
  // Step 1: skip when operator adaptation is disabled.
  if (!internal.options.operatorAdaptation?.enabled) return;

  // Step 2: load or initialize stats for this operator.
  const statsRecord = internal._operatorStats.get(mutationMethod.name) || {
    success: 0,
    attempts: 0,
  };
  statsRecord.attempts++;

  // Step 3: record growth when nodes or connections increase.
  const afterNodes = genome.nodes.length;
  const afterConns = genome.connections.length;
  if (
    afterNodes > beforeSizes.beforeNodes ||
    afterConns > beforeSizes.beforeConns
  ) {
    statsRecord.success++;
  }
  internal._operatorStats.set(mutationMethod.name, statsRecord);
}
