import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
} from '../shared/mutation.types';
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
 */

/**
 * Mutate a single genome based on configured mutation policies.
 *
 * @param genome - genome to mutate
 * @param internal - neat controller context
 * @param methods - mutation methods module
 * @returns Promise resolving after mutation attempts complete
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
    applyMutationOperator(genome, mutationMethod, internal, methods);
    maybeAddExtraConnection(genome, internal);
    updateOperatorStatsIfNeeded(genome, mutationMethod, beforeSizes, internal);
  }
}

/**
 * Initialize per-genome adaptive mutation parameters if configured.
 *
 * @param genome - genome to initialize
 * @param internal - neat controller context
 * @returns void
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
 * @param genome - genome to resolve for
 * @param internal - neat controller context
 * @returns effective mutation rate
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
 * @param genome - genome to resolve for
 * @param internal - neat controller context
 * @returns effective mutation amount
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
 * @param effectiveRate - effective mutation probability
 * @param internal - neat controller context
 * @returns true when the genome should be mutated
 */
export function shouldMutateGenome(
  effectiveRate: number,
  internal: NeatControllerForMutation,
): boolean {
  // Step 1: compare RNG draw to effective rate.
  return internal._getRNG()() <= effectiveRate;
}

/**
 * Select a concrete mutation method, resolving any legacy arrays.
 *
 * @param genome - genome to select for
 * @param internal - neat controller context
 * @returns resolved mutation method or null
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
 * @param genome - genome to inspect
 * @returns structural size snapshot
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
 * @param genome - genome to mutate
 * @param mutationMethod - mutation operator to apply
 * @param internal - neat controller context
 * @param methods - mutation methods module
 * @returns void
 */
export function applyMutationOperator(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): void {
  // Step 1: handle structural operators that require innovation reuse.
  if (
    mutationMethod ===
    (methods.mutation as Record<string, MutationMethod>).ADD_NODE
  ) {
    applyAddNodeMutation(genome, internal, methods);
    return;
  }
  if (
    mutationMethod ===
    (methods.mutation as Record<string, MutationMethod>).ADD_CONN
  ) {
    applyAddConnMutation(genome, internal, methods);
    return;
  }

  // Step 2: defer to genome.mutate for other operators.
  genome.mutate?.(mutationMethod);

  // Step 3: invalidate caches for likely structural changes.
  if (shouldInvalidateCaches(mutationMethod, methods)) {
    internal._invalidateGenomeCaches(genome);
  }
}

/**
 * Apply an ADD_NODE mutation with reuse and weight nudging.
 *
 * @param genome - genome to mutate
 * @param internal - neat controller context
 * @param methods - mutation methods module
 * @returns void
 */
export function applyAddNodeMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): void {
  // Step 1: perform the structural mutation via reuse helper.
  internal._mutateAddNodeReuse(genome);

  // Step 2: nudge weights to make the change observable in tests.
  try {
    const mut = methods.mutation as Record<string, MutationMethod>;
    genome.mutate?.(mut.MOD_WEIGHT as MutationMethod);
  } catch {
    // Intentionally ignore: mutation may fail if genome structure is invalid.
  }

  // Step 3: invalidate caches after structural change.
  internal._invalidateGenomeCaches(genome);
}

/**
 * Apply an ADD_CONN mutation with reuse and weight nudging.
 *
 * @param genome - genome to mutate
 * @param internal - neat controller context
 * @param methods - mutation methods module
 * @returns void
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
    genome.mutate?.(mut.MOD_WEIGHT as MutationMethod);
  } catch {
    // Intentionally ignore: mutation may fail if genome structure is invalid.
  }

  // Step 3: invalidate caches after structural change.
  internal._invalidateGenomeCaches(genome);
}

/**
 * Determine whether a mutation method invalidates cached structures.
 *
 * @param mutationMethod - mutation operator to inspect
 * @param methods - mutation methods module
 * @returns true when caches should be invalidated
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
 * Optionally add an extra connection to increase exploration.
 *
 * @param genome - genome to mutate
 * @param internal - neat controller context
 * @returns void
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
 * @param genome - genome used to compute after-sizes
 * @param mutationMethod - operator being recorded
 * @param beforeSizes - structural sizes captured before mutation
 * @param internal - neat controller context
 * @returns void
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
