import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
  OperatorStats,
} from './neat.mutation.types';

/** Default operator adaptation boost factor. */
const DEFAULT_OPERATOR_ADAPTATION_BOOST = 2;
/** Minimum attempts before boosting operators. */
const MIN_OPERATOR_ADAPTATION_ATTEMPTS = 5;
/** Minimum success ratio required to boost operators. */
const MIN_OPERATOR_ADAPTATION_SUCCESS_RATIO = 0.55;
/** Default exploration coefficient for operator bandit. */
const DEFAULT_OPERATOR_BANDIT_EXPLORATION_COEFFICIENT = 1.4;
/** Default minimum attempts for operator bandit. */
const DEFAULT_OPERATOR_BANDIT_MIN_ATTEMPTS = 5;
/** Small epsilon for numerical stability. */
const EPSILON = 1e-10;

// ============================================================================
// Helpers for selectMutationMethod() function
// ============================================================================

/**
 * Resolve legacy FFW policy behavior, including test-specific returns.
 *
 * @param internal - neat controller context
 * @param methods - methods module
 * @param rawReturnForTest - whether to return raw FFW array for tests
 * @returns mutation method or null when not handled
 */
export function resolveFFWPolicyForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
  rawReturnForTest: boolean,
): MutationMethod | MutationMethod[] | null {
  // Step 1: detect direct or nested FFW configurations.
  const mutationMethods = methods.mutation as Record<string, MutationMethod>;
  const isFFWDirect = internal.options.mutation === mutationMethods.FFW;
  const isFFWNested =
    Array.isArray(internal.options.mutation) &&
    (internal.options.mutation as unknown[]).length === 1 &&
    (internal.options.mutation as unknown[])[0] === mutationMethods.FFW;

  // Step 2: return full FFW array when required for tests.
  if ((isFFWDirect || isFFWNested) && rawReturnForTest) {
    return mutationMethods.FFW as unknown as MutationMethod[];
  }

  // Step 3: sample a concrete FFW method when configured.
  if (isFFWDirect || isFFWNested) {
    const ffwArray = mutationMethods.FFW as unknown as MutationMethod[];
    return sampleFromPoolForSelect(ffwArray, internal);
  }

  return null;
}

/**
 * Normalize the configured mutation pool to a flat operator list.
 *
 * @param internal - neat controller context
 * @param methods - methods module
 * @param rawReturnForTest - whether to return raw FFW for tests
 * @returns normalized mutation pool
 */
export function normalizeMutationPoolForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
  rawReturnForTest: boolean,
): MutationMethod[] {
  // Step 1: read the configured pool.
  const configuredPool = internal.options.mutation as MutationMethod[];

  // Step 2: return FFW when the pool matches legacy FFW ordering.
  if (rawReturnForTest && isLegacyFFWPoolForSelect(configuredPool, methods)) {
    const mutationMethods = methods.mutation as Record<string, MutationMethod>;
    return mutationMethods.FFW as unknown as MutationMethod[];
  }

  // Step 3: unwrap nested pool arrays when present.
  if (
    configuredPool.length === 1 &&
    Array.isArray(configuredPool[0]) &&
    configuredPool[0].length
  ) {
    return configuredPool[0] as unknown as MutationMethod[];
  }

  return configuredPool;
}

/**
 * Check whether a pool matches the legacy FFW operator ordering.
 *
 * @param configuredPool - configured operator pool
 * @param methods - methods module
 * @returns true when the pool matches FFW
 */
export function isLegacyFFWPoolForSelect(
  configuredPool: MutationMethod[],
  methods: { mutation: unknown },
): boolean {
  // Step 1: compare lengths to the canonical FFW pool.
  const mutationMethods = methods.mutation as Record<string, MutationMethod>;
  const ffwPool = mutationMethods.FFW as unknown as MutationMethod[];
  if (!Array.isArray(configuredPool)) return false;
  if (configuredPool.length !== ffwPool.length) return false;

  // Step 2: ensure all operator names align.
  return configuredPool.every(
    (method, methodIndex) => method?.name === ffwPool[methodIndex]?.name,
  );
}

/**
 * Apply phased complexity adjustments to the pool when enabled.
 *
 * @param pool - base operator pool
 * @param internal - neat controller context
 * @returns pool with phased complexity adjustments
 */
export function applyPhasedComplexityForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[] {
  // Step 1: skip when phased complexity is disabled.
  if (!internal.options.phasedComplexity?.enabled || !internal._phase) {
    return pool;
  }

  // Step 2: filter invalid entries and augment based on phase.
  const filteredPool = pool.filter((method) => !!method);
  if (internal._phase === 'simplify') {
    const simplifyPool = filteredPool.filter((method) =>
      isOperatorNamePrefixedForSelect(method, 'SUB_'),
    );
    return simplifyPool.length
      ? [...filteredPool, ...simplifyPool]
      : filteredPool;
  }

  if (internal._phase === 'complexify') {
    const addPool = filteredPool.filter((method) =>
      isOperatorNamePrefixedForSelect(method, 'ADD_'),
    );
    return addPool.length ? [...filteredPool, ...addPool] : filteredPool;
  }

  return filteredPool;
}

/**
 * Apply operator adaptation weighting to the pool when enabled.
 *
 * @param pool - base pool
 * @param internal - neat controller context
 * @returns augmented pool
 */
export function applyOperatorAdaptationForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[] {
  // Step 1: skip when operator adaptation is disabled.
  if (!internal.options.operatorAdaptation?.enabled) return pool;

  // Step 2: build the augmented pool with boosted operators.
  const boostFactor =
    internal.options.operatorAdaptation.boost ??
    DEFAULT_OPERATOR_ADAPTATION_BOOST;
  const augmentedPool: MutationMethod[] = [];
  for (const method of pool) {
    augmentedPool.push(method);
    const operatorStats = internal._operatorStats.get(method.name);
    if (
      !operatorStats ||
      operatorStats.attempts <= MIN_OPERATOR_ADAPTATION_ATTEMPTS
    ) {
      continue;
    }

    const successRatio = operatorStats.success / operatorStats.attempts;
    const boostCount = Math.min(
      boostFactor,
      Math.floor(successRatio * boostFactor),
    );
    if (successRatio <= MIN_OPERATOR_ADAPTATION_SUCCESS_RATIO) continue;
    for (let boostIndex = 0; boostIndex < boostCount; boostIndex++) {
      augmentedPool.push(method);
    }
  }
  return augmentedPool;
}

/**
 * Sample a random method from the pool.
 *
 * @param pool - operator pool
 * @param internal - neat controller context
 * @returns sampled method or null
 */
export function sampleFromPoolForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod | null {
  // Step 1: return null when no pool entries exist.
  if (!pool.length) return null;

  // Step 2: sample using controller RNG.
  const randomValue = internal._getRNG()();
  const chosenIndex = Math.floor(randomValue * pool.length);
  return pool[chosenIndex] ?? null;
}

/**
 * Check whether an operator name uses a specific prefix.
 *
 * @param method - mutation operator
 * @param prefix - name prefix to match
 * @returns true when the operator name matches the prefix
 */
export function isOperatorNamePrefixedForSelect(
  method: MutationMethod,
  prefix: string,
): boolean {
  // Step 1: guard against missing names.
  if (!method?.name) return false;
  return method.name.startsWith(prefix);
}

/**
 * Check whether a mutation is blocked by structural limits.
 *
 * @param mutationMethod - mutation operator to check
 * @param genome - genome to inspect
 * @param internal - neat controller context
 * @param methods - methods module
 * @returns true when the mutation should be blocked
 */
export function isBlockedByStructuralLimitsForSelect(
  mutationMethod: MutationMethod,
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): boolean {
  // Step 1: guard against capacity limits.
  const mutationMethods = methods.mutation as Record<string, MutationMethod>;
  if (
    mutationMethod === mutationMethods.ADD_GATE &&
    genome.gates.length >= (internal.options.maxGates || Infinity)
  ) {
    return true;
  }
  if (
    mutationMethod === mutationMethods.ADD_NODE &&
    genome.nodes.length >= (internal.options.maxNodes || Infinity)
  ) {
    return true;
  }
  if (
    mutationMethod === mutationMethods.ADD_CONN &&
    genome.connections.length >= (internal.options.maxConns || Infinity)
  ) {
    return true;
  }
  return false;
}

/**
 * Apply operator bandit selection if enabled.
 *
 * @param pool - operator pool
 * @param fallbackMethod - method used when bandit is disabled
 * @param internal - neat controller context
 * @returns selected method
 */
export function applyOperatorBanditForSelect(
  pool: MutationMethod[],
  fallbackMethod: MutationMethod,
  internal: NeatControllerForMutation,
): MutationMethod {
  // Step 1: skip when operator bandit is disabled.
  if (!internal.options.operatorBandit?.enabled) return fallbackMethod;

  // Step 2: initialize operator stats for all methods.
  const operatorStats = internal._operatorStats;
  for (const method of pool) {
    if (!operatorStats.has(method.name)) {
      operatorStats.set(method.name, { success: 0, attempts: 0 });
    }
  }

  // Step 3: select the best method using a UCB-like score.
  const explorationCoefficient =
    internal.options.operatorBandit.c ??
    DEFAULT_OPERATOR_BANDIT_EXPLORATION_COEFFICIENT;
  const minAttempts =
    internal.options.operatorBandit.minAttempts ??
    DEFAULT_OPERATOR_BANDIT_MIN_ATTEMPTS;
  const totalAttempts =
    (Array.from(operatorStats.values()) as OperatorStats[]).reduce(
      (accumulator, operatorStat) => accumulator + operatorStat.attempts,
      0,
    ) + EPSILON;

  let bestMethod = fallbackMethod;
  let bestScore = -Infinity;
  for (const method of pool) {
    const methodStats = operatorStats.get(method.name)!;
    const meanScore =
      methodStats.attempts > 0 ? methodStats.success / methodStats.attempts : 0;
    const explorationBonus =
      methodStats.attempts < minAttempts
        ? Infinity
        : explorationCoefficient *
          Math.sqrt(Math.log(totalAttempts) / (methodStats.attempts + EPSILON));
    const combinedScore = meanScore + explorationBonus;
    if (combinedScore > bestScore) {
      bestScore = combinedScore;
      bestMethod = method;
    }
  }
  return bestMethod;
}

/**
 * Check whether a mutation is blocked by recurrent connection policy.
 *
 * @param mutationMethod - mutation operator to check
 * @param internal - neat controller context
 * @param methods - methods module
 * @returns true when the mutation should be blocked
 */
export function isBlockedByRecurrentPolicyForSelect(
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown },
): boolean {
  // Step 1: enforce recurrent restrictions when disabled.
  if (internal.options.allowRecurrent) return false;
  const mutationMethods = methods.mutation as Record<string, MutationMethod>;
  return (
    mutationMethod === mutationMethods.ADD_BACK_CONN ||
    mutationMethod === mutationMethods.ADD_SELF_CONN
  );
}
