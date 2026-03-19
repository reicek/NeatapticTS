import type {
  GenomeWithMetadata,
  MutationMethod,
  NeatControllerForMutation,
  OperatorStats,
} from '../shared/mutation.types';

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

/**
 * Mutation-method selection helpers.
 *
 * This chapter owns policy resolution for mutation operator choice: normalize
 * legacy pools, bias the pool for simplify/complexify phases, boost successful
 * operators, and optionally hand final choice to the operator bandit.
 *
 * The selection boundary answers a controller question that is easy to phrase
 * but subtle to implement: given the current genome, controller mode, and
 * mutation policy history, which operator should the flow execute right now?
 *
 * That decision is intentionally staged rather than monolithic:
 *
 * 1. recognize legacy policy shapes such as FFW presets,
 * 2. normalize the configured operator pool into one flat candidate list,
 * 3. bias that pool for the current complexity phase,
 * 4. optionally boost operators that have recently produced structure,
 * 5. optionally let a bandit choose among the candidate operators,
 * 6. block operators that violate hard structural or recurrent-policy limits.
 *
 * Splitting the logic this way gives the mutation root a clean public story:
 * `flow/` executes one operator at a time, while `select/` explains why that
 * operator was even eligible in the first place.
 *
 * Read this chapter from top to bottom when debugging operator choice. The
 * early helpers explain pool shape and legacy compatibility. The middle helpers
 * explain policy bias. The final helpers explain the hard stop conditions that
 * keep illegal operators from reaching execution.
 *
 * ```mermaid
 * flowchart TD
 *   Config[Configured mutation policy] --> Legacy[Handle legacy FFW cases]
 *   Legacy --> Pool[Normalize candidate pool]
 *   Pool --> Phase[Bias by simplify or complexify phase]
 *   Phase --> Adapt[Boost operators with successful history]
 *   Adapt --> Bandit[Optional bandit chooses final candidate]
 *   Bandit --> Guards{Blocked by hard limits?}
 *   Guards -->|yes| Reject[Skip this candidate]
 *   Guards -->|no| Choice[Return executable operator]
 * ```
 */

/**
 * Resolve legacy FFW policy behavior, including test-specific returns.
 *
 * Older feed-forward configuration paths can encode mutation policy in shapes
 * that do not look like the newer flat operator pool. This helper isolates that
 * compatibility layer so the rest of selection can operate on a predictable
 * surface.
 *
 * The `rawReturnForTest` flag exists because some tests assert the preserved
 * legacy pool shape rather than one sampled method. Production mutation flow,
 * by contrast, normally wants one concrete operator.
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
 * This helper is the bridge from loose policy configuration to a concrete
 * candidate shelf. It flattens nested legacy arrays and preserves the special
 * FFW path when tests explicitly need the historical representation.
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
 * The selection boundary uses this check as a compatibility detector, not as a
 * generic equality helper. Matching the canonical FFW ordering tells the caller
 * that the configured pool is really a feed-forward preset that may deserve
 * special handling.
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
 * Phased complexity nudges operator choice without banning the rest of the
 * pool outright. In simplify mode the helper duplicates subtractive operators.
 * In complexify mode it duplicates additive operators. That duplication acts as
 * a soft bias rather than a hard switch, which keeps the controller from losing
 * all policy diversity when it changes search phase.
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
 * Operator adaptation uses recent structural success as a lightweight teaching
 * signal. Operators that have enough attempts and a sufficiently high success
 * ratio are duplicated into the candidate pool, increasing their sampling
 * probability without changing the operator interface itself.
 *
 * The helper intentionally does not compute a fancy score. It just reshapes the
 * pool, which keeps the later selection steps simple and preserves compatibility
 * with random sampling and bandit-based final choice.
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
 * This is the uncomplicated fallback selector. It is useful both for direct
 * legacy sampling paths and for any configuration that wants weighted random
 * choice without the stronger opinion of the operator bandit.
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
 * Prefix checks let the selection layer describe families of operators without
 * hard-coding every method in multiple places. That keeps phased complexity
 * logic concise while still making the intent readable in the generated docs.
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
 * This helper is one of the hard guard rails after softer policy shaping has
 * already happened. Even if an operator is favored by phase bias or adaptation,
 * it must still respect controller-wide caps such as maximum nodes,
 * connections, or gates.
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
 * The operator bandit is the strongest opinionated selector in this chapter.
 * Instead of drawing randomly from the shaped pool, it ranks candidate
 * operators with a UCB-style score that balances observed success and
 * exploration.
 *
 * Early in a run, under-sampled operators receive effectively unbounded
 * exploration pressure. Later, the choice shifts toward operators with better
 * observed structural payoff. That makes this helper the bridge between simple
 * mutation policy configuration and online operator learning.
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
 * Structural capacity limits are not the only hard guard. Some runs prohibit
 * recurrent structure entirely, and this helper keeps that policy localized so
 * the rest of the selection code can stay focused on scoring and weighting.
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
