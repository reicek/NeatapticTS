import type Network from '../../network/network';
import { defaultMemoryManager } from '../../../memory/manager';
import { captureEnvironmentMetrics } from '../../../utils/memory.utils';
import type {
  NetworkSparsityBudgetProps,
  NetworkSparsityBudgetSnapshot,
  PruningMethod,
} from '../network.types';
import {
  buildEvolutionaryPruneSelection,
  disconnectEvolutionaryConnections,
  markEvolutionaryTopologyDirty,
} from './network.prune.evolutionary.utils';
import {
  NetworkPruneBudgetGrowthGraceFractionError,
  NetworkPruneBudgetMaxConnectionsError,
} from './network.prune.budget.errors';
import {
  MIN_REMAINING_CONNECTION_COUNT,
  PRUNING_METHOD_MAGNITUDE,
} from './network.prune.utils.types';

type SparsityBudgetConfiguration = {
  maxConnections: number;
  growthGraceFraction?: number;
  method?: PruningMethod;
};

type DeniedGrowthBackoffFingerprint = {
  allowedConnectionLimit: number;
  connectionCountBeforeDecision: number;
  method: PruningMethod;
  requiredAdditionalConnections: number;
  softBudgetEnvironment?: 'browser' | 'node';
  softBudgetTriggered: boolean;
};

type DeniedGrowthBackoffState = {
  consecutiveEvaluatedDenials: number;
  fingerprint: DeniedGrowthBackoffFingerprint;
  remainingSkippedAttempts: number;
};

type TriggeredSoftBudgetState = {
  environment: 'browser' | 'node';
};

type NetworkSparsityBudgetRuntimeProps = NetworkSparsityBudgetProps & {
  _deniedGrowthBackoffState?: DeniedGrowthBackoffState;
};

/** Cap the deny-backoff exponent so dead-end growth still rechecks occasionally. */
const MAX_DENIED_GROWTH_BACKOFF_EXPONENT = 8;

/**
 * Configure a total-connection growth sparsity budget on one network.
 *
 * The budget is expressed as an absolute cap across forward and self
 * connections plus an optional grace fraction. Growth helpers can then prune
 * before mutation or deny the request when the graph cannot stay within the
 * allowed envelope.
 *
 * @param this - Target network instance.
 * @param configuration - Budget settings for future structural growth.
 * @returns Nothing.
 */
export function configureSparsityBudget(
  this: Network,
  configuration: SparsityBudgetConfiguration,
): void {
  const networkBudgetProps = asSparsityBudgetProps(this);
  const normalizedMaxConnections = normalizeMaxConnections(
    configuration.maxConnections,
  );
  const normalizedGrowthGraceFraction = normalizeGrowthGraceFraction(
    configuration.growthGraceFraction,
  );

  networkBudgetProps._sparsityBudgetConfig = {
    growthGraceFraction: normalizedGrowthGraceFraction,
    maxConnections: normalizedMaxConnections,
    method: configuration.method ?? PRUNING_METHOD_MAGNITUDE,
  };
  networkBudgetProps._lastSparsityBudgetSnapshot = undefined;
}

/**
 * Read the last recorded sparsity-budget decision snapshot.
 *
 * @param currentNetwork - Network to inspect.
 * @returns Snapshot clone when one exists; otherwise undefined.
 */
export function getSparsityBudgetSnapshot(
  currentNetwork: Network,
): NetworkSparsityBudgetSnapshot | undefined {
  const latestSnapshot = asSparsityBudgetProps(currentNetwork)
    ._lastSparsityBudgetSnapshot;

  return latestSnapshot ? { ...latestSnapshot } : undefined;
}

/**
 * Ensure enough total-connection budget remains before a growth mutation writes.
 *
 * Behavior:
 * - allow immediately when the projected total connection count fits the budget,
 * - prune lowest-priority connections first when the budget can be satisfied by
 *   freeing space,
 * - temporarily stop net-new growth when the active Node/browser heap already exceeds
 *   its soft memory target,
 * - deny without structural writes when the request cannot stay within the
 *   minimum remaining-connection invariant.
 *
 * @param currentNetwork - Network about to grow.
 * @param requiredAdditionalConnections - Net total-connection increase requested by the caller.
 * @returns True when growth may proceed.
 */
export function ensureGrowthBudget(
  currentNetwork: Network,
  requiredAdditionalConnections: number,
): boolean {
  const networkBudgetProps = asSparsityBudgetRuntimeProps(currentNetwork);
  const budgetConfig = networkBudgetProps._sparsityBudgetConfig;

  if (!budgetConfig) {
    return true;
  }

  const budgetedConnections = collectBudgetedConnections(currentNetwork);
  const connectionCountBeforeDecision = budgetedConnections.length;
  const triggeredSoftBudgetState = resolveTriggeredSoftBudgetState();
  const allowedConnectionLimit = resolveEffectiveAllowedConnectionLimit(
    resolveAllowedConnectionLimit(budgetConfig),
    connectionCountBeforeDecision,
    triggeredSoftBudgetState,
  );
  const projectedConnectionCount =
    connectionCountBeforeDecision + requiredAdditionalConnections;
  const deniedGrowthBackoffFingerprint =
    createDeniedGrowthBackoffFingerprint({
      allowedConnectionLimit,
      budgetConfig,
      connectionCountBeforeDecision,
      requiredAdditionalConnections,
      triggeredSoftBudgetState,
    });

  if (projectedConnectionCount <= allowedConnectionLimit) {
    clearDeniedGrowthBackoffState(currentNetwork);
    recordSparsityBudgetSnapshot(currentNetwork, {
      allowedConnectionLimit,
      connectionCountBeforeDecision,
      connectionCountBeforeGrowth: connectionCountBeforeDecision,
      decision: 'allow',
      desiredConnectionCountBeforeGrowth: connectionCountBeforeDecision,
      plannedPruneCount: 0,
      projectedConnectionCount,
      remainingHeadroom: allowedConnectionLimit - projectedConnectionCount,
      requiredAdditionalConnections,
      softBudgetEnvironment: triggeredSoftBudgetState?.environment,
      softBudgetTriggered: triggeredSoftBudgetState !== undefined,
    });
    return true;
  }

  if (
    shouldSkipDeniedGrowthAttempt(
      currentNetwork,
      deniedGrowthBackoffFingerprint,
    )
  ) {
    recordSparsityBudgetSnapshot(currentNetwork, {
      allowedConnectionLimit,
      connectionCountBeforeDecision,
      connectionCountBeforeGrowth: connectionCountBeforeDecision,
      decision: 'deny',
      desiredConnectionCountBeforeGrowth: connectionCountBeforeDecision,
      plannedPruneCount: 0,
      projectedConnectionCount,
      remainingHeadroom: allowedConnectionLimit - projectedConnectionCount,
      requiredAdditionalConnections,
      softBudgetEnvironment: triggeredSoftBudgetState?.environment,
      softBudgetTriggered: triggeredSoftBudgetState !== undefined,
    });
    return false;
  }

  const desiredConnectionCountBeforeGrowth =
    allowedConnectionLimit - requiredAdditionalConnections;
  const plannedPruneCount = Math.max(
    0,
    connectionCountBeforeDecision - desiredConnectionCountBeforeGrowth,
  );
  const maxPrunableConnectionCount = Math.max(
    0,
    connectionCountBeforeDecision - MIN_REMAINING_CONNECTION_COUNT,
  );

  if (
    desiredConnectionCountBeforeGrowth < MIN_REMAINING_CONNECTION_COUNT ||
    plannedPruneCount > maxPrunableConnectionCount
  ) {
    registerDeniedGrowthBackoff(currentNetwork, deniedGrowthBackoffFingerprint);
    recordSparsityBudgetSnapshot(currentNetwork, {
      allowedConnectionLimit,
      connectionCountBeforeDecision,
      connectionCountBeforeGrowth: connectionCountBeforeDecision,
      decision: 'deny',
      desiredConnectionCountBeforeGrowth: connectionCountBeforeDecision,
      plannedPruneCount: 0,
      projectedConnectionCount,
      remainingHeadroom: allowedConnectionLimit - projectedConnectionCount,
      requiredAdditionalConnections,
      softBudgetEnvironment: triggeredSoftBudgetState?.environment,
      softBudgetTriggered: triggeredSoftBudgetState !== undefined,
    });
    return false;
  }

  const pruneSelection = buildEvolutionaryPruneSelection({
    connections: budgetedConnections,
    method: budgetConfig.method,
    removalCount: plannedPruneCount,
  });

  disconnectEvolutionaryConnections(
    currentNetwork,
    pruneSelection.connectionsToPrune,
  );
  markEvolutionaryTopologyDirty(currentNetwork);

  const connectionCountBeforeGrowth = countBudgetedConnections(currentNetwork);
  const growthStillFitsBudget =
    connectionCountBeforeGrowth + requiredAdditionalConnections <=
    allowedConnectionLimit;

  if (growthStillFitsBudget) {
    clearDeniedGrowthBackoffState(currentNetwork);
  } else {
    registerDeniedGrowthBackoff(currentNetwork, deniedGrowthBackoffFingerprint);
  }

  recordSparsityBudgetSnapshot(currentNetwork, {
    allowedConnectionLimit,
    connectionCountBeforeDecision,
    connectionCountBeforeGrowth,
    decision: growthStillFitsBudget ? 'prune-then-allow' : 'deny',
    desiredConnectionCountBeforeGrowth,
    plannedPruneCount,
    projectedConnectionCount:
      connectionCountBeforeGrowth + requiredAdditionalConnections,
    remainingHeadroom:
      allowedConnectionLimit -
      (connectionCountBeforeGrowth + requiredAdditionalConnections),
    requiredAdditionalConnections,
    softBudgetEnvironment: triggeredSoftBudgetState?.environment,
    softBudgetTriggered: triggeredSoftBudgetState !== undefined,
  });

  return growthStillFitsBudget;
}

/**
 * Count all connection objects that contribute to structural sparsity.
 *
 * @param currentNetwork - Network being inspected.
 * @returns Total number of forward and self connections.
 */
function countBudgetedConnections(currentNetwork: Network): number {
  return collectBudgetedConnections(currentNetwork).length;
}

/**
 * Collect all live connection objects that may be pruned to free budget.
 *
 * @param currentNetwork - Network being inspected.
 * @returns Forward and self connections.
 */
function collectBudgetedConnections(currentNetwork: Network) {
  return [...currentNetwork.connections, ...currentNetwork.selfconns];
}

/**
 * Interpret one network as a sparsity-budget host.
 *
 * @param currentNetwork - Network being inspected.
 * @returns Runtime budget props bridge.
 */
function asSparsityBudgetProps(
  currentNetwork: Network,
): NetworkSparsityBudgetProps {
  return currentNetwork as unknown as NetworkSparsityBudgetProps;
}

/**
 * Interpret one network as a sparsity-budget host with internal retry state.
 *
 * @param currentNetwork - Network being inspected.
 * @returns Runtime budget props bridge including deny-backoff state.
 */
function asSparsityBudgetRuntimeProps(
  currentNetwork: Network,
): NetworkSparsityBudgetRuntimeProps {
  return currentNetwork as unknown as NetworkSparsityBudgetRuntimeProps;
}

/**
 * Validate and normalize the configured max-connection cap.
 *
 * @param maxConnections - Raw configured connection cap.
 * @returns Safe integer cap.
 */
function normalizeMaxConnections(maxConnections: number): number {
  if (!Number.isInteger(maxConnections) || maxConnections < 1) {
    throw new NetworkPruneBudgetMaxConnectionsError(
      'maxConnections must be an integer >= 1',
    );
  }

  return maxConnections;
}

/**
 * Validate and normalize the configured growth-grace fraction.
 *
 * @param growthGraceFraction - Raw configured grace fraction.
 * @returns Safe non-negative fraction.
 */
function normalizeGrowthGraceFraction(
  growthGraceFraction: number | undefined,
): number {
  if (growthGraceFraction === undefined) {
    return 0;
  }

  if (!Number.isFinite(growthGraceFraction) || growthGraceFraction < 0) {
    throw new NetworkPruneBudgetGrowthGraceFractionError(
      'growthGraceFraction must be >= 0',
    );
  }

  return growthGraceFraction;
}

/**
 * Resolve the effective connection budget including grace headroom.
 *
 * @param budgetConfig - Normalized sparsity-budget configuration.
 * @returns Effective allowed connection limit.
 */
function resolveAllowedConnectionLimit(
  budgetConfig: NonNullable<NetworkSparsityBudgetProps['_sparsityBudgetConfig']>,
): number {
  const graceConnectionCount = Math.floor(
    budgetConfig.maxConnections * budgetConfig.growthGraceFraction,
  );

  return Math.max(
    MIN_REMAINING_CONNECTION_COUNT,
    budgetConfig.maxConnections + graceConnectionCount,
  );
}

/**
 * Tighten the effective connection cap when the runtime is already over a soft heap target.
 *
 * @param allowedConnectionLimit - Hard connection cap resolved from the network budget.
 * @param connectionCountBeforeDecision - Current total connection count.
 * @param triggeredSoftBudgetState - Active soft-budget pressure, if any.
 * @returns Effective cap for this growth decision.
 */
function resolveEffectiveAllowedConnectionLimit(
  allowedConnectionLimit: number,
  connectionCountBeforeDecision: number,
  triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined,
): number {
  if (!triggeredSoftBudgetState) {
    return allowedConnectionLimit;
  }

  return Math.max(
    MIN_REMAINING_CONNECTION_COUNT,
    Math.min(allowedConnectionLimit, connectionCountBeforeDecision),
  );
}

/**
 * Build the fingerprint used to decide whether one deny state still matches.
 *
 * @param input - Current decision inputs that define one growth dead-end.
 * @returns Stable fingerprint for deny-backoff reuse.
 */
function createDeniedGrowthBackoffFingerprint(input: {
  allowedConnectionLimit: number;
  budgetConfig: NonNullable<NetworkSparsityBudgetProps['_sparsityBudgetConfig']>;
  connectionCountBeforeDecision: number;
  requiredAdditionalConnections: number;
  triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined;
}): DeniedGrowthBackoffFingerprint {
  return {
    allowedConnectionLimit: input.allowedConnectionLimit,
    connectionCountBeforeDecision: input.connectionCountBeforeDecision,
    method: input.budgetConfig.method,
    requiredAdditionalConnections: input.requiredAdditionalConnections,
    softBudgetEnvironment: input.triggeredSoftBudgetState?.environment,
    softBudgetTriggered: input.triggeredSoftBudgetState !== undefined,
  };
}

/**
 * Skip one repeated growth attempt when the network is still in the same dead-end state.
 *
 * @param currentNetwork - Network about to retry growth.
 * @param fingerprint - Current deny fingerprint.
 * @returns True when the retry should be denied without reevaluating prune work.
 */
function shouldSkipDeniedGrowthAttempt(
  currentNetwork: Network,
  fingerprint: DeniedGrowthBackoffFingerprint,
): boolean {
  const runtimeBudgetProps = asSparsityBudgetRuntimeProps(currentNetwork);
  const deniedGrowthBackoffState = runtimeBudgetProps._deniedGrowthBackoffState;

  if (!deniedGrowthBackoffState) {
    return false;
  }

  if (!isSameDeniedGrowthFingerprint(deniedGrowthBackoffState, fingerprint)) {
    runtimeBudgetProps._deniedGrowthBackoffState = undefined;
    return false;
  }

  if (deniedGrowthBackoffState.remainingSkippedAttempts <= 0) {
    return false;
  }

  deniedGrowthBackoffState.remainingSkippedAttempts -= 1;
  return true;
}

/**
 * Record one evaluated deny and expand the retry window for unchanged future attempts.
 *
 * @param currentNetwork - Network that just denied growth.
 * @param fingerprint - Current deny fingerprint.
 * @returns Nothing.
 */
function registerDeniedGrowthBackoff(
  currentNetwork: Network,
  fingerprint: DeniedGrowthBackoffFingerprint,
): void {
  const runtimeBudgetProps = asSparsityBudgetRuntimeProps(currentNetwork);
  const previousDeniedGrowthBackoffState =
    runtimeBudgetProps._deniedGrowthBackoffState;
  const consecutiveEvaluatedDenials =
    previousDeniedGrowthBackoffState &&
    isSameDeniedGrowthFingerprint(previousDeniedGrowthBackoffState, fingerprint)
      ? previousDeniedGrowthBackoffState.consecutiveEvaluatedDenials + 1
      : 1;

  runtimeBudgetProps._deniedGrowthBackoffState = {
    consecutiveEvaluatedDenials,
    fingerprint,
    remainingSkippedAttempts: resolveDeniedGrowthBackoffWindow(
      consecutiveEvaluatedDenials,
    ),
  };
}

/**
 * Reset any deny-backoff state after growth becomes viable again.
 *
 * @param currentNetwork - Network whose retry state should be cleared.
 * @returns Nothing.
 */
function clearDeniedGrowthBackoffState(currentNetwork: Network): void {
  asSparsityBudgetRuntimeProps(currentNetwork)._deniedGrowthBackoffState =
    undefined;
}

/**
 * Compare the current deny fingerprint against the stored backoff state.
 *
 * @param deniedGrowthBackoffState - Previously recorded deny-backoff state.
 * @param fingerprint - Current decision fingerprint.
 * @returns True when the deny state is unchanged.
 */
function isSameDeniedGrowthFingerprint(
  deniedGrowthBackoffState: DeniedGrowthBackoffState,
  fingerprint: DeniedGrowthBackoffFingerprint,
): boolean {
  return (
    deniedGrowthBackoffState.fingerprint.allowedConnectionLimit ===
      fingerprint.allowedConnectionLimit &&
    deniedGrowthBackoffState.fingerprint.connectionCountBeforeDecision ===
      fingerprint.connectionCountBeforeDecision &&
    deniedGrowthBackoffState.fingerprint.method === fingerprint.method &&
    deniedGrowthBackoffState.fingerprint.requiredAdditionalConnections ===
      fingerprint.requiredAdditionalConnections &&
    deniedGrowthBackoffState.fingerprint.softBudgetEnvironment ===
      fingerprint.softBudgetEnvironment &&
    deniedGrowthBackoffState.fingerprint.softBudgetTriggered ===
      fingerprint.softBudgetTriggered
  );
}

/**
 * Resolve how many repeated requests to skip after one evaluated deny.
 *
 * @param consecutiveEvaluatedDenials - Number of full reevaluated denies in the same state.
 * @returns Remaining retry slots to skip before the next reevaluation.
 */
function resolveDeniedGrowthBackoffWindow(
  consecutiveEvaluatedDenials: number,
): number {
  const deniedGrowthBackoffExponent = Math.min(
    consecutiveEvaluatedDenials - 1,
    MAX_DENIED_GROWTH_BACKOFF_EXPONENT,
  );

  return 2 ** deniedGrowthBackoffExponent;
}

/**
 * Detect whether the current runtime heap already exceeds one active soft-memory target.
 *
 * @returns Triggered soft-budget details when one environment is over budget.
 */
function resolveTriggeredSoftBudgetState(): TriggeredSoftBudgetState | undefined {
  const memoryConfig = defaultMemoryManager.getConfig();
  const environmentMetrics = captureEnvironmentMetrics();

  if (environmentMetrics.isBrowser) {
    return isSoftBudgetExceeded(
      environmentMetrics.usedJSHeapSize,
      memoryConfig.browserMemoryBudgetMB,
    )
      ? { environment: 'browser' }
      : undefined;
  }

  return isSoftBudgetExceeded(
    environmentMetrics.heapUsed,
    memoryConfig.nodeHeapSoftLimitMB,
  )
    ? { environment: 'node' }
    : undefined;
}

/**
 * Compare one runtime memory reading against a configured soft target.
 *
 * @param measuredBytes - Heap bytes reported by the runtime.
 * @param configuredLimitMegabytes - Soft target in megabytes.
 * @returns True when the runtime is already over the configured soft target.
 */
function isSoftBudgetExceeded(
  measuredBytes: number | undefined,
  configuredLimitMegabytes: number | undefined,
): boolean {
  if (
    typeof measuredBytes !== 'number' ||
    typeof configuredLimitMegabytes !== 'number'
  ) {
    return false;
  }

  return measuredBytes > convertMegabytesToBytes(configuredLimitMegabytes);
}

/**
 * Convert megabytes to bytes for runtime-heap comparisons.
 *
 * @param megabytes - Soft target expressed in megabytes.
 * @returns Equivalent byte count.
 */
function convertMegabytesToBytes(megabytes: number): number {
  return megabytes * 1024 * 1024;
}

/**
 * Persist the latest read-only decision snapshot.
 *
 * @param currentNetwork - Network being updated.
 * @param snapshot - Snapshot to store.
 * @returns Nothing.
 */
function recordSparsityBudgetSnapshot(
  currentNetwork: Network,
  snapshot: NetworkSparsityBudgetSnapshot,
): void {
  asSparsityBudgetProps(currentNetwork)._lastSparsityBudgetSnapshot = snapshot;
}