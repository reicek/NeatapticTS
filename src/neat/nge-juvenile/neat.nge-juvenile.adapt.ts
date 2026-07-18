/**
 * Score-gated adaptation API for the NGE juvenile phase.
 *
 * This module extracts the commit/rollback loop used by the racing curriculum's
 * runtime adaptation engine into a small, testable, domain-agnostic core
 * function. The caller supplies a live network, a score history, and an
 * injected {@link NgeCandidateEvaluator}. `adapt()` snapshots the network and
 * the global connection innovation counter, evaluates a baseline score, applies
 * the candidate mutation, evaluates a candidate score, and either commits the
 * mutation or rolls back to the snapshot based on a configurable improvement
 * threshold.
 *
 * The first structural growth may be exempted from the improvement check via
 * `config.overrides.firstGrowthExemption`. All policy values are defaults that
 * callers can override; no thresholds are hardcoded.
 *
 * ```mermaid
 * sequenceDiagram
 *   participant Caller
 *   participant adapt
 *   participant Network
 *   participant Evaluator
 *   Caller->>adapt: adapt({ network, evaluator, config })
 *   adapt->>Network: snapshot + capture innovation
 *   adapt->>Evaluator: baseline(network, scoreHistory)
 *   adapt->>Network: evaluator.apply(network)
 *   adapt->>Evaluator: candidate(network, scoreHistory)
 *   alt candidate improved or first-growth exemption
 *     adapt->>Network: keep mutation
 *   else candidate not improved
 *     adapt->>Network: restoreNetworkSnapshot
 *   end
 *   adapt-->>Caller: { baseline, candidate, accepted, telemetry }
 * ```
 */

import Connection from '../../architecture/connection/connection';
import type Network from '../../architecture/network';
import { restoreNetworkSnapshot } from '../neat.nge-lifecycle';
import type {
  NgeAdaptConfig,
  NgeAdaptOptions,
  NgeAdaptResult,
  NgeCandidateEvaluator,
} from './neat.nge-juvenile.types';

/**
 * Default score improvement required over baseline before a candidate
 * mutation is committed.
 *
 * A candidate must satisfy `candidate > baseline + improvementThreshold` when
 * no first-growth exemption applies.
 */
const DEFAULT_IMPROVEMENT_THRESHOLD = 0.05;

/**
 * Default first-growth exemption policy.
 *
 * When `true`, the first structural growth (`hasGrownBefore === false`) commits
 * regardless of score improvement. This gives the network capacity before
 * stabilization can tune it.
 */
const DEFAULT_FIRST_GROWTH_EXEMPTION = true;

/**
 * Snapshot the current network state and global connection innovation counter.
 *
 * The returned snapshot is a deep JSON representation of the network plus the
 * counter value captured before any candidate mutation. Restoring uses
 * {@link restoreNetworkSnapshot} so the live network reference and the global
 * innovation cursor can both be rewound.
 *
 * @param network - Live network whose state should be preserved.
 * @returns Object holding the JSON snapshot and captured innovation counter.
 */
function captureNetworkSnapshot(network: Network): {
  snapshot: Record<string, unknown>;
  capturedInnovation: number;
} {
  return {
    snapshot: network.toJSON() as Record<string, unknown>,
    capturedInnovation: Connection.nextInnovation,
  };
}

/**
 * Fully resolved adaptation config where all override fields are non-optional.
 *
 * This is the internal return type of {@link resolveAdaptConfig}, guaranteeing
 * that downstream consumers never need nullish coalescing on resolved values.
 */
interface ResolvedAdaptConfig {
  /** Resolved override values (never undefined). */
  overrides: {
    improvementThreshold: number;
    firstGrowthExemption: boolean;
  };
}

/**
 * Resolve the effective adaptation config from optional caller overrides.
 *
 * Any missing override falls back to the documented default constants. This
 * keeps the core free of hardcoded thresholds while still allowing red-test
 * fixtures to inject precise values.
 *
 * @param partial - Caller-supplied config overrides.
 * @returns Fully resolved adaptation config with non-optional override values.
 */
function resolveAdaptConfig(
  partial?: Partial<NgeAdaptConfig>,
): ResolvedAdaptConfig {
  return {
    overrides: {
      improvementThreshold:
        partial?.overrides?.improvementThreshold ??
        DEFAULT_IMPROVEMENT_THRESHOLD,
      firstGrowthExemption:
        partial?.overrides?.firstGrowthExemption ??
        DEFAULT_FIRST_GROWTH_EXEMPTION,
    },
  };
}

/**
 * Decide whether the candidate score should be committed.
 *
 * A candidate commits when it improves over the baseline by at least the
 * configured threshold, or when the first-growth exemption applies and the
 * network has not grown before.
 *
 * @param baseline - Score captured before the mutation.
 * @param candidate - Score captured after the mutation.
 * @param improvementThreshold - Minimum improvement required over baseline.
 * @param hasGrownBefore - Whether the network already committed growth.
 * @param firstGrowthExemption - Whether the first growth bypasses improvement.
 * @returns `true` when the mutation should be kept, `false` when it should roll
 *   back.
 */
function shouldCommitCandidate(
  baseline: number,
  candidate: number,
  improvementThreshold: number,
  hasGrownBefore: boolean,
  firstGrowthExemption: boolean,
): boolean {
  if (!hasGrownBefore && firstGrowthExemption) {
    return true;
  }

  return candidate > baseline + improvementThreshold;
}

/**
 * Run one score-gated adaptation window.
 *
 * Steps:
 * 1. Snapshot the live network and global connection innovation counter.
 * 2. Evaluate the baseline score using the injected evaluator.
 * 3. Apply the candidate mutation using `evaluator.apply(network)`.
 * 4. Evaluate the candidate score using the injected evaluator.
 * 5. If the candidate improves enough (or the first-growth exemption applies),
 *    keep the mutation; otherwise restore the snapshot and roll back the
 *    global innovation counter.
 *
 * @param options - Inputs for the adaptation window, including the live
 *   network, score history, injected evaluator, and optional config overrides.
 * @returns Result containing the baseline and candidate scores, whether the
 *   mutation was accepted, and telemetry for the window.
 *
 * @example
 * ```ts
 * import { adapt } from './neat.nge-juvenile.adapt';
 * import Network from '../../architecture/network';
 *
 * const network = new Network(4, 2, { seed: 42 });
 * const result = adapt({
 *   network,
 *   scoreHistory: [0.5, 0.55, 0.52],
 *   evaluator: {
 *     baseline: () => 0.5,
 *     apply: (net) => net.mutate(mutation.ADD_NODE),
 *     candidate: () => 0.7,
 *   },
 * });
 * console.log(result.accepted); // true
 * ```
 */
export function adapt(options: NgeAdaptOptions): NgeAdaptResult {
  const startTime = Date.now();
  const config = resolveAdaptConfig(options.config);
  const network = options.network;
  const scoreHistory = options.scoreHistory;
  const evaluator: NgeCandidateEvaluator = options.evaluator;

  // Step 0: Call pluggable providers when supplied.
  if (options.metricsProvider) {
    options.metricsProvider.getMetrics();
  }
  if (options.cadencePolicy) {
    options.cadencePolicy.decideCadence();
  }
  if (options.observationEncoder) {
    options.observationEncoder.encode(
      options.observation ?? {},
      network.nodes.length,
    );
  }
  if (options.lifecycleRunner) {
    options.lifecycleRunner();
  }

  const { snapshot, capturedInnovation } = captureNetworkSnapshot(network);

  const baseline = evaluator.baseline(network, scoreHistory);
  evaluator.apply(network);
  const candidate = evaluator.candidate(network, scoreHistory);

  const accepted = shouldCommitCandidate(
    baseline,
    candidate,
    config.overrides.improvementThreshold,
    options.hasGrownBefore ?? true,
    config.overrides.firstGrowthExemption,
  );

  let rollbackOccurred = false;
  if (!accepted) {
    restoreNetworkSnapshot(network, snapshot, capturedInnovation);
    rollbackOccurred = true;
  }

  const duration = Date.now() - startTime;

  return {
    baseline,
    candidate,
    accepted,
    telemetry: {
      snapshotTaken: true,
      rollbackOccurred,
      duration,
    },
  };
}
