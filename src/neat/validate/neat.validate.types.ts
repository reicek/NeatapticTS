import type { NetworkTopologyIntent } from '../../architecture/network/network.types';

/**
 * Stable issue codes reported by the proper-NEAT native-genome validator.
 *
 * The codes are intentionally narrow so regression tests can assert the exact
 * invariant that failed without string-matching full human-readable messages.
 */
export type NativeGenomeValidationIssueCode =
  | 'missing-node-gene-id'
  | 'duplicate-node-gene-id'
  | 'missing-connection-innovation'
  | 'duplicate-connection-innovation'
  | 'endpoint-resolution-failed'
  | 'gater-resolution-failed'
  | 'gated-connection-registration-mismatch'
  | 'topology-intent-mismatch'
  | 'feed-forward-recurrent-connection'
  | 'compat-cache-mismatch'
  | 'stale-derived-cache';

/**
 * One validator finding describing a broken native-genome invariant.
 *
 * `path` points to the runtime field that triggered the issue so test and
 * debug callers can move directly from the report to the malformed state.
 */
export interface NativeGenomeValidationIssue {
  /** Stable machine-readable issue code. */
  code: NativeGenomeValidationIssueCode;
  /** Approximate runtime location of the malformed field. */
  path: string;
  /** Human-readable explanation of the failed invariant. */
  message: string;
  /** Optional structured details for debugging or snapshot tests. */
  details?: Record<string, string | number | boolean | null>;
}

/**
 * Validation summary for one native NEAT genome.
 *
 * The report is designed for dev/test use before speciation, compatibility, or
 * crossover work begins so malformed native genomes fail loudly and locally.
 */
export interface NativeGenomeValidationReport {
  /** True when the validator found no invariant violations. */
  isValid: boolean;
  /** Optional genome id when the candidate already belongs to a NEAT population. */
  genomeId?: number;
  /** Current public topology intent carried by the network. */
  topologyIntent: NetworkTopologyIntent;
  /** Runtime node count inspected by the validator. */
  nodeCount: number;
  /** Runtime connection count inspected by the validator. */
  connectionCount: number;
  /** Ordered list of invariant violations. */
  issues: NativeGenomeValidationIssue[];
}
