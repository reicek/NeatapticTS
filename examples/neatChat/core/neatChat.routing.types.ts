/**
 * Identifies which generation strategy produced a routing candidate.
 *
 * - `'base'`: the live session network with no weight modification. Always present.
 * - `'personalized'`: the session network with a promoted adaptation vector applied
 *   on a detached clone. Present only when a pending adaptation candidate exists.
 * - `'retrieval-grounded'`: the live session network with retrieved episodic memories
 *   folded into the generation prompt. Present only when retrieved memories are non-empty.
 *
 * The routing layer compares candidates from each active path and selects the
 * highest-scoring one. Selecting a candidate does not promote weights or mutate
 * session state; the log produced by `appendNeatChatRoutingDecision` is the only
 * durable side-effect of a routing pass.
 */
export type NeatChatRoutingPath =
  | 'base'
  | 'personalized'
  | 'retrieval-grounded';

/**
 * One generated response candidate produced by the routing layer.
 *
 * Each candidate is a pure comparison artifact: it captures the response text,
 * the selected path, a simple scalar score, and any retrieved-memory
 * provenance used to build the candidate. Selecting a candidate does not imply
 * promotion or weight mutation.
 */
export interface NeatChatRoutingCandidate {
  /** Response path that produced this candidate. */
  readonly routingPath: NeatChatRoutingPath;
  /** Generated response text for the path. */
  readonly response: string;
  /** Tokenized form kept for reuse by the session exchange path. */
  readonly responseTokens: readonly string[];
  /** Deterministic scalar score used for path selection. */
  readonly score: number;
  /** Number of retrieved memories folded into this candidate. */
  readonly retrievedMemoryCount: number;
  /** Stable list of retrieved-memory keys used to build this candidate. */
  readonly retrievedMemoryKeys: readonly string[];
}

/**
 * Durable log entry for one routing decision.
 *
 * The log captures enough explicit context to replay the decision boundary in a
 * regression or debugging pass: which candidates were compared, what scores
 * they received, which path won, and which retrieved memories contributed to
 * any grounded candidate.
 */
export interface NeatChatRoutingDecisionLogEntry {
  /** Epoch-millisecond timestamp recorded when the decision was made. */
  readonly decidedAt: number;
  /** Path that won the comparison. */
  readonly selectedPath: NeatChatRoutingPath;
  /** Ordered list of candidate paths compared for this decision. */
  readonly comparedCandidatePaths: readonly NeatChatRoutingPath[];
  /** Number of candidates that participated in the comparison. */
  readonly candidateCount: number;
  /** Scalar score per compared routing path. */
  readonly scores: Readonly<Partial<Record<NeatChatRoutingPath, number>>>;
  /** Retrieved-memory key provenance grouped by path when applicable. */
  readonly retrievedMemoryKeysByPath?: Readonly<
    Partial<Record<NeatChatRoutingPath, readonly string[]>>
  >;
  /** Retrieved-memory counts grouped by path when applicable. */
  readonly retrievedMemoryCountByPath?: Readonly<
    Partial<Record<NeatChatRoutingPath, number>>
  >;
  /** Response text per candidate for reproducible debugging. */
  readonly responsesByPath?: Readonly<
    Partial<Record<NeatChatRoutingPath, string>>
  >;
}
