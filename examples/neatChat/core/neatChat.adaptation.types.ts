import type { ParameterVector } from '../../../src/neataptic.ts';

/**
 * Numeric scores recorded after one detached adaptation pass completes.
 *
 * Each key is a metric name such as `'heldOutAccuracy'`, `'repetitionRate'`, or
 * `'stabilityScore'`. Callers inspect these values before deciding whether to
 * promote or reject the candidate. The score set is open-ended so future
 * fine-tune passes can extend the evaluation suite without breaking callers.
 *
 * @example
 * ```ts
 * const scores: NeatChatCandidateEvaluationScores = {
 *   heldOutAccuracy: 0.72,
 *   repetitionRate: 0.04,
 * };
 * if ((scores['heldOutAccuracy'] ?? 0) > 0.6) {
 *   // consider promoting the candidate
 * }
 * ```
 */
export type NeatChatCandidateEvaluationScores = Record<string, number>;

/**
 * Detached adaptation candidate produced from the current live session.
 *
 * A candidate is an immutable snapshot of the weights produced by one isolated
 * fine-tune pass. The base session network is **never mutated** during or after
 * scheduling: the candidate carries a fully independent trained vector that
 * callers inspect before deciding whether to promote, reject, or discard it.
 *
 * Candidates are created by `scheduleNeatChatAdaptation` and consumed by
 * `promoteNeatChatAdaptationCandidate` or `rejectNeatChatAdaptationCandidate`.
 */
export interface NeatChatAdaptationCandidate {
  /** Full replacement weight vector produced by the isolated fine-tune pass. Compatible with `fromParameterVector` for promotion. */
  readonly trainedVector: ParameterVector;
  /** Lightweight numeric summaries for hold-out accuracy, repetition rate, and stability. Inspect before promoting or rejecting. */
  readonly evaluationScores: NeatChatCandidateEvaluationScores;
  /** Number of most-recent session exchanges replayed as the fine-tune dataset. */
  readonly trainedOnExchangeCount: number;
  /** Epoch-millisecond timestamp captured when the candidate became ready for review. */
  readonly proposedAt: number;
}

/**
 * Durable decision record captured after a candidate is promoted or rejected.
 *
 * Log entries are appended to both the in-memory
 * `NeatChatAdaptationManager.candidateLog` and the session-level
 * `NeatChatSession.candidateLog`. The session-level log survives
 * `exportNeatChatSessionV2` / `importNeatChatSessionV2` round-trips, giving
 * callers a persistent audit trail of when personalized checkpoints were
 * proposed, accepted, or rejected and what evaluation scores drove each decision.
 */
export interface NeatChatCandidateLogEntry {
  /** Outcome of the review: `'promoted'` means weights were applied to the live session; `'rejected'` means they were discarded. */
  readonly status: 'promoted' | 'rejected';
  /** Number of exchanges that served as the fine-tune dataset for the reviewed candidate. */
  readonly trainedOnExchangeCount: number;
  /** Evaluation scores captured at the time the candidate was proposed, preserved for post-decision audit. */
  readonly evaluationScores: NeatChatCandidateEvaluationScores;
  /** Epoch-millisecond timestamp recorded at the moment the promote or reject decision was applied. */
  readonly decidedAt: number;
}

/**
 * In-memory candidate queue plus the decisions recorded during one review session.
 *
 * The manager is an immutable value object: every mutating operation —
 * `scheduleNeatChatAdaptation`, `promoteNeatChatAdaptationCandidate`, and
 * `rejectNeatChatAdaptationCandidate` — returns a **new** manager instance
 * rather than mutating the current one. Callers should always reassign the
 * variable that holds the manager after each operation.
 *
 * @remarks
 * The in-memory `pendingCandidates` queue is intentionally ephemeral and is not
 * persisted in session snapshots. Use `exportNeatChatSessionV2` to durably
 * checkpoint the `candidateLog` entries from the corresponding session.
 *
 * @example
 * ```ts
 * let manager = createNeatChatAdaptationManager(session);
 * manager = await scheduleNeatChatAdaptation(manager, session);
 * console.log(manager.pendingCandidates.length); // 1
 * ```
 */
export interface NeatChatAdaptationManager {
  /** Candidates waiting for an explicit promote or reject decision. Ephemeral — not persisted in session snapshots. */
  readonly pendingCandidates: readonly NeatChatAdaptationCandidate[];
  /** Decision log accumulated by the current manager instance. Mirrors the session-level `candidateLog` which is persisted via `exportNeatChatSessionV2`. */
  readonly candidateLog: readonly NeatChatCandidateLogEntry[];
}

/**
 * Options for one detached NEATchat adaptation pass.
 *
 * All fields are optional. Omitting both `epochs` and `steps` defaults to a
 * single fine-tuning step, which is sufficient for a basic regression-correctness
 * check. For meaningful personalization, use 5–20 epochs with a small
 * `learningRate` such as `0.01`–`0.05` and cap `maxExchanges` to the most-recent
 * 10–20 exchanges to keep the dataset fresh and bounded.
 *
 * `steps` and `seed` stay available so tests can map directly to the underlying
 * `fineTuneVector` contract, while `epochs` is the alias callers should prefer
 * in production usage.
 */
export interface ScheduleNeatChatAdaptationOptions {
  /** High-level alias for the fine-tuning step count. Takes precedence over `steps` when both are provided. */
  readonly epochs?: number;
  /** Learning rate forwarded to the isolated fine-tune helper. Defaults to `0.05` when omitted. */
  readonly learningRate?: number;
  /** Maximum number of most-recent exchanges to replay during adaptation. When omitted, all session exchanges are used. */
  readonly maxExchanges?: number;
  /** Optional deterministic seed forwarded to the isolated fine-tune helper for reproducible adaptation passes. */
  readonly seed?: number;
  /** Explicit fine-tuning step count. Prefer `epochs` for production usage; `steps` is primarily for test contracts. */
  readonly steps?: number;
}
