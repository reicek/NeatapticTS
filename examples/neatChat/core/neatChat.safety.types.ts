/**
 * Safety violation category for NEATchat response quality gates.
 *
 * Identifies one of four known failure modes that the safety gate checks
 * before a response reaches the user. Each violation maps to a concrete
 * detection strategy in `neatChat.safety.services.ts`.
 *
 * - `'unknown-token'` — the response or input contains a token that falls
 *   outside the session vocabulary; the session cannot represent or learn from
 *   it safely.
 * - `'incomplete-fragment'` — the response stops on a dangling possessive or
 *   modal tail, suggesting the model emitted a clipped local phrase rather
 *   than a complete reply.
 * - `'repetition-collapse'` — the response consists largely of repeated n-gram
 *   sequences, indicating that the generative network has collapsed into a
 *   looping output pattern.
 * - `'degenerate-response'` — the response is empty, blank-only, or too short
 *   to carry meaningful content (zero or one non-whitespace token).
 *
 * @example
 * ```ts
 * const violation: SafetyViolation = 'repetition-collapse';
 * ```
 */
export type SafetyViolation =
  | 'unknown-token'
  | 'incomplete-fragment'
  | 'repetition-collapse'
  | 'degenerate-response';

/**
 * Result of a single safety check against a NEATchat session and response.
 *
 * When `ok` is `true`, `violation` is `null` and `detail` is an empty string.
 * When `ok` is `false`, `violation` names the first detected failure mode and
 * `detail` provides a human-readable explanation suitable for logs and
 * regression reports.
 *
 * The check is pass/fail only. For continuous-valued quality signals use the
 * scoring helpers in `neatChat.evaluation.services.ts` instead.
 *
 * @example
 * ```ts
 * const result: SafetyCheckResult = {
 *   ok: false,
 *   violation: 'unknown-token',
 *   detail: 'Token "xyzzy" is not in the session vocabulary.',
 * };
 * ```
 */
export interface SafetyCheckResult {
  /** Whether the response passed all safety checks. */
  readonly ok: boolean;
  /** The first detected violation, or `null` when all checks pass. */
  readonly violation: SafetyViolation | null;
  /** Human-readable explanation for the violation, or empty string when `ok`. */
  readonly detail: string;
}
