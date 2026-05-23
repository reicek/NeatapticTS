import type { NeatChatSession } from './neatChat.types';
import type {
  SafetyCheckResult,
  SafetyViolation,
} from './neatChat.safety.types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Minimum non-whitespace token count for a response to be considered
 * non-degenerate. Responses with zero or one token carry no meaningful
 * conversational content.
 */
const DEGENERATE_RESPONSE_MIN_TOKEN_COUNT = 2;

/**
 * Sliding bigram window size used for repetition-collapse detection.
 *
 * Bigrams (consecutive two-token pairs) catch single-word repetition loops
 * and two-word phrase loops without penalizing common short phrases in
 * otherwise varied text.
 */
const REPETITION_BIGRAM_WINDOW = 2;

/**
 * Default fraction of bigrams that must be repeated before a response is
 * classified as a repetition collapse.
 *
 * A value of `0.5` means more than half of all bigrams in the response are
 * duplicates of an earlier bigram in the same response.
 */
const REPETITION_COLLAPSE_DEFAULT_THRESHOLD = 0.5;

/**
 * Response-surface token prefix reserved for internal punctuation placeholders.
 *
 * These tokens are useful inside the compact token vocabulary, but they should
 * never reach the user-visible response surface as literal text.
 */
const RESPONSE_PUNCTUATION_PLACEHOLDER_PREFIX = 'PUNC_';

/**
 * Narrow set of terminal tokens that usually indicate a clipped local phrase.
 *
 * This is intentionally small and local to the live NEATchat boundary. It is
 * not a full grammar parser; it only catches the dangling possessive and modal
 * tails currently surfaced by the red fragment-response contract.
 */
const INCOMPLETE_FRAGMENT_TERMINAL_TOKENS = new Set([
  'his',
  'its',
  'my',
  'our',
  'their',
  'will',
  'would',
  'could',
  'should',
  'your',
]);

/**
 * Splits a response string into tokens without normalizing case.
 *
 * Used by the unknown-token check so that uppercase special tokens such as
 * `'UNK'`, `'BOS'`, and `'EOS'` are preserved and can be matched against
 * the vocabulary's original-form entries.
 *
 * @param text - Raw response text to split.
 * @returns Ordered array of whitespace-delimited tokens; empty for blank input.
 */
function splitRawTokens(text: string): readonly string[] {
  return text
    .trim()
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/**
 * Splits a response string into normalized lowercase tokens.
 *
 * Trims whitespace, lowercases, splits on one or more whitespace characters,
 * and removes empty strings produced by edge cases.
 *
 * @param text - Raw response text to tokenize.
 * @returns Ordered array of normalized token strings; empty when `text` is blank.
 */
function tokenizeResponse(text: string): readonly string[] {
  return text
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

/**
 * Returns `true` when a response ends on a small set of dangling tail tokens
 * that usually indicate the model emitted an incomplete phrase.
 *
 * This bounded heuristic is intentionally local: it only rejects responses
 * that end on known clipped-tail markers such as possessives (`'your'`) or
 * modal verbs (`'will'`) that require a continuation to read as a complete
 * reply.
 *
 * @param response - Raw response text to inspect.
 * @returns `true` when the response ends on a dangling fragment marker.
 */
function isIncompleteFragment(response: string): boolean {
  const tokens = tokenizeResponse(response);
  if (tokens.length < DEGENERATE_RESPONSE_MIN_TOKEN_COUNT) return false;

  const lastToken = tokens.at(-1);
  return (
    lastToken !== undefined &&
    INCOMPLETE_FRAGMENT_TERMINAL_TOKENS.has(lastToken)
  );
}

/**
 * Builds all bigrams (consecutive token pairs) from a token sequence.
 *
 * @param tokens - Ordered token sequence to scan.
 * @returns Array of stringified bigram keys in `"a b"` format.
 */
function buildBigrams(tokens: readonly string[]): readonly string[] {
  if (tokens.length < REPETITION_BIGRAM_WINDOW) return [];
  return tokens
    .slice(0, tokens.length - 1)
    .map((token, tokenIndex) => `${token} ${tokens[tokenIndex + 1]}`);
}

// ---------------------------------------------------------------------------
// Exported predicates
// ---------------------------------------------------------------------------

/**
 * Returns `true` when a response is empty, blank-only, or contains fewer than
 * two tokens — indicating the output carries no meaningful content.
 *
 * A single-token response is considered degenerate because a conversational
 * reply should contain at least a subject and a predicate.
 *
 * @param response - Raw response text to inspect.
 * @returns `true` when the response has zero or one meaningful tokens.
 *
 * @example
 * ```ts
 * isDegenerateResponse('');       // true
 * isDegenerateResponse('hello');  // true
 * isDegenerateResponse('hi!  '); // true  (single token after trim)
 * isDegenerateResponse('hello there'); // false
 * ```
 */
export function isDegenerateResponse(response: string): boolean {
  const tokens = tokenizeResponse(response);
  return tokens.length < DEGENERATE_RESPONSE_MIN_TOKEN_COUNT;
}

/**
 * Returns `true` when the fraction of repeated bigrams in `response` exceeds
 * `threshold`, indicating a repetition-collapse failure mode.
 *
 * Bigrams are computed across the full response. If the total number of
 * bigrams is zero (empty or single-token response), the function returns
 * `false` because there is nothing to compare.
 *
 * @param response - Raw response text to inspect.
 * @param threshold - Minimum repeated-bigram fraction that triggers a `true`
 *   result. Defaults to `0.5` (more than half of bigrams repeated).
 * @returns `true` when the response is dominated by repeated bigrams.
 *
 * @example
 * ```ts
 * isRepetitionCollapse('hello hello hello hello'); // true (default threshold)
 * isRepetitionCollapse('hello there how are you'); // false
 * ```
 */
export function isRepetitionCollapse(
  response: string,
  threshold: number = REPETITION_COLLAPSE_DEFAULT_THRESHOLD,
): boolean {
  const tokens = tokenizeResponse(response);
  const bigrams = buildBigrams(tokens);
  if (bigrams.length === 0) return false;

  // Step 1: Count unique bigrams and how many appear more than once.
  const bigramSeenSet = new Set<string>();
  let repeatedBigramCount = 0;
  for (const bigram of bigrams) {
    if (bigramSeenSet.has(bigram)) {
      repeatedBigramCount += 1;
    } else {
      bigramSeenSet.add(bigram);
    }
  }

  // Step 2: Compare repeated fraction against threshold.
  const repeatedFraction = repeatedBigramCount / bigrams.length;
  return repeatedFraction > threshold;
}

/**
 * Returns `true` when `token` is absent from the session vocabulary.
 *
 * Normalizes `token` to lowercase before looking it up so the check is
 * consistent with the rest of the tokenization pipeline. An empty string
 * is always treated as unknown.
 *
 * @param token - Raw token string to look up.
 * @param session - Live NEATchat session owning the vocabulary.
 * @returns `true` when the token is not found in `session.vocabulary.termToIndex`.
 *
 * @example
 * ```ts
 * isUnknownToken('hello', session); // false (if 'hello' is in vocabulary)
 * isUnknownToken('xyzzy', session); // true  (absent from vocabulary)
 * isUnknownToken('HELLO', session); // false (lowercased match)
 * ```
 */
export function isUnknownToken(
  token: string,
  session: NeatChatSession,
): boolean {
  if (token.length === 0) return true;
  // Step 1: Try original form first (catches uppercase special tokens: UNK, BOS, EOS, TURN_BREAK).
  if (session.vocabulary.termToIndex.has(token)) return false;
  // Step 2: Try lowercase form (catches corpus terms stored as lowercase).
  return !session.vocabulary.termToIndex.has(token.toLowerCase());
}

// ---------------------------------------------------------------------------
// Primary safety gate
// ---------------------------------------------------------------------------

/**
 * Checks a generated response against four known failure modes and returns a
 * single pass/fail result.
 *
 * Violation priority order (first match wins):
 * 1. `'degenerate-response'` — empty, blank, or single-token output.
 * 2. `'repetition-collapse'` — bigram repetition fraction exceeds `0.5`.
 * 3. `'incomplete-fragment'` — response ends on a dangling possessive or modal tail.
 * 4. `'unknown-token'` — a placeholder token or any response token is absent from the session vocabulary.
 *
 * When all checks pass, `ok` is `true`, `violation` is `null`, and `detail`
 * is an empty string. The safety gate does not mutate session state.
 *
 * For continuous-valued quality signals, use the scoring helpers in
 * `neatChat.evaluation.services.ts` instead.
 *
 * @param session - Live NEATchat session providing the vocabulary for OOV lookup.
 * @param response - Generated response text to check.
 * @returns A `SafetyCheckResult` describing the first detected violation or pass.
 *
 * @example
 * ```ts
 * const result = checkSafety(session, 'hello hello hello hello');
 * // result.ok === false, result.violation === 'repetition-collapse'
 * ```
 */
export function checkSafety(
  session: NeatChatSession,
  response: string,
): SafetyCheckResult {
  // Step 1: Check for degenerate response (highest priority).
  const degenerateResult = checkDegenerateResponse(response);
  if (degenerateResult !== null) return degenerateResult;

  // Step 2: Check for repetition collapse.
  const repetitionResult = checkRepetitionCollapse(response);
  if (repetitionResult !== null) return repetitionResult;

  // Step 3: Reject obviously clipped local fragments before surface delivery.
  const incompleteFragmentResult = checkIncompleteFragment(response);
  if (incompleteFragmentResult !== null) return incompleteFragmentResult;

  // Step 4: Reject internal punctuation placeholder tokens before surface delivery.
  const placeholderTokenResult = checkPlaceholderSurfaceTokens(response);
  if (placeholderTokenResult !== null) return placeholderTokenResult;

  // Step 5: Check for unknown tokens.
  const unknownTokenResult = checkUnknownTokens(session, response);
  if (unknownTokenResult !== null) return unknownTokenResult;

  // Step 6: All checks passed.
  return buildPassResult();

  /** @returns A fail result for degenerate response, or null. */
  function checkDegenerateResponse(text: string): SafetyCheckResult | null {
    if (!isDegenerateResponse(text)) return null;
    return buildFailResult(
      'degenerate-response',
      `Response has fewer than ${DEGENERATE_RESPONSE_MIN_TOKEN_COUNT} tokens and carries no meaningful content.`,
    );
  }

  /** @returns A fail result for repetition collapse, or null. */
  function checkRepetitionCollapse(text: string): SafetyCheckResult | null {
    if (!isRepetitionCollapse(text)) return null;
    return buildFailResult(
      'repetition-collapse',
      'Response bigram repetition fraction exceeds the collapse threshold; the network may be looping.',
    );
  }

  /** @returns A fail result for clipped local fragments, or null. */
  function checkIncompleteFragment(text: string): SafetyCheckResult | null {
    if (!isIncompleteFragment(text)) return null;
    return buildFailResult(
      'incomplete-fragment',
      'Response ends on a dangling possessive or modal tail and looks like an incomplete local fragment.',
    );
  }

  /** @returns A fail result for literal placeholder tokens, or null. */
  function checkPlaceholderSurfaceTokens(
    text: string,
  ): SafetyCheckResult | null {
    const tokens = splitRawTokens(text);
    const firstPlaceholderToken = tokens.find(isPunctuationPlaceholderToken);
    if (firstPlaceholderToken === undefined) return null;
    return buildFailResult(
      'unknown-token',
      `Token "${firstPlaceholderToken}" is an internal punctuation placeholder and should not reach the live response surface.`,
    );
  }

  /** @returns A fail result for the first OOV token found, or null. */
  function checkUnknownTokens(
    chatSession: NeatChatSession,
    text: string,
  ): SafetyCheckResult | null {
    // Use splitRawTokens (preserves case) so uppercase special tokens such as
    // 'UNK' are matched against the vocabulary's original-form entries.
    const tokens = splitRawTokens(text);
    const firstUnknown = tokens.find((token) =>
      isUnknownToken(token, chatSession),
    );
    if (firstUnknown === undefined) return null;
    return buildFailResult(
      'unknown-token',
      `Token "${firstUnknown}" is not in the session vocabulary.`,
    );
  }

  /** @returns A SafetyCheckResult with ok:true. */
  function buildPassResult(): SafetyCheckResult {
    return { ok: true, violation: null, detail: '' };
  }

  /** @returns A SafetyCheckResult with ok:false. */
  function buildFailResult(
    violation: SafetyViolation,
    detail: string,
  ): SafetyCheckResult {
    return { ok: false, violation, detail };
  }
}

function isPunctuationPlaceholderToken(token: string): boolean {
  return token.startsWith(RESPONSE_PUNCTUATION_PLACEHOLDER_PREFIX);
}
