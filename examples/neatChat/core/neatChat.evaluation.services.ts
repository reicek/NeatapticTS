import type { NeatChatEpisodicMemoryBank } from './neatChat.memory.types';
import type { NeatChatSession } from './neatChat.types';
import type {
  EvaluationHarnessInput,
  EvaluationMetric,
  FailureBucket,
  RegressionEntry,
  RegressionSuiteResult,
} from './neatChat.evaluation.types';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/**
 * Default OOV-fraction threshold used when `maxUnknownTokenFraction` is omitted
 * from `EvaluationHarnessInput`. A value of `0` means any OOV token triggers
 * the `'unknown-handling'` metric score.
 */
const DEFAULT_MAX_UNKNOWN_TOKEN_FRACTION = 0;

/**
 * Sliding window size for bigram-based repetition detection.
 *
 * Bigrams (consecutive token pairs) are the smallest repeatable unit that
 * catches single-word repetition loops such as "hello hello hello" without
 * penalising common repeated words in otherwise varied text.
 */
const REPETITION_BIGRAM_WINDOW_SIZE = 2;

/**
 * All `EvaluationMetric` values in canonical order.
 *
 * Used when iterating over metrics during aggregation so the order is
 * deterministic and independent of object key ordering.
 */
const ALL_EVALUATION_METRICS: readonly EvaluationMetric[] = [
  'next-token-accuracy',
  'factual-consistency',
  'repetition-rate',
  'response-length-stability',
  'unknown-handling',
] as const;

// ---------------------------------------------------------------------------
// Tokenization helper
// ---------------------------------------------------------------------------

/**
 * Splits a text string into normalized lowercase tokens.
 *
 * Trims leading and trailing whitespace, converts to lower-case, splits on
 * one or more whitespace characters, and removes any empty strings that
 * result from edge cases such as multiple adjacent spaces.
 *
 * @param text - Raw text to tokenize.
 * @returns Ordered array of normalized token strings; empty array for blank input.
 *
 * @example
 * ```ts
 * tokenizeText('Hello   World'); // ['hello', 'world']
 * tokenizeText('');              // []
 * ```
 */
function tokenizeText(text: string): readonly string[] {
  return text
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((token) => token.length > 0);
}

// ---------------------------------------------------------------------------
// Score helpers (exported per plan contract)
// ---------------------------------------------------------------------------

/**
 * Scores per-position next-token accuracy between two token sequences.
 *
 * Compares `predicted` and `expected` token-by-token up to the shorter
 * sequence length, then divides by the expected length. Returns `1` when
 * `expected` is empty (no tokens to get wrong) and `0` when `predicted` is
 * empty but `expected` is not.
 *
 * @param predicted - The predicted response text.
 * @param expected - The expected response text from the held-out corpus.
 * @returns Score in `[0, 1]` where `1` indicates perfect per-position match.
 *
 * @example
 * ```ts
 * scoreNextTokenAccuracy('hello world', 'hello world'); // 1
 * scoreNextTokenAccuracy('hi there', 'hello world');    // 0
 * ```
 */
export function scoreNextTokenAccuracy(
  predicted: string,
  expected: string,
): number {
  const predictedTokens = tokenizeText(predicted);
  const expectedTokens = tokenizeText(expected);

  if (expectedTokens.length === 0) return 1;
  if (predictedTokens.length === 0) return 0;

  // Step 1: Compare token-by-token up to min(predicted, expected) length.
  const comparisonLength = Math.min(
    predictedTokens.length,
    expectedTokens.length,
  );
  const matchCount = predictedTokens
    .slice(0, comparisonLength)
    .filter((token, tokenIndex) => token === expectedTokens[tokenIndex]).length;

  // Step 2: Normalise by expected length so missing tokens count as errors.
  return matchCount / expectedTokens.length;
}

/**
 * Scores the bigram repetition rate of a response.
 *
 * Builds consecutive token pairs (bigrams) and counts how many are
 * duplicates of a pair already seen. The result is the fraction of bigrams
 * that were repeated.
 *
 * Lower is better: `0` means no bigram appeared more than once, `1` means
 * every bigram after the first was a repeat.
 *
 * @param response - The generated or expected response text to evaluate.
 * @returns Score in `[0, 1]` where `0` indicates no repeated bigrams.
 *
 * @example
 * ```ts
 * scoreRepetitionRate('hello there general kenobi'); // 0
 * scoreRepetitionRate('hello hello hello hello');    // > 0
 * ```
 */
export function scoreRepetitionRate(response: string): number {
  const responseTokens = tokenizeText(response);

  if (responseTokens.length < REPETITION_BIGRAM_WINDOW_SIZE) return 0;

  const totalBigrams = responseTokens.length - 1;
  const seenBigrams = new Set<string>();
  let repeatedBigramCount = 0;

  // Step 1: Slide over all consecutive pairs and count duplicates.
  for (let bigramIndex = 0; bigramIndex < totalBigrams; bigramIndex++) {
    const bigram = `${responseTokens[bigramIndex]} ${responseTokens[bigramIndex + 1]}`;
    if (seenBigrams.has(bigram)) {
      repeatedBigramCount++;
    } else {
      seenBigrams.add(bigram);
    }
  }

  // Step 2: Normalise by total bigrams.
  return repeatedBigramCount / totalBigrams;
}

/**
 * Scores response length stability as a proximity measure between actual and
 * expected token counts.
 *
 * Uses a symmetric distance formula bounded to `[0, 1]`:
 * `1 - |actual - expected| / max(actual, expected)`.
 * Returns `1` when actual and expected lengths are both zero (perfect trivial match)
 * or when they are identical. Returns `0` when one of them is zero and the other is not.
 *
 * @param response - The actual response text to evaluate.
 * @param expectedLength - The expected token count from the held-out corpus.
 * @returns Score in `[0, 1]` where `1` indicates exact length match.
 *
 * @example
 * ```ts
 * scoreResponseLengthStability('hello world', 2); // 1
 * scoreResponseLengthStability('', 5);            // 0
 * ```
 */
export function scoreResponseLengthStability(
  response: string,
  expectedLength: number,
): number {
  const actualLength = tokenizeText(response).length;

  const longerLength = Math.max(actualLength, expectedLength);

  // Step 1: Both empty — perfect stability.
  if (longerLength === 0) return 1;

  // Step 2: Normalised symmetric distance.
  return 1 - Math.abs(actualLength - expectedLength) / longerLength;
}

/**
 * Scores the fraction of in-vocabulary tokens in a text given the session vocabulary.
 *
 * Returns `1` when all tokens are known, `0` when none are known, and a
 * proportional value for partial overlap. An empty input text returns `1`
 * (no tokens to be unknown).
 *
 * @param text - The text to check for out-of-vocabulary tokens.
 * @param session - The live session whose vocabulary is used for OOV detection.
 * @returns Score in `[0, 1]` where `1` means all tokens are in-vocabulary.
 *
 * @example
 * ```ts
 * scoreUnknownHandling('hello world', session); // 1 if both tokens are in vocab
 * scoreUnknownHandling('zzquux', session);      // 0 if token is out-of-vocabulary
 * ```
 */
export function scoreUnknownHandling(
  text: string,
  session: NeatChatSession,
): number {
  const textTokens = tokenizeText(text);

  if (textTokens.length === 0) return 1;

  // Step 1: Count tokens present in the session vocabulary.
  const inVocabCount = textTokens.filter((token) =>
    session.vocabulary.termToIndex.has(token),
  ).length;

  // Step 2: Normalise by total token count.
  return inVocabCount / textTokens.length;
}

/**
 * Scores factual consistency of a response against saved memory-bank records.
 *
 * Collects all token strings from every memory record's `key` and `value`
 * fields, then counts how many appear in the response. Returns `1` when the
 * memory bank is empty (no facts to violate) or when all fact tokens are
 * present in the response.
 *
 * @param response - The response text to evaluate.
 * @param memoryBank - The episodic memory bank containing user-specific facts.
 * @returns Score in `[0, 1]` where `1` means all fact tokens appear in the response.
 *
 * @example
 * ```ts
 * // Memory has { key: 'name', value: 'alice' }
 * scoreFactualConsistency('alice is here', memoryBank);  // 0.5 (only 'alice' found)
 * scoreFactualConsistency('', emptyMemoryBank);          // 1
 * ```
 */
export function scoreFactualConsistency(
  response: string,
  memoryBank: NeatChatEpisodicMemoryBank,
): number {
  const { records } = memoryBank;

  // Step 1: No facts to violate — trivially consistent.
  if (records.length === 0) return 1;

  // Step 2: Collect all fact tokens from all memory record key/value pairs.
  const factTokens = records.flatMap((record) => [
    ...tokenizeText(record.key),
    ...tokenizeText(record.value),
  ]);

  if (factTokens.length === 0) return 1;

  // Step 3: Check which fact tokens appear in the response token set.
  const responseTokenSet = new Set(tokenizeText(response));
  const matchedFactCount = factTokens.filter((token) =>
    responseTokenSet.has(token),
  ).length;

  // Step 4: Normalise matched count by total fact token count.
  return matchedFactCount / factTokens.length;
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------

/**
 * Attributes a regression entry to the most likely failure bucket.
 *
 * Examines `session.routingLog` to determine which candidate path last
 * influenced the session's output. The attribution is deterministic: the
 * same session state and entry always yield the same bucket.
 *
 * Routing observability invariant from Workstream 5 is preserved: this
 * function reads `routingLog` but never mutates any session state or
 * promotes weights.
 *
 * Attribution priority:
 * 1. If the last routing decision selected `'personalized'` → `'routing'`.
 * 2. If the last routing decision selected `'retrieval-grounded'` → `'retrieval'`.
 * 3. If the last routing decision selected `'base'` or the log is empty → `'base-seed'`.
 * 4. Otherwise → `'unattributed'`.
 *
 * @param session - The live session providing routing-log context.
 * @param _entry - The regression entry (reserved for future metric-aware attribution).
 * @returns The attributed `FailureBucket` for this regression.
 *
 * @example
 * ```ts
 * const bucket = attributeToFailureBucket(session, entry);
 * // Returns 'routing' if session used a personalized path most recently.
 * ```
 */
export function attributeToFailureBucket(
  session: NeatChatSession,
  _entry: RegressionEntry,
): FailureBucket {
  // Step 1: Inspect the most recent routing decision if any exist.
  const lastRoutingDecision = session.routingLog.at(-1);

  if (lastRoutingDecision === undefined) {
    // Step 2: No routing history — base seed is the only responsible subsystem.
    return 'base-seed';
  }

  // Step 3: Map selected routing path to its attributed failure bucket.
  const { selectedPath } = lastRoutingDecision;

  if (selectedPath === 'personalized') return 'routing';
  if (selectedPath === 'retrieval-grounded') return 'retrieval';

  // Step 4: selectedPath === 'base' is the only remaining NeatChatRoutingPath case.
  return 'base-seed';
}

// ---------------------------------------------------------------------------
// Regression harness
// ---------------------------------------------------------------------------

/**
 * Determines whether a given metric score is a regression relative to its
 * declared baseline.
 *
 * For `'repetition-rate'` (lower is better), a score *above* the baseline is
 * a regression. For all other metrics (higher is better), a score *below* the
 * baseline is a regression.
 *
 * @param metric - The evaluation metric being checked.
 * @param score - The computed score for this entry.
 * @param baseline - The minimum (or maximum for repetition) acceptable score.
 * @returns `true` when the entry's score fails the baseline threshold.
 */
function isRegressionScore(
  metric: EvaluationMetric,
  score: number,
  baseline: number,
): boolean {
  if (metric === 'repetition-rate') {
    // Lower is better: any score above the declared ceiling is a regression.
    return score > baseline;
  }
  // Higher is better: any score below the declared floor is a regression.
  return score < baseline;
}

/**
 * Computes all five metric scores for a single corpus entry.
 *
 * The `actualResponse` for corpus-quality evaluation is taken as the
 * `expectedResponse` itself (non-mutating baseline evaluation). The
 * `'unknown-handling'` metric uses the `inputPrompt` tokens so that
 * out-of-vocabulary inputs are flagged regardless of the expected output.
 *
 * @param corpusEntry - The held-out input/expected pair to score.
 * @param session - The session providing vocabulary and memory context.
 * @returns Partial metric scores map for the entry.
 */
function scoreAllMetricsForEntry(
  corpusEntry: {
    readonly inputPrompt: string;
    readonly expectedResponse: string;
  },
  session: NeatChatSession,
): Readonly<Partial<Record<EvaluationMetric, number>>> {
  const { inputPrompt, expectedResponse } = corpusEntry;
  const expectedTokenLength = tokenizeText(expectedResponse).length;

  // Step 1: Token-level accuracy against the expected corpus baseline.
  const nextTokenAccuracy = scoreNextTokenAccuracy(
    expectedResponse,
    expectedResponse,
  );

  // Step 2: Bigram repetition in the expected response.
  const repetitionRate = scoreRepetitionRate(expectedResponse);

  // Step 3: Length proximity (expected length vs itself → always 1 for well-formed entries).
  const responseLengthStability = scoreResponseLengthStability(
    expectedResponse,
    expectedTokenLength,
  );

  // Step 4: Factual consistency of expected response against memory bank facts.
  const factualConsistency = scoreFactualConsistency(
    expectedResponse,
    session.memoryBank,
  );

  // Step 5: OOV fraction of the input prompt (catching unknown inputs before generation).
  const unknownHandling = scoreUnknownHandling(inputPrompt, session);

  return {
    'next-token-accuracy': nextTokenAccuracy,
    'factual-consistency': factualConsistency,
    'repetition-rate': repetitionRate,
    'response-length-stability': responseLengthStability,
    'unknown-handling': unknownHandling,
  };
}

/**
 * Evaluates a single corpus entry: scores all metrics, detects regressions,
 * and attributes each regression to a failure bucket.
 *
 * @param corpusEntry - The held-out input/expected pair.
 * @param session - The session providing evaluation context.
 * @param metricBaselines - Per-metric baseline thresholds for regression detection.
 * @returns Scored entry shape including the attributed bucket.
 */
function evaluateCorpusEntry(
  corpusEntry: {
    readonly inputPrompt: string;
    readonly expectedResponse: string;
  },
  session: NeatChatSession,
  metricBaselines: Readonly<Partial<Record<EvaluationMetric, number>>>,
): {
  readonly metricScores: Readonly<Partial<Record<EvaluationMetric, number>>>;
  readonly isRegression: boolean;
  readonly attributedBucket: FailureBucket;
} {
  // Step 1: Score all metrics for this entry.
  const metricScores = scoreAllMetricsForEntry(corpusEntry, session);

  // Step 2: Check only metrics that have a declared baseline.
  // baseline is guaranteed to be defined by Object.keys iteration;
  // score is always populated by scoreAllMetricsForEntry for all EvaluationMetric keys.
  const hasRegression = (
    Object.keys(metricBaselines) as EvaluationMetric[]
  ).some((metric) => {
    const score = metricScores[metric];
    const baseline = metricBaselines[metric]!;
    return score !== undefined && isRegressionScore(metric, score, baseline);
  });

  if (!hasRegression) {
    return {
      metricScores,
      isRegression: false,
      attributedBucket: 'unattributed',
    };
  }

  // Step 3: Build a minimal regression entry to pass to the attribution function.
  const regressionEntry: RegressionEntry = {
    inputPrompt: corpusEntry.inputPrompt,
    expectedResponse: corpusEntry.expectedResponse,
    actualResponse: corpusEntry.expectedResponse,
    metricScores,
    attributedBucket: 'unattributed',
  };

  const attributedBucket = attributeToFailureBucket(session, regressionEntry);
  return { metricScores, isRegression: true, attributedBucket };
}

/**
 * Computes the arithmetic mean of each metric across all entries that evaluated it.
 *
 * Metrics absent from all entries are omitted from the result rather than set
 * to `0`, preserving the `Partial` contract from `RegressionSuiteResult`.
 *
 * @param allEntryScores - Array of per-entry partial metric score maps.
 * @returns Aggregated per-metric mean scores.
 */
function aggregatePerMetricMeans(
  allEntryScores: readonly Readonly<
    Partial<Record<EvaluationMetric, number>>
  >[],
): Readonly<Partial<Record<EvaluationMetric, number>>> {
  const perMetricMeans: Partial<Record<EvaluationMetric, number>> = {};

  for (const metric of ALL_EVALUATION_METRICS) {
    // Step 1: Collect scores for this metric across all entries.
    const metricScores = allEntryScores
      .map((entryScores) => entryScores[metric])
      .filter((score): score is number => score !== undefined);

    // Step 2: Compute mean only when at least one entry evaluated this metric.
    if (metricScores.length > 0) {
      perMetricMeans[metric] =
        metricScores.reduce((sum, score) => sum + score, 0) /
        metricScores.length;
    }
  }

  return perMetricMeans;
}

/**
 * Builds a bucket-to-count breakdown map from an array of attributed buckets.
 *
 * Buckets with zero regressions are omitted from the result consistent with
 * `RegressionSuiteResult.bucketBreakdown`.
 *
 * @param regressionBuckets - Ordered list of attributed buckets from regression entries.
 * @returns Partial breakdown of regression counts per failure bucket.
 */
function buildBucketBreakdown(
  regressionBuckets: readonly FailureBucket[],
): Readonly<Partial<Record<FailureBucket, number>>> {
  const breakdown: Partial<Record<FailureBucket, number>> = {};

  for (const bucket of regressionBuckets) {
    breakdown[bucket] = (breakdown[bucket] ?? 0) + 1;
  }

  return breakdown;
}

/**
 * Runs the NEATchat regression harness against a held-out evaluation corpus.
 *
 * For each corpus entry, the harness scores all five evaluation metrics
 * (`'next-token-accuracy'`, `'factual-consistency'`, `'repetition-rate'`,
 * `'response-length-stability'`, `'unknown-handling'`), detects regressions
 * against declared baselines, and attributes each regression to a
 * `FailureBucket`.
 *
 * The session is never mutated during evaluation: no weights, memory records,
 * routing log entries, or candidate log entries are modified. The harness
 * uses the expected corpus responses as the evaluation baseline rather than
 * running live inference, preserving the non-mutating contract.
 *
 * The `'repetition-rate'` metric treats the baseline as a ceiling (lower is
 * better): a regression is detected when the measured rate *exceeds* the
 * declared baseline. All other metrics treat the baseline as a floor.
 *
 * @param session - The live session providing vocabulary, memory, and routing context.
 *   Read-only during evaluation.
 * @param input - The harness input including corpus entries, metric baselines,
 *   and an optional OOV fraction threshold.
 * @returns Aggregated `RegressionSuiteResult` with per-metric means, total
 *   regression count, bucket breakdown, and entry count.
 *
 * @example
 * ```ts
 * const result = runNeatChatRegressionSuite(session, {
 *   session,
 *   corpusEntries: [{ inputPrompt: 'hello', expectedResponse: 'hi there' }],
 *   metricBaselines: { 'next-token-accuracy': 0.1, 'repetition-rate': 0.5 },
 * });
 * console.log(result.totalRegressions, result.perMetricMeans);
 * ```
 */
export function runNeatChatRegressionSuite(
  session: NeatChatSession,
  input: EvaluationHarnessInput,
): RegressionSuiteResult {
  // Step 1: Score and attribute each corpus entry individually.
  const evaluatedEntries = input.corpusEntries.map((corpusEntry) =>
    evaluateCorpusEntry(corpusEntry, session, input.metricBaselines),
  );

  // Step 2: Aggregate per-metric means across all entries.
  const perMetricMeans = aggregatePerMetricMeans(
    evaluatedEntries.map((evaluatedEntry) => evaluatedEntry.metricScores),
  );

  // Step 3: Collect regression entries and build the bucket breakdown.
  const regressionBuckets = evaluatedEntries
    .filter((evaluatedEntry) => evaluatedEntry.isRegression)
    .map((evaluatedEntry) => evaluatedEntry.attributedBucket);

  const bucketBreakdown = buildBucketBreakdown(regressionBuckets);

  return {
    perMetricMeans,
    totalRegressions: regressionBuckets.length,
    bucketBreakdown,
    entryCount: input.corpusEntries.length,
  };
}
