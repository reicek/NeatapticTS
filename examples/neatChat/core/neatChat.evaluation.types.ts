import type { NeatChatSession } from './neatChat.types';

/**
 * All scored dimensions used by the NEATchat regression harness.
 *
 * Each metric maps to a numeric score in `[0, 1]` (higher is better) or, for
 * `'repetition-rate'`, in `[0, 1]` where lower is better. Helper functions
 * in `neatChat.evaluation.services.ts` compute one score per metric from a
 * generated response and the held-out expected output.
 *
 * - `'next-token-accuracy'` — fraction of tokens that match the expected next
 *   token in the held-out exchange slice.
 * - `'factual-consistency'` — token-overlap score against user-profile memory
 *   facts present in the session at evaluation time.
 * - `'repetition-rate'` — fraction of n-grams repeated in the response;
 *   lower is better.
 * - `'response-length-stability'` — proximity of the actual token count to the
 *   expected token count, bounded to `[0, 1]`.
 * - `'unknown-handling'` — whether tokens absent from the vocabulary are
 *   gracefully handled rather than propagating a runtime error.
 *
 * @example
 * ```ts
 * const metric: EvaluationMetric = 'factual-consistency';
 * ```
 */
export type EvaluationMetric =
  | 'next-token-accuracy'
  | 'factual-consistency'
  | 'repetition-rate'
  | 'response-length-stability'
  | 'unknown-handling';

/**
 * Attribution category for a scored regression entry.
 *
 * When a regression (score below the expected baseline) is detected, the
 * harness assigns it to the subsystem most likely responsible. The
 * `'unattributed'` bucket is used when the failure cannot be confidently
 * traced to a single subsystem.
 *
 * - `'base-seed'` — regression likely caused by the pretrained seed weights or
 *   the default recurrent architecture.
 * - `'retrieval'` — regression correlated with episodic memory retrieval:
 *   either wrong records were recalled or recall changed generation behavior.
 * - `'routing'` — regression tied to the multi-path routing selection:
 *   the winning candidate path produced worse output than the base path would
 *   have.
 * - `'memory-compression'` — regression tied to memory-bank pruning or
 *   consolidation that dropped records needed by the current exchange.
 * - `'background-adaptation'` — regression observed after a background
 *   fine-tune candidate was promoted to the live session.
 * - `'unattributed'` — harness could not narrow the regression to a single
 *   subsystem; inspect `RegressionEntry.metricScores` manually.
 *
 * @example
 * ```ts
 * const bucket: FailureBucket = 'routing';
 * ```
 */
export type FailureBucket =
  | 'base-seed'
  | 'retrieval'
  | 'routing'
  | 'memory-compression'
  | 'background-adaptation'
  | 'unattributed';

/**
 * One scored evaluation entry covering a single held-out input/output pair.
 *
 * Each entry records the raw input prompt, the expected response from the
 * held-out corpus, the actual generated response, per-metric numeric scores,
 * and the attributed failure bucket when one or more metric scores fall below
 * the expected baseline.
 *
 * @remarks
 * - `metricScores` contains at most one entry per `EvaluationMetric`. Metrics
 *   that were not evaluated for this entry are omitted rather than set to `0`.
 * - `attributedBucket` is `'unattributed'` when all metrics pass or when the
 *   harness cannot narrow the failure to a single subsystem.
 *
 * @example
 * ```ts
 * const entry: RegressionEntry = {
 *   inputPrompt: 'What is my name?',
 *   expectedResponse: 'Your name is Alice.',
 *   actualResponse: 'I do not know.',
 *   metricScores: { 'factual-consistency': 0.1, 'next-token-accuracy': 0.2 },
 *   attributedBucket: 'retrieval',
 * };
 * ```
 */
export interface RegressionEntry {
  /** The raw prompt token string submitted to the session under evaluation. */
  readonly inputPrompt: string;
  /** The held-out expected response text from the evaluation corpus. */
  readonly expectedResponse: string;
  /** The actual generated response text produced by the session. */
  readonly actualResponse: string;
  /**
   * Numeric score in `[0, 1]` per evaluated metric.
   *
   * For `'repetition-rate'`, lower is better. For all other metrics, higher is
   * better. Omitted keys indicate the metric was not evaluated for this entry.
   */
  readonly metricScores: Readonly<Partial<Record<EvaluationMetric, number>>>;
  /**
   * Subsystem attributed as the most likely cause of any regression in this entry.
   *
   * Set to `'unattributed'` when all metrics pass the expected baseline or
   * when the harness cannot narrow the failure to a single subsystem.
   */
  readonly attributedBucket: FailureBucket;
}

/**
 * Aggregate result returned by `runNeatChatRegressionSuite`.
 *
 * The result bundles per-metric mean scores across all evaluated entries,
 * the total count of regression entries (entries with at least one metric
 * below the declared baseline), and a breakdown of regressions by attributed
 * failure bucket.
 *
 * @remarks
 * - `perMetricMeans` contains the arithmetic mean of each metric across all
 *   entries for which that metric was evaluated.
 * - `totalRegressions` is the count of entries with at least one metric score
 *   below the per-metric baseline declared in `EvaluationHarnessInput`.
 * - `bucketBreakdown` is a partial map from `FailureBucket` to integer count;
 *   buckets with zero regressions are omitted.
 *
 * @example
 * ```ts
 * const result: RegressionSuiteResult = {
 *   perMetricMeans: { 'next-token-accuracy': 0.82, 'repetition-rate': 0.03 },
 *   totalRegressions: 1,
 *   bucketBreakdown: { 'base-seed': 1 },
 *   entryCount: 10,
 * };
 * ```
 */
export interface RegressionSuiteResult {
  /**
   * Arithmetic mean score per metric across all entries in which that metric
   * was evaluated. Omitted keys indicate no entry evaluated that metric.
   */
  readonly perMetricMeans: Readonly<Partial<Record<EvaluationMetric, number>>>;
  /**
   * Total number of entries with at least one metric score below the declared
   * per-metric baseline.
   */
  readonly totalRegressions: number;
  /**
   * Per-bucket regression count. Buckets with zero regressions are omitted.
   */
  readonly bucketBreakdown: Readonly<Partial<Record<FailureBucket, number>>>;
  /**
   * Total number of entries evaluated in this suite run.
   */
  readonly entryCount: number;
}

/**
 * Minimal input contract for invoking `runNeatChatRegressionSuite`.
 *
 * The harness needs a live session (for vocabulary, memory bank, routing log,
 * and adaptation context), a held-out corpus of input/expected pairs, and the
 * per-metric score baselines below which an entry is flagged as a regression.
 *
 * @remarks
 * - `session` is used read-only during evaluation: no weights, memory records,
 *   or routing log entries are mutated during a harness run.
 * - `corpusEntries` is the held-out slice to evaluate; each entry carries the
 *   raw prompt and the expected response from the corpus.
 * - `metricBaselines` declares the minimum passing score per metric. Entries
 *   scoring below the baseline for any metric contribute to `totalRegressions`.
 *   Omitted baselines are treated as `0` (i.e., always passing) for that metric.
 * - `maxUnknownTokenFraction` is an optional threshold `[0, 1]` capping the
 *   fraction of out-of-vocabulary tokens before the `'unknown-handling'` metric
 *   is flagged. Defaults to `0` (any OOV token triggers the metric).
 *
 * @example
 * ```ts
 * const harnessInput: EvaluationHarnessInput = {
 *   session,
 *   corpusEntries: [
 *     { inputPrompt: 'Hello', expectedResponse: 'Hi there' },
 *   ],
 *   metricBaselines: {
 *     'next-token-accuracy': 0.1,
 *     'repetition-rate': 0.5,
 *     'response-length-stability': 0.5,
 *   },
 * };
 * ```
 */
export interface EvaluationHarnessInput {
  /**
   * The live NEATchat session to evaluate against.
   *
   * Read-only during evaluation: the harness never mutates session state.
   */
  readonly session: NeatChatSession;
  /**
   * Held-out input/expected pairs from the evaluation corpus.
   *
   * Each entry must supply the raw prompt string and the expected response
   * string as they appear in the held-out corpus.
   */
  readonly corpusEntries: readonly {
    /** Raw prompt text submitted to the session. */
    readonly inputPrompt: string;
    /** Expected response text from the held-out corpus. */
    readonly expectedResponse: string;
  }[];
  /**
   * Minimum passing score per metric.
   *
   * Entries scoring below the declared baseline for any metric are counted as
   * regressions. Omitted baselines default to `0` (always passing).
   */
  readonly metricBaselines: Readonly<Partial<Record<EvaluationMetric, number>>>;
  /**
   * Optional maximum fraction of out-of-vocabulary tokens before the
   * `'unknown-handling'` metric is flagged. Defaults to `0` when omitted.
   */
  readonly maxUnknownTokenFraction?: number;
}
