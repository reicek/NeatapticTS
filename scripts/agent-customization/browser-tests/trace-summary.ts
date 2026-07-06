/**
 * @file Browser testing harness — compact, deterministic trace summary for
 * smoke scenario results.
 *
 * A trace summary preserves the scenario URL, wall-clock duration, success flag,
 * and a small set of derived metrics so an orchestrator can receive JSON output
 * without carrying the full browser-side state.
 */

/**
 * Result object emitted by the browser scenario under test.
 *
 * Concrete fields vary by scenario, but a well-behaved scenario should report
 * at least a `success` boolean and any numeric metrics it computed.
 */
export interface ScenarioResult {
  success: boolean;
  [key: string]: unknown;
}

/**
 * Metrics derived from a scenario result by {@link createTraceSummary}.
 */
export interface TraceMetrics {
  /** Number of top-level keys present in the scenario result. */
  keyCount?: number;
  /** `true` when the scenario reported `success === true`. */
  success?: boolean;
  /** Any additional metric reported by the scenario. */
  [key: string]: unknown;
}

/**
 * Output of {@link createTraceSummary}: a compact, serializable summary of a
 * browser scenario trace.
 */
export interface TraceSummary {
  /** URL of the scenario page that produced the trace. */
  scenarioUrl: string;
  /** Wall-clock duration of the scenario in milliseconds. */
  durationMs: number;
  /** Whether the scenario reported success. */
  success: boolean;
  /** Derived metrics from the scenario result. */
  metrics: TraceMetrics;
}

/**
 * Input captured by the harness while running a browser scenario.
 *
 * This is the canonical shape consumed by {@link createTraceSummary}. Callers
 * may either supply this directly or compute `durationMs` and `success`
 * themselves from richer browser-side timestamps.
 */
export interface TraceInput {
  /** URL of the scenario page. */
  scenarioUrl: string;
  /** Wall-clock duration of the scenario in milliseconds. */
  durationMs: number;
  /** Whether the scenario reported success. */
  success: boolean;
  /** Optional metrics object; a normalized object is provided if omitted. */
  metrics?: TraceMetrics;
}

/**
 * Build a compact, deterministic summary of a browser scenario trace.
 *
 * The summary is intentionally small so it can be emitted back to an
 * orchestrator as JSON without carrying full browser-side state. It preserves
 * the scenario URL, runtime, success flag, and a stable set of derived
 * metrics.
 *
 * @param input - Duration, success flag, and optional metrics from the browser.
 * @returns A serializable trace summary.
 * @throws {Error} if `input` is missing required fields.
 *
 * @example
 * ```ts
 * const summary = createTraceSummary({
 *   scenarioUrl: 'http://localhost:8080/docs/browser-tests/webgpu-inference-smoke.html',
 *   durationMs: 500,
 *   success: true,
 *   metrics: { maxAbsDiff: 0.05 },
 * });
 * console.log(summary.durationMs); // 500
 * ```
 */
export function createTraceSummary(input: TraceInput): TraceSummary {
  if (input == null || typeof input.scenarioUrl !== 'string') {
    throw new Error('Trace input must include a scenarioUrl string.');
  }
  if (typeof input.durationMs !== 'number') {
    throw new Error('Trace input must include a durationMs number.');
  }

  const success = input.success === true;
  const metrics: TraceMetrics = input.metrics ?? {
    keyCount: 0,
    success,
  };

  return {
    scenarioUrl: input.scenarioUrl,
    durationMs: Math.max(0, input.durationMs),
    success,
    metrics,
  };
}
