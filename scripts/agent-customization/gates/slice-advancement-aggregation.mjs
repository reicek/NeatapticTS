#!/usr/bin/env node
/**
 * Aggregation rules for the consolidated `slice-advancement` gate.
 *
 * The consolidated gate runs several sub-gates sequentially and must decide
 * whether the slice as a whole passed. Three outcomes must stay distinct:
 *
 * - **Content failure** — the sub-gate ran and reported `pass: false`. The
 *   slice has a real problem and the consolidated gate fails.
 * - **Tooling error** — the sub-gate crashed or produced unparseable output.
 *   The check could not run, but nothing about the slice's content is known
 *   to be wrong, so this is reported and skipped.
 * - **Timeout** — the sub-gate was KILLED before it produced a verdict. The
 *   work it covers was never validated, so reporting the slice as green
 *   would be a FALSE GREEN. A timeout therefore fails the consolidated gate.
 *
 * This module is extracted from the gate script so the rules can be unit
 * tested without spawning the real sub-gates.
 */

/**
 * Summarizes the sub-gate results of a consolidated gate run.
 *
 * @param {object[]} results - Sub-gate results produced by the gate runner.
 * @returns {{ allPassed: boolean, failedGates: string[], erroredGates: string[], timedOutGates: string[], sub_gates: object[] }} Aggregation summary.
 */
export function summarizeSubGateResults(results) {
  const safeResults = Array.isArray(results) ? results : [];

  const erroredGates = safeResults.filter((result) => result.gate_error);
  const timedOutGates = safeResults.filter((result) => result.timed_out);
  const contentGates = safeResults.filter((result) => !result.gate_error);
  const failedGates = contentGates.filter((result) => !result.pass);

  // A timed-out sub-gate never produced a verdict, so it fails the
  // consolidated gate even though it is not a "content failure".
  const allPassed = failedGates.length === 0 && timedOutGates.length === 0;

  const sub_gates = safeResults.map((result) => ({
    name: result.gate,
    pass: result.pass,
    fixHint: result.fixHint,
    gate_error: result.gate_error,
    timed_out: result.timed_out,
    timeout_ms: result.timeoutMs,
    ...(result.gate_error ? { error: result.evidence?.error } : {}),
  }));

  return {
    allPassed,
    failedGates: failedGates.map((result) => result.gate),
    erroredGates: erroredGates.map((result) => result.gate),
    timedOutGates: timedOutGates.map((result) => result.gate),
    sub_gates,
  };
}
