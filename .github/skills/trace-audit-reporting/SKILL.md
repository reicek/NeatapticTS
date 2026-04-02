---
name: trace-audit-reporting
description: 'Analyze Chrome trace or Perfetto trace captures, run scripts/analyze-trace/analyze-trace.ts, map hotspots to NeatapticTS source files, and generate a detailed performance report with findings, evidence, and an action plan. Use when auditing renderer, worker, GPU, requestAnimationFrame, postMessage, or long-task regressions.'
argument-hint: 'Describe the trace file, feature area, and target report file.'
user-invocable: true
disable-model-invocation: false
---

# Trace Audit Reporting

Use this skill when an agent needs to inspect a Chrome or Perfetto trace,
summarize hotspots, connect them back to workspace code, and produce a durable
performance report.

When the report lives in a continuing tracker file, `tracker-handoff` owns the
canonical plan/log structure and copy-paste continuation format.

## When to Use

- Analyze a Chrome trace export or Perfetto JSON capture.
- Investigate renderer, worker, GPU, `requestAnimationFrame`, or `postMessage`
  bottlenecks.
- Explain whether a regression is demo-layer, protocol-layer, or core-library.
- Write a report into a plan file such as `performance.plan.md`.
- Reuse the repository trace tooling instead of manually inspecting raw JSON.

## Demo-to-library policy

When a trace captured from a demo or example exposes a mismatch between obvious
user intent and the library's public behavior, treat the demo as a diagnostic
surface for the library rather than the final destination for a workaround.

- Prefer recommending a library-level API/default/runtime fix when the same gap
  could affect downstream users.
- Recommend demo-local optimizations only for genuinely demo-specific rendering,
  protocol, or presentation issues.
- In reports and action plans, state clearly whether a finding should be fixed
  in the library, in shared infrastructure, or only in the demo.

## Primary Resources

- [Trace analysis workflow](./references/trace-analysis-workflow.md)
- [Performance report template](./assets/performance-report-template.md)
- Companion skill: `trace-analyzer-extension` for modifying
  `scripts/analyze-trace/analyze-trace.ts` itself when new rollups or comparisons are needed.

## Standard Workflow

1. Confirm the trace file path and the target area being audited.
2. Read the nearest folder `README.md` files before opening implementation
   files so module boundaries are clear.
3. Run the repo analyzer:

   ```bash
   npm run trace:analyze -- <trace-path> --top=15
   ```

4. Extract the high-signal metrics first:
   - trace window and event count,
   - dropped frames,
   - thread totals,
   - longest `RunTask`, `FunctionCall`, `FireAnimationFrame`, and
     `HandlePostMessage` events,
   - hottest bundle or script names.
5. Read only the code needed to explain the top hotspots.
6. Separate findings into layers:
   - demo or app rendering,
   - worker or protocol,
   - core NeatapticTS runtime.
7. Write a report using the template resource, then tailor it to the actual
   trace evidence.

## Required Output Standards

- Do not stop at trace numbers. Tie each major hotspot to concrete source files.
- Distinguish user-visible stutter from background or scalability issues.
- State what is primary versus secondary in the trace.
- End with a prioritized action plan, not a flat list of ideas.
- Prefer reusable scripts under `scripts/` if the existing analyzer is missing
  an important aggregation.

## Guardrails

- Do not hand-wave from one long task. Use repeated totals and longest-event
  summaries together.
- Do not assume the worker is the bottleneck if renderer main-thread
  `FunctionCall` or `FireAnimationFrame` dominates.
- Do not prepend specific calendar dates to report headings, action-plan logs,
  or follow-up sections. Use stable undated titles that remain readable after
  later revisions.
- Do not invent a custom tracker format for continuing reports; use
  `tracker-handoff` for `[PLANNED]`, `[WIP]`, `[DONE]`, compression, and
  `Handoff query` structure.
- Do not edit generated `src/**/README.md` files. Improve source JSDoc instead.
- If you modify trace tooling, keep the script deterministic and documented with
  JSDoc.

## Repo-Specific Notes

- This repository already includes `scripts/analyze-trace/analyze-trace.ts` for compact,
  thread-aware trace audits.
- For work in `src/`, `examples/`, `benchmarks/`, or `testing/`, consult the nearest folder `README.md` before
  deep file reads.
- For substantial architecture findings, align the report with the relevant
  `plans/` document when one exists.
