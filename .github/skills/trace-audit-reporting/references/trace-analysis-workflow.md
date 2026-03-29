# Trace Analysis Workflow

This reference explains how to use the repository trace analyzer and how to turn
its output into a useful engineering report.

## Inputs

Collect these inputs before starting:

- Trace path.
- Feature or subsystem being audited.
- Target report path.
- Any known user concern such as dropped frames, worker saturation, or GPU load.

## Command

Run the standard analyzer from the repository root:

```bash
npm run trace:analyze -- <trace-path> --top=15
```

Example:

```bash
npm run trace:analyze -- test/examples/flappy_bird/Trace-20260309T191949.json --top=15
```

## What the Analyzer Produces

The script prints these sections:

- `Trace`: file path, total time window, event count, dropped frames.
- `Thread Summary`: per-thread total duration, `RunTask` totals, long-task
  counts, and max task length.
- `Animation Frames`: aggregate duration distribution for
  `FireAnimationFrame` and `FunctionCall`.
- `Longest Events`: exact worst stalls with thread and script attribution.
- `Top Events`: event-name rollups such as `RunTask`, `GPUTask`, and
  `HandlePostMessage`.
- `Top Function Calls`: hottest bundles or script URLs.

## Interpretation Order

Read the output in this order.

### 1. Establish the bottleneck layer

Use `Thread Summary` and `Longest Events` to decide which layer dominates:

- Renderer main thread dominant:
  likely render-loop or main-thread orchestration cost.
- Dedicated worker dominant:
  likely simulation, inference, serialization, or protocol cost.
- GPU threads dominant:
  likely heavy canvas, image upload, filters, or compositing cost.

Do not rely on one section alone.

### 2. Identify user-visible symptoms

Focus on:

- dropped frames,
- long `RunTask`,
- long `FireAnimationFrame`,
- large renderer-side `FunctionCall` totals.

These are the best signals for visible jank.

### 3. Identify scalability risks

Focus on:

- worker-side `FunctionCall`,
- `HandlePostMessage`,
- repeated clone or transport costs,
- core activation or inference loops.

These may not be the current visible bottleneck but often become the next one.

### 4. Map hotspots to code

Read only the files that explain the top offenders. Start from the nearest
folder `README.md`, then inspect the implementation entry points.

Typical mapping examples:

- `FireAnimationFrame` plus renderer `FunctionCall`:
  inspect animation loop and render services.
- `HandlePostMessage`:
  inspect worker protocol, request helpers, and snapshot serialization.
- Worker `FunctionCall`:
  inspect simulation stepping, inference calls, and hot math loops.

## Report Writing Rules

Every report should include:

1. Scope.
2. Executive summary.
3. Trace summary with concrete numbers.
4. Detailed findings tied to source files.
5. Root-cause summary split by layer.
6. Prioritized action plan.
7. Validation plan for the next trace capture.

## Repo-Specific Heuristics for NeatapticTS

Use these heuristics when the trace touches evaluation or inference paths:

- Check whether a feed-forward workload is actually using the fast slab path.
- Verify whether acyclic constraints are explicit or merely implied.
- Treat renderer pain and runtime pain as separate questions.
- If the trace is app-heavy, do not over-rotate into core-library fixes.

## When to Extend the Analyzer

Modify `scripts/analyze-trace/analyze-trace.ts` only when the existing output cannot answer a
meaningful engineering question.

Good reasons to extend it:

- need a new event rollup,
- need better script attribution,
- need percentile or histogram data,
- need a deterministic comparison-friendly output section.

Poor reasons to extend it:

- one-off formatting preference,
- data that can already be inferred from the existing sections.
