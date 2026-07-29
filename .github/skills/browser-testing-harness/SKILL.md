---
name: browser-testing-harness
description: 'Use when: launching a visible browser test harness for smoke tests, GPU parity checks, or demo UI validation.'
argument-hint: 'Describe the browser test target, the active plan step, and any constraints such as headless vs visible window, GPU requirements, or CSP/worker delivery.'
user-invocable: false
disable-model-invocation: false
skills:
  - browser-build
  - chrome-devtools-mcp
  - visualizer-workflow
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Browser Testing Harness Playbook

Use this skill when NeatapticTS work needs a real browser window loaded to
validate runtime behavior that cannot be checked by Node-based unit tests alone.

This skill owns the durable workflow for visible-browser smoke tests,
demo UI validation, and GPU parity measurements. It is the companion to
`browser-build` (artifact generation) and `chrome-devtools-mcp` (profiling and
DOM inspection). When tracker files need updating, `tracker-handoff` owns the
plan/log shape.

## When to Use

- A slice requires real visible-window validation before it can be marked green.
- A browser demo or example needs a smoke test that loads the page and checks
  for runtime errors.
- GPU or rendering output must be measured on a real foreground browser window.
- Recurrent or worker-driven browser code needs an end-to-end activation check.

## When NOT to use

Do NOT use for Node-only unit tests - use `running-unit-tests` instead. Do NOT
use for performance optimization without a captured trace - use
`performance-optimization` instead. Do NOT use for bundle size or build
validation - use `browser-build` instead.

## Workflow Diagram

```text
Flowchart summary: "Need browser validation" → "Visible window required?"; "Visible window required?" → "Use browser-harness-specialist" (Yes), "Use Node tests" (No); "Use browser-harness-specialist" → "Capture console / DOM / trace"; "Capture console / DOM / trace" → "Report pass/fail".
```

## Required Workflow

1. Confirm the validation genuinely needs a browser runtime rather than a Node
   unit test or a `tsc`/`lint` check.
2. Identify whether the task needs a visible foreground window (GPU/rendering)
   or can run headless (pure console smoke tests).
3. Delegate to `browser-harness-specialist` for multi-step browser scenarios.
4. Record the browser version, visibility mode, and any console errors in the
   active plan's `VALIDATION_EVIDENCE`.

## Guardrails

- Do not claim a browser slice is green without a real window load when the
  plan requires visible-window validation.
- Do not use headless mode for GPU parity or rendering timing measurements.
- Do not let browser-specific message shapes leak into shared library APIs.

## Expected Final Output

A strong browser harness pass should report:

- the browser visibility mode used,
- the page or artifact loaded,
- any console errors or runtime failures,
- the pass/fail verdict for the validation target.
