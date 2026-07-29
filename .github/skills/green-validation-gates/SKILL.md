---
name: green-validation-gates
description: 'Use when: running or interpreting green-phase validation gates.'
argument-hint: 'Describe changed files, expected validations, latest failures, and whether strict customization validation should pass yet.'
user-invocable: false
disable-model-invocation: false
skills:
  - plan-sync-validation
  - tracker-handoff
  - coverage-guard
  - triaging-test-failures
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Green Validation Gates Playbook

Use this skill after implementation edits to confirm that the correct validation
surfaces pass before a workflow step can be marked `[DONE]`.

This skill owns the durable routing logic for green-phase validation in
NeatapticTS agent workflows. It matches the type of change to the narrowest
adequate validation surface, enforces the gate contract defined in
`.github/flows/`, and routes failures back to implementation rather than forward
to documentation.

When tracker updates are needed, `tracker-handoff` owns the plan/log shape.
When a plan or roadmap entry needs alignment confirmation, `plan-sync-validation`
owns that gate.

## When to Use

- A flow step declares exit gates and the agent must prove each gate returns
  `"pass": true` before marking the step done.
- Customization scripts, plan files, agents, skills, or flows were changed and
  need targeted validation before proceeding.
- A gate failed and the failure must be recorded and routed back to the
  implementation phase rather than bypassed.
- A gate has failed and the failure must be routed back to the
  implementation or red-testing phase for continued repair without an
  artificial retry threshold.
- A TypeScript, source, or package-script change requires build or lint
  confirmation alongside coverage verification.

## When NOT to use

Do NOT use for test repair - use `test-fix-workflow` instead. Do NOT use for plan consistency checking - use `plan-sync-validation` instead.

## Workflow Diagram

```text
Flowchart summary: "Implementation complete" → "Run tsc"; "Run tsc" → "Pass?"; "Pass?" → "Fix type errors" (No), "Run lint" (Yes); "Fix type errors" → "Run tsc"; "Run lint" → "Pass?"; "Pass?" → "Fix lint issues" (No), "Run focused jest" (Yes); "Fix lint issues" → "Run lint"; "Run focused jest" → "Pass?"; "Pass?" → "Triage failures" (No), "Run coverage-guard" (Yes); "Triage failures" → "Fix or escalate"; "Run coverage-guard" → "100%?"; "Fix or escalate"; "100%?" → "Gates passed" (Yes), "Fix coverage gap" (No); "Gates passed"; "Fix coverage gap".
```

## Task Packet

Pass a compact packet that names the changed files, the expected validation
surfaces, and whether strict customization validation should pass for the current
state.

```text
Use green-validation-gates after editing .github/flows/07.tracker-closure.flow.yml.
Changed files: 07.tracker-closure.flow.yml.
Expected validations: agent-graph gate (all references resolve), plan-sync gate.
Strict customization validation: yes, all declared gates should pass.
Latest failure: agent-graph gate — missing skill reference in step 3.
```

## Required Workflow

Run the narrowest validation that matches the changed surface. The full regression
matrix (`npm test`, `npm run test:silent`) chains multiple heavy test suites and
has crashed the host IDE; **never run it speculatively and never as a single shell
invocation**. Prefer focused Jest slices such as
`npx jest --config=jest.config.mjs --no-cache --testPathPattern=<path>`. Only run
the full matrix when explicitly requested by the user or required by the active
step packet's `validation` list; when required, execute it as separate, sequential
batched calls (`npm run build`, `npm run jest:base`, `npm run jest:esm-ts`,
`npm run jest:mjs`, `npm run lint`), each in its own shell invocation.

1. Run the narrowest validation that matches the changed surface:
   - TypeScript / source / package-script changes: `npm run build` or
     `npm run lint` as appropriate.
   - Docs-generation input changes: `npm run docs`.
   - Plan-only changes: markdown whitespace checks and `plan-sync-validation`.
   - `.github/agents/` changes: agent frontmatter and graph validation.
   - `.github/skills/` changes: skill frontmatter validation.
   - `.github/flows/` changes: flow gate resolution checks.
   - `src/` or `scripts/agent-customization/` source-file changes: the
     orchestrator MUST first run
     `node scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs --json --changed-files=<paths>`
     and confirm `pass: true` before dispatching Tier-3 specialists for
     review. After the smoke gate passes, run
     `node scripts/agent-customization/gates/code-coverage.gate.mjs --json`
     and confirm `pass: true` before marking the step `[DONE]`.

2. For customization scripts, run each script in audit mode first, then in
   targeted strict mode when the target state should hold.

3. Interpret gate output. Every gate in `.github/flows/` must return this
   structure:

   ```json
   {
     "pass": true,
     "evidence": { "...": "..." },
     "fixHint": "What to do if this gate fails",
     "owner": "gate-script-name.gate.mjs or validate-*.mjs"
   }
   ```

   A flow step is only complete when every declared gate returns `"pass": true`.

4. On gate failure:
   - Record the exception with:
     ```bash
     node scripts/agent-customization/gates/record-gate-exception.mjs
     ```
   - The exception is appended to `.github/ai-learning/learning-log.jsonl`.
   - Route the failure back to the implementation or red-testing phase.
   - Do not continue forward to documentation or plan closure.

5. Continue validation loop-backs until the issue is fully resolved or a true
   technical limit is reached. No concessions, no loop-back threshold.

6. Record all gate evidence (pass or fail) in the active plan before marking the
   step `[DONE]`.

## Tier-1 Gate Catalog

| Gate ID                | Check                                                                                      | Owner                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `plan-sync`            | Plan registered in README + Roadmap; status coherent                                       | `scripts/agent-customization/gates/plan-sync.gate.mjs`            |
| `step-packet`          | Active step has yaml block, status, next_step, validation, stop conditions                 | `scripts/agent-customization/gates/step-packet.gate.mjs`          |
| `plan-command-lint`    | Plan validation commands reference real CLI flags (no flag drift)                          | `scripts/agent-customization/gates/plan-command-lint.gate.mjs`    |
| `agent-graph`          | All flow/gate/agent references resolve to real files                                       | `scripts/agent-customization/gates/agent-graph.gate.mjs`          |
| `learning-event`       | A learning event exists for any gate exception or cross-tier call                          | `scripts/agent-customization/gates/learning-event.gate.mjs`       |
| `pre-specialist-smoke` | Narrowest Jest selection for changed files passes before Tier-3 specialists are dispatched | `scripts/agent-customization/gates/pre-specialist-smoke.gate.mjs` |
| `code-coverage`        | Changed `src/` and `scripts/agent-customization/` files are at 100%                        | `scripts/agent-customization/gates/code-coverage.gate.mjs`        |

## GPU Real-Device Gate

Any slice that touches `src/architecture/network/gpu/*` must pass the **GPU
Real-Device Gate** before it can be marked green. This gate is mandatory and
supersedes mock-only Jest validation.

Required gate evidence:

- Browser test file used (e.g., `docs/browser-tests/webgpu-nge-tier-benchmark.html`).
- GPU adapter info, including `vendor` and `architecture` from `navigator.gpu.requestAdapter().info`.
- Maximum absolute difference between CPU and GPU outputs (or latency/throughput metric when the slice is performance-oriented).
- Explicit confirmation that the browser window was **visible and in the foreground**
  during measurement (`browserVisibility: visible-foreground`).
- A statement that headless or minimized execution was **not** used.

Mock GPU test results (mock `GPUAdapter`/`GPUDevice` in Jest) do NOT satisfy this
gate. They are acceptable as pre-flight unit tests, but they cannot replace the
real-device visible-window measurement. If the gate cannot pass because no real
GPU is available in the execution environment, record `NOT OK — no real GPU
measurement` and route the slice back to implementation via the orchestrator.

## Decision Tree

```text
Flowchart summary: "Implementation complete" → "Changed surface?"; "Changed surface?" → "Run tsc + lint + coverage-guard" (src/ TypeScript), "Run agent-graph + frontmatter gates" (.github/agents/), "Run skill frontmatter validation" (.github/skills/), "Run plan-sync gate" (plans/); "Run tsc + lint + coverage-guard" → "All gates pass?"; "Run agent-graph + frontmatter gates" → "All gates pass?"; "Run skill frontmatter validation" → "All gates pass?"; "Run plan-sync gate" → "All gates pass?"; "All gates pass?".
```

## Before / After Examples

**Before:**

```json
{ "pass": false, "evidence": "tsc failed", "owner": "unknown" }
```

**After:**

```json
{
  "pass": true,
  "evidence": "tsc exit 0, 0 errors",
  "fixHint": "n/a",
  "owner": "npx tsc --noEmit"
}
```

## Guardrails

- Do not mark a phase step `[DONE]` if any declared gate has `"pass": false`.
- Do not route a gate failure forward to the next phase — always return to
  implementation or red testing first.
- Do not skip recording gate exceptions; every failure must be logged to
  `.github/ai-learning/learning-log.jsonl`.
- Do not run `npm run docs` unless docs-generation inputs were actually touched.
- Do not treat a partial gate pass (some gates green, some not yet run) as
  sufficient evidence to proceed.
- Do not invent ad hoc gate structure; use only the four-field JSON contract.

## Expected Final Output

A green-validation-gates pass should report:

- the changed surface and the validation commands run,
- the gate results for each declared gate (`pass`, `evidence`, `fixHint`,
  `owner`),
- whether any failures were recorded in `.github/ai-learning/learning-log.jsonl`,
- whether the failure was routed back to implementation or the step was confirmed
  complete,
- the final gate evidence recorded in the active plan.
