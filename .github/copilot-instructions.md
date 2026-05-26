# Repo Copilot Instructions — NeatapticTS

See [agent-skill-routing-table.md](agent-skill-routing-table.md) for the generated canonical agent and skill routing table.

## Mandatory routing policy

> **This section is mandatory and takes precedence over every other section in this file.**

The default/main Copilot agent **MUST NOT** perform substantive work directly. All implementation, refactoring, research, planning, testing, documentation, and logging work **MUST** be routed to the smallest relevant numbered SDLC orchestrator before any file is read, written, or modified.

### Numbered SDLC orchestrators (route here exclusively)

| Agent              | Domain                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `00-helping`       | AI-system maintenance, workflow gaps, CI/configuration, safe customization fixes         |
| `01-planning`      | Planning, decomposition, risk analysis, acceptance criteria, test strategy               |
| `02-researching`   | Codebase research, API/dependency exploration, architecture reconnaissance               |
| `03-red-testing`   | Failing tests, test plans, fixtures, assertions, coverage strategy before implementation |
| `04-implementing`  | Scoped code changes, focused implementation, pattern reuse                               |
| `05-green-testing` | Running tests, triaging failures, fixing regressions, validating behavior                |
| `06-documenting`   | JSDoc, user-facing docs, API docs, examples, changelogs                                  |
| `07-logging`       | Session summaries, decisions, evidence, files touched, next steps                        |

### Routing rules

1. **Identify the smallest relevant orchestrator** for the request before doing anything else.
2. **Delegate immediately** — do not begin substantive work before routing.
3. **For whole-plan execution**: call `01-planning` first so it authors step packets, then dispatch remaining numbered orchestrators in order, waiting for completion before advancing.
4. **Narrow exceptions** (no routing required): trivial factual answers (single-sentence lookups with no file changes) and direct operator commands that involve zero file reads/writes (e.g., "what is the test command?").

### Flow, gate, and universal-helper routing

- Numbered SDLC agents select a named flow from `.github/flows/` to execute their body work.
- Every flow declares exit gates (from the Tier-1/Tier-2 gate catalog) that must return `{pass: true, evidence, fixHint, owner}` JSON before completion.
- Post-phase fan-out runs after the flow body completes.
- Gate exceptions are recorded via `record_gate_exception` and appended to `.github/ai-learning/learning-log.jsonl`.
- Three consecutive gate failures in a session trigger automatic escalation to `00-helping` via `00.cross-tier-helper`.
- Cross-tier helper calls from any numbered agent (01-07) route to `00-helping`, which resolves the blocker and returns a resolution summary; all cross-tier calls are logged as learning events visible to `00.workflow-gap-audit`.

### Certainty and investigation thresholds

These thresholds govern every routing decision and every user-facing response.

- End every user-facing response with `(Certainty: NN%)`.
- If certainty is below 90%, stop and investigate before proceeding.
- If certainty is below 95%, investigate further and ask follow-up questions until the requirements and environment are clear enough.

## Agent delegation tier graph

The NeatapticTS repo enforces a **5-layer agent delegation tier graph**. Every `.github/agents/*.agent.md` file must carry a `tier: <N>` YAML frontmatter field. The tier field is the delegation-policy field; it is distinct from the uppercase `TIER:` key that appears inside structured-v1 output-contract body text.

| Tier | Label                                  | Examples                                                | `user-invocable` |
| ---- | -------------------------------------- | ------------------------------------------------------- | ---------------- |
| 0    | Default / Main                         | Default VS Code Copilot agent                           | —                |
| 1    | Numbered SDLC Orchestrators            | `00-helping` through `07-logging` (8 agents)            | `true`           |
| 2    | Named coordinators / sub-orchestrators | `planning-context-coordinator`, `solid-split`, etc.     | `false`          |
| 3    | Hidden scouts and specialists          | `Boundary Mapper`, `Coverage Scout`, `Plan Scout`, etc. | `false`          |
| 4    | Auxiliaries and one-shot helpers       | `acceptance-criteria-writer`, `file-change-summarizer`  | `false`          |

### Enforced delegation rules

- Tier 1 may delegate to Tier 2, 3, or 4.
- Tier 2 may delegate to Tier 3 or 4.
- Tier 3 may delegate to Tier 4 only.
- Tier 4 may not delegate to any agent.
- No tier may call a higher-numbered tier except via the `00.cross-tier-helper` escalation path.
- `user-invocable: true` is valid **only** for Tier 1 agents (the 8 SDLC orchestrators).

**Validated counts (enforced, 57 agents total):** Tier 1 = 8, Tier 2 = 10, Tier 3 = 35, Tier 4 = 4.

### Operator commands

```sh
# Full tier inventory (JSON)
node scripts/agent-customization/tier-inventory.mjs --json

# Validate delegation graph against tier rules
node scripts/agent-customization/validate-agent-graph.mjs --json

# Validate agent frontmatter fields (strict mode — fails on any gap)
node scripts/agent-customization/validate-agent-frontmatter.mjs --json --strict

# Tier enforcement gate (standard { pass, evidence, fixHint, owner } contract)
node scripts/agent-customization/gates/tier-enforcement-gate.mjs --json

# Agent body structure and output-contract compliance validator
node scripts/agent-customization/validate-agent-quality.mjs --json
# npm alias: npm run agents:validate-quality

# Agent-quality gate
node scripts/agent-customization/gates/agent-quality.gate.mjs --json
# npm alias: npm run agents:quality:gate

# Human-readable tier audit report (markdown)
node scripts/agent-customization/tier-audit-report.mjs

# Live MCP query (via neataptic-gate-mcp server, tool: query_tier_graph)
# Returns current inventory, violation list, and summary counts at runtime.
```

When adding or reshaping any agent, re-run `validate-agent-graph.mjs` to confirm the tier contract is intact. Any new agent without a valid `tier:` field, or with a `user-invocable: true` flag at Tier 2–4, will cause the gate to fail with a structured violation report.

## Skill and companion-agent routing

- Skills own durable knowledge: workflow, standards, guardrails, tone models, source-mapping rules, validation expectations, and handoff contracts.
- Companion agents stay thin and task-shaped: they gather evidence, map boundaries, scout drift, or execute one narrow workflow step while deferring durable policy to the relevant skill.
- When a skill and a companion agent overlap, update the agent to follow the skill rather than copying the overlap forward.
- Only the eight numbered SDLC orchestrators are directly user-invocable; route through the smallest relevant orchestrator and let it delegate into hidden coordinators, specialists, and skills.

### Skills

| Skill                          | Invoke when                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `solid-split`                  | Splitting or refactoring a medium/large module into folder-based submodules with stepwise sequencing                                       |
| `educational-docs`             | Documentation quality, JSDoc improvement, generated-README tone, Mermaid diagrams, citations, Wikimedia-safe visuals                       |
| `flappy-architecture-polish`   | Tuning, rerunning, or hardening one Flappy Bird architecture profile in the browser-worker path                                            |
| `visualizer-workflow`          | Browser-demo visualizer layout, overflow, hover/tooltip reliability, or cross-demo parity                                                  |
| `test-fix-workflow`            | Systematically repairing multiple failing tests (red suite)                                                                                |
| `plan-alignment`               | Aligning a change with roadmap intent or a specific plan document                                                                          |
| `tracker-handoff`              | Creating, compressing, or closing `.plans.md` / `.logs.md` files, `Handoff query` sections, terminal closure                               |
| `agent-frontmatter-standards`  | `.agent.md` naming, tools, model arrays, handoffs, visibility, frontmatter validation                                                      |
| `model-routing-and-budget`     | Assigning Full vs Mini model tiers and validating qualified Copilot model strings                                                          |
| `phase-handoff-workflow`       | Numbered SDLC handoff prompts, stop conditions, forward-only phase transitions                                                             |
| `subagent-delegation-patterns` | Compact specialist task packets, sequential vs parallel delegation, output contracts                                                       |
| `agent-inventory-audit`        | Agent/skill inventory and customization drift evidence                                                                                     |
| `skill-description-evals`      | Should-trigger and should-not-trigger description evals for skills                                                                         |
| `skill-output-evals`           | Evidence-backed skill output grading and baseline comparisons                                                                              |
| `agent-script-tooling`         | Noninteractive customization scripts with `--help`, JSON output, stderr diagnostics, idempotency                                           |
| `license-attribution-audit`    | Source attribution and license notes when external standards inform repo customizations                                                    |
| `plan-sync-validation`         | Keeping active trackers, `plans/README.md`, and `plans/Roadmap.md` aligned                                                                 |
| `red-test-contracts`           | Red-phase tests or eval assertions authored before behavior changes                                                                        |
| `green-validation-gates`       | Focused post-change validation and rerouting failed checks                                                                                 |
| `docs-academic-citation-audit` | Educational docs, Mermaid, citations, and generated-README quality audits                                                                  |
| `worker-inference-transport`   | Worker-friendly inference payloads, transport ladders, transfer semantics, browser/Node worker parity                                      |
| `multithread-evaluation`       | Ordered worker-pool batch evaluation, queueing, dataset shipping, single-thread fallback                                                   |
| `checkpointing-persistence`    | Versioned full/light checkpoints, strict restore behavior, durable resume state                                                            |
| `hybrid-training-interop`      | Deterministic parameter-vector layouts, isolated fine-tuning, persistence policy                                                           |
| `reproducibility-contracts`    | Seed/replay/ordering language and exact-vs-bounded determinism contracts                                                                   |
| `nge-core-algorithm`           | Phase 7 `NGE_DNA`, deterministic development, lifecycle policy, memory tiers, neuromodulation, reproduction                                |
| `nge-benchmark-workflow`       | Phase 7 benchmark methodology, curricula, fairness contracts, observability, demo-harness evaluation                                       |
| `neatchat-systems`             | Dependency-gated NEATchat follow-up system: memory tiers, retrieval/routing, branchable conversational state                               |
| `repo-cortex-workflow`         | Cortex index freshness, corpus rebuild, snapshot regeneration, `cortex-index.gate.mjs`, MCP plan path override                             |
| `coverage-tranche`             | Expanding coverage on passing code toward 100%; file-by-file tranche progression from the lowest-covered boundary                          |
| `coverage-guard`               | Enforcing 100% statements, branches, functions, and lines after any `src/` file is touched; an enforcement gate, not an expansion workflow |
| `browser-build`                | Browser runtime artifacts, docs asset bundles, smoke gates, CDN/runtime packaging, and browser-env aliasing                                |

### Companion agents

| Agent                     | Hands off to                                                      |
| ------------------------- | ----------------------------------------------------------------- |
| `Boundary Mapper`         | `solid-split`                                                     |
| `Coverage Scout`          | `coverage-tranche`                                                |
| `Coverage Guard` (agent)  | `coverage-guard`                                                  |
| `Docs Scout`              | `educational-docs`                                                |
| `Plan Scout`              | `plan-alignment`                                                  |
| `Visualizer Scout`        | `visualizer-workflow`                                             |
| `Worker Payload Scout`    | `worker-inference-transport`                                      |
| `Evaluation Pool Scout`   | `multithread-evaluation`                                          |
| `Checkpoint Scout`        | `checkpointing-persistence`                                       |
| `Hybrid Interop Scout`    | `hybrid-training-interop`                                         |
| `Browser Runtime Scout`   | `browser-build`                                                   |
| `Determinism Scout`       | `reproducibility-contracts`                                       |
| `NGE Core Scout`          | `nge-core-algorithm`                                              |
| `NGE Benchmark Scout`     | `nge-benchmark-workflow`                                          |
| `NEATchat Scout`          | `neatchat-systems`                                                |
| `Repo Cortex Scout`       | `repo-cortex-workflow`                                            |
| `Cortex Embeddings Scout` | `Semantic_Knowledge_Embeddings` (planned — skill not yet on disk) |

### Canonical routing table

- The generated canonical routing table lives at [agent-skill-routing-table.md](agent-skill-routing-table.md) and is the shared read-only snapshot for current agent and skill routing metadata.
- Refresh it with `npm run agents:routing-table` after changing `.github/agents/*.agent.md` or `.github/skills/*/SKILL.md`.
- Validate freshness with `npm run agents:routing-table:gate` or `node scripts/agent-customization/gates/routing-table-freshness.gate.mjs --json`.
- Every `.github/agents/*.agent.md` file must declare a `skills: [...]` frontmatter field, even when the list is empty.

## Workflow protocols

### MCP workflow snapshot protocol (session-start handling)

When `neataptic-workflow-mcp: get_active_workflow_snapshot` returns `scope: "no-active-phase"`:

- **Do not treat it as a blocking error.** It is a normal startup condition when the active plan has no `[WIP]` phase/step yet (new workstream just started or session resumed before any phase was advanced).
- **Fall back immediately** to a direct read of the plan file (read the `## Implementation phases` section) to determine which phase and step to activate next.
- **Auto-advance Phase 1 Step 01** to `[WIP]` in the plan file if the plan is brand-new (all phases `[PLANNED]`), then retry the snapshot or continue with direct file context.
- When using `plan-session-redirect.mjs` to switch sessions to a new workstream plan: run the redirect **after** Phase 1 Step 01 has been marked `[WIP]` in the plan, not before.
- The perpetual binding (`plans/mcp-active-binding.plans.md`) always has a valid `[WIP]` phase and is the safe fallback if `plan-session-redirect.mjs --clear` is run or no session override exists.
- `neataptic-workflow-mcp` is the only MCP that degrades gracefully for this case; `neataptic-validation-mcp` still requires a strict `[WIP]` step and will error correctly.

### Long-task logging and communication

- Use the compressed logging convention already established in this repo for long tasks: prefer short pass-style entries that record what changed, what remains, and the next concrete target without replaying full transcript detail.
- Keep chat communication to brief confirmations and step transitions only. Prefer one or two short sentences when moving to the next step unless the user explicitly asks for more detail.
- Prefer communicating ongoing work through markdown tracker files instead of chat when the task spans multiple steps.
- Treat `plans/` as the active tracker surface and `plans/completed/` as the archive for terminally closed tracker baselines and their matching logs.
- Use `.plans.md` files for work in progress, pending decisions, next steps, and handoff context.
- Use `.logs.md` files for completed work, concise pass history, and done-state records.
- When creating or reshaping tracker files, use `tracker-handoff` as the canonical workflow for `[PLANNED]`, `[WIP]`, `[DONE]`, compression, and `Handoff query` structure.
- When a workstream reaches `[DONE]`, the final tracker step is to compress the completed `.plans.md` into a short closed tracker, add or update a same-boundary `.logs.md` audit record, and move both files into `plans/completed/` before beginning the next workstream.
- Handoff prompts are strict for active trackers and blocker recovery, but omit them for fully closed trackers unless the user explicitly asks for reopen guidance.
- When both chat and tracker files are available, treat the tracker files as the primary source of detailed continuity and keep chat as a thin status layer.

### TDD-first execution policy

When a task changes behavior, fixes a regression, or deepens a refactor with meaningful runtime risk, prefer a TDD sequence instead of implementation-first work.

- Start by adding or updating the smallest targeted test that should fail for the intended behavior or bug fix.
- Make that narrow surface go red first when the task is not purely documentation, search, or mechanical rename work.
- Then implement the code change until the targeted test or test slice goes green.
- After the green step, expand coverage for the new code and the directly related boundary toward >95% when practical and safe.
- Keep the red/green loop narrow. Do not jump to broad suite runs before the active boundary is green.
- For coverage passes, prefer dedicated owner-local `*.test.ts` files and keep tests aligned with repo conventions such as AAA structure, nested `describe` blocks, and one top-level `expect(...)` per test.

### Multi-test failure repair

Invoke `test-fix-workflow`; do not re-state its protocol here.

## Code standards

### ES2023-first policy (strict)

For educational clarity and a modern look, always prefer idiomatic ES2023 syntax when it improves readability or safety without changing behavior. This repo is intentionally opinionated: use the immutable array methods and modern language constructs by default.

Prefer (non-exhaustive):

- Arrays: `toSorted`, `toReversed`, `toSpliced`, `with`, `.at(-1)`, `findLast`, `findLastIndex`.
- Objects: spread/rest over `Object.assign` for shallow copies and merges.
- Optional chaining `?.` and nullish coalescing `??` (avoid `||` for defaulting unless you mean falsy semantics).
- Deep clone: `structuredClone` (or the project helper `safeStructuredClone` when cross-env safety is needed).
- Errors: `Error` with `{ cause }` (e.g., `new Error(msg, { cause })`).
- Numerics: numeric separators for long literals (for readability only, not to change values).
- Modules: ES modules `import`/`export` over CommonJS `require`/`module.exports` (follow the repo's phase plan; new code should be ESM).

Avoid (legacy/less clear):

- In-place `sort`, `reverse`, `splice` in code paths that expect immutability; use the ES2023 immutable variants above.
- `Object.assign({}, obj)` or `Object.assign([], arr)` for cloning; use object/array spread.
- `JSON.parse(JSON.stringify(x))` for deep clone; use `structuredClone`/`safeStructuredClone`.
- Index math like `arr[arr.length - 1]`; prefer `arr.at(-1)` when readability benefits.
- CommonJS `require()` in new or refactored modules; prefer ESM.

### Standard module architecture

When splitting medium or large modules in `src/`, `examples/`, `benchmarks/`, or `testing/`, prefer a dedicated folder-based module boundary instead of accumulating many sibling files at the parent level.

Use this naming convention for a module named `module`:

- `module/module.ts`
- `module/module.utils.ts`
- `module/module.types.ts`
- `module/module.errors.ts`
- `module/module.services.ts`
- `module/module.constants.ts`

Use this naming convention for a nested sub-module named `sub-module` inside `module`:

- `module/sub-module/module.sub-module.ts`
- `module/sub-module/module.sub-module.utils.ts`
- `module/sub-module/module.sub-module.types.ts`
- `module/sub-module/module.sub-module.errors.ts`
- `module/sub-module/module.sub-module.services.ts`
- `module/sub-module/module.sub-module.constants.ts`

Interpretation rules:

- `*.ts`: main orchestration and primary public surface for the module.
- `*.utils.ts`: helper logic that is not the main orchestration path.
- `*.types.ts`: interfaces, DTOs, context/result objects, and narrow contracts.
- `*.errors.ts`: module-local error classes and error helpers.
- `*.services.ts`: side-effecting or stateful services used by orchestration.
- `*.constants.ts`: named constants, lookup tables, and local config values.

Architecture rules:

- Do not create a folder for every tiny file; use this pattern when a file has become a real subsystem.
- Keep the main `module/module.ts` file orchestration-first and declarative.
- Prefer subfolders over continued file sprawl when a module develops a clear internal subsystem.
- Avoid one-off naming patterns during refactors; once a module is folderized, keep all follow-up files in the same naming scheme.
- Replace broad catch-all files with narrower module-owned `*.types.ts`, `*.services.ts`, or `*.utils.ts` files rather than recreating another hub.

### Strict rules to enforce

Apply to any suggestion touching `src/`, `testing/`, `benchmarks/`, or `examples/`.

1. Naming: avoid short local identifiers. Do not use these short names for non-trivial locals: `dx`, `dy`, `d`, `i`, `a`, `b`, `c`, `p`, `o`, `cand`, `tries`, `idx`.
   - If the original code uses a short name in a tiny loop (1–3 lines) and it is clearly idiomatic, allow `i`, `j` only.
   - Prefer descriptive names: `candidateDirection`, `bestDistance`, `currentPosition`.

2. JSDoc: exported classes/functions/constants and public methods must have JSDoc with `@param` and `@returns` where appropriate. Add short `@example` when behavior is non-obvious.

   JSDoc-for-constants rule: All exported or shared default constants in `src/`, `testing/`, `benchmarks/`, and `examples/` must include a concise educational JSDoc explaining what the value controls (e.g., decay factor meaning, floor rates). Keep descriptions short and clarifying.

3. Tests: follow the single-expect rule. Each `it()` (or `test()`) must have exactly one top-level `expect(...)` statement. If multiple assertions are needed, split into multiple `it()` cases or use helper assertions.

4. Constants: replace magic numbers with named `export const` or class-private `static #` constants with a short JSDoc.

5. Comments: methods should have step-level inline comments explaining intent (not every line). Use numbered steps where helpful.

6. Lookup tables and enums: prefer a single table/enum for small fixed mappings (for example direction deltas) and helper methods like `#opposite(direction)` rather than scattered arithmetic.

7. Types: avoid `any` and `unknown` in `src/`, `testing/`, `benchmarks/`, and `examples/`. Use precise types or `// eslint-disable-next-line @typescript-eslint/no-explicit-any` with a short justification comment.

8. Local helper structure preference:
   - For new or refactored functions that introduce internal helpers, order the function as:
     1. Local variables/constants at top
     2. Declarative logic (calls to helpers)
     3. Return (fold)
     4. Internal helper function declarations at the end of the parent function
   - Helpers should be small and pure where practical, with step-level inline comments and JSDoc.

9. Mandatory implementation pattern (always; keep cognitive complexity low):
   - Applies to all new code and any modified/refactored code in `src/`, `testing/`, `benchmarks/`, and `examples/`.
   - Prefer a _declarative top-level flow_ ("collect → transform → fold/return") over deeply nested control flow.
   - Avoid ternary chains (especially nested) for multi-branch fallback logic; use named resolver helpers with early returns instead.
   - When normalizing legacy/loose data, isolate type assertions/casting into a single helper and keep the rest strongly typed.
   - Keep helpers after the fold, and give each helper a single responsibility (SOLID: SRP). If the logic reads like a decision tree, it likely wants 2–4 small helpers.

10. General multi-pass decomposition requirements (apply to all medium/large refactors):
    - Perform refactors in explicit passes, in this order unless unsafe:
      1. Stabilize current behavior and identify seams
      2. Extract pure helpers by responsibility
      3. Introduce typed context/result objects to reduce parameter sprawl
      4. Simplify top-level flow to orchestration only
      5. Fold repeated logic into collect/transform/fold helpers
    - All new complex methods should follow a declarative above-the-fold structure: keep the exported or top-level method as step-oriented orchestration, and place the actual logic in small SRP private helpers below the fold.
    - Keep the top-level method declarative and linear, with numbered inline comments (`Step 1`, `Step 2`, ...).
    - Ensure each helper has one reason to change (SRP), very low cognitive complexity, and descriptive naming.
    - Place helper declarations after the top-level return/fold where language/style allows.
    - Prefer immutable pass-style transforms (`map`, `filter`, `reduce`, index-collection helpers) over mixed mutation-heavy loops.
    - When passing more than 3-4 arguments repeatedly, introduce a typed context object and shared result types.

### Code structure example

```ts
export function exampleMethod(input: Input) {
   const constants = /* ... */;
   const locals = /* ... */;

   if (/* guard */) return /* fold */;

   const stepOne = helperOne(input, locals, constants);
   const stepTwo = helperTwo(stepOne, locals, constants);
   return helperThree(stepTwo, locals, constants);

   /** @param value - Input. @returns Intermediate. */
   function helperOne(value: Input, _locals: unknown, _constants: unknown): Intermediate {
      // Step 1: ...
      return /* ... */;
   }

   /** @param value - Intermediate. @returns Intermediate. */
   function helperTwo(value: Intermediate, _locals: unknown, _constants: unknown): Intermediate {
      // Step 1: ...
      return /* ... */;
   }

   /** @param value - Intermediate. @returns Output. */
   function helperThree(value: Intermediate, _locals: unknown, _constants: unknown): Output {
      // Step 1: Fold/return.
      return /* ... */;
   }
}

type Input = unknown;
type Intermediate = unknown;
type Output = unknown;
```

### Before-submit validation checklist

When you modify or create files under `src/`, `testing/`, `benchmarks/`, or `examples/`, run (or advise running) these quick validations. If you cannot run them, still ensure your suggestion would pass them.

- **TypeScript**: run `npm run build` (or `npx tsc --noEmit -p tsconfig.json`) and report pass/fail.
- **Clean install**: when `package.json`, `package-lock.json`, or workflow/runtime tooling changes, run `npm ci` and report pass/fail.
- **Test-expect heuristic**: flag test files that contain more than one top-level `expect(` per `it()` — split into multiple `it()` blocks.
- **JSDoc**: for new exported symbols, ensure a JSDoc block with `@param`/`@returns` exists (or flag if missing).
- **CI Linux/Chromium**: when the change can invoke Mermaid, Puppeteer, or Chromium in CI, validate `npm run docs`; do not assume local non-Linux success generalizes to GitHub-hosted Linux runners.
- **ES2023 modernization**: list any flagged legacy patterns and intended replacements (`Object.assign` → spread, `arr[arr.length-1]` → `arr.at(-1)`, `JSON.parse(JSON.stringify())` → `structuredClone`).

Local typecheck one-liner (PowerShell):

```powershell
npx tsc --noEmit -p tsconfig.json
```

## Documentation standards

### Educational docs preference (JSDoc)

This is a public-facing, educational library. JSDoc comments are compiled into user-facing documentation (for example the aggregated READMEs under `src/**/README.md` generated by the docs workflow).

When you touch code under `src/`, `testing/`, `benchmarks/`, or `examples/`, prefer improving JSDoc so the generated docs are:

- **Interesting and explanatory**, not just type signatures.
- **Example-driven**: include small examples in the main description (prefer fenced code blocks like ```ts) so the docs generator preserves them.
- **Conceptual**: include a brief "what/why" explanation and any important semantics (defaults, invariants, error cases, performance notes).
- **Atemporal**: public docs should read as current conceptual guidance, not as repo chronology.

Public documentation rule for `src/**`, `examples/**`, `benchmarks/**`, and `testing/**` README or JSDoc surfaces:

- never reference internal plans, tracker steps, roadmap phases, pass labels, or chat-only context unless the user explicitly asks for process, migration, or historical documentation,
- never structure public docs as repo before/after comparisons,
- keep public docs focused on current concepts, boundaries, invariants, tradeoffs, and reading paths,
- keep plan, rollout, migration, and tracker language in `plans/`, `.logs.md`, PR text, or release notes instead of README openings.

When a diagram would teach faster than prose, prefer Mermaid Markdown in the documentation surface. Use diagrams for architecture overviews, data flows, decision flows, state transitions, entity relationships, and simple quantitative views when they materially improve comprehension.

When styling those documentation visuals, match Astro Bird's neon-retro-arcade direction: dark backgrounds, blue and cyan structural lines, high-contrast readable labels, and restrained warm neon accents or glow only for the primary highlight. Prefer consistency and contrast over decorative intensity.

Keep examples short, dependency-light, and consistent with the current public API (avoid imaginary helpers or absolute file paths).

Keep this file focused on repo-level policy and invocation rules. The richer tone model, source mapping workflow, visual style guide, Mermaid diagram playbook, citation guidance, and Wikimedia Commons media rules belong in `educational-docs`.

### Generated README handling

Folder `README.md` files inside `src/` are generated artifacts and should be treated as read-only during normal editing.

When a generated `src/**/README.md` is lacking, improve the associated JSDoc in the source files that feed it instead of editing the README directly.

When a generated `src/**/README.md` appears outdated relative to the code or JSDoc:

- do not hand-edit the generated README,
- run `npm run docs` to refresh generated documentation when needed,
- consider `educational-docs` pre-approved to run `npm run docs` after doc-affecting edits so README files stay synchronized and drift does not confuse later work.

### Generated example publication handling

Published browser demo pages under `docs/examples/**/index.html` are generated artifacts copied from `examples/**/index.html` by `scripts/copy-examples.ts` during `npm run docs`.

When changing browser-demo HTML, CSS, or loader behavior:

- treat `docs/examples/**` as read-only generated output,
- edit the source entrypoint under `examples/**/index.html`,
- run `npm run docs` to republish the generated copy,
- verify the generated page after the docs run instead of patching it directly.

If a change appears to require touching both the source example page and the published docs copy, stop and confirm the generation path first. Do not edit `docs/examples/**` as a shortcut.

### CI-sensitive docs and tooling validation

When a task touches `.github/workflows/**`, `package.json`, `package-lock.json`, `scripts/**` that launch docs or browser tooling, Mermaid rendering, Puppeteer/Chromium, or any dependency change that can affect those paths:

- do not treat local Windows success as sufficient evidence for GitHub-hosted Linux runners,
- run `npm ci` after manifest or lockfile edits and report pass/fail before claiming the workflow is fixed,
- run `npm run docs` when Mermaid, Puppeteer, docs generation, or related launch scripts are affected,
- when browser-based docs tooling runs in Linux CI, explicitly account for Chromium sandbox restrictions and prefer durable script-level launch configuration over workflow-only ad hoc flags.

### Folder README reconnaissance (read this before deep code search)

Because JSDoc is auto-compiled into each folder's `README.md`, those README files are the fastest condensed overview of a module's purpose, exported surface, neighboring files, and intended usage.

Before exploring or editing a folder in `src/`, `examples/`, `benchmarks/`, or `testing/`, agents should:

1. Read the nearest folder `README.md` first.
   - Example: before changing `src/architecture/network/genetic/*`, read `src/architecture/network/genetic/README.md`.
2. Read the nearest useful parent README when the task spans multiple sibling areas.
   - Example: pair `src/architecture/network/genetic/README.md` with `src/architecture/network/README.md`.
3. Only then read individual source files.

Use that README-first pass to identify responsibility, likely orchestration files, documented invariants, neighboring modules, and whether the touched source should also receive JSDoc improvement.

If the folder README appears stale, incomplete, or in tension with the code, treat that as a signal to improve the underlying JSDoc in the touched source files when it is safe to do so. If the goal is to make that README materially more educational rather than merely less stale, invoke `educational-docs`.

## Cross-cutting policies

### Plan-aware execution

This repository has an active `plans/` directory with roadmap and design intent.

When a task touches architecture, roadmap items, major refactors, export formats, or new subsystems, invoke `plan-alignment` instead of re-stating the plan-selection workflow in ad hoc instructions.

Keep this file as the invocation layer. The detailed plan-selection sequence, terminology preservation rules, bounded reading workflow, and mismatch handling belong in that skill.

For substantial work, agent prompts and final summaries should briefly note which README and which plan document informed the change. After substantial edits, keep summaries short and high level by default, and only expand into detailed walkthroughs when the user asks.

### Demo-first library gap policy (critical)

Examples and demos in `examples/` are not places to normalize library ergonomics gaps. They are probes that should reveal where the public API, defaults, or runtime contracts fall short of world-class expectations.

When demo work exposes a mismatch between obvious user intent and the library behavior:

- treat the demo as evidence of a library DX gap first,
- prefer fixing the library, public API, defaults, or shared runtime semantics,
- use demo-local compensation only when the issue is genuinely demo-specific or a library fix would be unsafe for the current task,
- if a temporary demo-local workaround is unavoidable, call it out explicitly as technical debt and note the preferred library-level fix.

Critical expectation for feed-forward examples:

- if a user selects a feed-forward builder or feed-forward mutation policy, agents should assume the expected DX is that feed-forward intent flows through to the runtime without extra demo-specific flags unless the codebase explicitly documents a different contract,
- when that expectation is not met, agents should frame the issue as a library-level design gap and update plans accordingly.

### Low context window mitigation

When you need to make a change that requires more context than you have available:

- Update the relevant source plan document with a `NEXT:` item describing the change and the reason for it, so that future work in that area has more context.
- Provide a handoff prompt in a text-copy box with the relevant context and a clear question about how to proceed, so that a companion agent can pick it up and investigate.
