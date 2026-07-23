---
name: solid-split
description: 'Use when: planning or executing a repo-consistent SOLID split.'
argument-hint: 'Provide split root, mode (map|plan|execute|close), current boundary, plan path, stable import or compatibility requirements, expected validations, documentation follow-up needs, and any blocker or worktree caution.'
user-invocable: true
disable-model-invocation: false
skills:
  - splitting-monolithic-agent
  - educational-docs
  - tracker-handoff
  - coverage-guard
---

> **Search policy:** Follow the Cortex-First Search Policy from the `research-methodology` skill. Prefer Cortex MCP tools (`search_corpus`, `search_context`, `search_advanced`, `load_chunk`, `traverse_graph`) over native tools (`grep`, `glob`, `view`). Use native tools only as fallback when Cortex is degraded.

# Solid Split Playbook

Use this skill to run the NeatapticTS house style for deliberate, resumable,
documentation-aware SOLID splits across `src/`, `examples/`, `benchmarks/`,
`testing/`, `scripts/`, and repo-tooling surfaces.

This skill complements the `solid-split` custom agent. The skill owns the
durable workflow, discovery order, split-shape rules, automation defaults,
blocker recovery, companion-skill coordination, validation cadence, and
reporting contract that should stay consistent across sessions.

A split step is only done when the boundary is clearer, the tracker state is
accurate, the required documentation follow-up has run or been explicitly
deferred, and the validations match the actual surface that changed.

Tracker shape, status markers, compression, and `Handoff query` structure are
owned by the companion skill `tracker-handoff`.

## Workflow At A Glance

```text
Flowchart summary: Task packet → README inventory; README inventory → Explicit todo list; Explicit todo list → Mode; Mode → Map seams and choose plan (map), Create or refine tracker (plan), Run one durable extraction (execute), Compress and archive tracker pair (close); Map seams and choose plan → tracker-handoff update; Create or refine tracker → tracker-handoff update; Run one durable extraction → educational-docs follow-up; Compress and archive tracker pair → tracker-handoff update; tracker-handoff update; educational-docs follow-up → Targeted validation; Targeted validation → tracker-handoff update.
```

## Core Promise

`solid-split` exists to make large or overloaded boundaries easier to explore,
teach, validate, and continue in a later session.

The default promise is:

- one durable step at a time,
- direct-path migration by default,
- orchestration-first facades only when they are truly useful,
- README-first reconnaissance before deep edits,
- small-chapter folder structures instead of growing monoliths,
- explicit blocker handling that stops in a stable, resumable state.

## When to Use

- The user wants a SOLID split, folderization pass, or orchestration-first
  cleanup from a specific root such as `#file:flappy_bird`.
- A top-level file is acting as a coordination sink and should become a thin
  stable facade or a smaller public entrypoint.
- A module or folder boundary needs to be mapped before extracting
  `*.services.ts`, `*.utils.ts`, `*.types.ts`, `*.errors.ts`, or
  `*.constants.ts` helpers.
- A folder README or generated docs surface is too monolithic to stay readable
  without smaller chapter folders.
- A multi-session split needs a durable plan, explicit todo list, strict step
  sequencing, and a high-quality handoff prompt.
- A split stalled because stable import rules, compatibility expectations,
  documentation follow-up, or targeted validation were not explicit enough.
- Documentation work revealed that the real fix is a boundary split rather than
  more prose on the current shape.

## When NOT to use

Do NOT use for splitting agents - use `splitting-monolithic-agent` instead. Do NOT use for general refactoring - use `implementation-standards` instead.

## Invocation Pattern

Typical direct use:

```text
/solid-split
SOLID split #file:flappy_bird
```

Useful optional detail to include up front:

- the split root,
- the current target module, file, or folder boundary,
- whether this invocation should map, plan, execute, or close,
- the plan file to follow if one already exists,
- whether stable import paths must remain unchanged,
- whether this is approval-based stepwise work or autonomous continuation.

## Task Packet For Users And Agents

Pass a compact packet that contains the current-session specifics the skill
should not have to guess.

| Field                   | Purpose                                                        | Default if omitted                                                    |
| ----------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |
| Split root              | Names the requested root or `#file:` handle                    | Required                                                              |
| Mode                    | `map`, `plan`, `execute`, or `close`                           | `map` when no active plan exists; otherwise `execute`                 |
| Target boundary         | Names the exact file, folder, or seam                          | Infer from the active `[WIP]` step when a plan already exists         |
| Plan path               | Gives the durable tracker to follow                            | Create one when the work is not safely resumable without it           |
| Current step            | Pins the exact active task                                     | Use the active `[WIP]` step if present                                |
| Stable import rule      | Says whether current paths must keep working                   | Preserve stable imports unless an explicit break is approved          |
| Compatibility shim rule | Says whether old flat files may remain                         | No shim by default; require a verified compatibility reason           |
| Validation expectations | Lists exact checks or gates to satisfy                         | Choose the narrowest credible validation lane for the touched surface |
| Documentation follow-up | States how broad the doc pass should be                        | Focused `educational-docs` follow-up on the touched boundary          |
| Blockers or cautions    | Records generated drift, unrelated edits, or unknown consumers | Record them explicitly before moving code                             |

### Ready-To-Paste Packets

Mapping mode:

```text
Use solid-split for #file:methods.
Mode: map.
Target boundary: top-level methods folder.
Need: README inventory, seam map, and durable plan only.
Keep stable imports working.
```

Single-step execution mode:

```text
Use solid-split for #file:methods.
Mode: execute.
Target boundary: activation chapter split.
Plan: plans/methods-solid-split.plans.md.
Current step: Step 3 - extract activation chapter folder.
Compatibility shims: no unless verified external consumers require one.
Validate with: npx tsc --noEmit -p tsconfig.json, npm run docs.
```

Blocked-state recovery mode:

```text
Use solid-split for #file:browser_build.
Mode: execute.
Target boundary: docs asset copy boundary.
Blocker: generated output drift plus unknown external import consumers.
Need: stable-state recovery packet and the next proof required before deleting old paths.
```

## Operating Modes And User Interaction

Use one mode intentionally:

- **Boundary-mapping mode**: inventory README surfaces, map likely seams,
  identify public entrypoints, and recommend the tracker plus validation shape.
  Do not start moving code unless the task is explicitly tiny and low-risk.
- **Plan mode**: create or refine the durable tracker using the bundled split
  plan template plus `tracker-handoff` rules.
- **Single-step execution mode**: execute exactly one durable extraction or one
  durable cleanup step, validate it, update the plan, and stop.
- **User-confirmed stepwise mode**: when the user explicitly wants approval
  between steps, stop after each completed durable step and wait.
- **Autonomous continuation mode**: only continue beyond one durable step when
  the request clearly asks for continued execution. Even then, checkpoint the
  tracker after each durable step rather than batching an untracked rewrite.
- **Close-out mode**: compress the tracker, refresh the matching `.logs.md`
  file, and archive the pair when the workstream reaches terminal `[DONE]`.

If the packet does not say otherwise, default to the smallest mode that keeps
the work safe and resumable.

## Responsibility Split

Use this boundary intentionally:

- The skill owns durable knowledge: repo conventions, README-first discovery,
  small-chapter standards, plan sequencing, blocker recovery, validation
  expectations, and split philosophy.
- The agent owns session-local execution: reading the request, choosing the
  single step to run now, applying edits, updating the plan, validating the
  touched surface, and producing the handoff or close-out output.
- The agent should not restate this entire playbook in every prompt; it should
  pass the current packet into the skill and keep the prompt execution-focused.
- If the skill and agent ever disagree, update the agent to follow this skill
  rather than copying the disagreement forward.

## Primary Resources

- [Split workflow checklist](./assets/split-workflow-checklist.md)
- [Split plan template](./assets/split-plan-template.md)
- [Documentation improvement checklist](./assets/docs-checklist.md)
- Companion skill: `tracker-handoff`
- Companion skill: `educational-docs`
- Existing split execution agent: `solid-split`

## Repo Discovery Order

For any non-trivial split, follow this order before editing:

1. Build a full README inventory for the requested root, including nested
   subfolder `README.md` files that shape the documentation surface.
2. Convert that README inventory into an explicit todo list so the session can
   proceed folder by folder with visible scope.
3. Read the nearest folder `README.md` for the target root.
4. Read the nearest useful parent `README.md` if the split spans sibling
   surfaces.
5. Read `plans/README.md`.
6. Read only the single most relevant detailed plan, plus at most one adjacent
   plan if the split clearly spans two initiatives.
7. Then inspect the smallest set of source files needed to confirm the real
   seams, public entrypoints, and migration risk.

Generated folder `README.md` files are reconnaissance artifacts. Do not
hand-edit them. Use them to infer responsibility boundaries, missing docs,
stale public-surface descriptions, likely neighboring consumers, and where
source JSDoc must improve.

Published `docs/examples/**/index.html` pages are generated copies. If an
example split changes browser demo structure, edit the source under
`examples/**/index.html` and regenerate docs instead of patching the published
copy.

For documentation-heavy passes over a root such as `#file:flappy_bird`, the
README inventory is mandatory. Do not start editing source until the full set
of README-owning folders is visible in the todo list.

## Automation And Todo Discipline

Strong `solid-split` runs are explicit, not ad hoc.

- Convert the README inventory into an ordered checklist before the first edit.
- Name each checklist item with the README-owning folder or exact seam it
  covers.
- Keep exactly one active item at a time.
- Recompute the checklist only when the requested scope materially changes.
- Use the bundled checklist and plan template instead of inventing a new tracker
  shape for each split.
- If a blocker stops the pass, leave the active item in a stable blocked state
  with the exact missing proof or validation, not a vague note such as "needs
  investigation."

This keeps autonomous runs reviewable and makes user-confirmed stepwise work
natural instead of improvisational.

## Companion Skill And Agent Coordination

Route adjacent concerns to the smallest owner instead of stretching
`solid-split` into every neighboring job.

| Need                                                       | Owner                            | What to pass                                                                                                                        |
| ---------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Ambiguous seam, unknown import graph, or unclear consumers | `Boundary Mapper`                | Split root, target boundary, stable import rule, and why the seam is unclear                                                        |
| Roadmap-sensitive boundary or terminology                  | `plan-alignment` or `Plan Scout` | Relevant plan path, terminology that must stay aligned, and any architectural constraint                                            |
| Tracker creation, compression, or closure                  | `tracker-handoff`                | Plan or log path, current status, and whether the workstream stays active or closes                                                 |
| Mandatory documentation follow-up                          | `educational-docs`               | Changed boundary, intended reader, whether docs are generated from source JSDoc, and validation expectations such as `npm run docs` |
| Failing post-split validation                              | `test-fix-workflow`              | Exact failing command, touched files, and a compact blocker summary                                                                 |
| Post-change `src/` coverage gate                           | `coverage-guard`                 | Changed `src/` files, latest green baseline, and whether this is routine post-change checking or regression repair                  |
| Pre-existing coverage debt after the split is green        | `coverage-tranche`               | Exact file and uncovered path, not the whole split packet                                                                           |

If `educational-docs` concludes that the README surface is still too large
after a focused doc pass, bring the work back to `solid-split` with a compact
packet rather than compensating with more prose.

## Split Philosophy

The project prefers small, durable, orchestration-first refactors over large
one-pass rewrites.

When a split step changes behavior or meaningfully risks runtime drift, prefer a
TDD lane for that boundary: add or reshape the narrow red-phase test first,
complete the split until that boundary turns green, then run `coverage-guard`
on every `src/` file touched by the split to enforce 100% coverage in all four
categories before the step is considered fully hardened.

Default principles:

- Default to direct-path migration once a boundary is folderized.
- Do not preserve the old flat path with placeholder, mirror, or compatibility
  re-export files unless the user explicitly asks for them or there is a
  verified external compatibility requirement.
- When a split lands, update repo-local imports and tests to the new folder path
  in the same step whenever it is safe to do so.
- Delete the old flat file after direct imports are migrated and validations
  pass.
- Only keep a thin stable facade when it is the actual public orchestration
  surface or when a verified compatibility need exists.
- Move one responsibility cluster at a time into focused helpers or subfolders.
- Prefer folder-based boundaries when a file has become a real subsystem.
- Prefer small-chapter folder trees that keep generated README files scoped,
  navigable, and educational rather than accumulating large mixed-topic folder
  READMEs.
- Keep top-level flows declarative: collect, transform, fold, return.
- Use descriptive names and ES2023-first style where the touched code benefits.

For demos under `examples/`, treat DX gaps as library or runtime evidence
first. Do not normalize demo-specific workarounds if the real issue is a shared
API, default, or runtime contract.

## Decision Tree

```text
Flowchart summary: "Overloaded public file" → "Stable entrypoint?"; "Stable entrypoint?" → "Keep thin facade, extract helpers" (Yes), "Folderize freely" (No — internal); "Keep thin facade, extract helpers" → "README too large?"; "Folderize freely" → "Single split is enough"; "README too large?" → "Split into chapter folders" (Yes), "Single split is enough" (No); "Single split is enough" → "Proceed with code split"; "Split into chapter folders" → "Real problem is docs?"; "Proceed with code split"; "Real problem is docs?" → "Use educational-docs instead" (Yes), "Proceed with code split" (No); "Use educational-docs instead".
```

## Required Split Workflow

1. Classify the invocation mode from the task packet and apply the default mode
   only when the packet is silent.
2. Build the README inventory for the requested root, including nested
   README-owning folders.
3. Convert that inventory into an explicit todo list or checklist with one
   active item.
4. Read the root README, the nearest useful parent README, `plans/README.md`,
   and only the most relevant detailed plan files.
5. Decide whether `Boundary Mapper`, `Plan Scout`, or `Docs Scout` should run
   before any extraction.
6. If the work is non-trivial and no durable plan exists, create one with the
   bundled split plan template before moving code.
7. Confirm the stable public surface, compatibility requirement, and whether the
   old flat path should disappear after direct-path migration.
8. For steps with behavior change or meaningful runtime risk, add or update the
   smallest boundary-local red test first.
9. Execute only one durable extraction or one durable plan-authoring step unless
   the user explicitly asks for continued multi-step execution.
10. Keep the root entrypoint orchestration-first and move one responsibility
    cluster at a time into focused helpers or chapter folders.
11. Update repo-local imports and tests to point at the new folder path directly
    unless a verified compatibility need requires a facade or shim.
12. Delete obsolete flat, mirror, or placeholder files once direct imports are
    in place and validations pass.
13. Improve JSDoc on touched exported and public surfaces so the generated
    README explains the new boundary naturally.
14. Immediately run `educational-docs` as the focused follow-up unless the user
    explicitly defers that policy.
15. Run the validation matrix for the touched surface. After any `src/` change,
    run `coverage-guard` on every touched file.
16. If validation fails, repair it before proceeding or record a blocker only
    after the boundary is back in a stable, resumable state.
17. Update the plan immediately, compress any completed phase, and refresh the
    `Handoff query` when the workstream remains active.
18. If the workstream is fully complete, use `tracker-handoff` to close and
    archive the tracker pair.

### Mandatory `educational-docs` Follow-Up

The documentation follow-up is part of the split done-state, not an optional
polish pass.

- This requirement still applies when the user invokes `solid-split` directly.
- Pass only the changed boundary and the exact doc needs discovered during the
  split.
- Treat the follow-up as a narrow pass on the split result, not a license to
  start a repo-wide docs rewrite.
- Do not report the split step as complete until that follow-up has run or the
  user has explicitly deferred it.

## Small-Chapter Standard

For all SOLID split work in this repository, the project standard is the
small-chapter split pattern.

- Treat dotted names and clear responsibility seams as real folder boundaries,
  not just filename decoration.
- Keep each folder README focused on one concept cluster whenever practical.
- Prefer a small root README that explains the boundary and points to chapter
  folders instead of repeating the full implementation narrative in one place.
- If a boundary still produces an oversized generated README after the first
  split, continue splitting into narrower chapter folders instead of accepting a
  large mixed-topic folder.
- Favor chapter names that explain responsibility, not implementation trivia.
- A good split outcome is one where a new reader can discover the boundary by
  drilling down through a few concise README pages rather than one monolithic
  README.
- Apply this standard everywhere the skill is used: `src/`, `examples/`,
  `benchmarks/`, `testing/`, `scripts/`, browser/demo surfaces, tooling
  folders, and any other project area that has become too broad for one file or
  one folder README.
- Do not treat examples, scripts, or support tooling as exceptions. If they are
  large enough to need a SOLID split, they should follow the same direct-path,
  no-shim, small-chapter standard.

Recommended shape for an overloaded boundary:

- `boundary/boundary.ts` for orchestration and public exports.
- `boundary/shared/` for shared constants, vocabulary, and context types.
- Additional chapter folders only when they represent a real seam.

This standard overrides older habits of leaving broad root READMEs or keeping
flat compatibility wrappers after folderization.

## Extending The Pattern Beyond Library Code

Use the same split discipline across repo surfaces, but choose validations and
follow-ups that match the surface.

- `src/`: follow the standard module architecture, run targeted tests,
  `npx tsc --noEmit -p tsconfig.json`,
  `npm run quality:folder -- --folder=<touched_folder>`, and
  `coverage-guard` after the split step.
- `testing/`: split fixtures, helpers, and owner-local utilities by subject
  boundary; keep single-expect tests and run focused Jest plus
  `tsconfig.test.json` validation as needed.
- `examples/`: treat demo friction as evidence of a library DX gap first; edit
  source example files instead of generated `docs/examples/**` copies.
- `benchmarks/`: separate scenario config, measurement, telemetry, and reporting
  layers so each remains inspectable and repeatable.
- `scripts/` and `.github/`: keep noninteractive CLI contracts, JSON output,
  validators, and generated artifact ownership explicit after the split.

## Active-Plan Handoff Standard

Every completed split step that leaves the workstream active must end with a
handoff prompt that is ready to use in a fresh chat with this skill attached.

Use `tracker-handoff` as the canonical policy for how that prompt is stored in
the plan file and how active versus completed tracker sections are marked.

When the split workstream is fully complete, the terminal closure rule takes
precedence instead: compress the `.plans.md` file, add or update the matching
`.logs.md` file, move both files into `plans/completed/`, and omit the handoff
prompt unless the user explicitly wants a reopen prompt.

The handoff prompt must:

- name the split root,
- name the current boundary just completed,
- state the next logical boundary or the exact blocker to resolve,
- mention the relevant plan file when one exists,
- state that the repo standard is direct-path migration with no compatibility
  shims by default,
- state that the repo standard is the small-chapter README pattern,
- summarize the validations already completed for the finished step,
- state the next proof or command needed when the pass stopped on a blocker,
- avoid specific calendar dates in headings or status labels,
- instruct the next session to continue without relying on prior chat history.

Preferred ending sentence:

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.
```

## Documentation Delegation

Use `educational-docs` as the documentation policy for split work.

`solid-split` should not redefine the educational writing bar. Instead:

- keep split sessions focused on boundary mapping, extraction order, facades,
  and validation,
- improve JSDoc enough to keep the touched boundary coherent,
- invoke `educational-docs` after every completed split step as the mandatory
  follow-up pass on the touched boundary,
- let that follow-up decide whether the change only needs source-JSDoc
  tightening or whether generated README output also needs a stronger tone
  lift, Mermaid, citations, or Wikimedia-safe visuals,
- run `npm run docs` after doc-affecting edits when the split changes the
  generated surface.

This keeps the split skill orchestration-focused and lets the companion skill
own tone, source mapping, citation handling, and media compliance.

Conversely, when `educational-docs` determines that a README surface is too
large, too monolithic, or structurally confused for a healthy docs-only pass,
that skill should hand the work back to `solid-split` with a compact boundary
packet instead of compensating with more prose.

## Naming And File-Shape Rules

Use the repo's standard architecture pattern where it fits the boundary:

- `module/module.ts`
- `module/module.utils.ts`
- `module/module.types.ts`
- `module/module.errors.ts`
- `module/module.services.ts`
- `module/module.constants.ts`

For nested boundaries:

- `module/sub-module/module.sub-module.ts`
- `module/sub-module/module.sub-module.utils.ts`
- `module/sub-module/module.sub-module.types.ts`
- `module/sub-module/module.sub-module.errors.ts`
- `module/sub-module/module.sub-module.services.ts`
- `module/sub-module/module.sub-module.constants.ts`

Interpretation rules:

- `*.ts`: public surface or orchestration-first entrypoint.
- `*.utils.ts`: pure or narrow helper logic.
- `*.services.ts`: stateful, coordinating, or side-effecting helpers.
- `*.types.ts`: typed contracts and context/result objects.
- `*.errors.ts`: local error classes and error helpers.
- `*.constants.ts`: named constants and local lookup tables.

Only create new folders when the seam is real enough to deserve its own
chapter, tests, and documentation story.

## Validation Cadence And Matrix

After a split step, run the minimum validation that proves the touched boundary
is healthy. Make that choice explicit in the plan or summary.

| Surface                                      | Minimum validation                                                                                            | Escalate when needed                                                                                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/`                                       | Focused tests, `npx tsc --noEmit -p tsconfig.json`, and `npm run quality:folder -- --folder=<touched_folder>` | `coverage-guard` for every touched file, plus `npm run test:silent` only when the coverage gate or change scope explicitly requires repo-wide confirmation |
| `testing/`                                   | Focused Jest plus `npx tsc --noEmit -p tsconfig.test.json` when types or helpers moved                        | Also run source-side typecheck if the split changed shared helpers imported from `src/`                                                                    |
| `examples/` or `benchmarks/`                 | Targeted smoke/build for the changed entrypoint plus `npm run quality:folder -- --folder=<touched_folder>`    | `npm run docs` when generated README surfaces or published example output changed                                                                          |
| `scripts/` or `.github/`                     | Exact validator or gate for the changed automation boundary                                                   | Rerun routing/frontmatter/docs gates when customization metadata or generation inputs moved                                                                |
| Docs-affecting split                         | `npm run docs`                                                                                                | Re-read the generated output and confirm the source files, not generated artifacts, remain the authoring surface                                           |
| Package, workflow, or runtime-tooling change | `npm ci`                                                                                                      | Follow with the build, docs, or gate command affected by the manifest or tooling shift                                                                     |

Preferred cadence:

- During implementation: use cheap diagnostics and narrow tests first.
- After the extraction turns green: run the targeted validator or gate for the
  touched surface.
- For close-out: only run broader suites when the user asks for them or when
  the step semantics require them.
- Do not treat a focused run as a substitute for a required gate such as
  `coverage-guard`, `npm run docs`, or a customization freshness gate.

## Blocker Recovery And Failure Handling

Treat blockers as part of the workflow, not as excuses to leave the boundary
half-moved.

1. **Unclear seam or import graph**  
   Stop extraction, use `Boundary Mapper` or a focused consumer read, and
   document what proof is missing before moving more code.
2. **Unknown compatibility requirement**  
   Keep the smallest stable facade, record the blocker, and prove consumers
   before deleting old paths.
3. **Generated README or published example drift**  
   Record the drift as a worktree caution, edit the source surface instead of
   the generated output, and rerun `npm run docs` if the step actually changes
   that generated surface.
4. **Green validation failure**  
   Fix it immediately or route to `test-fix-workflow`. After repairing a `src/`
   change, rerun `coverage-guard`.
5. **Plan drift or stale handoff**  
   Update the tracker with `tracker-handoff` before doing more extraction work.
6. **Repeated gate or workflow failures**  
   Record the failure with
   `node scripts/agent-customization/gates/record-gate-exception.mjs`. Continue
   retrying until resolved or a true technical limit is reached. No
   concessions, no threshold. Only stop when a documented theoretical or
   practical absolute blocks further progress.
7. **Session must stop early**  
   Leave the boundary compiling or otherwise stable, mark the blocker
   explicitly in the plan, and provide a fenced `text` handoff prompt that says
   what to read first, what failed, and what exact proof is needed next.

A blocker summary should always name the missing proof, failing command, or
compatibility uncertainty directly. Do not hide it behind vague language such
as `needs more work`.

## Large-File Split Execution Mode

When the task is a large-file split into submodules, this skill also owns the
stricter step-by-step execution protocol.

Use this mode when the boundary is large enough that a normal single-step pass
would still be too risky.

1. Plan first.
   - Propose the file map before moving code.
   - Keep exported APIs in the main file unless the user explicitly approves a
     surface change.
   - Define the intended seams clearly.
2. Create a durable checklist.
   - Use ordered steps with one active step at a time.
3. Execute in strict passes.
   - Create target files first when needed.
   - Move one responsibility category at a time.
   - Delete moved source from the old file immediately after each category is
     transferred.
   - Do not batch multiple helper categories in one pass unless the user
     explicitly overrides the safer sequence.
4. Keep the main boundary orchestration-first.
   - The main file should remain the readable public flow, not a pile of thin
     wrappers.
5. Respect the interaction mode.
   - In user-confirmed stepwise mode, stop after each completed durable step
     and wait.
   - In autonomous continuation mode, checkpoint the tracker after each durable
     step before moving on.
6. Validate after the planned move set for the current step is complete.

This mode exists so large splits remain resumable, reviewable, and low-risk.

## Companion Agent Contract

If a companion agent uses this skill, it should:

1. Name this skill explicitly as `solid-split`.
2. Pass the current task packet into the skill instead of paraphrasing it away.
3. Reuse this skill's discovery order, defaults, and guardrails instead of
   copying them into the agent prompt at full length.
4. Keep the agent prompt focused on execution-only concerns: chosen mode,
   one-step scope, output shape, blocker handling, and handoff quality.
5. Hand tracker-only work back to `tracker-handoff`, docs-only work back to
   `educational-docs`, and validation repair to the appropriate testing
   workflow instead of broadening the prompt.
6. Update the agent when this skill changes materially so both remain aligned.

## Before / After Examples

**Before:**

```ts
// src/neat/mutation.ts — 500-line monolith
export function mutate(genome: Genome): Genome {
  // ... 200 lines of ADD mutation logic ...
  // ... 150 lines of SUB mutation logic ...
  // ... 150 lines of SWAP mutation logic ...
}
```

**After:**

```ts
// src/neat/mutation/neat.mutation.ts — orchestration
export function mutate(genome: Genome): Genome {
  const picked = pickMutationType();
  return applyMutation(genome, picked);
}

// src/neat/mutation/neat.mutation.utils.ts — focused helpers
function applyMutation(genome: Genome, type: MutationType): Genome {
  /* ... */
}
function pickMutationType(): MutationType {
  /* ... */
}
```

## Guardrails

- Do not hand-edit generated README files.
- Do not hand-edit published `docs/examples/**` copies when the source lives
  under `examples/**`.
- Do not revert unrelated user or generated changes.
- Do not complete multiple durable plan steps in one invocation unless the user
  explicitly asks.
- Do not keep compatibility shims or mirror re-export files by habit; require a
  specific compatibility reason before leaving them behind.
- Do not break stable import paths when a verified public compatibility
  boundary is required.
- Do not leave a plan stale after reshaping a step.
- Do not skip the `educational-docs` follow-up pass after a completed split
  step unless the user explicitly overrides that policy.
- Do not describe a split as fully complete when the documentation follow-up or
  required validation is still pending.
- Do not bury blocker state; name the missing proof, failing command, or
  compatibility uncertainty directly.
- Do not use `solid-split` when `educational-docs` or `tracker-handoff` is the
  real smaller tool for the current request.

## Expected Final Output

A strong split report should state:

- the mode and split root,
- the plan file used or created,
- the completed step label or blocker label,
- the resulting boundary shape,
- the import-path decision: direct-path migration, thin stable facade, or
  verified compatibility shim,
- whether the required `educational-docs` follow-up was completed or explicitly
  deferred,
- validation evidence,
- any blocker or remaining risk plus the exact next proof needed,
- either a fenced `text` handoff prompt for the next step, or the closed
  tracker plus matching `.logs.md` paths when the workstream was terminally
  closed.

Recommended active-workstream shape:

```text
Mode: execute
Plan: <path>
Step: <completed step>
Boundary result: <what moved where>
Import impact: <direct-path migration | thin facade retained | compatibility shim retained with reason>
Docs follow-up: <completed | deferred by user>
Validation: <commands and pass/fail>
Next: <next boundary or blocker proof>
```

If the pass ended blocked, replace the normal `Next` line with a fenced `text`
recovery prompt that starts with:

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and any declared pre_execute_hook/get_slice_context.
```

If the workstream closed, report the archived `.plans.md` and `.logs.md` paths
instead of a next-session prompt.
