---
name: solid-split
description: 'Plan and execute repo-consistent SOLID splits in NeatapticTS across src/ or test/, starting from a user-specified root such as #file:flappy_bird. Use when folderizing a module, thinning a compatibility facade, improving JSDoc so generated README files read naturally, or continuing an incremental split pass with stable imports.'
argument-hint: 'Describe the split root, target module or folder, any relevant plan file, and optional detail about the next boundary to extract.'
user-invocable: true
disable-model-invocation: false
---

# Solid Split Playbook

Use this skill to run the NeatapticTS house style for deliberate, resumable,
documentation-aware SOLID splits in either library code under `src/` or demos
and examples under `test/`.

This skill is designed to complement the existing `solid-split` custom agent.
The skill defines the repository-specific split protocol, documentation bar,
plan expectations, and validation checklist that should remain consistent across
split sessions.

This file is the canonical knowledge surface for solid-split work. The skill
owns the durable repository workflow, discovery order, plan discipline,
documentation expectations, and validation policy. Companion agents should be
thin executors: they should pass the current task details into this skill,
follow it, and avoid duplicating the same repository rules in their own prompt.

## When to Use

- The user wants a SOLID split, folderization pass, or orchestration-first
  cleanup from a specific root such as `#file:flappy_bird`.
- A top-level file is acting as a coordination sink and should become a thin
  compatibility facade or stable public entrypoint.
- A module boundary needs to be mapped before extracting `*.services.ts`,
  `*.utils.ts`, `*.types.ts`, `*.errors.ts`, or `*.constants.ts` helpers.
- Generated folder README output looks stale, thin, or not educational enough
  after a refactor.
- A multi-session split needs a durable plan, strict step sequencing, and a
  high-quality handoff prompt for the next session.

## Invocation Pattern

Typical use:

```text
/solid-split
SOLID split #file:flappy_bird
```

Useful optional detail to include:

- the root from which to start splitting,
- the current file or boundary believed to be overloaded,
- the plan file to follow if one already exists,
- whether this session should map boundaries, create the plan, or complete one
  specific split step.

## Responsibility Split

Use this boundary intentionally:

- The skill owns durable knowledge: repo conventions, README-first discovery,
  plan sequencing, documentation guardrails, validation expectations, and the
  split philosophy.
- The agent owns execution mechanics: reading the current request, deciding the
  single step to run now, applying edits, updating the plan, validating the
  touched surface, and producing the handoff prompt.
- The agent should not restate the full workflow from this skill unless a
  shorter execution summary is needed for the current session.
- If the skill and agent ever disagree, update the agent to follow this skill
  rather than copying the disagreement forward.

## Task Packet For Users And Agents

When invoking this skill directly or through a companion agent, pass a compact
task packet that includes all current-session specifics the skill itself should
not have to infer.

Preferred packet fields:

- split root,
- current target module, file, or folder boundary,
- requested mode: boundary mapping, plan creation, or execution,
- existing plan path if known,
- exact current step if already defined,
- whether stable import paths must remain unchanged,
- expected validations for this step,
- documentation expectations beyond the default JSDoc bar,
- known worktree cautions such as generated README drift or unrelated edits.

Compact example:

```text
Use solid-split for #file:flappy_bird.
Mode: execute one durable step.
Target boundary: browser-entry/playback.
Plan: plans/flappy_browser_entry_split.md.
Current step: Step 4 - extract playback snapshot boundary.
Keep stable imports working.
Validate with: npx tsc --noEmit -p tsconfig.test.json
Worktree caution: generated README files may already be dirty from npm run docs.
```

## Primary Resources

- [Split workflow checklist](./assets/split-workflow-checklist.md)
- [Split plan template](./assets/split-plan-template.md)
- [Documentation improvement checklist](./assets/docs-checklist.md)
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
7. Then inspect the smallest set of source files needed to confirm the actual
  seams.

Generated folder README files are reconnaissance artifacts. Do not hand-edit
them. Use them to infer responsibility boundaries, missing docs, stale public
surface descriptions, likely neighboring consumers, and where source JSDoc must
improve.

For documentation-heavy passes over a root such as `#file:flappy_bird`, the
README inventory is mandatory. Do not start editing source until the full set of
README-owning folders is visible in the todo list.

## Split Philosophy

The project prefers small, durable, orchestration-first refactors over large
one-pass rewrites.

- Keep stable public import paths working unless the user explicitly approves a
  breaking change.
- Reduce the old top-level file to a compatibility facade or public orchestration
  surface when stable imports matter.
- Move one responsibility cluster at a time into focused helpers or subfolders.
- Prefer folder-based boundaries when a file has become a real subsystem.
- Keep top-level flows declarative: collect, transform, fold, return.
- Use descriptive names and ES2023-first style where the touched code benefits.

For demos under `test/examples/`, treat DX gaps as library/runtime evidence
first. Do not normalize demo-specific workarounds if the real issue is a shared
API, default, or runtime contract.

## Required Split Workflow

1. Confirm the split root and whether the user wants planning, execution, or
   continuation of an existing pass.
2. Build a full inventory of every `README.md` under that root.
3. Convert the inventory into a todo list that names each README-owning folder.
4. Read the README files in that inventory before deep code search, starting at
  the root and then proceeding folder by folder.
5. Read `plans/README.md`, then the single most relevant plan file when the work
   is architectural or part of an ongoing roadmap stream.
6. If the split spans an unfamiliar area, use the existing `Boundary Mapper`,
   `Plan Scout`, or `Docs Scout` agents as needed.
7. If no durable plan exists, create one using the bundled template.
8. Keep the README todo explicit and folder-focused, with exactly one active
  README or folder documentation item at a time.
9. Execute only one durable step unless the user explicitly asks for more.
10. Improve JSDoc on touched exported and public surfaces so generated README
   output remains educational, example-driven, and conceptually clear.
11. Update the plan immediately after the step completes.
12. Immediately run `educational-docs` as the next step on the touched
  surface.
  - This is mandatory even when the user invokes `solid-split` directly.
  - Pass the changed boundary, the intended reader, whether the surface is
    generated from source JSDoc, and any relevant doc needs such as tone
    shaping, source mapping, Mermaid, citations, or media constraints.
  - Treat this as a focused follow-up pass on the exact split changes, not as
    permission to start a broad unrelated docs rewrite.
13. Run the minimum validation needed for touched files, documentation output,
  and the step's done criteria.
14. End with a next-session handoff prompt that can continue from the next step
  without depending on prior chat history.

## Companion Agent Contract

If a companion agent uses this skill, it should:

1. Name this skill explicitly as `solid-split`.
2. Pass the current task packet into the skill instead of paraphrasing it away.
3. Reuse this skill's discovery order and guardrails instead of copying them
  into the agent prompt at full length.
4. Keep the agent prompt focused on execution-only concerns: one-step scope,
  output shape, blocker handling, and handoff quality.
5. Update the agent when this skill changes materially so both remain aligned.

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

## Naming and File-Shape Rules

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

## Validation Rules

After a split step, run only the minimum validation needed for the touched
surface and the plan step done criteria.

Typical options:

- file-level diagnostics for touched files,
- focused Jest tests for the changed surface,
- `npx tsc --noEmit -p tsconfig.json` or `tsconfig.test.json` when the step is
  type-heavy,
- `npm run docs` after doc-affecting changes,
- targeted build validation when the split changes bundling or runtime entry
  behavior.

Do not run broad test suites unless they are truly needed for the current step.

## Large-File Split Execution Mode

When the task is a large-file split into submodules, this skill also owns the
step-by-step execution protocol.

Use this stricter mode:

1. Plan first.
   - Propose the file map before moving code.
   - Keep exported APIs in the main file unless the user explicitly approves a
     surface change.
   - Define the intended seams clearly.
2. Create a durable TODO checklist.
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
5. When the user requests user-confirmed stepwise execution, stop after each
   completed durable step and wait for confirmation before continuing.
6. Validate after the planned move set for the current step is complete.

This mode exists so large splits remain resumable, reviewable, and low-risk.

## Guardrails

- Do not hand-edit generated README files.
- Do not revert unrelated user or generated changes.
- Do not complete multiple durable plan steps in one invocation unless the user
  explicitly asks.
- Do not break stable import paths when a facade or compatibility shim is
  appropriate.
- Do not leave a plan stale after reshaping a step.
- Do not skip the `educational-docs` follow-up pass after a completed split
  step unless the user explicitly overrides that policy.
- Do not treat examples as the final place to hide library ergonomics gaps.
- Do not lower the documentation bar during refactors; public-facing docs should
  get better as boundaries improve.

## Expected Final Output

The final response for a split step should include:

- the plan file used or created,
- the completed step label,
- a short summary of the new boundary shape,
- validation results,
- the durable plan update,
- a fenced `text` handoff prompt for the next step.