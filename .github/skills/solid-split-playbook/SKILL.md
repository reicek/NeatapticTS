---
name: solid-split-playbook
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

## Primary Resources

- [Split workflow checklist](./assets/split-workflow-checklist.md)
- [Split plan template](./assets/split-plan-template.md)
- [Documentation improvement checklist](./assets/docs-checklist.md)
- Existing split execution agent: `solid-split`

## Repo Discovery Order

For any non-trivial split, follow this order before editing:

1. Read the nearest folder `README.md` for the target root.
2. Read the nearest useful parent `README.md` if the split spans sibling
   surfaces.
3. Read `plans/README.md`.
4. Read only the single most relevant detailed plan, plus at most one adjacent
   plan if the split clearly spans two initiatives.
5. Then inspect the smallest set of source files needed to confirm the actual
   seams.

Generated folder README files are reconnaissance artifacts. Do not hand-edit
them. Use them to infer responsibility boundaries, missing docs, stale public
surface descriptions, likely neighboring consumers, and where source JSDoc must
improve.

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
2. Read the nearest README files before deep code search.
3. Read `plans/README.md`, then the single most relevant plan file when the work
   is architectural or part of an ongoing roadmap stream.
4. If the split spans an unfamiliar area, use the existing `Boundary Mapper`,
   `Plan Scout`, or `Docs Scout` agents as needed.
5. If no durable plan exists, create one using the bundled template.
6. Convert the current step into a todo list with exactly one active
   implementation item.
7. Execute only one durable step unless the user explicitly asks for more.
8. Improve JSDoc on touched exported and public surfaces so generated README
   output remains educational, example-driven, and conceptually clear.
9. Run the minimum validation needed for touched files and the step's done
   criteria.
10. Update the plan immediately after the step completes.
11. End with a next-session handoff prompt that can continue from the next step
   without depending on prior chat history.

## Documentation Standards

This repository is a public-facing educational library. Splits must improve the
generated documentation story, not only the file structure.

When touching exported or public surfaces:

- Add or improve JSDoc so the generated README reads naturally.
- Include concise “what/why” explanations, not only type signatures.
- Add short `@example` blocks when behavior or intended usage is not obvious.
- Mention invariants, defaults, performance costs, or error semantics when they
  materially affect downstream users.
- When the topic is conceptually important, suggest useful background reading in
  plain prose inside the JSDoc description, for example a Wikipedia article on
  a concept such as parallax, graph theory, dynamic programming, or NEAT.
- Keep examples short, dependency-light, and aligned with the actual public API.

Use the generated README as a doc gap detector:

- If the README sounds too terse, improve source JSDoc.
- If neighboring modules are undocumented in the README, consider whether the
  touched source needs clearer exported comments.
- If a new subfolder is introduced, run `npm run docs` after doc-affecting edits
  so the generated README surface stays synchronized.

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

## Guardrails

- Do not hand-edit generated README files.
- Do not revert unrelated user or generated changes.
- Do not complete multiple durable plan steps in one invocation unless the user
  explicitly asks.
- Do not break stable import paths when a facade or compatibility shim is
  appropriate.
- Do not leave a plan stale after reshaping a step.
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