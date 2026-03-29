# <Workstream Name> SOLID Split

**Status:** [WIP]

## Scope

Describe the coordination sink being reduced, the stable surface that must keep
working, and the target architectural shape after the split.

## Root

- Split root: `<path>`
- Nearest README reviewed: `<path>`
- Parent README reviewed: `<path or n/a>`
- Relevant plan: `<path or n/a>`

## Durable Rules

- Keep exactly one active `[WIP]` step at a time.
- Update this plan immediately after each completed step.
- Preserve stable imports unless a breaking change is explicitly approved.
- Do not hand-edit generated README files; improve source JSDoc and run docs.
- Keep the true public facade orchestration-first.
- When this workstream reaches terminal `[DONE]`, compress this file into a
  short closed tracker and add or update the matching `.logs.md` file.

## Target Shape

- `<facade path>` remains the public facade or compatibility shim.
- `<subfolder path>` owns extracted helper categories.
- Public responsibilities remain visible in the facade.
- Implementation detail moves behind focused `*.services.ts`, `*.utils.ts`,
  `*.types.ts`, `*.errors.ts`, or `*.constants.ts` files.

## Current state

- Active boundary: `<path or seam>`
- Current pressure: `<why this boundary still needs work>`
- Worktree cautions: `<generated docs drift, unrelated edits, or n/a>`

## Coverage backlog

- [WIP] Step 1: Map the current boundary, stable imports, and likely helper
  seams.
- [PLANNED] Step 2: Extract the first focused responsibility cluster behind the
  stable facade.
- [PLANNED] Step 3: Extract the next responsibility cluster and reduce the
  facade to orchestration-only behavior.
- [PLANNED] Step 4: Improve JSDoc so the generated README reads naturally for
  the new boundary.
- [PLANNED] Step 5: Validate the touched surface and record the durable
  boundary note.

## Immediate next steps

- Execute only the single active `[WIP]` step.
- Refresh `## Handoff query` whenever the workstream remains active.
- If the whole workstream closes, replace active-session scaffolding with a
  short closed tracker and add or update the matching `.logs.md` file.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
<Describe the next narrow split step, the files to read first, and the required validations.>
```

## Done Criteria

- The target file is no longer the obvious coordination sink.
- The public facade remains stable and orchestration-first.
- Extracted helper files own the detailed responsibilities.
- Generated README output reflects the new shape after docs regeneration.
- If more work remains, the boundary can be resumed safely in a later session
  from this plan alone.
- If no work remains, this plan is compressed into a short closed tracker and a
  matching `.logs.md` file records the durable audit history.
