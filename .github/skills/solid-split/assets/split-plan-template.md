# <Workstream Name> SOLID Split

## Purpose

Describe the coordination sink being reduced, the stable surface that must keep
working, and the target architectural shape after the split.

## Root

- Split root: `<path>`
- Nearest README reviewed: `<path>`
- Parent README reviewed: `<path or n/a>`
- Relevant plan: `<path or n/a>`

## Durable Rules

- Keep exactly one active step at a time.
- Update this plan immediately after each completed step.
- Preserve stable imports unless a breaking change is explicitly approved.
- Do not hand-edit generated README files; improve source JSDoc and run docs.
- Keep the true public facade orchestration-first.

## Target Shape

- `<facade path>` remains the public facade or compatibility shim.
- `<subfolder path>` owns extracted helper categories.
- Public responsibilities remain visible in the facade.
- Implementation detail moves behind focused `*.services.ts`, `*.utils.ts`,
  `*.types.ts`, `*.errors.ts`, or `*.constants.ts` files.

## Steps

- [] Step 1: Map the current boundary, stable imports, and likely helper seams.
- [] Step 2: Extract the first focused responsibility cluster behind the stable
  facade.
- [] Step 3: Extract the next responsibility cluster and reduce the facade to
  orchestration-only behavior.
- [] Step 4: Improve JSDoc so the generated README reads naturally for the new
  boundary.
- [] Step 5: Validate the touched surface and record the durable boundary note.

## Done Criteria

- The target file is no longer the obvious coordination sink.
- The public facade remains stable and orchestration-first.
- Extracted helper files own the detailed responsibilities.
- Generated README output reflects the new shape after docs regeneration.
- The boundary can be resumed safely in a later session from this plan alone.