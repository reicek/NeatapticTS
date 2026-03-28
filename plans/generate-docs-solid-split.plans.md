# Generate Docs SOLID Split

**Status:** [DONE]

## Scope

- Split the docs generator from the old flat [scripts/generate-docs.ts](../scripts/generate-docs/generate-docs.ts) boundary into a folder-owned module under `scripts/`.
- Keep the `docs:folders:*` CLI behavior stable by updating repo-local callers to the new built entrypoint instead of leaving a flat compatibility shim.
- Improve source readability with orchestration-first flow, focused helper ownership, and comprehensive JSDoc on the new exported surfaces.

## Current state

- Split root: `scripts/generate-docs/`
- Nearest README reviewed: `n/a` (no `scripts/**/README.md` files currently exist)
- Parent README reviewed: `n/a`
- Relevant plan: [plans/neat-docs.plans.md](../plans/neat-docs.plans.md)
- Adjacent reference checked: [plans/architecture-solid-split.plans.md](../plans/architecture-solid-split.plans.md)
- Current boundary: the root entrypoint now delegates to focused chapter files for targets, symbols, ordering, output, constants, state, and shared contracts.
- Active split frontier: `generate-docs.symbols.ts` and `generate-docs.output.ts` have each grown into their own responsibility clusters and now need direct-path subfolderization.

## Durable rules

- Keep exactly one active workstream section.
- Prefer direct-path migration to `scripts/generate-docs/generate-docs.ts`; do not leave a flat shim unless a real compatibility need appears.
- Do not hand-edit generated README files as part of this split.
- Run docs-tooling validation after the split because this boundary feeds `npm run docs`.

## Target shape

- `scripts/generate-docs/generate-docs.ts` owns CLI orchestration.
- `scripts/generate-docs/generate-docs.types.ts` owns shared docs-generator contracts.
- `scripts/generate-docs/generate-docs.constants.ts` owns stable constants and target definitions.
- `scripts/generate-docs/generate-docs.state.ts` owns shared ts-morph and docs-order cache state.
- `scripts/generate-docs/generate-docs.targets.ts` owns target resolution and source-tree preparation.
- `scripts/generate-docs/generate-docs.symbols.ts` owns symbol collection, JSDoc extraction, and dedupe rules.
- `scripts/generate-docs/generate-docs.order.ts` owns `docs.order.json` loading, validation, caching, and warnings.
- `scripts/generate-docs/generate-docs.output.ts` owns README rendering, folder-index rendering, and file emission.

## Coverage backlog

### [DONE] Generator root split

- Landed the folder-owned module at [scripts/generate-docs/generate-docs.ts](../scripts/generate-docs/generate-docs.ts) with dedicated chapter files for constants, types, state, targets, symbols, ordering, and output.
- Retargeted the repo-local docs scripts to `dist-docs/scripts/generate-docs/generate-docs.js` so the split uses direct-path migration with no flat compatibility shim.
- Updated the existing plan references that pointed at the deleted flat script path.
- Completed the required educational-docs follow-up by adding chapter-level source introductions to the new files so maintainers can read the split as a guided boundary map instead of a raw helper shelf.

### [DONE] Symbols and output chapter split

- Reopen the generator boundary to move `generate-docs.symbols.ts` into `scripts/generate-docs/symbols/` and `generate-docs.output.ts` into `scripts/generate-docs/output/`.
- Keep direct-path migration: update repo-local imports to the new subfolder entrypoints and delete the flat files after validations pass.
- Keep the new chapter roots orchestration-first, with narrower helper files for collection/rendering/signature work in the symbols boundary and README/index ordering work in the output boundary.
- Completed the direct-path move into `scripts/generate-docs/symbols/` and `scripts/generate-docs/output/`, removed the flat chapter files, and retargeted the root generator imports to the new nested entrypoints.
- Fixed the post-split behavioral regressions in the output boundary so `introFile` once again controls directory intro promotion and folder-index nodes keep correct source and link paths.
- Completed the mandatory educational-docs follow-up with contract-level JSDoc covering signature storage, file-summary precedence, intro promotion, and fallback ordering.

## Validation

- Completed in prior step:
  - `npm run docs:build-scripts`
  - `npm run docs`
  - `npm run docs:build-scripts` after the educational-docs follow-up
- Completed for this step:
  - `npm run docs:build-scripts`
  - `npm run docs`
  - `npm run docs:build-scripts` after the post-validation educational-docs follow-up

## Immediate next steps

- No further split work is queued in this boundary right now.
- Future work, if needed, should start from the nested `symbols/` and `output/` folders rather than recreating flat chapter files.

## Handoff query

```text
Continue docs-tooling work from the current repo state only. Do not rely on prior chat history.

Plan: plans/generate-docs-solid-split.plans.md
Current boundary: scripts/generate-docs/
Repo standard: migrate directly to scripts/generate-docs/generate-docs.ts with no compatibility shim by default, and keep the split in small responsibility chapters.
Small-chapter standard: this applies to scripts and tooling boundaries too, not only library code.
Already covered: the old flat generator script was removed, package.json now points at dist-docs/scripts/generate-docs/generate-docs.js, and the current root chapter files own targets, symbols, ordering, output, state, constants, and types.
Next task if this boundary reopens: continue from the nested scripts/generate-docs/symbols/ and scripts/generate-docs/output/ folders, preserving direct-path imports and the current docs-order behavior.
Required validations: npm run docs:build-scripts, npm run docs.
Worktree caution: .github/copilot-instructions.md may already have unrelated local edits.
Continue from the current repo state only. Do not rely on prior chat history.
```
