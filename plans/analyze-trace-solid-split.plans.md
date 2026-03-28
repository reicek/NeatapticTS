# Analyze Trace SOLID Split

**Status:** [DONE]

## Scope

- Folderize the trace analyzer into [scripts/analyze-trace](../scripts/analyze-trace) with a stable entrypoint at [scripts/analyze-trace/analyze-trace.ts](../scripts/analyze-trace/analyze-trace.ts).
- Keep `npm run trace:analyze` stable by retargeting it to the folder-owned entrypoint instead of leaving a flat compatibility shim.
- Improve JSDoc, naming clarity, and deterministic report assembly without changing the report's overall scope.

## Current state

- Split root: `scripts/analyze-trace/`
- README inventory for `scripts/`: none under `scripts/**/README.md`
- Relevant plan: `n/a` (tooling-only trace analyzer tidy; no direct roadmap lane)
- Stable entrypoint: `scripts/analyze-trace/analyze-trace.ts`
- Landed folder files:
  - `scripts/analyze-trace/analyze-trace.ts`
  - `scripts/analyze-trace/analyze-trace.constants.ts`
  - `scripts/analyze-trace/analyze-trace.types.ts`
  - `scripts/analyze-trace/analyze-trace.shared.ts`
  - `scripts/analyze-trace/analyze-trace.io.ts`
  - `scripts/analyze-trace/analyze-trace.analysis.ts`
  - `scripts/analyze-trace/analyze-trace.report.ts`
- Closed issues in this step:
  - CLI parsing, trace loading, analysis, shared helpers, and report rendering no longer live in one file,
  - the split now uses a real folder boundary instead of scattering module files at the scripts root,
  - deterministic sort tie-breaks now make equal-duration sections more stable across runs,
  - the stable min/max timestamp scan remains single-pass for large traces.

## Coverage backlog

### [DONE] Folder-owned analyzer split

- Moved the analyzer into [scripts/analyze-trace](../scripts/analyze-trace) and retargeted `npm run trace:analyze` to [scripts/analyze-trace/analyze-trace.ts](../scripts/analyze-trace/analyze-trace.ts).
- Separated constants, contracts, shared utility helpers, CLI/file I/O, analysis passes, and text-report rendering inside the folder boundary.
- Preserved the existing report sections while making the orchestration flow read as resolve -> load -> analyze -> print.

### [DONE] Educational docs follow-up

- Added educational JSDoc to the folder-owned constants, contracts, helpers, and report surface so future trace-tooling changes can start from an intentional boundary map instead of a single oversized file.

## Validation

- File diagnostics: clean for the touched `scripts/analyze-trace/*.ts` files.
- `npx tsc --noEmit -p tsconfig.docs.json`
- `npm run trace:analyze -- test/examples/flappy_bird/Trace-20260309T191949.json --top=5`

## Immediate next steps

- No further split work is queued for this boundary right now.
- Future analyzer changes should continue inside `scripts/analyze-trace/` instead of re-growing a flat scripts-root boundary.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Use solid-split for #file:analyze-trace.
Plan: plans/analyze-trace-solid-split.plans.md.
Current boundary: scripts/analyze-trace/.
Completed context: scripts has no local README surface, scripts/analyze-trace/analyze-trace.ts is the stable CLI entrypoint, and the analyzer now splits responsibilities across constants, types, shared helpers, CLI/file I/O, analysis, and report rendering inside one owned folder.
Repo standard: direct-path migration with no compatibility shims by default, and the small-chapter mindset applies to scripts/tooling boundaries.
Required validations: file diagnostics for touched files, npx tsc --noEmit -p tsconfig.docs.json, then npm run trace:analyze -- test/examples/flappy_bird/Trace-20260309T191949.json --top=5.
Worktree caution: there may already be unrelated generated README drift elsewhere in the repo.
```
