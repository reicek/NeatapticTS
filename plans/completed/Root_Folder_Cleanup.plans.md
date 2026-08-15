**Status:** [DONE]
**Plan ID:** ROOT_FOLDER_CLEANUP
**Created:** 2026-08-12
**Source of truth:** `plans/Root_Folder_Cleanup.plans.md`
**Research artifact:** `plans/Root_Folder_Cleanup.research.md`

## Scope

Audit and clean the repository root of temporal, debug, and snapshot files; relocate root-level scripts and documentation to their canonical subdirectories; verify no build, lint, or documentation references are broken.

**Non-goals:** Changing source code behavior, renaming `src/`/`examples/`/`benchmarks/` directories, editing `package.json` scripts, or removing tracked build outputs (`dist/`, `dist-docs/`, `coverage/`).

## Final state

Repository root contains only expected top-level files and directories. All `delete`-marked items from the research artifact were removed; surviving root documentation and harnesses were relocated to `docs/`, `testing/`, and `scripts/agent-customization/mcp/facade-contract/`. Dangling references were fixed in source, docs, skills, tests, and the post-write-reindex hook log path. `npm run build`, `npm run lint`, and targeted Jest tests pass. Coverage on changed source files is 100%; the final `slice-advancement` gate reported 7/7 sub-gates passing (severity FULL).

## Implementation phases

### Phase 1 — Root folder cleanup [DONE]

- Step 01 — Planning and step packet authoring [DONE]
- Step 02 — Audit temporal/unused files and classify relocation targets [DONE] (research artifact: `plans/Root_Folder_Cleanup.research.md`)
- Step 03 — Delete tmp/debug/snapshot files [DONE]
- Step 04 — Relocate surviving root scripts and docs to proper dirs [DONE]
- Step 05 — Verify cleanup and update indexes [DONE]
- Step 06 — Documentation and tracker closure [DONE]

Detailed step/slice transcripts and validation evidence are archived in `plans/Root_Folder_Cleanup.logs.md`.

## Validation gates

- `log-completion-marker` gate: PASS
- `stale-wip-plans` gate: PASS
- `validate-plan-sync` (README/Roadmap consistency): PASS

## Reopen conditions

Reopen if new temporal/debug files accumulate at root or if relocated paths break downstream consumers.

## Audit log

- 2026-08-12 — Plan created and registered in `plans/README.md` and `plans/Roadmap.md`.
- 2026-08-12 — Research artifact produced by Step 02.
- 2026-08-13 — Steps 03–05 completed; residual root items removed and references fixed.
- 2026-08-12 — Step 06 closed; tracker compressed to `plans/Root_Folder_Cleanup.logs.md`.
