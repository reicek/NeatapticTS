# Log: Neatenstein SOLID Split

**Plan:** `neatenstein-solid-split.plans.md`
**Date:** 2026-08-15
**Session ID:** f3803217-b924-4fce-9637-a0d727e718e1
**Status:** [DONE] — all phases complete, archived.

---

## Summary

Reduced every oversized function in the `examples/neatenstein` codebase into a
declarative orchestrator (holds data, passes through steps calling imported
executors, returns). Pure logic moved to sibling `*.utils.ts` executor files
(cognitive complexity ≤ 5). Util files kept well under 800 lines. Public import
paths preserved via re-exports; `__testOnly*` hooks remained exported from main
modules. No compatibility shims — direct-path migration only.

## Phases Completed

| Phase | Scope | Outcome |
|---|---|---|
| Phase 0 | Validate the pattern (low-risk leaf extractions) | enemy-controller leaf functions, sprites.ts pure guards + atlas caches, floor.ts band/shade/projection executors — all [DONE] |
| Phase 1 | Renderer layer (isolated consumers) | sprites.ts projection + column utils finished, floor.ts + drawNeatensteinGrid refactored, bolt-render.ts split — all [DONE] |
| Phase 2 | Host/Game layer | `host/game/tick.ts` split into 9 util files, `host/hud.ts` split into 6 util files + constants — all [DONE] |
| Phase 3 | Scripts layer (index-alignment contract) | `updateControlledEnemy` decomposed (HIGH risk), `enemy-sprite.ts` utils extracted, `generate-enemy-sprites.ts` split, `enemy-navigation.ts` utils extracted — all [DONE] |
| Phase 4 | Worker layer (2392-line monster) | `display.worker.ts` 2392→865 lines, pure zero-state executors extracted, render-paint executors extracted, `DisplayWorkerState` introduced with `onmessage` rewrite (HIGH risk) — all [DONE] |
| Phase 5 | Harness + browser-entry | `enemy-warmstart.ts` split into 3 utils (mlp-math/backprop/curriculum), `browser-entry.ts` split into 3 util files (render-loop, bootstrap, canvas-dimensions) — all [DONE] |

## Final Validation

- `npx tsc --noEmit`: **0 errors**
- Full neatenstein test suite: **all tests pass** (index-renumbering after de-rez compaction, worker determinism contract preserved)
- Folder quality metrics: **PASS**
- JSDoc: 537/537

## Key Decisions & Patterns

- **Sibling util files, no subfolders**: every util file is a sibling of its owning main module.
- **Public import paths preserved**: consumers kept importing `../renderer/sprites`, `./enemy-controller`, etc. — no consumer changed an import.
- **`__testOnly*` hooks stayed exported** from main modules via re-export; test files unchanged.
- **Module-level caches/buffers moved WITH their owning util** — never duplicated.
- **No compatibility shims**: the main module IS the facade; no separate barrel/shim file.
- **Legacy noise removed**: obsolete/dead stubs (e.g. `scripts/voxel-gun.ts`) deleted rather than left.
- **Pragmatic mode**: broad slices, bypass legacy ceremony, single-model mandate (`glm-5.2:cloud`), follow-ups via `write_agent` to the same idle agent.

## Worker Reduction Highlight

`display.worker.ts`: **2392 → 865 lines** (7 util files extracted) — the largest
single-file reduction in the workstream, achieved without breaking the worker
determinism contract or the `DisplayWorkerState` encapsulation.