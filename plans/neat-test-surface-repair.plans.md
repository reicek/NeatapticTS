# NEAT Test Surface Repair

**Status:** [DONE]

## Scope

Repair the NEAT public TypeScript compatibility surface so the current runtime
features and root helper facades compile against the existing test suite.

## Current state

- [DONE] Grouped the failures into public-surface regressions instead of runtime assertion failures.
- [DONE] Confirmed the runtime still reads option fields such as `telemetry.enabled`, `multiObjective.dominanceEpsilon`, `multiObjective.adaptiveEpsilon`, `novelty.descriptor`, `novelty.archiveAddThreshold`, `adaptiveMutation.initialRate`, and `lineagePressure.strength`.
- [DONE] Confirmed legacy root imports `src/neat/neat.diversity` and `src/neat/neat.lineage` no longer exist even though the folderized modules still do.
- [DONE] Restored the public type/facade boundary with compatibility-first edits and validated the NEAT test slice.
- [DONE] Re-ran `npx tsc --noEmit -p tsconfig.test.json` from the current repo state and confirmed the remaining diagnostics are only in `test/examples/asciiMaze/**`.
- [DONE] Re-ran `npm test` from the current repo state and confirmed the full Jest suite passes.

## Coverage backlog

- [DONE] Widened `src/neat/shared/neat.shared.types.ts` so the public `NeatOptions` surface matches the live implementation-supported fields used by tests.
- [DONE] Reintroduced thin root compatibility facades for diversity and lineage.
- [DONE] Cleared the NEAT-specific TypeScript errors in `npx tsc --noEmit -p tsconfig.test.json` output.
- [DONE] Ran `npx jest test/neat --runInBand` successfully.
- [DONE] Noted that the repository-wide test TypeScript build still reports unrelated `test/examples/asciiMaze/**` branch errors outside this repair.
- [DONE] Ran `npm test` successfully across the full repository.

## Immediate next steps

1. No active NEAT-surface work remains in this tracker.
2. If needed, handle the unrelated `asciiMaze` branch diagnostics in a separate repair pass.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.
The NEAT public-surface repair tracked in plans/neat-test-surface-repair.plans.md is complete. Do not make further NEAT-surface changes unless new evidence appears. If follow-up work is needed, start a separate pass for the remaining TypeScript diagnostics under test/examples/asciiMaze/**. Preserve unrelated user changes.
```
