# Tests Fix Plan

## Context
TypeScript compilation failed during `npm test` after telemetry type tightening.

## Priority Order
1. **HIGH** Fix telemetry event type mismatches (missing `gen`, stale `ObjEvent` references).
2. **MEDIUM** Validate no lingering `ObjEvent` usages in telemetry code paths.
3. **HIGH** Initialize speciation and objective state maps/arrays to avoid runtime crashes.
4. **MEDIUM** Update strict lineage telemetry fixtures in tests.
5. **MEDIUM** Make mutation weight-change test robust to different connections.
6. **LOW** Re-run TypeScript build once fixes applied.

## Checklist
- [x] Add `gen` to telemetry objective events when recorded in `neat.telemetry.ts`.
- [x] Remove `ObjEvent` type references in telemetry event mapping.
- [x] Initialize `_nodeSplitInnovations` and related maps on `Neat` instances.
- [x] Initialize `_species`, `_nextSpeciesId`, species maps, and objective tracking maps/arrays.
- [x] Update lineage telemetry test fixtures to include required fields.
- [x] Adjust mutation test to assert any weight change.
- [x] Run `npx tsc --noEmit -p tsconfig.json` after code changes.
