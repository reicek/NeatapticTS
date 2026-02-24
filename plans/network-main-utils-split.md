# Network Main File Utility Split Plan

## Goal
Refactor `src/architecture/network.ts` so the main file remains orchestration-focused while preserving behavior and public API shape.

## Ordered TODO

1. ✅ Define migration targets and destination utility files.
2. ✅ Move forward-pass core logic from `Network.activate()` and `_gaussianRand` into `activate/` helpers.
3. ✅ Move backward-pass helpers from `propagate()` and `clear()` into `training/` helpers.
4. ✅ Move evaluation logic from `test()` into `stats/` helper.
5. ✅ Move static graph-construction helpers (`createMLP`, `rebuildConnections`) into `topology/` helper.
6. ✅ Update `network.utils.ts` exports and `network.ts` imports.
7. ⬜ Run `npx tsc --noEmit -p tsconfig.json`.
8. ⬜ Run `npm test`.
9. ⬜ Final risk and follow-up review.

## Destination Files

- `src/architecture/network/activate/network.activate.core.utils.ts`
- `src/architecture/network/training/network.training.backprop.utils.ts`
- `src/architecture/network/stats/network.stats.test.utils.ts`
- `src/architecture/network/topology/network.topology.factory.utils.ts`

## Notes

- Keep signatures and behavior unchanged.
- Move one group at a time and remove moved source from `network.ts` immediately.
- Avoid circular imports by using `this`-bound static helper patterns where needed.
