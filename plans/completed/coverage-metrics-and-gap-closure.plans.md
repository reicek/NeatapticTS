# Coverage Metrics + Gap Closure

**Status:** [DONE]

## Scope

- Extend `npm run docs:quality:metrics` so it reports additive coverage summary data from existing coverage artifacts only.
- Close the remaining below-100 coverage gaps in `src/` and refresh the authoritative repo-wide coverage artifact.
- Keep the docs-quality metrics contract additive while avoiding unrelated source, test, or docs-lane reopen work.

## Final state

- [DONE] `npm run docs:quality:metrics` now emits additive `summary.coverage` data, including deterministic `filesBelow100Detail` rows whenever any file is below `100%`.
- [DONE] The last training-finalize coverage miss was closed, and the owner-local tests were aligned to the final grad-norm passthrough contract.
- [DONE] The authoritative repo-wide rerun is green at `427` suites / `4654` tests, and the refreshed coverage table reports `All files` and `src` at `100/100/100/100`.
- [DONE] `summary.coverage.filesBelow100` is now `0`, so no active coverage-metrics or gap-closure work remains in `plans/` for this lane.

## Audit summary

- Files touched in this lane: `scripts/semantic-index/docs-quality/docs-quality.metrics.mjs`, `scripts/semantic-index/docs-quality/docs-quality.metrics.test.ts`, `src/architecture/network/training/network.training.finalize.utils.ts`, `src/architecture/network/training/network.training.finalize.utils.test.ts`, `src/architecture/network/training/network.training.basic.test.ts`, `src/architecture/onnx.test.ts`, `src/neat/telemetry/metrics/telemetry.metrics.barrel.test.ts`, `src/architecture/network/slab/network.slab.fast-path.helpers.utils.test.ts`, `src/architecture/architect/architect.test.ts`, and `src/architecture/group/group.utils.test.ts`.
- `npm run --silent docs:quality:metrics` now reports `coverage = { available: true, totalFiles: 360, filesBelow100: 0, filesBelow100Detail: [], overallLines: 100, overallBranches: 100, overallFunctions: 100 }`.
- `npm run test:silent` passed with refreshed authoritative coverage at `100/100/100/100` for `All files` and `src`.
- `npm run --silent docs:quality:gate` passed after the final artifact refresh.
- Active-plan sync was validated before archival, and the closed tracker plus same-boundary log now live under `plans/completed/`.

## Reopen conditions

Reopen this archive only if `npm run docs:quality:metrics` stops reporting the coverage detail contract, a future full-suite coverage artifact reports any file below `100%`, or a follow-on docs-quality contract change needs a new parity or coverage-summary pass.

## Audit log

See `plans/completed/coverage-metrics-and-gap-closure.logs.md`.
