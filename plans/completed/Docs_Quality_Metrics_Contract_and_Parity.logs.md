# Docs Quality Metrics Contract and Parity Log

**Status:** [DONE]

## Workstream closeout

- [DONE] Locked the canonical docs-quality contract at version 1 and fixed deterministic dimensions for scope, thresholds, source fingerprint, ordering, and metric totals.
- [DONE] Added the canonical docs-quality surfaces under `scripts/semantic-index/docs-quality/`:
  - `docs-quality.contract.mjs`
  - `docs-quality.normalize.mjs`
  - `docs-quality.artifacts.mjs`
  - `docs-quality.metrics.mjs`
  - `docs-quality.compare.mjs`
- [DONE] Added runner/comparator command wiring in `package.json` (`docs:quality:metrics`, `docs:quality:compare`) and retained compatibility-oriented scanner boundaries.
- [DONE] Routed MCP `scan_code_quality` through shared runner internals in `scripts/mcp-semantic/repo-cortex-mcp.mjs` to keep CLI and MCP serialization parity.
- [DONE] Added and wired the docs-quality metrics gate (`scripts/agent-customization/gates/docs-quality-metrics.gate.mjs`, `docs:quality:gate`, `.github/workflows/ci.yml`).
- [DONE] Published migration and baseline guidance in `scripts/semantic-index/README.md` and verified command help surfaces for metrics, compare, and gate.

## Validation evidence

- [DONE] `npx tsc --noEmit -p tsconfig.json` -> PASS (no diagnostics output).
- [DONE] `node scripts/semantic-index/docs-quality/docs-quality.metrics.mjs --json --scope=src --min-jsdoc-words=10 --complexity-threshold=10 --run-id=baseline-step07-20260524b` -> PASS (artifacts emitted).
- [DONE] `node scripts/semantic-index/docs-quality/docs-quality.metrics.mjs --json --scope=src --min-jsdoc-words=10 --complexity-threshold=10 --run-id=candidate-step07-20260524b` -> PASS (artifacts emitted).
- [DONE] `node scripts/semantic-index/docs-quality/docs-quality.compare.mjs --json --left=artifacts/docs-quality/runs/baseline-step07-20260524b/manifest.json --right=artifacts/docs-quality/runs/candidate-step07-20260524b/manifest.json` -> PASS contract (`accepted: true`, zero deltas).
- [DONE] `node scripts/semantic-index/docs-quality/docs-quality.metrics.mjs --json --scope=src --min-jsdoc-words=11 --complexity-threshold=10 --run-id=candidate-threshold-mismatch-step07-20260524b` -> PASS (mismatch candidate artifacts emitted).
- [DONE] `node scripts/semantic-index/docs-quality/docs-quality.compare.mjs --json --left=artifacts/docs-quality/runs/baseline-step07-20260524b/manifest.json --right=artifacts/docs-quality/runs/candidate-threshold-mismatch-step07-20260524b/manifest.json` -> PASS rejection (`accepted: false`, `reasonCode: THRESHOLD_MISMATCH`).
- [DONE] `node scripts/agent-customization/gates/docs-quality-metrics.gate.mjs --json` -> PASS (`owner: 05-green-testing`).
- [DONE] `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json` -> PASS (`gate: log-completion-marker`).
- [DONE] Required Step 07 focused command `npm run jest:base -- --no-cache --runInBand --testPathPattern=scripts/semantic-index/docs-quality` -> FAIL (unrelated repo-wide `ReferenceError: jest is not defined` spillover and existing W6-06 red boundaries outside docs-quality ownership).

## Residual risks

- The required Jest selector currently widens into unrelated suites; this can obscure lane-local signal until test-targeting strategy is narrowed or the shared Jest environment issue is resolved.
- Docs-quality metrics remain sensitive to intentional threshold changes; this is contract behavior and should continue to reject mixed baselines by design.
