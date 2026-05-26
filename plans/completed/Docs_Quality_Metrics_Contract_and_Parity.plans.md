# Docs Quality Metrics Contract and Parity

**Status:** [DONE]

## Scope

Closed the permanent docs-quality metrics consistency lane by delivering one canonical versioned contract and shared runner path for CLI plus MCP, with strict compare guards, deterministic run artifacts, CI gate wiring, and migration guidance away from ad hoc metric flows.

Out of scope: broader docs quality remediation, non-docs metric systems, and unrelated runtime or architecture refactors.

## Final state

- [DONE] Phase 1 Step 01 packetized the seven-step execution sequence with explicit validation ordering and closure criteria.
- [DONE] Phase 1 Step 02 froze `CONTRACT_VERSION=1` dimensions (scope tuple, thresholds, source fingerprint, deterministic ordering, metric dimensions, and mismatch reason-code contract).
- [DONE] Phase 1 Step 03 established red tests for runner artifacts, strict compare guards, and MCP or CLI parity expectations.
- [DONE] Phase 1 Step 04 implemented canonical docs-quality modules, package scripts, and MCP routing through the shared runner path.
- [DONE] Phase 1 Step 05 validated comparator behavior, parity expectations, and CI gate wiring (`docs:quality:gate`).
- [DONE] Phase 1 Step 06 published runbook and migration guidance under the semantic-index docs surface.
- [DONE] Phase 1 Step 07 captured closure validation evidence, documented one unrelated focused-Jest failure boundary, and archived this tracker with a matching audit log.

## Audit summary

- Canonical docs-quality runner artifacts are emitted under `artifacts/docs-quality/runs/<run-id>/` with the required files (`summary.json`, `evidence.json`, `manifest.json`).
- Comparator contract behavior is closed and validated: identical baseline/candidate manifests are accepted with zero delta; threshold drift is rejected with `THRESHOLD_MISMATCH`.
- The docs-quality metrics gate reports pass in machine-readable form and remains wired for CI usage.
- Tracker closure validation includes `log-completion-marker` gate pass evidence.
- The required Step 07 focused Jest command (`--testPathPattern=scripts/semantic-index/docs-quality`) currently pulls in unrelated suites and fails on existing repo-wide `ReferenceError: jest is not defined` issues outside this workstream boundary; the docs-quality contract validations above still pass.

## Reopen conditions

Reopen this archive only when changing docs-quality metric contract versioning, artifact schema shape, comparator mismatch semantics, MCP or CLI shared-path behavior, gate contract ownership, or migration guidance for canonical docs-quality commands.

Do not reopen for routine docs debt cleanup or unrelated Jest environment repairs unless those changes directly alter this docs-quality contract lane.

## Audit log

See `plans/completed/Docs_Quality_Metrics_Contract_and_Parity.logs.md`.
