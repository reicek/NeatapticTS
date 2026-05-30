# Folder Quality Gate and Racing Curriculum Hotfix

**Status:** [DONE]

## Scope

Closed the standalone lane that fixed the racing curriculum browser crash caused by typed-array rejection in `network.activate(...)` and shipped the repo-wide `quality:folder` static gate for fast folder-scoped post-edit checks.

Out of scope remained browser-host DOM layout coverage for the racing curriculum demo and broad cleanup of unrelated pre-existing folder-quality smells.

## Final state

- [DONE] `Network.activate(...)`, `Network.noTraceActivate(...)`, the batch-row guard, and the window-row guard now accept matching-width `Float32Array` inputs instead of misreporting `expected N, got N`.
- [DONE] The public activation contract was widened only where approved: `number[] | Float32Array` on the runtime entry points and matching internal type surfaces.
- [DONE] `scripts/folder-quality-metrics.mjs`, `scripts/agent-customization/gates/folder-quality.gate.mjs`, and the `quality:folder` package script shipped with help text, JSON output, blocking-smell exits, and workflow documentation.
- [DONE] The owned validation slice stayed green, `npm run test:silent` stayed green, and the touched `src/` files remained at `100/100/100/100`.
- [DONE] No active step remains for this workstream; future follow-up should reopen from this archived baseline instead of reviving an active tracker in place.

## Audit summary

- `00-helping` repaired the Step 07 validation packet mismatch before closure by restoring the missing `Required validation` prose block, after which the validation MCP returned `validationCommandsMatch: true`.
- The stale local `data/mcp-session-override.json` binding was cleared via `plan-session-redirect.mjs --clear`, leaving fresh MCP sessions on the perpetual `plans/mcp-active-binding.plans.md` fallback instead of the archived tracker path.
- The archive preserves the honest residual caveats from green validation: one pre-existing `lint-error` plus eight pre-existing `missing-test-file` smells under `examples/racing_curriculum/`, six pre-existing `missing-test-file` smells under `src/architecture/network/activate/`, and the PowerShell `npm run quality:folder` forwarding quirk where the direct `node ...` form remains safer.
- Browser-host DOM layout coverage for the racing curriculum demo stayed intentionally out of scope for this hotfix lane.
- The compressed tracker and same-boundary log now live together under `plans/completed/`.

## Reopen conditions

Reopen this archive only if the typed-array activation fix regresses, the folder-quality gate contract or workflow integration needs another bounded pass, or the deferred pre-existing racing curriculum or activate-folder smells are explicitly adopted as owned scope.

Do not reopen this lane for unrelated racing curriculum feature work such as Tier 3 roles.

## Audit log

See `plans/completed/Folder_Quality_Gate_and_Racing_Hotfix.logs.md`.
