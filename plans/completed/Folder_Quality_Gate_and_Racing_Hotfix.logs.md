# Folder Quality Gate and Racing Curriculum Hotfix Log

**Status:** [DONE]

## Workstream closeout

- [DONE] Closed the library-level fix for typed-array activation acceptance across the owned activate and window guards and the bounded `number[] | Float32Array` public contract update.
- [DONE] Shipped `scripts/folder-quality-metrics.mjs`, `scripts/agent-customization/gates/folder-quality.gate.mjs`, the `quality:folder` alias, and the mandatory workflow documentation updates in `.github/copilot-instructions.md` and `CLAUDE.md`.
- [DONE] Captured the Step 07 tracker repair: `00-helping` added the missing `Required validation` prose block so the validation MCP allowlist matched the active packet before closure.
- [DONE] Archived the closed tracker under `plans/completed/` with no remaining in-scope next step.

## Validation evidence

- [DONE] Direct validation MCP `get_active_validation_allowlist` for the live Step 07 packet -> `validationCommandsMatch: true` after the `00-helping` repair.
- [DONE] `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md` -> PASS (`0 errors, 0 warnings`).
- [DONE] `node scripts/agent-customization/plan-session-redirect.mjs --clear --json` -> PASS (`override_cleared: true`), and the stale local `data/mcp-session-override.json` file was removed.
- [DONE] Workflow MCP `get_active_workflow_snapshot` with `plan_path: "plans/completed/Folder_Quality_Gate_and_Racing_Hotfix.plans.md"` -> PASS with post-closure `scope: "no-active-phase"`, `activePhase: null`, and `activeStep: null`.

## Residual risks

- [DONE] The archive intentionally carries forward the pre-existing `examples/racing_curriculum` lint and `missing-test-file` smells plus the pre-existing `src/architecture/network/activate` `missing-test-file` smells as deferred static debt, not regressions from this lane.
- [DONE] PowerShell npm argument forwarding remains a repo-local usage caveat for `quality:folder`; the direct `node scripts/folder-quality-metrics.mjs --folder=... --json` form is the safer command when shells differ.
