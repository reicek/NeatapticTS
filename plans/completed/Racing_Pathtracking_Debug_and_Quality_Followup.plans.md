# Racing Path-Tracking Debug and Quality Followup

**Status:** [DONE]

## Scope

Closed the standalone follow-up lane that joined the racing curriculum Catmull-Rom path-tracking
repair with the deferred folder-quality cleanup for `examples/racing_curriculum` and
`src/architecture/network/activate`.

Out of scope remained Racing Curriculum Phase 3 and unrelated NGE feature work.

## Final state

- [DONE] Shared spline geometry now aligns the renderer, observation assembler, and scripted
  controller inside `examples/racing_curriculum/`, removing the raw-chord vs Catmull-Rom mismatch
  that caused phantom straight-edge steering.
- [DONE] The visual fix is user-confirmed: the car now follows the rounded circuit correctly and
  the lane returned to standard workflow after the user confirmed the repaired behavior.
- [DONE] The deferred folder-quality cleanup completed across `examples/racing_curriculum` and
  `src/architecture/network/activate`, reducing the inherited smells to the single approved
  residual debt only.
- [DONE] `examples/racing_curriculum/browser-entry/browser-entry.ts` still lacks a sibling
  `browser-entry.test.ts`; this remains explicit accepted static debt rather than a blocker because
  the browser bootstrap boundary owns side effects (`requestAnimationFrame`, host wiring, global
  listeners, live teardown) and a narrow Node-environment sibling test would be mostly scaffolding
  instead of a principled behavior check.
- [DONE] The accepted validation record for the workstream stayed green before closure: repo-wide
  `npm run test:silent` passed at `447` suites / `5071` tests, TypeScript stayed clean, the
  `src/architecture/network/activate` folder-quality gate passed, and the
  `examples/racing_curriculum` folder-quality result reduced to the single accepted
  `browser-entry.ts` smell.
- [DONE] No active step remains for this workstream; future follow-up should reopen from this
  archived baseline instead of reviving an active tracker in place.

## Audit summary

- Step 02 confirmed the root cause: the controller observed raw polygon chord geometry while the
  renderer drew a Catmull-Rom spline ribbon.
- Step 04 implemented the shared `splineSamples` fix entirely inside `examples/racing_curriculum`,
  and the repaired rounded-lane behavior was later confirmed by the user during manual validation.
- Step 06 closed the inherited folder-quality smells in both folders, leaving only the explicitly
  accepted `browser-entry.ts` missing-test-file debt.
- Step 07 required no new `00-helping` escalation: the workflow MCP packet was clean, the Step 07
  plan-sync validation passed before archival, the closed tracker and same-boundary log were moved
  under `plans/completed/`, and the live session override was cleared back to the perpetual
  `plans/mcp-active-binding.plans.md` fallback.

## Reopen conditions

Reopen this archive only if the racing demo regresses back to chord-following or lane-edge drift,
the accepted `browser-entry.ts` debt is explicitly adopted as owned scope, or the folder-quality
rationale for that browser bootstrap boundary changes.

Do not reopen this lane for Racing Curriculum Phase 3 work or unrelated NGE benchmark features.

## Audit log

See `plans/completed/Racing_Pathtracking_Debug_and_Quality_Followup.logs.md`.
