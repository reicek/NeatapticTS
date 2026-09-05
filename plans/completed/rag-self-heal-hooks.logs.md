# RAG / Cortex Self-Healing Hooks log

**Status:** [DONE]

Durable done-state archive for plans/rag-self-heal-hooks.plans.md (same boundary). Populated per the phase-compression rule: the plan file keeps compact `[DONE]` step headers and references this file; the original step packets, probe checklists, and raw validation evidence live here. Append-only — new phase records are added below existing ones.

## Phase 1 — Planning and research confirmation [DONE]

[DONE] Phase 1: Planning and research confirmation — plan authored and registered, research probes confirmed the Option B (server-internal core) + Option C (shared state file) mechanism with hooks demoted to additive-only, and a fresh independent 01-planning verification instance recorded green-light: true (2026-09-04). Active frontier moved to Phase 2 — Repair scheduler state and self-heal orchestrator.

### Phase 1 packet (as completed)

```yaml
phase: 1
title: Planning and research confirmation
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_phase: 'Phase 2 — Repair scheduler state and self-heal orchestrator'
skills:
  - 'plan-registration'
  - 'phase-handoff-workflow'
  - 'plan-sync-validation'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json'
acceptance_criteria:
  - 'Step 01 authors and registers this plan and passes the consolidated slice-advancement self-check'
  - 'Step 02 produces plans/rag-self-heal-hooks.research.md with empirical answers to the open assumptions'
  - 'Step 03 records green-light: true from an independent verification instance before any Phase 2 step activates'
placeholder_steps:
  - step: 1
    title: 'Author and register the self-heal plan (this pass)'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Research confirmation probes (hook feasibility, probe cost, duration calibration)'
    goal: researching
    status: '[DONE]'
  - step: 3
    title: 'Independent plan verification and green-light'
    goal: planning
    status: '[DONE]'
```

### Step 01 — Author and register the self-heal plan (this pass) [DONE]

[DONE] Step 01: Authored plans/rag-self-heal-hooks.plans.md (mechanism decision, detection inventory, state/lock design, T1-T4 guidance templates, risks, AC block) and registered it in plans/README.md (trigger phrases + tracker bullet) and plans/Roadmap.md (Standalone Meta-Workflow Lane). Fixed the heading/parser-contract defect before the gate self-check. slice-advancement PASS 4/4 sub-gates (P1-S01, TRIVIAL, plans-only); validate-plan-sync ok: true (0 errors, 0 warnings).

Original step packet (moved from plan):

```yaml
phase: 1
step: 1
title: 'Author and register the self-heal plan (this pass)'
status: '[DONE]'
goal: planning
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 1, Step 02 — Research confirmation probes'
skills:
  - 'plan-registration'
  - 'phase-handoff-workflow'
  - 'plan-sync-validation'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S01 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json'
acceptance_criteria:
  - 'Plan file exists with mechanism decision, detection inventory, state/lock design, templates, risks, and AC block'
  - 'Plan registered in plans/README.md trigger list and plans/Roadmap.md standalone meta-workflow lane with matching [WIP] status'
  - 'Consolidated slice-advancement gate passes with plans-only changed-files (TRIVIAL slice)'
  - 'Latest validation evidence updated honestly; no green-light self-awarded'
```

Required validation (moved from plan):

- `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S01 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md`
- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json`

Deliverable note (moved from plan): the plan file plus the two registration edits. Evidence went to `## Latest validation evidence`.

### Step 02 — Research confirmation probes [DONE]

[DONE] Step 02: Created plans/rag-self-heal-hooks.research.md with empirical answers. (a) Hook observability: PostToolUse hooks fire for native tools and successful MCP tool calls and deliver tool_name/tool_input/tool_result on stdin; new hook registrations are not picked up mid-session; failing `cortex-cortex` calls skip hooks — the hook layer was demoted to additive-only and Option B + C was confirmed by four specialists (boundary-mapper, implementation-pattern-scout, repo-cortex-scout, performance-reviewer). (b) checkDenseReadiness() latency (21 samples, model-only corpus): min 173 ms / p50 190 ms / p95 215 ms / max 220 ms. (c) Prewarm duration for ~25k chunks was honestly partial (attempt 1 failed at embed-index with SQLITE_BUSY: database is locked; attempt 2 progressed through embed-index for at least ~5 minutes before being stopped) — production repairs record real durations in the state-file history. Env-knob defaults confirmed: cooldown 600 s, backoff ×2, max 3 attempts per 24 h window, stale-lock 2700 s, probe-cost threshold 600 ms. slice-advancement PASS 4/4 sub-gates (P1-S02).

Original step packet (moved from plan):

```yaml
phase: 1
step: 2
title: 'Research confirmation probes (hook feasibility, probe cost, duration calibration)'
status: '[DONE]'
goal: researching
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 1, Step 03 — Independent plan verification and green-light'
skills:
  - 'research-methodology'
  - 'repo-cortex-workflow'
  - 'repo-cortex-embeddings'
  - 'evidence-based-changes'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: '02'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S02 --changed-files=plans/rag-self-heal-hooks.research.md,plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'Research artifact plans/rag-self-heal-hooks.research.md exists answering: (a) whether hooks fire for MCP tool names with result inspection, (b) measured checkDenseReadiness latency, (c) observed prewarm duration(s) for ~25k chunks from the currently running job and artifacts'
  - 'Mechanism recommendation re-confirmed or revised with one-paragraph justification recorded in the plan'
  - 'Env-knob defaults confirmed or adjusted based on measurements'
```

Probe checklist (moved from plan):

1. Empirically determine whether PreToolUse/PostToolUse hooks observe MCP calls: add a temporary, reversible diagnostic hook entry that records `tool_name` into `rag-index/data/hook-context/`, make one `cortex-cortex` call, inspect the capture, then revert the diagnostic registration.
2. Time `node rag-index/dense-readiness.mjs --json` (and `index_stats`) on this machine; record p50/p95-style numbers.
3. Collect prewarm duration evidence: the currently running `npm run index:prewarm` background job and any duration fields in artifacts/logs.
4. Re-confirm the insertion points listed in Phase 3/4 against current sources (line-level notes in the artifact).

### Step 03 — Independent plan verification and green-light [DONE]

[DONE] Step 03: A fresh 01-planning verification instance validated packets, AC traceability (AC-001..AC-008, 100% coverage), scope honesty, and research answers; recorded green-light: true (2026-09-04). slice-advancement PASS 4/4 sub-gates (P1-S03, TRIVIAL). Non-blocking: full prewarm duration not collected (accepted as honest partial evidence). Step flips were explicit edits — workflow-update-sync was deliberately NOT invoked (its yaml-block regex matches 0 of 13 fenced blocks on this CRLF plan; its substring matcher would hit the Phase 1 phase packet placeholder first and flip the phase status; hazards recorded as learning events). The verifier reported blockers only and did not self-dispatch patch cycles.

Original step packet (moved from plan):

```yaml
phase: 1
step: 3
title: 'Independent plan verification and green-light'
status: '[DONE]'
goal: planning
expansion: 'none'
mode: verification
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet) + plans/rag-self-heal-hooks.research.md'
copy_paste: ready
next_step: 'Phase 2, Step 01 — phase kickoff'
skills:
  - 'plan-sync-validation'
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S03 --changed-files=plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'Fresh 01-planning verification instance validates packets, AC traceability, scope honesty, and research answers; records green-light: true with evidence in ## Latest validation evidence, or records blockers'
  - 'On blockers: verification instance reports them only — it must not self-dispatch patch cycles'
```

## Phase 1 raw validation evidence (moved from plan)

Verbatim record of the plan's `## Latest validation evidence` blocks as they stood at compression time (2026-09-04). The plan keeps a compact summary; this is the durable raw detail.

```text
Authoring pass (2026-07-14, 01-planning authoring instance):
- Plan authored from live research (hooks contract, facade/server sources, gates, skills); files read listed in the authoring session log.
- Mechanism recommendation: Option B (server-internal core) + Option C shared state file; Option A demoted pending Step 02 empirical verification.
- Registration: pending at authoring time — README.md and Roadmap.md edits land in the same pass before the gate self-check.
Gate self-check (run after the repairs below; plans-only changed files, severity TRIVIAL):
- PASS: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S01 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md
  - Sub-gates: plan-sync: pass (all WIP plans registered in README and Roadmap); step-packet: pass (active WIP phase/step packets conform); plan-slice-quality: pass; plan-command-lint: pass. 4/4 gates, 0 failures, 0 gate errors.
- PASS: node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/rag-self-heal-hooks.plans.md --json → ok: true, 0 errors, 0 warnings.
- Parser-contract proof: get_active_workflow_snapshot (neataptic-workflow-mcp, plan_path=this tracker) now resolves activePhase = Phase 1 'Planning and research confirmation' [WIP] and activeStep = Step 01 [WIP] with both validation commands — the heading defect that previously blocked plan parsing is fixed.
Structure repair and registration (P1-S01 completion pass, 01-planning authoring instance):
- Heading defect fixed per the parser contract in scripts/agent-customization/mcp/mcp-plan-utils.mjs: added `## Implementation phases` wrapper and `## Validation gates` terminator; phase headings are now `### Phase N — Title [STATUS]` and step headings `#### Step NN — Title [STATUS]` with inline statuses (separate per-heading `**Status:**` lines removed; the document-level `**Status:** [WIP]` line is retained). A `**Required validation:**` prose block was added to the Step 01 packet so YAML and prose validation commands match.
- Registered in plans/README.md (trigger phrases + tracker bullet) and plans/Roadmap.md (Standalone Meta-Workflow Lane) with [WIP] status.
Orchestrator transition (Agent Zero, 2026-09-04):
- Step 01 flipped [WIP] → [DONE] (gate evidence above); Step 02 flipped [PLANNED] → [WIP] (now the active step).
- Fixed Step 02 pre_execute_hook slice_id 'P1-S02' → '02': mcp-plan-utils resolves whole-step packets by bare step number (deriveStepLabel('P1-S02') = 'P1' matches no step); verified get_slice_context now returns the Step 02 packet.

Research pass (Phase 1 Step 02, 02-researching, 2026-09-04):
- Empirical hook observability: `PostToolUse` hooks fire for native tools and successful MCP tools (e.g. `neataptic-gate-mcp-list_gates`) and deliver `tool_name`, `tool_input`, `tool_result` on stdin. New hook registrations are not picked up mid-session; failing `cortex-cortex` calls skip hooks.
- `checkDenseReadiness()` latency (21 samples, `model-only` corpus): min 173 ms / p50 190 ms / p95 215 ms / max 220 ms.
- Prewarm evidence: first `npm run index:prewarm` attempt failed at `embed-index` with `SQLITE_BUSY: database is locked`; a second attempt progressed through `embed-index` for at least ~5 minutes before being stopped. A full end-to-end duration for ~25k chunks was not collected because the long-running job was still in progress; real durations should be recorded in the state-file history once production repairs run.
- Specialist consensus: boundary-mapper, implementation-pattern-scout, repo-cortex-scout, and performance-reviewer all converged on Option B (server-internal core) + Option C shared state file.
- Env-knob defaults confirmed and adjusted: cooldown 600 s, backoff ×2, max 3 attempts per 24 h window, stale-lock 2700 s, probe-cost threshold 600 ms.
- Research artifact: `plans/rag-self-heal-hooks.research.md` created.
- Gate self-check: PASS — `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S02 --changed-files=plans/rag-self-heal-hooks.research.md,plans/rag-self-heal-hooks.plans.md` → 4/4 sub-gates passed (plan-sync, step-packet, plan-slice-quality, plan-command-lint).
- Step 02 flipped [WIP] → [DONE].

Verification pass (Phase 1 Step 03, fresh 01-planning instance, 2026-09-04):
- green-light: true. Phase 1 verified; Phase 2 activation is now allowed. The orchestrator's Phase 2 kickoff pass must still mark Phase 1 [DONE] and compress Phase 1 history per the phase-compression rule before activating Phase 2.
- Gate evidence: PASS - node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P1-S03 --changed-files=plans/rag-self-heal-hooks.plans.md (run via neataptic-gate-mcp run_gate_check) -> pass: true, 4/4 sub-gates green (plan-sync, step-packet, plan-slice-quality, plan-command-lint), 0 failed, 0 errored, severity TRIVIAL.
- Packet validation: Phase 1 phase/step packets conform and resolve (get_active_workflow_snapshot: activePhase = Phase 1 [WIP], activeStep = Step 03); Step 03 (mode=verification, expansion=none) is a valid whole-step packet. Phase 2/3/4 slice packets conform: 4/4/3 slices per step (<= 5 cap), all estimate_hours <= 2.5 (< 4 h cap), per-slice goals ordered red -> implementing -> green, dependency chains linear with cross-phase wiring (P3-S02-A deps P2-S02-D; P4-S02-A deps P3-S02-D), tdd_sequence red-green on implementing steps.
- AC traceability: AC-001..AC-008 all carry id/text/files/validation/red_test. Phase coverage: AC-002/003/004/007 -> Phase 2 slices; AC-001 guard -> Phase 2, AC-001 response wiring + AC-005 -> Phase 3; AC-006 -> Phase 4; AC-008 -> per-slice gate discipline. Every slice files_to_change entry is covered by a named --testPathPatterns command; no missing / partial / contradicts / unrequested findings. Coverage 100% of acceptance criteria (threshold 80%).
- Scope honesty: in-scope items map 1:1 to Phase 2-4 slices; non-goals hold (no slice adds hook registrations, periodic schedulers, BM25/embedding-model changes, or gate behavior changes; all triggers are signal-based). The authoring pass did not self-award this green light - the previous entry recorded status: pending.
- Research answers verified against plans/rag-self-heal-hooks.research.md; all 3 NEEDS CLARIFICATION assumptions are resolved with user sign-offs recorded in the plan. (a) Hook observability answered empirically: PostToolUse fires for native tools and successful MCP tool calls and delivers tool_name/tool_input/tool_result, but is SKIPPED when the MCP call itself errors, and mid-session hook registrations are frozen -> hook layer correctly demoted to additive-only (Option B + C server-internal mechanism confirmed; consistent with non-goals). (b) Env-knob defaults confirmed by 4 specialists; PROBE_COST_THRESHOLD_MS=600 added (~3x measured p50 190 ms, ~2.8x p95 215 ms, 21 samples) - satisfies the exceptional-scenarios-only requirement. (c) checkDenseReadiness latency measured: min 173 / p50 190 / p95 215 / max 220 ms.
- Non-blocking observation: full end-to-end prewarm duration for ~25k chunks was not collected (attempt 1 failed SQLITE_BUSY; attempt 2 ran >= 5 min in embed-index before being stopped). Accepted as honest partial evidence: the core mechanism does not depend on the exact duration (T1 estimate supports basis=history|chunk-count heuristic; the state file records real repair durations once production repairs run; stale-lock 2700 s ~ 9x the observed embed-index floor; all knobs env-tunable). No blocker.
- Transition: Step 03 flipped [WIP] -> [DONE] (heading + packet YAML + placeholder_steps refreshed to [DONE]; the placeholder statuses had been stale since the manual Step 01/02 transitions). workflow-update-sync deliberately NOT invoked - verified empirically: (a) its yaml-block regex (triple-backtick yaml anchor + bare LF) matches 0 of 13 fenced blocks on this CRLF plan, so the hook would flip the heading but silently skip the YAML status update; (b) with CRLF-tolerant matching its substring block matcher hits the Phase 1 phase packet first (placeholder `step: 3` entry + phase-level status: '[WIP]' satisfy the containment checks before the Step 03 packet), which would flip the PHASE status to [DONE] and skip the compression-first rule. The phase status transition stays explicit for the orchestrator.
- Post-edit snapshot expectation: the Phase 1 packet declares expansion: steps + auto_expand: false, so with all steps [DONE] the workflow snapshot resolves activePhase = Phase 1 [WIP], activeStep = null (between-steps state; mcp-plan-utils throws only when a non-pasted phase lacks a [WIP] step).
- Learning events: recorded the Step 02 pending gap (PostToolUse skips hook capture on MCP errors; mid-session hook registration freeze) and the workflow-update-sync CRLF/placeholder_steps matcher hazard in .github/ai-learning/learning-log.jsonl.
```

## Phase 2 — Repair scheduler state and self-heal orchestrator [DONE]

### Step 01 — Phase kickoff [DONE]

[DONE] Step 01 (2026-09-04, 01-planning kickoff pass): Re-read plans/rag-self-heal-hooks.plans.md and plans/rag-self-heal-hooks.research.md; confirmed the three research answers are absorbed by the Phase 2 packet. (1) Mechanism: hooks demoted to additive-only — Phase 2 slices contain zero hook registrations (Option B server-internal + Option C shared state file scope holds). (2) Env knobs: all seven defaults present in the Architecture env-knob table (COOLDOWN_S=600, BACKOFF_FACTOR=2, MAX_ATTEMPTS=3, ATTEMPT_WINDOW_S=86400, LOCK_STALE_S=2700, PROBE_COST_THRESHOLD_MS=600, DISABLE=0) and enforced via the P2-S02-B env-knobs contract. (3) Probe calibration: 600 ms threshold ≈ 3x the measured p50 190 ms / p95 215 ms, so the state-file-only fallback fires only under exceptional contention. Absorption gaps patched into the Step 02 packet before activation: P2-S02-A red coverage now explicitly requires probe-cost-threshold, kill-switch, and pid-liveness-reclaim tests; P2-S02-B now names the 600 ms probe-cost behavior with its measured calibration basis. Prewarm honesty (SQLITE_BUSY serialization via whole-run lock, real durations in state history) was already covered by P2-S02-C. Gate: PASS — node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P2-S01 --changed-files=plans/rag-self-heal-hooks.plans.md → 4/4 sub-gates (plan-sync, step-packet, plan-slice-quality, plan-command-lint), severity TRIVIAL, plans-only. Snapshot: get_active_workflow_snapshot resolves activePhase = Phase 2 [WIP], activeStep = null (Step 01 [DONE], Step 02 [PLANNED]). Step 02 was NOT started; kickoff did not begin implementation.

[DONE] Phase 2: Repair scheduler state and self-heal orchestrator — red-green TDD delivered cortex-health-guard.mjs (evaluateSelfHeal decision engine) and cortex-self-heal.mjs (orchestrator CLI) with colocated suites; 30/30 mjs tests green; dry-run composes 6 existing-automation commands classifier-first with zero --force; slice-advancement PASS 4/4 (P2-S02-A, TRIVIAL) and PASS 7/7 (P2-S02-C/D, FULL incl. shared-validation, code-coverage, specialist-review); AC-002/003/004/007 green with evidence. Phase compression executed 2026-09-04 (07-logging); active frontier moved to Phase 3 — Server-integrated detection and model-facing guidance.

### Phase 2 packet (as completed)

```yaml
phase: 2
title: 'Repair scheduler state and self-heal orchestrator'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'Phase 3 — Server-integrated detection and model-facing guidance'
skills:
  - 'phase-handoff-workflow'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P2-S02 --changed-files=scripts/agent-customization/cortex/cortex-health-guard.mjs,scripts/agent-customization/cortex/cortex-self-heal.mjs'
acceptance_criteria:
  - 'AC-001 guard/scheduler portion, AC-002, AC-003, AC-004 satisfied with green evidence'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff (re-read plan, confirm research answers absorbed)'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Guard module + orchestrator (red-green, 4 slices)'
    goal: implementing
    status: '[DONE]'
```

### Step 01 — Phase kickoff [DONE]

(Recorded above at kickoff time; unchanged. The plan keeps a compact coverage note.)

### Step 02 — Guard module and self-heal orchestrator (TDD) [DONE]

[DONE] Step 02 (2026-09-04): Four-slice red-green TDD pass implemented the self-heal core. P2-S02-A (red): colocated suites scripts/agent-customization/cortex/cortex-health-guard.test.mjs and cortex-self-heal.test.mjs written plan-exact after a drifted first writer pass (invented createGuard().consult() API, camelCase fixtures) was deleted; RED confirmed on missing modules; slice-advancement PASS 4/4 (TRIVIAL) after converting 10 inline dependency lists to block-style (parser quote-strip defect). P2-S02-B (implementing): cortex-health-guard.mjs — evaluateSelfHeal({probe, now, stateDir, spawner}) -> {action, guidanceFields, spawnDecision}, state schema, cooldown/backoff/ceiling, lock management, kill switch, probe-cost fallback (600 ms ≈ 3x measured p50 190 ms / p95 215 ms), zero-cost warm path; guard suite 15/15 PASS. P2-S02-C (implementing): cortex-self-heal.mjs (runRepair/createLock/refreshLock/releaseLock/reclaimStaleLock/dryRunReport + dry-run CLI) + .gitignore for rag-index/data/cortex-self-heal-state.json, cortex-self-heal.repair.lock, *.lock; two red-fixture repairs disclosed (missing fsCounts Proxy block; dead-pid live-holder fixture); 15/15 PASS; dry-run 6-command classifier-first list, zero --force; slice-advancement PASS 7/7 (FULL). P2-S02-D (green): both suites 15/15 PASS, dry-run JSON recorded (below), convergence-tracker PASS, slice-advancement PASS 7/7 (FULL; four Step 02 source/test files).

Original step packet (moved from plan):

```yaml
phase: 2
step: 2
title: 'Guard module and self-heal orchestrator (TDD)'
status: '[DONE]'
goal: implementing
expansion: 'slices'
auto_expand: true
tdd_sequence: 'red-green'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 3, Step 01 — phase kickoff'
skills:
  - 'red-testing'
  - 'implementation-standards'
  - 'testing'
  - 'green-testing'
  - 'repo-cortex-workflow'
specialists:
  - 'unit-test-writer'
  - 'property-based-test-writer'
  - 'implementation-executor'
constitution_check: 'pass — two new small modules composing existing scripts; no new subsystem or periodic scheduler'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S02-A'
validation:
  - 'npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage'
  - 'npm run jest:mjs -- --testPathPatterns=cortex-self-heal --runInBand --no-coverage'
  - 'node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json'
acceptance_criteria:
  - 'AC-002 cooldown/backoff/ceiling'
  - 'AC-003 lock lifecycle and stale reclamation'
  - 'AC-004 composed-command dry-run in classifier-first order'
  - 'AC-007 zero-cost warm path'
slices:
  - slice_id: 'P2-S02-A'
    title: 'Red: failing tests for guard decision engine and orchestrator contract'
    status: '[DONE]'
    goal: red-testing
    estimate_hours: 1.5
    files_to_change:
      - 'scripts/agent-customization/cortex/cortex-health-guard.test.mjs'
      - 'scripts/agent-customization/cortex/cortex-self-heal.test.mjs'
    acceptance_criteria:
      - 'Suites fail with import/contract errors before implementation exists'
      - 'Cover: trigger-once under cooldown, backoff doubling, max-attempts T3 guidance data, lock create/heartbeat/release/reclaim (age-based + pid-liveness staleness per Architecture), atomic JSON write-then-rename, dry-run command list, warm-path zero fs/spawn instrumentation'
      - 'Cover research-derived knobs: probe-cost threshold — injected probe slower than CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS (default 600 ms) switches subsequent evaluations to state-file-only checks while a fast probe stays on the live-probe path; kill switch — CORTEX_SELFHEAL_DISABLE=1 blocks triggering/spawning while read-side degraded guidance still reports'
    parallelizable: false
    dependencies: []
  - slice_id: 'P2-S02-B'
    title: 'Implement cortex-health-guard.mjs (probe adapter, state file, cooldown/backoff, lock mgmt)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2.5
    files_to_change:
      - 'scripts/agent-customization/cortex/cortex-health-guard.mjs'
    acceptance_criteria:
      - 'Exported API: evaluateSelfHeal({probe, now, stateDir, spawner}) returning {action, guidanceFields, spawnDecision} per the Architecture section'
      - 'State file schema exactly as specified; env knobs honored; kill switch respected'
      - 'Probe-cost threshold honored per research calibration: probe duration is measured; when it exceeds CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS (default 600 ms, ~3x the measured p50 190 ms / p95 215 ms) the guard switches to state-file-only checks between repairs so the normal ~190 ms degraded-path probe never trips it'
    parallelizable: false
    dependencies:
      - 'P2-S02-A'
  - slice_id: 'P2-S02-C'
    title: 'Implement cortex-self-heal.mjs orchestrator CLI + gitignore for state/lock files'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2.5
    files_to_change:
      - 'scripts/agent-customization/cortex/cortex-self-heal.mjs'
      - '.gitignore'
    acceptance_criteria:
      - '--dry-run --json reports composed commands touching only existing automation, classifier-first ordering'
      - 'Lock held whole run with heartbeat; outcome + measured duration appended to state history'
      - 'rag-index/data/cortex-self-heal-state.json, rag-index/data/cortex-self-heal.repair.lock, rag-index/data/*.lock ignored'
    parallelizable: false
    dependencies:
      - 'P2-S02-B'
  - slice_id: 'P2-S02-D'
    title: 'Green: run suites, dry-run orchestrator, record evidence'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/rag-self-heal-hooks.plans.md'
    acceptance_criteria:
      - 'Both suites green under npm run jest:mjs'
      - 'Dry-run JSON output pasted into Latest validation evidence'
      - 'slice-advancement gate green for P2-S02 with the four changed files'
    parallelizable: false
    dependencies:
      - 'P2-S02-C'
```

## Phase 2 raw validation evidence (moved from plan)

Verbatim record of the plan's `## Latest validation evidence` Phase 2 blocks as they stood at compression time (2026-09-04). The plan keeps a compact summary; this is the durable raw detail.

```text
Phase 2 kickoff evidence (Step 01, 01-planning, 2026-09-04):
- P2-S01 (Phase kickoff): PASS slice-advancement 4/4 sub-gates (TRIVIAL, plans-only; changed: plans/rag-self-heal-hooks.plans.md) — plan-sync, step-packet, plan-slice-quality, plan-command-lint all green.
- Research absorption confirmed against plans/rag-self-heal-hooks.research.md: probe-cost threshold 600 ms (~3x measured p50 190 ms / p95 215 ms), all 7 env knobs present in the Architecture table, hooks demoted to additive-only with zero hook registrations in Phase 2 slices.
- Absorption gaps patched into the Step 02 packet before activation: P2-S02-A red coverage now explicitly names probe-cost-threshold fallback, kill-switch blocking, and pid-liveness-reclaim tests; P2-S02-B now names the 600 ms probe-cost behavior with its measured calibration basis. Prewarm honesty (SQLITE_BUSY serialization, measured durations in state history) already covered by P2-S02-C.
- Kickoff edits added no new shell commands, so the command-lint surface is unchanged from the green P1-S01/P1-S03 baseline.
- Snapshot verification: get_active_workflow_snapshot resolves activePhase = Phase 2 [WIP], activeStep = null (between-steps; Step 01 [DONE], Step 02 [PLANNED], auto_expand: false).
- Step 02 NOT started: P2-S02-A..D remain [PLANNED]; no implementation began in this pass.
```

```text
Phase 2 / P2-S02-A red evidence (03-red-testing, 2026-09-04):
- Files changed: scripts/agent-customization/cortex/cortex-health-guard.test.mjs, scripts/agent-customization/cortex/cortex-self-heal.test.mjs (a first writer pass drifted from the mandated contracts — invented createGuard().consult() API and camelCase fixtures — and was deleted; the committed suites encode the plan-exact evaluateSelfHeal({probe, now, stateDir, spawner}) -> {action, guidanceFields, spawnDecision} API and snake_case state/lock schemas).
- RED confirmed (correct reason — missing modules, not syntax/fixture errors):
  - npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage → exit 1: "Cannot find module './cortex-health-guard.mjs'" (top-level await import, line 61).
  - npm run jest:mjs -- --testPathPatterns=cortex-self-heal --runInBand --no-coverage → exit 1: "Cannot find module './cortex-self-heal.mjs'" (top-level await import, line 22).
- Guard suite coverage (15 tests): AC-001 first-trigger start + lock record schema {pid, started_at, heartbeat_at, attempt, reason} + state schema {version, last_probe{state,dense_reason,chunk_count,embedding_count,probed_at}, cooldown, backoff, max_attempts, in_flight, history}; AC-002 in-flight no-respawn, cooldown no-respawn, backoff doubling (1 failure × factor 2 → next_allowed_at = now + 1_200_000), window reset, T3 exhausted + exact manual_recovery triple [npm run index:session-start, npm run index:prewarm, node .../cortex-index.gate.mjs --auto-rebuild --json], kill switch CORTEX_SELFHEAL_DISABLE=1 (action 'disabled', degraded guidance still reports, zero spawns), CORTEX_SELFHEAL_MAX_ATTEMPTS and CORTEX_SELFHEAL_COOLDOWN_S overrides, probe-cost fallback (slow 700 ms probe called once across two evaluates; fast probe stays live at 2 calls; CORTEX_SELFHEAL_PROBE_COST_THRESHOLD_MS=100 override honored), AC-007 warm path (counted-fs Proxy instrumentation: zero fs ops, zero spawns, decision deepEqual {action:null, guidanceFields:null, spawnDecision:null}).
- Orchestrator suite coverage (15 tests): AC-004 dry-run classifier-first full list (6 commands, NO --force) + warm-state list (snapshot+prewarm dropped); lock create schema + null-on-existing-holder; refreshLock heartbeat-only bump; releaseLock + foreign-pid refusal; reclaimStaleLock age-based (LOCK_STALE_S), dead-pid (live-pid inverse), reclaim-once; runRepair success (lock verified present DURING execution via spawner mock, released after), failure release, history record, atomic write-then-rename (rename counted ≥ 1).
- Test mechanics: jest.unstable_mockModule('node:fs'/'node:fs/promises') with counted pass-through Proxy registered BEFORE top-level await import (static imports stay real); fixture helpers write raw fs; env keys CORTEX_SELFHEAL_* saved/restored per test; deterministic BASE_NOW = 1_000_000_000_000; freshImport (jest.resetModules + re-register mocks + re-import) isolates env-read-at-import and probe-cost memo per test.
- Red-defined seams for P2-S02-B/C implementers: backoff formula assumed next_allowed_at = now + cooldown_s × backoff_factor^attempts_in_window; probe-cost persistence assumed in-process memo (fresh per process); dryRunReport({state}) returns the command string array (CLI wrapper validated in P2-S02-D).
- Expected green target: both suites pass under the two focused jest:mjs commands once P2-S02-B implements cortex-health-guard.mjs and P2-S02-C implements cortex-self-heal.mjs; then P2-S02-D runs node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json and pastes output here.
- Slice status: P2-S02-A marked [DONE]. Consolidated slice-advancement gate PASS 4/4 sub-gates for P2-S02-A (TRIVIAL severity; changed files: the two test suites + this plan): plan-sync ok, step-packet ok, plan-slice-quality ok, plan-command-lint ok. First run failed step-packet only ("Slice dependencies must be a list" ×3) — root cause: the plan parser's normalizeYamlScalar turns inline `dependencies: ['X']` into the string "[X]" via its quote-strip regex; fixed by converting all 10 inline dependency lists (P2-S02-B..D, P3-S02-A..D, P4-S02-A..B) to block-style lists, then the gate passed. No implementation modules were touched (P2-S02-B/C strictly untouched).
```

```text
Phase 2 / P2-S02-C implementation evidence (04-implementing, 2026-09-04):
- Files changed: scripts/agent-customization/cortex/cortex-self-heal.mjs (new — six exported seams runRepair/createLock/refreshLock/releaseLock/reclaimStaleLock/dryRunReport + CLI), .gitignore (+ rag-index/data/cortex-self-heal-state.json, + rag-index/data/cortex-self-heal.repair.lock, + rag-index/data/*.lock).
- Red-fixture repairs (deviation from files_to_change, disclosed for orchestrator confirmation; both defects originate in the P2-S02-A writer pass and are unfixable from production code): (1) added the missing fsCounts counted-Proxy block — the line-408 assertion `fsCounts.rename >= 1` referenced an undefined identifier, and the block is proven intended by the file's own trailing comment plus P2-S02-A's atomic-write AC — mirroring the proven cortex-health-guard.test.mjs pattern; (2) the live-holder fixture used spawnSync(...).pid, a just-exited (dead) child — process.kill(pid, 0) → ESRCH, verified empirically on this machine — so no correct liveness implementation could return null; replaced with a genuinely-live spawn() child killed in finally. No assertion weakened; test semantics preserved.
- Validation 1: npm run jest:mjs -- --testPathPatterns=cortex-self-heal --runInBand --no-coverage → exit 0, 15/15 PASS (dry-run command lists, lock lifecycle, stale reclaim incl. the inclusive 2700 s boundary, live-pid no-reclaim, reclaim-once, runRepair success/failure/history/atomic-rename).
- Validation 2: npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage → exit 0, 15/15 PASS (P2-S02-B regression-free).
- Validation 3: node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json → exit 0; 6-command classifier-first list (validate → corpus build → snapshot → prewarm → index gate --auto-rebuild → MCP smoke), zero --force flags; the dry run takes no lock and leaves no state file.
- Coverage: focused run (--testPathPatterns=cortex-self-heal --coverage --coverageDirectory=coverage/project-agent-customization-mjs-selfheal/) → lines 70.73 / statements 68.91 / functions 73.07 / branches 41.93; the uncovered surface is the CLI main entry plus defensive branches (the P2-S02-A red record explicitly defers CLI-wrapper validation to P2-S02-D). merge-coverage-summaries.mjs --baseline --source-files=<self-heal,guard>: merged summary preserves neataptic-dispatch-mcp 100% + guard 89.22% and adds self-heal; baseline records both cortex modules (same mechanism and precedent as P2-S02-B's guard entry).
- slice-advancement gate: PASS 7/7 sub-gates for P2-S02-C (FULL severity; changed files: module + test file + .gitignore): plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review. Note: the gate MCP wrapper rejected the args twice ("did not return valid JSON"); the gate CLI contract requires equals-form --slice-id=P2-S02-C (recorded as a learning event).
- Slice status: P2-S02-C marked [DONE]. Remaining in Step 02: P2-S02-D (green) — re-run both suites, paste the dry-run JSON into Latest validation evidence, and run the step-level slice-advancement gate with all Step 02 changed files.
```

```text
Phase 2 / P2-S02-B implementation evidence (04-implementing, 2026-09-04):
- Files changed: scripts/agent-customization/cortex/cortex-health-guard.mjs (new — evaluateSelfHeal API with probe adapter, state-file read/write, cooldown/backoff/ceiling math, lock management, kill-switch, probe-cost fallback, zero-cost warm path).
- Validation: npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage → exit 0, 15/15 PASS (first trigger, lock schema, state schema, in-flight/cooldown no-respawn, backoff doubling, window reset, T3 pause-and-ask guidance, env overrides, kill switch, probe-cost fallback, warm-path zero fs/spawn).
- Coverage: focused run included in the consolidated agent-customization-mjs project baseline; module recorded as covered by the same merge-coverage-summaries mechanism used for P2-S02-C.
- Slice status: P2-S02-B marked [DONE]; P2-S02-C subsequently verified P2-S02-B regression-free (15/15 guard tests still PASS).
```

```text
Phase 2 / P2-S02-D green evidence (05-green-testing, 2026-09-04):
- Step 02 file surface covered by this gate pass: scripts/agent-customization/cortex/cortex-health-guard.mjs, scripts/agent-customization/cortex/cortex-health-guard.test.mjs, scripts/agent-customization/cortex/cortex-self-heal.mjs, scripts/agent-customization/cortex/cortex-self-heal.test.mjs (P2-S02-B + P2-S02-C source/test files).
- Validation 1: npm run jest:mjs -- --testPathPatterns=cortex-health-guard --runInBand --no-coverage → exit 0, 15/15 PASS.
- Validation 2: npm run jest:mjs -- --testPathPatterns=cortex-self-heal --runInBand --no-coverage → exit 0, 15/15 PASS.
- Validation 3: node scripts/agent-customization/cortex/cortex-self-heal.mjs --dry-run --json → exit 0; 6-command classifier-first list (validate → corpus build → snapshot → prewarm → index gate --auto-rebuild → MCP smoke), zero --force flags; dry run takes no lock and leaves no state file.
- Dry-run JSON:
{
  "dry_run": true,
  "state_dir": "C:\\NeatapticTS\\rag-index\\data",
  "state_file": "C:\\NeatapticTS\\rag-index\\data\\cortex-self-heal-state.json",
  "lock_file": "C:\\NeatapticTS\\rag-index\\data\\cortex-self-heal.repair.lock",
  "last_probe_state": null,
  "commands": [
    "node rag-index/validate-index.mjs --json",
    "node rag-index/build-index.mjs --json",
    "npm run index:build-snapshot",
    "npm run index:prewarm",
    "node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json",
    "node scripts/agent-customization/gates/cortex-mcp-smoke.mjs --json"
  ],
  "ordering": "classifier-first",
  "notes": [
    "Every command shells out to existing automation; no repair logic is reimplemented.",
    "No --force flag is composed; stale rebuilds go through the auto-rebuild gate.",
    "A dry run executes nothing and takes no lock."
  ]
}
- Convergence tracker gate: PASS (no excessive fix-loop iterations for P2-S02-D).
- slice-advancement gate: PASS 7/7 sub-gates for P2-S02-D (FULL severity; changed files: the four Step 02 source/test files): plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review all green.
- Slice status: P2-S02-D marked [DONE]. Step 02 marked [DONE]. Phase 2 remains [WIP] until Phase 2 close-out is executed.
```

## Phase 3 — Server-integrated detection and model-facing guidance [DONE]

[DONE] Phase 3: Server-integrated detection and model-facing guidance — four-slice red-green TDD pass wired `evaluateSelfHeal` into search-corpus.mjs, search-context.mjs, index-stats.mjs, and repo-cortex-mcp.mjs; 349/349 mjs tests green across four suites; live DENSE_FORCE_STATE=model-only smoke produced a structured self_heal block with T3 exhausted guidance and BM25 fallback; slice-advancement PASS 7/7 for P3-S02; docs-scout README/JSDoc drift fixed. Original step packets + raw evidence below.

### Phase 3 packet (as completed)

```yaml
phase: 3
title: 'Server-integrated detection and model-facing guidance'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'Phase 4 — Server-unavailable guidance seam (facade)'
skills:
  - 'phase-handoff-workflow'
  - 'repo-cortex-embeddings'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02 --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-corpus.test.mjs,scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/tools/search-context.test.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/tools/index-stats.test.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs,scripts/mcp-semantic/repo-cortex-mcp.test.mjs'
acceptance_criteria:
  - 'AC-001 response augmentation and trigger-once wiring'
  - 'AC-005 post-repair warm observation without server restart'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Search-tool and index_stats wiring (red-green, 4 slices)'
    goal: implementing
    status: '[DONE]'
```

### Step 01 — Phase kickoff (server-integration absorption) [DONE]

[DONE] Step 01: Research and Phase 2 module absorption confirmed; gaps patched into P3-S02 packet before activation. Reconstructed step packet below.

```yaml
phase: 3
step: 1
title: 'Phase kickoff (server-integration absorption)'
status: '[DONE]'
goal: planning
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 3, Step 02 — Search-tool and index_stats wiring'
skills:
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S01 --changed-files=plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'Research answers absorbed; gaps patched into P3-S02 packet before activation'
files_to_change:
  - 'plans/rag-self-heal-hooks.plans.md'
```

### Step 02 — Search-tool and index_stats wiring (TDD) [DONE]

[DONE] Step 02: Four-slice red-green TDD pass delivered search-tool/index_stats wiring and docs-quality fixes. Reconstructed step packet below.

```yaml
phase: 3
step: 2
title: 'Search-tool and index_stats wiring (TDD)'
status: '[DONE]'
goal: implementing
expansion: 'slices'
auto_expand: true
tdd_sequence: 'red-green'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 4, Step 01 — Phase kickoff'
skills:
  - 'red-testing'
  - 'implementation-standards'
  - 'green-testing'
specialists:
  - 'unit-test-writer'
  - 'implementation-executor'
constitution_check: 'pass — response augmentation only; no behavior change on the success path'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P3-S02-A'
validation:
  - 'npm run jest:mjs -- --testPathPatterns=search-corpus|search-context|index-stats|repo-cortex-mcp --runInBand --no-coverage'
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02 --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-corpus.test.mjs,scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/tools/search-context.test.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/tools/index-stats.test.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs,scripts/mcp-semantic/repo-cortex-mcp.test.mjs'
acceptance_criteria:
  - 'AC-001 response augmentation and trigger-once wiring'
  - 'AC-005 post-repair warm observation without server restart'
files_to_change:
  - 'scripts/mcp-semantic/tools/search-corpus.mjs'
  - 'scripts/mcp-semantic/tools/search-corpus.test.mjs'
  - 'scripts/mcp-semantic/tools/search-context.mjs'
  - 'scripts/mcp-semantic/tools/search-context.test.mjs'
  - 'scripts/mcp-semantic/tools/index-stats.mjs'
  - 'scripts/mcp-semantic/tools/index-stats.test.mjs'
  - 'scripts/mcp-semantic/repo-cortex-mcp.mjs'
  - 'scripts/mcp-semantic/repo-cortex-mcp.test.mjs'
slices:
  - slice_id: 'P3-S02-A'
    title: 'Red: failing tests for search-tool self-heal wiring'
    status: '[DONE]'
    goal: red-testing
    estimate_hours: 1
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-corpus.test.mjs'
      - 'scripts/mcp-semantic/tools/search-context.test.mjs'
      - 'scripts/mcp-semantic/tools/index-stats.test.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.test.mjs'
    acceptance_criteria:
      - 'Red suites fail for missing implementation seams, not syntax/fixture/import errors'
    parallelizable: false
    dependencies:
      - 'P2-S02-D'
  - slice_id: 'P3-S02-B'
    title: 'Implement search-corpus.mjs self-heal wiring'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 1
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-corpus.mjs'
      - 'scripts/mcp-semantic/tools/search-corpus.test.mjs'
    acceptance_criteria:
      - 'evaluateSelfHeal consulted on degraded dense outcomes; invalidateDenseReadinessCache seam exposed; self_heal block merged'
    parallelizable: false
    dependencies:
      - 'P3-S02-A'
  - slice_id: 'P3-S02-C'
    title: 'Implement search-context.mjs, index-stats.mjs, and repo-cortex-mcp.mjs wiring'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'scripts/mcp-semantic/tools/search-context.mjs'
      - 'scripts/mcp-semantic/tools/index-stats.mjs'
      - 'scripts/mcp-semantic/repo-cortex-mcp.mjs'
    acceptance_criteria:
      - 'All three surfaces emit self_heal block on degraded paths; repo-cortex-mcp dead import removed'
    parallelizable: false
    dependencies:
      - 'P3-S02-B'
  - slice_id: 'P3-S02-D'
    title: 'Green: full smoke + plan update'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 0.5
    files_to_change:
      - 'plans/rag-self-heal-hooks.plans.md'
    acceptance_criteria:
      - 'All four suites green; slice-advancement PASS 7/7 for P3-S02'
    parallelizable: false
    dependencies:
      - 'P3-S02-C'
```

### Raw validation evidence (moved from plan)

```text
Phase 3 Step 02 — P3-S02-A red evidence (2026-09-04):
- Files changed: scripts/mcp-semantic/tools/search-corpus.test.mjs (created), scripts/mcp-semantic/tools/search-context.test.mjs (created), scripts/mcp-semantic/tools/index-stats.test.mjs (edited), scripts/mcp-semantic/repo-cortex-mcp.test.mjs (edited).
- Focused Jest commands confirm RED for the right reason (missing implementation seams, not syntax/fixture/import errors):
  * npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage → exit 1 (evaluateSelfHeal not called; invalidateDenseReadinessCache undefined)
  * npm run jest:mjs -- --testPathPatterns=search-context --runInBand --no-coverage → exit 1 (response.self_heal undefined)
  * npm run jest:mjs -- --testPathPatterns=index-stats --runInBand --no-coverage → exit 1 (dense_state/self_heal undefined)
  * npm run jest:mjs -- --testPathPatterns=repo-cortex-mcp --runInBand --no-coverage → exit 1 (handler responses lack self_heal augmentation)
- slice-advancement gate PASS for P3-S02-A (plan-sync, step-packet, plan-slice-quality, plan-command-lint) via neataptic-gate-mcp.
- Green target for 04-implementing (P3-S02-B/C): wire evaluateSelfHeal from scripts/agent-customization/cortex/cortex-health-guard.mjs into search-corpus.mjs, search-context.mjs, index-stats.mjs, and repo-cortex-mcp.mjs; expose invalidateDenseReadinessCache seam in search-corpus.mjs; reuse state/lock paths from cortex-self-heal.mjs.
- Structural note: slice-validator flagged that P3-S02-A files_to_change has 4 files (>3 boundary) and AC #5 references guard/orchestrator modules not listed in files_to_change. The step packet was authored and approved in pragmatic mode with this broad slice; treat as a documented plan-quality caveat, not a red-phase blocker.
```

```text
Phase 3 Step 02 — P3-S02-B implementation evidence (2026-09-04):
- File changed: scripts/mcp-semantic/tools/search-corpus.mjs.
- Added `import { evaluateSelfHeal } from '../../agent-customization/cortex/cortex-health-guard.mjs';`.
- Exported `invalidateDenseReadinessCache()` seam to clear the process-lifetime `cachedDenseReadiness` memo without a server restart.
- `createDegradedBm25Response` now consults `evaluateSelfHeal({ probe: () => Promise.resolve(readinessReport) })` only on degraded dense outcomes, merges `decision.guidanceFields` into `response.self_heal`, and invalidates the dense cache when `action === 'started'`.
- Guard failures are caught and swallowed: the plain BM25 response is still returned.
- Focused Jest smoke: `npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage` → 3 suites, 154 tests PASS.
- Preflight: `npx tsc --noEmit -p tsconfig.json` PASS; `npm run lint` PASS.
- slice-advancement gate: PASS 4/4 sub-gates for P3-S02-B (plan-sync, step-packet, plan-slice-quality, plan-command-lint).
- Out-of-scope (P3-S02-C/D red tests): search-context, index-stats, and repo-cortex-mcp suites still fail as expected because those surfaces have not been wired yet.
```

```text
Phase 3 Step 01 evidence (kickoff re-run, fresh 01-planning instance):
- P3-S01 (Phase kickoff): PASS slice-advancement 4/4 sub-gates (TRIVIAL, plans-only); validate-plan-sync ok: true (0 errors, 0 warnings); research artifact `plans/rag-self-heal-hooks.research.md` absorbed (hooks additive-only / zero new registrations; probe-cost threshold 600 ms ≈ 3× measured p50 190 ms / p95 215 ms; insertion points in search-corpus/search-context/index-stats/repo-cortex-mcp.mjs).
- Phase 2 modules absorbed for reuse (not reimplementation): `cortex-health-guard.mjs` (`evaluateSelfHeal` → {action,guidanceFields,spawnDecision}) and `cortex-self-heal.mjs` (lock/state paths, dry-run report, composed repair commands).
- Gaps patched in P3-S02 packets:
  * P3-S02-A red-test files now include `scripts/mcp-semantic/tools/index-stats.test.mjs` and explicit guard/orchestrator reuse criteria.
  * P3-S02-C implementation files now include `scripts/mcp-semantic/tools/index-stats.mjs` with mismatch-field + backward-compatible registration criteria.
  * Step-level validation commands now include an `index-stats` Jest run.
  * P3-S02 gate changed-files list expanded to include all source + test files touched by the step.
- Step 02 remains `[PLANNED]`; not started per user mandate.
```

#### Documentation evidence (P3-S02 step-closure docs-quality pass)

- Files audited: `scripts/mcp-semantic/README.md`, `scripts/mcp-semantic/tools/search-corpus.mjs`, `scripts/mcp-semantic/tools/search-context.mjs`, `scripts/mcp-semantic/tools/index-stats.mjs`, `scripts/mcp-semantic/repo-cortex-mcp.mjs`.
- Drift detected by `docs-scout`:
  - README `search_corpus` example omits new `dense_degraded`, `dense_reason`, and `self_heal` fields emitted on degraded paths.
  - README `search_context` schema lists `context_format` enum as `['text', 'json']` with default `'text'`; source accepts only `'markdown'` / `'json'` and defaults to `'markdown'`.
  - README `search_context` example uses `context_tokens` instead of `token_count` and omits `self_heal`, `rerank_state`, and `dense_degraded`.
  - README `search_context` schema lists a boolean `follow_up_refs` input that the MCP server does not expose; `follow_up_refs` is always returned in the response.
  - README `index_stats` says `'Schema: no arguments'` and example only shows counts/timestamp; source now accepts `include_metadata_coverage` and returns `feedback_stats`, `ann`, `metadata_coverage`, dense-readiness fields (`dense_state`, `dense_reason`, `dense_degraded`, `chunk_count`, `embedding_count`), and `self_heal`.
  - Source JSDoc gaps: floating `searchCorpus`/`searchCorpusImpl` JSDoc block in `search-corpus.mjs`; `createDegradedBm25Response` `@param` omits `alpha` and `client`; exported `searchCorpus` `@returns` omits new fields; `indexStats` `@returns` omits dense/self-heal fields; `SearchContextResult` typedef omits several response fields.
- Fixes applied:
  - `scripts/mcp-semantic/README.md`: corrected `search_context` schema (`context_format` enum `markdown`/`json`, default `markdown`); removed spurious `follow_up_refs` input; updated example output (`token_count`, `rerank_state`, degraded-path note); rewrote `index_stats` schema + example to include `include_metadata_coverage`, `feedback_stats`, `ann`, `metadata_coverage`, dense-readiness/self-heal fields and the non-enumerable property note.
  - `scripts/mcp-semantic/tools/search-corpus.mjs`: removed floating JSDoc block; expanded exported `searchCorpus` JSDoc with full options and return fields (`dense_state`, `dense_degraded`, `dense_reason`, `self_heal`, `rerank_state`, `diskann_used`, `rrf_used`, `response_tokens`, `freshness`); added `alpha` and `client` to `createDegradedBm25Response` `@param`.
  - `scripts/mcp-semantic/tools/search-context.mjs`: expanded `SearchContextResult` typedef with `compact`, `total_chunks_retrieved`, `chunks_in_context`, `tokens_used`, `context_budget_consumed`, `budget_remaining`, `dedup_strategy`, `results`, `top_result`, `follow_up_refs`, `self_heal`, `rerank_state`, and `freshness`.
  - `scripts/mcp-semantic/tools/index-stats.mjs`: updated `indexStats` `@returns` to include `metadata_coverage`, dense-readiness fields, and `self_heal`; noted non-enumerable property contract.
- Latest validation evidence:
  - `node --check` passed for all four changed `.mjs` files.
  - `npm run jest:mjs -- --runInBand --testPathPatterns='search-corpus|search-context|index-stats|repo-cortex'`: 10 suites, 349 tests passed.
  - `npm run docs:quality:gate`: pass=true (schema_valid, deterministic_ordering, comparator_guards, parity_cli_mcp, contract_invalid_rejected).
  - `node rag-index/build-index.mjs` and `npm run index:build-snapshot` completed; `cortex-index` gate reports `index_fresh=true`, all families fresh, snapshot_age=0. Remaining `cortex-index` pass=false is due to `workflow_mcp_alive=false` (MCP server binding), not documentation content.
  - `node scripts/agent-customization/gates/slice-advancement.gate.mjs --slice-id=P3-S02`: pass=true; all 7 sub-gates (plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review) green.
- Step-closure docs note: All `docs-scout` drift findings for the four changed tool files and the hand-maintained README have been addressed. No generated READMEs were hand-edited. Residual gap: low-priority `@example`/`@throws` completeness on small internal helpers remains out of scope for this slice.

#### PlanUpdate P3-S02-B

```yaml
plan_update:
  plan: plans/rag-self-heal-hooks.plans.md
  slice_id: P3-S02-B
  status: DONE
  files_changed:
    - scripts/mcp-semantic/tools/search-corpus.mjs
    - scripts/mcp-semantic/tools/search-corpus.test.mjs
  evidence:
    - command: npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage
      result: PASS (3 suites, 154 tests)
    - command: npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --coverage
      result: PASS (3 suites, 155 tests); search-corpus.mjs coverage 100/100/100/100 (stmts/branch/funcs/lines)
    - command: npx tsc --noEmit -p tsconfig.json
      result: PASS
    - command: npm run lint
      result: PASS
    - command: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02-B --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-corpus.test.mjs
      result: PASS 7/7 sub-gates (FULL severity: plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review)
  next_slice: P3-S02-C
  note: search-context/index-stats/repo-cortex-mcp suites still red as expected (P3-S02-C scope).
```

#### PlanUpdate P3-S02-C

```yaml
plan_update:
  plan: plans/rag-self-heal-hooks.plans.md
  slice_id: P3-S02-C
  status: DONE
  files_changed:
    - scripts/mcp-semantic/tools/search-context.mjs
    - scripts/mcp-semantic/tools/index-stats.mjs
    - scripts/mcp-semantic/repo-cortex-mcp.mjs
    - scripts/mcp-semantic/__tests__/search-context.coverage.test.mjs
    - scripts/mcp-semantic/tools/index-stats.test.mjs
    - scripts/mcp-semantic/repo-cortex-mcp.test.mjs
  evidence:
    - command: npm run jest:mjs -- --testPathPatterns='search-context|index-stats|repo-cortex-mcp|search-corpus' --runInBand --no-coverage
      result: PASS (10 suites, 349 tests)
    - command: npx tsc --noEmit -p tsconfig.json
      result: PASS
    - command: npm run lint
      result: PASS
    - command: npm run jest:mjs -- --testPathPatterns='search-context|index-stats|repo-cortex-mcp|search-corpus' --runInBand --coverage
      result: |
        PASS (10 suites, 349 tests).
        Per-file coverage (stmts/branch/funcs/lines):
          - scripts/mcp-semantic/tools/search-context.mjs: 100/100/100/100
          - scripts/mcp-semantic/tools/index-stats.mjs: 100/100/100/100
          - scripts/mcp-semantic/repo-cortex-mcp.mjs: 100/100/100/100
          - scripts/mcp-semantic/tools/search-corpus.mjs: 100/100/100/100
    - command: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02-C --changed-files=scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs
      result: PASS 7/7 sub-gates (FULL severity: plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review)
  next_slice: P3-S02-D
  tests_for_green_testing:
    - npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand
    - npm run jest:mjs -- --testPathPatterns=search-context --runInBand
    - npm run jest:mjs -- --testPathPatterns=index-stats --runInBand
    - npm run jest:mjs -- --testPathPatterns=repo-cortex-mcp --runInBand
    - node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02-C --changed-files=scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs
  note: 'Second follow-up complete. search-context.mjs now prepends the human self-heal guidance paragraph to the assembled markdown context when degraded, satisfying the design requirement that text-only harnesses see the warning. search-corpus.mjs and index-stats.mjs have no natural text body (results array / structured stats), so no prepend was added. repo-cortex-mcp.mjs had the dead checkDenseReadiness import removed. Focused tests assert the context text begins with the guidance paragraph and cover the no-prepend branches (no self_heal, JSON context, already-prefixed context, non-string guidance). All P3-S02-A red suites remain green, the three changed source files still have 100% statements/branches/functions/lines coverage, and the slice-advancement consolidated gate passes 7/7.'
```

#### PlanUpdate P3-S02-D green evidence

```text
Phase 3 Step 02 — P3-S02-D green evidence (2026-09-04):
- Allow-listed validation commands (run via neataptic-validation-mcp-run_allowlisted_validation):
  * npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage → PASS (3 suites, 155 tests)
  * npm run jest:mjs -- --testPathPatterns=search-context --runInBand --no-coverage → PASS (4 suites, 114 tests)
  * npm run jest:mjs -- --testPathPatterns=index-stats --runInBand --no-coverage → PASS (2 suites, 25 tests)
  * npm run jest:mjs -- --testPathPatterns=repo-cortex-mcp --runInBand --no-coverage → PASS (1 suite, 55 tests)
- Live DENSE_FORCE_STATE=model-only smoke (tmp/dense-smoke-P3-S02-D.mjs):
  * dense_degraded: true, dense_state: model-only, dense_reason: DENSE_FORCE_STATE forced model-only readiness.
  * self_heal.action: exhausted (attempt 3/3, max_attempts 3, cooldown_s 600, next_allowed_at 1788659608946).
  * manual_recovery: [npm run index:session-start, npm run index:prewarm, node scripts/agent-customization/gates/cortex-index.gate.mjs --auto-rebuild --json]
  * guidance begins: 'Cortex dense search is degraded ... Auto-repair is paused to avoid a blocker loop.'
  * BM25 fallback returned 3 results in ~27 ms.
- Consolidated Tier-1 gate: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02 --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-corpus.test.mjs,scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/__tests__/search-context.coverage.test.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/tools/index-stats.test.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs,scripts/mcp-semantic/repo-cortex-mcp.test.mjs,plans/rag-self-heal-hooks.plans.md → PASS 7/7 sub-gates (FULL severity).
- Smoke script created at tmp/dense-smoke-P3-S02-D.mjs (artifact); may be removed after evidence is accepted.
```

#### PlanUpdate P3-S02-D

```yaml
plan_update:
  plan: plans/rag-self-heal-hooks.plans.md
  slice_id: P3-S02-D
  status: DONE
  files_changed:
    - plans/rag-self-heal-hooks.plans.md
  evidence:
    - command: npm run jest:mjs -- --testPathPatterns=search-corpus --runInBand --no-coverage
      result: PASS (3 suites, 155 tests)
    - command: npm run jest:mjs -- --testPathPatterns=search-context --runInBand --no-coverage
      result: PASS (4 suites, 114 tests)
    - command: npm run jest:mjs -- --testPathPatterns=index-stats --runInBand --no-coverage
      result: PASS (2 suites, 25 tests)
    - command: npm run jest:mjs -- --testPathPatterns=repo-cortex-mcp --runInBand --no-coverage
      result: PASS (1 suite, 55 tests)
    - command: node tmp/dense-smoke-P3-S02-D.mjs
      result: |
        dense_degraded: true, dense_state: model-only, action: exhausted (3/3 attempts),
        self_heal.manual_recovery commands emitted, BM25 fallback 3 results, latency ~27 ms.
    - command: node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P3-S02 --changed-files=scripts/mcp-semantic/tools/search-corpus.mjs,scripts/mcp-semantic/tools/search-corpus.test.mjs,scripts/mcp-semantic/tools/search-context.mjs,scripts/mcp-semantic/__tests__/search-context.coverage.test.mjs,scripts/mcp-semantic/tools/index-stats.mjs,scripts/mcp-semantic/tools/index-stats.test.mjs,scripts/mcp-semantic/repo-cortex-mcp.mjs,scripts/mcp-semantic/repo-cortex-mcp.test.mjs,plans/rag-self-heal-hooks.plans.md
      result: PASS 7/7 sub-gates (FULL severity: plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review)
  next_slice: none
  note: 'P3-S02-D green phase complete. All four declared suites are green, the live DENSE_FORCE_STATE=model-only smoke produced a structured self_heal block via the real shared state file, and the P3-S02 consolidated slice-advancement gate passes 7/7.'
```

## Phase 4 — Server-unavailable guidance seam (facade) [DONE]

[DONE] Phase 4 (2026-09-05): Server-unavailable guidance seam — three-slice red-green TDD pass delivered T4 spawn-failure guidance in `scripts/agent-customization/mcp/lazy-facade-core.mjs` and focused contract tests in `scripts/agent-customization/mcp/lazy-facade-core.test.mjs`; 71/71 lazy-facade tests green, 82/82 combined facade suites green, `lazy-facade-core.mjs` 100% statements/branches/functions/lines coverage, slice-advancement PASS 7/7 (FULL). Active frontier moved to Phase 5 — Documentation and closure.

### Phase 4 packet (as completed)

```yaml
phase: 4
title: 'Server-unavailable guidance seam (facade)'
status: '[DONE]'
goal: planning
expansion: steps
auto_expand: false
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (phase packet)'
copy_paste: ready
next_phase: 'Phase 5 — Documentation and closure'
skills:
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P4-S02 --changed-files=scripts/agent-customization/mcp/lazy-facade-core.mjs,scripts/agent-customization/mcp/lazy-facade-core.test.mjs,plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'AC-006 T4 guidance on spawn failure'
placeholder_steps:
  - step: 1
    title: 'Phase kickoff'
    goal: planning
    status: '[DONE]'
  - step: 2
    title: 'Facade failure guidance (red-green, 3 slices)'
    goal: implementing
    status: '[DONE]'
```

### Step 01 — Phase kickoff (facade failure guidance absorption) [DONE]

[DONE] Step 01 (2026-09-05): Phase kickoff confirmed Phase 3 evidence intact and P4-S02 packet consistent; no absorption gaps required. slice-advancement PASS 4/4 sub-gates (P4-S01, TRIVIAL, plans-only).

Original step packet (moved from plan):

```yaml
phase: 4
step: 1
title: 'Phase kickoff (facade failure guidance absorption)'
status: '[DONE]'
goal: planning
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Step 02 — Facade failure guidance (TDD)'
skills:
  - 'phase-handoff-workflow'
  - 'repo-cortex-workflow'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P4-S01 --changed-files=plans/rag-self-heal-hooks.plans.md'
acceptance_criteria:
  - 'Phase 3 [DONE] compression and evidence preserved'
  - 'Phase 4 status flipped to [WIP], Step 01 to [DONE], Step 02 to [WIP]'
  - 'P4-S02 packet reviewed for absorption gaps; no patches required'
```

Step objective (moved from plan): Confirm Phase 3 server-integration evidence is intact, review the P4-S02 facade-failure packet for any absorption gaps or stale assumptions, and update the tracker so Step 02 can activate cleanly.

Context the agent must know (moved from plan): Phase 3 wired `evaluateSelfHeal` into `search-corpus.mjs`, `search-context.mjs`, `index-stats.mjs`, and `repo-cortex-mcp.mjs`; 349/349 mjs tests green; live `DENSE_FORCE_STATE=model-only` smoke produced T3 exhausted guidance. Step 02 is already authored with three slices (P4-S02-A red, P4-S02-B implement, P4-S02-C green). No new research is expected; the kickoff is a readiness gate before activation.

Stop conditions (moved from plan): Done when the plan file shows Phase 4 [WIP], Step 01 [WIP] (or [DONE] after this pass), P4-S02 packet is consistent, and the slice-advancement gate passes for P4-S01. Blocked if Phase 3 evidence is missing or the Step 02 packet has unresolved contradictions — route to `00-helping`.

### Step 02 — Facade failure guidance (TDD) [DONE]

[DONE] Step 02 (2026-09-05): Three-slice red-green TDD pass delivered T4 facade guidance and green validation. Files changed: `scripts/agent-customization/mcp/lazy-facade-core.mjs`, `scripts/agent-customization/mcp/lazy-facade-core.test.mjs`, `plans/rag-self-heal-hooks.plans.md`. Validations: 71/71 lazy-facade tests green, 82/82 combined facade suites green, 100% branch coverage, slice-advancement PASS 7/7 (FULL).

Original step packet (moved from plan):

```yaml
phase: 4
step: 2
title: 'Facade failure guidance (TDD)'
status: '[DONE]'
goal: implementing
expansion: 'slices'
auto_expand: true
tdd_sequence: 'red-green'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 5, Step 01 — documentation'
skills:
  - 'red-testing'
  - 'implementation-standards'
  - 'green-testing'
specialists:
  - 'unit-test-writer'
  - 'implementation-executor'
constitution_check: 'pass — error-path enrichment only; no behavior change on the success path'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P4-S02-A'
validation:
  - 'npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage'
acceptance_criteria:
  - 'Spawn failure renders T4 template with manual recovery commands; error still propagates'
slices:
  - slice_id: 'P4-S02-A'
    title: 'Red: failing test for T4 facade guidance'
    status: '[DONE]'
    goal: red-testing
    estimate_hours: 1
    files_to_change:
      - 'scripts/agent-customization/mcp/lazy-facade-core.test.mjs'
    acceptance_criteria:
      - 'Injected spawn failure asserts T4 text, self_heal action unavailable, error propagation, and no retry loop'
    parallelizable: false
    dependencies:
      - 'P3-S02-D'
  - slice_id: 'P4-S02-B'
    title: 'Implement T4 guidance in lazy-facade-core.mjs'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 1.5
    files_to_change:
      - 'scripts/agent-customization/mcp/lazy-facade-core.mjs'
    acceptance_criteria:
      - 'fallbackHint path replaced by template-rendered guidance while preserving JSON-RPC error semantics'
    parallelizable: false
    dependencies:
      - 'P4-S02-A'
  - slice_id: 'P4-S02-C'
    title: 'Green: suite green + evidence recorded'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 0.5
    files_to_change:
      - 'plans/rag-self-heal-hooks.plans.md'
    acceptance_criteria:
      - 'slice-advancement gate green for P4-S02'
    parallelizable: false
    dependencies:
      - 'P4-S02-B'
```

### Raw validation evidence (moved from plan)

Verbatim record of the plan's `## Latest validation evidence` Phase 4 blocks as they stood at compression time (2026-09-05). The plan keeps a compact summary; this is the durable raw detail.

```text
Phase 4 Step 01 kickoff (2026-09-05):
- Phase 3 [DONE] compression and evidence preserved; P4-S02 packet reviewed for absorption gaps — no patches required.
- Phase 4 status flipped [WIP]; Step 01 [DONE]; Step 02 [WIP].
- slice-advancement PASS 4/4 sub-gates (P4-S01, TRIVIAL, plans-only): plan-sync, step-packet, plan-slice-quality, plan-command-lint.
- README/Roadmap already register plan [WIP], Phases 1-3 [DONE], Phase 4 next; no index edits required.
```

```text
Phase 4 Step 02 P4-S02-A red (2026-09-05):
- RED confirmed for the right reason in scripts/agent-customization/mcp/lazy-facade-core.test.mjs: 6 new focused tests inside describe('createLazyFacade') covering the T4 facade contract — T4 guidance markers ('Cortex MCP server failed to start', 'RAG search is UNAVAILABLE', 'grep/glob/view'); manual-recovery markers ('TURSO_DATABASE_URL', 'npm run index:session-start', 'npm run index:prewarm', 'node scripts/mcp-semantic/repo-cortex-mcp.mjs'); error summary 'spawn failed' embedded in guidance; self_heal action 'unavailable' with full 10-field schema and manual_recovery subset; fallbackHint removed; error propagation + no spawn retry across two dispatches.
- Focused validation (allow-listed) `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage`: exit code 1 — 5 failed / 65 passed / 70 total. All 5 failures are missing-implementation assertions (4x TypeError reading 'includes'/'action' of undefined payload.guidance/payload.self_heal; 1x fallbackHint still present: "Start the real cortex server manually or check the spawn command."). The no-retry/propagation test passes as a regression guard over already-correct behavior (cached rejected initPromise).
- Fixture: existing mock seam — childProcess.spawn throws Error('spawn failed'), fs.readFileSync returns router snapshot with tool name 'cortex', new cortexConfig (target 'cortex', spawn command scripts/mcp-semantic/repo-cortex-mcp.mjs); makeRouterSnapshot extended to accept a router tool name (backward-compatible default); beforeEach mock resets unchanged; no timers/files.
- slice-advancement PASS 4/4 sub-gates (P4-S02-A, TRIVIAL) via run_gate_check: plan-sync, step-packet, plan-slice-quality, plan-command-lint.
- Delegation: tests authored by unit-test-writer (Tier 3); contract-fidelity corrections applied by 03-red-testing orchestrator review (added 'grep/glob/view' + 'TURSO_DATABASE_URL' markers, added error-summary test, relaxed manual_recovery deep-equal to subset assertion per designed contract).
- Slice status intentionally left [PLANNED] — orchestrator owns transitions; P4-S02-C records green evidence.
- Expected green (P4-S02-B): spawn-failure payload renders T4 template guidance with <error_summary> filled by the spawn failure message plus a full self_heal block (action 'unavailable', manual_recovery containing both npm commands), fallbackHint removed, error still propagates, spawn called exactly once.
```

```text
Phase 4 Step 02 P4-S02-B implementing (2026-09-05):
- Implemented T4 spawn-failure guidance in scripts/agent-customization/mcp/lazy-facade-core.mjs: `spawnCommand` propagated through dispatch/serverDispatch context; new `isSpawnFailureError()` classifies spawn-time errors; new `buildCortexSpawnFailurePayload()` renders T4 guidance string, error summary, and full self_heal block; catch block emits T4 payload only for target 'cortex' + spawn failure, keeping generic fallbackHint path for all other errors and preserving JSON-RPC tool-error semantics (isError: true, content[0].text JSON).
- Added focused regression test for the 'Spawn returned an invalid child process.' branch to reach 100% branch coverage on lazy-facade-core.mjs.
- Targeted tests: `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage` → 71 passed / 71 total; combined lazy-facade + cortex-facade + devtools-facade → 82 passed / 82 total.
- Type/lint preflight: `npx tsc --noEmit -p tsconfig.json` PASS; `npm run quality:folder -- --folder=scripts/agent-customization/mcp` PASS (0 ESLint errors across 5 files).
- Coverage: `npm run jest:mjs -- --runInBand --coverage --testPathPatterns=lazy-facade` + `node scripts/agent-customization/gates/merge-coverage-summaries.mjs` produced 100% lines/statements/functions/branches for scripts/agent-customization/mcp/lazy-facade-core.mjs.
- slice-advancement PASS 7/7 sub-gates (P4-S02, FULL): plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review.
- Slice status flipped [DONE]; P4-S02-C set [WIP] for green-testing handoff.
```

```text
Phase 4 Step 02 P4-S02-C green-testing (2026-09-05):
- Allow-listed focused validation `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage`: 71 passed / 71 total.
- Regression guard across related facade surfaces `npm run jest:mjs -- --testPathPatterns='lazy-facade|cortex-facade|devtools-facade' --runInBand --no-coverage`: 82 passed / 82 total.
- Re-ran consolidated `slice-advancement` gate for P4-S02 with changed files (lazy-facade-core.mjs, lazy-facade-core.test.mjs, plans/rag-self-heal-hooks.plans.md): PASS 7/7 sub-gates (plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review), severity FULL.
- No new source changes in this slice; coverage and specialist review evidence carried forward from P4-S02-B and confirmed passing by the consolidated gate.
- Slice status flipped [DONE].
```

### Phase 4 compressed session summary

### Phase 4 — Server-unavailable guidance seam (facade)

- Files changed:
  - `scripts/agent-customization/mcp/lazy-facade-core.mjs`: added `isSpawnFailureError()` classifier and `buildCortexSpawnFailurePayload()` T4 guidance renderer on the spawn-failure path
  - `scripts/agent-customization/mcp/lazy-facade-core.test.mjs`: added 6 focused T4 contract tests plus a 100% branch-coverage regression test for invalid child process
  - `plans/rag-self-heal-hooks.plans.md`: recorded P4-S02-A/B/C validation evidence under `## Latest validation evidence`
- Validations run:
  - `npm run jest:mjs -- --testPathPatterns=lazy-facade --runInBand --no-coverage` — initial red run 65 passed / 5 failed (expected missing-implementation assertions); final green run 71 passed / 0 failed
  - `npm run jest:mjs -- --testPathPatterns='lazy-facade|cortex-facade|devtools-facade' --runInBand --no-coverage` — 82 passed / 0 failed
  - `npx tsc --noEmit -p tsconfig.json` — PASS
  - `npm run quality:folder -- --folder=scripts/agent-customization/mcp` — PASS (0 ESLint errors)
  - `npm run jest:mjs -- --runInBand --coverage --testPathPatterns=lazy-facade` + merge coverage — 100% lines/statements/functions/branches for `lazy-facade-core.mjs`
  - `slice-advancement` gate for P4-S02 — PASS 7/7 sub-gates (FULL)
- Learning events:
  - `.github/ai-learning/learning-log.jsonl`: workflow-update-sync matcher hazards (CRLF / `placeholder_steps`) recorded as learning events
- Decisions:
  - T4 guidance emitted only when `targetName === 'cortex'` and error classifies as spawn failure; generic `fallbackHint` path preserved for all other errors
  - Error-path enrichment only; success path behavior unchanged
  - 100% branch coverage required before marking green slice complete
- Risks / residual gaps:
  - Phase 5 documentation alignment for `.github/skills/repo-cortex-workflow/SKILL.md` and `.github/skills/repo-cortex-embeddings/SKILL.md` not yet started
  - Plan remains active until Phase 5 Step 02 closure archive; do not run `workflow-update-sync` on this plan (CRLF/placeholder_steps matcher hazards recorded as learning events)
- Next resume point:
  - Dispatch `06-documenting` for Phase 5 Step 01 to add short self-heal surface pointers to `.github/skills/repo-cortex-workflow/SKILL.md` and `.github/skills/repo-cortex-embeddings/SKILL.md` without duplicating repair sequences.

## Phase 5 — Documentation and closure [DONE]

[DONE] Phase 5: Documentation alignment completed and plan closed (2026-09-05). README/Roadmap flipped to `[DONE]`, plan/logs pair archived to `plans/completed/`. Evidence below and in `plans/completed/rag-self-heal-hooks.plans.md`.

### Phase 5 Step 01 packet (moved from plan)

```yaml
phase: 5
step: 1
title: 'Documentation alignment'
status: '[DONE]'
goal: documenting
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'Phase 5, Step 02 — compression and closure'
skills:
  - '6-documentation-style'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P5-S01 --changed-files=.github/skills/repo-cortex-workflow/SKILL.md,.github/skills/repo-cortex-embeddings/SKILL.md'
acceptance_criteria:
  - 'repo-cortex-workflow gains a short Self-heal surface section (pointer only; canonical repair sequences stay single-sourced there)'
  - 'repo-cortex-embeddings degraded-state section links to the self_heal response contract'
  - 'No duplication of repair sequences into other docs'
files_to_change:
  - '.github/skills/repo-cortex-workflow/SKILL.md'
  - '.github/skills/repo-cortex-embeddings/SKILL.md'
```

### Phase 5 Step 02 packet (moved from plan)

```yaml
phase: 5
step: 2
title: 'Compression and closure'
status: '[DONE]'
goal: logging
expansion: 'none'
mode: pragmatic
source_of_truth: 'plans/rag-self-heal-hooks.plans.md (this packet)'
copy_paste: ready
next_step: 'plan archive'
skills:
  - 'logging'
validation:
  - 'node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=P5-S02 --changed-files=plans/rag-self-heal-hooks.plans.md,plans/README.md,plans/Roadmap.md'
acceptance_criteria:
  - 'PlanUpdate block appended; README/Roadmap flipped to [DONE] only when every phase is DONE; tracker moves to plans/completed per index policy'
```

### Phase 5 final session summary

- Files changed:
  - `.github/skills/repo-cortex-workflow/SKILL.md`: added "Self-heal surface" pointer section
  - `.github/skills/repo-cortex-embeddings/SKILL.md`: added "Degraded-state response" section linking to `self_heal` response contract
  - `plans/rag-self-heal-hooks.plans.md`: compressed Phase 5, flipped top status to `[DONE]`, removed `## Handoff query`, added final-state/audit/reopen sections
  - `plans/rag-self-heal-hooks.logs.md`: appended Phase 5 closure record, flipped status to `[DONE]`
  - `plans/README.md`: moved plan entry to `plans/completed/`, status `[DONE]`
  - `plans/Roadmap.md`: moved plan entry to `plans/completed/`, status `[DONE]`
- Validations run:
  - `slice-advancement` gate for P5-S02 — PASS (plan-level sub-gates)
  - `validate-plan-sync` — PASS
  - `log-completion-marker` gate — PASS
  - `stale-wip-plans` gate — PASS
- Learning events:
  - none new in this slice (existing workflow-update-sync matcher hazards already recorded)
- Decisions:
  - No new hook registrations; mechanism remains server-internal + shared-state only
  - Repair sequences stay single-sourced in `cortex-self-heal.mjs` / `cortex-health-guard.mjs`; skill docs contain only pointers
- Risks / residual gaps:
  - Full end-to-end prewarm duration for ~25k chunks still not empirically measured; production repairs will record real durations in `cortex-self-heal-state.json`
  - `workflow_mcp_alive=false` remains a live host-side MCP server binding issue, not a plan gap
- Next resume point:
  - Workstream complete; no active next step. Reopen only via a successor tracker or by moving the archived pair back to `plans/`.
