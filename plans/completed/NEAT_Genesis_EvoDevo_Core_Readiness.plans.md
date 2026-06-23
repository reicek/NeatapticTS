# NEAT Genesis EvoDevo: Core Readiness Audit

**Status:** [DONE] (Phase 4 [DONE]; Phase 5 [DONE]; Phase 6 [DONE]; Phase 7 [DONE]; Phase 7 Step 07 [DONE])

Claim: 07-logging @ 2026-06-23T02:31:55.654Z

```yaml
PlanUpdate:
  step_id: '06-step-01'
  status: '[DONE]'
  decision: 'Experimental NGE root API frozen as a single nge namespace from src/neataptic.ts via a new src/neat/nge-experimental.ts re-export module.'
  surface:
    - 'nge.adult: advanceAdultState + types (src/neat/nge-adult/neat.nge-adult.ts)'
    - 'nge.lifecycle: runNgeLifecycle + types (src/neat/neat.nge-lifecycle.ts)'
    - 'nge.juvenile: namespace re-export (src/neat/nge-juvenile/neat.nge-juvenile.ts)'
    - 'nge.assimilation: assimilateEquilibriumCandidate (src/neat/nge-assimilation/neat.nge-assimilation.ts)'
  non_goals:
    - 'No top-level NGE exports in src/neataptic.ts.'
    - 'No Phase G motifs (ModulatorBroadcaster, EpisodicSlot, GatingRouter, polyandric reproduction).'
    - 'No stability guarantees beyond experimental.'
  next: 'Step 02 — Public-surface and dependency mapping (02-researching).'
```

```yaml
PlanUpdate:
  step_id: '06-step-02'
  status: '[DONE]'
  decision: 'Seam map complete. src/neataptic.ts has no NGE exports today; the frozen nge namespace can be added safely via export * as nge from ./neat/nge-experimental. Four sub-namespaces (adult, lifecycle, juvenile, assimilation) are collision-free. Downstream benchmark plans do not currently consume the frozen nge entrypoint and honestly defer Phase G/E motifs.'
  surface:
    - 'nge.adult: advanceAdultState + types (src/neat/nge-adult/neat.nge-adult.ts)'
    - 'nge.lifecycle: runNgeLifecycle + types (src/neat/neat.nge-lifecycle.ts)'
    - 'nge.juvenile: namespace re-export (src/neat/nge-juvenile/neat.nge-juvenile.ts)'
    - 'nge.assimilation: assimilateEquilibriumCandidate (src/neat/nge-assimilation/neat.nge-assimilation.ts)'
  seam:
    - 'Create src/neat/nge-experimental.ts as a JSDoc barrel re-exporting adult, juvenile, lifecycle, assimilation sub-namespaces.'
    - 'Add export * as nge from ./neat/nge-experimental to src/neataptic.ts after the multi namespace export.'
  downstream_blockers: 0
  next: 'Step 03 — Red tests for experimental root API exposure (03-red-testing).'
```

```yaml
PlanUpdate:
  step_id: '06-step-04'
  status: '[DONE]'
  decision: 'Implemented the frozen experimental NGE namespace as a single nge export from src/neataptic.ts via a new src/neat/nge-experimental.ts re-export barrel. No top-level NGE names were added.'
  changed_files:
    - 'src/neat/nge-experimental.ts'
    - 'src/neataptic.ts'
    - 'testing/neataptic.nge-experimental.test.ts (formatting only)'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json: PASS'
    - 'npm run build: PASS'
    - 'npm run lint: PASS'
    - 'npx prettier --check src/neat/nge-experimental.ts src/neataptic.ts testing/neataptic.nge-experimental.test.ts: PASS'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=testing/neataptic.nge-experimental.test.ts'
      exit: 0
      result: '9/9 tests pass'
  coverage_guard:
    files:
      - 'src/neat/nge-experimental.ts'
    summary: 'statements:100,branches:100,functions:100,lines:100'
  coverage_nuance:
    - 'src/neataptic.ts reports 0% functions in a focused run because it is a root re-export barrel that assigns imported implementations to exported constants. The new re-export statement is exercised by the test; no new functions were added, so this is not a regression caused by the slice.'
  gates:
    - 'validate-plan-sync.mjs: PASS'
    - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage: PASS'
  next: 'Step 05 — Green validation and export-surface audit (05-green-testing).'
```

## Audit summary

Phase 3 closed 2026-06-07: independent-population harness and generation-barrier semantics implemented in NGE core (`src/neat/nge-collective/neat.nge-collective.two-population.ts`), validated with 10/10 targeted tests + 89/89 nge-collective regression tests passing, transport-neutral contract documented, deterministic transport normalization deferred to Phase 4. Racing is the first proving ground; Predator/Prey is the second consumer with stronger synchronized-two-population pressure.

## Scope

Audit the readiness of NGE core primitives before benchmark demo work advances,
and select the smallest aligned first implementation tranche that removes the
highest-leverage blocker. Demos are downstream end-to-end tests of the core;
when a demo exposes a gap, the gap routes back to NGE core rather than being
solved with demo-local compensation. Racing is the first e2e proving ground
after core readiness is sufficient.

Upstream authority: [plans/completed/NEAT_Genesis_EvoDevo.md](completed/NEAT_Genesis_EvoDevo.md)

Active downstream benchmarks:

- [plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md](NEAT_Genesis_EvoDevo_Racing_Curriculum.md) [WIP]
- [plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md](NEAT_Genesis_EvoDevo_AntHive_Demo.md) [PLANNED]
- [plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md](NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md) [PLANNED]

## MCP tracking plan

```yaml
workstream: nge_core_readiness_audit
source_reference: plans/completed/NEAT_Genesis_EvoDevo.md
active_tracker: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
primary_boundary: nge_primitive_readiness_and_core_gap_closure
reason:
  - 'NGE core primitives must be audited and the highest-leverage gaps closed before benchmark demos can act as honest e2e tests.'
  - 'Demo-local compensation for missing core capabilities produces benchmark results that reflect the workaround rather than the algorithm.'
  - 'Racing is the first e2e proving ground; any racing gap must route back to NGE core rather than being solved inside the racing demo.'
preserve_terms:
  - rolling opponent snapshot
  - polyandric reproduction
  - deterministic race packs
  - Phase A/B/D/E/G
  - team-level fitness
  - generation barriers
  - lifecycle staging
  - juvenile and assimilation boundaries
  - nge-adult
mcp_services:
  workflow:
    - neataptic-workflow-mcp.get_active_workflow_snapshot
    - neataptic-workflow-mcp.get_customization_inventory
  cortex:
    - neataptic-cortex-mcp.search_corpus
    - neataptic-cortex-mcp.freshness_check
  gates:
    - neataptic-gate-mcp.list_gates
    - neataptic-gate-mcp.run_gate_check
    - neataptic-gate-mcp.query_customization_routing_table
  validation:
    - neataptic-validation-mcp.get_active_validation_allowlist
    - neataptic-validation-mcp.run_allowlisted_validation
specialist_delegation:
  research:
    - NGE Core Scout
    - Repo Cortex Scout
  implementation:
    - 04-implementing
    - NGE Core Scout
  validation:
    - 05-green-testing
    - Coverage Guard
  escalation:
    - '00-helping only when an MCP/tool/agent/flow gap blocks the active step.'
non_goals:
  - 'Do not implement missing NGE primitives during the audit phase.'
  - 'Do not patch benchmark-local workarounds into demos to hide missing core capabilities.'
  - 'Do not optimize the first tranche around racing-, predator/prey-, or ant-hive-specific semantics before generic NGE foundation gaps are closed.'
  - 'Do not treat demo completion as proof of readiness if the demo only works via benchmark-local compensation.'
  - 'Do not export every internal NGE module at once; prefer a narrow experimental namespace for the initial public entrypoint.'
acceptance_criteria:
  - id: primitive_classification
    criterion: 'Given the requested primitive list, when the audit completes, then each primitive is classified as implemented, partial, plan-only, or missing with at least one file-backed citation.'
    validation: 'readiness matrix with file citations'
  - id: gap_ownership
    criterion: 'Given a primitive is not fully ready, when it is recorded in the audit, then it is assigned to the correct owner boundary: upstream NGE core, benchmark-local runtime, worker protocol, or policy decision.'
    validation: 'gap classification table'
  - id: terminology_preservation
    criterion: 'Given plan alignment matters, when the audit names a gap, then it preserves existing upstream terminology such as rolling opponent snapshot, polyandric reproduction, deterministic race packs, and Phase A/B/D/E/G.'
    validation: 'audit doc review'
  - id: benchmark_decomposition
    criterion: 'Given downstream benchmarks depend on different subsets of NGE, when the audit compares readiness, then it distinguishes reusable core support from racing-only, predator/prey-only, and ant-hive-only requirements.'
    validation: 'per-benchmark readiness column in matrix'
  - id: smallest_tranche
    criterion: 'Given the user wants an implementation plan after the audit, when the audit closes, then it identifies the smallest next tranche that removes the highest-leverage blocker.'
    validation: 'tranche selection rationale doc'
  - id: lifecycle_honesty
    criterion: 'Given lifecycle staging is easy to overclaim, when the audit evaluates lifecycle support, then it explicitly calls out whether the current boundary is production-ready or still scaffolded or red-phase.'
    validation: 'lifecycle status row in matrix'
  - id: core_first_routing
    criterion: 'Given demos are end-to-end tests, when an e2e exposes a missing capability or failure, then the gap is routed back to NGE core rather than solved with demo-local compensation.'
    validation: 'routing policy documented in tranche handoff'
stop_conditions:
  done: 'All Phase 1 steps are complete and the implementation handoff packet is ready for Phase 2.'
  hold: 'A benchmark-policy decision is needed before the first tranche can be selected.'
  blocked: 'A missing upstream primitive or MCP/tooling gap prevents honest audit completion.'
```

## Current-state audit summary

### Step 02 readiness matrix (verbatim archival copy)

| Primitive                                                            | Status          | Owner           | Depends on / unblocks           |
| -------------------------------------------------------------------- | --------------- | --------------- | ------------------------------- |
| `nge-dna` deterministic envelope / realized phenotype                | **partial**     | NGE core        | racing, ant-hive, predator/prey |
| Computation motif catalogue (`ResidualStream`, `WeightSharedCohort`) | **partial**     | NGE core        | all                             |
| Memory-tier params/runtime                                           | **partial**     | NGE core        | all, strongest for ant-hive     |
| Neuromodulator zones / broadcast economics                           | **partial**     | NGE core        | racing, ant-hive, predator/prey |
| Phase E reproduction incl. **polyandric reproduction**               | **implemented** | NGE core        | all                             |
| Phase B juvenile focus / grow boundary                               | **partial**     | NGE core        | all                             |
| `nge-adult` / adult-side **lifecycle staging**                       | **partial**     | NGE core        | all                             |
| Phase D assimilation write-back                                      | **implemented** | NGE core        | all                             |
| `nge-collective` shared field + role divergence                      | **implemented** | NGE core        | ant-hive, racing                |
| **rolling opponent snapshot** pool                                   | **implemented** | NGE core        | racing, predator/prey           |
| Two-population coevolution harness                                   | **partial**     | NGE core        | racing, predator/prey           |
| Team/group fitness aggregation seam                                  | **missing**     | policy decision | racing, ant-hive                |
| Generation barriers                                                  | **missing**     | worker protocol | racing, predator/prey           |
| Deterministic race packs / packed `race-step` transport              | **partial**     | worker protocol | racing first                    |
| Root public API experimental entrypoint (`src/neataptic.ts`)         | **missing**     | NGE core        | all external demos/tests        |

### Step 02 gap inventory

- team-level fitness core evaluator
- independent populations and generation barriers
- deterministic evaluation packs / deterministic race packs normalization
- lifecycle staging closure and `nge-adult` readiness reconciliation
- experimental root public API exposure
- downstream benchmark dependency + MCP synchronization

### Implemented primitives with code evidence

- `src/neat/nge-dna/*` provides deterministic DNA schema, fingerprints, virtual-plan build, and phenotype realization.
- `src/neat/nge-evolution/*` provides parthenogenesis, polyandric, and sexual reproduction operators plus reproduction-policy types.
- `src/neat/nge-collective/*` provides shared typed-array fields, collective evaluation context, rolling opponent snapshot pools, and a narrow two-population harness scaffold.
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.tier3.ts`, `.tier4.ts`, and `.tier5.ts` define packed race-pack transport shapes for 2v2 and 3v3 worker slices.

### Partial or not yet benchmark-ready

- **Team-level fitness** — described in plans but not implemented as a reusable core evaluator; the current two-population harness only partitions results and cross-registers snapshots.
- **Independent populations** — exist as separate `Neat` controllers in the two-population harness, but generation barriers and richer coevolution logic remain plan-owned gaps.
- **Deterministic evaluation packs** — partially present in racing worker helpers, but the reusable contract is not yet closed as a benchmark-ready core primitive.
- **Lifecycle staging** — uneven: juvenile and assimilation boundaries exist, but `src/neat/nge-adult/*` still advertises red-phase or placeholder behavior; end-to-end lifecycle readiness is not complete.
- **Root public API exposure** — NGE surfaces are not exported from `src/neataptic.ts`; support is still internal or experimental.
- **Racing plan gap confirmation** — the active racing plan records team-level fitness, generation barriers, and deterministic race packs as current gaps, confirming that racing should validate core readiness rather than define it.

## Open assumptions and planning gaps

- The exact Step 02 readiness table is preserved here verbatim rather than regenerated so every later phase can anchor back to the same audit baseline.
- The deterministic evaluation-pack owner boundary is frozen (Phase 4 Step 01): core (Layer 2) owns the deterministic pack contract; racing (Layer 3) owns frame assembly; worker protocol shares Layer 2 transfer/clone contracts.
- The lifecycle contradiction between archived closure claims and the current readiness audit remains open until file-backed staging evidence and tracker language reconcile.
- The exact public entrypoint shape is still open: a narrow experimental namespace may be safer than exporting every internal NGE module at once.
- The first execution tranche still needs a final priority decision between lifecycle closure, reusable collective-coevolution semantics, and public-surface wiring if all three cannot fit cleanly into one bounded pass.

## Tracker invariants

- Mandatory MCP flow: before starting any step, re-read `neataptic-workflow-mcp.get_active_workflow_snapshot` for this tracker and for any downstream tracker named by the active phase.
- Mandatory Cortex flow: use `neataptic-cortex-mcp.search_corpus` and freshness checks during Step 02 seam/evidence mapping before phase scope is treated as stable.
- No later bucket: every Step 02 gap remains a named phase in this tracker until it reaches `[DONE]` with file-backed evidence and explicit user confirmation.
- Cross-plan movement must be explicit: when work moves to Racing, Predator/Prey, or Ant Hive, record the target tracker, target step, and return condition here before leaving this plan.
- A phase cannot close by re-labeling unfinished work as deferred; unresolved work must remain `[WIP]`, move to a newly named readiness phase, or be marked `[BLOCKED]`.
- If workflow snapshot, Cortex freshness, validation allowlist, or routing evidence is unavailable, mark the active step `[BLOCKED]` and escalate to `00-helping`; do not continue on memory alone.

## Implementation phases

### Phase 1 — Readiness audit, matrix preservation, and phase packetization [DONE]

```yaml
phase: 1
title: 'Readiness audit, matrix preservation, and phase packetization'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Team-level fitness core evaluator'
skills:
  - execute
  - nge-core-algorithm
  - plan-alignment
  - phase-handoff-workflow
validation:
  - 'node scripts/agent-customization/validate-agent-graph.mjs'
acceptance_criteria:
  - 'All readiness audit steps complete'
placeholder_steps:
  - 'Step 01 — Boundary mapping'
  - 'Step 02 — Prior art scan'
  - 'Step 03 — Core algorithm design'
  - 'Step 04 — Benchmark methodology'
  - 'Step 05 — Documentation baseline'
```

#### Step 01 — Initial survey and findings [DONE]

[DONE] Initial audit compressed: upstream plan evidence, code exploration, and
racing curriculum gap review confirm that DNA, reproduction operators, collective
harness, and race-pack worker shapes are partially present; team-level fitness,
generation barriers, deterministic evaluation packs, lifecycle completeness
(`nge-adult`), and root public API exposure remain the highest-leverage gaps.
See Current-state audit summary above for file-backed evidence.

#### Step 02 — Readiness matrix and gap classification [DONE]

```yaml
phase: 1
step: 2
goal: 'researching'
expansion: none
auto_expand: false
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 03 — Gap-to-phase mapping'
skills:
  - execute
  - nge-core-algorithm
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**User instruction:** Paste this full step packet.

**Step objective:** Produce a primitive-by-primitive readiness matrix, paste the exact Step 02 readiness matrix into this tracker, and classify each gap by owner boundary so the named readiness phases can be selected with confidence.

**Context the agent must know:**

- The upstream authority is `plans/completed/NEAT_Genesis_EvoDevo.md` (archived baseline); preserve its terminology including rolling opponent snapshot, polyandric reproduction, deterministic race packs, Phase A/B/D/E/G, lifecycle staging, and nge-adult.
- The core-first rule is hard: any gap found during this audit belongs to NGE core unless it is demonstrably racing-, predator/prey-, or ant-hive-only.
- The current-state survey above is a starting point; the readiness matrix must be backed by file citations.
- Racing is the first e2e proving ground; any racing gap that is not demo-local must route back to NGE core before the racing curriculum advances.
- The public API exposure gap (`src/neataptic.ts`) is in scope as a core gap.

**Execution steps:**

1. Use `neataptic-workflow-mcp.get_active_workflow_snapshot` to confirm this tracker is active and Step 02 is [WIP].
2. Use `neataptic-cortex-mcp.search_corpus` for: team-level fitness, generation barriers, deterministic race packs, lifecycle staging, nge-adult, nge-collective, nge-evolution, nge-dna, rolling opponent snapshot, polyandric reproduction, public API entrypoint.
3. Read `src/neat/nge-dna/README.md`, `src/neat/nge-evolution/README.md`, `src/neat/nge-collective/README.md`, and `src/neat/nge-adult/README.md` (or nearest available READMEs) as the first-pass scope.
4. Produce a readiness matrix with columns: primitive, status (implemented/partial/plan-only/missing), file citation, owner boundary (NGE core/benchmark-local/worker protocol/policy decision), and which benchmark(s) depend on it.
5. Classify each partial/missing primitive by owner boundary; do not route benchmark-local gaps into the core tranche.
6. Identify the highest-leverage gap cluster: the set of partial/missing primitives whose closure unblocks the most downstream benchmark capability.
7. Record the matrix and gap classification in this tracker, including the verbatim archival matrix section and the Step 02 gap inventory, before ending the step.

**Stop conditions:**

- **Done:** readiness matrix covers all primitives from the upstream plan with file citations, the exact Step 02 matrix is pasted into this tracker, and gap classification is owner-specific.
- **Hold:** a policy decision is needed before a gap can be classified.
- **Blocked:** an MCP/tool/agent gap prevents honest research; escalate to `00-helping`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 03 — Gap-to-phase mapping [DONE]

```yaml
phase: 1
step: 3
goal: 'researching'
expansion: none
auto_expand: false
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 04 — First active implementation cycle selection'
skills:
  - execute
  - plan-alignment
  - phase-handoff-workflow
  - nge-core-algorithm
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**Step objective:** Convert every Step 02 gap into an explicit named phase in this tracker with no deferred or later bucket, and record which owner boundary each new phase closes.

**Execution focus:**

1. Re-read the archived Step 02 matrix plus the file-backed evidence behind each partial or missing primitive.
2. Map each distinct gap cluster into one named readiness phase and preserve the approved phase titles already declared in this tracker.
3. Record any shared prerequisites or sequencing rules without collapsing multiple gaps into a deferred bucket.
4. Mark downstream benchmark dependencies and the return condition for any gap that needs work in Racing, Predator/Prey, or Ant Hive after core closure.

**Stop conditions:**

- **Done:** every Step 02 gap is represented by an explicit named phase in this tracker, sequencing notes are recorded, and no unresolved work is labeled for later.
- **Hold:** a user priority decision is needed before the tranche can be finalized.
- **Blocked:** Step 02 matrix is incomplete; return to Step 02.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 04 — First active implementation cycle selection [DONE]

```yaml
phase: 1
step: 4
goal: 'researching'
expansion: none
auto_expand: false
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 05 — Cross-plan MCP handoff packet'
skills:
  - execute
  - plan-alignment
  - phase-handoff-workflow
  - nge-core-algorithm
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**Step objective:** Select the first active implementation cycle from the named readiness phases using the Step 02 evidence, the no-later-bucket rule, and the highest-leverage downstream unblock.

**Execution focus:**

1. Compare the named readiness phases against benchmark dependencies, owner boundaries, and current file-backed maturity.
2. Pick the smallest implementation cycle that removes the highest-leverage blocker without hiding unfinished work behind a deferred label.
3. Record why the non-selected phases remain queued and what evidence or prerequisite they still need before promotion to `[WIP]`.
4. Align the selected cycle with downstream benchmark return conditions so Racing remains the first proving ground rather than the design owner.

**Stop conditions:**

- **Done:** the first implementation cycle is explicitly selected, justified, and aligned to the named phases in this tracker.
- **Hold:** a user priority decision is still needed between equally valid first cycles.
- **Blocked:** Step 03 output is incomplete.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

[DONE] Selection compressed: the first active implementation cycle is **Phase 2 —
Team-level fitness core evaluator**. It is the smallest reusable core cycle that
closes a **missing** Step 02 primitive, unblocks **Racing** and **Ant Hive**
simultaneously, and prevents Racing from becoming the design owner for team
scoring semantics. Phase 3 explicitly depends on this evaluator boundary staying
stable, so promoting Phase 2 first removes the highest-leverage downstream
blocker without hiding any unfinished work behind a later bucket.

**Cycle selection and queue rationale:**

- **Selected first cycle — Phase 2 / Team-level fitness core evaluator:** Step 02
  records the team/group fitness aggregation seam as **missing** and needed by
  `racing, ant-hive`; the current code only partitions two-population results
  and cross-registers snapshots. This is a narrower first cycle than lifecycle
  reconciliation or public-surface exposure, but it still removes the most
  immediate shared benchmark blocker and freezes the policy seam before any
  benchmark-local compensation appears.
- **Phase 3 remains queued — Independent populations and generation barriers:**
  the gap is real, but the phase already declares it is blocked if **Phase 2
  still changes prerequisite evaluator semantics**. Step 02 also shows partial
  two-population scaffolding plus missing generation barriers, so Phase 3 should
  start only after the reusable team-level fitness contract is fixed in NGE
  core.
- **Phase 4 remains queued — Deterministic evaluation packs / deterministic race
  packs normalization:** Step 02 still marks the owner boundary as provisional,
  and Phase 4 already says its contract should keep Racing as the proving ground
  rather than the owner. It needs the Phase 3 coevolution/barrier seams to
  settle before transport normalization can be scoped honestly.
- **Phase 5 remains queued — Lifecycle staging closure and `nge-adult`
  readiness reconciliation:** lifecycle is benchmark-critical, but this tracker
  still carries an open contradiction between archived claims and current
  readiness evidence. The phase itself says it is blocked if earlier phases are
  still changing prerequisite lifecycle contracts, so it stays named and queued
  rather than being promoted ahead of the shared coevolution seam.
- **Phase 6 remains queued — Experimental root public API exposure:** the root
  export gap is still real, but exposing a public experimental namespace before
  the first reusable evaluator and lifecycle seams stabilize would advertise a
  false-ready surface. Its own phase already treats lifecycle closure as a
  prerequisite.
- **Phase 7 remains queued — Downstream benchmark dependency + MCP
  synchronization:** Step 05 owns the first cross-plan handoff packet. This
  phase should not move until there is completed core-phase evidence to hand to
  Racing, Predator/Prey, or Ant Hive.

**Downstream benchmark alignment:**

- **Racing remains the first proving ground, not the design owner:** the active
  racing tracker is still on **Phase 1 Step 02 — Research boundary mapping
  [WIP]**, where missing prerequisites must be classified as demo-local,
  reusable library/API, or upstream NGE core. The selected Phase 2 cycle keeps
  team-level fitness owned by NGE core, so Racing only validates that Team A/B
  benchmark flow can consume the reusable evaluator once it exists.
- **Return condition for the next handoff packet:** when Phase 2 closes with
  green validation, Step 05 should route the result back into
  `plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md` so Racing can continue as
  the first e2e consumer of core-owned team-level fitness without implying that
  unresolved Phase 3–6 gaps are solved.
- **Ant Hive stays queued as the second direct consumer:** it shares the same
  missing team-level fitness dependency, so selecting Phase 2 first improves its
  eventual readiness without making Ant Hive the policy owner either.

#### Step 05 — Cross-plan MCP handoff packet [DONE]

```yaml
phase: 1
step: 5
goal: 'researching'
expansion: none
auto_expand: false
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Phase 2 — Team-level fitness core evaluator'
skills:
  - execute
  - plan-alignment
  - phase-handoff-workflow
  - repo-cortex-workflow
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**Step objective:** Record the MCP-aware cross-plan handoff packet for any downstream benchmark tracker touched by the selected first implementation cycle, including target tracker, target step, and return condition.

**Execution focus:**

1. Name every downstream tracker that depends on the selected cycle and record its current step.
2. Re-read `neataptic-workflow-mcp.get_active_workflow_snapshot` for this tracker and each named downstream tracker before finalizing handoff text.
3. Record the return condition that hands control back to this readiness tracker after downstream coordination completes.
4. If workflow or Cortex evidence is unavailable for a downstream tracker, mark the step `[BLOCKED]` and escalate to `00-helping`.

**Stop conditions:**

- **Done:** downstream tracker targets, MCP snapshot evidence, and return conditions are recorded in this tracker.
- **Hold:** no downstream tracker needs immediate coordination for the selected first cycle.
- **Blocked:** workflow snapshot or routing evidence is unavailable for a required downstream tracker.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

[DONE] Cross-plan handoff packet recorded. MCP snapshot evidence, downstream tracker targets, and return conditions are below.

---

#### Cross-plan handoff packet — Phase 2 / Team-level fitness core evaluator

**MCP snapshot evidence (queried before finalizing this packet):**

| Tracker                                              | MCP scope                            | Active phase/step                                                                                              | Agent            |
| ---------------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ---------------- |
| `plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` | `no-active-phase`                    | Step 05 is now [DONE]; Phase 2 has not been promoted to [WIP] yet, so direct plan-file context is the fallback | `01-planning`    |
| `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`    | `repo-static`                        | Phase 1 Step 02 [WIP]                                                                                          | `02-researching` |
| `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`         | `no-active-phase`                    | [PLANNED] — no WIP marker                                                                                      | —                |
| `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`    | not queried — NOT a Phase 2 consumer | [PLANNED]                                                                                                      | —                |

**Downstream trackers directly touched by Phase 2 (team/group fitness aggregation seam):**

Step 02 readiness matrix assigns team/group fitness aggregation seam → **missing**, owner: policy decision, depends: `racing, ant-hive`. Predator/Prey is not a direct consumer of this primitive.

1. **Racing** — `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
   - Current step: Phase 1 Step 02 [WIP] — "Research boundary mapping"
   - The step objective is to classify every missing prerequisite as demo-local, reusable library/API, upstream NGE core, MCP tooling, or user-policy decision.
   - Team-level fitness is an upstream NGE core gap that Racing Step 02 cannot resolve locally. Racing must not advance to Step 03 (red tests for worker runtime and coevolution contracts) before the Phase 2 evaluator exists.
   - **Handoff target**: Racing Phase 1 Step 02. When Phase 2 closes, Racing Step 02 should record the evaluator as a resolved prerequisite and advance to Step 03.

2. **Ant Hive** — `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md`
   - MCP snapshot: `no-active-phase` — fully [PLANNED], no WIP marker.
   - Shares the team-level fitness dependency (Step 02 matrix). Cannot start implementation until the Phase 2 evaluator is available.
   - **Handoff target**: none yet. Ant Hive remains [PLANNED] until Phase 2 closes and Racing has proved the evaluator e2e. At that point, Ant Hive Phase 1 Step 01 may be promoted to [WIP].

3. **Predator/Prey** — `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`
   - Not a direct consumer of team-level fitness (Step 02 matrix: `racing, ant-hive` only).
   - **No Phase 2 handoff required.** Predator/Prey remains [PLANNED] and is not touched by the Phase 2 implementation cycle.

**Return condition (hands control back to this tracker):**

When Phase 2 Step 05 (Green validation) completes with passing focused tests and a clean `validate-plan-sync` result, control returns here for Phase 3 promotion. The explicit trigger is:

> Racing Phase 1 Step 02 records the team-level fitness evaluator as a resolved upstream prerequisite AND advances to Step 03; this confirms that the Phase 2 evaluator is live and benchmark-consumable. Phase 3 (independent populations and generation barriers) is then eligible for promotion to [WIP] in this tracker.

Until that confirmation exists, Phase 3 remains [PLANNED] and no unresolved Phase 3–6 gaps are implied to be solved.

**Stop condition met:** downstream tracker targets, MCP snapshot evidence, and return conditions are recorded. Phase 2 is the confirmed next active step.

---

### Phase 2 — Team-level fitness core evaluator [DONE]

[DONE] Phase 2 compressed: NGE core now owns the reusable team/group
aggregation seam via `createTeamFitnessEvaluator`, Racing consumes that seam
through an injected policy instead of benchmark-local aggregation logic, and
the touched NGE collective docs/readme now describe the contract without
claiming later barrier or transport work is complete.

**Durable coverage notes:**

- **Scope frozen:** planning and boundary mapping kept team/group fitness in
  `src/neat/nge-collective` as the reusable core owner boundary, with Racing
  and Ant Hive as consumers only and generation-barrier / deterministic
  transport work left for later phases.
- **Red-to-green delivery landed:** owner-local tests proved the missing
  evaluator/export seam; implementation added
  `src/neat/nge-collective/neat.nge-collective.team-fitness.ts`, updated the
  barrel/types/tests, and routed
  `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts`
  through the shared evaluator.
- **Docs aligned honestly:** `src/neat/nge-collective/README.md` was regenerated
  from source-first JSDoc and explicitly keeps Phase 3+ generation barriers and
  Phase 4 deterministic transport work outside the landed contract.

**Durable validation evidence:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
  passed after the Phase 2 compression edit.
- The implementation phase already recorded green validation for `npm run
build`, `npm run quality:folder -- --folder=src/neat/nge-collective`, `npm
run quality:folder -- --folder=examples/racing_curriculum/workers/simulation-worker`,
  the focused Jest slices, and `npm run test:silent` with repo-wide 100%
  `src/` coverage.
- No gate exceptions were recorded across the Phase 2 implementation,
  validation, or documentation passes.

**Remaining queue state and Phase 3 readiness:**

- **Phase 3 stays the explicit next phase:** independent populations and
  generation barriers remain unfinished core work and are not relabeled as
  deferred.
- **Phase 3 remains `[PLANNED]` after Phase 2 closure:** its planning packet is
  the next frontier once the next session promotes that boundary intentionally.
- **Downstream readiness remains visible:** Racing is the first proven consumer
  of the shared evaluator, Ant Hive remains queued behind the same reusable
  seam, and Phase 4–7 stay queued behind the Phase 3 barrier boundary rather
  than being implied complete.

### Phase 3 — Independent populations and generation barriers [DONE]

[DONE] Phase 3 compressed: reusable independent-population harness and generation-barrier semantics implemented in NGE core (`src/neat/nge-collective/neat.nge-collective.two-population.ts`), validated with 10/10 targeted tests + 89/89 nge-collective regression tests passing, transport-neutral barrier contract documented, deterministic transport normalization deferred to Phase 4.

**Durable coverage notes:**

- **Scope frozen:** planning kept independent-population semantics and transport-neutral generation-barrier semantics in NGE core; Racing and Predator/Prey treated as downstream consumers only.
- **Implementation boundary:** `createTwoPopulationHarness`, `runTwoTeamEvaluationTick`, `advanceTwoPopulations` in `src/neat/nge-collective/neat.nge-collective.two-population.ts`.
- **Test evidence:** 10/10 two-population tests green (barrier semantics, snapshot cross-registration); 89/89 nge-collective regression tests passing.
- **Gate evidence:** plan-sync PASS, agent-graph PASS (61 agents, 0 issues), step-packet PASS, phase-compression PASS, stale-wip-plans PASS.
- **Deferred to Phase 4:** deterministic evaluation-pack normalization, packed `race-step` transport, transfer-list rules, replay guarantees.
- **Downstream consumers:** Racing (first proving ground), Predator/Prey (second consumer with stronger synchronized-two-population pressure).

### Phase 4 — Deterministic evaluation packs / deterministic race packs normalization [DONE]

[DONE] Phase 4: Deterministic evaluation-pack normalization implemented and validated.

**Durable coverage notes:**

- **Scope frozen:** Froze Layer 1/2/3 ownership: NGE core Layer 1 owns transport-neutral collective evaluation and generation barriers; NGE core Layer 2 owns the generic deterministic pack contract (`createDeterministicEvaluationPack`, `resolveTransferList`, `assertSchemaVersion`); Racing Layer 3 owns `RacingRenderFrame` assembly, per-tier construction, and track physics; Racing stays the first proving ground, not the contract owner.
- **Implementation boundary:** `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` (Layer 2 core); `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts` (Layer 3 racing wrapper).
- **Test evidence:** 8/8 core pack tests green (`network.evaluation-pack.test.ts`); `network.evaluation-pack.ts` at 100% statements / branches / functions / lines. 7/7 racing normalizer tests green (`simulation-worker.evaluation-pack.normalizer.test.ts`).
- **Docs evidence:** Source JSDoc updated with bounded same-runtime determinism, generic reproducibility tuple (`seed, agentCount, schemaVersion`), racing reproducibility tuple (`seed, agentCount, packSchemaVersion, opponentSnapshot, trackId, featureFlags`), citations, and runnable examples; generated `src/architecture/network/evaluation-pack/README.md` regenerated via `npm run docs:folders:src`.
- **Gate evidence:** plan-sync PASS; step-packet PASS; agent-graph PASS (65 agents, 0 issues); phase-compression PASS; log-completion-marker PASS; stale-wip-plans PASS; delegate-skill-coverage PASS; chrome-devtools-mcp-coverage PASS (browser-determinism audit skipped because no browser render-loop/transfer-list surface was touched in this phase).
- **Deferred to Phase 5:** lifecycle staging closure and `nge-adult` readiness reconciliation.
- **Downstream consumers:** Racing (first proving ground), Predator/Prey and Ant Hive (will reuse the generic Layer 2 contract once their own worker seams are in scope).

### Phase 5 — Lifecycle staging closure and nge-adult readiness reconciliation [DONE]

**Closed:** 2026-06-23

**Scope:** Close lifecycle staging gaps, reconcile `nge-adult` readiness with file-backed evidence, and resolve contradictions with archived closure claims.

**Implementation boundary:**

- `src/neat/nge-adult/neat.nge-adult.cooling.ts` ? runtime growth-cooling decision.
- `src/neat/nge-adult/neat.nge-adult.utils.ts` ? adult-state seeding helpers.
- `src/neat/neat.nge-lifecycle.ts` ? lifecycle staging runner.
- `src/neat/nge-juvenile/neat.nge-juvenile.ts` ? juvenile root orchestrator.

**Validation evidence:**

- Phase/step packet validation PASS (`validate-plan-sync.mjs`, `validate-plan-phase-packets.mjs`).
- Red tests reproduced hardcoded `growthCoolingActive: true` and missing lifecycle runner (Step 03).
- Green validation: 164/164 focused lifecycle tests pass; touched `src/` files at 100% statements/branches/functions/lines.
- `npm run lint` exit 0; `npx tsc --noEmit -p tsconfig.json` exit 0.
- `npm run docs` regenerated READMEs; `docs-quality-metrics.gate.mjs` PASS; `cortex-index` gate PASS.
- plan-sync, phase-compression, log-completion-marker, stale-wip-plans, delegate-skill-coverage gates PASS.

**Decisions:**

- Runtime cooling decision uses focus floor; stale placeholder JSDoc corrected in adult cooling and utils.
- `runNgeLifecycle` sequences juvenile ? assimilation ? adult stages.
- Juvenile root orchestrator re-exports owner-local helpers.
- C5 snapshot-vs-emission semantic gap documented as experimental caveat; full resolution deferred to assimilation-wiring follow-up.

**Risks:**

- C5 adult-state snapshot semantics remain unresolved beyond documented caveat.
- Lifecycle integration beyond the staged runner is not yet exercised by a downstream benchmark.

**Next resume point:** Phase 6 ? Experimental root public API exposure [WIP].

### Phase 6 — Experimental root public API exposure [DONE]

[DONE] Phase 6: Experimental NGE namespace exposed from `src/neataptic.ts` via a narrow `nge` re-export barrel (`src/neat/nge-experimental.ts`). Four sub-namespaces (`adult`, `juvenile`, `lifecycle`, `assimilation`) are reachable, red-to-green tests pass, and docs/build/lint/typecheck gates are green. See `plans/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md` for detailed step evidence.

**Durable coverage notes:**

- Scope frozen: single `nge` namespace export; no top-level NGE names; no Phase G/E motifs; no stability guarantees.
- Implementation boundary: `src/neat/nge-experimental.ts` barrel; `src/neataptic.ts` re-export line; `testing/neataptic.nge-experimental.test.ts`.
- Validation: 9/9 focused tests pass; `src/neat/nge-experimental.ts` 100% coverage; build/tsc/lint/prettier green; plan-sync, plan-phase-packets, step-packet, agent-graph, chrome-devtools-mcp-coverage, delegate-skill-coverage, cortex-index gates PASS.
- Browser audit: skipped because no browser demo currently imports the root `nge` namespace.
- Docs: source JSDoc updated in `src/neataptic.ts` and `src/neat/nge-experimental.ts`; `npm run docs` regenerated `src/README.md` and `src/neat/README.md`; no generated READMEs edited directly.

**Next resume point:** Phase 7 — Downstream benchmark dependency + MCP synchronization [WIP].

### Phase 7 — Downstream benchmark dependency + MCP synchronization [DONE]

[DONE] Phase 7 compressed — cross-plan synchronization scope recorded for Racing, Predator/Prey, and Ant Hive; workflow tooling now emits downstreamTrackers; focused Jest contract tests pass; documentation and handoff alignment complete. Full details moved to plans/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md.

## Final state

All phases complete:
- Phase 1 — Readiness audit, matrix preservation, and phase packetization [DONE]
- Phase 2 — Team-level fitness core evaluator [DONE]
- Phase 3 — Independent populations and generation barriers [DONE]
- Phase 4 — Deterministic evaluation packs / deterministic race packs normalization [DONE]
- Phase 5 — Lifecycle staging closure and nge-adult readiness reconciliation [DONE]
- Phase 6 — Experimental root public API exposure [DONE]
- Phase 7 — Downstream benchmark dependency + MCP synchronization [DONE]

The NGE core readiness audit is closed. Downstream benchmark trackers (Racing, Predator/Prey, Ant Hive) have been handed the completed core seams with explicit return conditions.

## Audit summary

- Core primitives closed across Phases 2-6 with file-backed evidence and gate validation.
- Phase 7 added MCP-aware cross-plan synchronization: downstream tracker extraction, downstreamTrackers emission in workflow tooling, synchronization contract tests, and documentation alignment.
- Browser validation performed for Racing demo; Ant-Hive and Predator/Prey demos skipped due to missing pages.
- Two downstream tracker status discrepancies held as HOLD items for respective benchmark owners.

## Reopen conditions

Reopen this plan if:
- A downstream benchmark discovers a new NGE core readiness gap that cannot be solved in the downstream tracker.
- Phase 0/E core primitives (computation motifs, modeIsEvolvable, polyandric support) need explicit readiness closure before Ant Hive or Predator/Prey can advance.
- The experimental `nge` public API needs stabilization beyond the current experimental namespace.

## Audit log

See plans/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md for per-phase durable done-state records, validation evidence, decisions, and risks.
