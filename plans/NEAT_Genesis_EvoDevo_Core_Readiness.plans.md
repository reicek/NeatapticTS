# NEAT Genesis EvoDevo: Core Readiness Audit

**Status:** [WIP] (Phase 4 [DONE]; Phase 5 [DONE]; Phase 6 ? Experimental root public API exposure [WIP]; Phase 6 Step 01 [WIP])

Claim: 04-implementing @ 2026-06-22T20:19:59-04:00

```yaml
PlanUpdate:
  slice_id: 04-impl
  changed_files:
    - src/neat/nge-adult/neat.nge-adult.cooling.ts
    - src/neat/nge-adult/neat.nge-adult.utils.ts
    - src/neat/neat.nge-lifecycle.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.ts
    - src/neat/nge-assimilation/neat.nge-assimilation.ts
    - src/neat/nge-adult/neat.nge-adult.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check <touched-files>'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns="src/neat/(nge-adult/neat.nge-adult|neat.nge-lifecycle|nge-assimilation/neat.nge-assimilation)"'
      expected_exit: 0
  rollback:
    - 'git checkout -- src/neat/nge-adult/neat.nge-adult.cooling.ts src/neat/nge-adult/neat.nge-adult.utils.ts src/neat/neat.nge-lifecycle.ts src/neat/nge-juvenile/neat.nge-juvenile.ts src/neat/nge-adult/neat.nge-adult.test.ts'
    - 'rm src/neat/neat.nge-lifecycle.ts src/neat/nge-juvenile/neat.nge-juvenile.ts'
  next: 'Run 05-green-testing slice 04-green and attach coverage-guard evidence.'
```

```yaml
PlanUpdate:
  slice_id: 04-green
  validator: 05-green-testing
  changed_files:
    - src/neat/nge-adult/neat.nge-adult.cooling.ts
    - src/neat/nge-adult/neat.nge-adult.utils.ts
    - src/neat/neat.nge-lifecycle.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.ts
    - src/neat/nge-adult/neat.nge-adult.cooling.test.ts
    - src/neat/neat.nge-lifecycle.test.ts
    - src/neat/nge-juvenile/neat.nge-juvenile.test.ts
  validation:
    - gate: plan-sync
      command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
      result: pass
    - command: 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns="src/neat/(nge-adult/neat.nge-adult|neat.nge-lifecycle|nge-juvenile/neat.nge-juvenile)"'
      result: pass
      suites: '4 passed, 4 total'
      tests: '164 passed, 164 total'
    - command: 'npm run lint'
      result: pass
    - gate: delegate-skill-coverage
      command: 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
      result: pass
  coverage_guard:
    src/neat/neat.nge-lifecycle.ts: { statements: 100, branches: 100, functions: 100, lines: 100 }
    src/neat/nge-adult/neat.nge-adult.cooling.ts: { statements: 100, branches: 100, functions: 100, lines: 100 }
    src/neat/nge-adult/neat.nge-adult.utils.ts: { statements: 100, branches: 100, functions: 100, lines: 100 }
    src/neat/nge-juvenile/neat.nge-juvenile.ts: { statements: 100, branches: 100, functions: 100, lines: 100 }
  status: '[DONE]'
  next: 'Advance to Step 05 — Green validation and lifecycle audit.'
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

### Phase 5 ? Lifecycle staging closure and nge-adult readiness reconciliation [DONE]

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

### Phase 6 — Experimental root public API exposure [WIP]

```yaml
phase: 6
title: 'Experimental root public API exposure'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_phase: 'Downstream benchmark dependency + MCP synchronization'
skills:
  - plan-alignment
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
placeholder_steps:
  - 'Step 01 — Planning packet and export-surface freeze'
  - 'Step 02 — Public-surface and dependency mapping'
  - 'Step 03 — Red tests for experimental root API exposure'
  - 'Step 04 — Implement experimental root API exposure'
  - 'Step 05 — Green validation and export-surface audit'
  - 'Step 06 — Documentation and API usage alignment'
  - 'Step 07 — Logging and phase closure packet'
```

**Phase objective:**

Expose a narrow experimental public entrypoint for NGE surfaces from src/neataptic.ts without exporting every internal module at once.

#### Step 01 — Planning packet and export-surface freeze [WIP]

```yaml
phase: 6
step: 1
title: 'Planning packet and export-surface freeze'
status: '[WIP]'
goal: planning
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 02 — Public-surface and dependency mapping'
skills:
  - 'plan-alignment, phase-handoff-workflow, nge-core-algorithm, execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Freeze the narrow experimental public API shape and the migration constraints around `src/neataptic.ts`.

**Execution focus:** Preserve the existing caution that a narrow experimental namespace may be safer than broad exports, and record which downstream demos/tests depend on that surface.

**Stop conditions:** Done when the export contract and non-goals are explicit; hold on API-shape ambiguity; blocked if lifecycle closure still changes required exports.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 02 — Public-surface and dependency mapping [PLANNED]

```yaml
phase: 6
step: 2
title: 'Public-surface and dependency mapping'
status: '[PLANNED]'
goal: researching
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 03 — Red tests for experimental root API exposure'
skills:
  - 'nge-core-algorithm, repo-cortex-workflow, execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Map the missing root export seam and every downstream consumer that should use the experimental entrypoint.

**Execution focus:** Use Repo Cortex to inspect `src/neataptic.ts`, current NGE imports, and dependent tests/demos; keep the boundary narrow and explicit.

**Stop conditions:** Done when export and consumer seams are recorded; hold on unresolved API naming; blocked on missing workflow or Cortex evidence.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
- `node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json`

#### Step 03 — Red tests for experimental root API exposure [PLANNED]

```yaml
phase: 6
step: 3
title: 'Red tests for experimental root API exposure'
status: '[PLANNED]'
goal: red-testing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 04 — Implement experimental root API exposure'
skills:
  - 'red-test-contracts, creating-unit-tests, execute, chrome-devtools-mcp'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=chrome-devtools-mcp-coverage --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'Red tests fail for the expected export-surface gap (not syntax/fixture errors).'
  - 'If a browser demo consumes the root API, red tests include a Chrome DevTools MCP specialist-delegated browser check that fails before the export is added.'
slices:
  - slice_id: step-3-red-tests
    title: 'Write red tests'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 4
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Red tests exist and fail for the expected behavior.'
    parallelizable: false
    dependencies:
    next_slice: step-3-core
  - slice_id: step-3-core
    title: 'Implement the core behavior'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 8
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Implementation satisfies the red tests and design.'
    parallelizable: false
    dependencies:
      - step-3-red-tests
    next_slice: step-3-green
  - slice_id: step-3-green
    title: 'Green validation and coverage guard'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - step-3-core
```

**User instruction:**

Paste this full step packet.

**Step objective:** Add the smallest failing tests that prove the experimental NGE root export surface is missing or incomplete.

**Execution focus:** Protect the intended namespace shape and consumer import path without widening into benchmark implementation work.

**Stop conditions:** Done when the missing export contract is reproduced in red; hold on unresolved naming; blocked if dependency mapping is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 04 — Implement experimental root API exposure [PLANNED]

```yaml
phase: 6
step: 4
title: 'Implement experimental root API exposure'
status: '[PLANNED]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 05 — Green validation and export-surface audit'
skills:
  - nge-core-algorithm
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-4-red-tests
    title: 'Write red tests'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 4
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Red tests exist and fail for the expected behavior.'
    parallelizable: false
    dependencies:
    next_slice: step-4-core
  - slice_id: step-4-core
    title: 'Implement the core behavior'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 8
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Implementation satisfies the red tests and design.'
    parallelizable: false
    dependencies:
      - step-4-red-tests
    next_slice: step-4-green
  - slice_id: step-4-green
    title: 'Green validation and coverage guard'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - step-4-core
```

**User instruction:**

Paste this full step packet.

**Step objective:** Implement the agreed experimental root NGE export surface in `src/neataptic.ts` and any required narrow supporting exports.

**Execution focus:** Keep the change experimental and bounded, do not over-export unfinished internals, and preserve downstream tracker dependencies explicitly.

**Stop conditions:** Done when red export tests turn green; hold on unresolved API naming; blocked on conflicting contract expectations.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 05 — Green validation and export-surface audit [PLANNED]

```yaml
phase: 6
step: 5
title: 'Green validation and export-surface audit'
status: '[PLANNED]'
goal: green-testing
tdd_sequence: green-only
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 06 — Documentation and API usage alignment'
skills:
  - green-validation-gates
  - execute
  - chrome-devtools-mcp
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=chrome-devtools-mcp-coverage --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'Focused Jest suites pass; coverage guard passes on touched src/ files.'
  - 'If a browser demo consumes the new root API, Chrome DevTools MCP specialist delegation (browser-ui-specialist) confirms the import path resolves and the demo renders without console errors.'
slices:
  - slice_id: step-5-impl-verify
    title: 'Confirm Step 04 export-surface implementation is stable and all prior red tests are green'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
    acceptance_criteria:
      - 'All Step 04 red tests now pass green.'
      - 'No regressions in export-surface tests.'
    parallelizable: false
    dependencies:
    next_slice: step-5-green-jest
  - slice_id: step-5-green-jest
    title: 'Green Jest validation and coverage guard'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All targeted Jest suites pass.'
      - '100% coverage on touched src/ files.'
    parallelizable: false
    dependencies:
      - step-5-impl-verify
    next_slice: step-5-browser-export
  - slice_id: step-5-browser-export
    title: 'Browser export-surface audit via Chrome DevTools MCP'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
    acceptance_criteria:
      - 'If a browser demo consumes the experimental root API, delegate browser-ui-specialist to verify the import path resolves in the browser and the demo renders without console errors.'
      - 'If export-surface API overhead matters, delegate performance-trace-specialist to capture a trace and confirm the import does not add measurable overhead.'
      - 'If no browser demo consumes this export yet, record an explicit skip note and exit.'
    parallelizable: false
    dependencies:
      - step-5-green-jest
```

**User instruction:**

Paste this full step packet.

**Step objective:** Verify the new experimental export surface works for focused consumer slices and does not regress unrelated public API behavior.

**Stop conditions:** Done when focused validation passes; hold on flaky consumer behavior; blocked if implementation remains red.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 06 — Documentation and API usage alignment [PLANNED]

```yaml
phase: 6
step: 6
title: 'Documentation and API usage alignment'
status: '[PLANNED]'
goal: documenting
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 07 — Logging and phase closure packet'
skills:
  - 'educational-docs, execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Align docs and examples with the experimental root entrypoint without teaching broader export support than the code actually provides.

**Stop conditions:** Done when touched docs match the experimental contract; hold on unresolved wording; blocked if validation evidence is missing.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 07 — Logging and phase closure packet [PLANNED]

```yaml
phase: 6
step: 7
title: 'Logging and phase closure packet'
status: '[PLANNED]'
goal: logging
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Phase 7 — Downstream benchmark dependency + MCP synchronization'
skills:
  - tracker-handoff
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Compress export-surface coverage and keep downstream benchmark/MCP synchronization visible as the last named readiness phase.

**Stop conditions:** Done when the tracker records what landed and what downstream coordination remains; hold on pending user confirmation; blocked if evidence is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

### Phase 7 — Downstream benchmark dependency + MCP synchronization [PLANNED]

```yaml
phase: 7
title: 'Downstream benchmark dependency + MCP synchronization'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_phase: null
skills:
  - plan-alignment
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
placeholder_steps:
  - 'Step 01 — Planning packet and cross-plan synchronization scope'
  - 'Step 02 — Downstream tracker and MCP seam mapping'
  - 'Step 03 — Red tests for synchronization contracts'
  - 'Step 04 — Implement synchronization hooks and tracker updates'
  - 'Step 05 — Green validation and MCP audit'
  - 'Step 06 — Documentation and handoff alignment'
  - 'Step 07 — Logging and phase closure packet'
```

**Phase objective:**

Synchronize downstream benchmark dependencies and hand completed core evidence back to Racing, Predator/Prey, and Ant Hive trackers via MCP-aware cross-plan packets.

#### Step 01 — Planning packet and cross-plan synchronization scope [PLANNED]

```yaml
phase: 7
step: 1
title: 'Planning packet and cross-plan synchronization scope'
status: '[PLANNED]'
goal: planning
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 02 — Downstream tracker and MCP seam mapping'
skills:
  - 'plan-alignment, phase-handoff-workflow, repo-cortex-workflow, execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Freeze the downstream tracker, MCP, and return-condition scope needed to synchronize Racing, Predator/Prey, and Ant Hive with completed core readiness work.

**Execution focus:** Record exact target trackers, target steps, return conditions, and the rule that no downstream closure can imply an unfinished core gap is solved.

**Stop conditions:** Done when synchronization scope is explicit; hold on downstream-priority ambiguity; blocked if prior phases still lack closure evidence.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 02 — Downstream tracker and MCP seam mapping [PLANNED]

```yaml
phase: 7
step: 2
title: 'Downstream tracker and MCP seam mapping'
status: '[PLANNED]'
goal: researching
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 03 — Red tests for synchronization contracts'
skills:
  - 'repo-cortex-workflow, nge-benchmark-workflow, execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Re-read workflow snapshots for this tracker and the downstream benchmark trackers, then map the exact MCP synchronization seams and dependency gaps.

**Execution focus:** Use Repo Cortex and workflow snapshots to confirm downstream tracker state, benchmark dependency points, and any remaining MCP or routing evidence gaps that must be fixed before work leaves this plan.

**Stop conditions:** Done when downstream seam mapping and return conditions are recorded; hold on ambiguous tracker ownership; blocked on missing workflow or Cortex evidence.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
- `node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json`

#### Step 03 — Red tests for synchronization contracts [PLANNED]

```yaml
phase: 7
step: 3
title: 'Red tests for synchronization contracts'
status: '[PLANNED]'
goal: red-testing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 04 — Implement synchronization hooks and tracker updates'
skills:
  - 'red-test-contracts, execute, chrome-devtools-mcp'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=chrome-devtools-mcp-coverage --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'Red tests fail for the expected synchronization contract gap (not syntax/fixture errors).'
  - 'If synchronization behavior is observable in a browser benchmark, red tests include a Chrome DevTools MCP specialist-delegated browser check that fails before synchronization is implemented.'
slices:
  - slice_id: step-3-red-tests
    title: 'Write red tests'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 4
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Red tests exist and fail for the expected behavior.'
    parallelizable: false
    dependencies:
    next_slice: step-3-core
  - slice_id: step-3-core
    title: 'Implement the core behavior'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 8
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Implementation satisfies the red tests and design.'
    parallelizable: false
    dependencies:
      - step-3-red-tests
    next_slice: step-3-green
  - slice_id: step-3-green
    title: 'Green validation and coverage guard'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - step-3-core
```

**User instruction:**

Paste this full step packet.

**Step objective:** Add the smallest failing tests or checks that prove the downstream synchronization contract is still missing.

**Execution focus:** Keep the red boundary on tracker synchronization, MCP wiring, and dependency handoff semantics; do not reopen resolved core implementation work inside this step.

**Stop conditions:** Done when missing synchronization behavior fails under focused checks; hold on unresolved tracker policy; blocked if Step 02 mapping is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 04 — Implement synchronization hooks and tracker updates [PLANNED]

```yaml
phase: 7
step: 4
title: 'Implement synchronization hooks and tracker updates'
status: '[PLANNED]'
goal: implementing
tdd_sequence: red-green
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 05 — Green validation and MCP audit'
skills:
  - repo-cortex-workflow
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-4-red-tests
    title: 'Write red tests'
    status: '[PLANNED]'
    goal: red-testing
    estimate_hours: 4
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Red tests exist and fail for the expected behavior.'
    parallelizable: false
    dependencies:
    next_slice: step-4-core
  - slice_id: step-4-core
    title: 'Implement the core behavior'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 8
    files_to_change:
      - TBD
    acceptance_criteria:
      - 'Implementation satisfies the red tests and design.'
    parallelizable: false
    dependencies:
      - step-4-red-tests
    next_slice: step-4-green
  - slice_id: step-4-green
    title: 'Green validation and coverage guard'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - step-4-core
```

**User instruction:**

Paste this full step packet.

**Step objective:** Implement the synchronization work needed to keep downstream benchmark trackers and mandatory MCP flow aligned with this readiness tracker.

**Execution focus:** Keep changes bounded to tracker state, MCP wiring, and dependency handoff surfaces; route any new workflow infrastructure gaps to `00-helping`.

**Stop conditions:** Done when the selected synchronization seam is implemented and focused red checks turn green; hold on new workflow-policy questions; blocked on MCP/tooling gaps.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 05 — Green validation and MCP audit [PLANNED]

```yaml
phase: 7
step: 5
title: 'Green validation and MCP audit'
status: '[PLANNED]'
goal: green-testing
tdd_sequence: green-only
expansion: slices
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 06 — Documentation and handoff alignment'
skills:
  - 'green-validation-gates, repo-cortex-workflow, execute, chrome-devtools-mcp'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=chrome-devtools-mcp-coverage --json'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
  - 'Focused Jest suites pass; coverage guard passes on touched src/ files.'
  - 'Chrome DevTools MCP specialist delegation REQUIRED for browser benchmark validation: performance-trace-specialist captures traces for racing/ant-hive/predator-prey benchmark render loops and confirms no unacceptable frame drops or jank.'
  - 'browser-memory-specialist captures heap snapshots for long-running benchmark sessions and confirms no unbounded memory growth or leak pattern.'
  - 'browser-ui-specialist verifies benchmark visualizers render correctly (DOM structure, canvas state, no console errors) after synchronization hooks are applied.'
slices:
  - slice_id: step-5-impl-verify
    title: 'Confirm Step 04 synchronization implementation is stable and all prior red tests are green'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
    acceptance_criteria:
      - 'All Step 04 red tests now pass green.'
      - 'No regressions in synchronization contract tests.'
    parallelizable: false
    dependencies:
    next_slice: step-5-green-jest
  - slice_id: step-5-green-jest
    title: 'Green Jest validation and coverage guard'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All targeted Jest suites pass.'
      - '100% coverage on touched src/ files.'
    parallelizable: false
    dependencies:
      - step-5-impl-verify
    next_slice: step-5-browser-perf
  - slice_id: step-5-browser-perf
    title: 'Browser benchmark performance trace audit via Chrome DevTools MCP'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 5
    files_to_change:
    acceptance_criteria:
      - 'Delegate performance-trace-specialist to capture Chrome DevTools performance traces for racing, ant-hive, and predator-prey benchmark render loops.'
      - 'Confirm no unacceptable frame drops (>5 consecutive dropped frames) or excessive layout thrashing in the benchmark render loop.'
      - 'Save traces to tmp/traces/ for archival evidence.'
    parallelizable: false
    dependencies:
      - step-5-green-jest
    next_slice: step-5-browser-memory
  - slice_id: step-5-browser-memory
    title: 'Browser benchmark memory leak audit via Chrome DevTools MCP'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
    acceptance_criteria:
      - 'Delegate browser-memory-specialist to capture heap snapshots before and after a long-running benchmark session (>=100 generations or equivalent).'
      - 'Confirm no unbounded memory growth pattern (retained objects should stabilize, not grow linearly with generations).'
      - 'Classify any retained object growth as leak or expected and record the classification.'
    parallelizable: false
    dependencies:
      - step-5-browser-perf
    next_slice: step-5-browser-ui
  - slice_id: step-5-browser-ui
    title: 'Browser benchmark UI validation via Chrome DevTools MCP'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 3
    files_to_change:
    acceptance_criteria:
      - 'Delegate browser-ui-specialist to verify benchmark visualizers render correctly: DOM structure, canvas state, no console errors.'
      - 'Verify benchmark controls (start/stop/reset) are accessible and functional after synchronization hooks are applied.'
      - 'If a benchmark demo is not browser-rendered, record an explicit skip note and exit.'
    parallelizable: false
    dependencies:
      - step-5-browser-memory
```

**User instruction:**

Paste this full step packet.

**Step objective:** Verify the synchronization implementation and MCP wiring stay green and report accurate evidence. This is the prime Chrome DevTools MCP validation step: browser benchmarks (racing, ant-hive, predator-prey) must be validated via specialist-delegated performance traces, memory profiling, and UI checks.

**Stop conditions:** Done when focused synchronization validation passes; hold on transient MCP failures; blocked if the implementation or gate evidence remains red.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
- `node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json`

#### Step 06 — Documentation and handoff alignment [PLANNED]

```yaml
phase: 7
step: 6
title: 'Documentation and handoff alignment'
status: '[PLANNED]'
goal: documenting
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'Step 07 — Logging and phase closure packet'
skills:
  - 'tracker-handoff, repo-cortex-workflow, execute'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Refresh tracker-facing docs and handoff language so downstream benchmark synchronization is explicit, current, and MCP-aware.

**Stop conditions:** Done when docs and handoff text align to file-backed workflow behavior; hold on unresolved wording; blocked if validation evidence is missing.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 07 — Logging and phase closure packet [PLANNED]

```yaml
phase: 7
step: 7
title: 'Logging and phase closure packet'
status: '[PLANNED]'
goal: logging
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_step: 'User confirmation before any named phase closes'
skills:
  - tracker-handoff
  - execute
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**User instruction:**

Paste this full step packet.

**Step objective:** Compress the final synchronization phase history while preserving the invariant that nothing closes until file-backed evidence exists and the user confirms completion.

**Stop conditions:** Done when the tracker records durable synchronization coverage and explicit remaining return conditions; hold on missing user confirmation; blocked if evidence is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

## Validation gates

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
- `neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json`
- `neataptic-gate-mcp:run_gate_check --gate=chrome-devtools-mcp-coverage --json`

### Latest validation evidence
- 2026-06-23: Phase 5 closed ? compression and logging complete; Phase 5 [DONE], Phase 6 promoted to [WIP] / Step 01 active; plan-sync, phase-compression, log-completion-marker, stale-wip-plans, and delegate-skill-coverage gates PASS.

- 2026-06-23: Workflow sync: Advanced Phase 5 Step 5 → [DONE]; Phase 5 Step 6 → [WIP]
- 2026-06-23: Workflow sync: Advanced Phase 5 Step 3 → [DONE]; Phase 5 Step 4 → [WIP]
- 2026-06-23: Workflow sync: Advanced Phase 5 Step 1 → [DONE]; Phase 5 Step 2 → [WIP]
- 2026-06-22: Phase 4 closed — compression and logging complete; phase-compression, log-completion-marker, stale-wip-plans, plan-sync, delegate-skill-coverage gates PASS; Phase 5 remains [PLANNED] as the explicit next phase.
- 2026-06-23: Phase 5 kickoff — promoted Phase 5 to [WIP] and Step 01 to [WIP]; reconciled archived lifecycle closure claims against `src/neat/nge-adult/*` file-backed evidence; authored machine-actionable Step 02-07 packets with valid YAML skill lists and TDD-conformant slices; `validate-plan-sync`, `validate-plan-phase-packets`, `delegate-skill-coverage`, and `step-packet` gates PASS; `plans/README.md` and `plans/Roadmap.md` updated to Phase 5 [WIP] / Step 01 active. Workflow sync advanced Step 01 → [DONE] and Step 02 → [WIP].
- 2026-06-20: Orchestration-alignment refresh: added `execute` skill to all Phase 4-7 WIP/PLANNED step packets; added `chrome-devtools-mcp` skill to Phase 5/6/7 Step 03 (red-testing) and Step 05 (green-testing); fixed green-only Step 05 slices to remove incorrect red-tests/core placeholder slices and replace with green-validation + Chrome DevTools MCP specialist-delegated browser audit slices; Phase 7 Step 05 rewritten as prime Chrome DevTools MCP candidate with performance-trace-specialist, browser-memory-specialist, and browser-ui-specialist slices for racing/ant-hive/predator-prey benchmark validation.
- 2026-06-17: Workflow sync: Advanced Phase 4 Step 4 → [DONE]; Phase 4 Step 5 → [WIP]
- 2026-06-16: Workflow sync: Advanced Phase 4 Step 4 → [DONE]; Phase 4 Step 5 → [WIP]
- 2026-06-08: Workflow sync: Advanced Phase 3 Step 6 → [DONE]; Phase 3 Step 7 → [WIP]
- 2026-06-03: `validate-plan-sync` PASS after recording the Phase 3 Step 02 seam map, marking Step 02 `[DONE]`, promoting Step 03 `[WIP]`, and refreshing the handoff query for the red-test packet.
- 2026-06-03: `cortex-first-search.gate` PASS (`database_path: C:\NeatapticTS\data\semantic-index.sqlite`; `index_documents: 1396`; `index_chunks: 31090`; `index_fresh: true`; `corpus_mcp_alive: true`; `corpus_search_results: 3`) after the Step 02 seam map update.
- 2026-06-03: `validate-plan-sync` PASS after freezing Phase 3 Step 01 barrier scope, promoting Phase 3 to `[WIP]`, and moving Step 02 to the active seam-mapping packet.
- 2026-06-03: `validate-plan-sync` PASS after compressing Phase 2 into durable coverage notes, marking Phase 2 `[DONE]`, refreshing the handoff query, and keeping Phase 3 `[PLANNED]` as the explicit next phase.
- 2026-06-02: Workflow sync: Advanced Phase 2 Step 5 → [DONE]; Phase 2 Step 6 → [WIP]
- 2026-06-02: Workflow sync: Advanced Phase 2 Step 4 → [DONE]; Phase 2 Step 5 → [WIP]
- 2026-06-02: Workflow sync: Advanced Phase 2 Step 3 → [DONE]; Phase 2 Step 4 → [WIP]
- 2026-06-02: Workflow sync: Advanced Phase 2 Step 2 → [DONE]; Phase 2 Step 3 → [WIP]
- (first run — see below)

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Context: NGE core readiness tracker ? Phase 5 lifecycle staging closure is [DONE]; Phase 6 experimental root public API exposure is now [WIP].
Current boundary: Phase 6 Step 01 ? Planning packet and export-surface freeze.
What is already covered:
- Phase 5 Steps 01-07 complete: lifecycle contradiction freeze, seam mapping, red tests, implementation (runtime cooling decision + lifecycle runner + juvenile orchestrator), green validation (164/164 tests, 100% coverage), documentation/README regeneration, and phase compression/logging.
- Phase 4 deterministic evaluation-pack normalization is [DONE].
- C5 snapshot-vs-emission semantic gap is documented as experimental caveat; full resolution deferred to future assimilation-wiring work.

Next narrow task:
- Paste and execute the Phase 6 Step 01 planning packet.
- Freeze the experimental root public API exposure acceptance criteria and identify the narrowest NGE surface to export from `src/neataptic.ts`.
- Do not implement code changes yet; plan and author Step 02-07 packets before execution continues.

Required validations:
- node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
- node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
- neataptic-gate-mcp:run_gate_check --gate=delegate-skill-coverage --json

Known worktree cautions:
- Phase 6 must not export every internal NGE module at once; keep the experimental surface narrow and honest.
- Phase 7 downstream benchmark synchronization remains queued; do not imply downstream closures are solved by core readiness work.
```

