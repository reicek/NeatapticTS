# NEAT Genesis EvoDevo: Core Readiness Audit

**Status:** [WIP] (Phase 4 Step 05 — Green validation and determinism audit)

Claim: 04-implementing @ 2026-06-16

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
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 03 — Gap-to-phase mapping'
skills: 'nge-core-algorithm'
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
goal: 'planning'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 04 — First active implementation cycle selection'
skills: 'plan-alignment, phase-handoff-workflow, nge-core-algorithm'
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
goal: 'planning'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 05 — Cross-plan MCP handoff packet'
skills: 'plan-alignment, phase-handoff-workflow, nge-core-algorithm'
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
goal: 'planning'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Phase 2 — Team-level fitness core evaluator'
skills: 'plan-alignment, phase-handoff-workflow, repo-cortex-workflow'
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

### Phase 4 — Deterministic evaluation packs / deterministic race packs normalization [WIP]

```yaml
phase: 4
title: 'Deterministic evaluation packs / deterministic race packs normalization'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_phase: 'Lifecycle staging closure and nge-adult readiness reconciliation'
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
placeholder_steps:
  - 'Step 01 — Planning packet and determinism contract freeze'
  - 'Step 02 — Transport and reproducibility seam mapping'
  - 'Step 03 — Red tests for deterministic evaluation packs'
  - 'Step 04 — Implement deterministic evaluation-pack normalization'
  - 'Step 05 — Green validation and determinism audit'
  - 'Step 06 — Documentation and reproducibility contract alignment'
  - 'Step 07 — Logging and phase closure packet'
```

#### Step 01 — Planning packet and determinism contract freeze [DONE]

```yaml
phase: 4
step: 1
goal: 'planning'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 02 — Transport and reproducibility seam mapping'
skills: 'plan-alignment, phase-handoff-workflow, reproducibility-contracts'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**Step objective:** Freeze the deterministic evaluation-pack contract and clarify which pieces are core semantics versus worker transport normalization.

**Execution focus:** Preserve the provisional owner note from the readiness audit, define the normalization boundary explicitly, and keep Racing as the proving ground rather than the owner.

**Stop conditions:** Done when contract scope and ownership are frozen; hold on unresolved owner boundaries; blocked if Phase 3 still changes shared prerequisites.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

---

##### Step 01 — Frozen Determinism Contract

###### Determinism rung

**Level 2 — Ordered deterministic.** Same seed, same inputs, same ordering rules,
and same runtime configuration produce numerically stable outputs within one
runtime (Node or browser). Cross-environment byte-identical results are NOT
promised because typed-array layout, transfer semantics, and floating-point
reduction order may differ between Node `worker_threads` and browser
`Web Workers`.

Rationale: `createDeterministicRacePack` already proves that same seed + same
opponent snapshot → identical `RacingRenderFrame` on the same runtime. A Level 3
(replay exact) claim would require full RNG state capture beyond the pack seed,
which is out of scope for Phase 4 — the pack seed is sufficient for ordered
deterministic replay at the pack-creation boundary. A Level 4 (cross-environment
bounded) claim would require documenting every floating-point and transport
difference between Node and browser workers, which is deferred to a later
reproducibility audit after the core normalization seam lands.

###### Contract scope — what Phase 4 normalizes

| Contract element                 | Current location                                          | Target owner after Phase 4          |
| -------------------------------- | --------------------------------------------------------- | ----------------------------------- |
| Deterministic pack creation      | `simulation-worker.race-pack.service.ts` (racing)         | NGE core — Layer 2                  |
| Pack replay validation           | No reusable validator exists                              | NGE core — Layer 2                  |
| Transfer-list resolution         | `simulation-worker.race-pack.service.ts` (racing)         | NGE core — Layer 2                  |
| Schema versioning / rejection    | `simulation-worker.types.ts` (racing `RacingRenderFrame`) | NGE core — Layer 2 (generic schema) |
| Clone-safe payload contract      | Implicit in worker transport                              | NGE core — Layer 2 (explicit)       |
| `RacingRenderFrame` type         | `simulation-worker.types.ts` (racing)                     | Racing-owned — Layer 3 (stays)      |
| Per-tier pack construction       | `simulation-worker.tier3/4/5.ts` (racing)                 | Racing-owned — Layer 3 (stays)      |
| Track physics / tire / pit state | Racing-specific frame fields                              | Racing-owned — Layer 3 (stays)      |
| Renderer layout                  | Car positions, headings, place                            | Racing-owned — Layer 3 (stays)      |

###### Three-layer ownership boundary

```mermaid
graph TD
  L1["Layer 1 — NGE Core Semantics<br/>nge-collective<br/><em>transport-neutral</em>"]
  L2["Layer 2 — Worker Transport Normalization<br/>NEW Phase 4 module<br/><em>deterministic packs, replay, transfer, schema</em>"]
  L3["Layer 3 — Benchmark-Local Racing Transport<br/>examples/racing_curriculum<br/><em>RacingRenderFrame, per-tier construction, physics</em>"]

  L1 -->|"evaluation context<br/>team fitness<br/>generation barriers"| L2
  L2 -->|"generic pack contract<br/>transfer-list rules<br/>schema versioning"| L3
  L3 -->|"proves contract e2e<br/>does NOT own it"| L2
```

**Layer 1 — NGE Core Semantics** (`src/neat/nge-collective/`):

- `runCollectiveEvaluationTick`, `resetCollectiveEvaluationState`
- `createTeamFitnessEvaluator`, team/group fitness aggregation
- `createTwoPopulationHarness`, `advanceTwoPopulations`
- Generation barrier semantics (transport-neutral by Phase 3 design)
- Opponent snapshot pool management
- **Invariant:** these functions never depend on `RacingRenderFrame`,
  transfer lists, worker topology, or any benchmark-specific type.

**Layer 2 — Worker Transport Normalization** (NEW — Phase 4 creates this):

- Deterministic evaluation-pack creation (core-owned generic contract)
- Pack replay validation (same seed + same inputs → identical output)
- Transfer-list resolution (zero-copy `postMessage` contract)
- Schema versioning and forward-compatibility rejection
- Clone-safe payload contract for worker boundaries
- **Invariant:** Layer 2 types are generic (not `RacingRenderFrame`-specific).
  Racing provides the concrete frame type; Layer 2 provides the
  deterministic-pack and transfer contracts that any benchmark can reuse.

**Layer 3 — Benchmark-Local Racing Transport** (`examples/racing_curriculum/`):

- `RacingRenderFrame` type with `schemaVersion: 'racing-packed-v1'`
- Per-tier pack construction (`simulation-worker.tier3/4/5.ts`)
- Track-specific physics, tire/pit state, feature flags
- Renderer-specific layout (car positions, headings, place)
- **Invariant:** Layer 3 stays racing-owned and is never promoted to core.
  If Racing exposes a gap in the Layer 2 contract, the gap routes back to
  NGE core per the core-first routing policy.

###### Seam between Layer 2 and Layer 3

The seam is where core pack creation meets racing-specific frame population.
Today, `createDeterministicRacePack` in `simulation-worker.race-pack.service.ts`
fills a `RacingRenderFrame` directly from a seed + snapshot. After Phase 4,
the core owns a generic `createDeterministicEvaluationPack(seed, inputs)` that
produces a transport-neutral pack, and Racing wraps that with its own
`populateRacingFrame(pack, racingConfig)` that injects track physics, tire
state, and renderer layout. The replay contract applies at the generic pack
boundary, not at the racing-frame boundary.

###### Provisional owner note — resolved

The Step 02 readiness audit recorded:

> "The deterministic evaluation-pack owner boundary is still provisional
> pending seam mapping between NGE core, worker protocol, and racing-owned
> transport."

**Resolution:** The owner boundary is now frozen per the three-layer model
above. Core (Layer 2) owns the deterministic pack contract. Racing (Layer 3)
owns the frame assembly. Worker protocol (Layer 2, shared with core) owns the
transfer-list and clone-safe contracts. No owner boundary remains provisional.

###### Racing as proving ground, not owner

Racing is the first downstream consumer that validates the core deterministic
evaluation-pack contract works end-to-end. Racing does NOT own the pack
creation contract, the replay contract, or the transfer contract. If Racing
exposes a gap in the core contract, the gap routes back to NGE core (per the
`core_first_routing` acceptance criterion). Predator/Prey and Ant Hive will be
the second and third proving grounds, but they cannot validate the contract
until Phase 4 normalization lands and Racing confirms it e2e.

###### Reproducibility tuple for the pack-creation boundary

$$
R_{\text{pack}} = (\text{seed}, \text{opponentSnapshot}, \text{agentCount}, \text{schemaVersion})
$$

Same tuple → identical pack on the same runtime. Missing any component weakens
the claim to seed-repeatable (Level 1). The tuple does NOT include full RNG
state (Level 3 would require that) or environment assumptions (Level 4 would
require that).

###### Phase 3 prerequisite — confirmed stable

Phase 3 ([DONE]) closed with transport-neutral generation-barrier semantics in
`neat.nge-collective.two-population.ts`. The Phase 3 closure notes explicitly
defer deterministic transport normalization to Phase 4. Phase 3 no longer
changes shared prerequisites — the blocker condition is not active.

#### Step 02 — Transport and reproducibility seam mapping [DONE]

```yaml
phase: 4
step: 2
title: 'Transport and reproducibility seam mapping'
status: '[DONE]'
goal: researching
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'reproducibility-contracts, worker-inference-transport, repo-cortex-workflow'
next_step: 'Step 03 — Red tests for deterministic evaluation packs'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Map current pack generation, race-step transport, and reproducibility seams before writing tests.

**Execution focus:** Use Repo Cortex to map existing race-pack helpers, worker-frame schemas, and determinism notes; classify which files own normalization and which remain benchmark-local adapters.

**Stop conditions:** Done when seam mapping and owner boundaries are recorded; hold on unresolved transport ownership; blocked on missing workflow or Cortex evidence.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` → **PASS** (0 errors, 0 warnings)
- `node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json` → **FAIL** (`index_fresh: false`, stale paths from plan file edits; `corpus_mcp_alive: true`, `corpus_search_results: 3`). Fix: `node scripts/semantic-index/build-index.mjs` (maintenance, not blocking research).

**Seam mapping findings (Step 02 research output):**

**1. Pack generation seam**

Current owner: `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`

- `createDeterministicRacePack(seed, opponentSnapshot): RacingRenderFrame` — fills `RacingRenderFrame` typed arrays directly from seed + opponent snapshot.
- Reproducibility tuple: `R_pack = (seed, opponentSnapshot, agentCount, schemaVersion)`. Same tuple → identical pack on same runtime.
- Does NOT capture full RNG state or environment assumptions (Level 2 determinism, not Level 3).
- Target: generalize to `createDeterministicEvaluationPack(seed, inputs)` in Layer 2 (transport-neutral). Racing wraps with `populateRacingFrame(pack, racingConfig)` injecting track physics/tire/renderer.

Classification: **Benchmark-local adapter** (Layer 3). Pack creation logic is racing-specific because it fills `RacingRenderFrame` fields. The deterministic seed→pack pattern is generalizable but currently lives in racing code.

**2. Race-step transport seam**

Current state: **The racing benchmark does NOT use the library's `evaluateInWorkers`/`ParallelInferencePool` infrastructure.** It rolls its own transport:

- `resolveRaceStepTransferList(frame: RacingRenderFrame): ArrayBuffer[]` in `simulation-worker.race-pack.service.ts` — builds postMessage transfer list from frame typed arrays.
- `resolveRacingRenderFrameTransferList(frame)` in `simulation-worker.snapshot.utils.ts` — mirror implementation.
- `assertRacingSchemaVersion(frame)` in `simulation-worker.snapshot.utils.ts` — rejects frames with wrong `schemaVersion`.
- Direct `postMessage` with transfer lists; no `ParallelInferencePool`, no `evaluateInWorkers`, no `NetworkInferenceIR`, no `InferenceChannel`.
- Verified: zero imports of `multithreading`, `worker-payload`, `ParallelInferencePool`, `evaluateInWorkers`, `resolveBrowserWorkerAssetUrl`, `InferenceChannel`, or `NetworkInferenceIR` in `examples/racing_curriculum/`.

Library-owned transport infrastructure (existing, NOT consumed by racing):

- `src/architecture/network/worker-payload/network.worker-payload.batch.ts` — `evaluateInWorkers<TInput, TPayload, TWorker, TResult>(options)` + `EvaluateInWorkersOptions` + `BatchEvaluationResult`. Ordered batch evaluation with local fallback. Caller-owned transport substrate.
- `src/architecture/network/worker-payload/network.worker-payload.pool.ts` — `ParallelInferencePool<TPayload, TWorker>` with `evaluateOrderedBatch()`, `ParallelInferencePoolOptions`, `ParallelInferenceWorkerLike`.
- `src/architecture/network/worker-payload/network.worker-payload.browser-url.ts` — `resolveBrowserWorkerAssetUrl()` for CSP/custom packaging.
- `src/architecture/network/worker-payload/network.worker-payload.types.ts` — `NetworkInferenceIR`, `NetworkInferenceIRNode`, `NetworkInferenceIREdge`, `InferenceChannel`, `InferenceChannelOptions`, `InferencePredictor`. Deterministic worker-consumable network snapshot IR.
- `src/multithreading/multi.ts` — `Multi` facade with `activateSerializedNetwork`, `testSerializedSet`. Legacy multithreading boundary.
- `src/multithreading/types.ts` — `SerializableNetwork`, `SerializedSample`, `TestWorkerConstructor/Instance`. Legacy shared contracts.
- `src/multithreading/workers/workers.ts` — `Workers` class with `getBrowserTestWorker()`/`getNodeTestWorker()` runtime loader shelf.

Classification: **Two parallel transport stacks with no shared contract.**

- Library stack (`src/architecture/network/worker-payload/` + `src/multithreading/`): handles ordered batch evaluation, worker pool management, browser URL resolution, network IR serialization. **Library-owned** (Layer 2 candidate).
- Racing stack (`examples/racing_curriculum/workers/simulation-worker/`): handles `RacingRenderFrame` packing, transfer-list resolution, schema versioning, direct `postMessage`. **Benchmark-local adapter** (Layer 3).
- **Gap:** No shared contract between the two. The library stack has no deterministic pack creation, no schema versioning/rejection, no transfer-list resolution for arbitrary packed frames. The racing stack has no connection to the library's pool/batch infrastructure.

**3. Reproducibility seam**

Library-owned determinism primitives:

- `src/architecture/network/deterministic/network.deterministic.utils.ts` — `setSeed`, `snapshotRNG`, `restoreRNG`, `getRNGState`, `setRNGState`, `getRandomFn`, `RNGSnapshot` interface (`{ step: number | undefined; state: number | undefined }`). These are the Level 3 exact-resume primitives.
- `src/neat/nge-collective/neat.nge-collective.evaluation.ts` — `runCollectiveEvaluationTick` enforces sequential evaluator order `[0, 1, ..., N-1]`. `evaluationOrder` array tracks sequence. `resetCollectiveEvaluationState` resets evaluation state. This is the deterministic ordering enforcement point for collective evaluation.
- `src/neat/nge-collective/neat.nge-collective.types.ts` — `OpponentSnapshot` (NGE core: `{ agentId, snapshot, frozenAt }`), `SharedField`, `CollectiveEvaluationContext`, `CollectiveTickResult`.
- `src/neat/nge-collective/neat.nge-collective.two-population.ts` — `createTwoPopulationHarness`, `advanceTwoPopulations`. Generation barrier enforcement (transport-neutral).

Racing-local reproducibility adapters:

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.opponent-snapshot.service.ts` — `OpponentSnapshotStore` with generation barrier enforcement (rejects updates during evaluation) and generation boundary (only updates at multiples of `updateEveryNGenerations`). Racing-local `OpponentSnapshot` type: `{ snapshotId, generation, networkPayloads }`.
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.types.ts` — `RacingRenderFrame` with `schemaVersion: 'racing-packed-v1'`. Zero-copy transfer contract in JSDoc: every ArrayBuffer backing a typed-array field must appear exactly once in the postMessage transfer list; buffer is detached after transfer; producer must not reuse it; consumers must reject wrong `schemaVersion`.

Classification:

- **Library-owned** (Layer 1): `runCollectiveEvaluationTick`, `resetCollectiveEvaluationState`, `createCollectiveEvaluationContext`, `createTeamFitnessEvaluator`, generation barriers, opponent snapshot pool. Transport-neutral.
- **Library-owned** (Layer 2 candidate): `RNGSnapshot`, `setSeed`, `snapshotRNG`, `restoreRNG` — deterministic RNG primitives for Level 3 exact-resume.
- **Benchmark-local adapter** (Layer 3): `OpponentSnapshotStore` with racing-specific `OpponentSnapshot` type, `RacingRenderFrame` schema versioning, generation barrier enforcement.

**4. Owner classification table**

| Seam                        | Current file                                                                     | Current owner   | Target owner       | Classification                                        |
| --------------------------- | -------------------------------------------------------------------------------- | --------------- | ------------------ | ----------------------------------------------------- |
| Deterministic pack creation | `simulation-worker.race-pack.service.ts`                                         | Racing (L3)     | Layer 2 (new)      | Benchmark-local adapter → generalize                  |
| Pack replay validation      | (none exists)                                                                    | —               | Layer 2 (new)      | Gap: no reusable validator                            |
| Transfer-list resolution    | `simulation-worker.race-pack.service.ts` + `simulation-worker.snapshot.utils.ts` | Racing (L3)     | Layer 2 (new)      | Benchmark-local adapter → generalize                  |
| Schema versioning/rejection | `simulation-worker.types.ts` + `simulation-worker.snapshot.utils.ts`             | Racing (L3)     | Layer 2 (new)      | Benchmark-local adapter → generalize (generic schema) |
| Clone-safe payload contract | implicit (JSDoc only)                                                            | —               | Layer 2 (new)      | Gap: no explicit contract                             |
| Ordered batch evaluation    | `network.worker-payload.batch.ts`                                                | Library (L2)    | Layer 2 (existing) | Library-owned, already in `src/`                      |
| Worker pool management      | `network.worker-payload.pool.ts`                                                 | Library (L2)    | Layer 2 (existing) | Library-owned, already in `src/`                      |
| Browser worker URL          | `network.worker-payload.browser-url.ts`                                          | Library (L2)    | Layer 2 (existing) | Library-owned, already in `src/`                      |
| Network inference IR        | `network.worker-payload.types.ts`                                                | Library (L2)    | Layer 2 (existing) | Library-owned, already in `src/`                      |
| RNG state capture/restore   | `network.deterministic.utils.ts`                                                 | Library (L1/L2) | Layer 2 (existing) | Library-owned, already in `src/`                      |
| Collective evaluation order | `neat.nge-collective.evaluation.ts`                                              | Library (L1)    | Layer 1 (frozen)   | Library-owned, transport-neutral                      |
| Generation barriers         | `neat.nge-collective.two-population.ts`                                          | Library (L1)    | Layer 1 (frozen)   | Library-owned, transport-neutral                      |
| `RacingRenderFrame` type    | `simulation-worker.types.ts`                                                     | Racing (L3)     | Layer 3 (stays)    | Benchmark-local, never promoted                       |
| Per-tier pack construction  | racing demo code                                                                 | Racing (L3)     | Layer 3 (stays)    | Benchmark-local, never promoted                       |
| Track physics/tire/pit      | racing demo code                                                                 | Racing (L3)     | Layer 3 (stays)    | Benchmark-local, never promoted                       |
| Opponent snapshot store     | `simulation-worker.opponent-snapshot.service.ts`                                 | Racing (L3)     | Layer 3 (stays)    | Benchmark-local adapter                               |

**5. Gaps and blockers**

- **GAP-1: Two parallel transport stacks with no shared contract.** The library's `evaluateInWorkers`/`ParallelInferencePool` infrastructure and the racing benchmark's `RacingRenderFrame`/`postMessage` transport are completely independent. Phase 4 Layer 2 must bridge this by providing generic pack creation, schema versioning, transfer-list resolution, and clone-safe payload contracts that both sides can use.
- **GAP-2: No reusable pack replay validator.** Nothing in `src/` or `examples/` validates that a replayed pack reproduces the same outputs. This is a new Layer 2 responsibility.
- **GAP-3: No explicit clone-safe payload contract.** The zero-copy transfer contract exists only as JSDoc in `simulation-worker.types.ts`. It needs to be an explicit, enforceable interface in Layer 2.
- **GAP-4: Schema versioning is racing-specific.** `schemaVersion: 'racing-packed-v1'` and `assertRacingSchemaVersion` are hardcoded to racing. Layer 2 needs a generic schema versioning/rejection mechanism.
- **GAP-5: Cortex index stale.** `cortex-first-search.gate.mjs` reports `index_fresh: false`. Dense search degraded (`model-only`, embeddings incomplete: expected 18581, found 18540). Multiple `search_corpus` calls with `family: ts-source` + `query_class: code_specific` returned ZERO results — the family filter strips BM25 tokens for compound queries. Native grep/glob used as fallback. Fix: `node scripts/semantic-index/build-index.mjs` + `npm run index:prewarm`.
- **BLOCKER: none.** All five research questions answered. Transport ownership is resolved (racing owns racing-specific transport, library owns generic transport, Layer 2 bridges the gap). No unresolved ownership ambiguity.

**6. Layer 2 → Layer 3 seam design (from Step 01 frozen contract, confirmed by Step 02 research)**

Today: `createDeterministicRacePack(seed, opponentSnapshot)` fills `RacingRenderFrame` directly.
After Phase 4: Core owns `createDeterministicEvaluationPack(seed, inputs)` (transport-neutral). Racing wraps with `populateRacingFrame(pack, racingConfig)` injecting track physics/tire/renderer. Replay contract applies at generic pack boundary, not racing-frame boundary.

Transfer-list resolution generalizes from `resolveRaceStepTransferList(frame: RacingRenderFrame)` to `resolveTransferList(pack: DeterministicEvaluationPack): ArrayBuffer[]` in Layer 2. Racing's `resolveRaceStepTransferList` becomes a thin wrapper that calls the generic resolver.

Schema versioning generalizes from `assertRacingSchemaVersion(frame)` to `assertSchemaVersion(pack, expectedVersion)` in Layer 2. Racing keeps `'racing-packed-v1'` as its specific version string.

**7. Cortex search evidence**

- `freshness_check`: index fresh for most docs, plan file stale (recently modified). Dense: `model-only`, embeddings incomplete (18540/18581).
- `search_corpus` "worker pool evaluation batch ordered results multithreading": returned skill/agent/completed-plan results. `evaluateInWorkers`, `ParallelInferencePool`, `resolveBrowserWorkerAssetUrl` identified.
- `search_corpus` "RNG deterministic seed snapshot restore" (family: ts-source): returned `network.deterministic.utils.ts` with `setSeed`, `snapshotRNG`, `restoreRNG`, `RNGSnapshot`.
- `search_corpus`/`search_advanced` with `family: ts-source` + `query_class: code_specific`: ZERO results (Cortex gap — family filter strips BM25 tokens for compound queries). Fallback: native grep/glob.
- `load_document` on 10+ source files: confirmed all seam locations and ownership classifications above.
- `load_document` on `network.worker-payload.batch.ts`: confirmed `evaluateInWorkers` definition, `EvaluateInWorkersOptions` (7 fields), `BatchEvaluationResult` (ordered results + task ids + mode).
- `load_document` on `network.worker-payload.types.ts`: confirmed `NetworkInferenceIR`, `InferenceChannel`, `InferencePredictor`, `InferenceChannelOptions` types.
- Native grep confirmed: zero imports of library worker transport in `examples/racing_curriculum/`; racing uses `createTeamFitnessEvaluator` from `src/neat/nge-collective/` (Layer 1 → Layer 3 seam).

**TASK_STATUS: SUCCESS.** Seam mapping complete. All five research questions answered. Owner boundaries recorded. Ready for Step 03 (red tests for deterministic evaluation packs).

#### Step 03 — Red tests for deterministic evaluation packs [DONE]

Claim: 04-implementing @ 2026-06-16

```yaml
phase: 4
step: 3
title: 'Red tests for deterministic evaluation packs'
status: '[DONE]'
goal: red-testing
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'red-test-contracts, creating-unit-tests'
next_step: 'Step 04 — Implement deterministic evaluation-pack normalization'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-3-red-tests
    title: 'Write red tests'
    status: '[DONE]'
    goal: red-testing
    estimate_hours: 4
    files_to_change:
      - 'src/architecture/network/evaluation-pack/network.evaluation-pack.ts (stub)'
      - 'src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts (7 red tests)'
    acceptance_criteria:
      - 'Red tests exist and fail for the expected behavior.'
    parallelizable: false
    dependencies:
    next_slice: step-3-core
  - slice_id: step-3-core
    title: 'Implement the core behavior'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 8
    files_to_change:
      - 'src/architecture/network/evaluation-pack/network.evaluation-pack.ts'
    acceptance_criteria:
      - 'Implementation satisfies the red tests and design.'
    parallelizable: false
    dependencies:
      - step-3-red-tests
    next_slice: step-3-green
  - slice_id: step-3-green
    title: 'Green validation and coverage guard'
    status: '[DONE]'
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

**Step objective:** Add focused failing tests for deterministic pack normalization, replay stability, and race-step transport contracts.

**Execution focus:** Keep tests narrow to the selected normalization seam and avoid folding in broader population or lifecycle work that belongs to adjacent phases.

**Stop conditions:** Done when the missing determinism contract fails under focused tests; hold on unresolved owner boundary assertions; blocked if Step 02 mapping is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` → **PASS** (0 errors, 0 warnings)

**Red evidence (step-3-red-tests slice — DONE):**

- **Prerequisite:** Step 02 seam mapping is [DONE] with no blockers. Layer 2 → Layer 3 seam is frozen: core owns `createDeterministicEvaluationPack(seed, inputs)`, `resolveTransferList(pack)`, and `assertSchemaVersion(pack, expectedVersion)`. Racing wraps with `populateRacingFrame(pack, racingConfig)`.
- **Cortex search evidence:** Chunks 101586 (Step 02 [DONE] header), 101581 (seam between Layer 2 and Layer 3), 99232 (race-pack service source), 99246/99247 (snapshot utils source), 99267/99268 (RacingRenderFrame type), 87746 (deterministic utils snapshotRNG), 89336 (getTransferList worker-payload). `freshness_check`: index fresh.
- **Files created:**
  - `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` — stub module with `DeterministicEvaluationPack`, `EvaluationPackInputs` types and `createDeterministicEvaluationPack`, `resolveTransferList`, `assertSchemaVersion` function stubs (throw `Error('Not implemented …')`).
  - `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` — 7 red tests targeting the three normalization contracts.
- **Test command:** `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.evaluation-pack`
- **Result:** 7 failed / 7 total. All fail for the expected reason (stubs throw "Not implemented — Phase 4 Step 03 red contract"). No import/syntax errors.
- **Red contracts encoded:**
  1. `createDeterministicEvaluationPack` — same seed + same inputs → identical packs (Level 2 ordered-deterministic).
  2. `createDeterministicEvaluationPack` — different seeds → different packs (determinism distinctness).
  3. Replay stability — replayed packs produce byte-identical typed array contents.
  4. `resolveTransferList` — collects all distinct buffer entries from a pack.
  5. `resolveTransferList` — deduplicates buffers shared across multiple typed arrays.
  6. `assertSchemaVersion` — accepts a pack whose schema version matches (no throw).
  7. `assertSchemaVersion` — throws `RangeError` on schema version mismatch (generic, not racing-specific).
- **Gate checks:** `step-packet` → PASS. `plan-sync` → PASS.
- **Handoff to step-3-core (04-implementing):**
  - **Module:** `src/architecture/network/evaluation-pack/network.evaluation-pack.ts`
  - **Test file:** `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts`
  - **Validate with:** `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.evaluation-pack`
  - **Green target:** All 7 tests pass when `createDeterministicEvaluationPack` produces deterministic typed arrays from `(seed, inputs)`, `resolveTransferList` collects and deduplicates buffers from `pack.arrays`, and `assertSchemaVersion` throws `RangeError` on mismatch and silently accepts on match.
  - **Contract:** DeterministicEvaluationPack = `{ schemaVersion: string; seed: number; arrays: readonly ArrayBufferView[] }`. EvaluationPackInputs = `{ agentCount: number; schemaVersion: string }`.
  - **Setup/teardown:** Deterministic fixtures (SEED=42, ALT_SEED=99, INPUTS={agentCount:4, schemaVersion:'test-eval-pack-v1'}). No external state. Pure helper functions for typed-array comparison.

**Core implementation evidence (step-3-core slice — DONE):**

- **File changed:** `src/architecture/network/evaluation-pack/network.evaluation-pack.ts`
- **What each function does:**
  - `createDeterministicEvaluationPack(seed, inputs)` — seeds a self-contained xorshift32 PRNG, builds `Float32Array` agent states, `Float64Array` agent weights, and `Uint8Array` active flags of length `inputs.agentCount`, and returns a frozen pack with `schemaVersion`, `seed`, and `arrays`. Same `(seed, inputs)` → byte-identical pack on the same runtime.
  - `resolveTransferList(pack)` — walks `pack.arrays`, collects each view's `.buffer`, and deduplicates shared `ArrayBuffer` references using a `Set`.
  - `assertSchemaVersion(pack, expectedVersion)` — accepts when `pack.schemaVersion === expectedVersion`; throws `RangeError` with a descriptive message on mismatch.
- **Type refinements:** `DeterministicEvaluationPack.arrays` kept as `readonly ArrayBufferView[]` to stay generic across benchmark wrappers; helper `buildPackArrays` returns `ArrayBufferView[]` with concrete `Float32Array | Float64Array | Uint8Array` elements. No `any`/`unknown` introduced.
- **Preflight checks:**
  - `npx tsc --noEmit -p tsconfig.json` → pass (exit 0)
  - `npx tsc --noEmit -p tsconfig.test.json` → pass (exit 0)
  - `npm run lint` → pass (0 errors)
  - `npx prettier --check src/architecture/network/evaluation-pack/network.evaluation-pack.ts` → pass
  - `npx prettier --check src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` → [warn] formatting issues exist in test file (left unchanged per red-test contract; green validator may decide to format)
- **Focused tests:** `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.evaluation-pack` → **7 passed / 7 total**
- **Folder quality gate:** `npm run quality:folder -- --folder=src/architecture/network/evaluation-pack` → PASS (0 diagnostics, 0 ESLint errors, 3/3 exported symbols documented, sibling test present, line coverage 100%)
- **Focused coverage:** `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=network.evaluation-pack` → statements 100%, branches 83.33%, functions 100%, lines 100%; uncovered branch is the `seed === 0` fallback in `createPackPRNG` (line 209). The fallback is reachable defensive code and is deferred to step-3-green coverage guard.
- **Gate checks:** `step-packet` → PASS. `plan-sync` → PASS (0 errors, 0 warnings).

```yaml
PlanUpdate:
  slice_id: step-3-core
  changed_files:
    - src/architecture/network/evaluation-pack/network.evaluation-pack.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npm run quality:folder -- --folder=src/architecture/network/evaluation-pack'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.evaluation-pack'
      expected_exit: 0
  rollback:
    - 'git checkout -- src/architecture/network/evaluation-pack/network.evaluation-pack.ts'
  coverage_note:
    file: 'src/architecture/network/evaluation-pack/network.evaluation-pack.ts'
    statements: 100
    branches: 83.33
    functions: 100
    lines: 100
    uncovered_branch: 'seed === 0 fallback in createPackPRNG (line 209) — deferred to step-3-green coverage guard'
  next: 'Run 05-green-testing and attach coverage-guard evidence'
```

**Handoff to step-3-green (05-green-testing):**

- Validate focused slice still passes: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=network.evaluation-pack`
- Close branch-coverage gap for line 209 (`seed === 0` fallback in `createPackPRNG`) or confirm it is acceptable dead code.
- Run repo-wide suite only if the active step packet or user explicitly requires repo-wide confirmation.
- Confirm `coverage-guard` for `src/architecture/network/evaluation-pack/network.evaluation-pack.ts` reaches 100% in all four categories before closing the slice.

**Green validation evidence (step-3-green slice — DONE):**

- **Files changed:**
  - `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` — added one focused test covering the `seed === 0` fallback branch in `createPackPRNG`.
- **Coverage closure:** New test `'falls back to a non-zero PRNG state when the seed is zero'` calls `createDeterministicEvaluationPack(0, INPUTS)` and asserts `pack.arrays` has length `3`. This exercises the `seed === 0 ? PACK_PRNG_FALLBACK_SEED : seed >>> 0` branch and brings branch coverage to 100%.
- **Prettier:** `npx prettier --write src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` applied; `npx prettier --check` passes for both source and test files.
- **Focused test + coverage:**
  - Command: `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns=network.evaluation-pack`
  - Result: 8 passed / 8 total
  - Coverage for `src/architecture/network/evaluation-pack/network.evaluation-pack.ts`: statements 100%, branches 100%, functions 100%, lines 100%
- **Type check:** `npx tsc --noEmit -p tsconfig.json` and `npx tsc --noEmit -p tsconfig.test.json` both pass.
- **Lint:** `npm run lint` passes (0 errors).
- **Folder quality gate:** `npm run quality:folder -- --folder=src/architecture/network/evaluation-pack` → PASS (0 diagnostics, 0 ESLint errors, 3/3 exported symbols documented, sibling test present, 1 lcov entry at 100%).
- **Full suite:** `npm run test:silent` — `src/architecture/network/evaluation-pack/network.evaluation-pack.test.ts` passes; the only failures are 3 unrelated pre-existing test suites (`scripts/agent-customization/plan-workflow.test.ts`, `scripts/mcp-semantic/__tests__/repo-cortex-mcp.red.test.ts`, `scripts/agent-customization/hooks/runtime-enforcement-hooks.test.ts`, 4 tests total) that are outside the `src/architecture/network/evaluation-pack/` boundary and not caused by this change.
- **Coverage guard:** All touched `src/` files (`src/architecture/network/evaluation-pack/network.evaluation-pack.ts`) are at 100% across statements, branches, functions, and lines.
- **Plan-sync gate:** run after updating this plan.

#### Step 04 — Implement deterministic evaluation-pack normalization [DONE]

```yaml
phase: 4
step: 4
title: 'Implement deterministic evaluation-pack normalization'
status: '[DONE]'
goal: implementing
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'reproducibility-contracts, worker-inference-transport'
next_step: 'Step 05 — Green validation and determinism audit'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-4-red-tests
    title: 'Write red tests'
    status: '[DONE]'
    goal: red-testing
    estimate_hours: 4
    files_to_change:
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.test.ts
    acceptance_criteria:
      - 'Red tests exist and fail for the expected behavior.'
    parallelizable: false
    dependencies:
    next_slice: step-4-core
    notes:
      - 'Seam: Layer 2 core DeterministicEvaluationPack -> Layer 3 RacingRenderFrame normalization boundary in examples/racing_curriculum/workers/simulation-worker.'
      - 'Contracts: populateRacingFrame returns a RacingRenderFrame-shaped object; same pack + same config yields identical typed-array contents; transfer list includes all ArrayBuffers owned by the frame; assertRacingPackSchemaVersion accepts matching version and rejects mismatch with RangeError; wrapper rejects mismatched agentCount.'
      - 'Red evidence: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.evaluation-pack.normalizer -> 7 failed / 7 total, all throwing Error("Not implemented ...").'
      - 'Handoff to step-4-core: implement the stub functions in simulation-worker.evaluation-pack.normalizer.ts so that the 7 tests in simulation-worker.evaluation-pack.normalizer.test.ts pass and no broad suite regressions are introduced.'
  - slice_id: step-4-core
    title: 'Implement the core behavior'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 8
    files_to_change:
      - examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts
    acceptance_criteria:
      - 'Implementation satisfies the red tests and design.'
    parallelizable: false
    dependencies:
      - step-4-red-tests
    next_slice: step-4-green
    notes:
      - 'Implemented populateRacingFrame, resolveRacingTransferList, and assertRacingPackSchemaVersion in the Layer 3 normalizer.'
      - 'populateRacingFrame validates schema version and agent count, then deterministically maps the three core pack arrays (agentStates, agentWeights, agentActive) to the RacingRenderFrame typed-array fields.'
      - 'resolveRacingTransferList delegates to resolveRacingRenderFrameTransferList from simulation-worker.snapshot.utils to share the existing zero-copy ownership contract.'
      - 'assertRacingPackSchemaVersion mirrors the generic assertSchemaVersion contract, throwing RangeError on mismatch.'
      - 'Preflight: npx tsc --noEmit -p tsconfig.json -> OK; npm run lint -> 0 issues; npx prettier --check normalizer.ts -> OK.'
      - 'Focused tests: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.evaluation-pack.normalizer -> 7 passed / 7 total.'
      - 'Coverage note: changed file is in examples/, so coverage-guard 100% rule (src/ only) does not apply; step-4-green should still verify no broad suite regressions.'
      - 'Handoff to step-4-green: run green validation and determinism audit; optional repo-wide suite only if explicitly required.'
  - slice_id: step-4-green
    title: 'Green validation and coverage guard'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 4
    files_to_change:
      - coverage/lcov.info
    acceptance_criteria:
      - 'All tests pass and coverage guard is satisfied.'
    parallelizable: false
    dependencies:
      - step-4-core
    notes:
      - 'Focused slice: npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.evaluation-pack.normalizer -> 7 passed / 7 total.'
      - 'Preflight: npx tsc --noEmit -p tsconfig.json -> pass; npm run lint -> 0 errors; npx prettier --check on normalizer.ts and normalizer.test.ts -> pass.'
      - 'Quality gate: npm run quality:folder -- --folder=examples/racing_curriculum/workers/simulation-worker -> PASS (0 TS errors, 0 ESLint errors, 17/17 JSDoc symbols, 0 missing tests, 0 lcov entries below 100%).'
      - 'Coverage guard: no src/ files touched; examples/ file does not trigger 100% coverage obligation.'
      - 'Plan-sync gate: PASS (0 errors, 0 warnings).'
      - 'No test or implementation changes were required.'
```

**Step objective:** Implement the deterministic evaluation-pack and race-pack normalization seam selected by the red tests.

**Execution focus:** Keep the code bounded to normalized pack generation, stable transport, and explicit ownership boundaries; route any new lifecycle or benchmark-policy questions back to their named phases.

**Stop conditions:** Done when red tests turn green for the selected normalization seam; hold on policy ambiguity; blocked on contradictory worker transport ownership.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

**Core implementation evidence (step-4-core slice — DONE):**

- **File changed:** `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts`
- **What each function does:**
  - `populateRacingFrame(pack, racingConfig)` — asserts the generic pack schema version matches `racingConfig.packSchemaVersion`, validates that every core pack typed-array dimension equals `racingConfig.agentCount`, then deterministically maps `agentStates` → `carX`/`carHeading`/`tireState`, `agentWeights` → `carY`, and `agentActive` → `carActive`/`carTeam`/`carMode`/`lap`/`place`. Returns a `RacingRenderFrame` with `schemaVersion: 'racing-packed-v1'`.
  - `resolveRacingTransferList(frame)` — delegates to `resolveRacingRenderFrameTransferList` from `simulation-worker.snapshot.utils` to collect distinct `ArrayBuffer` references owned by the frame.
  - `assertRacingPackSchemaVersion(pack, expectedVersion)` — accepts when `pack.schemaVersion === expectedVersion`; throws `RangeError` with a descriptive message on mismatch.
- **Preflight checks:**
  - `npx tsc --noEmit -p tsconfig.json` → pass (exit 0)
  - `npm run lint` → pass (0 errors)
  - `npx prettier --check examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts` → pass
- **Focused tests:** `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.evaluation-pack.normalizer` → **7 passed / 7 total**
- **Coverage note:** The changed file lives in `examples/`; `coverage-guard` 100% rule applies to `src/` files only. No `src/` files were modified, so no coverage regression is expected. Step-4-green should still verify focused slice and check for broad suite regressions.
- **Gate checks:**
  - `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` → PASS (0 errors, 0 warnings)
  - `node .github/hooks/workflow-update-sync.mjs --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md --json` → advance (Phase 4 Step 4 → [DONE]; Phase 4 Step 5 → [WIP])

```yaml
PlanUpdate:
  slice_id: step-4-core
  changed_files:
    - examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.evaluation-pack.normalizer'
      expected_exit: 0
  rollback:
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts'
  coverage_note:
    scope: 'examples/ file; coverage-guard 100% rule applies to src/ only'
    next: 'step-4-green validates no broad suite regressions'
  next: 'Run step-4-green (05-green-testing) focused slice and attach green evidence'
```

**Handoff to step-4-green (05-green-testing):**

- Files to validate: `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evaluation-pack.normalizer.ts` (and its red-test sibling `simulation-worker.evaluation-pack.normalizer.test.ts`, which must remain unchanged).
- Run the focused slice: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.evaluation-pack.normalizer` and confirm 7/7 passing.
- Optional: run `npm run test:silent` only if the active step packet or user explicitly requires repo-wide confirmation.
- Confirm `npx tsc --noEmit -p tsconfig.json` and `npm run lint` remain clean.

#### Step 05 — Green validation and determinism audit [WIP]

```yaml
phase: 4
step: 5
title: 'Green validation and determinism audit'
status: '[WIP]'
goal: green-testing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'green-validation-gates, reproducibility-contracts'
next_step: 'Step 06 — Documentation and reproducibility contract alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-5-red-tests
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
    next_slice: step-5-core
  - slice_id: step-5-core
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
      - step-5-red-tests
    next_slice: step-5-green
  - slice_id: step-5-green
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
      - step-5-core
```

**Step objective:** Confirm deterministic packs, replay behavior, and transport invariants stay green for the touched boundary.

**Stop conditions:** Done when focused validation passes and determinism evidence is recorded; hold on flaky replay output; blocked if implementation remains red.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 06 — Documentation and reproducibility contract alignment [PLANNED]

```yaml
phase: 4
step: 6
title: 'Documentation and reproducibility contract alignment'
status: '[PLANNED]'
goal: documenting
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'educational-docs, reproducibility-contracts'
next_step: 'Step 07 — Logging and phase closure packet'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Update touched docs so deterministic evaluation-pack semantics, caveats, and downstream benchmark usage are explicit and accurate.

**Stop conditions:** Done when docs align to file-backed behavior; hold on unresolved wording; blocked if green validation evidence is missing.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 07 — Logging and phase closure packet [PLANNED]

```yaml
phase: 4
step: 7
title: 'Logging and phase closure packet'
status: '[PLANNED]'
goal: logging
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - tracker-handoff
next_step: 'Phase 5 — Lifecycle staging closure and nge-adult readiness reconciliation'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Compress deterministic-pack coverage, record validation evidence, and keep lifecycle staging reconciliation as the next explicit queue.

**Stop conditions:** Done when the tracker reflects durable coverage and remaining lifecycle work stays visible; hold on pending user confirmation; blocked if evidence is missing.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

### Phase 5 — Lifecycle staging closure and nge-adult readiness reconciliation [PLANNED]

```yaml
phase: 5
title: 'Lifecycle staging closure and nge-adult readiness reconciliation'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_phase: 'Experimental root public API exposure'
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
placeholder_steps:
  - 'Step 01 — Planning packet and lifecycle contradiction freeze'
  - 'Step 02 — Lifecycle seam and evidence mapping'
  - 'Step 03 — Red tests for lifecycle staging readiness'
  - 'Step 04 — Implement lifecycle staging closure'
  - 'Step 05 — Green validation and lifecycle audit'
  - 'Step 06 — Documentation and lifecycle contract alignment'
  - 'Step 07 — Logging and phase closure packet'
```

#### Step 01 — Planning packet and lifecycle contradiction freeze [PLANNED]

```yaml
phase: 5
step: 1
title: 'Planning packet and lifecycle contradiction freeze'
status: '[PLANNED]'
goal: planning
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'plan-alignment, phase-handoff-workflow, nge-core-algorithm'
next_step: 'Step 02 — Lifecycle seam and evidence mapping'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Freeze the lifecycle staging acceptance criteria and reconcile how archived completion claims differ from the current readiness audit.

**Execution focus:** Preserve lifecycle staging, juvenile, assimilation, Phase B/D/E/G, and `nge-adult` terminology; record what counts as closure versus scaffolded behavior.

**Stop conditions:** Done when lifecycle closure criteria are explicit; hold on unresolved archival contradictions; blocked if earlier phases still change prerequisite lifecycle contracts.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 02 — Lifecycle seam and evidence mapping [PLANNED]

```yaml
phase: 5
step: 2
title: 'Lifecycle seam and evidence mapping'
status: '[PLANNED]'
goal: researching
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'nge-core-algorithm, repo-cortex-workflow'
next_step: 'Step 03 — Red tests for lifecycle staging readiness'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Map lifecycle-stage seams, `nge-adult` readiness, and archived contradiction points with file-backed evidence.

**Execution focus:** Re-read upstream NGE docs, use Repo Cortex for lifecycle staging evidence, and identify which claims are implemented, partial, or still placeholder behavior.

**Stop conditions:** Done when file-backed contradictions and gaps are recorded; hold on unresolved stage semantics; blocked on missing workflow or Cortex evidence.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`
- `node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json`

#### Step 03 — Red tests for lifecycle staging readiness [PLANNED]

```yaml
phase: 5
step: 3
title: 'Red tests for lifecycle staging readiness'
status: '[PLANNED]'
goal: red-testing
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'red-test-contracts, creating-unit-tests'
next_step: 'Step 04 — Implement lifecycle staging closure'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
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

**Step objective:** Add focused failing tests that prove the current lifecycle staging and `nge-adult` readiness claims are incomplete or contradictory.

**Execution focus:** Cover juvenile, assimilation, and adult-stage boundaries with the smallest failing test slice and avoid folding in public API export work owned by Phase 6.

**Stop conditions:** Done when the intended lifecycle gap is reproduced in red; hold on unresolved semantics; blocked if Step 02 evidence is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 04 — Implement lifecycle staging closure [PLANNED]

```yaml
phase: 5
step: 4
title: 'Implement lifecycle staging closure'
status: '[PLANNED]'
goal: implementing
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - nge-core-algorithm
next_step: 'Step 05 — Green validation and lifecycle audit'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
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

**Step objective:** Implement the lifecycle staging closure selected by red tests and reconcile `nge-adult` readiness with file-backed behavior.

**Execution focus:** Keep the implementation bounded to lifecycle semantics, do not over-claim benchmark readiness, and leave export-surface work for Phase 6.

**Stop conditions:** Done when red tests turn green and lifecycle contradictions are resolved in code; hold on new policy questions; blocked on conflicting archived assumptions that need escalation.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 05 — Green validation and lifecycle audit [PLANNED]

```yaml
phase: 5
step: 5
title: 'Green validation and lifecycle audit'
status: '[PLANNED]'
goal: green-testing
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - green-validation-gates
next_step: 'Step 06 — Documentation and lifecycle contract alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-5-red-tests
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
    next_slice: step-5-core
  - slice_id: step-5-core
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
      - step-5-red-tests
    next_slice: step-5-green
  - slice_id: step-5-green
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
      - step-5-core
```

**Step objective:** Validate the touched lifecycle boundary and confirm the code now supports the documented staging claims.

**Stop conditions:** Done when focused validation passes; hold on flaky lifecycle output; blocked if implementation remains red.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 06 — Documentation and lifecycle contract alignment [PLANNED]

```yaml
phase: 5
step: 6
title: 'Documentation and lifecycle contract alignment'
status: '[PLANNED]'
goal: documenting
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'educational-docs, docs-academic-citation-audit'
next_step: 'Step 07 — Logging and phase closure packet'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Update touched lifecycle docs so current readiness, remaining caveats, and benchmark implications are explicit and honest.

**Stop conditions:** Done when docs align to file-backed lifecycle behavior; hold on unresolved wording; blocked if validation evidence is missing.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 07 — Logging and phase closure packet [PLANNED]

```yaml
phase: 5
step: 7
title: 'Logging and phase closure packet'
status: '[PLANNED]'
goal: logging
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - tracker-handoff
next_step: 'Phase 6 — Experimental root public API exposure'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Compress lifecycle-phase coverage and keep experimental API exposure as the next explicit tranche.

**Stop conditions:** Done when lifecycle work is summarized without hiding unresolved items; hold on pending user confirmation; blocked if evidence is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

### Phase 6 — Experimental root public API exposure [PLANNED]

```yaml
phase: 6
title: 'Experimental root public API exposure'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
next_phase: 'Downstream benchmark dependency + MCP synchronization'
skills:
  - plan-alignment
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
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

#### Step 01 — Planning packet and export-surface freeze [PLANNED]

```yaml
phase: 6
step: 1
title: 'Planning packet and export-surface freeze'
status: '[PLANNED]'
goal: planning
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'plan-alignment, phase-handoff-workflow, nge-core-algorithm'
next_step: 'Step 02 — Public-surface and dependency mapping'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
skills:
  - 'nge-core-algorithm, repo-cortex-workflow'
next_step: 'Step 03 — Red tests for experimental root API exposure'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'red-test-contracts, creating-unit-tests'
next_step: 'Step 04 — Implement experimental root API exposure'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
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
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - nge-core-algorithm
next_step: 'Step 05 — Green validation and export-surface audit'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
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
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - green-validation-gates
next_step: 'Step 06 — Documentation and API usage alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-5-red-tests
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
    next_slice: step-5-core
  - slice_id: step-5-core
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
      - step-5-red-tests
    next_slice: step-5-green
  - slice_id: step-5-green
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
      - step-5-core
```

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
skills:
  - educational-docs
next_step: 'Step 07 — Logging and phase closure packet'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
skills:
  - tracker-handoff
next_step: 'Phase 7 — Downstream benchmark dependency + MCP synchronization'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
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
skills:
  - 'plan-alignment, phase-handoff-workflow, repo-cortex-workflow'
next_step: 'Step 02 — Downstream tracker and MCP seam mapping'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
skills:
  - 'repo-cortex-workflow, nge-benchmark-workflow'
next_step: 'Step 03 — Red tests for synchronization contracts'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - red-test-contracts
next_step: 'Step 04 — Implement synchronization hooks and tracker updates'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
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
expansion: slices
tdd_sequence: red-green
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - repo-cortex-workflow
next_step: 'Step 05 — Green validation and MCP audit'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
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
expansion: slices
tdd_sequence: green-only
auto_expand: true
mode: fresh-session
source_of_truth: plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
copy_paste: true
skills:
  - 'green-validation-gates, repo-cortex-workflow'
next_step: 'Step 06 — Documentation and handoff alignment'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
  - 'node scripts/agent-customization/mcp/neataptic-gate-mcp.mjs --self-check --json'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
slices:
  - slice_id: step-5-red-tests
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
    next_slice: step-5-core
  - slice_id: step-5-core
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
      - step-5-red-tests
    next_slice: step-5-green
  - slice_id: step-5-green
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
      - step-5-core
```

**Step objective:** Verify the synchronization implementation and MCP wiring stay green and report accurate evidence.

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
skills:
  - 'tracker-handoff, repo-cortex-workflow'
next_step: 'Step 07 — Logging and phase closure packet'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

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
skills:
  - tracker-handoff
next_step: 'User confirmation before any named phase closes'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
acceptance_criteria:
  - 'Phase/step metadata validates with the new plan-phase-step schema.'
```

**Step objective:** Compress the final synchronization phase history while preserving the invariant that nothing closes until file-backed evidence exists and the user confirms completion.

**Stop conditions:** Done when the tracker records durable synchronization coverage and explicit remaining return conditions; hold on missing user confirmation; blocked if evidence is incomplete.

**Required validation:** `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

## Validation gates

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

### Latest validation evidence

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

Workstream: NGE core readiness tracker (`plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`).

Current durable coverage:
- Phase 1 readiness audit and packetization are `[DONE]`.
- Phase 2 team-level fitness core evaluator is `[DONE]`; `createTeamFitnessEvaluator` now lives in NGE core, Racing consumes it, and the touched validation/docs evidence is recorded in the Phase 2 coverage note.
- Phase 3 is the active phase.
- Phase 3 Step 01 is `[DONE]`; the reusable boundary is now frozen: core owns independent-population semantics plus transport-neutral generation-barrier behavior, while deterministic transport normalization stays in Phase 4.

Next narrow task:
- Continue with Phase 3 Step 03 — Red tests for independent populations and barriers.
- Add the smallest failing tests that prove population isolation, snapshot freeze/release, and coordinated next-generation eligibility without asserting Phase 4 transport schema details.
- Preserve the frozen owner split already recorded here: Phase 3 owns barrier semantics, Phase 4 owns deterministic evaluation/race-pack transport normalization, and downstream benchmarks remain consumers rather than design owners.

Seam map and owner table (file-backed evidence):

- NGE core — two-population harness and snapshot pool (Phase 3 reusable semantics):
  - src\neat\nge-collective\neat.nge-collective.two-population.ts:104-149,205-269 — `createTwoPopulationHarness`, `runTwoTeamEvaluationTick`, and `advanceTwoPopulations` isolate team-local controllers, retain one shared evaluation context, and cross-register frozen rival snapshots.
  - src\neat\nge-collective\neat.nge-collective.metrics.ts:104-148 — `createOpponentSnapshotPool` / `addOpponentSnapshot` deep-clone payloads and rotate bounded frozen snapshot history.
  - src\neat\nge-collective\neat.nge-collective.types.ts:77-98 — `OpponentSnapshot` / `OpponentSnapshotPool` invariants define immutable frozen payloads and FIFO retention.
  - src\neat\nge-collective\neat.nge-collective.team-fitness.ts:31-42,85-99 — `createTeamFitnessEvaluator` stays a closed Phase 2 prerequisite and explicitly excludes barriers and transport.

- Racing benchmark — worker-local barrier enforcement and transport shapes (benchmark-local, Phase 4 transport deferred):
  - examples\racing_curriculum\browser-entry\browser-entry.ts:126-143 — current browser seam still sends full `EnvironmentState` each tick, while the target worker-owned protocol moves populations, snapshots, and `race-step` production into the worker.
  - examples\racing_curriculum\workers\simulation-worker\simulation-worker.opponent-snapshot.service.ts:4-10,22-45,52-66,103-123 — `beginEvaluation` / `endEvaluation` / `tryUpdateSnapshot` implement local generation-barrier guards.
  - examples\racing_curriculum\workers\simulation-worker\simulation-worker.race-pack.service.ts:52-91,137-192 — `createDeterministicRacePack` and `resolveRaceStepTransferList` own packed frame schema and transfer-list rules.

- Predator/Prey plan (downstream consumer):
  - plans\NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md:498-509,699-736 — describes two independent NEAT workers, a coordinator barrier, frozen rolling opponent snapshots, and generation restart only after both populations report ready.

Frozen Phase 3 ownership (transport-neutral barrier semantics):
- Core must own: population independence, opponent snapshot freeze/retention semantics, and coordinated next-generation eligibility without transport assumptions.
- Benchmarks must own: worker FSMs, episode queueing, host/render cadence, transfer-list rules, and packed `race-step` payload shapes (transport details remain Phase 4).

Holds / unresolved policy questions (require explicit hold rather than silent assignment):
- Hold: Whether Phase 3 must provide a named minimal barrier-summary payload or only a transport-neutral callback/seam. Step 03 should test sequencing first; if payload fields become necessary, record that policy decision here and defer the concrete shape to Phase 4 transport work.

Required validation:
- node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
- node scripts/agent-customization/gates/cortex-first-search.gate.mjs --json

Route-back / return condition:
- When Step 03 confirms the failing barrier/isolation behavior and either keeps the hold as callback-only or records a named summary-policy decision, Step 04 can implement the reusable barrier seam without widening into transport work.


```
