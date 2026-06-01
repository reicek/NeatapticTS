# NEAT Genesis EvoDevo: Core Readiness Audit

**Status:** [WIP]

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
  - "NGE core primitives must be audited and the highest-leverage gaps closed before benchmark demos can act as honest e2e tests."
  - "Demo-local compensation for missing core capabilities produces benchmark results that reflect the workaround rather than the algorithm."
  - "Racing is the first e2e proving ground; any racing gap must route back to NGE core rather than being solved inside the racing demo."
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
    - "00-helping only when an MCP/tool/agent/flow gap blocks the active step."
non_goals:
  - "Do not implement missing NGE primitives during the audit phase."
  - "Do not patch benchmark-local workarounds into demos to hide missing core capabilities."
  - "Do not optimize the first tranche around racing-, predator/prey-, or ant-hive-specific semantics before generic NGE foundation gaps are closed."
  - "Do not treat demo completion as proof of readiness if the demo only works via benchmark-local compensation."
  - "Do not export every internal NGE module at once; prefer a narrow experimental namespace for the initial public entrypoint."
acceptance_criteria:
  - id: primitive_classification
    criterion: "Given the requested primitive list, when the audit completes, then each primitive is classified as implemented, partial, plan-only, or missing with at least one file-backed citation."
    validation: "readiness matrix with file citations"
  - id: gap_ownership
    criterion: "Given a primitive is not fully ready, when it is recorded in the audit, then it is assigned to the correct owner boundary: upstream NGE core, benchmark-local runtime, worker protocol, or policy decision."
    validation: "gap classification table"
  - id: terminology_preservation
    criterion: "Given plan alignment matters, when the audit names a gap, then it preserves existing upstream terminology such as rolling opponent snapshot, polyandric reproduction, deterministic race packs, and Phase A/B/D/E/G."
    validation: "audit doc review"
  - id: benchmark_decomposition
    criterion: "Given downstream benchmarks depend on different subsets of NGE, when the audit compares readiness, then it distinguishes reusable core support from racing-only, predator/prey-only, and ant-hive-only requirements."
    validation: "per-benchmark readiness column in matrix"
  - id: smallest_tranche
    criterion: "Given the user wants an implementation plan after the audit, when the audit closes, then it identifies the smallest next tranche that removes the highest-leverage blocker."
    validation: "tranche selection rationale doc"
  - id: lifecycle_honesty
    criterion: "Given lifecycle staging is easy to overclaim, when the audit evaluates lifecycle support, then it explicitly calls out whether the current boundary is production-ready or still scaffolded or red-phase."
    validation: "lifecycle status row in matrix"
  - id: core_first_routing
    criterion: "Given demos are end-to-end tests, when an e2e exposes a missing capability or failure, then the gap is routed back to NGE core rather than solved with demo-local compensation."
    validation: "routing policy documented in tranche handoff"
stop_conditions:
  done: "All Phase 1 steps are complete and the implementation handoff packet is ready for Phase 2."
  hold: "A benchmark-policy decision is needed before the first tranche can be selected."
  blocked: "A missing upstream primitive or MCP/tooling gap prevents honest audit completion."
```

## Current-state audit summary

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

## Open assumptions

- The exact public entrypoint shape is still open: a narrow experimental namespace may be safer than exporting every internal NGE module at once.
- The first execution tranche still needs a final priority decision between lifecycle closure, reusable collective-coevolution semantics, and public-surface wiring if all three cannot fit cleanly into one bounded pass.

## Implementation phases

### Phase 1 — Readiness audit and tranche selection [WIP]

#### Step 01 — Initial survey and findings [DONE]

[DONE] Initial audit compressed: upstream plan evidence, code exploration, and
racing curriculum gap review confirm that DNA, reproduction operators, collective
harness, and race-pack worker shapes are partially present; team-level fitness,
generation barriers, deterministic evaluation packs, lifecycle completeness
(`nge-adult`), and root public API exposure remain the highest-leverage gaps.
See Current-state audit summary above for file-backed evidence.

#### Step 02 — Readiness matrix and gap classification [WIP]

```yaml
phase: 1
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 03 — First tranche selection'
skills: 'nge-core-algorithm'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Produce a primitive-by-primitive readiness matrix and classify each gap by owner boundary so the first implementation tranche can be selected with confidence.

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
7. Record the matrix and gap classification in a compact format ready for Step 03.

**Stop conditions:**

- **Done:** readiness matrix covers all primitives from the upstream plan with file citations; gap classification is owner-specific.
- **Hold:** a policy decision is needed before a gap can be classified.
- **Blocked:** an MCP/tool/agent gap prevents honest research; escalate to `00-helping`.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 03 — First tranche selection [PLANNED]

```yaml
phase: 1
step: 3
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Step 04 — Implementation handoff packet'
skills: 'nge-core-algorithm'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**Step objective:** Select the smallest aligned first implementation tranche based on the readiness matrix from Step 02. The tranche must close the highest-leverage NGE core gap rather than optimizing for any single benchmark.

**Stop conditions:**

- **Done:** tranche is named, scoped, and justified against the readiness matrix; one-line rationale per excluded candidate.
- **Hold:** a user priority decision is needed before the tranche can be finalized.
- **Blocked:** Step 02 matrix is incomplete; return to Step 02.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

#### Step 04 — Implementation handoff packet [PLANNED]

```yaml
phase: 1
step: 4
agent: '01-planning'
agent_file: '.github/agents/01-planning.agent.md'
status: '[PLANNED]'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md'
copy_paste: 'true'
next_step: 'Phase 2 — Core NGE implementation (steps TBD from tranche selection)'
skills: 'nge-core-algorithm'
validation:
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md
```

**Step objective:** Draft a compact execution-oriented handoff packet with scope, acceptance criteria, non-goals, open assumptions, and validation expectations ready for the `04-implementing` agent to consume as Phase 2 of this plan.

**Stop conditions:**

- **Done:** handoff packet is complete; Phase 2 step packets are drafted in this tracker and ready for copy-paste delegation.
- **Hold:** tranche selection from Step 03 is still open.
- **Blocked:** Step 03 output is incomplete.

**Required validation:**

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

## Validation gates

`node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md`

### Latest validation evidence

- (first run — see below)

## Handoff query

NGE core readiness audit — Phase 1 Step 02 is active. Run
`neataptic-workflow-mcp.get_active_workflow_snapshot` or open this file to
confirm the current step. The readiness matrix and gap classification (Step 02)
must complete before tranche selection (Step 03) can proceed. Racing benchmark
work should pause on any missing NGE primitive until the core tranche closes the
gap; any racing-exposed gap that is not demo-local routes back here first.
