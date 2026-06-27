# NGE Core Growth Engine Wiring

**Status:** [DONE]

## Scope

Bridge the two disconnected growth systems in the NGE (Neuro-Genesis Engine)
architecture so that focus-scored growth morphs flow from planning through
application to runtime adaptation as a single continuous pipeline.

Six audit gaps are addressed:

1. **[P0] No morph applier** — `planGrowthMorphs` produces `NgeMorphDelta[]`
   but nothing consumes it. No function translates deltas into
   `network.mutate()` calls.
2. **[P0] runNgeLifecycle has zero production callers** — the lifecycle runner
   only plans, never executes, and nothing calls it in production.
3. **[P0] adaptOnTick doesn't call runNgeLifecycle** — runtime adaptation uses
   random operations instead of NGE focus-scoring. Two disconnected systems.
4. **[CRITICAL] No morph-to-mutation mapping** — NgeMorphDelta kinds have no
   mapping to NEAT mutation methods.
5. **[HIGH] Runtime adaptation limits too low** — `maxNodes: 256`,
   `maxConnections: 1_024` vs. the target 8,000 nodes / 32,000+ connections.
6. **[HIGH] commitGrowth only updates hysteresis** — never called after actual
   morph application; hysteresis state drifts.

**Priority:** This plan takes PRIORITY over the Racing Curriculum demo. The
growth engine must be fully wired before racing curriculum work continues.

## User Vision

- Continuous real-time adaptation is the primary mode — not batch evolution.
- Browser cap: 8,000 nodes, 32,000+ connections.
- Growth is organic from a seed network — not pre-structured.
- The NGE lifecycle (focus scoring → morph planning → morph application →
  hysteresis commit) should be the single growth pipeline used by all
  adaptation surfaces.

## Non-Goals

- Removing or replacing the NGE focus-scoring or morph-planning logic — those
  are correct and well-tested. This plan wires the output, not the planner.
- Adding new morph kinds — the existing five kinds are sufficient.
- Changing the NEAT mutation handlers (`addNode`, `addConn`, `subConn`,
  `subNode`) — those are correct. This plan calls them correctly.
- Episodic slot allocation at the module/DNA level — `slotExpand` will be a
  documented no-op at the Network level until the episodic slot primitive
  exists. The plan records this explicitly.

## No Deferred Cleanup Policy

When bridging `adaptOnTick` to use `runNgeLifecycle` (Phase 3), the standalone
random operation proposal engine (`proposeCandidateOperations`,
`resolveStructuralPool`) MUST be removed in the same step that introduces the
NGE lifecycle bridge. No dual-path code, no backward-compatibility wrappers.

## Morph-to-Mutation Mapping

| Morph Kind  | NEAT Mutation | Direct `mutate()` OK?  | Notes                                                                         |
| ----------- | ------------- | ---------------------- | ----------------------------------------------------------------------------- |
| edgeDensify | `ADD_CONN`    | Yes (call N×)          | N = `detail.proposedAdditions` (currently 1)                                  |
| nodeAdd     | `ADD_NODE`    | Yes                    | `addNode` adds +1 connection (plan's `wiringCostDelta:0` is misleading)       |
| slotExpand  | (none)        | No — no-op             | Episodic slot primitive doesn't exist yet                                     |
| edgePrune   | `SUB_CONN`    | No — custom disconnect | `SUB_CONN` picks random edge; must disconnect `detail.candidateId` explicitly |
| compact     | `SUB_NODE`    | Partially              | `SUB_NODE` picks random node (plan has no target id)                          |

Budget enforcement: The applier MUST re-validate `NgeGrowthBudget` (maxNodes,
maxEdges) and `NgePruneBudget` (minEdges, minNodes) before mutating, because
the network's internal `ensureGrowthBudget` only tracks connection sparsity,
not NGE-level budgets.

---

## Implementation phases

### Phase 1 — Morph Applier [DONE]

```yaml
phase: 1
title: 'Morph Applier'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Growth_Engine_Wiring.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Lifecycle Wiring'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'phase-handoff-workflow'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Growth_Engine_Wiring.plans.md'
acceptance_criteria:
  - 'Morph applier function exists and translates all 5 NgeMorphDelta kinds to appropriate network mutations'
  - 'edgeDensify calls network.mutate(ADD_CONN) N times where N = detail.proposedAdditions'
  - 'nodeAdd calls network.mutate(ADD_NODE)'
  - 'edgePrune disconnects the specific connection identified by detail.candidateId (not random SUB_CONN)'
  - 'compact calls network.mutate(SUB_NODE)'
  - 'slotExpand is a documented no-op (episodic slot primitive does not exist yet)'
  - 'Applier re-validates NgeGrowthBudget and NgePruneBudget before mutating'
  - '100% coverage on all touched src/ files'
placeholder_steps:
  - 'Step 01 — Plan morph applier phase'
  - 'Step 02 — Red tests for morph applier'
  - 'Step 03 — Implement morph applier'
  - 'Step 04 — Green validation for morph applier'
```

**Phase objective:** Create the morph applier that translates `NgeMorphDelta[]` into `network.mutate()` calls, establishing the missing link between morph planning and network modification.

**Stop conditions:**

- **Done:** Morph applier function exists, handles all 5 morph kinds, passes red-green validation with 100% coverage.
- **Blocked:** Cannot disconnect specific connections by `candidateId` — investigate the Network API for connection removal.

**Required validation:**

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Growth_Engine_Wiring.plans.md`

---

[DONE] Phase 1 steps 01-04 compressed to logs. See `plans/NGE_Core_Growth_Engine_Wiring.logs.md` for detailed step packets and validation evidence.

---

### Phase 2 — Lifecycle Wiring [DONE]

**Phase objective:** Wire `runNgeLifecycle` to call `applyMorphDeltas` after `planGrowthMorphs`, and call `commitGrowth` after morph application so hysteresis state stays in sync.

[DONE] Phase 2: Implemented lifecycle wiring (plan → apply → commit), 8/8 tests pass, 100% coverage on all touched files, plan-sync gate passed. See `NGE_Core_Growth_Engine_Wiring.logs.md` for detailed step evidence.

---

### Phase 3 — Runtime Integration [DONE]

**Phase objective:** Bridge `adaptOnTick` to call `runNgeLifecycle` instead of using random operations, removing the standalone proposal engine in the same step (No Deferred Cleanup policy).

[DONE] Phase 3: Wired `adaptOnTick` to `runNgeLifecycle`, removed standalone proposal engine (`proposeCandidateOperations`, `resolveStructuralPool`, `applyOperations`), 21/21 tests pass (3 suites), tsc/lint/build all clean, plan-sync gate passed. See `NGE_Core_Growth_Engine_Wiring.logs.md` for detailed step evidence.

---

### Phase 4 — Capacity and Limits [DONE]

**Phase objective:** Raise runtime adaptation limits from 256 nodes / 1024
connections to 8000 nodes / 32000+ connections, and ensure growth throttling
preserves real-time performance at scale.

[DONE] Phase 4: Raised DEFAULT_LIMITS to 8000/32000, added NGE_MAX_NODE_CAPACITY/NGE_MAX_EDGE_CAPACITY constants, added computeGrowthThrottle for tick-based throttling, 8 capacity tests + 21 runtime adaptation tests pass, 100% coverage on src/ constants file, tsc/lint/build clean, all gates pass. See `plans/NGE_Core_Growth_Engine_Wiring.logs.md` for detailed step evidence.

---

### Phase 5 — End-to-End Growth Verification [DONE]

[DONE] Phase 5: 6 E2E growth tests created, all pass immediately confirming pipeline is fully wired. Step 18 skipped (no wiring gaps found). Step 19 green validation: 6/6 E2E growth tests, 8/8 capacity tests, 21/21 runtime adaptation tests pass, tsc/lint/build clean, all gates pass. See `plans/NGE_Core_Growth_Engine_Wiring.logs.md` for detailed step packets and validation evidence.

---

## Decision Records

```yaml
decision_record:
  id: 'DR-NGE-001'
  context: 'edgePrune morph kind has a candidateId but network.mutate(SUB_CONN) picks a random edge. Need custom disconnect logic.'
  options:
    - id: optA
      desc: 'Use network.disconnect(from, to) directly with the candidateId'
    - id: optB
      desc: 'Add a new mutation method SUB_CONN_BY_ID to the mutation handlers'
  chosen: optA
  rationale: 'Avoids changing the NEAT mutation handler API. The applier can look up the connection by candidateId and call disconnect directly.'
  owner: '01-planning'
  rollback_plan: 'If disconnect API is insufficient, fall back to optB and add SUB_CONN_BY_ID.'
  created_at: '2025-01-15T00:00:00Z'
```

```yaml
decision_record:
  id: 'DR-NGE-002'
  context: 'slotExpand morph kind has no NEAT mutation equivalent. The episodic slot primitive does not exist at the Network level.'
  options:
    - id: optA
      desc: 'Document slotExpand as a no-op in the morph applier'
    - id: optB
      desc: 'Create a stub episodic slot primitive in the Network API'
  chosen: optA
  rationale: 'Creating a stub would violate the Non-Goals. The slot primitive is a module/DNA level concern, not a Network level concern. Document it as a no-op until the primitive exists.'
  owner: '01-planning'
  rollback_plan: 'If episodic slots are needed sooner, create a separate plan for the slot primitive.'
  created_at: '2025-01-15T00:00:00Z'
```

```yaml
decision_record:
  id: 'DR-NGE-003'
  context: 'compact morph kind calls SUB_NODE which picks a random node. The morph delta has no target node id.'
  options:
    - id: optA
      desc: 'Accept random node removal for compact (use network.mutate(SUB_NODE))'
    - id: optB
      desc: 'Add targetNodeId to the compact morph delta detail'
  chosen: optA
  rationale: 'The compact morph is planned by the NGE focus-scoring logic which already selects what to compact. Adding targetNodeId would require changing the morph planner, which is a Non-Goal. Accept random removal within the focus-selected area.'
  owner: '01-planning'
  rollback_plan: 'If random removal causes issues, add targetNodeId in a follow-up plan.'
  created_at: '2025-01-15T00:00:00Z'
```

## Final state

The NGE growth pipeline is fully connected end-to-end:
`adaptOnTick` → `computeGrowthThrottle` → `runNgeLifecycle` → `computeFocusScores` → `planGrowthMorphs` → `applyMorphDeltas` → `commitGrowth`

**All files changed across 5 phases:**

- `src/neat/nge-juvenile/neat.nge-juvenile.apply.ts` (new — Phase 1: morph applier)
- `src/neat/nge-juvenile/neat.nge-juvenile.ts` (barrel export — Phase 1)
- `src/neat/nge-juvenile/neat.nge-juvenile.apply.test.ts` (new — Phase 1: 7 tests)
- `src/neat/neat.nge-lifecycle.ts` (Phase 2: wired applyMorphDeltas + commitGrowth)
- `src/neat/neat.nge-lifecycle.apply.test.ts` (Phase 2: lifecycle tests)
- `examples/racing_curriculum/controller/runtime.adaptation.ts` (Phase 3: rewired adaptOnTick to runNgeLifecycle, removed standalone proposal engine; Phase 4: raised DEFAULT_LIMITS, added throttle integration)
- `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts` (Phase 4: NGE_MAX_NODE_CAPACITY/NGE_MAX_EDGE_CAPACITY, updated growth budget defaults)
- `examples/racing_curriculum/controller/nge-e2e-growth.test.ts` (new — Phase 5: 6 E2E growth tests)

## Audit summary

- All 5 phases [DONE] and compressed to `NGE_Core_Growth_Engine_Wiring.logs.md`
- 50 tests across the growth pipeline, all passing
- tsc, lint, build:racing-curriculum all clean
- plan-sync gate: pass
- step-packet gate: pass
- plan-phase-packets validator: pass
- stale-wip-plans gate: pass
- log-completion-marker gate: pass

## Downstream Tracker Impacts

- **AntHive** — may benefit from the wired growth engine for colony scaling
- **PredatorPrey** — may benefit from NGE-driven adaptation
- **Racing Curriculum** — unblocked; `adaptOnTick` changes in Phase 3 are complete

## Reopen conditions

- If the episodic slot primitive is created and `slotExpand` needs a real implementation
- If `compact` random node removal causes issues and `targetNodeId` needs to be added to the morph delta
- If capacity limits (8000/32000) prove insufficient for a benchmark demo

## Audit log

- 2026-06-26: Phase 1 (Morph Applier) [DONE] [COMPRESSED]
- 2026-06-26: Phase 2 (Lifecycle Wiring) [DONE] [COMPRESSED]
- 2026-06-27: Phase 3 (Runtime Integration) [DONE] [COMPRESSED]
- 2026-06-27: Phase 4 (Capacity Limits & Throttling) [DONE] [COMPRESSED]
- 2026-06-27: Phase 5 (End-to-End Growth Verification) [DONE] [COMPRESSED]
- 2026-06-27: Plan archived to `plans/completed/`
