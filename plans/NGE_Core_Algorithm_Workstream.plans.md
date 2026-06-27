# NGE Core Algorithm Workstream

**Status:** [WIP]

## Scope

Fully complete the NGE (Neuro-Genesis Engine) core algorithm in `src/neat/` before any
further demo work. This is a **library-core workstream**, not an `examples/` workstream.
The racing curriculum, ant hive, and predator/prey demos are all blocked until this
workstream is complete.

This workstream owns the carry-forward blockers documented in the racing curriculum
logs (P1–P5, DR-2026-06-27-05 `modeIsEvolvable`) plus the user-reported growth gap:
agents only reached 101 nodes / 388 connections by end of simulation when the target is
organic growth from seed toward 8,000+ neurons with continuous real-time adaptation.

This plan is downstream of:

- `plans/completed/NEAT_Genesis_EvoDevo.md` (archived concept + architecture plan)
- `plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.plans.md` (prior readiness audit)
- `plans/completed/NGE_Core_Growth_Engine_Wiring.plans.md` (growth engine wiring — DONE but not producing expected growth)
- `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` (racing curriculum — [DONE], blocked on this workstream)

If any upstream plan conflicts with this one, the upstream plan wins on architecture,
this plan wins on the specific carry-forward blockers and the growth-gap diagnosis.

## User Vision

- **NGE must be FULLY complete before any more demo work.** Stop moving into demos with
  parts of NGE missing.
- Networks grow **organically from a seed** — not pre-structured.
- **Continuous real-time adaptation** is the primary mode, not batch-only evolution.
- Generations are for **multiplication and fusing successful networks** (one per lap in
  racing). Generations should NOT be mandatory for an agent to evolve — start small and
  grow and adapt.
- Inspired by ant brain architecture at a smaller scale: compact seed unfolding into a
  far larger grown network.
- Browser target cap: **8,000 nodes, 32,000+ connections**.
- **WebGPU acceleration** is a future phase (RTX 4070 Super available). High node counts
  (8,000+) will need GPU acceleration. This is noted in the roadmap as a future lane,
  not a blocker for this workstream.

## Carry-Forward Blockers (from racing curriculum logs)

- **P1 (CRITICAL): NGE_DNA adoption gap.** Racing uses `Network`, but polyandric
  reproduction needs `NgeDnaCanonicalEnvelope`. No bridge exists between the runtime
  `Network` phenotype and the canonical DNA envelope.
- **P2 (CRITICAL): Polyandric type exports missing.** `NgePolyandricInput` /
  `NgePolyandricDroneInput` are not exported from `reproduction.ts`. 3 polyandric tests
  remain skipped because of this.
- **P3: Racing FSM reproduction step is a placeholder.** The reproduction step in the
  racing FSM is not wired to a real polyandric call site.
- **P4: Schema mismatch.** Reference spec uses non-overlapping / queen-weighted region
  assignment; the implemented core uses `roundRobin` / `byFitness` / `bySpecialization`.
- **P5: `queenBias` not honored.** The `queenBias` parameter is not honored by the merge
  logic in the reproduction pipeline.
- **DR-2026-06-27-05: `modeIsEvolvable` is a dead boolean field.** No operator reads it.
  `ModulatorBroadcaster` / `EpisodicSlot` / `GatingRouter` are descriptor-only. No
  phenotype→Network bridge exists.

## Growth Gap (user observation)

- Agents only reached **101 nodes / 388 connections** by end of simulation.
- NGE was supposed to grow organically from seed toward 8,000+ neurons with continuous
  real-time adaptation.
- The growth engine wiring plan (`NGE_Core_Growth_Engine_Wiring`) is marked [DONE] — the
  morph applier, lifecycle wiring, runtime integration, capacity limits, and E2E
  verification were all implemented — yet observed growth stalled far below target.
- **The growth engine is clearly not working as intended.** This workstream must diagnose
  why and fix it so networks demonstrably grow from seed to 8,000+ neurons.

## Non-Goals

- Replacing the NGE focus-scoring or morph-planning logic unless the growth-gap diagnosis
  proves the planner itself is the defect.
- Changing classic NEAT behavior when NGE is disabled (opt-in isolation must hold).
- Demo-specific compensation. Demo work stays blocked until this workstream completes.
- WebGPU implementation (future lane, noted in roadmap only).

## No Deferred Cleanup Policy

When this workstream activates a previously dead field (e.g. `modeIsEvolvable`) or bridges
a previously disconnected system (phenotype → Network), the step MUST remove any
placeholder, no-op, or descriptor-only shim in the same step that introduces the real
operator. No backward-compatibility wrappers, no dual-path code, no deferred cleanup.

## Determinism Contract

- Same DNA + same seed + same experience stream MUST reproduce identical lifecycle
  checkpoints when the plan says it should.
- Same network + same inputs MUST produce bitwise-identical activation output.
- Reproduction-mode region assignment (polyandric) MUST be deterministic.

## Coverage Contract

- 100% statements, branches, functions, lines on all touched `src/` files.
- Classic NEAT opt-in isolation tests MUST remain green (NGE disabled = unchanged NEAT).

---

## Implementation phases

### Phase 1 — NGE_DNA Adoption & Canonical Envelope Bridge (P1) [DONE]

```yaml
phase: 1
title: 'NGE_DNA Adoption & Canonical Envelope Bridge (P1)'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Polyandric Reproduction Exports & Activation (P2, P5)'
skills:
  - 'plan-alignment'
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'A phenotype→Network bridge exists: NgeDnaCanonicalEnvelope materializes into a runtime Network'
  - 'A Network→canonical envelope round-trip preserves identity and substrate metadata'
  - 'Racing worker can construct a NgeDnaCanonicalEnvelope from a live Network without lossy ad-hoc conversion'
  - 'Determinism: same DNA + same seed produces identical Network topology'
  - 'Opt-in isolation: classic NEAT unchanged when NGE disabled'
  - '100% coverage on touched src/neat/ files'
placeholder_steps:
  - 'Step 01 — Plan Phase 1 and author Step 02-07 packets'
  - 'Step 02 — Research the NGE_DNA ↔ Network boundary'
  - 'Step 03 — Red tests for phenotype→Network bridge'
  - 'Step 04 — Implement the canonical envelope bridge'
  - 'Step 05 — Green validation and coverage guard'
  - 'Step 06 — Document the bridge contract'
  - 'Step 07 — Compress Phase 1 into logs'
```

**Phase objective:** Bridge the runtime `Network` phenotype to the canonical
`NgeDnaCanonicalEnvelope` so that polyandric reproduction (which requires the envelope)
can operate on racing agents that currently only hold a `Network`. This resolves P1.

> **Detailed Phase 1 step packets, research brief, validation evidence, and boundary map
> have been compressed to `plans/NGE_Core_Algorithm_Workstream.logs.md`.**

[DONE] Step 01: Plan Phase 1 — step packets authored, boundary map produced, step-packet gate passed.
[DONE] Step 02: Research — 14-section research brief produced; both bridge gaps confirmed (descriptor→Network, Network→envelope); genome↔Network precedent mapped.
[DONE] Step 03: Red tests — 18 failing tests created in `src/neat/nge-dna/neat.nge-dna.bridge.test.ts`; all failed with TS2307 (missing module).
[DONE] Step 04: Implementation — bridge module created (`materializeNetworkFromPhenotype` + `extractCanonicalEnvelopeFromNetwork`); `NGE_DNA_BridgeError` added; 20 tests pass; 100% coverage on bridge.ts + errors.ts.
[DONE] Step 05: Green validation — 301 tests across 10 suites pass, zero regressions; bridge.ts + errors.ts 100/100/100/100; tsc 0 diagnostics; lint 0 errors; plan-sync gate pass.
[DONE] Step 06: Documentation — JSDoc on bridge API, Mermaid diagram, academic citations; `npm run docs` exit 0; generated README reflects bridge module.
[DONE] Step 07: Phase compression — detailed content moved to logs file; plan file trimmed; phase-compression and step-packet gates validated.

**Artifacts:**

- `src/neat/nge-dna/neat.nge-dna.bridge.ts` — `materializeNetworkFromPhenotype` + `extractCanonicalEnvelopeFromNetwork`
- `src/neat/nge-dna/neat.nge-dna.errors.ts` — `NGE_DNA_BridgeError` class
- `src/neat/nge-dna/neat.nge-dna.bridge.test.ts` — 20 tests, 100% coverage
- `src/neat/nge-dna/README.md` — regenerated with bridge documentation, Mermaid, citations

### Phase 2 — Polyandric Reproduction Exports & Activation (P2, P5) [WIP]

```yaml
phase: 2
title: 'Polyandric Reproduction Exports & Activation (P2, P5)'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Phase 3 — modeIsEvolvable Activation & Phenotype→Network Operator'
skills:
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'NgePolyandricInput and NgePolyandricDroneInput are exported from reproduction.ts and re-exported via the neat.nge-evolution facade (importable from both paths)'
  - 'queenBias=1.0 preserves current behavior: every patched offspring region deep-equals the queen region (regression anchor)'
  - 'queenBias=0.0 → drone data overrides queen for every patched region (drone wins all conflicts)'
  - '0.0 < queenBias < 1.0 → deterministic per-region winner gate computed solely from regionId + queenBias (no external RNG); queen wins with proportion queenBias; a library test mirrors skip-contract #3 queenBias=0.85'
  - 'Determinism: two reproducePolyandric calls with identical queen + drones + queenBias + seed produce deep-equal offspring and regionAssignment'
  - 'No deferred cleanup: policy.queenBias is read in the merge path; no dead queenBias placeholder or unconditional queen-wins spread remains'
  - '100% statements/branches/functions/lines on touched src/neat/nge-evolution/ files'
  - 'Opt-in isolation: reproducePolyandric with ngeEnabled=false still throws NgeEvolution_ModeError; no file under examples/ is touched'
  - 'Scope boundary: the 3 racing-worker skip-contracts remain .skip (owned by Phase 7, blocked on P3 Phase 6 + P4 Phase 5)'
placeholder_steps:
  - 'Step 01 — Plan Phase 2 and author remaining step packets'
  - 'Step 02 — Research polyandric export surface and queenBias merge path'
  - 'Step 03 — Red tests for polyandric exports and queenBias honoring'
  - 'Step 04 — Export types, wire queenBias, activate polyandric path'
  - 'Step 05 — Green validation and coverage guard'
  - 'Step 06 — Document the polyandric contract'
  - 'Step 07 — Compress Phase 2 into logs'
```

**Phase objective:** Export the missing polyandric input types (P2) and make
`queenBias` an honored parameter in the merge/patch logic (P5), so the racing worker can
construct and call `reproducePolyandric` with typed inputs and queen-based biasing actually
affects reproduction outcomes. This is a **library-core `src/neat/` phase** — it does NOT
touch `examples/` or wire the racing FSM (that is P3, Phase 6).

**Stop conditions:**

- **Done:** Types exported + facade re-exported, queenBias honored at the three boundary values (1.0 / 0.0 / partial), library-level red/green tests pass, determinism verified, 100% coverage on touched `src/neat/nge-evolution/` files, opt-in isolation holds, no deferred cleanup.
- **Blocked:** If polyandric reproduction cannot operate without the Phase 1 canonical envelope bridge, reorder after Phase 1 (already DONE — prerequisite met).
- **Route-back:** If queenBias honoring requires `NgeAssignedRegionStrategy` schema changes (new strategy enum values), route to Phase 5 (P4) — do NOT fold schema extension into Phase 2.

**Required validation:**

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`

### Phase 2 Boundary Map (produced by Step 01 reconnaissance)

**Input type definition sites (export status):**

| Symbol                    | Defined at                                                     | Export status                  |
| ------------------------- | -------------------------------------------------------------- | ------------------------------ |
| `NgePolyandricDroneInput` | `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts:28` | NON-exported `interface`       |
| `NgePolyandricInput`      | `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts:35` | NON-exported `interface`       |
| `NgeParthenogenesisInput` | same file, L21                                                 | NON-exported (out of P2 scope) |
| `NgeSexualInput`          | same file, L43                                                 | NON-exported (out of P2 scope) |

**reproduction.ts export surface — has vs missing:**

- Exports ONLY: `reproduceParthenogenesis` (L79), `reproducePolyandric` (L143), `reproduceSexual` (L215).
- Missing exports: `NgePolyandricInput`, `NgePolyandricDroneInput` (the P2 gap).
- Facade `neat.nge-evolution.ts` re-exports the 3 functions and types from `./neat.nge-evolution.types`, but does NOT re-export the input interfaces (they are not exported from the source). P2 seam: add `export` to the two `interface` declarations + add a facade `export type { ... } from './neat.nge-evolution.reproduction'` line.

**queenBias — definition, default, honoring gap (P5):**

- Defined: `NgeReproductionPolicy.queenBias: number` in `src/neat/nge-dna/neat.nge-dna.types.ts` ("Bias toward queen dominance where 1 means queen wins all conflicts").
- Default: `NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS = 1.0` in `src/neat/nge-evolution/neat.nge-evolution.constants.ts` (re-exported via facade).
- Gap: `queenBias` is NEVER referenced in `neat.nge-evolution.reproduction.ts` (grep count = 0). It is a fully dead field in the reproduction pipeline.

**Reproduction pipeline flow (queenBias gap marked):**

```
reproducePolyandric(input)                                    [L143, exported]
  ├─ resolveOperatorPolicy(input.policy ?? queen.reproductionPolicy, 'polyandric')  [L146]
  │     ↳ returns full NgeReproductionPolicy INCLUDING queenBias
  │     ⚠ GAP: queenBias returned but never read downstream
  ├─ collectPolyandricRegionIds(input.queen)                  [L158]
  ├─ patchableRegionIds = queenRegionIds.slice(0, ceil(len * polyandricDroneContributionFraction))  [L159]
  ├─ eligibleDrones = input.drones.slice(0, polyandricDroneCount)  [L166]
  ├─ assignPolyandricRegions(patchableRegionIds, eligibleDrones, resolvedPolicy)  [L170]
  │     ↳ uses assignedRegionStrategy (roundRobin|byFitness|bySpecialization) — NOT queenBias
  ├─ applyPolyandricAssignments(input.queen, eligibleDrones, regionAssignment)  [L178]
  │     └─ patchPolyandricRegion(currentEnvelope, drone.dna, regionId)  [L315]
  │          ├─ moduleArchetypes: mergeModuleArchetypeWithQueenPriority(queen, drone)  [L564, L482]
  │          │    ⚠ GAP: hard {...drone, ...queen} = queen wins ALL (≡ queenBias=1.0 always)
  │          └─ cppnPrograms | rulePasses: {...droneRegion, ...queenRegion}  [L571-574]
  │               ⚠ GAP: hard queen-wins-all (≡ queenBias=1.0 always)
  ├─ buildCanonicalEnvelope(patched, { reproductionPolicy: resolvedPolicy })  [L177]
  └─ return { offspring, outcome:'queen-template-patched', parentContributions, policy, regionAssignment }
```

P5 seam: thread `resolvedPolicy.queenBias` from `reproducePolyandric` (L146) through `applyPolyandricAssignments` → `patchPolyandricRegion` → `mergeModuleArchetypeWithQueenPriority` and the non-archetype spread branch. Single-file internal plumbing edit.

**Carry-forward blockers affecting Phase 2:**

| Blocker                                                                                        | Phase          | Status | Phase 2 dependency    |
| ---------------------------------------------------------------------------------------------- | -------------- | ------ | --------------------- |
| P1 — phenotype→Network canonical envelope bridge                                               | Phase 1        | DONE   | Prerequisite MET.     |
| P2 — types not exported                                                                        | Phase 2 target | OPEN   | In scope.             |
| P5 — queenBias not honored                                                                     | Phase 2 target | OPEN   | In scope.             |
| P3 — racing FSM reproduction step is a placeholder                                             | Phase 6        | OPEN   | OUT of Phase 2 scope. |
| P4 — schema mismatch (non-overlapping/queen-weighted vs roundRobin/byFitness/bySpecialization) | Phase 5        | OPEN   | OUT of Phase 2 scope. |

### Phase 2 Decision Record

```yaml
decision_record:
  id: 'DR-2026-06-27-P2'
  context: >
    The Phase 2 phase-level packet originally listed "3 previously-skipped
    polyandric tests un-skip and pass" as an acceptance criterion. The 3
    skip-contracts live in examples/racing_curriculum/workers/simulation-worker/
    simulation-worker.race-pack.tier5.test.ts (L179, L188, L197). Reconnaissance
    shows they require P3 (racing FSM wiring — Phase 6) to select a queen and
    call reproducePolyandric, and test #3 also references assignedRegionStrategy
    =non-overlapping and seedPolicy=queen-weighted which are P4 (Phase 5) schema
    values that do not exist in the current NgeAssignedRegionStrategy enum.
    Phase 7 owns the final "All previously-skipped polyandric tests pass"
    criterion. Phase 2 is a library-core src/neat/ phase and must not touch
    examples/ or wire the racing FSM.
  options:
    - id: optA
      desc: 'Keep "un-skip 3 tests" in Phase 2 and force examples/ + FSM wiring into Phase 2 (scope creep into P3/P4).'
    - id: optB
      desc: 'Revise Phase 2 criteria to library-core only; add library-level queenBias=0.85 test mirroring skip-contract #3; leave the 3 example tests skipped for Phase 6/7.'
  chosen: optB
  rationale: >
    optB preserves the workstream scope boundary (library-core, no examples/
    until Phase 7), avoids conflating P2/P5 with P3/P4, and still delivers the
    spirit of skip-contract #3 (queenBias=0.85 honoring) at the library level
    where Phase 7 can lift it unchanged. optA would violate the "no demo work
    until Phase 7" rule and create a false-green gate.
  owner: '01-planning'
  rollback_plan: >
    Revert the Phase 2 acceptance_criteria block to the original 6-item list and
    delete DR-2026-06-27-P2. Only do this if the user explicitly wants Phase 2
    to absorb P3/P4 scope.
  created_at: '2026-06-27T15:32:49-04:00'
```

### Phase 2 Step Packets

#### Step 01: Plan Phase 2 — author Step 02-07 packets [DONE]

```yaml
phase: 2
step: 1
title: 'Plan Phase 2 — author Step 02-07 packets'
status: '[DONE]'
goal: 'planning'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 02 — Research polyandric export surface and queenBias merge path'
skills:
  - 'plan-alignment'
  - 'nge-core-algorithm'
specialists:
  - 'boundary-mapper'
  - 'acceptance-criteria-writer'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'Step 02-07 packets authored in the plan file'
  - 'Boundary map produced (input type sites, export surface, queenBias gap, pipeline flow, blocker dependencies)'
  - 'step-packet gate returns pass: true'
  - 'plan-sync gate returns pass: true'
```

**Step objective:** Activate Phase 2, produce the boundary map, record the scope-conflict
decision (DR-2026-06-27-P2), and author the remaining six step packets.

**Outcome:** Boundary map produced via Cortex RAG + boundary-mapper specialist. Acceptance
criteria authored via acceptance-criteria-writer specialist. The original "3 skipped tests
un-skip" criterion was revised to a library-core criterion set per DR-2026-06-27-P2. Step
02-07 packets authored below. Gates to be run after the plan edit.

#### Step 02: Research polyandric export surface and queenBias merge path [DONE]

```yaml
phase: 2
step: 2
title: 'Research polyandric export surface and queenBias merge path'
status: '[DONE]'
goal: 'researching'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 03 — Red tests for polyandric exports and queenBias honoring'
skills:
  - 'research-methodology'
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
specialists:
  - 'nge-core-scout'
  - 'determinism-scout'
validation:
  - 'Research brief committed to the plan file under Step 02 evidence'
acceptance_criteria:
  - 'Confirm NgePolyandricInput/NgePolyandricDroneInput non-export status and the exact facade re-export seam'
  - 'Confirm queenBias is dead in reproduction.ts (0 references) and identify the 3 helper functions that must read it'
  - 'Resolve open assumption A1: pick a deterministic string-hash of regionId → [0,1) for partial queenBias gating and record it in the determinism contract'
  - 'Resolve open assumption A2: decide whether queenBias gates the nested moduleArchetype.parameterSchema sub-merge or only the top-level region winner'
  - 'Resolve open assumption A4: decide clamp-vs-throw for queenBias outside [0,1]'
  - 'Confirm no other consumers depend on the non-exported status of the input types'
```

**User instruction:** Paste this full step packet.

**Step objective:** Produce a research brief that fixes the export seam, the queenBias merge
path, and the three open design assumptions (A1 deterministic weighting, A2 parameterSchema
gating, A4 clamp-vs-throw) so Step 03 red tests and Step 04 implementation have no ambiguity.

**Context the agent must know:**

- Phase 1 bridge is DONE — `NgeDnaCanonicalEnvelope` is constructable from a `Network`.
- `NgePolyandricInput` (L35) and `NgePolyandricDroneInput` (L28) are non-exported in `src/neat/nge-evolution/neat.nge-evolution.reproduction.ts`.
- `queenBias` is on `NgeReproductionPolicy` (default 1.0) but never read; the merge is hard queen-wins-all at `mergeModuleArchetypeWithQueenPriority` (L482) and `patchPolyandricRegion` non-archetype branch (L571).
- Determinism is mandatory: no external RNG, no Math.random, no Date.now.

**Execution steps:**

1. Use Cortex RAG to confirm the export surface and queenBias dead-field status.
2. Enumerate every function in the polyandric merge path and mark where queenBias must flow.
3. Choose a deterministic regionId→[0,1) hash for partial queenBias gating; document it.
4. Decide A2 (parameterSchema gating) and A4 (clamp-vs-throw).
5. Record the research brief and resolved assumptions in the plan under Step 02 evidence.

**Stop conditions:**

- **Done:** Research brief + resolved A1/A2/A4 recorded; next step unblocked.
- **Blocked:** If a consumer depends on the non-exported status, record it and escalate.

**Required validation:** Research brief committed to the plan file.

**Plan update requirement:** Update the plan with the research brief, resolved assumptions,
and Step 02 [DONE] marker before ending.

#### Step 02 Evidence — Research Brief: Polyandric Export Surface & queenBias Merge Path

**Evidence sources:** Cortex RAG (search_corpus, search_advanced, load_chunk, load_document —
all fresh, dense_state=warm, DiskANN active) + direct source verification of
`src/neat/nge-evolution/neat.nge-evolution.reproduction.ts` (full file, 711 lines),
`src/neat/nge-evolution/neat.nge-evolution.ts` (facade, L1–120),
`src/neat/nge-dna/neat.nge-dna.types.ts` (L37–82, L285–298),
`src/neat/nge-evolution/neat.nge-evolution.constants.ts` (L1–47),
`src/neat/nge-evolution/neat.nge-evolution.test.ts` (L538–800, L1650–1657),
`src/neat/nge-evolution/neat.nge-evolution.facade.test.ts` (L1–54),
`src/neat/nge-evolution/neat.nge-evolution.utils.ts` (L18–45),
`examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.tier5.test.ts`
(L158–204). No scout conflicts; all claims verified against static code.

##### 1. Full Polyandric Reproduction Pipeline Flow (with function signatures)

```
reproducePolyandric(input: NgePolyandricInput): NgeEvolutionReproductionResult    [L143, exported]
  │
  ├─ resolveOperatorPolicy(policy: NgeReproductionPolicy, mode): NgeReproductionPolicy  [L577, internal]
  │    ↳ returns { ...policy, mode } — INCLUDES queenBias but ⚠ NEVER READ DOWNSTREAM
  │
  ├─ collectPolyandricRegionIds(queen: NgeDnaCanonicalEnvelope): string[]  [L432, internal]
  │    ↳ produces region IDs: "cppnPrograms:0", "moduleArchetypes:0", "rulePasses:0", etc.
  │
  ├─ patchableRegionIds = queenRegionIds.slice(0, ceil(len * polyandricDroneContributionFraction))  [L159]
  ├─ eligibleDrones = input.drones.slice(0, polyandricDroneCount)  [L166]
  │
  ├─ assignPolyandricRegions(patchableRegionIds, drones, policy): NgeEvolutionPolyandricRegionAssignmentResult  [L324, internal]
  │    ├─ uses policy.assignedRegionStrategy ('roundRobin'|'byFitness'|'bySpecialization')  [L332]
  │    ├─ selectPolyandricDroneForRegion(regionId, index, drones, policy): NgePolyandricDroneInput | undefined  [L587]
  │    └─ returns { assignedRegions, patchableRegionIds, strategy, unassignedRegionIds }
  │
  ├─ applyPolyandricAssignments(queen, drones, regionAssignment): NgeDnaCanonicalEnvelope  [L304, internal]
  │    └─ regionAssignment.assignedRegions.reduce → patchPolyandricRegion(current, drone.dna, regionId)  [L315]
  │         ⚠ DOES NOT receive queenBias or resolvedPolicy — P5 gap
  │
  ├─ patchPolyandricRegion(queenEnvelope, droneEnvelope, regionId): NgeDnaCanonicalEnvelope  [L514, internal]
  │    ├─ parsePolyandricRegionId(regionId): { family, index }  [L496]
  │    ├─ if family === 'moduleArchetypes':
  │    │    └─ mergeModuleArchetypeWithQueenPriority(queenRegion, droneRegion): NgeDnaModuleArchetype  [L482]
  │    │         ⚠ HARD {...drone, ...queen} = queen wins ALL (≡ queenBias=1.0 always) — P5 gap
  │    │         ⚠ parameterSchema: {...dronePS, ...queenPS} = queen wins ALL sub-merge — P5 gap
  │    └─ else (cppnPrograms | rulePasses):
  │         └─ {...structuredClone(droneRegion), ...structuredClone(queenRegion)}  [L571-574]
  │              ⚠ HARD queen-wins-all spread (≡ queenBias=1.0 always) — P5 gap
  │
  ├─ buildCanonicalEnvelope(patched, { reproductionPolicy: resolvedPolicy }): NgeDnaCanonicalEnvelope  [L373, internal]
  └─ return { offspring, outcome:'queen-template-patched', parentContributions, policy, regionAssignment }
```

**P5 seam (single-file internal plumbing):** Thread `resolvedPolicy.queenBias` from
`reproducePolyandric` (L146) through `applyPolyandricAssignments` → `patchPolyandricRegion` →
`mergeModuleArchetypeWithQueenPriority` and the non-archetype spread branch. Three internal
function signatures gain a `queenBias: number` parameter; one gains `regionId: string` (already
available at the call site).

##### 2. Export Surface Analysis (P2 Gap)

**Input type definition sites (confirmed non-exported):**

| Symbol                    | File              | Line | Declaration                         | Export status                  |
| ------------------------- | ----------------- | ---- | ----------------------------------- | ------------------------------ |
| `NgePolyandricDroneInput` | `reproduction.ts` | L28  | `interface NgePolyandricDroneInput` | **NON-exported**               |
| `NgePolyandricInput`      | `reproduction.ts` | L35  | `interface NgePolyandricInput`      | **NON-exported**               |
| `NgeParthenogenesisInput` | `reproduction.ts` | L21  | `interface NgeParthenogenesisInput` | NON-exported (out of P2 scope) |
| `NgeSexualInput`          | `reproduction.ts` | L43  | `interface NgeSexualInput`          | NON-exported (out of P2 scope) |

**reproduction.ts exports (confirmed):** Only 3 functions — `reproduceParthenogenesis` (L79),
`reproducePolyandric` (L143), `reproduceSexual` (L215). No types are exported.

**Facade `neat.nge-evolution.ts` re-export surface:**

- Imports the 3 functions from `./neat.nge-evolution.reproduction` (L19–22).
- Re-exports the 3 functions as `export const` (L61–71).
- Re-exports types from `./neat.nge-evolution.types` (L24–44): `NgeEvolutionPolyandricAssignedRegion`,
  `NgeEvolutionPolyandricRegionAssignmentResult`, `NgeEvolutionReproductionResult`, etc.
- **Does NOT re-export** `NgePolyandricInput` or `NgePolyandricDroneInput` — they are not exported
  from the source module, so the facade cannot re-export them.

**Utils file `neat.nge-evolution.utils.ts`:** Imports only the 3 functions (L18–22), groups them
under `ngeEvolutionReproductionUtils` (L41–45). No type re-exports.

**P2 seam (two edits):**

1. Add `export` keyword to `interface NgePolyandricDroneInput` (L28) and `interface NgePolyandricInput` (L35) in `reproduction.ts`.
2. Add `export type { NgePolyandricInput, NgePolyandricDroneInput } from './neat.nge-evolution.reproduction';` to the facade `neat.nge-evolution.ts` (after L44).

**No consumer depends on the non-exported status:** The types are module-private; no external
file imports them. The test file (`neat.nge-evolution.test.ts`) constructs `reproducePolyandric`
calls with inline object literals that structurally match `NgePolyandricInput` — it does not
import the type. The racing worker skip-contracts (`simulation-worker.race-pack.tier5.test.ts`)
document the expected behavior but cannot construct typed inputs because of the P2 gap. Exporting
the types is a pure additive change with zero breaking risk.

##### 3. queenBias Definition, Default, and Honoring Gap (P5)

**Definition:** `NgeReproductionPolicy.queenBias: number` at
`src/neat/nge-dna/neat.nge-dna.types.ts` L75.
JSDoc: "Bias toward queen dominance where `1` means queen wins all conflicts."

**Full NgeReproductionPolicy interface (L65–82):**

```typescript
export interface NgeReproductionPolicy {
  mode: NgeReproductionPolicyMode; // L67
  parthenogenesisMutationRate: number; // L69
  polyandricDroneCount: number; // L71
  polyandricDroneContributionFraction: number; // L73
  queenBias: number; // L75  ← P5 target
  assignedRegionStrategy: NgeAssignedRegionStrategy; // L77
  modeIsEvolvable: boolean; // L79
  seedPolicy: NgeSeedPolicy; // L81
}
```

**Default:** `NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS = 1.0` in
`neat.nge-evolution.constants.ts` L46. Re-exported via facade L118–119.

**NgeAssignedRegionStrategy** (L46–49): `'roundRobin' | 'byFitness' | 'bySpecialization'`.
Note: `non-overlapping` and `queen-weighted` are P4 (Phase 5) values that do NOT exist in the
current enum — OUT of Phase 2 scope.

**Honoring gap (confirmed):** `queenBias` is NEVER referenced in `reproduction.ts`. The
`resolveOperatorPolicy` function (L577) returns the full policy including `queenBias`, but no
downstream function reads it. The merge is hard queen-wins-all at two sites:

1. `mergeModuleArchetypeWithQueenPriority` (L482–494): `{...drone, ...queen}` + parameterSchema sub-merge
2. `patchPolyandricRegion` non-archetype branch (L571–574): `{...droneRegion, ...queenRegion}`

Both are equivalent to `queenBias=1.0` always, regardless of the actual policy value.

##### 4. Current Merge Path — mergeModuleArchetypeWithQueenPriority (L482–494)

```typescript
function mergeModuleArchetypeWithQueenPriority(
  queenRegion: NgeDnaModuleArchetype,
  droneRegion: NgeDnaModuleArchetype,
): NgeDnaModuleArchetype {
  return {
    ...structuredClone(droneRegion),
    ...structuredClone(queenRegion), // ← queen wins ALL top-level keys
    parameterSchema: {
      ...(droneRegion.parameterSchema ?? {}),
      ...(queenRegion.parameterSchema ?? {}), // ← queen wins ALL parameterSchema keys
    },
  };
}
```

**Semantics:** Shallow merge with queen priority. Queen values override drone values for
overlapping keys. Drone-only keys (e.g., `drift: 5` when queen has no `drift`) survive. The
parameterSchema sub-merge follows the same queen-priority pattern.

**Existing test anchor (L539–612):** Test "keeps queen conflicts while patching non-overlapping
round-robin regions" confirms: `firstBias: 1` (queen wins), `firstDrift: 5` (drone-only property
survives), `secondBias: 2` (queen wins), `rulePassPriority: 3` (queen wins). This test uses
`createReproductionPolicy({ ... })` which defaults `queenBias` to
`NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS` (1.0) via the fixture spread (L1650–1657).

**P5 change required:** Add `queenBias: number` and `regionId: string` parameters. When
`queenBias < 1.0`, use a deterministic per-region winner gate to choose between queen-priority
merge and drone-priority merge (see §7 and §11).

##### 5. Current Merge Path — Non-archetype Spread Branch (L571–574)

```typescript
// In patchPolyandricRegion, for cppnPrograms and rulePasses families:
return resolvedRegionAccessor.write(queenEnvelope, {
  ...structuredClone(droneRegion),
  ...structuredClone(queenRegion), // ← queen wins ALL keys
} as never);
```

**Semantics:** Same queen-priority shallow merge as the archetype branch, but without a
parameterSchema sub-merge (cppnPrograms and rulePasses do not have parameterSchema).

**P5 change required:** Add `queenBias: number` and `regionId: string` parameters to
`patchPolyandricRegion`. When `queenBias < 1.0`, apply the same deterministic per-region winner
gate. Queen-priority: `{...drone, ...queen}`. Drone-priority: `{...queen, ...drone}`.

##### 6. NgeDnaModuleArchetype and parameterSchema Structure

**NgeDnaModuleArchetype** (`nge-dna.types.ts` L285–298):

```typescript
export interface NgeDnaModuleArchetype {
  archetypeId: string; // L287 — stable identity
  computationType: NeatGenomeComputationType; // L289 — computation motif
  receivesCoordinates?: boolean; // L291
  residualStreamId?: string; // L293
  weightSharedCohortId?: string; // L295
  parameterSchema?: Record<string, unknown>; // L297 — optional governance parameters
}
```

`parameterSchema` is an optional `Record<string, unknown>` representing archetype-local
governance parameters forwarded into the realized descriptor. It is the only nested object in
the archetype that gets a dedicated sub-merge in `mergeModuleArchetypeWithQueenPriority`.

##### 7. Assumption A1 Resolution — Deterministic regionId→[0,1) Hash

**Decision:** Use FNV-1a 32-bit hash of the `regionId` string, normalized to [0,1) by dividing
the unsigned 32-bit result by 4294967296 (2^32).

**Algorithm:**

```typescript
function hashRegionIdToUnitInterval(regionId: string): number {
  let hash = 0x811c9dc5; // FNV-1a 32-bit offset basis
  for (let i = 0; i < regionId.length; i++) {
    hash ^= regionId.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV-1a 32-bit prime
  }
  return (hash >>> 0) / 4294967296; // unsigned normalize → [0, 1)
}
```

**Properties:**

- **Deterministic:** Pure function of `regionId` string. No RNG, no `Date.now`, no external state.
- **Well-known:** FNV-1a is a standard non-cryptographic hash with uniform distribution.
- **Reproducible across runtimes:** Uses only `charCodeAt`, `Math.imul`, and bitwise ops —
  no platform-dependent primitives.
- **Uniform in [0,1):** The `>>> 0` converts to unsigned 32-bit; division by 2^32 maps to [0,1).
- **Gate semantics:** `hash < queenBias` → queen wins; `hash >= queenBias` → drone wins.
  - `queenBias=1.0`: `hash < 1.0` is always true (hash ∈ [0,1), 1.0 ∉ [0,1)) → queen always wins. ✓
  - `queenBias=0.0`: `hash < 0.0` is never true → drone always wins. ✓
  - `queenBias=0.85`: queen wins ~85% of regions, deterministically per regionId. ✓

**Determinism contract update:** The regionId→[0,1) hash is a pure function of the regionId
string using FNV-1a 32-bit. Same queen + same drones + same queenBias + same seed → identical
offspring and regionAssignment. The hash does NOT incorporate the seed — the seed is already
consumed upstream by `buildCanonicalEnvelope` and the region assignment is deterministic given
the queen envelope shape. The hash is only used for the queen/drone winner gate within each
patched region.

##### 8. Assumption A2 Resolution — parameterSchema Sub-merge Gating

**Decision:** **No independent parameterSchema gating.** The per-region winner gate determines
the merge priority for the entire region, including parameterSchema.

**Rationale:**

- parameterSchema is an optional `Record<string, unknown>` of governance parameters. Per-key
  independent gating would produce fragmented offspring where some parameterSchema keys come
  from queen and others from drone within the same archetype — this could create incoherent
  parameter combinations (e.g., queen's `learningRate` with drone's `momentum`).
- The region-level winner gate produces coherent offspring: when queen wins a region, all of
  queen's parameterSchema keys override drone's; when drone wins, all of drone's override queen's.
- This is simpler, avoids per-key hash complexity, and is consistent with the "whole region"
  concept that the patch pipeline already uses.
- The current parameterSchema sub-merge (`{...dronePS, ...queenPS}`) becomes the queen-priority
  path. The drone-priority path reverses to `{...queenPS, ...dronePS}`.

**Implementation:** When queen wins the region gate:

```typescript
{ ...structuredClone(droneRegion), ...structuredClone(queenRegion),
  parameterSchema: { ...(droneRegion.parameterSchema ?? {}), ...(queenRegion.parameterSchema ?? {}) } }
```

When drone wins the region gate:

```typescript
{ ...structuredClone(queenRegion), ...structuredClone(droneRegion),
  parameterSchema: { ...(queenRegion.parameterSchema ?? {}), ...(droneRegion.parameterSchema ?? {}) } }
```

##### 9. Assumption A3 Confirmation — Deep-equals for queenBias=1.0

**Previously RESOLVED:** Use deep-equals (not byte-identical) comparison for queenBias=1.0
regression tests.

**Confirmed:** `structuredClone` produces new object instances, so referential equality
(`===`) would fail. Deep-equals (`expect(...).toEqual(...)` in Jest) is the correct comparison
method. The existing test at L539 already uses `.toEqual()` for this purpose.

**Important nuance:** "queenBias=1.0 preserves current behavior" means the existing test
expectations (e.g., `firstBias: 1, firstDrift: 5`) remain the regression anchor. The current
behavior is a shallow merge with queen priority, NOT a clone of the queen region. Drone-only
properties survive. The acceptance criterion phrase "deep-equals the queen region" refers to
the deep-equal comparison method, not a literal clone of the queen region.

##### 10. Assumption A4 Resolution — Clamp vs Throw for queenBias outside [0,1]

**Decision:** **Clamp to [0,1].** Apply `Math.max(0, Math.min(1, queenBias))` at the point of
use (inside the merge functions, not at policy resolution).

**Rationale:**

- queenBias is a `number` field on a user-facing policy interface. Callers may pass values
  slightly outside [0,1] due to floating-point arithmetic or configuration errors.
- The merge path is deep in the reproduction pipeline. Throwing would abort reproduction
  mid-pipeline, losing the partially-patched envelope.
- Clamping to 0 (drone wins all) or 1 (queen wins all) is the natural extension of the bias
  semantics. A value of -0.5 → 0.0 (drone wins all); 1.5 → 1.0 (queen wins all).
- The determinism contract is preserved: clamping is a deterministic pure function.
- The plan's acceptance criteria specify behavior for 0.0, 1.0, and (0,1) — clamping extends
  these naturally to out-of-range values without requiring additional test cases.

**Implementation:** Clamp at the top of `patchPolyandricRegion` (or at the winner-gate site):

```typescript
const clampedQueenBias = Math.max(0, Math.min(1, queenBias));
```

##### 11. queenBias Merge Design — Per-Region Winner Gate

**Design:** For each patched region, compute `hashRegionIdToUnitInterval(regionId)`. If
`hash < clampedQueenBias`, use queen-priority merge (current behavior). If
`hash >= clampedQueenBias`, use drone-priority merge (reversed spread).

**queenBias=1.0 behavior preservation:** `hash ∈ [0,1)` and `1.0 ∉ [0,1)`, so
`hash < 1.0` is always true. Queen-priority merge is always used. The existing test at L539
continues to pass unchanged. ✓

**queenBias=0.0 behavior:** `hash < 0.0` is never true. Drone-priority merge is always used.
Drone wins all overlapping keys; queen-only keys survive. ✓

**queenBias=0.85 behavior:** ~85% of regions use queen-priority merge; ~15% use drone-priority.
The specific winners are deterministic per regionId. A library test can encode fixed expected
winners by computing `hashRegionIdToUnitInterval(regionId)` for known region IDs and asserting
the expected merge direction. ✓

**Signature changes (all internal, single-file):**

| Function                                | Current signature                   | New signature                                          |
| --------------------------------------- | ----------------------------------- | ------------------------------------------------------ |
| `applyPolyandricAssignments`            | `(queen, drones, regionAssignment)` | `(queen, drones, regionAssignment, queenBias: number)` |
| `patchPolyandricRegion`                 | `(queen, drone, regionId)`          | `(queen, drone, regionId, queenBias: number)`          |
| `mergeModuleArchetypeWithQueenPriority` | `(queen, drone)`                    | `(queen, drone, queenBias: number, regionId: string)`  |

**New internal helper:**

```typescript
function hashRegionIdToUnitInterval(regionId: string): number;
```

**Threading path:** `reproducePolyandric` (L146, has `resolvedPolicy.queenBias`) →
`applyPolyandricAssignments` (add `queenBias` param) →
`patchPolyandricRegion` (add `queenBias` param, already has `regionId`) →
`mergeModuleArchetypeWithQueenPriority` (add `queenBias` + `regionId` params) + non-archetype branch.

##### 12. Test Coverage Map

**Existing library tests** (`src/neat/nge-evolution/neat.nge-evolution.test.ts`):

| Test                                                                       | Line | queenBias     | What it asserts                                                             |
| -------------------------------------------------------------------------- | ---- | ------------- | --------------------------------------------------------------------------- |
| "keeps queen conflicts while patching non-overlapping round-robin regions" | L539 | 1.0 (default) | Queen wins overlapping keys, drone-only keys survive, region assignment IDs |
| "assigns donors by descending fitness when requested"                      | L615 | 1.0 (default) | byFitness drone ordering and rank                                           |
| "breaks equal-fitness ties by donor id"                                    | L670 | 1.0 (default) | Tie-break by parentId ascending                                             |
| "prefers specialization matches and falls back when none exist"            | L706 | 1.0 (default) | bySpecialization matching with fallback                                     |
| "throws a mode error when NGE is disabled"                                 | L766 | 1.0 (default) | NgeEvolution_ModeError when ngeEnabled=false                                |
| "reports unassigned regions when no drones are available"                  | L778 | 1.0 (default) | Unassigned region IDs when polyandricDroneCount=0                           |

**Existing facade tests** (`src/neat/nge-evolution/neat.nge-evolution.facade.test.ts`):

- L23–53: "bundles the public runtime surface" — checks function identity, does NOT test input types.

**Racing curriculum skip-contracts** (`examples/.../simulation-worker.race-pack.tier5.test.ts`):

- L179: `it.skip('selects the best-finishing car as queen for polyandric reproduction')` — blocked by P1/P2
- L188: `it.skip('calls reproducePolyandric with queen envelope and 2 drone envelopes')` — blocked by P1/P2
- L197: `it.skip('passes queenBias = 0.85 in the polyandric reproduction policy')` — blocked by P1/P2

**Step 03 red tests needed (library-level, in `src/neat/nge-evolution/`):**

1. Export visibility: import `NgePolyandricInput` and `NgePolyandricDroneInput` from both
   `./neat.nge-evolution.reproduction` and `./neat.nge-evolution` (facade) — fails to compile
   before Step 04 (TS2305/TS2497).
2. queenBias=1.0 regression anchor — queen wins all overlapping keys (mirror existing test L539).
3. queenBias=0.0 — drone wins all overlapping keys; queen-only keys survive (NEW — fails before Step 04).
4. queenBias=0.85 — deterministic winner map per FNV-1a hash contract (NEW — fails before Step 04).
5. Determinism — two identical calls produce deep-equal offspring + regionAssignment (NEW).
6. ngeEnabled=false throws NgeEvolution_ModeError (already green — keep as regression anchor).

##### 13. Implementation Recommendations for Step 03/04

**Step 03 (red tests):**

- Add tests to `src/neat/nge-evolution/neat.nge-evolution.test.ts` in the `reproducePolyandric`
  describe block (after L800) or create a dedicated `neat.nge-evolution.reproduction.queen-bias.test.ts`.
- For the queenBias=0.85 deterministic test, compute expected winners using the FNV-1a hash
  contract from §7. Use region IDs from a known queen envelope (e.g., the fixture from L540)
  and assert which regions have queen values vs drone values.
- For the export visibility test, use a type-level import assertion: `import type {
NgePolyandricInput, NgePolyandricDroneInput } from './neat.nge-evolution.reproduction'` and
  `import type { NgePolyandricInput, NgePolyandricDroneInput } from './neat.nge-evolution'`.
  These will fail with TS2305/TS2497 before Step 04.

**Step 04 (implementation):**

1. Add `export` to `interface NgePolyandricDroneInput` (L28) and `interface NgePolyandricInput` (L35).
2. Add `export type { NgePolyandricInput, NgePolyandricDroneInput } from './neat.nge-evolution.reproduction';`
   to facade `neat.nge-evolution.ts` after L44.
3. Add `hashRegionIdToUnitInterval` function (§7 algorithm).
4. Thread `queenBias` through `applyPolyandricAssignments` → `patchPolyandricRegion` →
   `mergeModuleArchetypeWithQueenPriority` and the non-archetype branch (§11 signatures).
5. Implement the per-region winner gate: queen-priority when `hash < clampedQueenBias`,
   drone-priority otherwise.
6. Clamp `queenBias` to [0,1] at the point of use (§10).
7. Remove the dead `queenBias` field status — it is now read in the merge path (no deferred cleanup).
8. Ensure 100% coverage on `reproduction.ts` (all new branches: queen wins, drone wins, clamp
   paths, hash function).

**No deferred cleanup:** The old unconditional queen-wins spread is replaced by the gated
merge. No backward-compatibility wrapper, no dual-path code. The `mergeModuleArchetypeWithQueenPriority`
function name remains appropriate (queen priority is the default at queenBias=1.0).

##### 14. Determinism Contract Update

**New determinism requirement for Phase 2:**

- The regionId→[0,1) hash uses FNV-1a 32-bit, a pure function of the regionId string.
- Same queen + same drones + same queenBias + same seed → identical offspring and regionAssignment.
- The hash does NOT incorporate the seed — the seed is consumed upstream by `buildCanonicalEnvelope`.
- The queen/drone winner gate is deterministic per regionId and queenBias value.
- No `Math.random`, `Date.now`, `performance.now`, or external RNG is used in the merge path.
- The clamp operation (`Math.max(0, Math.min(1, queenBias))`) is a deterministic pure function.

**Existing determinism primitives confirmed clean:**

- `reproducePolyandric` uses no nondeterministic primitives.
- `assignPolyandricRegions` uses deterministic strategies (roundRobin index modulo, byFitness
  sort with tie-break, bySpecialization matching with fallback).
- `structuredClone` is deterministic (deep copy, no RNG).
- The sexual reproduction path uses `createDefaultSexualRandomGenerator` which returns a
  constant `DEFAULT_SEXUAL_RANDOM_SAMPLE = 0.75` — deterministic but OUT of Phase 2 scope.

[DONE] Step 02: Research — 14-section research brief produced; export surface confirmed (P2);
queenBias dead-field confirmed (P5); 4 assumptions resolved (A1: FNV-1a hash, A2: no independent
parameterSchema gating, A3: deep-equals confirmed, A4: clamp to [0,1]); pipeline flow mapped;
test coverage mapped; implementation recommendations for Step 03/04 authored.

#### Step 03: Red tests for polyandric exports and queenBias honoring [DONE]

```yaml
phase: 2
step: 3
title: 'Red tests for polyandric exports and queenBias honoring'
status: '[DONE]'
goal: 'red-testing'
tdd_sequence: 'red-green'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 04 — Export types, wire queenBias, activate polyandric path'
skills:
  - 'red-test-contracts'
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-evolution'
acceptance_criteria:
  - 'A library test imports NgePolyandricInput and NgePolyandricDroneInput from ./neat.nge-evolution.reproduction AND from the facade ./neat.nge-evolution — fails to compile before Step 04 (TS2305/TS2497)'
  - 'A red test asserts queenBias=1.0 yields queen-wins-all offspring (current behavior anchor)'
  - 'A red test asserts queenBias=0.0 yields drone-wins-all offspring — fails before Step 04 (currently queen always wins)'
  - 'A red test asserts queenBias=0.85 yields a deterministic winner map per the Step 02 hash contract — fails before Step 04'
  - 'A red test asserts determinism: two identical calls produce deep-equal offspring + regionAssignment'
  - 'A red test asserts ngeEnabled=false throws NgeEvolution_ModeError (already green — keep as regression anchor)'
  - 'All new red tests fail for the right reason (missing export / missing queenBias honoring), not syntax/fixture errors'
```

**User instruction:** Paste this full step packet.

**Step objective:** Create failing library-level tests in `src/neat/nge-evolution/` that define
the expected export visibility and queenBias honoring behavior. These replace the placeholder
skip-contracts at the library level (the example skip-contracts stay skipped per DR-2026-06-27-P2).

**Context the agent must know:**

- Tests live in `src/neat/nge-evolution/` (co-located with the reproduction boundary). Do NOT touch `examples/`.
- Use the existing `src/neat/nge-evolution/neat.nge-evolution.test.ts` polyandric section (around L538) as the fixture precedent for constructing `NgeDnaCanonicalEnvelope` queen/drone inputs.
- The Step 02 hash contract for partial queenBias must be encoded as fixed expected winners.

**Execution steps:**

1. Add a new test file or extend the existing polyandric test section with export-visibility tests.
2. Add queenBias=1.0 / 0.0 / 0.85 / determinism tests.
3. Run the targeted suite and confirm the new tests fail for the right reason.
4. Record the failing-test evidence in the plan.

**Stop conditions:**

- **Done:** New red tests exist and fail for the right reason; evidence recorded.
- **Route-back:** If a test fails for a syntax/fixture reason, fix the fixture and re-run.

**Required validation:** `npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-evolution`

**Plan update requirement:** Record failing-test evidence and Step 03 [DONE] marker.

**Step 03 red evidence (recorded):**

Files created:

- `src/neat/nge-evolution/neat.nge-evolution.polyandric-exports.test.ts` — 4 export visibility tests (P2 gap)
- `src/neat/nge-evolution/neat.nge-evolution.reproduction.queen-bias.test.ts` — 7 queenBias honoring tests (P5 gap)

Red results:

- File 1 (polyandric-exports): 4 TS compilation errors (TS2459 x2 for non-exported interfaces in reproduction.ts, TS2614 x2 for missing re-exports in facade). 0 tests run — compilation failure IS the red contract.
- File 2 (queen-bias): 4 failed, 3 passed (7 total). 3 regression anchors pass (queenBias=1.0, 1.5 clamped, determinism). 4 red contracts fail (queenBias=0.0, 0.5, 0.85, -0.3 clamped) because queenBias is currently ignored — queen always wins.

Fixture/cleanup notes:

- Shared fixture uses marker keys (queenTrait/droneTrait) for non-overlapping survival assertions
- queenBias=0.85 test requires 101 moduleArchetypes (index 100 has FNV-1a hash 0.939 >= 0.85, first region with hash >= 0.85)
- All fixtures use createDnaEnvelope/createModuleArchetype/createRulePass/createCppnProgram helpers replicated from existing test file
- Deterministic: no random seeds, all values are static

Validation commands run:

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neat.nge-evolution.reproduction.queen-bias` → 4 failed, 3 passed
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=neat.nge-evolution.polyandric-exports` → 0 tests run, 4 TS errors

Expected green condition for Step 04:

- File 1: All 4 tests pass (interfaces exported from reproduction.ts, re-exported via facade)
- File 2: All 7 tests pass (queenBias honored with FNV-1a per-region winner gate, clamped to [0,1])

#### Step 04: Export types, wire queenBias, activate polyandric path [PLANNED]

```yaml
phase: 2
step: 4
title: 'Export types, wire queenBias, activate polyandric path'
status: '[PLANNED]'
goal: 'implementing'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 05 — Green validation and coverage guard'
skills:
  - 'implementation-standards'
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-evolution'
  - 'npx tsc --noEmit'
acceptance_criteria:
  - 'NgePolyandricInput and NgePolyandricDroneInput are exported from reproduction.ts and re-exported via the facade'
  - 'queenBias is read in patchPolyandricRegion / mergeModuleArchetypeWithQueenPriority merge path (no dead path remains)'
  - 'queenBias=1.0 → queen wins all (deep-equal to pre-Phase-2 baseline)'
  - 'queenBias=0.0 → drone wins all patched regions'
  - '0.0 < queenBias < 1.0 → deterministic per-region gate per the Step 02 hash contract'
  - 'No deferred cleanup: the old unconditional {...drone, ...queen} spread is replaced, not wrapped'
  - 'All Step 03 red tests pass'
```

**User instruction:** Paste this full step packet.

**Step objective:** Implement P2 (export the two input interfaces + facade re-export) and P5
(thread queenBias through the merge path with deterministic gating), removing the old
hard-queen-wins spread in the same step (no deferred cleanup).

**Context the agent must know:**

- The two slices are independent enough to run sequentially; the export slice is tiny and the queenBias slice is the substantive change.
- Determinism is mandatory — use the Step 02 hash contract verbatim.
- Do NOT touch `examples/`, `NgeAssignedRegionStrategy`, `reproduceSexual`, or `reproduceParthenogenesis`.

**Execution steps:**

1. Slice 04-export-types: add `export` to the two interface declarations (L28, L35) and the facade re-export line.
2. Slice 04-wire-queenBias: thread `resolvedPolicy.queenBias` through the three helpers and implement the deterministic gate.
3. Run the targeted suite + tsc after each slice.

**Stop conditions:**

- **Done:** Both slices pass; all Step 03 red tests green; tsc clean; no dual-path code.
- **Route-back:** If queenBias honoring requires a new strategy enum value, route to Phase 5 (P4).

**Required validation:** `npx jest --config=jest.config.mjs --no-cache --testPathPattern=src/neat/nge-evolution` and `npx tsc --noEmit`.

**Plan update requirement:** Record implementation evidence, slice statuses, and Step 04 [DONE] marker.

#### Step 05: Green validation and coverage guard [PLANNED]

```yaml
phase: 2
step: 5
title: 'Green validation and coverage guard'
status: '[PLANNED]'
goal: 'green-testing'
tdd_sequence: 'red-green'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 06 — Document the polyandric contract'
skills:
  - 'green-testing-standards'
  - 'coverage-guard'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --coverage --testPathPattern=src/neat/nge-evolution --collectCoverageFrom=src/neat/nge-evolution/**/*.ts'
  - 'npx eslint src/neat/nge-evolution'
acceptance_criteria:
  - 'All Step 03/04 tests remain green; zero regressions in the nge-evolution suite'
  - '100% statements/branches/functions/lines on touched src/neat/nge-evolution/ files'
  - 'Opt-in isolation: ngeEnabled=false still throws; no examples/ file touched'
  - 'Lint exit code 0 on touched files'
  - 'Scope boundary confirmed: the 3 racing-worker skip-contracts remain .skip'
```

**User instruction:** Paste this full step packet.

**Step objective:** Validate the Phase 2 implementation with broader coverage and confirm no
regressions, no scope creep into examples/, and 100% coverage on touched files.

**Context the agent must know:**

- Run targeted coverage on `src/neat/nge-evolution/` only — do NOT run the full test suite unprompted.
- Confirm via `git diff --name-only` that no `examples/` path was touched and the 3 skip-contracts are still `.skip`.

**Execution steps:**

1. Run the targeted coverage suite.
2. Verify 100% coverage on touched files; add tests for any uncovered branch.
3. Run lint on touched files.
4. Confirm scope boundary (examples/ untouched, skip-contracts intact).

**Stop conditions:**

- **Done:** Coverage 100% on touched files, lint clean, no regressions, scope boundary confirmed.
- **Route-back:** If a regression appears, route back to Step 04 with the failure evidence.

**Required validation:** Coverage + lint commands above.

**Plan update requirement:** Record coverage/lint evidence and Step 05 [DONE] marker.

#### Step 06: Document the polyandric contract [PLANNED]

```yaml
phase: 2
step: 6
title: 'Document the polyandric contract'
status: '[PLANNED]'
goal: 'documenting'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Step 07 — Compress Phase 2 into logs'
skills:
  - 'educational-docs'
  - 'nge-core-algorithm'
validation:
  - 'npm run docs'
  - 'npm run lint'
acceptance_criteria:
  - 'JSDoc on NgePolyandricInput/NgePolyandricDroneInput and the queenBias merge path'
  - 'Mermaid diagram of the polyandric reproduction pipeline with queenBias gating'
  - 'Academic citations for polyandric reproduction / queen-bias biology where applicable'
  - 'npm run docs exit 0; generated README reflects the exported types and queenBias contract'
```

**User instruction:** Paste this full step packet.

**Step objective:** Document the polyandric reproduction contract — exported input types,
queenBias semantics (1.0 / 0.0 / partial deterministic gating), and the pipeline flow — in
JSDoc, Mermaid, and the generated README.

**Context the agent must know:**

- Document at the library level only (src/neat/nge-evolution/README.md regeneration via `npm run docs`).
- Cite prior art for polyandric / queen-bias concepts where the docs reference them.

**Execution steps:**

1. Add/update JSDoc on the exported interfaces and the queenBias merge functions.
2. Add a Mermaid diagram of the pipeline with the queenBias gate.
3. Run `npm run docs` and verify the generated README reflects the changes.

**Stop conditions:**

- **Done:** JSDoc + Mermaid + citations present; `npm run docs` exit 0.
- **Route-back:** If docs generation fails, fix the doc source and re-run.

**Required validation:** `npm run docs` and `npm run lint`.

**Plan update requirement:** Record docs evidence and Step 06 [DONE] marker.

#### Step 07: Compress Phase 2 into logs [PLANNED]

```yaml
phase: 2
step: 7
title: 'Compress Phase 2 into logs'
status: '[PLANNED]'
goal: 'logging'
tdd_sequence: 'green-only'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_step: 'Phase 3 Step 01 — Plan Phase 3 and author remaining step packets'
skills:
  - 'tracker-handoff'
  - 'summarizing-session-log'
validation:
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'Phase 2 detailed step/slice/VALIDATION_EVIDENCE blocks moved to plans/NGE_Core_Algorithm_Workstream.logs.md'
  - 'Plan file Phase 2 section replaced with compact [DONE] marker + logs reference'
  - 'Phase 2 header, goal, and status [DONE] retained in the plan file'
  - 'phase-compression gate returns pass: true'
  - 'step-packet gate returns pass: true'
```

**User instruction:** Paste this full step packet.

**Step objective:** Compress the completed Phase 2 history into the logs file and leave a
compact [DONE] marker in the plan file, then advance to Phase 3.

**Context the agent must know:**

- All Phase 2 steps must be [DONE] and green validation passed before compression.
- Move verbose transcripts to `plans/NGE_Core_Algorithm_Workstream.logs.md`; keep only the phase header + [DONE] summary in the plan.

**Execution steps:**

1. Move detailed Phase 2 content to the logs file.
2. Replace the plan Phase 2 section with a compact [DONE] marker + reference.
3. Run the phase-compression and step-packet gates.
4. Update the plan "Current state" and "Handoff query" to point at Phase 3.

**Stop conditions:**

- **Done:** Compression complete; gates pass; plan points at Phase 3.
- **Blocked:** If any Phase 2 step is not [DONE], route back to that step first.

**Required validation:** `phase-compression.gate.mjs --json` and `validate-plan-phase-packets.mjs --json`.

**Plan update requirement:** Record compression evidence, advance active phase to Phase 3, refresh the Handoff query.

### Phase 3 — modeIsEvolvable Activation & Phenotype→Network Operator (DR-2026-06-27-05) [PLANNED]

```yaml
phase: 3
title: 'modeIsEvolvable Activation & Phenotype→Network Operator (DR-2026-06-27-05)'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Phase 4 — Growth Engine Diagnosis & Fix (101 → 8,000+ nodes)'
skills:
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'A real operator reads modeIsEvolvable and activates the NGE evolution path when true'
  - 'The dead boolean field is removed or replaced by the operator (no dual-path)'
  - 'A phenotype→Network bridge operator materializes a Network from the canonical envelope'
  - 'ModulatorBroadcaster / EpisodicSlot / GatingRouter are either activated or explicitly scoped as future work with a recorded blocker'
  - 'Determinism: modeIsEvolvable=true with same DNA + seed produces identical activation'
  - '100% coverage on touched src/neat/ files'
placeholder_steps:
  - 'Step 01 — Plan Phase 3 and author remaining step packets'
  - 'Step 02 — Research modeIsEvolvable call sites and descriptor-only primitives'
  - 'Step 03 — Red tests for the modeIsEvolvable operator'
  - 'Step 04 — Implement the operator and phenotype→Network bridge'
  - 'Step 05 — Green validation and coverage guard'
  - 'Step 06 — Document the activation contract'
  - 'Step 07 — Compress Phase 3 into logs'
```

**Phase objective:** Activate the dead `modeIsEvolvable` boolean field with a real
operator that reads it and bridges phenotype to `Network`. This resolves
DR-2026-06-27-05 and unblocks the descriptor-only neuromodulation primitives.

**Stop conditions:**

- **Done:** Real operator reads modeIsEvolvable, phenotype→Network bridge exists, dead field removed or replaced, coverage gate passes.
- **Blocked:** If the neuromodulation primitives (ModulatorBroadcaster/EpisodicSlot/GatingRouter) require their own dedicated phase, record a decision and scope them as a follow-up.
- **Route-back:** If the phenotype→Network bridge depends on Phase 1's canonical envelope, reorder after Phase 1.

**Required validation:** `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`

### Phase 4 — Growth Engine Diagnosis & Fix (101 → 8,000+ nodes) [PLANNED]

```yaml
phase: 4
title: 'Growth Engine Diagnosis & Fix (101 → 8,000+ nodes)'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Phase 5 — Schema Alignment: Core NGE ↔ Racing Worker (P4)'
skills:
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
  - 'performance-optimization'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'Root cause identified: why agents stall at ~101 nodes despite the growth engine wiring being DONE'
  - 'Growth engine produces continuous organic growth from seed toward the 8,000-node cap'
  - 'Continuous real-time adaptation works without requiring a generation boundary'
  - 'Generations multiply and fuse successful networks but are NOT mandatory for an agent to grow'
  - 'Determinism: same DNA + seed + experience stream produces identical growth checkpoints'
  - '100% coverage on touched src/neat/ files'
placeholder_steps:
  - 'Step 01 — Plan Phase 4 and author remaining step packets'
  - 'Step 02 — Diagnose the growth stall (instrument morph applier, lifecycle, adaptOnTick, budgets)'
  - 'Step 03 — Red tests for continuous growth toward 8,000 nodes'
  - 'Step 04 — Fix the growth pipeline so organic growth reaches target'
  - 'Step 05 — Green validation with a growth-curve verification harness'
  - 'Step 06 — Document the growth contract and tuning knobs'
  - 'Step 07 — Compress Phase 4 into logs'
```

**Phase objective:** Diagnose why the growth engine — despite the
`NGE_Core_Growth_Engine_Wiring` plan being [DONE] — only grew agents to 101 nodes / 388
connections, and fix it so networks demonstrably grow from seed toward 8,000+ neurons with
continuous real-time adaptation. Generations are for multiplication and fusing successful
networks, not a prerequisite for growth.

**Stop conditions:**

- **Done:** Root cause identified, growth engine produces continuous organic growth toward 8,000-node cap, continuous adaptation works without generation boundaries, coverage gate passes.
- **Blocked:** If the root cause is in the NGE focus-scoring planner itself (not the wiring), record a decision and scope a planner fix.
- **Route-back:** If the growth stall is caused by the missing phenotype→Network bridge (Phase 3), reorder after Phase 3.

**Required validation:** `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`

### Phase 5 — Schema Alignment: Core NGE ↔ Racing Worker (P4) [PLANNED]

```yaml
phase: 5
title: 'Schema Alignment: Core NGE ↔ Racing Worker (P4)'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Phase 6 — Reproduction FSM Integration (P3)'
skills:
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'Core NGE region-assignment schema and racing worker schema agree (or a documented, owned adapter exists)'
  - 'Reference spec non-overlapping / queen-weighted semantics are either implemented or explicitly superseded with a recorded decision'
  - 'No silent schema drift between src/neat/nge-evolution/ and examples/racing_curriculum/'
  - 'Determinism: schema-aligned reproduction produces identical results in core and worker'
  - '100% coverage on touched src/neat/ files'
placeholder_steps:
  - 'Step 01 — Plan Phase 5 and author remaining step packets'
  - 'Step 02 — Research the schema mismatch between core NGE types and racing worker types'
  - 'Step 03 — Red tests for schema alignment'
  - 'Step 04 — Align schemas (remove old mismatched code in the same step)'
  - 'Step 05 — Green validation and coverage guard'
  - 'Step 06 — Document the aligned schema contract'
  - 'Step 07 — Compress Phase 5 into logs'
```

**Phase objective:** Resolve the schema mismatch (P4) between core NGE
region-assignment types and the racing worker types so reproduction results are
consistent across core and worker.

**Stop conditions:**

- **Done:** Schema aligned or documented adapter exists, no silent drift, determinism verified, coverage gate passes.
- **Blocked:** If schema alignment requires changing the reference spec semantics, record a decision with the user.
- **Route-back:** If the mismatch is actually a polyandric type export issue (Phase 2), merge into Phase 2.

**Required validation:** `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`

### Phase 6 — Reproduction FSM Integration (P3) [PLANNED]

```yaml
phase: 6
title: 'Reproduction FSM Integration (P3)'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Phase 7 — Verification: Seed → 8,000+ Neurons with Continuous Adaptation'
skills:
  - 'nge-core-algorithm'
  - 'nge-benchmark-workflow'
  - 'reproducibility-contracts'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'Racing FSM reproduction step is wired to a real polyandric call site (no placeholder)'
  - 'Placeholder reproduction step is removed in the same step (no dual-path)'
  - 'Reproduction produces a valid offspring Network from queen + drones'
  - 'Determinism: same queen + drones + seed produces identical offspring across FSM runs'
  - '100% coverage on touched src/neat/ and examples/racing_curriculum/ files'
placeholder_steps:
  - 'Step 01 — Plan Phase 6 and author remaining step packets'
  - 'Step 02 — Research the racing FSM reproduction step and call site'
  - 'Step 03 — Red tests for FSM reproduction integration'
  - 'Step 04 — Wire the FSM reproduction step to reproducePolyandric'
  - 'Step 05 — Green validation and coverage guard'
  - 'Step 06 — Document the FSM reproduction contract'
  - 'Step 07 — Compress Phase 6 into logs'
```

**Phase objective:** Replace the placeholder reproduction step in the racing FSM with a
real polyandric reproduction call site (P3). This is the integration point where core NGE
reproduction meets the racing benchmark.

**Stop conditions:**

- **Done:** FSM reproduction step wired to real polyandric call, placeholder removed, offspring Network valid, coverage gate passes.
- **Blocked:** If FSM integration requires Phase 1 (canonical envelope) and Phase 2 (polyandric exports) to be complete first, reorder after both.
- **Route-back:** If the FSM reproduction step exposes a new core NGE bug, route back to the relevant prior phase.

**Required validation:** `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`

### Phase 7 — Verification: Seed → 8,000+ Neurons with Continuous Adaptation [PLANNED]

```yaml
phase: 7
title: 'Verification: Seed → 8,000+ Neurons with Continuous Adaptation'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NGE_Core_Algorithm_Workstream.plans.md'
copy_paste: true
next_phase: 'Archive — NGE core complete; unblock racing curriculum v2 and demo lanes'
skills:
  - 'nge-core-algorithm'
  - 'reproducibility-contracts'
  - 'performance-optimization'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md'
acceptance_criteria:
  - 'NGE demonstrably grows a network from seed to 8,000+ neurons with continuous real-time adaptation'
  - 'Growth is organic and does not require a generation boundary to progress'
  - 'Generations multiply and fuse successful networks (one per lap equivalent)'
  - 'Polyandric reproduction produces valid offspring that continue growing'
  - 'Determinism: same DNA + seed + experience stream reproduces identical growth curve'
  - 'Opt-in isolation: classic NEAT unchanged when NGE disabled'
  - 'All touched src/neat/ files at 100% coverage'
  - 'All previously-skipped polyandric tests pass'
placeholder_steps:
  - 'Step 01 — Plan Phase 7 and author remaining step packets'
  - 'Step 02 — Research the verification harness boundary (growth-curve + reproduction + determinism)'
  - 'Step 03 — Red tests for the end-to-end growth + reproduction verification'
  - 'Step 04 — Implement the verification harness and any final fixes'
  - 'Step 05 — Green validation: seed→8,000+ neurons demonstrated and captured'
  - 'Step 06 — Document the NGE core completion contract'
  - 'Step 07 — Compress Phase 7 into logs and close the workstream'
```

**Phase objective:** Prove end-to-end that NGE grows a network from seed to 8,000+ neurons
with continuous real-time adaptation, that generations multiply and fuse successful
networks, and that polyandric reproduction produces valid offspring. This is the
workstream's completion gate: NGE is fully complete only when this is demonstrated.

**Stop conditions:**

- **Done:** Seed→8,000+ neurons demonstrated, continuous adaptation verified, polyandric offspring valid, all skipped tests passing, coverage gate passes, workstream ready for closure.
- **Blocked:** If 8,000+ neurons cannot be reached on CPU/worker without WebGPU acceleration, record a decision and adjust the target for the CPU verification phase.
- **Route-back:** If any prior phase's work is found incomplete, route back to that phase.

**Required validation:** `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md`

## Future Lane — WebGPU Acceleration (not blocking)

The user plans WebGPU acceleration for high node counts (8,000+ neurons) on an
RTX 4070 Super GPU. This is a **future lane**, noted in the roadmap, and is NOT a
blocker for this workstream. It should be planned as a separate workstream after NGE
core completion and after the growth engine is demonstrably reaching 8,000+ nodes on
CPU/worker first.

## Validation gates

The following gates apply to this workstream:

- **plan-sync** — after any plan status change.
- **step-packet** — after authoring or revising step packets.
- **agent-graph** — after any agent delegation change.
- **phase-compression** — after marking a phase [DONE] but before advancing.
- **log-completion-marker** — when the workstream is closing.
- **stale-wip-plans** — when the plan is closed or archived.

Gate commands:

```bash
neataptic-gate-mcp:run_gate_check --gate=plan-sync --json
neataptic-gate-mcp:run_gate_check --gate=step-packet --json
neataptic-gate-mcp:run_gate_check --gate=agent-graph --json
node scripts/agent-customization/gates/phase-compression.gate.mjs --json
node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json
node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json
```

## Current state

- Phase 1 [DONE]: NGE_DNA Adoption & Canonical Envelope Bridge (P1) — all 7 steps complete.
  Bridge module created, 20 tests pass, 301 tests across 10 suites pass with zero regressions,
  100% coverage on touched files, docs regenerated. Detailed history compressed to
  `plans/NGE_Core_Algorithm_Workstream.logs.md`.
- Phase 2 [WIP]: Polyandric Reproduction Exports & Activation (P2, P5) — active phase.
  Step 01 [DONE]: boundary map produced, DR-2026-06-27-P2 recorded (scope conflict: 3 example
  skip-contracts require P3/P4, not Phase 2), acceptance criteria revised to library-core,
  Step 02–07 packets authored. Next active step = Phase 2 Step 02 (researching).
- Phases 3–7 [PLANNED]: phase-level packets authored; per-phase Step 02–07 packets will be
  authored by each phase's Step 01.
- Carry-forward blockers P1–P5 and DR-2026-06-27-05 mapped to phases.
- Growth gap (101 → 8,000+ nodes) mapped to Phase 4.

## Coverage backlog

- ~~NGE_DNA ↔ Network bridge (Phase 1)~~ [DONE]
- Polyandric exports + queenBias (Phase 2)
- modeIsEvolvable operator + phenotype→Network bridge (Phase 3)
- Growth engine diagnosis + fix (Phase 4)
- Schema alignment (Phase 5)
- Reproduction FSM integration (Phase 6)
- End-to-end verification (Phase 7)

## Immediate next steps

1. Execute Phase 2 Step 02: research the polyandric export surface, the queenBias merge
   path, and resolve open assumptions A1 (deterministic weighting), A2 (parameterSchema
   gating), A4 (clamp-vs-throw for queenBias outside [0,1]).
2. After Step 02, proceed through Steps 03–07 (red tests → implement → green → docs →
   compress) using the authored step packets in the plan file.
3. Do NOT start any demo work (racing v2, ant hive, predator/prey) until Phase 7
   verification passes.

## Deferred questions

- Whether `ModulatorBroadcaster` / `EpisodicSlot` / `GatingRouter` activation belongs in
  Phase 3 or a later phase — to be resolved by Phase 3 Step 01 research.
- Whether the schema mismatch (P4) requires a core change or a worker-side adapter — to
  be resolved by Phase 5 Step 02 research.
- Whether the growth stall root cause is in the morph applier, the lifecycle wiring, the
  budget enforcement, or the planner itself — to be resolved by Phase 4 Step 02 diagnosis.

```text
Continue from the current repo state only. Do not rely on prior chat history.

Context: NGE Core Algorithm Workstream — plans/NGE_Core_Algorithm_Workstream.plans.md.
Status: [WIP]. Phase 1 [DONE] (compressed to plans/NGE_Core_Algorithm_Workstream.logs.md).
Phase 2 (Polyandric Reproduction Exports & Activation, P2/P5) is the active phase.

==================== PHASE STATUS SNAPSHOT ====================

Phase 1 — NGE_DNA Adoption & Canonical Envelope Bridge (P1) — [DONE]
  - 7 steps complete, compressed to logs.
  - Deliverables:
    * src/neat/nge-dna/neat.nge-dna.bridge.ts
        - materializeNetworkFromPhenotype(envelope, plan, descriptor, runtimeHints?)
        - extractCanonicalEnvelopeFromNetwork(network)
    * src/neat/nge-dna/neat.nge-dna.errors.ts — NGE_DNA_BridgeError class
    * src/neat/nge-dna/neat.nge-dna.bridge.test.ts — 20 tests, 100% coverage
  - Routes through NetworkJSON intermediate; extension carrier
    { version: 1, ngeDescriptor, ngeEnvelope } in NetworkJSONExtensions.values
  - Opt-in isolation via ngeEnabled flag on GenomeMaterializationRuntimeHints
  - JSDoc with Mermaid diagram + academic citations
    (Stanley & Miikkulainen NEAT 2002)

Phase 2 — Polyandric Reproduction Exports & Activation (P2, P5) — [WIP]
  - Step 01 — Planning: [DONE]
      7 step packets authored, boundary map produced, scope conflict resolved
      (DR-2026-06-27-P2: 3 racing skip-contracts require P3/P4, NOT Phase 2)
  - Step 02 — Research: [DONE]
      14-section research brief, 4 assumptions resolved:
        A1: FNV-1a 32-bit hash of regionId ÷ 2^32 → deterministic [0,1) normalization
        A2: no paramSchema gating
        A3: deep-equals
        A4: clamp to [0,1] at point of use
  - Step 03 — Red tests: IN PROGRESS
      03-red-testing agent dispatched (running in background).
      Creating failing tests for P2 export visibility + P5 queenBias honoring.
      Test file: src/neat/nge-evolution/neat.nge-evolution.reproduction.queen-bias.test.ts
      6 red tests needed: export visibility (3), queenBias honoring (3+)
  - Step 04 — Implementation: [PLANNED]
      Export types + wire queenBias through merge pipeline
  - Step 05 — Green validation: [PLANNED]
  - Step 06 — Documentation: [PLANNED]
  - Step 07 — Compress Phase 2 into logs: [PLANNED]

Phases 3-7 — [PLANNED]
  - Phase 3: modeIsEvolvable Activation
  - Phase 4: Growth Engine Diagnosis & Fix (101→8000+ nodes) — CRITICAL
  - Phase 5: Schema Alignment (P4)
  - Phase 6: Reproduction FSM Integration (P3)
  - Phase 7: Verification: Seed→8000+ Neurons

==================== KEY RESEARCH FINDINGS (Phase 2) ====================

P2 — Export visibility:
  - NgePolyandricInput (L35) and NgePolyandricDroneInput (L28) are
    non-exported in reproduction.ts
  - The facade does not re-export them
  - Fix: add `export` keyword + facade re-export

P5 — queenBias honoring:
  - queenBias on NgeReproductionPolicy (default 1.0) is NEVER referenced
    in reproduction.ts
  - Hard merge at L482 (archetype) and L571 (non-archetype)
  - Fix path: thread queenBias through
      applyPolyandricAssignments → patchPolyandricRegion →
      mergeModuleArchetypeWithQueenPriority + non-archetype branch

  - A1: FNV-1a 32-bit hash of regionId ÷ 2^32 → deterministic [0,1)
  - A4: clamp to [0,1] at point of use
  - queenBias=1.0 regression safe: hash ∈ [0,1) < 1.0 is always true

==================== CARRY-FORWARD BLOCKERS ====================

  - P2 (CRITICAL): NGE polyandric types not exported — RESOLVING in Phase 2
  - P5 (CRITICAL): queenBias not honored — RESOLVING in Phase 2
  - P3: Racing FSM reproduction step is placeholder — Phase 6
  - P4: Schema mismatch between core NGE types and racing worker types — Phase 5
  - modeIsEvolvable: dead boolean field, no operator reads it — Phase 3
  - 27 tsc.test.json duplicate-identifier errors in 3 test files (pre-existing)
  - 3 polyandric tests remain skipped (require P3/P4, NOT Phase 2)

==================== USER'S VISION ====================

  - Networks grow organically from seed to 8,000+ neurons with continuous
    real-time adaptation
  - Generations for multiplication/fusing, not mandatory for evolution
  - Start small and grow and adapt
  - WebGPU planned for acceleration at high node counts (RTX 4070 Super)
  - Visualizer shows only blue team #1 network; all cars coevolve independently
  - Racing curriculum stays [DONE] until NGE Core is fully complete

==================== WHAT THE NEXT SESSION MUST DO FIRST ====================

  1. Check if Phase 2 Step 03 (red tests) completed.
     - If [DONE], proceed to Step 04 (implementation).
     - If still in progress, wait for the 03-red-testing agent to finish,
       then verify the red test file exists and tests fail for the right
       reason (missing implementation, not syntax/fixture error).
  2. Dispatch 04-implementing for Step 04:
     - Export NgePolyandricInput / NgePolyandricDroneInput from
       src/neat/nge-evolution/reproduction.ts
     - Add facade re-export in the reproduction facade
     - Wire queenBias through the merge pipeline:
         applyPolyandricAssignments → patchPolyandricRegion →
         mergeModuleArchetypeWithQueenPriority + non-archetype branch
     - Use FNV-1a 32-bit hash of regionId ÷ 2^32 for deterministic
       [0,1) normalization; clamp to [0,1] at point of use
     - Ensure queenBias=1.0 preserves current hard-queen-wins behavior
  3. Then dispatch 05-green-testing for Step 05 validation.
     - Targeted tests only: --testPathPattern=queen-bias
     - Verify 100% coverage on touched src/neat/ files
     - Do NOT run full suite until phase complete
  4. Continue through Steps 06 (documentation) and 07 (compress Phase 2
     into logs) to complete Phase 2.
  5. Then proceed to Phase 3 (modeIsEvolvable Activation).

==================== ORCHESTRATION PROTOCOL REMINDERS ====================

  - MUST call neataptic-dispatch-mcp / build_dispatch_packet before EVERY
    task dispatch. Direct task use without a prior dispatch packet is a
    workflow violation.
  - Follow the RED → IMPLEMENT → GREEN loop for sliced implementation.
    Each iteration uses a NEW agent instance (fresh context).
  - 100% coverage-guard applies (src/neat/ workstream).
  - Use Cortex RAG (neataptic-cortex-mcp) as the PRIMARY search mechanism.
    Native tools (grep/glob/view) are LAST RESORT only.
  - Targeted tests only — never run the full suite until the phase is
    complete. Use --testPathPattern or --testNamePattern selectors.
  - Phase compression is MANDATORY when all steps in a phase are [DONE]
    and green validation has passed. Dispatch 07-logging to compress
    before advancing to the next phase.
  - No deferred cleanup: dead fields and placeholders must be removed
    in the same step that introduces the real operator.
  - Classic NEAT must remain unchanged when NGE is disabled (opt-in
    isolation).

==================== REQUIRED VALIDATIONS ====================

  - node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md
  - node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NGE_Core_Algorithm_Workstream.plans.md

==================== KNOWN CONSTRAINTS ====================

  - This is a src/neat/ workstream — do NOT touch examples/ until Phase 7
    passes.
  - The 3 racing-worker skip-contracts
    (simulation-worker.race-pack.tier5.test.ts L179/188/197) must remain
    .skip — they are owned by Phase 7, blocked on P3 (Phase 6) + P4 (Phase 5).
  - NGE must be FULLY complete before any demo work resumes.
  - No deferred cleanup: dead fields and placeholders must be removed in
    the same step that introduces the real operator.
  - Classic NEAT must remain unchanged when NGE is disabled (opt-in
    isolation).
  - WebGPU acceleration is a future lane, not part of this workstream.
```
