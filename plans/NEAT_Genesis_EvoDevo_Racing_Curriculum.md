# NEAT Genesis EvoDevo: Racing Curriculum Audit

**Status:** [WIP]

## Scope

Refactor `examples/racing_curriculum` from a proof of concept into the Team
Racing Curriculum benchmark described by `examples/racing_curriculum/reference.plans.md`.
The benchmark remains downstream of `plans/completed/NEAT_Genesis_EvoDevo.md`
and `plans/completed/Memory_Optimization.md`; when those upstream plans conflict
with the demo plan, the upstream plan wins.

## Current state

- The current POC is strongest in deterministic runtime scaffolding, local
  runtime adaptation, track generation, tire/pit mechanics, and renderer
  plumbing.
- The largest plan-fidelity gap is still runtime/evolution semantics: the POC is
  single-controller centric, not two independent Team A/B NEAT populations with
  rolling opponent snapshots, generation barriers, and team-level fitness.
- The worker seam exists, but the racing demo does not yet follow the
  Flappy-style pattern where a worker owns evolution/playback authority and the
  host receives compact real-time `race-step` frames.
- Track and physics are partial: surface-type semantics, sand/wall/off-track
  lifecycle, pit-entrance blocking, braking-distance tire effects, and slip-onset
  tire effects are not yet complete.
- UI/observability are partial: benchmark-grade charts for Team A vs Team B
  fitness, radio heatmaps, pit/tire strategy, role divergence, reproduction mode,
  and playback controls remain incomplete.

## MCP tracking plan

```yaml
workstream: racing_curriculum_reference_completion
source_reference: examples/racing_curriculum/reference.plans.md
active_tracker: plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
primary_boundary: full_team_a_b_coevolution_worker_loop
reason:
  - 'The worker-authoritative runtime foundation is complete, so the next highest-fidelity gap is full Team A/B coevolution.'
  - 'Worker-owned controller inference and generation looping are required before the benchmark can claim reference-plan depth.'
  - 'Flappy Bird already proves the host/worker split: generation-ready summaries plus real-time playback-step snapshots.'
preserve_terms:
  - Team A and Team B
  - identical-DNA teams
  - team radio as stigmergy
  - rolling opponent snapshot
  - polyandric reproduction
  - category ladder
  - carry-state and reset-state semantics
  - deterministic race packs
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
    - NGE Benchmark Scout
    - Worker Payload Scout
    - Evaluation Pool Scout
    - Repo Cortex Scout
  implementation:
    - 04-implementing
    - NGE Core Scout
    - Visualizer Scout
  validation:
    - 05-green-testing
    - Coverage Guard
  escalation:
    - '00-helping only when an MCP/tool/agent/flow gap blocks the active step.'
non_goals:
  - 'Do not hand-code queen, blocker, pacer, or pit-strategy roles.'
  - 'Do not claim full Tier 5 or Tier 6 completion before Team A/B coevolution and observability gates exist.'
  - 'Do not move browser rendering authority into workers; workers own simulation/evolution, host owns DOM/canvas presentation.'
  - 'Do not patch missing NGE primitives locally inside the demo; route missing core primitives to NGE Core Scout.'
  - 'Do not edit generated docs/examples output directly.'
acceptance_criteria:
  - id: worker_authority
    criterion: 'Given the browser demo is running, when evolution or race playback advances, then controller inference and simulation state advance in a worker rather than on the main thread.'
    validation: 'focused worker protocol tests plus browser bundle build'
  - id: streaming_frames
    criterion: 'Given a race is active, when the host requests live display, then it receives compact race-step snapshots suitable for render-cadence streaming.'
    validation: 'packed snapshot transfer-list tests'
  - id: independent_populations
    criterion: 'Given Team A and Team B evolve, then each team keeps independent population/species/fitness state and only interacts through the race environment.'
    validation: 'coevolution container tests'
  - id: frozen_opponents
    criterion: 'Given a generation starts, then evaluation uses a frozen rolling opponent snapshot until the configured snapshot boundary.'
    validation: 'snapshot selection and generation-barrier tests'
  - id: deterministic_packs
    criterion: 'Given the same race pack seed and opponent snapshot, then repeated evaluation returns stable comparable episode inputs.'
    validation: 'deterministic replay tests'
  - id: tier_ladder
    criterion: 'Given a team promotes, then the category ladder advances only through deterministic pack reliability and records explicit carry-state/reset-state behavior.'
    validation: 'tier transition tests'
  - id: observability
    criterion: 'Given training progresses beyond early tiers, then UI telemetry exposes Team A vs Team B fitness, radio/tire/pit signals, role divergence, and reproduction-mode summaries.'
    validation: 'renderer/telemetry tests and manual browser smoke'
stop_conditions:
  done: 'All Step 02-07 packets have completed and the tracker is closed by 07-logging.'
  hold: 'The active step needs user prioritization or benchmark-policy clarification.'
  blocked: 'A missing upstream NGE primitive or MCP/tooling gap prevents honest implementation.'
```

## Implementation phases

### Phase 1 — Racing curriculum refactor packetization [WIP]

#### Step 01 — Gap audit and first-boundary selection [DONE]

[DONE] Gap audit compressed: `examples/racing_curriculum` is a deterministic
POC with partial track, tire, pit, renderer, and worker seams; the selected first
boundary is runtime/evolution plus worker streaming because it is the highest
plan-fidelity and performance prerequisite for Team A/B coevolution.

#### Step 02 — Research boundary mapping [DONE]

```yaml
phase: 1
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 03 — Red tests for worker runtime and coevolution contracts'
```

[DONE] Boundary map complete. Validation gates all passed (plan-sync ✓,
plan-phase-packets ✓, routing-table-freshness ✓). No escalation required.
Research outcome recorded below.

---

##### Step 02 — Research Output

###### Boundary Map

**Host-owned (never move to worker):**
- DOM/canvas rendering, HUD panels, telemetry display, network-view canvas
- User interaction (keyboard shortcuts, tier selector UI)
- `requestAnimationFrame` loop cadence and viewport resize handling
- Receiving and decoding compact typed-array `race-step` snapshots for render

**Worker-owned (target state after Step 04):**
- Team A and Team B `Neat` population instances + species/fitness history
- Rolling opponent snapshot store: hall-of-fame sample + recent-population
  sample; frozen at generation start, updated every N generations (configurable)
- Generation lifecycle: evaluate against frozen snapshot → select → evolve →
  emit `generation-ready` summary
- Race episode lifecycle: construct deterministic race pack → tick simulation →
  run controller inference per car per tick → pack and stream `race-step`
  typed-array snapshot
- Controller inference (network.activate per car each simulation tick)
- Packed `race-step` snapshot construction and transfer-list resolution
- Deterministic race-pack construction (seed + opponent snapshot → initial state)
- Generation barrier enforcement (no opponent update while evaluation runs)
- Optional nested evaluation-pool dispatch (SharedArrayBuffer / nested workers;
  capability-gated, with single-thread fallback)

**Host↔Worker message seam (target protocol):**
```
Host → Worker:
  init        { populationSize, rngSeed, tier, archProfileId? }
  request-generation { }
  start-race  { tierConfig, opponentSnapshotId }
  request-race-step { requestId, stepsToAdvance }
  stop        { }

Worker → Host:
  generation-ready { generation, teamABestFitness, teamBBestFitness,
                     bestNetworkPayload?, populationNetworkPayloads? }
  race-step        { requestId, snapshot: RacingRenderFrame, done }
  runtime-status   { phase, statusText }
  error            { message }
```

**Current POC seam (to be replaced):**
`{type:'step', requestId, envState, control}` → `{type:'step-result', requestId, envState}`

The host sends full `EnvironmentState` every tick and the worker returns a new
`EnvironmentState`. Controller inference, curriculum progress, and evolution all
run on the main thread. The worker is a one-shot physics delegate, not an
authority.

###### File-Level Seams

| File | Role | Gap |
|------|------|-----|
| `browser-entry/browser-entry.ts` | POC main loop, DOM, controller, curriculum | Owns evolution and controller — must delegate both to worker |
| `workers/simulation-worker/simulation-worker.types.ts` | `RacingRenderFrame` packed struct | Schema correct; needs evolution-worker message types (new file) |
| `workers/simulation-worker/simulation-worker.snapshot.utils.ts` | Transfer-list resolver, schema assert | Correct; reusable as-is |
| `workers/simulation-worker/simulation-worker.tier3.ts` | Tier 3 frame factory + radio resolver | Correct; usable by worker-owned episode stepper |
| `workers/simulation-worker/simulation-worker.tier4.ts` | Tier 4 frame factory + pit status | Correct; usable by worker-owned episode stepper |
| `workers/simulation-worker/simulation-worker.tier5.ts` | Tier 5 frame factory + radio resolver | Correct; usable by worker-owned episode stepper |
| `environment/environment.step.service.ts` | Deterministic physics stepper | Complete; worker will call this inside the episode loop |
| `environment/environment.types.ts` | Shared state types | Complete |
| `controller/nge.controller.ts` | Network-based car controller | Must move to worker; host receives snapshots not control output |
| `controller/runtime.adaptation.ts` | Within-episode mutation engine | Demo-local; keep or discard in worker rewrite |
| `browser-entry/host/` | DOM panel layout | Host-owned; no change |
| `evaluation/` (not yet present) | Evaluation harness | Needs: team fitness resolver, race-pack factory, opponent snapshot container |
| `workers/simulation-worker/simulation-worker.evolution.types.ts` | Evolution worker message types | **Missing — new file needed** |
| `workers/simulation-worker/simulation-worker.evolution.protocol.service.ts` | FSM router (init→evolve→race→stop) | **Missing — new file needed** |
| `workers/simulation-worker/simulation-worker.coevolution.service.ts` | Team A/B container | **Missing — new file needed** |
| `workers/simulation-worker/simulation-worker.race-pack.service.ts` | Deterministic pack factory | **Missing — new file needed** |
| `workers/simulation-worker/simulation-worker.opponent-snapshot.service.ts` | Rolling snapshot store + barrier | **Missing — new file needed** |

###### POC Gap Classification

| Gap | Classification | Owner |
|-----|----------------|-------|
| Worker evolution protocol FSM (init/generation/start-race/race-step/stop) | demo-local | racing_curriculum worker |
| Evolution worker message types (typed union like Flappy) | demo-local | racing_curriculum worker |
| Packed `race-step` snapshot with generation telemetry | demo-local | racing_curriculum worker |
| Transfer-list buffer-detach enforcement for race-step frames | demo-local | racing_curriculum worker |
| Team A/B independent `Neat` population containers | demo-local (2× Neat instances) | racing_curriculum worker |
| Team fitness: best-finishing car position = team score | demo-local | racing_curriculum worker |
| Rolling opponent snapshot store (hall-of-fame + recent sample) | demo-local policy | racing_curriculum worker |
| Generation barrier: freeze snapshot before evaluation, release after | demo-local orchestration | racing_curriculum worker |
| Deterministic race-pack construction (seed + snapshot → initial conditions) | demo-local | racing_curriculum worker |
| Radio field transport in RacingRenderFrame | partially exists (type defined, not written) | racing_curriculum worker |
| Team tier-ladder state management (carry-state / reset-state) | demo-local | racing_curriculum worker |
| Benchmark observability charts (Team A vs B fitness, radio heatmaps) | demo-local renderer | racing_curriculum host |
| Stigmergy typed-array field primitive (shared pheromone analog) | **upstream NGE Phase G Step 04** | NGE Core Scout |
| `ModulatorBroadcaster` neuromodulation | **upstream NGE** | NGE Core Scout |
| `EpisodicSlot` medium-term memory | **upstream NGE** | NGE Core Scout |
| `GatingRouter` hard task-switching | **upstream NGE** | NGE Core Scout |
| Polyandric reproduction mode + `modeIsEvolvable` | **upstream NGE Phase E** | NGE Core Scout |

**NGE Core Scout escalation:** NOT required for Step 03 or Step 04 tranche.
All upstream NGE primitives (ModulatorBroadcaster, EpisodicSlot, GatingRouter,
polyandric) are advanced-tier features needed by Tier 5/6. The first tranche
targets Tier 1–3 worker authority, which only needs standard `Neat` populations
and network inference. Step 04 must document NGE primitive dependencies with
explicit TODO comments rather than compensating locally.

###### Step 03 Red-Test Targets

Smallest observable failures that prove the current POC gaps:

**File: `workers/simulation-worker/simulation-worker.evolution.protocol.test.ts`** (new)
- `routeRacingWorkerProtocolMessage` rejects `request-generation` before `init`
- `routeRacingWorkerProtocolMessage` rejects `start-race` before a generation exists
- `routeRacingWorkerProtocolMessage` rejects `request-race-step` before `start-race`
- `routeRacingWorkerProtocolMessage` routes `stop` first regardless of state

**File: `workers/simulation-worker/simulation-worker.coevolution.test.ts`** (new)
- Team A population evolves independently from Team B (separate innovation counters)
- Team fitness resolves as the best-finishing car index (not sum, not average)
- Frozen opponent snapshot does not update during active evaluation
- Snapshot is replaced only after the configured generation boundary

**File: `workers/simulation-worker/simulation-worker.race-pack.test.ts`** (new)
- `createDeterministicRacePack(seed, opponentSnapshot)` returns identical initial
  conditions for identical inputs
- Pack initial car positions are not all zeros (must spread cars on track grid)
- Transfer list from a `race-step` snapshot contains no duplicate buffers
- Detached buffer (after transfer-list postMessage) cannot be reused without
  allocating a new array

**Already-passing test coverage to protect:**
- `simulation-worker.snapshot.utils.test.ts` (schema version + transfer list)
- `simulation-worker.tier3.test.ts`, `tier4.test.ts`, `tier5.test.ts` (frame factories)
- `simulation-worker.tier5-renderer.test.ts` (Tier 5 renderer parity)
- `environment.step.service.test.ts` (physics stepper)

###### Validation Gate Results (Step 02)

- `validate-plan-sync`: PASS (0 errors, 0 warnings)
- `validate-plan-phase-packets`: PASS (0 errors, 0 warnings; Phase 1 WIP, agent 02-researching)
- `routing-table-freshness.gate`: PASS (hash match, 57 agents, 51 skills)

#### Step 03 — Red tests for worker runtime and coevolution contracts [DONE]

```yaml
phase: 1
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-testing.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 04 — Implement worker-authoritative racing runtime foundation'
```

**User instruction:** Start a fresh session, select `03-red-testing`, and paste this full step packet.

**Step objective:** Add the smallest failing owner-local tests that prove the
current POC lacks the worker-authoritative runtime and Team A/B coevolution
contracts selected by Step 02.

**Context the agent must know:**

- Step 02 boundary map is in the `Step 02 — Research Output` section of this
  tracker. Read it before writing any test.
- The current worker seam (`createRacingSimulationWorker` in `browser-entry.ts`)
  is a one-shot step-delegation worker, not an evolution-authority worker.
  Current protocol: `{type:'step', envState, control}` → `{type:'step-result', envState}`.
  Target protocol: `init → request-generation → start-race → request-race-step → stop`.
- New files needed (all in `workers/simulation-worker/`):
  - `simulation-worker.evolution.types.ts` — typed message union (like Flappy's `types.ts`)
  - `simulation-worker.evolution.protocol.service.ts` — FSM router
  - `simulation-worker.coevolution.service.ts` — Team A/B container
  - `simulation-worker.race-pack.service.ts` — deterministic pack factory
  - `simulation-worker.opponent-snapshot.service.ts` — rolling snapshot store + barrier
- Tests must follow the single-expect rule and should fail for the intended POC
  gap before implementation begins.
- Prioritize behavior that unlocks performance and plan fidelity: worker-owned
  simulation/evolution, compact race-step streaming, Team A/B population state,
  rolling opponent snapshots, generation barriers, and deterministic race packs.
- Do not write tests that require hard-coded roles or prescribed radio
  semantics; role differentiation and radio meaning must remain emergent.
- NGE Core Scout escalation NOT needed for this tranche. The first tranche uses
  standard `Neat` populations and network inference only. Document NGE motif
  dependencies (ModulatorBroadcaster, EpisodicSlot, GatingRouter, polyandric) as
  explicit TODO comments rather than compensating locally.

**Execution steps:**

1. Read the Step 02 boundary map in this tracker and nearest owner-local racing
   worker/runtime tests (`simulation-worker.snapshot.utils.test.ts` etc.).
2. Add failing tests for worker protocol lifecycle FSM in
   `simulation-worker.evolution.protocol.test.ts`:
   `init → request-generation → start-race → request-race-step → stop`.
3. Add failing tests for packed `race-step` snapshot schemas and transfer-list
   ownership in `simulation-worker.race-pack.test.ts` so detached buffers
   cannot be reused accidentally.
4. Add failing tests for independent Team A/B population containers and team-level
   fitness (best-finishing car = team score) in
   `simulation-worker.coevolution.test.ts`.
5. Add failing tests for frozen rolling opponent snapshot selection and
   generation-barrier semantics (snapshot is not updated during evaluation).
6. Add failing tests for deterministic race-pack replay: same seed + snapshot =
   identical initial conditions.
7. Run the focused red-test slice and record exact failing tests plus expected
   failure reasons.

**Stop conditions:**

- **Done:** focused tests fail for the intended missing contracts and are ready
  for Step 04 implementation.
- **Hold:** Step 02 did not provide enough boundary detail to write targeted
  failing tests.
- **Blocked:** a required test helper depends on an upstream NGE primitive that
  is not available; route to NGE Core Scout.
- **Route-back:** return to `02-researching` if the red tests expose an
  unmapped file boundary.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `npx jest --config=jest.config.mjs --no-cache --runInBand --runTestsByPath examples/racing_curriculum/index.test.ts`

**Plan update requirement:** Record red-test files, expected failing assertions,
single-expect compliance notes, and the exact Step 04 implementation target in
this tracker before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

---

##### Step 03 — Red-Phase Evidence

###### Red-Test Files Added

All three files are new, owner-local, in
`examples/racing_curriculum/workers/simulation-worker/`.

| File | Tests | Target service boundary |
|------|-------|------------------------|
| `simulation-worker.evolution.protocol.test.ts` | 4 | `simulation-worker.evolution.protocol.service.ts` (FSM router) |
| `simulation-worker.race-pack.test.ts` | 5 | `simulation-worker.race-pack.service.ts` (deterministic pack factory + transfer ownership) |
| `simulation-worker.coevolution.test.ts` | 7 | `simulation-worker.coevolution.service.ts` + `simulation-worker.opponent-snapshot.service.ts` |

###### Minimal Red-Phase Scaffolding Added

The following placeholder boundaries were added only so Jest can load the
intended Step 04 seams while still failing on the missing contracts:

- `simulation-worker.evolution.types.ts`
- `simulation-worker.evolution.protocol.service.ts`
- `simulation-worker.coevolution.service.ts`
- `simulation-worker.race-pack.service.ts`
- `simulation-worker.opponent-snapshot.service.ts`

Each scaffold is intentionally incomplete and must be replaced or hardened in
Step 04.

###### Focused Red-Test Run Output (13 FAILED / 3 PASSED / 16 TOTAL)

```
FAIL simulation-worker.evolution.protocol.test.ts
  ● request-generation before init → expect(result.error).toBeDefined()
      Received: undefined
  ● start-race before generation-ready → expect(result.error).toBeDefined()
      Received: undefined
  ● request-race-step before start-race → expect(result.error).toBeDefined()
      Received: undefined
  ● stop routing → expect(result.nextState.phase).toBe('stopped')
      Received: 'idle'

FAIL simulation-worker.race-pack.test.ts
  ● identical carX for identical seed + snapshot → expect(Array.from(packA.carX)).toEqual(Array.from(packB.carX))
      Received first differing value: 1 vs 2
  ● packed schema sentinel → expect(pack.schemaVersion).toBe('racing-packed-v1')
      Received: 'racing-packed-v0'
  ● transfer ownership → expect({ containsCarXBuffer, entryCount, uniqueBufferCount }).toEqual(...)
      Received: { containsCarXBuffer: false, entryCount: 1, uniqueBufferCount: 1 }
  ● detached-buffer safety → expect(pack.carX.byteLength).toBe(0)
      Received: 16

FAIL simulation-worker.coevolution.test.ts
  ● teamA/teamB isolation → expect(container.teamA).not.toBe(container.teamB)
      Received shared object: { populationId: 'shared-population' }
  ● population identity isolation → expect(container.teamA.populationId).not.toBe(container.teamB.populationId)
      Received: 'shared-population'
  ● team-level fitness uses best finisher → expect(teamFitness).toBe(3)
      Received: 5
  ● team-level fitness does not average finishes → expect(teamFitness).toBe(1)
      Received: 5
  ● generation barrier before boundary → expect(updateApplied).toBe(false)
      Received: true

Test Suites: 3 failed, 3 total
Tests:       13 failed, 3 passed, 16 total
```

**Failure reason:** Jest now resolves the intended Step 04 seams, and the
focused red slice fails for the actual missing contracts: lifecycle rejection
rules are absent, `stop` does not transition to `stopped`, deterministic
race-pack replay is not stable, the packed snapshot schema/transfer ownership is
incomplete, Team A/B containers alias shared state, team fitness averages
finishes instead of taking the best finisher, and opponent snapshots ignore the
configured generation barrier.

###### Single-Expect Compliance

Every `it()` block contains exactly one top-level `expect(...)` call.
No nested expects, no multi-assertion helpers.

###### Validation Gate Results (Step 03)

- `validate-plan-sync`: **PASS** (0 errors, 0 warnings)
- `validate-plan-phase-packets`: **PASS** (0 errors, 0 warnings)
- `index.test.ts`: **PASS** (1 passing — existing suite unaffected)

###### Step 04 Implementation Target

Step 04 must create these five files in `workers/simulation-worker/` so that all
16 red tests pass:

| File | Minimum contract |
|------|-----------------|
| `simulation-worker.evolution.types.ts` | Typed inbound/outbound message union |
| `simulation-worker.evolution.protocol.service.ts` | `createInitialProtocolState()` + `routeRacingWorkerProtocolMessage()` must enforce `init → request-generation → start-race → request-race-step → stop` |
| `simulation-worker.coevolution.service.ts` | `createCoevolutionContainer()` must allocate independent Team A/B containers and resolve team fitness from the best-finishing team car |
| `simulation-worker.race-pack.service.ts` | `createDeterministicRacePack(seed, snapshot)` must replay identical initial conditions and emit `'racing-packed-v1'` frames |
| `simulation-worker.opponent-snapshot.service.ts` | `createOpponentSnapshotStore(config)` must freeze rolling opponent snapshots during evaluation and only rotate them at the configured generation barrier |
| Transfer-list handling inside `simulation-worker.race-pack.service.ts` | Must include every packed frame buffer exactly once and leave transferred buffers detached after handoff |

NGE primitive dependencies (NOT to be compensated locally in Step 04 — record
as `TODO: NGE_TODO` comments):

- `ModulatorBroadcaster` for team radio neuromodulation (upstream Phase G)
- `EpisodicSlot` for opponent snapshot payload enrichment (upstream Phase G)
- `GatingRouter` for `carMode` hard task-switch (upstream Phase G)
- Polyandric reproduction + `modeIsEvolvable` (upstream Phase E)

#### Step 04 — Implement worker-authoritative racing runtime foundation [DONE]

```yaml
phase: 1
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 05 — Green validation and regression triage'
```

**User instruction:** Start a fresh session, select `04-implementing`, and paste this full step packet.

**Step objective:** Implement the minimum runtime/evolution foundation that
makes the Step 03 tests pass while moving racing toward the reference plan's
worker-authoritative Team A/B benchmark.

**Context the agent must know:**

- Host-owned responsibilities: DOM controls, HUD, canvas rendering, telemetry
  panel wiring, and user interaction.
- Worker-owned responsibilities: environment state, controller inference,
  generation and race lifecycle, Team A/B runtime state, race-step snapshot
  packing, and optional nested evaluation-pool dispatch.
- Reuse Flappy-style patterns where they fit: generation-ready summaries,
  playback/race-step streaming, typed-array snapshot schemas, transfer-list
  resolution, capability-gated optimized transports, and fallback paths when
  SharedArrayBuffer or nested workers are unavailable.
- Keep benchmark-local code honest. If implementation requires a new NGE motif,
  reproduction primitive, or memory primitive, stop and route the missing core
  requirement to NGE Core Scout.

**Execution steps:**

1. Implement the smallest protocol and type boundary that satisfies the worker
   lifecycle red tests without moving renderer authority into the worker.
2. Introduce or harden packed race-step snapshots for car positions, headings,
   team ids, tire summaries, radio summaries, alive/pit status, and generation
   telemetry needed by the host.
3. Add transfer-list handling that mirrors Flappy's safe transferable-payload
   pattern and prevents reused detached buffers.
4. Introduce independent Team A/B runtime containers with population/species/
   fitness separation, even if the first pass uses narrow placeholder population
   adapters behind explicit TODO-free interfaces.
5. Implement rolling opponent snapshot selection and generation-barrier
   orchestration at the benchmark layer, using existing NGE collective
   primitives where available.
6. Implement deterministic race-pack setup for repeatable evaluation across
   teams and opponent snapshots.
7. Keep all new public or exported symbols documented, replace magic numbers
   with named constants, and keep helper flow declarative.

**Stop conditions:**

- **Done:** Step 03 tests pass through the implemented worker/co-evolution
  foundation without hiding failures behind demo-local fallbacks.
- **Hold:** a behavior choice affects benchmark semantics and requires planning
  clarification.
- **Blocked:** missing upstream core primitive, browser-build issue, or MCP/tool
  issue prevents implementation; route to the correct owner before continuing.
- **Route-back:** return to `03-red-testing` if implementation reveals that the
  red contract was too broad or not observable.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `npm run quality:folder -- --folder=examples/racing_curriculum`
- `npm run build:racing-curriculum`
- `npx jest --config=jest.config.mjs --no-cache --runInBand --runTestsByPath examples/racing_curriculum/index.test.ts`

**Plan update requirement:** Record changed files, behavior implemented,
remaining reference-plan gaps, and the focused validation result before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

---

##### Step 04 — Implementation Record

###### Files Changed

| File | Change type | Summary |
|------|-------------|---------|
| `workers/simulation-worker/simulation-worker.evolution.protocol.service.ts` | Rewritten | Full FSM router: idle→initialised→generation-ready→racing→stopped; `stop` always transitions to `stopped`; invalid messages for the current phase return a rejection `error` string |
| `workers/simulation-worker/simulation-worker.coevolution.service.ts` | Rewritten | Distinct `teamA`/`teamB` population handles with unique `populationId` per container; `resolveTeamFitness` uses `Math.min` (best position), not average |
| `workers/simulation-worker/simulation-worker.opponent-snapshot.service.ts` | Rewritten | Generation barrier (evaluationActive guard) + generation boundary check (`generation % updateEveryNGenerations === 0`); stores frozen payload; exposes `frozenPayload` field |
| `workers/simulation-worker/simulation-worker.race-pack.service.ts` | Rewritten | Deterministic `createDeterministicRacePack(seed, snapshot)` using index-based grid offsets; correct `schemaVersion: 'racing-packed-v1'`; `resolveRaceStepTransferList` collects all 10 distinct typed-array buffers |
| `workers/simulation-worker/simulation-worker.coevolution.service.test.ts` | Created | Sibling smoke tests for quality gate |
| `workers/simulation-worker/simulation-worker.evolution.protocol.service.test.ts` | Created | Sibling smoke tests for quality gate |
| `workers/simulation-worker/simulation-worker.opponent-snapshot.service.test.ts` | Created | Sibling smoke tests for quality gate |
| `workers/simulation-worker/simulation-worker.race-pack.service.test.ts` | Created | Sibling smoke tests for quality gate |
| `controller/runtime.adaptation.test.ts` | Created | Sibling smoke tests for quality gate (pre-existing gap) |

###### Behavior Implemented

- **Evolution protocol FSM:** `routeRacingWorkerProtocolMessage` now enforces phase-gated message routing. `stop` always reaches `stopped`. All other messages outside the current phase return `error` rather than passing through.
- **Team A/B coevolution containers:** `createCoevolutionContainer` allocates two independent objects with distinct `populationId` strings (using a monotonic serial counter seeded by `rngSeed`). `resolveTeamFitness` returns `Math.min(...positions)`.
- **Generation barrier + boundary enforcement:** `createOpponentSnapshotStore` applies two guards before updating: (1) evaluation must not be active, (2) `generation % updateEveryNGenerations === 0`. Failed updates return `false`; successful updates return `true` and store the frozen payload.
- **Deterministic race-pack:** `createDeterministicRacePack(seed, snapshot)` builds a stable `RacingRenderFrame` using grid offsets computed from agent index and seed alone — no counter, no mutable state. Same inputs always produce the same `carX`/`carY` values. `resolveRaceStepTransferList` collects all 10 typed-array buffers (carX, carY, carHeading, carActive, carTeam, carMode, tireState, radioField, lap, place) into a deduplicated transfer list.
- **Named constants:** `AGENT_COUNT`, `GRID_COLUMN_SPACING`, `GRID_ROW_SPACING`, `SEED_X_SCALE`, `EXPECTED_TRANSFER_BUFFER_COUNT` replace all magic numbers.
- **NGE primitive TODOs:** `ModulatorBroadcaster`, `EpisodicSlot`, `GatingRouter`, and polyandric reproduction are documented as `NGE_TODO` comments in the coevolution and race-pack services.

###### Remaining Reference-Plan Gaps

| Gap | Status |
|-----|--------|
| Real `Neat` population wiring (Team A/B with actual speciation + fitness tracking) | `TeamPopulationContainer` is an opaque handle; wiring deferred to later pass |
| Controller inference inside worker (worker owns `network.activate` per car per tick) | Not yet implemented; worker still delegates physics only |
| Generation-ready summary (`generation-ready` worker→host message) | Protocol types exist; actual generation loop not yet wired |
| Rolling opponent snapshot hall-of-fame + recent sample selection | Placeholder ID only; no actual network payload sampling |
| Deterministic race-pack seed → actual grid positions on the real track geometry | Grid positions are index-based, not track-spline-based |
| Packed `race-step` streaming to host at render cadence | Protocol exists; step loop not yet wired |
| Team A vs B fitness charts, radio heatmaps, role-divergence observability | Renderer/telemetry pass not yet started |
| Sand/wall/off-track lifecycle, pit-entrance blocking, braking/slip-onset tire effects | Surface physics not yet complete |
| Tier-ladder carry-state/reset-state transitions | Not yet implemented |

###### Focused Validation Results

```
validate-plan-sync:             PASS (0 errors, 0 warnings)
npm run quality:folder:         PASS (0 TS errors, 0 ESLint errors, 64/64 JSDoc, 0 missing test files)
npm run build:racing-curriculum PASS (679.1 kb bundle in 132 ms)
npx jest (Step 03 target tests): PASS (17/17 tests, 4 suites)
```

#### Step 05 — Green validation and regression triage [DONE]

```yaml
phase: 1
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 06 — Documentation and educational runtime contract'
```

**User instruction:** Start a fresh session, select `05-green-testing`, and paste this full step packet.

**Step objective:** Prove the implemented tranche is green, does not regress
existing racing behavior, and remains MCP-trackable through plan and gate
validation.

**Context the agent must know:**

- Validate the exact Step 03/04 boundary first, then expand only to nearby
  racing slices and standard build gates.
- Use `test-fix-workflow` if multiple unrelated failures appear.
- If any touched files under `src/` appear, run `coverage-guard` for those
  source files before declaring completion.
- Do not claim full benchmark completion from green runtime foundation tests;
  surface remaining tier, physics, radio, reproduction, and observability gaps.

**Execution steps:**

1. Use `neataptic-validation-mcp.get_active_validation_allowlist` for the active
   allow-listed validation commands when the active step packet exposes them.
2. Run the focused worker/runtime red-test slice from Step 03 and confirm it is
   green.
3. Run owner-local racing tests for environment, track, renderer, worker, and
   browser-entry boundaries touched by Step 04.
4. Run `npm run quality:folder -- --folder=examples/racing_curriculum` and
   `npm run build:racing-curriculum`.
5. Run `npm run build:ts` if TypeScript surfaces outside the bundle-specific
   example build changed.
6. Triage failures to the smallest owner: implementation regression,
   test-contract issue, browser-build issue, upstream NGE primitive, or
   unrelated baseline failure.
7. Record concise validation evidence and any remaining risks.

**Stop conditions:**

- **Done:** focused tests, folder quality, and bundle/build validation pass or
  unrelated baseline failures are clearly isolated with evidence.
- **Hold:** validation exposes a benchmark-policy ambiguity rather than a code
  defect.
- **Blocked:** repeated validation failures require `test-fix-workflow` or
  `00-helping` escalation.
- **Route-back:** return to `04-implementing` for implementation-owned
  regressions or to `03-red-testing` for incorrect red contracts.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `npm run quality:folder -- --folder=examples/racing_curriculum`
- `npm run build:racing-curriculum`
- `npm run build:ts`

**Plan update requirement:** Record pass/fail evidence, regression ownership,
and whether Step 06 may proceed. Compress completed validation detail to a
concise coverage note.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

---

##### Step 05 — Green-Phase Evidence

###### MCP Validation Status

- `plan-session-redirect --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`: PASS
  (`mcp-session-override.json` now points at this tracker, so workflow MCP reads
  Phase 1 / Step 05 correctly).
- `neataptic-validation-mcp.get_active_validation_allowlist`: PASS
  (`requiredValidationCommands` matched the step packet for this tracker).
- `neataptic-validation-mcp.run_allowlisted_validation`: not usable for this step
  because the active packet exposes no executable `validationCommands`; direct
  command execution was used for the required validations instead.

###### Focused Step 03/04 Boundary Validation

- `npx jest --config=jest.config.mjs --no-cache --runInBand --runTestsByPath <Step 03/04 slice>`:
  PASS (10 suites, 41 tests). Worker protocol, coevolution, deterministic
  race-pack, transfer-list safety, packed snapshot helpers, Tier 3/4/5 frame
  factories, renderer parity, environment step service, and `index.test.ts`
  stayed green together.

###### Nearby Owner-Local Regression Validation

- `npx jest --config=jest.config.mjs --no-cache --runInBand --runTestsByPath <owner-local racing slice>`:
  PASS (12 suites, 64 tests). Browser-entry, browser-entry progression, host,
  Tier 5 environment, renderer, track generator, runtime adaptation, worker
  snapshot, and sibling worker-service smoke tests all remained green.

###### Required Validation Results

- `validate-plan-sync`: PASS (0 errors, 0 warnings).
- `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- `npm run quality:folder -- --folder=examples/racing_curriculum`: PASS
  (0 TS errors, 0 ESLint errors, 64/64 exported symbols documented, 0 missing
  sibling tests).
- `npm run build:racing-curriculum`: PASS (browser bundle emitted:
  `docs/assets/racing-curriculum.bundle.js`).
- `npm run build:ts`: PASS.

###### Coverage Note

- No `src/` files changed in this tranche, so `coverage-guard` was not required.

###### Regression Ownership and Remaining Risks

- **Regression ownership:** none detected. No Step 04 implementation regression,
  no Step 03 contract issue, no browser-build failure, no upstream NGE blocker,
  and no unrelated baseline failure were found in the validated slices.
- **Failure recording status:** no gate exception was recorded; `.github/ai-learning/learning-log.jsonl`
  was not updated by Step 05.
- **Step 06 readiness:** yes — documentation may proceed.
- **Still out of scope / still pending:** real Team A/B `Neat` wiring, worker-owned
  controller inference and generation loop, rolling hall-of-fame snapshot
  sampling, track-geometry race-pack placement, tier ladder carry/reset rules,
  surface-physics completion, radio semantics, reproduction analytics, and
  observability charts/heatmaps.

#### Step 06 — Documentation and educational runtime contract [DONE]

```yaml
phase: 1
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[DONE]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 07 — Logging and tracker handoff'
```

**User instruction:** Start a fresh session, select `06-documenting`, and paste this full step packet.

**Step objective:** Document the racing runtime contract so future benchmark
work can understand worker authority, fallback transport, Team A/B semantics,
and the remaining path to full reference-plan completion.

**Context the agent must know:**

- Public docs should explain current concepts, boundaries, invariants, and
  tradeoffs; do not describe chat history or internal plan chronology.
- Generated `docs/examples/**` output must not be edited directly. Edit source
  example docs or JSDoc only when documentation changes are needed.
- If JSDoc or generated README inputs change, run the appropriate docs workflow
  rather than hand-editing generated `src/**/README.md` files.
- Keep the docs honest: worker runtime foundation is not the same as full
  category ladder, radio semantics, polyandric reproduction, or Tier 6 arms-race
  completion.

**Execution steps:**

1. Read Step 04 changed files and Step 05 validation evidence.
2. Update source-facing docs or JSDoc to explain host-owned versus worker-owned
   racing responsibilities, packed race-step snapshots, fallback transport, and
   deterministic race-pack expectations.
3. Document remaining reference-plan gaps: surface physics, full Team A/B
   coevolution depth, radio heatmaps, role-divergence observability, tier
   promotion, and reproduction-mode analytics.
4. Add short examples only where they reflect the actual public/runtime API.
5. If browser-demo source docs changed, run the source generation/build command
   required by the changed surface.
6. Record documentation files touched and any generated artifacts intentionally
   refreshed.

**Stop conditions:**

- **Done:** docs explain the implemented runtime contract and clearly distinguish
  completed foundation from remaining benchmark work.
- **Hold:** documentation would need to describe behavior not yet validated.
- **Blocked:** docs generation or browser-build tooling fails; route to
  `browser-build` or `00-helping` as appropriate.
- **Route-back:** return to `04-implementing` if docs reveal an API or runtime
  contract inconsistency.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `npm run build:racing-curriculum`
- `npm run docs`

**Plan update requirement:** Record docs updated, commands run, generated
artifacts refreshed or intentionally untouched, and the final Step 07 logging
target before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

---

##### Step 06 — Documentation Evidence

###### Files Changed

| File | Change type | Summary |
|------|-------------|---------|
| `workers/simulation-worker/simulation-worker.evolution.types.ts` | Updated | Replaced plan-tracking module comment with atemporal JSDoc explaining worker authority, host/worker split, protocol lifecycle, and remaining gaps; added `RacingWorkerOutboundMessage` typed union with JSDoc; expanded all type JSDoc with invariants and semantics |
| `workers/simulation-worker/simulation-worker.evolution.protocol.service.ts` | Updated | Added module-level JSDoc explaining FSM router, fallback transport, and host/worker authority boundary |
| `browser-entry/browser-entry.ts` | Updated | Added comment block near POC seam types documenting the current step-delegation seam versus the target worker-authoritative protocol |
| `workers/simulation-worker/README.md` | Created | Hand-written educational README documenting host/worker authority split, protocol FSM with Mermaid stateDiagram, race-step snapshot schema, team A/B coevolution semantics, rolling opponent snapshot store, deterministic race-pack contract, fallback transport, and remaining reference-plan gaps |

###### Generated Artifacts

- `docs/assets/racing-curriculum.bundle.js` — refreshed by `npm run build:racing-curriculum` (679.1 kb, no content change from JSDoc-only edits).
- `src/**/README.md` files — `npm run docs` completed successfully; no `src/` JSDoc was changed so no generated README changed.
- `docs/examples/**` — not touched (source example HTML unchanged).

###### Validation Results

- `validate-plan-sync`: **PASS** (0 errors, 0 warnings).
- `npm run build:racing-curriculum`: **PASS** (679.1 kb bundle).
- `npm run docs`: **PASS** (HTML docs generated, 180 Mermaid diagrams validated).
- `npm run quality:folder -- --folder=examples/racing_curriculum`: **PASS** (0 TS errors, 0 ESLint errors, 64/64 exported symbols documented, 0 missing sibling tests).
- Focused Step 03/04 tests: **PASS** (17/17 across 4 suites).

###### Documentation Scope and Gaps

The documentation pass explains the implemented runtime foundation — protocol
lifecycle, host/worker authority, team A/B container semantics, rolling snapshot
guards, deterministic race-pack contract, and transfer-list safety — without
claiming the full category ladder, radio semantics, polyandric reproduction, or
Tier 6 arms-race completion. Remaining gaps are tabulated in the README and
referenced as `NGE_TODO` in the service files.

The boundary was comfortably within a docs-only pass. No `solid-split`
escalation was required.

#### Step 07 — Logging and tracker handoff [WIP]

```yaml
phase: 1
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Close or re-scope the next racing curriculum tranche'
```

**User instruction:** Start a fresh session, select `07-logging`, and paste this full step packet.

**Step objective:** Compress the completed tranche into durable tracker evidence,
record remaining benchmark gaps, and prepare either closure or the next
MCP-tracked racing-curriculum tranche.

**Context the agent must know:**

- This workstream should remain active unless the full reference-plan completion
  boundary is genuinely done; a worker-runtime foundation alone is not terminal
  benchmark completion.
- Tracker history should be compressed to concise coverage notes once each phase
  or step completes.
- If the workstream closes, follow `tracker-handoff`: refresh the same-boundary
  `.logs.md`, remove stale handoff scaffolding, and archive the closed tracker
  pair under `plans/completed/`.
- If the workstream continues, keep a fresh `Handoff query` naming the next
  narrow boundary.

**Execution steps:**

1. Summarize completed Step 02-06 evidence into compact coverage notes.
2. Record durable changed-file groups, validation outcomes, and residual risks.
3. Run MCP gate checks for plan-sync, step-packet, routing-table freshness, and
   stale-WIP risk where applicable.
4. If no customization gap occurred, record that no agent/skill/hook changes
   were required; if a gap occurred, ensure the learning event was recorded.
5. Decide whether the next tranche should target full Team A/B coevolution,
   surface/boundary physics, category ladder promotion, radio/observability, or
   docs/browser polish.
6. Refresh the `Handoff query` for the next active boundary or close/archive the
   tracker only if the plan is terminally complete.

**Stop conditions:**

- **Done:** tracker evidence is compressed, the active next boundary is clear,
  and required gates pass or have owner-routed blockers.
- **Hold:** user must choose the next tranche after the worker-runtime foundation.
- **Blocked:** stale-WIP, plan-sync, or step-packet gates fail and cannot be
  repaired locally; escalate to `00-helping`.
- **Route-back:** return to the owner of any incomplete Step 02-06 evidence.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `node scripts/agent-customization/gates/routing-table-freshness.gate.mjs --json`
- `node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json`

**Plan update requirement:** Update this tracker with the compressed log summary,
next tranche or closure state, gate evidence, and handoff prompt before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

##### Step 07 — Logging Evidence

- Step 02 compressed the boundary map: worker-owned simulation/evolution plus
  packed race-step streaming was selected because it is the first plan-fidelity
  prerequisite for Team A/B coevolution.
- Step 03 added the smallest failing owner-local tests for the worker FSM,
  deterministic race-pack replay, transfer-list ownership, Team A/B isolation,
  best-finisher team fitness, and the opponent-snapshot barrier.
- Step 04 implemented the worker-authoritative foundation: typed protocol
  boundaries, independent Team A/B containers, deterministic race packs,
  packed frame ownership, and frozen rolling-opponent snapshot rotation.
- Step 05 validated the focused Step 03/04 slice plus nearby owner-local
  regressions; no Step 04 regression, upstream NGE blocker, or browser-build
  failure was detected.
- Step 06 documented host-owned versus worker-owned responsibilities, fallback
  transport, packed snapshot semantics, and the honest remaining benchmark gaps.
- Changed-file groups stayed localized to `examples/racing_curriculum/workers/
  simulation-worker/`, the browser-entry/runtime contract, and the source-facing
  docs/JSDoc that feed generated README and browser assets.
- No agent, skill, hook, or MCP customization changes were required, and no
  learning-event entry was needed for this tranche.
- Residual risks remain in worker-owned controller inference and generation
  loop depth, full Team A/B coevolution, surface/boundary physics, tier ladder
  promotion, radio/observability, and reproduction analytics.
- Same-boundary durable log: `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`
  captures the compressed Step 02-06 evidence and current next-tranche boundary.

## Validation gates

- `plan-sync`: confirms the active [WIP] plan is registered in plan indexes.
- `step-packet`: confirms Step 02-07 packets are copy-pasteable and MCP-readable.
- `routing-table-freshness`: confirms agent/skill routing metadata is current.
- `stale-wip-plans`: used by Step 07 before closure or archival handoff.

## Latest validation evidence

- MCP workflow snapshot confirms `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
  is the active repo-static workflow source.
- MCP Cortex index statistics: 1379 documents, 30909 chunks, latest build
  timestamp `2026-06-01T00:15:50.727Z`.
- MCP routing-table freshness reports pass for `.github/agent-skill-routing-table.md`.
- 00-helping gap audit found no critical agent, skill, hook, or MCP tooling
  blocker; the required fix was plan packet content, not customization
  infrastructure.
- `validate-plan-sync.mjs` passes for
  `plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`.
- `validate-plan-phase-packets.mjs` passes once Step 07 is marked [WIP].
- `routing-table-freshness.gate.mjs` passes with a fresh generated routing table.
- `stale-wip-plans.gate.mjs` passes; no stale terminal WIP plans are present.
- No customization gap occurred, so no learning-event record was written.
- Same-boundary log refreshed at `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active workstream:
- plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
- Phase 1 Step 07 — Logging and tracker handoff [WIP]
- Steps 02–06 complete: boundary map, red tests, worker-runtime implementation,
  green validation, and documentation are compressed into durable tracker
  evidence.

Use 04-implementing. The current tranche is logged; keep the workstream open and
route the next tranche to full Team A/B coevolution with worker-owned controller
inference and the generation loop. Do not close/archive yet: the reference-plan
boundary is still incomplete. Refresh the Handoff query after the next tranche
is selected.

Required validations:
- node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md
- node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md
- node scripts/agent-customization/gates/routing-table-freshness.gate.mjs --json
- node scripts/agent-customization/gates/stale-wip-plans.gate.mjs --json
```
