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
primary_boundary: runtime_evolution_worker_streaming_foundation
reason:
  - "Runtime/evolution semantics are the highest-fidelity gap against the reference plan."
  - "Worker-owned authority is required before Team A/B coevolution can run at browser frame rates."
  - "Flappy Bird already proves the host/worker split: generation-ready summaries plus real-time playback-step snapshots."
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
    - "00-helping only when an MCP/tool/agent/flow gap blocks the active step."
non_goals:
  - "Do not hand-code queen, blocker, pacer, or pit-strategy roles."
  - "Do not claim full Tier 5 or Tier 6 completion before Team A/B coevolution and observability gates exist."
  - "Do not move browser rendering authority into workers; workers own simulation/evolution, host owns DOM/canvas presentation."
  - "Do not patch missing NGE primitives locally inside the demo; route missing core primitives to NGE Core Scout."
  - "Do not edit generated docs/examples output directly."
acceptance_criteria:
  - id: worker_authority
    criterion: "Given the browser demo is running, when evolution or race playback advances, then controller inference and simulation state advance in a worker rather than on the main thread."
    validation: "focused worker protocol tests plus browser bundle build"
  - id: streaming_frames
    criterion: "Given a race is active, when the host requests live display, then it receives compact race-step snapshots suitable for render-cadence streaming."
    validation: "packed snapshot transfer-list tests"
  - id: independent_populations
    criterion: "Given Team A and Team B evolve, then each team keeps independent population/species/fitness state and only interacts through the race environment."
    validation: "coevolution container tests"
  - id: frozen_opponents
    criterion: "Given a generation starts, then evaluation uses a frozen rolling opponent snapshot until the configured snapshot boundary."
    validation: "snapshot selection and generation-barrier tests"
  - id: deterministic_packs
    criterion: "Given the same race pack seed and opponent snapshot, then repeated evaluation returns stable comparable episode inputs."
    validation: "deterministic replay tests"
  - id: tier_ladder
    criterion: "Given a team promotes, then the category ladder advances only through deterministic pack reliability and records explicit carry-state/reset-state behavior."
    validation: "tier transition tests"
  - id: observability
    criterion: "Given training progresses beyond early tiers, then UI telemetry exposes Team A vs Team B fitness, radio/tire/pit signals, role divergence, and reproduction-mode summaries."
    validation: "renderer/telemetry tests and manual browser smoke"
stop_conditions:
  done: "All Step 02-07 packets have completed and the tracker is closed by 07-logging."
  hold: "The active step needs user prioritization or benchmark-policy clarification."
  blocked: "A missing upstream NGE primitive or MCP/tooling gap prevents honest implementation."
```

## Implementation phases

### Phase 1 — Racing curriculum refactor packetization [WIP]

#### Step 01 — Gap audit and first-boundary selection [DONE]

[DONE] Gap audit compressed: `examples/racing_curriculum` is a deterministic
POC with partial track, tire, pit, renderer, and worker seams; the selected first
boundary is runtime/evolution plus worker streaming because it is the highest
plan-fidelity and performance prerequisite for Team A/B coevolution.

#### Step 02 — Research boundary mapping [WIP]

```yaml
phase: 1
step: 2
agent: '02-researching'
agent_file: '.github/agents/02-researching.agent.md'
status: '[WIP]'
mode: 'fresh-session'
source_of_truth: 'plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md'
copy_paste: 'true'
next_step: 'Step 03 — Red tests for worker runtime and coevolution contracts'
```

**User instruction:** Start a fresh session, select `02-researching`, and paste this full step packet.

**Step objective:** Map the exact racing-curriculum runtime, worker, and
coevolution boundaries needed to replace the current POC loop with an
MCP-trackable Team A/B worker-authoritative benchmark foundation.

**Context the agent must know:**

- Use `examples/racing_curriculum/reference.plans.md` as the benchmark intent
  reference and this tracker as the active MCP source of truth.
- Preserve Team A/B independence, identical-DNA teammates, team radio as
  stigmergy, rolling opponent snapshots, category ladder, deterministic race
  packs, carry-state/reset-state semantics, and polyandric reproduction terms.
- Flappy Bird is the worker-streaming reference: the host sends coarse
  generation/playback requests while the worker owns evolution/playback state and
  streams compact playback-step snapshots.
- Cortex evidence already points at worker authority, packed typed arrays,
  generation-ready summaries, nested worker-pool capability gates, and
  Predator/Prey generation barriers as relevant patterns.
- 00-helping found no customization blocker; if a later MCP/tool/agent gap
  blocks the step, escalate to `00-helping` before continuing.

**Execution steps:**

1. Use `neataptic-workflow-mcp.get_active_workflow_snapshot` to confirm this
   tracker is the active source and Step 02 is the WIP packet.
2. Use `neataptic-cortex-mcp.search_corpus` for worker streaming, Flappy
   evolution worker, racing curriculum, Predator/Prey generation barrier,
   `createNeatParallelPopulationEvaluator`, packed typed-array transport, and
   zero-copy render-frame transfer context.
3. Read nearest relevant READMEs first, then map source files under
   `examples/racing_curriculum/browser-entry`, `examples/racing_curriculum/workers`,
   `examples/racing_curriculum/evaluation`, `examples/racing_curriculum/environment`,
   and `examples/flappy_bird/flappy-evolution-worker`.
4. Produce a boundary map separating host-owned DOM/canvas/HUD state from
   worker-owned environment, controller inference, generation lifecycle,
   snapshot packing, and optional nested evaluation pools.
5. Identify current POC gaps against the reference plan: Team A/B independent
   populations, rolling snapshots, generation barrier, deterministic race packs,
   packed race-step frames, radio field transport, tier ladder state, and
   benchmark observability.
6. Classify every missing prerequisite as demo-local, reusable library/API,
   upstream NGE core, MCP tooling, or user-policy decision.
7. Record the smallest implementable red-test boundary for Step 03 and route
   any missing NGE primitive to NGE Core Scout instead of compensating in demo
   code.

**Stop conditions:**

- **Done:** research records file-level seams, worker protocol candidates,
  packed-frame schema gaps, test targets, and prerequisite classifications for
  Step 03.
- **Hold:** a benchmark-policy decision is needed before red tests can name
  expected behavior.
- **Blocked:** an MCP/tool/agent gap or missing upstream NGE primitive prevents
  honest boundary mapping; escalate to `00-helping` or NGE Core Scout.
- **Route-back:** return to `01-planning` if the selected first boundary proves
  too broad for one implementation tranche.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`
- `node scripts/agent-customization/gates/routing-table-freshness.gate.mjs --json`

**Plan update requirement:** Update this tracker with the boundary map,
classified prerequisites, selected Step 03 red-test targets, Cortex evidence,
and any escalation outcome before ending.

**Whole-step copy rule:** The entire step block above is the prompt. Do not append a second nested `Copy-paste prompt` subsection.

#### Step 03 — Red tests for worker runtime and coevolution contracts [PLANNED]

```yaml
phase: 1
step: 3
agent: '03-red-testing'
agent_file: '.github/agents/03-red-testing.agent.md'
status: '[PLANNED]'
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

- Step 02 owns the final boundary map; do not widen beyond its selected test
  targets.
- Tests must follow the single-expect rule and should fail for the intended POC
  gap before implementation begins.
- Prioritize behavior that unlocks performance and plan fidelity: worker-owned
  simulation/evolution, compact race-step streaming, Team A/B population state,
  rolling opponent snapshots, generation barriers, and deterministic race packs.
- Do not write tests that require hard-coded roles or prescribed radio
  semantics; role differentiation and radio meaning must remain emergent.

**Execution steps:**

1. Read the Step 02 boundary map and nearest owner-local racing worker/runtime
   tests.
2. Add or update failing tests for worker protocol lifecycle:
   `init -> request-generation -> start-race -> request-race-step -> stop`.
3. Add failing tests for packed `race-step` snapshot schemas and transfer-list
   ownership so detached buffers cannot be reused accidentally.
4. Add failing tests for independent Team A/B population containers, team-level
   fitness based on best-finishing teammate, and frozen rolling opponent
   snapshot selection.
5. Add failing tests for deterministic race-pack replay and generation barrier
   semantics, using the smallest seed pack that exposes the current gap.
6. If Step 02 classified surface physics or observability as inside the first
   tranche, add only the minimum failing tests needed to protect that boundary.
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

#### Step 04 — Implement worker-authoritative racing runtime foundation [PLANNED]

```yaml
phase: 1
step: 4
agent: '04-implementing'
agent_file: '.github/agents/04-implementing.agent.md'
status: '[PLANNED]'
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

#### Step 05 — Green validation and regression triage [PLANNED]

```yaml
phase: 1
step: 5
agent: '05-green-testing'
agent_file: '.github/agents/05-green-testing.agent.md'
status: '[PLANNED]'
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

#### Step 06 — Documentation and educational runtime contract [PLANNED]

```yaml
phase: 1
step: 6
agent: '06-documenting'
agent_file: '.github/agents/06-documenting.agent.md'
status: '[PLANNED]'
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

#### Step 07 — Logging and tracker handoff [PLANNED]

```yaml
phase: 1
step: 7
agent: '07-logging'
agent_file: '.github/agents/07-logging.agent.md'
status: '[PLANNED]'
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
  blocker; the required fix is plan packet content, not customization
  infrastructure.
- `validate-plan-sync.mjs` passes for
  `plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md`.
- `validate-plan-phase-packets.mjs` passes for Phase 1 Step 02-07 packets.
- `neataptic-validation-mcp.get_active_validation_allowlist` recognizes Step 02
  required validation commands from the active plan packet.

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Active workstream:
- plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.md
- Phase 1 Step 02 — Research boundary mapping [WIP]
- Source reference: examples/racing_curriculum/reference.plans.md
- Selected first boundary: runtime/evolution plus worker streaming foundation.

Use 02-researching. Start by calling neataptic-workflow-mcp.get_active_workflow_snapshot,
then use Cortex search for racing curriculum, Flappy worker streaming, packed typed-array
transport, Predator/Prey generation barriers, and Neataptic browser worker evaluation.

Produce a file-level boundary map for host-owned versus worker-owned racing runtime state,
classify missing prerequisites, and choose the smallest Step 03 red-test targets.
Required validations:
- node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md
- node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans\NEAT_Genesis_EvoDevo_Racing_Curriculum.md
```
