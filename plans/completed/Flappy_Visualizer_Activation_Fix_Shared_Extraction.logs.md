# Flappy Visualizer Activation Fix & Shared Extraction — Phase 1 log

**Status:** [DONE]

This file archives the detailed Phase 1 record (Steps 01–03, their step packets, slices, and validation evidence) after compression in the active tracker. The high-level outcome is that the playback protocol now carries the frame winner's node activations to the host and the host overlay copies them into the visualization network, with activation-aware cache invalidation. AC-005 is CLOSED (fix-loop iteration 3, 2026-09-12): the worker inference-channel write-back was fixed in iteration 1, two false-negative smokes were traced to a stale worker bundle (host-only build command), and after a full `npm run build:flappy-bird` the user manually confirmed live non-zero, changing node labels on their deployment at 11:43; green re-run iteration 3 passed all sweeps and gates with AC-005 accepted via that manual confirmation (closure evidence in the active plan tail).

### Learning event — unit suites green while production read zeros

**Trigger:** Phase 1 Step 03 visible-window browser smoke.
**Gap:** Jest suites passed while real browser frames showed all
`winnerNodeActivations` as `0.00`. Root causes: (1) the worker's
`bird.inferenceChannel?.predict(...)` fast path returned outputs without
writing `node.activation` back, so the worker snapshot packer streamed
zeros; (2) `npm run build:flappy-bird:host` refreshed only the host bundle
and left the evolution-worker bundle stale, producing a second
false-negative smoke. Only real visible-window browser validation caught
both.
**Resolution:** Recorded this durable learning event and refreshed the
demo/UI slice validation policy: require a full bundle build that refreshes
both host and worker entries plus a visible-window browser smoke before
green.

### Phase 1 — Live champion activations (fix the 0.0 regression) [DONE]

**Phase objective:** The flappy network visualizer renders the frame
winner's real per-node activations live during playback; no node shows a
stale `0.00` after the first streamed frame; racing_curriculum and
asciiMaze visualizers are unaffected.

```yaml
phase: 1
title: 'Live champion activations (fix the 0.0 regression)'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Shared network visualization extraction'
skills:
  - 'plan-alignment'
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
acceptance_criteria:
  - 'Frame winner node activations reach the host and label values update live (AC-001..AC-005)'
  - 'All flappy / racing / asciiMaze Jest suites green (AC-006)'
placeholder_steps:
  - 'Step 01 — Plan the fix and extraction phases (this tracker)'
  - 'Step 02 — Stream winner activations through the playback protocol'
  - 'Step 03 — Green validation and live browser smoke'
```

**Phase progression rule:** Start with Step 01. Step 01 must author the
remaining numbered step packets before the phase can advance. Per the
author-verify loop, a fresh 01-planning verification instance must record
`green-light` in `## Latest validation evidence` before any Step 02 slice
flips to `[WIP]`.

---

#### Step 01: Plan the fix and extraction phases [DONE]

```yaml
phase: 1
step: 1
title: 'Plan the fix and extraction phases'
status: '[DONE]'
goal: 'planning'
mode: 'fresh-session'
source_of_truth: 'plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
copy_paste: true
next_step: 'Await fresh verification green-light in ## Latest validation evidence, then Step 02 — Stream winner activations through the playback protocol'
skills:
  - 'plan-alignment'
  - 'planning-acceptance-criteria'
  - 'phase-handoff-workflow'
  - 'tracker-handoff'
validation:
  - 'neataptic-gate-mcp:run_gate_check — gate: slice-advancement, args: { slice-id: 01-plan-visualizer-fix, changed-files: [plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md, plans/README.md, plans/Roadmap.md] }'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
acceptance_criteria:
  - id: AC-101
    text: 'Plan contains Phase 1 + Phase 2 kickoff packets and Steps 02-07 packets with valid goal/tdd_sequence/slice shapes'
    validation: 'neataptic-gate-mcp:run_gate_check — gate: slice-advancement'
  - id: AC-102
    text: 'plans/README.md and plans/Roadmap.md carry an active entry for this workstream'
    validation: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
```

**Step objective:** Author a machine-readable tracker that records the
root-cause diagnosis, the streaming fix design, and the extraction boundary
so specialists can execute without re-research.

**Evidence recorded this session:** root cause verified against source
(`fromJSON` activation drop; host cache invalidation sites); visualizer
contract proven green (23/23 network-view Jest tests pass, including
"renders the input node activation value after a forward pass"); importer
sweep and extraction boundary mapped by `boundary-mapper`; protocol hop map
and SoA conventions mapped by `implementation-pattern-scout`.

---

#### Step 02: Stream winner activations through the playback protocol [DONE]

```yaml
phase: 1
step: 2
title: 'Stream winner activations through the playback protocol'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
copy_paste: true
next_step: 'Step 03 — Green validation and live browser smoke'
skills:
  - 'implementation-standards'
  - 'green-validation-gates'
  - 'plan-sync-validation'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/flappy-evolution-worker" --runInBand'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/playback" --runInBand'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/runtime" --runInBand'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/host" --runInBand'
acceptance_criteria:
  - id: AC-001
    text: 'createWorkerPlaybackSnapshot packs winnerNodeActivations equal to the winner bird post-step node activations and the transfer list carries its buffer'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/flappy-evolution-worker/flappy-evolution-worker.snapshot.utils" --runInBand'
  - id: AC-002
    text: 'Per-frame (done:false) playback messages carry winnerNodeActivations + winnerBirdIndex and values change across frames'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/flappy-evolution-worker/flappy-evolution-worker.playback.service" --runInBand'
  - id: AC-003
    text: 'Main-thread applyPlaybackSnapshot and frameStats expose winnerNodeActivations + winnerBirdIndex to runtime callbacks'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/playback" --runInBand'
  - id: AC-004
    text: 'NetworkVisualizationHandle.applyNetworkActivationOverlay copies values into node.activation, clears the cached frame, and requests redraw'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/host" --runInBand'
  - id: AC-252
    text: 'Runtime champion callback applies the overlay each frameStats tick using the existing champion network instance'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/runtime" --runInBand'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
slices:
  - slice_id: '02-red-activation-stream'
    title: 'Red tests for winner activation streaming and host overlay'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.test.ts'
      - 'examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.playback.service.test.ts'
      - 'examples/flappy_bird/browser-entry/playback/playback.snapshot.utils.test.ts'
      - 'examples/flappy_bird/browser-entry/playback/playback.iteration.services.test.ts'
      - 'examples/flappy_bird/browser-entry/host/host.network-visualization.controller.test.ts'
      - 'examples/flappy_bird/browser-entry/runtime/runtime.evolution-loop.service.test.ts'
    acceptance_criteria:
      - id: AC-201
        text: 'Failing tests assert packed winnerNodeActivations + transfer-list membership'
      - id: AC-202
        text: 'Failing tests assert per-frame carry of winner activations + winnerBirdIndex through playback frames'
      - id: AC-203
        text: 'New host controller test asserts overlay copy, cached-frame invalidation, and redraw request'
      - id: AC-204
        text: 'Failing runtime test asserts overlay applied on champion visualization network per frameStats tick'
    parallelizable: false
    dependencies: []
    next_slice: '02-worker-capture'
    red_evidence:
      recorded_by: '03-red-testing'
      status: 'RED CONFIRMED — 16 new/updated contracts fail for the intended behavior; every pre-existing test still passes'
      focused_commands:
        - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/flappy-evolution-worker" --runInBand → exit 1 (4 failed, 51 passed, ~46s)'
        - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/playback" --runInBand → exit 1 (3 failed, 29 passed, ~10.5s)'
        - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/runtime" --runInBand → exit 1 (4 failed, 24 passed, ~16s)'
        - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/host" --runInBand → exit 1 (4 failed, 5 passed, ~14.4s)'
      failure_highlights:
        - 'AC-201: worker snapshot packing lacks winnerNodeActivations (Received without the key); transferListLength 7 vs 8 with winnerActivationsBuffer missing; winner-packing test receives undefined instead of the winner activations'
        - 'AC-202: live done:false frames carry no winnerBirdIndex/winnerNodeActivations (both undefined while firstFrameDone false proves the done:false path ran)'
        - 'AC-003a: applyPlaybackSnapshot render state lacks the winner fields entirely; stale pre-seeded winnerBirdIndex 1 / [9,9,9] survive instead of being cleared when the snapshot omits the stream'
        - 'AC-003b: emitPlaybackFrameStats forwards only legacy stats keys (Received 6 legacy fields, Expected winner fields)'
        - 'AC-203: createHostNetworkVisualizationController is not exported from ./host yet — all 4 tests fail on the descriptive resolver guard (implement per AC-212 in slice 02-host-overlay)'
        - 'AC-204: applyRuntimeChampionActivationOverlay is not exported from ./runtime.evolution-loop.service yet — all 4 tests fail on the descriptive resolver guard (implement per AC-213 in slice 02-host-overlay)'
      fixture_notes:
        - 'New-field reads/writes use narrowly-scoped single casts so all tests compile under ts-jest diagnostics:true and fail at RUNTIME assertion (missing implementation), never at TS compile'
        - 'Deterministic fixtures: seed 42; worker 2-bird fixture flips winner bird 1 → bird 0 between frames via pipesPassed (5>3 then 9 vs 3 after mutation) to prove per-frame streaming; playback AC-202 delegates to the real createWorkerPlaybackSnapshot so the contract is agnostic to payload-sourcing strategy'
        - 'Host test harness: document.createElement + requestAnimationFrame stubs installed in beforeEach and deleted in afterEach; rAF callbacks captured and drained manually (flushPendingRedraws) to defeat redraw coalescing; full 2D-context stub follows the network-view.test.ts createStubContext pattern; real Network(2, 1, { seed: 42 }) instances'
        - 'Host/runtime export resolvers throw clear "not exported yet" errors so RED fails on missing implementation, not opaque undefined-calls'
      handoff_notes:
        - 'Host draw-path stubs were only exercised up to the resolver guard in RED; the first green flush of the render rAF exercises the full draw pipeline and may reveal additional stub needs for 05-green-testing'
        - 'Cached-frame invalidation (latestResolvedFrame) is internal — the unit-level proxy is the redraw request; true relabeling is verified by Step 03 browser smoke (AC-005)'
        - 'runRuntimeEvolutionLoop must CALL applyRuntimeChampionActivationOverlay per frameStats tick; unit tests pin the helper contract, wiring is verified in green/browser smoke'
      expected_green: '04-implementing makes all 16 contracts pass via 02-worker-capture → 02-main-thread-plumb → 02-host-overlay; 02-green-activation-stream then runs all four focused suites green with no skips'
  - slice_id: '02-worker-capture'
    title: 'Worker captures and packs winner node activations'
    status: '[DONE] (succeeded — AC-205/206/207 green after fix-packet-02-worker-capture-iteration-1; worker focused suite 56/56, tsc/lint clean, slice-advancement 7/7)'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types.ts'
      - 'examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts'
      - 'examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.test.ts'
      - 'examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.playback.service.ts'
    acceptance_criteria:
      - id: AC-205
        text: 'WorkerPackedPlaybackBirdSnapshot/message types gain winnerNodeActivations (Float32Array) + winnerBirdIndex'
      - id: AC-206
        text: 'Winner activations captured post-simulation, winner-only, ordered by network node index'
      - id: AC-207
        text: 'Transfer list includes the activations buffer when transferable'
    parallelizable: false
    dependencies:
      - '02-red-activation-stream'
    next_slice: '02-main-thread-plumb'
  - slice_id: '02-main-thread-plumb'
    title: 'Main-thread unpack and frameStats plumbing'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/flappy_bird/browser-entry/browser-entry.worker.types.ts'
      - 'examples/flappy_bird/browser-entry/playback/snapshot/playback.snapshot.services.ts'
      - 'examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.summary.services.ts'
      - 'examples/flappy_bird/browser-entry/playback/playback.iteration.services.ts'
    acceptance_criteria:
      - id: AC-208
        text: 'EvolutionPlaybackStepSnapshot mirror types gain the new fields'
      - id: AC-209
        text: 'applyPlaybackSnapshot surfaces winner activations on main-thread playback state'
      - id: AC-210
        text: 'resolvePlaybackFrameStats + emitPlaybackFrameStats forward winnerNodeActivations + winnerBirdIndex to runtime callbacks'
    parallelizable: false
    dependencies:
      - '02-worker-capture'
    next_slice: '02-host-overlay'
  - slice_id: '02-host-overlay'
    title: 'Host activation overlay with cache invalidation'
    status: '[DONE] (succeeded — AC-211/212/213 green after fix-packet-02-host-overlay-iteration-1; green re-validation OK incl. launch-service forwarding contract test)'
    goal: 'implementing'
    estimate_hours: 2
    files_to_change:
      - 'examples/flappy_bird/browser-entry/browser-entry.visualization.types.ts'
      - 'examples/flappy_bird/browser-entry/host/host.ts'
      - 'examples/flappy_bird/browser-entry/runtime/runtime.evolution-loop.service.ts'
    acceptance_criteria:
      - id: AC-211
        text: 'NetworkVisualizationHandle interface gains applyNetworkActivationOverlay(network, activations)'
      - id: AC-212
        text: 'Host overlay copies activations into node.activation with identity + length guards, clears latestResolvedFrame, requests redraw'
      - id: AC-213
        text: 'Runtime champion callback applies overlay using the existing champion network instance each frameStats tick'
    parallelizable: false
    dependencies:
      - '02-main-thread-plumb'
    next_slice: '02-green-activation-stream'
  - slice_id: '02-green-activation-stream'
    title: 'Green focused suites for the activation stream'
    status: '[DONE] (succeeded — AC-214/215 green, slice-advancement 7/7)'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change: []
    acceptance_criteria:
      - id: AC-214
        text: 'All four focused validation suites green with no skips'
      - id: AC-215
        text: 'No regressions in the full flappy network-view suite (23 existing tests)'
    parallelizable: false
    dependencies:
      - '02-host-overlay'
    next_slice: null
```

**User instruction:** Paste this full step packet.

**Step objective:** Extend the playback protocol so the frame winner's real
node activations reach the host and label values update live.

**Context the agent must know:**

- Winner ↔ champion parity: both sides use the shared
  `resolveFramePrimaryWinnerIndex`; the `currentPopulation[0]` champion fold
  happens ONLY in the `done:true` path, so index mapping is stable during
  playback.
- `Network.activate` returns OUTPUTS only — capture node activations by
  reading `network.nodes[i].activation` after the simulation step, not from
  the activate() return value.
- SoA conventions: Float32Array for continuous values, unit-suffix names,
  defensive length fallbacks, `isTransferableArrayBuffer` filters
  SharedArrayBuffers (never transfer shared buffers).
- Host guards: overlay applies only when the target network instance matches
  the visualized champion (`network === previousNetworkForVisualization`)
  and array length matches node count; otherwise no-op.
- The transient `0.00` state before the first streamed frame (generation
  preview) is expected and acceptable.

**Execution steps:**

1. Author the failing tests listed in `02-red-activation-stream`.
2. Implement worker-side capture + packing (`02-worker-capture`).
3. Thread the fields through main-thread unpack + frameStats
   (`02-main-thread-plumb`).
4. Implement the handle overlay + host invalidation + runtime wiring
   (`02-host-overlay`).
5. Run the four focused suites; fix regressions until green
   (`02-green-activation-stream`).

**Stop conditions:**

- Blocked if the per-frame message size or transfer behavior regresses the
  playback service contract in a way the red tests cannot express; record
  in the tracker and escalate via `00.cross-tier-helper`.

**Required validation:** the four `npx jest` commands in the packet.

**Plan update requirement:** Record slice status flips and validation
output in this tracker before ending the session.

### Validation evidence — Step 02 closure and Step 03 browser smoke (2026-09-12)

- **Step closure:** 06-documenting closed Step 02 after all four
  implementation slices reported green.
- **Deferred specialist observations resolved:**
  - Stale RED-phase comments removed from test helpers in
    `examples/flappy_bird/browser-entry/host/host.network-visualization.controller.test.ts`,
    `examples/flappy_bird/browser-entry/runtime/runtime.evolution-loop.service.test.ts`,
    `examples/flappy_bird/browser-entry/playback/playback.iteration.services.test.ts`,
    `examples/flappy_bird/browser-entry/playback/playback.snapshot.utils.test.ts`, and
    `examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.test.ts`.
  - Added verifiable `@example` blocks and explicit `@param`/`@returns` tags to
    new public exports: `NetworkVisualizationHandle`,
    `createHostNetworkVisualizationController`, `applyRuntimeChampionActivationOverlay`,
    `createWorkerPlaybackSnapshot`, and `resolveWorkerPlaybackWinnerBirdIndex`.
  - Added missing `@param options` tag to `createCanvasHostInternal`.
- **Generated docs refreshed:** `npm run docs` completed successfully; generated
  READMEs under `examples/flappy_bird/browser-entry/host`, `runtime`, `playback`, and
  `flappy-evolution-worker` updated.
- **Focused Jest suites green (2026-09-12):**
  - `npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/(flappy-evolution-worker|browser-entry/(playback|runtime|host))" --runInBand`
  - Result: **24 suites passed, 125 tests passed, 0 skipped**.
- **Docs-quality metrics scan (2026-09-12):**
  - Command: `node rag-index/docs-quality/docs-quality.metrics.mjs --json --scope=paths --run-id step-02-docs-close-final ...`
  - Result: **0 missing JSDoc, 0 weak JSDoc, 0 incomplete tags**; coverage pass.
  - Residual: **3 pre-existing high-complexity findings** (cyclomatic complexity > 10):
    - `createHostNetworkVisualizationController` (18)
    - `createWorkerPlaybackSnapshot` (13)
    - `runRuntimeEvolutionLoop` (12)
  - These functions pre-date or are largely unchanged by the Step 02 documentation pass; complexity reduction is out of scope for 06-documenting and should be owned by 04-implementing if prioritized.
- **Tier-1 gates:**
  - `slice-advancement` for `02-host-overlay` passed (7/7 sub-gates: plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review).
  - `cortex-index` rebuilt (`node rag-index/build-index.mjs`); all index families fresh. Final gate invocation reports `workflow_mcp_alive: false` (infrastructure/MCP bind issue, not content); index content itself is fresh.
- **Step 03 browser smoke (AC-005) -- BLOCKED (2026-09-12):**
  - Command: node tmp_flappy_smoke_run.mjs
  - Browser: Chrome/152.0.0.0, visible-foreground window, http://localhost:3000/examples/flappy_bird/index.html
  - Probe: intercepted 102 playback-step worker frames; winnerNodeActivations length 19 in every frame.
  - Finding: **all winnerNodeActivations values are 0.00 and do not change across frames**.
  - Screenshot: C:\NeatapticTS\tmp\flappy-smoke-1789223686370.png
  - Console errors: one 404 on load (non-blocking).
  - Verdict: AC-005 **FAIL**; network visualizer labels remain frozen at 0.00.
  - Root-cause hypothesis: the worker uses bird.inferenceChannel?.predict(...) when a transferable inference channel is available (flappy-evolution-worker.simulation.frame.service.ts), which returns outputs but does **not** write node activations back to bird.network.nodes[i].activation. The existing worker snapshot capture reads network.nodes[i].activation, so it streams zeros even though the bird is making real control decisions.
  - Action: flipped 02-worker-capture back to [WIP]; requires 04-implementing to ensure post-inference node activations are captured and packed before Step 03 can be unblocked.
- **Step 03 Jest sweeps (AC-006 partial):**
  - flappy_bird: PASS (36 suites, 174 tests).
  - racing_curriculum: PASS (55 suites, 648 tests).
  - asciiMaze: SKIPPED -- jest.config.mjs testPathIgnorePatterns excludes examples/asciiMaze/ in the default project, so the plan's testPathPatterns "asciiMaze" command matches 0 tests. Existing asciiMaze tests are present under examples/asciiMaze/ but are not wired into the default project.
- **Residual risks carried to Step 03:**
  - Pre-existing complexity in the three functions above; no documentation gaps remain.
  - True live-label verification remains the responsibility of Step 03 browser smoke (AC-005).

### Validation evidence — Step 03 re-run after 02-worker-capture fix iteration 2 (2026-09-12)

- **fix-packet-02-worker-capture-iteration-2 status:** PASSED (specialist APPROVE recorded).
- **Step 03 Jest sweeps (AC-006):**
  - flappy_bird: PASS (36 suites, 175 tests).
  - racing_curriculum: PASS (55 suites, 648 tests).
  - asciiMaze-browser: PASS (2 suites, 8 tests).
- **Build gate (AC-216):**
  - `npm run build:flappy-bird:host` → exit 0; `docs/assets/flappy-bird.bundle.js` refreshed (748.1kb).
- **Visible-window browser smoke (AC-005) — STILL FAILING:**
  - Specialist: `browser-ui-specialist` (Tier 3).
  - URL: `http://localhost:8080/examples/flappy_bird/index.html`.
  - Browser: Chrome with `--remote-debugging-port=9222`, user-data-dir `C:\temp\chrome-debug-step03`, `browserVisibility: visible-foreground`.
  - Intercepted playback-step frames: **8,148 frames**, `winnerNodeActivations` Float32Array length 19 in every frame.
  - Finding: **all activation values remain 0.00 and do not change across frames**.
  - DOM probe: 3 canvases present; 3rd canvas (network panel) visible at 786×586.
  - Screenshot: `C:\NeatapticTS\tmp\step03-browser-smoke.png` (296,383 bytes).
  - Console: 0 errors, 0 warnings.
  - Network: all requests 200.
  - Verdict: **AC-005 FAIL**; network visualizer labels stay frozen at 0.00.
  - Root cause: per-bird inference-channel predict path in the worker returns outputs but does not write node activations back into `bird.network.nodes[i].activation`; the snapshot packer reads those zeroed values.
  - Reroute: open new fix packet for `02-worker-capture` (or its successor) and dispatch a fresh `04-implementing` instance to repair the worker-side activation write-back before re-running Step 03.
- **Tier-1 gates:**
  - `convergence-tracker` for `02-worker-capture` → pass (iteration 2 status=passed; within max 4).
  - `slice-advancement` for `03-browser-smoke` → pass with tooling note: `shared-validation` sub-gate errored (`spawnSync node ETIMEDOUT`), treated as tooling failure; content gates (plan-sync, step-packet, plan-slice-quality, plan-command-lint, code-coverage, specialist-review) passed.

---

#### Step 03: Green validation and live browser smoke [DONE]

```yaml
phase: 1
step: 3
title: 'Green validation and live browser smoke'
status: '[DONE]'
goal: 'green-testing'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/Flappy_Visualizer_Activation_Fix_Shared_Extraction.plans.md'
copy_paste: true
next_step: 'Phase 2 kickoff — flip Phase 2 to [WIP]; Step 04 — Settings seam'
skills:
  - 'green-validation-gates'
  - 'plan-sync-validation'
  - 'tracker-handoff'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird" --runInBand'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "racing_curriculum" --runInBand'
  - 'npx jest --config=jest.config.mjs --no-cache --selectProjects asciiMaze-browser --testPathPatterns "asciiMaze/(networkVisualization|browser-entry/network-view)" --runInBand'
  - 'npm run build:flappy-bird'
acceptance_criteria:
  - id: AC-006
    text: 'Full flappy suite plus racing + asciiMaze suites green'
    validation: 'npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird" --runInBand'
  - id: AC-005
    text: 'Real visible-window browser smoke: flappy network panel shows non-zero, changing activation labels after >=2 streamed frames (screenshot + probe evidence in tracker)'
    validation: 'Browser smoke evidence recorded in ## Latest validation evidence'
  - id: AC-216
    text: 'flappy-bird.bundle.js builds clean from the repointed entry'
    validation: 'npm run build:flappy-bird'
constitution_check:
  - 'principle-4-small-slices'
  - 'principle-5-unique-ids'
```

**User instruction:** Paste this full step packet.

**Step objective:** Prove the fix end-to-end: all suites green and the
visualizer shows live non-zero labels in a real visible browser window.

**Context the agent must know:**

- This is a DEMO/UI change under `examples/flappy_bird/browser-entry`:
  mock-only Jest evidence is INSUFFICIENT. Run a real visible-window browser
  smoke (Chrome via the browser harness; see
  `plans/completed/Chrome_MCP_Browser_Tests.plans.md`): serve the flappy
  host page (bundle via `npm run build:flappy-bird` — refreshes the worker
  AND host bundles; a host-only build leaves the evolution-worker bundle
  stale), wait for ≥2
  playback frames, then probe the visualizer state (e.g. the controller's
  last-applied overlay or node label source values) and capture a
  screenshot showing non-zero labels.
- Evidence: screenshot path + probe transcript + frame count recorded in
  `## Latest validation evidence`.
- If the smoke reveals the labels still frozen at 0.00, do NOT patch in
  the green step: record the blocker, flip the owning slice back to [WIP],
  and hand back through the plan tracker.

**Execution steps:**

1. Run the three Jest sweeps; fix nothing here — triage failures back to
   owning slices.
2. Build the host bundle.
3. Run the visible-window browser smoke; capture screenshot + probe.
4. Record evidence; flip statuses; set up Phase 2 kickoff handoff.

---

### Phase 2 — Shared network visualization extraction [DONE]

**Phase objective:** The rich flappy network visualizer now lives in `examples/shared/network-visualization/`, is self-contained (zero shared→flappy import edges), and is consumed by flappy, racing_curriculum, asciiMaze, and neatChat with identical rendering.

**Files changed:**
- `examples/flappy_bird/browser-entry/network-view/*` and `visualization/*`: de-branded via `NetworkVisualizationSettings` and wired to flappy host settings.
- `examples/flappy_bird/browser-entry/host/host.network-view.settings.ts`: new flappy-branded settings module.
- `examples/shared/network-visualization/`: new shared domain containing `network-view/`, `visualization/`, and domain-root `network-visualization.{types,constants,math.utils}.ts`.
- `examples/flappy_bird/browser-entry/browser-entry.types.ts` and `constants/constants.ts`: barrels updated; deleted `browser-entry.visualization.types.ts`, `constants.network-view.ts`, and the dead facades `browser-entry.network-view.utils.ts` + `browser-entry.visualization.utils.ts`.
- `examples/racing_curriculum/browser-entry/network-view/network-view.ts`, `network-view.lod.ts`, `network-view.test.ts`; `examples/asciiMaze/browser-entry/network-view/network-view.ts`; `examples/neatChat/core/neatChat.constants.ts`, `neatChat.test.ts`, `README.md`: repointed to shared paths.
- `examples/asciiMaze/browser-visualizer.ts`: orphaned legacy consumer deleted.
- Generated bundles refreshed: `docs/assets/flappy-bird.bundle.js`, `docs/assets/neat-chat.bundle.js`.

**Validation evidence:**
- `npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird/browser-entry/(network-view|host)" --runInBand` → **PASS** (36/36).
- `npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "racing_curriculum/browser-entry/(network-view|host)" --runInBand` → **PASS** (30/30).
- `npx jest --config=jest.config.mjs --no-cache --selectProjects asciiMaze-browser --testPathPatterns "asciiMaze/browser-entry/(network-view|host)" --runInBand` → **PASS** (7/7).
- `npx tsc --noEmit -p tsconfig.json` → **PASS**; `npm run lint` → **PASS** (0 errors; pre-existing `any` warnings).
- `node scripts/agent-customization/gates/folder-quality.gate.mjs --folder=examples/shared/network-visualization --json` → **PASS** (0 gaps).
- `npm run build:flappy-bird` and `npm run build:neat-chat` → **PASS**.
- Importer sweep → **0** live references to old flappy visualizer import paths.

**Key decisions:** Settings seam built on the existing `InputLabelGroupDefinition[]` surface; value-identical `NETWORK_*` defaults keep zero-settings consumers unchanged; flat 1:1 physical move with dependency relocation (not repoint-back) preserved the zero shared→flappy edge invariant; sibling tests and dead-facade deletion landed in the same step as the move.

**Residual risks:** Generated bundle/snapshot paths can go stale after later edits; future consumers must not reintroduce flappy-branded imports into the shared domain.

---

### Phase 3 — racing_curriculum visualizer migration [DONE]

**Phase objective:** Racing’s LOD renderer, tooltip resolver, and racing semantic settings moved into the shared visualizer; redundant racing-specific modules deleted; racing behavior preserved via host-passed settings.

**Files changed:**
- New `examples/shared/network-visualization/network-visualization.tooltip.service.ts` + `.test.ts`; new `examples/shared/network-visualization/network-visualization.lod.service.ts` + `.test.ts`.
- `examples/shared/network-visualization/network-visualization.{types,constants}.ts`: extended for tooltip and LOD settings.
- Deleted: `examples/racing_curriculum/browser-entry/network-view/network-view.lod.ts`, `network-view.fixture.ts`; `examples/racing_curriculum/browser-entry/host/host.network-tooltip.service.ts` + `.test.ts`; `examples/flappy_bird/browser-entry/host/host.network-tooltip.service.ts` + `.test.ts`.
- `examples/racing_curriculum/browser-entry/network-view/network-view.ts`, `network-view.constants.ts`, `network-view.test.ts`; `examples/racing_curriculum/browser-entry/host/host.ts`, `host.types.ts`: thin adapter repoint to shared LOD/tooltip/draw.
- `examples/flappy_bird/browser-entry/host/host.ts`: repointed to shared tooltip resolver.
- Regenerated docs: `examples/flappy_bird/browser-entry/host/README.md`, `examples/racing_curriculum/browser-entry/host/README.md`, `examples/racing_curriculum/browser-entry/network-view/README.md`; `rag-index/snapshots/semantic-snapshot.json` and `docs/assets/semantic-snapshot.json` refreshed.

**Validation evidence:**
- Step 09 tooltip migration: shared 6 suites/59 tests, racing 56 suites/653 tests, flappy host 2 suites/9 tests; `npx tsc`, `npx eslint`, `npm run build`, `api-contract-reviewer` APPROVE → **PASS**.
- Step 10 LOD migration: red tests resolved; shared + racing suites green.
- Step 11 racing adapter: `npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "racing_curriculum" --runInBand` → **PASS** (54 suites, 643 tests); `folder-quality` → **PASS**; `slice-advancement` → **PASS**.
- Step 12 cross-demo: `npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird|shared/network-visualization|racing_curriculum|asciiMaze" --runInBand` → **PASS** (96 suites, 861 tests); `npx jest ... --selectProjects asciiMaze-browser ...` → **PASS** (2 suites, 8 tests); `npm run build:flappy-bird:host` (759.3 kb) and `npm run build:racing-curriculum` (799.7 kb) → **PASS**; `folder-quality` → **PASS**; `slice-advancement` → **PASS**.
- Docs closure: `npm run docs`, `npm run index:build`, `npm run index:build-snapshot`, `npm run docs:quality:gate`, `npm run docs:quality:metrics`, and slice-advancement → **PASS**.

**AC-280 SUSPENDED note:** AC-280 (visible-window browser smokes for flappy and racing network panels after the migration) is **SUSPENDED** per user mandate; acceptance authority remains manual runtime confirmation. No automated browser run was performed. Manual confirmation expected at `http://192.168.1.209:8080/docs/examples/`.

**Key decisions:** Tooltip resolver unified from the nearly-identical flappy and racing host services into a shared settings-driven service; LOD renderer moved to the shared domain as a reusable, threshold-configurable feature; racing-specific input groups, output labels, and connection style remain as racing host-passed settings.

**Residual risks:** AC-280 is the only unverified acceptance gate; future browser smokes must use a full bundle build that refreshes both host and worker bundles to avoid stale-worker false negatives. No automated regression coverage exists for the live network-panel rendering path.
