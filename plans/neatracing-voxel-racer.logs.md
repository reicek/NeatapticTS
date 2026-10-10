# Neat Racing Voxel Racer log

**Status:** [WIP] — companion log for `plans/neatracing-voxel-racer.plans.md` (same workstream boundary; created per the tracker-handoff convention). Completed-step packets and compressed evidence summaries are archived here; the tracker keeps a compact [DONE] marker plus a reference to this file. This file is not gate-parsed — the tracker's fenced YAML packets remain the machine contract.

## Step 01 — Author step packets and register plan [DONE]

Archived verbatim from the tracker at the Step 01 → [DONE] flip (Evidence batch 8, items 28-29).

```yaml
phase: 1
step: 1
title: 'Author step packets and register plan'
status: '[WIP]'
goal: planning
expansion: none
auto_expand: false
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 02 - Shell canvas + WebGL2 boot + render loop (spec P1-S1) - dispatch after a verification instance records green-light'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - spec-checklist
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement with args slice-id=P1-step-01-author changed-files=plans/neatracing-voxel-racer.plans.md,plans/README.md,plans/Roadmap.md,.vscode/mcp.json'
  - 'neataptic-gate-mcp:run_gate_check gate=cortex-index'
constitution_check: 'principle-5-unique-ids; section-development-workflow'
acceptance_criteria:
  - 'AC-PLN-001: spec section 17 slices P1-S1..P1-S7 appear verbatim (slice_id + files_to_change + AC binding) in Steps 02-12 packets'
  - 'AC-PLN-002: all 19 step packets and 3 phase packets conform to the step-packet gate schema (WIP blocks gate-validated; PLANNED blocks authored gate-compliant for later WIP flips)'
  - 'AC-PLN-003: coverage backlog maps AC-001..AC-043 (43/43) to steps/slices; section 18 traceability preserved'
  - 'AC-PLN-004: plans/README.md + plans/Roadmap.md carry the [WIP] entry; plan-session-redirect.mjs ran with --plan=plans/neatracing-voxel-racer.plans.md; .vscode/mcp.json MCP --plan args rebound'
  - 'AC-PLN-005: RAG plan family reindexed (BM25 sync + embeddings queued) and cortex-index gate result recorded in Latest validation evidence'
```

**User instruction:** Author all NeatRacing phase/step/slice packets from the spec, register the plan as the active [WIP] workstream, rebind the session, reindex RAG, and record gate evidence; planning only, no production code edits.

**Stop conditions:**
- **Done:** 3 phase packets + 19 step packets gate-conformant; plan registered + rebound; cortex-index gate green; evidence recorded in Latest validation evidence.
- **Blocked:** validator or gate crash — set TASK_STATUS: PARTIAL, record evidence, escalate via 00.cross-tier-helper.

**Step objective:** Author all 19 step packets and 3 phase packets with gate-compliant slice packets derived verbatim from spec section 17; register the plan in plans/README.md and plans/Roadmap.md as the active [WIP] workstream; rebind plan-session-redirect.mjs and .vscode/mcp.json MCP --plan args; reindex the RAG plan family; record all gate evidence in Latest validation evidence. Planning only: no production code changes.

**Required validation:**
- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md`
- `neataptic-gate-mcp:run_gate_check gate=slice-advancement with args slice-id=P1-step-01-author changed-files=plans/neatracing-voxel-racer.plans.md,plans/README.md,plans/Roadmap.md,.vscode/mcp.json`
- `neataptic-gate-mcp:run_gate_check gate=cortex-index`

### Step 01 evidence summary (batches 3-7, compressed)

Full itemized records live in the tracker's `## Latest validation evidence` (items 1-27); this is the compressed archive.

- **Batch 3 — final authoring pass:** dual-parser heading restructure (H3 phases + H4 steps with em-dash and trailing status, `## Implementation phases` wrapper, `## Validation gates` H2 terminator); workflow MCP self-check `ok: true, 0 errors, 0 warnings`; plan-sync PASS; reindex chain → 1892 documents / 25,276 chunks; slice-advancement FULL 7/7 PASS via direct CLI (`tmp/slice-advancement-final.json`, final-state rerun `tmp/slice-advancement-final2.json`); cortex-index gate PASS (all 10 families fresh, corpus + workflow MCP alive). Workflow gap found: gate-MCP 15 s relay timeout vs ~37-46 s FULL runs.
- **Batch 4 — maintenance fix (00-helping):** gate-MCP `GATE_SPAWN_TIMEOUT_MS` raised 15 s → 120 s; slice-advancement gained `--plan` forwarding to plan-command-lint (recorded as `evidence.planCommandLintPlan`; previously silently linted the stale `plans/orchestration-fixes.plans.md` default); FULL rerun PASS in 46.1 s with correct `planCommandLintPlan`; gate-MCP relay run PASS in 37.0 s; targeted gate suites 24 + 23 tests pass; plan-command-lint 33 commands / 0 issues. Open (pre-existing, unfixed): `neataptic-validation-mcp` self-check ENOENT — `run_allowlisted_validation` has no routing for `neataptic-gate-mcp:*` pseudo-commands.
- **Batch 5 — packet-contract completion:** 48 inline `dependencies: ['X','Y']` flow arrays converted to YAML block lists; a scripted re-indent pass swallowed 30 subsequent `- slice_id:` headers (7 steps reported slice-merge errors) — all restored and verified; `validate-plan-phase-packets.mjs` 0 errors / 33 warnings (tolerated classes only: unexpected pre_execute_hook / constitution_check keys; non-path validation entries relocated verbatim to prose); plan-sync PASS; tracker reindexed; final-state slice-advancement + cortex-index PASS.
- **Batch 6 — verification pass (independent 01-planning instance):** scripted structural verification — 18 execution steps, 68 slices = 50 impl (1:1 vs spec section 17) + 18 step-green; every step ≤5 slices; every slice ≤3 non-trivial files; estimates 1-4 h; AC-001..AC-043 mapped 43/43; registration + session rebind verified (`plans/README.md`, `plans/Roadmap.md`, `data/mcp-session-override.json`, `.vscode/mcp.json`); FULL slice-advancement with `--plan` 7/7 PASS (gate-MCP transport closed both attempts → direct CLI authoritative); **green-light: true**. Observations: (a) steps 09-12 prose **User instruction** drift vs YAML packets; (b) split-step green ids use `-G1`/`-G2` vs derivation note 1 `<spec-step>-green`; (c) gate-MCP transport instability recurred.
- **Batch 7 — confirmation pass (second independent instance):** re-verification against the post-batch-6 file; identical direct-CLI slice-advancement invocation → `pass: true`, 7/7 sub-gates PASS, `fixHint: "All 7 gates passed for slice P1-step-01-author (FULL)."`; phase-packets rerun 0 errors / 33 warnings; structural reconciliation PASS; **green-light: true — CONFIRMED (double-verified)**; `<!-- EVIDENCE-APPEND-POINT -->` marker restored; concurrency warning recorded — dispatch at most one verification instance per plan at a time.

## Step 02 — Shell canvas + WebGL2 boot + render loop (spec P1-S1) [DONE]

Archived verbatim from the tracker at the Step 02 → [DONE] flip (workflow-update-sync advance + orchestrator compression pass; evidence batch 14 items 63-68, item 68 YAML-lag note).

```yaml
phase: 1
step: 2
title: 'Shell canvas + WebGL2 boot + render loop (spec P1-S1)'
status: '[WIP]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 03 - Neon background passes (spec P1-S2)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S1-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
  - 'examples/neatRacing/browser-entry/canvas-dimensions.utils.test.ts'
  - 'examples/neatRacing/engine/context.service.test.ts'
  - 'examples/neatRacing/host/render-loop.utils.test.ts'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit (tsconfig introduced by P1-S1-B)'
  - 'npm run build:neatRacing (script introduced by P1-S1-B) then manual: serve and confirm frame 1 renders'
acceptance_criteria:
  - 'AC-001 (partial): boot stub window.neatRacingStart mounts the game canvas (P1-S1-A); fixed render height 720 + max backing width 1920 canvas strategy with letterboxed CSS aspect ratio and no page scroll (P1-S1-B)'
  - 'AC-002: WebGL2 context service initializes and graceful-falls-back when WebGL2 is unavailable (P1-S1-C)'
  - 'AC-003: fixed-timestep accumulator render loop at 1000/60 ms with max-catchup 4 per frame (P1-S1-D)'
  - 'AC-037 (partial): npm run build:neatRacing script registered (P1-S1-B; full esbuild bundle pipeline lands in Phase 3 Step 02 / P3-S2-A)'
  - 'AC-040: shader warmup sweep compiles+links all current programs with one off-screen draw per pass before frame 1, with a named program-count constant (P1-S1-C; count grows as passes land through Step 07)'
  - 'AC-042 (partial): render loop allocation-count fixture over preallocated matrices/vectors (P1-S1-D; packed-frames transfer half lands in Phase 2 Step 01 / P2-S1-B)'
slices:
  - slice_id: 'P1-S1-A'
    title: 'Entry shell + boot stub: index.html + browser-entry.ts window.neatRacingStart'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/index.html'
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - 'AC-001 (partial): boot stub window.neatRacingStart mounts the game canvas into the page shell with no page-level scroll'
      - 'AC-037 (partial): page loads the host entry from src/ per the flagship demo contract (bundle pipeline lands in Phase 3 Step 02)'
    red_test:
      - 'RED: browser-entry.test.ts fails before the boot stub exists; asserts window.neatRacingStart is a function and the canvas mounts (spec section 17 RED -> IMPLEMENT -> GREEN)'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S1-B'
    title: 'Fixed-canvas strategy + tsconfig.neatRacing.json + build:neatRacing script'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/browser-entry/canvas-dimensions.utils.ts'
      - 'examples/neatRacing/browser-entry/constants.ts'
      - 'examples/neatRacing/browser-entry/canvas-dimensions.utils.test.ts'
      - 'tsconfig.neatRacing.json (trivial registration)'
      - 'package.json build:neatRacing script (trivial registration)'
    acceptance_criteria:
      - 'AC-001: canvas strategy fixes render height to NEATRACING_FIXED_RENDER_HEIGHT=720 and caps backing store at NEATRACING_MAX_BACKING_WIDTH=1920 with letterboxed CSS aspect ratio and no page scroll'
      - 'AC-037 (partial): package.json gains build:neatRacing; tsconfig.neatRacing.json includes examples/neatRacing and type checks clean'
    red_test:
      - 'RED: canvas-dimensions.utils.test.ts fails before the strategy exists; asserts 720 fixed height, 1920 backing cap, and letterbox math at odd viewport sizes'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S1-C'
    title: 'WebGL2 context service + shader warmup sweep'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/context.service.ts'
      - 'examples/neatRacing/engine/context.service.test.ts'
    acceptance_criteria:
      - 'AC-002: WebGL2 context service initializes once and graceful-falls-back with an on-page notice when WebGL2 is unavailable'
      - 'AC-040: warmup sweep compiles+links every registered program and performs one off-screen draw per pass before frame 1; warmed-program count is a named constant asserted by test (grows as passes land through Step 07)'
    red_test:
      - 'RED: context.service.test.ts fails before the service; asserts single init, fallback path, and warmup sweep count constant with a mock GL'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S1-D'
    title: 'Fixed-timestep accumulator render loop, zero per-frame host allocations'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/host/render-loop.utils.ts'
      - 'examples/neatRacing/host/render-loop.utils.test.ts'
    acceptance_criteria:
      - 'AC-003: fixed-timestep accumulator at 1000/60 ms with NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME=4 and no spiral-of-death'
      - 'AC-042 (partial): allocation-count fixture runs N frames with zero per-frame heap allocations via preallocated matrices/vectors'
    red_test:
      - 'RED: render-loop.utils.test.ts fails before the loop; asserts catchup cap at 4 and zero per-frame allocations with a counting allocator test double'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S1-green'
    title: 'Step green validation: shell + canvas + boot + render loop'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 02 AC subset green: AC-001, AC-002, AC-003 plus AC-037/AC-040/AC-042 partials; browser-entry + engine jest patterns green; tsconfig.neatRacing.json type check clean; build:neatRacing script present'
    parallelizable: false
    dependencies:
      - 'P1-S1-A'
      - 'P1-S1-B'
      - 'P1-S1-C'
      - 'P1-S1-D'
```

**User instruction:** Implement spec slice P1-S1: browser shell canvas, WebGL2 context service with graceful fallback, and the fixed-timestep render loop; red -> implement -> green per slice.

**Step objective:** Deliver the bootable shell: index.html + browser-entry boot stub mounting the game canvas, canvas-dimension utils, the WebGL2 context service, and the preallocated fixed-timestep render loop at 1000/60 ms with max-catchup 4.

**Stop conditions:**
- **Done:** browser-entry + engine + host jest suites green; tsconfig.neatRacing.json typecheck clean; AC-001..AC-003 plus AC-037/AC-040/AC-042 partials recorded.
- **Blocked:** WebGL2 unavailable in the environment — record the blocker in Latest validation evidence and continue with CPU-fallback fixtures.

**Required validation:**
- `examples/neatRacing/browser-entry/browser-entry.test.ts`
- `examples/neatRacing/browser-entry/canvas-dimensions.utils.test.ts`
- `examples/neatRacing/engine/context.service.test.ts`
- `examples/neatRacing/host/render-loop.utils.test.ts`
- `npx tsc -p tsconfig.neatRacing.json --noEmit (tsconfig introduced by P1-S1-B)`
- `npm run build:neatRacing (script introduced by P1-S1-B) then manual: serve and confirm frame 1 renders`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"`

### Step 02 evidence summary (batches 9-14, compressed)

Full itemized records live in the tracker's `## Latest validation evidence` (items 36-68); this is the compressed archive.

- **Batch 9 — P1-S1-A (boot shell + boot stub):** RED TS2307 (missing module) -> GREEN 8/8; `window.neatRacingStart` stop-handle-mounted `#neatracing-canvas`, `declare global` Window augmentation, index.html flagship-contract shell (overflow hidden, `#status` aria-live, repo-path bundle loader). Slice-advancement FULL 7/7 (`slice-id=P1-S1-A`); A flipped [DONE].
- **Batch 10 — P1-S1-B (canvas strategy + registrations):** RED -> GREEN 16/16; `round(720 x refW/refH)` clamped to 1920 backing cap, 3-rung ladder (client -> viewport -> 1280x720), letterbox min-scale fit; `tsconfig.neatRacing.json` + `package.json build:neatRacing` registered, tsc exit 0, bundle emitted to `docs/assets/neatracing.bundle.js`. Slice-advancement FULL 7/7 (`slice-id=P1-S1-B`); B flipped [DONE].
- **Batch 11 — P1-S1-C (WebGL2 context service + warmup):** RED -> GREEN 9/9; single WebGL2 init with WeakMap memoization, graceful fallback notice (no throw), G5 context-loss handling (preventDefault + lost-flag predicate + restore hook), 9-program warmup registry (`NEATRACING_WARMED_PROGRAM_COUNT` derived from registry keys; starfield, grid, pulse, track, car, glass, bloom-threshold, bloom-blur, bloom-additive). Slice-advancement FULL 7/7 (`slice-id=P1-S1-C`); C flipped [DONE]. Session note: dense Cortex embeddings degraded (expected 25293, found 0); BM25-only retrieval sufficed; housekeeping suggestions recorded (item 43).
- **Batch 12 — P1-S1-D (render loop) + orchestrator reconciliation:** RED -> GREEN 16/16; counting-allocator zero-alloc contracts, ulp-robust exact-60 Hz accumulator (1e-9 step-unit boundary epsilon), catchup cap 4, no spiral-of-death. D's own slice-advancement FULL 7/7 (`slice-id=P1-S1-D`); D deferred the tracker flip — orchestrator flipped P1-S1-D [DONE]. Union step-level shared-validation gate: `slice-advancement` with `slice-id=P1-S1-green` + the 12-file union `--plan` forwarded -> FULL 7/7 PASS.
- **Batch 13 — pre-green specialist review (implementation-pattern-scout, kimi-k2.7-code:cloud):** verdict **APPROVE** — all four slices match spec AC-001/002/003/037/040/042 intent; 49 contract tests pin the right behaviors; implementation-standards compliance verified (full JSDoc, named constants, orchestration-first helpers, ES2023, zero `any` in production, extensionless imports, single-expect tests). Four non-blocking observations: (1) boot-glue wiring owner unassigned (initRenderingContext / runShaderWarmup / advanceFrameSteps / letterbox apply / fallback append) — assign before the Phase 1 exit check; (2) green-test scoping: manual check proves bundle load + canvas mount + console clean; full frame-1 render is a later-wiring checkpoint; (3) `image-rendering: auto` unpinned — add with letterbox CSS wiring; (4) spec-internal host/ path inconsistency (documentation-only, plan is source of truth).
- **Batch 14 — P1-S1-green (step green validation, 05-green-testing):** GREEN OK. Pre-specialist-smoke gate PASS (4 suites/49 tests); focused Jest 49/49; `tsc -p tsconfig.neatRacing.json --noEmit` exit 0; `npm run build:neatRacing` exit 0 (bundle 339 B); DEMO/UI real visible-window browser smoke PASS via delegated browser-harness-specialist (bundle HTTP 200, `neatRacingStart` defined, `#neatracing-canvas` mounted, zero console errors, visible-foreground Chrome) + direct real-shell check of `index.html` (status "Neat Racing booted.", canvas in output, zero console messages). `slice-advancement` FULL 7/7 (`slice-id=P1-S1-green`); `workflow-update-sync` hook advanced Step 02 -> [DONE] and Step 03 -> [WIP]; smoke fixture created at `docs/browser-tests/scenarios/neatracing-shell-smoke.html`; environment torn down. Open item: boot-glue wiring owner unassigned (observation 1).

## Step 03 — Neon background passes (spec P1-S2) [DONE]

```yaml
phase: 1
step: 3
title: 'Neon background passes (spec P1-S2)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 04 - Voxel data pipeline part 1 (spec P1-S3 A-C)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S2-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/engine/passes'
  - 'manual: npm run dev:neatRacing (from Step 08) later shows starfield + grid + pulse over the sky; until then validate via jest fixtures'
acceptance_criteria:
  - 'AC-004: starfield pass renders 3 deterministic layers (seeds 1337 / 2777 / 4242; densities 0.03 / 0.04 / 0.05; sizes 35 / 28 / 23) with cyan rgba(95,255,255,1) and scroll rates matching constants.starfield.ts'
  - 'AC-005: grid pass draws fwidth-based constant-screen-width lines with depth fading and pulse sync'
  - 'AC-006: pulse pass draws the #B7FF00 pulse ring at alpha 0.42 with expansion/decay behavior'
slices:
  - slice_id: 'P1-S2-A'
    title: 'Starfield pass + constants.starfield.ts'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/engine/passes/starfield.pass.ts'
      - 'examples/neatRacing/browser-entry/constants.starfield.ts'
      - 'examples/neatRacing/engine/passes/starfield.pass.test.ts'
    acceptance_criteria:
      - 'AC-004: 3-layer deterministic starfield per spec section 13 AC-004 constants; visual-only NEATRACING_VISUAL_SEED exempt from byte parity'
    red_test:
      - 'RED: starfield.pass.test.ts fails before the pass; asserts layer constants (seed/density/size) and deterministic point layout per seed'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S2-B'
    title: 'Grid pass with fwidth-based constant line widths'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/passes/grid.pass.ts'
      - 'examples/neatRacing/engine/passes/grid.pass.test.ts'
    acceptance_criteria:
      - 'AC-005: grid lines keep constant screen-space width via fwidth with depth fade and no z-fighting against the track'
    red_test:
      - 'RED: grid.pass.test.ts fails before the pass; asserts fwidth-derived widths and depth-fade function over fixture viewports'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S2-C'
    title: 'Pulse pass (#B7FF00 ring, alpha 0.42)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/engine/passes/pulse.pass.ts'
      - 'examples/neatRacing/engine/passes/pulse.pass.test.ts'
    acceptance_criteria:
      - 'AC-006: pulse ring color #B7FF00 at alpha 0.42; ring expands and decays deterministically with distance/time'
    red_test:
      - 'RED: pulse.pass.test.ts fails before the pass; asserts ring expansion/decay curve and alpha 0.42'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S2-green'
    title: 'Step green validation: neon background passes'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 03 AC subset green: AC-004, AC-005, AC-006; engine jest pattern green'
    parallelizable: false
    dependencies:
      - 'P1-S2-A'
      - 'P1-S2-B'
      - 'P1-S2-C'
```

## Step 04 — Voxel data pipeline part 1: schema + mesher + bin codec (spec P1-S3 A-C) [DONE]

Archived from the tracker at close-out (batch 34). Full itemized records live
in the tracker's `## Latest validation evidence` (items 105-134); this is the
compressed archive.

```yaml
phase: 1
step: 4
title: 'Voxel data pipeline part 1: schema + mesher + bin codec (spec P1-S3 A-C)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 05 - Voxel data pipeline part 2: LOD + distance swap (spec P1-S3 D-E)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S3-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/voxel/model'
  - 'npm run voxel:build (script introduced by P1-S3-C)'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-007 + AC-008: model.schema validates single-index rule (color+emissive in one byte grid; black exception), lod128/lod32/lod8 levels, and decode recovers typed arrays (P1-S3-A)'
  - 'AC-009: greedy mesher emits correct quads with per-voxel color/emissive lookup and no interior faces (P1-S3-B)'
  - 'AC-010: bin codec packs/unpacks pre-baked meshes losslessly; scripts/build-voxel-bins.mjs + npm run voxel:build generate committed bins (P1-S3-C)'
slices:
  - slice_id: 'P1-S3-A'
    title: 'Voxel model schema + decode utils'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/voxel/model.schema.ts'
      - 'examples/neatRacing/voxel/model.decode.utils.ts'
      - 'examples/neatRacing/voxel/model.decode.utils.test.ts'
    acceptance_criteria:
      - 'AC-007: single-index rule with the black-is-solid exception; lod128/lod32/lod8 level schema validated'
      - 'AC-008: decode emits typed arrays sized exactly from the schema header; malformed inputs rejected with named errors'
    red_test:
      - 'RED: model.decode.utils.test.ts fails before schema/decode exist; asserts single-index constraint, level sizing, and malformed-input rejection'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S3-B'
    title: 'Greedy mesher (interior-face culling, per-voxel color/emissive)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/voxel/greedy-mesher.ts'
      - 'examples/neatRacing/voxel/greedy-mesher.test.ts'
    acceptance_criteria:
      - 'AC-009: greedy mesher produces quad-reduced meshes with correct per-voxel color/emissive lookup and zero interior faces on fixture models'
    red_test:
      - 'RED: greedy-mesher.test.ts fails before the mesher; asserts quad counts and face culling on a canonical fixture model'
    parallelizable: true
    dependencies:
      - 'P1-S3-A'
  - slice_id: 'P1-S3-C'
    title: 'Bin codec + build-voxel-bins script + voxel:build script'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/voxel/bin.codec.ts'
      - 'scripts/build-voxel-bins.mjs'
      - 'examples/neatRacing/voxel/bin.codec.test.ts'
      - 'package.json voxel:build script (trivial registration)'
    acceptance_criteria:
      - 'AC-010: bin codec round-trips pre-baked meshes byte-exactly; build-voxel-bins.mjs emits deterministic bins under assets/models/bins/'
    red_test:
      - 'RED: bin.codec.test.ts fails before the codec; asserts round-trip byte parity and deterministic header layout'
    parallelizable: true
    dependencies:
      - 'P1-S3-A'
  - slice_id: 'P1-S3-G1'
    title: 'Part 1 green validation: schema + mesher + bin codec'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 04 AC subset green: AC-007, AC-008, AC-009, AC-010; voxel jest pattern green; voxel:build runs clean'
    parallelizable: false
    dependencies:
      - 'P1-S3-A'
      - 'P1-S3-B'
      - 'P1-S3-C'
```

**User instruction:** Implement spec slice P1-S3 A-C: voxel schema, greedy
mesher, and bin codec; red -> implement -> green per slice.

**Step objective:** Deliver the voxel data pipeline foundation: typed voxel
schema, greedy mesher emitting chunk geometry, and the .bin codec with a
deterministic header.

**Stop conditions:**
- **Done:** voxel jest suite green; npm run voxel:build produces a readable
  .bin; AC-007..AC-010 recorded.
- **Blocked:** codec format dispute vs spec — stop, record, escalate via
  00.cross-tier-helper.

**Required validation:**
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/voxel"`
- `npm run voxel:build`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`

### Batch summaries (batches 28-34, items 105-134)

- **Batch 28 — orchestrator:** Step 04 flipped [PLANNED] -> [WIP] (heading +
  YAML + slices A/B/C -> [WIP]; G1 stayed [PLANNED]); hygiene chain green
  (107 chunks, plan-session-redirect pass). P1-S3-A dispatched
  (04-implementing, background, kimi-k2.7-code:cloud, 36429b64,
  complexity=complex). Items 105-106.
- **Items 107-108 — P1-S3-A (36429b64) via implementation-executor:**
  `model.schema.ts` + `model.decode.utils.ts` + 33-test file. Fix packet:
  `validateMetaDims` no longer hard-pins the 128³ cube — positive-integer
  declared dims (`VOXEL_DIMS_AXIS_MIN=1`) per spec §7.2 "dims != 128³ *or the
  declared dims*" + §7.4/TP-14 (wheels are 32³ same-format models that must
  pass schema validation); new SRP `validateSymmetricWidth` rejects odd
  widths on symmetric docs (`VOXEL_SYMMETRIC_WIDTH_MODULUS=2`, LD-28);
  `VOXEL_GRID_DIMS` kept as normative car-grid constant (JSDoc: NOT a
  validation pin). 33/33 jest, tsc 0, eslint 0; api-contract-reviewer
  APPROVE first cycle; slice-advancement 7/7.
- **Batch 32 — orchestrator reconciliation:** A consolidated [DONE] (item
  109) with carried gaps (voxel:build unvalidatable until C;
  quality:folder missing-test-file debt for model.schema.ts — owner-local
  test; tracker AC wording lod128/lod32/lod8 vs spec §7.3.3
  lod128+lod64-only drift flagged for P1-S3-D reconciliation). B dispatched
  (97059988, item 110), C dispatched (c5a8abb4, item 111) in parallel.
- **Batch 29 — P1-S3-C design resolutions (items 112-114):** Node 26 native
  type-stripping does not resolve extensionless TS specifiers -> module
  resolve hook retrying extensionless relative specifiers with .ts (no edits
  to landed A files); seam binding = `meshVoxelModel(model): VoxelMesh
  {dims, quads}`; path drift resolved per spec wins: script at
  `examples/neatRacing/scripts/`, bins at `examples/neatRacing/assets/bins/`.
- **Batch 30 — P1-S3-B (items 115-116):** `greedy-mesher.ts` + 11-test file,
  full RED -> GREEN (RED TS2307, then 11/11 first run). Merge key (palette
  index, normal axis, orientation) via signed Int16 mask; interior-face
  culling vs naive exposed-face oracle; glass (alpha 0)/EMPTY non-occluding
  air; LD-23 black exception in resolveVoxelShade; deterministic emission
  order. tsc 0, eslint 0, gate 7/7. ESCALATION: quad normalAxis UNSIGNED
  0..2 in mesher output AND codec validation (VOXEL_BIN_NORMAL_AXIS_MIN/MAX)
  — spec §7.3.1 parenthetical leaves standalone bins unable to reconstruct
  signed per-fragment normal (opposite-orientation faces emit
  field-identical records).
- **Batch 31 — P1-S3-C (item 117):** `bin.codec.ts` (NXVB little-endian:
  16B header magic/version/dims/paletteCount, 5B palette entries, 12B quads
  u16x3 pos @0 + u8 axis @6 + u16 w @7 + u16 h @9 + u8 paletteIndex @11;
  byte-identical round-trip; named-error family with stable codes) +
  24-test file + `examples/neatRacing/scripts/build-voxel-bins.mjs` (resolve
  hook; zero-model exit-0 short-circuit) + package.json `voxel:build`.
  24/24 jest, tsc 0, eslint 0, voxel:build exit 0; api-contract-reviewer
  APPROVE first cycle; gate 7/7. Co-signed the normalAxis escalation.
- **Batch 32 — reconciliation (items 118-120):** B and C consolidated
  [DONE] on agent evidence. normalAxis escalation ADJUDICATED: decision
  (spec amendment to signed/6-value axis vs renderer-side convention)
  DEFERRED to a 01-planning patch that MUST run before P1-S4/P1-S5 renderer
  slices are dispatched; P1-S3-green NOT blocked (both slices pin unsigned
  0..2 consistently per the locked spec). Standing dispatch constraint.
- **Items 121-122 — union gate + specialist dispatch:** consolidated
  slice-advancement `slice-id=P1-S3-green` over the 9-file impl union
  (model trio + mesher pair + codec pair + build-voxel-bins.mjs +
  package.json), `--plan` forwarded -> FULL 7/7 PASS. Union specialist
  dispatched (implementation-pattern-scout c882cbf4).
- **Item 123 — union specialist verdict: APPROVE.** All 9 union files read
  end-to-end: mesher<->codec seam exact (meshVoxelModel/VoxelMesh preferred
  export; VoxelQuad ~= VoxelBinQuad); NXVB layout cross-checked vs named
  constants + byte fixtures; ES2023/JSDoc/naming/owner-local tests
  conformant; No-Deferred-Cleanup not violated (resolveMesherExport =
  defensive dynamic-import seam per resolution 113d; lod64/wheel = capability
  sequencing). No fix packet.
- **Batch 33 — P1-S3-green (05-green-testing 8816840e, items 124-133):**
  **GREEN: OK.** tsc exit 0; `jest:base --testPathPatterns=
  "examples/neatRacing/voxel"` 3 suites / 68 tests exit 0; `npm run
  voxel:build` exit 0 with designed zero-model short-circuit (resolution
  113c; NXVB emission deferred to P1-S6-D); convergence-tracker pass (zero
  fix-loop markers); slice-advancement post-append 7/7 (10-file union incl.
  plan). Relay exact-match limitation recurred (jest:base + voxel:build
  rejected as prose; direct execution per Batch 24 precedent — recorded as
  workflow-improvement candidate). GPU/browser/pre-specialist gates N/A
  (no such files touched).
- **Batch 34 — orchestrator close-out:** P1-S3-G1 + Step 04 flipped [DONE];
  Step 04 archived verbatim to logs.md and compressed to a tracker Done
  record; Current state / Immediate next steps / Handoff query refreshed to
  the Step 05 boundary; hygiene chain green.

### Carried flags

- Signed-face-orientation contract gap (normalAxis u8 0..2, no winding
  field in normative §7.3.2): 01-planning patch MUST run before P1-S4/P1-S5
  renderer slices consume bins (standing dispatch constraint).
- Tracker AC wording (AC-007/AC-011/AC-022 lod128/lod32/lod8) vs locked spec
  §7.3.3 (lod128 + lod64 only; 32³ tier default NOT built): reconcile at
  P1-S3-D (Step 05) dispatch.
- Pre-existing quality:folder missing-test-file heuristic FAIL for
  model.schema.ts (schema covered via owner-local decode tests).
- Mesher seam runtime path unexercised until P1-S6-D model assets land
  (voxel:build zero-model short-circuit is the G1 validation surface).
- Node 26 DEP0205 module.register() deprecation warning (registerHooks()
  migration follow-up; warning-only).
- Validation-MCP relay cannot execute pattern-shorthand allowlist entries
  (2nd occurrence); direct execution per Batch 24 precedent.

**User instruction:** Implement spec slice P1-S2: starfield, grid, and pulse neon background passes; red -> implement -> green per slice.

**Step objective:** Deliver the three neon background passes as composable WebGL2 draw passes layered over the shell canvas.

**Stop conditions:**
- **Done:** engine background-pass jest suites green; AC-004..AC-006 recorded.
- **Blocked:** GPU-dependent fixtures fail in CI — record and continue with deterministic fixture renders.

**Required validation:**
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"`
- manual: npm run dev:neatRacing (from Step 08) later shows starfield + grid + pulse over the sky; until then validate via jest fixtures

**Plan-hygiene notes (carried by the compression):**
- The AC-005 tracker paraphrase added "pulse sync"; spec AC-005 contains no such requirement — the implementer correctly dropped it (batch 17 item 73) and the paraphrase is retired with this archive.
- Spec section 6.4.3 writes `#B7FF00 (185, 255, 0)` but 0xB7 = 183; code/tests pin the spec's decimal 185 (specialist flagged to the spec owner — batch 22, code value unchanged per fix packet).

### Step 03 evidence summary (batches 15-24, compressed)

Full itemized records live in the tracker's `## Latest validation evidence` (items 69-98); this is the compressed archive.

- **Batch 15 — orchestrator:** Step 02 compressed to tracker Done record + archived here; Step 03 flipped [WIP] with slices A/B/C [WIP]; three parallel implementers dispatched (kimi-k2.7-code:cloud): A starfield `2ee3bf3a`, B grid `bf828813`, C pulse `57b75668`.
- **Batch 16 — 01-planning patch (boot-glue owner, spec Step-02 observation 1):** new Step 13 "Boot-glue wiring" [PLANNED] inserted after Step 12 (slices P1-boot-glue implementing 2h + P1-boot-glue-green green-testing 1h); Step 12's manual keyboard-lap exit check re-sequenced behind Step 13; Phase 1 packet next_phase 02-13; coverage backlog AC-002 row updated; plans/README.md boot-glue trigger phrases registered; consolidated gate 4/4 PASS (TRIVIAL).
- **Batch 17 — P1-S2-B grid (bf828813):** `grid.pass.ts` + test, RED -> GREEN 22/22; fwidth-derived constant widths (2px core / 4px halo at 0.8x), depth fade 0.58 -> 0.29 -> 0.05 over 320 m, y=-0.01 depth-neutral plane (no z-fighting), zero `gl.lineWidth`; slice-advancement FULL 7/7; B flipped [DONE]. Recorded the AC-005 "pulse sync" paraphrase discrepancy (spec has no such requirement; dropped).
- **Batch 18 — P1-S2-C pulse (57b75668):** `pulse.pass.ts` + test, RED -> GREEN 29/29; #B7FF00 ring alpha 0.42, deterministic expansion/decay envelope, GL_LESS depth predicate, premultiplied-additive draw, dispose; gate 7/7 pre+post flip; project-wide tsc fully clean; C flipped [DONE].
- **Batch 19 — 01-planning verification (fresh instance):** `green-light: true` recorded; consolidated gate 4/4 PASS. Three non-blocking observations: (1) coverage backlog Phase 2/3 rows off-by-one + "13" double-bound — renumber before Phase 2 kickoff (Phase 2 = steps 14-18, Phase 3 = 19-20); (2) concurrent-append race on the evidence section is real — hygiene after every batch, one verification instance at a time; (3) pre-existing mangled quotes near item 29 (optional cleanup).
- **Batch 20 — P1-S2-A starfield (2ee3bf3a):** `constants.starfield.ts` + `starfield.pass.ts` + 20-test file; exact flappy parity xorshift32 (13/17/5), 3 layers (seeds 1337/2777/4242), G6 scroll math verbatim, GLSL ES 3.00 point sprites, warmup seam keyed NEATRACING_WARMED_PROGRAM_KEYS[0]; 4 suites / 80 engine tests green, tsc/eslint clean; per-slice Tier-3 review via review-coordinator -> webgpu-parity-reviewer **APPROVE**; slice-advancement FULL 7/7; A flipped [DONE].
- **Batch 21 — orchestrator reconciliation + union shared-validation:** all three slices [DONE] reconciled (interleaved item numbers from concurrent batches 17-20 pinned by the union gate); consolidated `slice-advancement` with `slice-id=P1-S2-green` over the 7-file impl union, `--plan` forwarded -> FULL 7/7 PASS (`planCommandLintPlan` correct, zero failed/errored).
- **Batch 22 — union specialist review (implementation-pattern-scout `2873bfe5`):** **REQUEST_CHANGES** — BLOCKING 1: pulse color uniform fed raw 0-255 [185,255,0] without /255 (clamps to yellow-white on real GPUs; invisible to mock harness); BLOCKING 2: grid ships no AC-040 warmup seam (no key, no factory, no warmup test; registry pins 'grid' as warmed key #2). 7 non-blocking observations (pulse GL_BLEND enable, P9 allocations, drawOffscreen asymmetry, starfield guard off-by-one, spec 6.4.3 hex/decimal defect 0xB7=183 vs 185, single-expect overruns, dots-vs-ring wording). Fix packet `fix-packet-P1-S2-green-iteration-1` created (`fix-loop: P1-S2-green iteration 1 status=failed`).
- **Batch 23 — fix packet executed (04-implementing `ab0c8529` + implementation-executor):** all 3 requested changes implemented across 4 files (22 hunks): pulse color normalized to [185/255, 1, 0] with decimal-185 spec pin kept + JSDoc flag of the 0xB7=183 defect; grid AC-040 warmup seam added (ShaderWarmupProgram import, `NEATRACING_GRID_PROGRAM_KEY='grid'` = registry index 1, zero-pose identity-projection draw through createGridPass) and ONE canonical standalone `to<X>WarmupProgram(gl)` factory unified across starfield/grid/pulse (old `pass.createWarmupProgram` seam removed same step — No-Deferred-Cleanup); `gl.enable(GL_BLEND)` precedes blendFunc(ONE, ONE). Preflight: pulse+grid 57/57, tsc exit 0, eslint 0. Fresh Tier-3 re-review via review-coordinator -> api-contract-reviewer **APPROVE** — fix loop CLOSED at iteration 1. Consolidated gate re-run: FULL 7/7. Plan updated (fix-packet follow-up + items 87-91); hygiene chain green. Carry-forward: write a fresh shared-validation artifact on the changed-file union before final advancement.
- **Batch 24 — P1-S2-green (05-green-testing `551ff173`):** **GREEN: OK.** Union jest `examples/neatRacing/engine/passes` with coverage: 3 suites / 77 tests, exit 0 (fix-packet pins re-verified: normalized uColor [185/255, 255/255, 0]; grid warmup key parity [1]; enable(BLEND) before blendFunc); `tsc -p tsconfig.neatRacing.json --noEmit` exit 0; consolidated slice-advancement FULL 7/7 post-append (fresh shared-validation on the changed-file union); independent slice-validator contract check PASS; `fix-loop: P1-S2-green iteration 2 status=passed` marker recorded; plan-family reindex stale:false. Tooling notes: MCP validation relay cannot execute this packet's pattern-shorthand allow-list entries (spawn ENOENT) — direct focused jest per running-unit-tests skill; jest --coverage table empty because jest.config.mjs does not scope coverage to examples/ (canonical coverage authority = code-coverage sub-gate, passed). Browser/manual sky-composite check deferred to Step 08 / P1-boot-glue-green by the packet itself; GPU real-device gate N/A (no src/architecture/network/gpu files). Evidence batches 15-24 items 69-98.

## Step 05 — Voxel data pipeline part 2: LOD + distance swap (spec P1-S3 D-E) [DONE]

Archived from the tracker at close-out (batch 41). Full itemized records live
in the tracker's `## Latest validation evidence` (items 138-158); this is the
compressed archive.

```yaml
phase: 1
step: 5
title: 'Voxel data pipeline part 2: LOD + distance swap (spec P1-S3 D-E)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 06 - Car materials + wheels part 1 (spec P1-S4 A-C)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S3-D'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/voxel/model'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-011: LOD builder derives lod32/lod8 from lod128 with G2 voxel-grid rules and P4-F6 face-count caps (>256 threshold), black exception preserved, mixed color/glow averages exclude black cells (P1-S3-D)'
  - 'AC-012: distance util switches LOD tiers with NEATRACING_VOXEL_WORLD_SIZE approx 0.035156 m scale and hysteresis swap (no thrash) (P1-S3-E)'
slices:
  - slice_id: 'P1-S3-D'
    title: 'LOD builder (lod32/lod8 derivation, black exception)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/voxel/lod.builder.ts'
      - 'examples/neatRacing/voxel/lod.builder.test.ts'
    acceptance_criteria:
      - 'AC-011: lod32/lod8 honor G2/P4-F6 rules including the black exception and mixed color/emissive average exclusions'
    red_test:
      - 'RED: lod.builder.test.ts fails before the builder; asserts tier shapes, black exception, and average exclusions on a fixture model'
    parallelizable: true
    dependencies:
      - 'P1-S3-G1'
  - slice_id: 'P1-S3-E'
    title: 'LOD distance util with hysteresis swap'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/voxel/lod.distance.utils.ts'
      - 'examples/neatRacing/voxel/lod.distance.utils.test.ts'
    acceptance_criteria:
      - 'AC-012: tier selection from camera distance at NEATRACING_VOXEL_WORLD_SIZE scale with hysteresis band preventing swap thrash'
    red_test:
      - 'RED: lod.distance.utils.test.ts fails before the util; asserts boundary distances and hysteresis behavior'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S3-G2'
    title: 'Part 2 green validation: LOD + distance swap'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 05 AC subset green: AC-011, AC-012; voxel jest pattern green'
    parallelizable: false
    dependencies:
      - 'P1-S3-D'
      - 'P1-S3-E'
```

**User instruction:** Implement spec slice P1-S3 D-E: LOD chunk builder and distance-based chunk swap; red -> implement -> green per slice.

**Step objective:** Deliver LOD chunk building and the distance-based swap so far chunks render lighter without popping artifacts.

**Stop conditions:**
- **Done:** voxel LOD jest suites green; typecheck clean; AC-011, AC-012 recorded.
- **Blocked:** LOD swap hysteresis unstable — record and escalate via 00.cross-tier-helper.

**Required validation:**
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/voxel"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`

### Batch summaries (batches 35-41, items 138-158)

- **Batch 35 — orchestrator dispatch hygiene (item 140):** D dispatched
  (04-implementing c3c3ef45, complexity=complex) and E (58ce447b) in
  parallel per declared dependencies. Aborted duplicate-retry episode
  recorded honestly: post-compaction read_agent false-negative (short
  ID prefix) triggered retries; user cancelled them; lesson recorded —
  verify with list_agents (FULL UUIDs) before re-dispatching.
- **Batch 36 — P1-S3-E (item 141):** lod.distance.utils.ts + 15-test
  suite (AC-012, spec §7.3.4: world size 4.5/128=0.03515625 m, focal
  623.54 px @720p/60°vFOV, swap 43.84 m @0.5 px, ±10% hysteresis;
  selectLodTier 'lod128'|'lod64', allocation-free, NaN-safe). 15/15
  jest, tsc 0; performance-reviewer APPROVE; slice-advancement 7/7.
  Cortex search_context payload routing defect hit twice by the
  executor → learning-event capture (one open agent-system-gap record
  in .github/ai-learning/learning-log.jsonl; fix routed 00-helping).
- **Batch 37 — orchestrator reconciliation (item 142):** E
  [WIP]→[DONE]; union gate deferred to D completion; carried notes
  (local NEATRACING_LOD_SWAP_FOV_DEG=60; learning-event dispatch).
- **Batch 38 — P1-S3-D (items 143-146):** lod.builder.ts +
  lod.builder.test.ts via implementation-executor (green-only TDD):
  all 9 §7.3.3 rules (contributing = index≠0; all-empty→EMPTY; black
  by color bytes excluded only from color+glow averages, alpha over
  ALL contributors; single-index verbatim; half-up
  floor((2·sum+count)/(2·count)); 5-bit quantize r/g/b before dedup;
  derived (0,0,0)→glow 0; alpha snap <128→glass(0) else 255; dedup
  lowest-index-wins, cap 256, nearest-squared-RGB merge, EMPTY never a
  merge target). Reconciliation locked: spec §7.3.3/LD-19 wins —
  tracker lod32/lod8 shorthand retired; only derived tier is 64³ from
  2×2×2 merges of 128³; 32³ tier default NOT built. 26/26 voxel lod
  jest, eslint 0, tsc 0; api-contract-reviewer APPROVE;
  slice-advancement 7/7.
- **Batch 39 — orchestrator reconciliation (item 147):** D
  [WIP]→[DONE]; item-141 integrity re-verified byte-exact after the
  agent-reported transient truncation+restore; union gate unblocked.
- **Batch 40 — P1-S3-G2 union validation (05-green-testing a515645a,
  items 148-157):** union gate slice-advancement slice-id=P1-S3-green-2
  (4-file impl union + --plan) FULL 7/7 PASS; union specialist
  implementation-pattern-scout (377d56e0) APPROVE — every §7.3.3 rule
  and the §7.3.4 formula chain verified with all arithmetic
  independently recomputed (worldSize 0.03515625, focal 623.538, swap
  43.8425, band 39.458/48.227, half-up + 5-bit quantization
  spot-checks); 5 non-blocking observations. GREEN: OK — tsc exit 0;
  `jest:base --testPathPatterns="examples/neatRacing/voxel"` 5 suites /
  94 tests exit 0; convergence-tracker pass; slice-advancement 7/7
  pre+post append; fix-loop: P1-S3-G2 iteration 1 status=passed.
  Allowlist relay pattern-shorthand limitation recurred (direct
  execution per Batch 24 precedent). GPU/browser gates N/A
  (examples/ + plan surface only).
- **Batch 41 — orchestrator close-out:** P1-S3-G2 + Step 05 flipped
  [DONE]; Step 05 archived verbatim here and compressed to a tracker
  Done record; Current state / Immediate next steps / Handoff query
  refreshed to the Step 06 boundary; hygiene chain green.

### Carried flags

- Signed-face-orientation contract gap (normalAxis u8 0..2, no winding
  field in normative §7.3.2): 01-planning patch MUST run before
  P1-S4/P1-S5 renderer slices consume bins (standing dispatch
  constraint; item 120).
- NEATRACING_LOD_SWAP_FOV_DEG (60) defined locally in
  lod.distance.utils.ts — shared-FOV constant reconciliation deferred
  to a later slice (item 142 note).
- No in-repo consumer imports buildLodModel yet — car-lod64.bin
  emission expected with P1-S6-D model assets (reviewer non-blocking
  note).
- Derived black-glass tuple (0,0,0,0,0) dedups into the EMPTY sentinel
  (coarse cell → air) — spec-silent edge, visually equivalent,
  deterministic (specialist observation).
- findNearestPaletteIndex tie-break/merge-cache branch implemented
  deterministically but not directly pinned by a dedicated test —
  coverage-tranche candidate.
- Open learning event: Cortex search_context payload routing defect
  (agent-system-gap record in .github/ai-learning/learning-log.jsonl;
  fix owned by 00-helping; flip to resolved on fix).
- Validation-MCP relay cannot execute pattern-shorthand allowlist
  entries (recurring); direct execution per Batch 24 precedent.

## Step 06 — Car materials + wheels part 1 (spec P1-S4 A-C) [DONE]

Archived from the tracker at close-out (batch 50). Full itemized records live
in the tracker's `## Latest validation evidence` (items 159-190); this is the
compressed archive.

```yaml
phase: 1
step: 6
title: 'Car materials + wheels part 1 (spec P1-S4 A-C)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 07 - Car materials + wheels part 2 (spec P1-S4 D-E)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S4-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/voxel/model'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-013 (partial): materials semantics module with black-never-emits-shine rule consumed by mesher/LOD/glass (P1-S4-A; glass-pass half in Step 07 / P1-S4-E)'
  - 'AC-014: team remap maps palette slots to cyan/red/green/blue team colors deterministically (P1-S4-B)'
  - 'AC-015: wheel mounts attach at 4 sockets with NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S=2 clamp (P1-S4-C)'
slices:
  - slice_id: 'P1-S4-A'
    title: 'Materials semantics (black never emits shine)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/voxel/materials.ts'
      - 'examples/neatRacing/voxel/materials.test.ts'
    acceptance_criteria:
      - 'AC-013 (partial): material table + black-exception emission rule; black cells contribute no glow through any consumer'
    red_test:
      - 'RED: materials.test.ts fails before the module; asserts black cells yield zero emissive in every material lookup'
    parallelizable: true
    dependencies:
      - 'P1-S3-G2'
  - slice_id: 'P1-S4-B'
    title: 'Team color remap'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/voxel/team-remap.ts'
      - 'examples/neatRacing/voxel/team-remap.test.ts'
    acceptance_criteria:
      - 'AC-014: deterministic palette-to-team remap for cyan/red/green/blue with no cross-team bleed on fixture models'
    red_test:
      - 'RED: team-remap.test.ts fails before the module; asserts per-team output colors and untouched non-palette voxels'
    parallelizable: true
    dependencies:
      - 'P1-S4-A'
  - slice_id: 'P1-S4-C'
    title: 'Wheel mount utils with visual rev clamp'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/voxel/wheel.mount.utils.ts'
      - 'examples/neatRacing/voxel/wheel.mount.utils.test.ts'
    acceptance_criteria:
      - 'AC-015: wheel sockets attach at spec-defined offsets; visual rotation clamped at NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S=2'
    red_test:
      - 'RED: wheel.mount.utils.test.ts fails before the module; asserts socket placement and the rev clamp'
    parallelizable: true
    dependencies:
      - 'P1-S4-A'
  - slice_id: 'P1-S4-G1'
    title: 'Part 1 green validation: materials + remap + wheel mount'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 06 AC subset green: AC-013 partial, AC-014, AC-015; voxel jest pattern green'
    parallelizable: false
    dependencies:
      - 'P1-S4-A'
      - 'P1-S4-B'
      - 'P1-S4-C'
```

**User instruction:** Implement spec slice P1-S4 A-C: car materials, team color remap, and wheel mount; red -> implement -> green per slice.

**Step objective:** Deliver the car material pipeline: palette-driven materials, team color remap, and wheel mount assembly on the voxel model.

**Signed-face-orientation contract (item 120, resolved batch 42):** bins carry `normalAxis` as u8 0..2 with NO winding/orientation field (normative §7.3.2 layout unchanged; AC-010 byte-exact round-trip and the landed mesher/codec are NOT modified). Renderer convention (spec §7.3.1 amendment): bins-derived meshes draw two-sided (`CULL_FACE` disabled); the effective signed normal is reconstructed per fragment as the axis normal flipped to face the viewer, and sun/hemisphere lighting consumes the flipped normal — no per-vertex lighting, preserving the §7.3.1 merge invariant. Binds all bins-consuming P1-S4/P1-S5 renderer slices (car pass, editor preview); the glass pass is exempt (model-derived geometry; §6.5 culling-ON stands).

**Stop conditions:**
- **Done:** voxel jest suites green; typecheck clean; AC-013 partial + AC-014, AC-015 recorded.
- **Blocked:** team palette mapping mismatch vs spec — record and escalate via 00.cross-tier-helper.

**Required validation:**
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/voxel"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`

### Batch summaries (batches 42-50, items 159-190)

- **Batch 42 (items 159-164) — item-120 adjudication:** tracker-side
  01-planning patch resolved the item-120 signed-face-orientation defect with a
  normative decision: `normalAxis` stays UNSIGNED u8 0..2 with NO winding field
  (§7.3.2 layout, AC-010 byte-exact round-trip, and the landed mesher/codec
  untouched); bins-derived meshes draw two-sided (CULL_FACE disabled), the
  effective signed normal is reconstructed per fragment via the view-flip
  convention, sun+hemisphere consume the flipped normal, no per-vertex lighting
  (§7.3.1 merge invariant preserved); glass pass EXEMPT (model-derived
  geometry; §6.5 culling-ON). Spec amendment text-only (§7.3.1/§7.3.2/§6.6, no
  AC churn). Tracker patches (161-162) verified: plan-session-redirect PASS;
  slice-advancement PASS (P1-S4-A TRIVIAL 4/4); hygiene reindex/build/embed/
  validate PASS (cortex-index FAIL on the workflow_mcp_alive env probe only,
  item 164 — returned to orchestrator).
- **Batch 43 (items 165-167) — verification green-light + maintenance
  remediation:** fresh 01-planning verification green-light: true; spec
  spot-checks confirm the §7.3.1/§7.3.2/§6.6 amendment claims;
  slice-advancement 4/4 plan-structure sub-gates PASS (TRIVIAL, plan-only).
  00-helping remediation (item 167): root cause of the item-164 cortex-index
  failure — the gate self-check hard-requires a [WIP] step, and between-steps
  windows (Step 05 [DONE], Step 06 [PLANNED]) are structural, NOT stale
  processes; fix advanced Step 06 [PLANNED] → [WIP] (heading + YAML) as the
  sanctioned active frontier, refreshed stale placeholder_steps, ended 4
  stale persistent MCP processes by PID, deleted a dead binding file;
  cortex-index gate PASS (all 10 family groups fresh) + fresh-subprocess
  self-check ok; learning event appended.
- **Batch 44 (items 168-170) — P1-S4-A green (green-only TDD):**
  examples/neatRacing/voxel/materials.ts — VoxelMaterialKind
  'empty' | 'black' | 'glass' | 'solid'; VoxelMaterial {kind, color, alpha,
  emissive}; resolveVoxelMaterial verbatim passthrough;
  buildVoxelMaterialTable O(palette) one-pass; LD-23 honored:
  VOXEL_MATERIAL_BLACK_EMISSIVE=0 at every glow magnitude; classification
  EMPTY → BLACK (wins over glass; zero emissive regardless of glow) → GLASS
  (alpha===0, glow-only; VOXEL_MATERIAL_GLASS_ALPHA_MAX=0) → SOLID;
  mesher/LOD/glass consumers untouched (no wiring this slice). 17/17 jest,
  tsc 0, eslint 0; slice-advancement 7/7 FULL (specialist-review sub-gate
  "evidence confirmed" — fresh re-review deferred to the union gate per
  173(i)).
- **Batch 45 (items 171-173) — consolidation + carried flags:** orchestrator
  consolidation and user-ordered session stop; workflow/RAG maintenance from
  item 167 completed and the plan family reindexed. Carried flags: (i) fresh
  P1-S4-A specialist re-review REQUIRED before P1-S4-G1 — satisfied batch 48;
  (ii) implementer's closed workflow-MCP transport — degraded-Cortex native
  fallback authorized with gate MCP healthy; (iii) P1-S4-B/C deps satisfied by
  A [DONE], P1-S4-G1 by A/B/C; (iv) §7.3.1 two-sided view-flip convention is
  a standing monitor-only constraint.
- **Batch 46 (items 174-176) — P1-S4-C green (green-only TDD):**
  examples/neatRacing/voxel/wheel.mount.utils.ts —
  NEATRACING_WHEEL_GRID_SIZE=32; NEATRACING_WHEEL_VOXEL_WORLD_SIZE=0.03515625 m
  (single scale owner: lod.distance.utils.ts); world 1.125 m / disc radius
  0.5625 m across 32 voxels; hub height = disc radius (G9: wheel bottoms at
  y=0 contact patch); pivot at grid center (voxel 16 of 32);
  NEATRACING_WHEEL_MOUNT_YAW_RAD=+π/2 (model depth → car lateral axis);
  NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S=2 (|ω| ≤ 4π rad/s, sign-preserving
  clamp, non-finite → 0); socket order [front-left, front-right, rear-left,
  rear-right]; resolveVoxelWheelMetrics with TP-14 format guards
  (VoxelWheelFormatError, code 'voxel-wheel/format'); ω = v/r; tuning points
  wheelbase 3.2 m / track 2.0 m documented, spec-silent. 36/36 jest (after a
  TS2345 fixture-tuple annotation fix), tsc 0, eslint 0; slice-advancement
  7/7 FULL (post-append re-run PASS; SQLITE_BUSY reindex contention resolved
  via the PID-kill precedent).
- **Batch 47 (items 177-178) — P1-S4-B green + tsc consolidation :**
  examples/neatRacing/voxel/team-remap.ts — NEATRACING_TEAM_REMAP_SLOT=1;
  NEATRACING_TEAM_COLORS cyan/red/green/blue ids 0-3; remapVoxelPaletteToTeam
  pure (no input/state mutation), alpha + glow preserved, EMPTY slot skipped,
  no cross-team bleed; red-first TS2307 confirmed. 10/10 jest, union scan
  7 suites / 121 tests, eslint 0, own slice-advancement PASS; union tsc 0 on
  the final on-disk state (item 178); no plan edits by the implementer
  (contract held).
- **Batch 48 (items 179-180) — 3/3 Tier-3 specialist reviews APPROVE:** via
  review-coordinator (kimi-k2.7-code:cloud): (i) P1-S4-A fresh re-review
  (api-contract-reviewer) — honors 173(i); LD-23 confirmed; additive surface;
  latent duplicate-EMPTY schema gap RECORDED for schema-side backlog (inert
  under AC-013 as implemented); (ii) P1-S4-B (determinism-reviewer) — AC-014
  satisfied; INFO: exported team-color constants readonly-typed but
  runtime-mutable (module never mutates; non-blocking); (iii) P1-S4-C
  (api-contract-reviewer) — §7.4/G9 conformance confirmed. P1-S4-G1
  [PLANNED] → [WIP]; union command reconciliation: shorthand + YAML list the
  same jest dir (allowlist compliant).
- **Batch 49 (items 181-189) — P1-S4-G1 union green:** fix-loop:
  P1-S4-G1 iteration 1 status=passed (convergence-tracker standalone PASS,
  zero markers at check time); env boundary: no GPU/browser dependency
  (node/jest only — GPU gate N/A); tsc 0 (~1.9 s); union Jest
  `--testPathPatterns="examples/neatRacing/voxel"` = 8 suites / 157 tests /
  0 failed, exit 0 (vs batch-40 baseline 5 suites / 94 tests: +3 suites,
  +63 tests); AC-013 partial + AC-014 + AC-015 evidence items recorded
  (185-187); slice-advancement 7/7 FULL pre- and post-append; GREEN: OK;
  validation-MCP relay limitation recurred — direct shell execution
  sanctioned per batches 24/33/40 precedent (standing 00-helping
  improve-candidate).
- **Batch 50 (item 190) — Step 06 close-out:** P1-S4-G1 [WIP] → [DONE] on
  GREEN: OK; Step 06 heading [WIP] → [DONE]; Current state / Immediate next
  steps / Handoff query refreshed; next boundary Step 07 (spec P1-S4 D-E)
  [PLANNED] with deps satisfied; batch tail 42-50 flagged
  compression-eligible → this section is the compression deliverable.
  Pre-existing workflow notes carried: duplicate-EMPTY schema backlog (inert);
  validation-MCP relay limitation.

### Carried flags

- Item-120 signed-face-orientation contract (RESOLVED batch 42) is a standing
  monitor-only constraint: every bins-consuming renderer slice (P1-S4 through
  P1-S5, plus P1-S7 track slabs via §7.3.1) draws two-sided with the
  per-fragment view-flip normal (no winding field, no signed axis byte without
  a new spec amendment; §7.3.2 layout normative). Glass pass exempt (§6.5
  culling-ON).
- Glass half of AC-013 (glass pass reading the material table) is deferred to
  Step 07 / P1-S4-E — intentional partial boundary honored (batch 49,
  item 185).
- Latent duplicate-EMPTY schema gap (item 179): model.schema.ts does not
  reject byte-identical EMPTY palette duplicates at non-zero indices while
  materials.ts classifies byte-based 'empty' and the mesher/LOD treat
  duplicates as non-contributing — inert under AC-013 as implemented;
  schema-side tightening owned outside P1-S4-A.
- Exported team-color constants are readonly-typed but runtime-mutable
  (determinism INFO note, item 179 ii); the module never mutates them —
  optional hardening, non-blocking.
- Standing 00-helping improve-candidate: validation-MCP cannot relay the
  union jest command / pattern-shorthand allowlist entries (recurring; direct
  execution per batches 24/33/40/49 precedent).
- Wheel tuning constants wheelbase 3.2 m / track 2.0 m intentionally pinned
  as documented, spec-silent points (item 174).
- Open Cortex search_context payload routing learning event from Step 05
  remains owned by 00-helping in .github/ai-learning/learning-log.jsonl —
  unrelated to Step 06 scope but carried for visibility.

## Step 07 — Car materials + wheels part 2 (spec P1-S4 D-E) [DONE]

Archived from the tracker at close-out (batches 52-54). Full itemized records
live in the tracker's `## Latest validation evidence` (items 192-204); this is
the compressed archive.

```yaml
phase: 1
step: 7
title: 'Car materials + wheels part 2 (spec P1-S4 D-E)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 08 - Dev server + editor core (spec P1-S5)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S4-D'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/engine/passes'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'manual: car renders with team colors, glass highlights, and bloom under dev server (from Step 08)'
acceptance_criteria:
  - 'AC-016 (partial): directional + emissive car lighting with wheel illumination rules (P1-S4-D)'
  - 'AC-013: glass pass honors the black-never-emits-shine rule with G4 state policy (P1-S4-E)'
  - 'AC-016: glass highlights + bloom (threshold/blur/composite) complete the car material pipeline (P1-S4-E)'
slices:
  - slice_id: 'P1-S4-D'
    title: 'Car lighting + car pass'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/lighting.ts'
      - 'examples/neatRacing/engine/passes/car.pass.ts'
      - 'examples/neatRacing/engine/lighting.test.ts'
    acceptance_criteria:
      - 'AC-016 (partial): directional light + emissive wheel illumination integrated into the car pass with deterministic light setup'
    red_test:
      - 'RED: lighting.test.ts fails before the modules; asserts light vectors and emissive wheel behavior with a mock GL'
    parallelizable: true
    dependencies:
      - 'P1-S4-G1'
  - slice_id: 'P1-S4-E'
    title: 'Glass pass + bloom pipeline'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/passes/glass.pass.ts'
      - 'examples/neatRacing/engine/passes/bloom.pass.ts'
      - 'examples/neatRacing/engine/bloom.pass.test.ts'
    acceptance_criteria:
      - 'AC-013: glass pass enforces the black-never-emits-shine rule with the G4 blend-state policy'
      - 'AC-016: bloom threshold/blur/composite completes the neon car look per spec section 13 AC-016'
    red_test:
      - 'RED: bloom.pass.test.ts fails before the passes; asserts glass blend states and bloom threshold behavior with a mock GL'
    parallelizable: true
    dependencies:
      - 'P1-S4-D'
  - slice_id: 'P1-S4-G2'
    title: 'Part 2 green validation: lighting + glass + bloom'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 07 AC subset green: AC-013, AC-016; engine jest pattern green; warmup program-count constant updated to include glass + bloom programs (AC-040 constant parity)'
    parallelizable: false
    dependencies:
      - 'P1-S4-D'
      - 'P1-S4-E'
```

**Latest validation evidence (P1-S4-G2):**
- `npx tsc -p tsconfig.neatRacing.json --noEmit` — exit 0 (via neataptic-validation-mcp + direct shell)
- `npm run jest:base -- --no-cache --testPathPatterns="examples/neatRacing/engine"` — 6 suites / 191 tests pass (exit 0)
- `npm run lint` — 0 errors, 5 pre-existing warnings
- Consolidated `slice-advancement` gate for P1-S4-G2 — pass true, 7/7 sub-gates pass, severity FULL
- AC-040 constant parity: `NEATRACING_WARMED_PROGRAM_COUNT` == 9 (registry includes glass + bloom-threshold/blur/additive), asserted in `context.service.test.ts`
- Manual dev-server visual check deferred to Step 08

### Batch summaries (batches 52-54, items 192-204)

- **Batch 52 (items 192-196) — P1-S4-D car lighting + car pass:**
  delivered green RED-first (`examples/neatRacing/engine/lighting.ts`,
  `passes/car.pass.ts`, `lighting.test.ts`). lighting.ts: directional
  sun (1,2,1 y-dominant) + hemisphere fill + exp2 fog mirror
  (`fogFactor exp2(-(density*d)^2)` TS<->GLSL), colors /255 (sun
  255,246,229 / hemi sky 140,180,255 / ground 60,55,50 / fog
  168,195,225), `FOG_DENSITY_1_PER_M = 0.006` (~50% haze at ~166.7 m),
  hemisphere mix `clamp(n.y*0.5+0.5)`, sunDiffuse `max(0,dot(n,sunDir))`,
  `shadeLitSurface = albedo*(hemi+sun)+albedo*emissive` (LD-23
  black -> 0). car.pass.ts: two programs under ONE warmup key 'car'
  (registry index 4); expandVoxelBinModelToCarMesh from mesher
  min-corner + tangent convention (9 floats/vertex); item-120 two-sided
  draws + per-fragment view-flip normal; blob shadows at world y+0.01
  (depthMask(false), SRC_ALPHA blend, radial falloff, full body XZ
  footprint x 1.15); four wheels resolved at creation, per-frame
  carModel x T(socket) x RY(mountYaw) x RZ(spin); far->near insertion
  sort into a grow-only scratch buffer (zero per-frame allocation);
  draw-time team remap (uTeamColor/uTeamSlot fragment swap, not baked
  into VBO); VP/cameraWorld validation throws; dispose removes
  2 programs / 4 shaders / 3 buffers. Specialist loop: no
  WebGL2/GLSL Tier-3 reviewer at round 1 -> provisional
  webgpu-parity-reviewer -> round-2 REQUEST_CHANGES (two real defects:
  pre-loop bodyBuffer binding wrong for wheel slots >= 1; blob-shadow
  scale used half the footprint) -> fix pass RED-first -> 5 suites /
  130 tests green -> round-3 fresh instance APPROVE.
  `slice-advancement` 7/7 FULL; tsc + eslint 0; learning-event gate
  pass (2 events). Item 196: an OUT-OF-SLICE real-GL defect was routed
  to the orchestrator (grid/pulse fragment shaders hardcode
  FRAGMENT_SHADER = 0x8b20; real value 0x8b30 — mock-invisible).
- **Batch 53 (items 197-200) — routed micro-fix + reviewer creation:**
  00-helping routed the item-196 GL-enum defect to a Tier-2
  implementation executor: grid.pass.ts + pulse.pass.ts now read
  `gl.VERTEX_SHADER` / `gl.FRAGMENT_SHADER` / `gl.COMPILE_STATUS` /
  `gl.LINK_STATUS` (and the remaining stage/status sites) from the GL
  context (starfield pattern); correct-valued buffer constants kept;
  zero behavior change in mock-GL tests (5 suites / 130 tests
  pre == post by an independent 00-helping re-run). Created
  `webgl2-glsl-render-reviewer` (Tier 3, model kimi-k2.7-code:cloud)
  and wired it into the review-coordinator allow-list (9 -> 10);
  repaired the coordinator's long-standing delegate-skill-coverage
  gap (missing execute skill); validators PASS (agent-frontmatter
  normal + strict, agent-graph, delegate-skill-coverage, routing-table
  regenerated 39 agents / 67 skills + freshness PASS). Note: the
  validation-MCP relay rejects the step packet's path-as-command
  validation entry, so the direct shell command remains the sanctioned
  executor (batches 24/33/40/49 precedent).
- **Batch 54 (items 201-204) — P1-S4-E glass pass + bloom:**
  delivered green (`passes/glass.pass.ts`, `passes/bloom.pass.ts`,
  `bloom.pass.test.ts` 61-test slice suite). glass.pass.ts: glass mesh
  7 floats/vertex at a 28-byte stride, all six faces per glass cell,
  configurable tint alpha (LD-23-adjacent spec constant), interior
  double-tint spec-accepted, glass-glass never co-draws and
  glass-opaque fragments drop to depth-LESS; GLASS_FACE_CORNERS
  pre-reversed outward; G4 blend-state order exact incl.
  blendFunc(ONE, ONE_MINUS_SRC_ALPHA) with the pass-local no-restore
  policy; premultiplied GLSL; far->near insertion sort; body/wheel
  matrix composition parity with the car pass; warmup keys[5] all-glass
  cube; item-120 exemption holds (culling-ON). bloom.pass.ts: three
  programs keyed [6..8] (bloom-threshold / bloom-blur /
  bloom-additive); half-res scene targets
  (checkFramebufferStatus-validated); Rec709 luma threshold with a
  linear ramp (black luma 0 -> 0 contribution, LD-23); separable blur
  run twice, TS `computeBloomBlurTaps()` mirrored to a GLSL const
  array; single-scene-transfer composite blendFunc(ONE, ONE) — scene
  + bloom*0.35 onto the default FB; AC-035 zero-op off-switch
  (test-proven draw-op counter); dispose 3 programs / 6 shaders /
  1 buffer / 3 FBOs / 3 textures; warmup restores viewport + default
  FBO. Review loop: webgl2-glsl-render-reviewer unspawnable this
  session (harness task-spawner enum staleness; 00-helping ticket open)
  -> substitute webgpu-parity-reviewer; round-1 REQUEST_CHANGES HIGH
  finding (double-scene claim) verified FALSE POSITIVE by
  04-implementing with source evidence — the threshold stage samples
  the host-owned scene FBO texture; the suggested alternatives would
  either black out the world or erase the additive background — fix
  round addressed comments + source-pin hardening and pinned the
  composite GLSL line in the test; round-2 fresh instance APPROVE.
  Validation: union engine scan 6 suites / 191 tests exit 0; tsc 0;
  eslint 0 (3 slice files); consolidated slice-advancement 7/7 FULL;
  learning-event gate pass (2 events: harness spawner-enum staleness
  open; jest-30 `--testPathPattern` -> `--testPathPatterns` drift —
  plan owner to correct remaining occurrences at next plan touch).

### Carried flags

- Step 08 host contract: keep the single-scene-transfer guarantee (world
  only in the host-owned scene FBO; no opaque world draws direct to the
  default FB); host owns frame-level GL state restore (G4/G8 pass-local
  no-restore policy). Manual dev-server visual check owns glass tint /
  bloom intensity tuning + real-GL validation.
- webgl2-glsl-render-reviewer was unspawnable this session (harness
  spawner-enum staleness; 00-helping ticket open) — re-review with the
  real reviewer recommended after fresh-session spawn confirmation,
  before the Step 08 visual lock.
- Mock-GL harness cannot evaluate enum-typed arguments (the starfield
  fixture still mocks FRAGMENT_SHADER 0x8b20; real value 0x8b30 —
  mock-invisible); systemic rule recorded: mock harnesses must validate
  enum-typed arguments.
- Plan-owner hygiene: step-packet validation entries of the path-as-command
  form ('examples/neatRacing/engine/passes') block the validation-MCP
  relay for the union jest command (direct shell sanctioned); jest-30
  flag drift (--testPathPattern -> --testPathPatterns) — remaining
  occurrences to be corrected by the plan owner at next plan touch
  (learning event line 28703).
- Warmup placeholder scene texture lacks texParameteri (real GL samples
  black, not white; harmless for warmup — polish nit).
- car.pass.ts has no sibling test file (pre-existing folder-quality
  flag, carried for a later coverage tranche).
- Manual dev-server visual check of car team colors, glass highlights,
  and bloom intensity (fog color 168,195,225 haze; blob-shadow margin
  1.15) deferred to Step 08 (unblocked after the batch 53 GL enum
  micro-fix).

## Step 08 — Dev server + editor core (spec P1-S5) [DONE]

Archived from the tracker at close-out (batches 56-61). Full itemized records
live in the tracker's `## Latest validation evidence` (items 209-226); this is
the compressed archive.

```yaml
phase: 1
step: 8
title: 'Dev server + editor core (spec P1-S5)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 09 - Editor features part 1 (spec P1-S6 A-C)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S5-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/server"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"'
  - 'manual: npm run dev:neatRacing then load /editor/ and the game page'
acceptance_criteria:
  - 'AC-018: dev server S1-S6 hardening (403 traversal, 403 traversal-2, 405 method, 413 size, 415 type, single-flight rebuilds, model name-shape validation, atomic backup) + model store CRUD (P1-S5-A/B)'
  - 'AC-019 (partial): editor page + camera controls render and orbit (P1-S5-C/D)'
  - 'AC-020 (partial): editor shell wires state container consumed by palette/undo/symmetry slices (P1-S5-C)'
slices:
  - slice_id: 'P1-S5-A'
    title: 'Model store CRUD + validation'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/server/model-store.utils.ts'
      - 'examples/neatRacing/server/model-store.utils.test.ts'
    acceptance_criteria:
      - 'AC-018 (support): model name-shape validation + store read/write/delete with atomic backup per spec S6'
      - 'AC-007 (support): model store round-trips model.schema payloads without mutation'
    red_test:
      - 'RED: model-store.utils.test.ts fails before the store; asserts name-shape rejection and atomic backup behavior'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S5-B'
    title: 'Dev server with S1-S6 hardening + dev:neatRacing script'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - 'examples/neatRacing/server/dev-server.ts'
      - 'examples/neatRacing/server/dev-server.test.ts'
      - 'package.json dev:neatRacing script (trivial registration)'
    acceptance_criteria:
      - 'AC-018: S1-S6 responses (403/403/405/413/415), single-flight rebuilds, and NEATRACING_DEV_SERVER_PORT binding asserted by tests'
    red_test:
      - 'RED: dev-server.test.ts fails before the server; asserts each hardening case (traversal, method, size, type, single-flight) with supertest-style fixtures'
    parallelizable: true
    dependencies:
      - 'P1-S5-A'
  - slice_id: 'P1-S5-C'
    title: 'Editor shell page + entry'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.html'
      - 'examples/neatRacing/editor/editor-entry.ts'
      - 'examples/neatRacing/editor/editor-entry.test.ts'
    acceptance_criteria:
      - 'AC-019 (partial): editor page mounts WebGL2 canvas + tool frame with no page scroll'
      - 'AC-020 (partial): editor state container initialized and exposed to tool slices'
    red_test:
      - 'RED: editor-entry.test.ts fails before the shell; asserts canvas mount + state container creation'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S5-D'
    title: 'Editor camera controls'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/editor/editor.camera.ts'
      - 'examples/neatRacing/editor/editor.camera.test.ts'
    acceptance_criteria:
      - 'AC-019: orbit/pan/zoom camera controls with clamped zoom and inertial smoothing per spec section 13 AC-019'
    red_test:
      - 'RED: editor.camera.test.ts fails before the camera; asserts orbit deltas, zoom clamp, and smoothing'
    parallelizable: true
    dependencies: []
  - slice_id: 'P1-S5-green'
    title: 'Step green validation: dev server + editor core'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 08 AC subset green: AC-018, AC-019, AC-020 partial; server + editor jest patterns green; manual dev server check of / and /editor/'
    parallelizable: false
    dependencies:
      - 'P1-S5-A'
      - 'P1-S5-B'
      - 'P1-S5-C'
      - 'P1-S5-D'
```

**Latest validation evidence (P1-S5-green):**
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/server"` — 2
  suites / 63 tests exit 0 (model-store 30 + dev-server 33 incl. the CLI
  self-start contract test added by fix-packet-P1-S5-B-iteration-2), executed
  through neataptic-validation-mcp
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"` — 2
  suites / 36 tests exit 0 (editor-entry 17 + editor.camera 19), executed
  through neataptic-validation-mcp
- Manual: `npm run dev:neatRacing` — clean boot verified twice on the real
  CLI path (probe PID 41116 + specialist boot PID 6392); listener
  127.0.0.1:8787; GET / → 200 2954B text/html; GET /editor/ → 200 2507B;
  GET /editor/editor.bundle.js → 200 1324B; GET /assets/neatracing.bundle.js
  → 200 339B (spec §8.2 stub); teardown verified (port 8787 closed)
- Visible-window browser smoke (browser-ui-specialist, chrome-devtools-mcp,
  visible-foreground, fresh profile on port 9222): Chrome 154, WebGL 2.0
  (ANGLE D3D11, NVIDIA RTX 4070 SUPER); /editor/ → 200, title "Neat Racing
  Model Editor (NeatapticTS)", #neatracing-editor-canvas (1398x1296) live
  WebGL 2.0, console clean; / → 200, title "Neat Racing (NeatapticTS)",
  #neatracing-canvas live WebGL 2.0, console clean; 0 failed network
  requests on both routes
- Convergence marker: `fix-loop: P1-S5-B iteration 2 status=passed.` — the
  declared-command CLI self-start defect resolved and verified twice
  independently on the real CLI path (not only at the jest contract layer)
- Consolidated `slice-advancement` gate for P1-S5-green — pass true, 4/4
  sub-gates PASS (plan-sync, step-packet, plan-slice-quality,
  plan-command-lint), severity TRIVIAL; convergence-tracker gate
  (--slice-id=P1-S5-B) pass:true pre/post marker
- P1-S5-B packet status was flipped by 04 mid-step; the step-level [DONE]
  consolidated at close (batch 62, 06-documenting)
- Close-out (batch 62, 06-documenting): docs-quality metrics + gate PASS;
  plan-session-redirect pass:true; plan-family + logs reindex PASS; workflow
  MCP self-check PASS (frontier Step 09); validate-plan-sync ok:true 0
  errors; cortex-index gate PASS; slice-advancement consolidated gate for
  P1-S5-green PASS on the compressed tracker

### Batch summaries (batches 57-61, items 214-226)

- **Batch 57 (item 214) — P1-S5-A green validation:** model-store focused
  lane 30 tests exit 0 via validation-MCP.
- **Batch 58 (items 215-216) — P1-S5-B green + P1-S5-D green:** dev-server
  focused lane green (routing/mime/single-flight/hardening); editor.camera
  lane green re-recorded across the batch 58/59 boundary (item 216
  duplicated with batch 59 — left verbatim).
- **Batch 59 (item 216-dup, item 217) — P1-S5-D green + lint-lane fix
  packet:** editor.camera focused lane green; orchestrator fix packet for
  the shared-validation lint lane (plan items 215-216) executed by
  04-implementing.
- **Batch 60 (items 218-219) — P1-S5-C green + manual smoke:**
  editor-entry focused lane green; first visible-window browser smoke pass;
  orchestrator fix packet for the manual-command CLI self-start
  (plan item 219).
- **Batch 61 (items 220-223 + 226; items 224-225 never assigned) — P1-S5-green union validation (final):**
  allow-listed union Jest lanes 2 suites / 63 tests (server) + 2 suites /
  36 tests (editor) exit 0 via validation-MCP; declared-command manual
  check re-verified on real boots; real visible-window browser smoke re-run
  via browser-ui-specialist; convergence marker recorded; post-write
  hygiene + gates all green (slice-advancement 4/4 PASS TRIVIAL).

### Carried flags

- Step 09 editor features (spec P1-S6 A-C) build on Step 08 surfaces:
  palette/undo/symmetry consume the editor state container (AC-020 partial
  from P1-S5-C) and the camera contract (AC-019 from P1-S5-D); the model
  store (CRUD + validation + AC-007 round-trip) is the persistence
  boundary; the dev server (dev:neatRacing, port 8787, single-flight
  rebuilds, atomic 3-slot backup rotation, name-shape validation) remains
  the manual-check vehicle — do not weaken hardening while extending routes.
- /assets/neatracing.bundle.js is still the spec §8.2 stub (339B); real
  asset emission belongs to the later generator/rebuild pipeline step
  (Step 10, spec P1-S6 D-E).
- Item 209 (batch 56 tail) routed the step-packet command-template drift to
  the plan owner; the jest-30 flag drift (--testPathPattern ->
  --testPathPatterns) learning event (line 28703) stays carried for the
  plan owner at next plan touch.
- Standing 00-helping improve-candidate: validation-MCP relay coverage for
  command shapes outside the step-packet allowlist path (union jest lanes
  ran via the allowlisted entries in batch 61; earlier batches used direct
  shell per batches 24/33/40/49 precedent).
- Duplicate item numbers 213 (the P1-S5-B landed and P1-S5-C red->green
  records share the number) and 216 (the P1-S5-D green record appears
  twice across batches 58/59) in the batch 56-59 era are historical — left
  verbatim for cross-reference fidelity; flagged for the 07-logging owner
  (no renumbering). Items 224-225 were never assigned: batch 61 numbering
  jumps 223 -> 226 (the P1-S5-green [DONE] blockquote sits between them);
  a stray bare "214." sub-reference also sits between items 214 and 215.
  Disclosed at close (batch 62). Item 220's count slip ("dev-server 32"
  under a 63-test headline) was corrected to 33 at close to match the 63
  union headline; the intact 62-era occurrences (pre-fix-packet
  step-regression records) stay verbatim.
- webgl2-glsl-render-reviewer re-review recommendation (00-helping ticket
  line 28701, batch 53) stays attached to GLSL-touching slices; the Step 08
  visual lock was exercised in batch 61 (visible-window browser smoke; no
  GLSL changes in Step 08 sources).
- Step 08 host contract still applies to editor render surfaces (batch 54
  carried): single-scene-transfer guarantee (world only in the host-owned
  scene FBO), host-owned frame-level GL state restore, G4/G8 pass-local
  no-restore policy.

## Session logging close-out (Step 08 learning-event capture)

- Workstream: neatracing-voxel-racer — 07-logging session pass; the
  workstream stays [WIP] at the Step 09 dispatch boundary (no terminal
  closure; this log continues accumulating).
- Files changed: `.github/ai-learning/learning-log.jsonl` (line 28701
  status open → resolved in place — fresh-session REAL
  webgl2-glsl-render-reviewer spawn confirmed in the P1-S5-C review
  round 1; line 28704 appended — `learn-20260929-151627-p1s5c-jestmockspy`,
  tooling-gotcha, open: jest-mock 30 spy accumulation under repo
  `clearMocks:false`, per-boot spy-count assertions unreliable until
  scoped clearing or `clearMocks:true`); `plans/neatracing-voxel-racer.plans.md`
  (P1-S5-C record block marker `LEARNING_EVENT_PENDING` → captured form;
  Handoff carried notes updated; evidence batch 63 appended).
- Validations run: `neataptic-gate-mcp run_gate_check learning-event` →
  pass:true (log exists, well-formed, all categories valid).
- Decisions: captured only the plan's pending-marker event (no duplicates
  — validation-MCP relay L28693 family and jest-30 flag drift L28702 keep
  their standing records); L28701 resolved per plan item 208's stated
  confirmation condition.
- Risks / residual gaps: jest-mock gotcha open until a convention is
  chosen (scoped `mockClear()`/`mockReset()` vs `clearMocks:true`);
  spawner-freshness gate remains an open systemic proposal (L28701
  resolution note); L28702 (jest-flag packet hygiene) and L28703
  (gate-MCP spawnSync hardening) stay open.
- Next resume point: orchestrator dispatches Step 09 (spec P1-S6 A-C);
  no logging debt pending for Steps 02-08.
- Capture sub-agent: learning-event-capturer (Tier 4, model
  kimi-k2.7-code:cloud per user mandate; dispatch MCP caller-tier
  validated).

## Step 09 — Editor features part 1 (spec P1-S6 A-C) [DONE]

Archived from the tracker at close-out (batches 64-71, close batch 72).
Full itemized records live in the tracker's `## Latest validation evidence`
(evidence batches 64-71); this is the compressed archive — no prose or
stop-condition blocks (same archive shape as the Step 08 record).

```yaml
phase: 1
step: 9
title: 'Editor features part 1 (spec P1-S6 A-C)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 10 - Editor features part 2 (spec P1-S6 D-E)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S6-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-020: 128-color palette (exact + nearest) + undo (3-step dense snapshots within NEATRACING_EDITOR_UNDO_BUDGET_BYTES) + symmetry modes complete (P1-S6-A/B)'
  - 'AC-021: x/y/z mirror symmetry painting modes (P1-S6-C)'
  - 'AC-036 (partial): undo snapshots are dense Uint8Array within the named byte budget (P1-S6-B; full memory accounting in Phase 3 Step 01 / P3-S1-B)'
slices:
  - slice_id: 'P1-S6-A'
    title: 'Palette + editor state (exact + nearest)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.palette.ts'
      - 'examples/neatRacing/editor/editor.state.ts'
      - 'examples/neatRacing/editor/editor.palette.test.ts'
    acceptance_criteria:
      - 'AC-020 (partial): 128-color palette with exact match and nearest-color fallback painting into the editor state volume'
    red_test:
      - 'RED: editor.palette.test.ts fails before the palette; asserts exact/nearest resolution against the 128-color table'
    parallelizable: true
    dependencies:
      - 'P1-S5-green'
  - slice_id: 'P1-S6-B'
    title: 'Undo stack (3 steps, dense Uint8Array budget)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/editor/editor.undo.ts'
      - 'examples/neatRacing/editor/editor.undo.test.ts'
    acceptance_criteria:
      - 'AC-020: 3-step undo as dense snapshots; total snapshot bytes within NEATRACING_EDITOR_UNDO_BUDGET_BYTES (approx 6.3 MB)'
      - 'AC-036 (partial): dense snapshot accounting asserted (full budget function in Phase 3 Step 01)'
    red_test:
      - 'RED: editor.undo.test.ts fails before undo exists; asserts 3-step window, byte budget, and restore correctness'
    parallelizable: true
    dependencies:
      - 'P1-S6-A'
  - slice_id: 'P1-S6-C'
    title: 'Mirror symmetry painting (x/y/z)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/editor/editor.symmetry.ts'
      - 'examples/neatRacing/editor/editor.symmetry.test.ts'
    acceptance_criteria:
      - 'AC-021: x/y/z mirror modes mirror paints deterministically; combined modes compose'
    red_test:
      - 'RED: editor.symmetry.test.ts fails before the module; asserts mirrored voxel coordinates per axis and combinations'
    parallelizable: true
    dependencies:
      - 'P1-S6-A'
  - slice_id: 'P1-S6-G1'
    title: 'Part 1 green validation: palette + undo + symmetry'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 09 AC subset green: AC-020, AC-021, AC-036 partial; editor jest pattern green'
    parallelizable: false
    dependencies:
      - 'P1-S6-A'
      - 'P1-S6-B'
      - 'P1-S6-C'
```

**Latest validation evidence (P1-S6-G1):**

- Step-green union lane `npm run jest:base --
  --testPathPatterns="examples/neatRacing/editor"`: exit 0, PASS 5 suites /
  72 tests (editor.palette.test.ts, editor.undo.test.ts,
  editor.symmetry.test.ts, editor.camera.test.ts, editor-entry.test.ts) —
  batch 70 (6.9 s) via neataptic-validation-mcp
  run_allowlisted_validation; live re-verify batch 71 (8.3 s, exit 0, no
  drift against the batch 70 evidence).
- Typecheck lane `npx tsc -p tsconfig.neatRacing.json --noEmit`: exit 0,
  zero type errors (explicit neatRacing project type-check; batches
  65/68/69/70/71).
- Per-slice RED→IMPLEMENT→GREEN provenance (batch 71 record): P1-S6-A
  batches 64/65, P1-S6-B batches 66/68, P1-S6-C batches 67/69, step
  P1-S6-G1 batch 70, close verification batch 71.
- Gates: slice P1-S6-A consolidated slice-advancement 7/7 sub-gates PASS,
  severity FULL (batch 65); P1-S6-B 7/7 FULL (batch 68); P1-S6-C 7/7 FULL
  (batch 69); step P1-S6-G1 4/4 TRIVIAL PASS — plan-record-only surface
  (plan-sync, step-packet, plan-slice-quality, plan-command-lint), matching
  the P1-S5-green batch 61 precedent (batch 70).
- Convergence markers: `fix-loop: P1-S6-{A,B,C,G1} iteration 1
  status=passed.` — all four slices first-green-round, zero fix loops.
- Shared validation baseline: shared-validation-P1-S6-BC.json 23/23
  (slices P1-S6-B + P1-S6-C focused selection green, build 0, lint 0;
  batches 68/69) plus the P1-S6-A palette tests inside the 72/72 superset
  run.
- Orchestrator status flips (batch 71, green never flips markers):
  slice P1-S6-G1 [PLANNED]→[DONE]; Step 09 packet status
  [PLANNED]→[DONE]; Step 09 heading [WIP]→[DONE].
- Blocked branch N/A (batch 71): the "picker math disagrees with camera
  controls" condition cannot trigger at Step 09 — grid picker + part
  placement live in Step 10 (spec P1-S6 D-E) per the packet next_step
  pointer.

### Batch summaries (batches 64-71)

- Batch 64 (P1-S6-A RED, 03-red-testing): editor.palette.test.ts authored
  — 13 `it()` blocks across 5 describes declaring the AC-020 (partial)
  API surface: `NEATRACING_EDITOR_PALETTE_MAX_ENTRIES = 128`,
  `NEATRACING_EDITOR_PALETTE_EMPTY_INDEX = 0` (sentinel `[0,0,0,0,0]`),
  `createEditorPalette()` addColor exact-match-dedup / first-free-index /
  full-palette nearest fallback (Manhattan RGB, deterministic
  lowest-index tie-break mirroring the LOD builder merge rule),
  setAlpha/setGlow 0-255 clamping + sentinel immutability, and
  `createEditorState({ palette? })` exposing the dense 128³ `Uint8Array`
  palette-index working volume (2,097,152 bytes ≈ 2.1 MB, R-07 P7 editor
  exception) with `paint(x, y, z, entry)` exact-else-nearest at layout
  `x + 128*y + 16384*z` (x col, y row 0 = down, z front slice) plus the
  model dropdown state (snapshot().model null until selectModel). RED
  confirmed via orchestrator rerun: ts-jest TS2307 "Cannot find module
  './editor.palette'" + "'./editor.state'"; 0 tests executed.
- Batch 65 (P1-S6-A GREEN, 05-green-testing): editor.palette.ts +
  editor.state.ts green — 3 suites / 49 tests exit 0 (6.5 s: palette 13/13
  plus the carried camera and entry suites) + tsc clean; batch-64 red
  contracts map 1:1 onto passing it() blocks; `fix-loop: P1-S6-A iteration
  1 status=passed.`; slice-advancement 7/7 sub-gates PASS (FULL).
- Batch 66 (P1-S6-B RED, 03-red-testing): editor.undo.test.ts authored —
  12 `it()` blocks across 5 describes: `createEditorUndo(volume)`
  bound stack (push dense detached snapshot, undo() newest-first restore
  returning false when empty without mutation, canUndo/size/bytesInUse),
  `NEATRACING_EDITOR_UNDO_BUDGET_BYTES` asserted <= 8 MiB / 8,388,608
  with the 3 x 2,097,152-byte fit (≈ 6.3 MB per AC-020), cap-3
  oldest-eviction window, newest-first restores, dense non-aliasing;
  checkpoint-before-edit fixture reshape (push BEFORE writing each value)
  documented by the orchestrator; redo semantics declared out of P1-S6-B
  scope (later-slice addition). RED confirmed: TS2307 "Cannot find module
  './editor.undo'" (line 18:8); 0 tests executed.
- Batch 67 (P1-S6-C RED, 03-red-testing): editor.symmetry.test.ts
  authored — 11 `it()` blocks across 5 describes:
  `createEditorSymmetry({ mirrorX?, mirrorY?, mirrorZ? })`, mirrorX
  default true / mirrorY + mirrorZ default false (LD-28 "mirror-X ON by
  default"), `map(x, y, z)` under the normative mirror rule
  `coordinate c ↔ coordinate 127−c` (spec §7.2: cols 0-63 mirror to
  cols 64-127; fixture (10, 20, 30) → x (117, 20, 30), y (10, 107, 30),
  z (10, 20, 97)), deterministic emission order pinned by the
  orchestrator reshape (primary voxel first, ascending subset order
  x < y < xy < z < xz < yz < xyz; writer's order-independent
  sortedVoxels helper removed), boundary coords 0 ↔ 127, 63 ↔ 64,
  127 ↔ 0, RangeError guard (same semantics as editor.state.paint). RED
  confirmed: TS2307 "Cannot find module './editor.symmetry'"; 0 tests
  executed.
- Batch 68 (P1-S6-B GREEN, 05-green-testing): editor.undo.ts green —
  the batch record's union claim (5 suites / 72 tests exit 0, 7.4 s) is
  a record error corrected at close-out batch 72: at batch 68 the
  symmetry suite was still in its batch-67 TS2307 RED state
  (`editor.symmetry.ts` landed only at batch 69, whose 8.2 s union run
  is the first attainable and authoritative 5 suites / 72 tests
  full-lane green), so no full-lane union could exit 0 with 72 passing
  tests at batch 68; the targeted editor.undo GREEN result is
  unaffected; `NEATRACING_EDITOR_UNDO_MAX_STEPS = 3` enforced by shift()
  oldest-eviction; push captures a dense detached copy
  (`new Uint8Array(volume)`, never aliases the live volume); undo()
  restores in place (volume reference stable for renderers);
  bytesInUse() sums retained `Uint8Array.byteLength`;
  `NEATRACING_EDITOR_UNDO_BUDGET_BYTES = 3 x 2,097,152 = 6,291,456`
  bytes (≤ 8 MiB asserted; ≈ 6.3 MB per AC-020); AC-020 (undo portion)
  + AC-036 (partial) green; `fix-loop: P1-S6-B iteration 1
  status=passed.`; slice-advancement 7/7 sub-gates PASS (FULL).
- Batch 69 (P1-S6-C GREEN, 05-green-testing): editor.symmetry.ts green —
  union lane 5 suites / 72 tests exit 0 (8.2 s) + tsc clean;
  `NEATRACING_EDITOR_MIRROR_MAX_COORDINATE = 127`, per-axis
  c ↔ 127−c mirrors, per-axis canonicalization above the mirror line
  into the stored half (map(127, y, z) emits the same ordered set as
  map(0, y, z)), 1-8 tuples per axis power set, duplicate-free, pure
  deterministic module (same flags + coordinates → same ordered tuple
  list on every call); AC-021 green (boundary 0 ↔ 127, 63 ↔ 64, guard);
  `fix-loop: P1-S6-C iteration 1 status=passed.`; slice-advancement 7/7
  sub-gates PASS (FULL).
- Batch 70 (P1-S6-G1 GREEN — step green, 05-green-testing): consolidated
  union run 5 suites / 72 tests exit 0 (6.9 s) + tsc clean, both via
  neataptic-validation-mcp; AC-020, AC-021, AC-036 (partial) documented
  green across the palette/state/undo/symmetry surface; consolidated
  slice-advancement 4/4 sub-gates PASS (TRIVIAL md-only surface);
  `fix-loop: P1-S6-G1 iteration 1 status=passed.` (zero fix loops).
- Batch 71 (orchestrator close verification re-run): both allow-listed
  commands re-run against the live tree via validation-MCP — jest exit 0,
  5 suites / 72 tests (8.3 s), tsc exit 0; no drift between batch 70
  evidence and the repo state; orchestrator flips slice P1-S6-G1
  [PLANNED]→[DONE], Step 09 packet [PLANNED]→[DONE], Step 09 heading
  [WIP]→[DONE]; blocked branch N/A; close-out (packet compression +
  docs-quality (user-mandated) + hygiene gates) dispatched to
  06-documenting per the batch-62 precedent.

### Carried flags

- Redo hooks are a later-slice addition out of P1-S6-B scope (batches
  66/68): spec 8.3 names undo/redo for the step, but the P1-S6-B AC list
  and files_to_change scope undo only; any later redo addition must not
  break the batch-66 12-contract undo surface.
- AC-036 remains partial by design: full memory accounting (budget
  function) stays with Phase 3 Step 01 / P3-S1-B per the AC phrasing
  (batches 66/68/70).
- Grid picker + part placement live in Step 10 (spec P1-S6 D-E): the Step
  09 packet blocked branch ("picker math disagrees with camera controls")
  cannot trigger at Step 09 (batch 71).
- Environment: Cortex dense search degraded during the Step 09 red
  phases (25,994 expected / 0 found embeddings; self-heal exhausted x3;
  BM25-only fallback; user recovery requested) — batches 64/66/67; stale
  plan index noted by the batch-66 writer delegate (non-blocking).

## Step 10 — Editor features part 2 (spec P1-S6 D-E) [DONE]

Archived from the tracker at close-out (batches 5/73-78, close batch 79).
Full itemized records live in the tracker's `## Latest validation evidence`
(evidence batches 5 and 73-78); this is the compressed archive — no prose
or stop-condition blocks (same archive shape as the Step 09 record; the
Red-contract design P1-S6-D block is archived below along with the packet
YAML verbatim, statuses as [DONE]; the packet's retired dispatch shorthand
(`assets/models/car-lod128.model.json`, bare
`scripts/generate-car-voxel.mjs`) stays verbatim per archive fidelity —
the delivered spec §10.2 names are
`examples/neatRacing/scripts/generate-car-voxel.mjs` and
`examples/neatRacing/assets/models/car-f1.model.json` + `wheel.model.json`).

```yaml
phase: 1
step: 10
title: 'Editor features part 2 (spec P1-S6 D-E)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 11 - Track + physics + race part 1 (spec P1-S7 A-C)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S6-D'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/voxel"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"'
  - 'npm run voxel:generate:car'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-022: deterministic car generator emits committed lod128/lod32/lod8 model JSONs validated against model.schema (P1-S6-D)'
  - 'AC-041: editor save->rebuild pipeline: 250 ms trailing debounce, latest-wins single-flight, dirty-slice remesh during drag, full rebuild on pointer-up, p95 <= NEATRACING_REBUILD_P95_BUDGET_MS=300 (P1-S6-E)'
slices:
  - slice_id: 'P1-S6-D'
    title: 'Deterministic car generator + committed assets'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'scripts/generate-car-voxel.mjs'
      - 'assets/models/car-lod128.model.json (generated)'
      - 'examples/neatRacing/voxel/generated-assets.test.ts'
      - 'package.json voxel:generate:car script (trivial registration)'
    acceptance_criteria:
      - 'AC-022: generator is deterministic (same inputs -> byte-identical JSON); committed assets validate against model.schema single-index rule'
    red_test:
      - 'RED: generated-assets.test.ts fails before generation; asserts schema validity + determinism (two runs byte-identical)'
    parallelizable: true
    dependencies:
      - 'P1-S3-A'
  - slice_id: 'P1-S6-E'
    title: 'Editor rebuild pipeline (debounce, latest-wins, p95 budget)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.rebuild.pipeline.ts'
      - 'examples/neatRacing/editor/editor.rebuild.pipeline.test.ts'
    acceptance_criteria:
      - 'AC-041: 250 ms trailing debounce, at most one in-flight rebuild (latest-wins), dirty-slice remesh during drag, full rebuild on pointer-up, p95 <= 300 ms on fixture with jest.useFakeTimers'
    red_test:
      - 'RED: editor.rebuild.pipeline.test.ts fails before the pipeline; asserts debounce window, latest-wins coalescing, and p95 budget with fake timers + timing double'
    parallelizable: true
    dependencies:
      - 'P1-S6-G1'
  - slice_id: 'P1-S6-G2'
    title: 'Part 2 green validation: generator + rebuild pipeline'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 10 AC subset green: AC-022, AC-041; voxel + editor jest patterns green; voxel:generate:car clean'
    parallelizable: false
    dependencies:
      - 'P1-S6-D'
      - 'P1-S6-E'
```

**Red-contract design P1-S6-D (recorded at the red phase before
implementation; archived verbatim):**
Owner-local RED test `examples/neatRacing/voxel/generated-assets.test.ts`, two describes.
(1) Committed-asset contracts (RED today: assets do not exist — right reason): load
`examples/neatRacing/assets/models/car-f1.model.json` and `wheel.model.json`
(spec §10.2 names for the generated initial assets; the packet shorthand
`car-lod128.model.json` is the retired lod128/lod32/lod8 shorthand drift, spec
wins per the locked reconciliation precedent), validate both via
`validateVoxelModelDocument` from `./model.schema` (pins the single-palette /
index-0 EMPTY sentinel rule), expect `meta.dims` `[128,128,128]` for the car
(`VOXEL_GRID_DIMS`) and `[32,32,32]` for the wheel (spec TP-14), and expect
non-empty solid rows (at least one run with paletteIdx != 0 — AC-022 non-empty
meshes, cheap RLE-string proxy without the mesher). (2) Determinism contract
(RED today: script does not exist — right reason): spawn
`spawn(process.execPath, ['<racing>/scripts/generate-car-voxel.mjs', '--out',
<tmpdir>])` with `shell: false` (dev-server CLI-spawn precedent), run twice,
expect exit 0 and byte-identical SHA-256 bytes for both emitted files
(`car-f1.model.json` + `wheel.model.json`), temp dirs removed afterEach. CLI
seam pin: `--out <dir>` overrides the output directory (default run with no
args targets the committed `examples/neatRacing/assets/models/` as
`voxel:generate:car`). Determinism test MUST NOT write the committed assets dir
(LD-29 hand-edits-are-authoritative: the generator is never re-run over tuned
models). Deterministic setup: no RNG/seeds; sha256 comparison only.

**Latest validation evidence (P1-S6-G2):**

- Step-green union lanes (batch 78): voxel jest 9 suites / 160 tests (8.16 s);
  editor jest 6 suites / 83 tests (10.86 s) incl. editor.rebuild.pipeline
  11/11; `npm run voxel:generate:car` sha256 byte-identity
  pre==post==batch-76 committed hashes (car-f1.model.json 240,278 bytes
  `492B7782…E8F86`; wheel.model.json 18,243 bytes `6FF0C655…EECA`);
  `npx tsc -p tsconfig.neatRacing.json --noEmit` clean.
- Provenance: P1-S6-D red batch 5 → green batch 76 (generator + committed
  assets written; dead-constant lint fix-forward recorded as `fix-loop:
  P1-S6-D iteration 1 status=passed.`); P1-S6-E red batch 74 → green batch
  75 (pipeline landed with two fix-forwards recorded as `fix-loop: P1-S6-E
  iteration 1 status=passed.` (drain-loop undefined guard) + `fix-loop:
  P1-S6-E iteration 2 status=passed.` (lint `_request` consumed via
  `void _request;`)); orchestrator union shared-validation
  `artifacts/shared-validation-P1-S6-DE.json` PASS (2 suites / 14 tests +
  build 0 + lint 0; supersedes the stale 2026-09-13
  `artifacts/shared-validation.json`) + severity FULL/FULL + two Tier-3
  specialist APPROVEs (implementation-pattern-scout P1-S6-D;
  performance-trace-specialist P1-S6-E) in batch 77; P1-S6-G2 green batch
  78 (four lanes).
- P1-S6-E mechanics (AC-041) verified on the fixture: 250 ms trailing
  debounce; latest-wins single-flight (`maxInFlight() === 1`,
  `supersededCount === 1` arrival counting); dirty-slice remesh during
  drag (clamped+dedup [z−1,z,z+1]: 0→[0,1], 127→[126,127], exact call
  order); pointer-up flush = one immediate full + pending debounce
  consumed; p95 fixture 19×290 ms + 1×350 ms → nearest-rank p95 = 290 ≤
  NEATRACING_REBUILD_P95_BUDGET_MS = 300 (budget constant 300).
  Real-browser p95 stays deferred to Step 13 wiring (batch-77 flag).
- AC verdicts: AC-022 PASS (deterministic generator; committed schema-valid
  assets; single-index rule) + AC-041 PASS (pipeline mechanics + p95
  budget) both recorded (batch 78).
- Gates (batch 78): consolidated `slice-advancement` with
  `--slice-id=P1-S6-G2 --plan plans/neatracing-voxel-racer.plans.md
  --changed-files plans/neatracing-voxel-racer.plans.md` → pass=true;
  TRIVIAL severity, 4/4 sub-gates (plan-sync, step-packet,
  plan-slice-quality, plan-command-lint) 0 failed / 0 errored; owner:
  orchestrator (Agent Zero). Batch-76 nuance: the consolidated gate ran
  5/6 PASS with shared-validation blocked only by the sibling P1-S6-E
  `_request` lint residual; cleared at batch 77 (union shared-validation).
- Convergence/gates: convergence-tracker N/A for P1-S6-G2 (no loop-back —
  all lanes green); code-coverage N/A (no src/ or
  scripts/agent-customization/ surfaces touched; examples/ excluded from
  jest coverage collection per batches 65/75); pre-specialist-smoke N/A
  (fresh Tier-3 specialists already dispatched and APPROVE per batch 77).
- Validation-MCP relay limitation (3rd occurrence, batch 78):
  `get_active_validation_allowlist` returned activeStep=null /
  validationCommands=[] for this pattern-shorthand packet → direct
  execution of the exact packet lanes per Batch 24 precedent.
- Status flips: the Step 10 heading, packet status, and all three slice
  statuses were flipped [DONE] by the orchestrator before the batch-79
  close-out; the batch-78 record's "left [PLANNED]" line reflects the
  pre-flip green-phase state (flips orchestrator-owned).

### Batch summaries (batches 5/73-78)

- Batch 5 (P1-S6-D RED + red-contract design, 03-red-testing): packet-slice
  RED contract recorded before implementation; owner-local
  `examples/neatRacing/voxel/generated-assets.test.ts` authored — (a)
  committed `car-f1.model.json` load + `validateVoxelModelDocument` schema
  validity (meta.dims [128,128,128], single-palette semantics, non-empty
  solid rows), (b) committed `wheel.model.json` same (meta.dims [32,32,32]),
  (c) determinism — two `--out <tmpdir>` spawns exit 0 + byte-identical
  SHA-256 for both emitted files, `shell: false`, temp dirs removed
  afterEach, never writes the committed assets dir (LD-29); focused RED run
  3/3 failing with the right reasons (committed files ENOENT; generator
  spawn exit 1); consolidated slice-advancement 4/4 PASS TRIVIAL. Packet
  shorthand drift retired by reconciliation: spec §10.2 names win
  (`examples/neatRacing/assets/models/car-f1.model.json` + `wheel.model.json`
  under `examples/neatRacing/scripts/generate-car-voxel.mjs`).
- Batch 73 (unplanned editor boot-clear fix, Agent 0):
  pre-Step-10-dispatch boot fix —
  `examples/neatRacing/editor/editor-entry.ts` one-time guarded clear after
  `initRenderingContext` success (PREVIEW_CLEAR_* named constants,
  optional-chained GL calls), fixing the served boot-slice-only bundle's
  never-drawn canvas ("black screen" report); rebuilt bundle re-served in
  place with PIDs preserved and chrome-devtools-mcp verified
  COLOR_CLEAR_VALUE == [0.08,0.09,0.14,1] on the live context; jest editor
  lane 5 suites / 72 tests + tsc clean.
- Batch 74 (P1-S6-E RED, 03-red-testing):
  `examples/neatRacing/editor/editor.rebuild.pipeline.test.ts` authored — 11
  it() blocks across 6 describes (T1-reshape deltas: AABB dirty-window
  contract clamped+dedup [z-1,z,z+1] with 0→[0,1] and 127→[126,127],
  250 ms trailing debounce, latest-wins single-flight with
  supersededCount===1, pointer-up flush, nearest-rank p95 assertion with
  the manual nowMs clock + two-sided tolerance, named constants); RED
  confirmed TS2307 at :33:8, 0 tests; consolidated slice-advancement 4/4
  TRIVIAL pass.
- Batch 75 (P1-S6-E GREEN, 04-implementing): editor.rebuild.pipeline.ts
  landed (FIFO + exclusive drain loop, trailing-reschedule debounce feeding
  the same queue, full-mode flight counting with durations recorded via the
  injected nowMs clock, nearest-rank p95); test alias swapped to the module
  import; preflight 11/11 + tsc clean; two fix-forwards (drain-loop
  undefined guard; lint `_request` consumed `void _request;`);
  consolidated slice-advancement 7/7 PASS FULL after the lint fix.
- Batch 76 (P1-S6-D GREEN, 04-implementing): `generate-car-voxel.mjs`
  created (zero-dependency deterministic generator; 21 pre-run self-fixes
  incl. ESM require removal, argv dead-block removal, wheel disc
  half-integer center, named palette-index constants); committed
  car-f1.model.json 240,278 bytes + wheel.model.json 18,243 bytes
  (schema-compliant palette semantics, shining glass a=0, solid black tire
  per LD-23, RLE half-row sums 64/16, meta names →
  car-f1-lod128.bin/wheel-lod128.bin); `voxel:generate:car` registered;
  focused generated-assets lane 3/3 green + tsc clean + two `--out` spawns
  byte-identical; dead-constant lint fix-forward (`fix-loop: P1-S6-D
  iteration 1 status=passed.`); gate 5/6 PASS with shared-validation
  blocked only by the sibling P1-S6-E `_request` lint residual (cleared at
  batch 77).
- Batch 77 (orchestrator union shared-validation + specialists, Agent 0):
  shared-validation-P1-S6-DE.json PASS (2 suites / 14 tests + build 0 +
  lint 0; the `_request` residual confirmed cleared; supersedes the stale
  2026-09-13 `artifacts/shared-validation.json`); severity FULL/FULL;
  implementation-pattern-scout APPROVE (P1-S6-D) + performance-trace-specialist
  APPROVE (P1-S6-E) with 4 minor non-blocking doc observations + the
  Step-13 real-browser p95 flag (escalate to performance-optimization if
  the real p95 > 300 ms); zero fix loops.
- Batch 78 (P1-S6-G2 GREEN, 05-green-testing): four packet lanes exit 0
  (voxel jest 9 suites / 160 tests, 8.16 s; editor jest 6 suites / 83
  tests, 10.86 s, editor.rebuild.pipeline 11/11; `voxel:generate:car`
  sha256 byte-identity pre==post==batch-76 committed hashes; tsc clean);
  AC-022 + AC-041 verdicts PASS; Validation-MCP relay pattern-shorthand
  limitation (3rd occurrence, direct execution per Batch 24 precedent);
  consolidated slice-advancement 4/4 TRIVIAL pass.

### Carried flags

- Real-browser p95 remains unmeasured until the Step 13 wiring: the
  step-green p95 verdict (290 ≤ 300) is the fixture verdict
  (jest.useFakeTimers + manual nowMs clock); escalate to the
  performance-optimization specialist if the real trace exceeds 300 ms
  (batches 77/78).
- 4 non-blocking doc-vs-code observations on editor.rebuild.pipeline.ts
  docs for the plan owner (batch 77): snapshot() "frozen copy" wording;
  durationsMs "completions only" vs finally-recording; test-header
  "coalesced" breadth; queued dirty windows un-coalesced within a debounce
  window.
- Validation-MCP relay pattern-shorthand limitation (3rd occurrence,
  batch 78): `get_active_validation_allowlist` misses pattern-shorthand
  packet lanes → direct execution of the exact packet lanes per Batch 24
  precedent; learning event NOT yet captured (standing 00-helping
  improve-candidate).
- LD-29 generator-vs-committed-assets seam (batches 5/76/78): the
  generator is authoritative at commit time (batch-76 models are the
  committed truth); a bare `voxel:generate:car` run rewrites the committed
  assets dir; `--out <dir>` never writes committed; determinism test never
  writes committed assets.
- Pre-Step-10 unplanned boot-clear fix (batch 73): editor-entry.ts
  one-time guarded clear; the `gl.CLEAR_COLOR` vs `COLOR_CLEAR_VALUE`
  pname learning-event note is NOT yet captured in the learning log;
  environment-level Node DEP0205 dev-server warning remains.
- Carried-open pool from Step 09 (unchanged, non-blocking): the
  jest-mock learning-log items (L28701/L28702/L28703) remain open for the
  learning owner; environment-level Node DEP0205 dev-server warning
  (batch 73); Cortex dense-search degradation episodes (batches 64/66/67 —
  BM25-only fallback, user recovery requested).

## Step 11 — Track + physics + race part 1 (spec P1-S7 A-C) [DONE]

Archived from the tracker at close-out (batches 80-88, close batch 88;
packet compression at close-out batch 89). Full itemized records live in
the tracker's `## Latest validation evidence` (evidence batches 80-88);
this is the compressed archive — no prose or stop-condition blocks (same
archive shape as the Step 09/10 records) and NO batch-summaries section
(per the close-out batch-88 directive the itemized batch records stay in
the tracker's Latest validation evidence). The packet's extended
red-contract red_test bullets — recorded in-plan by batch 85, incl. the
P1-S7-C outer-apron-edge boundary interpretation pin — are archived
verbatim inside the packet YAML below, statuses as [DONE].

```yaml
phase: 1
step: 11
title: 'Track + physics + race part 1 (spec P1-S7 A-C)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 12 - Track + physics + race part 2 (spec P1-S7 D-E)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S7-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/track"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-017: track slabs with 0.5 m pitch per TP-19 incl. banked turns and boost pads (P1-S7-A)'
  - 'AC-023: deterministic fixed-step physics with wheel slip + drift model (P1-S7-B)'
  - 'AC-024 (partial): keyboard input mapping + collisions/boost/off-track surfaces (P1-S7-C)'
slices:
  - slice_id: 'P1-S7-A'
    title: 'Track slab builder (0.5 m pitch, banks, boost pads)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/track/track.slabs.ts'
      - 'examples/neatRacing/track/track.slabs.test.ts'
    acceptance_criteria:
      - 'AC-017: slab layout at 0.5 m pitch with banked turns and boost-pad surfaces per TP-19 fixture track'
    red_test:
      - 'RED: track.slabs.test.ts fails before the builder; asserts slab pitch, bank angles, and boost-pad placement on the fixture track'
    parallelizable: true
    dependencies:
      - 'P1-S6-G2'
  - slice_id: 'P1-S7-B'
    title: 'Physics step service (deterministic, wheel slip + drift)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/physics/physics.step.service.ts'
      - 'examples/neatRacing/physics/physics.step.service.test.ts'
    acceptance_criteria:
      - 'AC-023: fixed-step deterministic physics with wheel slip and drift model; no Math.random sim-side'
    red_test:
      - 'RED: physics.step.service.test.ts fails before the service; asserts determinism across runs and slip/drift envelopes'
    parallelizable: true
    dependencies:
      - 'P1-S7-A'
  - slice_id: 'P1-S7-C'
    title: 'Keyboard input + collision/boost/off-track utils'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/host/input.ts'
      - 'examples/neatRacing/physics/collision.utils.ts'
      - 'examples/neatRacing/physics/collision.utils.test.ts'
    acceptance_criteria:
      - 'AC-024 (partial): keyboard input map (D5) + collision and off-track detection (D6) wired into physics surfaces'
    red_test:
      - 'RED: collision.utils.test.ts fails before the utils; asserts collision responses and off-track detection on fixture geometry'
      - 'RED CONTRACT (recorded by 03-red-testing): owner-local test file = examples/neatRacing/physics/collision.utils.test.ts only; it imports {NEATRACING_CAR_COLLISION_RADIUS_M, NEATRACING_TRACK_CURB_WIDTH_M, NEATRACING_TRACK_GRASS_APRON_WIDTH_M, createTrackSurfaceResolver, resolveCarCollisions, resolveTrackBoundaryCar} from ./collision.utils, {sampleKeyboardInput, type NeatRacingKeyboardSample} from ../host/input, step service exports from ./physics.step.service, and TrackSegment from ../../racing_curriculum/track/track.generator.types; expected failure reason = Cannot find module for both unimplemented modules under ts-jest (no fixture typos); fixture = single straight TrackSegment (startX -50, startY 0, endX 50, endY 0, width 8) + seed 42 derived states + JSON.stringify byte-parity purity checks; assertions are radius/band-relative (computed from exported constants, no magic geometry thresholds)'
      - 'RED green contract - collision.utils.ts exports: NEATRACING_CAR_COLLISION_RADIUS_M finite > 0 (D6 named fixed constant, never voxel-derived); NEATRACING_TRACK_CURB_WIDTH_M = 1.0 and NEATRACING_TRACK_GRASS_APRON_WIDTH_M = 3.0 mirroring the private track.slabs band constants; createTrackSurfaceResolver(segments) returning a pure (car: PhysicsCarState) => PhysicsSurfaceType whose nearest-segment lateral distance bands mirror the slab rasterizer - road when |lat| <= halfWidth - curb, curb when <= halfWidth, grass when <= halfWidth + apron and beyond (off-track floor zone); resolveCarCollisions(cars: readonly PhysicsCarState[]): PhysicsCarState[] - pairwise circular overlap detection by the D6 radius, equal-mass frictionless ARCADE ELASTIC exchange along the collision normal (kinetic energy and normal momentum conserved, relative normal velocity reversed, no deformation, no rewards per LD-04), centers separated to at least the sum of radii, pure + deterministic, returns a NEW array and never mutates inputs; resolveTrackBoundaryCar(car, segments): PhysicsCarState - bound = the OUTER grass-apron edge (halfWidth + apron) - INTERPRETATION ~88 percent confidence: outer apron edge rather than the road edge, because §5.2 keeps the grass apron reachable as the off-road zone while racing_curriculum instead clamped at the road edge; outside the bound the car is reseated inside and the outward world-velocity component is reflected elastically with magnitude preserved, inside the bound the car returns byte-identical - the untouched-at-lat-5 assertion deliberately pins this interpretation for 04-implementing review'
      - 'RED green contract - host/input.ts (D5 map, ships in 04-implementing, NOT part of the red artifact) exports a pure DOM-free sampleKeyboardInput(keys: NeatRacingKeyboardSample): PhysicsStepInput with NeatRacingKeyboardSample = {up, down, left, right: boolean}: up -> throttle 1; down -> brake 1; left -> steer +1 (counterclockwise = left per the stepPhysics steer contract); right -> steer -1; no keys -> neutral zeros; D5 one-sample-per-rAF-frame replay is demonstrated by reusing ONE returned sample object across ALL catch-up ticks of a 4-tick burst with byte-identical outcome versus fresh per-tick samples; DOM event adapters belong to later host wiring and must not enter this pure part'
      - 'RED test blocks: (1) constants - radius finite > 0, band widths toBe 1.0 and 3.0; (2) car-car head-on closing pair (speeds 10 vs 10, heading 0 vs PI, centers 1 cm apart on x) - elastic exchange with relative normal velocity closeTo -20, momentum approx 0, energy approx 100 (tolerance 1e-6; world velocity of a state computed as (cos(h + slip) * speed - sin(h) * lat, sin(h + slip) * speed + cos(h) * lat)); (3) centers separated to >= sum of radii - 1e-6; (4) far states (2r + 1 apart) returned byte-identical; (5) purity - repeat resolution byte-identical with inputs unchanged; (6) boundary bounce - car at lat 8, heading PI/2, speed 10: |y| <= 7 + 1e-6, outward velocity closeTo -10, speed magnitude closeTo 10; (7) boundary inside - car at lat 5 returned byte-identical; (8) resolver bands - road at lat 0 and 2.5, curb at 3.5, grass at 5 and 20; (9) wiring - 4 idle catch-up ticks via stepPhysics with the resolver: grass start (lat 20) decelerates faster than road start (lat 2) using one reused idle input object; (10) D5 keyboard map - distinct it() per contract: throttle, brake, left steer +1, right steer -1, neutral zeros, plus determinism byte-parity and the one-sample-replayed-per-tick burst'
    parallelizable: true
    dependencies:
      - 'P1-S7-B'
  - slice_id: 'P1-S7-G1'
    title: 'Part 1 green validation: track + physics + input collisions'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 11 AC subset green: AC-017, AC-023, AC-024 partial; track + physics jest patterns green'
    parallelizable: false
    dependencies:
      - 'P1-S7-A'
      - 'P1-S7-B'
      - 'P1-S7-C'
```

**Latest validation evidence (P1-S7-G1):**

- Step-green lanes (batch 87, all three live via the official
  validation-MCP relay with exact allowlist entries — the earlier relay
  limitation did not recur; no direct-shell fallback needed): track jest
  1 suite / 10 tests; physics jest 2 suites / 28 tests
  (physics.step.service.test.ts + collision.utils.test.ts);
  `npx tsc -p tsconfig.neatRacing.json --noEmit` exit 0, 0 errors — the
  neatRacing-project tsc lane captured explicitly (batch-86 item 4b; the
  root build tsc does not cover examples/**). Union: 3 suites / 38 tests,
  zero content failures, no flake evidence across the three runs.
- Provenance: P1-S7-A red batch 81 (10 it() blocks under describe
  'buildTrackSlabs (spec §6.7 LD-25, AC-017)'; RED TS2307 via the
  allowlisted relay) -> green batch 82 (track.slabs.ts landed; green on
  the first implementation pass 10/10; slice-advancement 7/7 FULL;
  api-contract-reviewer APPROVE, zero REQUEST_CHANGES) with `fix-loop:
  P1-S7-A iteration 1 status=passed.`; P1-S7-B red batch 83 (9 it()
  blocks; RED TS2307 confirmed twice — direct and relayed) -> verify
  batch 84 (orchestrator shared-validation artifact
  coverage/shared-validation-p1s7b.json PASS: jest 9/9, build 0, lint 0;
  severity classification FULL with the corrected `--input=` CLI form;
  implementation-pattern-scout APPROVE, zero blockers) with `fix-loop:
  P1-S7-B iteration 1 status=passed.`; P1-S7-C red batch 85 (19 it()
  blocks / 6 describes; sibling physics suite 9/9 unaffected) -> verify
  batch 86 (host/input.ts + collision.utils.ts landed with one
  contract-fidelity fix round — rebound guarded by `v . outward > 0`;
  coverage/shared-validation-p1s7c.json PASS: jest 19/19, build 0,
  lint 0; severity FULL; implementation-pattern-scout APPROVE with five
  non-blocking observations) with `fix-loop: P1-S7-C iteration 1
  status=passed.`; P1-S7-G1 green batch 87 with `fix-loop: P1-S7-G1
  iteration 1 status=passed.` (zero fix loops across the step); step
  CLOSED batch 88 (P1-S7-G1 + Step 11 heading + packet status +
  placeholder_steps entry all flipped [DONE] by the orchestrator).
- AC verdicts: AC-017 green (track slabs — 0.5 m pitch TP-19 with the
  bank/boost-pad packet wording adjudicated out of spec scope at batches
  80-81, spec wins); AC-023 green (fixed-step determinism + slip/drift
  envelopes); AC-024 PARTIAL green (D5 keyboard map + D6
  collision/off-track surfaces) — full AC-024 closure (HUD, checkpoints,
  lap counting) remains Step 12 per the packet's partial binding.
- Gates (batch 87): consolidated `slice-advancement` with
  `--slice-id=P1-S7-G1 --plan plans/neatracing-voxel-racer.plans.md` →
  pass=true, TRIVIAL severity, 4/4 sub-gates (plan-sync, step-packet,
  plan-slice-quality, plan-command-lint) 0 failed / 0 errored; workflow
  MCP self-check PASS (0 errors, 0 warnings; snapshot phase 1 step 11,
  validationCommandCount 3); plan-session-redirect → pass, override
  written; cortex-index first run pass:false (plan family stale from the
  batch-87 append — the carried batch-85 item 7 flag) -> fixHint targeted
  rebuild executed (build-index.mjs + embed-index.mjs for the plan file:
  203 chunks indexed + embedded) -> gate re-run PASS, index_fresh true,
  all families fresh — the long-carried staleness flag is CLOSED; the
  exception was recorded via record-gate-exception.mjs to
  .github/ai-learning/learning-log.jsonl (sessionId
  P1-S7-G1-green-batch87). No content failures in any gate.
- Gate scoping (batch 87, no content gaps): code-coverage gate N/A — no
  src/ or scripts/agent-customization/ files touched (Step 11 is
  examples/- + plans/- only; examples/ excluded from coverage collection
  per the batch 99 item 99 precedent). GPU real-device gate N/A — no
  src/architecture/network/gpu/* paths. Browser smoke N/A — no browser
  runtime behavior changed; the manual full-lap check stays deferred to
  the Phase 1 exit (Step 13 boot-glue) per the step packet.
- Specialist reviews (one Tier-3 reviewer per FULL implementing slice):
  api-contract-reviewer P1-S7-A APPROVE (batch 82);
  implementation-pattern-scout P1-S7-B APPROVE (batch 84);
  implementation-pattern-scout P1-S7-C APPROVE (batch 86). Orchestrator
  artifacts on record (batch 88): coverage/shared-validation-p1s7a.json,
  p1s7b.json, p1s7c.json — all PASS firsthand; the severity gate consumes
  `--input=` (not `--changed-files=`) per the batch 84 item 2 standing
  note.

### Carried flags

- Full AC-024 closure (HUD wiring, checkpoints, lap counting) stays with
  Step 12 (spec P1-S7 D-E); the manual full-lap browser check is deferred
  to the Phase 1 exit (Step 13 boot-glue) per the step packet (batches
  87/88).
- JSDoc mojibake (double-encoded UTF-8: `Â§`, `â€"`) shared across the
  Step 11 test files (batch 84) — pre-existing repo-wide authoring
  artifact; a separate encoding-hygiene pass is recommended for a later
  slice, not a P1-S7 rework.
- Track rasterization bounding box is measured from segment endpoints
  only, so grass-apron cells of sharply diagonal chords can be clipped
  outside the endpoint hull — no contract assertion affected; visual
  follow-up optional (batch 82, api-contract-reviewer non-blocking note).
- Later-slice candidates from batch 86 (observations a/c/d/e; item b
  resolved by batch 87 lane 3): empty-segments micro-tests only if
  examples coverage is ever enforced; mirror band constants +
  wrapHeading duplication single-sourcing; test-side local interface
  instead of importing the exported type; standing 1e-6-branch slipAngle
  literal nit.
- Model mandate (batch 84 item 5): the batch-83 red subdispatch used
  kimi-k2.7-code:cloud instead of the user-mandated kimi-k2.7-code:cloud;
  results were fully validated and all subsequent dispatches complied;
  going forward every subdispatch carries the flash override.
- Packet wording drift on record (batches 80-81): the Step 11 packet's
  User instruction/Step objective/Blocked prose named heightfield
  sampling, raycast queries, banked turns, and boost pads — none exist in
  spec P1-S7 A-C; spec wins, no bank/boost surface was implemented, and
  the heightfield/raycast seam-artifact stop condition was moot.
- NeatRacingKeyboardSample naming (batch 85 item 4): the test pins it as
  the returned axes sample ({throttle, brake, steer}) with the
  pressed-key map as the argument type — self-consistent; the
  implementation followed the test file's header green contract.

## Step 12 — Track + physics + race part 2 (spec P1-S7 D-E) [DONE]

Archived from the tracker at close-out (green batches 90-95, close batch
96; packet compression at the same close-out batch 96). Full itemized
records live in the tracker's `## Latest validation evidence` (evidence
batches 90-95); this is the compressed archive — no prose or
stop-condition blocks (same archive shape as the Step 09/10/11 records)
and NO batch-summaries section (the itemized batch records stay in the
tracker's Latest validation evidence per the batches
25/34/41/51/56-61/73-78/80-88 itemized convention). The packet YAML is
archived verbatim below, statuses as [DONE] (incl. the P1-S7-D/E
red_test bullets).

```yaml
phase: 1
step: 12
title: 'Track + physics + race part 2 (spec P1-S7 D-E)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 13 - Boot-glue wiring: live render path + fallback append (batch 13 item 61 obs 1)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-S7-D'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host"'
  - 'manual: npm run dev:neatRacing then complete a full keyboard-driven lap (Phase 1 exit check)'
acceptance_criteria:
  - 'AC-024: full manual driving loop green: HUD speed/lap, race rules (laps, checkpoints with initial 60 s budget + 25 s per-lap extension), mode selector AI/manual toggle with restart, and a complete manual keyboard lap end-to-end'
slices:
  - slice_id: 'P1-S7-D'
    title: 'Race rules (laps, checkpoints, OQ-001/002 budgets)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/physics/race-rules.ts'
      - 'examples/neatRacing/physics/race-rules.test.ts'
    acceptance_criteria:
      - 'AC-024 (partial): checkpoint budget NEATRACING_CHECKPOINT_INITIAL_BUDGET_S=60 extended by NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S=25 per lap; lap counting at finish line'
    red_test:
      - 'RED: race-rules.test.ts fails before the module; asserts checkpoint budget math and lap counting'
    parallelizable: true
    dependencies:
      - 'P1-S7-G1'
  - slice_id: 'P1-S7-E'
    title: 'HUD + mode selector'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/host/hud.ts'
      - 'examples/neatRacing/host/mode-selector.ts'
      - 'examples/neatRacing/host/mode-selector.test.ts'
    acceptance_criteria:
      - 'AC-024: HUD speed/lap display; mode selector toggles AI vs manual and restarts the race on switch (AC-033 restart behavior pre-verified here)'
    red_test:
      - 'RED: mode-selector.test.ts fails before the selector; asserts toggle + restart behavior'
    parallelizable: true
    dependencies:
      - 'P1-S7-D'
  - slice_id: 'P1-S7-G2'
    title: 'Part 2 green validation: race rules + HUD + manual lap'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 12 AC subset green: AC-024 complete; manual full keyboard lap asserted at the Phase 1 exit check after Step 13 boot-glue wiring makes the render path live (batch 13 item 61 obs 1)'
    parallelizable: false
    dependencies:
      - 'P1-S7-D'
      - 'P1-S7-E'
```

**Latest validation evidence (P1-S7-G2):**

- Step-green union lanes (batch 95, direct shell invocations — the
  validation-MCP allowlist returned empty with activeStep null, batch-90/
  92/94/95 tooling-gap precedent): physics jest 3 suites / 49 tests
  (collision.utils, physics.step.service, race-rules); host jest 2
  suites / 32 tests (render-loop.utils 16 + mode-selector 16);
  `npx tsc --noEmit -p tsconfig.neatRacing.json` exit 0, 0 errors (the
  neatRacing-project lane is the type evidence of record — batch-86
  item 4b precedent; the root build tsc does not cover examples/**).
  Union: 5 suites / 81 tests, zero content failures, no flake evidence;
  lane ceilings match the batch-94 recorded values exactly — no
  P1-S7-D/P1-S7-E regressions.
- Provenance: P1-S7-D red batch 90 (TS2307 module-absence right-reason
  failures) -> implementation batch 91 (race-rules.ts +
  race-rules.test.ts; shared-validation artifact
  coverage/shared-validation-p1s7d.json PASS: race-rules.test.ts 21/21,
  build exit 0, lint exit 0; slice-advancement severity FULL 7/7
  sub-gates incl. specialist-review, 0 failed / 0 errored) -> green
  batch 92 with `fix-loop: P1-S7-D iteration 1 status=passed.`;
  P1-S7-E red batch 93 (executed the packet's declared red contract,
  TS2307 right-reason confirmed) -> implementation batch 94 (hud.ts +
  mode-selector.ts; coverage/shared-validation-p1s7e.json PASS:
  mode-selector.test.ts 16/16, build exit 0, lint exit 0;
  slice-advancement severity FULL 7/7 sub-gates, specialist-review
  evidence confirmed) -> green batch 94 with `fix-loop: P1-S7-E
  iteration 1 status=passed.`; P1-S7-G2 union green batch 95 with
  `fix-loop: P1-S7-G2 iteration 1 status=passed.` (zero fix loops
  across the step); step CLOSE batch 96 (orchestrator-owned status
  flips: heading + packet + all slice statuses [DONE]; packet
  compression to this record at the same batch).
- AC verdicts: AC-024 green for the closable scope (race rules + HUD +
  mode selector): OQ-001 checkpoint budget 60 s initial + 25 s per-lap
  extension, lap counting at finish-line forward crossings, OQ-002
  level advancement cap 1-2-3, terminal inertness; HUD speed/lap/time
  formatters with cyan/magenta overlay tokens; Neatenstein-pattern
  ai/manual selector with the once-per-switch onSwitch restart carrier
  pre-verifying AC-033 (manual default). The complete end-to-end manual
  keyboard lap clause lands at the Phase 1 exit check after Step 13
  boot-glue wiring (batch 13 item 61 obs 1) — deferred by design, not a
  skipped lane.
- Gates: consolidated slice-advancement with slice-id P1-S7-G2 +
  changed-files plans/neatracing-voxel-racer.plans.md (batch 95) →
  pass=true, severity TRIVIAL (md-only surface), 4/4 sub-gates
  (plan-sync, step-packet, plan-slice-quality, plan-command-lint),
  0 failed / 0 errored; convergence-tracker scan pass=true
  (iterationCount 0, pre-append).
- Gate scoping (batch 95, no content gaps): shared-validation/
  code-coverage/specialist-review N/A for the plan-only G2 slice —
  files_to_change contain no src/ or scripts/agent-customization/
  paths; GPU real-device gate N/A — no src/architecture/network/gpu/*
  surface; browser/manual lane N/A by design this batch — the manual
  full-lap check is the deliberately deferred Phase 1 exit check.
- Specialist reviews: the two FULL implementing slices closed with
  slice-advancement severity FULL 7/7 sub-gates (incl. shared-validation,
  code-coverage, specialist-review; 0 failed / 0 errored) at
  implementation batches 91 (P1-S7-D) and 94 (P1-S7-E), specialist-
  review evidence confirmed; the P1-S7-G2 surface was plan-only
  md TRIVIAL (batch 95). The batch-94 specialist observation binds
  Step 13 to the slice-internal `neatracing-hud-output` HUD
  container id. Orchestrator baseline artifacts on record and inspected
  at their recorded timestamps/states: coverage/shared-validation-
  p1s7d.json (ts 2026-09-30T11:49:44.146Z) + p1s7e.json
  (ts 2026-09-30T12:14:28.987Z).
- Orchestrator close-out context (batch 96): the plan-family RAG index
  went stale after batches 90-91 and was rebuilt via
  `node rag-index/reindex-plan-family.mjs --files=
  plans/neatracing-voxel-racer.plans.md` + embed catch-up +
  validate-index (all 10 families fresh; BM25 + hybrid-dense retrieval
  of the batch-90/91 chunks verified). Standing gap recurred after
  batch 87 and again at 90-91 — the recommended durable fix is
  automating the plan-family reindex into the post-append hook chain
  (learning event capture commissioned, separate record).

### Carried flags

- Manual full keyboard lap DEFERRED BY DESIGN to the Phase 1 exit check
  after Step 13 boot-glue wiring (batch 13 item 61 obs 1); the remaining
  deferred manual browser checks (Steps 03/06/07) share that exit lane.
- Step 13 must adopt the slice-internal `neatracing-hud-output` HUD
  container id (specialist obs, batch 94).
- 5 pre-existing no-explicit-any eslint warnings in
  shared/network-visualization/network-view.test.ts (batch 95 item 6) —
  unrelated surfaces outside both slices' files_to_change; standing
  lint-noise note.
- validation-MCP allowlist empty during the Step 12 red/green/close-out
  batches (activeStep null during the [PLANNED] -> [DONE] handoff;
  batch 95 item 12) — packet-declared commands ran via direct shell;
  standing tooling gap recorded, no content failure.