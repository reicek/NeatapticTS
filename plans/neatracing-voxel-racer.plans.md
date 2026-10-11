# Neat Racing Voxel Racer (neatRacing)

**Status:** [WIP]

<!-- Derived from plans/neatracing-voxel-racer.spec.md (spec sections 13, 17, 18, 12.4-12.5).
     This is the tracker for the neatRacing voxel racer flagship demo. Phase/step/slice
     structure mirrors spec section 17 phases P1/P2/P3 1:1 with slice_ids kept verbatim
     so section 18 traceability holds without renumbering. Planning-only authoring pass:
     no src/, docs/, examples/, or scripts/ production edits. -->

## Scope

- Implement the neatRacing voxel racer per `plans/neatracing-voxel-racer.spec.md`:
  Phase 1 graphics engine + manual keyboard driving (AC-001..AC-024 + AC-040..AC-042),
  Phase 2 AI migration with NGE growth engine (AC-025..AC-034 + AC-043),
  Phase 3 polish + deprecation prep (AC-035..AC-039).
- Planning-only pass in this instance: author full machine-readable step packets,
  register the plan, rebind the session, reindex RAG, prove gates. Execution is
  dispatched step-by-step to implementing/testing specialists after verification.
- Single source of truth for acceptance criteria and validation commands is the spec
  (section 13 ACs, section 12.5 conventions); this tracker binds each spec slice to its
  ACs, files, red-test notes, and validation commands.

## Mandates

- **Planning-only:** this authoring pass edits only `plans/neatracing-voxel-racer.plans.md`,
  `plans/README.md`, `plans/Roadmap.md`, `.vscode/mcp.json`, and RAG index artifacts.
- **Spec lock:** spec section 12 locked decisions register (LD) and resolved open
  questions are not reopened by the tracker. Deprecation prep is notices-only
  (LD-33); removal of `examples/racing_curriculum` is a separate user-confirmed
  workstream and out of scope.
- **Machine-readable packets:** every execution step carries a YAML packet with
  `expansion: slices`, `auto_expand: true`, `tdd_sequence: green-only`, and typed
  slices (slice_id / title / status / goal / estimate_hours / files_to_change /
  acceptance_criteria / parallelizable / dependencies) per the step-packet gate schema.
- **Slice discipline:** each slice is one atomic behavioral intent bound to spec
  section 17; at most 3 code files per slice (sibling `*.test.ts` files count;
  trivial one-line config registrations such as a package.json script line or a
  tsconfig include may accompany without counting); estimate_hours <= 4.
- **Red-first:** every implementation slice carries `red_test` notes (the failing
  assertion to author first) honoring the spec RED -> IMPLEMENT -> GREEN rule; each
  step ends with a green-testing slice that runs the section 12.5 validation list.
- **Named constants:** tuning values use the named constants from the spec open
  questions patch (NEATRACING_FIXED_RENDER_HEIGHT, NEATRACING_MAX_BACKING_WIDTH,
  NEATRACING_AI_MIN_SESSION_LAPS, NEATRACING_EDITOR_UNDO_BUDGET_BYTES,
  NEATRACING_REBUILD_P95_BUDGET_MS, NEATRACING_ADAPTATION_TICK_BUDGET_MS, etc.).
- **Zero per-frame allocation (P9)** and **capacity limits preserved** (maxNodes 8000 /
  maxConnections 32000, AC-029) are standing constraints on all Phase 1/2 slices.
- **No deferred cleanup:** when a slice replaces an existing module or script, the old
  code is removed in the same slice.
- **No git:** agents never run git commands; all changes land via direct file edits.
- **Evidence:** gate runs are recorded in `## Latest validation evidence`; green-light
  recording is reserved for the separate verification instance.

## Current state

- Spec `plans/neatracing-voxel-racer.spec.md` is complete (55 impl slices preview,
  43 ACs, section 18 traceability 43/43, all open questions RESOLVED).
- This tracker is authored and registered as [WIP]; Phase 1 is the active phase;
  Steps 02-12 are [DONE] (Step 08 compressed in batch 62; Step 09 compressed
  in batch 72; Step 10 compressed at close-out batch 79; Step 11 packet
  compressed at close-out batch 89; Step 12 packet compressed at close-out
  batch 96); Step 13
  (boot-glue wiring: live render path + fallback append) is [DONE] (closed
  2026-10-03 on user approval, batch 118 — boot wiring green, car-pass GLSL
  defect repaired batch 114, dev-server staleness fixed batch 115, served
  page shows `Neat Racing booted.` with HUD + selector). Step 14
  (browser composition) is [DONE] on automated evidence (batch 134) after
  fixing the blank-screen root causes and passing all neatRacing jest lanes,
  tsc, build, and visible-window smoke; the user-gate confirmation was not
  available in the closing session. Step 14B
  (composition correctness pass: grid flat horizon + visible track + F1 car
  shape + four wheels + drivable physics) is the newly inserted [WIP]
  fix-up step between Step 14 and Step 15, opened by the user after review
  observed the grid tilted, track missing, car all-black with only two
  visible wheels, and no keyboard drive response. Step 15
  (editor editing-loop composition) remains [PLANNED] after Step 14B; Steps
  15-22 are [PLANNED] (Phase 2 = global 16-20, Phase 3 = global 21-22
  after the batch-105 editing-loop renumber, batch-26 precedent).
- Step 04 delivered the voxel pipeline part 1 (batches 28-34, GREEN: OK):
  model.schema.ts + model.decode.utils.ts (33 tests), greedy-mesher.ts
  (11 tests), bin.codec.ts (24 tests), build-voxel-bins.mjs, voxel:build;
  68 union tests green; union specialist APPROVE (item 123).
- Step 05 delivered the voxel pipeline part 2 (batches 35-41, GREEN: OK):
  lod.builder.ts (11 tests, all 9 spec §7.3.3 rules; spec §7.3.3/LD-19 wins
  over the retired tracker lod32/lod8 shorthand) + lod.distance.utils.ts
  (15 tests, §7.3.4 distance swap: 0.03515625 m world, focal 623.54 px,
  swap ≈43.84 m, ±10% hysteresis); 94 union voxel tests green; union
  specialist APPROVE (P1-S3-green-2); union gate FULL 7/7.
- Step 06 (spec P1-S4 A-C) is [DONE] STEP-GREEN (batches 42-50):
  P1-S4-A materials semantics (17 tests, AC-013 partial, LD-23),
  P1-S4-B team-remap (10 tests, AC-014; cyan/red/green/blue table,
  pure/non-mutating, EMPTY-slot skip, no cross-team bleed), and
  P1-S4-C wheel.mount.utils (36 tests, AC-015; 32³ wheel grid,
  hub=radius contact patch, +90° mount, 4 sockets,
  NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S=2 sign-preserving clamp).
  Union voxel scan 8 suites / 157 tests green; tsc + eslint clean;
  3/3 fresh Tier-3 specialist reviews APPROVE (batch 48 item 179;
  closes item 173(i)); consolidated slice-advancement 7/7 FULL;
  convergence-tracker PASS (zero fix loops).
- Step 07 (spec P1-S4 D-E) is [DONE] STEP-GREEN (batches 52-54):
  P1-S4-D car lighting + car pass (directional sun + hemisphere
  ambient + exp2 fog mirror, two-sided bins meshes via the item-120
  view-flip contract, blob shadows at +0.01 m, draw-time team remap,
  four wheel sockets via G9, warmup key 'car') and P1-S4-E glass pass
  - bloom pipeline (G4 glass blend state machine, LD-23
    black-never-emits-shine, half-res bloom threshold/blur/additive,
    AC-035 zero-op off-switch, keys[6..8]); engine union scan 6 suites /
    191 tests green; tsc + lint clean; NEATRACING_WARMED_PROGRAM_COUNT
    = 9 (keys 0-8, AC-040 parity); specialist rounds APPROVE
    (webgpu-parity-reviewer substitute + fresh instances;
    webgl2-glsl-render-reviewer added batch 53); consolidated
    slice-advancement 7/7 FULL. Manual dev-server visual check of car team
    colors, glass highlights, and bloom intensity landed in Step 08
    (batch 61: declared command + visible-window smoke green).
- Step 08 (spec P1-S5) is [DONE] STEP-GREEN (batches 56-61): P1-S5-A model
  store CRUD + validation (30 tests), P1-S5-B dev server with S1-S6
  hardening (33 tests incl. the CLI self-start contract after
  fix-packet-P1-S5-B-iteration-2; convergence marker
  `fix-loop: P1-S5-B iteration 2 status=passed.`), P1-S5-C editor shell
  page + entry (17 tests; fix-packet-P1-S5-C-iteration-1 closed), P1-S5-D
  editor camera controls (19 tests); union lanes 2 suites / 63 tests
  (server) + 2 suites / 36 tests (editor) green; declared-command manual
  check + visible-window browser smoke green (batch 61); consolidated
  slice-advancement 4/4 PASS; step packet compressed to
  plans/neatracing-voxel-racer.logs.md ("Step 08" section) in batch 62.
- Standing dispatch constraints: (a) RESOLVED (batch 42, closes item 120):
  signed-face-orientation contract pinned — normalAxis stays u8 0..2 with NO
  winding field; renderers draw bins-derived meshes two-sided and reconstruct
  the effective signed normal via the per-fragment view-flip convention
  (spec §7.3.1 amended text-only; §7.3.2 layout + landed mesher/codec
  unchanged; all bins-consuming renderer slices follow spec §7.3.1); (b)
  RESOLVED: lod wording drift reconciled at P1-S3-D, spec §7.3.3/LD-19 wins
  (items 109/135/143/147; 32³ tier default NOT built).
- Coverage-backlog renumber patch applied (batch 26; batch 19 item 78 obs a):
  Phase 2/3 backlog rows now carry the post-insertion global numbers
  (Phase 2 = 14-18, Phase 3 = 19-20); AC-002 keeps "13" uniquely (Phase 1
  Step 13 boot-glue). Independently verified green-light: true by the fresh
  `01-planning` verification instance (batch 27 items 103-104); Step 04
  dispatch unblocked. Re-renumbered after the batch-102 Step 14 insertion
  (batch 103; batch-26 precedent): Phase 2 = 15-19, Phase 3 = 20-21; AC-002
  adds "14" (composition wiring per batch 102), AC-042 rebinds old Phase 2
  Step 01 "14" to "15", and the other Phase 2/3 row bindings shift by +1.
  Re-renumbered again after the batch-105 Step 15 editing-loop insertion
  (batch 107, patch #2; batch-26 precedent): Phase 2 = 16-20, Phase 3 =
  21-22; AC-019 adds "15" (P1-editor-controls CAD ortho wiring), AC-020 +
  AC-021 add "15" (editing-loop browser wiring + live-save per batch 105),
  AC-042 rebinds old Phase 2 Step 01 "15" to "16", and the other Phase 2/3
  row bindings shift by +1 again.
- Session binding: `plan-session-redirect.mjs` + `.vscode/mcp.json` point
  neataptic-validation-mcp and neataptic-workflow-mcp at this plan.
- RAG plan family reindexed after authoring; cortex-index gate is the freshness proof.
- Batch 102 integration-gap classification (user manual retest of `/` and
  `/editor/` + orchestrator analysis, 2026-09-30): the nine Phase 1 render
  passes are module-green (warmup keys `starfield`/`grid`/`pulse`/`track`/
  `car`/`glass`/`bloom-threshold`/`bloom-blur`/`bloom-additive` via
  NEATRACING_WARMED_PROGRAM_KEYS), but NO frame composer/render-pipeline
  module exists, the boot warmup registry is `readonly never[] = []`, the
  batch-94 `neatracing-hud-output` container handoff is unadopted, the HUD/
  mode-selector/race/physics modules are browser-unwired, and the editor
  page composes only canvas + context + one-time clear. Classified a
  PLANNING DEFECT (missing browser-composition wiring step between Step 13
  and the exit check), not an AC-002 regression. Remediation = the inserted
  Step 14 packet below (frame composer + shell HUD adoption + 9-key warmup
  registry + live drive loop + the recorded editor-shell composition
  decision). The Phase 1 exit check is unreachable until Step 14 is DONE
  and green; the user re-runs the manual exit check afterward.
- Batch 105 Phase-1 scope correction (user manual-test follow-up +
  orchestrator analysis, patch #2, batch 107): Phase 1 must close with a
  RUNNING engine AND a WORKING editor; the batch-103 Step 14 packet's
  P1-editor-preview slice wires only the passive preview lane, leaving the
  editor editing composition (pointer-pick edit -> rebuild pipeline ->
  preview redraw; palette + undo/redo + model-dropdown + CAD-view controls;
  mirror-X symmetry, default ON; immediate live-save through the dev-server
  model-store API plus the AC-021 manual edit -> refresh -> persisted
  check) owned by NO slice — the same planning-defect class as the batch
  102 gap (v) second half. Remediation = the inserted Step 15 packet below
  (tool-frame wiring + pointer-pick edit loop + controls + symmetry/live-save,
  with editor jest lanes + DEMO/UI visible-window smoke on `/editor/` and
  AC-020/AC-021 traceability); P1-editor-preview stays preview-only. The
  Phase 1 exit check is now the comprehensive TWO-PAGE USER manual retest
  (game page full keyboard lap; editor page full editing round-trip) and is
  unreachable until Step 15 is DONE and green; no Phase-1 user-facing
  functionality may be deferred to Phase 2/3.
- **Step 14 IN FLIGHT — session paused 2026-10-04 17:11, resume handoff at
  plan EOF under `handoff:step-14-resume-2026-10-04` (batch 129).** All five
  Step-14 slices reverted [WIP] by user mandate (batches 127/128 — statuses
  follow served-page behavior ONLY; no [DONE] on code evidence; no
  postponement to Step 15 or later phases is permitted). Two green
  iterations on P1-composition-green FAILED (batches 124/125): game `/`
  composites ~99.8% black (root cause 1 = warmup-sweep viewport 1x1 residue,
  fixed + served, ineffective; scene FBO complete-but-empty — cause 2 still
  unfixed); editor `/editor/` shows only a 22x15 px car on a flat canvas
  (user adjudication: counts as NOT visible). Iteration-2 fix packet
  (real-GL present-path tripwire + empty-FBO localization + letterbox +
  editor visibility) was dispatched twice (05438124 instance cleared by the
  session stop WITHOUT writing any files — verified 2026-10-04 17:12: all
  Step-14 file mtimes predate the dispatch). RESUME: dispatch a fresh
  04-implementing on `fix-packet-P1-composition-green-iteration-2` (+ the
  batch-128 full-page evidence mandate), then shared-validation, then a
  fresh 05-green (rebuild + server restart + BOTH-page visible Chrome
  smoke), then USER manual confirmation. Step 14 closes ONLY on user-
  confirmed visible driving (game page) + visible editor render.

## Coverage backlog (AC -> step/slice map)

| AC     | Step       | Slice(s)                                                                                                                                                                 |
| ------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| AC-001 | 02         | P1-S1-A (partial), P1-S1-B                                                                                                                                               |
| AC-002 | 02, 13, 14 | P1-S1-C (service), P1-boot-glue (wiring), P1-frame-composer + P1-boot-composition + P1-drive-loop-live (composition wiring per batch 102)                                |
| AC-003 | 02         | P1-S1-D                                                                                                                                                                  |
| AC-004 | 03         | P1-S2-A                                                                                                                                                                  |
| AC-005 | 03         | P1-S2-B                                                                                                                                                                  |
| AC-006 | 03         | P1-S2-C                                                                                                                                                                  |
| AC-007 | 04         | P1-S3-A (P1-S5-A reuses validation)                                                                                                                                      |
| AC-008 | 04         | P1-S3-A                                                                                                                                                                  |
| AC-009 | 04         | P1-S3-B                                                                                                                                                                  |
| AC-010 | 04         | P1-S3-C                                                                                                                                                                  |
| AC-011 | 05         | P1-S3-D                                                                                                                                                                  |
| AC-012 | 05         | P1-S3-E                                                                                                                                                                  |
| AC-013 | 06, 07     | P1-S4-A (partial), P1-S4-E                                                                                                                                               |
| AC-014 | 06         | P1-S4-B                                                                                                                                                                  |
| AC-015 | 06         | P1-S4-C                                                                                                                                                                  |
| AC-016 | 07         | P1-S4-D (partial), P1-S4-E                                                                                                                                               |
| AC-017 | 11         | P1-S7-A                                                                                                                                                                  |
| AC-018 | 08         | P1-S5-A (support), P1-S5-B                                                                                                                                               |
| AC-019 | 08, 14, 15 | P1-S5-C (partial), P1-S5-D, P1-editor-preview (preview render per spec §8.3), P1-editor-controls (CAD ortho wiring per batch 105)                                        |
| AC-020 | 09, 15     | P1-S5-C (partial), P1-S6-A, P1-S6-B, P1-editor-toolframe + P1-editor-controls (editing-loop browser wiring per batch 105)                                                |
| AC-021 | 09, 15     | P1-S6-C, P1-editor-toolframe (state wiring), P1-editor-editloop + P1-editor-symmetry-save (edit-loop browser wiring + live-save + the AC-021 manual check per batch 105) |
| AC-022 | 10         | P1-S6-D                                                                                                                                                                  |
| AC-023 | 11         | P1-S7-B                                                                                                                                                                  |
| AC-024 | 11, 12, 14 | P1-S7-C (partial), P1-S7-D (partial), P1-S7-E, P1-drive-loop-live (browser wiring; manual lap stays the exit check)                                                      |
| AC-025 | 16         | P2-S1-A, P2-S1-B                                                                                                                                                         |
| AC-026 | 16, 18     | P2-S1-C, P2-S3-A (support)                                                                                                                                               |
| AC-027 | 17         | P2-S2-B                                                                                                                                                                  |
| AC-028 | 17         | P2-S2-C                                                                                                                                                                  |
| AC-029 | 17, 18, 19 | P2-S2-A (partial), P2-S3-A, P2-S3-E (support)                                                                                                                            |
| AC-030 | 18         | P2-S3-B                                                                                                                                                                  |
| AC-031 | 18         | P2-S3-C                                                                                                                                                                  |
| AC-032 | 19         | P2-S3-D                                                                                                                                                                  |
| AC-033 | 20         | P2-S4-A, P2-S4-C                                                                                                                                                         |
| AC-034 | 20         | P2-S4-B                                                                                                                                                                  |
| AC-035 | 21         | P3-S1-A                                                                                                                                                                  |
| AC-036 | 09, 21     | P1-S6-B (partial), P3-S1-B                                                                                                                                               |
| AC-037 | 02, 22     | P1-S1-B (partial), P3-S2-A                                                                                                                                               |
| AC-038 | 22         | P3-S2-B                                                                                                                                                                  |
| AC-039 | 22         | P3-S2-C                                                                                                                                                                  |
| AC-040 | 02, 14     | P1-S1-C, P1-boot-composition (boot warmup registry population)                                                                                                           |
| AC-041 | 10         | P1-S6-E                                                                                                                                                                  |
| AC-042 | 02, 16     | P1-S1-D (partial), P2-S1-B (partial)                                                                                                                                     |
| AC-043 | 17, 18     | P2-S2-A (partial), P2-S3-A                                                                                                                                               |

### Derivation notes

1. **1:1 slice preservation:** spec section 17 slice_ids (P1-S1-A .. P3-S2-C) appear
   verbatim with their spec file lists and AC bindings; section 18 traceability holds
   without renumbering. The tracker adds one trailing green-testing slice per execution
   step (id `<spec-step>-green`, e.g. `P1-S1-green`) to host the section 12.5
   validation sweep; spec impl slices keep goal `implementing`.
2. **a/b step splits:** spec steps with 5 impl slices (P1-S3, P1-S4, P1-S6, P1-S7,
   P2-S3) are split into part 1 (A-C) and part 2 (D-E) sub-steps because a
   gate-conformant green-only step needs at least 2 and at most 5 slices and a
   trailing green-testing slice (5 impl + 1 green = 6 > 5). Spec slice_ids, file
   lists, and AC bindings are unchanged by the split.
3. **Uniform green-only tdd_sequence:** spec slices are RED -> IMPLEMENT -> GREEN
   atomic units; per-slice RED notes live in the slice `red_test` field so the
   gate-conformant step shape is uniformly `green-only` (all-but-last slices
   `implementing`, last slice `green-testing`).
4. **Planning ACs** use the AC-PLN-001..AC-PLN-005 prefix to avoid collision with the
   spec AC-001..AC-043 numbering.
5. **Dual-parser layout contract:** two parsers coexist and this tracker satisfies both.
   The step-packet GATE parses only the fenced YAML packets (heading-agnostic). The
   neataptic-workflow MCP additionally parses HEADINGS scoped to `## Implementation
phases`: `### Phase <N> — <Title> [STATUS]` (H3 + em-dash, terminated by a
   `## …validation gates…` H2) and `#### Step <N> — <Title> [STATUS]` (H4) inside the
   phase body; the active [WIP] step also carries `**Step objective:**` and
   `**Required validation:**` prose sections (backtick commands prefix-matching the
   YAML `validation` list). Keep this shape on every future step flip.
6. **Global step numbering:** the coverage backlog `Step` column uses global
   step numbers across phases — Phase 1 = 02-15 (Step 13 = P1-boot-glue
   wiring; Step 14 = browser composition, inserted between Step 13 and the
   exit check per the batch-102 integration-gap classification; Step 15 =
   editor editing-loop composition, inserted between Step 14 and the exit
   check per the batch-105 editing-loop scope correction, patch #2),
   Phase 2 = 16-20, Phase 3 = 21-22. Phase 2/3 rows were renumbered from
   the pre-insertion global sequence (batch 19 item 78 obs a), re-renumbered
   after the batch-102 Step 14 insertion, and re-renumbered again after the
   batch-105 Step 15 insertion (the batch-26 renumber precedent) so
   "13"/"14"/"15" bind uniquely to Phase 1 steps and Phase 2 starts at
   global 16.

## Implementation phases

### Phase 1 — Graphics engine + manual keyboard driving [WIP]

```yaml
phase: 1
title: 'Graphics engine + manual keyboard driving'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_phase: 'Phase 2 - AI migration (NGE) - dispatch after Phase 1 steps 02-15 are DONE (the Phase 1 exit check follows Step 15 per the batch-105 editing-loop scope correction, patch #2)'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - spec-checklist
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated; never run plan-sync / step-packet / plan-slice-quality / plan-command-lint individually)'
  - 'neataptic-gate-mcp:run_gate_check gate=cortex-index (RAG plan-family freshness proof)'
constitution_check: 'principle-2-human-mission-ai-method; principle-4-breadth-first-recoverable; principle-5-unique-ids; section-development-workflow'
acceptance_criteria:
  - 'AC-PLN-001: tracker derives 1:1 from spec section 17 - every spec slice P1-S1..P1-S7 appears with verbatim slice_id, files_to_change, and AC binding'
  - 'AC-PLN-002: every step packet is step-packet-gate conformant (goal enum, expansion, tdd_sequence green-only, max 5 slices, max 4h per slice, slice-goal ordering)'
  - 'AC-PLN-003: coverage backlog maps all Phase 1 spec ACs (AC-001..AC-024 + AC-040..AC-042) to steps/slices'
  - 'AC-PLN-004: plan registered in plans/README.md + plans/Roadmap.md as [WIP]; .vscode/mcp.json + plan-session-redirect.mjs rebound; cortex-index gate green'
  - 'AC-PLN-005: planning-only mandate held - no production code edits; validation evidence + handoff query recorded; no calendar dates in headings'
placeholder_steps:
  - 'Step 01 - Author step packets and register plan [DONE]'
  - 'Step 02 - Shell canvas + WebGL2 boot + render loop (spec P1-S1) [DONE]'
  - 'Step 03 - Neon background passes (spec P1-S2) [DONE]'
  - 'Step 04 - Voxel data pipeline part 1: schema + mesher + bin codec (spec P1-S3 A-C) [DONE]'
  - 'Step 05 - Voxel data pipeline part 2: LOD + distance swap (spec P1-S3 D-E) [DONE]'
  - 'Step 06 - Car materials + wheels part 1: materials + remap + wheel mount (spec P1-S4 A-C) [DONE]'
  - 'Step 07 - Car materials + wheels part 2: lighting + glass + bloom (spec P1-S4 D-E) [DONE]'
  - 'Step 08 - Dev server + editor core (spec P1-S5) [DONE]'
  - 'Step 09 - Editor features part 1: palette + undo + symmetry (spec P1-S6 A-C) [DONE]'
  - 'Step 10 - Editor features part 2: generator + rebuild pipeline (spec P1-S6 D-E) [DONE]'
  - 'Step 11 - Track + physics + race part 1: track + physics + input collisions (spec P1-S7 A-C) [DONE]'
  - 'Step 12 - Track + physics + race part 2: race rules + HUD (spec P1-S7 D-E) [DONE]'
  - 'Step 13 - Boot-glue wiring: live render path + fallback append (batch 13 item 61 obs 1) [DONE]'
  - 'Step 14 - Browser composition: frame composer + shell HUD adoption + 9-key warmup registry + live drive loop (batch 102 patch) [DONE] (closed on automated evidence; user-gate confirmation unavailable in this session)'
  - 'Step 14B - Composition correctness pass: grid flat horizon + visible track + F1 car shape + four wheels + drivable physics (batch 108 user-gap patch) [WIP]'
  - 'Step 15 - Editor editing-loop composition: tool-frame wiring + pointer-pick edit loop + palette/undo/model-dropdown/CAD-view controls + mirror-X symmetry + live-save (batch 105 patch, patch #2) [PLANNED]'
```

**Phase objective:** Deliver the playable voxel racer shell end to end: browser canvas + WebGL2 boot + neon background passes, voxel data pipeline (schema/mesher/LOD/codec), car materials + wheels + full car pass, dev server + editor core, editor features, and track + physics + race flow with manual keyboard driving.

**Stop conditions:**

- **Done:** Steps 02-15 DONE with AC-001..AC-024 + AC-040..AC-042 green; Step 13 boot-glue wiring makes the render path live (batches 97-100), Step 14 browser composition wires the composition + shell HUD + drive loop (batch 102 patch), and Step 15 editing-loop composition wires the editor tool frame + pointer-pick edit loop + palette/undo/model-dropdown/CAD-view controls + mirror-X symmetry + live-save (batch 105 patch, patch #2); the comprehensive two-page USER manual retest (game page full keyboard lap; editor page full editing round-trip — palette colors, visible mirror-X symmetry, undo/redo, model dropdown, CAD front/side/top views, live-save + reload persistence) recorded as the user-gated Phase 1 exit check after Step 15.
- **Blocked:** any step packet fails its gate after two fix attempts — record the blocker in Latest validation evidence and escalate via 00.cross-tier-helper.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md`
- Consolidated slice gate per step: `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=<slice_id> --changed-files=<comma-separated files>`
- RAG freshness: reindex via `rag-index/build-index.mjs`, then `node scripts/agent-customization/gates/cortex-index.gate.mjs --json`

#### Step 01 — Author step packets and register plan [DONE]

**Done record:** planning pass complete; double-verified `green-light: true` by two independent verification instances (evidence batches 6-7 in `## Latest validation evidence`). Gate-compliant outputs: 3 phase + 19 step packets authored from spec section 17; plan registered as the active [WIP] workstream in plans/README.md + plans/Roadmap.md; `plan-session-redirect.mjs` and `.vscode/mcp.json` MCP `--plan` args rebound; RAG plan family reindexed; slice-advancement FULL pass for slice `P1-step-01-author`. Per tracker-handoff compression, the full YAML packet, stop conditions, objective, and validation commands are archived at `plans/neatracing-voxel-racer.logs.md` (Step 01 record).

#### Step 02 — Shell canvas + WebGL2 boot + render loop (spec P1-S1) [DONE]

**Done record:** spec P1-S1 delivered green by four implementing slices + one step-green slice (49 tests, evidence batches 9-14): P1-S1-A boot shell + boot stub; P1-S1-B fixed-canvas strategy (720/1920 + letterbox) + tsconfig.neatRacing.json + build:neatRacing; P1-S1-C WebGL2 context service (WeakMap single init, graceful fallback, G5 loss/restore, 9-program warmup); P1-S1-D fixed-timestep 1000/60 ms accumulator loop (catchup cap 4, zero per-frame allocations). Step green: 49/49 jest, tsc clean, build clean, slice-advancement FULL 7/7 (union + per-slice runs), specialist APPROVE (batch 13, four non-blocking observations), DEMO/UI real visible-window browser smoke PASS (bundle load + canvas mount + console clean; full frame-1 render = later-wiring checkpoint per batch 13 obs 2). Boot-glue wiring owner assigned (batch 16 item 71): targeted Step 13 slices P1-boot-glue + P1-boot-glue-green before the Phase 1 exit check. Per tracker-handoff compression, the full YAML packet, stop conditions, objective, validation commands, and evidence batches 9-14 are archived at `plans/neatracing-voxel-racer.logs.md` (Step 02 record).

#### Step 03 — Neon background passes (spec P1-S2) [DONE]

**Done record:** spec P1-S2 delivered green by three implementing slices + one step-green slice (71 slice tests -> 77 union tests, evidence batches 15-24): P1-S2-A starfield pass + constants.starfield.ts (AC-004, exact flappy parity xorshift32, 3 layers seeds 1337/2777/4242, G6 scroll math, GLSL ES 3.00 point sprites, warmup seam key [0]; per-slice webgpu-parity-reviewer APPROVE); P1-S2-B grid pass (AC-005, fwidth constant widths 2px/4px, depth fade 0.58->0.29->0.05 over 320 m, y=-0.01 plane); P1-S2-C pulse pass (AC-006, #B7FF00 ring alpha 0.42, deterministic envelope, GL_LESS + premultiplied-additive). Union green: 3 suites / 77 tests, tsc exit 0, slice-advancement FULL 7/7 (union + per-slice runs), slice-validator PASS. Union specialist REQUEST_CHANGES -> fix-packet-P1-S2-green-iteration-1 executed (pulse color normalized [185/255, 1, 0] — spec 0xB7=183 hex/decimal defect flagged to spec owner; grid AC-040 warmup seam added keyed 'grid'; canonical to<X>WarmupProgram(gl) factory unified across the three passes; GL_BLEND enabled) -> fresh Tier-3 re-review APPROVE -> GREEN OK (batches 22-24). Manual sky-composite browser check deferred to Step 08 / P1-boot-glue-green per packet. Plan-hygiene: the AC-005 "pulse sync" tracker paraphrase (spec has no such requirement) is retired with the archive. Full YAML packet, prose, stop conditions, validation, and evidence batches 15-24 archived at `plans/neatracing-voxel-racer.logs.md` (Step 03 record).

#### Step 04 — Voxel data pipeline part 1: schema + mesher + bin codec (spec P1-S3 A-C) [DONE]

**Done record:** All four slices [DONE] + P1-S3-green GREEN: OK (batch 33
items 124-133): jest:base voxel pattern 3 suites / 68 tests exit 0; tsc
exit 0; `npm run voxel:build` exit 0 (designed zero-model short-circuit,
resolution 113c). Union slice-advancement FULL 7/7 (9-file impl union,
item 121); union specialist implementation-pattern-scout APPROVE (item
123); api-contract-reviewer APPROVE per slice (items 108, 116, 117); fix
packet P1-S3-A dims conformance closed at iteration 1 (item 107 —
declared-dims schema per §7.2/§7.4/TP-14, validateSymmetricWidth LD-28).
Deliverables: model.schema.ts, model.decode.utils.ts (+33 tests),
greedy-mesher.ts (+11 tests), bin.codec.ts (+24 tests),
examples/neatRacing/scripts/build-voxel-bins.mjs, package.json
voxel:build. Standing constraint carried: signed-face-orientation gap
(normalAxis u8 0..2, no winding field) — 01-planning patch REQUIRED before
P1-S4/P1-S5 renderer dispatch (item 120). Tracker AC wording lod drift
(lod128/lod32/lod8 vs spec §7.3.3 lod128+lod64-only) reconciles at
P1-S3-D. Full packet + batches 28-34 archived in
plans/neatracing-voxel-racer.logs.md ("Step 04" section).

#### Step 05 — Voxel data pipeline part 2: LOD + distance swap (spec P1-S3 D-E) [DONE]

**Done record:** All three slices [DONE] + P1-S3-G2 GREEN: OK (batch 40
items 148-157): jest:base voxel pattern 5 suites / 94 tests exit 0; tsc
exit 0. Union slice-advancement FULL 7/7 (slice-id=P1-S3-green-2, 4-file
impl union); union specialist implementation-pattern-scout APPROVE with
every §7.3.3 rule and the full §7.3.4 formula chain verified and all
arithmetic independently recomputed (5 non-blocking observations carried:
glow-quantization interpretation, local FOV constant, black-glass→EMPTY
dedup edge, palette aliasing, unpinned merge tie-break branch);
api-contract-reviewer APPROVE per slice (items 141, 145). Deliverables:
lod.builder.ts (+11 tests — all 9 §7.3.3 rules; spec §7.3.3/LD-19 wins
over the retired tracker lod32/lod8 shorthand, reconciliation recorded in
items 109/135/143; 32³ tier default NOT built) and lod.distance.utils.ts
(+15 tests — §7.3.4 chain 0.03515625 m world → focal 623.54 px → swap
≈43.84 m → ±10% hysteresis; NaN-total, allocation-free). Carried notes:
NEATRACING_LOD_SWAP_FOV_DEG (60) defined locally — shared-FOV
reconciliation deferred; no in-repo consumer of buildLodModel yet
(car-lod64.bin emission deferred to P1-S6-D assets). Full packet + batches
35-41 archived in plans/neatracing-voxel-racer.logs.md ("Step 05" section).

#### Step 06 — Car materials + wheels part 1 (spec P1-S4 A-C) [DONE]

**Done record:** spec P1-S4 A-C delivered green by three implementing slices + one step-green slice (63 slice tests -> union voxel scan 8 suites / 157 tests, evidence batches 42-50): P1-S4-A materials semantics — examples/neatRacing/voxel/materials.ts (17 tests, AC-013 partial, LD-23; VoxelMaterialKind/VoxelMaterial, resolveVoxelMaterial verbatim passthrough, buildVoxelMaterialTable O(palette), VOXEL_MATERIAL_BLACK_EMISSIVE=0 proven at every glow magnitude, VOXEL_MATERIAL_GLASS_ALPHA_MAX=0); P1-S4-B team-remap — team-remap.ts (10 tests, AC-014; remapVoxelPaletteToTeam pure/non-mutating, cyan/red/green/blue ids 0-3, EMPTY-slot skip, alpha+glow preserved, no cross-team bleed); P1-S4-C wheel.mount.utils — wheel.mount.utils.ts (36 tests, AC-015; 32³ wheel grid at 0.03515625 m/voxel, hub height = disc radius contact patch, pivot at grid center, NEATRACING_WHEEL_MOUNT_YAW_RAD=+π/2, 4 sockets, NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S=2 sign-preserving clamp, resolveVoxelWheelMetrics ω=v/r TP-14). Item-120 signed-face-orientation contract RESOLVED batch 42 (spec §7.3.1 amended text-only: bins-derived meshes draw two-sided with the per-fragment view-flip normal; glass pass §6.5 culling-ON exempt; normalAxis u8 0..2 with NO winding field — §7.3.2 normative, AC-010 + landed mesher/codec unchanged). Step green (batch 49): union `jest --testPathPatterns="examples/neatRacing/voxel"` 8 suites / 157 tests exit 0, `npx tsc -p tsconfig.neatRacing.json --noEmit` clean, eslint clean; 3/3 fresh Tier-3 specialist reviews APPROVE (batch 48 item 179; closes item 173(i)); consolidated slice-advancement 7/7 FULL; convergence-tracker PASS with `fix-loop: P1-S4-G1 iteration 1 status=passed` (zero fix loops). Per tracker-handoff compression, the full YAML packet, prose (incl. the item-120 contract), stop conditions, validation commands, and batch summaries (batches 42-50, items 159-190) are archived at `plans/neatracing-voxel-racer.logs.md` (Step 06 record).

#### Step 07 — Car materials + wheels part 2 (spec P1-S4 D-E) [DONE]

**Done record:** spec P1-S4 D-E delivered green by two implementing slices + one step-green slice (191 engine tests -> union engine scan 6 suites / 191 tests): P1-S4-D car lighting + car pass — `examples/neatRacing/engine/lighting.ts` + `passes/car.pass.ts` + `lighting.test.ts` (directional sun + hemisphere ambient + exp2 fog mirror, two-sided bins meshes, blob shadows at +0.01 m, draw-time team remap, four wheel sockets via G9 contract, warmup key 'car' index 4); P1-S4-E glass pass + bloom pipeline — `passes/glass.pass.ts` + `passes/bloom.pass.ts` + `bloom.pass.test.ts` (G4 glass blend state machine, black-never-emits-shine LD-23 rule, post-review winding fix, half-res bloom threshold/blur/additive, AC-035 off-switch, keys[6..8] warmup seam). Step green (batch 52-54 equivalent): `npx tsc -p tsconfig.neatRacing.json --noEmit` exit 0; `npm run jest:base -- --no-cache --testPathPatterns="examples/neatRacing/engine"` 6 suites / 191 tests exit 0; `npm run lint` exit 0 (0 errors, 5 pre-existing warnings in `examples/neatRacing/shared/network-visualization`); AC-040 constant parity verified — `NEATRACING_WARMED_PROGRAM_COUNT` is derived from the 9-key registry `['starfield','grid','pulse','track','car','glass','bloom-threshold','bloom-blur','bloom-additive']` and asserted within 7–9. Consolidated slice-advancement gate 7/7 FULL pass (plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review). Manual dev-server visual check of car team colors, glass highlights, and bloom intensity deferred to Step 08 dev server. Per tracker-handoff compression, the packet YAML, P1-S4-G2 validation record, batch summaries, and carried flags are archived at `plans/neatracing-voxel-racer.logs.md` (Step 07 record; no prose/stop-conditions blocks — same compression shape as the Step 08 record).

#### Step 08 — Dev server + editor core (spec P1-S5) [DONE]

**Done record:** spec P1-S5 delivered green by four implementing slices plus
the step-green lane (batches 56-61): P1-S5-A model store CRUD + validation
(`model-store.utils.ts`, 30 tests — AC-018 support + AC-007 round-trip),
P1-S5-B dev server with S1-S6 hardening + `dev:neatRacing` script
(33 tests incl. the CLI self-start contract after
fix-packet-P1-S5-B-iteration-2; convergence marker
`fix-loop: P1-S5-B iteration 2 status=passed.`), P1-S5-C editor shell page +
entry (17 tests; fix-packet-P1-S5-C-iteration-1 closed), P1-S5-D editor
camera controls (19 tests; AC-019 orbit/pan/zoom), and P1-S5-green step
green (batch 61): union lanes 2 suites / 63 tests (server) + 2 suites /
36 tests (editor) exit 0 via validation-MCP; declared-command manual check
`npm run dev:neatRacing` green on real boots (GET /, /editor/,
/editor/editor.bundle.js 200s); visible-window browser smoke green
(visible-foreground Chrome 154, WebGL2 canvases on /editor/ and /, 0 console
errors, 0 failed requests); consolidated slice-advancement 4/4 PASS
(TRIVIAL, md-only surface, item 226). AC-018 green; AC-019/AC-020 partials
are consumed by Steps 09-10. Per tracker-handoff compression, the packet YAML, P1-S5-green validation record,
batch summaries (green batches
57-61, items 214-226), and carried flags are archived at
`plans/neatracing-voxel-racer.logs.md` (Step 08 record; close span batches
56-61, items 209-226; batch-56 implementation items 209-213 stay itemized in
the tracker — no prose/stop-conditions blocks in this archive, matching the
Step 07 shape).

#### Step 09 — Editor features part 1 (spec P1-S6 A-C) [DONE]

**Done record:** spec P1-S6 A-C delivered green by three implementing slices + one step-green slice (evidence batches 64-70; RED->IMPLEMENT->GREEN per slice, union editor lane 5 suites / 72 tests): P1-S6-A palette + editor state (`editor.palette.ts` + `editor.state.ts`, 13 red contracts; NEATRACING_EDITOR_PALETTE_MAX_ENTRIES=128 with exact-match-dedup -> first-free-index -> deterministic nearest fallback (Manhattan RGB, lowest-index tie-break), NEATRACING_EDITOR_PALETTE_EMPTY_INDEX=0 sentinel `[0,0,0,0,0]`, setAlpha/setGlow 0-255 clamp; dense 128³ Uint8Array palette-index volume approx 2.1 MB (R-07 P7 editor exception) painted exact-else-nearest at layout `x + 128*y + 16384*z` (x col, y row 0 = down, z front slice); model dropdown state (snapshot().model null until selectModel)), P1-S6-B 3-step dense-snapshot undo (`editor.undo.ts`, 12 red contracts; createEditorUndo bound stack — push() detached `new Uint8Array(volume)` snapshot (check-before-edit convention, no aliasing), cap-3 oldest-eviction window, newest-first in-place restore (volume reference stays stable for renderers), bytesInUse() live accounting; NEATRACING_EDITOR_UNDO_BUDGET_BYTES = 3 x 2,097,152 = 6,291,456 bytes approx 6.3 MB, <= 8 MiB asserted; AC-036 partial; redo hooks are a later-slice addition out of P1-S6-B scope), P1-S6-C mirror symmetry (`editor.symmetry.ts`, 11 red contracts; x/y/z mirror modes under the normative spec §7.2 rule (coordinate c mirrored to coordinate 127 minus c; cols 0-63 mirror to cols 64-127), mirrorX default ON per LD-28, deterministic emission order (primary voxel first, then ascending subset composition x < y < xy < z < xz < yz < xyz), boundary coords 0/63/64/127, RangeError guard). Step green P1-S6-G1 (batch 70; live re-verify batch 71): allow-listed `npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"` 5 suites / 72 tests exit 0 + `npx tsc -p tsconfig.neatRacing.json --noEmit` clean via neataptic-validation-mcp; consolidated slice-advancement 4/4 PASS (plan-sync, step-packet, plan-slice-quality, plan-command-lint; TRIVIAL md-only surface); convergence markers `fix-loop: P1-S6-{A,B,C,G1} iteration 1 status=passed.` (zero fix loops). AC-020, AC-021, AC-036 (partial) green; AC-036 full memory accounting stays with Phase 3 Step 01 / P3-S1-B per the partial scope. Blocked branch ("picker math disagrees with camera controls") N/A — grid picker + part placement live in Step 10 (spec P1-S6 D-E, [PLANNED]). Per tracker-handoff compression, the packet YAML, P1-S6-G1 validation record, batch summaries (batches 64-71), and carried flags are archived at `plans/neatracing-voxel-racer.logs.md` (Step 09 record; no prose/stop-condition blocks — same compression shape as the Step 08 record; itemized batches 64-71 stay in the tracker's Latest validation evidence per the batches 25/34/41/51/56-61 itemized convention).

#### Step 10 — Editor features part 2 (spec P1-S6 D-E) [DONE]

**Done record:** spec P1-S6 D-E delivered green by two implementing slices + one
step-green slice (evidence batches 5/73-78; P1-S6-D RED landed at dispatch,
batch 5, P1-S6-E RED batch 74): P1-S6-D deterministic car generator + committed
assets (batch 76; zero-dependency `examples/neatRacing/scripts/generate-car-voxel.mjs`
— no RNG/clock, fixed key order + trailing newline; schema-valid committed
assets `car-f1.model.json` 240,278 bytes (128³, RLE half-rows sum 64) /
`wheel.model.json` 18,243 bytes (32³, RLE half-rows sum 16) under spec §10.2
names with the index-0 EMPTY sentinel `[0,0,0,0,0]` + idx-1 team-remap palette
semantics (NEATRACING_TEAM_REMAP_SLOT livery placeholder in the car,
duplicated-EMPTY team-remap opt-out in the wheel); LD-29 seam — bare run
targets the committed assets dir, `--out <dir>` never writes committed;
`voxel:generate:car` registered in package.json; AC-022 PASS) and P1-S6-E
editor save→rebuild pipeline (batch 75;
`examples/neatRacing/editor/editor.rebuild.pipeline.ts`; 11 red contracts →
11/11 green: 250 ms trailing debounce, latest-wins single-flight
(`supersededCount === 1`), dirty-slice remesh during drag clamped+dedup
[z-1,z,z+1] (0→[0,1], 127→[126,127]), pointer-up flush = one immediate full +
pending debounce consumed, durations recorded via the injected nowMs clock,
nearest-rank p95 = 290 ≤ NEATRACING_REBUILD_P95_BUDGET_MS=300 on the
19×290 + 1×350 ms fixture; AC-041 PASS). Step green P1-S6-G2 (batch 78)
re-verified all four packet lanes exit 0 — voxel jest 9 suites / 160 tests;
editor jest 6 suites / 83 tests (editor.rebuild.pipeline 11/11);
`voxel:generate:car` sha256 pre==post==batch-76 committed hashes (car
`492B7782…E8F86` 240,278 bytes, wheel `6FF0C655…EECA` 18,243 bytes);
`npx tsc -p tsconfig.neatRacing.json --noEmit` clean — after the batch-77
orchestrator shared-validation union PASS
(`artifacts/shared-validation-P1-S6-DE.json`, 2 suites / 14 tests + build 0 +
lint 0) + severity FULL/FULL + two Tier-3 specialist APPROVEs
(implementation-pattern-scout P1-S6-D; performance-trace-specialist P1-S6-E
with 4 non-blocking doc observations); consolidated slice-advancement 4/4
PASS; convergence markers `fix-loop: P1-S6-D iteration 1 status=passed.` +
`fix-loop: P1-S6-E iteration 1 status=passed.` + `iteration 2 status=passed.`
(lint fix-forwards; zero loop-back regressions). AC-022 + AC-041 both recorded
PASS; real-browser p95 stays deferred to Step 13 wiring (batches 77/78). Per
tracker-handoff compression, the packet YAML, the Red-contract design
P1-S6-D block, and carried flags are archived at
`plans/neatracing-voxel-racer.logs.md` (Step 10 record; no prose/stop-condition
blocks — same compression shape as the Step 09 record; itemized batches
5/73-78 stay in the tracker's Latest validation evidence per the
batches 25/34/41/51/56-61 itemized convention).

#### Step 11 — Track + physics + race part 1 (spec P1-S7 A-C) [DONE]

**Done record:** spec P1-S7 A-C delivered green by three implementing slices +
step-green P1-S7-G1 (evidence batches 80-88; RED -> implement -> verify ->
green per slice). P1-S7-A — `examples/neatRacing/track/track.slabs.ts`
(AC-017: named `NEATRACING_TRACK_SLAB_PITCH_M = 0.5` (TP-19), pure
`buildTrackSlabs({ seed, layoutVersion, sizeBucket })` -> `TrackSlabMesh
{ palette, quads }` via the recycled racing_curriculum closed-loop generator

- shared greedy mesher (LD-28), road/curb/grass slab palette (curb 1.0 m /
  grass apron 3.0 m bands, road > curb > grass priority), §7.3.1 world-meter
  quads with the EMPTY index-0 sentinel stripped, byte-identical same-tuple
  determinism, seed-43 negative differentiator, widest axis span >=
  NEATRACING_DRAW_DISTANCE_M = 320 (TP-16); no bank/boost-pad feature — the
  packet's bank/boost wording was spec-drift adjudicated out of scope at
  batch 80, spec wins). P1-S7-B —
  `examples/neatRacing/physics/physics.step.service.ts` (AC-023:
  deterministic 60 Hz fixed timestep 1/60 s; §5.2 constants
  MAX_SPEED_UNITS_PER_S = 89 (TP-01) with derived accel/braking/natural
  decel/off-road decel + speed floor, CENTRIFUGAL_MULTIPLIER 0.3 (TP-02),
  STEERING_RATE 2 (TP-04); `stepPhysics` over `PhysicsCarState { tick, carX,
carY, carHeading, forwardSpeedWorld, speedWorld, yawRate, slipAngle }` +
  `PhysicsStepInput { throttle, brake, steer }`; seeded
  `createInitialPhysicsCarState` (Math.imul avalanche); no Math.random;
  JSON.stringify byte-parity). P1-S7-C — `examples/neatRacing/host/input.ts`
- `examples/neatRacing/physics/collision.utils.ts` (AC-024 partial: D5 pure
  DOM-free `sampleKeyboardInput` (up -> throttle 1, down -> brake 1,
  left/right -> steer ±1) with the one-sample-per-rAF-frame replay across
  catch-up ticks; D6 NEATRACING_CAR_COLLISION_RADIUS_M = 1.0 with band
  mirrors curb 1.0 / grass apron 3.0; slab-mirroring road-priority
  `createTrackSurfaceResolver`; approaching-only equal-mass elastic
  `resolveCarCollisions`; outer-apron-edge `resolveTrackBoundaryCar` (bound =
  halfWidth + apron; the ~88% red-phase boundary interpretation pinned by
  the untouched-at-lat-5 assertion, confirmed by implementation batch 86)).
  Step green (batch 87): three lanes live via the validation-MCP relay
  (track 1 suite / 10 tests; physics 2 suites / 28 tests; neatRacing tsc
  exit 0); union 3 suites / 38 tests; consolidated slice-advancement 4/4
  PASS (TRIVIAL); orchestrator shared-validation artifacts
  coverage/shared-validation-p1s7a.json / -p1s7b.json / -p1s7c.json PASS +
  Tier-3 specialist APPROVE per FULL slice (batches 82/84/86); convergence
  markers `fix-loop: P1-S7-{A,B,C,G1} iteration 1 status=passed.` (zero fix
  loops). Full AC-024 closure (HUD, checkpoints, lap counting) stays with
  Step 12; deviations on record: batch-83 red subdispatch model mandate
  violation (batch 84 item 5), severity-gate CLI form `--input=` standing
  note (batch 84 item 2), cortex-index staleness flag CLOSED at batch 87
  item 8. Per tracker-handoff compression, the packet YAML (incl. the
  red-contract red_test bullets) and carried flags are archived at
  `plans/neatracing-voxel-racer.logs.md` (Step 11 record; no prose or
  stop-condition blocks — same compression shape as the Step 09/10 records;
  itemized batches 80-88 stay in this tracker's Latest validation evidence
  per the batches 25/34/41/51/56-61/73-78 itemized convention).

#### Step 12 — Track + physics + race part 2 (spec P1-S7 D-E) [DONE]

**Done record:** Step 12 [DONE] STEP-GREEN (batches 90-95): P1-S7-D race rules
(AC-024 partial — OQ-001 checkpoint budget 60 s + 25 s/lap extension, lap
counting at forward finish-line crossings, OQ-002 level advancement 1→2→3 cap,
terminal inertness; 21 contracts, physics lane 49/49) + P1-S7-E HUD + mode
selector (AC-024 — speed/lap/time formatters, cyan/magenta overlay tokens,
Neatenstein-pattern ai/manual selector, once-per-switch onSwitch restart
carrier pre-verifying AC-033; 16 contracts, host lane 32/32) + P1-S7-G2 union
green (5 suites / 81 tests; neatRacing tsc exit 0). Red phases confirmed
TS2307 right-reason module-absence failures (batches 90/93); shared-validation
artifacts coverage/shared-validation-p1s7d.json + p1s7e.json PASS; single
Tier-3 implementation-pattern-scout APPROVE per FULL slice; convergence
markers `fix-loop: P1-S7-{D,E,G2} iteration 1 status=passed.` (zero fix
loops). Manual full keyboard lap DEFERRED BY DESIGN to the Phase 1 exit check
after Step 13 boot-glue wiring (batch 13 item 61 obs 1; superseded by the
batch-102 Step 14 insertion, then re-anchored after the batch-105 Step 15
editing-loop insertion, patch #2 — the comprehensive two-page exit check
follows Step 15). Specialist
handoffs: Step 13 must adopt the slice-internal `neatracing-hud-output`
container id (specialist obs batch 94; the batch-102 patch moves the
adoption to Step 14 P1-boot-composition). Per tracker-handoff compression,
the packet YAML (incl. the
P1-S7-D/E red_test bullets) is archived verbatim at
`plans/neatracing-voxel-racer.logs.md` (Step 12 record; no prose or
stop-condition blocks — same compression shape as the Step 09/10/11
records; itemized batches 90-95 stay in this tracker's Latest validation
evidence per the batches 25/34/41/51/56-61/73-78/80-88 itemized
convention).

#### Step 13 — Boot-glue wiring: live render path + fallback append (batch 13 item 61 obs 1) [DONE]

```yaml
phase: 1
step: 13
title: 'Boot-glue wiring: live render path + fallback append (batch 13 item 61 obs 1)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 14 - Browser composition (frame composer + shell HUD adoption + 9-key warmup registry + live drive loop + editor preview, batch 102), then the Phase 1 exit check (manual full keyboard lap + deferred manual browser checks, Steps 03/06/07), then dispatch Phase 2'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-boot-glue'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
  - 'manual: npm run build:neatRacing then confirm the served page renders frame 1 with WebGL2 or shows the graceful fallback message without it'
acceptance_criteria:
  - 'AC-002 (completion): window.neatRacingStart wires the Step 02 boot modules - initRenderingContext (single init; graceful fallback element append when WebGL2 is unavailable), runShaderWarmup before frame 1, updateRendererSize letterbox apply at boot + resize, and an RAF loop driving advanceFrameSteps - so the first frame renders or the fallback message appears (batch 13 item 61 obs 1; spec AC-002)'
slices:
  - slice_id: 'P1-boot-glue'
    title: 'Wire boot path: context init + shader warmup + RAF advanceFrameSteps + letterbox apply + fallback append'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
      - 'examples/neatRacing/index.html (conditional: only if the fallback container needs a dedicated element beyond the existing #status shell)'
    acceptance_criteria:
      - 'AC-002 (completion): boot path order asserted - context init -> shader warmup -> size/letterbox apply -> RAF loop driving advanceFrameSteps; fallback element appended with NEATRACING_FALLBACK_MESSAGE when WebGL2 is unavailable (no throw, page stays usable)'
    red_test:
      - 'RED: browser-entry.test.ts wiring contract fails while the boot stub only mounts the canvas; spy doubles over context.service + render-loop + canvas-dimensions assert each wiring call and its order'
    parallelizable: false
    dependencies: []
  - slice_id: 'P1-boot-glue-green'
    title: 'Step green validation: boot-glue wiring + first-frame render proof'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 13 AC subset green: AC-002 wiring complete (first frame renders / graceful fallback message); DEMO/UI real visible-window browser smoke per section 9; closes the batch 13 item 61 obs 2 later-wiring checkpoint'
    parallelizable: false
    dependencies:
      - 'P1-boot-glue'
```

**User instruction:** Implement slice P1-boot-glue: wire `window.neatRacingStart` to the Step 02 boot modules so the first frame renders (or the graceful fallback message appears without WebGL2); red -> implement -> green.

**Step objective:** Close the batch 13 item 61 obs 1 gap: wire initRenderingContext, runShaderWarmup, advanceFrameSteps, updateRendererSize letterbox apply, and the fallback-element append into the boot path, making spec AC-002 observable and enabling the Phase 1 exit check (manual full keyboard lap). Composition of later pass/track/car render modules stays with their owning steps; this step owns only the boot wiring.

**Stop conditions:**

- **Done:** browser-entry wiring jest suite green; served `build:neatRacing` page renders frame 1 (or shows the graceful fallback message without WebGL2); AC-002 completion recorded; Step 14 browser composition is now [DONE] (automated validation + visible-window smoke green; user-gate assumed in this session because the user was unavailable to confirm); Step 14B (composition correctness pass: grid/track/car/wheels/physics fix-up) is [PLANNED] after Step 14 before the Phase 1 exit check; the comprehensive two-page Phase 1 exit check (game page full keyboard lap + editor page editing round-trip + deferred manual checks from Steps 03/06/07) runs after Step 14B + Step 15.
- **Blocked:** wiring reveals an API mismatch with the Step 02 modules — stop, record the mismatch in Latest validation evidence, escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"`
- manual: npm run build:neatRacing then confirm the served page renders frame 1 with WebGL2 or shows the graceful fallback message without it

#### Step 14 — Browser composition: frame composer + shell HUD adoption + 9-key warmup registry + live drive loop (batch 102 patch) [DONE]

```yaml
phase: 1
step: 14
title: 'Browser composition: frame composer + shell HUD adoption + 9-key warmup registry + live drive loop (batch 102 patch)'
status: '[DONE]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 14B - Composition correctness pass: grid/track/car/physics fix-up (user-reported gaps in AC-005, AC-015, AC-017, AC-022, AC-023, AC-024), then Step 15 - Editor editing-loop composition (tool-frame wiring + pointer-pick edit loop + palette/undo/model-dropdown/CAD-view controls + mirror-X symmetry + live-save, batch 105 patch, patch #2), then the Phase 1 exit check - comprehensive user manual retest on both pages (game page full keyboard lap + editor page editing round-trip + deferred manual browser checks, Steps 03/06/07), then Step 13 + Step 14 + Step 14B + Step 15 close, phase compression, and the Phase 2 dispatch (global steps 16-20)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-frame-composer'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'manual: npm run build:neatRacing + visible-window browser smoke on the served / and /editor/ pages (composed scene visible on both) ahead of the user exit-check retest'
acceptance_criteria:
  - 'AC-002 (extension): a host-owned frame composer draws the nine composed passes (starfield, grid, pulse, track, car, glass, bloom-threshold, bloom-blur, bloom-additive) into the scene frame buffer every tick in the pinned warmup-key order with frame-level GL state restored host-side, no lazy mid-race program compiles, and zero per-frame allocations on the hot path (batch 102 gap i; spec AC-002)'
  - 'AC-040 (completion): NEATRACING_BOOT_PROGRAM_REGISTRY is populated with the nine composite programs keyed by NEATRACING_WARMED_PROGRAM_KEYS and the boot warmup count matches NEATRACING_WARMED_PROGRAM_COUNT before frame 1 so no program compiles mid-race (batch 102 gap iii; the AC-040 seam in browser-entry.ts)'
  - 'AC-024 (completion, browser wiring): keyboard input drives the live race loop on the served game page - RAF tick accumulator -> sampleKeyboardInput -> stepPhysics at the 60 Hz fixed cadence -> race-rules budget/lap/level/terminal transitions -> hud.update + race mode selector refresh - with the HUD and mode selector mounted at boot into the reserved neatracing-hud-output container (batch 94 handoff adoption, gap ii); the manual full keyboard lap stays the user-gated Phase 1 exit check (batch 102 gap iv; spec AC-024)'
  - 'AC-019 (completion, preview lane): the editor page renders the selected model preview (decode -> greedy mesher -> bins draw) RAF-driven under the Step 08 orbit/pan/zoom camera, composing ONLY the model preview per the recorded plan-owner editor-shell decision (batch 102 gap v; spec AC-019, section 8.3 dogfooding)'
slices:
  - slice_id: 'P1-frame-composer'
    title: 'Host frame composer: per-tick ordered nine-pass draw into the scene frame buffer'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/host/frame-composer.ts'
      - 'examples/neatRacing/host/frame-composer.test.ts'
    acceptance_criteria:
      - 'AC-002 (extension): createFrameComposer draws the nine passes per tick in the pinned warmup-key order (starfield -> grid -> pulse -> track -> car -> glass -> bloom-threshold -> bloom-blur -> bloom-additive) against the host-owned scene frame buffer, restores frame-level GL state host-side (batch 54 carried), keeps zero per-frame allocations on the hot path, and honors the AC-035 zero-op off-switch'
    red_test:
      - 'RED: frame-composer.test.ts fails with TS2307 while examples/neatRacing/host/frame-composer.ts is absent (batch 90/93 right-reason precedent); spy doubles assert the call order, exactly one draw per pass per tick, the bloom half-res chain (threshold -> blur -> additive), and zero allocations'
    parallelizable: true
    dependencies:
      - 'P1-boot-glue'
  - slice_id: 'P1-boot-composition'
    title: 'Boot composition: populate the 9-key warmup registry + mount the HUD and race mode selector into the reserved container'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
      - 'examples/neatRacing/index.html (conditional: reserve the neatracing-hud-output host container per the batch 94 handoff)'
    acceptance_criteria:
      - 'AC-040 (completion): NEATRACING_BOOT_PROGRAM_REGISTRY keys equal NEATRACING_WARMED_PROGRAM_KEYS and the boot warmup count matches NEATRACING_WARMED_PROGRAM_COUNT before frame 1 (nine-program parity)'
      - 'AC-024 (partial): createHud and createRaceModeSelector mount into the reserved neatracing-hud-output container at boot, honoring the resolve-or-throw host contract (batch 94 handoff)'
    red_test:
      - 'RED: browser-entry.test.ts fails while NEATRACING_BOOT_PROGRAM_REGISTRY is still the empty readonly never[] and index.html lacks the neatracing-hud-output container; asserts the registry keys equal NEATRACING_WARMED_PROGRAM_KEYS, the count parity vs NEATRACING_WARMED_PROGRAM_COUNT, and the createHud/createRaceModeSelector mount calls with the reserved id'
    parallelizable: true
    dependencies:
      - 'P1-frame-composer'
  - slice_id: 'P1-drive-loop-live'
    title: 'Live drive loop: keyboard input -> 60 Hz fixed-step physics -> race rules -> HUD and mode selector, wired to the composer'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/host/race-session.ts'
      - 'examples/neatRacing/host/race-session.test.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - 'AC-024 (completion, browser wiring): each RAF tick accumulates elapsed time and drives sampleKeyboardInput -> stepPhysics at the 60 Hz cadence (catch-up ticks included) -> race-rules budget/lap/level/terminal transitions -> hud.update + mode-selector refresh live on the served game page; the onSwitch hook wires the AC-033 restart carrier pre-verify; the frame composer is invoked per frame after the simulation accumulation; zero per-frame allocations'
    red_test:
      - 'RED: race-session.test.ts keyboard-event spy asserts the 60 Hz cadence across catch-up ticks, the race-rules transition calls (budget/lap/level/terminal), hud receiving updates only after the race transitions, and the composer invoked in frame order'
    parallelizable: false
    dependencies:
      - 'P1-boot-composition'
  - slice_id: 'P1-editor-preview'
    title: 'Editor preview lane: RAF-driven model preview render loop under the Step 08 camera (editor-shell decision in-step)'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor-entry.ts'
      - 'examples/neatRacing/editor/editor-preview.render.ts'
      - 'examples/neatRacing/editor/editor-preview.render.test.ts'
    acceptance_criteria:
      - 'AC-019 (completion, preview lane): the editor page runs a RAF loop rendering the selected model (decode -> greedy mesher -> bins draw) under the Step 08 orbit/pan/zoom camera, reusing the game decoder + mesher + renderer per spec section 8.3 dogfooding; recorded plan-owner decision: the editor page composes ONLY the model preview (no game-scene race passes), materializing the never-materialized P1-S5-C later-slices render-loop promise (deferral to Phase 2/3 rejected as a would-be planning defect)'
    red_test:
      - 'RED: editor-preview.render.test.ts fails before the module exists; asserts the RAF-driven decode -> mesher -> draw chain and the editor camera transform feeding the preview draw'
    parallelizable: true
    dependencies:
      - 'P1-S5-D'
  - slice_id: 'P1-composition-green'
    title: 'Step green validation: composition + warmup registry + live drive loop + editor preview'
    status: '[DONE]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 14 AC subset green: AC-002 composer + AC-040 nine-program warmup parity + AC-024 browser live wiring + AC-019 preview lane; union jest lanes (browser-entry + host + editor patterns) and neatRacing tsc clean; DEMO/UI real visible-window browser smoke green on BOTH served pages (sky composite, track, car, HUD overlay, editor model preview observable) clearing the batch 102 observation ahead of the user exit-check retest'
    parallelizable: false
    dependencies:
      - 'P1-frame-composer'
      - 'P1-boot-composition'
      - 'P1-drive-loop-live'
      - 'P1-editor-preview'
```

**User instruction:** Implement Step 14 per the batch 102 integration-gap classification: the frame composer drawing the nine delivered passes into the RAF loop, the 9-key boot warmup registry populating `NEATRACING_BOOT_PROGRAM_REGISTRY`, the `neatracing-hud-output` container adoption (batch 94 handoff), the live input -> physics -> race-state -> HUD drive loop, and the dogfooded editor model preview; red -> implement -> green per slice.

**Step objective:** Close the batch 102 planning defect: the served game page composes the nine passes per frame, warms the nine programs at boot into `NEATRACING_BOOT_PROGRAM_REGISTRY`, mounts the HUD + race mode selector into the reserved `neatracing-hud-output` container, and drives live keyboard input through the 60 Hz fixed-step physics into race rules and HUD updates; the editor page renders the RAF-driven model preview under the Step 08 camera. The user-gated Phase 1 exit check (manual full keyboard lap + deferred manual browser checks from Steps 03/06/07) follows Step 15 (batch-105 patch, patch #2 — the inserted editor editing-loop step supersedes the earlier after-this-step boundary; P1-editor-preview stays the preview-only lane).

**Editor-shell composition decision (batch 102 item (v), plan-owner, recorded):** P1-editor-preview keeps the editor preview IN-STEP in Phase 1: a RAF-driven render loop (decode -> greedy mesher -> bins draw) with the Step 08 orbit/pan/zoom camera, reusing the game decoder + mesher + renderer per spec section 8.3 dogfooding (AC-019 preview lane). The editor page composes ONLY the model preview — no game-scene race passes on the editor page. Deferring editor rendering to Phase 2 or Phase 3 is adjudicated a would-be planning defect: neither phase owns editor rendering, so the deferral would recreate the batch 102 gap at the Phase 1 exit check. Per the batch-105 patch (patch #2, batch 107), P1-editor-preview stays PREVIEW-ONLY and the Step 14 slice boundary is unchanged: the editing-loop composition (editor tool-frame wiring, pointer-pick edit loop through the rebuild pipeline, palette/undo/model-dropdown/CAD-view control wiring, mirror-X symmetry + live-save through the dev-server model-store API) is the inserted Step 15's scope, and no game-scene race passes are added to the editor page.

**Stop conditions:**

- **Done:** composition jest lanes green (`browser-entry`, `host`, `editor` testPathPatterns) and `npx tsc -p tsconfig.neatRacing.json --noEmit` clean; the served `/` game page composes the nine passes per frame with the live drive loop and `/editor/` renders the model preview; DEMO/UI visible-window browser smoke green on both pages. Step 14 closed on automated evidence; the comprehensive user-gated Phase 1 exit check (game page full keyboard lap + editor page editing round-trip + deferred manual checks from Steps 03/06/07) runs after Step 14B + Step 15; then Step 13 + Step 14 + Step 14B + Step 15 close and Phase 1 phase compression runs.
- **Blocked:** a pass-module API mismatch or a missing composition seam not closable inside files_to_change — stop, record in Latest validation evidence, escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/host"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`
- manual: npm run build:neatRacing + visible-window browser smoke on the served / and /editor/ pages (composed scene visible on both) ahead of the user exit-check retest

#### Step 14B — Composition correctness pass: grid flat horizon + visible track + F1 car shape + four wheels + drivable physics (batch 108 user-gap patch) [WIP]

```yaml
phase: 1
step: 14B
title: 'Composition correctness pass: grid flat horizon + visible track + F1 car shape + four wheels + drivable physics (batch 108 user-gap patch; rounds 1-3 fixes applied: flat grid + ribbon + red body + ai boot race + axis-1 transposition + world handedness + wheel spin cap; awaiting the final user visual gate)'
status: '[WIP]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'USER GATE FIRST: ask the user to confirm on the served / page (F1 car nose-forward, solid four wheels, no strip artifacts, car rotates with the chase camera, A = left / D = right, race runs continuously). On confirmation mark P1-correctness-green [DONE] and close Step 14B, then Step 15 - Editor editing-loop composition (tool-frame wiring + pointer-pick edit loop + palette/undo/model-dropdown/CAD-view controls + mirror-X symmetry + live-save, batch 105 patch, patch #2), then the Phase 1 exit check - comprehensive user manual retest on both pages (game page full keyboard lap + editor page editing round-trip + deferred manual browser checks, Steps 03/06/07), then Step 13 + Step 14 + Step 14B + Step 15 close, phase compression, and the Phase 2 dispatch (global steps 16-20)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-ground-composition'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="grid.pass|track.pass|track.slabs"'
  - 'npm run jest:base -- --testPathPatterns="car.pass|materials|generate-car"'
  - 'npm run jest:base -- --testPathPatterns="engine/lighting"'
  - 'npm run jest:base -- --testPathPatterns="wheel.mount"'
  - 'npm run jest:base -- --testPathPatterns="physics|input|race-session"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry|examples/neatRacing/host"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'manual: npm run build:neatRacing + visible-window browser smoke on the served / page: grid sits flat at the horizon, track ribbon is visible on the ground, the F1-style car shows four wheels and recognizable body color (not all black), and keyboard input makes the car move/drive; record screenshot/readback evidence'
acceptance_criteria:
  - 'AC-005 (completion): the ground grid is world-space flat, horizon-aligned, and world-anchored (not camera-relative or tilted), with stable pitch/direction as the player moves; grid passes its fragment-shader fwidth math tests and visible-browser check'
  - 'AC-017 (completion): the track ribbon snaps to the 0.5 m coarse voxel slab grid, is greedy-meshed, extends across NEATRACING_DRAW_DISTANCE_M, and is plainly visible on the served game page (not missing behind the grid)'
  - 'AC-022 (completion): npm run voxel:generate:car emits an F1-style car model + wheel model that pass schema validation, produce non-empty meshes, and render with body color/material contrast so the car is recognizable and not all black'
  - 'AC-015 (completion): four wheel models are mounted at the car hub positions (pivot at disc geometric center, hub height = wheel radius, contact patch at y = 0), spin axis lateral, and spin capped at NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S'
  - 'AC-013 / AC-016 (completion): car materials apply lighting (sun + hemisphere + fog) and glass/glow semantics correctly; black voxels never glow, glass pass tinted, blob shadows under cars visible'
  - 'AC-023 (completion): the physics step service implements the §5.2 arcade constants at 60 Hz fixed timestep and advances car speed/yawRate/slipAngle deterministically'
  - 'AC-024 (completion): keyboard steer/throttle/brake drive the car end to end through the 60 Hz fixed-step loop; the car visibly moves on the served page; HUD updates speed/lap/time; mode selector present'
slices:
  - slice_id: 'P1-ground-composition'
    title: 'Ground composition: flat horizon grid + visible track ribbon'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/passes/grid.pass.ts'
      - 'examples/neatRacing/engine/passes/track.pass.ts'
      - 'examples/neatRacing/track/track.slabs.ts'
    acceptance_criteria:
      - 'AC-005 (completion): grid pass draws a world-space, horizon-aligned ground plane with stable pitch/direction; the grid is not tilted or camera-relative; fragment-shader fwidth line-width math correct'
      - 'AC-017 (completion): track slabs generate a closed-loop ribbon on the 0.5 m slab grid, track.pass draws it into the scene frame buffer, and it is visible behind/on the ground in the served page'
    red_test:
      - 'RED: grid.pass.test.ts / track.pass.test.ts / track.slabs.test.ts fail with the expected right-reason assertions: grid transform places lines flat at the horizon with no tilt, track slab mesh is non-empty and deterministic, both passes are called in the composer order'
    parallelizable: true
    dependencies:
      - 'P1-composition-green'
  - slice_id: 'P1-car-shape'
    title: 'Car shape + materials: F1-style voxel body, team color contrast, glass/glow semantics'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/scripts/generate-car-voxel.mjs'
      - 'examples/neatRacing/engine/passes/car.pass.ts'
      - 'examples/neatRacing/voxel/materials.ts'
    acceptance_criteria:
      - 'AC-022 (completion): generated car model is F1-shaped (nose, cockpit, rear wing, body sides), passes schema validation, produces non-empty greedy mesh'
      - 'AC-013 / AC-016 (partial): car body renders with team-color remap, glass windows go through glass pass, black voxels never emit glow, blob shadows drawn under the car'
    red_test:
      - 'RED: generate-car-voxel + car.pass tests fail with TS2307 or missing fixture; asserts the generated model palette has body/glass/wheel colors, the car mesh is non-empty, and the car pass sets the correct GL state (depth-test, cull) before drawing'
    parallelizable: true
    dependencies:
      - 'P1-composition-green'
  - slice_id: 'P1-wheel-mount'
    title: 'Four-wheel mount: hub positioning, lateral spin axis, spin cap'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/voxel/wheel.mount.utils.ts'
      - 'examples/neatRacing/assets/models/wheel.model.json'
    acceptance_criteria:
      - 'AC-015 (completion): four wheel instances are computed for the car (front-left, front-right, rear-left, rear-right), pivot at disc center, hub height = wheel radius, contact patch y = 0, spin axis lateral after 90° mount, spin quantized/capped at NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S'
    red_test:
      - 'RED: wheel.mount.utils.test.ts fails while the four-hub transform table is missing or while wheel.model.json does not decode to a wheel-shaped mesh; asserts the 4 hub matrices, the lateral spin axis, and the rev/s cap'
    parallelizable: true
    dependencies:
      - 'P1-car-shape'
  - slice_id: 'P1-physics-drive'
    title: 'Drivable physics: 60 Hz arcade step + keyboard input + race-session integration'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/physics/physics.step.service.ts'
      - 'examples/neatRacing/host/input.ts'
      - 'examples/neatRacing/host/race-session.ts'
    acceptance_criteria:
      - 'AC-023 (completion): physics.step.service advances speed, yawRate, slipAngle with the §5.2 constants at the 60 Hz fixed timestep; deterministic for same seed + inputs'
      - 'AC-024 (completion): each rAF frame samples keyboard input and applies it to every catch-up physics tick; the car position/heading update visibly; HUD receives speed/lap/time updates; mode selector present'
    red_test:
      - 'RED: physics.step.service.test.ts / input.test.ts / race-session.test.ts fail with the expected right-reason assertions: keyboard throttle applied across catch-up ticks, speed increases with throttle, car position changes over time, HUD updated after transitions'
    parallelizable: false
    dependencies:
      - 'P1-ground-composition'
      - 'P1-car-shape'
      - 'P1-wheel-mount'
  - slice_id: 'P1-ai-boot-race'
    title: 'Round 2 fix loop: AI autopilot boot race + red team body + lap-detector axis fix'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/physics/ai.autopilot.ts'
      - 'examples/neatRacing/physics/ai.autopilot.fixtures.ts'
      - 'examples/neatRacing/physics/ai.autopilot.test.ts'
      - 'examples/neatRacing/host/race-session.ts'
      - 'examples/neatRacing/host/mode-selector.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/host/race-session.test.ts'
      - 'examples/neatRacing/host/mode-selector.test.ts'
    acceptance_criteria:
      - 'AC-024 (completion): the served page boots as a RUNNING race — race-session samples stepAutopilotInput(car, trackSpec.splineSamples) while mode = ai (INITIAL_RACE_MODE = MODE_OPTION_AI); manual stays selectable and its switch restarts the race (AC-033 carrier); HUD speed/lap/time advance live'
      - 'AC-014 (completion): carPoseBag remaps NEATRACING_TEAM_ID_RED (slot-1 livery) at draw time so the F1 chassis reads against the cyan ground (was hardcoded cyan = invisible against the sky/starfield)'
      - 'Lap counter: resolveForwardLapLineCrossing = along-track dot sign-flip gated by a 5 m lateral disc + 0.5 m along dead-zone with a latched baseline (no spurious flips at the spawn point, which lies ON segment 0 chord)'
    red_test:
      - 'RED (round 2): ai.autopilot 6 contracts + race-session mode-branch tests + mode-selector initial-mode pin failed before implementation; visible page parked at 0 km/h in manual. Evidence: logs archive batch 137.'
    parallelizable: false
    dependencies:
      - 'P1-physics-drive'
  - slice_id: 'P1-correctness-fixup'
    title: 'Round 3 fix loop: axis-1 expansion transposition + world handedness + wheel spin cap + track mirror'
    status: '[DONE]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/passes/car.pass.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/track/track.slabs.ts'
      - 'examples/neatRacing/voxel/wheel.mount.utils.ts'
      - 'examples/neatRacing/engine/lighting.test.ts'
      - 'examples/neatRacing/track/track.slabs.test.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
      - 'examples/neatRacing/voxel/wheel.mount.utils.test.ts'
    acceptance_criteria:
      - 'AC-013/016 (wrap): car mesh expansion follows the shared greedy-mesher tangent rule on all axes — quadCornerGridX/quadCornerGridZ return position[0]+u / position[2]+v for axis 1 (was transposed: every non-square y-normal quad stretched to swapped extents = center/nose strips + see-through tires); engine/lighting pins the non-square case'
      - 'AC-005/017 (wrap): scene handedness unit-pinned in browser-entry — carSceneWorldZ = -topDownY (+0 normalized), carSceneYawRad = PI/2 + heading (NEATRACING_CAR_SCENE_YAW_OFFSET_RAD), composer sceneForwardZ = -sin(heading), camera eye/lookAt/carPoseBag all mirrored; track.slabs paints/mirrors the ribbon (pointZ = -worldZ, mirrored lattice span) so ribbon + car + camera share one frame; physics/lap/autopilot/surface stay top-down untouched'
      - 'AC-015 (wrap): visual spin = computeVisualWheelSpinDeltaRad (true omega v/r clamped to the 2 rev/s cap) accumulated per frame with session-swap + tick-wind-back reset — smoke shows solid tires, no strip artifacts, car rotates with the chase camera, A = screen-left / D = screen-right'
    red_test:
      - 'RED (round 3): engine/lighting axis-1 non-square fixture + track.slabs mirror test + browser-entry handedness TS2305 + wheel.mount spin-delta TS2305 — all four confirmed failing for the right reasons before the fixes. Evidence: logs archive batch 138.'
    parallelizable: false
    dependencies:
      - 'P1-ai-boot-race'
  - slice_id: 'P1-correctness-green'
    title: 'Step 14B green validation: FINAL USER VISUAL GATE (round 3 evidence complete)'
    status: '[WIP]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Union lane green (rounds 1-3 evidence all recorded): AC-005 grid + AC-017 track + AC-022 car shape + AC-015 wheels + AC-013/016 materials + AC-023 physics + AC-024 driving; jest lanes grid.pass|track.pass|track.slabs 42/42, car.pass|materials|generate-car 17/17, wheel.mount 40/40, physics|input|race-session 108/108, examples/neatRacing/browser-entry|host 118/118, engine/lighting 45/45; npx tsc -p tsconfig.neatRacing.json --noEmit clean; visible smoke green (flat horizon at pixel row 433, visible ribbon, red F1 body, HUD moving, zero page errors)'
      - 'USER GATE (the only remaining work): the user confirms on the served / page — F1 car nose-forward, solid four wheels, no strip artifacts at center/nose/wheels, car rotates with the chase camera on turn-in, A = left / D = right, race runs continuously — THEN mark this slice [DONE] and close Step 14B; never flip a user-gated status without user confirmation (standing do-not-self-close rule)'
    parallelizable: false
    dependencies:
      - 'P1-ground-composition'
      - 'P1-car-shape'
      - 'P1-wheel-mount'
      - 'P1-physics-drive'
      - 'P1-ai-boot-race'
      - 'P1-correctness-fixup'
```

**User instruction:** Implement Step 14B to close the user-reported composition gaps observed on the served `/` page: the grid must be flat at the horizon (not tilted), the track ribbon must be visible, the car must look like an F1 car with four wheels and body-color contrast (not all black), and keyboard input must drive the car. red -> implement -> green per slice.

**Current status (end of round 3, 2026-10-06; tidy pass 2026-10-11):** Rounds 1-3 are
implemented and every validation lane is green: grid.pass|track.pass|track.slabs 42/42,
car.pass|materials|generate-car 17/17, wheel.mount 40/40, physics|input|race-session
108/108, examples/neatRacing/browser-entry|host 118/118, engine/lighting 45/45,
`npx tsc -p tsconfig.neatRacing.json --noEmit` clean. The dev server was restarted after
the edits and the puppeteer visible-window smoke passed (boot race accelerating,
manual drive 0 -> 160 km/h, A/D distinct frames, flat horizon at pixel row 433 across
the width, red body + road ribbon present, zero page errors). Remaining work = the
P1-correctness-green user gate ONLY (see its AC). After the user confirms: mark
P1-correctness-green [DONE], close Step 14B, then Step 15. Do not re-run lanes or touch
source before that gate.

**Frame seam map (all pinned by tests; do not re-litigate):**

- Shared greedy mesher tangent rule `FACE_TANGENT_AXES` = {0:[1,2], 1:[0,2], 2:[0,1]};
  the mesher records `position[axis]` = plane, `position[widthAxis]` = u,
  `position[heightAxis]` = v; quad corner expansion order 00/10/11/01 (corner index 2 is
  the (u,v) DIAGONAL corner; triangle order 0,1,2,0,2,3 makes vertex 11 the pure (0,1)
  corner). car.pass / glass.pass / track.pass corner getters and
  expandVoxelBinModelToCarMesh all follow it — NON-SQUARE quads are the transposition
  detector (the round-3 artifact root cause).
- Physics is top-down math: y = ground plane, heading CCW, velocity = heading +
  slipAngle, spawn heading 0 along +x. The visual scene mirrors ground y -> world z = -y.
- browser-entry.ts mapping: `carSceneWorldZ(topDownY) = -topDownY` (+0 normalized),
  `carSceneYawRad(h) = PI/2 + h` (`NEATRACING_CAR_SCENE_YAW_OFFSET_RAD = PI/2`);
  buildComposerFrame uses `sceneForwardZ = -sin(heading)` for camera eye/lookAt and fills
  carPoseBag.worldZ/yawRad + cameraPose.yawRadians from the mirrored frame. Positive
  steer tips the nose toward -z = screen LEFT (A = left, D = right); the car yaws in
  lockstep with the chase camera.
- Track ribbon mirrors identically: track.slabs paintTrackSlabSlices queries the
  generator at pointZ = -worldZ and synthesizeTrackSlabLattice mirrors the span;
  physics, lap detector, autopilot, and the surface resolver consume trackSpec.segments
  top-down — untouched.
- Wheels: mount pivot at disc center, hub height = radius, contact patch y = 0; visual
  spin = computeVisualWheelSpinDeltaRad(deltaTicks, dt, speed, radius) — true omega v/r
  clamped to NEATRACING_MAX_VISUAL_WHEEL_REV_PER_S — accumulated in browser-entry with
  session-swap + tick-wind-back reset (spin must never rewind on restart).
- Camera: lookAtInto is algebraically correct; do NOT flip its forward row
  (batch 136 probe, rolled back).

**Dev / validation lifecycle for the next agent:**

- esbuild bundles at server STARTUP ONLY: after ANY served-page source edit, stop the
  dev server, relaunch `npm run dev:neatRacing` (async), and poll http://127.0.0.1:8787/
  for HTTP 200 before validating.
- Visible-window smoke recipe (write tmp__.cjs probes, DELETE them after — tmp__ count
  must end at 0): puppeteer launch `headless: false` with args
  ['--no-sandbox','--use-gl=angle','--enable-unsafe-webgpu'], viewport 1280x1024; HUD
  read `#neatracing-hud-output` innerText (km/h | Lap N/20 | timer | WASD/ARROWS |
  ai/manual); mode toggle via page.select('#neatracing-race-mode-selector', 'manual');
  the page must boot as a RUNNING ai race (`INITIAL_RACE_MODE = 'ai'`); the favicon 404
  is known-harmless.
- Standing rules: NEVER run any git command (git is uninstalled). Do not flip user-gated
  slice/step statuses without explicit user confirmation. Batch MCP gate calls only once
  there is something concrete to validate (user mandate).
- Blocked path: a missing pass-module API or a car/track data seam not closable inside a
  slice's files_to_change — stop, record in the logs archive, escalate via
  00.cross-tier-helper.
- New evidence batches append to the logs file under the moved-archive heading (see the
  pointer after EVIDENCE-APPEND-POINT), not to this step body.

#### Step 15 — Editor editing-loop composition: tool-frame wiring + pointer-pick edit loop + palette/undo/model-dropdown/CAD-view controls + mirror-X symmetry + live-save (batch 105 patch, patch #2) [PLANNED]

```yaml
phase: 1
step: 15
title: 'Editor editing-loop composition: tool-frame wiring + pointer-pick edit loop + palette/undo/model-dropdown/CAD-view controls + mirror-X symmetry + live-save (batch 105 patch, patch #2)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 14B - Composition correctness pass: grid/track/car/physics fix-up (user-reported gaps in AC-005, AC-015, AC-017, AC-022, AC-023, AC-024) is executed BEFORE this step; Phase 1 exit check - comprehensive user manual retest on both pages (game page full keyboard lap + editor page editing round-trip per AC-021, Steps 03/06/07 deferred manual browser checks), then Step 13 + Step 14 + Step 14B + Step 15 close, phase compression, and the Phase 2 dispatch (global steps 16-20)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P1-editor-toolframe'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/server"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'manual: npm run build:neatRacing + visible-window browser smoke on the served /editor/ page (tool-frame controls visible; pointer-pick edit with visible mirror-X symmetry; palette, undo/redo, model dropdown, CAD front/side/top toggles observable; edits live-saved, page reload shows the persisted model) ahead of the user exit-check retest'
acceptance_criteria:
  - 'AC-020 (extension, browser wiring): the editor page mounts the editing tool frame - the createEditorState working-volume container and the palette list, undo/redo buttons, model dropdown (LD-28), CAD-view buttons, and symmetry toggle into the reserved neatracing-editor-tools container - and wires palette entry add/select (createEditorPalette under NEATRACING_EDITOR_PALETTE_MAX_ENTRIES), the 3-step bounded undo/redo stack (createEditorUndo, oldest evicted beyond the cap), and the model dropdown switching the edited model via editor.state.selectModel, observable on the served page (batch 105 scope correction, patch #2; spec AC-020)'
  - 'AC-021 (extension, browser wiring): pointer picks on the editor canvas paint the working volume through createEditorSymmetry canonicalized paint with mirror-X ON by default (NEATRACING_EDITOR_DEFAULT_MIRROR_X), edits flow through the save-to-rebuild pipeline (createEditorRebuildPipeline edit debounce + pointerUp flush; the AC-041 p95 budget stays with the pipeline), every edit saves immediately through the dev-server model-store API (store writeVoxelModelDocument behind PUT /api/models/:name with the RebuildCoordinator rebuild flight; no import/export UI per LD-28), and a reload shows the saved model unchanged (the AC-021 manual edit -> refresh -> persisted browser check; batch 105 scope correction, patch #2)'
  - 'AC-019 (extension, CAD views): the editor tool frame wires the CAD orthographic front/side/top buttons to the Step 08 editor camera setOrthoView(front/side/top) and the orbit path back to setPerspectiveView on the served page (spec AC-019)'
slices:
  - slice_id: 'P1-editor-toolframe'
    title: 'Editor tool-frame boot wiring: state container + palette/undo/model-dropdown/CAD/symmetry control mounts + initial model load'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.toolframe.ts'
      - 'examples/neatRacing/editor/editor.toolframe.test.ts'
      - 'examples/neatRacing/editor/editor-entry.ts'
    acceptance_criteria:
      - 'AC-020 (partial, tool-frame boot): createEditorToolframe extends the window.neatRacingEditorStart session - createEditorState + createEditorPalette volume/palette wiring, the palette list, undo/redo, model dropdown, CAD-view buttons, and symmetry toggle mounted into the reserved neatracing-editor-tools container (resolve-or-throw per the batch-94 handoff contract), and the initial selected model loaded from the model store (list + read + decode into the working volume)'
    red_test:
      - 'RED: editor.toolframe.test.ts fails with TS2307 while examples/neatRacing/editor/editor.toolframe.ts is absent (batch 90/93 right-reason precedent); asserts the editor-state binding, the control mounts under the reserved neatracing-editor-tools id, and the initial model-volume load'
    parallelizable: false
    dependencies:
      - 'P1-editor-preview'
  - slice_id: 'P1-editor-editloop'
    title: 'Pointer-pick edit loop: canvas pointer events -> symmetry-canonicalized paint -> undo push -> rebuild pipeline edit -> RAF preview redraw'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.edit.loop.ts'
      - 'examples/neatRacing/editor/editor.edit.loop.test.ts'
      - 'examples/neatRacing/editor/editor-entry.ts'
    acceptance_criteria:
      - 'AC-021 (partial, edit loop): canvas pointerdown/pointermove pick -> editor.state.paint through createEditorSymmetry canonicalized paint with mirror-X default ON (NEATRACING_EDITOR_DEFAULT_MIRROR_X), editor.undo push captured at edit boundaries, createEditorRebuildPipeline edit(sliceZ) trailing debounce (NEATRACING_REBUILD_DEBOUNCE_MS) with pointerUp full flush, and the Step 14 preview lane redrawing the edited volume per frame'
    red_test:
      - 'RED: editor.edit.loop.test.ts fails with TS2307 while examples/neatRacing/editor/editor.edit.loop.ts is absent; spy doubles assert the pick -> symmetry-canonicalized paint -> undo push -> pipeline edit(sliceZ) -> preview-redraw chain and the pointer-up full flush'
    parallelizable: false
    dependencies:
      - 'P1-editor-toolframe'
  - slice_id: 'P1-editor-controls'
    title: 'Palette/undo/model-dropdown/CAD-view control wiring on the served editor page'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.controls.ts'
      - 'examples/neatRacing/editor/editor.controls.test.ts'
      - 'examples/neatRacing/editor/editor-entry.ts'
    acceptance_criteria:
      - 'AC-020 (completion, browser wiring): palette add/select handlers (createEditorPalette add/resolve under NEATRACING_EDITOR_PALETTE_MAX_ENTRIES), undo/redo buttons bound to the 3-step bounded editor undo stack (createEditorUndo push/undo seam, oldest snapshot evicted beyond the cap; the redo path re-applies through the same snapshot mechanism), the model dropdown change -> editor.state.selectModel + model-store read/decode reload, and the CAD-view buttons -> createEditorCamera setOrthoView(front/side/top) + setPerspectiveView on the served page'
    red_test:
      - 'RED: editor.controls.test.ts fails with TS2307 while examples/neatRacing/editor/editor.controls.ts is absent; asserts the palette cap behavior at NEATRACING_EDITOR_PALETTE_MAX_ENTRIES, the undo/redo click contracts with oldest eviction, the selectModel + working-volume reload, and the CAD view toggles'
    parallelizable: false
    dependencies:
      - 'P1-editor-editloop'
  - slice_id: 'P1-editor-symmetry-save'
    title: 'Mirror-X symmetry toggle + immediate live-save through the dev-server model-store API'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/editor/editor.live.save.ts'
      - 'examples/neatRacing/editor/editor.live.save.test.ts'
      - 'examples/neatRacing/editor/editor-entry.ts'
    acceptance_criteria:
      - 'AC-021 (completion, browser wiring): the symmetry toggle defaults ON for X (NEATRACING_EDITOR_DEFAULT_MIRROR_X) with the canonicalized paint path applied to every edit, every edit enqueues the immediate save through the dev-server model-store API (PUT /api/models/:name via the model store + RebuildCoordinator rebuild flight; no import/export UI per LD-28), the save payload round-trips the edited volume under NEATRACING_MAX_SAVE_BODY_BYTES, and a page reload shows the saved model unchanged (the AC-021 manual edit -> refresh -> persisted check, manual browser lane)'
    red_test:
      - 'RED: editor.live.save.test.ts fails with TS2307 while examples/neatRacing/editor/editor.live.save.ts is absent; asserts the immediate save enqueue per edit, the model-store PUT payload round-trip, and the restored volume read back after a reload'
    parallelizable: false
    dependencies:
      - 'P1-editor-controls'
  - slice_id: 'P1-editor-loop-green'
    title: 'Step green validation: editing loop + controls + symmetry/live-save'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 15 AC subset green: AC-020 palette/undo/model-dropdown browser wiring + AC-021 edit loop + mirror-X symmetry live-save (manual edit -> refresh -> persisted) + AC-019 CAD views; union jest lanes (editor + server testPathPatterns) and neatRacing tsc clean; DEMO/UI visible-window browser smoke green on the served /editor/ page (tool frame, symmetric paint, undo/redo, model dropdown, CAD front/side/top, live-save + reload persistence observable) ahead of the comprehensive two-page user exit-check retest'
    parallelizable: false
    dependencies:
      - 'P1-editor-toolframe'
      - 'P1-editor-editloop'
      - 'P1-editor-controls'
      - 'P1-editor-symmetry-save'
```

**User instruction:** Implement Step 15 per the batch 105 editing-loop scope correction (patch #2): mount the editing tool frame into the reserved `neatracing-editor-tools` container over the `window.neatRacingEditorStart` boot path, wire the pointer-pick -> symmetry-canonicalized paint -> undo push -> rebuild-pipeline edit -> RAF preview-redraw editing loop, the palette/undo-redo/model-dropdown/CAD-view controls, and the mirror-X symmetry with immediate live-save through the dev-server model-store API; red -> implement -> green per slice.

**Step objective:** Close the batch 105 editing-composition ownership gap (patch #2; the same planning-defect class as the batch 102 gap (v) second half): the served `/editor/` page gains the working editing loop this step — tool-frame state + control mounts, pointer-pick edit loop (pick -> editor.state.paint via symmetry canonicalization -> editor.undo push -> createEditorRebuildPipeline edit(sliceZ) trailing debounce with pointerUp flush -> RAF preview redraw through the Step 14 preview lane), palette/undo/redo/model-dropdown/CAD-view controls (AC-020 + AC-019), and mirror-X symmetry default-ON toggle + immediate live-save through the dev-server model-store API (AC-021). The comprehensive two-page USER exit check (game page full keyboard lap + editor page full editing round-trip) follows this step.

**AC-020/AC-021 traceability (batch 105 mandate, patch #2):** coverage backlog rows AC-020 + AC-021 now bind Step 15 (P1-editor-toolframe + P1-editor-controls for AC-020; P1-editor-toolframe state wiring + P1-editor-editloop + P1-editor-symmetry-save for AC-021's browser half, with the AC-021 manual edit -> refresh -> persisted check as this step's green/exit boundary evidence), and AC-019 gains the "15" binding for the P1-editor-controls CAD ortho wiring. Module-level AC coverage stays with Step 09 (P1-S6-A/B/C; jest editor lanes per spec traceability). P1-editor-preview (Step 14) stays the preview-only lane — no game-scene race passes on the editor page.

**Stop conditions:**

- **Done:** editor + server jest lanes green (`testPathPatterns="examples/neatRacing/editor"` and `"examples/neatRacing/server"`) and `npx tsc -p tsconfig.neatRacing.json --noEmit` clean; the served `/editor/` page exposes the working editing loop (pointer-pick paints through the palette with visible mirror-X symmetry, undo/redo bounded at 3 steps, model dropdown switches + reloads the volume, CAD front/side/top views toggle, every edit live-saves, reload restores the model); DEMO/UI visible-window browser smoke green on `/editor/`; the comprehensive two-page user exit check follows this step; then Step 13 + Step 14 + Step 15 close and Phase 1 phase compression runs.
- **Blocked:** an editor state/pipeline/camera API mismatch or a missing editor seam not closable inside files_to_change — stop, record in Latest validation evidence, escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/editor"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/server"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`
- manual: npm run build:neatRacing + visible-window browser smoke on the served /editor/ page (tool-frame controls, symmetric paint, undo/redo, model dropdown, CAD views, live-save + reload persistence) ahead of the user exit-check retest

### Phase 2 — AI migration (NGE) [PLANNED]

```yaml
phase: 2
title: 'AI migration (NGE)'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_phase: 'Phase 3 - Polish + deprecation prep - dispatch after Phase 2 steps 01-05 are DONE'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - spec-checklist
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated)'
constitution_check: 'principle-2-human-mission-ai-method; principle-5-unique-ids'
acceptance_criteria:
  - 'AC-PLN-006: Phase 2 packets bind spec slices P2-S1..P2-S4 verbatim to AC-025..AC-034 + AC-043'
  - 'AC-PLN-007: evolution-correctness constraints carried (promotion three-part predicate AC-031, shared oscillation predicate AC-030, reproducibility tuple AC-026, eval-worker split AC-029/AC-043)'
placeholder_steps:
  - 'Step 01 - Simulation worker + packed frames + reproducibility tuple (spec P2-S1) [PLANNED]'
  - 'Step 02 - Eval worker + observation assembler + scripted rivals (spec P2-S2) [PLANNED]'
  - 'Step 03 - NGE growth + promotion part 1 (spec P2-S3 A-C) [PLANNED]'
  - 'Step 04 - NGE growth + promotion part 2 (spec P2-S3 D-E) [PLANNED]'
  - 'Step 05 - Mode integration + network visualizer (spec P2-S4) [PLANNED]'
```

**Phase objective:** Migrate the NGE brain to workers: simulation worker + packed frames + reproducibility tuple, eval worker + observation assembler + scripted rivals, NGE growth + promotion (controller/oscillation/promotion, then rival propagation/fitness shaping), and AI-mode integration with the network visualizer.

**Stop conditions:**

- **Done:** Phase 2 Steps 01-05 DONE with AC-025..AC-034 + AC-043 green and the reproducibility tuple recorded for a full AI-mode race.
- **Blocked:** any step packet fails its gate after two fix attempts — record the blocker in Latest validation evidence and escalate via 00.cross-tier-helper.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md`
- Consolidated slice gate per step: `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=<slice_id> --changed-files=<comma-separated files>`

#### Step 01 — Simulation worker + packed frames + reproducibility tuple (spec P2-S1) [PLANNED]

```yaml
phase: 2
step: 1
title: 'Simulation worker + packed frames + reproducibility tuple (spec P2-S1)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 02 - Eval worker + observation assembler + scripted rivals (spec P2-S2)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S1-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/workers'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-025: sim worker owns all sim state incl. physics + controllers; single in-band live-worker file; packed frames schema neatracing-packed-v1 (P2-S1-A/B)'
  - 'AC-026: reproducibility tuple (seed, agentCount, packSchemaVersion, opponentSnapshot, trackId, featureFlags) reproduces identical frames for the same tuple on the same runtime with no live human input (P2-S1-C)'
  - 'AC-042 (partial): per-frame host<->worker transfer over preallocated buffers (P2-S1-B)'
slices:
  - slice_id: 'P2-S1-A'
    title: 'Simulation worker (in-band live-worker file) + worker bundle registration'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/workers/simulation-worker/simulation-worker.ts'
      - 'examples/neatRacing/workers/simulation-worker/simulation-worker.test.ts'
      - 'package.json worker bundle addition (trivial registration)'
    acceptance_criteria:
      - 'AC-025 (partial): sim worker boots and runs the fixed-step loop; single in-band live-worker file per LD'
    red_test:
      - 'RED: simulation-worker.test.ts fails before the worker; asserts boot + message protocol contract'
    parallelizable: true
    dependencies:
      - 'P1-S7-G2'
  - slice_id: 'P2-S1-B'
    title: 'Frame pack service (neatracing-packed-v1, preallocated buffers)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/workers/simulation-worker/frame.pack.service.ts'
      - 'examples/neatRacing/workers/simulation-worker/frame.pack.service.test.ts'
    acceptance_criteria:
      - 'AC-025: packed frame schema neatracing-packed-v1; AC-042 partial: transfer uses preallocated buffers with offsets'
    red_test:
      - 'RED: frame.pack.service.test.ts fails before the service; asserts pack/unpack round-trip and zero-alloc transfer'
    parallelizable: true
    dependencies:
      - 'P2-S1-A'
  - slice_id: 'P2-S1-C'
    title: 'Reproducibility tuple utils (child seeds, byte parity)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/workers/simulation-worker/reproducibility.utils.ts'
      - 'examples/neatRacing/workers/simulation-worker/reproducibility.utils.test.ts'
    acceptance_criteria:
      - 'AC-026: same-tuple runs produce byte-identical frames across >= NEATRACING_REPLAY_ADAPTATION_TICKS=32 adaptation ticks; visual-only surfaces exempt; no Math.random sim-side (D1)'
    red_test:
      - 'RED: reproducibility.utils.test.ts fails before the utils; asserts byte parity over 32 adaptation ticks for a fixed tuple'
    parallelizable: true
    dependencies:
      - 'P2-S1-A'
  - slice_id: 'P2-S1-green'
    title: 'Step green validation: sim worker + packed frames + tuple'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 01 AC subset green: AC-025, AC-026, AC-042 partial; workers jest pattern green'
    parallelizable: false
    dependencies:
      - 'P2-S1-A'
      - 'P2-S1-B'
      - 'P2-S1-C'
```

**User instruction:** Implement spec slice P2-S1: simulation worker, packed frames (neatracing-packed-v1), and the reproducibility tuple; red -> implement -> green per slice.

**Step objective:** Move all sim state incl. physics + controllers into an in-band live-worker file with packed frames over preallocated buffers and byte-identical tuple reproduction.

**Stop conditions:**

- **Done:** workers jest suite green; typecheck clean; AC-025, AC-026, AC-042 partial recorded.
- **Blocked:** worker bundle registration breaks the dev server — record and escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/workers"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`

#### Step 02 — Eval worker + observation assembler + scripted rivals (spec P2-S2) [PLANNED]

```yaml
phase: 2
step: 2
title: 'Eval worker + observation assembler + scripted rivals (spec P2-S2)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 03 - NGE growth + promotion part 1 (spec P2-S3 A-C)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S2-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/workers'
  - 'examples/neatRacing/ai'
acceptance_criteria:
  - 'AC-029 (partial): eval worker protocol with bounded parallelVariantCount (P2-S2-A; growth engine lands in Phase 2 Step 03)'
  - 'AC-027: observation assembler produces the 82/89/110-channel tier vector with track-relative features and recycled scales (P2-S2-B)'
  - 'AC-028: scripted waypoint rivals with per-course deterministic throttle table (SCRIPTED_THROTTLE_CONSTANT=0.82 + proportional steer) (P2-S2-C)'
  - 'AC-043 (partial): eval-worker protocol budget hooks (P2-S2-A; budget tests complete in Phase 2 Step 03)'
slices:
  - slice_id: 'P2-S2-A'
    title: 'Eval worker + protocol'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/workers/eval-worker/eval-worker.ts'
      - 'examples/neatRacing/workers/eval-worker/eval.protocol.ts'
      - 'examples/neatRacing/workers/eval-worker/eval.protocol.test.ts'
    acceptance_criteria:
      - 'AC-029 (partial) + AC-043 (partial): two-worker split protocol with bounded parallelVariantCount named constant'
    red_test:
      - 'RED: eval.protocol.test.ts fails before the worker; asserts request/eval/result protocol and variant-count bound'
    parallelizable: true
    dependencies:
      - 'P2-S1-green'
  - slice_id: 'P2-S2-B'
    title: 'Observation assembler (82/89/110 channels)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/ai/observation.assembler.ts'
      - 'examples/neatRacing/ai/observation.assembler.test.ts'
    acceptance_criteria:
      - 'AC-027: channel counts 82/89/110 per tier with rival-gap >= 4 at lvl 3 and teammate tail 3x7=21; one tier per test fixture'
    red_test:
      - 'RED: observation.assembler.test.ts fails before the assembler; asserts per-tier channel counts and semantics fixtures'
    parallelizable: true
    dependencies: []
  - slice_id: 'P2-S2-C'
    title: 'Scripted waypoint rivals (per-course throttle table)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/ai/scripted.controller.ts'
      - 'examples/neatRacing/ai/scripted.controller.test.ts'
    acceptance_criteria:
      - 'AC-028: deterministic scripted controller with per-course speed/throttle table; throttle constant 0.82 + proportional steer'
    red_test:
      - 'RED: scripted.controller.test.ts fails before the controller; asserts per-course table determinism'
    parallelizable: true
    dependencies: []
  - slice_id: 'P2-S2-green'
    title: 'Step green validation: eval worker + assembler + rivals'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 02 AC subset green: AC-027, AC-028, AC-029/AC-043 partial; workers + ai jest patterns green'
    parallelizable: false
    dependencies:
      - 'P2-S2-A'
      - 'P2-S2-B'
      - 'P2-S2-C'
```

**User instruction:** Implement spec slice P2-S2: eval worker + protocol, observation assembler (82/89/110 channels), and scripted waypoint rivals; red -> implement -> green per slice.

**Step objective:** Deliver the eval-worker split protocol with bounded parallelVariantCount, the tiered observation vector, and deterministic scripted rivals.

**Stop conditions:**

- **Done:** workers + ai jest suites green; AC-027, AC-028 + AC-029/AC-043 partials recorded.
- **Blocked:** channel-count fixtures disagree with spec tables — record and escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/workers"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai"`

#### Step 03 — NGE growth + promotion part 1 (spec P2-S3 A-C) [PLANNED]

```yaml
phase: 2
step: 3
title: 'NGE growth + promotion part 1 (spec P2-S3 A-C)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 04 - NGE growth + promotion part 2 (spec P2-S3 D-E)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S3-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/ai'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - 'AC-029: NGE controller + runtime adaptation migrated with maxNodes 8000 / maxConnections 32000; eval in separate worker; <= 2 ms per-tick sim budget; jest.useFakeTimers for adaptation/timer tests (P2-S3-A)'
  - 'AC-030: shared oscillation predicate drives penalty AND commit boost with hysteresis + symmetry + deadband 0.15 (P2-S3-B)'
  - 'AC-031: three-part promotion predicate with 20-lap floor, drivability gate, seedNetwork + carriedInnovationCounter tuple fields, round-boundary state reset (P2-S3-C)'
slices:
  - slice_id: 'P2-S3-A'
    title: 'NGE controller + runtime adaptation (seeded, capacity-limited)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - 'examples/neatRacing/ai/nge.controller.ts'
      - 'examples/neatRacing/ai/runtime.adaptation.ts'
      - 'examples/neatRacing/ai/runtime.adaptation.test.ts'
    acceptance_criteria:
      - 'AC-029: capacity limits 8000/32000 preserved; seeded xorshift32 random from (tupleSeed, carIndex); per-tick budget <= NEATRACING_ADAPTATION_TICK_BUDGET_MS=2'
      - 'AC-026 (support): adaptation engine consumes the reproducibility tuple child-seed table (D1)'
      - 'AC-043: budget test with jest.useFakeTimers asserts sim-worker residual work within budget'
    red_test:
      - 'RED: runtime.adaptation.test.ts fails before migration; asserts capacity caps, seed derivation, and 2 ms budget with fake timers'
    parallelizable: true
    dependencies:
      - 'P2-S2-green'
  - slice_id: 'P2-S3-B'
    title: 'Shared oscillation predicate (penalty + commit boost)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/ai/oscillation-penalty.utils.ts'
      - 'examples/neatRacing/ai/oscillation-penalty.utils.test.ts'
    acceptance_criteria:
      - 'AC-030: one shared predicate replaces RACING_OSCILLATION_MIN_NEURONS=200; below-threshold never penalized; hysteresis; symmetric for baseline + variant; deadband 0.15'
    red_test:
      - 'RED: oscillation-penalty.utils.test.ts fails before the utils; asserts below-threshold no-penalty, grace window, hysteresis, and symmetry cases'
    parallelizable: true
    dependencies: []
  - slice_id: 'P2-S3-C'
    title: 'Promotion rules (three-part predicate + round boundary)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 4
    files_to_change:
      - 'examples/neatRacing/ai/promotion.rules.ts'
      - 'examples/neatRacing/ai/promotion.rules.test.ts'
    acceptance_criteria:
      - 'AC-031: one red test per condition + combined; lap floor NEATRACING_AI_MIN_SESSION_LAPS=20 precondition; seedNetwork (serialized Float32Array + NGE envelope) + carriedInnovationCounter tuple fields; scripted-baseline probe episode pinned; no mid-session materialization'
    red_test:
      - 'RED: promotion.rules.test.ts fails before the module; asserts per-condition cases + all-conditions + round-boundary replay-from-own-tuple'
    parallelizable: true
    dependencies:
      - 'P2-S3-A'
      - 'P2-S3-B'
  - slice_id: 'P2-S3-G1'
    title: 'Part 1 green validation: controller + oscillation + promotion'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 03 AC subset green: AC-029, AC-030, AC-031, AC-043; ai jest pattern green'
    parallelizable: false
    dependencies:
      - 'P2-S3-A'
      - 'P2-S3-B'
      - 'P2-S3-C'
```

**User instruction:** Implement spec slice P2-S3 A-C: NGE controller + runtime adaptation migration, shared oscillation predicate, and the three-part promotion predicate; red -> implement -> green per slice.

**Step objective:** Migrate the NGE brain into the eval-worker protocol with capacity limits, seeded RNG, oscillation predicate, and promotion rules.

**Stop conditions:**

- **Done:** ai jest suite green; typecheck clean; AC-029, AC-030, AC-031, AC-043 recorded.
- **Blocked:** adaptation budget exceeded under jest.useFakeTimers — record and continue only once the named budget constant is respected.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`

#### Step 04 — NGE growth + promotion part 2 (spec P2-S3 D-E) [PLANNED]

```yaml
phase: 2
step: 4
title: 'NGE growth + promotion part 2 (spec P2-S3 D-E)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 05 - Mode integration + network visualizer (spec P2-S4)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S3-D'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/ai'
acceptance_criteria:
  - 'AC-032: rival winner propagation as bounded seeded weight-nudge mutations on serialized weights; LAP/line-boundary swaps only; lvl-2 no-op tested; main agent never fully reset (P2-S3-D)'
  - 'AC-029 (support): fitness shaping keeps variant evaluation in the eval worker per TP-08 (P2-S3-E)'
slices:
  - slice_id: 'P2-S3-D'
    title: 'Rival propagation operator (weight nudges + snapshot serialization)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/ai/rival.propagation.ts'
      - 'examples/neatRacing/ai/rival.propagation.test.ts'
    acceptance_criteria:
      - 'AC-032: lvl-2 no-op; lvl-3 (4 rivals) propagation; boundary timing; crossing-network snapshot + mutation RNG seed serialized into opponentSnapshot'
    red_test:
      - 'RED: rival.propagation.test.ts fails before the module; asserts lvl-2 no-op, lvl-3 propagation, boundary timing, snapshot serialization'
    parallelizable: true
    dependencies:
      - 'P2-S3-G1'
  - slice_id: 'P2-S3-E'
    title: 'Fitness shaping utils (TP-08)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/ai/fitness.shaping.utils.ts'
      - 'examples/neatRacing/ai/fitness.shaping.utils.test.ts'
    acceptance_criteria:
      - 'AC-029 (support): shaping terms per TP-08 keep variant evaluation inside the eval worker protocol'
    red_test:
      - 'RED: fitness.shaping.utils.test.ts fails before the utils; asserts shaping term math and bounded outputs'
    parallelizable: true
    dependencies: []
  - slice_id: 'P2-S3-G2'
    title: 'Part 2 green validation: propagation + fitness shaping'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 04 AC subset green: AC-032, AC-029 support; ai jest pattern green'
    parallelizable: false
    dependencies:
      - 'P2-S3-D'
      - 'P2-S3-E'
```

**User instruction:** Implement spec slice P2-S3 D-E: rival winner propagation and TP-08 fitness shaping; red -> implement -> green per slice.

**Step objective:** Deliver bounded seeded weight-nudge propagation with LAP/line-boundary swaps and fitness shaping kept inside the eval worker.

**Stop conditions:**

- **Done:** ai jest suite green; AC-032 + AC-029 support recorded.
- **Blocked:** propagation touches the main agent state — stop, record, escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai"`

#### Step 05 — Mode integration + network visualizer (spec P2-S4) [PLANNED]

```yaml
phase: 2
step: 5
title: 'Mode integration + network visualizer (spec P2-S4)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Phase 2 complete - dispatch Phase 3 (Step 01) after green light'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P2-S4-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/ai'
  - 'examples/neatRacing/host'
  - 'manual: AI mode race with evolving agents + right-side network panel'
acceptance_criteria:
  - 'AC-033: AI mode runs >= NEATRACING_AI_MIN_SESSION_LAPS=20 laps until growth goal met; mode toggle restarts the race; human takes car 0 vs evolving agents (P2-S4-A/C)'
  - 'AC-034: right-side network panel renders only the most-forward agent via thin adapter over examples/shared/network-visualization (P2-S4-B)'
slices:
  - slice_id: 'P2-S4-A'
    title: 'AI mode goals + mode selector integration'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/ai/ai-mode.goals.ts'
      - 'examples/neatRacing/host/mode-selector.ts'
      - 'examples/neatRacing/ai/ai-mode.goals.test.ts'
    acceptance_criteria:
      - 'AC-033: 20-lap minimum then promotion-threshold-driven continuation; lvl 3 continues until user stops'
    red_test:
      - 'RED: ai-mode.goals.test.ts fails before the goals module; asserts lap floor and growth-goal continuation'
    parallelizable: true
    dependencies:
      - 'P2-S3-G2'
  - slice_id: 'P2-S4-B'
    title: 'Network panel adapter (most-forward agent only)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/visualizer/network-panel.adapter.ts'
      - 'examples/neatRacing/visualizer/network-panel.adapter.test.ts'
    acceptance_criteria:
      - 'AC-034: adapter renders only the current winner; resize service behavior matches installRacingNetworkResize pattern'
    red_test:
      - 'RED: network-panel.adapter.test.ts fails before the adapter; asserts most-forward selection + resize wiring'
    parallelizable: true
    dependencies: []
  - slice_id: 'P2-S4-C'
    title: 'Human car 0 vs evolving agents host wiring'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/host/host.wiring.test.ts'
    acceptance_criteria:
      - 'AC-033: end-to-end wiring of human car 0 racing evolving agents through the sim worker protocol'
    red_test:
      - 'RED: host.wiring.test.ts fails before wiring; asserts car 0 human control + rival evolution loop integration'
    parallelizable: true
    dependencies:
      - 'P2-S4-A'
  - slice_id: 'P2-S4-green'
    title: 'Step green validation: mode integration + visualizer'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 05 AC subset green: AC-033, AC-034; ai + host jest patterns green; manual AI-mode check'
    parallelizable: false
    dependencies:
      - 'P2-S4-A'
      - 'P2-S4-B'
      - 'P2-S4-C'
```

**User instruction:** Implement spec slice P2-S4: AI-mode integration (20-lap goal floor, mode toggle) and the most-forward-agent network panel; red -> implement -> green per slice.

**Step objective:** Wire AI mode end-to-end: human car 0 vs evolving agents through the sim worker protocol, plus the right-side network panel adapter.

**Stop conditions:**

- **Done:** ai + host jest suites green; manual AI-mode race with the network panel works; AC-033, AC-034 recorded.
- **Blocked:** visualizer adapter breaks the examples/shared contract — record and escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/host"`
- manual: AI mode race with evolving agents + right-side network panel

### Phase 3 — Polish + deprecation prep [PLANNED]

```yaml
phase: 3
title: 'Polish + deprecation prep'
status: '[PLANNED]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_phase: 'Terminal - user-confirmed racing_curriculum removal workstream is explicitly out of scope (LD-33)'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - spec-checklist
  - tracker-handoff
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated)'
constitution_check: 'principle-4-breadth-first-recoverable; principle-5-unique-ids'
acceptance_criteria:
  - 'AC-PLN-008: Phase 3 packets bind spec slices P3-S1..P3-S2 verbatim to AC-035..AC-039'
  - 'AC-PLN-009: deprecation prep is notices-only; no removal, no dual-path code (LD-33)'
placeholder_steps:
  - 'Step 01 - Adaptive quality + memory budget (spec P3-S1) [PLANNED]'
  - 'Step 02 - Build + docs + deprecation prep (spec P3-S2) [PLANNED]'
```

**Phase objective:** Polish + deprecation prep: adaptive quality switching + full memory budget accounting, then the build pipeline + docs + deprecation notices so the neatRacing example ships cleanly ahead of the separate user-confirmed racing_curriculum removal workstream.

**Stop conditions:**

- **Done:** Phase 3 Steps 01-02 DONE with AC-035..AC-039 green; build + docs + racing_curriculum suites green.
- **Blocked:** any step packet fails its gate after two fix attempts — record the blocker in Latest validation evidence and escalate via 00.cross-tier-helper.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md`
- Consolidated slice gate per step: `node scripts/agent-customization/gates/slice-advancement.gate.mjs --json --slice-id=<slice_id> --changed-files=<comma-separated files>`

#### Step 01 — Adaptive quality + memory budget (spec P3-S1) [PLANNED]

```yaml
phase: 3
step: 1
title: 'Adaptive quality + memory budget (spec P3-S1)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 02 - Build + docs + deprecation prep (spec P3-S2)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P3-S1-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'examples/neatRacing/engine'
  - 'examples/neatRacing/voxel'
acceptance_criteria:
  - 'AC-035: quality-first adaptive degradation order (bloom off -> LOD tier -> prop density) with deterministic hysteresis-bounded decision function (P3-S1-A)'
  - 'AC-036: memory accounting proves no dense 128^3 grid retained at game runtime and peak within named budget; editor dense volume + undo snapshots accounted (P3-S1-B)'
slices:
  - slice_id: 'P3-S1-A'
    title: 'Adaptive quality decision function'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'examples/neatRacing/engine/adaptive-quality.ts'
      - 'examples/neatRacing/engine/adaptive-quality.test.ts'
    acceptance_criteria:
      - 'AC-035: drop order bloom -> LOD -> props; explicit bloom off-switch (G8); hysteresis-bounded and deterministic'
    red_test:
      - 'RED: adaptive-quality.test.ts fails before the module; asserts drop order + hysteresis + explicit bloom off'
    parallelizable: true
    dependencies:
      - 'P2-S4-green'
  - slice_id: 'P3-S1-B'
    title: 'Memory budget accounting (runtime + editor)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/voxel/memory.budget.utils.ts'
      - 'examples/neatRacing/voxel/memory.budget.utils.test.ts'
    acceptance_criteria:
      - 'AC-036: accounting proves sparse-only 128^3 retention at runtime; editor dense 2.1 MB working volume + <= 8 MB undo budget accounted'
    red_test:
      - 'RED: memory.budget.utils.test.ts fails before the utils; asserts accounting math and budget assertions on fixtures'
    parallelizable: true
    dependencies: []
  - slice_id: 'P3-S1-green'
    title: 'Step green validation: adaptive quality + memory'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 01 AC subset green: AC-035, AC-036; engine + voxel jest patterns green'
    parallelizable: false
    dependencies:
      - 'P3-S1-A'
      - 'P3-S1-B'
```

**User instruction:** Implement spec slice P3-S1: adaptive quality decision function and full memory budget accounting; red -> implement -> green per slice.

**Step objective:** Deliver hysteresis-bounded adaptive degradation (bloom -> LOD -> props) and the memory accounting that proves sparse-only runtime retention.

**Stop conditions:**

- **Done:** engine + voxel jest suites green; AC-035, AC-036 recorded.
- **Blocked:** budget accounting contradicts editor undo snapshot figures — record and reconcile against the AC-020 constants.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"`
- `npm run jest:base -- --testPathPatterns="examples/neatRacing/voxel"`

#### Step 02 — Build + docs + deprecation prep (spec P3-S2) [PLANNED]

```yaml
phase: 3
step: 2
title: 'Build + docs + deprecation prep (spec P3-S2)'
status: '[PLANNED]'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: green-only
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Terminal - neatRacing tracker complete; deprecation removal is a separate user-confirmed workstream'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P3-S2-A'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
validation:
  - 'npm run build:neatRacing'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'npm run docs'
  - 'examples/racing_curriculum'
  - 'manual: browser check of bundle-served game page'
acceptance_criteria:
  - 'AC-037: esbuild pipeline emits docs/assets/neatracing.bundle.js (host) + neatracing.worker.js with alias neataptic -> src/browser-entry.ts and external fs/child_process/path (P3-S2-A)'
  - 'AC-038: examples/neatRacing docs generated/refreshed via npm run docs with target registered (P3-S2-B)'
  - 'AC-039: deprecation notices in examples/racing_curriculum/README.md + examples/README.md pointing to examples/neatRacing; no removal, no dual-path (P3-S2-C)'
slices:
  - slice_id: 'P3-S2-A'
    title: 'esbuild bundle pipeline + build test'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    files_to_change:
      - 'scripts/build-neatracing.mjs'
      - 'scripts/build-neatracing.test.mjs'
    acceptance_criteria:
      - 'AC-037: host + worker bundles emitted to docs/assets with IIFE format, externals, and alias; type check clean'
    red_test:
      - 'RED: build-neatracing.test.mjs fails before the script; asserts bundle outputs, externals, and alias wiring (build-variants style)'
    parallelizable: true
    dependencies:
      - 'P3-S1-green'
  - slice_id: 'P3-S2-B'
    title: 'neatRacing folder docs via docs target'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/neatRacing/README.md'
      - 'scripts/quality/docs-generator config (neatRacing target registration)'
    acceptance_criteria:
      - 'AC-038: npm run docs emits/refreshes neatRacing docs with educational-docs tone; no stale docs'
    red_test:
      - 'RED: docs spot-check fails while neatRacing target is missing from the docs generator config'
    parallelizable: true
    dependencies: []
  - slice_id: 'P3-S2-C'
    title: 'Deprecation notices only (no removal)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/README.md'
      - 'examples/README.md'
      - 'examples/racing_curriculum/deprecation-notice.test.ts'
    acceptance_criteria:
      - 'AC-039: both notices point to examples/neatRacing as successor; racing_curriculum suites remain green; no dual-path code'
    red_test:
      - 'RED: deprecation-notice.test.ts fails before notices exist; asserts both notice texts via source inspection'
    parallelizable: true
    dependencies: []
  - slice_id: 'P3-S2-green'
    title: 'Step green validation: build + docs + deprecation prep'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 1
    files_to_change:
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - 'Step 02 AC subset green: AC-037, AC-038, AC-039; build + docs + racing_curriculum suites green; manual browser check'
    parallelizable: false
    dependencies:
      - 'P3-S2-A'
      - 'P3-S2-B'
      - 'P3-S2-C'
```

**User instruction:** Implement spec slice P3-S2: esbuild bundle pipeline, neatRacing docs target, and deprecation notices only; red -> implement -> green per slice.

**Step objective:** Ship the example: host + worker bundles to docs/assets, docs generated via the docs target, and successor-pointing deprecation notices with no removal.

**Stop conditions:**

- **Done:** build + docs + racing_curriculum suites green; manual browser check of the bundle-served page; AC-037..AC-039 recorded.
- **Blocked:** bundle externals or alias wiring fail — record and escalate via 00.cross-tier-helper.

**Required validation:**

- `npm run build:neatRacing`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`
- `npm run docs`
- `npm run jest:base -- --testPathPatterns="examples/racing_curriculum"`
- manual: browser check of bundle-served game page

## Validation gates

Step-level `validation` lists inside each packet are the authoritative per-step command
sources (MCP allowlist authority: `active-step.validation`). Consolidated gate routing:

- `slice-advancement` — consolidates plan-sync + step-packet + plan-slice-quality +
  plan-command-lint + shared-validation + code-coverage + specialist-review; run per
  active slice/step handoff. The Step 01 authoring pass is captured in
  `## Latest validation evidence`.
- `cortex-index` — RAG freshness + workflow/ corpus MCP liveness; rerun after every
  tracker edit batch (see `rag-index/reindex-plan-family.mjs` + `build-index` +
  `validate-index` recovery loop).
- `agent-graph` — only after any agent delegation change. `learning-event` — only after
  discovering a workflow gap or improvement.

## Deferred questions

None. The spec records all open questions RESOLVED (section 20); the tracker introduces
no new NEEDS CLARIFICATION markers. If execution uncovers a spec conflict, stop and
escalate via `00.cross-tier-helper` before deviating.

## Latest validation evidence

- Planning pass (this instance, 01-planning author role): tracker authored from spec
  section 17 (55 impl slices preserved verbatim + 19 step-green slices = 74 gate slices
  across 20 steps and 3 phases); registration + session rebind + RAG reindex performed.
- Gate results and green-light determination are appended below by this authoring pass
  (authoring self-check) and must be independently confirmed by a fresh verification
  instance; this section intentionally does NOT record green-light: true.

<!-- EVIDENCE-APPEND-POINT -->

### Evidence archive pointer (plan compressed 2026-10-11)

Every historical `### Evidence batch` / `### Batch` record for this plan (steps 02-14B,
batches up to 138, 2026-09-27 -> 2026-10-06) was moved verbatim to
`plans/neatracing-voxel-racer.logs.md` -> near EOF under
"## Evidence-batch archive (moved verbatim from plans/neatracing-voxel-racer.plans.md on
2026-10-11)". New batches append THERE (below the archive heading); this file stays
body-only. Compact resume index for the current Step 14B work:

- batch 136 - lookAtInto sign-flip probe: the matrix is algebraically correct; probe
  rolled back; do NOT flip its forward row (inconsistent with perspectiveInto
  `out[11] = -1` and the regression test).
- batch 137 - Step 14B round 2: AI autopilot boot race + red team body + lap-detector
  axis fix; the page boots as a RUNNING ai race; lanes green.
- batch 138 - Step 14B round 3: car-pass axis-1 expansion transposition + world
  handedness (world z = -y + PI/2 nose yaw) + wheel spin cap + track ribbon mirror;
  all lanes + tsc green; visible smoke green; FINAL USER GATE OPEN
  (P1-correctness-green [WIP]).
- batch 139 - Step 14B user gate attempt 2026-10-11: dev server relaunched
  (HTTP 200 on first poll); gate prompt issued but user unavailable; per the
  do-not-self-close rule P1-correctness-green stays [WIP]; no lanes re-run, no
  source touched, tmp_* count 0. Next boundary: user gate → close Step 14B → Step 15.
