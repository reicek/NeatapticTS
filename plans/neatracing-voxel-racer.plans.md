# Neat Racing Voxel Racer (neatRacing)

**Status:** [WIP]

<!-- Derived from plans/neatracing-voxel-racer.spec.md (spec sections 13, 17, 18, 12.4-12.5).
     This is the tracker for the neatRacing voxel racer flagship demo. Phase/step/slice
     structure mirrors spec section 17 phases P1/P2/P3 1:1 with slice_ids kept verbatim
     so section 18 traceability holds without renumbering. Planning-only authoring pass:
     no src/, docs/, examples/, or scripts/ production edits. -->

<!-- COUNTDOWN-FIRST HANDOFF AMENDMENT (01-planning, tracker-only).
 The Handoff query below is the durable resume instruction. The active
 frontier is Step 27A: a transparent Astrobird-like countdown overlay,
 no black/opaque backdrop during or after it, and simulation resume at
 GO. Step 30 drift work and its blocked verdict are retained behind
 this repair; no racer code, tests, builds, or browser actions were run.
 The prior handoff and pause detail remain archived in the companion log. -->

## Scope

- Implement the neatRacing voxel racer per `plans/neatracing-voxel-racer.spec.md`:
  Phase 1 graphics engine + manual keyboard driving (AC-001..AC-024 + AC-040..AC-042),
  Phase 2 AI migration with NGE growth engine (AC-025..AC-034 + AC-043),
  Phase 3 polish + deprecation prep (AC-035..AC-039),
  Phase 4 AI-core integration (USER VERDICT 2026-11-07 — the plan is NOT complete:
  the AI core is missing; the Phase-2 [DONE] claims hold at the MODULE level only;
  Phase 4 re-opens AC-027..AC-034 + AC-043 in the LIVE lane and authors the NEW
  user-verdict ACs AC-100..AC-103; see the `## Phase-4 AUDIT` section).
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
  `expansion: slices`, `auto_expand: true`, and a `tdd_sequence` that MATCHES the
  slice roles the packet carries (`red-green` when the FIRST slice is a
  red-testing slice, `green-only` when the first slice is an implementing slice),
  and typed slices (slice_id / title / status / goal / estimate_hours /
  files_to_change / acceptance_criteria / parallelizable / dependencies) per the
  step-packet gate schema.
- **Slice discipline:** each slice is one atomic behavioral intent bound to spec
  section 17; at most 3 code files per slice (sibling `*.test.ts` files count;
  trivial one-line config registrations such as a package.json script line or a
  tsconfig include may accompany without counting); estimate_hours <= 4; at most
  5 slices per step - a step that needs a SIXTH slice is SPLIT into two steps
  and the split is recorded in the wave note, in the phase step list and in the
  plan tail (the at-most-5 limit is a MANDATE: a deviation is never recorded in
  its place, and a split NEVER defers a re-pin away from the production change
  that invalidated the pin).
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
- **Visible-browser validation (the STANDING USER RULE, recorded 2026-11-21, verbatim):**
  "visible-browser validation is required; topmost/foreground/focus-stealing is prohibited;
  background is valid; FPS is not required." EVERY browser-backed slice MUST validate in a REAL
  VISIBLE browser window; the harness MUST NOT require an always-on-top / foreground /
  focus-stealing window and MUST NOT change the z-order — a BACKGROUNDED window is VALID; NO FPS
  ASSERTION may be required or asserted (the measured frame rate of a throttled background window
  is an ENVIRONMENT property, NOT a defect — the `P5-S27-5` lane measured ~1.9-2.4 fps in a
  backgrounded window and asserted NOTHING about FPS); the pixel proof MAY be read from the
  COMPOSER SCENE RENDER TARGET via `readPixels` when the adaptive-quality ladder disables bloom
  in a backgrounded window. THE RULE IMPOSES NO PRODUCTION BEHAVIOUR — it is a VALIDATION-STANDARD
  rule for every future slice of this plan.
- **Browser-tooling minimization + rare FPS exception (USER DIRECTIVE):** minimize
  Chrome MCP and other browser automation. For the countdown repair, use no
  browser during planning or implementation; ideally reserve ONE final real
  visible-browser validation after the focused checks are green. Never force
  foreground, focus, always-on-top, or z-order changes. The ONLY exception is a
  rare, explicitly FPS-scoped test authorized for that purpose; it is NOT part
  of Step 27A and must not become a routine countdown or FPS requirement. A
  visible background window is valid evidence and needs no activation or
  topmost change.
- **Evidence:** gate runs are recorded in `## Latest validation evidence`; green-light
  recording is reserved for the separate verification instance.

## Current state

- **ACTIVE USER PRIORITY — COUNTDOWN REPAIR BEFORE ANY OTHER RACER IMPLEMENTATION.** Step 27A is the only `[WIP]` step and `P5-S27A-red` is the only active slice; Step 30 and `P5-S30-repin` are `[PLANNED]`, with their blocked independent verdict retained rather than cleared. Execution order is: fresh `01-planning` verification of Step 27A, `03-red-testing` for `P5-S27A-red`, then `04-implementing` for `P5-S27A-overlay`, then `05-green-testing` for `P5-S27A-green`. Only after Step 27A is DONE may the existing six-edit Step-30 author amendment and fresh `P5-S30-repin` verification resume. The first production implementation is therefore `P5-S27A-overlay`, not the drift re-pin.
- **COUNTDOWN REPAIR CONTRACT.** Preserve the completed 3-2-1-GO resolver and HOLD-until-GO semantics, but repair the presentation to a transparent Astrobird-like text/glow overlay with the race scene visible behind it. No black, opaque, or full-screen dimming backdrop may appear during 3/2/1, GO, dismissal, or afterward. Dismissal must remove/hide the overlay and any backdrop without leaving an input-capturing layer. At GO the input gate must release and the simulation must observably resume (simulation time/frame progression and post-GO throttle accepted), rather than remaining stopped behind a cleared overlay. Acceptance lives in AC-106-R1..R3 and Step 27A below.
- **THIS AMENDMENT IS PLANNING-ONLY.** No racer implementation, test, build, or browser/Chrome action was performed or authorized for this pass. Chrome MCP is minimized, ideally to one final visible-background validation after focused checks; no foreground/focus/topmost forcing, except the separately authorized rare FPS-only exception recorded in Mandates. Step 27A has NO FPS assertion and no foreground exception.
- **HISTORICAL PAUSE RECORD — SUPERSEDED FOR RESUME ORDER BY THE ACTIVE USER PRIORITY ABOVE; ITS STEP-30 BLOCKED EVIDENCE REMAINS VALID. THE OLD STATUS/RESUME CLAIMS BELOW ARE NOT OPERATIVE.**
- **PAUSED (2026-11-23, THE USER-DIRECTED TRANSITION TO A NEW OPENTELEMETRY WORKSTREAM — THE PLAN IS PAUSED AT THE `P5-S30-repin` BLOCKED VERDICT; NO `04-implementing`, `03-red-testing` OR `05-green-testing` LANE IS AUTHORIZED; NO STATUS TOKEN WAS FLIPPED; THE PLAN IS **NOT** CLOSED, **NOT** COMPRESSED AND **NOT** ARCHIVED; TRACKER-ONLY — THIS PLAN PLUS `plans/README.md` PLUS `plans/Roadmap.md` PLUS THE SAME-BOUNDARY `plans/neatracing-voxel-racer.logs.md` ARE THE ONLY BYTES WRITTEN; NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE; NO MUTATING GIT COMMAND):**
  0. **THE PAUSE BOUNDARY IS THE 2026-11-23 `P5-S30-repin` INDEPENDENT VERIFICATION VERDICT — **BLOCKED** (BLOCKERS: TWO — B-1 THE SCOPE / LEGACY-FLOOR COLLISION, B-2 THE AI-CHANNEL AUTHORITY; BOUNDED CLARIFICATIONS: TWO — C-1 THE CHATTER BAND, C-2 THE VALIDATION LIST; EXPLICITLY SCOPED TO THE SLICE `P5-S30-repin` ONLY). NOTHING RAN AFTER THAT VERDICT: NO `01-planning` AMENDMENT PASS, NO `04-implementing` DISPATCH, NO FRESH RE-VERIFICATION, NO TEST RUN, NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE. THE FULL VERDICT DETAIL IS THE NEWEST `## Latest validation evidence` ROW, WHICH SITS DIRECTLY BELOW THE `<!-- EVIDENCE-APPEND-POINT -->` MARKER — **THAT ROW IS THE AUTHORITATIVE PAUSE STATE AND IS NOT REWRITTEN HERE.**
  1. **THE TOKENS ARE UNCHANGED AND MUST STAY UNCHANGED WHILE PAUSED:** THE PLAN `**Status:**` IS `[WIP]`; PHASE 5 IS `[WIP]`; STEP 30 IS `[WIP]`; THE ONE `[WIP]` SLICE IS `P5-S30-repin`; `P5-S30-slip`, `P5-S30-green`, `P5-S30A-crossing-rebase` AND STEPS 31 / 32 REMAIN `[PLANNED]`. **THE PAUSE IS RECORDED IN PROSE AND NEVER BY A TOKEN FLIP** — THE MCP PLAN PARSER (`scripts/agent-customization/mcp/mcp-plan-utils.mjs:153`) THROWS WHEN A `[WIP]` PHASE HAS NO `[WIP]` STEP, SO FLIPPING STEP 30 TO `[PLANNED]` WOULD BREAK THE TRACKER; THE PLAN-SYNC VALIDATOR (`scripts/agent-customization/customization-utils.mjs:1067`) READS THE LITERAL `[WIP]` TOKEN, AND `plans/README.md` PLUS `plans/Roadmap.md` THEREFORE RETAIN THE LITERAL `[WIP]` WHILE CARRYING A `PAUSED` ANNOTATION.
  2. **THE EXACT RESUME POINT — THE FIRST ACT ON RESUME, IN ORDER:** (a) A TRACKER-ONLY `01-planning` **AUTHOR** PASS APPLIES THE SIX-EDIT PLAN-SIDE AMENDMENT TO THE STEP-30 PACKET THAT THE 2026-11-23 VERIFICATION RECORDED IN ITS CLAUSE 7 — (1) STATE THE FIRST-TICK SEEDING RULE (THE FIRST TICK OF A LANE IS **UNSMOOTHED**, THE SMOOTHING AND THE FEEDBACK ENGAGE FROM THE **SECOND** TICK ONWARD); (2) STATE THE CHANNEL (THE SMOOTHING AND THE FEEDBACK APPLY TO THE **USER** CHANNEL **ONLY**, SO EVERY AI NUMBER STAYS BYTE-IDENTICAL); (3) STATE THE DRIFT-LAW INPUT AND RECORD THE AC-091 "ONE RESOLVED NUMBER" SUPERSESSION FOR THE HEADING ADVANCE ONLY; (4) NAME THE COUPLED FIRST-TICK ASSERTIONS (THE VERIFICATION CLAUSE 2 (i)-(vii)) AND THE RE-PIN POSTURE (DISCLOSE THE RE-PIN OBLIGATION AND WIDEN `files_to_change` TO `examples/neatRacing/physics/physics.step.service.test.ts` IF A RE-PIN EVER PROVES UNAVOIDABLE); (5) NAME THE CHATTER-BAND HOME (`examples/neatRacing/physics/physics.step.service.ts`, AN EXPORTED `NEATRACING_PHYSICS_*` CONSTANT) AND THE MEASURED WINDOW (AT LEAST ~0.0005 rad, AT MOST ~0.00158 rad); (6) ADD `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai/scripted.controller"` TO THE SLICE VALIDATION LIST. (b) A **FRESH** `01-planning` VERIFICATION INSTANCE RE-RUNS THE ONE CONSOLIDATED `slice-advancement` GATE OVER THE AMENDED PACKET. (c) **ONLY** A `green-light: true` VERDICT THERE UNBLOCKS THE `04-implementing` LANE FOR `P5-S30-repin`. **THE AMENDMENT IS A PLAN-SIDE EDIT ONLY — IT IS NOT A PRODUCTION OR TEST EDIT, AND IT IS NOT DISPATCHABLE AS `04-implementing` BEFORE THE FRESH VERIFICATION.**
  3. **NO NEW MARKER AND NO NEW DEFERRED QUESTION:** THE PLAN-WIDE `NEEDS CLARIFICATION` BUDGET STANDS AT **3 OF 3 ACTIVE** (CLAR-4-01, CLAR-4-02, CLAR-5-01) AND THIS PAUSE CREATES **NO** FOURTH MARKER. THE 2026-11-23 BOUNDED CLARIFICATIONS C-1 AND C-2 ARE **NOT** MARKERS — THEY ARE CLOSED BY THE CLAUSE-7 AMENDMENT IN ITEM 2.
  4. **THE PAUSE IS DURABLE IN FOUR PLACES:** THIS `## Current state` ROW; THE `## Handoff query` PAUSE BANNER; THE SAME-BOUNDARY `plans/neatracing-voxel-racer.logs.md` PAUSE CLOSE-OUT NOTE; AND THE `PAUSED` ANNOTATIONS IN `plans/README.md` PLUS `plans/Roadmap.md`. THE SAME-BOUNDARY NOTE IS WHAT MAKES THE PAUSE SURVIVE A LATER TRACKER COMPRESSION.
  5. **THE PAUSE IS NOT A DEFERRED-CLEANUP ESCAPE:** WHEN THE WORKSTREAM RESUMES, EVERY SLICE BOUNDARY STILL HOLDS — AT MOST 3 CODE FILES PER SLICE (SIBLING `*.test.ts` FILES COUNT), `estimate_hours` <= 4, AT MOST 5 SLICES PER STEP, RED-FIRST, AND NO OLD CODE LEFT BEHIND. THE PAUSE INTRODUCES NO NEW CODE, SO IT OWES NO CLEANUP.
  6. **THE STANDING CARRIES ARE UNCHANGED AND NONE IS ABSORBED:** THE AI-LANE LIVE-LANE BUDGET RE-PIN (THE LIVE FIGURE IS **283 / 285**); COVERAGE GAP g1 (REACHABLE-BUT-UNTESTED — ADD A TEST, DO NOT REMOVE THE BRANCH); AND THE STANDING VISIBLE-BROWSER USER RULE (VERBATIM, RECORDED 2026-11-21) — WHICH BINDS EVERY BROWSER-BACKED SLICE ON RESUME.
  7. **THE ACTIVE LANE MOVES, NOT THE WORK:** A **NEW, SEPARATE** OPENTELEMETRY WORKSTREAM IS THE ACTIVE LANE WHILE THIS ONE IS PAUSED. IT IS A DISTINCT PLAN TRACKER, NOT A PHASE OF THIS ONE — IT OWNS ITS OWN SCOPE, ITS OWN ACCEPTANCE CRITERIA AND ITS OWN SLICE PACKETS, AND IT MUST NOT WIDEN STEP 30 / 30A / 31 / 32 OR TOUCH THE RACER PRODUCTION BYTES.
- **THE OPERATIVE GREEN VERDICT (2026-11-22, THE SLICE-SCOPED `05-green-testing` (Tier 1) LANE — THE SLICE `P5-S30-damping` IS `GREEN: OK` (BLOCKERS: NONE, EXPLICITLY SCOPED TO THAT SLICE ONLY); THE `P5-S30-damping` TOKEN IS NOW `[DONE 2026-11-22 - 05-green-testing GREEN: OK]`; THE ONE `[WIP]` SLICE IS NOW `P5-S30-repin` AND THE STEP-30 `pre_execute_hook` IS RE-BOUND TO IT; NO CHROME WAS LAUNCHED, NO FOCUS WAS FORCED AND NO FPS WAS MEASURED; TRACKER-ONLY — THIS PLAN PLUS `plans/README.md` PLUS `plans/Roadmap.md` ARE THE ONLY BYTES WRITTEN; NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE; NO IMPLEMENTATION DISPATCHED; NO MUTATING GIT COMMAND; THE FULL DETAIL IS THE NEWEST `## Latest validation evidence` ROW, WHICH SITS DIRECTLY BELOW THE `<!-- EVIDENCE-APPEND-POINT -->` MARKER):**
  0. **THE VERDICT AND THE SLICE-LEVEL GATE OBJECT:** `GREEN: OK` — **BLOCKERS: NONE** — EXPLICITLY SCOPED TO THE SLICE `P5-S30-damping` ONLY. THE SLICE-LEVEL GATE CONTRACT OBJECT IS `{ "pass": true, "slice_id": "P5-S30-damping", "evidence": { "coverage_summary": { "statements": 98.58, "branches": 96.77, "functions": 100, "lines": 98.58 }, "test_results": "examples/neatRacing/physics/drift.model.test.ts -> 64/64 PASS; examples/neatRacing/physics/drift.smoothness.test.ts -> RED-S30-1 PASS + RED-S30-2 PASS; examples/neatRacing/browser-entry -> 117/117 PASS" }, "fixHint": null, "owner": "05-green-testing" }`.
  1. **THE TWO OWNED RED CONTRACTS ARE GREEN BY NAME:** `RED-S30-1` (THE DAMPING RATIO) AT `examples/neatRacing/physics/drift.smoothness.test.ts:187` **PASSES** AND `RED-S30-2` (THE COUNTERSTEER CONTINUITY) AT `:210` **PASSES**. THE LATER-SLICE REDS ARE **EXPECTED, NOT REGRESSIONS**: `RED-S30-3` (`:264`) AND `RED-S30-4` (`:307`) REMAIN RED **BY DESIGN** (OWNED BY `P5-S30-repin`) AND `RED-S30-5` (`examples/neatRacing/engine/passes/car.rear-slip.test.ts:444`) REMAINS RED **BY DESIGN** (OWNED BY `P5-S30-slip`). NEITHER CLOSED RED HOME WAS RE-AUTHORED OR WEAKENED.
  2. **THE EXACT LEGACY PROTECTIONS HOLD:** EVERY COUPLED CORPUS FLOOR IS **BYTE-IDENTICAL**; THE **ONLY** ASSERTION CHANGED IS THE DAMPING PIN, RE-PINNED BY **EXACT EQUALITY** TO 43.43 AT `examples/neatRacing/physics/drift.model.test.ts:889`; `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER` IS STILL **0.9** (NOW THE RAMP **KNEE**), `NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2` IS STILL **70** (NOW THE **FULL-SCALE REFERENCE**), THE STIFFNESS IS STILL **324**; AND THE **ONLY** PRODUCTION CONSTANT CHANGED IS `NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S` (12.6 → **43.43**).
  3. **THE FIVE MEASURED MARGINS REPRODUCE EXACTLY:** TRAVEL VISIBILITY **7.9881910941984415 DEG** (FLOOR 2.0), TRUE FAR-SIDE OVERSHOOT **EXACTLY 0.0000 DEG** (FLOAT-EXACT), SETTLE **31 TICKS** (WINDOW [31, 33]), MAXIMUM PER-TICK STEP **2.0435494692999328 DEG** (CEILING 2.2), ZETA **1.206388888888889** (FLOOR 0.7). THE COUNTERSTEER RAMP SWEEP IS **MONOTONE** WITH THE KNEE **EXACTLY 0**, FULL LOCK **EXACTLY 1** AND A MAXIMUM ADJACENT JUMP OF **0.70000000000001** AGAINST THE LIMIT OF **7**.
  4. **THE BROWSER OBLIGATION IS N/A FOR THIS SLICE:** CLAUSE (6) FORBIDS THIS LANE FROM TOUCHING ANY BROWSER BYTE AND CLAUSE (7) HANDS THE BROWSER CROSSING RE-BASE TO THE **STEP-30A BOUNDARY LANE `P5-S30A-crossing-rebase`**; THE **VISIBLE-BACKGROUND BROWSER PROOF** IS OWNED BY THE **WHOLE-STEP LANE `P5-S30-green`** (AC-110-GREEN). **NO CHROME WAS LAUNCHED, NO FOCUS WAS FORCED, NO TOPMOST WAS SET AND NO FPS WAS MEASURED.**
  5. **THE VALIDATION LEDGER (TARGETED TESTS ONLY — NEVER A BROAD SUITE):** PHYSICS **EXIT 1 — 2 FAILED / 254 PASSED / 256** (ONLY `RED-S30-3` + `RED-S30-4`, BY DESIGN); ENGINE **EXIT 1 — 1 FAILED / 266 PASSED / 267** (ONLY `RED-S30-5`, BY DESIGN); BROWSER-ENTRY **EXIT 0 — 117/117**; `npx tsc -p tsconfig.neatRacing.json --noEmit` **EXIT 0**; `npm run build:neatRacing` **EXIT 0**; `npm run lint` **EXIT 0** (0 ERRORS, 7 WARNINGS, 2 OF THEM PRE-EXISTING AT `drift.model.test.ts:203`/`:214`); THE PLAN-SYNC VALIDATOR **PASS — 0 ERRORS, 0 WARNINGS, EXIT 0**; THE CONSOLIDATED `slice-advancement` GATE **pass: true, 7 OF 7, `failedGates: []`, `erroredGates: []`, `timedOutGates: []`**.
  6. **THE TIER-3 CROSS-CHECKS (TWO, BOTH READ-ONLY):** `slice-validator` **PASS — NO BLOCKERS**; `coverage-analyst` **NON-BLOCKING** — THE TWO UNCOVERED LINES (875 / 898) ARE **PRE-EXISTING DEFENSIVE-GUARD ARMS**, NOT DEAD CODE, AND MUST NOT BE DELETED IN THIS SLICE.
  7. **THE NON-BLOCKING OBSERVATIONS (DISCLOSED, NOT PATCHED):** THE STALE `zeta = 0.35` TEXT IN THE **GENERATED** `examples/neatRacing/physics/README.md:812` AND IN THE **OUT-OF-SET** `examples/neatRacing/engine/passes/car.pass.ts:501` (CLAUSE (4a) FORBIDS PATCHING EITHER); THE 6-ULP PRINT DELTA ON THE MAX-STEP HEADROOM; AND THE 34 SIBLING-LANE DIRTY FILES THAT MAKE WORKTREE-LEVEL BYTE-IDENTITY UNATTRIBUTABLE (THE PROTECTIONS WERE VERIFIED **PER FILE** AND PER HUNK INSTEAD).
  8. **THE NEXT ACT (SUPERSEDED 2026-11-23 BY THE `## Current state` PAUSE ROW ABOVE — THE ORIGINAL LINE IS ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md`, EVIDENCE BATCH 188, SUB-BLOCK J): THE `04-implementing` DISPATCH OF `P5-S30-repin`, WHICH IS NOW THE ONE `[WIP]` SLICE. THE WHOLE-STEP CLOSE AND THE VISIBLE-BACKGROUND BROWSER PROOF REMAIN OWED TO THE `P5-S30-green` LANE; `P5-S30-slip` AND `P5-S30A-crossing-rebase` REMAIN OPEN; AND STEP 30, STEP 30A, PHASE 5 AND THE PLAN `**Status:**` ALL REMAIN `[WIP]`.
- **[ARCHIVED 2026-11-23 to `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block A) — the SUPERSEDED `## Current state` ROWS (the batch-175 .. batch-180 narrative rows) ARE ARCHIVED BYTE-EXACT. THE OPERATIVE ROWS ARE THE TWO ABOVE: THE 2026-11-23 PAUSE ROW AND THE 2026-11-22 `P5-S30-damping` OPERATIVE GREEN ROW.]**

## Coverage backlog (AC -> step/slice map)

<!-- THE COVERAGE BACKLOG WAS COMPRESSED ON 2026-11-23 (the 01-planning tracker-tidy pass). THE LIVE
     Phase-5 ROWS (AC-104 .. AC-111) STAY IN THIS TABLE. THE Phase-1 .. Phase-4 ROWS (AC-001 ..
     AC-103 — EVERY ONE OF THEM [DONE]) ARE ARCHIVED BYTE-EXACT AT
     `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block B), TOGETHER WITH THE
     ORIGINAL WIDE TABLE HEADER AND THE ORIGINAL INTRO. THE AC-087 .. AC-090 RE-DERIVATION LAW BELOW
     IS UNCHANGED AND WAS NOT COMPRESSED. -->
| AC     | Step | Slice(s) |
| ------ | ---- | -------- |
| AC-104 | 27 | P5-S27-1 + P5-S27-2 + P5-S27-5 (the GREEN 2-HEX START LINE - the USER directive, verbatim: "green 2-hex start line" - a GREEN band TWO hexes DEEP along the track axis (2 x NEATRACING_ROAD_EDGE_HEX_DIAMETER_M = 4.0 m) spans the FULL road width and is VISIBLE in the rendered track; the band is a NEW ALONG-TRACK classification and NOT a fifth TileColorway member, because the EXPORTED TileColorway surface stays at its FOUR members (a COMPILE-TIME type-level assertion - colorwayToColors + COLORWAY_PRECEDENCE are MODULE-PRIVATE and are NOT importable, so their UNCHANGED-ness is an EXPORT-SURFACE + BEHAVIORAL invariant) and the road-edge pins stay green; the band uses the EXISTING green fill + edge color tokens; the RED lane measures the CURRENT band classification BEFORE any production edit and the GREEN lane verifies it BY NAME) |
| AC-105 | 27 | P5-S27-1 + P5-S27-3 + P5-S27-5 (the SIDE-BY-SIDE RACERS FULLY BEHIND THE GREEN START LINE - the USER directive, verbatim: "side-by-side racers", AMENDED by the 2026-11-12 USER clarification, verbatim: "every car, including player and rival, must be side by side FULLY BEHIND the green start line, not on/above it" - the fleet - the PLAYER car 0 AND every RIVAL - is placed SIDE BY SIDE on ONE shared along-track station that is FULLY BEHIND the green start line instead of LINE-ASTERN: the station is UPSTREAM (a NEGATIVE along-track offset in the walk frame), NEVER on the line (offset 0) and NEVER on/above it (a positive offset); the PLANE is `splineSamples[0]` (segment 0's start point - the AC-104 band's arc-0 origin AND the lap-detector plane), so in the closed-loop arc frame the upstream direction is the lap TAIL (`arcM = loopLengthM - clearance`); the CLEARANCE is DERIVED, never spelled out: `NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M = NEATRACING_VOXEL_CAR_LENGTH_M` (4.5 m, the ONE published car length in the example, at examples/neatRacing/voxel/lod.distance.utils.ts:48), measured to the car's CENTRE, so the nose stands 2.25 m (half a car length) strictly upstream of the plane, the tail 6.75 m upstream, and the WHOLE footprint is behind the line and can NEVER overlap the forward green band [0, 4.0] m; the placement is a PURE function of (trackSpec, agentCount) with NO RNG; every pose stays INSIDE the published ASYMMETRIC half-width (resolveTrackEdgeHalfWidthM + resolveTrackEdgeSideAt) READ AT THE SPAWN STATION (the half-width is station-dependent - the road widens on the OUTER side of a corner); the existing fleet-pose pin (resolveSimWorkerFleetSpawnPoses(trackSpec, 3)) is RE-PINNED to the side-by-side-behind-the-line expectation IN THE SAME SLICE (no deferred cleanup, no dual path) and the RED home is re-opened (P5-S27-1) to re-pin the RED-S27-2 clause BEFORE the implementation; the line-astern NEATRACING_SIM_WORKER_SPAWN_ALONG_LEAD_M (2 m) + NEATRACING_SIM_WORKER_SPAWN_GRID_PITCH_M (12 m) are RETIRED IN THE SAME SLICE (no deferred cleanup); the NEATRACING_PHYSICS_SPAWN_LATERAL_SPREAD_M path is NOT revived - it stays a LIVE FALLBACK (declared at physics.step.service.ts:433, read at :1388 inside resolveSpawnLateralOffset :1386, reached from the `pose?.x ?? resolveSpawnLateralOffset(resolvedSeed)` fallback at :552) that the explicit worker pose BYPASSES on the live path; the side-by-side grid is placed by the worker's explicit pose, NOT by that fallback) |
| AC-106 | 27 | P5-S27-1 + P5-S27-4 + P5-S27-5 (the FLAPPY-STYLE 3-2-1-GO COUNTDOWN - the USER directive, verbatim: "Flappy-style 3,2,1,Go countdown" - the HUD shows 3, then 2, then 1, then GO BEFORE the green; the clock is a PURE resolver over raceState.elapsedS (NEVER Date.now()); the host input gate HOLDS the throttle until GO; the text uses the two-pass neon recipe (the glow pass then the crisp pass) over the DOM overlay surface; the countdown state is added to the read model WITHOUT changing any published physics state; NO countdown EXISTS in examples/flappy_bird today - only the fading centered startup legend - so the recipe is lifted, not copied) |
| AC-106-R1 | 27A | P5-S27A-red + P5-S27A-overlay + P5-S27A-green (TRANSPARENT ASTROBIRD-LIKE OVERLAY: throughout 3, 2, 1 and GO, only the countdown text/glow treatment overlays the live race scene; overlay/backdrop surfaces stay transparent and the scene remains visible; NO black, opaque, or full-screen dimming backdrop during any phase, on GO/dismissal, or afterward. Validation: `npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown"`) |
| AC-106-R2 | 27A | P5-S27A-red + P5-S27A-overlay + P5-S27A-green (CLEAN DISMISSAL: after GO/dismissal the countdown overlay and any backdrop are removed or hidden, leave no black/opaque remnant or persistent input-capturing layer, and leave the unchanged race scene visible and usable. Validation: `npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown"`) |
| AC-106-R3 | 27A | P5-S27A-red + P5-S27A-overlay + P5-S27A-green (SIMULATION RESUMES: before GO the established human throttle HOLD still applies; at GO it releases, post-GO throttle reaches the existing worker-handoff seam, and the simulation observably advances after the countdown rather than remaining stopped; prove BOTH pre-GO hold and post-GO release/progression so a permanently stopped simulation cannot pass. Validation: `npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown|examples/neatRacing/browser-entry/browser-entry"`) |
| AC-107 | 28 | P5-S28-red + P5-S28-law + P5-S28-repin + P5-S28-green (the TRACK-DEPARTURE FIX - the USER directive, verbatim: "fix track departure" - the car does NOT wander off the track through the corners; GAP-5-04: applyTrackContainment() is a deliberate NO-OP (the AC-090 VERDICT 8 law) and the off-road band IS drivable (the 20 km/h cap), so the departure fix is PURELY ADVISORY per the recorded default Q3 (a HUD "return to the track" hint + the existing off-track penalty, NO auto-reset and NO teleport); the RED lane measures the CURRENT departure rate BEFORE any production edit) — **[CLOSED 2026-11-22 - 05-green-testing GREEN: OK: the whole-step GREEN lane `P5-S28-green` returned GREEN: OK (all eight allow-listed commands PASS, the two Step-28 RED homes are GREEN BY NAME and were NOT re-authored, the consolidated `slice-advancement` gate PASSED 4 of 4 with `failedGates: []`, the visible-browser smoke PASSED under the standing USER RULE); the completed detail is archived at logs evidence batch 183]** |
| AC-108 | 29 | P5-S29-red + P5-S29-neon + P5-S29-repin + P5-S29-green (the NEON DROPDOWN TEXT - the USER directive, verbatim: "neon dropdown text" - the option text of the CLOSED select carries the neon treatment; GAP-5-05: the native <option> popup is OS-rendered and IGNORES text-shadow/filter, so the neon lands on the CLOSED <select> per the recorded default Q2 (a CSS glow + a neon CLASS hook), and a custom non-native listbox is OUT OF SCOPE; there is NO neon technique in neatRacing today and NO shared neon-text helper exists - Step 29 authors its OWN helper module (and its OWN tokens: the repo has NO shared neon-TOKEN module either - the two neon palettes are DUPLICATED value-identical copies, FLAPPY_NEON_PALETTE at examples/flappy_bird/constants/constants.palette.ts:9 - EXAMPLE-LOCAL by declaration - and NETWORK_NEON_PALETTE at examples/shared/network-visualization/network-view/network-view.constants.ts:115, consumed only under examples/shared/); the Neatenstein WAVE-announcement precedent is the in-repo reference; the bloom pass samples the WebGL FBO only and CANNOT serve DOM text) — **[CLOSED 2026-11-22 - 05-green-testing GREEN: OK: the 01-planning STEP-29 CLOSURE PASS CLOSED THIS STEP — all four slices are `P5-S29-red` ([DONE 2026-11-22 - 03-red-testing (Tier 1) RED AUTHORED]), `P5-S29-neon` ([DONE 2026-11-22 - 05-green-testing GREEN: OK] — GREEN-LIT BY ITS OWN FRESH INDEPENDENT-VERIFICATION INSTANCE ON 2026-11-22 (`green-light: true` EXPLICITLY SCOPED TO `P5-S29-neon` ONLY, BLOCKERS NONE) AND THE 05-green-testing LANE RETURNED `GREEN: OK` — EVERY ALLOW-LISTED COMMAND PASS, THE RED HOME `examples/neatRacing/host/hud.neon.test.ts` IS GREEN BY NAME (4/4) AND WAS NOT RE-AUTHORED, THE FULL-SEVERITY CONSOLIDATED `slice-advancement` GATE PASSED 7 OF 7 WITH `failedGates: []`, AND THE VISIBLE-BACKGROUND BROWSER PROOF PASSED UNDER THE STANDING USER RULE), `P5-S29-repin` ([DONE 2026-11-22 - 05-green-testing GREEN: OK] — THE TRACK-SELECT ADOPTION ONTO THE SAME ONE HELPER, WITH THE GRANTED TRACK-SELECT NEON WITNESS GREEN BY NAME) and `P5-S29-green` ([DONE 2026-11-22 - 05-green-testing GREEN: OK] — THE WHOLE-STEP GREEN LANE: EVERY ALLOW-LISTED COMMAND PASS, THE RED HOME IS GREEN BY NAME, THE TRIVIAL FOUR-SUB-GATE CONSOLIDATED `slice-advancement` GATE PASSED 4 OF 4 WITH `failedGates: []`, AND THE VISIBLE-BACKGROUND BROWSER PROOF PASSED UNDER THE STANDING USER RULE); THE COMPLETED DETAIL IS ARCHIVED AT LOGS EVIDENCE BATCH 186; the RECORDED DEFAULT Q5 = EVERY closed neatRacing `<select>` receives the neon through the ONE helper (the native popup option rendering stays OUT OF SCOPE and the cross-example palette consolidation stays OUT OF SCOPE); the RECORDED DECISION (2026-11-22) = THE INLINE NEON CORE COLOUR IS INTENTIONAL, BECAUSE IT GUARANTEES THE IDENTICAL CLOSED-SELECT RENDERING ON BOTH SELECTS REGARDLESS OF THE STYLESHEET CASCADE STATE]** |
| AC-109 | 30 | P5-S30-red + P5-S30-damping + P5-S30-repin + P5-S30-slip + P5-S30-green (the SMOOTH NO-WOBBLE DRIFT - the USER directive, verbatim: "smooth no-wobble drift" - the four wobble sources of GAP-5-06 are addressed: the under-damped visual pose spring (zeta 0.35), the bang-bang countersteer (the 70 m/s2 lateral impulse behind the hard |steer| >= 0.9 gate), the per-tick slip re-derivation with NO depth hysteresis, and the absence of steering smoothing / yaw-rate feedback; the AC-087 "never mirrors" invariant and the UNCLAMPED pose spring are PRESERVED; the drift model stays PURE (the drift.model.ts law) and the ai/oscillation-penalty.utils.ts fitness domain is NOT coupled) — **[WIP 2026-11-22 - the 01-planning STEP-29 CLOSURE PASS AUTHORED THE STEP-30 PACKET AND PROMOTED `P5-S30-red` TO THE ONE [WIP] SLICE: the FIVE slices are `P5-S30-red` (the ONE [WIP] slice — the RED lane; the TWO NEW red homes are `examples/neatRacing/physics/drift.smoothness.test.ts` and `examples/neatRacing/engine/passes/car.rear-slip.test.ts`), `P5-S30-damping` (the zeta-0.35 pose spring + the bang-bang countersteer in `examples/neatRacing/physics/drift.model.ts`), `P5-S30-repin` (the per-tick slip re-derivation hysteresis + the steering smoothing / yaw-rate feedback in `examples/neatRacing/physics/physics.step.service.ts`), `P5-S30-slip` (the rear wheels yaw with the PUBLISHED slip in `examples/neatRacing/engine/passes/car.pass.ts` + the browser seam in `examples/neatRacing/browser-entry/browser-entry.ts`) and `P5-S30-green` (the whole-step GREEN lane); the step `pre_execute_hook` is RE-BOUND to `P5-S30-red`; THE PACKET IS AUTHORED, NOT GREEN-LIT AND NOT DISPATCHED]** |
| AC-110 | 30 | P5-S30-red + P5-S30-slip + P5-S30-green (the VISIBLE REAR SLIP - the USER directive, verbatim: "with visible rear slip" - GAP-5-07: the rear wheels are yawed by EXACTLY 0 today (the whole body + all four wheels share ONE visual offset) and there are NO skid marks, NO smoke, NO tire trails and NO particles; the renderer MUST consume the PUBLISHED slip value verbatim (the drift model publishes atan2(lateralVel, speed)); the new visual channels stay OPTIONAL; the browser debug seam is pinned to EXACTLY EIGHT fields and is NOT widened) — **[WIP 2026-11-22 - THE `P5-S30-render` SLICE IS FOLDED INTO `P5-S30-slip` UNDER THE HARD FIVE-SLICES-PER-STEP CAP (`MAX_SLICES_PER_STEP = 5` in `scripts/agent-customization/gates/plan-slice-quality.gate.mjs`): the AC-109 + AC-110 union is SIX unique slice names (red, damping, repin, slip, render, green) and the cap is FIVE, so THE RENDER WORK (the rear-wheel yaw + the renderer consumption + the debug seam) IS THE `P5-S30-slip` SLICE. THE AC-109 ROW IS THE RICHER CONTRACT AND ITS SLICE LIST IS CORRECTED IN THE SAME PASS TO NAME ALL FIVE SLICES (IT NOW NAMES `P5-S30-slip` AS WELL), WHILE THIS ROW'S SLICE LIST IS RE-POINTED AT THE FOLDED `P5-S30-slip`, AND THE FOLD IS DISCLOSED HERE AND IN THE STEP-30 PROSE. THE NEW VISUAL CHANNELS (skid marks / smoke / tire trails / particles) STAY OPTIONAL AND ARE **NOT** TAKEN IN THIS STEP — THE RECOMMENDED VISIBILITY IS THE REAR-WHEEL YAW ITSELF]** |
| AC-111 | 31 | P5-S31-red + P5-S31-law + P5-S31-ai + P5-S31-green (the TRACTION LAW - the USER directive, verbatim: "traction 100% normally, falling with speed/tightness, fun arcade spins and AI challenge" - the traction is 100% at a straight-line speed below NEATRACING_DRIFT_MIN_SPEED_MPS and at any zero-demand turn, and FALLS as a MONOTONIC function of (the speed x the demanded turn rate x the tightness); the "tightness" is a NEW OPTIONAL DriftStepInput field computed by the SERVICE (the drift model stays pure and never reads the track); the arcade spin is a SEPARATE NAMED over-rotation branch that only arms above STEER_ENGAGE_THRESHOLD (0.5) and stays inside the MAX_SLIP_ANGLE_RAD (0.45) envelope; the AI CHALLENGE is a FITNESS term (the shaped fitness gains a slip/drift term), NOT a behaviour change; the RATIFIED GRIP_FALLOFF_THRESHOLD_RAD (0.2) + GRIP_FALLOFF_RATE_PER_RAD (2) stay the anchors) |

### PHASE-4 RE-OPEN NOTE (2026-11-07, the USER VERDICT via the 01-planning PHASE-4 AUTHORING pass)

The USER VERDICT (2026-11-07) declared the plan NOT complete - the AI core is missing - and RESCINDS the Phase-3 TERMINAL verdict + the plans/completed/ archive-pair move. The Phase-2 [DONE] holds on AC-027..AC-034 + AC-043 are validated at the MODULE level only; the LIVE integration never landed (the `## Phase-4 AUDIT` section holds the file:line audit: the worker boots manual-always, the fleet runs the scripted autopilot in BOTH worker modes, the eval-worker session is booted by no host, the goal machine is fed nodeCount = 0, one pinned track, the visualizer rides the idle stub). The re-opened live-lane ACs (AC-027..AC-034 + AC-043) are routed to the Phase 4 steps 23, 23B, 24-26 above (the re-open note rides the plan, NOT the backlog rows - the rows above keep their phase 1/2/3 step maps); the FOUR NEW ACs are AC-100..AC-103 (steps 25/26). The next planning boundary is the fresh VERIFICATION instance green-light on the P4-S1-red staged step, then the 04-implementing dispatch (author -> verify loop; no self-dispatch).

### AC-087..AC-090 RE-DERIVATION (2026-10-27, verdict-4 + verdict-5, 01-planning AUTHOR)

**Scope of this subsection.** The author_handoff item 5 of the `verdict5-merged-gap-map` record: _keep AC-087..AC-090 semantics and re-derive their numeric pins from the equilibrium + the manual-mode rule_. The four SEMANTICS are unchanged (slide prominence with a recorded feel direction, a measurable traction signal, a no-regression ledger, and a user visual gate). The NUMERIC PINS are re-derived, because every pin that was written against the pre-wave model, against the AI-mode page, or against a transient reading is either vacuous or unreachable after the re-wiring. This subsection is the single home of the re-derivation; the row text above only points here.

**The two rules that drive every re-derived pin (the merged-map `acceptance_rules_for_the_author`):**

- **R-MANUAL (the measurement lane):** every slide, camera and traction measurement is taken with the page in **MANUAL** mode and the car **ON THE ROAD**. The ai-mode page is never acceptable evidence (R-M3). The page used to boot in AI mode (`INITIAL_RACE_MODE = MODE_OPTION_AI`, `examples/neatRacing/host/mode-selector.ts:63`), and the autopilot only crosses the `|steer| >= 0.5` engage cliff on **176 of 7200 ticks (2.44 %)** — so the AI page barely ever produces the slide the user is judging. The off-road excursion is the second mask: the demo track puts the car **OFF-ROAD at tick 280 at 82.3 m/s** (921 of 1200 ticks), where the `22.25 m/s` floor pins the car at `80.1 km/h` and the off-road deceleration is **32 : 1** the drift scrub — the slide is arithmetically erased. The re-wiring therefore starts with the two masks: the boot mode (Step 14P, slice `P1-drift-rewire-boot-mode`) and a ROAD-ONLY surface classification (Step 14P, slice `P1-drift-rewire-road-surface`).
- **R-EQUILIBRIUM (the deficit pin):** the speed deficit is pinned at the **EQUILIBRIUM 32.911 km/h** (the envelope audit), **never** at a transient 600-tick reading (which reads `32.555 km/h`). The repeat-offence in this plan has been to pin a transient; the re-derived contract waits for a convergence criterion and then measures.

**AC-087 (slide prominence) — semantics retained, pins re-derived.** The semantics are retained exactly: a **named** slide-prominence floor; the feel **direction decided and recorded in this plan before any production edit** (a bigger slide — decision D-11); the user-approved body-animation **shape** retained (the second-order pose spring with the steer-articulation term); and a sustained-slide contract. The re-derived pins:

1. The floor stays asserted on the **deterministic unit proxy** (`computeDriftSlideProxyRad(slipAngleRad)`), and the browser reading stays a loose **presence** observable (`driftFeelSlideProminence`) that is never numerically compared against a constant. (Unchanged — this pin was already correct.)
2. **The slide DEPTH becomes SPEED-PROPORTIONAL.** The measured defect is that the committed slide is the **same 0.45 rad at 5 m/s and at 89 m/s** (the envelope audit): the entry ramp is a fixed `3.5 rad/s` (speed-independent) and the clamp is reached in 8 ticks at every speed. The re-derived contract: the depth at the top speed is at least `0.45 rad` **and** the depth at a low speed is at most `0.20 rad` (the exact split is a RED decision, `P1-drift-law-red` / RED-2). This is the physics half of the user's "the tires should lose traction also proportionally to the speed".
3. The **sustained-slide window** is measured **on the road, in MANUAL mode** (the off-road lane pins the car at 80.1 km/h and masks the slide).
4. The **speed deficit** is pinned at the **equilibrium 32.911 km/h**, never at the transient.
5. The camera and the body read the **same published slip** (the one shared scalar), so the frame can never be internally inconsistent — a travel-locked camera with a crabbing body is the exact defect the audits forbid. The camera aim is the **travel heading published on the physics state** (the new named field `travelHeadingRad`), and the body yaw source is switched from the **committed** slip to the **published** slip (the call site `examples/neatRacing/physics/physics.step.service.ts:457`).
6. **The 3D VIEW must MOVE.** A new red contract (`P1-drift-rewire-red` / RED-4, and the W4b contract A12) asserts the **camera basis** — the look target and the forward vector — differs from the nose-aligned values by the travel divergence, and is **bit-identical to the pre-wave basis** when the published slip is zero. The assertion never targets the seam or a channel value: the A1–A11 contracts and all 43 fixture checks can pass with a **frozen** camera, which is the trap the F3 finding names.

**AC-088 (traction) — semantics retained, substance corrected.** The semantics are retained: the tire traction loss is a **measurable, non-degenerate** contract rather than a feel claim. The substance is corrected, because the old contract measured two things that are not traction at all:

| Old measured quantity                                   | Why it is not a traction signal                                    | Re-derived quantity                                                                       |
| ------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| a **fixed-sun lighting band** (the tire gradient, G-V4) | it is a lighting term, not a friction term                         | the per-tire grip and the tire gradient are driven by the **published traction severity** |
| the **raw steer echo** (the front-wheel yaw, G-V5)      | it echoes the input, not the physics                               | the front-wheel yaw is driven by the **physics state**                                    |
| the **committed slip** (the crab, G-V3)                 | the traction law reads the **published** slip, so the two disagree | **one** shared scalar, published on the state as `tractionSeverity`                       |

The re-derived contract: the severity is a **named function of the speed and the published slip** (the user law S-2/S-3), published on the state, `0` on a straight line, `1` at the full slide, **monotonic** in the slide depth, and computed on the **post-traction `car.speed`** — the same basis the traction law uses (the F11 rule: the basis must match, or the camera and the tires disagree). **No consumer re-derives it**: the car pass, the HUD, the camera and the debug seam all read the same field.

**AC-089 (no regression) — preserve set retained, re-baselined set extended.** The **PRESERVE SET** is unchanged and is never relaxed: AC-047A..AC-047D (the AC-047D byte-identity is re-anchored on the **production serializer** — a stronger contract than the whole-object `JSON.stringify` pin), the determinism twin-runs, the straight-line exact zeros, the low-speed exact-zero gate, AC-078 and its tyre-follow tolerance, and the 24 non-drift fixture checks. The **RE-BASELINED SET** gains, each with exactly **one ledger row** whose single home is **this plan** (the pin, the old baseline, the new baseline, the **measured** value, the owner file):

| Pin                                                                                                                                                                                                                                     | Invalidated by                                                                                                    | Owner slice                                                                                                                                                                                                                                                                         | Home                               |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| the physics state shape (the new `travelHeadingRad` and `tractionSeverity` fields)                                                                                                                                                      | the state contract change                                                                                         | `P1-drift-rewire-travel-camera`, `P1-drift-traction-scalar`                                                                                                                                                                                                                         | the ledger in this plan            |
| the **DECLARED POSE ENVELOPE** (re-derived from the **published** slip ceiling, so the `0.002 rad` published overshoot can never breach it)                                                                                             | the body-yaw source change + the re-derived gain                                                                  | `P1-drift-rewire-travel-camera` (source), `P1-drift-law-proportional` (the gain and the envelope)                                                                                                                                                                                   | the ledger in this plan            |
| the body yaw gain `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` (the `0.5` cosmetic gain; the RED-decided value is **`0.45`** — RE-PINNED 2026-10-27 by the 01-planning PATCH from the INFEASIBLE `0.75` recommendation: the anti-null guard ` | GAIN - 1                                                                                                          | >= 0.5`caps GAIN at 0.5, the 0.2 rad prominence floor against the 0.45 rad max slip forces`GAIN >= 0.4444`below 1, and the pose settle band [30, 34] ticks needs`GAIN >= ~0.386`, so the FEASIBLE WINDOW is [0.386, 0.5] and 0.45 is chosen inside it — the guard still passes at ` | 0.45 - 1                           | = 0.55 >= 0.5`; a slight animation, never a full tilt) | the proportional law | `P1-drift-law-proportional` (the production gain) + `P1-drift-law-repin` (the FIVE gain-coupled TEST literals) | the ledger in this plan |
| the **speed-blind grip budget** (the ratified `[RED-G2]` pin: the grip factor is EQUAL at speed 1 and at speed 0.2)                                                                                                                     | the proportional law — the pin is **re-opened** (deleted and replaced in the same change, never silently relaxed) | `P1-drift-law-red` / `P1-drift-law-proportional`                                                                                                                                                                                                                                    | the ledger + the **contested set** |
| the `MIN_SPEED 15 m/s` cliff and the speed-independent `3.5 rad/s` entry ramp                                                                                                                                                           | the proportional law                                                                                              | `P1-drift-law-proportional`                                                                                                                                                                                                                                                         | the ledger in this plan            |
| the `0.5` engage threshold (its **existence** is ratified; only its **binarity** is contested, G-M9)                                                                                                                                    | the proportional law (the commit gate stays)                                                                      | `P1-drift-law-proportional`                                                                                                                                                                                                                                                         | the ledger in this plan            |
| the `0.45 rad` commit clamp                                                                                                                                                                                                             | the proportional law                                                                                              | `P1-drift-law-proportional`                                                                                                                                                                                                                                                         | the ledger in this plan            |
| the `CarPassCarState` contract (the traction channel) and the `NeatRacingHudState` contract (the traction readout)                                                                                                                      | the presentation wave                                                                                             | `P1-drift-presentation-car`, `P1-drift-presentation-hud`                                                                                                                                                                                                                            | the ledger in this plan            |
| the browser drift seam field count (`5 → 6 → 7 → 8`) and the fixture check count (`43 → 44 → 45 → 46`)                                                                                                                                  | each wave step adds exactly **one** field and exactly **one** check                                               | the three green slices                                                                                                                                                                                                                                                              | the ledger in this plan            |

**AC-090 (the user visual gate) — the verdict is REJECT and the gate is OPEN.** The **semantics** of AC-090 are retained in full: the automated contracts are necessary but not sufficient (a FEEL verdict needs a human), the gate asks the **six verbatim questions** transcribed from `plans/neatracing-voxel-racer.plans.md (the folded fixture-gate section FOLD-1)` section 5.3, the verdict is recorded on **three surfaces** (this plan, `plans/README.md`, `plans/Roadmap.md`), the verdict is bound to the **identified bundle** (byte length + sha256), and a **REJECT re-opens** the step instead of closing it. The **re-derived** part is the state of the gate itself:

- **VERDICT 2026-10-27 = REJECT** (verdicts 4 and 5). The step is **NOT** closed, the step stays `[WIP]`, and the six questions are **RE-ASKED** against the re-wired build, in **MANUAL** mode, on the **ROAD**.
- The earlier `fix-packet-P1-drift-feel-green-iteration-4` close-out was scoped to the **HARNESS** (the fixture's own instrumentation — the one-line `resetFrameCapture()` per-frame call) and was **NEVER an AC-090 pass**. It must not be read as one, in this plan or in `plans/README.md` or `plans/Roadmap.md`.
- The deterministic verdict rule is **unchanged** (ACCEPT iff Q2 == CRISP and Q3 == CATCHES and Q4 == NO and Q5 == YES and Q6 == YES and Q1 in {YES, TOO MUCH}, with the envelope check green when Q1 == TOO MUCH; a REJECT follows on a Q2/Q3/Q6 failure, on Q1 == NO, or on Q5 == NO).
- **Who may record the verdict:** the **user**, or a fresh verification instance that is **explicitly authorised** to record the user verdict. An implementation slice and a green-testing slice **may not** record an AC-090 pass — that is a plan violation, and the Step 14R green slice carries this prohibition explicitly.

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

### Phase 1 — Graphics engine + manual keyboard driving [DONE 2026-11-05 — the AC-090 FULL gate ACCEPTED by VERDICT 10 (the USER's six §5.3 answers satisfy the DETERMINISTIC rule); Phase 1 CLOSED — the Phase-1 compression, the Cortex rebuild and the Phase 2 staging follow]
**Done record:** EVERY STEP [DONE] — the graphics engine, the manual keyboard driving, the voxel pipeline + the LOD, the car materials + the wheels, the dev server + the editor, the track + the physics + the race rules + the HUD, the boot-glue render path, the composition correctness, the turn + drift waves, the right-world decoration + crowd waves, the drift-feel sharpening wave, the drift re-wiring waves and the AC-090 verdict waves. THE AC-090 FULL GATE ACCEPTED THE PHASE (VERDICT 10). THE FULL PHASE PACKET, EVERY STEP PACKET AND THE PHASE PROSE ARE ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block C); THE HISTORICAL EVIDENCE ROWS ARE AT EVIDENCE BATCH 188, SUB-BLOCK D.

### Phase 2 — AI migration (NGE) [DONE 2026-11-07 — the Step-05 close verified Phase 2 holds ZERO [WIP] steps (Steps 01-05 ALL [DONE]); Phase 2 CLOSED — the Phase-2 close compression (logs evidence batch 164), the Phase-2 evidence-ledger disposition and the Phase 3 staging per the `next_phase` mandate follow]
**Done record:** EVERY STEP [DONE] — Step 01 (the simulation worker + the packed frames + the reproducibility tuple), Step 02 (the eval worker + the observation assembler + the scripted rivals), Step 03 (the NGE growth + promotion part 1), Step 04 (the NGE growth + promotion part 2) and Step 05 (the mode integration + the network visualizer). THE PHASE HOLDS **ZERO** `[WIP]` STEPS. THE FULL PHASE PACKET, EVERY STEP PACKET AND THE PHASE PROSE ARE ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block C); THE HISTORICAL EVIDENCE ROWS ARE AT EVIDENCE BATCH 188, SUB-BLOCK D.

### Phase 3 — Polish + deprecation prep [DONE 2026-11-07 — the ORCHESTRATOR-DIRECTED 07-logging PHASE-3 CLOSE (batch-158 phase-close shape); BOTH steps CLOSED [DONE 2026-11-07] via the [COMPRESSED] pointers + the retained evidence rows; the phase archives discharged to logs evidence batch 166; terminal per the next_phase mandate (LD-33)]
**Done record:** BOTH STEPS [DONE] — Step 01 (the adaptive quality + the memory budget) and Step 02 (the build + the docs + the deprecation prep). THE FULL PHASE PACKET, EVERY STEP PACKET AND THE PHASE PROSE ARE ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block C); THE HISTORICAL EVIDENCE ROWS ARE AT EVIDENCE BATCH 188, SUB-BLOCK D.

### Phase 4 — AI-core integration (NGE staged evolution drives the racer) [DONE 2026-11-11 — the ORCHESTRATOR-DIRECTED 07-logging PHASE-4 CLOSE (the batch-158 phase-close shape): all TEN steps [DONE] (Steps 23, 23B, 24, 24B, 24C, 24A, 25, 25A, 26, 26A); the NINE open packets compressed IN PLACE to the [COMPRESSED] pointers (Step 23 stays byte-exact-archived at logs batch 167); the close record = evidence batch 168 in `plans/neatracing-voxel-racer.logs.md`; the NEXT boundary is the USER verdict re-test, THEN terminal (LD-33 out of scope)]
**Done record:** ALL TEN STEPS [DONE] — Steps 23, 23B, 24, 24A, 24B, 24C, 25, 25A, 26 and 26A (the NGE staged evolution, the AC-029 no-dual-path closure, the concurrent evolving agents + the live ladder, the harness fixes for the user AI-mode bug wave, the user AI-mode bug fixes, the W2 debug rung seam, the multiple tracks, the two docs-regen packets and the AI-mode live visualizer). THE FULL PHASE PACKET, EVERY STEP PACKET AND THE PHASE PROSE ARE ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block C); THE HISTORICAL EVIDENCE ROWS ARE AT EVIDENCE BATCH 188, SUB-BLOCK D. LD-33 (the `examples/racing_curriculum` removal) STAYS OUT OF SCOPE.

### Phase 5 — Race presentation + arcade driving feel [WIP]

**Status note:** AUTHORED 2026-11-12 by the 01-planning (AUTHOR) instance (NOT green-lit, NO execution dispatched); the heading above carries the BARE `[WIP]` token per the PHASE_PATTERN law (`scripts/agent-customization/mcp/mcp-plan-utils.mjs:39-40`, which requires the heading to END with the bare status token) and THIS paragraph carries the annotation — the plan's own F1 remedy (plan line 7615). The pass was driven by the USER directive (verbatim): "Update plans/neatracing-voxel-racer.plans.md: compact DONE details into its .logs.md. Add one WIP phase/step with bounded slices, acceptance and validation for: green 2-hex start line, side-by-side racers; Flappy-style 3,2,1,Go countdown; fix track departure; neon dropdown text; smooth no-wobble drift with visible rear slip; traction 100% normally, falling with speed/tightness, fun arcade spins and AI challenge. Inspect code/Flappy, do not implement. Record gaps, evidence and next boundary." NOTHING WAS IMPLEMENTED in this pass: the neatRacing sources and the Flappy example were INSPECTED READ-ONLY (the four-scout evidence + the gap list are archived at logs evidence batch 169), and ONLY the plan + the logs + the registration bytes changed. NO green light is recorded here and NO execution is dispatched — a FRESH 01-planning (VERIFICATION) instance must record `green-light: true` in `## Latest validation evidence` before the Step-27 packet dispatches (the author does NOT grant the green light and MUST NOT self-dispatch a patch cycle).

**Phase objective:** (the six USER-named gameplay desires, ONE wave) the racer must READ like a race — (1) a VISIBLE green 2-hex start line painted ACROSS the road, (2) the field SIDE BY SIDE on that line, (3) a Flappy-style 3-2-1-GO countdown before the green, (4) the car must NOT wander off the track, (5) the dropdown text must be NEON, (6) the drift must be SMOOTH (no wobble) with a VISIBLE rear slip — and the traction law must be 100% normally, FALL with the speed and the tightness, keep the arcade spin FUN and keep the AI CHALLENGED.
**THE LIVE-STATE CORRECTION (COUNTDOWN-FIRST USER AMENDMENT — TRACKER-ONLY):** the authoring record above is historical. Step 27A is now the ONLY `[WIP]` step and `P5-S27A-red` its ONLY active slice. Step 30 and `P5-S30-repin` are `[PLANNED]`; the earlier green lights remain limited to their named closed slices and the `P5-S30-repin` verdict remains **BLOCKED**, not cleared. No execution is authorized until a fresh `01-planning` verification records a green light for Step 27A. The first production implementation must be `P5-S27A-overlay`; no Step-30 drift implementation may run first.


**The slice-boundary law for this phase (the at-most-FIVE-slices mandate is a MANDATE — the 14Q/14Q2 SPLIT precedent):** the phase now carries EIGHT bounded steps: Steps 27..32, the NEW Step-27A countdown-repair step, and the existing Step-30A prerequisite boundary. Step 27A reopens only the AC-106 presentation/resume defect and has three slices (RED, implementation, GREEN); it does not reopen the closed start-line/grid work or any drift behavior. Every step keeps at most five slices and every slice at most 4 h and three code/test files. Steps 27, 28 and 29 remain CLOSED; Step 27A is the ONLY `[WIP]` step; Steps 30, 30A, 31 and 32 are `[PLANNED]` and not dispatched. The existing Step-30 blocked verdict and six-edit author amendment remain queued behind Step 27A.

**The Phase-5 step map:** Step 27 = start line + side-by-side grid + original countdown (AC-104..AC-106) — **[DONE]**, archived at logs evidence batch 179; **Step 27A = the transparent Astrobird-like countdown repair, clean dismissal and simulation resume (AC-106-R1..R3) — [WIP], the ACTIVE STEP, active slice `P5-S27A-red`, not yet green-lit**; Step 28 = track-departure fix — **[DONE]**, archived at logs evidence batch 183; Step 29 = neon dropdown — **[DONE]**, archived at logs evidence batch 186; **Step 30 = smooth drift + visible rear slip — [PLANNED] behind Step 27A, with `P5-S30-repin` [PLANNED] and its independent verdict still BLOCKED**; Step 30A = browser crossing re-base — **[PLANNED]** after its existing Step-30 dependency; Step 31 = traction law — **[PLANNED]**; Step 32 = docs regeneration — **[PLANNED]**. The stale `P5-S30-damping`-as-active wording is retired: that slice is DONE, and neither it nor `P5-S30-repin` is the current frontier.

**Stop conditions:**

- **Done:** Phase 5 Steps 27-32 DONE with AC-104..AC-111 green on the code lanes; the USER feel verdict recorded on the served build.
- **Blocked:** any step packet fails its gate after two fix attempts — record the blocker in `## Latest validation evidence` and escalate via `00-cross-tier-helper`; NEVER widen a slice past 5 files / 4 h.

**Required validation:**

- `npm run jest:base -- --testPathPatterns="examples/neatRacing/track|examples/neatRacing/physics|examples/neatRacing/workers|examples/neatRacing/host"`
- `npx tsc -p tsconfig.neatRacing.json --noEmit`
- `npm run build:neatRacing`
- `npm run lint`
- `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md`
- Consolidated slice gate per step: `neataptic-gate-mcp:run_gate_check gate=slice-advancement --slice-id=<slice_id> --changed-files=<comma-separated files>`

```yaml
phase: 5
title: 'Race presentation + arcade driving feel'
status: '[WIP]'
goal: planning
expansion: steps
auto_expand: false
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_phase: 'Terminal after Phase 5 - the racing_curriculum removal stays the separate user-confirmed workstream (LD-33)'
skills:
  - plan-alignment
  - planning-acceptance-criteria
  - spec-checklist
  - tracker-handoff
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/track|examples/neatRacing/physics|examples/neatRacing/workers|examples/neatRacing/host"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'npm run build:neatRacing'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated)'
constitution_check: 'principle-4-breadth-first-recoverable; principle-5-unique-ids'
acceptance_criteria:
  - 'AC-PLN-012: the phase carries EIGHT bounded steps (Steps 27..32 PLUS Step 27A for the countdown repair and Step 30A for the browser crossing re-base); Step 27A has THREE slices and Step 30A has ONE; every step has at most FIVE slices and every slice at most 4 h and three code/test files'
  - 'AC-PLN-013: every new criterion (AC-104..AC-111) names an OBSERVABLE outcome AND a validation command; NO placeholder testing step is authored for the tracker-only, the deterministic-customization or the documentation-only work'
  - 'AC-PLN-014: the plan carries the nine inspection gaps (GAP-5-01..GAP-5-09), the read-only evidence pointer and the next boundary; the authoring pass changed NO production, test, spec or bundle byte'
  - 'AC-PLN-015: the plan stays REGISTERED [WIP] in plans/README.md and plans/Roadmap.md with the Phase-5 lane named in both'
placeholder_steps:
  - 'Step 27 - THE RACE START: the green 2-hex start line + the side-by-side grid + the Flappy-style 3-2-1-Go countdown [DONE]'
  - 'Step 27A - The transparent Astrobird-like countdown overlay, clean dismissal and simulation resume [WIP]'
  - 'Step 28 - The track-departure fix: the car stays ON the road through the corners [DONE 2026-11-22 - 05-green-testing GREEN: OK]'
  - 'Step 29 - The neon dropdown text: the neon option text on the CLOSED select [DONE 2026-11-22 - 05-green-testing GREEN: OK]'
  - 'Step 30 - The smooth drift: no wobble + the VISIBLE rear slip [PLANNED]'
  - 'Step 30A - The browser crossing contract re-base against the PINNED ramp [PLANNED]'
  - 'Step 31 - The traction law: 100% normally, falling with the speed and the tightness [PLANNED]'
  - 'Step 32 - The docs-regen packet (06-documenting) for the Phase-5 modules [PLANNED]'
```

### Phase-5 GAPS + EVIDENCE (2026-11-12, the read-only inspection — NOTHING was implemented)

**The evidence home:** the full inspection evidence, the measured file:line citations, the gap list, the five recorded defaults and the next boundary are archived at `plans/neatracing-voxel-racer.logs.md` evidence batch 169. The nine gaps are the ones that SHAPE this lane:

- **GAP-5-01 (the start line has NO renderer):** the start/finish exists only as a LOGICAL detector (the worker's `resolveSimWorkerLapLineDetectorStep` at `examples/neatRacing/workers/simulation-worker/simulation-worker.ts:649`, called at :1265 and :1340, gating on `NEATRACING_SIM_WORKER_LAP_LINE_LATERAL_GATE_M` = 5 declared at :160 and consumed at :672, and `NEATRACING_SIM_WORKER_LAP_LINE_ALONG_EPS_M` = 0.5 declared at :167 and consumed at :676); there is NO start-line paint, and GREEN is ALREADY the innermost off-road band (`TileColorway = 'road'|'red'|'yellow'|'green'`, a CLOSED union) — so a green start line needs a NEW ALONG-TRACK classification axis (a fifth `TileColorway` member would change `colorwayToColors` + `COLORWAY_PRECEDENCE` and break the `road-edge.test.ts` pins). (ATTRIBUTION CORRECTED 2026-11-12 by the Phase-5 B-3 patch: `isFinishLineCrossing` is NOT the start/finish detector — it is a SPLINE-SAMPLE-WRAP check at `examples/neatRacing/ai/promotion.rules.ts:1193`, called at :634; the AC-104 design intent is UNAFFECTED and only the citation changes, so the RED lane measures the RIGHT seam.)
- **GAP-5-02 (the grid is LINE-ASTERN):** `resolveSimWorkerSplineSpawnGrid` has NO lateral term (the spawn lead + the grid pitch are ALONG-TRACK only); the only lateral spawn constant (`NEATRACING_PHYSICS_SPAWN_LATERAL_SPREAD_M`, declared at `examples/neatRacing/physics/physics.step.service.ts:433`) is a LIVE FALLBACK, not dead code: it is read at :1388 inside `resolveSpawnLateralOffset` (:1386), which is reached from the live fallback `pose?.x ?? resolveSpawnLateralOffset(resolvedSeed)` at :552 — and the worker DOES supply an explicit pose, so the fallback is BYPASSED on the live path (a SINGLE-REFERENCE fallback, not an inactive constant). **THE GRID'S ALONG-TRACK TERM ALSO POINTS THE WRONG WAY (the 2026-11-12 USER clarification):** the spawn lead (2 m) and the grid pitch (12 m) are measured FORWARD from the start plane (`splineSamples[0]` = segment 0's start point), so the first car stands 2 m ON/ABOVE the green start line rather than BEHIND it, and the pitch puts the rest of the field even further along the road. THE AC-105 GEOMETRY LAW (the plane, the direction, the derived clearance) is recorded below.
- **GAP-5-03 (there is NO countdown anywhere):** flappy has NO countdown (only the fading centered startup legend); there is no ready-state overlay, no start gate and no race-start phase in the neatRacing read model.
- **GAP-5-04 (the departure has NO recovery):** `applyTrackContainment()` is a deliberate NO-OP (declared at `examples/neatRacing/physics/physics.step.service.ts:984`; its in-body comment at :988-990 reads "THE RETIRED BOUND: there is NO containment left to apply - no clamp, no reseat, no hold" — the AC-090 VERDICT-8 law; the string "NO BOUND (AC-090 VERDICT 8)" does NOT appear in the file and the only "NO BOUND" occurrence is at :948); the off-road band IS drivable (the 20 km/h cap) but there is no reset, no respawn and no recovery — a car that leaves the road can stay off it indefinitely.
- **GAP-5-05 (the native dropdown cannot be neon):** the OS-rendered `<option>` popup ignores `text-shadow`/`filter`; there is NO shared neon-text helper in the repo and the bloom pass samples the WebGL FBO only, so it cannot serve DOM text.
- **GAP-5-06 (the wobble has FOUR sources):** the under-damped visual pose spring (zeta 0.35), the bang-bang countersteer (the 70 m/s2 lateral impulse behind the hard `|steer| >= 0.9` gate), the per-tick slip re-derivation with NO depth hysteresis, and the absence of steering smoothing / yaw-rate feedback.
- **GAP-5-07 (the rear slip is neither published nor rendered):** the rear wheels are yawed by EXACTLY 0 (the whole body + all four wheels share ONE visual offset); the rear vs the front differ only in SHADING; there are NO skid marks, NO smoke, NO tire trails and NO particles.
- **GAP-5-08 (the traction law has NO tightness term):** the drift model reads the speed and the demanded turn rate only; the curvature lives in the AI autopilot; the grip budget is the rational falloff of the demand + the severity.
- **GAP-5-09 (the AI has NO drift challenge):** the shaped fitness has NO slip/drift term and the driving-quality scalar has NO slip term; the AI turn rate is authoritative, so the AI lane drives a DIFFERENT feel from the user lane.

**THE 2026-11-12 B-1..B-8 PATCH NOTE (the gap list above was CORRECTED IN PLACE by the Phase-5 AUTHOR patch cycle — plan bytes only, NO production/test/spec byte):** GAP-5-01 (the start/finish detector is `resolveSimWorkerLapLineDetectorStep` at `simulation-worker.ts:649`, NOT `isFinishLineCrossing` — B-3); GAP-5-02 (the lateral spawn constant is a LIVE FALLBACK, not an INACTIVE constant — B-6); GAP-5-04 (the quoted comment is the "THE RETIRED BOUND: there is NO containment left to apply ..." text at `physics.step.service.ts:988-990`, NOT "NO BOUND (AC-090 VERDICT 8)" — B-6); the FLAPPY REFERENCE paragraph (the neon PALETTE is EXAMPLE-LOCAL and there is NO shared neon-TOKEN module — B-5); and the AC-104 / AC-105 / AC-106 / AC-108 coverage-backlog rows (the `TileColorway` pin is a COMPILE-TIME type-level assertion over the EXPORTED type — B-4; the throttle-hold clause is re-anchored to the REAL throttle surfaces — B-2). THE DESIGN INTENT OF EVERY GAP IS UNCHANGED; only the CITATIONS, the COUNTS and the WORDING move. THE FULL DISPOSITION IS IN `## Latest validation evidence`.

**The FLAPPY REFERENCE (what is reusable and what is NOT):** there is NO 3-2-1-Go countdown and NO ready-state overlay anywhere in `examples/flappy_bird` or in `examples/shared`. The closest analogue is the canvas-2D centered startup legend with its opacity fade (`FLAPPY_STARTUP_PREVIEW_LEGEND_TEXT`; `renderRuntimeStartupPreviewLegend()`; `resolveRuntimeStartupPreviewVisualState()`; the fade `FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS = 300`). Flappy's phase is a STATUS STRING. The REUSABLE recipe is the TWO-PASS NEON TEXT (the glow pass: `globalCompositeOperation='lighter'` + `shadowColor` + `shadowBlur` + `fillText`; the crisp pass: `source-over` + `shadowBlur=0` + `fillText`; the measured passes are at `examples/flappy_bird/browser-entry/runtime/runtime.startup-preview.service.ts:309-322` (glow) and :324-337 (crisp), inside `renderRuntimeStartupPreviewLegend` :284) over the `FLAPPY_NEON_PALETTE` tokens — a PALETTE that is EXAMPLE-LOCAL (`examples/flappy_bird/constants/constants.palette.ts:9`; it is imported cross-example by `examples/racing_curriculum/browser-entry/host/host.ts:2` and `examples/neatenstein/browser-entry/renderer/floor.ts:53`, and the only OTHER neon palette, `NETWORK_NEON_PALETTE` at `examples/shared/network-visualization/network-view/network-view.constants.ts:115`, is a DUPLICATED value-identical copy consumed only under `examples/shared/`) — there is NO `strokeText`, NO shared neon-text HELPER and NO shared neon-TOKEN module, so the Phase-5 neon text needs its OWN helper module and its OWN tokens (the Step-29 home). Flappy has NO dropdown (a `<button>` grid), so the dropdown work has NO in-repo precedent and follows the Neatenstein WAVE-announcement precedent instead.

**The FIVE RECORDED DEFAULTS (the open decisions — the defaults are RECORDED so the execution proceeds unless the user answers otherwise):** Q1 = **CLAR-5-01** (the start-line geometry: TWO hexes DEEP along the track = a 4.0 m green stripe across the FULL road width; the `TileColorway` union stays UNCHANGED — the ONLY new NEEDS CLARIFICATION marker, so the budget stands at 3 of 3); Q2 = the neon lands on the CLOSED `<select>` (the CSS glow + a neon CLASS hook on the option text; a custom non-native listbox is OUT OF SCOPE for Step 29); Q3 = the departure recovery is PURELY ADVISORY (a HUD "return to the track" hint + the existing off-track penalty; NO auto-reset and NO teleport — the AC-090 VERDICT-7/8 law stays); Q4 = the drift "tightness" is an OPTIONAL `DriftStepInput` field computed by the SERVICE from the demanded turn rate (the drift model stays pure and never reads the track); Q5 = the traction is 100% at a straight-line speed below `NEATRACING_DRIFT_MIN_SPEED_MPS` and at any zero-demand turn and falls as a MONOTONIC function of (the speed x the demanded turn rate x the tightness) with the RATIFIED `GRIP_FALLOFF_THRESHOLD_RAD` (0.2) + `GRIP_FALLOFF_RATE_PER_RAD` (2) as the anchors; the arcade spin is a SEPARATE NAMED over-rotation branch that only arms above `STEER_ENGAIN_THRESHOLD` (0.5) and stays inside the `MAX_SLIP_ANGLE_RAD` (0.45) envelope; the AI CHALLENGE is a fitness term, NOT a behaviour change.

**THE Q1 AMENDMENT (2026-11-12, the USER clarification — a RESOLVED default, NOT a new NEEDS CLARIFICATION marker, so the marker budget STAYS at 3 of 3 ACTIVE):** the USER's verbatim words are "every car, including player and rival, must be side by side FULLY BEHIND the green start line, not on/above it". THE AMENDED Q1 (CLAR-5-01's own default STANDS — the band is still TWO hexes DEEP and still spans the FULL road width): the fleet's shared along-track station is FULLY BEHIND the green line — UPSTREAM, NEVER on it and NEVER on/above it — and the clearance behind the line is DERIVED from the ONE published car length (4.5 m, measured to the car's CENTRE). The Q1 AMENDMENT RESOLVES the USER's own words, so it creates NO marker and the budget stands at 3 of 3 (CLAR-4-01 + CLAR-4-02 + CLAR-5-01).

**THE AC-105 START-LINE GEOMETRY LAW (the 2026-11-12 USER clarification, recorded as the load-bearing geometry of the AC-105 re-pin — PLAN BYTES ONLY, NOTHING IMPLEMENTED):**

1. **THE PLANE.** The green start line is the ALONG-TRACK PLANE at `splineSamples[0]` — segment 0's start point. It is simultaneously the ARC-0 ORIGIN of the AC-104 start-line band (`projectOntoClosedSpline` measures `arcM` FORWARD from `splineSamples[0]`; `isWithinStartLineBand(arcM)` = `arcM >= 0 && arcM <= 4.0`, so the band occupies the FORWARD window `[0, 4.0] m`), the PLANE the live lap detector `resolveSimWorkerLapLineDetectorStep` counts the crossing on (its `alongM` is signed from segment 0's start point, NEGATIVE = BEHIND, with a `|alongM| < 0.5` deadband), and the ORIGIN of the ROAD-ANCHORED spawn walk.
2. **THE DIRECTION.** "FULLY BEHIND" = UPSTREAM = a NEGATIVE along-track offset in the walk frame. NEVER offset 0 (that is ON the line) and NEVER a positive offset (that is ON or ABOVE it). In the closed-loop arc frame the upstream direction is the lap TAIL: the station is `loopLengthM - clearance` (the wrap), NEVER inside `[0, 4.0] m`.
3. **THE CLEARANCE (named + DERIVED, never a literal).** `NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M = NEATRACING_VOXEL_CAR_LENGTH_M` = **4.5 m** (the ONE published car length in the example — `examples/neatRacing/voxel/lod.distance.utils.ts:48`; a repo-wide grep confirms it is the ONLY car dimension published under `examples/neatRacing`), measured to the car's CENTRE. THE CONSEQUENCES: the NOSE stands `4.5 - 2.25 = 2.25 m` (half a car length) strictly upstream of the plane; the TAIL stands `4.5 + 2.25 = 6.75 m` upstream; therefore the WHOLE footprint is behind the line and can NEVER overlap the forward green band `[0, 4.0] m` (the nose-to-band gap is 2.25 m). The clearance is DERIVED from the published car length so the footprint and the placement can never drift apart; a second hard-coded 4.5 m would be a SECOND source of truth and is FORBIDDEN.
4. **THE TWO FRAMES (ONE physical placement, TWO measures).** The clearance applies in EACH walk frame's own along-track measure, always BACKWARD from that frame's plane: the ROAD-ANCHORED SPLINE walk (`resolveSimWorkerSplineSpawnGrid`) ⇒ `alongM = (loopLengthM - 4.5) % loopLengthM`; the LEGACY CHORD walk (`resolveSimWorkerFleetSpawnPoses`'s fallback, which walks segment 0's chord from `firstSegment.start`) ⇒ `alongM = -4.5`. On the ROAD-ANCHORED layouts (the generated layouts the demo races) the two planes are the SAME POINT — `splineSamples[0]` sits on segment 0's start point within `NEATRACING_SIM_WORKER_SPAWN_PATH_ANCHOR_EPS_M` (0.01 m) — so the law is exact against the painted green band, the lap detector and the spawn walk alike.
5. **THE LEGACY-FRAME RESIDUAL (RECORDED, NOT HIDDEN).** On a layout whose spline is NOT road-anchored (the hand-authored fixture rings in the test home ONLY — the square fixture's `splineSamples[0]` is the MIDPOINT of segment 0, 400 m from `firstSegment.start`), the painted band's plane (the spline origin) and the chord plane (segment 0's start) are DIFFERENT points, so the green band sits AWAY from the chord plane — a PRE-EXISTING AC-104 property of non-road-anchored layouts, NOT created by this patch. The behind-the-line clearance is therefore asserted ON THAT FIXTURE against the CHORD plane (the plane the legacy walk and the lap detector share), and the GREEN lane RECORDS the measured band origin vs the chord plane on BOTH fixtures. The chord-frame point `-4.5 m` lands in the corner cap just upstream of segment 0's start, which the chord-capsule edge model still classifies as road (the fixture's published width is 60 m).
6. **THE LATERAL LAW (UNCHANGED, but read AT the new station).** The side-by-side lateral offsets stay as authored by `P5-S27-3`, but the published ASYMMETRIC half-width is STATION-DEPENDENT (the road widens on the OUTER side of a corner), so the containment bound MUST be read AT the spawn station (the lap tail), NOT at the start line.
7. **THE LAP-DETECTOR CONSEQUENCE (RECORDED, NOT SILENT).** The fleet now starts BEHIND the plane, so the detector's FIRST FORWARD crossing after GO is the LAP-1 crossing — the desired race-start reading. The detector's own RED-home case constructs its poses directly and is UNCHANGED.
8. **THE RETIREMENT (no deferred cleanup).** `P5-S27-3` retires `NEATRACING_SIM_WORKER_SPAWN_ALONG_LEAD_M` (2 m) and `NEATRACING_SIM_WORKER_SPAWN_GRID_PITCH_M` (12 m) IN THE SAME SLICE — the line-astern lead + pitch die with the line-astern walk. Both are worker-local (a repo-wide grep finds NO consumer outside `simulation-worker.ts`), so the retirement breaks NO published surface.
9. **THE CROSS-LAYER IMPORT (the recorded risk + the decision).** The derivation means `examples/neatRacing/workers/simulation-worker/simulation-worker.ts` imports `NEATRACING_VOXEL_CAR_LENGTH_M` from `examples/neatRacing/voxel/lod.distance.utils.ts` — a `workers -> voxel` edge. DECISION: **ACCEPTED** — the worker ALREADY reaches across the same example's layers (`workers -> physics`, `workers -> host`, `workers -> ai`, `workers -> racing_curriculum`), so the edge is NOT a new architectural violation; and the alternative (a NEW published car-length constant in the physics layer) would create a SECOND source of truth for the car's length, which the single-source law forbids (the `P1-blob-shadow` / `P1-traction` precedent).
10. **THE TEST-LOCAL CONSTANT (the RED-lane rule).** The re-pinned RED home uses a TEST-LOCAL `SIM_WORKER_TEST_SPAWN_BEHIND_CLEARANCE_M = 4.5` LITERAL — a RED lane must NEVER create an import failure, and a literal is the STRONGER pin (a later production drift of the car length is then CAUGHT by the RED home instead of being silently followed). The GREEN lane (`P5-S27-5`) imports the production `NEATRACING_SIM_WORKER_SPAWN_BEHIND_CLEARANCE_M` and cross-checks it against the test-local 4.5 m, so the two can never drift.

**The ACTIVE NEXT BOUNDARY (COUNTDOWN-FIRST USER AMENDMENT):** a FRESH `01-planning` VERIFICATION instance must validate Step 27A and record `green-light: true` scoped to `P5-S27A-red` with BLOCKERS NONE. Then run `03-red-testing` for `P5-S27A-red`; the FIRST production implementation is `04-implementing` for `P5-S27A-overlay`, followed by `05-green-testing` for `P5-S27A-green`. No Step-30 implementation may run before Step 27A is DONE. Afterwards, apply the preserved six-edit Step-30 author amendment, freshly verify `P5-S30-repin`, and dispatch it only on a new slice-scoped green light. Minimize Chrome MCP, ideally to the final Step-27A visible-background validation; no foreground/focus/topmost forcing and no FPS assertion for this repair.

**HISTORICAL STEP-30 RESUME RECORD — SUPERSEDED AS THE ACTIVE FRONTIER BY THE COUNTDOWN-FIRST PARAGRAPH ABOVE; RETAINED FOR THE BLOCKED VERDICT AND SIX-EDIT AMENDMENT DETAIL (original heading: The NEXT BOUNDARY (the resume point of record — RE-HEADED 2026-11-23 by the 01-planning HANDOFF-RESTRUCTURE pass; the 2026-11-22 STEP-29-CLOSURE boundary paragraph is ARCHIVED BYTE-EXACT at `plans/neatracing-voxel-racer.logs.md`, evidence batch 188, sub-block J):** THE PLAN IS **PAUSED** (2026-11-23, THE USER-DIRECTED TRANSITION TO A NEW OPENTELEMETRY WORKSTREAM — THE OPEN TELEMETRY WORKSTREAM IS A SEPARATE PLAN TRACKER, NOT A PHASE OF THIS ONE). THE STATUS TOKENS ARE UNCHANGED — THE PLAN `[WIP]`, PHASE 5 `[WIP]`, STEP 30 `[WIP]`, THE ONE `[WIP]` SLICE `P5-S30-repin`; `P5-S30-slip`, `P5-S30-green`, `P5-S30A-crossing-rebase`, STEP 31 AND STEP 32 ARE `[PLANNED]`. THE PAUSE BOUNDARY IS THE 2026-11-23 `P5-S30-repin` INDEPENDENT VERIFICATION VERDICT — **BLOCKED** (B-1 THE SCOPE / LEGACY-FLOOR COLLISION, B-2 THE AI-CHANNEL AUTHORITY; THE BOUNDED CLARIFICATIONS C-1 THE CHATTER BAND AND C-2 THE VALIDATION LIST ARE **NOT** NEEDS-CLARIFICATION MARKERS — THE PLAN-WIDE BUDGET STANDS AT 3 OF 3). THE FIRST ACT ON RESUME IS A TRACKER-ONLY `01-planning` **AUTHOR** PASS THAT APPLIES THE SIX-EDIT CLAUSE-7 AMENDMENT TO THE STEP-30 PACKET — (1) THE FIRST-TICK SEEDING RULE (THE FIRST TICK IS UNSMOOTHED; THE SMOOTHING AND THE YAW-RATE FEEDBACK START FROM THE SECOND TICK); (2) THE CHANNEL IS THE USER CHANNEL ONLY (EVERY AI NUMBER STAYS BYTE-IDENTICAL); (3) THE DRIFT-LAW INPUT PLUS THE AC-091 "ONE RESOLVED NUMBER" SUPERSESSION (FOR THE HEADING ADVANCE ONLY); (4) THE COUPLED FIRST-TICK ASSERTIONS (VERIFICATION CLAUSE 2 (i) THROUGH (vii)) AND THE RE-PIN POSTURE (WIDEN `files_to_change` TO `examples/neatRacing/physics/physics.step.service.test.ts` IF A RE-PIN PROVES UNAVOIDABLE); (5) THE CHATTER-BAND HOME (`examples/neatRacing/physics/physics.step.service.ts`, AN EXPORTED `NEATRACING_PHYSICS_*` CONSTANT) AND THE MEASURED WINDOW (AT OR ABOVE ~0.0005 RAD, AT OR BELOW ~0.00158 RAD); (6) THE AI-LANE VALIDATION COMMAND `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai/scripted.controller"` ADDED TO THE SLICE VALIDATION LIST. THEN A **FRESH** `01-planning` VERIFICATION INSTANCE RE-RUNS THE ONE CONSOLIDATED `slice-advancement` GATE OVER THE AMENDED PACKET. **ONLY** A `green-light: true` VERDICT THERE UNBLOCKS THE `04-implementing` LANE FOR `P5-S30-repin`. THE AMENDMENT IS A PLAN-SIDE EDIT ONLY — IT IS NOT AN IMPLEMENTATION DISPATCH, AND THIS PASS APPLIED NONE OF IT. THE STANDING CARRIES ARE UNCHANGED AND NONE IS ABSORBED — THE AI-LANE LIVE-LANE BUDGET RE-PIN (THE LIVE FIGURE IS **283 / 285**); COVERAGE GAP g1 (REACHABLE-BUT-UNTESTED — ADD A TEST, DO NOT REMOVE THE BRANCH); THE STANDING VISIBLE-BROWSER USER RULE (2026-11-21); THE W-1 `checkPlanGreenLight` SCOPING DEFECT; THE MCP VALIDATION-ALLOWLIST DEGRADATION; THE STANDING 00-helping CORTEX INDEX REBUILD OWNERSHIP.

#### Step 27 — THE RACE START: the green 2-hex start line + the side-by-side grid + the Flappy-style 3-2-1-Go countdown [DONE]

**Done record:** Step 27 CLOSED [DONE 2026-11-21 - 05-green-testing GREEN: OK] — the green 2-hex start line, the side-by-side start grid and the Flappy-style 3-2-1-Go countdown (AC-104 + AC-105 + AC-106) ARE GREEN BY NAME AND EVERY SLICE (`P5-S27-1`, `P5-S27-2`, `P5-S27-3`, `P5-S27-4`, `P5-S27-5`) IS [DONE]. THE FULL DONE RECORD IS ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block K).

#### Step 28 — THE TRACK-DEPARTURE FIX: the car stays ON the road through the corners [DONE]

**Done record:** Step 28 CLOSED [DONE 2026-11-22 - 05-green-testing GREEN: OK] — the track-departure fix (AC-107) IS GREEN BY NAME AND EVERY SLICE (`P5-S28-red`, `P5-S28-law`, `P5-S28-repin`, `P5-S28-green`) IS [DONE]. THE FULL DONE RECORD IS ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block K).

#### Step 29 — THE NEON DROPDOWN TEXT: the neon option text on the CLOSED select [DONE]

**Done record:** Step 29 CLOSED [DONE 2026-11-22 - 05-green-testing GREEN: OK] — the neon dropdown text (AC-108) IS GREEN BY NAME AND EVERY SLICE (`P5-S29-red`, `P5-S29-neon`, `P5-S29-repin`, `P5-S29-green`) IS [DONE]. THE FULL DONE RECORD IS ARCHIVED BYTE-EXACT AT `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block K).

#### Step 27A — THE COUNTDOWN REPAIR: transparent Astrobird-like overlay + simulation resume [WIP]

**Objective (the FIRST implementation priority).** Repair the already-delivered AC-106
countdown without reopening the closed start-line, grid, track, neon, or drift work. The
3-2-1-GO sequence must read as a lightweight Astrobird-like text/glow overlay above the
visible race scene. There must be NO black, opaque, or full-screen dimming backdrop during
the countdown, at GO/dismissal, or afterward. The pre-GO human throttle HOLD remains, but
at GO the overlay dismisses cleanly, the gate releases, and the simulation advances.

**Known seam and bounded ownership.** The closed Step-27 contract names
`examples/neatRacing/host/hud.countdown.ts` for the pure countdown/presentation resolver,
`examples/neatRacing/host/hud.ts` for the DOM overlay surface, and
`examples/neatRacing/browser-entry/browser-entry.ts` for the existing HOLD/release and
worker-handoff seam. The RED homes are the existing
`examples/neatRacing/host/hud.countdown.test.ts` and
`examples/neatRacing/browser-entry/browser-entry.test.ts`. This planning pass did NOT
inspect racer source or tests and does not assume a root cause. The implementing slice
must inspect the live bytes, make the smallest repair inside this three-file production
boundary, and leave `browser-entry.ts` unchanged if the existing release/progression path
is already correct. Do not widen into physics, workers, rendering passes, stylesheets, or
new dependencies without returning to planning.

**Validation and browser policy.** RED must fail by named assertions, never by a missing
import or compile error. Implementation must turn AC-106-R1..R3 green by name. The GREEN
slice runs the focused host/browser-entry checks and the step validation list. Browser
automation is minimized: no Chrome during planning or implementation; ideally ONE final
real visible-browser validation after focused checks are green. The browser window may
stay in the background. Never force foreground, focus, always-on-top, or z-order for this
step; the rare explicitly FPS-scoped exception in Mandates does NOT apply here, and no FPS
assertion is required. The final visual check must cover 3/2/1, GO, dismissal, and a short
post-GO interval, proving a visible scene, no black backdrop, and resumed motion.

```yaml
phase: 5
step: '27A'
title: 'THE COUNTDOWN REPAIR: transparent Astrobird-like overlay + simulation resume'
status: '[WIP]'
wave_note: 'USER-DIRECTED COUNTDOWN-FIRST AMENDMENT; TRACKER-ONLY. This is the first implementation boundary. The closed Step-27 behavior is preserved except for the transparent presentation, clean dismissal, and post-GO resume defects named here. Step 30 and P5-S30-repin move to PLANNED behind this step; their blocked verdict is retained. No racer code, test, build, browser, gate, or subagent action ran in this authoring pass. Fresh independent verification is required before RED execution.'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: red-green
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'After Step 27A is DONE, apply the preserved six-edit Step-30 author amendment, freshly verify P5-S30-repin, and only then consider drift implementation'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P5-S27A-red'
skills:
  - red-test-contracts
  - implementation-standards
  - green-validation-gates
  - tracker-handoff
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown|examples/neatRacing/browser-entry/browser-entry"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'npm run build:neatRacing'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated)'
acceptance_criteria:
  - 'AC-106-R1: transparent Astrobird-like text/glow over the visible scene; no black, opaque, or dimming backdrop during 3/2/1, GO, dismissal, or afterward'
  - 'AC-106-R2: GO/dismissal removes or hides the overlay and any backdrop, with no residual black surface or persistent input-capturing layer'
  - 'AC-106-R3: pre-GO human throttle remains held; at GO the gate releases and simulation time/frame progression resumes, with post-GO throttle reaching the existing worker handoff'
  - 'MINIMAL CHROME: ideally one final visible-background browser validation only; no foreground, focus, topmost, z-order, or FPS requirement for this repair'
slices:
  - slice_id: 'P5-S27A-red'
    title: 'RED: transparent overlay, clean dismissal, and simulation-resume contracts'
    status: '[WIP]'
    goal: red-testing
    estimate_hours: 2
    research_artifact: 'Coverage rows AC-106 and AC-106-R1..R3 plus the closed Step-27 countdown seam map in plans/neatracing-voxel-racer.logs.md; live source inspection belongs to the bounded implementation lane, not this planning pass'
    files_to_change:
      - 'examples/neatRacing/host/hud.countdown.test.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - id: 'RED-27A-OVERLAY'
        text: 'Extend the existing countdown home with named assertions that the scene-facing overlay and backdrop stay transparent for 3/2/1/GO and after dismissal; assert no black/opaque/dimming surface and no persistent input-capturing overlay. Fail by assertion against existing imports, never by a missing module or compile error.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown"'
      - id: 'RED-27A-RESUME'
        text: 'Extend the existing host/browser-entry homes with a non-vacuous sequence: before GO human throttle is held and simulation progression is in the countdown state; after GO throttle is released to the existing handoff and simulation time/frame or tick progression advances. A gate that never releases or a simulation that stays stopped must fail.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown|examples/neatRacing/browser-entry/browser-entry"'
    red_test:
      - 'Record the exact named failures and before-counts in Latest validation evidence before production edits; do not weaken or delete the original AC-106 resolver, neon-phase, or pre-GO HOLD assertions'
      - 'No browser, build, production edit, or broad suite is needed to establish the focused RED baseline'
    parallelizable: false
    dependencies:
      - 'P5-S27-5'
  - slice_id: 'P5-S27A-overlay'
    title: 'IMPLEMENT FIRST: transparent Astrobird-like countdown and post-GO resume'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    research_artifact: 'P5-S27A-red named failures plus the existing hud.countdown.ts, hud.ts, and browser-entry.ts seams; inspect current live bytes before choosing the smallest repair'
    files_to_change:
      - 'examples/neatRacing/host/hud.countdown.ts'
      - 'examples/neatRacing/host/hud.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - id: 'AC-106-R1-OVERLAY'
        text: 'Render the 3/2/1/GO text/glow as an Astrobird-like transparent overlay above the visible race scene. No black, opaque, or full-screen dimming backdrop may be introduced or left behind during or after any countdown phase.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown"'
      - id: 'AC-106-R2-DISMISSAL'
        text: 'At GO/dismissal remove or hide the overlay and any backdrop, restore ordinary input behavior, and leave no black/opaque remnant or persistent input-capturing layer; the race scene stays visible afterward.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown"'
      - id: 'AC-106-R3-RESUME'
        text: 'Preserve the pure raceState.elapsedS resolver and pre-GO human throttle HOLD; at GO release the existing gate and ensure the simulation advances after countdown. Keep AI samples and unrelated physics/worker behavior unchanged.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown|examples/neatRacing/browser-entry/browser-entry"'
    red_test:
      - 'Make only the P5-S27A-red contracts green by name; do not re-author, weaken, or delete RED assertions'
      - 'Use the smallest repair within the three-file boundary; if browser-entry.ts already releases and advances correctly, leave it unchanged and record that evidence rather than making a speculative edit'
      - 'Do not use Chrome during implementation; final visible-background validation belongs to P5-S27A-green'
    parallelizable: false
    dependencies:
      - 'P5-S27A-red'
  - slice_id: 'P5-S27A-green'
    title: 'GREEN: focused regression proof and one minimized final visual validation'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 2
    research_artifact: 'P5-S27A-red baseline, P5-S27A-overlay implementation evidence, and the standing visible-background browser rule as amended by the minimal-Chrome directive'
    files_to_change:
      - 'examples/neatRacing/host/hud.countdown.test.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.test.ts'
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - id: 'AC-106-R-GREEN'
        text: 'AC-106 and AC-106-R1..R3 are green by name; focused and step validation commands pass; the transparent scene remains visible through countdown and after dismissal, and post-GO simulation progression is observed. No black backdrop is accepted at any observed phase.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host/hud.countdown|examples/neatRacing/browser-entry/browser-entry"'
      - id: 'AC-106-R-BROWSER'
        text: 'Ideally run exactly one final real visible-browser validation after focused checks are green; cover 3/2/1, GO, dismissal, and post-GO motion. Keep the window backgrounded where possible; never force foreground, focus, topmost, or z-order. No FPS assertion and no rare-FPS exception for this step.'
        validation: 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated), with final visible-background browser evidence attached'
    red_test:
      - 'Verify the existing RED homes by name; do not re-author contracts in the GREEN lane'
      - 'Record exact pass counts and the minimized browser evidence; if final browser validation is unavailable, record the blocker rather than forcing window visibility or claiming visual success'
    parallelizable: false
    dependencies:
      - 'P5-S27A-overlay'
evidence:
  gate_outputs: []
```

#### Step 30 — THE SMOOTH DRIFT: no wobble + the VISIBLE rear slip [PLANNED]

**Objective (Step 30).** TWO user-visible defects are closed in ONE step, because they share
the same seam and the same published numbers: (1) **the wobble** — the drift currently rings
and chatters instead of reading as a smooth slide (the four sources are enumerated in the
GAP-5-06 row below), and (2) **the invisible rear slip** — the car is visibly sideways but
the rear wheels still point straight ahead, so the slide is unreadable at a glance
(GAP-5-07). The step delivers BOTH the SMOOTH drift and the VISIBLE rear slip, and it does
NOT add a new physics channel: the rear slip is rendered from the ALREADY-PUBLISHED slip
angle, so the renderer and the physics can never disagree about the slide.

**The seam map (every citation read off the LIVE bytes on 2026-11-22; this authoring pass
edited NO source byte).**

- `examples/neatRacing/physics/drift.model.ts` (1280 lines) — the PURE drift model. The
  visual pose spring constants: `NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2 = 324` (L463)
  and `NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S = 12.6` (L468), which is a damping ratio of
  `12.6 / (2 * 18) = 0.35` — **UNDER-DAMPED** (wobble source 1). The pose spring itself is
  `stepDriftPoseSpring()` (L1219) and the articulation slew is
  `NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S = 20` (L485). The countersteer: the
  authority gate `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER = 0.9` (L392), the impulse
  `NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2 = 70` (L380) and the entry function
  `computeCountersteerAuthority()` (L1075) — the hard `|steer| >= 0.9` gate plus the
  70 m/s² step is **BANG-BANG** (wobble source 2). The published slip:
  `computeSlipAngle()` (L1271) is `Math.atan2(lateralVelocity, speed)` (L1278).
  **THE PUBLISHED SLIP'S ON-ROAD CEILING IS THE DERIVED `NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD`
  (L164-166) = 0.452** (the committed slide budget 0.45 plus the 0.002
  `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` at L130), and the OFF-ROAD ceiling is
  `NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD = 0.493` (L199). **BOTH PUBLISHED-SLIP
  CEILINGS ARE ASSERTION BOUNDS, NEVER CLIPS — nothing in the service clamps the published
  slip.** `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45` (L102) **IS AN ASSERTION BOUND ON THE
  COMMITTED SLIDE AND IS NEVER A CLIP ON THE PUBLISHED SLIP** (it hard-clamps `committed.slip`
  inside `stepDriftCommitment` and is NEVER re-pinned) — **SO A RED HOME THAT ASSERTS
  `Math.abs(publishedSlip) <= 0.45` FAILS FOR THE WRONG REASON AND MUST NOT BE WRITTEN.**
  The traction-severity inputs `GRIP_FALLOFF_THRESHOLD_RAD = 0.2` (L205),
  `GRIP_FALLOFF_RATE_PER_RAD = 2` (L215), `MIN_SPEED_MPS = 15` (L227),
  `STEER_ENGAGE_THRESHOLD = 0.5` (L274) and `SLIP_RESPONSE_TAU_S = 0.045` (L374) are
  UNCHANGED by this step.
- `examples/neatRacing/physics/physics.step.service.ts` (1447 lines) — the per-tick step.
  The demanded turn rate is `resolveDemandedTurnRateRadPerS(clampedInput.steer, speedPercent,
  clampedInput.turnRatePercent)` (L781-785), the yaw rate is `resolveYawRatePerTick(...)`
  (L786) and the heading advance is `nextHeading = wrapHeading(state.heading + yawRatePerTick)`
  (L787) — **THERE IS NO STEERING SMOOTHING AND NO YAW-RATE FEEDBACK TODAY** (wobble source
  4). The steering rate constant is `NEATRACING_PHYSICS_STEERING_RATE = 2` (L348) and the
  speed-proportional turn-rate reduction is
  `TURN_RATE_SPEED_REDUCTION_FRACTION = 0.2` (L359). The slip re-derivation is
  `computeSlipAngle(...)` (L852-855) called EVERY TICK with **NO DEPTH HYSTERESIS**
  (wobble source 3), the pose spring is `stepDriftPoseSpring(...)` (L863-869), the travel
  direction is `velocityHeading = wrapHeading(nextHeading + nextSlipAngle)` (L875) and the
  published slip is `slipAngle: nextSlipAngle` (L904).
- `examples/neatRacing/engine/passes/car.pass.ts` (1833 lines) — the renderer. The pose
  state `CarPassCarState` (L467-541) carries `yawRad` (L473), `wheelSpinRad` (L477), the
  front-wheel steer `steerAngleRad` (L483) and the drift-yaw channel `driftYawOffsetRad`
  (L522). **THESE ARE THE LIVE EXPORTED NAMES — `CarPoseState` AND `visualDriftYawRad` HAVE
  ZERO REPO-WIDE OCCURRENCES AND MUST NOT BE TYPED BY A RED HOME (A TYPE ERROR THERE IS RED
  BY COMPILE ERROR, THE ONE FAILURE MODE THIS STEP FORBIDS).** The wheel draw loop
  (L1748-1799) applies the mount translation, the mount yaw
  (`rotationYMat4(..., mount.rotationYawRad)` L1760), the steer yaw
  (`isFrontWheel ? computeFrontWheelSteerYawRad(car) : 0` L1771-1774 — **THE REAR WHEELS
  ARE YAWED BY EXACTLY 0 TODAY**), the per-wheel grip (L1779-1782) and the wheel spin
  (L1788). The front axle is the steered axle under AC-088 and that law is UNCHANGED.
- `examples/neatRacing/browser-entry/browser-entry.ts` (3910 lines) — the debug seam. The
  drift seam interface carries **EXACTLY EIGHT** `readonly` numeric fields:
  `followOffsetRad` (L253), `poseRad` (L255), `poseRateRadPerS` (L273), `slipAngleRad`
  (L275), `speedMps` (L277), `tickHz` (L279), `tick` (L293) and `tractionSeverity` (L304).
  **THE SEAM IS PINNED AT EIGHT AND IS NOT WIDENED BY THIS STEP** — the rear slip reaches
  the renderer through the EXISTING published slip channel, never through a NINTH field.
- `examples/neatRacing/voxel/wheel.mount.utils.ts` — the wheel sockets
  (`NEATRACING_WHEEL_SOCKET_ORDER` L140-144), the mount yaw
  `NEATRACING_WHEEL_MOUNT_YAW_RAD` (L107) and the mount transforms (L296, L325). The mount
  layout is READ ONLY by this step.

**The invariants that MUST survive the green (carried forward from the closed steps).** The
AC-087 "never mirrors" law; the UNCLAMPED pose spring (a clamp would fight the spring and
flatten the pop); the PURE drift model (it never reads the track and never reads a mutable
global); `examples/neatRacing/ai/oscillation-penalty.utils.ts` (the AI FITNESS DOMAIN) stays
UNCOUPLED — this step touches NO AI file; the published slip, the travel heading and the
traction severity stay the SAME numbers for EVERY consumer; and the new visual channels
(skid marks, smoke, tire trails, particles) stay OPTIONAL and are **NOT TAKEN** — the
recommended visibility IS the rear-wheel yaw itself.

```yaml
phase: 5
step: 30
title: 'THE SMOOTH DRIFT: no wobble + the VISIBLE rear slip'
status: '[PLANNED]'
wave_note: 'AUTHORED 2026-11-22 by the 01-planning (AUTHOR) instance from the two Phase-5 coverage-backlog rows AC-109 (the SMOOTH NO-WOBBLE DRIFT) and AC-110 (the VISIBLE REAR SLIP) - the USER directive, verbatim: a smooth no-wobble drift WITH a visible rear slip. THE STEP OBJECTIVE: the drift FEELS smooth (the car no longer wobbles through a corner) AND the rear slip is VISIBLE (the rear wheels point where the car is actually sliding). THE PACKET IS AUTHORED, NOT GREEN-LIT AND NOT DISPATCHED - the author/verify separation holds: this instance writes the packet and self-checks it with the ONE consolidated slice-advancement gate, and a FRESH 01-planning VERIFICATION instance records the green light. THE ONE [WIP] SLICE WAS P5-S30-red (the RED lane) and the step pre_execute_hook was RE-BOUND to it. AMENDED 2026-11-22 BY THE 01-planning P5-S30-damping PREPARATION PASS (TRACKER-ONLY): THE RED LANE RAN, WAS GREEN-LIT AT THE CONTRACT LEVEL (EXPLICITLY SCOPED TO P5-S30-red, BLOCKERS: NONE) AND IS NOW CLOSED WITH THE ANNOTATED TOKEN [DONE 2026-11-22 - 03-red-testing (Tier 1) RED AUTHORED]; THE ONE [WIP] SLICE IS NOW P5-S30-damping AND THE STEP pre_execute_hook IS RE-BOUND TO IT; THE DAMPING LANE FILE SET AND CRITERION ARE WIDENED BY THE DISCLOSED RE-PIN OBLIGATION (R-2) AND THE DIRECTLY COUPLED EXISTING ASSERTIONS (THE NEW AC-109-DAMPING-REPIN CRITERION) - THAT PASS CHANGED NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE. THE STEP IS THE FIVE-SLICE HARD CAP: the AC-109 + AC-110 union names SIX unique slice names (red, damping, repin, slip, render, green) while BOTH gates cap a step at FIVE slices (MAX_SLICES_PER_STEP = 5 in scripts/agent-customization/gates/plan-slice-quality.gate.mjs and in scripts/agent-customization/gates/step-packet.gate.mjs), so THE RENDER WORK IS FOLDED INTO P5-S30-slip and the standalone P5-S30-render slice DOES NOT EXIST - the fold is DISCLOSED in the AC-110 row of the coverage backlog. EVERY CITATION BELOW WAS READ OFF THE LIVE BYTES ON 2026-11-22 - THIS AUTHORING PASS EDITED NO SOURCE BYTE. AMENDED 2026-11-22 BY THE 01-planning (AUTHOR) INSTANCE ON THE ACCEPTANCE BY AGENT ZERO OF THE 00-helping CROSS-TIER DECISION (TRACKER-ONLY: THIS PLAN IS THE ONLY BYTE WRITTEN; NO PRODUCTION BYTE, NO TEST BYTE, NO SPEC BYTE, NO BUNDLE BYTE, NO GENERATED-DOC BYTE, NO COMMIT, NO BRANCH, NO PR, NO MUTATING GIT COMMAND, AND NO STATUS TOKEN FLIPPED): THE AC-109-DAMPING-REPIN CLAUSE (3) (THE RESOLUTION LADDER) IS REWRITTEN, THE CLAUSES (2a), (2b), (4a) AND (7) ARE ADDED, THE research_artifact AND THE red_test BULLETS OF P5-S30-damping ARE EXTENDED, AND THE BROWSER CROSSING CONTRACT RE-BASE IS HANDED TO THE NEW STEP-30A BOUNDARY LANE P5-S30A-crossing-rebase (WHICH DEPENDS ON P5-S30-damping) - THE STEP-30 FIVE-SLICE CAP IS PRESERVED BY THE BOUNDARY STEP AND NOT BY WIDENING A SLICE OR BY FOLDING TWO BEHAVIOURAL INTENTS INTO ONE SLICE. AMENDED 2026-11-22 BY THE 05-green-testing (Tier 1) SLICE-SCOPED GREEN LANE (THE P5-S30-damping ACCEPTANCE VERDICT; TRACKER-ONLY - THIS PLAN PLUS plans/README.md AND plans/Roadmap.md ARE THE ONLY BYTES WRITTEN; NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE; NO IMPLEMENTATION DISPATCHED; NO MUTATING GIT COMMAND): THE VERDICT IS GREEN: OK, SCOPED TO THE SLICE P5-S30-damping ONLY - RED-S30-1 AND RED-S30-2 ARE GREEN BY NAME, EVERY COUPLED CORPUS FLOOR IS BYTE-IDENTICAL, THE FIVE MEASURED MARGINS REPRODUCE EXACTLY AND THE COUNTERSTEER RAMP SWEEP IS MONOTONE WITH THE KNEE EXACTLY 0 AND FULL LOCK EXACTLY 1; THE P5-S30-damping TOKEN IS NOW [DONE 2026-11-22 - 05-green-testing GREEN: OK], THE ONE [WIP] SLICE IS NOW P5-S30-repin (THE STEP-SERVICE RE-PIN) AND THE STEP pre_execute_hook IS RE-BOUND TO IT; THE BROWSER OBLIGATION IS N/A FOR THIS SLICE (CLAUSE (7) HANDS THE BROWSER CROSSING RE-BASE TO THE STEP-30A BOUNDARY LANE P5-S30A-crossing-rebase AND THE VISIBLE-BACKGROUND BROWSER PROOF IS OWNED BY THE WHOLE-STEP LANE P5-S30-green), SO NO CHROME WAS LAUNCHED, NO FOCUS WAS FORCED AND NO FPS WAS MEASURED.'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: red-green
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 30A - The browser crossing contract re-base against the PINNED ramp'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P5-S30-repin'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
  - spec-checklist
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
  - 'npm run build:neatRacing'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md'
  - 'neataptic-gate-mcp:run_gate_check gate=slice-advancement (consolidated)'
acceptance_criteria:
  - 'AC-109 (the SMOOTH NO-WOBBLE DRIFT): all FOUR wobble sources of the GAP-5-06 row are closed or explicitly justified - the under-damped visual pose spring (zeta 0.35), the bang-bang countersteer (the hard |steer| >= 0.9 gate plus the 70 m/s2 step), the per-tick slip re-derivation with NO depth hysteresis, and the absent steering smoothing / yaw-rate feedback. The RED home is authored FIRST and fails BY ASSERTION, then the green makes it pass BY NAME. The drift model stays PURE, the pose spring stays UNCLAMPED, the AC-087 never-mirrors law holds, the ai/oscillation-penalty.utils.ts fitness domain stays UNCOUPLED and the published slip, the travel heading and the traction severity stay the SAME numbers for every consumer.'
  - 'AC-110 (the VISIBLE REAR SLIP): the rear wheels carry the PUBLISHED slip angle verbatim, so the rear vs the front differ by MORE than shading; the front-wheel steer yaw under AC-088 is UNCHANGED; the browser debug seam still carries EXACTLY EIGHT readonly fields; the optional visual channels (skid marks, smoke, tire trails, particles) are NOT taken - the rear-wheel yaw IS the visibility; and the VISIBLE-BACKGROUND browser proof runs under the standing USER RULE.'
  - 'NO DEFERRED CLEANUP: the wobble sources this step removes are removed IN THE SAME SLICE that introduces the replacement - there is no dead flag, no orphaned constant, no superseded branch and no "TODO remove" left behind in examples/neatRacing/physics/drift.model.ts, examples/neatRacing/physics/physics.step.service.ts or examples/neatRacing/engine/passes/car.pass.ts.'
  - 'NO REGRESSION: the two closed-step invariants that this step touches stay intact - the AC-087 never-mirrors law and the UNCLAMPED pose spring - and the published slip, the travel heading and the traction severity are asserted to be the SAME numbers for every consumer after the change.'
slices:
  - slice_id: 'P5-S30-red'
    title: 'The RED lane: the no-wobble + the visible rear-slip contracts'
    status: '[DONE 2026-11-22 - 03-red-testing (Tier 1) RED AUTHORED]'
    goal: red-testing
    estimate_hours: 3
    research_artifact: 'ARCHIVED 2026-11-23 (the 01-planning handoff restructure, evidence batch 188, sub-block H): the full AC-109 + AC-110 contract set, the GAP-5-06 four wobble sources, the GAP-5-07 rear-slip gap and the Step-30 seam map are byte-exact in the companion log. THIS SLICE IS CLOSED [DONE 2026-11-22 - 03-red-testing (Tier 1) RED AUTHORED].'
    files_to_change:
      - 'examples/neatRacing/physics/drift.smoothness.test.ts'
      - 'examples/neatRacing/engine/passes/car.rear-slip.test.ts'
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - id: 'AC-109-RED (the no-wobble contracts)'
        text: 'ARCHIVED 2026-11-23 (the 01-planning handoff restructure, evidence batch 188, sub-block H): the full AC-109-RED contract text is byte-exact in the companion log. THE CONTRACT IN ONE LINE: the TWO new red homes are RED BY ASSERTION and NEVER by an import failure - examples/neatRacing/physics/drift.smoothness.test.ts probes the EXISTING drift model and the EXISTING step service only, and examples/neatRacing/engine/passes/car.rear-slip.test.ts probes the EXISTING rear-slip seam only - so every probe targets an EXISTING module and each red is a NAMED RUNTIME failure.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
      - id: 'AC-110-RED (the visible rear-slip contract)'
        text: 'ARCHIVED 2026-11-23 (the 01-planning handoff restructure, evidence batch 188, sub-block H): the full AC-110-RED contract text is byte-exact in the companion log. THE CONTRACT IN ONE LINE: the rear-slip red home is RED BY ASSERTION - it asserts that the rear wheels carry the PUBLISHED slip angle verbatim (the drift model publishes atan2(lateralVelocity, speed)) and that the rear vs the front differ by MORE than shading.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"'
    red_test:
      - 'the lane is RED BY ASSERTION and NEVER by an import failure: every probe targets an EXISTING module, so each red is a NAMED RUNTIME failure'
      - 'the lane RECORDS the MEASURED current wobble baseline (the damping ratio, the countersteer discontinuity, the slip chatter, the rear-wheel yaw of exactly 0) in this plan BEFORE any production edit'
      - 'the lane pins the invariants that MUST survive the green: the AC-087 never-mirrors law, the UNCLAMPED pose spring, the PURE drift model, the ai/oscillation-penalty.utils.ts fitness domain stays UNCOUPLED - THE DEBUG SEAM IS NOT PROBED BY THIS LANE: THE EIGHT-FIELD PIN IS AN ALREADY-PASSING INVARIANT OWNED BY `examples/neatRacing/browser-entry/browser-entry.test.ts`, IT IS NOT RE-AUTHORED HERE AND THE SEAM-NOT-WIDENED DUTY STAYS WITH THE GREEN LANE'
    parallelizable: false
    dependencies: []
  - slice_id: 'P5-S30-damping'
    title: 'THE DAMPING LAW: the zeta-0.35 pose spring + the bang-bang countersteer'
    status: '[DONE 2026-11-22 - 05-green-testing GREEN: OK: AC-109-DAMPING + AC-109-DAMPING-REPIN - THE TWO OWNED RED CONTRACTS RED-S30-1 AND RED-S30-2 ARE GREEN BY NAME, THE COUPLED CORPUS FLOORS ARE BYTE-IDENTICAL, THE FIVE MEASURED MARGINS REPRODUCE EXACTLY, THE COUNTERSTEER RAMP SWEEP IS MONOTONE WITH THE KNEE EXACTLY 0 AND FULL LOCK EXACTLY 1, THE BROWSER OBLIGATION IS N/A FOR THIS SLICE]'
    goal: implementing
    estimate_hours: 3
    research_artifact: 'ARCHIVED 2026-11-23 (the 01-planning handoff restructure, evidence batch 188, sub-block H): the AC-109-RED contracts authored by P5-S30-red, the R-2 disclosed re-pin obligation and the directly coupled existing assertions are byte-exact in the companion log. THIS SLICE IS CLOSED [DONE 2026-11-22 - 05-green-testing GREEN: OK].'
    files_to_change:
      - 'examples/neatRacing/physics/drift.model.ts'
      - 'examples/neatRacing/physics/drift.model.test.ts'
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - id: 'AC-109-DAMPING (the pose spring + the countersteer)'
        text: 'ARCHIVED 2026-11-23 (the 01-planning handoff restructure, evidence batch 188, sub-block H): the full AC-109-DAMPING contract text is byte-exact in the companion log. THE SLICE IS CLOSED AND GREEN BY NAME (RED-S30-1 AND RED-S30-2) AND THE COUPLED CORPUS FLOORS ARE BYTE-IDENTICAL.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
      - id: 'AC-109-DAMPING-REPIN (the disclosed re-pin + the coupled existing assertions)'
        text: 'ARCHIVED 2026-11-23 (the 01-planning handoff restructure, evidence batch 188, sub-block H): the full AC-109-DAMPING-REPIN contract text - THE DISCLOSED RE-PIN OBLIGATION (the carried item R-2) AND THE DIRECTLY COUPLED EXISTING ASSERTIONS - IS BYTE-EXACT IN THE COMPANION LOG. THE SLICE IS CLOSED AND GREEN BY NAME (RED-S30-1 AND RED-S30-2) AND NO BEHAVIOUR WAS WEAKENED.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
    red_test:
      - 'the RED contracts this lane makes GREEN BY NAME are the ones the closed RED lane authored: RED-S30-1 (the pose-spring damping ratio, measured 0.35 against a contract of at least 0.7) and RED-S30-2 (the countersteer continuity, measured a 63 m/s^2 adjacent jump against a contract of at most 7 m/s^2) in examples/neatRacing/physics/drift.smoothness.test.ts; the RED home is VERIFIED, never re-authored'
      - 'the failing assertion to land FIRST is the damping-dependent pin in the OWNER-LOCAL home: examples/neatRacing/physics/drift.model.test.ts:842 (the 12.6 damping pin) is RE-PINNED to the NEW published value in the SAME lane that changes the coefficient, and the coupled corpus floors are RE-READ and reported, never lowered'
      - 'the NEW overshoot assertion lands FIRST in the SAME owner-local home: the TRUE far-side overshoot (the excursion PAST the final pose on the far side, NOT the coded total excursion) is MEASURED and asserted at EXACTLY 0.0000 DEG at the pinned point, and the ADDITIVE assertion NEVER rewrites, renames, re-points or deletes the corpus travel-visibility floor at :1196 or :1240 (toBeGreaterThanOrEqual(2.0)), the settle window at :1207 or :1208, the per-tick step ceiling at :1219 or :1242 or the anti-teardown guard at :1239'
      - 'the RAMP FORM IS PINNED BEFORE THE COEFFICIENT MOVES: the 1001-sample sweep over |steer| in [0, 1] is RECORDED in this plan with the knee value (0), the full-lock value (1), the monotonicity proof, the maximum adjacent jump (at most 7 m/s2 at full speed) and the continuity at the knee, and the production JSDoc at examples/neatRacing/physics/drift.model.ts:466 is CORRECTED IN THE SAME SLICE (no deferred cleanup); the lane then hands the browser crossing re-base to P5-S30A-crossing-rebase and touches NO browser byte'
    parallelizable: false
    dependencies:
      - 'P5-S30-red'
  - slice_id: 'P5-S30-repin'
    title: 'THE STEP-SERVICE RE-PIN: the slip hysteresis + the steering smoothing and yaw-rate feedback'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    research_artifact: 'the AC-109-RED contracts authored by P5-S30-red + the step-service seam map (examples/neatRacing/physics/physics.step.service.ts:781 the demanded turn rate, :787 the heading advance, :852 the per-tick slip re-derivation, :863 the pose spring call, :875 the travel heading, :904 the published slip)'
    files_to_change:
      - 'examples/neatRacing/physics/physics.step.service.ts'
    acceptance_criteria:
      - id: 'AC-109-REPIN (the hysteresis + the yaw-rate feedback)'
        text: 'the re-pin lands in examples/neatRacing/physics/physics.step.service.ts: the per-tick slip re-derivation gains a DEPTH HYSTERESIS (a published slip inside the published chatter band is HELD instead of re-derived, so the sign no longer flips tick to tick) and the demanded turn rate gains a STEERING SMOOTHING and YAW-RATE FEEDBACK (the heading advance is fed back from the MEASURED yaw rate instead of stepping from the raw demand); the published slipAngle, the travel heading (the nose heading plus the PUBLISHED slip) and the traction severity stay the SAME numbers for EVERY consumer, and the fixed timestep, the published steering rate and the AI turn-rate authority stay UNCHANGED.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
    parallelizable: false
    dependencies:
      - 'P5-S30-damping'
  - slice_id: 'P5-S30-slip'
    title: 'THE VISIBLE REAR SLIP: the rear wheels yaw with the PUBLISHED slip (the FOLDED RENDER WORK)'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    research_artifact: 'the AC-110-RED contracts authored by P5-S30-red + the renderer seam map (examples/neatRacing/engine/passes/car.pass.ts:466 the pose state, :605 computeFrontWheelSteerYawRad, :1408 the wheel mounts, :1748 the wheel draw loop, :1771 the rear-wheel yaw of exactly 0) + the debug-seam field list (examples/neatRacing/browser-entry/browser-entry.ts:253 to :304, EXACTLY EIGHT fields)'
    files_to_change:
      - 'examples/neatRacing/engine/passes/car.pass.ts'
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - id: 'AC-110-SLIP (the rear wheels carry the PUBLISHED slip)'
        text: 'the renderer consumes the PUBLISHED slip VERBATIM: in examples/neatRacing/engine/passes/car.pass.ts the wheel draw loop yaws the REAR wheels with the published slip instead of EXACTLY 0 (the front wheels keep their steer yaw under AC-088, the mount yaw stays the mount yaw, the per-wheel tire grip stays per wheel and the wheel spin stays the wheel spin), and the drift pose state carries the published slip WITHOUT widening the browser debug seam (examples/neatRacing/browser-entry/browser-entry.ts keeps EXACTLY EIGHT readonly fields); the optional visual channels (skid marks, smoke, tire trails, particles) stay OPTIONAL and are NOT taken - the recommended visibility IS the rear-wheel yaw itself.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"'
      - id: 'AC-110-SLIP-SEAM (the debug seam is NOT widened)'
        text: 'the browser debug seam keeps EXACTLY EIGHT readonly fields (followOffsetRad, poseRad, poseRateRadPerS, slipAngleRad, speedMps, tickHz, tick, tractionSeverity) - the rear slip reaches the renderer through the EXISTING published slip channel and NOT through a NINTH field, so the seam-pinning contract that passed BEFORE this step still passes AFTER it.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
    parallelizable: false
    dependencies:
      - 'P5-S30-repin'
  - slice_id: 'P5-S30-green'
    title: 'The GREEN lane: the whole Step-30 validation'
    status: '[PLANNED]'
    goal: green-testing
    estimate_hours: 3
    research_artifact: 'the AC-109 + AC-110 red homes (authored by P5-S30-red) + the four implementing lanes + the standing USER RULE for the visible-browser validation + the Step-30 invariants list'
    files_to_change:
      - 'examples/neatRacing/physics/drift.smoothness.test.ts'
      - 'examples/neatRacing/engine/passes/car.rear-slip.test.ts'
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - id: 'AC-109-GREEN (the whole-step no-wobble green)'
        text: 'the no-wobble lane is GREEN BY NAME (the RED home is VERIFIED, never re-authored) and the MEASURED wobble baseline moved: the visual pose spring no longer rings, the countersteer is continuous, the published slip no longer chatters across zero and the heading no longer steps; the drift model stays PURE, the ai/oscillation-penalty.utils.ts fitness domain stays UNCOUPLED, the AC-087 never-mirrors invariant and the UNCLAMPED pose spring are UNCHANGED, and the published slip, the travel heading and the traction severity stay the SAME numbers for every consumer.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"'
      - id: 'AC-110-GREEN (the whole-step visible rear slip)'
        text: 'the visible rear slip is GREEN BY NAME: the rear wheels carry the PUBLISHED slip angle verbatim, the rear vs the front differ by MORE than shading, the front-wheel steer yaw is UNCHANGED, the debug seam still carries EXACTLY EIGHT fields, and the VISIBLE-BACKGROUND browser proof PASSES under the standing USER RULE (the rear slip is SEEN in the rendered frame).'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"'
    red_test:
      - 'the green lane VERIFIES the TWO AC-109 and AC-110 RED homes BY NAME and never re-authors them; a RED home edited by an implementing slice is a CONTRACT VIOLATION and is REPORTED, not absorbed'
      - 'the green lane runs the VISIBLE-BROWSER validation under the STANDING USER RULE: the window is VISIBLE, the harness NEVER makes the window always-on-top, NEVER requires the foreground, NEVER steals focus, and FPS is NOT required'
      - 'the green lane records the MEASURED after-baseline against the recorded before-baseline in this plan and compares the NUMBERS, not the adjectives'
    parallelizable: false
    dependencies:
      - 'P5-S30-damping'
      - 'P5-S30-repin'
      - 'P5-S30-slip'
evidence:
  gate_outputs:
    - 'THE GATE-TRANSCRIPT ARCHIVE (2026-11-23, the 01-planning handoff restructure): the SIXTEEN SINGLE-QUOTED GATE-TRANSCRIPT ENTRIES THAT THIS BLOCK HELD ARE BYTE-EXACT IN plans/neatracing-voxel-racer.logs.md (evidence batch 188, sub-block I) - the CONSOLIDATED slice-advancement GATE TRANSCRIPTS, the HOUSE DOUBLE-RUN ROUNDS AND THE PLAN-SYNC VALIDATOR RUNS ARE ALL PRESERVED THERE. THIS ENTRY IS AN INDEX, NOT A NEW GATE RESULT.'
```

#### Step 30A — THE BROWSER CROSSING CONTRACT RE-BASE: the AI-mode goal-crossing rebase surface against the PINNED ramp [PLANNED]

**Why this step exists (the lawful prerequisite boundary).** Step 30 already carries exactly
FIVE slices (`P5-S30-red`, `P5-S30-damping`, `P5-S30-repin`, `P5-S30-slip`, `P5-S30-green`),
which is the hard `MAX_SLICES_PER_STEP = 5` cap, and the browser crossing contract needs
`examples/neatRacing/browser-entry/browser-entry.ts` plus
`examples/neatRacing/host/host.wiring.test.ts` — neither of which is in the
`P5-S30-damping` slice file set, and `P5-S30-slip` (which owns the browser file) carries the
VISIBLE REAR SLIP intent, so folding the re-base into it would put TWO behavioural intents
into ONE slice. The `00-helping` cross-tier decision ACCEPTED by Agent Zero on 2026-11-22
therefore splits the re-base OUT into this prerequisite boundary step, whose ONE slice
DEPENDS on `P5-S30-damping`. This is the `25A`/`26A` precedent again: a boundary step
carries NO new user desire — it exists only to keep every slice atomic, every file set at
most three files, and every cap intact.

**The ordering IS the point (the accepted decision, clause (7) of `AC-109-DAMPING-REPIN`).**
The pinned countersteer ramp changes the MEASURED crossing-lap drivability
(`meanForwardSpeed`, `oscillationMetric`, `offTrackFraction`), those facts feed the TP-17
drivability gate, the gate decides the promotion, the promotion produces `rebaseRung`, and
`rebaseRung` is applied by `rebaseRoundAtPromotion`. Re-basing the crossing contract against
a ramp that is not pinned yet would bake a pre-amendment number into the contract, so the
re-base runs ONLY AFTER `P5-S30-damping` has pinned the ramp form and the damping value.

**The seam map (every citation read off the LIVE bytes on 2026-11-22).**
- `examples/neatRacing/browser-entry/browser-entry.ts` — `:1132` the
  `NeatRacingAiModeRebaseDecision` interface (the `rebaseRung` field and the level the host
  round must rebase onto), `:1418` `resolveNeatRacingAiModeRebaseRung`, `:2672`
  `rebaseRoundAtPromotion`, `:2725` `feedAiModeGoalCrossing` (the crossing-lap drivability
  facts: `meanForwardSpeed`, `scriptedBaselineLapPace`, `oscillationMetric` and
  `offTrackFraction`).
- `examples/neatRacing/ai/ai-mode.goals.ts` — `:395` `recordNeatRacingAiModeGoalCrossing`
  (the goal machine that records the crossing) and `:464` the drivability normaliser.
- `examples/neatRacing/ai/promotion.rules.ts` — `:457` `evaluateNeatRacingDrivabilityGate`
  (TP-17) and `:517` `evaluateNeatRacingPromotion` (the conditions a/b/c).
- `examples/neatRacing/ai/runtime.adaptation.ts` — `detectSteeringOscillation`, the
  steering-oscillation metric the crossing lap consumes.
- `examples/neatRacing/host/host.wiring.test.ts` — `:995-1031` the FOUR host-side
  `rebaseRung` contract assertions (null for a non-forward crossing, null for a no-lap
  crossing, null for a non-promoting lap, and the promoted rung equals
  `promoted.session.level`).

**The invariants that MUST survive the green.** The eight-field browser debug seam is NOT
widened; the TP-17 drivability gate is NOT weakened; the exported countersteer contracts
stay owned by `P5-S30-damping` and are NOT touched here; the AI fitness domain
(`examples/neatRacing/ai/oscillation-penalty.utils.ts`) stays UNCOUPLED; and the four
host-side assertions are EXTENDED, never relaxed, never deleted and never re-pointed.

```yaml
phase: 5
step: '30A'
title: 'THE BROWSER CROSSING CONTRACT RE-BASE: the AI-mode goal-crossing rebase surface against the PINNED ramp'
status: '[PLANNED]'
wave_note: 'AUTHORED 2026-11-22 BY THE 01-planning (AUTHOR) INSTANCE AS THE LAWFUL PREREQUISITE BOUNDARY OF THE 00-helping CROSS-TIER DECISION ACCEPTED BY AGENT ZERO (TRACKER-ONLY: THIS PLAN IS THE ONLY BYTE WRITTEN; NO PRODUCTION BYTE, NO TEST BYTE, NO SPEC BYTE, NO BUNDLE BYTE, NO GENERATED-DOC BYTE, NO COMMIT, NO BRANCH, NO PR, NO MUTATING GIT COMMAND, AND NO STATUS TOKEN FLIPPED - THIS PACKET IS [PLANNED] AND NOTHING HERE IS [WIP]): STEP 30 ALREADY CARRIES EXACTLY FIVE SLICES (THE MAX_SLICES_PER_STEP = 5 HARD CAP) AND THE BROWSER CROSSING CONTRACT NEEDS examples/neatRacing/browser-entry/browser-entry.ts PLUS examples/neatRacing/host/host.wiring.test.ts, WHICH ARE OUTSIDE EVERY STEP-30 SLICE FILE SET EXCEPT P5-S30-slip (WHOSE ONE BEHAVIOURAL INTENT IS THE VISIBLE REAR SLIP) - SO THE RE-BASE IS SPLIT OUT HERE INSTEAD OF FOLDING TWO INTENTS INTO ONE SLICE OR WIDENING A SLICE; THE HARD ORDERING IS THAT THE RE-BASE RUNS ONLY AFTER P5-S30-damping HAS PINNED THE COUNTERSTEER RAMP FORM AND THE DAMPING VALUE, BECAUSE THE PINNED RAMP CHANGES THE MEASURED CROSSING-LAP DRIVABILITY, THE DRIVABILITY FACTS FEED THE TP-17 GATE, THE GATE DECIDES THE PROMOTION, THE PROMOTION PRODUCES rebaseRung AND rebaseRung IS APPLIED BY rebaseRoundAtPromotion - SO THE CONTRACT MUST NEVER BE RE-BASED AGAINST AN UNPINNED RAMP; THIS STEP CARRIES NO NEW USER DESIRE AND IS THE 25A/26A PRECEDENT; THE FOLD-OUT IS DISCLOSED IN THE STEP-30 wave_note, IN THE PHASE-5 STEP MAP, IN AC-PLN-012 AND IN THE placeholder_steps LIST.'
goal: implementing
expansion: slices
auto_expand: true
tdd_sequence: red-green
mode: fresh-session
source_of_truth: 'plans/neatracing-voxel-racer.plans.md'
copy_paste: true
next_step: 'Step 31 - The traction law (AC-111)'
pre_execute_hook:
  tool: 'neataptic-workflow-mcp-get_slice_context'
  args:
    slice_id: 'P5-S30A-crossing-rebase'
skills:
  - implementation-standards
  - red-test-contracts
  - green-validation-gates
  - spec-checklist
validation:
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
  - 'npm run jest:base -- --testPathPatterns="examples/neatRacing/host"'
  - 'npx tsc -p tsconfig.neatRacing.json --noEmit'
acceptance_criteria:
  - id: 'AC-109A-CROSSING-REBASE (the browser crossing contract, re-based against the PINNED ramp)'
    text: 'THE BROWSER CROSSING CONTRACT IS THE AI-MODE GOAL-CROSSING REBASE SURFACE AND IT IS RE-BASED AGAINST THE COUNTERSTEER RAMP PINNED BY P5-S30-damping: THE PINNED RAMP FORM, THE PINNED DAMPING VALUE AND THE MEASURED PRE-RE-BASE CROSSING-LAP BASELINE ARE READ FROM THIS PLAN, NEVER RE-DERIVED AND NEVER ASSUMED; THE ORDERING IS HARD - THIS LANE RUNS ONLY AFTER P5-S30-damping IS GREEN (THE dependencies LIST NAMES IT) AND THE PINNED RAMP IS RECORDED IN THIS PLAN BEFORE ANY BYTE IS TOUCHED; THE CROSSING SURFACE STAYS THE SAME SURFACE - examples/neatRacing/browser-entry/browser-entry.ts:1132 (NeatRacingAiModeRebaseDecision), :1418 (resolveNeatRacingAiModeRebaseRung), :2672 (rebaseRoundAtPromotion) AND :2725 (feedAiModeGoalCrossing) KEEP THEIR IDENTITIES, THEIR SIGNATURES AND THEIR EXPORTED SHAPES, AND THE RE-BASE CHANGES THE SELECTED RUNG (rebaseRung) AGAINST THE PINNED RAMP, NOT THE SHAPE OF THE CONTRACT; THE HOST-SIDE CONTRACT IS EXTENDED AND NEVER WEAKENED - THE FOUR ASSERTIONS AT examples/neatRacing/host/host.wiring.test.ts:995-1031 (rebaseRung IS NULL FOR A NON-FORWARD CROSSING, NULL FOR A NO-LAP CROSSING, NULL FOR A NON-PROMOTING LAP, AND THE PROMOTED RUNG EQUALS promoted.session.level) KEEP THEIR MEANING AND A NEW ASSERTION PINS THE RE-BASED RUNG AGAINST THE PINNED RAMP; THE MEASURED CROSSING-LAP DRIVABILITY IS THE EVIDENCE - THE LANE RE-MEASURES meanForwardSpeed, scriptedBaselineLapPace, oscillationMetric AND offTrackFraction ON THE CROSSING LAP UNDER THE PINNED RAMP, RECORDS THE BEFORE AND AFTER NUMBERS IN THIS PLAN AND COMPARES THE NUMBERS, NOT THE ADJECTIVES; THE BOUNDARIES HOLD - THE LANE TOUCHES ONLY THE THREE FILES IN ITS files_to_change LIST, THE EIGHT-FIELD BROWSER DEBUG SEAM IS NOT WIDENED, THE TP-17 DRIVABILITY GATE AT examples/neatRacing/ai/promotion.rules.ts:457 AND :517 IS NOT WEAKENED, THE AI FITNESS DOMAIN (examples/neatRacing/ai/oscillation-penalty.utils.ts) STAYS UNCOUPLED, AND THE EXPORTED COUNTERSTEER CONTRACTS STAY OWNED BY P5-S30-damping.'
    validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
slices:
  - slice_id: 'P5-S30A-crossing-rebase'
    title: 'THE CROSSING RE-BASE: the AI-mode goal-crossing rebase surface against the PINNED ramp'
    status: '[PLANNED]'
    goal: implementing
    estimate_hours: 3
    research_artifact: 'the AI-MODE GOAL-CROSSING REBASE SEAM MAP (examples/neatRacing/browser-entry/browser-entry.ts:1132 the NeatRacingAiModeRebaseDecision interface, :1418 resolveNeatRacingAiModeRebaseRung, :2672 rebaseRoundAtPromotion, :2725 feedAiModeGoalCrossing; examples/neatRacing/ai/ai-mode.goals.ts:395 recordNeatRacingAiModeGoalCrossing AND :464 the drivability normaliser; examples/neatRacing/ai/promotion.rules.ts:457 the TP-17 drivability gate AND :517 evaluateNeatRacingPromotion; examples/neatRacing/ai/runtime.adaptation.ts detectSteeringOscillation; examples/neatRacing/host/host.wiring.test.ts:995-1031 the four host-side rebaseRung contract assertions) + the PINNED RAMP FORM AND THE PINNED DAMPING VALUE FROM P5-S30-damping (READ FROM THIS PLAN, NEVER RE-DERIVED) + the accepted 00-helping decision of 2026-11-22 (the re-base runs AFTER the ramp is pinned)'
    files_to_change:
      - 'examples/neatRacing/browser-entry/browser-entry.ts'
      - 'examples/neatRacing/host/host.wiring.test.ts'
      - 'plans/neatracing-voxel-racer.plans.md'
    acceptance_criteria:
      - id: 'AC-109A-CROSSING-REBASE'
        text: 'THE RE-BASED RUNG IS SELECTED AGAINST THE PINNED RAMP AND THE HOST-SIDE CONTRACT IS EXTENDED, NEVER WEAKENED: resolveNeatRacingAiModeRebaseRung KEEPS ITS IDENTITY AND ITS SIGNATURE, ITS RESULT IS RE-BASED AGAINST THE PINNED RAMP AND THE PINNED DAMPING VALUE READ FROM THIS PLAN, THE FOUR EXISTING rebaseRung ASSERTIONS AT examples/neatRacing/host/host.wiring.test.ts:995-1031 STAY MEANINGFUL AND ARE NEVER RELAXED, DELETED OR RE-POINTED, A NEW ASSERTION PINS THE RE-BASED RUNG AGAINST THE PINNED RAMP, AND THE MEASURED BEFORE-AND-AFTER CROSSING-LAP DRIVABILITY (meanForwardSpeed, scriptedBaselineLapPace, oscillationMetric, offTrackFraction) IS RECORDED IN THIS PLAN; THE LANE TOUCHES ONLY ITS THREE DECLARED FILES, THE EIGHT-FIELD DEBUG SEAM IS NOT WIDENED AND THE TP-17 GATE IS NOT WEAKENED.'
        validation: 'npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"'
    red_test:
      - 'the lane is RED BY ASSERTION and NEVER by an import failure: every probe targets an EXISTING module (the browser-entry debug seam and the host wiring), so each red is a NAMED RUNTIME failure'
      - 'the FIRST failing assertion to land is the RE-BASED RUNG assertion in examples/neatRacing/host/host.wiring.test.ts: the rung selected by resolveNeatRacingAiModeRebaseRung UNDER THE PINNED RAMP is asserted against the pinned value, and the FOUR EXISTING assertions at :995-1031 are NEVER rewritten, relaxed, deleted or re-pointed'
      - 'the lane RECORDS THE MEASURED CROSSING-LAP DRIVABILITY (meanForwardSpeed, scriptedBaselineLapPace, oscillationMetric, offTrackFraction) BEFORE AND AFTER the re-base AGAINST THE PINNED RAMP, AND COMPARES THE NUMBERS, NOT THE ADJECTIVES'
    parallelizable: false
    dependencies:
      - 'P5-S30-damping'
```

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

THREE carry NEEDS CLARIFICATION markers with RECORDED DEFAULTS (CLAR-4-01, CLAR-4-02 and
CLAR-5-01), so the plan-wide marker budget stands at 3 of 3 ACTIVE and NO further marker may
be created; the spec records every OTHER open question RESOLVED (section 20). If execution
uncovers a spec conflict, stop and escalate via `00.cross-tier-helper` before deviating.

- **[ARCHIVED 2026-11-23 to `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block G) — the RESOLVED AND SUPERSEDED `## Deferred questions` ENTRIES (the 2026-10-26 Step-14N RESOLVED ENTRIES — the drift-feel DIRECTION, THE P0-4 ENTRY-ATTACK CONTRACT, THE A-8 STATE-SHAPE AMENDMENT AND THE ONE-OWNER-PER-FILE DECISION — THE WITHDRAWN FOUR-FIELD DEBUG-SEAM CLAUSE, THE 2026-11-12 `P5-S27-3` PAUSE NOTE AND ITS 2026-11-21 SUPERSEDING ROW) ARE ARCHIVED BYTE-EXACT. THE STILL-OPEN ENTRIES ARE BELOW: THE CANOPY DESYNC, THE PRODUCER-DECLARED AI INTENT BIT, THE OPEN W-1 GATE-MAINTENANCE ITEM, THE OPEN O-11 CORTEX-INDEX ITEM AND THE THREE NEEDS-CLARIFICATION MARKERS WITH THEIR RECORDED DEFAULTS.]**

- **OPEN (2026-10-26, Step 14N) — the canopy desync.** `examples/neatRacing/engine/passes/glass.pass.ts` reads `car.yawRad` only and never the drift pose (integration proposal R-03), so a 0.3050 rad (17.5 deg) body slide is a canopy error. The fix needs a new render-bag field - a RENDER-CONTRACT change spanning car.pass.ts, glass.pass.ts and the bag type - and it has NO test home inside 14N, so it is DEFERRED to a NAMED FOLLOW-UP step (the integration proposal recommends a step 14O) rather than widening 14N. Mitigation inside 14N: the AC-090 user gate asks Q4 (no canopy or visual REGRESSION) explicitly, so the gap cannot ship unnoticed. The follow-up RE-OPENS if the user gate reports a canopy mismatch.
- **OPEN (2026-10-26, Step 14N) — the producer-declared AI intent bit.** The 3-clause Phase-2-safe intent contract (the countersteer trigger reads a DEDICATED intent signal rather than sign(steer); the intent is PRODUCER-DECLARED - the human lane reads the key state, the AI lanes explicitly declare no intent; the intent is HELD for a minimum window) is RECORDED as the model specification, and 14N implements only the IN-MODEL FALLBACK (D2.3: `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER` band 0.85-0.95 + `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS` band 6-10; the latency trade-off against the GF-15 4 Hz chatter is recorded). The producer-declared bit itself (a new `PhysicsStepInput` field + `examples/neatRacing/host/input.ts` + `examples/neatRacing/physics/ai.autopilot.ts`) is DEFERRED to a follow-up. RESIDUAL RISK (RB-6): an AI saturated |steer| = 1 correction still satisfies the countersteer gate, so an AI heading-error zero-crossing can still become an accidental countersteer until the bit lands. The Phase-2 seam to amend is `examples/racing_curriculum/controller/observation.assembler.ts` (NOT `examples/neatRacing/ai/observation.assembler.ts`, which does not exist), and the observation channel-74 normalisation compresses 3.5x under the new MAX_SLIP (0.45 rad) and must be re-pinned before any Phase-2 lateral-velocity model is adopted.

- **OPEN (2026-10-26, Step 14N PATCH) — W-1, the `plan-readiness` gate's SCOPING defect.** The readiness scan reads the WHOLE `## Latest validation evidence` section, so it finds the STALE green-light markers of OTHER steps (Step 14J at the plan line ~4377, block `verified-14J-user-clar-9-reverify`, and Steps 14F/14G/14H at ~4636, block `green-light-14F-14G-14H`) and reports a false-positive `green-light: true` for Step 14N. The 14J marker MUST NOT be deleted (it is a true historical record), so the fix is a SCOPING change in the gate - the scan must be restricted to the block belonging to the step under evaluation. This is a WORKFLOW-GATE defect, NOT a packet defect: it is ROUTED to `00-helping` / `agent-maintenance-coordinator` as a separate gate-maintenance item, and it is NOT a blocker on the Step 14N packet itself.
- **OPEN (2026-10-26, Step 14N PATCH) — O-11, the stale Cortex index for the plan family.** The Cortex index for the plan family is STALE (`index_fresh: false`, stale path `plans/neatracing-voxel-racer.plans.md`); every other source family is fresh. The re-index is the ORCHESTRATOR's call (re-indexing is content-based, so it goes stale again after the next plan edit): `node rag-index/build-index.mjs --files=plans/neatracing-voxel-racer.plans.md && node rag-index/embed-index.mjs --files=plans/neatracing-voxel-racer.plans.md`. This instance used native view/grep as the degraded-Cortex fallback for the exact line evidence.
- **OPEN (2026-11-07, PHASE-4 AUTHORING) — CLAR-4-01 (NEEDS CLARIFICATION; RECORDED DEFAULT — execution proceeds on the default unless the user answers otherwise at review).** At ladder rung 1 the fleet layout is the lone car 0 (the human's car per LD-12; `NEATRACING_TEAM_LAYOUTS` lvl 1 = 1 car), so WHO is the evolving agent whose growth feeds the promotion predicate at rung 1? RECORDED DEFAULT: AI mode joins ONE dedicated background NGE carrier alongside car 0 at EVERY rung — its laps/crossings feed the predicate, it becomes the round's `seedNetwork` carrier, and it is the panel's visible agent from rung 1; the rung's listed car count keeps counting the RIVAL cars (2 at lvl 2, 4 rivals at lvl 3). The marker budget stands at 2 of 3 active.
- **OPEN (2026-11-07, PHASE-4 AUTHORING) — CLAR-4-02 (NEEDS CLARIFICATION; RECORDED DEFAULT — execution proceeds on the default unless the user answers otherwise at review).** The AI-mode visualizer surface the user verdict names ("fullscreen") — is it a TOGGLABLE fullscreen overlay over the current right-side panel, or the DEFAULT AI-mode layout (the panel retired in AI mode)? RECORDED DEFAULT: a fullscreen-capable surface, VISIBLE BY DEFAULT in AI mode, togglable, with manual mode unchanged (no manual-mode regression, no dual-path). The marker budget stands at 2 of 3 active.
- **OPEN (2026-11-12, PHASE-5 AUTHORING) — CLAR-5-01 (NEEDS CLARIFICATION; RECORDED DEFAULT — execution proceeds on the default unless the user answers otherwise at review).** The USER directive, verbatim: "green 2-hex start line". WHAT DOES "2-HEX" MEAN — TWO hexes DEEP along the TRACK axis (a 4.0 m green stripe across the FULL road width, 2 x NEATRACING_ROAD_EDGE_HEX_DIAMETER_M), ONE hex deep (2.0 m), or TWO hexes WIDE across the road (a lateral pair)? RECORDED DEFAULT: **TWO hexes DEEP along the track axis** (a 4.0 m band spanning the FULL road width), and the band is a NEW ALONG-TRACK classification — the `TileColorway` union stays UNCHANGED at its FOUR members, because a fifth member would change `colorwayToColors` + `COLORWAY_PRECEDENCE` and break the `road-edge.test.ts` pins (GAP-5-01). This is the ONLY new NEEDS CLARIFICATION marker of this pass, so the marker budget stands at **3 of 3 active** (CLAR-4-01 + CLAR-4-02 + CLAR-5-01) — no further marker may be created in this phase, and every further open decision is a RECORDED DEFAULT (Q2..Q5 in the Phase-5 GAPS + EVIDENCE section).

## Latest validation evidence

<!-- EVIDENCE-APPEND-POINT: the compact resume index lives HERE (append new evidence rows BELOW this marker); the full evidence history is archived byte-exact in plans/neatracing-voxel-racer.logs.md (evidence batch 188, sub-blocks A-M, AND THE FULL HISTORY BACK TO BATCH 175.

- **VALIDATION_EVIDENCE — COUNTDOWN-FIRST PLANNING AMENDMENT (01-planning, TRACKER-ONLY):** authored Step 27A / `P5-S27A-red` / `P5-S27A-overlay` / `P5-S27A-green` and AC-106-R1..R3; made Step 27A and its RED slice the sole active WIP frontier; moved Step 30 and `P5-S30-repin` to `[PLANNED]` without clearing the existing BLOCKED verdict. The first production implementation is now `P5-S27A-overlay`: transparent Astrobird-like overlay, scene visible throughout, no black/opaque backdrop during or after countdown, clean dismissal, and post-GO simulation/input resume. Chrome MCP is minimized, ideally to one final visible-background validation; no foreground/focus/topmost forcing for this repair and no FPS requirement. **NOT RUN:** racer code edits, test edits or runs, builds, browser/Chrome, plan/slice gates, `get_slice_context`, and subagents were all intentionally omitted under the user planning-only constraint. This author records NO green light; a fresh verification instance is the next boundary. The two permitted post-edit commands are the RAG build-index and Cortex-index freshness check, each once with no retries; their results are returned in the task output rather than followed by another tracker edit.

**THE COMPACT VERDICT INDEX (2026-11-23, the 01-planning HANDOFF-RESTRUCTURE pass — the authoritative verdict markers of this plan, carried forward TRUTHFULLY FROM THE ROWS BELOW; the FULL evidence history is archived byte-exact at `plans/neatracing-voxel-racer.logs.md`, evidence batch 188, sub-block D):**

- **THE ACTIVE STEP-27A SLICE HAS NO GREEN LIGHT. THE ONE `[WIP]` SLICE IS `P5-S27A-red`; FRESH VERIFICATION IS REQUIRED BEFORE RED EXECUTION, AND THE FIRST PRODUCTION IMPLEMENTATION MUST BE `P5-S27A-overlay`. `P5-S30-repin` IS NOW `[PLANNED]` BEHIND STEP 27A; ITS PRIOR INDEPENDENT VERDICT REMAINS **BLOCKED** (B-1 SCOPE/LEGACY-FLOOR, B-2 AI-CHANNEL AUTHORITY), IT HAS NO GREEN LIGHT, AND THIS INDEX UNBLOCKS NO `04-implementing` LANE.**
- **THE SCOPED GREEN LIGHTS THAT ARE RECORDED (EACH ONE IS SCOPED TO ITS OWN SLICE AND TO NOTHING ELSE):** `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S30-damping` ONLY (2026-11-22, the fresh independent verification of the closed damping lane, BLOCKERS NONE); `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S30-red` ONLY (2026-11-22, the fresh independent re-verification after the three-blocker patch cycle, BLOCKERS NONE); `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S29-green` ONLY; `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S29-neon` ONLY; `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S29-red` ONLY; `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S28-green` ONLY; `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S28-repin` ONLY; `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S28-law` ONLY; `green-light: true` — EXPLICITLY SCOPED TO THE SLICE `P5-S28-red` ONLY. (THE PHASE-1 .. PHASE-4 AND STEP-27 .. STEP-29 CLOSING VERDICTS ARE ARCHIVED BYTE-EXACT IN THE COMPANION LOG.)
- **THE CONVERGENCE MARKERS (THE `fix-loop` ITERATIONS, CARRIED FORWARD BYTE-EXACT AND IN THE ORIGINAL ORDER — A `passed` MARKER RESETS THE COUNTER; THE FULL TRANSCRIPTS ARE IN THE COMPANION LOG, SUB-BLOCK D, AND THE MARKER SOURCE IS IN SUB-BLOCK L):**

    - fix-loop: P5-S30-red iteration 2 status=passed (THE ITERATION CONVERGED: THE FRESH INDEPENDENT RE-VERIFICATION OF 2026-11-22 CONFIRMED ALL THREE OF THE PRIOR BLOCKERS CORRECTED AGAINST THE LIVE WORKING-TREE BYTES, CONFIRMED ALL FIVE RED CONTRACTS FAIL TODAY BY ASSERTION (NEVER BY AN IMPORT FAILURE), CONFIRMED THE EIGHT-FIELD SEAM PIN IS AN ALREADY-PASSING INVARIANT OWNED BY `examples/neatRacing/browser-entry/browser-entry.test.ts` WITH THE SEAM-NOT-WIDENED DUTY IN THE GREEN LANE, AND CONFIRMED THE PUBLISHED-SLIP BOUND IS STATED IN THE RIGHT DIRECTION — SO THE `P5-S30-red` RED LANE IS GREEN-LIT AT THE CONTRACT LEVEL, `green-light: true`, SCOPED TO `P5-S30-red` ONLY, BLOCKERS: NONE, AND THE `03-red-testing` LANE IS AUTHORIZED TO AUTHOR THE TWO RED HOMES. THE CONVERGENCE COUNTER RESETS TO 0 ON THIS `passed` MARKER. NO STATUS TOKEN WAS FLIPPED: `P5-S30-red` REMAINS THE BARE `[WIP]`, THE OTHER FOUR STEP-30 SLICES REMAIN `[PLANNED]`, STEP 30 AND PHASE 5 REMAIN `[WIP]` AND THE PLAN `**Status:**` REMAINS `[WIP]`. RECORDED 2026-11-22 BY THE 01-planning `P5-S30-red` FRESH-INDEPENDENT RE-VERIFICATION PASS.)
    - fix-loop: P5-S30-red iteration 1 status=failed (THE ITERATION DID NOT CONVERGE: THE CONSOLIDATED GATE PASSED 4 OF 4 BUT THE FRESH INDEPENDENT VERIFICATION FOUND THREE PLAN-TEXT BLOCKERS — THE `car.pass.ts` IDENTIFIER DRIFT, THE BACKWARDS PUBLISHED-SLIP BOUND AND THE UNPLACEABLE `RED-S30-6` CONTRACT — SO THE `P5-S30-red` RED LANE IS NOT GREEN-LIT AND IS NOT DISPATCHED. THE REMEDY IS A TRACKER-ONLY `01-planning` PATCH CYCLE FOLLOWED BY A NEW FRESH VERIFICATION INSTANCE; THE ORCHESTRATOR OWNS THE PATCH-TO-VERIFY LOOP. RECORDED 2026-11-22 BY THE 01-planning `P5-S30-red` FRESH-INDEPENDENT-VERIFICATION PASS.)
  - fix-loop: P5-S29-neon iteration 1 status=passed (THE ITERATION IS CLOSED ON THE FIRST PASS — EVERY ALLOW-LISTED COMMAND PASSED, THE CONSOLIDATED `slice-advancement` GATE PASSED 7 OF 7 WITH `failedGates: []`, THE VISIBLE-BACKGROUND BROWSER PROOF PASSED BOTH NEON ASSERTIONS WITH NO FOCUS STEAL AND NO FPS, AND NO LOOP-BACK TO `04-implementing` WAS NEEDED; THE SLICE-STATUS FLIP REMAINS THE ORCHESTRATOR'S CLOSURE ACT).
  - fix-loop: P5-S28-repin iteration 1 status=passed (THE ITERATION IS CLOSED — EVERY ALLOW-LISTED COMMAND PASSED, THE CONSOLIDATED GATE PASSED 7 OF 7 AND THE VISIBLE-BROWSER SMOKE PASSED ALL FIVE ASSERTIONS; NO LOOP-BACK TO `04-implementing` WAS NEEDED; THE SLICE-STATUS FLIP REMAINS THE ORCHESTRATOR'S).
    - fix-loop: P5-S28-law iteration 1 status=passed (THE ITERATION IS CLOSED - THE AC-107-LAW BEHAVIOUR IS GREEN (GREEN: OK) AND THE SC-02 BOUNDARY BLOCKER WAS CLEARED BY THE TRACKER AMENDMENT ITSELF, NOT BY A ROLLBACK; THE FRESH TIER-3 slice-validator RE-VERDICT IS PASS WITH NO REGRESSION AND THE 2026-11-22 INDEPENDENT-VERIFICATION ROW IS THE AUTHORITATIVE green-light: true MARKER, EXPLICITLY SCOPED TO THIS SLICE. ADVANCED 2026-11-22 BY THE 01-planning P5-S28-law CLOSURE PASS.)
  - fix-loop: P5-S27-3 iteration 1 status=passed

- **THE `P5-S30-repin` SLICE HAS NO `fix-loop` MARKER — NO ITERATION OCCURRED, AND IT HAS NO GREEN LIGHT. THE EARLIER `P5-S27-3` FAILED ITERATION AND THE Phase-1 .. Phase-4 MARKERS ARE ARCHIVED BYTE-EXACT IN THE COMPANION LOG.**

- **VALIDATION_EVIDENCE — THE INDEPENDENT VERIFICATION OF `P5-S30-repin` (2026-11-23, THE FRESH `01-planning` (Tier 1) VERIFICATION INSTANCE — THE VERDICT IS **BLOCKED**, EXPLICITLY SCOPED TO THE SLICE `P5-S30-repin` ONLY; TWO BLOCKERS (B-1 SCOPE/LEGACY-FLOOR, B-2 AI-CHANNEL AUTHORITY) AND TWO BOUNDED CLARIFICATIONS (C-1 THE CHATTER BAND, C-2 THE VALIDATION LIST); THE `P5-S30-repin` TOKEN STAYS THE BARE `[WIP]`, PHASE 5 STAYS `[WIP]`, STEP 30 STAYS `[WIP]` AND THE STEP-30 `pre_execute_hook` STAYS RE-BOUND TO `P5-S30-repin`; NO STATUS TOKEN WAS FLIPPED; NO IMPLEMENTATION WAS DISPATCHED AND NO SELF-DISPATCHED PATCH CYCLE WAS RUN; TRACKER-ONLY — THIS PLAN IS THE ONLY BYTE WRITTEN; NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE; NO COMMIT, NO BRANCH, NO PR, NO MUTATING GIT COMMAND):**
  0. **THE VERDICT AND ITS SCOPE:** **BLOCKED** — **BLOCKERS: TWO (B-1, B-2)** — EXPLICITLY SCOPED TO THE SLICE `P5-S30-repin` ONLY. THE PACKET'S STRUCTURE IS SOUND (THE ONE CONSOLIDATED GATE PASSES 4 OF 4 — CLAUSE 6) AND THE CONTRACT FAMILIES ARE RE-DERIVED AGAINST THE LIVE BYTES (CLAUSE 1), BUT THE TEXT OF `AC-109-REPIN` AS WRITTEN COLLIDES WITH AT LEAST ELEVEN RATIFIED, LIVE, EXACT-VALUE ASSERTIONS THAT LIVE IN A FILE THE LANE MAY NOT EDIT, AND IT LEAVES THE AI-CHANNEL QUESTION — THE ONE POINT THAT DECIDES THE AI's OWN NUMBERS — UNRESOLVED. THESE ARE **CONTRACT-CONTENT** BLOCKERS, NOT GATE BLOCKERS, AND BOTH ARE CLOSED BY A PLAN-SIDE AMENDMENT (CLAUSE 7) — **NOT** BY A PRODUCTION OR TEST EDIT.
  1. **THE FIVE CONTRACT FAMILIES, RE-DERIVED AGAINST THE LIVE BYTES (EVERY CITATION READ OFF THE LIVE BYTES ON 2026-11-23):** (a) **HYSTERESIS — SOUND, AND IT FITS INSIDE THE ONE FILE:** `computeSlipAngle` (`examples/neatRacing/physics/drift.model.ts:1348`) IS A PURE `atan2(lateralVelocity, speed)`; THE SERVICE RE-DERIVES IT ON EVERY TICK (`examples/neatRacing/physics/physics.step.service.ts:852`) AND PUBLISHES IT (`:904`) FROM THE **POST-STEP** `lateralVelocity` AND `speed`; THE RED HOME RE-DERIVES FROM THE **SAME** PUBLISHED PAIR (`examples/neatRacing/physics/drift.smoothness.test.ts:280-283`), SO A HOLD IS HONESTLY OBSERVABLE AS `state.slipAngle !== computeSlipAngle(state.lateralVelocity, state.speed)` (`:277-292`) AND NOT AS AN ARTEFACT OF A DIVERGENT INPUT PAIR. A DEPTH HYSTERESIS IS IMPLEMENTABLE **STATELESSLY** FROM THE PREVIOUS PUBLISHED SLIP (`state.slipAngle`): NO NEW STATE FIELD, NO `drift.model.ts` EDIT, THE ONE-FILE SCOPE HOLDS. (b) **YAW FEEDBACK — SATISFIABLE IN THE ONE FILE:** THE FEEDBACK MEMORY IS THE **PUBLISHED** `state.yawRate` (`physics.step.service.ts:196-202` — RADIANS PER 60 Hz TICK; `rad/s = yawRate * 60`), AND THE PER-TICK DEMAND IS `demandedTurnRateRadPerS * dt` (`:781`, `:786`, `resolveYawRatePerTick` AT `:1319`); A FIRST-ORDER LAG `nextYaw = prevYaw + (rawYaw - prevYaw) * a` WITH `a IN (0, 1]` GIVES A RELEASE-TICK ADVANCE OF `prevYaw * (1 - a) > 0` THAT IS ALSO `<=` THE PRE-RELEASE ADVANCE — EXACTLY RED-S30-4's TWO FLAGS. (c) **FIXTURE/TICK — HONEST AND NOT OVER-CONSTRAINED:** `DRIFT_LANE_TICKS = 300` (`drift.smoothness.test.ts:130`), `YAW_STEER_PHASE_TICKS = 120` (`:133`), `YAW_STRAIGHT_TICKS = 3` (`:136`), THE ROAD RESOLVER (`:139`), THE DETERMINISTIC `wrapToPi` (`:152`), AND NO `Math.random` / `Date.now` / TIMER / ASYNC (`:65-66`); THE RELEASE TICK IS A **REAL** ZERO-DEMAND TICK (`steer: 0`, NO PERCENT → `resolveDemandedTurnRateRadPerS` RETURNS EXACTLY 0), SO THE CONTRACT IS NOT VACUOUS — THE MEASURED RELEASE ADVANCE IS **EXACTLY 0 rad** AGAINST A PRE-RELEASE ADVANCE OF **0.02554904338790065 rad** (`drift.smoothness.test.ts:60-63`), RED BY ASSERTION TODAY; THE THREE STRAIGHT TICKS ARE **MEASURED BUT NOT ASSERTED** (`:333-337` — THEY ONLY RIDE ALONG IN THE FAILURE MESSAGE), SO THE LANE MAY NOT CLAIM A "THE YAW ONLY DECAYS" CONTRACT FROM THEM; AND THE FIXED TIMESTEP (`NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS = 1 / 60`, `:272`) IS UNCHANGED BY THE CRITERION, SO THE TICK SEMANTICS ARE STABLE. (d) **LEGACY FLOOR — SPLITS CLEANLY IN TWO:** THE **MODEL-SIDE** FLOORS ARE **STRUCTURALLY IMMUNE** — `drift.model.ts` IS **NOT** IN THE LANE'S FILE SET AND THE MODEL IS PURE (IT RECEIVES THE DEMANDED TURN RATE AS A DATA VALUE), SO THE FAR-SIDE POP FLOOR (`drift.model.test.ts:1196`, `:1240`), THE SETTLE WINDOW (`:1207`, `:1208`), THE PER-TICK STEP CEILING (`:1219`, `:1242`), THE ANTI-TEARDOWN `toEqual(POSE_POINT)` GUARD (`:1239`) AND THE DAMPING EQUALITY PIN (`:889`) CANNOT MOVE; THE **SERVICE-SIDE** FLOORS ARE THE COUPLED ONES — THE PUBLISHED-SLIP ENVELOPE (`physics.step.service.test.ts:577-583`: `<= 0.45 + 0.002 = 0.452 rad` AGAINST THE MEASURED 180-TICK CAPTURE **0.45041978773870417 rad**) PLUS THE FIRST-TICK TURN-RATE CORPUS OF CLAUSE 2. (e) **SCOPE — ADEQUATE FOR THE IMPLEMENTATION, INCOMPLETE AS A DISCLOSURE:** BOTH CHANGES LAND IN THE ONE DECLARED FILE (`physics.step.service.ts:852` FOR THE HYSTERESIS, `:781-787` FOR THE YAW LAW), NEITHER NEEDS A NEW STATE FIELD, AND THE AI REACH IS **DATA-ONLY** (`turnRatePercent?: number` AT `:179`; RESOLVED AT `:1277-1289`, WHERE A PRESENT FINITE PERCENT IS AUTHORITATIVE AND RETURNS `(turnRatePercent / 100) * NEATRACING_PHYSICS_STEERING_RATE`) — BUT THE SLICE DISCLOSES NEITHER ITS COUPLED ASSERTIONS NOR A RE-PIN OBLIGATION (CLAUSES 2 AND 7).
  2. **BLOCKER B-1 (SCOPE + LEGACY FLOOR — THE FIRST-TICK TURN-RATE CORPUS IS COUPLED AND UNDISCLOSED):** THE RATIFIED AC-091/AC-092/AC-093 CORPUS MEASURES THE TURN RATE **THROUGH THE FIRST TICK'S PUBLISHED HEADING DELTA** AND PINS IT WITH EXACT OR NEAR-EXACT FLOORS, AND EVERY ONE OF THOSE ASSERTIONS LIVES IN `examples/neatRacing/physics/physics.step.service.test.ts` — **A FILE THE SLICE'S `files_to_change` DOES NOT NAME** (THE SLICE NAMES ONLY `examples/neatRacing/physics/physics.step.service.ts`): (i) `physics.step.service.test.ts:2070-2078` — AC-091 PINNED-UNIT: `expect(demandedTurnRateOf(lane.firstDriftInput)).toBe(MOMENTUM_SERVICE_STEER_AXIS * NEATRACING_PHYSICS_STEERING_RATE * MOMENTUM_SERVICE_SPEED_PERCENT * MOMENTUM_SERVICE_TURN_RATE_SPEED_FACTOR)` — AN EXACT `toBe` WITH **NO EPSILON** ON THE NUMBER THE DRIFT LAW CONSUMES; (ii) `:2049-2052` — AC-091 ONE-RESOLVED-NUMBER: `expect(lane.firstHeadingDeltaRad).toBeCloseTo(demandedTurnRateOf(lane.firstDriftInput) * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS, 12)`; (iii) `:2422-2424` AND `:2447-2449` — AC-092 TOP-SPEED USER RATE `< 2` AND `>= 0.75 * 2 = 1.5 rad/s`; (iv) `:2477-2489` — AC-092 FACTOR MONOTONICITY OVER THE FIVE-SPEED LADDER (FIRST TICK OF EACH LANE); (v) `:2533-2541` — AC-092 CRAWL BAND `[0.9 * crawlNominalLaw - eps, crawlNominalLaw + eps]`; (vi) `:2622-2625`, `:2648-2651`, `:2723-2730`, `:2751-2754` — THE AC-093 AI PERCENT LANES AT **1e-12**; (vii) `examples/neatRacing/physics/ai.autopilot.test.ts:344-352` WITH THE TOLERANCE AT `:205` (`1e-9 rad/s`) — THE AI RATE-ADOPTION RAIL, ALSO A FIRST-TICK MEASUREMENT (`ai.autopilot.test.ts:255-259`). THE TWO MEASUREMENT HELPERS ARE `physics.step.service.test.ts:1957-1961` (`firstHeadingDeltaRad = next.heading - state.heading` ON TICK 0) AND `:2342-2347` (`commandedRateRadPerS = (next.heading - state.heading) / NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS` ON TICK 0). **CONSEQUENCE:** A SMOOTHING OR FEEDBACK THAT TOUCHES THE **FIRST** TICK BREAKS (i)-(vii); A SMOOTHING THAT LEAVES THE FIRST TICK RAW BUT DIVERGES FROM THE DRIFT-CONSUMED NUMBER FROM TICK 1 ONWARD SILENTLY **SUPERSEDES** THE AC-091 "ONE RESOLVED NUMBER" LAW (WHOSE CORPUS ASSERTION COVERS TICK 0 ONLY). AS WRITTEN, `AC-109-REPIN` NEITHER NAMES THESE ASSERTIONS NOR DECLARES A RE-PIN OBLIGATION NOR STATES A FIRST-TICK SEEDING RULE — **WHILE THE HOUSE PRECEDENT (`AC-109-DAMPING-REPIN`) DOES EXACTLY THAT** (IT NAMES THE COUPLED PIN, DECLARES THE RE-PIN OBLIGATION AND WIDENS THE FILE SET). SO THE LANE AS WRITTEN EITHER BREAKS RATIFIED FLOORS IT MAY NOT REPAIR (ITS FILE SET) OR EDITS A TEST FILE OUTSIDE ITS DECLARED SCOPE — WHICH IS A **TEST WEAKENING**, AND THIS VERIFICATION REJECTS IT. EVERY ONE OF THESE FLOORS IS **LIVE AND GREEN TODAY** (`physics.step.service.test.ts` PASSES ENTIRELY — CLAUSE 5).
  3. **BLOCKER B-2 (AI-CHANNEL AUTHORITY — THE CRITERION IS AMBIGUOUS ON THE ONE POINT THAT DECIDES THE AI's OWN NUMBERS):** `AC-109-REPIN` REQUIRES BOTH "THE DEMANDED TURN RATE GAINS A STEERING SMOOTHING AND YAW-RATE FEEDBACK" **AND** "THE AI TURN-RATE AUTHORITY STAY UNCHANGED". IF THE SMOOTHING COVERS THE **AI** CHANNEL, THE AI's COMMANDED-RATE NUMBERS **CHANGE** AND THE 1e-12 AI LANES (CLAUSE 2 (vi)) AND THE 1e-9 AI RATE-ADOPTION RAIL (CLAUSE 2 (vii)) BREAK — WHICH IS EXACTLY THE AI-AUTHORITY CHANGE THE CRITERION FORBIDS; IF THE SMOOTHING COVERS THE **USER** CHANNEL ONLY, EVERY AI NUMBER STAYS **BYTE-IDENTICAL**. THE TWO READINGS ARE MATERIALLY DIFFERENT AND THE CRITERION DOES NOT CHOOSE, SO THE AC MUST SAY WHICH CHANNEL THE SMOOTHING TOUCHES. (THE AI-COUPLING LAW ITSELF IS **GREEN BY CONSTRUCTION** INSIDE THE DECLARED FILE SET: `RED-S30-AI-UNCOUPLED` (`drift.smoothness.test.ts:365-402`) SCANS THE FLAT `examples/neatRacing/physics/` NON-TEST SOURCES FOR `../ai/` IMPORTS AND THE LANE'S ONLY FILE IS ONE OF THEM, SO **NO AI IMPORT IS REACHABLE** FROM THE LANE AND NO AI COUPLING CAN ENTER WITHOUT ONE.)
  4. **CLARIFICATION C-1 (THE CHATTER BAND — THE CRITERION PRESUPPOSES A CONSTANT THAT DOES NOT EXIST, AND ITS WINDOW IS BOUNDED):** `AC-109-REPIN` REFERS TO "THE PUBLISHED CHATTER BAND" AS IF IT EXISTED; **NO SUCH CONSTANT EXISTS IN THE CURRENT BYTES** (THE PLAN ITSELF RECORDS THIS). THE LANE MUST **PUBLISH** ONE INSIDE ITS SINGLE DECLARED FILE — THE LAWFUL HOME IS `examples/neatRacing/physics/physics.step.service.ts` BESIDE `NEATRACING_PHYSICS_STEERING_RATE` (`:348`), `NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S` (`:281`) AND `NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS` (`:272`), **NOT** `drift.model.ts` (WHOSE CONSTANTS ARE THE `NEATRACING_DRIFT_*` FAMILY AND WHICH IS **NOT** IN THE FILE SET). THE WINDOW IS BOUNDED BY TWO MEASURED NUMBERS: **LOWER** — THE RED HOME'S OWN MEASUREMENT (`drift.smoothness.test.ts:54-59`) RECORDS THAT 273 OF THE 300 PER-TICK RE-DERIVATION STEPS MOVE THE SLIP BY **AT MOST 0.0005 rad**, SO A BAND BELOW THAT SIZE PRODUCES **NO** HOLD AND RED-S30-3 STAYS RED (THE HOME'S PHRASE "ANY NON-ZERO CHATTER BAND" IS LOOSE — THE LANE MUST **MEASURE**, NOT ASSUME); **UPPER** — THE PUBLISHED-SLIP ENVELOPE (CLAUSE 1 (d)) LEAVES ROUGHLY **0.00158 rad** OF MARGIN AT THE 180-TICK CAPTURE, SO THE BAND MUST STAY BELOW THAT OR THE LANE MUST MEASURE THAT NO HOLD COINCIDES WITH THE ENVELOPE PROBE. THE AC MUST NAME THE HOME AND THE MEASURED WINDOW, OR STATE THAT THE LANE MEASURES AND RECORDS THE SMALLEST BAND THAT PRODUCES HOLDS AND KEEPS THE ENVELOPE GREEN. (NO `NEEDS CLARIFICATION` MARKER IS CREATED — THE PLAN-WIDE BUDGET STANDS AT 3 OF 3.)
  5. **CLARIFICATION C-2 (VALIDATION COMPLETENESS — THE AI PATH IS COUPLED AND OUT OF THE ALLOW-LISTED VALIDATION):** THE STEP SERVICE IS ALSO DRIVEN BY `examples/neatRacing/ai/scripted.controller.test.ts:352-354` (`expect(suppressed.heading).toBe(proofCar.heading)` AND `expect(suppressed.yawRate).toBe(0)` — THE AI's ZERO-PERCENT TURN-RATE AUTHORITY, MEASURED THROUGH THE LIVE TICK, WITH THE PROOF CAR'S `yawRate: 0` AT `:317`). THAT FILE IS **GREEN TODAY**, BUT THE `examples/neatRacing/ai` PATH IS **NOT** IN THE SLICE'S ALLOW-LISTED VALIDATION (WHICH COVERS ONLY THE `physics` / `engine` / `browser-entry` PATHS), SO THE LANE CANNOT PROVE IT STAYS GREEN. NOTE THAT THE WHOLE `ai` PATH IS **NOT** A CLEAN BASELINE TODAY (TWO PRE-EXISTING FAILURES IN `examples/neatRacing/ai/rival.propagation.test.ts:1161` — THE STALE LANE THE PLAN ALREADY RECORDS), SO THE CLOSURE MUST NAME THE **FILE**, NOT THE PATH: ADD `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai/scripted.controller"` TO THE SLICE'S VALIDATION LIST. (UNDER A FEEDBACK LAW THAT READS THE **PREVIOUS** PUBLISHED YAW RATE, THAT ASSERTION SURVIVES: THE PROOF CAR's YAW RATE IS 0, SO A ZERO-PERCENT DEMAND KEEPS THE YAW AT EXACTLY 0.)
  6. **THE MEASURED BASELINE AND THE ONE CONSOLIDATED GATE (RUN BY THIS INSTANCE, 2026-11-23):** (a) `npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"` → **EXIT 1, 2 failed / 254 passed / 256**; THE ONLY TWO FAILURES ARE `RED-S30-3` (`drift.smoothness.test.ts:264`) AND `RED-S30-4` (`:307`) — **RED BY ASSERTION, NOT BY AN IMPORT FAILURE** — AND `physics.step.service.test.ts` **PASSES ENTIRELY**, SO EVERY FLOOR NAMED IN CLAUSE 2 IS A LIVE, GREEN FLOOR; (b) `npm run jest:base -- --testPathPatterns="examples/neatRacing/physics/drift.smoothness"` → **EXIT 1, 2 failed / 3 passed**; THE MEASURED NUMBERS ARE `heldTicks = 0` (`holdsPublishedSlip: false`), `releaseAdvanceRad = 0`, `preReleaseAdvanceRad = 0.02554904338790065`, `firstStraightAdvanceRad = 0`; (c) `npm run jest:base -- --testPathPatterns="examples/neatRacing/ai"` → **EXIT 1, 2 failed / 283 passed / 285**, THE FAILURES BEING THE UNRELATED PRE-EXISTING `examples/neatRacing/ai/rival.propagation.test.ts:1161` LANE, WHILE `ai/scripted.controller.test.ts` **PASSES**; (d) THE **ONE** CONSOLIDATED `slice-advancement` GATE (`--slice-id P5-S30-repin --changed-files plans/neatracing-voxel-racer.plans.md`) → **PASS, 4 OF 4 SUB-GATES**: `plan-sync` PASS, `step-packet` PASS, `plan-slice-quality` PASS, `plan-command-lint` PASS, WITH `failedGates: []`, `erroredGates: []`, `timedOutGates: []` — THE PACKET'S STRUCTURE IS SOUND; (e) THE CORTEX INDEX WAS **FRESH** AT VERIFICATION TIME (`fresh: true`, `last_update_source: filesystem`).
  7. **THE EXACT, MINIMAL, PLAN-SIDE AMENDMENT THAT WOULD UNBLOCK `P5-S30-repin` (SIX EDITS TO `AC-109-REPIN` — NO PRODUCTION AND NO TEST BYTE IS NEEDED FOR ANY OF THEM):** (1) **STATE THE FIRST-TICK SEEDING RULE:** "THE **FIRST** TICK OF A LANE IS **UNSMOOTHED** — THE HEADING ADVANCE IS THE RAW DEMAND TIMES THE FIXED `dt` — SO THE RATIFIED AC-091/AC-092/AC-093 FIRST-TICK FLOORS STAY BYTE-IDENTICAL; THE SMOOTHING AND THE FEEDBACK ENGAGE FROM THE **SECOND** TICK ONWARD, AS A FIRST-ORDER LAG ON THE **PUBLISHED** `state.yawRate` WITH `a IN (0, 1]`"; (2) **STATE THE CHANNEL:** "THE SMOOTHING AND THE FEEDBACK APPLY TO THE **USER** CHANNEL **ONLY** — THE AI PERCENT CHANNEL RESOLVES THE DEMANDED TURN RATE **UNCHANGED** AND ADVANCES THE HEADING **UNCHANGED**, SO THE AI TURN-RATE AUTHORITY AND EVERY AI NUMBER STAY BYTE-IDENTICAL"; (3) **STATE THE DRIFT-LAW INPUT AND RECORD THE ONE SUPERSESSION:** "THE DRIFT LAW KEEPS CONSUMING THE **RAW** DEMANDED TURN RATE (WHICH PRESERVES THE AC-091 PINNED-UNIT EXACTNESS AT `physics.step.service.test.ts:2070-2078` VERBATIM) — THE HEADING ADVANCE IS THE **ONLY** QUANTITY FED BACK FROM THE MEASURED YAW RATE, AND THE AC-091 "ONE RESOLVED NUMBER" LAW IS THEREFORE **EXPLICITLY SUPERSEDED FOR THE HEADING ADVANCE ONLY** (THE DEMAND AND THE DRIFT LAW STILL READ ONE NUMBER)"; (4) **NAME THE COUPLED ASSERTIONS AND THE RE-PIN POSTURE:** LIST CLAUSE 2 (i)-(vii) VERBATIM AND STATE THAT **NONE** OF THEM MAY CHANGE (OR, IF A RE-PIN EVER PROVES UNAVOIDABLE, DISCLOSE THE RE-PIN OBLIGATION AND WIDEN `files_to_change` TO INCLUDE `examples/neatRacing/physics/physics.step.service.test.ts`, EXACTLY AS `AC-109-DAMPING-REPIN` DID); (5) **NAME THE CHATTER BAND:** ITS HOME (`examples/neatRacing/physics/physics.step.service.ts`, AN EXPORTED `NEATRACING_PHYSICS_*` CONSTANT) AND THE MEASURED WINDOW (CLAUSE 4: AT LEAST ~0.0005 rad, AT MOST ~0.00158 rad), WITH THE LANE MEASURING AND RECORDING THE SMALLEST BAND THAT PRODUCES HOLDS AND KEEPS THE PUBLISHED-SLIP ENVELOPE GREEN; (6) **ADD THE AI FILE TO THE VALIDATION LIST** (`npm run jest:base -- --testPathPatterns="examples/neatRacing/ai/scripted.controller"`).
  8. **THE NEXT AGENT AND THIS INSTANCE'S BOUNDARIES:** THE NEXT AGENT IS THE `01-planning` **AUTHOR** INSTANCE, WHICH APPLIES THE CLAUSE-7 AMENDMENT TO THE STEP-30 PACKET AND THE `P5-S30-repin` SLICE **IN THE PLAN ONLY** — **NOT** `04-implementing`, BECAUSE THE LANE IS **NOT** EXECUTABLE UNTIL B-1 AND B-2 ARE CLOSED; AFTER THE AMENDMENT, A **FRESH** `01-planning` VERIFICATION INSTANCE RE-RUNS THE ONE CONSOLIDATED `slice-advancement` GATE AND RECORDS THE VERDICT. THIS INSTANCE RAN **NO** IMPLEMENTATION, WROTE **NO** PRODUCTION / TEST / SPEC / BUNDLE / GENERATED-DOC BYTE, FLIPPED **NO** STATUS TOKEN, RAN **NO** SELF-DISPATCHED PATCH CYCLE, AND RAN **NO** MUTATING GIT COMMAND.

- **VALIDATION_EVIDENCE — THE SLICE-SCOPED GREEN VALIDATION OF `P5-S30-damping` (2026-11-22, THE `05-green-testing` (Tier 1) INSTANCE — THE VERDICT IS `GREEN: OK`, EXPLICITLY SCOPED TO THE SLICE `P5-S30-damping` ONLY; THE SLICE TOKEN IS NOW `[DONE 2026-11-22 - 05-green-testing GREEN: OK]`, THE ONE `[WIP]` SLICE IS NOW `P5-S30-repin` AND THE STEP-30 `pre_execute_hook` IS RE-BOUND TO IT; TRACKER-ONLY — THIS PLAN PLUS `plans/README.md` PLUS `plans/Roadmap.md` ARE THE ONLY BYTES WRITTEN; NO PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE; NO IMPLEMENTATION DISPATCHED; NO MUTATING GIT COMMAND):**
  0. **THE VERDICT AND THE SLICE-LEVEL GATE OBJECT:** `GREEN: OK` — **BLOCKERS: NONE** — EXPLICITLY SCOPED TO THE SLICE `P5-S30-damping` ONLY. THE SLICE-LEVEL GATE CONTRACT OBJECT IS `{ "pass": true, "slice_id": "P5-S30-damping", "evidence": { "coverage_summary": { "statements": 98.58, "branches": 96.77, "functions": 100, "lines": 98.58 }, "test_results": "examples/neatRacing/physics/drift.model.test.ts -> 64/64 PASS; examples/neatRacing/physics/drift.smoothness.test.ts -> RED-S30-1 PASS + RED-S30-2 PASS; examples/neatRacing/browser-entry -> 117/117 PASS" }, "fixHint": null, "owner": "05-green-testing" }`.
  1. **THE TWO OWNED RED CONTRACTS ARE GREEN BY NAME:** `RED-S30-1` (THE DAMPING RATIO) AT `examples/neatRacing/physics/drift.smoothness.test.ts:187` **PASSES** AND `RED-S30-2` (THE COUNTERSTEER CONTINUITY) AT `:210` **PASSES**. THE LATER-SLICE REDS ARE **EXPECTED, NOT REGRESSIONS**: `RED-S30-3` (`:264`) AND `RED-S30-4` (`:307`) REMAIN RED **BY DESIGN** (OWNED BY `P5-S30-repin`) AND `RED-S30-5` (`examples/neatRacing/engine/passes/car.rear-slip.test.ts:444`) REMAINS RED **BY DESIGN** (OWNED BY `P5-S30-slip`). NEITHER CLOSED RED HOME WAS RE-AUTHORED OR WEAKENED.
  2. **THE EXACT LEGACY PROTECTIONS (CLAUSES (1), (2), (2a), (2b), (4), (4a), (5), (6)) HOLD:** (a) EVERY COUPLED CORPUS FLOOR IS **BYTE-IDENTICAL** — THE POP FLOOR, THE SETTLE-WINDOW ASSERTIONS, THE PER-TICK STEP CEILING AND THE ANTI-TEARDOWN `toEqual(POSE_POINT)` ASSERTION; (b) THE **ONLY** ASSERTION CHANGED IS THE DAMPING PIN, RE-PINNED BY **EXACT EQUALITY** TO 43.43 AT `examples/neatRacing/physics/drift.model.test.ts:889`; (c) `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER` IS STILL EXPORTED AT **0.9** (`drift.model.ts:406` — NOW NAMED THE **KNEE**), `NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2` IS STILL **70** (`:385` — NOW NAMED THE **FULL-SCALE REFERENCE**), THE STIFFNESS IS STILL **324** (`:477`) AND THE VISUAL YAW GAIN IS STILL **0.45** (`:547`); (d) THE **ONLY** PRODUCTION CONSTANT CHANGED IS `NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S` (12.6 → **43.43**, `:497`) — `MIN_SLIP_RAD` (0.02), `SLIP_RESPONSE_TAU_S` (0.045), `MAX_SLIP_ANGLE_RAD` (0.45) AND `ARTICULATION_SLEW` (20) ARE UNCHANGED; (e) THE RETIRED-NAME CONTRACT AND THE `not.toBe(1)` HOLD-TICKS ASSERTION (`:1746`) ARE UNCHANGED AND `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS` HAS **ZERO** PRODUCTION OCCURRENCES; (f) THE RAMP REPLACES **ONLY** THE `|steer|` FACTOR OF `turnSign * 70 * ramp(|steer|) * speedPercent` — THE `speedPercent` SCALING, THE LATCHED `turnSign`, THE `sign(steer)` GATE AND THE `|slip| > MIN_SLIP_RAD` GATE ARE **PRESERVED VERBATIM** AND THE HARD `|steer| >= 0.9` GATE IS **REMOVED** BECAUSE THE KNEE **IS** THE GUARD NOW; (g) THE MODEL STAYS **PURE** AND THE POSE SPRING STAYS **UNCLAMPED**; (h) `examples/neatRacing/physics/physics.step.service.test.ts` **PASSES** (THE AC-098 GUARD AT `:4841` IS GREEN) AND IS BYTE-IDENTICAL, `examples/neatRacing/engine/passes/car.pass.test.ts` **PASSES**, `examples/neatRacing/ai/oscillation-penalty.utils.ts` IS UNMODIFIED AND THE **EIGHT-FIELD** BROWSER-DEBUG SEAM IS INTACT (`examples/neatRacing/browser-entry` — **117 / 117 PASS**).
  3. **THE FIVE MEASURED MARGINS REPRODUCE EXACTLY (CLAUSE (3), THE FIVE-BOUND CONJUNCTION):** TRAVEL VISIBILITY **7.9881910941984415 DEG** (FLOOR 2.0, **+5.9881910941984415**); TRUE FAR-SIDE OVERSHOOT **EXACTLY 0.0000 DEG** (FLOAT-EXACT — `Object.is(0, 0)` IS TRUE); SETTLE **31 TICKS** (WINDOW [31, 33] — **+1** OVER THE 30-TICK FLOOR, **3** UNDER THE 34-TICK CEILING); MAXIMUM PER-TICK STEP **2.0435494692999328 DEG** (CEILING 2.2, **0.1564505307000672** OF HEADROOM); ZETA **1.206388888888889** (FLOOR 0.7, **+0.506388888888889**). 43.43 IS THE SMALLEST VALUE ON THE DECLARED 0.01 RAD/S GRID INSIDE THE LEGAL BAND [43.426422, ~48.83]; 43.42 MEASURES SETTLE 30 AND IS **REJECTED, NEVER RELAXED**. THE OVERSHOOT WITNESS IS **ADDITIVE** (`drift.model.test.ts:1247-1271`, `toBe(0)` PLUS `Object.is(..., 0)`) AND MEASURES THE TRUE FAR-SIDE OVERSHOOT OVER THE COUNTER PHASE ONLY (`:574`), WITH `DriftScenarioRun.overshootDeg` ADDED AT `:392`.
  4. **THE COUNTERSTEER RAMP SWEEP REPRODUCES (CLAUSE (4a)):** THE PINNED FORM IS `ramp(s) = clamp((s - 0.9) / (1 - 0.9), 0, 1)` WITH `s = |steer|`; OVER A 1001-SAMPLE SWEEP AT `speedPercent` = 1 THE KNEE VALUE IS **EXACTLY 0**, EVERY BELOW-KNEE SAMPLE IS **EXACTLY 0**, THE FULL-LOCK VALUE IS **EXACTLY 1**, THE RAMP IS **MONOTONE NON-DECREASING**, IT IS **CONTINUOUS** AT THE KNEE (THE ADJACENT JUMP THERE IS **EXACTLY 0**) AND THE MAXIMUM ADJACENT JUMP IS **0.70000000000001** AGAINST THE RED-S30-2 LIMIT OF **7** — ROUGHLY **10x** HEADROOM. THE OLD **63 M/S^2** STEP ACROSS THE KNEE IS **GONE**.
  5. **THE ALLOW-LISTED VALIDATION LEDGER (TARGETED TESTS ONLY — NEVER A BROAD SUITE):** (1) `npm run jest:base -- --testPathPatterns="examples/neatRacing/physics"` → **EXIT 1** — **2 FAILED / 254 PASSED / 256**, AND THE ONLY TWO FAILURES ARE **RED-S30-3** AND **RED-S30-4** (OWNED BY `P5-S30-repin`, RED BY DESIGN); (2) `npm run jest:base -- --testPathPatterns="examples/neatRacing/engine"` → **EXIT 1** — **1 FAILED / 266 PASSED / 267**, AND THE ONLY FAILURE IS **RED-S30-5** (OWNED BY `P5-S30-slip`, RED BY DESIGN); (3) `npm run jest:base -- --testPathPatterns="examples/neatRacing/browser-entry"` → **EXIT 0** — **5 SUITES, 117 / 117**; (4) `npx tsc -p tsconfig.neatRacing.json --noEmit` → **EXIT 0**, CLEAN; (5) `npm run build:neatRacing` → **EXIT 0**; (6) `npm run lint` → **EXIT 0** — **0 ERRORS, 7 WARNINGS**, OF WHICH THE TWO `drift.model.test.ts` ONES (`:203`, `:214`, `no-explicit-any`) ARE **PRE-EXISTING** (THE FILE'S DIFF HUNKS START AT `:124`); (7) `node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md` → **`PASS plan sync: 0 errors, 0 warnings`**, **EXIT 0**; (8) THE CONSOLIDATED `slice-advancement` GATE — SEE ITEM 7.
  6. **THE BROWSER OBLIGATION IS N/A FOR THIS SLICE (NO CHROME WAS LAUNCHED, NO FOCUS WAS FORCED, NO TOPMOST WAS SET AND NO FPS WAS MEASURED):** CLAUSE (6) FORBIDS THIS LANE FROM TOUCHING ANY BROWSER BYTE AND CLAUSE (7) HANDS THE BROWSER CROSSING RE-BASE TO THE **STEP-30A BOUNDARY LANE `P5-S30A-crossing-rebase`**; THE **VISIBLE-BACKGROUND BROWSER PROOF** IS OWNED BY THE **WHOLE-STEP LANE `P5-S30-green`** (AC-110-GREEN). THE SLICE OWNS **NO** BROWSER BEHAVIOUR, SO NO BROWSER RUN WAS REQUIRED AND NONE WAS PERFORMED. THE THREE LIVE TP-17 FACTS (`meanForwardSpeed`, `oscillationMetric`, `offTrackFraction`) HAVE **NO HEADLESS SEAM** (`goalState` IS CLOSURE-LOCAL INSIDE `startLiveDriveLoop`) AND MUST BE MEASURED BY THE REBASE LANE THAT OWNS THE BROWSER BYTES; THE PLAN'S RECORDED CROSSING BASELINE (`scripted_baseline_lap_pace 63.404817826632915`, `lap_ticks 1122`, `completed_lap true`, BIT-IDENTICAL ACROSS THREE VARIANTS BECAUSE THE COUNTERSTEER AUTHORITY IS NEVER NON-ZERO IN THAT LANE — 1046 CALLS, 42 PASSING THE GATES, 0 NON-ZERO, THE LARGEST GATED `|steer|` **0.5483766620966092** < THE 0.9 KNEE) IS CARRIED FORWARD UNCHANGED.
  7. **THE GATES (THE FOUR-FIELD GATE CONTRACT, ONE ENTRY PER GATE, NO `gate_error` RAISED ANYWHERE):** (a) `slice-advancement` — `{ "gate": "slice-advancement", "pass": true, "evidence": "neataptic-gate-mcp:run_gate_check gate=slice-advancement --slice-id=P5-S30-damping --changed-files=examples/neatRacing/physics/drift.model.ts,examples/neatRacing/physics/drift.model.test.ts,plans/neatracing-voxel-racer.plans.md --plan=plans/neatracing-voxel-racer.plans.md -> pass:true, severity:FULL, gateCount:7, gatesRun:[plan-sync, step-packet, plan-slice-quality, plan-command-lint, shared-validation, code-coverage, specialist-review], ALL SEVEN pass:true, failedGates:[], erroredGates:[], timedOutGates:[]", "fixHint": "All 7 gates passed for slice P5-S30-damping (FULL).", "owner": "orchestrator (Agent Zero)" }`; (b) `plan-sync` — `{ "gate": "plan-sync", "pass": true, "evidence": "node scripts/agent-customization/validate-plan-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md -> PASS plan sync: 0 errors, 0 warnings (EXIT 0)", "fixHint": "n/a", "owner": "scripts/agent-customization/validate-plan-sync.mjs" }`; (c) `cortex-index` — `{ "gate": "cortex-index", "pass": true, "evidence": "rag-index/validate-index.mjs -> Semantic index valid: 1857 documents, 22604 chunks (EXIT 0) and the plan-family refresh returned {\"syncFresh\":true,\"planFresh\":true,\"stale\":false}", "fixHint": "n/a", "owner": "rag-index" }`. **THE `code-coverage` GATE IS N/A AS A CONTENT CLAIM FOR THIS SLICE**: `jest.config.mjs` PUTS `/examples/` INSIDE `coveragePathIgnorePatterns`, SO `examples/` IS OUTSIDE THE TRACKED COVERAGE SURFACE — THE REAL NUMBERS WERE MEASURED WITH `--coveragePathIgnorePatterns="/node_modules/"` AND ARE RECORDED IN ITEM 0 AND ITEM 8.
  8. **THE COVERAGE NUMBERS FOR THE TOUCHED PRODUCTION FILE, MEASURED IN ISOLATION:** `examples/neatRacing/physics/drift.model.ts` — STATEMENTS **98.58%** (139/141), BRANCHES **96.77%** (60/62), FUNCTIONS **100%** (17/17), LINES **98.58%** (139/141). THE TWO UNCOVERED LINES ARE **875** AND **898** — THE IF-TRUE ARMS OF TWO **PRE-EXISTING** DEFENSIVE GUARDS (`computeDriftSlipResponseTauS` `:874-875` AND `computeDriftReArmReleaseRateRadPerS` `:897-898`), WHICH HAVE **NO** SLICE HUNK AT 873-899 (`git log -S` DATES BOTH TO `4c874f3f`) AND ARE UNREACHABLE FROM THE PRODUCTION CALL SITES BUT LEGALLY REACHABLE THROUGH THE EXPORTED README-PUBLISHED API — SO THEY ARE **DEFENSIVE GUARDS BY DESIGN, NOT DEAD CODE, AND MUST NOT BE DELETED IN THIS SLICE**. **THE SLICE'S OWN NEW CODE (`:379-406`, `:480-497`, `:1092-1141`, `:1157-1175`) IS 100 % COVERED, AND BOTH ARMS OF ALL THREE SURVIVING GATES ARE COVERED.**
  9. **THE TIER-3 DELEGATIONS (TWO, BOTH READ-ONLY, NEITHER WROTE A FILE):** (a) `slice-validator` — **PASS, NO BLOCKERS** — CONFIRMED THE FILE SET, THE EXACT-EQUALITY RE-PIN AT `:889`, THE BYTE-IDENTICAL CORPUS FLOORS, THE ADDITIVE OVERSHOOT WITNESS, THE PUBLISHED SEAMS (43.43 / 324 / 0.9 / 70), THE RAMP FORM `clamp((s-0.9)/(1-0.9),0,1)`, THE **ZERO** `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS` PRODUCTION OCCURRENCES, THE PURE MODEL AND THE UNCLAMPED SPRING, THAT NEITHER RED HOME WAS RE-AUTHORED (ATTRIBUTED BY MTIME ORDERING), AND THE PACKET COMPLETENESS (`step-packet` PASS, `plan-slice-quality` PASS, `cortex-index` PASS); (b) `coverage-analyst` — **NON-BLOCKING** — CLASSIFIED THE TWO UNCOVERED LINES AS THE DEFENSIVE-GUARD ARMS DESCRIBED IN ITEM 8 AND RAN THE `cortex-index` GATE (PASS, INDEX FRESH, 1857 DOCS).
  10. **THE NON-BLOCKING OBSERVATIONS (DISCLOSED, NOT PATCHED):** (a) THE STALE `zeta = 0.35` TEXT SURVIVES IN THE **GENERATED** `examples/neatRacing/physics/README.md:812` AND IN THE **OUT-OF-SET** `examples/neatRacing/engine/passes/car.pass.ts:501` — CLAUSE (4a) FORBIDS PATCHING EITHER, SO BOTH ARE RECORDED AS OUT-OF-SCOPE OBSERVATIONS (THE README MUST BE REGENERATED, NEVER HAND-EDITED); (b) THE LEDGER'S PRINTED MAX-STEP HEADROOM **0.1564505307000672** IS **6 ULP** FROM THE EXACT **0.15645053070006743** — A PRINTING DELTA THAT CHANGES NO BOUND, NO MARGIN SIGN AND NO VERDICT; (c) THE PLAN RECORDS THE **ANALYTIC NOMINAL STEP** `0.7000000000000001` AT LINES 84 / 1558 / 2742 AND THE **MEASURED SWEEP MAXIMUM** `0.70000000000001` AT LINE 2812 — BOTH ARE ALREADY IN THE PLAN AND BOTH ARE ROUGHLY **10x** UNDER THE LIMIT OF 7, SO THE BOUND HOLDS UNDER EITHER FIGURE; (d) **34 TRACKED FILES ARE DIRTY IN THE WORKTREE FROM SIBLING LANES**, SO WORKTREE-LEVEL BYTE-IDENTITY IS UNATTRIBUTABLE — THE PROTECTIONS WERE VERIFIED **PER FILE** AND PER HUNK INSTEAD; (e) **ONE SIBLING-LANE MARKDOWN DEFECT WAS REPAIRED, TRACKER-ONLY AND FULLY DISCLOSED:** THE `## PlanUpdate` YAML BLOCK OF THE `P5-S29-neon` IMPLEMENTATION LANE (THE ` ```yaml ` FENCE AT LINE 2318) WAS **NEVER CLOSED**, SO THE SIBLING `## PlanUpdate — THE P5-S29-neon NEON-CLOSURE PASS` HEADING AND ITS PROSE WERE SWALLOWED INTO A CODE BLOCK; THIS LANE INSERTED **ONE** MISSING CLOSING FENCE AT LINE 2401 AND CHANGED **NO** CLAIM TEXT, **NO** STATUS TOKEN, **NO** EVIDENCE AND **NO** SIBLING LANE CLAIM — THE FENCE COUNT IS NOW **28** (EVEN) AND THE PLAN RENDERS CORRECTLY; THE REPAIR IS **OUT OF THIS SLICE'S SCOPE** AND IS DISCLOSED HERE FOR THE ORCHESTRATOR.
  11. **THE STATUS-TOKEN MOVES (EXECUTED UNDER THE USER'S EXPLICIT "CLOSE/PROMOTE ONLY IF OK" DIRECTIVE, WITH THE OWNERSHIP DISCLOSED):** `P5-S30-damping` IS NOW `[DONE 2026-11-22 - 05-green-testing GREEN: OK: AC-109-DAMPING + AC-109-DAMPING-REPIN - THE TWO OWNED RED CONTRACTS RED-S30-1 AND RED-S30-2 ARE GREEN BY NAME, THE COUPLED CORPUS FLOORS ARE BYTE-IDENTICAL, THE FIVE MEASURED MARGINS REPRODUCE EXACTLY, THE COUNTERSTEER RAMP SWEEP IS MONOTONE WITH THE KNEE EXACTLY 0 AND FULL LOCK EXACTLY 1, THE BROWSER OBLIGATION IS N/A FOR THIS SLICE]`; `P5-S30-repin` IS PROMOTED FROM `[PLANNED]` TO `[WIP]` AND IS NOW THE **ONE** `[WIP]` SLICE; THE STEP-30 `pre_execute_hook.args.slice_id` IS RE-BOUND FROM `P5-S30-damping` TO `P5-S30-repin`; THE STEP-30 `wave_note` IS AMENDED WITH A DATED `05-green-testing` CLAUSE; AND `plans/README.md` PLUS `plans/Roadmap.md` ARE UPDATED TO NAME THE NEW ONE `[WIP]` SLICE. **THE DISCLOSURE:** THE STEP-30 WAVE NOTE RESERVES THE REMAINING **WHOLE-STEP CLOSE** AND THE **VISIBLE-BACKGROUND BROWSER PROOF** TO THE `P5-S30-green` LANE (THE STEP-28 / STEP-29 PRECEDENT) — THOSE OBLIGATIONS ARE **NOT** DISCHARGED BY THIS ROW; `P5-S30-repin`, `P5-S30-slip`, `P5-S30-green` AND `P5-S30A-crossing-rebase` ALL REMAIN OPEN; AND STEP 30, STEP 30A, PHASE 5 AND THE PLAN `**Status:**` ALL REMAIN `[WIP]`.
  12. **WHAT THIS LANE DID NOT DO:** IT DID NOT EDIT ANY PRODUCTION, TEST, SPEC, BUNDLE OR GENERATED-DOC BYTE; IT DID NOT RE-AUTHOR OR WEAKEN ANY CLOSED RED HOME; IT DID NOT RUN A BROAD SUITE, A `coverage` RUN OR A REPO-WIDE JEST COMMAND; IT DID NOT LAUNCH CHROME, FORCE FOCUS OR SET TOPMOST, AND IT DID NOT MEASURE FPS; IT DID NOT DISPATCH `04-implementing` OR ANY OTHER IMPLEMENTING LANE; IT DID NOT RUN A MUTATING GIT COMMAND; AND IT DID NOT CLOSE STEP 30, STEP 30A, PHASE 5 OR THE PLAN.
- **[ARCHIVED 2026-11-23 to `plans/neatracing-voxel-racer.logs.md` (evidence batch 188, sub-block D) — the OLDER `## Latest validation evidence` ROWS (the Phase-1 .. Phase-4 AND THE Step-27 .. Step-29 EVIDENCE TRANSCRIPTS) ARE ARCHIVED BYTE-EXACT. THE TWO ROWS ABOVE ARE THE OPERATIVE ONES: THE 2026-11-23 `P5-S30-repin` INDEPENDENT VERIFICATION (BLOCKED) AND THE 2026-11-22 `P5-S30-damping` GREEN VALIDATION.]**

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history. Load context via Cortex MCP and the declared pre_execute_hook when an execution lane is authorized.

ACTIVE USER PRIORITY — COUNTDOWN FIRST: the next racer implementation must repair the countdown before any Step-30 drift work. This tracker is [WIP], not closed and not archived. The OpenTelemetry pause is superseded for resume ordering by this newer countdown directive.

Context: neatracing-voxel-racer — Phases 1-4 are [DONE]. Phase 5 is [WIP]: Steps 27, 28 and 29 are [DONE]; Step 27A is the ONLY [WIP] step and P5-S27A-red its ONLY active slice; Steps 30, 30A, 31 and 32 are [PLANNED]. P5-S30-repin is [PLANNED] and its prior independent verdict remains BLOCKED (B-1 scope/legacy-floor, B-2 AI-channel authority). Completed packets, evidence and old handoffs are archived at plans/neatracing-voxel-racer.logs.md (evidence batches 180-188).

Required countdown outcome (AC-106-R1..R3): preserve the pure raceState.elapsedS 3-2-1-GO sequence and pre-GO human throttle HOLD, but present an Astrobird-like transparent text/glow overlay over the visible race scene. No black, opaque, or full-screen dimming backdrop during 3/2/1, GO, dismissal, or afterward. Dismissal leaves no overlay/backdrop or input-capturing remnant. At GO the human gate releases, post-GO throttle reaches the existing worker handoff, and simulation time/frame/tick progression observably resumes. A permanently stopped simulation or any persistent black backdrop is a failure.

Bounded repair surface: RED homes examples/neatRacing/host/hud.countdown.test.ts and examples/neatRacing/browser-entry/browser-entry.test.ts; implementation limit examples/neatRacing/host/hud.countdown.ts + examples/neatRacing/host/hud.ts + examples/neatRacing/browser-entry/browser-entry.ts. Inspect live bytes in the execution lane; this planning pass did not inspect or modify racer code/tests. Make the smallest repair, leave browser-entry.ts unchanged if its release/progression path is already correct, and return to planning rather than widening into physics, workers, render passes, stylesheets, or dependencies.

THE ORDERED NEXT BOUNDARY:
1. A FRESH 01-planning VERIFICATION instance validates Step 27A and records green-light: true explicitly scoped to P5-S27A-red with BLOCKERS NONE. The author pass did not self-grant a green light.
2. 03-red-testing runs P5-S27A-red: author named transparent-overlay, clean-dismissal, and non-vacuous pre-GO-hold/post-GO-release-and-progression assertions; record the RED evidence before production edits.
3. 04-implementing runs P5-S27A-overlay. This is the FIRST production implementation and must turn those contracts green without weakening the original AC-106 behavior.
4. 05-green-testing runs P5-S27A-green and closes Step 27A only after focused/step validation and visual evidence pass.
5. Only after Step 27A is DONE, resume the preserved tracker-only six-edit Step-30 clause-7 amendment, then a FRESH P5-S30-repin verification. Its implementation requires a new slice-scoped green light; the old closed-slice green lights do not authorize it.

Chrome/browser rule: MINIMIZE Chrome MCP and browser automation. No browser during planning or implementation; ideally use exactly ONE final real visible-browser validation in P5-S27A-green after focused checks are green. Cover 3/2/1, GO, dismissal, and post-GO motion. A visible background window is valid. Never force foreground, focus, always-on-top, or z-order. The only standing exception is a rare explicitly FPS-scoped test authorized for that purpose; it does NOT apply to Step 27A, which has no FPS assertion. If visual validation is unavailable, record a blocker rather than forcing visibility or claiming success.

Required future gates: the ONE consolidated slice-advancement gate, never decomposed; plan-sync validation where required; cortex-index freshness after tracker edits. The countdown-first author pass intentionally ran no racer tests/builds/browser, no plan/slice gates, no get_slice_context and no subagents; only the user-permitted RAG build-index and cortex-index commands run once after edits.

Known worktree cautions: do not edit the stale duplicate examples/neatRacing/assets/neatracing.bundle.js directly. The AI-lane live baseline re-pin (283/285), coverage gap g1, W-1 checkPlanGreenLight scoping defect, MCP validation-allowlist degradation, and standing 00-helping Cortex rebuild ownership remain carried and unchanged. NEEDS-CLARIFICATION budget remains 3 of 3; this amendment adds no marker.

Archive target: plans/completed/ (plan + logs pair only after the future exit check and Cortex rebuild).
```
