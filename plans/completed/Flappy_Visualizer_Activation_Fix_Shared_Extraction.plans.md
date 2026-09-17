# Flappy Visualizer Activation Fix & Shared Network Visualization

**Status:** [DONE]

## Scope

Fix the Astro Bird demo regression where the network visualizer shows `0.00`
on every node, and complete the shared-visualization intent: make the rich
network visualizer (activation labels, input groups, legend, tooltips, LOD)
reusable from `examples/shared/` with minimal settings by any example.

Three phases:

1. **Phase 1 — Live champion activations (bug fix).** Stream the frame
   winner's real per-node activations from the evolution worker across the
   playback protocol and overlay them onto the host's visualization network,
   with activation-aware cache invalidation so labels update live during
   playback.
2. **Phase 2 — Shared network visualization extraction.** Physically move
   `examples/flappy_bird/browser-entry/network-view/` and
   `examples/flappy_bird/browser-entry/visualization/` (16 source files +
   tests, READMEs, docs.order.json) to
   `examples/shared/network-visualization/` as ONE domain, relocate the
   three visualizer-owned dependencies the folders import from flappy
   (the visualization types, `clamp`, and the 110 visualizer geometry
   constants) into the domain root so it is self-contained, introduce the
   minimal-settings seam (flappy-branded theme values become settings),
   repoint every consumer (flappy, racing_curriculum, asciiMaze, neatChat
   path strings), delete the two dead legacy facades, add the four missing
   sibling tests, and rebuild the generated bundles.
3. **Phase 3 — racing_curriculum visualizer migration.** Audit the racing
   module's working visualizer (`examples/racing_curriculum/browser-entry/network-view/*`
   with LOD + `host.network-tooltip.service`), absorb its distinguishing
   features (LOD renderer for dense networks, tooltip resolver, and racing
   semantic settings) into `examples/shared/network-visualization/`, repoint
   the racing adapter to the enhanced shared domain, delete redundant
   racing-specific code, and keep all racing suites green.

## Root cause record

### Defect D1 — never-activated main-thread networks

The flappy main thread builds visualization networks via
`Network.fromJSON(...)` (`runtime.evolution-loop.service.ts` ~line 182 for the
best network, ~line 491 for the generation population). `fromJSON`
(`src/architecture/network/serialize/network.serialize.json.utils.ts`, lines 267/271/368/374)
restores node type, bias, squash, and gene-id — but NOT `activation` values.
Nothing on the main thread ever calls `activate()` on these networks, so
every node renders `formatNodeActivationLabel(node.activation ?? 0)` →
`0.00`. The worker computes live activations every frame
(`bird.network.activate(observationVector)` in
`flappy-evolution-worker.simulation.frame.service.ts` `resolveBirdControlActions`),
but those values never cross the worker boundary.

### Defect D2 — no activation-aware host cache invalidation

`createHostNetworkVisualizationController` (`host/host.ts` ~lines 544-825)
caches `latestResolvedFrame` and invalidates it ONLY on canvas resize
(~line 586) and on `renderNetworkArchitecture` (~line 803). Even if node
activations were mutated in place, a stale cached frame would keep rendering
old labels. racing_curriculum works because its host activates networks live
on the main thread AND recomputes `computeNetworkStateFingerprint`
(bias + activation + weights) every frame.

### Why the chosen fix

Re-activating the champion network on the main thread would desynchronize
recurrent state from the simulation (the worker's network carries the
honest post-step recurrent memory). Streaming the worker winner's actual
activations preserves simulation honesty at trivial cost (one Float32Array
of node-count length, ~1.2 KB per frame). Winner ↔ champion index parity is
guaranteed: both sides resolve the winner with the shared
`resolveFramePrimaryWinnerIndex`, and the `currentPopulation[0]` champion
fold happens only in the `done: true` completion path, so the bird ↔ genome ↔
main-thread-network mapping is stable during playback. A transient `0.00`
before the first streamed frame (generation preview) is acceptable.

### Protocol hop map (Phase 1 implementation reference)

Worker types (`flappy-evolution-worker.types.ts`, mirrored by
`browser-entry.worker.types.ts` + `worker-channel/worker-channel.types.ts`)
→ packing (`createWorkerPlaybackSnapshot` +
`resolveWorkerPlaybackSnapshotTransferList` in
`flappy-evolution-worker.snapshot.utils.ts`) → postMessage
(`flappy-evolution-worker.playback.service.ts` `processWorkerPlaybackStep`)
→ main-thread unpack (`playback/snapshot/playback.snapshot.services.ts`
`applyPlaybackSnapshot`) → frameStats
(`playback/worker-channel/playback.worker-channel.summary.services.ts`
`resolvePlaybackFrameStats` → `playback/playback.iteration.services.ts`
`emitPlaybackFrameStats`) → runtime champion callback
(`runtime.evolution-loop.service.ts`) → host overlay
(`NetworkVisualizationHandle` extension + `host.ts` invalidation).

Conventions: SoA typed arrays (Float32Array for continuous values), unit
suffixes, defensive length fallbacks, `isTransferableArrayBuffer` filtering
SharedArrayBuffers. Capture is winner-only, post-simulation (read
`birds[winnerIndex].network.nodes[i].activation` after
`stepWorkerPopulationFrame` — zero-copy during simulation).

## Non-goals

- No changes to `src/visualization/network-view/` (the minimal
  demo-agnostic renderer from the archived Network_Visualization_Export
  _Schema plan). It is a separate, smaller API and is NOT the extraction
  target.
- Phase 2 changes racing_curriculum's visualizer imports only, not behavior.
  Phase 3 then migrates and removes the racing-specific visualizer code that
  Phase 2 leaves behind, while keeping racing behavior identical.
- No main-thread re-activation of champion networks (recurrent-state
  honesty; see Decision record).
- No new visualization features (no new label formats, palettes, or layout
  work beyond what settings threading requires).
- No worker→main-thread architecture changes beyond additive protocol
  fields (existing SoA snapshot shape is preserved).

## No Deferred Cleanup policy

- The two legacy facades
  (`browser-entry.network-view.utils.ts`,
  `browser-entry.visualization.utils.ts`) have zero consumers (verified by
  importer sweep) and MUST be deleted in the same step that introduces the
  shared folder (Step 05), not left behind.
- The flappy-side originals of every relocated dependency
  (`browser-entry.visualization.types.ts` plus its type-barrel re-export
  line, `constants.network-view.ts` plus its constants-barrel line) MUST
  be deleted in the same step that creates the shared replacements
  (Step 05), and the flappy default input-group data MUST move to the
  flappy host side in the same step that empties the domain default
  (Step 04) — no relocated original may survive its replacement.
- The four moved files with missing sibling tests
  (`network-view.draw.service.ts`, `visualization.ts`,
  `visualization.draw.service.ts`, `visualization.errors.ts`) MUST gain
  sibling tests in the SAME step that creates
  `examples/shared/network-visualization/` so the new folder passes
  `folder-quality` on creation, not later.
- The neatChat path-string constants, their test assertions, README links,
  and generated bundles (docs/assets/neat-chat.bundle.js,
  flappy-bird.bundle.js, semantic-snapshot.json) MUST all be repointed /
  rebuilt in Step 06, not deferred.

## Acceptance criteria traceability

| ID     | Intent             | Observable criterion                                                                                                                                                                                                                                                |
| ------ | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AC-001 | Worker capture     | `createWorkerPlaybackSnapshot` packs `winnerNodeActivations` (Float32Array) equal to the frame winner's post-step node activations; transfer list includes its buffer                                                                                               |
| AC-002 | Protocol carry     | `processWorkerPlaybackStep` emits `done:false` frames carrying `winnerNodeActivations` + `winnerBirdIndex`; values differ across frames                                                                                                                             |
| AC-003 | Main-thread unpack | `applyPlaybackSnapshot` / frameStats expose `winnerNodeActivations` + `winnerBirdIndex`                                                                                                                                                                             |
| AC-004 | Host overlay       | `NetworkVisualizationHandle` gains `applyNetworkActivationOverlay`; host copies values into `node.activation`, clears `latestResolvedFrame`, requests redraw                                                                                                        |
| AC-005 | Live labels        | Real visible-window browser smoke: after ≥2 streamed frames at least one node label shows a non-zero activation value, and label values change across frames                                                                                                        |
| AC-006 | Regression safety  | flappy Jest suites (network-view, playback, runtime, host, flappy-evolution-worker) all green; racing + asciiMaze suites green                                                                                                                                      |
| AC-007 | Settings seam      | Network orchestrator accepts `NetworkVisualizationSettings` (incl. `InputLabelGroupDefinition[]` groups, palette, hover duration) with value-identical `NETWORK_*` defaults and empty default input groups; flappy renders identically via its host-passed settings |
| AC-008 | Shared extraction  | `examples/shared/network-visualization/` contains both folders plus the relocated domain-root modules (types, math utils, shared constants); flappy host visualizes identically (screenshot); `folder-quality` gate PASSes on the new folder                        |
| AC-009 | Consumer repoint   | racing adapter (guard test at `racing_curriculum/browser-entry/browser-entry.test.ts:457` still passes), asciiMaze adapter, and neatChat constants all import from the shared path                                                                                  |
| AC-010 | Bundle freshness   | Generated bundles rebuilt; no live file under examples/, scripts/, or docs/assets references the old `flappy_bird/browser-entry/network-view` or `.../visualization` import paths (plan archives excluded)                                                          |
| AC-011 | Cross-demo smoke   | Visible-window browser smokes for flappy, racing_curriculum, and asciiMaze network panels after the move                                                                                                                                                            |

## Mandates

> **Session-scoped budget mandate (recorded 2026-09-12, user directive).**
> Applies to every dispatch and sub-dispatch made under this plan until the
> plan closes. Non-negotiable. Agents already running at mandate time are
> grandfathered until they complete; all FUTURE dispatches must comply.

```yaml
mandate: session-model-override
scope: 'this plan execution (budget-constrained session only)'
model: 'kimi-k2.7-code:cloud'
applies_to:
  - 'all Tier-1 dispatches (00-helping through 07-logging)'
  - 'all Tier-2 coordinator dispatches'
  - 'all Tier-3 specialist dispatches'
  - 'all Tier-4 helper dispatches'
enforcement: 'orchestrator passes model kimi-k2.7-code:cloud on every task dispatch; any agent sub-dispatching specialists MUST override its agents[] frontmatter defaults and dispatch them with kimi-k2.7-code:cloud'
overrides: 'per-agent frontmatter model defaults, for this plan only'
expires: 'when this plan completes and is archived'
```

> **Session-scoped visible-window suspension (recorded 2026-09-12 11:45,
> user directive).** Overrides the execute-skill DEMO/UI visible-window
> validation rule for the remainder of this session. Non-negotiable.

```yaml
mandate: session-visible-window-suspension
scope: 'this plan execution (current session only)'
recorded: '2026-09-12 11:45, user directive'
rule: 'visible-window browser tests are SUSPENDED for the rest of the session'
substitute_evidence: 'user manual confirmation on the live deployment is the acceptance authority for demo/UI runtime behavior'
applies_to:
  - '05-green-testing browser smokes (AC-005-style validations)'
  - 'any agent considering browser-harness-specialist or browser-ui-specialist sub-dispatches'
enforcement: 'no agent may launch a visible browser window or dispatch browser specialists; record the user manual confirmation (timestamp + live URL) as the runtime-behavior evidence instead; final approval or follow-up goes back to the user'
expires: 'when this plan completes and is archived'
```

---

## Implementation summary

All three phases are **[DONE]**. Detailed step packets, slice evidence,
validation transcripts, and fix-loop records are archived in
`plans/completed/Flappy_Visualizer_Activation_Fix_Shared_Extraction.logs.md`.

- **Phase 1 — Live champion activations (fix the 0.0 regression):** [DONE]
- **Phase 2 — Shared network visualization extraction:** [DONE]
- **Phase 3 — racing_curriculum visualizer migration:** [DONE]

## Final validation evidence

- `npx jest --config=jest.config.mjs --no-cache --selectProjects default --testPathPatterns "flappy_bird|shared/network-visualization|racing_curriculum|asciiMaze" --runInBand` → **PASS — 96 suites, 861 tests**.
- `npx jest --config=jest.config.mjs --no-cache --selectProjects asciiMaze-browser --testPathPatterns "asciiMaze/(networkVisualization|browser-entry/network-view)" --runInBand` → **PASS — 2 suites, 8 tests**.
- `npm run build:flappy-bird:host` → **PASS — docs/assets/flappy-bird.bundle.js 759.3 kb**.
- `npm run build:racing-curriculum` → **PASS — docs/assets/racing-curriculum.bundle.js 799.7 kb**.
- `node scripts/agent-customization/gates/folder-quality.gate.mjs --folder=examples/shared/network-visualization --json` → **PASS — 0 TS/ESLint/JSDoc/test/coverage gaps**.
- Visible-window browser smoke (flappy + racing network panels): **SUSPENDED** per session mandate — user manually confirms runtime at `http://192.168.1.209:8080/docs/examples/`.
- Closure gates (`slice-advancement`): plan-sync, step-packet, plan-slice-quality, plan-command-lint all green.

## Archive note

This tracker and its logs have been moved to `plans/completed/`.
AC-280 remains **SUSPENDED** pending the user's manual runtime confirmation.

