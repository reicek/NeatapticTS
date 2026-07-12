# Racing Curriculum Pit Parity, Pit-Stop Behavior, Tire Wear, and Tier 5 Activation

## Question

Why does the racing curriculum demo show these related symptoms at Tier 3+?

1. Red-team (Team B) pit boxes are not visible on the track, so pit parity looks broken.
2. Cars appear to repair tires far away from any pit box.
3. Tire wear feels too fast once tire wear is enabled.
4. Tier 5 never activates through normal lap-based progression.
5. The red team often stalls/fails when tire wear is active.

This research step does **not** modify production code; it identifies root causes and recommends implementation changes.

## Evidence

### 1. Pit geometry is implemented as 6 alternating boxes, not one per team

`examples/racing_curriculum/track/track.generator.ts` defines six lap-progress anchors (`ALTERNATING_PIT_PROGRESS_SAMPLES`, lines 36–38) and builds six pit boxes in `[0,1,0,1,0,1]` team order (lines 216–230).

This contradicts `examples/racing_curriculum/reference.plans.md`, which describes one dedicated pit per team. The current implementation therefore has three pit boxes per team, but the rest of the runtime treats pits as a compact per-team shelf.

### 2. Renderer hides the non-focused team's pits

`examples/racing_curriculum/renderer/racing.renderer.ts`:

- `resolveVisiblePitTeamIndex` (lines 1758–1769) returns the focused car's team index.
- `drawPitOverlays` (lines 1218–1274) skips any `pitBox` whose `teamIndex` differs from `visiblePitTeamIndex`.

Because the default focus car is car 0 (Team A / blue), only blue pit overlays render. Red pit boxes exist in the `TrackSpec` but are not drawn, which directly causes the "red team has no pits" symptom.

### 3. Worker race-pack cars keep driving while their pit timer runs down

`examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts` `tick()` (lines 525–645):

- Step 1 ticks the pit lifecycle and Step 7 resolves pit entries.
- Between those steps, every car still receives throttle, forward movement, and tire decay (lines 578–628).
- There is **no** `isCarStoppedInPit` guard comparable to the one in `environment.step.service.ts`.

When the pit timer expires, `tickPitLifecycle` (lines 900–921) resets tires and releases the car, but the car has already moved far past the pit because it never stopped.

### 4. Environment path stops the car but leaves it at the entrance corridor

`examples/racing_curriculum/environment/environment.step.service.ts`:

- `isCarStoppedInPit` (lines 628–636) freezes a car in place during its pit stop.
- `resolvePitEntries` (lines 743–776) assigns an available team pit slot when the car center is inside the entrance corridor, but it **does not** move the car to the pit-box center.
- As a result, stopped cars sit at the entrance rather than inside the box.

### 5. Tire decay constants are too aggressive

`examples/racing_curriculum/environment/environment.step.service.ts` lines 38–42:

```ts
const TIRE_DECAY_LATERAL_FACTOR = 0.00012;
const TIRE_DECAY_LONGITUDINAL_FACTOR = 0.00006;
const TIRE_DECAY_SPEED_FACTOR = 0.000006;
```

These constants drive `decayTireState` (line 108) and are consumed by both the environment probe (`browser-entry.ts` line 1860) and the worker stepping loop (`race-pack.service.ts` lines 619–628). They are the single source of truth for the "tires degrade too fast" symptom.

### 6. Auto-promotion is hard-capped at Tier 4

`examples/racing_curriculum/browser-entry/browser-entry.ts`:

- `MAX_FALLBACK_AUTOPROMOTION_TIER = 4` (line 339).
- `resolveTierPromotionFromLapCount` (lines 2267–2281) returns the current tier unchanged when `currentTier >= MAX_FALLBACK_AUTOPROMOTION_TIER`.

Consequently, once the player reaches Tier 4, lap completions no longer advance the tier. Tier 5 can only be reached by explicitly calling `start(container, { tier: 5 })`.

### 7. Tire wear is masked below Tier 4

`examples/racing_curriculum/browser-entry/browser-entry.ts`:

- `TIRE_WEAR_START_TIER = 4` (line 166).
- `stabilizeCurriculumTierTireGrip` (lines 2130–2148) forces full tire health for any tier below 4.

Therefore:

- Pit parity at Tier 3 is purely a rendering issue (red pits are hidden).
- Fast tire wear and pit-stop movement bugs only become visible at Tier 4+.

### 8. Worker `pitStatus` initializer for 6-car packs is malformed

`examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts` lines 1132–1144:

```ts
const pitStatus =
  agentCount >= TIER_FIVE_CAR_COUNT
    ? new Uint8Array([
        NO_CAR_INDEX,
        0,
        NO_CAR_INDEX,
        NO_CAR_INDEX,
        0,
        NO_CAR_INDEX,
      ])
```

The documented compact layout for 6-car packs is `[teamA_car, teamA_ticks, teamA_wait, teamB_car, teamB_ticks, teamB_wait]`. The initializer places the waiting slot incorrectly, which is a latent bug if the wait slot is ever used for re-entry timing.

## Decision

All five symptoms are caused by **demo/runtime implementation issues**, not by the core NEAT/NGE library or the track generator algorithm itself. The root causes are:

| Symptom | Root cause | Primary file |
|---|---|---|
| Red pits invisible | Renderer filters pit overlays to the focused car's team | `renderer/racing.renderer.ts` |
| Cars repair far from pits | Worker `tick()` lacks a pit-stop movement guard | `workers/simulation-worker/simulation-worker.race-pack.service.ts` |
| Cars sit at entrance instead of in box | Environment `resolvePitEntries` never teleports the car to the box center | `environment/environment.step.service.ts` |
| Tire wear too fast | Fixed high decay constants in `environment.step.service.ts` | `environment/environment.step.service.ts` |
| Tier 5 never activates | `MAX_FALLBACK_AUTOPROMOTION_TIER = 4` | `browser-entry/browser-entry.ts` |
| Red team stalls with tire wear | Combined effect of hidden pits + no worker stop guard + fast decay | multiple files |

The most important architectural tension is the mismatch between:

- **Environment pit model**: 6 slots (`PIT_SLOTS_PER_TEAM = 3`, `PIT_SLOT_COUNT = 6`), matching the six generated pit boxes.
- **Worker/renderer pit model**: compact per-team shelf with stride 2 (4-car packs) or stride 3 (6-car packs), independent of how many physical boxes exist.

Any implementation pass must first decide whether to keep the six-box geometry or redesign to one-box-per-team, then unify the pit-state shelf so the environment and worker agree.

## Risks

1. **Pit-state layout mismatch.** Changing the renderer without also aligning the worker/environment pit shelf may cause occupancy indicators to lie or cause two cars to be assigned the same slot.
2. **Tier 5 exposure.** Raising `MAX_FALLBACK_AUTOPROMOTION_TIER` to 5 will surface the 6-car pack and the malformed `pitStatus` initializer. The worker pit-stop guard must be fixed before Tier 5 is reachable by normal play.
3. **Shared tire-decay constants.** The same constants are used by the deterministic controller probe and the live worker. Tuning them affects both regression tests and the browser demo.
4. **Pit geometry redesign.** Switching to one pit per team changes `TrackSpec.pitBoxes` length and may break any code that assumes six boxes (renderer stride logic, environment slot mapping).
5. **Focus-car coupling.** The renderer pit-filtering is tied to `focusCarIndex`; any fix must consider spectator/follow modes where no single car should dominate pit visibility.

## Implementation Recommendations

1. **Renderer parity** — Remove the team filter in `drawPitOverlays` so both teams' pit boxes always render when pits are enabled. If focus-car highlighting is still desired, tint or outline the focused team's boxes without hiding the others.
2. **Worker pit-stop guard** — In `race-pack.service.ts` `tick()`, skip forward movement and tire decay for any car whose `pitStatus` slot shows an active stop (`pitStatus[teamBase + 1] > 0`). Restore fresh tires only when the timer reaches zero and the car is released.
3. **Environment pit positioning** — In `environment.step.service.ts` `resolvePitEntries`, move the entering car's center to `pitBox.boxCenter` and keep it there until the stop timer expires.
4. **Tire wear tuning** — Reduce `TIRE_DECAY_LATERAL_FACTOR`, `TIRE_DECAY_LONGITUDINAL_FACTOR`, and `TIRE_DECAY_SPEED_FACTOR` by roughly 30% (or scale them by curriculum tier), then verify with the deterministic controller probe and a live Tier 4/5 browser run.
5. **Tier 5 progression** — Raise `MAX_FALLBACK_AUTOPROMOTION_TIER` to 5, or remove the cap, so lap-based promotion can reach Tier 5. Fix the 6-car `pitStatus` initializer to match `[teamA_car, teamA_ticks, teamA_wait, teamB_car, teamB_ticks, teamB_wait]`.
6. **Pit geometry decision** — Decide whether the final design is:
   - **One pit per team** (aligns with `reference.plans.md`; requires shrinking `pitBoxes`, `PIT_SLOT_COUNT`, and renderer worker models), or
   - **Multiple pits per team as capacity** (keeps current six-box layout; document it as a 3-slot-per-team capacity model and unify the worker shelf length with `PIT_SLOT_COUNT`).

   Either choice is acceptable, but mixing the two models is what produced the current bugs.

## Browser Evidence

A live Tier 5 screenshot was captured to show the missing red pit overlays:

`docs/research/racing-curriculum-pit-parity-evidence.png`

The capture page is `tmp/racing-tier5-evidence.html`, which starts the demo at Tier 5 with the default focus car. Because of the `resolveVisiblePitTeamIndex` filter, only blue pit overlays are expected to appear in this screenshot.
