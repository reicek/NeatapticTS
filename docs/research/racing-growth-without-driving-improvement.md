# Racing Growth Without Driving Improvement — Root-Cause Analysis

## Question

Why do NGE racing curriculum networks grow (N109→N523) but agents don't improve at driving? Specifically:

1. What is the "guide" mechanism on Tier 1?
2. How does the evaluator score driving quality?
3. How does tier promotion and agent selection work?
4. What are the tier completion criteria per the plan?
5. Why doesn't growth translate to driving improvement?

## Evidence

### 1. The "Guide" Mechanism

**What it is:** The "guide" is a visual optimal-line overlay rendered on the track canvas with
adjustable transparency (`guidanceAlpha`).

**How it works:** `resolveGuidanceAlphaForTier(tier)` in `nge.controller.ts` (line 202) controls
rendering transparency:
- Tier 0 (scripted baseline): alpha = 1.0 (full overlay)
- Tier 1: alpha = 0.35 (`TIER_ONE_GUIDANCE_ALPHA`) — faint hint
- Tier 2+: alpha = 0 — overlay removed

**Critical insight:** The guide is a **visual rendering** effect only. The observation vector
still carries optimal-line data at ALL tiers. Channels 16 (`optimalLineLateralOffsetWorld`) and
17 (`optimalLineHeadingErrorRadians`) in `observation.assembler.ts` exist at every tier. The
sensory data does not change between tiers 1 and 2 — only the visual rendering fades.

**Source files:**
- `nge.controller.ts` line 202: `resolveGuidanceAlphaForTier`
- `observation.assembler.ts`: channel 16/17 present at all tiers
- `browser-entry.ts` line 2653: `resolveGuidanceAlphaForCurriculumTier`

### 2. How the Evaluator Scores Driving Quality

**The evaluator is `evaluateRacingTrendScore`** (`runtime.adaptation.ts` line 522):

```
score = scoreMean + scoreTrend * 0.5 + complexityBonus
```

Where:
- `scoreMean` = mean of `toDrivingQuality(signal)` over the rolling window (60 ticks)
- `scoreTrend` = last - first quality in the window
- `complexityBonus` = (nodes + connections) × 0.0001

**`toDrivingQuality`** (line 555):
```
quality = trackProgress * 0.35 + forwardSpeed * 0.25 + headingAlignment * 0.3 - offTrackPenalty * 0.1
```

**How `RacingQualitySignal` is constructed** (`browser-entry.ts` lines 570-587):
- `trackProgress` = `lastClosestSplineSampleIndex / splineSamples.length` — **shared across ALL
  cars** via `curriculumProgress.lapProgress`, not per-car. Does not change within a tick.
- `forwardSpeed` = `Math.max(0, perCarTickResult.control.throttle)` — **commanded throttle**, not
  actual physics speed
- `headingAlignment` = `perCarTickResult.evidence.headingAlignment01` — from observation channels 16/17
- `offTrackPenalty` = `Math.min(1, perCarTickResult.evidence.lateralErrorNormalized)` — from
  observation channel 16

**ROOT CAUSE — The evaluator does NOT re-run the network to measure behavioral change.**

The adaptation engine (`createRuntimeAdaptationEngine`, line 200) evaluates:
1. `baselineScore = evaluateScore(network, evidenceWindow)` — line 231
2. `runNgeLifecycle` mutates `network` in place — line 338
3. `candidateScore = evaluateScore(network, evidenceWindow)` — line 372

Both calls use the **same `evidenceWindow`**. Since `evaluateRacingTrendScore` only uses
`scoreHistory` (for `scoreMean` and `scoreTrend`) and `network.nodes.length +
network.connections.length` (for `complexityBonus`), the ONLY difference between baseline and
candidate is the `complexityBonus`.

**The score is never computed from a forward pass.** The evaluator never feeds an observation
through the network to see if the mutated topology actually changes the controller's outputs.
It only compares the same historical score window plus a complexity bonus that always favors the
larger network.

**`improvementThreshold: 0`** (`browser-entry.ts` line 474): The browser demo overrides the
default threshold of 0.01 to 0. With threshold 0, ANY positive score delta commits. Since the
complexityBonus always increases when a mutation adds nodes/connections, **every structural
mutation that adds capacity is automatically committed**, regardless of driving quality.

**Physics rewards are disconnected:** The environment step service computes real rewards
(`OFF_TRACK_CLAMP_REWARD = -1`, `WRONG_DIRECTION_REWARD = -1`, line 196-204) and stores them in
`car.reward`. But `browser-entry.ts` never reads `.reward`. The adaptation engine only sees the
observation-vector-derived `RacingQualitySignal`, not the physics reward.

**Source files:**
- `runtime.adaptation.ts` lines 200-426: `createRuntimeAdaptationEngine`
- `runtime.adaptation.ts` lines 522-566: `evaluateRacingTrendScore`, `toDrivingQuality`
- `runtime.adaptation.ts` line 499: `RACING_COMPLEXITY_WEIGHT = 0.000_1`
- `runtime.adaptation.ts` line 170: `DEFAULT_IMPROVEMENT_THRESHOLD = 0.01`
- `browser-entry.ts` line 474: `improvementThreshold: 0`
- `browser-entry.ts` lines 570-587: `RacingQualitySignal` construction
- `environment.step.service.ts` lines 36-38, 196-204: physics rewards (unused by evaluator)

### 3. Tier Promotion and Agent Selection

**`resolveTierPromotionFromLapCount`** (`browser-entry.ts` line 2588): Advances tier when
`completedLaps >= 3` (`LAP_COMPLETIONS_REQUIRED_FOR_TIER_ADVANCE`).

**No agent selection exists.** Promotion is purely lap-count-based. ALL cars advance to the next
tier simultaneously. There is no:
- Fitness comparison between agents
- Tournament selection
- Best-agent retention
- Within-team queen selection
- Cross-team coordination

The plan's reference design (`reference.plans.md` lines 379-389) describes rich promotion rules:
within-team refill (best car becomes queen), cross-team promotion (both teams must reach
threshold), capacity floor gate, growth-velocity gate. **None of these are implemented in the
browser demo.**

**Source files:**
- `browser-entry.ts` line 2588: `resolveTierPromotionFromLapCount`
- `reference.plans.md` lines 379-389: designed but unimplemented promotion rules

### 4. Tier Completion Criteria Per the Plan

The plan (`NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` lines 64-107) specifies:

- **Capacity floor gate (DR-003):** median hidden-node count must meet tier's N_floor (Tier 1:
  90→500, Tier 2: 2,000, etc.)
- **Growth-velocity gate (DR-003):** minimum growth-velocity floor (median nodes gained per
  generation)
- **Reliability requirement:** team must complete current tier reliably over deterministic race
  variants
- **Cross-team requirement:** both teams must reach promotion threshold

The reference design (`reference.plans.md` lines 591-597) also specifies an optimal-line guidance
fade policy:
- Tier 1-2: explicit optimal line, lateral error, heading error, target speed envelope
- Tier 3-4: guidance available but less reliable
- Tier 5-6: guidance reduced or removed

The plan specifies (lines 166-168) that the `complexityBonus` should be **performance-gated** —
only awarded when driving quality actually improves.

**Actual implementation:** Just `completedLaps >= 3` → auto-promote. No node count check, no lap
time improvement check, no performance gate, no cross-team coordination. The complexityBonus is
unconditional.

**Source files:**
- `NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` lines 64-107: tier ladder, promotion rules
- `NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md` lines 166-168: performance-gated complexity bonus
- `reference.plans.md` lines 591-597: guidance fade policy

### 5. Why Growth Doesn't Translate to Driving Improvement

**Summary of the causal chain:**

1. The evaluator compares pre-mutation and post-mutation scores using the **same rolling score
   history window** — the driving-quality components (`scoreMean`, `scoreTrend`) are identical
   for both.

2. The only score difference is `complexityBonus = (nodes + connections) × 0.0001`, which always
   increases when a mutation adds structure.

3. `improvementThreshold: 0` means any positive delta commits — so every growth mutation is
   accepted.

4. The evaluator **never runs a forward pass** to check whether the mutated topology actually
   changes the controller's outputs on any observation.

5. The `RacingQualitySignal` components are **weak proxies**:
   - `trackProgress` is shared across all cars and doesn't change within a tick
   - `forwardSpeed` is commanded throttle, not actual physics speed
   - `headingAlignment` and `offTrackPenalty` come from the observation vector, not from physics

6. Real physics rewards (`car.reward`: off-track penalty, wrong-direction penalty) are computed
   but **never fed** to the adaptation evaluator.

7. NGE mutations (addNode, addConnection) add **dead-weight structure** because nothing validates
   that new nodes/connections actually change controller behavior. There is no behavioral delta
   check, no forward-pass comparison, no fitness evaluation against real track performance.

8. Tier promotion requires only 3 completed laps — no performance gate, no node-count floor, no
   lap-time improvement check. Networks auto-promote regardless of driving quality.

**Result:** Networks grow because the evaluator rewards structural complexity unconditionally.
Driving doesn't improve because the evaluator never measures actual driving behavior change from
a mutation.

## Decision

### Required Fixes (7)

1. **Set `improvementThreshold` > 0** — Restore to at least the default 0.01, or higher. This
   alone prevents zero-delta commits, but is insufficient alone since complexityBonus still
   provides a positive delta for any structural growth.

2. **Make the evaluator re-run the network on observations** — The candidate evaluation must
   forward-pass the mutated network on the current observation vector and compare the output
   delta against the pre-mutation output. If the mutation doesn't change controller behavior on
   any observation in the evidence window, it should not be committed. This is the most important
   fix.

3. **Feed physics rewards into `RacingQualitySignal`** — Replace or augment the
   observation-vector-derived proxies with actual physics rewards (`car.reward` from
   `stepEnvironment`). The off-track penalty and wrong-direction penalty are already computed by
   the physics engine but are never read by the adaptation engine.

4. **Make `trackProgress` per-car** — Currently shared across all cars via
   `curriculumProgress.lapProgress.lastClosestSplineSampleIndex`. Each car should report its own
   spline progress based on its own position. This makes the progress signal actually
   differentiate between cars that drive well and cars that don't.

5. **Use actual forward speed from physics** — Replace `forwardSpeed = Math.max(0,
   control.throttle)` with actual speed from the physics state
   (`Math.abs(effectiveThrottle) * MAX_FORWARD_SPEED_UNITS_PER_SECOND` or the equivalent from
   the car state). This measures actual movement, not commanded intent.

6. **Gate `complexityBonus` on actual performance improvement** — As the plan specifies (lines
   166-168), the complexity bonus should only be awarded when driving quality actually improves.
   Implementation: only add `complexityBonus` when `scoreTrend > 0` or when
   `toDrivingQuality(candidate) > toDrivingQuality(baseline)` on a forward-pass comparison.

7. **Implement the plan's capacity floor gate (N_floor) and growth-velocity gate in tier
   promotion** — Tier promotion should require:
   - median hidden-node count ≥ tier's N_floor
   - minimum growth-velocity (nodes gained per generation)
   - lap time improvement (or at minimum, no lap time regression) on the current tier
   - cross-team coordination (both teams must reach threshold)

   This replaces the current `completedLaps >= 3 → auto-promote` logic.

### Optional Fixes

8. **Implement within-team agent selection** — When promoting, the best-performing car should
   become the "queen" whose DNA dominates the next generation, as specified in the reference
   design. This creates actual selection pressure for driving quality.

9. **Vary observation channels by tier** — The guidance fade policy should remove
   `optimalLineLateralOffsetWorld` and `optimalLineHeadingErrorRadians` from the observation
   vector at higher tiers (not just the visual overlay), forcing the network to internalize
   line-following. Currently these channels exist at all tiers, so the "guide removal" is
   purely cosmetic.

10. **Add a behavioral delta check** — Before committing a mutation, forward-pass both the
    pre-mutation and post-mutation networks on a sample of recent observations. If the output
    delta is below a threshold (e.g., cosine similarity > 0.99), reject the mutation as
    behaviorally inert — it added structure without changing behavior.

## Risks

- **Fix 2 (forward-pass evaluation) has a performance cost.** Running forward passes on every
  mutation candidate may reduce real-time FPS. Mitigation: only forward-pass on a small sample
  (3-5) of recent observations, not the full evidence window.

- **Fix 7 (N_floor gates) may stall progression.** If the network can't reach the N_floor
  through valid mutations alone (without the complexityBonus crutch), the curriculum may get
  stuck. Mitigation: combine with fix 6 to gate complexityBonus on performance, allowing genuine
  improvements to accumulate toward the N_floor.

- **Fix 9 (removing observation channels) may break the remap logic.** The
  `remapControllerNetworkForObservationTier` function only handles adding inputs, not removing
  them. Removing channels would require a different remap strategy (e.g., zero-weighting the
  removed channels rather than deleting input nodes).

- **The reference design is aspirational.** Many NGE prerequisites (GatedRecurrentCell,
  EpisodicSlot, ModulatorBroadcaster, GatingRouter, polyandric reproduction) are not yet
  implemented. The browser demo is a simplified single-population version. Fixes should be
  scoped to the current demo's architecture, not the full reference design.

## Source-of-Truth Order

Evidence was gathered by direct file reads (runtime code > plan documents > reference design).
Cortex RAG search was attempted first but returned mostly `src/` results, not the
`examples/racing_curriculum/` demo files. Direct file reads were used as fallback per the
Cortex-First Search Policy.

Confidence: 98% — all findings are from direct source code reading with line-number citations.