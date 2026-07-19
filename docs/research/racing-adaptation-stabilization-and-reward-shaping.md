# Racing Adaptation Stabilization & Reward Shaping — Research Report

## Question

Two user-reported issues with the NGE racing curriculum:

1. **Stabilization period after node addition:** Weights should refine and stabilize
   after adding nodes BEFORE adding more. Currently nodes are added too frequently
   without giving the network time to learn to use existing structure.

2. **Stronger reward/penalty shaping:** Agents should receive STRONG positive rewards
   for following the guide line and STRONG negative penalties for crashing against
   borders, going in the wrong direction, or diverging from the guide. Currently
   agents go against the border for a long time and eventually go in the wrong
   direction.

## Evidence

### 1. Current Adaptation Cycle Flow

**Source:** `examples/racing_curriculum/controller/runtime.adaptation.ts`
**Source:** `examples/racing_curriculum/browser-entry/browser-entry.ts` lines 506-521

#### Adaptation Engine Configuration (browser-entry.ts:511-516)

```typescript
const engines = createPerCarAdaptationEngines(rosterSize, {
  evaluateScore: evaluateRacingTrendScore,
  improvementThreshold: 0,          // ANY non-negative improvement commits
  cadence: { mode: 'every_n_ticks', everyNTicks: 4 },  // check every 4 ticks
  limits: { mutationCooldownTicks: 5, rollbackCooldownTicks: 5 },
});
```

#### Default Limits (runtime.adaptation.ts:164-170)

```typescript
const DEFAULT_LIMITS: RuntimeAdaptationLimits = {
  maxStructuralEditsPerStep: 1,
  maxNodes: 8_000,
  maxConnections: 32_000,
  mutationCooldownTicks: 5,
  rollbackCooldownTicks: 5,
};
```

#### Key Constants

- `MAX_EPISODIC_SLOTS = 100` (line 181) — allows up to 100 episodic growth slots per cycle
- `LARGE_NETWORK_NODE_THRESHOLD = 1_000` (line 188) — growth throttle only activates above 1000 nodes
- `GROWTH_THROTTLE_BASE_INTERVAL_TICKS = 3` (line 194) — base throttle interval at scale
- `RACING_RUNTIME_SCORE_HISTORY_WINDOW = 60` (browser-entry.ts:139) — rolling window size
- `DEFAULT_MINIMUM_EVIDENCE_WINDOW = 4` (line 173) — minimum 4 entries before evaluation
- `DEFAULT_IMPROVEMENT_THRESHOLD = 0.01` (line 172) — but browser overrides to 0

#### Adaptation Cycle Steps (adaptOnTick, lines 228-416)

1. **Cadence gate** (Step 1, line 236): Checks if `tick % everyNTicks === 0`. With
   `everyNTicks=4`, adaptation is evaluated every 4 simulation ticks (~66ms at 60fps).

2. **Evidence window gate** (Step 2, line 261): Requires at least 4 entries in the
   score history. The window holds up to 60 entries (RACING_RUNTIME_SCORE_HISTORY_WINDOW).

3. **Hysteresis advancement** (Step 3, line 274-279): Advances growth hysteresis.
   The `isPositiveFocusWindow` is determined by comparing the last vs first quality
   in the evidence window. The hysteresis config is set with
   `hysteresisWindowCount: 0` (line 346), which means `canGrowNow()` always passes
   the positive-streak check (0 >= 0 is always true).

4. **Cooldown check** (line 281): If `hysteresis.cooldownWindowsRemaining > 0`,
   returns `mutation_cooldown_active`. After a commit, `commitGrowth()` sets
   `cooldownWindowsRemaining = config.cooldownWindowCount = 5`. Each subsequent
   cadence tick decrements this by 1. So **5 cadence ticks = 20 simulation ticks
   of cooldown** after a successful growth commit.

5. **Rollback cooldown check** (line 294): If `tick < nextRollbackTick`, returns
   `rollback_cooldown_active`. After a rollback, `nextRollbackTick = tick + 5`.
   So **5 simulation ticks of rollback cooldown**.

6. **Growth throttle** (Step 3.5, line 309): Only throttles when
   `network.nodes.length > 1000`. Below 1000 nodes, no throttling at all.

7. **Lifecycle execution** (Step 6, line 340): Calls `runNgeLifecycle` which plans
   growth morphs (edge densify, slot expand, node add) and applies them to the
   live network.

8. **Commit/rollback** (Steps 8-10, lines 371-415): Computes candidate score,
   checks if `improvement >= improvementThreshold`. With `improvementThreshold=0`,
   **any non-negative score delta commits the mutation**.

#### Effective Growth Frequency

- Growth attempt: every 4 ticks
- After successful commit: 20 ticks cooldown (5 cadence windows × 4 ticks)
- After rollback: 5 ticks cooldown
- At 60fps: growth attempt roughly every 24 ticks = ~0.4 seconds
- **No fitness plateau detection**, no "epochs between growth", no weight convergence
  check, no stabilization period based on learning progress

#### buildGrowthBudget (line 860-872)

Passes `MAX_EPISODIC_SLOTS = 100` as `maxEpisodicSlots`. This allows up to 100
episodic slot expansions per growth cycle. The budget also carries
`currentEpisodicSlotCount: 0` — always starts at zero, meaning the slot budget
is fully available every cycle.

#### evaluateRacingTrendScore (line 655-686)

```
score = scoreMean + scoreTrend * 0.5 + complexityBonus
```

Where:
- `scoreMean` = mean of `toDrivingQuality(signal)` over the evidence window
- `scoreTrend` = last quality - first quality in the window
- `complexityBonus` = forward-pass output variance × 0.0001, gated on non-negative trend

The evaluator uses the **same scoreHistory** for both baseline and candidate
networks. Only the complexityBonus differs (because the candidate network has
different forward-pass outputs). This means the driving-quality components
(scoreMean, scoreTrend) are identical between baseline and candidate — the
mutation commit decision is driven almost entirely by the complexity bonus.

### 2. Current Reward/Penalty Signal Values

**Source:** `examples/racing_curriculum/environment/environment.step.service.ts`

#### Physics Penalties (lines 35-38)

```typescript
const OFF_TRACK_CLAMP_REWARD = -1;   // when car is clamped back onto track
const WRONG_DIRECTION_REWARD = -1;  // when moving opposite to track tangent
```

These are **weak penalties** (magnitude 1 each). A car that is both off-track AND
going the wrong direction receives a total reward of -2.

#### Penalty Application (lines 188-204)

```typescript
const clamped = clampCarToTrackBounds(car, trackSpec);
const wasClamped = clamped.carX !== car.carX || clamped.carY !== car.carY;
let reward = 0;
if (wasClamped) {
  reward += OFF_TRACK_CLAMP_REWARD;   // -1
}
if (wrongDirectionFlags[carIndex]) {
  reward += WRONG_DIRECTION_REWARD;   // -1
}
return reward === 0 ? clamped : { ...clamped, reward };
```

#### toDrivingQuality Weighting (runtime.adaptation.ts:699-711)

```typescript
return (
  entry.trackProgress * 0.35 +
  entry.forwardSpeed * 0.25 +
  entry.headingAlignment * 0.3 -
  entry.offTrackPenalty * 0.1 +       // off-track penalty weighted at only 0.1
  (entry.physicsReward ?? 0) * 0.1    // physics reward weighted at only 0.1
);
```

**Critical finding:** The physicsReward (which carries the -1 border penalty and
-1 wrong-direction penalty) is weighted at only **0.1**. The offTrackPenalty (0..1
lateral distance fraction) is also weighted at only **0.1**. The dominant signals
are trackProgress (0.35), headingAlignment (0.3), and forwardSpeed (0.25).

Effective penalty contribution to quality score:
- Off-track border collision: physicsReward × 0.1 = -1 × 0.1 = **-0.1**
- Wrong direction: physicsReward × 0.1 = -1 × 0.1 = **-0.1**
- Both: -2 × 0.1 = **-0.2**
- Off-track lateral distance: offTrackPenalty × 0.1 = up to **-0.1**

Compare to positive signals:
- Track progress: up to 0.35
- Forward speed: up to 0.25
- Heading alignment: up to 0.3
- Total positive range: up to 0.9

**The penalties are 4.5× to 9× weaker than the positive signals.** A car that is
off-track but still making forward progress along the track can easily have a net
positive quality score because trackProgress (0.35) + forwardSpeed (0.25) = 0.6
dwarfs the -0.1 border penalty.

### 3. Guide Line System

**Source:** `examples/racing_curriculum/controller/nge.controller.ts`
**Source:** `examples/racing_curriculum/controller/observation.assembler.ts`
**Source:** `examples/racing_curriculum/renderer/racing.renderer.ts`

#### What the Guide Line Is

The "guide line" is a **visual rendering overlay** — a dashed line drawn parallel
to the road centerline at the team's lane center. Team 0 (blue) gets the inner-lane
centerline; Team 1 (red) gets the outer-lane centerline.

- `buildGuidingLineForTeam()` in `racing.renderer.ts` (line 285) constructs the
  per-team guiding line points.
- `drawTeamGuidingLines()` in `racing.renderer.ts` (line 723) renders them with
  `guidanceAlpha` controlling transparency.

#### Guidance Alpha by Tier (nge.controller.ts:202-212)

```typescript
export function resolveGuidanceAlphaForTier(tier: 0 | 1 | 2): number {
  if (tier === 0) return 1;       // full overlay (scripted baseline)
  if (tier === 1) return 0.35;    // faint hint
  return 0;                       // off at Tier 2+
}
```

In browser-entry.ts (line 3215-3224), the curriculum tier maps to guidance tier:
- Curriculum Tier 1 → guidance tier 1 → alpha 0.35
- Curriculum Tier 2+ → guidance tier 2 → alpha 0

#### Guide Line Signals in the Observation Vector

The observation vector includes two guide-line-related channels at ALL tiers:

- **Channel 16** (`optimalLineLateralOffsetWorld`): Signed lateral distance from
  the car to the team's target lane centerline, normalized by
  `OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE = 18` world units.
  Source: `observation.assembler.ts` line 901-914, 764-766.

- **Channel 17** (`optimalLineHeadingErrorRadians`): Angular difference between
  the track tangent heading and the car heading, wrapped to [-π, π], normalized
  by π.
  Source: `observation.assembler.ts` line 952-956, 768-770.

#### Critical Finding: No Positive Reward for Following the Guide

There is **NO positive reward signal** for following the guide line. The guide line
data is only available as **observation inputs** (channels 16 and 17). The reward
system does not compute a "guide-following" bonus. The `RacingQualitySignal` does
not include a guide-following component — only trackProgress, forwardSpeed,
headingAlignment, offTrackPenalty, and physicsReward.

The `headingAlignment` signal (dot product of car heading and track tangent) is the
closest proxy, but it measures alignment with the track direction, not alignment
with the guide line specifically.

### 4. Border Collision Detection and Penalty Strength

**Source:** `environment.step.service.ts` lines 655-692

#### Detection Mechanism

`clampCarToTrackBounds()` (line 655) finds the nearest spline sample, computes the
signed lateral offset from the sample center, and if `|signedOffset| > halfWidth`,
clamps the car back to the track edge. The clamping is detected by comparing
pre/post positions.

#### Penalty

- `OFF_TRACK_CLAMP_REWARD = -1` — applied once per tick when clamping occurs
- In `toDrivingQuality`: weighted at `physicsReward × 0.1 = -0.1`
- **This is far too weak.** The car can drive against the border indefinitely,
  getting clamped every tick, and the -0.1 penalty is negligible compared to the
  0.35 trackProgress + 0.25 forwardSpeed positive signals.

#### No Progressive/Escalating Penalty

The penalty is a flat -1 per tick. There is no escalation for prolonged border
contact, no cumulative penalty, and no speed-dependent penalty.

### 5. Direction Tracking and Wrong-Direction Penalty

**Source:** `environment.step.service.ts` lines 385-426

#### Detection Mechanism

`detectWrongDirection()` compares each car's pre-step and post-step positions,
computes the displacement vector, and checks if it has a negative dot product with
the track tangent at the pre-step nearest sample. Stationary cars are not flagged.

#### Penalty

- `WRONG_DIRECTION_REWARD = -1` — applied once per tick when wrong direction detected
- In `toDrivingQuality`: weighted at `physicsReward × 0.1 = -0.1`
- **Far too weak.** Combined with border collision, the maximum penalty is -0.2,
  still dwarfed by positive progress signals.

#### No Direction History or Accumulation

There is no accumulation of wrong-direction penalties over time, no "wrong direction
streak" counter, and no escalating penalty for persistent wrong-way driving.

### 6. Network Observation Vector

**Source:** `observation.assembler.ts` lines 703-780

#### Tier 1 Base Vector (70 channels)

| Index | Channel | Description |
|-------|---------|-------------|
| 0 | carX (normalized) | Car X position relative to track center |
| 1 | carY (normalized) | Car Y position relative to track center |
| 2 | sin(carHeading) | Heading sine |
| 3 | cos(carHeading) | Heading cosine |
| 4 | forwardSpeed | Forward speed (normalized by 108) |
| 5 | lateralSpeed | Lateral speed (normalized by 54) |
| 6 | speed | Total speed (normalized by 108) |
| 7 | yawRate | Yaw rate |
| 8 | slipAngle | Slip angle (normalized by π/2) |
| 9 | progress01 | Track progress fraction |
| 10 | lapProgress01 | Lap progress fraction |
| 11 | boundaryDistanceLeft | Distance to left border (normalized by 24) |
| 12 | boundaryDistanceRight | Distance to right border (normalized by 24) |
| 13 | boundaryBalance | Left/right balance |
| 14 | hazardDistance | Hazard distance (normalized by 64) |
| 15 | waypointDistance | Waypoint distance (normalized by 64) |
| **16** | **optimalLineLateralOffset** | **Lateral offset from guide line (normalized by 18)** |
| **17** | **optimalLineHeadingError** | **Heading error to guide (normalized by π)** |
| 18 | targetSpeed | Target speed (normalized by 108) |
| 19 | speedError | Target speed - actual speed |
| 20-59 | 5×8 look-ahead segments | 5 look-ahead track segments, 8 channels each |
| 60-69 | 10 memory trace channels | Recurrent memory trace |

The network **does see** the guide line (channels 16, 17), border proximity
(channels 11, 12, 13), and direction (channels 2, 3, 17). The problem is not
observational — it is that the **reward signal does not strongly penalize bad
behavior or reward good behavior**.

### 7. Root Cause Analysis

#### Why Agents Go Against Borders

1. **Border penalty is negligible**: The -1 physicsReward weighted at 0.1 produces
   only -0.1 quality impact. The car can drive against the border and still have a
   net positive quality score from trackProgress and forwardSpeed.

2. **No positive reward for staying on track**: There is no bonus for maintaining
   centerline position or staying away from borders. The only signal is the absence
   of the penalty.

3. **Clamping prevents catastrophic failure**: `clampCarToTrackBounds` physically
   prevents the car from leaving the track, so there is no natural consequence
   (crashing, stopping) — just a weak -1 reward. The car can "lean" against the
   border indefinitely without significant quality score impact.

#### Why Agents Eventually Go Wrong Direction

1. **Wrong-direction penalty is negligible**: Same -1 × 0.1 = -0.1 quality impact.

2. **No escalating penalty**: A car going the wrong direction for 100 ticks
   receives the same -1 per tick. There is no accumulation or escalation.

3. **Growth mutations can disrupt learned behavior**: With `improvementThreshold=0`
   and `hysteresisWindowCount=0`, mutations are committed freely. A structural
   mutation that happens to produce wrong-direction behavior can be committed if
   the complexity bonus is non-negative, even if driving quality degrades.

4. **The evaluator uses the same scoreHistory for baseline and candidate**: Only
   the complexityBonus differs between the two. The driving quality components
   (scoreMean, scoreTrend) are identical. This means the commit decision is almost
   entirely driven by structural complexity, not driving performance.

#### Why Nodes Are Added Too Frequently

1. **`hysteresisWindowCount: 0`**: No consecutive positive windows required before
   growth. The `canGrowNow()` check always passes (0 >= 0).

2. **`improvementThreshold: 0`**: Any non-negative score delta commits. Since the
   complexityBonus is typically positive for structural mutations (they increase
   forward-pass variance), mutations almost always commit.

3. **Short cooldown (20 ticks = 0.33s)**: After a commit, only 20 simulation ticks
   of cooldown before the next growth attempt. At 60fps, that's ~0.33 seconds.

4. **No weight stabilization check**: There is no mechanism to detect whether
   existing weights have converged or stabilized before adding new structure.
   The system does not check if the network has "learned to use" the existing
   nodes.

5. **No fitness plateau detection**: The system does not detect when quality
   improvement has plateaued. Growth happens on a fixed cadence regardless of
   whether the network is actually improving.

6. **Growth throttle only above 1000 nodes**: Below 1000 nodes, there is no
   throttle at all. The starting network has ~109 nodes (per prior research),
   so growth runs at full speed for a long time.

## Decision

### Recommended Fixes for Issue 1 (Stabilization Period)

1. **Raise `hysteresisWindowCount`** from 0 to at least 3-5 in the lifecycle config
   at `runtime.adaptation.ts:346`. This requires 3-5 consecutive positive-quality
   windows before growth is allowed.

2. **Raise `mutationCooldownTicks`** from 5 to at least 30-60 (0.5-1 second at 60fps)
   in `browser-entry.ts:515`. This gives the network time to stabilize weights after
   a structural change.

3. **Raise `improvementThreshold`** from 0 to at least 0.01-0.05 in
   `browser-entry.ts:513`. This ensures only meaningful improvements commit
   mutations, not noise.

4. **Add a fitness plateau detector**: Before allowing growth, check if the quality
   score has been stable (low variance) over the evidence window for N consecutive
   ticks. Only grow when the network has plateaued — meaning it needs more capacity
   to improve further.

5. **Reduce `MAX_EPISODIC_SLOTS`** from 100 to 10-20 to limit growth rate per cycle.

### Recommended Fixes for Issue 2 (Stronger Reward/Penalty Shaping)

1. **Increase `OFF_TRACK_CLAMP_REWARD`** from -1 to at least -5 to -10 in
   `environment.step.service.ts:36`. This makes border collision a strong negative
   signal.

2. **Increase `WRONG_DIRECTION_REWARD`** from -1 to at least -5 to -10 in
   `environment.step.service.ts:38`. Wrong-direction driving should be strongly
   penalized.

3. **Increase the `physicsReward` weight** in `toDrivingQuality` from 0.1 to at
   least 0.3-0.5 at `runtime.adaptation.ts:709`. This makes physics penalties
   meaningful relative to the positive signals.

4. **Add a positive guide-following reward**: Compute a guide-following bonus in
   the environment or browser-entry that rewards the car for being close to the
   team's lane centerline. Add it to `carState.reward` or as a new
   `RacingQualitySignal` component. Suggested: `+2 to +5` when the car is within
   a small threshold of the guide line.

5. **Add a diverging-from-guide penalty**: When the guide line is available (Tier 1,
   alpha > 0), penalize lateral divergence from the guide line more strongly than
   the generic `offTrackPenalty × 0.1`. Suggested: increase the
   `offTrackPenalty` weight from 0.1 to 0.3-0.5, or add a separate
   `guideDivergencePenalty` component.

6. **Add escalating penalties**: Track consecutive ticks of border contact or
   wrong-direction and escalate the penalty. E.g., `-1 × consecutiveTicks` for
   border contact, capped at some maximum.

## Risks

- **Performance risk**: Increasing cooldown and hysteresis will slow network growth.
  This is the desired behavior but may make the demo less visually dynamic.

- **Tuning risk**: The exact penalty/reward magnitudes need empirical tuning. The
  suggested values are starting points, not final values.

- **Evaluator architecture risk**: The fundamental issue that baseline and candidate
  use the same scoreHistory (only complexityBonus differs) means driving quality
  changes from mutations are not properly evaluated. This is a deeper architectural
  issue beyond just reward shaping.

- **Test breakage**: Changing reward constants will break existing tests that assert
  specific reward values. Tests in `environment.step.test.ts` and
  `runtime.adaptation.test.ts` will need updating.

- **No weight-level learning**: The current system only does structural adaptation
  (adding nodes/edges). There is no backpropagation or weight update mechanism in
  the runtime adaptation loop. Weight "stabilization" in the traditional sense does
  not occur — the network's weights are set at initialization and only change
  through structural mutations. The stabilization period would need to be defined
  as "quality score plateau" rather than "weight convergence."