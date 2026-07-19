# Memory-Tier Signal Availability for Juvenile Focus Scoring

## Question

Can memory-tier signals (episodic hit-rate, recurrent refresh, short/medium-term usage) be wired into juvenile focus scoring today, or should this work be deferred?

Originating plan step: `plans/NGE_Grow_Stabilize_Cycle.plans.md` Phase 3 Step 02.

## Evidence

### Source 1: NgeModuleMetricsSnapshot — the focus scoring input (static code, high authority)

`src/neat/nge-juvenile/neat.nge-juvenile.types.ts` lines 24–37:

The metrics snapshot consumed by `computeFocusScores` has exactly five fields:
`moduleId`, `utilization`, `rewardDelta`, `novelty`, `stabilityAge`, `wiringCost`.

**None of these are memory-tier signals.** There is no episodic hit-rate field, no
recurrent refresh field, and no short/medium-term usage field on the snapshot type.

### Source 2: Focus scoring formula — no memory-tier terms (static code, high authority)

`src/neat/nge-juvenile/neat.nge-juvenile.focus.ts` lines 196–201:

```ts
const rawScore =
  focusWeights.w_u * normalizedUtilization +
  focusWeights.w_r * normalizedRewardDelta +
  focusWeights.w_n * normalizedNovelty +
  focusWeights.w_s * normalizedStabilityAge -
  focusWeights.w_c * normalizedWiringCost;
```

The weighted formula operates exclusively on the five snapshot metrics. No
memory-tier signal participates in the raw score computation.

### Source 3: Episodic hit-rate is proxied, not measured (static code + comment, high authority)

`src/neat/nge-juvenile/neat.nge-juvenile.grow.ts` lines 133–136 and 491–499:

The `planSlotExpansion` planner accepts a `hitRate` parameter and gates on
`config.episodicHitRateThreshold` (default 0.65). However, the caller
(`planGrowthMorphs`) passes `metrics.utilization` as the hit-rate proxy:

```ts
const slotDelta = planSlotExpansion(
  moduleId,
  metrics.utilization, // ← proxy, not a real episodic hit-rate signal
  budget,
  focusScore,
  config,
);
```

The code explicitly documents this: *"This planner uses `metrics.utilization`
as the episodic hit-rate proxy until a dedicated hit-rate metric is added to
the module snapshot."*

There is no runtime telemetry that produces a real episodic recall hit-rate.

### Source 4: recurrentRefreshFloor is declared but never consumed (static code, high authority)

`recurrentRefreshFloor` (default 0.3) is resolved in `NgeJuvenilePhaseConfig`
via `resolveFocusConfig` in `focus.ts` lines 50–52, but a codebase-wide search
shows it is **never read by any runtime logic** — only tested for resolution
and echoed in test fixtures. It is a declared config field with no consumer.

There is no runtime signal source that measures recurrent hidden-state refresh.

### Source 5: Short/medium-term usage does not exist in juvenile code (static code, high authority)

A search across all `src/neat/nge-juvenile/*.ts` files for "short-term",
"medium-term", "shortTerm", "mediumTerm", or "memoryTier" returned zero hits
inside the juvenile module. The only `memoryTier` references in `src/neat/` are
in the assimilation layer (`nge-assimilation`) and evolution distance
(`nge-evolution`), which operate on DNA genotype deltas — not on runtime
per-module metrics.

The assimilation delta type (`NgeAssimilationDelta.memoryTier`) has
`hiddenDim`, `slotCount`, and `decayRate` fields, but these are evolutionary
target values for DNA writeback, not live runtime telemetry signals that could
feed focus scoring.

### Source 6: slotExpand apply is a no-op (static code, high authority)

`src/neat/nge-juvenile/neat.nge-juvenile.apply.ts` lines 119–124:

```ts
case 'slotExpand':
  return {
    kind: 'slotExpand',
    status: 'skipped',
    reason: 'No NEAT mutation equivalent for episodic slot expansion.',
  };
```

Even if memory-tier signals were wired into focus scoring, the `slotExpand`
morph delta — the primary consumer of episodic hit-rate — is currently a
no-op on apply. Operationalizing it would require defining a NEAT mutation
equivalent for episodic slot expansion, which is a separate design question.

### Source 7: Memory_Optimization.md is about low-level memory management, not cognitive memory tiers (documentation, medium authority)

`plans/completed/Memory_Optimization.md` covers centralized memory ownership,
buffer pooling, sparsity governance, adaptive precision, compression, and
activation windows — all low-level memory management for very large networks.
It does not address cognitive memory tiers (episodic/short-term/medium-term)
or any runtime signal infrastructure for per-module memory usage tracking.

## Decision

**DEFER.** Wiring memory-tier signals into juvenile focus scoring is not
feasible today. The three required signals do not exist as runtime telemetry:

1. **Episodic hit-rate** — only exists as a config threshold
   (`episodicHitRateThreshold`); the actual value is proxied by
   `metrics.utilization`. No runtime measurement infrastructure produces a
   real episodic recall hit-rate per module.

2. **Recurrent refresh** — only exists as a config floor
   (`recurrentRefreshFloor`) that is declared but never consumed by any
   runtime logic. No signal source measures recurrent hidden-state refresh.

3. **Short/medium-term usage** — does not exist anywhere in the juvenile
   codebase or its types. The assimilation layer's `memoryTier` fields are
   evolutionary DNA deltas, not runtime telemetry.

To make this feasible, a future workstream would need to:
- Extend `NgeModuleMetricsSnapshot` with dedicated memory-tier fields.
- Build a runtime telemetry layer that tracks per-module episodic recall,
  recurrent refresh rates, and short/medium-term usage counts.
- Define the NEAT mutation equivalent for `slotExpand` so it is no longer a
  no-op on apply.
- Add focus weights for the new memory-tier signals to `NgeJuvenileFocusWeights`.

This is a multi-step infrastructure effort that exceeds the scope of a single
A4 slice and should be planned as a separate workstream when runtime memory-tier
telemetry is prioritized.

### Rejected alternatives

- **Wiring with proxy signals only** — extending focus weights to consume
  `utilization` under a new name would be cosmetic and would not deliver the
  cognitive-memory-informed scoring the plan intends. Rejected as providing
  no real behavioral change.

## Risks

- **Deferral gap**: The A4 slice (memory-tier focus wiring) is skipped, which
  means focus scoring continues to use the 5-metric formula without memory-tier
  awareness. This is the status quo and carries no regression risk, but the
  brain-like stabilization vision remains partially unrealized.
  **Owner**: Future workstream for runtime memory-tier telemetry.

- **slotExpand remains a no-op**: Even with the grow planner producing
  `slotExpand` deltas, the apply layer skips them. This is documented and
  tested as current behavior, so no silent regression exists.
  **Owner**: Future workstream that defines the NEAT mutation equivalent
  for episodic slot expansion.

- **recurrentRefreshFloor unused config**: The config field exists but has no
  consumer. It should be retained as a forward-compatible declaration for when
  runtime telemetry is added, or removed if the API surface is cleaned up.
  **Owner**: Future workstream or a cleanup slice.