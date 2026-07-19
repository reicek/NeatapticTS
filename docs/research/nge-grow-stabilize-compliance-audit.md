# NGE Grow-Stabilize Compliance Audit — Brain-Inspired Fidelity & Acceleration Recommendations

**Date:** 2026-07-12  
**Author:** `02-researching` → delegated to `nge-core-scout`  
**Companion artifact:** [`nge-grow-stabilize-boundary-map.md`](./nge-grow-stabilize-boundary-map.md)

## Question

Does the current NGE grow-stabilize implementation in `src/neat/nge-juvenile/` faithfully emulate organic brain growth, satisfy the NGE vision, and safely support a “baby phase” of faster early growth? Which acceleration ideas borrowed from training are core-NGE-compatible, and which should stay as training/performance overlays?

## Evidence

### Code surfaces audited

| Surface | Purpose |
|---|---|
| `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:282-377` | `runNgeGrowStabilizeCycle` orchestrator |
| `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:187-201` | `applyWeightMutations` — random-only perturbation |
| `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:83-91` | Adaptive hysteresis thresholds |
| `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:221-239` | `computeGrowthThrottle` — currently unused by core cycle |
| `src/neat/nge-juvenile/neat.nge-juvenile.grow.ts:196-209` | Growth-signal formula |
| `src/neat/nge-juvenile/neat.nge-juvenile.grow.ts:247-249` | Node-addition count scaling |
| `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:125-210` | Hard-coded capacity, plateau, mutation, throttle constants |
| `src/neat/nge-juvenile/neat.nge-juvenile.apply.ts:119-124` | `slotExpand` no-op — memory tiers not operational |
| `src/neat/nge-juvenile/neat.nge-juvenile.focus.ts:106-138` | Focus scoring inputs |
| `src/neat/neat.nge-lifecycle.ts:262-277` | `restoreNetworkSnapshot` exists in lifecycle |
| `examples/racing_curriculum/controller/runtime.adaptation.ts:394-503,538-607,716-773` | Demo-local commit/rollback, forward-pass scoring |
| `plans/completed/NEAT_Genesis_EvoDevo.md` | Canonical NGE plan: memory architecture, neuromodulation, morphogenesis, phased roadmap |

### Verified facts

1. **Stabilization is random-only weight noise.** `applyWeightMutations` mutates 30 % of connection weights by ±0.1 with uniform random deltas. It does **not** touch biases, use activity signals, or apply reward-gated nudges.  
   Source: `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:187-201`.

2. **Growth signal is reward-history-driven, not activity-driven.** Composite = 0.25·util + 0.3·rewardΔ + 0.2·novelty + 0.15·stabilityAge − 0.1·wiringCost. No per-module activity burst, episodic hit-rate, or recurrent refresh signal enters the calculation.  
   Source: `src/neat/nge-juvenile/neat.nge-juvenile.grow.ts:196-209`.

3. **“Baby phase” is embryonic, not explicit.** Hysteresis shortens to 2 windows below 200 nodes and 3 windows below 500 nodes, but there is no named early-burst policy and the 1,000-node throttle threshold is not driven by a lifecycle stage.  
   Source: `src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts:83-91`, `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:200-205`.

4. **Large-network throttle is dead code in the core cycle.** `computeGrowthThrottle` is exported, but the only caller found is the racing demo (`runtime.adaptation.ts:352`); `runNgeGrowStabilizeCycle` does not invoke it.  
   Source: `Select-String` confirmation across `src/` and `examples/racing_curriculum/`.

5. **Candidate-evaluation / commit / rollback is demo-local.** The core lifecycle exports `restoreNetworkSnapshot`, but the racing demo owns the snapshot, forward-pass scoring, improvement gating, and rollback logic.  
   Source: `examples/racing_curriculum/controller/runtime.adaptation.ts:454-503`, `src/neat/neat.nge-lifecycle.ts:262-277`.

6. **Memory tiers and neuromodulation are present in types/plan but not operational in juvenile growth.** `slotExpand` is a no-op; juvenile focus scoring does not consume episodic/recurrent metrics or neuromodulator signals.  
   Sources: `src/neat/nge-juvenile/neat.nge-juvenile.apply.ts:119-124`, `plans/completed/NEAT_Genesis_EvoDevo.md:191-230`.

7. **Knobs are hard-coded constants.** Plateau window, variance threshold, weight mutation rate/magnitude, stabilization min/max ticks, hysteresis thresholds, cooldown, focus weights, and capacity are all exported constants rather than DNA/config-overridable parameters.  
   Source: `src/neat/nge-juvenile/neat.nge-juvenile.constants.ts:125-210`.

## Decision

The implementation is **structurally on-policy** but **functionally not yet brain-like** in its stabilization and growth-gating.

- **Keep:** edge-first morphogenesis, dry-run rollbackable deltas, budget enforcement, adaptive hysteresis, and the lifecycle stage abstraction.
- **Fix in core (nge-core-algorithm):** make stabilization activity/bias-aware, extract the candidate-evaluation/rollback seam, introduce an explicit configurable baby-phase policy, and make all grow-stabilize knobs DNA/config-overridable.
- **Keep as overlays:** parallel weight evaluation, GPU batch evaluation of candidate variants, racing-specific forward-pass scoring, and other training-style optimizers. They belong in performance/training layers and must not leak into the core lifecycle.

## Prioritized Recommendations

### P0 — Core NGE primitives needed now

| # | Recommendation | Rationale | Owner | Biological analogy |
|---|---|---|---|---|
| 1 | Replace random-only stabilization with activity/bias-aware plasticity. Add bias adjustment and reward-gated nudges; keep random noise only as a small exploratory component. | Current stabilization is the main blocker to brain-like growth and may explain the 82→86 difficulty wall. | `nge-core-algorithm` / `src/neat/nge-juvenile/` | Real neurons tune synaptic weights *and* thresholds; plasticity is activity and neuromodulator-gated (Hebbian / heterosynaptic), not uniform noise. |
| 2 | Extract the candidate-evaluation/rollback seam into core. The core lifecycle should snapshot the network, apply a candidate morph/parameter window, let the caller inject an evaluator, and rollback if not improved. | Safety invariant in the plan; currently duplicated/demo-local, making it hard to test and risky for new demos. | `nge-core-algorithm` / `src/neat/nge-juvenile/` | Developmental mutations are tested against the organism’s current phenotype; unsuccessful changes are discarded before fixation. |
| 3 | Make all grow-stabilize constants DNA/config-overridable (plateau window, variance threshold, weight/bias mutation rate & magnitude, stabilization min/max, hysteresis thresholds/cadence, cooldown, focus weights, capacity). | The plan states wiring-cost weights live in DNA and evolve; hard-coded constants prevent evo-devo adaptation of the growth policy. | `nge-core-algorithm` / types + config | Evolution tunes neurogenesis/synaptogenesis timing in different lineages. |
| 4 | Introduce an explicit `baby → juvenile → adult` lifecycle stage policy with configurable node-count thresholds and distinct growth/stabilization cadences. Default baby burst should target the ~1k-node density band. | User wants faster early growth; biology supports exuberant early neurogenesis followed by pruning/refinement. | `nge-core-algorithm` / lifecycle | Early brains undergo rapid neurogenesis and exuberant synaptogenesis; later stages emphasize pruning and myelination. |

### P1 — Strongly aligned core improvements

| # | Recommendation | Rationale | Owner | Biological analogy |
|---|---|---|---|---|
| 5 | Wire memory-tier signals into juvenile focus scoring (episodic hit-rate, recurrent refresh, short/medium-term usage). Operationalize `slotExpand`. | Plan declares memory architecture as a first-class NGE primitive; currently it is inert in growth decisions. | `nge-core-algorithm` + memory plan `plans/Memory_Optimization.md` | Mushroom bodies and hippocampus gate plasticity by replay/consolidation signals. |
| 6 | Add neuromodulator fast-switching before structural edits. Use a quick gain/learning-rate modulation step to test whether the current structure can already solve the new task. | Plan distinguishes fast behavioral switching from slow structural adaptation; this prevents unnecessary growth. | `nge-core-algorithm` | Neuromodulators (dopamine, acetylcholine) switch behavioral modes in milliseconds before rewiring occurs. |
| 7 | Re-enable `computeGrowthThrottle` inside `runNgeGrowStabilizeCycle` so the large-network cadence is a core policy, not demo-local. | NGE vision explicitly targets 2×256k GPU-scale networks; core must own performance-aware growth cadence. | `nge-core-algorithm` | Metabolic cost of adding neurons rises with brain size; larger brains slow structural change. |

### P2 — Performance/training overlays (not core)

| # | Recommendation | Rationale | Owner |
|---|---|---|---|
| 8 | Build a reusable “evaluate N parameter-variants of one topology in a single GPU/worker dispatch” primitive. | Accelerates early baby-phase trial-and-error; environment-agnostic if it accepts a generic evaluator. | `performance-optimization` / `worker-inference-transport` |
| 9 | Provide an optional training-style weight/bias fine-tune overlay (e.g., `fineTuneVector`-style detached clone tuning) that the demo can invoke between growth windows. | Borrowing from training is safe because it mutates a detached parameter vector, preserving the “DNA encodes structure, not weights” invariant. | training utilities / demo adapter |
| 10 | Auto-detect or recommend GPU activation based on network size, and surface a fallback notification when WebGPU is unavailable. | Supports the NGE dense-network vision without forcing GPU on all users. | browser-runtime-scout / GPU layer |

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Demo continues to own commit/rollback safety logic, so core invariants remain untested and future demos diverge. | High | P0 #2 — extract the rollback seam into core with deterministic hash/candidate tests. |
| Random-only stabilization plateaus on tasks where small perturbations cannot discover needed representational change. | High | P0 #1 — activity/bias-aware plasticity + optional neuromodulator pre-step. |
| Hard-coded constants prevent DNA-driven adaptation of the growth policy, contradicting plan intent. | Medium | P0 #3 — config-overridable/DNA-backed parameters. |
| Large-network throttle is unused in core, risking performance collapse at 2×256k scale. | Medium | P1 #7 — integrate `computeGrowthThrottle` into the core cycle. |
| Blurring core vs. training responsibilities could pollute classic NEAT with NGE-only assumptions. | Medium | Keep parallel weight evaluation and fine-tune overlays outside `nge-core-algorithm`; use opt-in adapters. |
| `slotExpand`/memory-tier work depends on `plans/Memory_Optimization.md`; premature changes may conflict with memory architecture. | Low | Read `plans/Memory_Optimization.md` before P1 #5 implementation. |

## Handoff

Route to **`nge-core-algorithm`** for Phase B juvenile stabilization and the candidate-evaluation seam. Primary canonical plan: `plans/completed/NEAT_Genesis_EvoDevo.md` (Phase B L713-719, Phase C L720-725, Morphogenesis L574-590, Memory Architecture L191-209, Neuromodulation L211-230). Read `plans/Memory_Optimization.md` before any memory-tier or `slotExpand` change.
