/**
 * Juvenile phase orchestration surface for the NGE lifecycle.
 *
 * The juvenile stage is the real-time growth engine of the Neuro-evolutionary
 * Genesis Engine (NGE). It looks at one module at a time, scores how promising
 * that module is, and plans small structural edits — denser edges, extra nodes,
 * or wider episodic slots — that the lifecycle can commit during a single
 * evaluation window. Growth is continuous: it does not wait for a generation
 * boundary, a breeding cycle, or any example-side scaffolding. Generations are
 * for multiplying and fusing successful networks, not a prerequisite for an
 * agent to grow.
 *
 * This boundary exists so the policy that decides *where* to grow (focus
 * scoring, hysteresis, cooldowns, and budgets) stays separate from the lower
 * level structural mutations that actually change the network. That separation
 * lets the same engine run inside a racing curriculum, an ant hive, a predator
 * simulation, or a headless unit test with no dependency on `examples/` or demo
 * code.
 *
 * ## The juvenile growth contract
 *
 * One lifecycle window follows a strict pipeline:
 *
 * 1. **Collect metrics.** The caller supplies one {@link NgeModuleMetricsSnapshot}
 *    per module: utilization, reward delta, novelty, stability age, and wiring
 *    cost.
 * 2. **Score focus.** {@link computeFocusScores} normalizes each metric column
 *    independently, folds in the configured {@link NgeJuvenileFocusWeights},
 *    and emits a probability-like focus vector via softmax normalization.
 * 3. **Plan dry-run morphs.** {@link planGrowthMorphs} produces a sorted list of
 *    {@link NgeMorphDelta} objects in edge-first priority order: edge densify,
 *    slot expand, then node add. Every plan is checked against the DNA
 *    {@link NgeGrowthBudget}.
 * 4. **Re-validate and mutate.** {@link applyMorphDeltas} translates each delta
 *    into the matching NEAT mutation operator, re-checks the live budget, and
 *    reports the outcome truthfully as `applied` or `skipped`. A saturated graph
 *    or a conflicting sparsity budget no longer produces a false-positive
 *    applied report.
 * 5. **Commit hysteresis.** When at least one growth morph is genuinely applied,
 *    {@link commitGrowth} resets the positive-focus streak and starts the
 *    cooldown timer so growth stays bursty rather than noisy.
 *
 * ```mermaid
 * flowchart LR
 *   Metrics["Module metrics"] --> Focus["computeFocusScores"]
 *   Focus --> Plan["planGrowthMorphs"]
 *   Plan --> Hyst{"Hysteresis gate\n& budget check"}
 *   Hyst -->|open| Apply["applyMorphDeltas"]
 *   Hyst -->|closed| Skip["Skip this window"]
 *   Apply --> Truth["Truthful outcome:\napplied / skipped"]
 *   Truth -->|applied| Commit["commitGrowth"]
 *   Truth -->|skipped| Keep["Keep hysteresis"]
 * ```
 *
 * ## Tuning knobs
 *
 * Most callers can use the seeded defaults. The constants below are the levers
 * you actually touch when the default growth personality is too aggressive or
 * too conservative.
 *
 * | Constant | What it controls | Default | When to change |
 * |---|---|---|---|
 * | {@link NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS} | Relative weight of utilization, reward, novelty, stability, and cost in the focus score | `w_u=0.25, w_r=0.3, w_n=0.2, w_s=0.15, w_c=0.1` | Increase `w_u` when underused modules should grow faster; increase `w_c` to penalize wiring. |
 * | {@link NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT} | Consecutive positive-focus windows required before growth can commit | `2` | Raise to reduce noise, lower to speed up response. |
 * | {@link NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT} | Forward edges added by one committed edge-densify step | `5` | Raise for faster saturation escape, lower for fine-grained growth. |
 * | {@link NGE_JUVENILE_DEFAULT_NODE_ADDITION_COUNT} | Hidden nodes inserted by one committed node-add step | `2` | Raise to break past local plateaus, lower to keep networks compact. |
 * | {@link NGE_JUVENILE_DEFAULT_NODE_GROWTH_SIGNAL_FLOOR} | Minimum composite growth signal that opens the node-add gate | `0.0` | Raise to make node addition rarer and more evidence-gated. |
 * | {@link NGE_MAX_NODE_CAPACITY} | Absolute node ceiling enforced by the growth budget | `8000` | Match to the memory/performance envelope of your runtime. |
 * | {@link NGE_MAX_EDGE_CAPACITY} | Absolute edge ceiling enforced by the growth budget | `32000` | Match to the memory/performance envelope of your runtime. |
 *
 * ## Determinism boundary
 *
 * The pipeline is deterministic for a fixed DNA, fixed seed, and fixed
 * experience stream. {@link runNgeLifecycle} seeds the network RNG before
 * morph application and pins the global connection innovation counter to the
 * network's current maximum innovation, so repeated runs produce the same edge
 * choices and innovation IDs. The `computedAt` timestamp in
 * {@link NgeFocusVector} is metadata only and must never participate in a
 * deterministic replay fingerprint.
 *
 * ## Background reading
 *
 * - NEAT itself: Kenneth O. Stanley and Risto Miikkulainen,
 *   [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/?stanley:ec02) (2002).
 * - Softmax normalization:
 *   [Wikipedia — Softmax function](https://en.wikipedia.org/wiki/Softmax_function).
 * - Feature scaling / min-max normalization:
 *   [Wikipedia — Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling).
 *
 * @example
 * Dry-run the focus scorer and growth planner for one window.
 * ```ts
 * import { nge } from 'neataptic';
 *
 * const metrics = {
 *   moduleId: 'module:alpha',
 *   utilization: 0.8,
 *   rewardDelta: 0.4,
 *   novelty: 0.2,
 *   stabilityAge: 5,
 *   wiringCost: 0.1,
 * };
 * const focus = nge.juvenile.computeFocusScores([metrics], {});
 * const config = nge.juvenile.resolveFocusConfig({});
 * const deltas = nge.juvenile.planGrowthMorphs(
 *   'module:alpha',
 *   focus.scores[0],
 *   metrics,
 *   { maxNodes: 8000, maxEdges: 32000, maxEpisodicSlots: 100, currentNodeCount: 10, currentEdgeCount: 20, currentEpisodicSlotCount: 0 },
 *   config,
 *   { growthPositiveWindowCount: 2, pruneUnderuseWindowCount: 0, lastMorphKind: 'none', cooldownWindowsRemaining: 0 },
 * );
 * ```
 *
 * @example
 * Apply planned growth to a live network with a deterministic seed.
 * ```ts
 * import { nge, Network } from 'neataptic';
 *
 * const network = new Network(2, 1, { seed: 42 });
 * const budget = {
 *   growth: { maxNodes: 8000, maxEdges: 32000, maxEpisodicSlots: 100, currentNodeCount: network.nodes.length, currentEdgeCount: network.connections.length, currentEpisodicSlotCount: 0 },
 *   prune: { minNodes: 1, minEdges: 1, costExemptEdgeIds: [], currentEdgeCount: network.connections.length, currentNodeCount: network.nodes.length, currentWiringCost: 0 },
 * };
 * const outcomes = nge.juvenile.applyMorphDeltas(network, deltas, budget);
 * ```
 */

export * from './neat.nge-juvenile.apply';
export * from './neat.nge-juvenile.constants';
export * from './neat.nge-juvenile.errors';
export * from './neat.nge-juvenile.focus';
export * from './neat.nge-juvenile.grow';
export * from './neat.nge-juvenile.probe';
export * from './neat.nge-juvenile.prune';
export * from './neat.nge-juvenile.types';
export * from './neat.nge-juvenile.utils';
