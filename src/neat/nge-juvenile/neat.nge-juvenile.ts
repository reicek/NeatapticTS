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
 * lets the same engine run inside an application curriculum, a collective
 * simulation, an agent-based scenario, or a headless unit test with no dependency
 * on `examples/` or demo code.
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
 * ## The grow-stabilize cycle
 *
 * Above the single-window growth pipeline sits the
 * {@link runNgeGrowStabilizeCycle} orchestrator, which sequences repeated
 * adaptation ticks. Each tick decides whether the network should **grow**
 * (add structural capacity via the lifecycle) or **stabilize** (tune existing
 * weights to exploit current capacity). The decision is driven by plateau
 * detection: when the rolling quality-score variance falls below a threshold,
 * the network has learned to use its current structure and further growth is
 * permitted.
 *
 * This separation keeps the engine from growing indiscriminately. A network
 * that adds structure every tick never learns to use what it already has. By
 * alternating growth with stabilization, the cycle lets the network consolidate
 * each structural investment before the next expansion, mirroring the
 * explore–exploit tradeoff common in reinforcement learning and
 * neuro-evolution.
 *
 * Key pure decision functions exported from the grow-stabilize module:
 *
 * - {@link isPlateauReached} — variance-based plateau detection with
 *   time-boxed min/max stabilization guards.
 * - {@link resolveAdaptiveHysteresis} — network-size-aware hysteresis window
 *   count that accelerates early growth and requires more sustained evidence
 *   at scale.
 * - {@link applyWeightMutations} — stochastic weight perturbation during the
 *   stabilization phase.
 * - {@link computeGrowthThrottle} — progressive back-off for large networks
 *   to preserve real-time performance.
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> PlateauCheck
 *   PlateauCheck --> Stabilization : not plateaued
 *   PlateauCheck --> Growth : plateaued or first growth
 *   Stabilization --> [*] : weight mutations applied
 *   Growth --> [*] : lifecycle morphs applied
 * ```
 *
 * ## Weight-exhaustion gate and variant scaling
 *
 * The grow-stabilize cycle does not rely on a fixed improvement threshold. When
 * weight mutation is active, the cycle compares the best weight variant against
 * an adaptive **exhaustion** bar that rises as the network ages. If no variant
 * beats the bar for too many consecutive ticks, the cycle treats structural growth
 * as the better investment and switches back to the lifecycle pipeline. The bar
 * is stage-aware: babies tolerate larger jumps, adults require finer evidence.
 *
 * Three additional signals shape the bar so it cannot be gamed by scale alone:
 *
 * - **Neuron-budget factor** — small networks are nudged toward structural
 *   growth because they have headroom; large networks face a tighter bar.
 * - **Noise-multiplier cap** — the statistical uplift from evaluating many
 *   variants is bounded so huge variant counts do not drown out real signal.
 * - **Score-ceiling vs. magnitude scaling** — when a known ceiling exists the
 *   threshold measures remaining headroom; otherwise it scales with the absolute
 *   score magnitude.
 *
 * After a bad growth event the bar doubles briefly (the post-growth anti-runaway
 * boost) so the cycle does not over-tune weights while ignoring the structural
 * mistake. The boost is capped and time-boxed so growth is never deferred forever.
 *
 * Variant patches scale their effective magnitude with lifecycle stage and
 * network size. A small baby network explores a wide symmetric range around each
 * weight, while a large adult network shrinks its perturbations so tuning stays
 * local. The scaling is bounded by clamps so very large variant counts or very
 * large connection counts cannot explode or collapse the step size.
 *
 * The exported helpers {@link resolveEffectiveMagnitude} and
 * {@link resolveRepresentativeDelta} materialize these scaled perturbations for
 * downstream weight mutation, while {@link resolveExhaustionImprovementThreshold}
 * and {@link resolveExhaustionForceGrowthThreshold} compute the adaptive
 * improvement bar and the structural-growth fallback threshold.
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
 * | {@link NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE} | Rolling window length for plateau detection | `5` | Raise for smoother plateau detection, lower for faster response. |
 * | {@link NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD} | Variance below which quality is considered plateaued | `0.1` | Raise to trigger growth sooner, lower to require tighter convergence. |
 * | {@link NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP} | Max structural edits per lifecycle call | `5` | Raise for batch growth, lower for fine-grained morphs. |
 * | {@link NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS} | Min ticks after growth before plateau can fire | `5` | Raise to give more learning time, lower for faster cycling. |
 * | {@link NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS} | Max ticks before growth is forced regardless of plateau | `25` | Raise to allow longer stabilization, lower to force growth sooner. |
 * | {@link NGE_EXHAUSTION_STAGE_FRACTION_BABY} | Relative improvement bar for weight variants in the baby stage | `0.02` | Lower to let baby networks commit smaller weight wins; raise to demand stronger evidence. |
 * | {@link NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE} | Relative improvement bar for weight variants in the juvenile stage | `0.01` | Raise to demand stronger variants, lower to commit smaller improvements. |
 * | {@link NGE_EXHAUSTION_STAGE_FRACTION_ADULT} | Relative improvement bar for weight variants in the adult stage | `0.006` | Raise to demand stronger variants; adult tuning is intentionally picky. |
 * | {@link NGE_EXHAUSTION_TICK_BUDGET} | Total exhaustion tick budget before structural growth is forced | `48` | Raise to give weight tuning more total ticks; the per-variant limit is `ceil(tickBudget / variantCount)`. |
 * | {@link NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS} | Floor on consecutive exhaustion ticks before forcing growth | `1` | Raise to prevent immediate growth fallback after a single bad variant tick. |
 * | {@link NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS} | Consecutive failed variant ticks before forcing structural growth | `8` | Raise to allow more tuning attempts, lower to switch to growth sooner. |
 * | {@link NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST} | Multiplier applied to the exhaustion limit after a bad growth event | `2.0` | Lower to reduce the post-growth anti-runaway back-off. |
 * | {@link NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS} | Same cap after a bad growth event triggered the exhaustion boost | `16` | Keeps the doubled back-off from deferring weight tuning forever. |
 * | {@link NGE_EXHAUSTION_NEURON_BUDGET_FACTOR} | Half-range multiplier for the neuron-budget factor | `0.5` | Raise to push small networks toward growth faster; zero disables the bias. |
 * | {@link NGE_EXHAUSTION_NOISE_MULTIPLIER_CAP} | Upper bound on the variant-count noise uplift | `2.0` | Lower to make huge variant counts less forgiving of noise. |
 * | {@link NGE_EXHAUSTION_SCORE_EPSILON} | Minimum absolute scale for the adaptive threshold | `1e-6` | Raise when very small scores need a larger minimum improvement bar. |
 * | {@link NGE_GROW_STABILIZE_BIAS_MUTATION_RATE} | Fraction of biases perturbed during stabilization | `0.3` | Raise for more aggressive bias exploration; lower for conservative tuning. |
 * | {@link NGE_GROW_STABILIZE_BIAS_MUTATION_MAGNITUDE} | Maximum bias perturbation during stabilization | `0.1` | Raise for larger bias steps; lower for fine-grained bias tuning. |
 * | {@link NGE_GROW_STABILIZE_MUTATION_COOLDOWN_TICKS} | Cooldown after a committed mutation attempt | `5` | Raise to make adaptation sparser; lower for faster response. |
 * | {@link NGE_GROW_STABILIZE_ROLLBACK_COOLDOWN_TICKS} | Cooldown after a rollback outcome | `5` | Raise to throttle retries after a rejected candidate. |
 * | {@link NGE_VARIANT_WIDTH_FACTOR_MAX} | Upper clamp on stage-driven variant-count magnitude scaling | `1.5` | Lower to cap exploration width for very large variant counts. |
 * | {@link NGE_VARIANT_SIZE_FACTOR_FLOOR} | Lower clamp on network-size magnitude scaling | `0.1` | Raise to prevent tiny perturbations in very large networks. |
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
 * - Hysteresis in control systems, which inspires the adaptive hysteresis
 *   window counts in the grow-stabilize cycle:
 *   [Wikipedia — Hysteresis](https://en.wikipedia.org/wiki/Hysteresis).
 * - The explore–exploit tradeoff that the grow-stabilize cycle mirrors:
 *   [Wikipedia — Exploration vs exploitation](https://en.wikipedia.org/wiki/Exploration_exploitation_dilemma).
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
export * from './neat.nge-juvenile.grow-stabilize';
export * from './neat.nge-juvenile.probe';
export * from './neat.nge-juvenile.prune';
export * from './neat.nge-juvenile.types';
export * from './neat.nge-juvenile.utils';
