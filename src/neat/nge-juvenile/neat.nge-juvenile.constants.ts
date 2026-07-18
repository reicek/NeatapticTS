import type {
  NgeJuvenileFocusWeights,
  NgeProbeKind,
} from './neat.nge-juvenile.types';

/**
 * Default focus weights used by the juvenile scorer.
 * Drives the weighted formula that ranks modules by utilization, reward, novelty, stability, and cost.
 */
export const NGE_JUVENILE_DEFAULT_FOCUS_WEIGHTS: NgeJuvenileFocusWeights = {
  w_u: 0.25,
  w_r: 0.3,
  w_n: 0.2,
  w_s: 0.15,
  w_c: 0.1,
};

/**
 * Hit-rate floor required before episodic slot growth becomes eligible for commit.
 * Slot expansion is blocked when a module's episodic recall rate falls at or below this value.
 */
export const NGE_JUVENILE_DEFAULT_EPISODIC_HIT_RATE_THRESHOLD = 0.65;

/**
 * Hidden-state refresh floor below which recurrent state becomes prune evidence.
 */
export const NGE_JUVENILE_DEFAULT_RECURRENT_REFRESH_FLOOR = 0.3;

/**
 * Consecutive evaluation windows required before a growth or prune action may commit.
 * Prevents premature structural changes caused by transient evaluation signal spikes.
 */
export const NGE_JUVENILE_DEFAULT_HYSTERESIS_WINDOW_COUNT = 2;

/**
 * Rolling-window length used for neuromodulator gain stabilization checks in the juvenile phase.
 * The mean gain is averaged over this many windows before stability is evaluated.
 */
export const NGE_JUVENILE_DEFAULT_GAIN_STABILITY_WINDOW = 5;

/**
 * Allowed mean-gain deviation before one module is treated as unstable.
 */
export const NGE_JUVENILE_DEFAULT_GAIN_STABILITY_TOLERANCE = 0.05;

/**
 * Probe scheduler cadence floor keeping expensive perturbations sparse across training epochs.
 * At least this many epochs must elapse between successive juvenile probe executions.
 */
export const NGE_JUVENILE_DEFAULT_PROBE_CADENCE_EPOCHS = 10;

/**
 * Default deterministic probe-kind rotation applied by the juvenile perturbation scheduler.
 * Each probe epoch advances the rotation index to cycle through lesion, noise, and gating kinds.
 */
export const NGE_JUVENILE_DEFAULT_PROBE_KINDS: readonly NgeProbeKind[] = [
  'lesion',
  'noise',
  'gating',
];

/**
 * Maximum number of append-only probe ledger entries preserved per episode for analysis.
 * Older entries are evicted in FIFO order when the ledger reaches this cap.
 */
export const NGE_JUVENILE_DEFAULT_PROBE_MAX_LEDGER_ENTRIES = 500;

/**
 * Default lesion severity where `1.0` suppresses the full target edge set.
 */
export const NGE_JUVENILE_DEFAULT_LESION_SEVERITY = 1.0;

/**
 * Default Gaussian standard deviation applied to module activations during noise probes.
 * Smaller values produce fine-grained perturbations; larger values create more disruptive noise.
 * See [Normal distribution (Wikipedia)](https://en.wikipedia.org/wiki/Normal_distribution)
 * for background on the bell-curve noise model.
 */
export const NGE_JUVENILE_DEFAULT_NOISE_SIGMA = 0.05;

/**
 * Default Euclidean edge-length threshold targeted by juvenile gating probes.
 * Edges shorter than this value are de-prioritized when selecting gating perturbation targets.
 */
export const NGE_JUVENILE_DEFAULT_GATING_EDGE_LENGTH_THRESHOLD = 2.0;

/**
 * Minimum viable edge increment applied by one approved juvenile densification step.
 * Each committed grow pass adds at least this many edges to the target module.
 */
export const NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT = 5;

/**
 * Minimum viable episodic slot increment applied by one expansion step.
 */
export const NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT = 1;

/**
 * Floor below which the composite node-growth signal cannot open the node-add gate.
 * The signal is derived from the focus-weighted module metrics, so growth evidence is
 * no longer tied to raw rewardDelta alone.
 */
export const NGE_JUVENILE_DEFAULT_NODE_GROWTH_SIGNAL_FLOOR = 0.0;

/**
 * Number of hidden nodes one approved node-addition step plans to insert.
 */
export const NGE_JUVENILE_DEFAULT_NODE_ADDITION_COUNT = 2;

/**
 * Cost-pressure fraction above which one window counts as prune evidence.
 */
export const NGE_JUVENILE_DEFAULT_PRUNE_COST_PRESSURE_THRESHOLD = 0.85;

/**
 * Safe absolute minimum edge floor when DNA supplies no tighter prune bound.
 */
export const NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR = 1;

/**
 * Maximum node capacity that the NGE growth budget supports.
 * Caps the total number of nodes a network may grow to during runtime adaptation.
 * Used by lifecycle runners and callers that need an explicit 8,000-node ceiling.
 *
 * Contract: NGE_MAX_NODE_CAPACITY=8_000
 */
export const NGE_MAX_NODE_CAPACITY = 8_000;

/**
 * Maximum edge capacity that the NGE growth budget supports.
 * Caps the total number of connections a network may grow to during runtime adaptation.
 * Used by lifecycle runners and callers that need an explicit 32,000-edge ceiling.
 *
 * Contract: NGE_MAX_EDGE_CAPACITY=32_000
 */
export const NGE_MAX_EDGE_CAPACITY = 32_000;

// ──────────────────────────────────────────────────────────────────────
// Grow-Stabilize Cycle Constants
// ──────────────────────────────────────────────────────────────────────

/**
 * Maximum number of quality-score entries retained for plateau detection.
 * The rolling window tracks the baseline score at each adaptation tick to
 * determine whether the network has stabilized before allowing growth.
 *
 * Contract: NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE=5
 */
export const NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE = 5;

/**
 * Variance threshold below which the quality score is considered plateaued.
 * When the rolling-window variance falls below this value, the network is
 * deemed to have learned to use its current structure and further growth
 * is permitted.
 *
 * Contract: NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD=0.1
 */
export const NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD = 0.1;

/**
 * Fraction of connections whose weights are perturbed during each
 * stabilization-phase adaptation tick.
 *
 * Contract: NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE=0.3
 */
export const NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE = 0.3;

/**
 * Maximum magnitude of weight perturbation applied during stabilization.
 * Each selected connection's weight is shifted by a random value in
 * [-MAGNITUDE, +MAGNITUDE].
 *
 * Contract: NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE=0.1
 */
export const NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE = 0.1;

/**
 * Minimum stabilization ticks that must elapse after structural growth
 * before plateau detection can fire.
 *
 * Contract: NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS=5
 */
export const NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS = 5;

/**
 * Maximum stabilization ticks after which growth is forced to re-enter
 * even if the quality score has not plateaued.
 *
 * Contract: NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS=25
 */
export const NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS = 25;

/**
 * Maximum number of episodic growth slots the NGE lifecycle may allocate.
 *
 * Contract: NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS=15
 */
export const NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS = 15;

/**
 * Node count above which the growth throttle engages.
 * Networks exceeding this threshold get progressively longer back-off intervals.
 *
 * Contract: NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD=1_000
 */
export const NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD = 1_000;

/**
 * Base throttle interval (in ticks) applied when the network exceeds the
 * large-network threshold.
 *
 * Contract: NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS=3
 */
export const NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS = 3;

/**
 * Maximum number of sample observations drawn from the score history for
 * forward-pass evaluation.
 *
 * Contract: NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES=5
 */
export const NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES = 5;

/**
 * Default maximum number of structural edits per lifecycle call.
 * Enables batch growth so multiple morphs can commit in a single tick.
 *
 * Contract: NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP=5
 */
export const NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP = 5;

/**
 * Default module identifier used by the grow-stabilize cycle when no
 * custom module ID is supplied.
 */
export const NGE_GROW_STABILIZE_DEFAULT_MODULE_ID = 'nge:runtime';

/**
 * Fraction of biases perturbed during each grow-stabilize stabilization tick.
 *
 * Contract: NGE_GROW_STABILIZE_BIAS_MUTATION_RATE=0.3
 */
export const NGE_GROW_STABILIZE_BIAS_MUTATION_RATE = 0.3;

/**
 * Maximum magnitude of bias perturbation applied during grow-stabilize
 * stabilization. Each selected bias is shifted by a random value in
 * [-MAGNITUDE, +MAGNITUDE].
 *
 * Contract: NGE_GROW_STABILIZE_BIAS_MUTATION_MAGNITUDE=0.1
 */
export const NGE_GROW_STABILIZE_BIAS_MUTATION_MAGNITUDE = 0.1;

/**
 * Number of consecutive lifecycle windows used for grow-stabilize cooldown
 * gating. A morph action may not commit again until this many windows have
 * elapsed.
 *
 * Contract: NGE_GROW_STABILIZE_LIFECYCLE_COOLDOWN_WINDOW_COUNT=5
 */
export const NGE_GROW_STABILIZE_LIFECYCLE_COOLDOWN_WINDOW_COUNT = 5;

/**
 * Cooldown ticks between consecutive weight mutations in the grow-stabilize
 * cycle. Prevents over-tuning within a single stabilization window.
 *
 * Contract: NGE_GROW_STABILIZE_MUTATION_COOLDOWN_TICKS=10
 */
export const NGE_GROW_STABILIZE_MUTATION_COOLDOWN_TICKS = 10;

/**
 * Cooldown ticks between structural rollbacks in the grow-stabilize cycle.
 *
 * Contract: NGE_GROW_STABILIZE_ROLLBACK_COOLDOWN_TICKS=3
 */
export const NGE_GROW_STABILIZE_ROLLBACK_COOLDOWN_TICKS = 3;

/**
 * Number of parallel weight variants evaluated during the stabilization phase.
 *
 * The racing demo's acceleration config uses 1024 variants for growth-oriented
 * evaluation, but stabilization is a local hill-climb around the current weights
 * and does not need that resolution. Capping stabilization variants at this count
 * preserves real-time frame budget without changing the growth-phase variant
 * budget.
 *
 * Contract: NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT=32
 */
export const NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT = 32;

/**
 * Number of consecutive failed stabilization ticks after which the grow-stabilize
 * cycle skips weight tuning and forces a structural growth attempt. This prevents
 * the network from remaining stuck in a local weight basin when the score is
 * no longer improving.
 *
 * Contract: NGE_GROW_STABILIZE_FORCE_GROWTH_AFTER_FAILED_STABILIZATIONS=3
 */
export const NGE_GROW_STABILIZE_FORCE_GROWTH_AFTER_FAILED_STABILIZATIONS = 3;

// ──────────────────────────────────────────────────────────────────────
// Lifecycle Stage Defaults
// ──────────────────────────────────────────────────────────────────────

/**
 * Default number of weight variants evaluated in the baby lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT=16
 */
export const NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT = 16;

/**
 * Default number of weight variants evaluated in the juvenile lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT=8
 */
export const NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT = 8;

/**
 * Default number of weight variants evaluated in the adult lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT=2
 */
export const NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT = 2;

/**
 * Default node-count threshold that separates the baby lifecycle stage from
 * the juvenile stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD=1_000
 */
export const NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD = 1_000;

/**
 * Default node-count threshold that separates the juvenile lifecycle stage
 * from the adult stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD=4_000
 */
export const NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD = 4_000;

/**
 * Default growth cadence for the baby lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE=0.8
 */
export const NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE = 0.8;

/**
 * Default weight-mutation magnitude for the baby lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE=0.15
 */
export const NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE = 0.15;

/**
 * Default stabilization intensity for the baby lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY=0.3
 */
export const NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY = 0.3;

/**
 * Default growth cadence for the adult lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE=0.2
 */
export const NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE = 0.2;

/**
 * Default weight-mutation magnitude for the adult lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE=0.05
 */
export const NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE = 0.05;

/**
 * Default stabilization intensity for the adult lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY=0.7
 */
export const NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY = 0.7;

/**
 * Default weight-mutation magnitude for the juvenile lifecycle stage.
 *
 * Contract: NGE_LIFECYCLE_DEFAULT_JUVENILE_MUTATION_MAGNITUDE=0.1
 */
export const NGE_LIFECYCLE_DEFAULT_JUVENILE_MUTATION_MAGNITUDE = 0.1;

// ──────────────────────────────────────────────────────────────────────
// Weight-Variant Patch Defaults
// ──────────────────────────────────────────────────────────────────────

/**
 * Minimum number of connections perturbed by a single multi-connection
 * weight variant patch. Patches always contain at least this many
 * perturbations when the network has any trainable connections.
 *
 * Contract: NGE_VARIANT_PATCH_MIN_CONNECTION_COUNT=1
 */
export const NGE_VARIANT_PATCH_MIN_CONNECTION_COUNT = 1;

/**
 * Maximum number of connections perturbed by a single multi-connection
 * weight variant patch. Caps patch size independently of network scale so
 * each variant remains a bounded local search step. Reduced from 16 to 8
 * because random-sign perturbations scale as sqrt(K), so smaller K loses
 * little signal while keeping patches local.
 *
 * Contract: NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT=8
 */
export const NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT = 8;

/**
 * Divisor used to derive the desired multi-connection patch size from the
 * total connection count. Combined with
 * {@link NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT} to keep patch size modest.
 * A divisor of 50 means a 500-connection network gets a 10-connection patch,
 * which is then clamped to the patch maximum.
 *
 * Contract: NGE_VARIANT_PATCH_SIZE_DIVISOR=50
 */
export const NGE_VARIANT_PATCH_SIZE_DIVISOR = 50;

/**
 * Deterministic seed stride multiplier between successive variant patches.
 * Each patch uses `baseSeed + index * strideFactor`, keeping streams short
 * and non-overlapping when the same base seed is reused across variant
 * indices.
 *
 * Contract: NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR=4
 */
export const NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR = 4;

/**
 * Minimum deterministic seed offset between successive variant patches.
 * The actual stride is `max(offset, patchSize * strideFactor)` so small
 * patches remain well-separated while large patches cannot overflow a 32-bit
 * seed space by using a per-count multiplier.
 *
 * Contract: NGE_VARIANT_PATCH_SEED_OFFSET=1000
 */
export const NGE_VARIANT_PATCH_SEED_OFFSET = 1_000;

// ──────────────────────────────────────────────────────────────────────
// Variant Magnitude Scaling
// ──────────────────────────────────────────────────────────────────────

/**
 * Total weight exploration range multiplier. Representative deltas are
 * symmetric around zero and span [-magnitude, +magnitude], so the
 * effective exploration width is two magnitudes.
 */
export const NGE_VARIANT_WEIGHT_RANGE = 2;

/**
 * Upper clamp for the width-factor scaling term. Keeps very large variant
 * counts (for example, the 1024 override) from inflating the effective
 * magnitude beyond a bounded multiple of the stage baseline. Reduced from 2.0
 * to 1.5 so the logarithmic width factor does not over-widen the search.
 *
 * Contract: NGE_VARIANT_WIDTH_FACTOR_MAX=1.5
 */
export const NGE_VARIANT_WIDTH_FACTOR_MAX = 1.5;

/**
 * Lower clamp for the size-factor scaling term. Prevents the magnitude from
 * collapsing to zero for extremely large networks.
 *
 * Contract: NGE_VARIANT_SIZE_FACTOR_FLOOR=0.1
 */
export const NGE_VARIANT_SIZE_FACTOR_FLOOR = 0.1;

/**
 * Reference connection count used to scale the effective mutation magnitude
 * by network size. Derived from the patch defaults as
 * ceil(NGE_VARIANT_PATCH_MAX_CONNECTION_COUNT / (1 / NGE_VARIANT_PATCH_SIZE_DIVISOR)).
 *
 * Contract: NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS=400
 */
export const NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS = 400;

// ──────────────────────────────────────────────────────────────────────
// Weight-Exhaustion Gate Constants
// ──────────────────────────────────────────────────────────────────────

/**
 * Minimum absolute scale used by the adaptive improvement threshold to avoid
 * a zero threshold when both baseline and best score are extremely small.
 *
 * Contract: NGE_EXHAUSTION_SCORE_EPSILON=1e-6
 */
export const NGE_EXHAUSTION_SCORE_EPSILON = 1e-6;

/**
 * Noise-sigma fraction for the baby lifecycle stage. Controls how much the
 * adaptive threshold is lifted by variant-count noise in early growth.
 *
 * Contract: NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY=0.003
 */
export const NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY = 0.003;

/**
 * Noise-sigma fraction for the juvenile lifecycle stage.
 *
 * Contract: NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE=0.002
 */
export const NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE = 0.002;

/**
 * Noise-sigma fraction for the adult lifecycle stage.
 *
 * Contract: NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT=0.001
 */
export const NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT = 0.001;

/**
 * Stage fraction for the baby lifecycle stage. Determines the relative
 * improvement bar used to decide whether a weight variant commits.
 *
 * Contract: NGE_EXHAUSTION_STAGE_FRACTION_BABY=0.02
 */
export const NGE_EXHAUSTION_STAGE_FRACTION_BABY = 0.02;

/**
 * Stage fraction for the juvenile lifecycle stage.
 *
 * Contract: NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE=0.01
 */
export const NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE = 0.01;

/**
 * Stage fraction for the adult lifecycle stage.
 *
 * Contract: NGE_EXHAUSTION_STAGE_FRACTION_ADULT=0.006
 */
export const NGE_EXHAUSTION_STAGE_FRACTION_ADULT = 0.006;

/**
 * Total tick budget allocated to weight-exhaustion before structural growth is
 * forced. The raw exhaustion count is `ceil(tickBudget / variantCount)`.
 *
 * Contract: NGE_EXHAUSTION_TICK_BUDGET=48
 */
export const NGE_EXHAUSTION_TICK_BUDGET = 48;

/**
 * Minimum consecutive weight-exhaustion ticks before forcing structural growth.
 *
 * Contract: NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS=1
 */
export const NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS = 1;

/**
 * Maximum consecutive weight-exhaustion ticks before forcing structural growth
 * under normal conditions.
 *
 * Contract: NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS=8
 */
export const NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS = 8;

/**
 * Maximum consecutive weight-exhaustion ticks after a post-growth boost is
 * active. Used to cap the doubled exhaustion limit so bad growth cannot defer
 * weight tuning indefinitely.
 *
 * Contract: NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS=16
 */
export const NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS = 16;

/**
 * Multiplier applied to the base exhaustion limit after a bad growth event.
 * The resulting limit is clamped between 4 and 16.
 *
 * Contract: NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST=2.0
 */
export const NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST = 2.0;

/**
 * Half-range multiplier used by the neuron-budget factor. The factor equals
 * `1.0 + factor * (1.0 - current / max)`, clamped to [0.5, 2.0].
 *
 * Contract: NGE_EXHAUSTION_NEURON_BUDGET_FACTOR=0.5
 */
export const NGE_EXHAUSTION_NEURON_BUDGET_FACTOR = 0.5;

/**
 * Maximum multiplier applied to the noise-sigma term when computing the
 * weight-exhaustion improvement threshold. Without a cap, very large variant
 * counts inflate the noise uplift without bound; this cap keeps the uplift
 * bounded to twice the per-stage noise sigma.
 *
 * Contract: NGE_EXHAUSTION_NOISE_MULTIPLIER_CAP=2.0
 */
export const NGE_EXHAUSTION_NOISE_MULTIPLIER_CAP = 2.0;

/**
 * Per-failure decay rate applied to the adaptive improvement threshold. Each
 * consecutive weight-exhaustion tick multiplies the threshold by
 * `(1 - NGE_EXHAUSTION_THRESHOLD_DECAY_RATE)` so that near-converged scores can
 * still commit small-but-useful weight variants.
 *
 * Contract: NGE_EXHAUSTION_THRESHOLD_DECAY_RATE=0.15
 */
export const NGE_EXHAUSTION_THRESHOLD_DECAY_RATE = 0.15;

/**
 * Floor for the adaptive improvement-threshold decay. The decay multiplier is
 * clamped to this value so the threshold never collapses to zero and allows
 * random noise to commit.
 *
 * Contract: NGE_EXHAUSTION_THRESHOLD_DECAY_FLOOR=0.40
 */
export const NGE_EXHAUSTION_THRESHOLD_DECAY_FLOOR = 0.40;
