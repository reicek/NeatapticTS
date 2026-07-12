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
