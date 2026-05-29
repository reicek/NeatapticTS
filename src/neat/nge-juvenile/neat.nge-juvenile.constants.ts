import type {
  NgeJuvenileFocusWeights,
  NgeProbeKind,
} from './neat.nge-juvenile.types';

/**
 * Seed focus weights from the plan's initial default-threshold section for the juvenile scorer.
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
export const NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT = 1;

/**
 * Minimum viable episodic slot increment applied by one expansion step.
 */
export const NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT = 1;

/**
 * Exclusive lower bound on reward delta required to evidence-gate a node addition.
 * Node growth is blocked when the module's focus reward delta is at or below this floor.
 */
export const NGE_JUVENILE_DEFAULT_NODE_REWARD_DELTA_FLOOR = 0.0;

/**
 * Cost-pressure fraction above which one window counts as prune evidence.
 */
export const NGE_JUVENILE_DEFAULT_PRUNE_COST_PRESSURE_THRESHOLD = 0.85;

/**
 * Safe absolute minimum edge floor when DNA supplies no tighter prune bound.
 */
export const NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR = 1;
