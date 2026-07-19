import type { VariantScorer } from '../../acceleration/acceleration.types';
import type Network from '../../architecture/network';
import type { AccelerationConfig } from '../../acceleration/acceleration.types';
import type { NgeRealizedModule } from '../nge-dna/neat.nge-dna.types';

/**
 * Focus-weight shelf consumed by the juvenile weighted focus formula for module scoring.
 * Each weight scales one normalized metric — utilization, reward, novelty, stability, or cost.
 */
export interface NgeJuvenileFocusWeights {
  /** Weight applied to normalized module utilization. */
  w_u: number;
  /** Weight applied to normalized reward delta. */
  w_r: number;
  /** Weight applied to normalized novelty. */
  w_n: number;
  /** Weight applied to normalized stability age. */
  w_s: number;
  /** Weight subtracted from normalized wiring cost. */
  w_c: number;
}

/**
 * Cheap per-module metrics snapshot consumed by the juvenile focus scorer.
 */
export interface NgeModuleMetricsSnapshot {
  /** Stable realized-module identifier whose metrics are being scored. */
  moduleId: NgeRealizedModule['moduleId'];
  /** Relative module usage across the active evaluation slice. */
  utilization: number;
  /** Reward improvement attributable to the module in the active slice. */
  rewardDelta: number;
  /** Novel behavior contribution attributable to the module. */
  novelty: number;
  /** Number of stable windows the module has persisted without churn. */
  stabilityAge: number;
  /** Current structural or runtime wiring cost assigned to the module. */
  wiringCost: number;
}

/**
 * One module-level focus result containing both the raw and normalized score.
 */
export interface NgeFocusScore {
  /** Stable realized-module identifier whose metrics were scored. */
  moduleId: NgeRealizedModule['moduleId'];
  /** Weighted scalar emitted by the plan focus formula before vector normalization. */
  rawScore: number;
  /** Probability-like normalized score used by later top-k selection. */
  normalizedScore: number;
  /** True when the raw score is positive enough to support a growth action. */
  supportsGrowth: boolean;
  /** Normalized utilization value that participated in the raw score. */
  normalizedUtilization: number;
  /** Normalized reward delta that participated in the raw score. */
  normalizedRewardDelta: number;
  /** Normalized novelty value that participated in the raw score. */
  normalizedNovelty: number;
  /** Normalized stability age that participated in the raw score. */
  normalizedStabilityAge: number;
  /** Normalized wiring cost that participated in the raw score. */
  normalizedWiringCost: number;
}

/**
 * Normalized focus vector emitted for one juvenile evaluation window.
 * Contains per-module probability-like scores produced by the weighted focus formula.
 */
export interface NgeFocusVector {
  /** One scored entry per module in the input snapshot order. */
  scores: NgeFocusScore[];
  /** Caller-supplied evaluation window identifier. */
  windowIndex: number;
  /** Runtime timestamp recorded as metadata only. */
  computedAt: number;
}

/**
 * Resolved juvenile-phase configuration for focus scoring and later morph guards.
 */
export interface NgeJuvenilePhaseConfig {
  /** Resolved focus weights used by the weighted focus formula. */
  focusWeights: NgeJuvenileFocusWeights;
  /** Hit-rate floor that later slot-expansion logic must exceed. */
  episodicHitRateThreshold: number;
  /** Refresh floor below which recurrent state becomes prune evidence. */
  recurrentRefreshFloor: number;
  /** Consecutive windows required before growth or prune can commit. */
  hysteresisWindowCount: number;
  /** Cooldown windows required between committed morph actions. */
  cooldownWindowCount: number;
  /** Mean-window size used by neuromodulator gain stability checks. */
  gainStabilityWindow: number;
  /** Allowed deviation around the mean gain before stability is considered broken. */
  gainStabilityTolerance: number;
  /** Evaluation window identifier echoed into the focus-vector metadata. */
  windowIndex: number;
  /**
   * Floor below which the composite node-growth signal cannot open the node-add gate.
   * The signal is derived from the focus-weighted module metrics, so it reflects
   * utilization, reward, novelty, stability, and cost jointly rather than reward alone.
   */
  nodeGrowthSignalFloor: number;
  /** Number of hidden nodes one approved node-addition step plans to insert. */
  nodeAdditionCount: number;
  /** Number of forward edges one approved edge-densification step plans to insert. */
  edgeDensificationCount: number;
  /**
   * Maximum number of structural edits (morph deltas) that one lifecycle call
   * may apply. When set, the lifecycle runner truncates the planned delta list
   * to this count before passing it to `applyMorphDeltas`, enabling batch
   * growth in a single tick rather than one-at-a-time.
   */
  maxStructuralEditsPerStep: number;
}

/**
 * DNA-configured structural caps and live counts for one locally growing module.
 */
export interface NgeGrowthBudget {
  /** Maximum node count the DNA genome allows for this module. */
  maxNodes: number;
  /** Maximum edge count the DNA genome allows for this module. */
  maxEdges: number;
  /** Maximum episodic slot count the DNA genome allows for this module. */
  maxEpisodicSlots: number;
  /** Current live node count inside the module. */
  currentNodeCount: number;
  /** Current live edge count inside the module. */
  currentEdgeCount: number;
  /** Current live episodic slot count inside the module. */
  currentEpisodicSlotCount: number;
}

/**
 * One scored prune candidate supplied by the caller for dry-run ranking.
 */
export interface NgePruneCandidate {
  /** Stable edge identifier for this candidate. */
  candidateId: string;
  /** Wiring cost attributed to this edge where higher values prune first. */
  wiringCost: number;
  /**
   * Euclidean or proxy edge length used as an inter-module pressure signal.
   */
  edgeLength: number;
}

/**
 * DNA-configured structural floors and permanent prune exemptions for one juvenile module.
 * Guards the minimum edge and node counts that no morph action may reduce below.
 */
export interface NgePruneBudget {
  /** Minimum edge count the DNA genome requires for this module. */
  minEdges: number;
  /** Minimum node count the DNA genome requires for this module. */
  minNodes: number;
  /**
   * Edge IDs permanently exempt from pruning regardless of cost pressure.
   */
  costExemptEdgeIds: readonly string[];
  /** Current live edge count inside the module. */
  currentEdgeCount: number;
  /** Current live node count inside the module. */
  currentNodeCount: number;
  /** Current aggregate wiring cost attributed to the module. */
  currentWiringCost: number;
}

/**
 * JSON-safe hysteresis state tracked persistently across juvenile morphology evaluation windows.
 * Persists growth and prune streak counts plus cooldown counters between successive windows.
 */
export interface NgeHysteresisState {
  /** Positive-focus window streak for growth decisions. */
  growthPositiveWindowCount: number;
  /** Sustained underuse window streak for prune decisions. */
  pruneUnderuseWindowCount: number;
  /** Most recent committed morph kind, if any. */
  lastMorphKind: NgeMorphDelta['kind'] | 'none';
  /** Remaining cooldown windows before another morph may commit. */
  cooldownWindowsRemaining: number;
}

/**
 * Dry-run structural delta that later juvenile passes can validate or roll back.
 */
export interface NgeMorphDelta {
  /** Planned structural action family. */
  kind: 'edgeDensify' | 'slotExpand' | 'nodeAdd' | 'edgePrune' | 'compact';
  /** Target module receiving the structural change. */
  targetModuleId: NgeRealizedModule['moduleId'];
  /** JSON-safe action details reserved for later morph passes. */
  detail: Record<string, unknown>;
  /** Signed wiring-cost delta that the dry run predicts. */
  wiringCostDelta: number;
}

/**
 * Canonical probe kinds cycled by the juvenile perturbation scheduler across episodes.
 * Each kind suppresses, perturbs, or gates a different aspect of module behavior.
 */
export type NgeProbeKind = 'lesion' | 'noise' | 'gating';

/**
 * Fully resolved configuration for the cadence-gated juvenile perturbation probe scheduler.
 * Controls probe-kind rotation, ledger size cap, and per-kind severity parameters.
 */
export interface NgeProbeSchedulerConfig {
  /** Probe kinds cycled in deterministic rotation order. */
  probeKinds: NgeProbeKind[];
  /** Minimum epoch distance between expensive probe executions. */
  cadenceEpochs: number;
  /** Maximum number of append-only ledger entries preserved per episode. */
  maxLedgerEntries: number;
  /** Lesion severity where `1.0` suppresses the whole target module edge set. */
  lesionSeverity: number;
  /** Standard deviation used by noise probes. */
  noiseSigma: number;
  /** Euclidean edge-length cutoff targeted by gating probes. */
  gatingEdgeLengthThreshold: number;
}

/**
 * JSON-safe probe scheduler state tracked persistently across juvenile evaluation windows.
 * Holds the last-fired epoch, probe-kind rotation index, and the append-only probe ledger.
 */
export interface NgeProbeSchedulerState {
  /** Most recent epoch that executed one probe, or `-1` before the first run. */
  lastProbeEpoch: number;
  /** Rotation index for the next probe kind selection. */
  nextProbeKindIndex: number;
  /** Append-only per-episode ledger of measured probe outcomes. */
  ledger: NgeProbeLedgerEntry[];
}

/**
 * Pure scheduler decision emitted for one epoch's cadence check in the juvenile phase.
 * Signals whether the gate is open and which probe kind has been selected for execution.
 */
export interface NgeProbeDecision {
  /** Whether the cadence gate allows one probe to run for the current epoch. */
  shouldRun: boolean;
  /** Probe kind selected for execution when the cadence gate opens. */
  probeKind: NgeProbeKind | undefined;
}

/**
 * Append-only probe ledger entry produced by one juvenile scheduled perturbation pass.
 * Records the probe kind, target module, epoch index, and signed reward delta for analysis.
 */
export interface NgeProbeLedgerEntry {
  /** Probe family applied to the target module. */
  probeKind: NgeProbeKind;
  /** Module whose local behavior was perturbed. */
  targetModuleId: NgeRealizedModule['moduleId'];
  /** Reward observed before the perturbation. */
  rewardBefore: number;
  /** Reward observed after the perturbation. */
  rewardAfter: number;
  /** Signed reward change caused by the perturbation. */
  delta: number;
  /** Evaluation epoch in which the probe ran. */
  epochIndex: number;
}

/**
 * Runtime sentinel confirming the juvenile types module has been loaded.
 * Ensures Istanbul instruments this file so it appears in coverage reports.
 */
export const NGE_JUVENILE_TYPES_LOADED = true;

// ──────────────────────────────────────────────────────────────────────
// Score-Gated Adaptation Types
// ──────────────────────────────────────────────────────────────────────

/**
 * Injected evaluator contract used by the score-gated adaptation window.
 *
 * The caller provides baseline scoring, mutation application, and candidate
 * scoring so `adapt()` can stay domain-agnostic and testable.
 */
export interface NgeCandidateEvaluator {
  /** Score the network before the candidate mutation is applied. */
  baseline: (network: Network, scoreHistory: readonly number[]) => number;
  /** Apply the candidate mutation to the live network in place. */
  apply: (network: Network) => void;
  /** Score the network after the candidate mutation is applied. */
  candidate: (network: Network, scoreHistory: readonly number[]) => number;
}

/**
 * Optional override values for a single `adapt()` call.
 */
export interface NgeAdaptConfig {
  /** Override values for the adaptation decision. */
  overrides?: {
    /** Minimum improvement over baseline required to commit a mutation. */
    improvementThreshold?: number;
    /** Whether the first structural growth bypasses the improvement check. */
    firstGrowthExemption?: boolean;
  };
}

/**
 * Pluggable metrics provider called when supplied to `adapt()`.
 *
 * The provider is intentionally minimal so tests and callers can inject a
 * simple spy without implementing a full telemetry surface.
 */
export interface NgeMetricsProvider {
  /** Return current metrics snapshot for the adaptation window. */
  getMetrics: () => NgeModuleMetricsSnapshot | undefined;
}

/**
 * Pluggable cadence policy called when supplied to `adapt()`.
 */
export interface NgeCadencePolicy {
  /** Decide whether the current tick should run an adaptation pass. */
  decideCadence: () => boolean;
}

/**
 * Domain-agnostic encoder that converts an application observation into a
 * network input vector.
 *
 * @typeParam T - Application-specific observation type.
 */
export interface NgeObservationEncoder<T = unknown> {
  /**
   * Encode one observation into a numeric input vector matching the supplied
   * input size.
   *
   * @param observation - Application-specific observation.
   * @param inputSize - Expected length of the returned input vector.
   * @returns Numeric input vector of length `inputSize`.
   */
  encode: (observation: T, inputSize: number) => number[];
}

/**
 * Optional lifecycle runner signature accepted by `adapt()` for dependency
 * injection and future cycle integration.
 */
export type NgeLifecycleRunner = () => void;

/**
 * Inputs for one score-gated adaptation window.
 */
export interface NgeAdaptOptions {
  /** Live mutable controller network. */
  network: Network;
  /** Rolling score history used by the evaluator. */
  scoreHistory: readonly number[];
  /** Injected evaluator implementing baseline, apply, and candidate steps. */
  evaluator: NgeCandidateEvaluator;
  /** Whether the network has already committed structural growth. */
  hasGrownBefore?: boolean;
  /** Optional override config for the adaptation decision. */
  config?: NgeAdaptConfig;
  /** Optional metrics provider invoked before the adaptation window. */
  metricsProvider?: NgeMetricsProvider;
  /** Optional cadence policy invoked before the adaptation window. */
  cadencePolicy?: NgeCadencePolicy;
  /** Optional observation encoder invoked before the adaptation window. */
  observationEncoder?: NgeObservationEncoder<unknown>;
  /** Optional raw observation passed to the observation encoder. */
  observation?: unknown;
  /** Optional lifecycle runner invoked before the adaptation window. */
  lifecycleRunner?: NgeLifecycleRunner;
}

/**
 * Telemetry recorded for one score-gated adaptation window.
 */
export interface NgeAdaptTelemetry {
  /** Whether a snapshot was captured before the candidate mutation. */
  snapshotTaken: boolean;
  /** Whether a rollback occurred because the candidate was rejected. */
  rollbackOccurred: boolean;
  /** Wall-clock duration of the adaptation window in milliseconds. */
  duration: number;
}

/**
 * Result of one score-gated adaptation window.
 */
export interface NgeAdaptResult {
  /** Score before the candidate mutation. */
  baseline: number;
  /** Score after the candidate mutation. */
  candidate: number;
  /** Whether the candidate mutation was committed. */
  accepted: boolean;
  /** Telemetry for the adaptation window. */
  telemetry: NgeAdaptTelemetry;
}

// ──────────────────────────────────────────────────────────────────────
// Candidate Scoring Types
// ──────────────────────────────────────────────────────────────────────

/**
 * Configuration for candidate score-window and sample-index resolution.
 */
export interface NgeCandidateScoringConfig {
  /** Length of the rolling score window used for plateau detection. */
  windowSize?: number;
  /** Maximum number of sample indices to draw from the evidence window. */
  maxSamples?: number;
}

// ──────────────────────────────────────────────────────────────────────
// Grow-Stabilize Cycle Types
// ──────────────────────────────────────────────────────────────────────

/**
 * Phase label for the NGE grow-stabilize cycle.
 *
 * The cycle alternates between structural `growth` (adding edges/nodes via the
 * NGE lifecycle) and `stabilization` (weight tuning to exploit current
 * capacity before further growth).
 */
export type NgeGrowStabilizePhase = 'growth' | 'stabilization';

/**
 * Quality signal entry accepted by the grow-stabilize cycle.
 *
 * The core module operates on plain numeric scores. App layers convert
 * composite signals (e.g. `RacingQualitySignal`) to scalar numbers before
 * calling the core cycle, keeping the core free of demo-specific types.
 */
export type NgeQualitySignal = number;

/**
 * Discrete NGE lifecycle stage. The progression is embryo → baby → juvenile →
 * adult → equilibrium. Each stage carries different defaults for growth cadence,
 * stabilization intensity, and weight-mutation magnitude.
 */
export type NgeLifecycleStage =
  'embryo' | 'baby' | 'juvenile' | 'adult' | 'equilibrium';

/**
 * Optional numeric overrides for lifecycle-stage thresholds and magnitudes.
 *
 * All fields are optional. When omitted, the lifecycle-stage resolver falls
 * back to the documented constants in `neat.nge-juvenile.constants.ts`.
 */
export interface NgeLifecycleStageConfig {
  /** Optional override for the baby-stage node threshold. */
  babyNodeThreshold?: number;
  /** Optional override for the juvenile-stage node threshold. */
  juvenileNodeThreshold?: number;
  /** Optional override for the baby-stage variant count. */
  babyVariantCount?: number;
  /** Optional override for the adult-stage variant count. */
  adultVariantCount?: number;
  /** Optional override for the juvenile-stage variant count midpoint. */
  juvenileVariantCount?: number;
  /** Optional override for the baby-stage growth cadence. */
  babyGrowthCadence?: number;
  /** Optional override for the adult-stage growth cadence. */
  adultGrowthCadence?: number;
  /** Optional override for the baby-stage stabilization intensity. */
  babyStabilizationIntensity?: number;
  /** Optional override for the adult-stage stabilization intensity. */
  adultStabilizationIntensity?: number;
  /** Optional override for the baby-stage weight mutation magnitude. */
  babyMutationMagnitude?: number;
  /** Optional override for the adult-stage weight mutation magnitude. */
  adultMutationMagnitude?: number;
}

/**
 * Configuration for one grow-stabilize cycle call.
 *
 * All fields have sensible defaults, so callers can override only the knobs
 * they need to tune.
 */
export interface NgeGrowStabilizeConfig {
  /** Maximum number of structural edits per lifecycle call. */
  maxStructuralEditsPerStep: number;
  /** Hard cap for node count after mutation. */
  maxNodes: number;
  /** Hard cap for connection count after mutation. */
  maxConnections: number;
  /** Maximum episodic growth slots the lifecycle may allocate. */
  maxEpisodicSlots: number;
  /** Module identifier passed to the lifecycle runner. */
  moduleId: string;

  // Stabilization plateau detection
  /** Rolling-window length for score-plateau detection. */
  plateauWindowSize: number;
  /** Variance threshold below which the score is treated as plateaued. */
  plateauVarianceThreshold: number;
  /** Minimum stabilization ticks before plateau detection may fire. */
  minStabilizationTicks: number;
  /** Maximum stabilization ticks before growth is forced to re-enter. */
  maxStabilizationTicks: number;

  // Weight and bias mutation tuning
  /** Fraction of connections perturbed during each stabilization tick. */
  weightMutationRate: number;
  /** Maximum magnitude of each weight perturbation. */
  weightMutationMagnitude: number;
  /** Fraction of biases perturbed during each stabilization tick. */
  biasMutationRate: number;
  /** Maximum magnitude of each bias perturbation. */
  biasMutationMagnitude: number;
  /** Minimum score improvement required to treat a mutation as beneficial. */
  improvementThreshold: number;
  /** Cooldown ticks between consecutive weight mutations. */
  mutationCooldownTicks: number;
  /** Cooldown ticks between structural rollbacks. */
  rollbackCooldownTicks: number;
  /** Number of consecutive lifecycle windows used for cooldown gating. */
  lifecycleCooldownWindowCount: number;

  // Growth throttling and sampling
  /** Node count above which growth throttling engages. */
  largeNetworkNodeThreshold: number;
  /** Base back-off interval for large-network throttling. */
  growthThrottleBaseIntervalTicks: number;
  /** Maximum observations drawn from score history for forward-pass evaluation. */
  maxForwardPassSamples: number;

  // Lifecycle stage band overrides
  /** Baby-stage node-count threshold. */
  babyNodeThreshold: number;
  /** Juvenile-stage node-count threshold. */
  juvenileNodeThreshold: number;
  /** Baby-stage variant count. */
  babyVariantCount: number;
  /** Juvenile-stage variant count. */
  juvenileVariantCount: number;
  /** Adult-stage variant count. */
  adultVariantCount: number;
  /** Baby-stage growth cadence. */
  babyGrowthCadence: number;
  /** Adult-stage growth cadence. */
  adultGrowthCadence: number;
  /** Baby-stage stabilization intensity. */
  babyStabilizationIntensity: number;
  /** Adult-stage stabilization intensity. */
  adultStabilizationIntensity: number;
  /** Baby-stage weight-mutation magnitude. */
  babyMutationMagnitude: number;
  /** Adult-stage weight-mutation magnitude. */
  adultMutationMagnitude: number;

  // Buffer-pool budget and acceleration controls
  /** Maximum bytes the connection buffer pool may pool for this cycle. */
  bufferPoolMaxPooledBytes: number;
  /** When true, skip GPU acceleration even if available. */
  disableGPU: boolean;
  /** When true, run evaluation on the main thread instead of workers. */
  disableWorkers: boolean;

  /** Optional acceleration configuration for parallel weight-variant evaluation. */
  accelerationConfig?: AccelerationConfig;
}

/**
 * Result of one grow-stabilize cycle call.
 *
 * @property committed - Whether the cycle committed a structural or weight mutation.
 * @property phase - Current phase after the cycle (`growth` or `stabilization`).
 * @property reason - Outcome reason for diagnostics and telemetry.
 * @property operations - Operations applied during this cycle.
 * @property stabilizationTicksSinceGrowth - Updated tick count since the last structural growth.
 * @property mutatedCount - Number of weight mutations applied (stabilization phase only).
 * @property networkSizeAfter - Network size snapshot after the cycle.
 */
export interface NgeGrowStabilizeResult {
  /** Whether the cycle committed a structural or weight mutation. */
  committed: boolean;
  /** Current phase after the cycle. */
  phase: NgeGrowStabilizePhase;
  /** Outcome reason for diagnostics and telemetry. */
  reason: string;
  /** Operations applied during this cycle. */
  operations: readonly string[];
  /** Updated tick count since the last structural growth. */
  stabilizationTicksSinceGrowth: number;
  /** Number of weight mutations applied (stabilization phase only). */
  mutatedCount: number;
  /** Network size snapshot after the cycle. */
  networkSizeAfter: { nodes: number; connections: number };
  /** Updated hysteresis state from the lifecycle runner (growth phase only). */
  hysteresis?: NgeHysteresisState;
  /**
   * Consecutive stabilization ticks where no weight variant improvement
   * exceeded the adaptive threshold.
   */
  consecutiveWeightExhaustion: number;
  /**
   * Whether the post-growth exhaustion boost is currently active. Doubles the
   * exhaustion limit (capped at 16, floored at 4) after a bad growth event.
   */
  postGrowthThresholdActive: boolean;
  /**
   * Baseline score captured at the start of the most recent growth phase. The
   * next stabilization tick compares the new baseline against this value to detect
   * a bad growth event and activate the post-growth anti-runaway boost.
   */
  preGrowthBaseline?: number;
  /**
   * Score of the best weight variant evaluated during the stabilization phase.
   * Only populated when weight variants were evaluated and a best score exists.
   */
  bestVariantScore?: number;
  /**
   * Adaptive improvement threshold used to decide whether the best variant
   * score justifies committing a weight mutation. Only populated when weight
   * variants were evaluated.
   */
  threshold?: number;
  /**
   * Actual number of weight variants used during the stabilization phase. This
   * may differ from the configured stage count because the stabilization path
   * overrides the count with `NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT`
   * and then derives patches from the live connection count. Populated whenever
   * variant evaluation runs; zero when the non-variant fallback path is used.
   */
  actualVariantCount?: number;
  /**
   * Number of consecutive failed stabilization ticks carried forward to the
   * caller. Reset to zero when the cycle enters the growth phase.
   */
  consecutiveStabilizationFailures?: number;
}

/**
 * Input for one grow-stabilize cycle call.
 *
 * The first four fields are required and match the minimal red-test contract.
 * All other fields are optional with sensible defaults, so the cycle can run
 * with just a network and score history.
 *
 * @property network - Live mutable controller network.
 * @property scoreHistory - Rolling score history used to compute module metrics.
 * @property hasGrownBefore - Whether the network has already undergone structural growth.
 * @property stabilizationTicksSinceGrowth - Ticks elapsed in stabilization since the last growth.
 * @property qualityScoreHistory - Rolling quality scores for plateau detection.
 * @property hysteresis - Current hysteresis state from the caller, used for subsequent growth.
 * @property random - Optional deterministic random source for weight mutations.
 * @property config - Optional configuration overrides.
 * @property lifecycleRunner - Optional lifecycle runner for dependency injection.
 * @property lifecycleStage - Optional lifecycle stage override; defaults to `'baby'`.
 * @property accelerationConfig - Optional acceleration config for parallel variant evaluation.
 * @property inputs - Optional training inputs for parallel variant evaluation.
 * @property target - Optional training target for parallel variant evaluation.
 */
export interface NgeGrowStabilizeInput {
  /** Live mutable controller network. */
  readonly network: Network;
  /** Rolling score history used to compute module metrics. */
  readonly scoreHistory: readonly number[];
  /** Whether the network has already undergone structural growth. */
  readonly hasGrownBefore: boolean;
  /** Ticks elapsed in stabilization since the last structural growth. */
  readonly stabilizationTicksSinceGrowth: number;
  /** Rolling quality scores for plateau detection. Defaults to an empty array. */
  readonly qualityScoreHistory?: readonly number[];
  /** Current hysteresis state from the caller, used for subsequent growth. */
  readonly hysteresis?: NgeHysteresisState;
  /** Optional deterministic random source for weight mutations. */
  readonly random?: () => number;
  /** Optional configuration overrides. */
  readonly config?: Partial<NgeGrowStabilizeConfig>;
  /**
   * Optional lifecycle runner override for dependency injection and cycle
   * breaking. When omitted, the default `runNgeLifecycle` is used.
   */
  readonly lifecycleRunner?: (input: {
    stage: NgeLifecycleStage;
    moduleId: string;
    metrics: NgeModuleMetricsSnapshot;
    budget: NgeGrowthBudget;
    config: Partial<NgeJuvenilePhaseConfig>;
    hysteresis: NgeHysteresisState;
    network?: Network;
    pruneBudget?: NgePruneBudget;
    seed?: number;
  }) => {
    stage: string;
    juvenileResult?: {
      focusScore: NgeFocusScore;
      deltas: NgeMorphDelta[];
    };
    applyOutcomes?: readonly { status: string; kind: string }[];
    hysteresis?: NgeHysteresisState;
  };
  /** Optional lifecycle stage override; defaults to `'baby'`. */
  readonly lifecycleStage?: NgeLifecycleStage;
  /** Optional acceleration config for parallel weight-variant evaluation. */
  readonly accelerationConfig?: AccelerationConfig;
  /** Optional training inputs for parallel variant evaluation. */
  readonly inputs?: number[][];
  /** Optional training target for parallel variant evaluation. */
  readonly target?: number[];
  /**
   * Optional custom scorer for parallel weight-variant evaluation.
   *
   * When supplied, both the baseline score and the variant scores are computed
   * with this scorer so the weight-exhaustion commit inequality compares values
   * in the same score space. The scorer must return values that share the same
   * sign, scale, and semantic direction as `baselineScore`; otherwise a variant
   * can never beat the threshold (for example, negative MSE versus a positive
   * driving-quality score). Callers that need a task-specific reduction (for
   * example, collapsing a 2-D controller output to a scalar driving-quality
   * score) should provide a {@link VariantScorer} here.
   *
   * Background reading:
   * - Mean squared error:
   *     [Wikipedia — Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
   * - NEAT:
   *     K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through
   *     Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2,
   *     pp. 99-127, 2002.
   *     [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
   */
  readonly scoreFn?: VariantScorer;
  /**
   * Baseline quality score used by the weight-exhaustion gate to decide
   * whether a variant improvement is large enough to commit. Defaults to the
   * last entry of `qualityScoreHistory` when omitted.
   */
  readonly baselineScore?: number;
  /**
   * Previous quality score recorded immediately before the last structural
   * growth. Used as the fallback baseline when `baselineScore` is omitted so
   * the weight-exhaustion gate measures improvement against the pre-growth
   * score.
   */
  readonly previousScore?: number;
  /**
   * Current count of consecutive weight-exhaustion ticks carried over from a
   * previous stabilization cycle.
   */
  readonly consecutiveWeightExhaustion?: number;
  /**
   * Whether a post-growth anti-runaway boost is already active from a previous
   * cycle.
   */
  readonly postGrowthThresholdActive?: boolean;
  /**
   * Baseline score captured at the start of the most recent growth phase. The
   * cycle compares the new stabilization baseline against this value to detect a
   * bad growth event and activate the post-growth anti-runaway boost.
   */
  readonly preGrowthBaseline?: number;
  /**
   * Maximum neuron budget for the adaptive improvement threshold. Defaults to
   * `config.maxNodes` when omitted.
   */
  readonly maxNeurons?: number;
  /**
   * Known score ceiling used by the adaptive improvement threshold. When
   * omitted or `Infinity`, the threshold scales with the absolute score
   * magnitude instead of headroom.
   */
  readonly scoreCeiling?: number;
  /**
   * Number of consecutive failed stabilization ticks observed by the caller.
   * When this count reaches the configured threshold, the next cycle skips
   * weight tuning and forces structural growth instead of plateau detection.
   */
  readonly consecutiveStabilizationFailures?: number;
}
