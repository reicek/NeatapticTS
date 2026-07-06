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
