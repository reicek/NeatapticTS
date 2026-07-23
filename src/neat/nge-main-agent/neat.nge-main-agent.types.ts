import type {
  NeatGenomeModuleArchetypeDescriptor,
  NeatGenomeSubstrateCoordinate,
} from '../genome/genome.types';
import type { NgeReproductionPolicyMode } from '../nge-dna/neat.nge-dna.types';

/**
 * Ordered lifecycle stages for the NGE main agent.
 *
 * The stage machine is intentionally distinct from the Racing Curriculum
 * juvenile/adult stage contract so that main-agent morphogenesis can evolve its
 * own cadence without being coupled to a different demo's lifecycle policy.
 */
export type NgeMainAgentLifecycleStage =
  'embryo' | 'juvenile' | 'adult' | 'reproducing';

/**
 * Configuration governing the main agent lifecycle and tier budget.
 *
 * The same config plus the same seed must always advance through the lifecycle
 * in the same order, so the seed is stored here rather than inferred from the
 * environment.
 */
export interface NgeMainAgentLifecycleConfig {
  /** Determinism seed for lifecycle transitions. */
  seed: number;
  /** Tier cap on the number of nodes. */
  maxNodes: number;
  /** Tier cap on the number of edges. */
  maxEdges: number;
}

/**
 * Shared state fields across all main-agent lifecycle stages.
 */
export interface NgeMainAgentLifecycleState {
  /** Current lifecycle stage. */
  stage: NgeMainAgentLifecycleStage;
  /** Completed generation count. */
  generation: number;
  /** Determinism seed carried forward. */
  seed: number;
}

/**
 * Topology budget enforced at every lifecycle transition.
 *
 * The budget is always positive and never exceeds the configured tier cap.
 */
export interface NgeMainAgentTopologyBudget {
  /** Maximum nodes allowed for the current stage. */
  maxNodes: number;
  /** Maximum edges allowed for the current stage. */
  maxEdges: number;
}

/**
 * Archetype descriptor specialized for the embryo stage.
 *
 * Every embryo archetype receives a deterministic substrate coordinate and zone
 * assignment from the coordinate allocator so that downstream materialization is
 * reproducible and zone-aware.
 */
export interface NgeMainAgentEmbryoArchetypeDescriptor extends NeatGenomeModuleArchetypeDescriptor {
  /** Three-axis unit-cube coordinate allocated to this archetype. */
  coordinate: NeatGenomeSubstrateCoordinate;
  /** Deterministic zone id matching the allocated coordinate. */
  zoneId: string;
}

/**
 * Main agent embryo state.
 *
 * The embryo is the smallest materialized stage. It carries the full motif
 * allowlist as archetypes but keeps node and edge counts tiny so that later
 * growth has headroom within the tier budget.
 */
export interface NgeMainAgentEmbryo extends NgeMainAgentLifecycleState {
  stage: 'embryo';
  /** Node count after embryo construction. */
  nodeCount: number;
  /** Edge count after embryo construction. */
  edgeCount: number;
  /** Module archetypes present in the embryo, each with an allocated coordinate. */
  archetypes: NgeMainAgentEmbryoArchetypeDescriptor[];
  /** DNA schema version; remains A.1.0 because this slice does not introduce new motifs. */
  schemaVersion: string;
  /**
   * Initial reproduction mode for the lineage.
   *
   * The mode starts as parthenogenesis and may be updated later by the
   * reproduction-mode hysteresis policy when {@link modeIsEvolvable} is true.
   */
  reproductionMode: NgeReproductionPolicyMode;
  /** Whether the reproduction mode itself may evolve via hysteresis. */
  modeIsEvolvable: boolean;
}

/**
 * Main agent juvenile state during local growth and focus gating.
 */
export interface NgeMainAgentJuvenile extends NgeMainAgentLifecycleState {
  stage: 'juvenile';
  /** Node count after juvenile growth. */
  nodeCount: number;
  /** Edge count after juvenile growth. */
  edgeCount: number;
  /** Module archetypes present in the juvenile. */
  archetypes: NeatGenomeModuleArchetypeDescriptor[];
  /** DNA schema version carried from the embryo. */
  schemaVersion: string;
}

/**
 * Main agent adult state after maturation.
 */
export interface NgeMainAgentAdult extends NgeMainAgentLifecycleState {
  stage: 'adult';
  /** Node count after adult pruning. */
  nodeCount: number;
  /** Edge count after adult pruning. */
  edgeCount: number;
  /** Module archetypes present in the adult. */
  archetypes: NeatGenomeModuleArchetypeDescriptor[];
  /** DNA schema version carried from earlier stages. */
  schemaVersion: string;
}

/**
 * Equilibrium candidate produced by adult optimization.
 */
export interface NgeMainAgentEquilibrium {
  /** Whether the candidate has stabilized. */
  isStable: boolean;
  /** Adult state captured at equilibrium. */
  adult: NgeMainAgentAdult;
}

/**
 * Main agent reproducing stage, ready to emit the next generation's embryo.
 */
export interface NgeMainAgentReproducing extends NgeMainAgentLifecycleState {
  stage: 'reproducing';
  /** Node count inherited from the stable equilibrium adult. */
  nodeCount: number;
  /** Edge count inherited from the stable equilibrium adult. */
  edgeCount: number;
  /** Module archetypes carried into reproduction. */
  archetypes: NeatGenomeModuleArchetypeDescriptor[];
  /** DNA schema version carried from earlier stages. */
  schemaVersion: string;
}

/**
 * Attention-head motif descriptor for threat prioritization.
 *
 * This is a typed tag used by the main agent allowlist; the actual realization
 * logic lives in later NGE phases.
 */
export interface NgeMainAgentAttentionHead {
  /** Computation motif tag. */
  computationType: 'AttentionHead';
}

/**
 * Gated recurrent cell motif descriptor for aim/strafe state retention.
 *
 * This is a typed tag used by the main agent allowlist; the actual realization
 * logic lives in later NGE phases.
 */
export interface NgeMainAgentGatedRecurrentCell {
  /** Computation motif tag. */
  computationType: 'GatedRecurrentCell';
}

/**
 * Episodic slot motif descriptor for spawn-pattern memory.
 *
 * This is a typed tag used by the main agent allowlist; the actual realization
 * logic lives in later NGE phases.
 */
export interface NgeMainAgentEpisodicSlot {
  /** Computation motif tag. */
  computationType: 'EpisodicSlot';
}
