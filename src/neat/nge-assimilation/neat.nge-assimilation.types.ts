import type { EquilibriumCandidate } from '../nge-adult/neat.nge-adult.types';
import type { NgeEncodingMode } from '../nge-dna/neat.nge-dna.types';

/**
 * Structural-prior delta captured for one realized module once the adult phase reaches equilibrium.
 * Each numeric field carries both the current DNA value and the equilibrium-informed target value.
 */
export interface NgeAssimilationModuleDelta {
  /** Stable module identifier receiving the structural-prior update. */
  moduleId: string;
  /** Stable zone identifier associated with the module at equilibrium time. */
  zoneId: string;
  /** Optional archetype identifier when the module belongs to one shared developmental template. */
  archetypeId?: string;
  /** Rule-parameter targets drifted during the adult lifetime. */
  ruleParameters?: Record<
    string,
    {
      /** Current DNA value before assimilation begins. */
      currentValue: number;
      /** Equilibrium-informed structural target that write-back should approach slowly. */
      targetValue: number;
    }
  >;
  /** CPPN topology-related structural targets discovered during the adult lifetime. */
  cppnTopology?: {
    /** Current versus target node count for the local CPPN block. */
    nodeCount?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target edge count for the local CPPN block. */
    edgeCount?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target enable threshold for sparse adjacency realization. */
    enableThreshold?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target connection-density prior for the local topology. */
    connectionDensity?: {
      currentValue: number;
      targetValue: number;
    };
  };
  /** Optional CPPN parameter block eligible for lossy owner-local write-back compression. */
  cppnParameterBlock?: {
    /** Current serialized CPPN parameter payload before assimilation updates. */
    currentValue: Float32Array | Int8Array;
    /** Equilibrium-informed CPPN parameter payload that write-back should approach slowly. */
    targetValue: Float32Array;
    /** Optional unit-interval scale recorded when the current payload is quantized. */
    quantizationScale?: number;
  };
  /** Wiring-economy coefficients and structural budgets affected by the equilibrium run. */
  wiringCostWeights?: {
    /** Current versus target node-cost weight. */
    nodeWeight?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target edge-cost weight. */
    edgeWeight?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target inter-zone penalty applied to long edges. */
    interZonePenalty?: {
      currentValue: number;
      targetValue: number;
    };
  };
  /** Lifecycle scheduling knobs that may be written back after equilibrium is confirmed. */
  lifecycleKnobs?: {
    /** Current versus target juvenile evaluation-window count. */
    juvenileWindow?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target adult evaluation-window count. */
    adultWindow?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target cooldown-window count between structural edits. */
    cooldownWindow?: {
      currentValue: number;
      targetValue: number;
    };
  };
  /** Memory-tier targets reserved for recurrent and episodic module assimilation. */
  memoryTier?: {
    /** Current versus target recurrent hidden-state dimensionality. */
    hiddenDim?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target episodic slot count. */
    slotCount?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target decay rate for short-term memory state. */
    decayRate?: {
      currentValue: number;
      targetValue: number;
    };
  };
  /** Neuromodulator-zone targets reserved for later broadcast tuning. */
  neuromodulatorZone?: {
    /** Current versus target lower gain bound for the zone. */
    gainMin?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target upper gain bound for the zone. */
    gainMax?: {
      currentValue: number;
      targetValue: number;
    };
    /** Current versus target broadcast radius. */
    broadcastRadius?: {
      currentValue: number;
      targetValue: number;
    };
  };
  /** Cheap projected budget impact used by the later guard and rollback pass. */
  estimatedBudgetImpact?: {
    /** Predicted node-count delta if the assimilation update is accepted. */
    nodeCountDelta: number;
    /** Predicted edge-count delta if the assimilation update is accepted. */
    edgeCountDelta: number;
    /** Predicted serialized byte-size delta if the assimilation update is accepted. */
    byteDelta: number;
  };
}

/**
 * Typed Phase D input assembled from one adult equilibrium signal and one module-local structural delta.
 * This boundary is structural only: it must never carry raw network weights.
 */
export interface NgeAssimilationCandidate {
  /** Stable equilibrium event emitted by the adult lifecycle boundary. */
  equilibriumCandidate: EquilibriumCandidate;
  /** Deterministic fingerprint of the source DNA envelope being updated. */
  sourceDnaFingerprint: string;
  /** Source DNA schema version used for schema validation before write-back. */
  sourceSchemaVersion: string;
  /** Module-local structural delta prepared for slow assimilation. */
  moduleDelta: NgeAssimilationModuleDelta;
}

/**
 * Resolved policy bag controlling one Phase D structural-prior write-back attempt.
 */
export interface NgeAssimilationPolicy {
  /** Fraction of the distance from current to target value applied per assimilation pass. */
  writeBackRate: number;
  /** Whether the budget guard must reject over-budget updates before any write-back applies. */
  budgetGuardEnabled: boolean;
  /** Encoding mode used when the assimilation boundary later serializes updated structural priors. */
  encodingMode: NgeEncodingMode;
  /** Maximum allowed node count after one accepted assimilation update. */
  maxNodes: number;
  /** Maximum allowed edge count after one accepted assimilation update. */
  maxEdges: number;
  /** Maximum allowed serialized byte size after one accepted assimilation update. */
  maxBytes: number;
}

/**
 * Outcome returned by one owner-local assimilation attempt for a single module candidate.
 */
export interface NgeAssimilationResult {
  /** Stable module identifier tied to the attempted write-back. */
  moduleId: NgeAssimilationModuleDelta['moduleId'];
  /** Whether the candidate was accepted, rejected by budget, or rejected by schema validation. */
  status: 'accepted' | 'budget-exceeded' | 'schema-invalid';
  /** Updated structural-prior shelf after the write-back pass, or null when rejected. */
  updatedModuleDelta: NgeAssimilationModuleDelta | null;
  /** Minimal telemetry flags surfaced by the later orchestration facade. */
  telemetry: {
    /** Whether lossy compression was used by the assimilation path. */
    lossy: boolean;
    /** Whether the budget guard was active for the attempted write-back. */
    budgetGuardEnabled: boolean;
  };
}
