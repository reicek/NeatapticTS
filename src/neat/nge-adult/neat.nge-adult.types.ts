/**
 * Rolling reward-delta evidence tracked while the adult phase checks for plateau behavior.
 */
export interface PlateauRecord {
  /** Number of evaluation windows participating in the plateau decision. */
  windowSize: number;
  /** Ordered reward deltas collected across the active plateau window. */
  rewardDeltas: number[];
  /** Whether the current reward-delta window satisfies the stagnation guard. */
  isStagnant: boolean;
}

/**
 * Rolling gain evidence tracked while the adult phase checks for stable optimization pressure.
 */
export interface GainStabilityRecord {
  /** Number of evaluation windows participating in the gain-stability decision. */
  windowSize: number;
  /** Ordered gain measurements collected across the active stability window. */
  gainHistory: number[];
  /** Whether the current gain-history window satisfies the stability tolerance. */
  isStable: boolean;
}

/**
 * Typed equilibrium snapshot emitted once one adult zone looks ready for compaction-first policy.
 */
export interface EquilibriumCandidate {
  /** Stable adult-zone identifier being evaluated for equilibrium. */
  zoneId: string;
  /** Whether the candidate zone satisfies the gain-stability guard. */
  isGainStable: boolean;
  /** Whether the candidate zone satisfies the plateau guard. */
  isPlateau: boolean;
}

/**
 * Persistent adult-phase state shelf reserved for plateau, cooling, and equilibrium passes.
 */
export interface AdultState {
  /** Plateau evidence carried across adult evaluation cycles. */
  plateauRecord: PlateauRecord;
  /** Gain-stability evidence carried across adult evaluation cycles. */
  gainStabilityRecord: GainStabilityRecord;
  /** Current equilibrium candidate snapshot for the active adult zone. */
  equilibriumCandidate: EquilibriumCandidate;
  /** Whether adult growth has been cooled to leave budget for prune and compact actions. */
  growthCoolingActive: boolean;
  /** Number of adult evaluation cycles completed so far. */
  cycleCount: number;
}
