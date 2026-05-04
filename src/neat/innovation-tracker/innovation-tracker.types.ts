/**
 * Contracts for the NEAT innovation-tracker boundary.
 *
 * NEAT needs two kinds of innovation memory at the same time:
 *
 * 1. a monotonic global cursor that guarantees every truly new structural gene
 *    gets a fresh historical marking,
 * 2. a generation-scoped de-duplication shelf that lets equivalent mutations
 *    created during the same generation share those markings.
 *
 * The tracker types in this file make that split explicit. The cursor survives
 * across the whole run. The de-duplication registries are only meaningful for
 * the currently active generation window and are expected to be cleared when a
 * later generation starts mutating.
 *
 * This boundary also owns the checkpoint shape needed for deterministic resume.
 * If a run is paused in the middle of mutating a generation, the serialized
 * tracker preserves both the next innovation id and the generation-scoped
 * registries so restore can continue without silently reassigning identities.
 */

/**
 * Runtime record for a historically aligned node split.
 *
 * When multiple genomes split the same historically marked connection during
 * one mutation window, they should reuse the same inserted-node id and the
 * same pair of replacement connection innovations. This record is the compact
 * payload that preserves that shared identity.
 */
export interface NodeSplitRecord {
  /** Gene id assigned to the inserted node. */
  newNodeGeneId: number;
  /** Innovation id for the incoming split connection. */
  inInnov: number;
  /** Innovation id for the outgoing split connection. */
  outInnov: number;
}

/** Generation-local registry of reusable node-split records keyed by canonical split-event identity. */
export type NodeSplitInnovationRegistry = Map<string, NodeSplitRecord>;

/** Generation-local registry of reusable connection innovations keyed by exact connection identity. */
export type ConnectionInnovationRegistry = Map<string, number>;

/**
 * Live innovation-tracker state owned by the NEAT controller.
 *
 * `activeGeneration` names the generation whose structural mutation window is
 * currently being tracked. The two registries only apply within that window.
 * `nextInnovationId` is the durable global cursor and must never move
 * backward, even when the generation-scoped registries are cleared.
 */
export interface InnovationTracker {
  /** Generation whose mutation window currently owns the registries. */
  activeGeneration: number;
  /** Next global innovation id to assign to a new structural gene. */
  nextInnovationId: number;
  /** Reusable split records discovered during the active generation. */
  nodeSplitRecords: NodeSplitInnovationRegistry;
  /** Reusable connection innovations discovered during the active generation. */
  connectionInnovations: ConnectionInnovationRegistry;
}

/** JSON tuple entry used when serializing node-split records. */
export type NodeSplitInnovationEntry = [string, NodeSplitRecord];

/** JSON tuple entry used when serializing connection innovations. */
export type ConnectionInnovationEntry = [string, number];

/**
 * Serialized innovation-tracker payload used by controller checkpoints.
 *
 * This is the minimum durable state needed to resume a run faithfully after a
 * pause, including mid-generation checkpoints where the current mutation window
 * still has outstanding structural reuse information.
 */
export interface InnovationTrackerJSON {
  /** Generation whose mutation window owns the serialized registries. */
  activeGeneration: number;
  /** Next global innovation id to allocate after restore. */
  nextInnovationId: number;
  /** Serialized split records for the active generation. */
  nodeSplitRecords: NodeSplitInnovationEntry[];
  /** Serialized connection innovations for the active generation. */
  connectionInnovations: ConnectionInnovationEntry[];
}
