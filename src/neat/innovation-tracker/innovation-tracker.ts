/**
 * Contracts for the NEAT innovation-tracker boundary.
 *
 * NEAT needs two kinds of innovation memory at the same time:
 *
 * 1. a monotonic global cursor that guarantees every truly new structural gene
 *    gets a fresh historical marking,
 * 2. a generation-scoped de-duplication shelf that lets equivalent mutations
 *    created during the same generation reuse those markings.
 *
 * This module is the smallest explicit owner of that split. Mutation helpers
 * write through this tracker, crossover reads the resulting innovation numbers
 * for alignment, and export/import persists the tracker so deterministic replay
 * does not silently reassign identities mid-run.
 *
 * Required teaching output: historical-markings lifecycle.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   gen0[Generation-zero template\nnode geneIds and connection innovations]:::base --> seed[Seed tracker above template maxima]:::accent
 *   seed --> tracker[InnovationTracker\nnextInnovationId + generation-local shelves]:::base
 *
 *   tracker --> addConn[add-conn mutation\nallocate or reuse innovation id]:::base
 *   tracker --> addNode[add-node mutation\nrecord split reuse + inserted node geneId]:::base
 *
 *   addConn --> crossover[Crossover aligns by innovation number\nmatch, disjoint, excess]:::accent
 *   addNode --> crossover
 *
 *   crossover --> checkpoint[Export checkpoint\npersist tracker + networks]:::base
 *   checkpoint --> restore[Restore checkpoint\nrestore tracker exactly]:::base
 *   restore --> tracker
 *
 *   tracker --> boundary{Advance generation?}:::base
 *   boundary -->|Yes| clear[Clear generation-local shelves\nkeep nextInnovationId monotonic]:::base
 *   boundary -->|No| tracker
 *   clear --> tracker
 * ```
 */
import type {
  InnovationTracker,
  InnovationTrackerJSON,
  NodeSplitRecord,
} from './innovation-tracker.types';

/** Initial active generation for a newly created tracker. */
export const INITIAL_TRACKER_GENERATION = 0;

/** Initial next-innovation cursor for a newly created tracker. */
export const INITIAL_NEXT_INNOVATION_ID = 0;

/**
 * Create an empty innovation tracker for a fresh controller.
 *
 * New controllers start with no generation-local structural reuse records, but
 * they still need a concrete tracker object so mutation, export, and restore
 * code can depend on one explicit owner instead of scattered maps.
 *
 * Conceptually, the tracker holds two different kinds of memory:
 *
 * - `nextInnovationId`: a monotonic global cursor (durable across generations)
 * - `nodeSplitRecords` / `connectionInnovations`: generation-local de-duplication
 *   shelves (cleared when mutation advances to a later generation)
 *
 * @returns Empty innovation tracker ready for generation-zero use.
 *
 * @example
 * ```ts
 * const tracker = createInnovationTracker();
 * tracker.nextInnovationId; // 0
 * ```
 */
export function createInnovationTracker(): InnovationTracker {
  return {
    activeGeneration: INITIAL_TRACKER_GENERATION,
    nextInnovationId: INITIAL_NEXT_INNOVATION_ID,
    nodeSplitRecords: new Map(),
    connectionInnovations: new Map(),
  };
}

/**
 * Serialize a live innovation tracker into a checkpoint payload.
 *
 * Checkpoints need more than the next innovation cursor. They also need the
 * current generation's de-duplication registries so a restored run can keep
 * assigning the same identities if it resumes mid-generation.
 *
 * @param tracker Live tracker to serialize.
 * @returns JSON-safe tracker payload.
 */
export function serializeInnovationTracker(
  tracker: InnovationTracker,
): InnovationTrackerJSON {
  return {
    activeGeneration: tracker.activeGeneration,
    nextInnovationId: tracker.nextInnovationId,
    nodeSplitRecords: Array.from(tracker.nodeSplitRecords.entries()),
    connectionInnovations: Array.from(tracker.connectionInnovations.entries()),
  };
}

/**
 * Restore a live innovation tracker from serialized checkpoint data.
 *
 * Restore is intentionally strict about using the explicit tracker payload
 * instead of reconstructing state from older controller fields. The restored
 * tracker becomes the canonical owner for both the global cursor and the
 * current generation's structural reuse registries.
 *
 * @param serializedTracker Tracker payload from controller export.
 * @returns Live innovation tracker ready for continued mutation.
 */
export function restoreInnovationTracker(
  serializedTracker: InnovationTrackerJSON,
): InnovationTracker {
  return {
    activeGeneration: serializedTracker.activeGeneration,
    nextInnovationId: serializedTracker.nextInnovationId,
    nodeSplitRecords: new Map(serializedTracker.nodeSplitRecords),
    connectionInnovations: new Map(serializedTracker.connectionInnovations),
  };
}

/**
 * Prepare a tracker for mutations targeting a specific generation.
 *
 * Moving into a later generation keeps the monotonic global cursor but clears
 * the generation-local registries. Re-preparing the same generation is a no-op,
 * which is what makes mid-generation checkpoint restore deterministic.
 *
 * @param tracker Live tracker.
 * @param targetGeneration Generation whose mutation window is about to run.
 * @returns Nothing.
 *
 * @example
 * ```ts
 * // On generation boundaries, prepare clears generation-local registries.
 * prepareInnovationTrackerForGeneration(tracker, 10);
 *
 * // Re-preparing the same generation is a no-op.
 * prepareInnovationTrackerForGeneration(tracker, 10);
 * ```
 */
export function prepareInnovationTrackerForGeneration(
  tracker: InnovationTracker,
  targetGeneration: number,
): void {
  if (tracker.activeGeneration === targetGeneration) return;

  tracker.activeGeneration = targetGeneration;
  tracker.nodeSplitRecords.clear();
  tracker.connectionInnovations.clear();
}

/**
 * Prepare a tracker for the controller's current public mutation pass.
 *
 * Public mutation runs usually target the controller's current generation, but
 * a restored or pre-prepared tracker may already be tracking a later mutation
 * window. Using the larger generation value preserves that in-flight state
 * instead of accidentally clearing it.
 *
 * @param tracker Live tracker.
 * @param controllerGeneration Generation currently recorded on the controller.
 * @returns Nothing.
 */
export function prepareInnovationTrackerForMutation(
  tracker: InnovationTracker,
  controllerGeneration: number,
): void {
  prepareInnovationTrackerForGeneration(
    tracker,
    Math.max(controllerGeneration, tracker.activeGeneration),
  );
}

/**
 * Look up a reusable split record for the active generation.
 *
 * @param tracker Live tracker.
 * @param splitKey Stable split-event identity, usually the split connection innovation.
 * @returns Previously recorded split data, if present.
 */
export function getNodeSplitRecord(
  tracker: InnovationTracker,
  splitKey: string,
): NodeSplitRecord | undefined {
  return tracker.nodeSplitRecords.get(splitKey);
}

/**
 * Record a reusable split result for the active generation.
 *
 * @param tracker Live tracker.
 * @param splitKey Stable split-event identity, usually the split connection innovation.
 * @param splitRecord Reusable node-split payload.
 * @returns Nothing.
 */
export function recordNodeSplitRecord(
  tracker: InnovationTracker,
  splitKey: string,
  splitRecord: NodeSplitRecord,
): void {
  tracker.nodeSplitRecords.set(splitKey, splitRecord);
}

/**
 * Look up a reusable connection innovation for the active generation.
 *
 * @param tracker Live tracker.
 * @param connectionKey Caller-defined exact connection identity.
 * @returns Reusable innovation id, if present.
 */
export function getConnectionInnovation(
  tracker: InnovationTracker,
  connectionKey: string,
): number | undefined {
  return tracker.connectionInnovations.get(connectionKey);
}

/**
 * Record a reusable connection innovation for the active generation.
 *
 * @param tracker Live tracker.
 * @param connectionKey Caller-defined exact connection identity.
 * @param innovationId Innovation id to reuse for matching mutations.
 * @returns Nothing.
 */
export function recordConnectionInnovation(
  tracker: InnovationTracker,
  connectionKey: string,
  innovationId: number,
): void {
  tracker.connectionInnovations.set(connectionKey, innovationId);
}

/**
 * Consume the next global innovation id from the tracker.
 *
 * @param tracker Live tracker.
 * @returns Newly reserved innovation id.
 *
 * @example
 * ```ts
 * const innovationId = takeNextInnovationId(tracker);
 * recordConnectionInnovation(tracker, 'from=12->to=27', innovationId);
 * ```
 */
export function takeNextInnovationId(tracker: InnovationTracker): number {
  const innovationId = tracker.nextInnovationId;
  tracker.nextInnovationId += 1;
  return innovationId;
}
