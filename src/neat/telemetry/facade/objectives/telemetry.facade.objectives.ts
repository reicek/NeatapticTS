/**
 * Objective-inspection and lifecycle helpers inside the public telemetry facade.
 *
 * This chapter is the smallest public surface for answering three related
 * questions about the active objective set:
 *
 * - which objectives are currently registered,
 * - what compact descriptor summary should a dashboard show,
 * - which add/remove events happened recently as the controller adapted.
 *
 * It also carries the two lifecycle controls that let a caller register a new
 * objective or clear custom objectives without digging into the lower-level
 * objective subsystem directly.
 *
 * Read this chapter after the root telemetry facade when the question is about
 * objective policy itself rather than telemetry buffers, species history, or
 * Pareto archive inspection.
 *
 * ```mermaid
 * flowchart TD
 *   Registry["Active objective registry"] --> Keys["getObjectiveKeys()<br/>stable objective names"]
 *   Registry --> Summary["getObjectives()<br/>compact descriptors"]
 *   Events["Objective lifecycle events"] --> History["getObjectiveEvents()<br/>recent add remove log"]
 *   Register["registerTelemetryObjective()"] --> Registry
 *   Clear["clearTelemetryObjectives()"] --> Registry
 * ```
 */
import {
  clearObjectives,
  registerObjective,
} from '../../../objectives/objectives';
import type { NeatLikeWithObjectives } from '../../../objectives/core/objectives.types';
import { getObjectiveEventsSnapshot } from '../../accessors/telemetry.accessors';
import type {
  GenomeLike,
  ObjectiveDescriptor,
} from '../../../shared/neat.shared.types';

/**
 * Narrow telemetry-facade host surface required by the objectives chapter.
 *
 * This chapter keeps objective lifecycle reads and registration helpers beside
 * each other so the root telemetry facade can delegate that policy cluster as a
 * single concept instead of mixing it with telemetry buffers or lineage reads.
 *
 * The host combines the read-side objective registry with the recent event log
 * because callers often want to inspect both the current state and the recent
 * policy changes in the same place.
 */
export interface TelemetryFacadeObjectivesHost extends NeatLikeWithObjectives {
  _objectiveEvents?: { gen: number; type: 'add' | 'remove'; key: string }[];
  _getObjectives(): ObjectiveDescriptor[];
}

/**
 * Return just the registered objective keys in stable order.
 *
 * This is the shortest inspection surface for tests and quick diagnostics that
 * only need to confirm which objectives are active, not the full descriptor
 * payload.
 *
 * @param host - `Neat` instance exposing objective descriptors.
 * @returns Ordered list of active objective keys.
 *
 * @example
 * ```ts
 * console.log(getObjectiveKeys(neat));
 * ```
 */
export function getObjectiveKeys(
  host: TelemetryFacadeObjectivesHost,
): string[] {
  return host._getObjectives().map((objective) => objective.key);
}

/**
 * Return a compact view of active objective descriptors.
 *
 * The full objective descriptor includes accessors and internal metadata. This
 * read model trims that down to the pieces most useful in UI surfaces and
 * debugging output: the key and whether the objective is minimized or
 * maximized.
 *
 * @param host - `Neat` instance exposing objective descriptors.
 * @returns Compact objective summaries in evaluation order.
 *
 * @example
 * ```ts
 * console.table(getObjectives(neat));
 * ```
 */
export function getObjectives(
  host: TelemetryFacadeObjectivesHost,
): { key: string; direction: 'max' | 'min' }[] {
  return host._getObjectives().map((objective) => ({
    key: objective.key,
    direction: objective.direction,
  }));
}

/**
 * Register or replace a custom objective.
 *
 * Use this when an experiment or dashboard wants to add a temporary objective
 * to the live registry without reaching into the deeper objective subsystem
 * directly through controller internals.
 *
 * @param host - `Neat` instance whose multi-objective registry should change.
 * @param key - Unique objective key.
 * @param direction - Whether lower or higher values are considered better.
 * @param accessor - Function that reads the objective value from a genome.
 * @returns Nothing. The objective registry on `host` is updated in place.
 */
export function registerTelemetryObjective(
  host: TelemetryFacadeObjectivesHost,
  key: string,
  direction: 'min' | 'max',
  accessor: (genome: GenomeLike) => number,
): void {
  registerObjective.call(host as never, key, direction, accessor);
}

/**
 * Remove all registered custom objectives so only the default objective path remains.
 *
 * Reach for this when an experiment wants to reset the objective surface to its
 * baseline state without rebuilding the whole controller.
 *
 * @param host - `Neat` instance whose objective registry should be cleared.
 * @returns Nothing. The helper mutates the objective registry in place.
 */
export function clearTelemetryObjectives(
  host: TelemetryFacadeObjectivesHost,
): void {
  clearObjectives.call(host as never);
}

/**
 * Snapshot recent objective add/remove events for telemetry consumers.
 *
 * This is the historical companion to {@link getObjectives}. The descriptor
 * summary tells you what is active now; the event log tells you how the active
 * set changed across recent generations.
 *
 * @param host - `Neat` instance storing objective lifecycle events.
 * @returns Shallow copy of the recorded objective events.
 *
 * @example
 * ```ts
 * const recentEvents = getObjectiveEvents(neat);
 * console.log(recentEvents.at(-1));
 * ```
 */
export function getObjectiveEvents(host: TelemetryFacadeObjectivesHost): {
  gen: number;
  type: 'add' | 'remove';
  key: string;
}[] {
  return getObjectiveEventsSnapshot(host);
}
