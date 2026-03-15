import {
  buildLineageSnapshot,
  LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
} from '../../accessors/telemetry.accessors';

/**
 * Compact lineage entry exposed by the telemetry facade.
 *
 * This read model stays intentionally small so inspection helpers can show
 * immediate ancestry without exporting full genealogy trees.
 */
export type TelemetryLineageSnapshotEntry = {
  id: number;
  parents: number[];
};

type TelemetryLineageGenome = {
  _id?: number;
  _parents?: number[];
};

/**
 * Narrow telemetry-facade host surface required by the lineage chapter.
 *
 * The lineage snapshot path only needs the current population and the
 * lightweight ancestry markers stored on each genome.
 */
export interface TelemetryFacadeLineageHost {
  population: unknown[];
}

export { LINEAGE_SNAPSHOT_DEFAULT_LIMIT };

/**
 * Return a compact lineage sample for the first genomes in the current population.
 *
 * This chapter keeps the public telemetry facade focused on orchestration while
 * lineage inspection lives beside the narrow host contract and default limit it
 * depends on.
 *
 * @param host - `Neat` instance whose population lineage should be sampled.
 * @param limit - Maximum number of genomes to include in the snapshot.
 * @returns Array of `{ id, parents }` lineage entries.
 *
 * @example
 * ```ts
 * const lineageSnapshot = getLineageSnapshot(neat);
 * console.log(lineageSnapshot.at(-1)?.parents);
 * ```
 */
export function getLineageSnapshot(
  host: TelemetryFacadeLineageHost,
  limit: number = LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
): TelemetryLineageSnapshotEntry[] {
  return buildLineageSnapshot(
    host.population as TelemetryLineageGenome[],
    limit,
  );
}
