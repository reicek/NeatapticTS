/**
 * Novelty-archive inspection and reset helpers inside the public telemetry facade.
 *
 * This subchapter exists for the part of experimentation where a caller wants
 * to check or reset the novelty-search memory without touching the broader
 * telemetry surface. The novelty archive is a small but important runtime
 * state: it records behavior descriptors that help determine whether a genome
 * is genuinely surprising relative to what the search has already seen.
 *
 * The public read side stays intentionally tiny:
 *
 * - `getNoveltyArchiveSize()` answers how much novelty memory is currently retained,
 * - `resetNoveltyArchive()` starts a fresh novelty observation window.
 *
 * Read this chapter after the root telemetry facade when the question is about
 * novelty-search memory itself rather than generation telemetry, species
 * history, Pareto fronts, or operator diagnostics.
 *
 * ```mermaid
 * flowchart TD
 *   Archive[Novelty archive] --> Size[getNoveltyArchiveSize()<br/>current memory size]
 *   Archive --> Reset[resetNoveltyArchive()<br/>fresh novelty window]
 * ```
 */
/**
 * Narrow telemetry-facade host surface required by the novelty chapter.
 *
 * This chapter owns the tiny novelty-archive maintenance surface directly so
 * the public telemetry facade can delegate both read and reset operations into
 * the same small novelty boundary without depending on a leftover root helper.
 *
 * Only the archive itself is required here because the novelty subchapter is a
 * pure maintenance/read boundary around stored behavior descriptors.
 */
export interface TelemetryFacadeNoveltyHost {
  _noveltyArchive?: unknown[];
}

/**
 * Return the current novelty archive size.
 *
 * This helper gives diagnostics and tests a small read path for novelty search
 * state without pulling the rest of the telemetry facade into view.
 *
 * @param host - `Neat` instance tracking novelty behavior descriptors.
 * @returns Number of archived novelty descriptors.
 *
 * @example
 * ```ts
 * console.log(getNoveltyArchiveSize(neat));
 * ```
 */
export function getNoveltyArchiveSize(
  host: TelemetryFacadeNoveltyHost,
): number {
  return host._noveltyArchive ? host._noveltyArchive.length : 0;
}

/**
 * Clear the novelty archive.
 *
 * Reach for this when you want novelty search to stop comparing against older
 * behavior descriptors and begin building a new archive from scratch.
 *
 * @param host - `Neat` instance whose novelty archive should be reset.
 * @returns Nothing. The archive is mutated in place.
 */
export function resetNoveltyArchive(host: TelemetryFacadeNoveltyHost): void {
  host._noveltyArchive = [];
}
