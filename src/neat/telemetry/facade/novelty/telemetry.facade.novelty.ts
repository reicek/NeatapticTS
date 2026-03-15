/**
 * Narrow telemetry-facade host surface required by the novelty chapter.
 *
 * This chapter owns the tiny novelty-archive maintenance surface directly so
 * the public telemetry facade can delegate both read and reset operations into
 * the same small novelty boundary without depending on a leftover root helper.
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
 */
export function getNoveltyArchiveSize(
  host: TelemetryFacadeNoveltyHost,
): number {
  return host._noveltyArchive ? host._noveltyArchive.length : 0;
}

/**
 * Clear the novelty archive.
 *
 * @param host - `Neat` instance whose novelty archive should be reset.
 * @returns Nothing. The archive is mutated in place.
 */
export function resetNoveltyArchive(host: TelemetryFacadeNoveltyHost): void {
  host._noveltyArchive = [];
}
