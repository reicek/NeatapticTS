/**
 * Return the current size of the novelty archive.
 */
export function getNoveltyArchiveSize(host: {
  _noveltyArchive?: unknown[];
}): number {
  return host._noveltyArchive ? host._noveltyArchive.length : 0;
}

/**
 * Reset the novelty archive in place.
 */
export function resetNoveltyArchive(host: {
  _noveltyArchive?: unknown[];
}): void {
  host._noveltyArchive = [];
}
