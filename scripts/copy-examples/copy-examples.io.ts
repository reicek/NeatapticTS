/*
 * Filesystem probe helpers and type guards for the copy-examples boundary.
 *
 * These utilities keep I/O concerns isolated from the copy and HTML-building
 * layers so that each layer can be understood and tested independently.
 */

import { access } from 'node:fs/promises';
import type { PublishedExample } from './copy-examples.types.js';

/**
 * Resolves whether a filesystem path exists.
 *
 * Uses `fs.access` rather than `fs.stat` so that the check is a pure
 * existence probe with no size or type information loaded unnecessarily.
 *
 * @param targetPath - Absolute or relative filesystem path to probe.
 * @returns `true` when the path is accessible, `false` when it is not found.
 */
export async function pathExists(targetPath: string): Promise<boolean> {
  try {
    await access(targetPath);
    return true;
  } catch {
    return false;
  }
}

/**
 * Type guard that narrows a `Promise.all` result item to the published-example
 * shape, filtering out entries where the source directory was missing.
 *
 * @param publishedExample - Candidate value returned by `copyExampleEntryPoint`.
 * @returns `true` when the example was published successfully.
 */
export function isPublishedExample(
  publishedExample: PublishedExample | null,
): publishedExample is PublishedExample {
  return publishedExample !== null;
}
