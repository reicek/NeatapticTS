/**
 * Z-buffer constants extracted from {@link module:./zbuffer}.
 *
 * @module
 */

/**
 * Sentinel value meaning "no wall in this column".
 *
 * Any finite positive sprite distance is closer than this value, so sprites are
 * visible wherever the z-buffer remains empty.
 */
export const NEATENSTEIN_ZBUFFER_EMPTY = Number.POSITIVE_INFINITY;