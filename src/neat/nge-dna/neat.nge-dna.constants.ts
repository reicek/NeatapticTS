import type { NgeSchemaVersion } from './neat.nge-dna.types';

/**
 * Initial schema version for the canonical NGE DNA envelope serialization format.
 */
export const NGE_DNA_SCHEMA_VERSION = 'A.1.0' as NgeSchemaVersion;

/**
 * Default node-budget sentinel used when no caller supplies a stricter budget.
 */
export const NGE_DNA_DEFAULT_BUDGET_MAX_NODES = 512;

/**
 * Default edge-budget sentinel used when no caller supplies a stricter budget.
 */
export const NGE_DNA_DEFAULT_BUDGET_MAX_EDGES = 1_024;

/**
 * Default per-axis partition count for the unit-cube substrate coordinate space.
 */
export const NGE_DNA_DEFAULT_ZONE_PARTITION_COUNT = 3;

/**
 * Default rule-pass priority applied when constructor inputs omit one explicitly.
 */
export const NGE_DNA_DEFAULT_RULE_PRIORITY = 0;

/**
 * Inclusive absolute CPPN weight floor below which one realized edge is discarded.
 */
export const NGE_DNA_DEFAULT_CPPN_ENABLE_THRESHOLD = 0.3;

/**
 * Fixed canonical CPPN input count: `x1`, `y1`, `z1`, `x2`, `y2`, `z2`, `dist`.
 */
export const NGE_DNA_CPPN_INPUT_COUNT = 7;

/**
 * Fixed canonical CPPN output count covering `weight` and `enableBias` output channels.
 */
export const NGE_DNA_CPPN_OUTPUT_COUNT = 2;
