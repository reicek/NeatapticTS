import type { NgeSchemaVersion } from './neat.nge-dna.types';

/**
 * Initial schema version for the Phase A canonical NGE DNA envelope.
 */
export const NGE_DNA_SCHEMA_VERSION = 'A.1.0' as NgeSchemaVersion;

/**
 * Default node-budget sentinel used until development passes own stricter budgets.
 */
export const NGE_DNA_DEFAULT_BUDGET_MAX_NODES = 512;

/**
 * Default edge-budget sentinel used until development passes own stricter budgets.
 */
export const NGE_DNA_DEFAULT_BUDGET_MAX_EDGES = 1_024;

/**
 * Default per-axis partition count for the Step 03 unit-cube substrate.
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
 * Fixed Phase A CPPN input count: `x1`, `y1`, `z1`, `x2`, `y2`, `z2`, `dist`.
 */
export const NGE_DNA_CPPN_INPUT_COUNT = 7;

/**
 * Fixed Phase A CPPN output count covering `weight` and `enableBias` output channels.
 */
export const NGE_DNA_CPPN_OUTPUT_COUNT = 2;
