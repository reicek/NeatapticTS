import type { NgeEncodingMode } from '../nge-dna/neat.nge-dna.types';

/**
 * Default fraction of one structural gap applied during one assimilation write-back pass.
 */
export const DEFAULT_ASSIMILATION_WRITE_BACK_RATE = 0.1;

/**
 * Default enabled state for the Phase D assimilation budget-guard enforcement switch.
 */
export const DEFAULT_BUDGET_GUARD_ENABLED = true;

/**
 * Supported lossless and lossy encoding modes for owner-local assimilation serialization.
 */
export const ASSIMILATION_ENCODING_MODES = [
  'lossless',
  'lossy',
] as const satisfies readonly NgeEncodingMode[];
