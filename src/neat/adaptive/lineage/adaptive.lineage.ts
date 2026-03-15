/**
 * Lineage-diversity adaptive controllers.
 *
 * This category explains how ancestor uniqueness telemetry feeds back into the
 * search so the population can recover when family trees become too uniform.
 */
export { applyAncestorUniqAdaptive } from '../adaptive';
export {
  applyUniquenessAdjustment,
  extractAncestorUniqueness,
  isCooldownSatisfied,
  resolveUniquenessThresholds,
} from './adaptive.ancestor-uniqueness.utils';
