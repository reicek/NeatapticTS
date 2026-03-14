import {
  PHASE_COMPLEXIFY,
  PHASE_LENGTH_DEFAULT,
  PHASE_SIMPLIFY,
  ZERO,
} from '../core/adaptive.core.constants';
import type {
  NeatLikeWithAdaptive,
  PhasedComplexityConfig,
} from '../core/adaptive.core.types';

/**
 * Ensure phase state is initialized.
 *
 * @param engine - NEAT engine instance.
 * @param config - Phased complexity configuration.
 * @returns {void}
 */
export function initializePhaseState(
  engine: NeatLikeWithAdaptive,
  config: PhasedComplexityConfig,
): void {
  if (engine._phase) return;

  // Step 1: Select initial phase and record start generation.
  const initialPhase = config.initialPhase ?? PHASE_COMPLEXIFY;
  engine._phase = initialPhase;
  engine._phaseStartGeneration = engine.generation;
}

/**
 * Toggle phase if the current phase has exceeded its length.
 *
 * @param engine - NEAT engine instance.
 * @param config - Phased complexity configuration.
 * @returns {void}
 */
export function togglePhaseIfNeeded(
  engine: NeatLikeWithAdaptive,
  config: PhasedComplexityConfig,
): void {
  const phaseLength = config.phaseLength ?? PHASE_LENGTH_DEFAULT;
  const elapsed = engine.generation - (engine._phaseStartGeneration ?? ZERO);
  if (elapsed < phaseLength) return;

  // Step 1: Toggle phase.
  engine._phase = resolveNextPhase(engine._phase ?? PHASE_COMPLEXIFY);
  // Step 2: Reset phase start generation.
  engine._phaseStartGeneration = engine.generation;
}

/**
 * Resolve next phase name.
 *
 * @param currentPhase - Current phase label.
 * @returns Next phase label.
 */
export function resolveNextPhase(currentPhase: string): string {
  return currentPhase === PHASE_COMPLEXIFY ? PHASE_SIMPLIFY : PHASE_COMPLEXIFY;
}