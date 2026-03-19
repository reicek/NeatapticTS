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
 * Phase helpers for adaptive complexity mode.
 *
 * These helpers keep the phase story intentionally small: initialize the first
 * structural mood, watch elapsed generations, then flip between complexify and
 * simplify when the configured window expires.
 */

/* Module introduction boundary for generated README output. */

/**
 * Ensure phase state is initialized.
 *
 * Phase initialization is intentionally lazy so controllers that never enable
 * phased complexity do not pay for extra runtime state. Once seeded, the phase
 * and its start generation persist across later evolve calls.
 *
 * @param engine - NEAT engine instance.
 * @param config - Phased complexity configuration.
 * @returns Nothing.
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
 * The phase helper only flips when the configured duration has actually elapsed.
 * That keeps the structural mood stable long enough for later mutation and
 * pruning passes to express it for several generations before the controller
 * switches direction.
 *
 * @param engine - NEAT engine instance.
 * @param config - Phased complexity configuration.
 * @returns Nothing.
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
 * The phase cycle is intentionally binary so the controller can alternate
 * between growth and simplification without inventing additional intermediate
 * moods here.
 *
 * @param currentPhase - Current phase label.
 * @returns Next phase label.
 */
export function resolveNextPhase(currentPhase: string): string {
  return currentPhase === PHASE_COMPLEXIFY ? PHASE_SIMPLIFY : PHASE_COMPLEXIFY;
}
