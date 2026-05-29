import type {
  NgeAssimilationCandidate,
  NgeAssimilationPolicy,
  NgeAssimilationResult,
} from './neat.nge-assimilation.types';
import {
  buildAssimilationResult,
  validateAssimilationCandidate,
} from './neat.nge-assimilation.utils';
import { applyAssimilationWriteback } from './neat.nge-assimilation.writeback';

/**
 * Orchestrate one equilibrium-candidate assimilation pass inside the owner-local Phase D boundary.
 *
 * @param candidate - Structured equilibrium candidate prepared by the adult boundary.
 * @param policy - Resolved policy controlling validation, budget handling, and encoding behavior.
 * @returns The public result packet for the current assimilation attempt.
 */
export function assimilateEquilibriumCandidate(
  candidate: NgeAssimilationCandidate,
  policy: NgeAssimilationPolicy,
): NgeAssimilationResult {
  // Step 1: Validate the incoming candidate envelope before any mutation-facing work begins.
  const schemaError = validateAssimilationCandidate(candidate);

  if (schemaError !== null) {
    return buildAssimilationResult(candidate, 'schema-invalid', policy, null);
  }

  // Step 2: Delegate budget checks, write-back, and optional compression to the local worker.
  const writebackResult = applyAssimilationWriteback(candidate, policy);

  // Step 3: Normalize the public result shape through the shared helper surface.
  return buildAssimilationResult(
    candidate,
    writebackResult.status,
    policy,
    writebackResult.updatedModuleDelta,
    {
      lossy: writebackResult.telemetry.lossy,
    },
  );
}
