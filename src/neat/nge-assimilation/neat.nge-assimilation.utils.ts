import { AssimilationSchemaError } from './neat.nge-assimilation.errors';
import type {
  NgeAssimilationCandidate,
  NgeAssimilationModuleDelta,
  NgeAssimilationPolicy,
  NgeAssimilationResult,
} from './neat.nge-assimilation.types';

/**
 * Validate that one equilibrium candidate can safely enter the owner-local assimilation path.
 *
 * @param candidate - Candidate assembled from the adult equilibrium boundary.
 * @returns A schema error when the envelope is inconsistent, otherwise `null`.
 */
export function validateAssimilationCandidate(
  candidate: NgeAssimilationCandidate,
): AssimilationSchemaError | null {
  if (candidate.sourceDnaFingerprint.trim().length === 0) {
    return new AssimilationSchemaError(
      'Assimilation candidate requires a source DNA fingerprint.',
    );
  }

  if (candidate.sourceSchemaVersion.trim().length === 0) {
    return new AssimilationSchemaError(
      'Assimilation candidate requires a source schema version.',
    );
  }

  if (candidate.equilibriumCandidate.zoneId !== candidate.moduleDelta.zoneId) {
    return new AssimilationSchemaError(
      'Assimilation candidate zone identifiers must match across the equilibrium and module shelves.',
    );
  }

  const cppnParameterBlock = candidate.moduleDelta.cppnParameterBlock;

  if (
    cppnParameterBlock !== undefined &&
    cppnParameterBlock.currentValue.length !==
      cppnParameterBlock.targetValue.length
  ) {
    return new AssimilationSchemaError(
      'Assimilation candidate CPPN parameter arrays must have matching lengths.',
    );
  }

  return null;
}

/**
 * Build the public telemetry payload emitted by the owner-local assimilation boundary.
 *
 * @param policy - Resolved policy for the current assimilation pass.
 * @param lossy - Whether lossy compression was used during write-back.
 * @returns The normalized telemetry packet for the caller-facing result.
 */
export function buildAssimilationTelemetry(
  policy: NgeAssimilationPolicy,
  lossy: boolean,
): NgeAssimilationResult['telemetry'] {
  return {
    budgetGuardEnabled: policy.budgetGuardEnabled,
    lossy,
  };
}

/**
 * Fold the canonical public result payload for one assimilation attempt.
 *
 * @param candidate - Candidate tied to the current owner-local assimilation pass.
 * @param status - Terminal status for the current pass.
 * @param policy - Resolved policy for the current assimilation pass.
 * @param updatedModuleDelta - Updated structural-prior payload, or `null` when rejected.
 * @param telemetryOptions - Optional telemetry overrides for the current result.
 * @returns The normalized result returned by the assimilation facade.
 */
export function buildAssimilationResult(
  candidate: NgeAssimilationCandidate,
  status: NgeAssimilationResult['status'],
  policy: NgeAssimilationPolicy,
  updatedModuleDelta: NgeAssimilationModuleDelta | null,
  telemetryOptions: {
    lossy?: boolean;
  } = {},
): NgeAssimilationResult {
  return {
    moduleId: candidate.moduleDelta.moduleId,
    status,
    telemetry: buildAssimilationTelemetry(
      policy,
      telemetryOptions.lossy ?? false,
    ),
    updatedModuleDelta,
  };
}
