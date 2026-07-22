/**
 * Internal assimilation prior write-back for the NGE main agent.
 *
 * This path is owner-local: it derives structural priors from the main agent's
 * own equilibrium candidate and writes them back to its own genome with a
 * weak, decaying update. Enemy weights and enemy structure are explicitly
 * ignored so that assimilation cannot pull foreign topology into the main
 * agent.
 */

import type {
  NgeMainAgentEquilibriumFitnessMetrics,
  NgeMainAgentStableCandidate,
} from '../nge-main-agent/neat.nge-main-agent.adult';
import { AssimilationSchemaError } from './neat.nge-assimilation.errors';
import type {
  NgeAssimilationCandidate,
  NgeAssimilationModuleDelta,
} from './neat.nge-assimilation.types';

/** Minimum value for a unit-interval hyperparameter (0%). */
const MIN_UNIT_INTERVAL = 0;

/** Maximum value for a unit-interval hyperparameter (100%). */
const MAX_UNIT_INTERVAL = 1;

/**
 * Input for one internal assimilation write-back pass.
 *
 * The candidate must come from the main agent's own equilibrium boundary.
 * Optional enemy-derived fields are accepted only so they can be rejected.
 * An optional `stableCandidate` can be supplied for cross-validation against
 * the adult equilibrium's deterministic structural summary.
 *
 * @example
 * ```ts
 * const input: InternalAssimilationInput = {
 *   candidate: equilibriumCandidate,
 *   writeBackRate: 0.1,
 *   decay: 0.9,
 * };
 * ```
 */
export interface InternalAssimilationInput {
  /** Structured equilibrium candidate produced by the main agent's adult phase. */
  candidate: NgeAssimilationCandidate;
  /** Optional deterministic structural summary from the adult equilibrium module. */
  stableCandidate?: NgeMainAgentStableCandidate;
  /** Fraction of the current→target gap applied during this pass. */
  writeBackRate: number;
  /** Decay factor that weakens the applied update over time. */
  decay: number;
  /** Enemy-derived weights, which internal assimilation must never incorporate. */
  enemyWeightTensor?: Float32Array;
  /** Enemy-derived structure checksum, which internal assimilation must never incorporate. */
  enemyStructureChecksum?: string;
  /** Enemy-derived topology hash, which internal assimilation must never incorporate. */
  enemyTopologyHash?: string;
}

/**
 * Result of one internal assimilation write-back pass.
 *
 * The updated module delta contains weak, decayed structural priors that
 * slowly drift the main agent's DNA toward its own equilibrium target.
 *
 * @example
 * ```ts
 * const result: InternalAssimilationResult = writeInternalAssimilationPriors(input);
 * console.log(result.updatedModuleDelta.ruleParameters.connectionDensity.currentValue);
 * ```
 */
export interface InternalAssimilationResult {
  /** Fingerprint of the source DNA that owned the equilibrium candidate. */
  sourceDnaFingerprint: string;
  /** Always false: enemy weights are never incorporated into the main agent. */
  enemyWeightsIncorporated: boolean;
  /** Always false: enemy structure is never incorporated into the main agent. */
  enemyStructureIncorporated: boolean;
  /** Always false: enemy topology is never incorporated into the main agent. */
  enemyTopologyIncorporated: boolean;
  /** True when a supplied stable candidate passed all cross-validation checks. */
  stableCandidateValidated: boolean;
  /** Decay factor actually applied to this update. */
  appliedDecay: number;
  /** Updated structural-prior delta after weak/decaying write-back. */
  updatedModuleDelta: NgeAssimilationModuleDelta;
}

type NumericDelta = {
  currentValue: number;
  targetValue: number;
};

type OptionalNumericDeltaRecord = Record<string, NumericDelta | undefined>;

/**
 * Write weak, decaying structural priors back to the main agent's own genome.
 *
 * The update is derived solely from the main agent's equilibrium candidate.
 * Any enemy-derived weights, structure checksum, or topology hash supplied in
 * the input are ignored. Hyperparameters are sanitized to the unit interval,
 * and non-finite numeric deltas are left untouched so that a single corrupt
 * prior cannot poison unrelated fields.
 *
 * @param input - Internal assimilation input with candidate, rate, and decay.
 * @returns The internal assimilation result with updated module priors.
 * @throws AssimilationSchemaError when the candidate envelope is malformed.
 *
 * @example
 * ```ts
 * const result = writeInternalAssimilationPriors({
 *   candidate,
 *   writeBackRate: 0.1,
 *   decay: 0.9,
 * });
 * expect(result.enemyWeightsIncorporated).toBe(false);
 * ```
 */
export function writeInternalAssimilationPriors(
  input: InternalAssimilationInput,
): InternalAssimilationResult {
  validateCandidateEnvelope(input.candidate);

  const { candidate, stableCandidate, writeBackRate, decay } = input;
  const sanitizedRate = sanitizeUnitInterval(writeBackRate);
  const sanitizedDecay = sanitizeUnitInterval(decay);

  return {
    sourceDnaFingerprint: candidate.sourceDnaFingerprint,
    enemyWeightsIncorporated: false,
    enemyStructureIncorporated: false,
    enemyTopologyIncorporated: false,
    stableCandidateValidated: isStableCandidateValid(
      candidate,
      stableCandidate,
    ),
    appliedDecay: sanitizedDecay,
    updatedModuleDelta: applyWeakDecayUpdate(
      candidate.moduleDelta,
      sanitizedRate,
      sanitizedDecay,
    ),
  };
}

function validateCandidateEnvelope(candidate: NgeAssimilationCandidate): void {
  if (
    candidate === null ||
    typeof candidate !== 'object' ||
    typeof candidate.sourceDnaFingerprint !== 'string' ||
    candidate.sourceDnaFingerprint.length === 0 ||
    typeof candidate.sourceSchemaVersion !== 'string' ||
    candidate.sourceSchemaVersion.length === 0
  ) {
    throw new AssimilationSchemaError(
      'Internal assimilation candidate must have a non-empty source fingerprint and schema version.',
    );
  }

  const { equilibriumCandidate, moduleDelta } = candidate;

  if (
    equilibriumCandidate === null ||
    typeof equilibriumCandidate !== 'object' ||
    typeof equilibriumCandidate.zoneId !== 'string' ||
    equilibriumCandidate.zoneId.length === 0
  ) {
    throw new AssimilationSchemaError(
      'Internal assimilation candidate must have a non-empty equilibrium zone identifier.',
    );
  }

  if (
    moduleDelta === null ||
    typeof moduleDelta !== 'object' ||
    typeof moduleDelta.moduleId !== 'string' ||
    moduleDelta.moduleId.length === 0 ||
    typeof moduleDelta.zoneId !== 'string' ||
    moduleDelta.zoneId.length === 0
  ) {
    throw new AssimilationSchemaError(
      'Internal assimilation module delta must have non-empty module and zone identifiers.',
    );
  }

  if (equilibriumCandidate.zoneId !== moduleDelta.zoneId) {
    throw new AssimilationSchemaError(
      `Zone mismatch: equilibrium zone ${equilibriumCandidate.zoneId} does not match module delta zone ${moduleDelta.zoneId}.`,
    );
  }
}

function isStableCandidateValid(
  candidate: NgeAssimilationCandidate,
  stableCandidate: NgeMainAgentStableCandidate | undefined,
): boolean {
  if (stableCandidate === undefined) {
    return false;
  }

  const { genomeState, fitnessMetrics, structuralInfo } = stableCandidate;

  if (
    genomeState === null ||
    typeof genomeState !== 'object' ||
    typeof genomeState.schemaVersion !== 'string' ||
    genomeState.schemaVersion !== candidate.sourceSchemaVersion
  ) {
    return false;
  }

  if (
    !Array.isArray(genomeState.archetypes) ||
    genomeState.archetypes.length === 0
  ) {
    return false;
  }

  const moduleArchetypeId = candidate.moduleDelta.archetypeId;
  if (
    moduleArchetypeId !== undefined &&
    !genomeState.archetypes.some(
      (descriptor) => descriptor.archetypeId === moduleArchetypeId,
    )
  ) {
    return false;
  }

  if (
    !Number.isFinite(genomeState.nodeCount) ||
    !Number.isFinite(genomeState.edgeCount) ||
    genomeState.nodeCount < 0 ||
    genomeState.edgeCount < 0
  ) {
    return false;
  }

  if (
    structuralInfo === null ||
    typeof structuralInfo !== 'object' ||
    typeof structuralInfo.maxNodes !== 'number' ||
    typeof structuralInfo.maxEdges !== 'number' ||
    typeof structuralInfo.withinBudget !== 'boolean' ||
    !Number.isFinite(structuralInfo.maxNodes) ||
    !Number.isFinite(structuralInfo.maxEdges)
  ) {
    return false;
  }

  if (
    genomeState.nodeCount > structuralInfo.maxNodes ||
    genomeState.edgeCount > structuralInfo.maxEdges
  ) {
    return false;
  }

  return areFitnessMetricsValid(fitnessMetrics) && structuralInfo.withinBudget;
}

function areFitnessMetricsValid(
  fitnessMetrics: NgeMainAgentEquilibriumFitnessMetrics | undefined,
): boolean {
  if (
    fitnessMetrics === null ||
    fitnessMetrics === undefined ||
    typeof fitnessMetrics !== 'object'
  ) {
    return false;
  }

  const requiredFields: (keyof NgeMainAgentEquilibriumFitnessMetrics)[] = [
    'survivalTicks',
    'damageDealt',
    'kills',
    'damageTaken',
    'aimMissRate',
    'complexityBonus',
    'parsimonyDensityPenalty',
  ];

  for (const field of requiredFields) {
    if (!Number.isFinite(fitnessMetrics[field])) {
      return false;
    }
  }

  return (
    fitnessMetrics.survivalTicks >= 0 &&
    fitnessMetrics.damageDealt >= 0 &&
    fitnessMetrics.kills >= 0 &&
    fitnessMetrics.damageTaken >= 0 &&
    fitnessMetrics.aimMissRate >= MIN_UNIT_INTERVAL &&
    fitnessMetrics.aimMissRate <= MAX_UNIT_INTERVAL
  );
}

function applyWeakDecayUpdate(
  moduleDelta: NgeAssimilationModuleDelta,
  writeBackRate: number,
  decay: number,
): NgeAssimilationModuleDelta {
  const base = structuredClone(moduleDelta);

  return {
    ...base,
    ruleParameters: updateOptionalDeltaRecord(
      base.ruleParameters,
      writeBackRate,
      decay,
    ),
    cppnTopology: updateOptionalDeltaRecord(
      base.cppnTopology,
      writeBackRate,
      decay,
    ),
    wiringCostWeights: updateOptionalDeltaRecord(
      base.wiringCostWeights,
      writeBackRate,
      decay,
    ),
    lifecycleKnobs: updateOptionalDeltaRecord(
      base.lifecycleKnobs,
      writeBackRate,
      decay,
    ),
    memoryTier: updateOptionalDeltaRecord(
      base.memoryTier,
      writeBackRate,
      decay,
    ),
    neuromodulatorZone: updateOptionalDeltaRecord(
      base.neuromodulatorZone,
      writeBackRate,
      decay,
    ),
    cppnParameterBlock: updateCppnParameterBlock(
      base.cppnParameterBlock,
      writeBackRate,
      decay,
    ),
  };
}

function sanitizeUnitInterval(value: number): number {
  if (!Number.isFinite(value)) {
    return value > 0 ? MAX_UNIT_INTERVAL : MIN_UNIT_INTERVAL;
  }

  return Math.max(MIN_UNIT_INTERVAL, Math.min(MAX_UNIT_INTERVAL, value));
}

function updateOptionalDeltaRecord<T extends OptionalNumericDeltaRecord>(
  record: T | undefined,
  writeBackRate: number,
  decay: number,
): T | undefined {
  if (record === undefined) {
    return undefined;
  }

  return Object.fromEntries(
    (Object.entries(record) as [string, NumericDelta | undefined][]).map(
      ([fieldName, delta]) => [
        fieldName,
        delta === undefined
          ? undefined
          : updateNumericDelta(delta, writeBackRate, decay),
      ],
    ),
  ) as T;
}

function updateNumericDelta(
  delta: NumericDelta,
  writeBackRate: number,
  decay: number,
): NumericDelta {
  if (
    !Number.isFinite(delta.currentValue) ||
    !Number.isFinite(delta.targetValue)
  ) {
    return {
      currentValue: delta.currentValue,
      targetValue: delta.targetValue,
    };
  }

  const nextValue =
    delta.currentValue +
    writeBackRate * decay * (delta.targetValue - delta.currentValue);

  return {
    currentValue: normalizeNumericValue(nextValue),
    targetValue: delta.targetValue,
  };
}

function updateCppnParameterBlock(
  parameterBlock: NgeAssimilationModuleDelta['cppnParameterBlock'],
  writeBackRate: number,
  decay: number,
): NgeAssimilationModuleDelta['cppnParameterBlock'] {
  if (parameterBlock === undefined) {
    return undefined;
  }

  const updatedValues = Float32Array.from(
    parameterBlock.targetValue,
    (targetValue, parameterIndex) => {
      const currentValue = parameterBlock.currentValue[parameterIndex];

      if (!Number.isFinite(currentValue) || !Number.isFinite(targetValue)) {
        return normalizeNumericValue(currentValue);
      }

      return normalizeNumericValue(
        currentValue + writeBackRate * decay * (targetValue - currentValue),
      );
    },
  );

  return {
    currentValue: updatedValues,
    targetValue: Float32Array.from(parameterBlock.targetValue),
  };
}

function normalizeNumericValue(value: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }

  return Number(value.toFixed(12));
}
