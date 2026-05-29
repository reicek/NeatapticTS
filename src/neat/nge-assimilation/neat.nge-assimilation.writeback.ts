import type {
  NgeAssimilationCandidate,
  NgeAssimilationModuleDelta,
  NgeAssimilationPolicy,
  NgeAssimilationResult,
} from './neat.nge-assimilation.types';
import { buildAssimilationResult } from './neat.nge-assimilation.utils';

type NumericDelta = {
  currentValue: number;
  targetValue: number;
};

type OptionalNumericDeltaRecord = Record<string, NumericDelta | undefined>;

const INTEGER_DELTA_FIELD_NAMES = new Set([
  'adultWindow',
  'cooldownWindow',
  'edgeCount',
  'hiddenDim',
  'juvenileWindow',
  'nodeCount',
  'slotCount',
]);
const NON_NEGATIVE_DELTA_FIELD_NAMES = new Set(['broadcastRadius']);
const UNIT_INTERVAL_DELTA_FIELD_NAMES = new Set(['decayRate']);

/**
 * Apply one slow, deterministic structural-prior write-back pass for a single equilibrium candidate.
 *
 * @param candidate - One owner-local equilibrium candidate carrying only structural deltas.
 * @param policy - Resolved write-back policy for the current assimilation pass.
 * @returns The accepted per-module structural-prior update for the DNA-facing shelf.
 */
export function applyAssimilationWriteback(
  candidate: NgeAssimilationCandidate,
  policy: NgeAssimilationPolicy,
): NgeAssimilationResult {
  // Step 1: Reject the candidate atomically when the projected budget impact exceeds the local caps.
  if (shouldRollbackForBudget(candidate, policy)) {
    return buildAssimilationResult(candidate, 'budget-exceeded', policy, null);
  }

  // Step 2: Update only the known structural-prior shelves inside the accepted owner-local boundary.
  const updatedCppnParameterBlock = updateCppnParameterBlock(
    candidate.moduleDelta.cppnParameterBlock,
    policy,
  );
  const updatedModuleDelta = omitUndefinedFields<NgeAssimilationModuleDelta>({
    moduleId: candidate.moduleDelta.moduleId,
    zoneId: candidate.moduleDelta.zoneId,
    archetypeId: candidate.moduleDelta.archetypeId,
    ruleParameters: updateRuleParameterRecord(
      candidate.moduleDelta.ruleParameters,
      policy,
    ),
    cppnTopology: updateOptionalDeltaRecord(
      candidate.moduleDelta.cppnTopology,
      policy,
    ),
    cppnParameterBlock: updatedCppnParameterBlock,
    wiringCostWeights: updateOptionalDeltaRecord(
      candidate.moduleDelta.wiringCostWeights,
      policy,
    ),
    lifecycleKnobs: updateOptionalDeltaRecord(
      candidate.moduleDelta.lifecycleKnobs,
      policy,
    ),
    memoryTier: updateOptionalDeltaRecord(
      candidate.moduleDelta.memoryTier,
      policy,
    ),
    neuromodulatorZone: updateOptionalDeltaRecord(
      candidate.moduleDelta.neuromodulatorZone,
      policy,
    ),
    estimatedBudgetImpact: structuredClone(
      candidate.moduleDelta.estimatedBudgetImpact,
    ),
  });

  // Step 3: Fold the accepted result with explicit lossy telemetry when CPPN quantization ran.
  return buildAssimilationResult(
    candidate,
    'accepted',
    policy,
    updatedModuleDelta,
    {
      lossy: updatedCppnParameterBlock?.currentValue instanceof Int8Array,
    },
  );
}

function shouldRollbackForBudget(
  candidate: NgeAssimilationCandidate,
  policy: NgeAssimilationPolicy,
): boolean {
  if (!policy.budgetGuardEnabled) {
    return false;
  }

  const budgetImpact = candidate.moduleDelta.estimatedBudgetImpact;

  if (budgetImpact === undefined) {
    return false;
  }

  return (
    budgetImpact.nodeCountDelta > policy.maxNodes ||
    budgetImpact.edgeCountDelta > policy.maxEdges ||
    budgetImpact.byteDelta > policy.maxBytes
  );
}

function updateRuleParameterRecord(
  ruleParameters: NgeAssimilationModuleDelta['ruleParameters'],
  policy: NgeAssimilationPolicy,
): NgeAssimilationModuleDelta['ruleParameters'] {
  if (ruleParameters === undefined) {
    return undefined;
  }

  return Object.fromEntries(
    Object.entries(ruleParameters).map(([fieldName, delta]) => [
      fieldName,
      updateNumericDelta(delta, fieldName, policy),
    ]),
  );
}

function updateOptionalDeltaRecord<T extends OptionalNumericDeltaRecord>(
  record: T | undefined,
  policy: NgeAssimilationPolicy,
): T | undefined {
  if (record === undefined) {
    return undefined;
  }

  return Object.fromEntries(
    (Object.entries(record) as [string, NumericDelta][]).map(
      ([fieldName, delta]) => [
        fieldName,
        updateNumericDelta(delta, fieldName, policy),
      ],
    ),
  ) as T;
}

function updateNumericDelta(
  delta: NumericDelta,
  fieldName: string,
  policy: NgeAssimilationPolicy,
): NumericDelta {
  const writeBackValue =
    delta.currentValue +
    policy.writeBackRate * (delta.targetValue - delta.currentValue);

  return {
    currentValue: clampWriteBackValue(writeBackValue, fieldName, policy),
    targetValue: delta.targetValue,
  };
}

function updateCppnParameterBlock(
  parameterBlock: NgeAssimilationModuleDelta['cppnParameterBlock'],
  policy: NgeAssimilationPolicy,
): NgeAssimilationModuleDelta['cppnParameterBlock'] {
  if (parameterBlock === undefined) {
    return undefined;
  }

  const updatedValues = Float32Array.from(
    parameterBlock.targetValue,
    (targetValue, parameterIndex) => {
      const currentValue = parameterBlock.currentValue[parameterIndex];
      return currentValue + policy.writeBackRate * (targetValue - currentValue);
    },
  );

  if (policy.encodingMode === 'lossless') {
    return {
      currentValue: updatedValues,
      targetValue: Float32Array.from(parameterBlock.targetValue),
    };
  }

  const quantizationScale = resolveCppnQuantizationScale(
    updatedValues,
    parameterBlock.targetValue,
  );

  return {
    currentValue: Int8Array.from(updatedValues, (parameterValue) =>
      quantizeCppnParameterValue(parameterValue, quantizationScale),
    ),
    targetValue: Float32Array.from(parameterBlock.targetValue),
    quantizationScale,
  };
}

function clampWriteBackValue(
  writeBackValue: number,
  fieldName: string,
  policy: NgeAssimilationPolicy,
): number {
  if (INTEGER_DELTA_FIELD_NAMES.has(fieldName)) {
    const fieldCap =
      fieldName === 'edgeCount' ? policy.maxEdges : policy.maxNodes;

    return Math.min(fieldCap, Math.max(1, Math.trunc(writeBackValue)));
  }

  if (UNIT_INTERVAL_DELTA_FIELD_NAMES.has(fieldName)) {
    return normalizeNumericValue(Math.min(1, Math.max(0, writeBackValue)));
  }

  if (NON_NEGATIVE_DELTA_FIELD_NAMES.has(fieldName)) {
    return normalizeNumericValue(Math.max(0, writeBackValue));
  }

  return normalizeNumericValue(writeBackValue);
}

function normalizeNumericValue(value: number): number {
  return Number(value.toFixed(12));
}

function resolveCppnQuantizationScale(
  updatedValues: Float32Array,
  targetValues: Float32Array,
): number {
  const maxUpdatedMagnitude = Math.max(...Array.from(updatedValues, Math.abs));
  const maxTargetMagnitude = Math.max(...Array.from(targetValues, Math.abs));

  return Math.max(maxUpdatedMagnitude, maxTargetMagnitude, 1);
}

function quantizeCppnParameterValue(
  parameterValue: number,
  quantizationScale: number,
): number {
  const normalizedValue = parameterValue / quantizationScale;
  const clampedValue = Math.min(1, Math.max(-1, normalizedValue));

  return Math.round(clampedValue * 127);
}

function omitUndefinedFields<T extends object>(record: T): T {
  return Object.fromEntries(
    Object.entries(record).filter(([, fieldValue]) => fieldValue !== undefined),
  ) as T;
}
