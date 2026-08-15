import type {
  OnnxBuildResolvedOptions,
  OnnxExportOptions,
  OnnxQuantizationCalibrationLayerTarget,
  OnnxQuantizationCalibrationRange,
  OnnxResolvedPrecisionOptions,
  OnnxResolvedQuantizationCalibrationOptions,
  OnnxResolvedQuantizationOptions,
} from './network.onnx.export.types';

type RawQuantizationPacket = {
  mode?: unknown;
  target?: unknown;
  targets?: unknown;
  calibration?: unknown;
  activationEncoding?: unknown;
  weightEncoding?: unknown;
  activationGranularity?: unknown;
  weightGranularity?: unknown;
  representation?: unknown;
};

/**
 * Resolve export options with all defaults required by model construction.
 *
 * @param sourceOptions - Raw export options.
 * @param networkLayerCount - Total number of layers in the export network.
 * @returns Resolved options used by this builder.
 * @throws {Error} When precision and quantization options are incompatible.
 */
export function resolveBuildOptions(
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxBuildResolvedOptions {
  const resolvedPrecision = resolvePrecisionOptions(sourceOptions);
  const resolvedQuantization = resolveQuantizationOptions(
    sourceOptions,
    networkLayerCount,
  );

  validatePrecisionAndQuantizationComposition(
    resolvedPrecision,
    resolvedQuantization,
  );

  return {
    includeMetadata: sourceOptions.includeMetadata ?? false,
    opset: sourceOptions.opset ?? 18,
    batchDimension: sourceOptions.batchDimension ?? false,
    legacyNodeOrdering: sourceOptions.legacyNodeOrdering ?? false,
    producerName: sourceOptions.producerName ?? 'neataptic-ts',
    producerVersion: sourceOptions.producerVersion,
    docString: sourceOptions.docString,
    precision: resolvedPrecision,
    quantization: resolvedQuantization,
  };
}

/**
 * Resolve precision options with stable defaults and supported modes only.
 *
 * @param sourceOptions - Raw export options.
 * @returns Normalized precision packet.
 * @throws {Error} When the precision mode is not `float32` or `storage-fp16`.
 */
export function resolvePrecisionOptions(
  sourceOptions: OnnxExportOptions,
): OnnxResolvedPrecisionOptions {
  if (!sourceOptions.precision) {
    return {
      requested: false,
      mode: 'float32',
      metadata: false,
    };
  }

  const precisionMode = sourceOptions.precision.mode ?? 'float32';
  if (precisionMode !== 'float32' && precisionMode !== 'storage-fp16') {
    throw new Error('Phase 7 precision mode must be float32 or storage-fp16.');
  }

  return {
    requested: true,
    mode: precisionMode,
    metadata: sourceOptions.precision.metadata ?? true,
  };
}

/**
 * Resolve quantization options with stable defaults and supported first-wave modes only.
 *
 * @param sourceOptions - Raw export options.
 * @param networkLayerCount - Total number of layers in the export network.
 * @returns Normalized quantization packet.
 * @throws {Error} When quantization mode is unrecognized or calibration data is invalid.
 */
export function resolveQuantizationOptions(
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxResolvedQuantizationOptions {
  if (!sourceOptions.quantization) {
    return createDefaultQuantizationOptions();
  }

  const quantizationPacket = asRawQuantizationPacket(sourceOptions);

  if (isDynamicQuantizationPacket(quantizationPacket)) {
    return resolveDynamicQuantizationOptions(quantizationPacket);
  }

  if (isStaticQuantizationPacket(quantizationPacket)) {
    return resolveStaticQuantizationPacket(
      quantizationPacket,
      sourceOptions,
      networkLayerCount,
    );
  }

  throw new Error(
    'Phase 7 dynamic quantization must use the documented dynamic-uint8 lane.',
  );
}

function createDefaultQuantizationOptions(): OnnxResolvedQuantizationOptions {
  return {
    requested: false,
    mode: null,
    fallbackReasons: [],
  };
}

function asRawQuantizationPacket(
  sourceOptions: OnnxExportOptions,
): RawQuantizationPacket {
  return sourceOptions.quantization as RawQuantizationPacket;
}

function isDynamicQuantizationPacket(
  quantizationPacket: RawQuantizationPacket,
): boolean {
  return quantizationPacket.mode === 'dynamic-uint8';
}

function isStaticQuantizationPacket(
  quantizationPacket: RawQuantizationPacket,
): boolean {
  return quantizationPacket.mode === 'static-8bit';
}

function resolveDynamicQuantizationOptions(
  quantizationPacket: RawQuantizationPacket,
): OnnxResolvedQuantizationOptions {
  validateDynamicQuantizationTarget(quantizationPacket.target ?? 'dense');

  const dynamicRepresentation = resolveDynamicQuantizationRepresentation(
    quantizationPacket.representation,
  );

  return {
    requested: true,
    mode: 'dynamic-uint8',
    target: 'dense',
    representation: dynamicRepresentation,
    fallbackReasons: [],
  };
}

function validateDynamicQuantizationTarget(dynamicTarget: unknown): void {
  if (dynamicTarget === 'dense') {
    return;
  }

  throw new Error(
    'Phase 7 dynamic quantization currently supports dense targets only.',
  );
}

function resolveDynamicQuantizationRepresentation(
  representation: unknown,
): 'DynamicQuantizeLinear' | 'metadata-only' {
  const dynamicRepresentation = representation ?? 'metadata-only';
  if (
    dynamicRepresentation === 'DynamicQuantizeLinear' ||
    dynamicRepresentation === 'metadata-only'
  ) {
    return dynamicRepresentation;
  }

  throw new Error(
    'Phase 7 dynamic quantization must use DynamicQuantizeLinear or metadata-only representation.',
  );
}

function resolveStaticQuantizationPacket(
  quantizationPacket: RawQuantizationPacket,
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxResolvedQuantizationOptions {
  const staticTargets = resolveStaticQuantizationTargets(
    quantizationPacket.targets,
  );
  assertStaticQuantizationTargets(staticTargets);

  const calibrationPacket = resolveExternalCalibrationPacket(
    quantizationPacket.calibration,
  );
  const resolvedWeightGranularity = resolveWeightGranularity(
    quantizationPacket.weightGranularity,
  );
  const resolvedCalibration = resolveStaticCalibrationOptions(
    calibrationPacket,
    staticTargets,
    resolvedWeightGranularity,
    sourceOptions,
    networkLayerCount,
  );

  return {
    requested: true,
    mode: 'static-8bit',
    targets: staticTargets,
    calibration: resolvedCalibration,
    activationEncoding:
      quantizationPacket.activationEncoding === 'int8' ? 'int8' : 'uint8',
    weightEncoding:
      quantizationPacket.weightEncoding === 'uint8' ? 'uint8' : 'int8',
    activationGranularity: 'per-tensor',
    weightGranularity: resolvedWeightGranularity,
    representation:
      quantizationPacket.representation === 'qdq' ? 'qdq' : 'qlinear',
    fallbackReasons: [],
  };
}

function resolveStaticQuantizationTargets(
  targets: unknown,
): Array<'dense' | 'conv'> {
  if (!Array.isArray(targets)) {
    return [];
  }

  return targets.filter(
    (target): target is 'dense' | 'conv' =>
      target === 'dense' || target === 'conv',
  );
}

function assertStaticQuantizationTargets(
  staticTargets: Array<'dense' | 'conv'>,
): void {
  if (staticTargets.length > 0) {
    return;
  }

  throw new Error(
    'Phase 7 static-8bit quantization requires at least one dense or conv target.',
  );
}

function resolveExternalCalibrationPacket(calibrationPacket: unknown): object {
  if (
    calibrationPacket &&
    typeof calibrationPacket === 'object' &&
    (calibrationPacket as { source?: unknown }).source === 'external'
  ) {
    return calibrationPacket;
  }

  throw new Error(
    'Phase 7 static-8bit quantization requires an external calibration packet.',
  );
}

function resolveWeightGranularity(
  weightGranularity: unknown,
): 'per-tensor' | 'per-output-channel' {
  return weightGranularity === 'per-output-channel'
    ? 'per-output-channel'
    : 'per-tensor';
}

function resolveStaticCalibrationOptions(
  calibrationPacket: unknown,
  staticTargets: Array<'dense' | 'conv'>,
  weightGranularity: 'per-tensor' | 'per-output-channel',
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxResolvedQuantizationCalibrationOptions {
  const calibrationRecord = calibrationPacket as {
    packetId?: unknown;
    sampleCount?: unknown;
    layerTargets?: unknown;
    weightRangePolicy?: unknown;
    zeroInclusion?: unknown;
    activationSymmetry?: unknown;
    weightSymmetry?: unknown;
    roundingMode?: unknown;
  };
  const resolvedLayerTargets = resolveCalibrationLayerTargets(
    calibrationRecord.layerTargets,
    staticTargets,
    sourceOptions,
    networkLayerCount,
  );

  if (
    weightGranularity === 'per-output-channel' &&
    resolvedLayerTargets.some((layerTarget) => layerTarget.target === 'dense')
  ) {
    throw new Error(
      'Phase 7 static-8bit per-output-channel weight granularity currently requires Conv calibration targets only.',
    );
  }

  if (
    calibrationRecord.weightRangePolicy !== undefined &&
    calibrationRecord.weightRangePolicy !== 'min-max'
  ) {
    throw new Error(
      'Phase 7 static-8bit calibration currently supports the min-max weight range policy only.',
    );
  }

  if (
    calibrationRecord.zeroInclusion !== undefined &&
    calibrationRecord.zeroInclusion !== 'required'
  ) {
    throw new Error(
      'Phase 7 static-8bit calibration currently requires zero-inclusive ranges.',
    );
  }

  if (
    calibrationRecord.roundingMode !== undefined &&
    calibrationRecord.roundingMode !== 'nearest-even'
  ) {
    throw new Error(
      'Phase 7 static-8bit calibration currently supports nearest-even rounding only.',
    );
  }

  return {
    source: 'external',
    packetId:
      typeof calibrationRecord.packetId === 'string'
        ? calibrationRecord.packetId
        : undefined,
    sampleCount: resolveCalibrationSampleCount(calibrationRecord.sampleCount),
    layerTargets: resolvedLayerTargets,
    weightRangePolicy: 'min-max',
    zeroInclusion: 'required',
    activationSymmetry:
      calibrationRecord.activationSymmetry === 'symmetric'
        ? 'symmetric'
        : 'asymmetric',
    weightSymmetry:
      calibrationRecord.weightSymmetry === 'asymmetric'
        ? 'asymmetric'
        : 'symmetric',
    roundingMode: 'nearest-even',
  };
}

function resolveCalibrationLayerTargets(
  layerTargetsValue: unknown,
  staticTargets: Array<'dense' | 'conv'>,
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxQuantizationCalibrationLayerTarget[] {
  if (!Array.isArray(layerTargetsValue) || layerTargetsValue.length === 0) {
    throw new Error(
      'Phase 7 static-8bit quantization requires explicit calibration layer targets.',
    );
  }

  const resolvedLayerTargets = layerTargetsValue
    .map((layerTargetValue) =>
      resolveCalibrationLayerTarget(
        layerTargetValue,
        staticTargets,
        sourceOptions,
        networkLayerCount,
      ),
    )
    .toSorted((leftLayerTarget, rightLayerTarget) => {
      if (leftLayerTarget.layerIndex !== rightLayerTarget.layerIndex) {
        return leftLayerTarget.layerIndex - rightLayerTarget.layerIndex;
      }

      return leftLayerTarget.target.localeCompare(rightLayerTarget.target);
    });
  const layerTargetKeys = resolvedLayerTargets.map(
    (layerTarget) => `${layerTarget.target}:${layerTarget.layerIndex}`,
  );
  const uniqueLayerTargetKeys = new Set(layerTargetKeys);

  if (uniqueLayerTargetKeys.size !== layerTargetKeys.length) {
    throw new Error(
      'Phase 7 static-8bit calibration layer targets must be unique per operator family and layer index.',
    );
  }

  return resolvedLayerTargets;
}

function resolveCalibrationLayerTarget(
  layerTargetValue: unknown,
  staticTargets: Array<'dense' | 'conv'>,
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxQuantizationCalibrationLayerTarget {
  if (!layerTargetValue || typeof layerTargetValue !== 'object') {
    throw new Error(
      'Phase 7 static-8bit calibration layer targets must be objects.',
    );
  }

  const layerTargetRecord = layerTargetValue as {
    target?: unknown;
    layerIndex?: unknown;
    inputRange?: unknown;
    outputRange?: unknown;
  };
  const target = layerTargetRecord.target;
  const layerIndex = layerTargetRecord.layerIndex;

  if (target !== 'dense' && target !== 'conv') {
    throw new Error(
      'Phase 7 static-8bit calibration layer targets must name a dense or conv operator family.',
    );
  }

  if (!staticTargets.includes(target)) {
    throw new Error(
      'Phase 7 static-8bit calibration layer targets must stay within the requested operator families.',
    );
  }

  if (
    typeof layerIndex !== 'number' ||
    !Number.isInteger(layerIndex) ||
    layerIndex <= 0 ||
    layerIndex >= networkLayerCount
  ) {
    throw new Error(
      'Phase 7 static-8bit calibration layer targets must reference a valid non-input export layer.',
    );
  }

  const hasResolvedConvMapping =
    sourceOptions.conv2dMappings?.some(
      (convMapping) => convMapping.layerIndex === layerIndex,
    ) ?? false;

  if (target === 'conv' && !hasResolvedConvMapping) {
    throw new Error(
      'Phase 7 static-8bit conv calibration requires a resolved Conv mapping for each target layer.',
    );
  }

  if (target === 'dense' && hasResolvedConvMapping) {
    throw new Error(
      'Phase 7 static-8bit dense calibration cannot target layers emitted as Conv operators.',
    );
  }

  return {
    target,
    layerIndex,
    inputRange: resolveCalibrationRange(
      layerTargetRecord.inputRange,
      'input',
      layerIndex,
    ),
    outputRange: resolveCalibrationRange(
      layerTargetRecord.outputRange,
      'output',
      layerIndex,
    ),
  };
}

function resolveCalibrationRange(
  rangeValue: unknown,
  rangeLabel: 'input' | 'output',
  layerIndex: number,
): OnnxQuantizationCalibrationRange {
  if (!rangeValue || typeof rangeValue !== 'object') {
    throw new Error(
      `Phase 7 static-8bit calibration layer ${layerIndex} requires a ${rangeLabel} range object.`,
    );
  }

  const rangeRecord = rangeValue as { min?: unknown; max?: unknown };

  if (
    typeof rangeRecord.min !== 'number' ||
    !Number.isFinite(rangeRecord.min) ||
    typeof rangeRecord.max !== 'number' ||
    !Number.isFinite(rangeRecord.max)
  ) {
    throw new Error(
      `Phase 7 static-8bit calibration layer ${layerIndex} requires finite ${rangeLabel} min and max values.`,
    );
  }

  if (rangeRecord.min >= rangeRecord.max) {
    throw new Error(
      `Phase 7 static-8bit calibration layer ${layerIndex} requires min strictly less than max for the ${rangeLabel} range.`,
    );
  }

  return {
    min: rangeRecord.min,
    max: rangeRecord.max,
  };
}

function resolveCalibrationSampleCount(
  sampleCountValue: unknown,
): number | undefined {
  if (sampleCountValue === undefined) {
    return undefined;
  }

  if (
    typeof sampleCountValue !== 'number' ||
    !Number.isInteger(sampleCountValue) ||
    sampleCountValue <= 0
  ) {
    throw new Error(
      'Phase 7 static-8bit calibration sample counts must be positive integers when provided.',
    );
  }

  return sampleCountValue;
}

function validatePrecisionAndQuantizationComposition(
  resolvedPrecision: OnnxResolvedPrecisionOptions,
  resolvedQuantization: OnnxResolvedQuantizationOptions,
): void {
  if (
    resolvedPrecision.mode === 'storage-fp16' &&
    resolvedQuantization.requested
  ) {
    throw new Error(
      'Phase 7 storage-fp16 precision cannot be combined with quantization until an explicit composition contract lands.',
    );
  }
}
