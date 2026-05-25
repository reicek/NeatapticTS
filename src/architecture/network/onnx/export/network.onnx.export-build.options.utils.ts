import type {
  OnnxBuildResolvedOptions,
  OnnxExportOptions,
  OnnxQuantizationCalibrationLayerTarget,
  OnnxQuantizationCalibrationRange,
  OnnxResolvedPrecisionOptions,
  OnnxResolvedQuantizationCalibrationOptions,
  OnnxResolvedQuantizationOptions,
} from './network.onnx.export.types';

/**
 * Resolve export options with defaults required by model construction.
 *
 * @param sourceOptions Raw export options.
 * @returns Resolved options used by this builder.
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
 * @param sourceOptions Raw export options.
 * @returns Normalized precision packet.
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
 * @param sourceOptions Raw export options.
 * @returns Normalized quantization packet.
 */
export function resolveQuantizationOptions(
  sourceOptions: OnnxExportOptions,
  networkLayerCount: number,
): OnnxResolvedQuantizationOptions {
  if (!sourceOptions.quantization) {
    return {
      requested: false,
      mode: null,
      fallbackReasons: [],
    };
  }

  const quantizationPacket = sourceOptions.quantization as {
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

  if (quantizationPacket.mode === 'dynamic-uint8') {
    const dynamicTarget = quantizationPacket.target ?? 'dense';
    if (dynamicTarget !== 'dense') {
      throw new Error(
        'Phase 7 dynamic quantization currently supports dense targets only.',
      );
    }

    const dynamicRepresentation =
      quantizationPacket.representation ?? 'metadata-only';
    if (
      dynamicRepresentation !== 'DynamicQuantizeLinear' &&
      dynamicRepresentation !== 'metadata-only'
    ) {
      throw new Error(
        'Phase 7 dynamic quantization must use DynamicQuantizeLinear or metadata-only representation.',
      );
    }

    return {
      requested: true,
      mode: 'dynamic-uint8',
      target: 'dense',
      representation: dynamicRepresentation,
      fallbackReasons: [],
    };
  }

  if (quantizationPacket.mode === 'static-8bit') {
    const staticTargets = Array.isArray(quantizationPacket.targets)
      ? quantizationPacket.targets.filter(
          (target): target is 'dense' | 'conv' =>
            target === 'dense' || target === 'conv',
        )
      : [];
    const calibrationPacket = quantizationPacket.calibration;
    const resolvedWeightGranularity =
      quantizationPacket.weightGranularity === 'per-output-channel'
        ? 'per-output-channel'
        : 'per-tensor';

    if (!staticTargets.length) {
      throw new Error(
        'Phase 7 static-8bit quantization requires at least one dense or conv target.',
      );
    }

    if (
      !calibrationPacket ||
      typeof calibrationPacket !== 'object' ||
      (calibrationPacket as { source?: unknown }).source !== 'external'
    ) {
      throw new Error(
        'Phase 7 static-8bit quantization requires an external calibration packet.',
      );
    }

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

  throw new Error(
    'Phase 7 dynamic quantization must use the documented dynamic-uint8 lane.',
  );
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
