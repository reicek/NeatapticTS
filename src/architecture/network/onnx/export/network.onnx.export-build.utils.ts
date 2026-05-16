import type Network from '../../network';
import type NeatapticNode from '../../../node';
import type { OnnxModel } from '../schema/network.onnx.schema.types';
import {
  createFloat16StoragePayload,
  ONNX_FLOAT_DATA_TYPE,
} from '../schema/network.onnx.schema.tensor-data.utils';
import type {
  OnnxBuildResolvedOptions,
  OnnxExportOptions,
  OnnxGraphDimensions,
  OnnxLayerEmissionContext,
  OnnxLayerEmissionResult,
  OnnxPostProcessingContext,
  OnnxQuantizationCalibrationLayerTarget,
  OnnxQuantizationCalibrationRange,
  OnnxResolvedPrecisionOptions,
  OnnxResolvedQuantizationCalibrationOptions,
  OnnxResolvedQuantizationOptions,
  OnnxRecurrentCollectionContext,
} from './network.onnx.export.types';
import {
  applyModelMetadata,
  collectRecurrentLayerIndices,
  createBaseModel,
  createGraphDimensions,
} from './network.onnx.export-setup.utils';
import { emitLayerGraph } from './layers/network.onnx.export-layer-graph.utils';
import {
  emitFusedRecurrentHeuristics,
  finalizeExportMetadata,
} from './network.onnx.export-postprocess.utils';
import { emitShadowAttentionMappings } from './network.onnx.export-attention.utils';
import { pruneIdentityActivationNodes } from './network.onnx.export-optimization.utils';
import { validateOnnxModelShapes } from './network.onnx.export-shape-validation.utils';

const STORAGE_FP16_CAST_SUFFIX = '_fp32';
const STORAGE_FP16_FLOAT_CAST_TARGET = 1;
const ONNX_UINT8_DATA_TYPE = 2;
const ONNX_INT8_DATA_TYPE = 3;
const STORAGE_FP16_ELIGIBLE_INITIALIZER_PATTERNS = [
  /^W\d+$/,
  /^B\d+$/,
  /^W\d+_n\d+$/,
  /^B\d+_n\d+$/,
  /^ConvW\d+$/,
  /^ConvB\d+$/,
];

/** Export-owned plan for lowering one dense layer into the current qlinear subset. */
type StaticDenseLoweringPlan = {
  layerIndex: number;
  activationNode: OnnxModel['graph']['node'][number];
  hasBiasBridge: boolean;
};

/**
 * Construct ONNX graph (initializers + nodes) from validated layered network structure.
 *
 * @param network Source network (retained for API compatibility).
 * @param layers Layered nodes including input and output layers.
 * @param options Export options.
 * @returns ONNX model.
 */
export function buildOnnxModel(
  network: Network,
  layers: NeatapticNode[][],
  options?: OnnxExportOptions,
): OnnxModel {
  // Step 1: Resolve stable export defaults and input options.
  const sourceOptions = options ?? {};
  const resolvedOptions = resolveBuildOptions(sourceOptions);
  void network;

  // Step 2: Initialize base ONNX model with graph dimensions and metadata.
  const model = createInitializedModel(layers, resolvedOptions);

  // Step 3: Collect recurrent layer indices required during graph emission.
  const recurrentLayerIndices = collectRecurrentIndices({
    model,
    layers,
    options: sourceOptions,
    batchDimension: resolvedOptions.batchDimension,
  });

  // Step 4: Emit all non-input layers and collect hidden-size metadata.
  const layerEmissionResult = emitNonInputLayers({
    model,
    layers,
    options: sourceOptions,
    recurrentLayerIndices,
    batchDimension: resolvedOptions.batchDimension,
    legacyNodeOrdering: resolvedOptions.legacyNodeOrdering,
  });

  // Step 5: Apply post-processing and finalize export metadata.
  applyPostProcessing({
    model,
    layers,
    options: sourceOptions,
    includeMetadata: resolvedOptions.includeMetadata,
    recurrentLayerIndices,
    layerEmissionResult,
  });

  // Step 6: Emit calibration-owned quantization parameters for the narrow supported subset.
  applyStaticQuantizationCalibrationPostProcessing(
    model,
    sourceOptions,
    resolvedOptions.quantization,
    recurrentLayerIndices,
  );

  // Step 7: Lower the narrow same-family dense subset into qlinear affine nodes.
  applyStaticDenseQuantizationPostProcessing(
    model,
    sourceOptions,
    resolvedOptions.quantization,
    recurrentLayerIndices,
  );

  // Step 8: Apply storage-fp16 rewrites only for the narrow supported subset.
  applyStorageFp16PostProcessing(
    model,
    sourceOptions,
    recurrentLayerIndices,
  );

  // Step 9: Prune exact Identity activation nodes before validation.
  pruneIdentityActivationNodes(model);

  // Step 10: Validate exporter-owned tensor dimensions before returning.
  validateOnnxModelShapes(model);

  // Step 11: Return fully constructed ONNX model.
  return model;

  /**
   * Resolve export options with defaults required by model construction.
   *
   * @param sourceOptions Raw export options.
   * @returns Resolved options used by this builder.
   */
  function resolveBuildOptions(
    sourceOptions: OnnxExportOptions,
  ): OnnxBuildResolvedOptions {
    // Step 1: Resolve caller-provided values with stable defaults.
    const resolvedPrecision = resolvePrecisionOptions(sourceOptions);
    const resolvedQuantization = resolveQuantizationOptions(sourceOptions);

    // Step 2: Reject unsupported cross-packet combinations before graph emission.
    validatePrecisionAndQuantizationComposition(
      resolvedPrecision,
      resolvedQuantization,
    );

    // Step 3: Resolve stable build defaults and normalized Phase 7 packets.
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
  function resolvePrecisionOptions(
    sourceOptions: OnnxExportOptions,
  ): OnnxResolvedPrecisionOptions {
    // Step 1: Treat absent precision as the canonical float32 baseline.
    if (!sourceOptions.precision) {
      return {
        requested: false,
        mode: 'float32',
        metadata: false,
      };
    }

    // Step 2: Validate the requested precision mode before returning it.
    const precisionMode = sourceOptions.precision.mode ?? 'float32';
    if (precisionMode !== 'float32' && precisionMode !== 'storage-fp16') {
      throw new Error(
        'Phase 7 precision mode must be float32 or storage-fp16.',
      );
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
  function resolveQuantizationOptions(
    sourceOptions: OnnxExportOptions,
  ): OnnxResolvedQuantizationOptions {
    // Step 1: Treat absent quantization as no quantized export request.
    if (!sourceOptions.quantization) {
      return {
        requested: false,
        mode: null,
        fallbackReasons: [],
      };
    }

    // Step 2: Validate the requested quantization mode against the Phase 7 lane.
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

  /**
   * Resolve static calibration policy defaults and validate explicit layer targets.
   *
   * @param calibrationPacket Raw calibration packet.
   * @param staticTargets Requested operator families.
   * @param weightGranularity Resolved weight granularity.
   * @returns Normalized calibration packet.
   */
  function resolveStaticCalibrationOptions(
    calibrationPacket: unknown,
    staticTargets: Array<'dense' | 'conv'>,
    weightGranularity: 'per-tensor' | 'per-output-channel',
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
    );

    if (
      weightGranularity === 'per-output-channel' &&
      resolvedLayerTargets.some(
        (layerTarget) => layerTarget.target === 'dense',
      )
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

  /**
   * Resolve and validate the explicit layer-target list carried by a calibration packet.
   *
   * @param layerTargetsValue Raw layer-target payload.
   * @param staticTargets Requested operator families.
   * @returns Sorted, validated calibration layer targets.
   */
  function resolveCalibrationLayerTargets(
    layerTargetsValue: unknown,
    staticTargets: Array<'dense' | 'conv'>,
  ): OnnxQuantizationCalibrationLayerTarget[] {
    if (!Array.isArray(layerTargetsValue) || layerTargetsValue.length === 0) {
      throw new Error(
        'Phase 7 static-8bit quantization requires explicit calibration layer targets.',
      );
    }

    const resolvedLayerTargets = layerTargetsValue
      .map((layerTargetValue) =>
        resolveCalibrationLayerTarget(layerTargetValue, staticTargets),
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

  /**
   * Resolve one calibration layer target against the current exportable layer stack.
   *
   * @param layerTargetValue Raw layer-target payload.
   * @param staticTargets Requested operator families.
   * @returns Normalized calibration layer target.
   */
  function resolveCalibrationLayerTarget(
    layerTargetValue: unknown,
    staticTargets: Array<'dense' | 'conv'>,
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
      layerIndex >= layers.length
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

  /**
   * Resolve one calibration range and reject unstable min/max windows.
   *
   * @param rangeValue Raw range payload.
   * @param rangeLabel Human-readable range label.
   * @param layerIndex Target layer index.
   * @returns Normalized calibration range.
   */
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

  /**
   * Resolve optional calibration sample counts to a stable positive integer.
   *
   * @param sampleCountValue Raw sample-count payload.
   * @returns Normalized sample count.
   */
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

  /**
   * Reject unsupported precision-plus-quantization composition before graph emission.
   *
   * @param resolvedPrecision Normalized precision packet.
   * @param resolvedQuantization Normalized quantization packet.
   * @returns Nothing.
   */
  function validatePrecisionAndQuantizationComposition(
    resolvedPrecision: OnnxResolvedPrecisionOptions,
    resolvedQuantization: OnnxResolvedQuantizationOptions,
  ): void {
    // Step 1: Reject composition until the exporter owns an explicit mixed contract.
    if (
      resolvedPrecision.mode === 'storage-fp16' &&
      resolvedQuantization.requested
    ) {
      throw new Error(
        'Phase 7 storage-fp16 precision cannot be combined with quantization until an explicit composition contract lands.',
      );
    }
  }

  /**
   * Emit exporter-owned calibration parameters for the narrow supported static subset.
   *
   * @param model Built ONNX model.
   * @param sourceOptions Raw export options.
   * @param resolvedQuantization Normalized quantization packet.
   * @param recurrentLayerIndices Collected recurrent layer indices.
   * @returns Nothing.
   */
  function applyStaticQuantizationCalibrationPostProcessing(
    model: OnnxModel,
    sourceOptions: OnnxExportOptions,
    resolvedQuantization: OnnxResolvedQuantizationOptions,
    recurrentLayerIndices: number[],
  ): void {
    if (
      !shouldEmitStaticQuantizationParameters(
        sourceOptions,
        resolvedQuantization,
        recurrentLayerIndices,
      )
    ) {
      return;
    }

    const parameterInitializers = resolvedQuantization.calibration.layerTargets.flatMap(
      (layerTarget) =>
        createStaticQuantizationParameterInitializers(
          model,
          resolvedQuantization,
          layerTarget,
        ),
    );
    model.graph.initializer.push(...parameterInitializers);
  }

  /**
   * Determine whether the current export request may emit calibration parameters.
   *
   * @param sourceOptions Raw export options.
   * @param resolvedQuantization Normalized quantization packet.
   * @param recurrentLayerIndices Collected recurrent layer indices.
   * @returns True when the current request fits the narrow supported subset.
   */
  function shouldEmitStaticQuantizationParameters(
    sourceOptions: OnnxExportOptions,
    resolvedQuantization: OnnxResolvedQuantizationOptions,
    recurrentLayerIndices: number[],
  ): resolvedQuantization is Extract<
    OnnxResolvedQuantizationOptions,
    { mode: 'static-8bit' }
  > {
    if (resolvedQuantization.mode !== 'static-8bit') {
      return false;
    }

    if (recurrentLayerIndices.length > 0) {
      return false;
    }

    if (sourceOptions.allowMixedActivations || sourceOptions.allowPartialConnectivity) {
      return false;
    }

    if (
      (sourceOptions.concatMappings?.length ?? 0) > 0 ||
      (sourceOptions.attentionMappings?.length ?? 0) > 0
    ) {
      return false;
    }

    return true;
  }

  /**
   * Lower the narrow first-wave dense subset into exporter-owned qlinear affine nodes.
   *
   * @param model Built ONNX model.
   * @param sourceOptions Raw export options.
   * @param resolvedQuantization Normalized quantization packet.
   * @param recurrentLayerIndices Collected recurrent layer indices.
   * @returns Nothing.
   */
  function applyStaticDenseQuantizationPostProcessing(
    model: OnnxModel,
    sourceOptions: OnnxExportOptions,
    resolvedQuantization: OnnxResolvedQuantizationOptions,
    recurrentLayerIndices: number[],
  ): void {
    if (
      !shouldApplyStaticDenseQuantizationLowering(
        sourceOptions,
        resolvedQuantization,
        recurrentLayerIndices,
      )
    ) {
      return;
    }

    const lowerableDenseLayerPlans =
      resolvedQuantization.calibration.layerTargets
        .filter((layerTarget) => layerTarget.target === 'dense')
        .map((layerTarget) =>
          resolveStaticDenseLoweringPlan(model, layerTarget.layerIndex),
        );

    model.graph.initializer.push(
      ...lowerableDenseLayerPlans.map((loweringPlan) =>
        createQuantizedDenseWeightInitializer(
          model,
          resolvedQuantization,
          loweringPlan.layerIndex,
        ),
      ),
    );

    const lowerableDenseLayerPlansByLayerIndex = new Map(
      lowerableDenseLayerPlans.map((loweringPlan) => [
        loweringPlan.layerIndex,
        loweringPlan,
      ]),
    );

    model.graph.node = model.graph.node.flatMap((graphNode) =>
      rewriteDenseGraphNodeForStaticQuantization(
        graphNode,
        lowerableDenseLayerPlansByLayerIndex,
      ),
    );
  }

  /**
   * Determine whether the current export request fits the first qlinear dense subset.
   *
   * @param sourceOptions Raw export options.
   * @param resolvedQuantization Normalized quantization packet.
   * @param recurrentLayerIndices Collected recurrent layer indices.
   * @returns True when qlinear dense lowering may be attempted.
   */
  function shouldApplyStaticDenseQuantizationLowering(
    sourceOptions: OnnxExportOptions,
    resolvedQuantization: OnnxResolvedQuantizationOptions,
    recurrentLayerIndices: number[],
  ): resolvedQuantization is Extract<
    OnnxResolvedQuantizationOptions,
    { mode: 'static-8bit' }
  > {
    if (
      !shouldEmitStaticQuantizationParameters(
        sourceOptions,
        resolvedQuantization,
        recurrentLayerIndices,
      )
    ) {
      return false;
    }

    if (resolvedQuantization.representation !== 'qlinear') {
      return false;
    }

    if ((sourceOptions.conv2dMappings?.length ?? 0) > 0) {
      return false;
    }

    if ((sourceOptions.pool2dMappings?.length ?? 0) > 0) {
      return false;
    }

    return true;
  }

  /**
   * Resolve one export-owned lowering plan for the current qlinear dense subset.
   *
   * @param model Built ONNX model.
   * @param layerIndex Dense export layer index.
   * @returns Lowering plan for one exporter-owned dense layer pair.
   */
  function resolveStaticDenseLoweringPlan(
    model: OnnxModel,
    layerIndex: number,
  ): StaticDenseLoweringPlan {
    findGraphNodeByName(model, `gemm_l${layerIndex}`)!;
    const activationNode = findGraphNodeByName(model, `act_l${layerIndex}`)!;
    findInitializerByName(model, `W${layerIndex - 1}`)!;
    const biasInitializer = findInitializerByName(model, `B${layerIndex - 1}`)!;

    return {
      layerIndex,
      activationNode,
      hasBiasBridge: hasNonZeroDenseBias(biasInitializer),
    };
  }

  /**
   * Determine whether one dense bias initializer requires an explicit float-domain bridge.
   *
   * @param biasInitializer Dense bias initializer.
   * @returns True when any dense bias entry is nonzero.
   */
  function hasNonZeroDenseBias(
    biasInitializer: NonNullable<ReturnType<typeof findInitializerByName>>,
  ): boolean {
    return biasInitializer.float_data.some((biasValue) => biasValue !== 0);
  }

  /**
  * Rewrite one supported dense Gemm node into qlinear affine nodes and drop the paired dense activation node.
   *
   * @param graphNode Current graph node.
   * @param lowerableDenseLayerPlansByLayerIndex Lowerable dense layer plans.
   * @returns Replacement node list.
   */
  function rewriteDenseGraphNodeForStaticQuantization(
    graphNode: OnnxModel['graph']['node'][number],
    lowerableDenseLayerPlansByLayerIndex: Map<number, StaticDenseLoweringPlan>,
  ): OnnxModel['graph']['node'] {
    const lowerableLayerIndex = [...lowerableDenseLayerPlansByLayerIndex.keys()].find(
      (layerIndex) =>
        graphNode.name === `gemm_l${layerIndex}` ||
        graphNode.name === `act_l${layerIndex}`,
    );

    if (lowerableLayerIndex === undefined) {
      return [graphNode];
    }

    const loweringPlan = lowerableDenseLayerPlansByLayerIndex.get(
      lowerableLayerIndex,
    )!;

    if (graphNode.name === `act_l${loweringPlan.layerIndex}`) {
      return [];
    }

    return createStaticDenseQuantizedNodes(graphNode, loweringPlan);
  }

  /**
   * Create the qlinear node sequence that replaces one dense Gemm plus activation pair.
   *
   * @param gemmNode Dense Gemm node being replaced.
   * @param loweringPlan Dense lowering plan.
   * @returns Quantize, qlinear, dequantize, and optional bridge nodes in deterministic order.
   */
  function createStaticDenseQuantizedNodes(
    gemmNode: OnnxModel['graph']['node'][number],
    loweringPlan: StaticDenseLoweringPlan,
  ): OnnxModel['graph']['node'] {
    const quantizedInputName = `QuantDenseInput_l${loweringPlan.layerIndex}`;
    const quantizedAffineOutputName = `QuantDenseAffine_l${loweringPlan.layerIndex}`;
    const dequantizedAffineOutputName = resolveStaticDenseDequantizedOutputName(
      loweringPlan,
    );
    const activationInputName = resolveStaticDenseActivationInputName(
      loweringPlan,
      dequantizedAffineOutputName,
    );

    return [
      {
        op_type: 'QuantizeLinear',
        input: [
          gemmNode.input[0],
          `QuantDenseInputScale_l${loweringPlan.layerIndex}`,
          `QuantDenseInputZeroPoint_l${loweringPlan.layerIndex}`,
        ],
        output: [quantizedInputName],
        name: `quantize_input_l${loweringPlan.layerIndex}`,
      },
      {
        op_type: 'QLinearMatMul',
        input: [
          quantizedInputName,
          `QuantDenseInputScale_l${loweringPlan.layerIndex}`,
          `QuantDenseInputZeroPoint_l${loweringPlan.layerIndex}`,
          `QuantDenseWeight_l${loweringPlan.layerIndex}`,
          `QuantDenseWeightScale_l${loweringPlan.layerIndex}`,
          `QuantDenseWeightZeroPoint_l${loweringPlan.layerIndex}`,
          `QuantDenseOutputScale_l${loweringPlan.layerIndex}`,
          `QuantDenseOutputZeroPoint_l${loweringPlan.layerIndex}`,
        ],
        output: [quantizedAffineOutputName],
        name: `qlinear_matmul_l${loweringPlan.layerIndex}`,
      },
      {
        op_type: 'DequantizeLinear',
        input: [
          quantizedAffineOutputName,
          `QuantDenseOutputScale_l${loweringPlan.layerIndex}`,
          `QuantDenseOutputZeroPoint_l${loweringPlan.layerIndex}`,
        ],
        output: [dequantizedAffineOutputName],
        name: `dequantize_output_l${loweringPlan.layerIndex}`,
      },
      ...(loweringPlan.hasBiasBridge
        ? [
            createStaticDenseBiasBridgeNode(
              loweringPlan,
              dequantizedAffineOutputName,
            ),
          ]
        : []),
      ...(shouldEmitStaticDenseActivationNode(loweringPlan)
        ? [
            createStaticDenseActivationNode(
              loweringPlan,
              activationInputName,
            ),
          ]
        : []),
    ];
  }

  /**
   * Resolve the float-domain tensor name produced immediately after dequantization.
   *
   * @param loweringPlan Dense lowering plan.
   * @returns Dequantized affine output name.
   */
  function resolveStaticDenseDequantizedOutputName(
    loweringPlan: StaticDenseLoweringPlan,
  ): string {
    if (
      !loweringPlan.hasBiasBridge &&
      !shouldEmitStaticDenseActivationNode(loweringPlan)
    ) {
      return `Layer_${loweringPlan.layerIndex}`;
    }

    return `DenseAffineFloat_l${loweringPlan.layerIndex}`;
  }

  /**
   * Resolve the tensor name that should feed the preserved post-affine activation.
   *
   * @param loweringPlan Dense lowering plan.
   * @param dequantizedAffineOutputName Dequantized affine output name.
   * @returns Activation input tensor name.
   */
  function resolveStaticDenseActivationInputName(
    loweringPlan: StaticDenseLoweringPlan,
    dequantizedAffineOutputName: string,
  ): string {
    if (!loweringPlan.hasBiasBridge) {
      return dequantizedAffineOutputName;
    }

    if (!shouldEmitStaticDenseActivationNode(loweringPlan)) {
      return `Layer_${loweringPlan.layerIndex}`;
    }

    return `DenseBiasedFloat_l${loweringPlan.layerIndex}`;
  }

  /**
   * Determine whether the paired dense activation node should survive after qlinear lowering.
   *
   * @param loweringPlan Dense lowering plan.
   * @returns True when the activation node is not Identity.
   */
  function shouldEmitStaticDenseActivationNode(
    loweringPlan: StaticDenseLoweringPlan,
  ): boolean {
    return loweringPlan.activationNode.op_type !== 'Identity';
  }

  /**
   * Create the explicit float-domain bias bridge required after qlinear affine lowering.
   *
   * @param loweringPlan Dense lowering plan.
   * @param dequantizedAffineOutputName Dequantized affine output name.
   * @returns Bias-bridge Add node.
   */
  function createStaticDenseBiasBridgeNode(
    loweringPlan: StaticDenseLoweringPlan,
    dequantizedAffineOutputName: string,
  ): OnnxModel['graph']['node'][number] {
    return {
      op_type: 'Add',
      input: [dequantizedAffineOutputName, `B${loweringPlan.layerIndex - 1}`],
      output: [resolveStaticDenseActivationInputName(loweringPlan, dequantizedAffineOutputName)],
      name: `bias_add_l${loweringPlan.layerIndex}`,
    };
  }

  /**
   * Recreate the exporter-owned post-affine activation node after the qlinear bridge.
   *
   * @param loweringPlan Dense lowering plan.
   * @param activationInputName Float-domain activation input name.
   * @returns Preserved activation node.
   */
  function createStaticDenseActivationNode(
    loweringPlan: StaticDenseLoweringPlan,
    activationInputName: string,
  ): OnnxModel['graph']['node'][number] {
    return {
      op_type: loweringPlan.activationNode.op_type,
      input: [activationInputName],
      output: [`Layer_${loweringPlan.layerIndex}`],
      name: loweringPlan.activationNode.name,
      attributes: loweringPlan.activationNode.attributes,
    };
  }

  /**
   * Create one quantized dense weight initializer for qlinear affine lowering.
   *
   * @param model Built ONNX model.
   * @param resolvedQuantization Normalized static quantization packet.
   * @param layerIndex Dense export layer index.
   * @returns Quantized dense weight initializer.
   */
  function createQuantizedDenseWeightInitializer(
    model: OnnxModel,
    resolvedQuantization: Extract<
      OnnxResolvedQuantizationOptions,
      { mode: 'static-8bit' }
    >,
    layerIndex: number,
  ) {
    const weightInitializer = findInitializerByName(model, `W${layerIndex - 1}`)!;
    const transposedWeightValues = transposeDenseWeightValues(weightInitializer);
    const weightScaleValue = readRequiredScalarFloatInitializer(
      model,
      `QuantDenseWeightScale_l${layerIndex}`,
    );
    const weightZeroPointValue = readRequiredScalarIntegerInitializer(
      model,
      `QuantDenseWeightZeroPoint_l${layerIndex}`,
    );

    return {
      name: `QuantDenseWeight_l${layerIndex}`,
      data_type:
        resolvedQuantization.weightEncoding === 'uint8'
          ? ONNX_UINT8_DATA_TYPE
          : ONNX_INT8_DATA_TYPE,
      dims: [weightInitializer.dims[1]!, weightInitializer.dims[0]!],
      float_data: [],
      int32_data: quantizeTensorValues(
        transposedWeightValues,
        weightScaleValue,
        weightZeroPointValue,
        resolvedQuantization.weightEncoding,
        resolvedQuantization.calibration.roundingMode,
      ),
    };
  }

  /**
   * Transpose one dense weight initializer from Gemm-owned [out, in] order into MatMul-owned [in, out] order.
   *
   * @param weightInitializer Dense float weight initializer.
   * @returns Transposed dense weight payload.
   */
  function transposeDenseWeightValues(
    weightInitializer: NonNullable<
      ReturnType<typeof findInitializerByName>
    >,
  ): number[] {
    const outputCount = weightInitializer.dims[0]!;
    const inputCount = weightInitializer.dims[1]!;

    return Array.from({ length: inputCount * outputCount }, (_unused, valueIndex) => {
      const inputIndex = Math.floor(valueIndex / outputCount);
      const outputIndex = valueIndex % outputCount;
      return weightInitializer.float_data[outputIndex * inputCount + inputIndex]!;
    });
  }

  /**
   * Find one graph node by its deterministic name.
   *
   * @param model Built ONNX model.
   * @param nodeName Graph node name.
   * @returns Matching graph node when present.
   */
  function findGraphNodeByName(
    model: OnnxModel,
    nodeName: string,
  ): OnnxModel['graph']['node'][number] | undefined {
    return model.graph.node.find((graphNode) => graphNode.name === nodeName);
  }

  /**
   * Find one initializer by its deterministic name.
   *
   * @param model Built ONNX model.
   * @param initializerName Initializer name.
   * @returns Matching initializer when present.
   */
  function findInitializerByName(
    model: OnnxModel,
    initializerName: string,
  ) {
    return model.graph.initializer.find(
      (initializerEntry) => initializerEntry.name === initializerName,
    );
  }

  /**
   * Read one required scalar float initializer.
   *
   * @param model Built ONNX model.
   * @param initializerName Initializer name.
   * @returns Scalar float value.
   */
  function readRequiredScalarFloatInitializer(
    model: OnnxModel,
    initializerName: string,
  ): number {
    const initializerEntry = findInitializerByName(model, initializerName)!;
    return initializerEntry.float_data[0]!;
  }

  /**
   * Read one required scalar integer initializer.
   *
   * @param model Built ONNX model.
   * @param initializerName Initializer name.
   * @returns Scalar integer value.
   */
  function readRequiredScalarIntegerInitializer(
    model: OnnxModel,
    initializerName: string,
  ): number {
    const initializerEntry = findInitializerByName(model, initializerName)!;
    return initializerEntry.int32_data![0]!;
  }

  /**
   * Quantize one float tensor payload into exporter-owned int8 or uint8 storage.
   *
   * @param floatValues Source float payload.
   * @param scaleValue Scalar quantization scale.
   * @param zeroPointValue Scalar quantization zero point.
   * @param encoding Integer encoding family.
   * @param roundingMode Rounding policy.
   * @returns Quantized integer payload.
   */
  function quantizeTensorValues(
    floatValues: number[],
    scaleValue: number,
    zeroPointValue: number,
    encoding: 'uint8' | 'int8',
    roundingMode: 'nearest-even',
  ): number[] {
    const quantizedMinimum = encoding === 'uint8' ? 0 : -128;
    const quantizedMaximum = encoding === 'uint8' ? 255 : 127;

    return floatValues.map((floatValue) =>
      clampInteger(
        roundQuantizedValue(floatValue / scaleValue + zeroPointValue, roundingMode),
        quantizedMinimum,
        quantizedMaximum,
      ),
    );
  }

  /**
   * Create the six scale or zero-point tensors for one calibrated operator target.
   *
   * @param model Built ONNX model.
   * @param resolvedQuantization Normalized static quantization packet.
   * @param layerTarget One calibrated operator target.
   * @returns Ordered parameter initializers.
   */
  function createStaticQuantizationParameterInitializers(
    model: OnnxModel,
    resolvedQuantization: Extract<
      OnnxResolvedQuantizationOptions,
      { mode: 'static-8bit' }
    >,
    layerTarget: OnnxQuantizationCalibrationLayerTarget,
  ) {
    const inputParameters = resolveRangeQuantizationParameters(
      layerTarget.inputRange,
      resolvedQuantization.activationEncoding,
      resolvedQuantization.calibration.activationSymmetry,
      resolvedQuantization.calibration.zeroInclusion,
      resolvedQuantization.calibration.roundingMode,
    );
    const outputParameters = resolveRangeQuantizationParameters(
      layerTarget.outputRange,
      resolvedQuantization.activationEncoding,
      resolvedQuantization.calibration.activationSymmetry,
      resolvedQuantization.calibration.zeroInclusion,
      resolvedQuantization.calibration.roundingMode,
    );
    const weightParameters = resolveWeightQuantizationParameters(
      model,
      resolvedQuantization,
      layerTarget,
    );

    return [
      createScaleInitializer(
        buildQuantizationParameterName(layerTarget, 'InputScale'),
        inputParameters.scales,
      ),
      createZeroPointInitializer(
        buildQuantizationParameterName(layerTarget, 'InputZeroPoint'),
        resolvedQuantization.activationEncoding,
        inputParameters.zeroPoints,
      ),
      createScaleInitializer(
        buildQuantizationParameterName(layerTarget, 'WeightScale'),
        weightParameters.scales,
      ),
      createZeroPointInitializer(
        buildQuantizationParameterName(layerTarget, 'WeightZeroPoint'),
        resolvedQuantization.weightEncoding,
        weightParameters.zeroPoints,
      ),
      createScaleInitializer(
        buildQuantizationParameterName(layerTarget, 'OutputScale'),
        outputParameters.scales,
      ),
      createZeroPointInitializer(
        buildQuantizationParameterName(layerTarget, 'OutputZeroPoint'),
        resolvedQuantization.activationEncoding,
        outputParameters.zeroPoints,
      ),
    ];
  }

  /**
   * Resolve scale and zero-point arrays for one weight tensor.
   *
   * @param model Built ONNX model.
   * @param resolvedQuantization Normalized static quantization packet.
   * @param layerTarget One calibrated operator target.
   * @returns Weight scales and zero points.
   */
  function resolveWeightQuantizationParameters(
    model: OnnxModel,
    resolvedQuantization: Extract<
      OnnxResolvedQuantizationOptions,
      { mode: 'static-8bit' }
    >,
    layerTarget: OnnxQuantizationCalibrationLayerTarget,
  ): { scales: number[]; zeroPoints: number[] } {
    const weightRanges = collectWeightRangesForLayer(
      model,
      layerTarget,
      resolvedQuantization.weightGranularity,
    );
    const weightParameterEntries = weightRanges.map((weightRange) =>
      resolveRangeQuantizationParameters(
        weightRange,
        resolvedQuantization.weightEncoding,
        resolvedQuantization.calibration.weightSymmetry,
        resolvedQuantization.calibration.zeroInclusion,
        resolvedQuantization.calibration.roundingMode,
      ),
    );

    return {
      scales: weightParameterEntries.map((parameterEntry) => parameterEntry.scales[0]),
      zeroPoints: weightParameterEntries.map(
        (parameterEntry) => parameterEntry.zeroPoints[0],
      ),
    };
  }

  /**
   * Collect per-tensor or per-output-channel weight ranges for one target layer.
   *
   * @param model Built ONNX model.
   * @param layerTarget One calibrated operator target.
   * @param weightGranularity Resolved weight granularity.
   * @returns Ordered weight ranges.
   */
  function collectWeightRangesForLayer(
    model: OnnxModel,
    layerTarget: OnnxQuantizationCalibrationLayerTarget,
    weightGranularity: 'per-tensor' | 'per-output-channel',
  ): OnnxQuantizationCalibrationRange[] {
    const weightInitializerName =
      layerTarget.target === 'conv'
        ? `ConvW${layerTarget.layerIndex - 1}`
        : `W${layerTarget.layerIndex - 1}`;
    const weightInitializer = model.graph.initializer.find(
      (initializerTensor) => initializerTensor.name === weightInitializerName,
    )!;

    if (weightGranularity === 'per-tensor') {
      return [createMinMaxRange(weightInitializer.float_data)];
    }

    const outputChannelCount = weightInitializer.dims[0]!;
    const elementsPerOutputChannel =
      weightInitializer.float_data.length / outputChannelCount;

    return Array.from(
      { length: outputChannelCount },
      (_, outputChannelIndex) => {
        const startOffset = outputChannelIndex * elementsPerOutputChannel;
        const endOffset = startOffset + elementsPerOutputChannel;
        return createMinMaxRange(
          weightInitializer.float_data.slice(startOffset, endOffset),
        );
      },
    );
  }

  /**
   * Resolve one range into scale and zero-point arrays.
   *
   * @param range Calibration range.
   * @param encoding Integer encoding family.
   * @param symmetry Symmetry policy.
   * @param zeroInclusion Zero-inclusion policy.
   * @param roundingMode Rounding policy.
   * @returns Scalar scale and zero-point arrays.
   */
  function resolveRangeQuantizationParameters(
    range: OnnxQuantizationCalibrationRange,
    encoding: 'uint8' | 'int8',
    symmetry: 'symmetric' | 'asymmetric',
    zeroInclusion: 'required',
    roundingMode: 'nearest-even',
  ): { scales: number[]; zeroPoints: number[] } {
    void zeroInclusion;
    const normalizedRange = {
      min: Math.min(range.min, 0),
      max: Math.max(range.max, 0),
    };
    const quantizationParameters =
      symmetry === 'symmetric'
        ? resolveSymmetricQuantizationParameters(normalizedRange, encoding)
        : resolveAsymmetricQuantizationParameters(
            normalizedRange,
            encoding,
            roundingMode,
          );

    return {
      scales: [quantizationParameters.scale],
      zeroPoints: [quantizationParameters.zeroPoint],
    };
  }

  /**
   * Resolve symmetric quantization parameters for one range.
   *
   * @param range Zero-inclusive calibration range.
   * @param encoding Integer encoding family.
   * @returns Symmetric scale and zero point.
   */
  function resolveSymmetricQuantizationParameters(
    range: OnnxQuantizationCalibrationRange,
    encoding: 'uint8' | 'int8',
  ): { scale: number; zeroPoint: number } {
    const maxAbsoluteMagnitude = Math.max(
      Math.abs(range.min),
      Math.abs(range.max),
    );

    if (maxAbsoluteMagnitude === 0) {
      return {
        scale: 1,
        zeroPoint: encoding === 'uint8' ? 128 : 0,
      };
    }

    return {
      scale: maxAbsoluteMagnitude / 127,
      zeroPoint: encoding === 'uint8' ? 128 : 0,
    };
  }

  /**
   * Resolve asymmetric quantization parameters for one range.
   *
   * @param range Zero-inclusive calibration range.
   * @param encoding Integer encoding family.
   * @param roundingMode Rounding policy.
   * @returns Asymmetric scale and zero point.
   */
  function resolveAsymmetricQuantizationParameters(
    range: OnnxQuantizationCalibrationRange,
    encoding: 'uint8' | 'int8',
    roundingMode: 'nearest-even',
  ): { scale: number; zeroPoint: number } {
    const quantizedMinimum = encoding === 'uint8' ? 0 : -128;
    const quantizedMaximum = encoding === 'uint8' ? 255 : 127;
    const scale = (range.max - range.min) / (quantizedMaximum - quantizedMinimum);
    const unclampedZeroPoint = quantizedMinimum - range.min / scale;

    return {
      scale,
      zeroPoint: clampInteger(
        roundQuantizedValue(unclampedZeroPoint, roundingMode),
        quantizedMinimum,
        quantizedMaximum,
      ),
    };
  }

  /**
   * Create a min/max range from one list of floating-point values.
   *
   * @param values Floating-point values.
   * @returns Min/max range.
   */
  function createMinMaxRange(
    values: number[],
  ): OnnxQuantizationCalibrationRange {
    const minimumValue = values.reduce(
      (currentMinimum, currentValue) =>
        Math.min(currentMinimum, currentValue),
      Number.POSITIVE_INFINITY,
    );
    const maximumValue = values.reduce(
      (currentMaximum, currentValue) =>
        Math.max(currentMaximum, currentValue),
      Number.NEGATIVE_INFINITY,
    );

    return {
      min: minimumValue,
      max: maximumValue,
    };
  }

  /**
   * Build a deterministic parameter initializer name for one layer target.
   *
   * @param layerTarget One calibrated operator target.
   * @param parameterKind Parameter suffix.
   * @returns Stable initializer name.
   */
  function buildQuantizationParameterName(
    layerTarget: OnnxQuantizationCalibrationLayerTarget,
    parameterKind:
      | 'InputScale'
      | 'InputZeroPoint'
      | 'WeightScale'
      | 'WeightZeroPoint'
      | 'OutputScale'
      | 'OutputZeroPoint',
  ): string {
    const operatorPrefix = layerTarget.target === 'conv' ? 'Conv' : 'Dense';
    return `Quant${operatorPrefix}${parameterKind}_l${layerTarget.layerIndex}`;
  }

  /**
   * Create one float scale initializer with scalar or vector shape.
   *
   * @param name Initializer name.
   * @param scaleValues Scalar or vector scale payload.
   * @returns Float initializer.
   */
  function createScaleInitializer(name: string, scaleValues: number[]) {
    return {
      name,
      data_type: ONNX_FLOAT_DATA_TYPE,
      dims: scaleValues.length === 1 ? [] : [scaleValues.length],
      float_data: scaleValues,
    };
  }

  /**
   * Create one integer zero-point initializer stored in int32 JSON payload words.
   *
   * @param name Initializer name.
   * @param encoding Integer encoding family.
   * @param zeroPointValues Scalar or vector zero-point payload.
   * @returns Integer initializer.
   */
  function createZeroPointInitializer(
    name: string,
    encoding: 'uint8' | 'int8',
    zeroPointValues: number[],
  ) {
    return {
      name,
      data_type: encoding === 'uint8' ? ONNX_UINT8_DATA_TYPE : ONNX_INT8_DATA_TYPE,
      dims: zeroPointValues.length === 1 ? [] : [zeroPointValues.length],
      float_data: [],
      int32_data: zeroPointValues,
    };
  }

  /**
   * Round one quantized value with the documented deterministic policy.
   *
   * @param value Floating-point value to round.
   * @param roundingMode Rounding policy.
   * @returns Rounded integer.
   */
  function roundQuantizedValue(
    value: number,
    roundingMode: 'nearest-even',
  ): number {
    void roundingMode;

    const lowerInteger = Math.floor(value);
    const fractionalOffset = value - lowerInteger;

    if (fractionalOffset < 0.5) {
      return lowerInteger;
    }

    if (fractionalOffset > 0.5) {
      return lowerInteger + 1;
    }

    return lowerInteger % 2 === 0 ? lowerInteger : lowerInteger + 1;
  }

  /**
   * Clamp one integer into the supported quantized numeric bounds.
   *
   * @param value Integer candidate.
   * @param minimumValue Minimum allowed integer.
   * @param maximumValue Maximum allowed integer.
   * @returns Clamped integer.
   */
  function clampInteger(
    value: number,
    minimumValue: number,
    maximumValue: number,
  ): number {
    return Math.min(Math.max(value, minimumValue), maximumValue);
  }

  /**
   * Apply storage-fp16 initializer rewrites and float32 cast bridges for the narrow supported subset.
   *
   * @param model Built ONNX model.
   * @param sourceOptions Raw export options.
   * @param recurrentLayerIndices Collected recurrent layer indices.
   * @returns Nothing.
   */
  function applyStorageFp16PostProcessing(
    model: OnnxModel,
    sourceOptions: OnnxExportOptions,
    recurrentLayerIndices: number[],
  ): void {
    // Step 1: Exit when the current export request is outside the supported storage-fp16 subset.
    if (!shouldApplyStorageFp16(sourceOptions, recurrentLayerIndices)) {
      return;
    }

    // Step 2: Rewrite eligible initializer payloads to packed float16 storage.
    const convertedInitializerNames = rewriteEligibleInitializersToFloat16(model);
    // Step 3: Prepend float32 cast bridges and retarget node inputs deterministically.
    prependFloat32CastBridges(model, convertedInitializerNames);
  }

  /**
   * Determine whether the current export request fits the first storage-fp16 closure.
   *
   * @param sourceOptions Raw export options.
   * @param recurrentLayerIndices Collected recurrent layer indices.
   * @returns True when storage-fp16 may be applied.
   */
  function shouldApplyStorageFp16(
    sourceOptions: OnnxExportOptions,
    recurrentLayerIndices: number[],
  ): boolean {
    if (sourceOptions.precision?.mode !== 'storage-fp16') {
      return false;
    }

    if (recurrentLayerIndices.length > 0) {
      return false;
    }

    if (sourceOptions.allowMixedActivations || sourceOptions.allowPartialConnectivity) {
      return false;
    }

    if (
      (sourceOptions.concatMappings?.length ?? 0) > 0 ||
      (sourceOptions.attentionMappings?.length ?? 0) > 0
    ) {
      return false;
    }

    return true;
  }

  /**
   * Rewrite eligible initializer tensors from float32 storage into packed float16 payloads.
   *
   * @param model Built ONNX model.
   * @returns Converted initializer names in deterministic order.
   */
  function rewriteEligibleInitializersToFloat16(model: OnnxModel): string[] {
    return model.graph.initializer.flatMap((initializerEntry) => {
      if (
        !isStorageFp16EligibleInitializer(initializerEntry.name) ||
        initializerEntry.data_type !== ONNX_FLOAT_DATA_TYPE ||
        initializerEntry.float_data.length === 0
      ) {
        return [];
      }

      const float16Payload = createFloat16StoragePayload(
        initializerEntry.float_data,
      );
      initializerEntry.data_type = float16Payload.data_type;
      initializerEntry.float_data = float16Payload.float_data;
      initializerEntry.int32_data = float16Payload.int32_data;
      return [initializerEntry.name];
    });
  }

  /**
   * Determine whether an initializer belongs to the first storage-fp16 subset.
   *
   * @param initializerName Initializer name.
   * @returns True when the initializer is eligible.
   */
  function isStorageFp16EligibleInitializer(initializerName: string): boolean {
    return STORAGE_FP16_ELIGIBLE_INITIALIZER_PATTERNS.some((initializerPattern) =>
      initializerPattern.test(initializerName),
    );
  }

  /**
   * Prepend float32 cast bridges for converted initializers and retarget all graph-node inputs.
   *
   * @param model Built ONNX model.
   * @param convertedInitializerNames Converted initializer names.
   * @returns Nothing.
   */
  function prependFloat32CastBridges(
    model: OnnxModel,
    convertedInitializerNames: string[],
  ): void {
    const castOutputNameByInitializerName = new Map(
      convertedInitializerNames.map((initializerName) => [
        initializerName,
        `${initializerName}${STORAGE_FP16_CAST_SUFFIX}`,
      ]),
    );

    const castNodes = convertedInitializerNames.map((initializerName) => ({
      op_type: 'Cast',
      input: [initializerName],
      output: [castOutputNameByInitializerName.get(initializerName)!],
      name: `cast_${initializerName}_to_fp32`,
      attributes: [
        {
          name: 'to',
          type: 'INT',
          i: STORAGE_FP16_FLOAT_CAST_TARGET,
        },
      ],
    }));

    model.graph.node = model.graph.node.map((graphNode) => ({
      ...graphNode,
      input: graphNode.input.map(
        (inputName) =>
          castOutputNameByInitializerName.get(inputName) ?? inputName,
      ),
    }));
    model.graph.node = [...castNodes, ...model.graph.node];
  }

  /**
   * Build base ONNX model and apply metadata.
   *
   * @param networkLayers Layered network topology.
   * @param currentOptions Resolved options.
   * @returns Initialized ONNX model.
   */
  function createInitializedModel(
    networkLayers: NeatapticNode[][],
    currentOptions: OnnxBuildResolvedOptions,
  ): OnnxModel {
    // Step 1: Resolve graph IO dimensions from layered topology.
    const graphDimensions = createModelGraphDimensions(
      networkLayers,
      currentOptions.batchDimension,
    );

    // Step 2: Create base ONNX model from resolved dimensions.
    const initializedModel = createBaseModel({
      inputDims: graphDimensions.inputDims,
      outputDims: graphDimensions.outputDims,
    });

    // Step 3: Attach metadata settings to initialized model.
    applyResolvedModelMetadata(initializedModel, currentOptions);
    return initializedModel;
  }

  /**
   * Resolve ONNX graph dimensions from input/output layer sizes.
   *
   * @param networkLayers Layered network topology.
   * @param batchDimension Whether batch dimension is enabled.
   * @returns Input and output dimensions.
   */
  function createModelGraphDimensions(
    networkLayers: NeatapticNode[][],
    batchDimension: boolean,
  ): OnnxGraphDimensions {
    // Step 1: Resolve first and last layers for IO sizing.
    const inputLayerNodes = networkLayers[0];
    const outputLayerNodes = networkLayers.at(-1)!;

    // Step 2: Build graph dimensions using setup utility.
    return createGraphDimensions({
      inputWidth: inputLayerNodes.length,
      outputWidth: outputLayerNodes.length,
      batchDimension,
    });
  }

  /**
   * Apply resolved build metadata to an initialized ONNX model.
   *
   * @param initializedModel Initialized ONNX model.
   * @param currentOptions Resolved build options.
   * @returns Nothing.
   */
  function applyResolvedModelMetadata(
    initializedModel: OnnxModel,
    currentOptions: OnnxBuildResolvedOptions,
  ): void {
    // Step 1: Delegate metadata assignment to setup utility.
    applyModelMetadata({
      model: initializedModel,
      includeMetadata: currentOptions.includeMetadata,
      opset: currentOptions.opset,
      producerName: currentOptions.producerName,
      producerVersion: currentOptions.producerVersion,
      docString: currentOptions.docString,
    });
  }

  /**
   * Collect recurrent layer indices needed during graph emission.
   *
   * @param context Recurrent collection context.
   * @returns Recurrent layer indices.
   */
  function collectRecurrentIndices(
    context: OnnxRecurrentCollectionContext,
  ): number[] {
    // Step 1: Delegate recurrent-layer discovery to setup utility.
    return collectRecurrentLayerIndices(context);
  }

  /**
   * Emit all non-input layers while tracking hidden-layer metadata.
   *
   * @param context Layer emission context.
   * @returns Final output name and hidden layer sizes metadata.
   */
  function emitNonInputLayers(
    context: OnnxLayerEmissionContext,
  ): OnnxLayerEmissionResult {
    // Step 1: Build deterministic layer index list for all non-input layers.
    const nonInputLayerIndices = createNonInputLayerIndices(
      context.layers.length,
    );

    // Step 2: Collect hidden layer sizes (excluding output layer).
    const hiddenSizesMetadata = collectHiddenLayerSizes(
      context.layers,
      nonInputLayerIndices,
    );

    // Step 3: Emit graph nodes for each non-input layer and fold output name.
    const layerEmissionState = emitLayerGraphsForIndices(
      context,
      nonInputLayerIndices,
      'input',
    );

    return {
      previousOutputName: layerEmissionState.previousOutputName,
      hiddenSizesMetadata,
      layerOutputNamesByLayerIndex:
        layerEmissionState.layerOutputNamesByLayerIndex,
    };
  }

  /**
   * Apply export post-processing and metadata finalization.
   *
   * @param context Post-processing context.
   * @returns Nothing.
   */
  function applyPostProcessing(context: OnnxPostProcessingContext): void {
    // Step 1: Emit optional fused recurrent output adjustments.
    emitFusedRecurrentHeuristics(
      context.model,
      context.layers,
      context.options.allowRecurrent,
      context.layerEmissionResult.previousOutputName,
    );

    // Step 2: Emit explicit shadow attention mappings without replacing the dense path.
    emitShadowAttentionMappings(
      context.model,
      context.layers,
      context.options,
      context.layerEmissionResult.layerOutputNamesByLayerIndex,
      context.includeMetadata,
    );

    // Step 3: Finalize metadata using collected build artifacts.
    finalizeExportMetadata(
      context.model,
      context.layers,
      context.options,
      context.includeMetadata,
      context.layerEmissionResult.hiddenSizesMetadata,
      context.recurrentLayerIndices,
    );
  }

  /**
   * Build ordered indices for all non-input layers.
   *
   * @param layerCount Total number of layers.
   * @returns Layer indices from first hidden to output.
   */
  function createNonInputLayerIndices(layerCount: number): number[] {
    // Step 1: Generate ordered indices for all layers after the input layer.
    return Array.from(
      { length: Math.max(layerCount - 1, 0) },
      (_, offset) => offset + 1,
    );
  }

  /**
   * Collect metadata sizes for hidden layers only.
   *
   * @param networkLayers Layered network topology.
   * @param nonInputLayerIndices Layer indices excluding input.
   * @returns Hidden layer sizes metadata.
   */
  function collectHiddenLayerSizes(
    networkLayers: NeatapticNode[][],
    nonInputLayerIndices: number[],
  ): number[] {
    // Step 1: Resolve output layer index for exclusion.
    const outputLayerIndex = networkLayers.length - 1;

    // Step 2: Keep hidden layers only and map to their unit counts.
    return nonInputLayerIndices
      .filter((layerIndex) => layerIndex !== outputLayerIndex)
      .map((layerIndex) => networkLayers[layerIndex].length);
  }

  /**
   * Emit layer graphs in index order while folding the output tensor name.
   *
   * @param context Layer emission context.
   * @param nonInputLayerIndices Layer indices excluding input.
   * @param initialOutputName Initial input tensor name.
   * @returns Final output tensor name.
   */
  function emitLayerGraphsForIndices(
    context: OnnxLayerEmissionContext,
    nonInputLayerIndices: number[],
    initialOutputName: string,
  ): {
    previousOutputName: string;
    layerOutputNamesByLayerIndex: Map<number, string>;
  } {
    // Step 1: Seed the emitted-layer output lookup with the graph input tensor.
    const initialLayerOutputNamesByLayerIndex = new Map<number, string>([
      [0, initialOutputName],
    ]);

    // Step 2: Fold layer emissions to the final output tensor name and layer map.
    return nonInputLayerIndices.reduce(
      (layerEmissionState, layerIndex) => {
        const currentOutputName = emitSingleLayerGraph(
          context,
          layerIndex,
          layerEmissionState.previousOutputName,
          layerEmissionState.layerOutputNamesByLayerIndex,
        );
        const nextLayerOutputNamesByLayerIndex = new Map(
          layerEmissionState.layerOutputNamesByLayerIndex,
        );
        nextLayerOutputNamesByLayerIndex.set(layerIndex, currentOutputName);
        return {
          previousOutputName: currentOutputName,
          layerOutputNamesByLayerIndex: nextLayerOutputNamesByLayerIndex,
        };
      },
      {
        previousOutputName: initialOutputName,
        layerOutputNamesByLayerIndex: initialLayerOutputNamesByLayerIndex,
      },
    );
  }

  /**
   * Emit graph nodes for a single layer index.
   *
   * @param context Layer emission context.
   * @param layerIndex Layer index to emit.
   * @param previousOutputName Previous tensor output name.
   * @returns Current layer output tensor name.
   */
  function emitSingleLayerGraph(
    context: OnnxLayerEmissionContext,
    layerIndex: number,
    previousOutputName: string,
    layerOutputNamesByLayerIndex: Map<number, string>,
  ): string {
    // Step 1: Emit ONNX operators for one layer and return its output tensor.
    return emitLayerGraph({
      model: context.model,
      layers: context.layers,
      options: context.options,
      layerIndex,
      previousOutputName,
      layerOutputNamesByLayerIndex,
      recurrentLayerIndices: context.recurrentLayerIndices,
      batchDimension: context.batchDimension,
      legacyNodeOrdering: context.legacyNodeOrdering,
    });
  }
}
