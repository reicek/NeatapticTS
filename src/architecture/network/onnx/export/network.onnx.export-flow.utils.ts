import type Network from '../../network';
import type NeatapticNode from '../../../node';
import { buildOnnxModel } from './network.onnx.export-build.utils';
import { appendAdvancedGraphMetadata } from './network.onnx.export-advanced-graph.utils';
import {
  appendConvInferenceMetadata,
  appendLstmPatternStubMetadata,
  assignExportNodeIndices,
  collectLstmPatternStubs,
  resolveEffectiveConvMappings,
} from './network.onnx.export-orchestrators.utils';
import {
  inferLayerOrdering,
  rebuildConnectionsLocal,
  validateLayerHomogeneityAndConnectivity,
} from '../network.onnx.layer-analysis.utils';
import type { OnnxExportOptions } from './network.onnx.export.types';
import type { OnnxModel } from '../schema/network.onnx.schema.types';

/**
 * Execute the complete ONNX export flow for one network instance.
 *
 * High-level behavior:
 *  1. Rebuild runtime connection caches and assign stable export indices.
 *  2. Infer layered ordering and collect recurrent-pattern stubs.
 *  3. Validate structural constraints for the requested export options.
 *  4. Build ONNX graph payload and append inference-oriented metadata.
 *
 * @param network Source network to serialize.
 * @param options Optional ONNX export controls.
 * @returns ONNX-like model payload.
 */
export function runOnnxExportFlow(
  network: Network,
  options: OnnxExportOptions = {},
): OnnxModel {
  // Step 1: Normalize runtime graph caches for deterministic export traversal.
  rebuildConnectionsLocal(network);
  assignExportNodeIndices(network);

  // Step 2: Infer ordered layers and collect heuristic recurrent metadata stubs.
  const layers = inferLayerOrdering(network);
  const lstmPatternStubs = collectLstmPatternStubs(
    layers,
    options.allowRecurrent,
  );

  // Step 3: Validate layer constraints before graph materialization.
  validateLayerHomogeneityAndConnectivity(layers, network, options);

  // Step 4: Resolve effective export options after optional Conv promotion.
  const effectiveOptions = resolveEffectiveExportOptions(layers, options);

  // Step 5: Build the ONNX-like payload and append post-build metadata.
  const model = buildOnnxModel(network, layers, effectiveOptions);
  appendAdvancedGraphMetadata(
    model,
    network,
    layers,
    effectiveOptions.includeMetadata ?? false,
  );
  appendConvInferenceMetadata(model, layers, effectiveOptions);
  appendLstmPatternStubMetadata(model, lstmPatternStubs);
  appendPhaseSevenRequestMetadata(model, layers, options);
  return model;

  /**
   * Resolve effective export options without mutating caller-owned state.
   *
   * @param networkLayers Layered network nodes.
   * @param sourceOptions Raw export options.
   * @returns Effective export options for this export pass.
   */
  function resolveEffectiveExportOptions(
    networkLayers: NeatapticNode[][],
    sourceOptions: OnnxExportOptions,
  ): OnnxExportOptions {
    const effectiveConvMappings = resolveEffectiveConvMappings(
      networkLayers,
      sourceOptions,
    );
    return {
      ...sourceOptions,
      conv2dMappings: effectiveConvMappings,
    };
  }

  /**
   * Append Phase 7 request metadata without implying that reduced-precision lowering has landed.
   *
   * @param model Built ONNX model.
   * @param networkLayers Layered network nodes.
   * @param sourceOptions Raw export options.
   * @returns Nothing.
   */
  function appendPhaseSevenRequestMetadata(
    model: OnnxModel,
    networkLayers: NeatapticNode[][],
    sourceOptions: OnnxExportOptions,
  ): void {
    // Step 1: Skip Phase 7 metadata unless top-level metadata emission is enabled.
    if (!sourceOptions.includeMetadata) {
      return;
    }

    // Step 2: Append requested/effective precision metadata when a precision packet exists.
    if (sourceOptions.precision && sourceOptions.precision.metadata !== false) {
      const requestedPrecisionMode = sourceOptions.precision.mode ?? 'float32';
      const precisionFallbackReasons = buildPrecisionFallbackReasons(
        networkLayers,
        sourceOptions,
      );
      const effectivePrecisionMode =
        requestedPrecisionMode === 'storage-fp16' &&
        precisionFallbackReasons.length === 0
          ? 'storage-fp16'
          : 'float32';

      appendMetadataEntry(
        model,
        'requested_precision_mode',
        requestedPrecisionMode,
      );
      appendMetadataEntry(
        model,
        'effective_precision_mode',
        effectivePrecisionMode,
      );
      if (precisionFallbackReasons.length > 0) {
        appendMetadataEntry(
          model,
          'precision_fallback_reasons',
          JSON.stringify(precisionFallbackReasons),
        );
      }
    }

    // Step 3: Append requested/effective quantization metadata when a quantization packet exists.
    if (sourceOptions.quantization) {
      appendQuantizationRequestMetadata(model, networkLayers, sourceOptions);
    }
  }

  /**
   * Append quantization request metadata and explicit float32 fallback reasons.
   *
   * @param model Built ONNX model.
   * @param networkLayers Layered network nodes.
   * @param sourceOptions Raw export options.
   * @returns Nothing.
   */
  function appendQuantizationRequestMetadata(
    model: OnnxModel,
    networkLayers: NeatapticNode[][],
    sourceOptions: OnnxExportOptions,
  ): void {
    const quantizationPacket = sourceOptions.quantization!;
    const effectiveQuantizationMode = resolveEffectiveQuantizationMode(
      model,
      quantizationPacket.mode,
    );

    const quantizationFallbackReasons = buildQuantizationFallbackReasons(
      model,
      networkLayers,
      sourceOptions,
    );

    appendMetadataEntry(
      model,
      'requested_quantization_mode',
      quantizationPacket.mode,
    );
    appendMetadataEntry(
      model,
      'effective_quantization_mode',
      effectiveQuantizationMode,
    );

    if (quantizationPacket.mode === 'static-8bit') {
      appendMetadataEntry(
        model,
        'quantization_activation_granularity',
        quantizationPacket.activationGranularity ?? 'per-tensor',
      );
      appendMetadataEntry(
        model,
        'quantization_weight_granularity',
        quantizationPacket.weightGranularity ?? 'per-tensor',
      );
      appendMetadataEntry(
        model,
        'quantization_calibration_source',
        quantizationPacket.calibration.source,
      );

      if (quantizationPacket.calibration.packetId) {
        appendMetadataEntry(
          model,
          'quantization_calibration_packet_id',
          quantizationPacket.calibration.packetId,
        );
      }

      if (quantizationPacket.calibration.sampleCount !== undefined) {
        appendMetadataEntry(
          model,
          'quantization_calibration_sample_count',
          String(quantizationPacket.calibration.sampleCount),
        );
      }

      appendMetadataEntry(
        model,
        'quantization_calibration_target_count',
        String(quantizationPacket.calibration.layerTargets.length),
      );
      appendMetadataEntry(
        model,
        'quantization_weight_range_policy',
        quantizationPacket.calibration.weightRangePolicy ?? 'min-max',
      );
      appendMetadataEntry(
        model,
        'quantization_zero_inclusion_policy',
        quantizationPacket.calibration.zeroInclusion ?? 'required',
      );
      appendMetadataEntry(
        model,
        'quantization_activation_symmetry',
        quantizationPacket.calibration.activationSymmetry ?? 'asymmetric',
      );
      appendMetadataEntry(
        model,
        'quantization_weight_symmetry',
        quantizationPacket.calibration.weightSymmetry ?? 'symmetric',
      );
      appendMetadataEntry(
        model,
        'quantization_rounding_mode',
        quantizationPacket.calibration.roundingMode ?? 'nearest-even',
      );
    }

    appendMetadataEntry(
      model,
      'quantization_fallback_reasons',
      JSON.stringify(quantizationFallbackReasons),
    );
  }

  /**
   * Resolve the effective quantization mode from the emitted graph payload.
   *
   * @param model Built ONNX model.
   * @param requestedQuantizationMode Requested quantization mode.
   * @returns Effective quantization mode for metadata emission.
   */
  function resolveEffectiveQuantizationMode(
    model: OnnxModel,
    requestedQuantizationMode: NonNullable<
      OnnxExportOptions['quantization']
    >['mode'],
  ): 'none' | 'static-8bit' {
    if (
      requestedQuantizationMode === 'static-8bit' &&
      model.graph.node.some(
        (graphNode) => graphNode.op_type === 'QLinearMatMul',
      )
    ) {
      return 'static-8bit';
    }

    return 'none';
  }

  /**
   * Build the explicit float32 fallback reasons for a Phase 7 precision request.
   *
   * @param networkLayers Layered network nodes.
   * @param sourceOptions Raw export options.
   * @returns Ordered fallback reason codes.
   */
  function buildPrecisionFallbackReasons(
    networkLayers: NeatapticNode[][],
    sourceOptions: OnnxExportOptions,
  ): string[] {
    if (sourceOptions.precision?.mode !== 'storage-fp16') {
      return [];
    }

    const fallbackReasons: string[] = [];

    if (hasRecurrentBoundary(networkLayers, sourceOptions)) {
      fallbackReasons.push('recurrent_boundary_requires_float32');
    }

    if (hasAdvancedGraphBoundary(sourceOptions)) {
      fallbackReasons.push('advanced_graph_boundary_requires_float32');
    }

    if (sourceOptions.allowMixedActivations) {
      fallbackReasons.push('mixed_activation_boundary_requires_float32');
    }

    if (sourceOptions.allowPartialConnectivity) {
      fallbackReasons.push('partial_connectivity_boundary_requires_float32');
    }

    return fallbackReasons;
  }

  /**
   * Build the explicit float32 fallback reasons for a Phase 7 quantization request.
   *
   * @param networkLayers Layered network nodes.
   * @param sourceOptions Raw export options.
   * @returns Ordered fallback reason codes.
   */
  function buildQuantizationFallbackReasons(
    model: OnnxModel,
    networkLayers: NeatapticNode[][],
    sourceOptions: OnnxExportOptions,
  ): string[] {
    const quantizationPacket = sourceOptions.quantization!;

    const fallbackReasons =
      quantizationPacket.mode === 'static-8bit'
        ? resolveStaticQuantizationFallbackReasons(model)
        : ['dynamic_uint8_not_implemented'];

    if (hasRecurrentBoundary(networkLayers, sourceOptions)) {
      fallbackReasons.push('recurrent_boundary_requires_float32');
    }

    if (hasAdvancedGraphBoundary(sourceOptions)) {
      fallbackReasons.push('advanced_graph_boundary_requires_float32');
    }

    if (sourceOptions.allowMixedActivations) {
      fallbackReasons.push('mixed_activation_boundary_requires_float32');
    }

    if (sourceOptions.allowPartialConnectivity) {
      fallbackReasons.push('partial_connectivity_boundary_requires_float32');
    }

    return fallbackReasons;
  }

  /**
   * Resolve static quantization fallback reasons after checking whether qlinear dense lowering landed.
   *
   * @param model Built ONNX model.
   * @returns Ordered static-quantization fallback reasons.
   */
  function resolveStaticQuantizationFallbackReasons(
    model: OnnxModel,
  ): string[] {
    return model.graph.node.some(
      (graphNode) => graphNode.op_type === 'QLinearMatMul',
    )
      ? []
      : ['static_8bit_not_implemented'];
  }

  /**
   * Detect whether the current export request crosses the recurrent boundary.
   *
   * @param networkLayers Layered network nodes.
   * @param sourceOptions Raw export options.
   * @returns True when recurrent export is requested and a hidden layer carries self-state.
   */
  function hasRecurrentBoundary(
    networkLayers: NeatapticNode[][],
    sourceOptions: OnnxExportOptions,
  ): boolean {
    if (!sourceOptions.allowRecurrent) {
      return false;
    }

    const hiddenLayers = networkLayers.slice(1, -1);
    return hiddenLayers.some((hiddenLayerNodes) =>
      hiddenLayerNodes.some(
        (hiddenNode) => hiddenNode.connections.self.length > 0,
      ),
    );
  }

  /**
   * Detect whether the current export request crosses the explicit advanced-graph boundary.
   *
   * @param sourceOptions Raw export options.
   * @returns True when explicit merge or attention mappings are requested.
   */
  function hasAdvancedGraphBoundary(sourceOptions: OnnxExportOptions): boolean {
    return (
      (sourceOptions.concatMappings?.length ?? 0) > 0 ||
      (sourceOptions.attentionMappings?.length ?? 0) > 0
    );
  }

  /**
   * Append one metadata entry to the model-level metadata registry.
   *
   * @param model Built ONNX model.
   * @param key Metadata key.
   * @param value Metadata value.
   * @returns Nothing.
   */
  function appendMetadataEntry(
    model: OnnxModel,
    key: string,
    value: string,
  ): void {
    model.metadata_props ??= [];
    model.metadata_props.push({
      key,
      value,
    });
  }
}
