import type NeatapticNode from '../../node';
import type {
  Conv2DMapping,
  NodeInternals,
  OnnxConvEmissionContext,
  OnnxConvEmissionParams,
  OnnxConvKernelCoordinate,
  OnnxConvParameters,
  OnnxConvTensorNames,
  OnnxExportOptions,
  Pool2DMapping,
} from './network.onnx.utils.types';
import {
  appendIndexedMetadata,
  appendMetadataSpec,
  emitOptionalPoolingAndFlatten,
} from './network.onnx.export-layer-common.utils';
import { mapActivationToOnnx } from './network.onnx.layer-analysis.utils';

/**
 * Try to emit a conv-mapped layer.
 *
 * @param params Conv emission parameters.
 * @returns New output tensor name when handled, otherwise undefined.
 */
export function tryEmitConvLayer(
  params: OnnxConvEmissionParams,
): string | undefined {
  // Step 1: Resolve Conv mapping for this layer index.
  const convSpec = resolveConvMapping(params.options, params.layerIndex);
  if (!convSpec) return undefined;

  // Step 2: Build typed emission context.
  const convContext = createConvContext(params, convSpec);

  // Step 3: Validate mapping shape and stop early when incompatible.
  if (!validateConvShapeOrWarn(convContext)) return undefined;

  // Step 4: Collect Conv parameters and emit core Conv graph.
  const convParameters = collectConvParameters(convContext);
  const convTensorNames = emitConvParameterInitializers(
    convContext,
    convParameters,
  );
  const activationOutputName = emitConvAndActivationGraph(
    convContext,
    convTensorNames,
  );

  // Step 5: Emit optional pooling/flatten post-processing.
  const pooledOutputName = emitOptionalPoolingAndFlattenForConv(
    convContext,
    activationOutputName,
  );

  // Step 6: Append export metadata and return final output tensor.
  appendConvExportMetadata(convContext);
  return pooledOutputName;

  /**
   * Resolve Conv mapping declaration for a layer index.
   *
   * @param options Export options.
   * @param layerIndex Layer index.
   * @returns Conv mapping, if configured.
   */
  function resolveConvMapping(
    options: OnnxExportOptions,
    layerIndex: number,
  ): Conv2DMapping | undefined {
    // Step 1: Find mapping that targets current layer index.
    return options.conv2dMappings?.find(
      (mapping) => mapping.layerIndex === layerIndex,
    );
  }

  /**
   * Build Conv emission context from params and mapping.
   *
   * @param sourceParams Raw emission params.
   * @param convSpec Conv mapping spec.
   * @returns Typed Conv emission context.
   */
  function createConvContext(
    sourceParams: OnnxConvEmissionParams,
    convSpec: Conv2DMapping,
  ): OnnxConvEmissionContext {
    // Step 1: Merge base params and resolved mapping.
    return {
      ...sourceParams,
      convSpec,
    };
  }

  /**
   * Validate Conv dimensions and log mismatch details when invalid.
   *
   * @param context Conv emission context.
   * @returns Whether Conv shape is compatible.
   */
  function validateConvShapeOrWarn(context: OnnxConvEmissionContext): boolean {
    // Step 1: Return success when dimensions align.
    if (isConvShapeCompatible(context)) return true;

    // Step 2: Log mismatch details and reject this mapping.
    logConvShapeMismatch(context);
    return false;
  }

  /**
   * Determine whether declared Conv dimensions match layer widths.
   *
   * @param context Conv emission context.
   * @returns Whether Conv dimensions match the network layers.
   */
  function isConvShapeCompatible(context: OnnxConvEmissionContext): boolean {
    // Step 1: Compare expected and actual previous/current widths.
    return (
      getExpectedPreviousWidth(context.convSpec) ===
        context.previousLayerNodes.length &&
      getExpectedCurrentWidth(context.convSpec) ===
        context.currentLayerNodes.length
    );
  }

  /**
   * Get expected previous-layer width for Conv mapping.
   *
   * @param convSpec Conv mapping spec.
   * @returns Expected previous-layer width.
   */
  function getExpectedPreviousWidth(convSpec: Conv2DMapping): number {
    // Step 1: Multiply input height, width, and channels.
    return convSpec.inHeight * convSpec.inWidth * convSpec.inChannels;
  }

  /**
   * Get expected current-layer width for Conv mapping.
   *
   * @param convSpec Conv mapping spec.
   * @returns Expected current-layer width.
   */
  function getExpectedCurrentWidth(convSpec: Conv2DMapping): number {
    // Step 1: Multiply output channels, height, and width.
    return convSpec.outChannels * convSpec.outHeight * convSpec.outWidth;
  }

  /**
   * Log Conv mapping shape mismatch warning.
   *
   * @param context Conv emission context.
   * @returns Nothing.
   */
  function logConvShapeMismatch(context: OnnxConvEmissionContext): void {
    // Step 1: Resolve expected dimensions for warning output.
    const previousWidthExpected = getExpectedPreviousWidth(context.convSpec);
    const currentWidthExpected = getExpectedCurrentWidth(context.convSpec);

    // Step 2: Emit warning with expected vs actual widths.
    console.warn(
      `Conv2D mapping for layer ${context.layerIndex} skipped: dimension mismatch (expected prev=${previousWidthExpected} got ${context.previousLayerNodes.length}; expected this=${currentWidthExpected} got ${context.currentLayerNodes.length}).`,
    );
  }

  /**
   * Collect flattened Conv weights and biases.
   *
   * @param context Conv emission context.
   * @returns Conv initializer parameters.
   */
  function collectConvParameters(
    context: OnnxConvEmissionContext,
  ): OnnxConvParameters {
    // Step 1: Collect representative internals for each output channel.
    const outputChannelIndices = createOutputChannelIndices(context.convSpec);
    const representativeNeuronInternals = collectRepresentativeNeuronInternals(
      context,
      outputChannelIndices,
    );

    // Step 2: Collect biases and flattened kernel weights.
    const biases = collectConvBiasValues(representativeNeuronInternals);
    const weights = collectConvWeightValues(
      context,
      representativeNeuronInternals,
    );
    return { weights, biases };
  }

  /**
   * Create output channel index list.
   *
   * @param convSpec Conv mapping spec.
   * @returns Output channel indices.
   */
  function createOutputChannelIndices(convSpec: Conv2DMapping): number[] {
    // Step 1: Build ordered output channel indices.
    return Array.from(
      { length: convSpec.outChannels },
      (_, channelOffset) => channelOffset,
    );
  }

  /**
   * Collect representative neuron internals for each output channel.
   *
   * @param context Conv emission context.
   * @param outputChannelIndices Output channel indices.
   * @returns Representative internals.
   */
  function collectRepresentativeNeuronInternals(
    context: OnnxConvEmissionContext,
    outputChannelIndices: number[],
  ): NodeInternals[] {
    // Step 1: Resolve one representative neuron per output channel.
    return outputChannelIndices.map((outputChannelIndex) =>
      resolveRepresentativeNeuronInternal(context, outputChannelIndex),
    );
  }

  /**
   * Resolve representative neuron internals for one output channel.
   *
   * @param context Conv emission context.
   * @param outputChannelIndex Output channel index.
   * @returns Representative neuron internals.
   */
  function resolveRepresentativeNeuronInternal(
    context: OnnxConvEmissionContext,
    outputChannelIndex: number,
  ): NodeInternals {
    // Step 1: Resolve representative neuron index for this channel.
    const representativeNeuronIndex = resolveRepresentativeNeuronIndex(
      context.convSpec,
      outputChannelIndex,
    );

    // Step 2: Cast representative neuron to runtime internals shape.
    return context.currentLayerNodes[
      representativeNeuronIndex
    ] as NodeInternals;
  }

  /**
   * Resolve representative neuron index for one output channel.
   *
   * @param convSpec Conv mapping spec.
   * @param outputChannelIndex Output channel index.
   * @returns Representative neuron index.
   */
  function resolveRepresentativeNeuronIndex(
    convSpec: Conv2DMapping,
    outputChannelIndex: number,
  ): number {
    // Step 1: Jump to first spatial position for the output channel.
    return outputChannelIndex * convSpec.outHeight * convSpec.outWidth;
  }

  /**
   * Collect Conv bias values from representative neurons.
   *
   * @param representativeNeuronInternals Representative internals.
   * @returns Bias values.
   */
  function collectConvBiasValues(
    representativeNeuronInternals: NodeInternals[],
  ): number[] {
    // Step 1: Map representative bias values.
    return representativeNeuronInternals.map(
      (representativeNeuronInternal) => representativeNeuronInternal.bias,
    );
  }

  /**
   * Collect flattened Conv weight values.
   *
   * @param context Conv emission context.
   * @param representativeNeuronInternals Representative internals.
   * @returns Flattened Conv weights.
   */
  function collectConvWeightValues(
    context: OnnxConvEmissionContext,
    representativeNeuronInternals: NodeInternals[],
  ): number[] {
    // Step 1: Build reusable kernel coordinates for all input channels.
    const kernelCoordinates = createKernelCoordinates(context.convSpec);

    // Step 2: Flatten weights for each output channel representative.
    return representativeNeuronInternals.flatMap(
      (representativeNeuronInternal) =>
        kernelCoordinates.map((kernelCoordinate) =>
          resolveWeightForCoordinate(
            context,
            representativeNeuronInternal,
            kernelCoordinate,
          ),
        ),
    );
  }

  /**
   * Create all Conv kernel coordinates across input channels.
   *
   * @param convSpec Conv mapping spec.
   * @returns Kernel coordinates.
   */
  function createKernelCoordinates(
    convSpec: Conv2DMapping,
  ): OnnxConvKernelCoordinate[] {
    // Step 1: Enumerate input channels then flatten per-channel coordinates.
    return Array.from(
      { length: convSpec.inChannels },
      (_, inputChannelIndex) => inputChannelIndex,
    ).flatMap((inputChannelIndex) =>
      buildKernelCoordinatesForInputChannel(convSpec, inputChannelIndex),
    );
  }

  /**
   * Build kernel coordinates for one input channel.
   *
   * @param convSpec Conv mapping spec.
   * @param inputChannelIndex Input channel index.
   * @returns Kernel coordinates for the input channel.
   */
  function buildKernelCoordinatesForInputChannel(
    convSpec: Conv2DMapping,
    inputChannelIndex: number,
  ): OnnxConvKernelCoordinate[] {
    // Step 1: Enumerate kernel rows.
    return Array.from(
      { length: convSpec.kernelHeight },
      (_, kernelRowIndex) => kernelRowIndex,
    ).flatMap((kernelRowIndex) =>
      Array.from({ length: convSpec.kernelWidth }, (_, kernelColumnIndex) => ({
        inChannelIndex: inputChannelIndex,
        kernelRowIndex,
        kernelColumnIndex,
      })),
    );
  }

  /**
   * Resolve weight for one kernel coordinate.
   *
   * @param context Conv emission context.
   * @param representativeNeuronInternal Representative neuron internals.
   * @param kernelCoordinate Kernel coordinate.
   * @returns Weight value or zero when connection is missing.
   */
  function resolveWeightForCoordinate(
    context: OnnxConvEmissionContext,
    representativeNeuronInternal: NodeInternals,
    kernelCoordinate: OnnxConvKernelCoordinate,
  ): number {
    // Step 1: Resolve source node for coordinate and read inbound weight.
    const sourceNode = resolveConvSourceNode(context, kernelCoordinate);
    return resolveInboundWeightOrZero(representativeNeuronInternal, sourceNode);
  }

  /**
   * Resolve source node referenced by one kernel coordinate.
   *
   * @param context Conv emission context.
   * @param kernelCoordinate Kernel coordinate.
   * @returns Source node.
   */
  function resolveConvSourceNode(
    context: OnnxConvEmissionContext,
    kernelCoordinate: OnnxConvKernelCoordinate,
  ): NeatapticNode {
    // Step 1: Resolve flattened feature index then return corresponding node.
    const inputFeatureIndex = resolveInputFeatureIndex(
      context.convSpec,
      kernelCoordinate,
    );
    return context.previousLayerNodes[inputFeatureIndex];
  }

  /**
   * Resolve flattened input feature index for one kernel coordinate.
   *
   * @param convSpec Conv mapping spec.
   * @param kernelCoordinate Kernel coordinate.
   * @returns Flattened input feature index.
   */
  function resolveInputFeatureIndex(
    convSpec: Conv2DMapping,
    kernelCoordinate: OnnxConvKernelCoordinate,
  ): number {
    // Step 1: Convert channel/row/column coordinate to flattened index.
    return (
      kernelCoordinate.inChannelIndex * (convSpec.inHeight * convSpec.inWidth) +
      kernelCoordinate.kernelRowIndex * convSpec.inWidth +
      kernelCoordinate.kernelColumnIndex
    );
  }

  /**
   * Resolve inbound weight or zero when missing.
   *
   * @param representativeNeuronInternal Representative neuron internals.
   * @param sourceNode Source node.
   * @returns Inbound weight value.
   */
  function resolveInboundWeightOrZero(
    representativeNeuronInternal: NodeInternals,
    sourceNode: NeatapticNode,
  ): number {
    // Step 1: Find inbound connection from source node.
    const inboundConnection = representativeNeuronInternal.connections.in.find(
      (connection) => connection.from === sourceNode,
    );

    // Step 2: Return connection weight or zero fallback.
    return inboundConnection?.weight ?? 0;
  }

  /**
   * Emit Conv parameter initializers and return tensor names.
   *
   * @param context Conv emission context.
   * @param convParameters Conv parameters.
   * @returns Conv tensor names.
   */
  function emitConvParameterInitializers(
    context: OnnxConvEmissionContext,
    convParameters: OnnxConvParameters,
  ): OnnxConvTensorNames {
    // Step 1: Create deterministic tensor names.
    const convTensorNames = createConvTensorNames(context.layerIndex);

    // Step 2: Emit weight and bias initializers.
    appendConvWeightInitializer(
      context,
      convTensorNames,
      convParameters.weights,
    );
    appendConvBiasInitializer(context, convTensorNames, convParameters.biases);
    return convTensorNames;
  }

  /**
   * Create deterministic Conv parameter tensor names.
   *
   * @param layerIndex Layer index.
   * @returns Conv tensor names.
   */
  function createConvTensorNames(layerIndex: number): OnnxConvTensorNames {
    // Step 1: Build names with existing layer-based convention.
    return {
      convWeightName: `ConvW${layerIndex - 1}`,
      convBiasName: `ConvB${layerIndex - 1}`,
    };
  }

  /**
   * Append Conv weight initializer.
   *
   * @param context Conv emission context.
   * @param convTensorNames Conv tensor names.
   * @param weightValues Conv weight values.
   * @returns Nothing.
   */
  function appendConvWeightInitializer(
    context: OnnxConvEmissionContext,
    convTensorNames: OnnxConvTensorNames,
    weightValues: number[],
  ): void {
    // Step 1: Push weight tensor initializer with Conv shape dimensions.
    context.model.graph.initializer.push({
      name: convTensorNames.convWeightName,
      data_type: 1,
      dims: [
        context.convSpec.outChannels,
        context.convSpec.inChannels,
        context.convSpec.kernelHeight,
        context.convSpec.kernelWidth,
      ],
      float_data: weightValues,
    });
  }

  /**
   * Append Conv bias initializer.
   *
   * @param context Conv emission context.
   * @param convTensorNames Conv tensor names.
   * @param biasValues Conv bias values.
   * @returns Nothing.
   */
  function appendConvBiasInitializer(
    context: OnnxConvEmissionContext,
    convTensorNames: OnnxConvTensorNames,
    biasValues: number[],
  ): void {
    // Step 1: Push bias tensor initializer.
    context.model.graph.initializer.push({
      name: convTensorNames.convBiasName,
      data_type: 1,
      dims: [context.convSpec.outChannels],
      float_data: biasValues,
    });
  }

  /**
   * Emit Conv and activation nodes and return activation output name.
   *
   * @param context Conv emission context.
   * @param convTensorNames Conv tensor names.
   * @returns Activation output name.
   */
  function emitConvAndActivationGraph(
    context: OnnxConvEmissionContext,
    convTensorNames: OnnxConvTensorNames,
  ): string {
    // Step 1: Resolve output tensor names.
    const convOutputName = `Conv_${context.layerIndex}`;
    const activationOutputName = `Layer_${context.layerIndex}`;

    // Step 2: Emit Conv and activation operators.
    emitConvNode(context, convTensorNames, convOutputName);
    emitActivationNode(context, convOutputName, activationOutputName);
    return activationOutputName;
  }

  /**
   * Emit ONNX Conv node.
   *
   * @param context Conv emission context.
   * @param convTensorNames Conv tensor names.
   * @param convOutputName Conv output tensor name.
   * @returns Nothing.
   */
  function emitConvNode(
    context: OnnxConvEmissionContext,
    convTensorNames: OnnxConvTensorNames,
    convOutputName: string,
  ): void {
    // Step 1: Resolve ONNX pads attribute values.
    const convPaddingValues = createConvPaddingValues(context.convSpec);

    // Step 2: Push Conv node definition.
    context.model.graph.node.push({
      op_type: 'Conv',
      input: [
        context.previousOutputName,
        convTensorNames.convWeightName,
        convTensorNames.convBiasName,
      ],
      output: [convOutputName],
      name: `conv_l${context.layerIndex}`,
      attributes: [
        {
          name: 'kernel_shape',
          type: 'INTS',
          ints: [context.convSpec.kernelHeight, context.convSpec.kernelWidth],
        },
        {
          name: 'strides',
          type: 'INTS',
          ints: [context.convSpec.strideHeight, context.convSpec.strideWidth],
        },
        { name: 'pads', type: 'INTS', ints: convPaddingValues },
      ],
    });
  }

  /**
   * Create ONNX pads values for Conv node.
   *
   * @param convSpec Conv mapping spec.
   * @returns Padding values in ONNX order.
   */
  function createConvPaddingValues(convSpec: Conv2DMapping): number[] {
    // Step 1: Resolve top/left/bottom/right pads with zero defaults.
    return [
      convSpec.padTop ?? 0,
      convSpec.padLeft ?? 0,
      convSpec.padBottom ?? 0,
      convSpec.padRight ?? 0,
    ];
  }

  /**
   * Emit activation node for Conv output.
   *
   * @param context Conv emission context.
   * @param convOutputName Conv output tensor name.
   * @param activationOutputName Activation output tensor name.
   * @returns Nothing.
   */
  function emitActivationNode(
    context: OnnxConvEmissionContext,
    convOutputName: string,
    activationOutputName: string,
  ): void {
    // Step 1: Resolve activation operator name.
    const activationOperator = resolveActivationOperator(context);

    // Step 2: Push activation node.
    context.model.graph.node.push({
      op_type: activationOperator,
      input: [convOutputName],
      output: [activationOutputName],
      name: `act_conv_l${context.layerIndex}`,
    });
  }

  /**
   * Resolve activation operator for Conv output.
   *
   * @param context Conv emission context.
   * @returns Activation operator name.
   */
  function resolveActivationOperator(context: OnnxConvEmissionContext): string {
    // Step 1: Prefer explicit mapping activation, otherwise infer from nodes.
    return (
      context.convSpec.activation ??
      mapActivationToOnnx(context.currentLayerNodes[0].squash)
    );
  }

  /**
   * Emit optional pooling and flatten nodes for Conv output.
   *
   * @param context Conv emission context.
   * @param activationOutputName Activation output name.
   * @returns Final output tensor name.
   */
  function emitOptionalPoolingAndFlattenForConv(
    context: OnnxConvEmissionContext,
    activationOutputName: string,
  ): string {
    // Step 1: Resolve matching pool spec for current layer.
    const poolSpec = resolvePoolingSpec(context);

    // Step 2: Emit optional pooling/flatten graph and return final output name.
    return emitOptionalPoolingAndFlatten({
      model: context.model,
      options: context.options,
      layerIndex: context.layerIndex,
      sourceOutputName: activationOutputName,
      poolSpec,
    });
  }

  /**
   * Resolve pooling spec for current layer.
   *
   * @param context Conv emission context.
   * @returns Pool mapping spec, if configured.
   */
  function resolvePoolingSpec(
    context: OnnxConvEmissionContext,
  ): Pool2DMapping | undefined {
    // Step 1: Match pooling mapping by after-layer index.
    return context.options.pool2dMappings?.find(
      (poolingMapping) => poolingMapping.afterLayerIndex === context.layerIndex,
    );
  }

  /**
   * Append Conv export metadata entries.
   *
   * @param context Conv emission context.
   * @returns Nothing.
   */
  function appendConvExportMetadata(context: OnnxConvEmissionContext): void {
    // Step 1: Append Conv layer index and mapping spec metadata.
    appendIndexedMetadata(context.model, 'conv2d_layers', context.layerIndex);
    appendMetadataSpec(context.model, 'conv2d_specs', context.convSpec);
  }
}
