import type NeatapticNode from '../../../../node';
import type { OnnxModel } from '../../schema/network.onnx.schema.types';
import type {
  DenseActivationContext,
  DenseActivationNodePayload,
  DenseGemmNodePayload,
  DenseGraphNames,
  DenseInitializerValues,
  DenseLayerContext,
  DenseLayerParams,
  DenseOrderedNodePayload,
  DenseTensorNames,
  OptionalLayerOutputParams,
  PerNeuronConcatNodePayload,
  PerNeuronGraphNames,
  PerNeuronLayerContext,
  PerNeuronLayerParams,
  PerNeuronNodeContext,
  PerNeuronSubgraphContext,
  PerNeuronTensorNames,
  ResidualAddLayerParams,
  SharedActivationNodeBuildParams,
  SharedGemmNodeBuildParams,
} from '../network.onnx.export.types';
import type { NodeInternals } from '../../network.onnx.utils.types';
import {
  buildDenseWeightsAndBiases,
  emitOptionalPoolingAndFlatten,
} from './network.onnx.export-layer-common.utils';
import { resolveOnnxActivationNodeConfig } from '../../network.onnx.layer-analysis.utils';

/**
 * Emit the compact dense export path for a layer whose neurons all share the
 * same activation.
 *
 * This is the cheapest ONNX shape the exporter can produce for a standard MLP
 * layer: one Gemm node for the affine transform and one activation node for the
 * whole layer. The same helper also preserves the library's legacy node-ordering
 * compatibility mode when older snapshots need deterministic graph ordering.
 *
 * @param params Dense emission parameters.
 * @returns Output tensor name.
 * @example
 * ```ts
 * const outputName = emitDenseLayer({
 *   model,
 *   layerIndex: 2,
 *   previousOutputName: 'Layer_1',
 *   previousLayerNodes,
 *   currentLayerNodes,
 *   options: {},
 *   legacyNodeOrdering: false,
 * });
 * ```
 */
export function emitDenseLayer(params: DenseLayerParams): string {
  const denseLayerContext = createDenseLayerContext(params);

  // Step 1: Emit dense initializers from layer traversal inputs.
  const denseTensorNames = emitDenseInitializers(denseLayerContext);

  // Step 2: Resolve graph tensor names and activation context.
  const denseGraphNames = createDenseGraphNames(denseLayerContext.layerIndex);
  const denseActivationContext = createDenseActivationContext(
    denseLayerContext,
    denseTensorNames,
    denseGraphNames,
  );

  // Step 3: Emit the Gemm + activation sequence using the requested ordering.
  emitDenseActivationSubgraph(denseLayerContext.model, denseActivationContext);

  // Step 4: Fold to optional pooling and flatten output.
  return emitDenseLayerOutput(denseLayerContext, denseGraphNames);

  /**
   * Build the compact dense-layer context.
   *
   * @param input Dense emission input.
   * @returns Dense layer context.
   */
  function createDenseLayerContext(input: DenseLayerParams): DenseLayerContext {
    const activationSquash = (
      input.currentLayerNodes[0] as unknown as NodeInternals
    ).squash;
    return {
      ...input,
      activationSquash,
    };
  }

  /**
   * Build names for Gemm and activation outputs.
   *
   * @param currentLayerIndex Dense layer index.
   * @returns Dense graph names.
   */
  function createDenseGraphNames(currentLayerIndex: number): DenseGraphNames {
    return {
      gemmOutputName: `Gemm_${currentLayerIndex}`,
      activationOutputName: `Layer_${currentLayerIndex}`,
    };
  }

  /**
   * Build activation emission context with minimal parameter sprawl.
   *
   * @param layerContext Dense layer context.
   * @param tensorNames Dense initializer names.
   * @param graphNames Dense graph names.
   * @returns Dense activation context.
   */
  function createDenseActivationContext(
    layerContext: DenseLayerContext,
    tensorNames: DenseTensorNames,
    graphNames: DenseGraphNames,
  ): DenseActivationContext {
    return {
      layerIndex: layerContext.layerIndex,
      previousOutputName: layerContext.previousOutputName,
      legacyNodeOrdering: layerContext.legacyNodeOrdering,
      tensorNames,
      graphNames,
      squash: layerContext.activationSquash,
      opset: layerContext.options.opset ?? 18,
    };
  }

  /**
   * Emit optional pooling/flatten output fold.
   *
   * @param layerContext Dense layer context.
   * @param graphNames Dense graph names.
   * @returns Output tensor name.
   */
  function emitDenseLayerOutput(
    layerContext: DenseLayerContext,
    graphNames: DenseGraphNames,
  ): string {
    return emitOptionalLayerOutput({
      model: layerContext.model,
      options: layerContext.options,
      layerIndex: layerContext.layerIndex,
      sourceOutputName: graphNames.activationOutputName,
    });
  }
}

/**
 * Emit the fallback dense-family representation for a layer whose target
 * neurons use different activations.
 *
 * Instead of pretending the layer is homogeneous, this path exports one tiny
 * Gemm-plus-activation subgraph per neuron and then concatenates the results.
 * The graph is larger, but it preserves mixed activation behavior that a single
 * layer-wide activation node cannot express.
 *
 * @param params Per-neuron emission parameters.
 * @returns Output tensor name.
 * @example
 * ```ts
 * const outputName = emitPerNeuronLayer({
 *   model,
 *   layerIndex: 2,
 *   previousOutputName: 'Layer_1',
 *   previousLayerNodes,
 *   currentLayerNodes,
 *   options: { allowMixedActivations: true },
 * });
 * ```
 */
export function emitPerNeuronLayer(params: PerNeuronLayerParams): string {
  const perNeuronLayerContext = createPerNeuronLayerContext(params);

  // Step 1: Build tiny subgraph contexts per target neuron.
  const perNeuronSubgraphContexts = createPerNeuronSubgraphContexts(
    perNeuronLayerContext,
  );

  // Step 2: Emit each per-neuron subgraph and collect outputs.
  const perNeuronActivationOutputs = emitPerNeuronSubgraphs(
    perNeuronSubgraphContexts,
  );

  // Step 3: Emit layer concat for per-neuron outputs.
  const layerOutputName = createPerNeuronLayerOutputName(
    perNeuronLayerContext.layerIndex,
  );
  const perNeuronConcatNode = createPerNeuronConcatNode(
    perNeuronLayerContext,
    perNeuronActivationOutputs,
    layerOutputName,
  );
  appendPerNeuronConcatNode(perNeuronLayerContext.model, perNeuronConcatNode);

  // Step 4: Fold to optional pooling and flatten output.
  return emitPerNeuronLayerOutput(perNeuronLayerContext, layerOutputName);

  /**
   * Build per-neuron layer context.
   *
   * @param input Per-neuron layer params.
   * @returns Per-neuron layer context.
   */
  function createPerNeuronLayerContext(
    input: PerNeuronLayerParams,
  ): PerNeuronLayerContext {
    return { ...input };
  }

  /**
   * Build per-neuron subgraph contexts from layer traversal inputs.
   *
   * @param layerContext Per-neuron layer context.
   * @returns Per-neuron subgraph contexts.
   */
  function createPerNeuronSubgraphContexts(
    layerContext: PerNeuronLayerContext,
  ): PerNeuronSubgraphContext[] {
    return layerContext.currentLayerNodes.map((targetNode, neuronIndex) => ({
      model: layerContext.model,
      layerIndex: layerContext.layerIndex,
      neuronIndex,
      previousOutputName: layerContext.previousOutputName,
      previousLayerNodes: layerContext.previousLayerNodes,
      targetNode,
      opset: layerContext.options.opset ?? 18,
    }));
  }

  /**
   * Emit all per-neuron subgraphs.
   *
   * @param subgraphContexts Per-neuron subgraph contexts.
   * @returns Per-neuron activation outputs.
   */
  function emitPerNeuronSubgraphs(
    subgraphContexts: PerNeuronSubgraphContext[],
  ): string[] {
    return subgraphContexts.map((subgraphContext) =>
      emitPerNeuronSubgraph(subgraphContext),
    );
  }

  /**
   * Build per-neuron layer output tensor name.
   *
   * @param currentLayerIndex Layer index.
   * @returns Layer output name.
   */
  function createPerNeuronLayerOutputName(currentLayerIndex: number): string {
    return `Layer_${currentLayerIndex}`;
  }

  /**
   * Create per-neuron concat node payload.
   *
   * @param layerContext Per-neuron layer context.
   * @param activationOutputs Per-neuron activation output names.
   * @param layerOutputName Concat output name.
   * @returns Concat node payload.
   */
  function createPerNeuronConcatNode(
    layerContext: PerNeuronLayerContext,
    activationOutputs: string[],
    layerOutputName: string,
  ): PerNeuronConcatNodePayload {
    return {
      op_type: 'Concat',
      input: activationOutputs,
      output: [layerOutputName],
      name: `concat_l${layerContext.layerIndex}`,
      attributes: [
        {
          name: 'axis',
          type: 'INT',
          i: layerContext.batchDimension ? 1 : 0,
        },
      ],
    };
  }

  /**
   * Append per-neuron concat node.
   *
   * @param model Target model.
   * @param concatNode Concat node payload.
   * @returns Nothing.
   */
  function appendPerNeuronConcatNode(
    model: OnnxModel,
    concatNode: PerNeuronConcatNodePayload,
  ): void {
    model.graph.node.push(concatNode);
  }

  /**
   * Emit optional pooling/flatten fold for per-neuron layer output.
   *
   * @param layerContext Per-neuron layer context.
   * @param layerOutputName Layer output name.
   * @returns Output tensor name.
   */
  function emitPerNeuronLayerOutput(
    layerContext: PerNeuronLayerContext,
    layerOutputName: string,
  ): string {
    return emitOptionalLayerOutput({
      model: layerContext.model,
      options: layerContext.options,
      layerIndex: layerContext.layerIndex,
      sourceOutputName: layerOutputName,
    });
  }
}

/**
 * Emit a one-hop residual-add dense layer.
 *
 * This subset preserves one skipped source layer by splitting the target layer
 * into two affine branches: the ordinary adjacent-layer Gemm keeps the original
 * bias term, and the skipped source layer emits a bias-free branch whose output
 * is summed before the layer activation.
 *
 * @param params Residual-add emission parameters.
 * @returns Output tensor name.
 */
export function emitResidualAddLayer(params: ResidualAddLayerParams): string {
  const activationSquash = (
    params.currentLayerNodes[0] as unknown as NodeInternals
  ).squash;
  const mainTensorNames = createDenseTensorNames(params.layerIndex);
  const mainInitializerValues = buildDenseWeightsAndBiases(
    params.previousLayerNodes,
    params.currentLayerNodes,
  );
  const residualInitializerValues = buildDenseWeightsAndBiases(
    params.residualSourceLayerNodes,
    params.currentLayerNodes,
  );
  const residualWeightTensorName = `ResidualW_l${params.layerIndex}`;
  const residualBiasTensorName = `ResidualB_l${params.layerIndex}`;
  const mainGemmOutputName = `Gemm_${params.layerIndex}_main`;
  const activationOutputName = `Layer_${params.layerIndex}`;

  // Step 1: Emit adjacent-layer dense initializers.
  appendDenseWeightInitializer(
    {
      ...params,
      activationSquash,
      legacyNodeOrdering: false,
    },
    mainTensorNames.weightTensorName,
    mainInitializerValues.weightMatrixValues,
  );
  appendDenseBiasInitializer(
    {
      ...params,
      activationSquash,
      legacyNodeOrdering: false,
    },
    mainTensorNames.biasTensorName,
    mainInitializerValues.biasVector,
  );

  // Step 2: Emit the skipped-source residual branch initializers.
  params.model.graph.initializer.push({
    name: residualWeightTensorName,
    data_type: 1,
    dims: [
      params.currentLayerNodes.length,
      params.residualSourceLayerNodes.length,
    ],
    float_data: residualInitializerValues.weightMatrixValues,
  });
  params.model.graph.initializer.push({
    name: residualBiasTensorName,
    data_type: 1,
    dims: [params.currentLayerNodes.length],
    float_data: Array.from(
      { length: params.currentLayerNodes.length },
      () => 0,
    ),
  });

  // Step 3: Emit the main Gemm, residual Gemm, Add, and activation nodes.
  params.model.graph.node.push(
    createSharedGemmNodePayload({
      previousOutputName: params.previousOutputName,
      weightTensorName: mainTensorNames.weightTensorName,
      biasTensorName: mainTensorNames.biasTensorName,
      gemmOutputName: mainGemmOutputName,
      nodeName: `gemm_l${params.layerIndex}_main`,
    }),
    createSharedGemmNodePayload({
      previousOutputName: params.residualSourceOutputName,
      weightTensorName: residualWeightTensorName,
      biasTensorName: residualBiasTensorName,
      gemmOutputName: params.branchTensorName,
      nodeName: `residual_gemm_l${params.layerIndex}`,
    }),
    {
      op_type: 'Add',
      input: [mainGemmOutputName, params.branchTensorName],
      output: [params.mergeOutputName],
      name: params.mergeNodeName,
    },
    createSharedActivationNodePayload({
      activationType: resolveOnnxActivationNodeConfig(
        activationSquash,
        params.options.opset ?? 18,
      ).operation,
      activationAttributes: resolveOnnxActivationNodeConfig(
        activationSquash,
        params.options.opset ?? 18,
      ).attributes,
      gemmOutputName: params.mergeOutputName,
      activationOutputName,
      nodeName: `act_l${params.layerIndex}`,
    }),
  );

  // Step 4: Fold to optional pooling and flatten output.
  return emitOptionalLayerOutput({
    model: params.model,
    options: params.options,
    layerIndex: params.layerIndex,
    sourceOutputName: activationOutputName,
  });
}

/**
 * Emit dense initializers and return tensor names.
 *
 * @param layerContext Dense layer context.
 * @returns Tensor names.
 */
function emitDenseInitializers(
  layerContext: DenseLayerContext,
): DenseTensorNames {
  const denseInitializerValues = collectDenseInitializerValues(layerContext);
  const denseTensorNames = createDenseTensorNames(layerContext.layerIndex);

  appendDenseWeightInitializer(
    layerContext,
    denseTensorNames.weightTensorName,
    denseInitializerValues.weightMatrixValues,
  );
  appendDenseBiasInitializer(
    layerContext,
    denseTensorNames.biasTensorName,
    denseInitializerValues.biasVector,
  );

  return denseTensorNames;
}

/**
 * Collect dense weight matrix and bias vector values.
 *
 * @param layerContext Dense layer context.
 * @returns Dense initializer values.
 */
function collectDenseInitializerValues(
  layerContext: DenseLayerContext,
): DenseInitializerValues {
  const { weightMatrixValues, biasVector } = buildDenseWeightsAndBiases(
    layerContext.previousLayerNodes,
    layerContext.currentLayerNodes,
  );
  return { weightMatrixValues, biasVector };
}

/**
 * Build dense tensor names for initializer emission.
 *
 * @param layerIndex Layer index.
 * @returns Dense tensor names.
 */
function createDenseTensorNames(layerIndex: number): DenseTensorNames {
  return {
    weightTensorName: `W${layerIndex - 1}`,
    biasTensorName: `B${layerIndex - 1}`,
  };
}

/**
 * Append dense weight initializer.
 *
 * @param layerContext Dense layer context.
 * @param weightTensorName Weight tensor name.
 * @param weightMatrixValues Weight values.
 * @returns Nothing.
 */
function appendDenseWeightInitializer(
  layerContext: DenseLayerContext,
  weightTensorName: string,
  weightMatrixValues: number[],
): void {
  layerContext.model.graph.initializer.push({
    name: weightTensorName,
    data_type: 1,
    dims: [
      layerContext.currentLayerNodes.length,
      layerContext.previousLayerNodes.length,
    ],
    float_data: weightMatrixValues,
  });
}

/**
 * Append dense bias initializer.
 *
 * @param layerContext Dense layer context.
 * @param biasTensorName Bias tensor name.
 * @param biasVector Bias vector values.
 * @returns Nothing.
 */
function appendDenseBiasInitializer(
  layerContext: DenseLayerContext,
  biasTensorName: string,
  biasVector: number[],
): void {
  layerContext.model.graph.initializer.push({
    name: biasTensorName,
    data_type: 1,
    dims: [layerContext.currentLayerNodes.length],
    float_data: biasVector,
  });
}

/**
 * Emit Gemm and activation nodes using requested ordering.
 *
 * @param model Target ONNX model.
 * @param denseActivationContext Dense activation context.
 * @returns Nothing.
 */
function emitDenseActivationSubgraph(
  model: OnnxModel,
  denseActivationContext: DenseActivationContext,
): void {
  // Step 1: Build node payloads.
  const activationNode = createActivationNode(denseActivationContext);
  const gemmNode = createGemmNode(denseActivationContext);

  // Step 2: Resolve ordered sequence.
  const orderedNodes = resolveDenseNodeOrder(
    gemmNode,
    activationNode,
    denseActivationContext.legacyNodeOrdering,
  );

  // Step 3: Emit the ordered nodes.
  appendDenseNodes(model, orderedNodes);
}

/**
 * Resolve dense node order for legacy and current exports.
 *
 * @param gemmNode Gemm node.
 * @param activationNode Activation node.
 * @param legacyNodeOrdering Whether legacy ordering is required.
 * @returns Ordered node list.
 */
function resolveDenseNodeOrder(
  gemmNode: DenseGemmNodePayload,
  activationNode: DenseActivationNodePayload,
  legacyNodeOrdering: boolean,
): DenseOrderedNodePayload[] {
  if (legacyNodeOrdering) {
    return [activationNode, gemmNode];
  }
  return [gemmNode, activationNode];
}

/**
 * Append ordered dense nodes to the model graph.
 *
 * @param model Target model.
 * @param orderedNodes Ordered dense nodes.
 * @returns Nothing.
 */
function appendDenseNodes(
  model: OnnxModel,
  orderedNodes: DenseOrderedNodePayload[],
): void {
  model.graph.node.push(...orderedNodes);
}

/**
 * Create dense Gemm node definition.
 *
 * @param denseActivationContext Dense activation context.
 * @returns ONNX Gemm node payload.
 */
function createGemmNode(
  denseActivationContext: DenseActivationContext,
): DenseGemmNodePayload {
  return createSharedGemmNodePayload({
    previousOutputName: denseActivationContext.previousOutputName,
    weightTensorName: denseActivationContext.tensorNames.weightTensorName,
    biasTensorName: denseActivationContext.tensorNames.biasTensorName,
    gemmOutputName: denseActivationContext.graphNames.gemmOutputName,
    nodeName: `gemm_l${denseActivationContext.layerIndex}`,
  });
}

/**
 * Create dense activation node definition.
 *
 * @param denseActivationContext Dense activation context.
 * @returns ONNX activation node payload.
 */
function createActivationNode(
  denseActivationContext: DenseActivationContext,
): DenseActivationNodePayload {
  const activationConfig = resolveOnnxActivationNodeConfig(
    denseActivationContext.squash,
    denseActivationContext.opset,
  );

  return createSharedActivationNodePayload({
    activationType: activationConfig.operation,
    activationAttributes: activationConfig.attributes,
    gemmOutputName: denseActivationContext.graphNames.gemmOutputName,
    activationOutputName:
      denseActivationContext.graphNames.activationOutputName,
    nodeName: `act_l${denseActivationContext.layerIndex}`,
  });
}

/**
 * Emit per-neuron Gemm + activation subgraph.
 *
 * @param perNeuronSubgraphContext Per-neuron subgraph context.
 * @returns Per-neuron activation output name.
 */
function emitPerNeuronSubgraph(
  perNeuronSubgraphContext: PerNeuronSubgraphContext,
): string {
  const perNeuronNodeContext = createPerNeuronNodeContext(
    perNeuronSubgraphContext,
  );

  // Step 1: Build per-neuron initializer values and names.
  const weightRow = buildSingleNeuronWeightRow(
    perNeuronNodeContext.targetNodeInternal,
    perNeuronNodeContext.previousLayerNodes,
  );
  const perNeuronTensorNames = createPerNeuronTensorNames(perNeuronNodeContext);
  const perNeuronGraphNames = createPerNeuronGraphNames(perNeuronNodeContext);

  // Step 2: Emit per-neuron initializers.
  appendPerNeuronWeightInitializer(
    perNeuronNodeContext,
    perNeuronTensorNames,
    weightRow,
  );
  appendPerNeuronBiasInitializer(perNeuronNodeContext, perNeuronTensorNames);

  // Step 3: Emit per-neuron node sequence.
  const perNeuronGemmNode = createPerNeuronGemmNode(
    perNeuronNodeContext,
    perNeuronTensorNames,
    perNeuronGraphNames,
  );
  const perNeuronActivationNode = createPerNeuronActivationNode(
    perNeuronNodeContext,
    perNeuronGraphNames,
  );
  appendPerNeuronSubgraphNodes(
    perNeuronNodeContext.model,
    perNeuronGemmNode,
    perNeuronActivationNode,
  );

  // Step 4: Fold to activation output.
  return perNeuronGraphNames.activationOutputName;

  /**
   * Build normalized per-neuron node context.
   *
   * @param subgraphContext Per-neuron subgraph context.
   * @returns Per-neuron node context.
   */
  function createPerNeuronNodeContext(
    subgraphContext: PerNeuronSubgraphContext,
  ): PerNeuronNodeContext {
    return {
      model: subgraphContext.model,
      layerIndex: subgraphContext.layerIndex,
      neuronIndex: subgraphContext.neuronIndex,
      previousOutputName: subgraphContext.previousOutputName,
      previousLayerNodes: subgraphContext.previousLayerNodes,
      targetNodeInternal:
        subgraphContext.targetNode as unknown as NodeInternals,
      opset: subgraphContext.opset,
    };
  }

  /**
   * Build per-neuron initializer tensor names.
   *
   * @param nodeContext Per-neuron node context.
   * @returns Per-neuron tensor names.
   */
  function createPerNeuronTensorNames(
    nodeContext: PerNeuronNodeContext,
  ): PerNeuronTensorNames {
    return {
      weightTensorName: `W${nodeContext.layerIndex - 1}_n${nodeContext.neuronIndex}`,
      biasTensorName: `B${nodeContext.layerIndex - 1}_n${nodeContext.neuronIndex}`,
    };
  }

  /**
   * Build per-neuron graph tensor names.
   *
   * @param nodeContext Per-neuron node context.
   * @returns Per-neuron graph names.
   */
  function createPerNeuronGraphNames(
    nodeContext: PerNeuronNodeContext,
  ): PerNeuronGraphNames {
    return {
      gemmOutputName: `Gemm_${nodeContext.layerIndex}_n${nodeContext.neuronIndex}`,
      activationOutputName: `Layer_${nodeContext.layerIndex}_n${nodeContext.neuronIndex}`,
    };
  }

  /**
   * Append per-neuron weight initializer.
   *
   * @param nodeContext Per-neuron node context.
   * @param tensorNames Per-neuron tensor names.
   * @param weightRow Weight row values.
   * @returns Nothing.
   */
  function appendPerNeuronWeightInitializer(
    nodeContext: PerNeuronNodeContext,
    tensorNames: PerNeuronTensorNames,
    weightRow: number[],
  ): void {
    nodeContext.model.graph.initializer.push({
      name: tensorNames.weightTensorName,
      data_type: 1,
      dims: [1, nodeContext.previousLayerNodes.length],
      float_data: weightRow,
    });
  }

  /**
   * Append per-neuron bias initializer.
   *
   * @param nodeContext Per-neuron node context.
   * @param tensorNames Per-neuron tensor names.
   * @returns Nothing.
   */
  function appendPerNeuronBiasInitializer(
    nodeContext: PerNeuronNodeContext,
    tensorNames: PerNeuronTensorNames,
  ): void {
    nodeContext.model.graph.initializer.push({
      name: tensorNames.biasTensorName,
      data_type: 1,
      dims: [1],
      float_data: [nodeContext.targetNodeInternal.bias],
    });
  }

  /**
   * Build per-neuron Gemm node payload.
   *
   * @param nodeContext Per-neuron node context.
   * @param tensorNames Per-neuron tensor names.
   * @param graphNames Per-neuron graph names.
   * @returns Gemm node payload.
   */
  function createPerNeuronGemmNode(
    nodeContext: PerNeuronNodeContext,
    tensorNames: PerNeuronTensorNames,
    graphNames: PerNeuronGraphNames,
  ): DenseGemmNodePayload {
    return createSharedGemmNodePayload({
      previousOutputName: nodeContext.previousOutputName,
      weightTensorName: tensorNames.weightTensorName,
      biasTensorName: tensorNames.biasTensorName,
      gemmOutputName: graphNames.gemmOutputName,
      nodeName: `gemm_l${nodeContext.layerIndex}_n${nodeContext.neuronIndex}`,
    });
  }

  /**
   * Build per-neuron activation node payload.
   *
   * @param nodeContext Per-neuron node context.
   * @param graphNames Per-neuron graph names.
   * @returns Activation node payload.
   */
  function createPerNeuronActivationNode(
    nodeContext: PerNeuronNodeContext,
    graphNames: PerNeuronGraphNames,
  ): DenseActivationNodePayload {
    const activationConfig = resolveOnnxActivationNodeConfig(
      nodeContext.targetNodeInternal.squash,
      nodeContext.opset,
    );

    return createSharedActivationNodePayload({
      activationType: activationConfig.operation,
      activationAttributes: activationConfig.attributes,
      gemmOutputName: graphNames.gemmOutputName,
      activationOutputName: graphNames.activationOutputName,
      nodeName: `act_l${nodeContext.layerIndex}_n${nodeContext.neuronIndex}`,
    });
  }

  /**
   * Append per-neuron Gemm and activation nodes.
   *
   * @param model Target model.
   * @param gemmNode Gemm node payload.
   * @param activationNode Activation node payload.
   * @returns Nothing.
   */
  function appendPerNeuronSubgraphNodes(
    model: OnnxModel,
    gemmNode: DenseGemmNodePayload,
    activationNode: DenseActivationNodePayload,
  ): void {
    model.graph.node.push(gemmNode);
    model.graph.node.push(activationNode);
  }
}

/**
 * Build one neuron's incoming weight row against previous layer.
 *
 * @param targetNodeInternal Target node internals.
 * @param previousLayerNodes Previous layer nodes.
 * @returns Weight row values.
 */
function buildSingleNeuronWeightRow(
  targetNodeInternal: NodeInternals,
  previousLayerNodes: NeatapticNode[],
): number[] {
  return previousLayerNodes.map((sourceNode) =>
    resolveSingleNeuronInboundWeight(targetNodeInternal, sourceNode),
  );
}

/**
 * Resolve one inbound connection weight for a source node.
 *
 * @param targetNodeInternal Target node internals.
 * @param sourceNode Source node.
 * @returns Inbound weight or zero when missing.
 */
function resolveSingleNeuronInboundWeight(
  targetNodeInternal: NodeInternals,
  sourceNode: NeatapticNode,
): number {
  const inboundConnection = targetNodeInternal.connections.in.find(
    (connection) => connection.from === sourceNode,
  );
  return inboundConnection?.weight ?? 0;
}

/**
 * Build a shared Gemm node payload.
 *
 * @param params Shared Gemm build parameters.
 * @returns Gemm node payload.
 */
function createSharedGemmNodePayload(
  params: SharedGemmNodeBuildParams,
): DenseGemmNodePayload {
  return {
    op_type: 'Gemm',
    input: [
      params.previousOutputName,
      params.weightTensorName,
      params.biasTensorName,
    ],
    output: [params.gemmOutputName],
    name: params.nodeName,
    attributes: createDefaultGemmAttributes(),
  };
}

/**
 * Build a shared activation node payload.
 *
 * @param params Shared activation build parameters.
 * @returns Activation node payload.
 */
function createSharedActivationNodePayload(
  params: SharedActivationNodeBuildParams,
): DenseActivationNodePayload {
  return {
    op_type: params.activationType,
    input: [params.gemmOutputName],
    output: [params.activationOutputName],
    name: params.nodeName,
    attributes: params.activationAttributes,
  };
}

/**
 * Build default Gemm attributes for ONNX export.
 *
 * @returns Default Gemm attribute list.
 */
function createDefaultGemmAttributes(): DenseGemmNodePayload['attributes'] {
  return [
    { name: 'alpha', type: 'FLOAT', f: 1 },
    { name: 'beta', type: 'FLOAT', f: 1 },
    { name: 'transB', type: 'INT', i: 1 },
  ];
}

/**
 * Emit optional pooling and flatten output fold.
 *
 * @param params Optional output parameters.
 * @returns Output tensor name.
 */
function emitOptionalLayerOutput(params: OptionalLayerOutputParams): string {
  return emitOptionalPoolingAndFlatten({
    model: params.model,
    options: params.options,
    layerIndex: params.layerIndex,
    sourceOutputName: params.sourceOutputName,
    poolSpec: params.options.pool2dMappings?.find(
      (pooling) => pooling.afterLayerIndex === params.layerIndex,
    ),
  });
}
