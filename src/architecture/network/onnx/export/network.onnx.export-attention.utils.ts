import type NeatapticNode from '../../../node';
import type {
  OnnxMetadataProperty,
  OnnxModel,
  OnnxNode,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import type {
  AttentionMapping,
  OnnxExportOptions,
} from './network.onnx.export.types';

const ADVANCED_GRAPH_ATTENTION_BLOCKS_KEY = 'advanced_graph_attention_blocks';
const ONNX_FLOAT_DATA_TYPE = 1;
const ONNX_INT64_DATA_TYPE = 7;

type AttentionShadowMetadata = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  sequenceLength: number;
  modelWidth: number;
  heads: number;
  shadowOutputName: string;
};

type AttentionEmissionContext = {
  sourceLayerIndex: number;
  targetLayerIndex: number;
  sequenceLength: number;
  modelWidth: number;
  heads: number;
  headWidth: number;
  sourceOutputName: string;
  queryWeightTensorName: string;
  keyWeightTensorName: string;
  valueWeightTensorName: string;
  queryBiasTensorName: string;
  keyBiasTensorName: string;
  valueBiasTensorName: string;
  inputShapeTensorName: string;
  splitShapeTensorName: string;
  mergeShapeTensorName: string;
  flatShapeTensorName: string;
  scaleTensorName: string;
  outputWeightTensorName: string;
  outputBiasTensorName: string;
  inputReshapeOutputName: string;
  queryOutputName: string;
  keyOutputName: string;
  valueOutputName: string;
  queryHeadsOutputName: string;
  keyHeadsOutputName: string;
  valueHeadsOutputName: string;
  scoreOutputName: string;
  scaledScoreOutputName: string;
  probabilityOutputName: string;
  contextHeadsOutputName: string;
  contextMergedOutputName: string;
  contextFlatOutputName: string;
  shadowOutputName: string;
  scaleScores: boolean;
  mapping: AttentionMapping;
};

/**
 * Emit explicit shadow attention blocks for the Phase 5E fixed-width subset.
 *
 * The current runtime does not execute native attention semantics, so this
 * pass follows the same strategy used by the fused recurrent heuristics: emit a
 * deterministic ONNX attention subgraph without replacing the stable dense path
 * that the importer already round-trips.
 *
 * @param model Target ONNX model.
 * @param layers Resolved layered network ordering.
 * @param options Export options.
 * @param layerOutputNamesByLayerIndex Emitted canonical layer output names.
 * @param includeMetadata Whether attention metadata should be recorded.
 * @returns Nothing.
 */
export function emitShadowAttentionMappings(
  model: OnnxModel,
  layers: NeatapticNode[][],
  options: OnnxExportOptions,
  layerOutputNamesByLayerIndex: ReadonlyMap<number, string>,
  includeMetadata: boolean,
): void {
  const attentionMappings = (options.attentionMappings ?? []).toSorted(
    (leftMapping, rightMapping) =>
      leftMapping.layerIndex - rightMapping.layerIndex,
  );
  const emittedLayerIndices = new Set<number>();

  attentionMappings.forEach((attentionMapping) => {
    if (emittedLayerIndices.has(attentionMapping.layerIndex)) {
      return;
    }

    const emissionContext = createAttentionEmissionContext(
      model,
      layers,
      attentionMapping,
      layerOutputNamesByLayerIndex,
    );
    if (!emissionContext) {
      return;
    }

    emittedLayerIndices.add(attentionMapping.layerIndex);
    appendAttentionInitializers(model, emissionContext);
    appendAttentionNodes(model, emissionContext);
    appendAttentionMetadata(model, emissionContext, includeMetadata);
  });
}

function createAttentionEmissionContext(
  model: OnnxModel,
  layers: NeatapticNode[][],
  attentionMapping: AttentionMapping,
  layerOutputNamesByLayerIndex: ReadonlyMap<number, string>,
): AttentionEmissionContext | null {
  const targetLayerIndex = attentionMapping.layerIndex;
  const sourceLayerIndex = targetLayerIndex - 1;

  if (targetLayerIndex <= 0 || targetLayerIndex >= layers.length) {
    return null;
  }

  if (
    attentionMapping.sequenceLength <= 0 ||
    attentionMapping.modelWidth <= 0 ||
    attentionMapping.heads <= 0
  ) {
    return null;
  }

  if (attentionMapping.modelWidth % attentionMapping.heads !== 0) {
    return null;
  }

  const sourceLayerNodes = layers[sourceLayerIndex] ?? [];
  if (
    sourceLayerNodes.length !==
    attentionMapping.sequenceLength * attentionMapping.modelWidth
  ) {
    return null;
  }

  if (
    !hasSquareProjection(
      attentionMapping.queryWeights,
      attentionMapping.modelWidth,
    ) ||
    !hasSquareProjection(
      attentionMapping.keyWeights,
      attentionMapping.modelWidth,
    ) ||
    !hasSquareProjection(
      attentionMapping.valueWeights,
      attentionMapping.modelWidth,
    ) ||
    !hasBiasWidth(attentionMapping.queryBias, attentionMapping.modelWidth) ||
    !hasBiasWidth(attentionMapping.keyBias, attentionMapping.modelWidth) ||
    !hasBiasWidth(attentionMapping.valueBias, attentionMapping.modelWidth)
  ) {
    return null;
  }

  const sourceOutputName = layerOutputNamesByLayerIndex.get(sourceLayerIndex);
  if (!sourceOutputName) {
    return null;
  }

  const outputWeightTensorName = `W${targetLayerIndex - 1}`;
  const outputBiasTensorName = `B${targetLayerIndex - 1}`;
  if (
    !hasInitializer(model, outputWeightTensorName) ||
    !hasInitializer(model, outputBiasTensorName)
  ) {
    return null;
  }

  const headWidth = attentionMapping.modelWidth / attentionMapping.heads;
  const suffix = `_l${targetLayerIndex}`;
  return {
    sourceLayerIndex,
    targetLayerIndex,
    sequenceLength: attentionMapping.sequenceLength,
    modelWidth: attentionMapping.modelWidth,
    heads: attentionMapping.heads,
    headWidth,
    sourceOutputName,
    queryWeightTensorName: `AttentionQW${suffix}`,
    keyWeightTensorName: `AttentionKW${suffix}`,
    valueWeightTensorName: `AttentionVW${suffix}`,
    queryBiasTensorName: `AttentionQB${suffix}`,
    keyBiasTensorName: `AttentionKB${suffix}`,
    valueBiasTensorName: `AttentionVB${suffix}`,
    inputShapeTensorName: `AttentionInputShape${suffix}`,
    splitShapeTensorName: `AttentionSplitShape${suffix}`,
    mergeShapeTensorName: `AttentionMergeShape${suffix}`,
    flatShapeTensorName: `AttentionFlatShape${suffix}`,
    scaleTensorName: `AttentionScale${suffix}`,
    outputWeightTensorName,
    outputBiasTensorName,
    inputReshapeOutputName: `AttentionInput_${targetLayerIndex}`,
    queryOutputName: `AttentionQ_${targetLayerIndex}`,
    keyOutputName: `AttentionK_${targetLayerIndex}`,
    valueOutputName: `AttentionV_${targetLayerIndex}`,
    queryHeadsOutputName: `AttentionQHeads_${targetLayerIndex}`,
    keyHeadsOutputName: `AttentionKHeads_${targetLayerIndex}`,
    valueHeadsOutputName: `AttentionVHeads_${targetLayerIndex}`,
    scoreOutputName: `AttentionScores_${targetLayerIndex}`,
    scaledScoreOutputName: `AttentionScaledScores_${targetLayerIndex}`,
    probabilityOutputName: `AttentionProbabilities_${targetLayerIndex}`,
    contextHeadsOutputName: `AttentionContextHeads_${targetLayerIndex}`,
    contextMergedOutputName: `AttentionContextMerged_${targetLayerIndex}`,
    contextFlatOutputName: `AttentionContextFlat_${targetLayerIndex}`,
    shadowOutputName: `AttentionShadow_${targetLayerIndex}`,
    scaleScores: attentionMapping.scaleScores ?? true,
    mapping: attentionMapping,
  };
}

function hasSquareProjection(weights: number[], modelWidth: number): boolean {
  return weights.length === modelWidth * modelWidth;
}

function hasBiasWidth(biasValues: number[], modelWidth: number): boolean {
  return biasValues.length === modelWidth;
}

function hasInitializer(model: OnnxModel, tensorName: string): boolean {
  return model.graph.initializer.some(
    (initializerEntry) => initializerEntry.name === tensorName,
  );
}

function appendAttentionInitializers(
  model: OnnxModel,
  context: AttentionEmissionContext,
): void {
  model.graph.initializer.push(
    createFloatTensor(
      context.queryWeightTensorName,
      [context.modelWidth, context.modelWidth],
      context.mapping.queryWeights,
    ),
    createFloatTensor(
      context.keyWeightTensorName,
      [context.modelWidth, context.modelWidth],
      context.mapping.keyWeights,
    ),
    createFloatTensor(
      context.valueWeightTensorName,
      [context.modelWidth, context.modelWidth],
      context.mapping.valueWeights,
    ),
    createFloatTensor(
      context.queryBiasTensorName,
      [context.modelWidth],
      context.mapping.queryBias,
    ),
    createFloatTensor(
      context.keyBiasTensorName,
      [context.modelWidth],
      context.mapping.keyBias,
    ),
    createFloatTensor(
      context.valueBiasTensorName,
      [context.modelWidth],
      context.mapping.valueBias,
    ),
    createInt64Tensor(
      context.inputShapeTensorName,
      [3],
      [1, context.sequenceLength, context.modelWidth],
    ),
    createInt64Tensor(
      context.splitShapeTensorName,
      [4],
      [1, context.sequenceLength, context.heads, context.headWidth],
    ),
    createInt64Tensor(
      context.mergeShapeTensorName,
      [3],
      [1, context.sequenceLength, context.modelWidth],
    ),
    createInt64Tensor(
      context.flatShapeTensorName,
      [1],
      [context.sequenceLength * context.modelWidth],
    ),
  );

  if (!context.scaleScores) {
    return;
  }

  model.graph.initializer.push(
    createFloatTensor(
      context.scaleTensorName,
      [1],
      [Math.sqrt(context.headWidth)],
    ),
  );
}

function createFloatTensor(
  tensorName: string,
  dims: number[],
  floatData: number[],
): OnnxTensor {
  return {
    name: tensorName,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims,
    float_data: floatData,
  };
}

function createInt64Tensor(
  tensorName: string,
  dims: number[],
  int64Data: number[],
): OnnxTensor {
  return {
    name: tensorName,
    data_type: ONNX_INT64_DATA_TYPE,
    dims,
    float_data: [],
    int64_data: int64Data,
  };
}

function appendAttentionNodes(
  model: OnnxModel,
  context: AttentionEmissionContext,
): void {
  model.graph.node.push(
    createReshapeNode(
      context.sourceOutputName,
      context.inputShapeTensorName,
      context.inputReshapeOutputName,
      `attention_l${context.targetLayerIndex}_input_reshape`,
    ),
    ...createProjectionBranchNodes(
      context,
      'query',
      context.queryWeightTensorName,
      context.queryBiasTensorName,
      context.queryOutputName,
      context.queryHeadsOutputName,
      [0, 2, 1, 3],
    ),
    ...createProjectionBranchNodes(
      context,
      'key',
      context.keyWeightTensorName,
      context.keyBiasTensorName,
      context.keyOutputName,
      context.keyHeadsOutputName,
      [0, 2, 3, 1],
    ),
    ...createProjectionBranchNodes(
      context,
      'value',
      context.valueWeightTensorName,
      context.valueBiasTensorName,
      context.valueOutputName,
      context.valueHeadsOutputName,
      [0, 2, 1, 3],
    ),
    createMatMulNode(
      context.queryHeadsOutputName,
      context.keyHeadsOutputName,
      context.scoreOutputName,
      `attention_l${context.targetLayerIndex}_score_matmul`,
    ),
    ...(context.scaleScores
      ? [
          createBinaryNode(
            'Div',
            context.scoreOutputName,
            context.scaleTensorName,
            context.scaledScoreOutputName,
            `attention_l${context.targetLayerIndex}_scale`,
          ),
        ]
      : []),
    createSoftmaxNode(
      context.scaleScores
        ? context.scaledScoreOutputName
        : context.scoreOutputName,
      context.probabilityOutputName,
      `attention_l${context.targetLayerIndex}_softmax`,
    ),
    createMatMulNode(
      context.probabilityOutputName,
      context.valueHeadsOutputName,
      context.contextHeadsOutputName,
      `attention_l${context.targetLayerIndex}_context_matmul`,
    ),
    createTransposeNode(
      context.contextHeadsOutputName,
      `AttentionContextTransposed_${context.targetLayerIndex}`,
      `attention_l${context.targetLayerIndex}_context_transpose`,
      [0, 2, 1, 3],
    ),
    createReshapeNode(
      `AttentionContextTransposed_${context.targetLayerIndex}`,
      context.mergeShapeTensorName,
      context.contextMergedOutputName,
      `attention_l${context.targetLayerIndex}_context_merge`,
    ),
    createReshapeNode(
      context.contextMergedOutputName,
      context.flatShapeTensorName,
      context.contextFlatOutputName,
      `attention_l${context.targetLayerIndex}_context_flatten`,
    ),
    createGemmNode(
      context.contextFlatOutputName,
      context.outputWeightTensorName,
      context.outputBiasTensorName,
      context.shadowOutputName,
      `attention_l${context.targetLayerIndex}_output_projection`,
    ),
  );
}

function createProjectionBranchNodes(
  context: AttentionEmissionContext,
  branchLabel: 'query' | 'key' | 'value',
  weightTensorName: string,
  biasTensorName: string,
  branchOutputName: string,
  headsOutputName: string,
  transposePermutation: number[],
): OnnxNode[] {
  const matMulOutputName = `${branchOutputName}_MatMul`;
  const splitOutputName = `${branchOutputName}_Split`;

  return [
    createMatMulNode(
      context.inputReshapeOutputName,
      weightTensorName,
      matMulOutputName,
      `attention_l${context.targetLayerIndex}_${branchLabel}_matmul`,
    ),
    createBinaryNode(
      'Add',
      matMulOutputName,
      biasTensorName,
      branchOutputName,
      `attention_l${context.targetLayerIndex}_${branchLabel}_bias`,
    ),
    createReshapeNode(
      branchOutputName,
      context.splitShapeTensorName,
      splitOutputName,
      `attention_l${context.targetLayerIndex}_${branchLabel}_split`,
    ),
    createTransposeNode(
      splitOutputName,
      headsOutputName,
      `attention_l${context.targetLayerIndex}_${branchLabel}_transpose`,
      transposePermutation,
    ),
  ];
}

function createReshapeNode(
  sourceInputName: string,
  shapeTensorName: string,
  outputName: string,
  nodeName: string,
): OnnxNode {
  return {
    op_type: 'Reshape',
    input: [sourceInputName, shapeTensorName],
    output: [outputName],
    name: nodeName,
  };
}

function createMatMulNode(
  leftInputName: string,
  rightInputName: string,
  outputName: string,
  nodeName: string,
): OnnxNode {
  return {
    op_type: 'MatMul',
    input: [leftInputName, rightInputName],
    output: [outputName],
    name: nodeName,
  };
}

function createBinaryNode(
  operatorType: 'Add' | 'Div',
  leftInputName: string,
  rightInputName: string,
  outputName: string,
  nodeName: string,
): OnnxNode {
  return {
    op_type: operatorType,
    input: [leftInputName, rightInputName],
    output: [outputName],
    name: nodeName,
  };
}

function createTransposeNode(
  sourceInputName: string,
  outputName: string,
  nodeName: string,
  permutation: number[],
): OnnxNode {
  return {
    op_type: 'Transpose',
    input: [sourceInputName],
    output: [outputName],
    name: nodeName,
    attributes: [
      {
        name: 'perm',
        type: 'INTS',
        ints: permutation,
      },
    ],
  };
}

function createSoftmaxNode(
  sourceInputName: string,
  outputName: string,
  nodeName: string,
): OnnxNode {
  return {
    op_type: 'Softmax',
    input: [sourceInputName],
    output: [outputName],
    name: nodeName,
    attributes: [
      {
        name: 'axis',
        type: 'INT',
        i: -1,
      },
    ],
  };
}

function createGemmNode(
  previousOutputName: string,
  weightTensorName: string,
  biasTensorName: string,
  outputName: string,
  nodeName: string,
): OnnxNode {
  return {
    op_type: 'Gemm',
    input: [previousOutputName, weightTensorName, biasTensorName],
    output: [outputName],
    name: nodeName,
    attributes: [
      { name: 'alpha', type: 'FLOAT', f: 1 },
      { name: 'beta', type: 'FLOAT', f: 1 },
      { name: 'transB', type: 'INT', i: 1 },
    ],
  };
}

function appendAttentionMetadata(
  model: OnnxModel,
  context: AttentionEmissionContext,
  includeMetadata: boolean,
): void {
  if (!includeMetadata) {
    return;
  }

  const attentionMetadata: AttentionShadowMetadata = {
    sourceLayerIndex: context.sourceLayerIndex,
    targetLayerIndex: context.targetLayerIndex,
    sequenceLength: context.sequenceLength,
    modelWidth: context.modelWidth,
    heads: context.heads,
    shadowOutputName: context.shadowOutputName,
  };
  const existingMetadataProperty = model.metadata_props?.find(
    (metadataProperty) =>
      metadataProperty.key === ADVANCED_GRAPH_ATTENTION_BLOCKS_KEY,
  );

  if (!existingMetadataProperty) {
    appendMetadataProperty(
      model,
      buildMetadataProperty(ADVANCED_GRAPH_ATTENTION_BLOCKS_KEY, [
        attentionMetadata,
      ]),
    );
    return;
  }

  try {
    const parsedAttentionMetadata = JSON.parse(existingMetadataProperty.value);
    const nextAttentionMetadata = Array.isArray(parsedAttentionMetadata)
      ? [...parsedAttentionMetadata, attentionMetadata]
      : [attentionMetadata];
    existingMetadataProperty.value = JSON.stringify(nextAttentionMetadata);
  } catch {
    existingMetadataProperty.value = JSON.stringify([attentionMetadata]);
  }
}

function buildMetadataProperty(
  key: string,
  metadataValue: AttentionShadowMetadata[],
): OnnxMetadataProperty {
  return {
    key,
    value: JSON.stringify(metadataValue),
  };
}

function appendMetadataProperty(
  model: OnnxModel,
  metadataProperty: OnnxMetadataProperty,
): void {
  model.metadata_props = [...(model.metadata_props ?? []), metadataProperty];
}
