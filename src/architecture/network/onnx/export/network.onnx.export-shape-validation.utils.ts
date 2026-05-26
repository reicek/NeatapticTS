import type {
  Conv2DMapping,
  OnnxAttribute,
  OnnxModel,
  OnnxNode,
  OnnxTensor,
} from '../schema/network.onnx.schema.types';
import { NetworkOnnxShapeValidationError } from '../network.onnx.errors';

const SAME_SHAPE_OPERATIONS = new Set([
  'Cast',
  'DequantizeLinear',
  'Gelu',
  'Identity',
  'Logistic',
  'Mish',
  'QuantizeLinear',
  'Relu',
  'Selu',
  'Sigmoid',
  'Softplus',
  'Softsign',
  'Tanh',
]);

const BINARY_BROADCAST_OPERATIONS = new Set(['Add', 'Div']);

const FUSED_RECURRENT_OPERATIONS = new Set(['GRU', 'LSTM']);

type ShapeDimension = number | string;
type TensorShape = ShapeDimension[];

type TensorShapeLedgerEntry = {
  resolvedShape: TensorShape;
  fallbackShape?: TensorShape;
};

type NodeShapeInferenceContext = {
  graphNode: OnnxNode;
  resolvedInputEntries: TensorShapeLedgerEntry[];
  resolvedInputShapes: TensorShape[];
  initializerTensorsByName: Map<string, OnnxTensor>;
  convSpecsByLayerIndex: Map<number, Conv2DMapping>;
};

/**
 * Validate the exporter-owned ONNX tensor ledger before the model leaves the builder.
 *
 * This validator is intentionally conservative and repo-shaped rather than a full
 * protobuf-level ONNX checker. It understands the operator subset that the
 * exporter already emits and verifies that the graph stays dimensionally
 * coherent across dense, residual, concat, recurrent, spatial, and attention
 * helper paths.
 *
 * @param model - ONNX-like model to validate.
 * @returns Nothing.
 */
export function validateOnnxModelShapes(model: OnnxModel): void {
  // Step 1: Seed the shape ledger from declared graph inputs and initializers.
  const initializerTensorsByName = createInitializerTensorMap(
    model.graph.initializer,
  );
  validateInitializerPayloadSizes(model.graph.initializer);
  const tensorShapesByName = createInitialTensorShapeLedger(model);
  const convSpecsByLayerIndex = collectConvSpecsByLayerIndex(model);

  // Step 2: Walk the ordered graph and infer output shapes one node at a time.
  let pendingGraphNodes = [...model.graph.node];

  while (pendingGraphNodes.length > 0) {
    const deferredGraphNodes: OnnxNode[] = [];
    let resolvedGraphNodeCount = 0;

    pendingGraphNodes.forEach((graphNode) => {
      const resolvedInputEntries = resolveInputShapes(
        tensorShapesByName,
        graphNode,
      );
      if (!resolvedInputEntries) {
        deferredGraphNodes.push(graphNode);
        return;
      }

      const resolvedInputShapes = resolvedInputEntries.map(
        (shapeLedgerEntry) => shapeLedgerEntry.resolvedShape,
      );
      const inferredOutputShapes = inferNodeOutputShapes({
        graphNode,
        resolvedInputEntries,
        resolvedInputShapes,
        initializerTensorsByName,
        convSpecsByLayerIndex,
      });

      if (inferredOutputShapes.length !== graphNode.output.length) {
        throw buildShapeValidationError(
          graphNode,
          `expected ${graphNode.output.length} outputs but inferred ${inferredOutputShapes.length}`,
        );
      }

      graphNode.output.forEach((outputName, outputIndex) => {
        tensorShapesByName.set(
          outputName,
          buildOutputShapeLedgerEntry({
            graphNode,
            inferredOutputShape: inferredOutputShapes[outputIndex],
            resolvedInputEntries,
          }),
        );
      });
      resolvedGraphNodeCount += 1;
    });

    if (deferredGraphNodes.length === 0) {
      return;
    }

    if (resolvedGraphNodeCount === 0) {
      throw buildUnresolvedInputError(
        deferredGraphNodes[0],
        tensorShapesByName,
      );
    }

    pendingGraphNodes = deferredGraphNodes;
  }
}

function createInitializerTensorMap(
  initializers: OnnxTensor[],
): Map<string, OnnxTensor> {
  return new Map(
    initializers.map((initializerTensor) => [
      initializerTensor.name,
      initializerTensor,
    ]),
  );
}

function validateInitializerPayloadSizes(initializers: OnnxTensor[]): void {
  initializers.forEach((initializerTensor) => {
    const expectedElementCount = multiplyNumericDimensions(
      initializerTensor.dims,
    );

    if (
      initializerTensor.int32_data &&
      initializerTensor.int32_data.length > 0 &&
      initializerTensor.int32_data.length !== expectedElementCount
    ) {
      throw new NetworkOnnxShapeValidationError(
        `Initializer ${initializerTensor.name} declares ${formatShape(initializerTensor.dims)} but stores ${initializerTensor.int32_data.length} int32 values.`,
      );
    }

    if (
      initializerTensor.int64_data &&
      initializerTensor.int64_data.length > 0 &&
      initializerTensor.int64_data.length !== expectedElementCount
    ) {
      throw new NetworkOnnxShapeValidationError(
        `Initializer ${initializerTensor.name} declares ${formatShape(initializerTensor.dims)} but stores ${initializerTensor.int64_data.length} int64 values.`,
      );
    }

    if (
      initializerTensor.float_data.length > 0 &&
      initializerTensor.float_data.length !== expectedElementCount
    ) {
      throw new NetworkOnnxShapeValidationError(
        `Initializer ${initializerTensor.name} declares ${formatShape(initializerTensor.dims)} but stores ${initializerTensor.float_data.length} float values.`,
      );
    }
  });
}

function createInitialTensorShapeLedger(
  model: OnnxModel,
): Map<string, TensorShapeLedgerEntry> {
  const tensorShapesByName = new Map<string, TensorShapeLedgerEntry>();

  model.graph.inputs.forEach((valueInfo) => {
    tensorShapesByName.set(valueInfo.name, {
      resolvedShape: valueInfo.type.tensor_type.shape.dim.map(
        (dimension) => dimension.dim_param ?? dimension.dim_value ?? 1,
      ),
    });
  });

  model.graph.initializer.forEach((initializerTensor) => {
    tensorShapesByName.set(initializerTensor.name, {
      resolvedShape: [...initializerTensor.dims],
    });
  });

  return tensorShapesByName;
}

function collectConvSpecsByLayerIndex(
  model: OnnxModel,
): Map<number, Conv2DMapping> {
  const rawSpecValue = model.metadata_props?.find(
    (property) => property.key === 'conv2d_specs',
  )?.value;
  if (!rawSpecValue) {
    return new Map();
  }

  const parsedSpecs = JSON.parse(rawSpecValue) as Conv2DMapping[];
  return new Map(
    parsedSpecs.map((convSpec) => [convSpec.layerIndex, convSpec]),
  );
}

function resolveInputShapes(
  tensorShapesByName: Map<string, TensorShapeLedgerEntry>,
  graphNode: OnnxNode,
): TensorShapeLedgerEntry[] | null {
  const resolvedInputEntries = graphNode.input.map((inputName) =>
    tensorShapesByName.get(inputName),
  );
  if (resolvedInputEntries.some((shapeLedgerEntry) => !shapeLedgerEntry)) {
    return null;
  }

  return resolvedInputEntries as TensorShapeLedgerEntry[];
}

function inferNodeOutputShapes(
  context: NodeShapeInferenceContext,
): TensorShape[] {
  if (SAME_SHAPE_OPERATIONS.has(context.graphNode.op_type)) {
    return [resolveUnaryOutputShape(context)];
  }

  if (BINARY_BROADCAST_OPERATIONS.has(context.graphNode.op_type)) {
    return [resolveBinaryBroadcastOutputShape(context)];
  }

  if (FUSED_RECURRENT_OPERATIONS.has(context.graphNode.op_type)) {
    return [resolveFusedRecurrentOutputShape(context)];
  }

  switch (context.graphNode.op_type) {
    case 'Concat':
      return [resolveConcatOutputShape(context)];
    case 'Conv':
      return [resolveConvOutputShape(context)];
    case 'DynamicQuantizeLinear':
      return resolveDynamicQuantizeLinearOutputShapes(context);
    case 'QLinearConv':
      return [resolveQLinearConvOutputShape(context)];
    case 'Flatten':
      return [resolveFlattenOutputShape(context)];
    case 'Gemm':
      return [resolveGemmOutputShape(context)];
    case 'MatMul':
      return [resolveMatMulOutputShape(context)];
    case 'QLinearMatMul':
      return [resolveQLinearMatMulOutputShape(context)];
    case 'MaxPool':
    case 'AveragePool':
      return [resolvePoolOutputShape(context)];
    case 'Reshape':
      return [resolveReshapeOutputShape(context)];
    case 'Softmax':
      return [resolveSoftmaxOutputShape(context)];
    case 'Transpose':
      return [resolveTransposeOutputShape(context)];
    default:
      throw buildShapeValidationError(
        context.graphNode,
        `uses unsupported operator ${context.graphNode.op_type}`,
      );
  }
}

function resolveUnaryOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const sourceShape = readRequiredInputShape(
    context,
    0,
    'one unary source tensor',
  );
  return [...sourceShape];
}

function resolveBinaryBroadcastOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const leftShape = readRequiredInputShape(context, 0, 'a left tensor');
  const rightShape = readRequiredInputShape(context, 1, 'a right tensor');
  return broadcastShapes(leftShape, rightShape, context.graphNode);
}

function resolveDynamicQuantizeLinearOutputShapes(
  context: NodeShapeInferenceContext,
): TensorShape[] {
  if (context.graphNode.output.length !== 3) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects exactly 3 outputs but received ${context.graphNode.output.length}`,
    );
  }

  const sourceShape = readRequiredInputShape(
    context,
    0,
    'a float source tensor',
  );
  return [[...sourceShape], [], []];
}

function resolveConcatOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  if (context.resolvedInputShapes.length === 0) {
    throw buildShapeValidationError(
      context.graphNode,
      'requires at least one input tensor',
    );
  }

  const firstInputShape = context.resolvedInputShapes[0];
  const axis = normalizeAxis(
    readAttributeInt(context.graphNode.attributes, 'axis', 0),
    firstInputShape.length,
    context.graphNode,
  );
  const resolvedOutputShape = [...firstInputShape];

  context.resolvedInputShapes.slice(1).forEach((inputShape) => {
    if (inputShape.length !== firstInputShape.length) {
      throw buildShapeValidationError(
        context.graphNode,
        `cannot concat ranks ${firstInputShape.length} and ${inputShape.length}`,
      );
    }

    firstInputShape.forEach((referenceDimension, dimensionIndex) => {
      if (dimensionIndex === axis) {
        return;
      }

      assertDimensionsCompatible(
        referenceDimension,
        inputShape[dimensionIndex],
        context.graphNode,
        `Concat input dimensions disagree outside axis ${axis}`,
      );
    });

    resolvedOutputShape[axis] = sumDimensions(
      resolvedOutputShape[axis],
      inputShape[axis],
    );
  });

  return resolvedOutputShape;
}

function resolveGemmOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const leftShape = readRequiredInputShape(context, 0, 'a left matrix');
  const weightShape = readRequiredInputShape(context, 1, 'a weight matrix');
  const biasShape = readOptionalInputShape(context, 2);
  const normalizedLeftShape = normalizeDenseInputShape(leftShape);
  const fallbackLeftShape = readFallbackInputShape(context, 0);

  if (normalizedLeftShape.length < 1) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects a rank-1 or rank-2 left input but received ${formatShape(leftShape)}`,
    );
  }

  if (weightShape.length !== 2) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects a rank-2 weight tensor but received ${formatShape(weightShape)}`,
    );
  }

  const transposeLeft =
    readAttributeInt(context.graphNode.attributes, 'transA', 0) === 1;
  const transposeWeight =
    readAttributeInt(context.graphNode.attributes, 'transB', 0) === 1;

  let effectiveLeftShape = normalizedLeftShape;
  let leftMatrixShape = resolveGemmMatrixShape(
    effectiveLeftShape,
    transposeLeft,
  );
  const weightMatrixShape = resolveGemmMatrixShape(
    weightShape,
    transposeWeight,
  );

  if (
    !dimensionsAreCompatible(
      leftMatrixShape.columnCount,
      weightMatrixShape.rowCount,
    ) &&
    fallbackLeftShape
  ) {
    const normalizedFallbackLeftShape =
      normalizeDenseInputShape(fallbackLeftShape);
    if (
      normalizedFallbackLeftShape.length >= 1 &&
      normalizedFallbackLeftShape.length <= 2
    ) {
      const fallbackMatrixShape = resolveGemmMatrixShape(
        normalizedFallbackLeftShape,
        transposeLeft,
      );
      if (
        dimensionsAreCompatible(
          fallbackMatrixShape.columnCount,
          weightMatrixShape.rowCount,
        )
      ) {
        effectiveLeftShape = normalizedFallbackLeftShape;
        leftMatrixShape = fallbackMatrixShape;
      }
    }
  }

  assertDimensionsCompatible(
    leftMatrixShape.columnCount,
    weightMatrixShape.rowCount,
    context.graphNode,
    'Gemm inner dimensions do not align',
  );

  const outputShape =
    effectiveLeftShape.length === 1
      ? [weightMatrixShape.columnCount]
      : [leftMatrixShape.rowCount, weightMatrixShape.columnCount];

  if (biasShape) {
    validateGemmBiasShape(context.graphNode, outputShape, biasShape);
  }

  return outputShape;
}

function validateGemmBiasShape(
  graphNode: OnnxNode,
  outputShape: TensorShape,
  biasShape: TensorShape,
): void {
  if (biasShape.length === 1) {
    assertDimensionsCompatible(
      biasShape[0],
      outputShape.at(-1)!,
      graphNode,
      'Gemm bias width does not match the inferred output width',
    );
    return;
  }

  if (biasShape.length === outputShape.length) {
    broadcastShapes(outputShape, biasShape, graphNode);
    return;
  }

  throw buildShapeValidationError(
    graphNode,
    `cannot broadcast Gemm bias ${formatShape(biasShape)} to ${formatShape(outputShape)}`,
  );
}

function resolveGemmMatrixShape(
  sourceShape: TensorShape,
  shouldTranspose: boolean,
): { rowCount: ShapeDimension; columnCount: ShapeDimension } {
  if (sourceShape.length === 1) {
    return shouldTranspose
      ? { rowCount: sourceShape[0], columnCount: 1 }
      : { rowCount: 1, columnCount: sourceShape[0] };
  }

  return shouldTranspose
    ? { rowCount: sourceShape[1], columnCount: sourceShape[0] }
    : { rowCount: sourceShape[0], columnCount: sourceShape[1] };
}

function resolveFlattenOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const inputShape = readRequiredInputShape(
    context,
    0,
    'one tensor to flatten',
  );
  if (inputShape.length < 2) {
    return [...inputShape];
  }

  const axis = normalizeAxis(
    readAttributeInt(context.graphNode.attributes, 'axis', 1),
    inputShape.length,
    context.graphNode,
  );

  return [
    foldDimensionsToToken(inputShape.slice(0, axis)),
    foldDimensionsToToken(inputShape.slice(axis)),
  ];
}

function resolveReshapeOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const inputShape = readRequiredInputShape(context, 0, 'a tensor to reshape');
  const shapeTensorName = context.graphNode.input[1];
  const shapeTensor = context.initializerTensorsByName.get(shapeTensorName);

  if (!shapeTensor?.int64_data) {
    throw buildShapeValidationError(
      context.graphNode,
      `requires an int64 shape tensor for ${shapeTensorName}`,
    );
  }

  const targetDimensions = [...shapeTensor.int64_data];
  const resolvedOutputShape = targetDimensions.map(
    (shapeValue, dimensionIndex) => {
      if (shapeValue === 0) {
        return inputShape[dimensionIndex] ?? 0;
      }

      if (shapeValue < -1) {
        throw buildShapeValidationError(
          context.graphNode,
          `uses unsupported reshape dimension ${shapeValue}`,
        );
      }

      return shapeValue;
    },
  );

  const inferDimensionIndex = resolvedOutputShape.findIndex(
    (dimensionValue) => dimensionValue === -1,
  );
  if (
    inferDimensionIndex !== -1 &&
    resolvedOutputShape.findLastIndex(
      (dimensionValue) => dimensionValue === -1,
    ) !== inferDimensionIndex
  ) {
    throw buildShapeValidationError(
      context.graphNode,
      'uses more than one inferred reshape dimension',
    );
  }

  if (inferDimensionIndex !== -1) {
    const knownOutputProduct = multiplyNumericShape(
      resolvedOutputShape.filter((dimensionValue) => dimensionValue !== -1),
    );
    const inputProduct = multiplyNumericShape(inputShape);

    if (
      knownOutputProduct !== null &&
      inputProduct !== null &&
      knownOutputProduct !== 0 &&
      inputProduct % knownOutputProduct === 0
    ) {
      resolvedOutputShape[inferDimensionIndex] =
        inputProduct / knownOutputProduct;
    }
  }

  const normalizedOutputShape = resolvedOutputShape.map((dimensionValue) =>
    dimensionValue === -1 ? foldDimensionsToToken(inputShape) : dimensionValue,
  );
  const inputProduct = multiplyNumericShape(inputShape);
  const outputProduct = multiplyNumericShape(normalizedOutputShape);

  if (
    inputProduct !== null &&
    outputProduct !== null &&
    inputProduct !== outputProduct
  ) {
    throw buildShapeValidationError(
      context.graphNode,
      `cannot reshape ${formatShape(inputShape)} to ${formatShape(normalizedOutputShape)}`,
    );
  }

  return normalizedOutputShape;
}

function resolveTransposeOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const inputShape = readRequiredInputShape(
    context,
    0,
    'a tensor to transpose',
  );
  const permutation =
    readAttributeInts(context.graphNode.attributes, 'perm') ??
    Array.from(
      { length: inputShape.length },
      (_, offset) => inputShape.length - 1 - offset,
    );

  if (permutation.length !== inputShape.length) {
    throw buildShapeValidationError(
      context.graphNode,
      `cannot apply permutation ${formatShape(permutation)} to ${formatShape(inputShape)}`,
    );
  }

  return permutation.map((dimensionIndex) => inputShape[dimensionIndex]);
}

function resolveSoftmaxOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const inputShape = readRequiredInputShape(
    context,
    0,
    'a tensor to normalize',
  );
  normalizeAxis(
    readAttributeInt(context.graphNode.attributes, 'axis', -1),
    inputShape.length,
    context.graphNode,
  );
  return [...inputShape];
}

function resolveMatMulOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const leftShape = readRequiredInputShape(context, 0, 'a left tensor');
  const rightShape = readRequiredInputShape(context, 1, 'a right tensor');

  const leftWasVector = leftShape.length === 1;
  const rightWasVector = rightShape.length === 1;
  const normalizedLeftShape = leftWasVector ? [1, ...leftShape] : leftShape;
  const normalizedRightShape = rightWasVector ? [...rightShape, 1] : rightShape;

  if (normalizedLeftShape.length < 2 || normalizedRightShape.length < 2) {
    throw buildShapeValidationError(
      context.graphNode,
      'requires rank-1 or higher tensors for MatMul',
    );
  }

  const batchOutputShape = broadcastShapes(
    normalizedLeftShape.slice(0, -2),
    normalizedRightShape.slice(0, -2),
    context.graphNode,
  );
  const leftRowCount = normalizedLeftShape.at(-2)!;
  const leftColumnCount = normalizedLeftShape.at(-1)!;
  const rightRowCount = normalizedRightShape.at(-2)!;
  const rightColumnCount = normalizedRightShape.at(-1)!;

  assertDimensionsCompatible(
    leftColumnCount,
    rightRowCount,
    context.graphNode,
    'MatMul inner dimensions do not align',
  );

  const matrixOutputShape = [
    ...batchOutputShape,
    leftRowCount,
    rightColumnCount,
  ];

  if (leftWasVector) {
    matrixOutputShape.splice(matrixOutputShape.length - 2, 1);
  }

  if (rightWasVector) {
    matrixOutputShape.splice(matrixOutputShape.length - 1, 1);
  }

  return matrixOutputShape;
}

function resolveQLinearMatMulOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const leftShape = readRequiredInputShape(
    context,
    0,
    'a left quantized tensor',
  );
  const rightShape = readRequiredInputShape(
    context,
    3,
    'a right quantized tensor',
  );

  const leftWasVector = leftShape.length === 1;
  const rightWasVector = rightShape.length === 1;
  const normalizedLeftShape = leftWasVector ? [1, ...leftShape] : leftShape;
  const normalizedRightShape = rightWasVector ? [...rightShape, 1] : rightShape;

  if (normalizedLeftShape.length < 2 || normalizedRightShape.length < 2) {
    throw buildShapeValidationError(
      context.graphNode,
      'requires rank-1 or higher tensors for QLinearMatMul',
    );
  }

  const batchOutputShape = broadcastShapes(
    normalizedLeftShape.slice(0, -2),
    normalizedRightShape.slice(0, -2),
    context.graphNode,
  );
  const leftColumnCount = normalizedLeftShape.at(-1)!;
  const rightRowCount = normalizedRightShape.at(-2)!;
  const matrixOutputShape = [
    ...batchOutputShape,
    normalizedLeftShape.at(-2)!,
    normalizedRightShape.at(-1)!,
  ];

  assertDimensionsCompatible(
    leftColumnCount,
    rightRowCount,
    context.graphNode,
    'QLinearMatMul inner dimensions do not align',
  );

  if (leftWasVector) {
    matrixOutputShape.splice(matrixOutputShape.length - 2, 1);
  }

  if (rightWasVector) {
    matrixOutputShape.splice(matrixOutputShape.length - 1, 1);
  }

  return matrixOutputShape;
}

function resolveConvOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const rawInputShape = readRequiredInputShape(
    context,
    0,
    'a Conv input tensor',
  );
  const weightShape = readRequiredInputShape(
    context,
    1,
    'a Conv weight tensor',
  );
  const biasShape = readOptionalInputShape(context, 2);
  const logicalInputShape = resolveLogicalConvInputShape(
    context,
    rawInputShape,
  );
  const layerIndex = parseLayerIndexFromNodeName(context.graphNode.name);
  const convSpec = layerIndex
    ? context.convSpecsByLayerIndex.get(layerIndex)
    : undefined;

  if (weightShape.length !== 4) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects rank-4 Conv weights but received ${formatShape(weightShape)}`,
    );
  }

  if (logicalInputShape.length !== 4) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects rank-4 Conv inputs but received ${formatShape(logicalInputShape)}`,
    );
  }

  assertDimensionsCompatible(
    logicalInputShape[1],
    weightShape[1],
    context.graphNode,
    'Conv channel dimensions do not align',
  );

  const outputShape = resolveConvOutputShapeFromMappingOrKernel({
    context,
    logicalInputShape,
    rawInputShape,
    weightShape,
    convSpec,
  });

  if (biasShape) {
    validateConvBiasShape(context.graphNode, weightShape[0], biasShape);
  }

  return outputShape;
}

function resolveQLinearConvOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const rawInputShape = readRequiredInputShape(
    context,
    0,
    'a QLinearConv input tensor',
  );
  const weightShape = readRequiredInputShape(
    context,
    3,
    'a QLinearConv weight tensor',
  );
  const biasShape = readOptionalInputShape(context, 8);
  const logicalInputShape = resolveLogicalConvInputShape(
    context,
    rawInputShape,
  );
  const layerIndex = parseLayerIndexFromNodeName(context.graphNode.name);
  const convSpec = layerIndex
    ? context.convSpecsByLayerIndex.get(layerIndex)
    : undefined;

  if (weightShape.length !== 4) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects rank-4 QLinearConv weights but received ${formatShape(weightShape)}`,
    );
  }

  if (logicalInputShape.length !== 4) {
    throw buildShapeValidationError(
      context.graphNode,
      `expects rank-4 QLinearConv inputs but received ${formatShape(logicalInputShape)}`,
    );
  }

  assertDimensionsCompatible(
    logicalInputShape[1],
    weightShape[1],
    context.graphNode,
    'QLinearConv channel dimensions do not align',
  );

  const outputShape = resolveConvOutputShapeFromMappingOrKernel({
    context,
    logicalInputShape,
    rawInputShape,
    weightShape,
    convSpec,
  });

  if (biasShape) {
    validateConvBiasShape(context.graphNode, weightShape[0], biasShape);
  }

  return outputShape;
}

function resolveConvOutputShapeFromMappingOrKernel(context: {
  context: NodeShapeInferenceContext;
  logicalInputShape: TensorShape;
  rawInputShape: TensorShape;
  weightShape: TensorShape;
  convSpec: Conv2DMapping | undefined;
}): TensorShape {
  const rawFeatureWidth = normalizeDenseInputShape(context.rawInputShape).at(
    -1,
  );
  const declaredInputWidth = context.convSpec
    ? context.convSpec.inChannels *
      context.convSpec.inHeight *
      context.convSpec.inWidth
    : undefined;

  if (
    context.convSpec &&
    rawFeatureWidth !== undefined &&
    declaredInputWidth !== undefined &&
    dimensionsAreCompatible(rawFeatureWidth, declaredInputWidth)
  ) {
    return [
      context.logicalInputShape[0],
      context.convSpec.outChannels,
      context.convSpec.outHeight,
      context.convSpec.outWidth,
    ];
  }

  const kernelShape = readAttributeInts(
    context.context.graphNode.attributes,
    'kernel_shape',
  ) ?? [
    asNumericDimension(
      context.weightShape[2],
      context.context.graphNode,
      'Conv kernel height',
    ),
    asNumericDimension(
      context.weightShape[3],
      context.context.graphNode,
      'Conv kernel width',
    ),
  ];
  const strides = readAttributeInts(
    context.context.graphNode.attributes,
    'strides',
  ) ?? [1, 1];
  const pads = readAttributeInts(
    context.context.graphNode.attributes,
    'pads',
  ) ?? [0, 0, 0, 0];

  return [
    context.logicalInputShape[0],
    context.weightShape[0],
    computeWindowedOutputDimension(
      context.logicalInputShape[2],
      kernelShape[0],
      strides[0],
      pads[0] + pads[2],
      context.context.graphNode,
      'height',
    ),
    computeWindowedOutputDimension(
      context.logicalInputShape[3],
      kernelShape[1],
      strides[1],
      pads[1] + pads[3],
      context.context.graphNode,
      'width',
    ),
  ];
}

function validateConvBiasShape(
  graphNode: OnnxNode,
  outputChannels: ShapeDimension,
  biasShape: TensorShape,
): void {
  if (biasShape.length !== 1) {
    throw buildShapeValidationError(
      graphNode,
      `expects rank-1 Conv bias but received ${formatShape(biasShape)}`,
    );
  }

  assertDimensionsCompatible(
    biasShape[0],
    outputChannels,
    graphNode,
    'Conv bias width does not match the output channel count',
  );
}

function resolveLogicalConvInputShape(
  context: NodeShapeInferenceContext,
  rawInputShape: TensorShape,
): TensorShape {
  const layerIndex = parseLayerIndexFromNodeName(context.graphNode.name);
  const convSpec = layerIndex
    ? context.convSpecsByLayerIndex.get(layerIndex)
    : undefined;

  if (convSpec) {
    const rawFeatureWidth = normalizeDenseInputShape(rawInputShape).at(-1);
    const declaredFeatureWidth =
      convSpec.inChannels * convSpec.inHeight * convSpec.inWidth;
    if (dimensionsAreCompatible(rawFeatureWidth!, declaredFeatureWidth)) {
      const batchDimension =
        rawInputShape.length >= 2
          ? normalizeDenseInputShape(rawInputShape)[0]
          : 1;
      return [
        batchDimension,
        convSpec.inChannels,
        convSpec.inHeight,
        convSpec.inWidth,
      ];
    }
  }

  if (rawInputShape.length >= 4 || !convSpec) {
    return rawInputShape;
  }

  return rawInputShape.length === 1
    ? [1, convSpec.inChannels, convSpec.inHeight, convSpec.inWidth]
    : [
        rawInputShape[0],
        convSpec.inChannels,
        convSpec.inHeight,
        convSpec.inWidth,
      ];
}

function resolvePoolOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const inputShape = readRequiredInputShape(context, 0, 'a pooled tensor');
  if (inputShape.length !== 4) {
    return [...inputShape];
  }

  const kernelShape = readAttributeInts(
    context.graphNode.attributes,
    'kernel_shape',
  ) ?? [1, 1];
  const strides = readAttributeInts(
    context.graphNode.attributes,
    'strides',
  ) ?? [1, 1];
  const pads = readAttributeInts(context.graphNode.attributes, 'pads') ?? [
    0, 0, 0, 0,
  ];

  return [
    inputShape[0],
    inputShape[1],
    computeWindowedOutputDimension(
      inputShape[2],
      kernelShape[0],
      strides[0],
      pads[0] + pads[2],
      context.graphNode,
      'height',
    ),
    computeWindowedOutputDimension(
      inputShape[3],
      kernelShape[1],
      strides[1],
      pads[1] + pads[3],
      context.graphNode,
      'width',
    ),
  ];
}

function resolveFusedRecurrentOutputShape(
  context: NodeShapeInferenceContext,
): TensorShape {
  const sourceShape = readRequiredInputShape(
    context,
    0,
    'a recurrent source tensor',
  );
  const hiddenSize = readAttributeInt(
    context.graphNode.attributes,
    'hidden_size',
  );
  if (hiddenSize === undefined) {
    throw buildShapeValidationError(
      context.graphNode,
      'requires a hidden_size attribute',
    );
  }

  return sourceShape.length === 2 ? [sourceShape[0], hiddenSize] : [hiddenSize];
}

function readRequiredInputShape(
  context: NodeShapeInferenceContext,
  inputIndex: number,
  description: string,
): TensorShape {
  const resolvedShape = context.resolvedInputShapes[inputIndex];
  if (!resolvedShape) {
    throw buildShapeValidationError(
      context.graphNode,
      `requires ${description}`,
    );
  }
  return resolvedShape;
}

function readOptionalInputShape(
  context: NodeShapeInferenceContext,
  inputIndex: number,
): TensorShape | undefined {
  return context.resolvedInputShapes[inputIndex];
}

function readFallbackInputShape(
  context: NodeShapeInferenceContext,
  inputIndex: number,
): TensorShape | undefined {
  return context.resolvedInputEntries[inputIndex]?.fallbackShape;
}

function readAttributeInt(
  attributes: OnnxAttribute[] | undefined,
  attributeName: string,
  fallbackValue: number,
): number;
function readAttributeInt(
  attributes: OnnxAttribute[] | undefined,
  attributeName: string,
): number | undefined;
function readAttributeInt(
  attributes: OnnxAttribute[] | undefined,
  attributeName: string,
  fallbackValue?: number,
): number | undefined {
  const attribute = findAttribute(attributes, attributeName);
  return attribute?.i ?? fallbackValue;
}

function readAttributeInts(
  attributes: OnnxAttribute[] | undefined,
  attributeName: string,
): number[] | undefined {
  const attribute = findAttribute(attributes, attributeName);
  return attribute?.ints;
}

function findAttribute(
  attributes: OnnxAttribute[] | undefined,
  attributeName: string,
): OnnxAttribute | undefined {
  return attributes?.find((attribute) => attribute.name === attributeName);
}

function normalizeAxis(
  axis: number,
  rank: number,
  graphNode: OnnxNode,
): number {
  const normalizedAxis = axis < 0 ? rank + axis : axis;
  if (normalizedAxis < 0 || normalizedAxis >= rank) {
    throw buildShapeValidationError(
      graphNode,
      `uses axis ${axis} for rank ${rank}`,
    );
  }
  return normalizedAxis;
}

function broadcastShapes(
  leftShape: TensorShape,
  rightShape: TensorShape,
  graphNode: OnnxNode,
): TensorShape {
  const maximumRank = Math.max(leftShape.length, rightShape.length);
  const paddedLeftShape = padShapeToRank(leftShape, maximumRank);
  const paddedRightShape = padShapeToRank(rightShape, maximumRank);

  return paddedLeftShape.map((leftDimension, dimensionIndex) =>
    broadcastDimensions(
      leftDimension,
      paddedRightShape[dimensionIndex],
      graphNode,
    ),
  );
}

function padShapeToRank(
  sourceShape: TensorShape,
  targetRank: number,
): TensorShape {
  return [
    ...Array.from({ length: targetRank - sourceShape.length }, () => 1),
    ...sourceShape,
  ];
}

function broadcastDimensions(
  leftDimension: ShapeDimension,
  rightDimension: ShapeDimension,
  graphNode: OnnxNode,
): ShapeDimension {
  if (leftDimension === 1) {
    return rightDimension;
  }

  if (rightDimension === 1) {
    return leftDimension;
  }

  if (dimensionsAreCompatible(leftDimension, rightDimension)) {
    return typeof leftDimension === 'string' ? leftDimension : rightDimension;
  }

  throw buildShapeValidationError(
    graphNode,
    `cannot broadcast ${String(leftDimension)} with ${String(rightDimension)}`,
  );
}

function assertDimensionsCompatible(
  leftDimension: ShapeDimension,
  rightDimension: ShapeDimension,
  graphNode: OnnxNode,
  message: string,
): void {
  if (dimensionsAreCompatible(leftDimension, rightDimension)) {
    return;
  }

  throw buildShapeValidationError(
    graphNode,
    `${message}: ${String(leftDimension)} vs ${String(rightDimension)}`,
  );
}

function dimensionsAreCompatible(
  leftDimension: ShapeDimension,
  rightDimension: ShapeDimension,
): boolean {
  if (typeof leftDimension === 'number' && typeof rightDimension === 'number') {
    return leftDimension === rightDimension;
  }

  if (typeof leftDimension === 'string' && typeof rightDimension === 'string') {
    return leftDimension === rightDimension;
  }

  return true;
}

function sumDimensions(
  leftDimension: ShapeDimension,
  rightDimension: ShapeDimension,
): ShapeDimension {
  if (typeof leftDimension === 'number' && typeof rightDimension === 'number') {
    return leftDimension + rightDimension;
  }

  const symbolicTerms = [leftDimension, rightDimension].map(String);
  return symbolicTerms.join('+');
}

function foldDimensionsToToken(sourceShape: TensorShape): ShapeDimension {
  if (sourceShape.length === 0) {
    return 1;
  }

  const numericProduct = sourceShape
    .filter((dimensionValue) => typeof dimensionValue === 'number')
    .reduce(
      (product, dimensionValue) => product * (dimensionValue as number),
      1,
    );
  const symbolicTerms = sourceShape.filter(
    (dimensionValue) => typeof dimensionValue === 'string',
  ) as string[];

  if (symbolicTerms.length === 0) {
    return numericProduct;
  }

  const numericPrefix = numericProduct === 1 ? [] : [String(numericProduct)];
  return [...numericPrefix, ...symbolicTerms].join('*');
}

function multiplyNumericDimensions(dimensions: number[]): number {
  return dimensions.reduce(
    (product, dimensionValue) => product * dimensionValue,
    1,
  );
}

function multiplyNumericShape(sourceShape: TensorShape): number | null {
  if (
    sourceShape.some((dimensionValue) => typeof dimensionValue === 'string')
  ) {
    return null;
  }

  return (sourceShape as number[]).reduce(
    (product, dimensionValue) => product * dimensionValue,
    1,
  );
}

function computeWindowedOutputDimension(
  inputDimension: ShapeDimension,
  kernelSize: number,
  stride: number,
  totalPadding: number,
  graphNode: OnnxNode,
  label: string,
): number {
  const numericInputDimension = asNumericDimension(
    inputDimension,
    graphNode,
    `numeric spatial ${label}`,
  );
  if (!Number.isFinite(kernelSize) || !Number.isFinite(stride) || stride <= 0) {
    return numericInputDimension;
  }

  const resolvedOutputDimension = Math.floor(
    (numericInputDimension + totalPadding - kernelSize) / stride + 1,
  );

  if (
    !Number.isFinite(resolvedOutputDimension) ||
    resolvedOutputDimension <= 0
  ) {
    return numericInputDimension;
  }

  return resolvedOutputDimension;
}

function normalizeDenseInputShape(sourceShape: TensorShape): TensorShape {
  if (sourceShape.length <= 2) {
    return sourceShape;
  }

  return [sourceShape[0], foldDimensionsToToken(sourceShape.slice(1))];
}

function buildOutputShapeLedgerEntry(context: {
  graphNode: OnnxNode;
  inferredOutputShape: TensorShape;
  resolvedInputEntries: TensorShapeLedgerEntry[];
}): TensorShapeLedgerEntry {
  const primaryInputEntry = context.resolvedInputEntries[0];
  const propagatedFallbackShape =
    primaryInputEntry?.fallbackShape ?? primaryInputEntry?.resolvedShape;

  if (
    SAME_SHAPE_OPERATIONS.has(context.graphNode.op_type) ||
    context.graphNode.op_type === 'Flatten' ||
    context.graphNode.op_type === 'MaxPool' ||
    context.graphNode.op_type === 'AveragePool' ||
    context.graphNode.op_type === 'Reshape' ||
    context.graphNode.op_type === 'Transpose'
  ) {
    return {
      resolvedShape: context.inferredOutputShape,
      fallbackShape: propagatedFallbackShape,
    };
  }

  return { resolvedShape: context.inferredOutputShape };
}

function asNumericDimension(
  dimensionValue: ShapeDimension,
  graphNode: OnnxNode,
  description: string,
): number {
  if (typeof dimensionValue === 'number') {
    return dimensionValue;
  }

  throw buildShapeValidationError(
    graphNode,
    `requires ${description} but received symbolic dimension ${dimensionValue}`,
  );
}

function parseLayerIndexFromNodeName(nodeName: string): number | null {
  const layerMatch = /_l(\d+)(?:_|$)/i.exec(nodeName);
  return layerMatch ? Number(layerMatch[1]) : null;
}

function formatShape(shape: Array<number | string>): string {
  return `[${shape.map(String).join(', ')}]`;
}

function buildShapeValidationError(
  graphNode: OnnxNode,
  message: string,
): NetworkOnnxShapeValidationError {
  const nodeName = graphNode.name || '<unnamed-node>';
  return new NetworkOnnxShapeValidationError(
    `${nodeName} (${graphNode.op_type}) ${message}.`,
  );
}

function buildUnresolvedInputError(
  graphNode: OnnxNode,
  tensorShapesByName: Map<string, TensorShapeLedgerEntry>,
): NetworkOnnxShapeValidationError {
  const missingInputName = graphNode.input.find(
    (inputName) => !tensorShapesByName.has(inputName),
  )!;
  return buildShapeValidationError(
    graphNode,
    `references unknown input tensor ${missingInputName}`,
  );
}
