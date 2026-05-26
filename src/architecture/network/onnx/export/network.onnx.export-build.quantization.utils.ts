import type { OnnxModel } from '../schema/network.onnx.schema.types';
import type {
  OnnxExportOptions,
  OnnxQuantizationCalibrationLayerTarget,
  OnnxQuantizationCalibrationRange,
  OnnxResolvedQuantizationOptions,
} from './network.onnx.export.types';

const ONNX_FLOAT_DATA_TYPE = 1;
const ONNX_UINT8_DATA_TYPE = 2;
const ONNX_INT8_DATA_TYPE = 3;
const ONNX_INT32_DATA_TYPE = 6;

type StaticDenseLoweringPlan = {
  layerIndex: number;
  activationNode: OnnxModel['graph']['node'][number];
  hasBiasBridge: boolean;
};

type StaticConvLoweringPlan = {
  layerIndex: number;
  activationNode: OnnxModel['graph']['node'][number];
  convNode: OnnxModel['graph']['node'][number];
};

type DynamicDenseGuidancePlan = {
  layerIndex: number;
};

/**
 * Appends calibrated static INT8 scale and zero-point initializers to the model graph for each supported layer target.
 */
export function applyStaticQuantizationCalibrationPostProcessing(
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

  const supportedLayerTargets =
    resolvedQuantization.calibration.layerTargets.filter((layerTarget) =>
      isSupportedStaticQuantizationLayerTarget(model, layerTarget),
    );
  const parameterInitializers = supportedLayerTargets.flatMap((layerTarget) =>
    createStaticQuantizationParameterInitializers(
      model,
      resolvedQuantization,
      layerTarget,
    ),
  );
  model.graph.initializer.push(...parameterInitializers);
}

/**
 * Rewrites dense GEMM graph nodes to QLinearMatMul sequences and appends quantized weight initializers for static INT8 lowering.
 */
export function applyStaticDenseQuantizationPostProcessing(
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

  const lowerableDenseLayerPlans = resolvedQuantization.calibration.layerTargets
    .filter(
      (layerTarget) =>
        layerTarget.target === 'dense' &&
        isSupportedStaticQuantizationLayerTarget(model, layerTarget),
    )
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
 * Rewrites Conv graph nodes to QLinearConv sequences and appends quantized weight and bias initializers for static INT8 lowering.
 */
export function applyStaticConvQuantizationPostProcessing(
  model: OnnxModel,
  sourceOptions: OnnxExportOptions,
  resolvedQuantization: OnnxResolvedQuantizationOptions,
  recurrentLayerIndices: number[],
): void {
  if (
    !shouldApplyStaticConvQuantizationLowering(
      sourceOptions,
      resolvedQuantization,
      recurrentLayerIndices,
    )
  ) {
    return;
  }

  const lowerableConvLayerPlans = resolvedQuantization.calibration.layerTargets
    .filter((layerTarget) => layerTarget.target === 'conv')
    .flatMap((layerTarget) => {
      const loweringPlan = resolveStaticConvLoweringPlan(
        model,
        layerTarget.layerIndex,
      );
      return loweringPlan ? [loweringPlan] : [];
    });

  model.graph.initializer.push(
    ...lowerableConvLayerPlans.flatMap((loweringPlan) => [
      createQuantizedConvWeightInitializer(
        model,
        resolvedQuantization,
        loweringPlan.layerIndex,
      ),
      createQuantizedConvBiasInitializer(
        model,
        resolvedQuantization,
        loweringPlan.layerIndex,
      ),
    ]),
  );

  const lowerableConvLayerPlansByLayerIndex = new Map(
    lowerableConvLayerPlans.map((loweringPlan) => [
      loweringPlan.layerIndex,
      loweringPlan,
    ]),
  );

  model.graph.node = model.graph.node.flatMap((graphNode) =>
    rewriteConvGraphNodeForStaticQuantization(
      graphNode,
      lowerableConvLayerPlansByLayerIndex,
    ),
  );
}

/**
 * Wraps each dense GEMM node with DynamicQuantizeLinear/DequantizeLinear guidance nodes for dynamic UINT8 quantization.
 */
export function applyDynamicDenseQuantizationGuidancePostProcessing(
  model: OnnxModel,
  sourceOptions: OnnxExportOptions,
  resolvedQuantization: OnnxResolvedQuantizationOptions,
  recurrentLayerIndices: number[],
): void {
  if (
    !shouldApplyDynamicDenseQuantizationGuidance(
      sourceOptions,
      resolvedQuantization,
      recurrentLayerIndices,
    )
  ) {
    return;
  }

  if (resolvedQuantization.representation === 'metadata-only') {
    return;
  }

  const lowerableDenseGuidancePlansByLayerIndex = new Map(
    model.graph.node
      .filter((graphNode) => graphNode.name.startsWith('gemm_l'))
      .map((graphNode) => {
        const layerIndex = Number.parseInt(
          graphNode.name.slice('gemm_l'.length),
          10,
        );

        return [
          layerIndex,
          {
            layerIndex,
          } satisfies DynamicDenseGuidancePlan,
        ] as const;
      }),
  );

  model.graph.node = model.graph.node.flatMap((graphNode) =>
    rewriteDenseGraphNodeForDynamicGuidance(
      graphNode,
      lowerableDenseGuidancePlansByLayerIndex,
    ),
  );
}

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

  if (
    sourceOptions.allowMixedActivations ||
    sourceOptions.allowPartialConnectivity
  ) {
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

function shouldApplyStaticConvQuantizationLowering(
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

  return (sourceOptions.conv2dMappings?.length ?? 0) > 0;
}

function shouldApplyDynamicDenseQuantizationGuidance(
  sourceOptions: OnnxExportOptions,
  resolvedQuantization: OnnxResolvedQuantizationOptions,
  recurrentLayerIndices: number[],
): resolvedQuantization is Extract<
  OnnxResolvedQuantizationOptions,
  { mode: 'dynamic-uint8' }
> {
  if (
    !resolvedQuantization.requested ||
    resolvedQuantization.mode !== 'dynamic-uint8'
  ) {
    return false;
  }

  if (recurrentLayerIndices.length > 0) {
    return false;
  }

  if ((sourceOptions.conv2dMappings?.length ?? 0) > 0) {
    return false;
  }

  if ((sourceOptions.pool2dMappings?.length ?? 0) > 0) {
    return false;
  }

  if (
    (sourceOptions.concatMappings?.length ?? 0) > 0 ||
    (sourceOptions.attentionMappings?.length ?? 0) > 0
  ) {
    return false;
  }

  if (sourceOptions.allowMixedActivations) {
    return false;
  }

  return !sourceOptions.allowPartialConnectivity;
}

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

function isSupportedStaticQuantizationLayerTarget(
  model: OnnxModel,
  layerTarget: OnnxQuantizationCalibrationLayerTarget,
): boolean {
  if (layerTarget.target === 'conv') {
    return true;
  }

  return resolveDenseLayerOutputWidth(model, layerTarget.layerIndex) === 1;
}

function resolveDenseLayerOutputWidth(
  model: OnnxModel,
  layerIndex: number,
): number {
  const biasInitializer = findInitializerByName(model, `B${layerIndex - 1}`)!;
  return biasInitializer.float_data.length;
}

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

function hasNonZeroDenseBias(
  biasInitializer: NonNullable<ReturnType<typeof findInitializerByName>>,
): boolean {
  return biasInitializer.float_data.some((biasValue) => biasValue !== 0);
}

function rewriteDenseGraphNodeForStaticQuantization(
  graphNode: OnnxModel['graph']['node'][number],
  lowerableDenseLayerPlansByLayerIndex: Map<number, StaticDenseLoweringPlan>,
): OnnxModel['graph']['node'] {
  const lowerableLayerIndex = [
    ...lowerableDenseLayerPlansByLayerIndex.keys(),
  ].find(
    (layerIndex) =>
      graphNode.name === `gemm_l${layerIndex}` ||
      graphNode.name === `act_l${layerIndex}`,
  );

  if (lowerableLayerIndex === undefined) {
    return [graphNode];
  }

  const loweringPlan =
    lowerableDenseLayerPlansByLayerIndex.get(lowerableLayerIndex)!;

  if (graphNode.name === `act_l${loweringPlan.layerIndex}`) {
    return [];
  }

  return createStaticDenseQuantizedNodes(graphNode, loweringPlan);
}

function createStaticDenseQuantizedNodes(
  gemmNode: OnnxModel['graph']['node'][number],
  loweringPlan: StaticDenseLoweringPlan,
): OnnxModel['graph']['node'] {
  const quantizedInputName = `QuantDenseInput_l${loweringPlan.layerIndex}`;
  const quantizedAffineOutputName = `QuantDenseAffine_l${loweringPlan.layerIndex}`;
  const dequantizedAffineOutputName =
    resolveStaticDenseDequantizedOutputName(loweringPlan);
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
      ? [createStaticDenseActivationNode(loweringPlan, activationInputName)]
      : []),
  ];
}

function rewriteDenseGraphNodeForDynamicGuidance(
  graphNode: OnnxModel['graph']['node'][number],
  lowerableDenseGuidancePlansByLayerIndex: Map<
    number,
    DynamicDenseGuidancePlan
  >,
): OnnxModel['graph']['node'] {
  if (!graphNode.name.startsWith('gemm_l')) {
    return [graphNode];
  }

  const layerIndex = Number.parseInt(graphNode.name.slice('gemm_l'.length), 10);
  const guidancePlan = lowerableDenseGuidancePlansByLayerIndex.get(layerIndex)!;

  return createDynamicDenseGuidanceNodes(graphNode, guidancePlan);
}

function createDynamicDenseGuidanceNodes(
  gemmNode: OnnxModel['graph']['node'][number],
  guidancePlan: DynamicDenseGuidancePlan,
): OnnxModel['graph']['node'] {
  const quantizedInputName = `DynamicDenseInput_l${guidancePlan.layerIndex}`;
  const quantizedInputScaleName = `DynamicDenseInputScale_l${guidancePlan.layerIndex}`;
  const quantizedInputZeroPointName = `DynamicDenseInputZeroPoint_l${guidancePlan.layerIndex}`;
  const dequantizedInputName = `DynamicDenseInputFloat_l${guidancePlan.layerIndex}`;

  return [
    {
      op_type: 'DynamicQuantizeLinear',
      input: [gemmNode.input[0]],
      output: [
        quantizedInputName,
        quantizedInputScaleName,
        quantizedInputZeroPointName,
      ],
      name: `dynamic_quantize_input_l${guidancePlan.layerIndex}`,
    },
    {
      op_type: 'DequantizeLinear',
      input: [
        quantizedInputName,
        quantizedInputScaleName,
        quantizedInputZeroPointName,
      ],
      output: [dequantizedInputName],
      name: `dynamic_dequantize_input_l${guidancePlan.layerIndex}`,
    },
    {
      ...gemmNode,
      input: [dequantizedInputName, ...gemmNode.input.slice(1)],
    },
  ];
}

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

function shouldEmitStaticDenseActivationNode(
  loweringPlan: StaticDenseLoweringPlan,
): boolean {
  return loweringPlan.activationNode.op_type !== 'Identity';
}

function createStaticDenseBiasBridgeNode(
  loweringPlan: StaticDenseLoweringPlan,
  dequantizedAffineOutputName: string,
): OnnxModel['graph']['node'][number] {
  return {
    op_type: 'Add',
    input: [dequantizedAffineOutputName, `B${loweringPlan.layerIndex - 1}`],
    output: [
      resolveStaticDenseActivationInputName(
        loweringPlan,
        dequantizedAffineOutputName,
      ),
    ],
    name: `bias_add_l${loweringPlan.layerIndex}`,
  };
}

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

function resolveStaticConvLoweringPlan(
  model: OnnxModel,
  layerIndex: number,
): StaticConvLoweringPlan | undefined {
  const convNode = findGraphNodeByName(model, `conv_l${layerIndex}`);
  const activationNode = findGraphNodeByName(model, `act_conv_l${layerIndex}`);

  if (!convNode || !activationNode) {
    return undefined;
  }

  findInitializerByName(model, `ConvW${layerIndex - 1}`)!;
  findInitializerByName(model, `ConvB${layerIndex - 1}`)!;

  return {
    layerIndex,
    activationNode,
    convNode,
  };
}

function rewriteConvGraphNodeForStaticQuantization(
  graphNode: OnnxModel['graph']['node'][number],
  lowerableConvLayerPlansByLayerIndex: Map<number, StaticConvLoweringPlan>,
): OnnxModel['graph']['node'] {
  const lowerableLayerIndex = [
    ...lowerableConvLayerPlansByLayerIndex.keys(),
  ].find(
    (layerIndex) =>
      graphNode.name === `conv_l${layerIndex}` ||
      graphNode.name === `act_conv_l${layerIndex}`,
  );

  if (lowerableLayerIndex === undefined) {
    return [graphNode];
  }

  const loweringPlan =
    lowerableConvLayerPlansByLayerIndex.get(lowerableLayerIndex)!;

  if (graphNode.name === `act_conv_l${loweringPlan.layerIndex}`) {
    return [];
  }

  return createStaticConvQuantizedNodes(loweringPlan);
}

function createStaticConvQuantizedNodes(
  loweringPlan: StaticConvLoweringPlan,
): OnnxModel['graph']['node'] {
  const quantizedInputName = `QuantConvInput_l${loweringPlan.layerIndex}`;
  const quantizedAffineOutputName = `QuantConvAffine_l${loweringPlan.layerIndex}`;
  const dequantizedAffineOutputName =
    resolveStaticConvDequantizedOutputName(loweringPlan);

  return [
    {
      op_type: 'QuantizeLinear',
      input: [
        loweringPlan.convNode.input[0],
        `QuantConvInputScale_l${loweringPlan.layerIndex}`,
        `QuantConvInputZeroPoint_l${loweringPlan.layerIndex}`,
      ],
      output: [quantizedInputName],
      name: `quantize_conv_input_l${loweringPlan.layerIndex}`,
    },
    {
      op_type: 'QLinearConv',
      input: [
        quantizedInputName,
        `QuantConvInputScale_l${loweringPlan.layerIndex}`,
        `QuantConvInputZeroPoint_l${loweringPlan.layerIndex}`,
        `QuantConvWeight_l${loweringPlan.layerIndex}`,
        `QuantConvWeightScale_l${loweringPlan.layerIndex}`,
        `QuantConvWeightZeroPoint_l${loweringPlan.layerIndex}`,
        `QuantConvOutputScale_l${loweringPlan.layerIndex}`,
        `QuantConvOutputZeroPoint_l${loweringPlan.layerIndex}`,
        `QuantConvBias_l${loweringPlan.layerIndex}`,
      ],
      output: [quantizedAffineOutputName],
      name: `qlinear_conv_l${loweringPlan.layerIndex}`,
      attributes: loweringPlan.convNode.attributes,
    },
    {
      op_type: 'DequantizeLinear',
      input: [
        quantizedAffineOutputName,
        `QuantConvOutputScale_l${loweringPlan.layerIndex}`,
        `QuantConvOutputZeroPoint_l${loweringPlan.layerIndex}`,
      ],
      output: [dequantizedAffineOutputName],
      name: `dequantize_conv_output_l${loweringPlan.layerIndex}`,
    },
    ...(shouldEmitStaticConvActivationNode(loweringPlan)
      ? [
          createStaticConvActivationNode(
            loweringPlan,
            dequantizedAffineOutputName,
          ),
        ]
      : []),
  ];
}

function resolveStaticConvDequantizedOutputName(
  loweringPlan: StaticConvLoweringPlan,
): string {
  return shouldEmitStaticConvActivationNode(loweringPlan)
    ? `ConvAffineFloat_l${loweringPlan.layerIndex}`
    : `Layer_${loweringPlan.layerIndex}`;
}

function shouldEmitStaticConvActivationNode(
  loweringPlan: StaticConvLoweringPlan,
): boolean {
  return loweringPlan.activationNode.op_type !== 'Identity';
}

function createStaticConvActivationNode(
  loweringPlan: StaticConvLoweringPlan,
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

function createQuantizedConvWeightInitializer(
  model: OnnxModel,
  resolvedQuantization: Extract<
    OnnxResolvedQuantizationOptions,
    { mode: 'static-8bit' }
  >,
  layerIndex: number,
) {
  const weightInitializer = findInitializerByName(
    model,
    `ConvW${layerIndex - 1}`,
  )!;
  const weightScaleValues = readRequiredFloatInitializerValues(
    model,
    `QuantConvWeightScale_l${layerIndex}`,
  );
  const weightZeroPointValues = readRequiredIntegerInitializerValues(
    model,
    `QuantConvWeightZeroPoint_l${layerIndex}`,
  );

  return {
    name: `QuantConvWeight_l${layerIndex}`,
    data_type:
      resolvedQuantization.weightEncoding === 'uint8'
        ? ONNX_UINT8_DATA_TYPE
        : ONNX_INT8_DATA_TYPE,
    dims: [...weightInitializer.dims],
    float_data: [],
    int32_data: quantizePerChannelTensorValues(
      weightInitializer.float_data,
      weightInitializer.dims[0]!,
      weightScaleValues,
      weightZeroPointValues,
      resolvedQuantization.weightEncoding,
      resolvedQuantization.calibration.roundingMode,
    ),
  };
}

function createQuantizedConvBiasInitializer(
  model: OnnxModel,
  resolvedQuantization: Extract<
    OnnxResolvedQuantizationOptions,
    { mode: 'static-8bit' }
  >,
  layerIndex: number,
) {
  const biasInitializer = findInitializerByName(
    model,
    `ConvB${layerIndex - 1}`,
  )!;
  const inputScaleValue = readRequiredScalarFloatInitializer(
    model,
    `QuantConvInputScale_l${layerIndex}`,
  );
  const weightScaleValues = readRequiredFloatInitializerValues(
    model,
    `QuantConvWeightScale_l${layerIndex}`,
  );
  const resolvedWeightScaleValues =
    weightScaleValues.length === 1
      ? Array.from(
          { length: biasInitializer.float_data.length },
          () => weightScaleValues[0]!,
        )
      : weightScaleValues;

  return {
    name: `QuantConvBias_l${layerIndex}`,
    data_type: ONNX_INT32_DATA_TYPE,
    dims: [biasInitializer.float_data.length],
    float_data: [],
    int32_data: biasInitializer.float_data.map(
      (biasValue, outputChannelIndex) =>
        clampInteger(
          roundQuantizedValue(
            biasValue /
              (inputScaleValue *
                resolvedWeightScaleValues[outputChannelIndex]!),
            resolvedQuantization.calibration.roundingMode,
          ),
          -2147483648,
          2147483647,
        ),
    ),
  };
}

function transposeDenseWeightValues(
  weightInitializer: NonNullable<ReturnType<typeof findInitializerByName>>,
): number[] {
  const outputCount = weightInitializer.dims[0]!;
  const inputCount = weightInitializer.dims[1]!;

  return Array.from(
    { length: inputCount * outputCount },
    (_unused, valueIndex) => {
      const inputIndex = Math.floor(valueIndex / outputCount);
      const outputIndex = valueIndex % outputCount;
      return weightInitializer.float_data[
        outputIndex * inputCount + inputIndex
      ]!;
    },
  );
}

function findGraphNodeByName(
  model: OnnxModel,
  nodeName: string,
): OnnxModel['graph']['node'][number] | undefined {
  return model.graph.node.find((graphNode) => graphNode.name === nodeName);
}

function findInitializerByName(model: OnnxModel, initializerName: string) {
  return model.graph.initializer.find(
    (initializerEntry) => initializerEntry.name === initializerName,
  );
}

function readRequiredScalarFloatInitializer(
  model: OnnxModel,
  initializerName: string,
): number {
  const initializerEntry = findInitializerByName(model, initializerName)!;
  return initializerEntry.float_data[0]!;
}

function readRequiredFloatInitializerValues(
  model: OnnxModel,
  initializerName: string,
): number[] {
  const initializerEntry = findInitializerByName(model, initializerName)!;
  return initializerEntry.float_data;
}

function readRequiredScalarIntegerInitializer(
  model: OnnxModel,
  initializerName: string,
): number {
  const initializerEntry = findInitializerByName(model, initializerName)!;
  return initializerEntry.int32_data![0]!;
}

function readRequiredIntegerInitializerValues(
  model: OnnxModel,
  initializerName: string,
): number[] {
  const initializerEntry = findInitializerByName(model, initializerName)!;
  return initializerEntry.int32_data!;
}

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
      roundQuantizedValue(
        floatValue / scaleValue + zeroPointValue,
        roundingMode,
      ),
      quantizedMinimum,
      quantizedMaximum,
    ),
  );
}

function quantizePerChannelTensorValues(
  floatValues: number[],
  outputChannelCount: number,
  scaleValues: number[],
  zeroPointValues: number[],
  encoding: 'uint8' | 'int8',
  roundingMode: 'nearest-even',
): number[] {
  const resolvedScaleValues =
    scaleValues.length === 1
      ? Array.from({ length: outputChannelCount }, () => scaleValues[0]!)
      : scaleValues;
  const resolvedZeroPointValues =
    zeroPointValues.length === 1
      ? Array.from({ length: outputChannelCount }, () => zeroPointValues[0]!)
      : zeroPointValues;
  const elementsPerOutputChannel = floatValues.length / outputChannelCount;

  return floatValues.map((floatValue, valueIndex) => {
    const outputChannelIndex = Math.floor(
      valueIndex / elementsPerOutputChannel,
    );
    return quantizeTensorValues(
      [floatValue],
      resolvedScaleValues[outputChannelIndex]!,
      resolvedZeroPointValues[outputChannelIndex]!,
      encoding,
      roundingMode,
    )[0]!;
  });
}

function createStaticQuantizationParameterInitializers(
  model: OnnxModel,
  resolvedQuantization: Extract<
    OnnxResolvedQuantizationOptions,
    { mode: 'static-8bit' }
  >,
  layerTarget: OnnxQuantizationCalibrationLayerTarget,
) {
  const weightParameters = resolveWeightQuantizationParameters(
    model,
    resolvedQuantization,
    layerTarget,
  );

  if (!weightParameters) {
    return [];
  }

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

function resolveWeightQuantizationParameters(
  model: OnnxModel,
  resolvedQuantization: Extract<
    OnnxResolvedQuantizationOptions,
    { mode: 'static-8bit' }
  >,
  layerTarget: OnnxQuantizationCalibrationLayerTarget,
): { scales: number[]; zeroPoints: number[] } | undefined {
  const weightRanges = collectWeightRangesForLayer(
    model,
    layerTarget,
    resolvedQuantization.weightGranularity,
  );

  if (!weightRanges) {
    return undefined;
  }

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
    scales: weightParameterEntries.map(
      (parameterEntry) => parameterEntry.scales[0],
    ),
    zeroPoints: weightParameterEntries.map(
      (parameterEntry) => parameterEntry.zeroPoints[0],
    ),
  };
}

function collectWeightRangesForLayer(
  model: OnnxModel,
  layerTarget: OnnxQuantizationCalibrationLayerTarget,
  weightGranularity: 'per-tensor' | 'per-output-channel',
): OnnxQuantizationCalibrationRange[] | undefined {
  const weightInitializerName =
    layerTarget.target === 'conv'
      ? `ConvW${layerTarget.layerIndex - 1}`
      : `W${layerTarget.layerIndex - 1}`;
  const weightInitializer = model.graph.initializer.find(
    (initializerTensor) => initializerTensor.name === weightInitializerName,
  );

  if (!weightInitializer) {
    return undefined;
  }

  if (weightGranularity === 'per-tensor') {
    return [createMinMaxRange(weightInitializer.float_data)];
  }

  const outputChannelCount = weightInitializer.dims[0]!;
  const elementsPerOutputChannel =
    weightInitializer.float_data.length / outputChannelCount;

  return Array.from({ length: outputChannelCount }, (_, outputChannelIndex) => {
    const startOffset = outputChannelIndex * elementsPerOutputChannel;
    const endOffset = startOffset + elementsPerOutputChannel;
    return createMinMaxRange(
      weightInitializer.float_data.slice(startOffset, endOffset),
    );
  });
}

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

function createMinMaxRange(values: number[]): OnnxQuantizationCalibrationRange {
  const minimumValue = values.reduce(
    (currentMinimum, currentValue) => Math.min(currentMinimum, currentValue),
    Number.POSITIVE_INFINITY,
  );
  const maximumValue = values.reduce(
    (currentMaximum, currentValue) => Math.max(currentMaximum, currentValue),
    Number.NEGATIVE_INFINITY,
  );

  return {
    min: minimumValue,
    max: maximumValue,
  };
}

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

function createScaleInitializer(name: string, scaleValues: number[]) {
  return {
    name,
    data_type: ONNX_FLOAT_DATA_TYPE,
    dims: scaleValues.length === 1 ? [] : [scaleValues.length],
    float_data: scaleValues,
  };
}

function createZeroPointInitializer(
  name: string,
  encoding: 'uint8' | 'int8',
  zeroPointValues: number[],
) {
  return {
    name,
    data_type:
      encoding === 'uint8' ? ONNX_UINT8_DATA_TYPE : ONNX_INT8_DATA_TYPE,
    dims: zeroPointValues.length === 1 ? [] : [zeroPointValues.length],
    float_data: [],
    int32_data: zeroPointValues,
  };
}

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

function clampInteger(
  value: number,
  minimumValue: number,
  maximumValue: number,
): number {
  return Math.min(Math.max(value, minimumValue), maximumValue);
}
