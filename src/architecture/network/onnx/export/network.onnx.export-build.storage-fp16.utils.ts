import type { OnnxModel } from '../schema/network.onnx.schema.types';
import type { OnnxExportOptions } from './network.onnx.export.types';
import {
  createFloat16StoragePayload,
  ONNX_FLOAT_DATA_TYPE,
} from '../schema/network.onnx.schema.tensor-data.utils';

const STORAGE_FP16_CAST_SUFFIX = '_fp32';
const STORAGE_FP16_FLOAT_CAST_TARGET = 1;
const STORAGE_FP16_ELIGIBLE_INITIALIZER_PATTERNS = [
  /^W\d+$/,
  /^B\d+$/,
  /^W\d+_n\d+$/,
  /^B\d+_n\d+$/,
  /^ConvW\d+$/,
  /^ConvB\d+$/,
];

/**
 * Rewrites eligible weight and bias initializers from FP32 to FP16 storage and prepends Cast-to-FP32 bridge nodes.
 */
export function applyStorageFp16PostProcessing(
  model: OnnxModel,
  sourceOptions: OnnxExportOptions,
  recurrentLayerIndices: number[],
): void {
  if (!shouldApplyStorageFp16(sourceOptions, recurrentLayerIndices)) {
    return;
  }

  const convertedInitializerNames = rewriteEligibleInitializersToFloat16(model);
  prependFloat32CastBridges(model, convertedInitializerNames);
}

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

function isStorageFp16EligibleInitializer(initializerName: string): boolean {
  return STORAGE_FP16_ELIGIBLE_INITIALIZER_PATTERNS.some((initializerPattern) =>
    initializerPattern.test(initializerName),
  );
}

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
