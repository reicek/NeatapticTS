import type NeatapticNode from '../../node';
import type {
  Conv2DMapping,
  DenseWeightBuildContext,
  DenseWeightBuildResult,
  DenseWeightRow,
  DenseWeightRowCollectionContext,
  DiagonalRecurrentBuildContext,
  FlattenAfterPoolingContext,
  IndexedMetadataAppendContext,
  NodeInternals,
  OnnxExportOptions,
  OnnxMetadataProperty,
  OnnxModel,
  OptionalPoolingAndFlattenParams,
  PoolingAttributes,
  PoolingEmissionContext,
  Pool2DMapping,
  RecurrentRowCollectionContext,
  SpecMetadataAppendContext,
} from './network.onnx.utils.types';

/**
 * Build dense-layer weight matrix and bias vector.
 *
 * @param previousLayerNodes Source layer nodes.
 * @param currentLayerNodes Destination layer nodes.
 * @returns Flattened row-major weight matrix and bias vector.
 */
export function buildDenseWeightsAndBiases(
  previousLayerNodes: NeatapticNode[],
  currentLayerNodes: NeatapticNode[],
): DenseWeightBuildResult {
  // Step 1: Build a typed collection context.
  const buildContext: DenseWeightBuildContext = {
    previousLayerNodes,
    currentLayerNodes,
  };

  // Step 2: Collect per-target dense rows.
  const denseRows = collectDenseRows(buildContext);

  // Step 3: Fold rows to flattened initializers.
  return foldDenseRowsToInitializers(denseRows);
}

/**
 * Build a diagonal recurrent matrix from self-connections.
 *
 * @param currentLayerNodes Layer nodes.
 * @returns Flattened row-major recurrent matrix.
 */
export function buildDiagonalRecurrentWeights(
  currentLayerNodes: NeatapticNode[],
): number[] {
  // Step 1: Build a typed collection context.
  const buildContext: DiagonalRecurrentBuildContext = { currentLayerNodes };

  // Step 2: Collect recurrent rows.
  const recurrentRows = collectRecurrentRows(buildContext);

  // Step 3: Fold to a flattened row-major matrix.
  return recurrentRows.flat();
}

/**
 * Emit optional pooling and flatten nodes after a layer output.
 *
 * @param params Pooling parameters.
 * @returns Final output tensor name after optional pooling/flatten.
 */
export function emitOptionalPoolingAndFlatten(
  params: OptionalPoolingAndFlattenParams,
): string {
  // Step 1: Exit early when pooling is not configured.
  if (!params.poolSpec) return params.sourceOutputName;

  // Step 2: Resolve a minimal pooling emission context.
  const poolingContext = toPoolingEmissionContext(params);

  // Step 3: Emit pooling and optional flatten nodes.
  const poolingOutputName = emitPoolingNode(poolingContext);
  const outputAfterFlatten = emitOptionalFlattenAfterPooling({
    model: poolingContext.model,
    flattenAfterPooling: poolingContext.options.flattenAfterPooling,
    layerIndex: poolingContext.layerIndex,
    sourceOutputName: poolingOutputName,
  });

  // Step 4: Append pooling metadata and fold the output name.
  appendPoolingMetadata(poolingContext);
  return outputAfterFlatten;
}

/**
 * Append an integer index to JSON-array metadata key.
 *
 * @param model Target model.
 * @param key Metadata key.
 * @param layerIndex Layer index to append.
 * @returns Nothing.
 */
export function appendIndexedMetadata(
  model: OnnxModel,
  key: string,
  layerIndex: number,
): void {
  // Step 1: Resolve append context and metadata registry.
  const appendContext: IndexedMetadataAppendContext = {
    model,
    key,
    layerIndex,
  };
  const metadataRegistry = ensureMetadataRegistry(model);
  const existingProperty = findMetadataProperty(metadataRegistry, key);

  // Step 2: Append uniquely to existing property or create a new one.
  if (!existingProperty) {
    metadataRegistry.push(buildIndexedMetadataProperty(key, layerIndex));
    return;
  }

  existingProperty.value = serializeIndexedMetadataValue(
    existingProperty.value,
    layerIndex,
  );
}

/**
 * Append a JSON object to JSON-array metadata key.
 *
 * @param model Target model.
 * @param key Metadata key.
 * @param spec Metadata object.
 * @returns Nothing.
 */
export function appendMetadataSpec(
  model: OnnxModel,
  key: string,
  spec: Conv2DMapping | Pool2DMapping,
): void {
  // Step 1: Resolve append context and metadata registry.
  const appendContext: SpecMetadataAppendContext = { model, key, spec };
  const metadataRegistry = ensureMetadataRegistry(model);
  const existingProperty = findMetadataProperty(metadataRegistry, key);

  // Step 2: Append to existing property or create a new one.
  if (!existingProperty) {
    metadataRegistry.push(buildSpecMetadataProperty(key, spec));
    return;
  }

  existingProperty.value = serializeSpecMetadataValue(
    existingProperty.value,
    spec,
  );
}

/**
 * Collect dense rows for each target node in current layer.
 *
 * @param context Dense row collection context.
 * @returns Dense rows containing per-target weights and bias.
 */
function collectDenseRows(context: DenseWeightBuildContext): DenseWeightRow[] {
  return context.currentLayerNodes.map((targetNode) => {
    const targetNodeInternal = asNodeInternals(targetNode);
    const rowCollectionContext: DenseWeightRowCollectionContext = {
      previousLayerNodes: context.previousLayerNodes,
      targetNodeInternal,
    };

    return {
      bias: targetNodeInternal.bias,
      weights: collectDenseRowWeights(rowCollectionContext),
    };
  });
}

/**
 * Fold dense rows into flattened ONNX initializer arrays.
 *
 * @param denseRows Dense rows.
 * @returns Flattened dense initializer result.
 */
function foldDenseRowsToInitializers(
  denseRows: DenseWeightRow[],
): DenseWeightBuildResult {
  const weightMatrixValues = denseRows.flatMap((denseRow) => denseRow.weights);
  const biasVector = denseRows.map((denseRow) => denseRow.bias);
  return { weightMatrixValues, biasVector };
}

/**
 * Collect source-to-target weights for one dense row.
 *
 * @param context Dense row collection context.
 * @returns Row weights in source-node order.
 */
function collectDenseRowWeights(
  context: DenseWeightRowCollectionContext,
): number[] {
  return context.previousLayerNodes.map((sourceNode) =>
    resolveInboundWeight(context.targetNodeInternal, sourceNode),
  );
}

/**
 * Resolve source-to-target inbound connection weight.
 *
 * @param targetNodeInternal Target node internals.
 * @param sourceNode Source node.
 * @returns Inbound weight or zero for disconnected edges.
 */
function resolveInboundWeight(
  targetNodeInternal: NodeInternals,
  sourceNode: NeatapticNode,
): number {
  const inboundConnection = targetNodeInternal.connections.in.find(
    (connection) => connection.from === sourceNode,
  );
  return inboundConnection?.weight ?? 0;
}

/**
 * Normalize a public node instance into ONNX export internals.
 *
 * @param node Source node.
 * @returns Internal runtime-facing node representation.
 */
function asNodeInternals(node: NeatapticNode): NodeInternals {
  return node as NodeInternals;
}

/**
 * Collect recurrent matrix rows for one layer.
 *
 * @param context Recurrent matrix build context.
 * @returns Recurrent row collection.
 */
function collectRecurrentRows(
  context: DiagonalRecurrentBuildContext,
): number[][] {
  return context.currentLayerNodes.map((_node, rowIndex) => {
    const rowCollectionContext: RecurrentRowCollectionContext = {
      currentLayerNodes: context.currentLayerNodes,
      rowIndex,
    };

    return collectRecurrentRow(rowCollectionContext);
  });
}

/**
 * Collect one recurrent matrix row.
 *
 * @param context Row collection context.
 * @returns Recurrent row values.
 */
function collectRecurrentRow(context: RecurrentRowCollectionContext): number[] {
  return context.currentLayerNodes.map((_node, columnIndex) =>
    resolveDiagonalRecurrentWeight(context, columnIndex),
  );
}

/**
 * Resolve recurrent weight value for one matrix coordinate.
 *
 * @param context Row collection context.
 * @param columnIndex Column index in row.
 * @returns Recurrent weight for diagonal entries, otherwise zero.
 */
function resolveDiagonalRecurrentWeight(
  context: RecurrentRowCollectionContext,
  columnIndex: number,
): number {
  if (context.rowIndex !== columnIndex) return 0;

  const selfConnection =
    context.currentLayerNodes[context.rowIndex].connections.self[0];
  return selfConnection?.weight ?? 0;
}

/**
 * Resolve pooling emission context from optional pooling parameters.
 *
 * @param params Optional pooling and flatten parameters.
 * @returns Pooling emission context.
 */
function toPoolingEmissionContext(
  params: OptionalPoolingAndFlattenParams,
): PoolingEmissionContext {
  return {
    model: params.model,
    options: params.options,
    layerIndex: params.layerIndex,
    sourceOutputName: params.sourceOutputName,
    poolSpec: params.poolSpec as Pool2DMapping,
  };
}

/**
 * Emit one pooling node and return its output tensor name.
 *
 * @param context Pooling emission context.
 * @returns Pooling output tensor name.
 */
function emitPoolingNode(context: PoolingEmissionContext): string {
  const poolingOutputName = `Pool_${context.layerIndex}`;
  const poolingAttributes = collectPoolingAttributes(context.poolSpec);

  context.model.graph.node.push({
    op_type: context.poolSpec.type,
    input: [context.sourceOutputName],
    output: [poolingOutputName],
    name: `pool_after_l${context.layerIndex}`,
    attributes: [
      {
        name: 'kernel_shape',
        type: 'INTS',
        ints: poolingAttributes.kernelShape,
      },
      { name: 'strides', type: 'INTS', ints: poolingAttributes.strides },
      { name: 'pads', type: 'INTS', ints: poolingAttributes.pads },
    ],
  });

  return poolingOutputName;
}

/**
 * Collect ONNX pooling attributes from one pooling spec.
 *
 * @param poolSpec Pooling spec.
 * @returns Pooling attributes for ONNX node payload.
 */
function collectPoolingAttributes(poolSpec: Pool2DMapping): PoolingAttributes {
  return {
    kernelShape: [poolSpec.kernelHeight, poolSpec.kernelWidth],
    strides: [poolSpec.strideHeight, poolSpec.strideWidth],
    pads: [
      poolSpec.padTop ?? 0,
      poolSpec.padLeft ?? 0,
      poolSpec.padBottom ?? 0,
      poolSpec.padRight ?? 0,
    ],
  };
}

/**
 * Conditionally emit flatten node after pooling.
 *
 * @param context Flatten emission context.
 * @returns Output tensor name after optional flatten.
 */
function emitOptionalFlattenAfterPooling(
  context: FlattenAfterPoolingContext,
): string {
  if (!context.flattenAfterPooling) return context.sourceOutputName;

  const flattenOutputName = `PoolFlat_${context.layerIndex}`;
  context.model.graph.node.push({
    op_type: 'Flatten',
    input: [context.sourceOutputName],
    output: [flattenOutputName],
    name: `flatten_after_l${context.layerIndex}`,
    attributes: [{ name: 'axis', type: 'INT', i: 1 }],
  });

  appendIndexedMetadata(context.model, 'flatten_layers', context.layerIndex);
  return flattenOutputName;
}

/**
 * Append pooling metadata for one emitted pooling layer.
 *
 * @param context Pooling emission context.
 * @returns Nothing.
 */
function appendPoolingMetadata(context: PoolingEmissionContext): void {
  appendIndexedMetadata(context.model, 'pool2d_layers', context.layerIndex);
  appendMetadataSpec(context.model, 'pool2d_specs', context.poolSpec);
}

/**
 * Ensure model metadata registry exists.
 *
 * @param model Target model.
 * @returns Mutable metadata registry.
 */
function ensureMetadataRegistry(model: OnnxModel): OnnxMetadataProperty[] {
  const metadataRegistry = model.metadata_props ?? [];
  model.metadata_props = metadataRegistry;
  return metadataRegistry;
}

/**
 * Find a metadata property by key.
 *
 * @param metadataRegistry Metadata registry.
 * @param key Metadata key.
 * @returns Matching metadata property if present.
 */
function findMetadataProperty(
  metadataRegistry: OnnxMetadataProperty[],
  key: string,
): OnnxMetadataProperty | undefined {
  return metadataRegistry.find((property) => property.key === key);
}

/**
 * Build a new index-array metadata property.
 *
 * @param key Metadata key.
 * @param layerIndex Layer index.
 * @returns Metadata property.
 */
function buildIndexedMetadataProperty(
  key: string,
  layerIndex: number,
): OnnxMetadataProperty {
  return { key, value: JSON.stringify([layerIndex]) };
}

/**
 * Build a new spec-array metadata property.
 *
 * @param key Metadata key.
 * @param spec Mapping spec.
 * @returns Metadata property.
 */
function buildSpecMetadataProperty(
  key: string,
  spec: Conv2DMapping | Pool2DMapping,
): OnnxMetadataProperty {
  return { key, value: JSON.stringify([{ ...spec }]) };
}

/**
 * Serialize index metadata after appending one unique index.
 *
 * @param currentValue Existing JSON value.
 * @param layerIndex Layer index.
 * @returns Serialized JSON value.
 */
function serializeIndexedMetadataValue(
  currentValue: string,
  layerIndex: number,
): string {
  const parsedIndexes = parseMetadataArray<number>(currentValue);
  if (!parsedIndexes) return JSON.stringify([layerIndex]);

  const uniqueIndexes = parsedIndexes.includes(layerIndex)
    ? parsedIndexes
    : [...parsedIndexes, layerIndex];
  return JSON.stringify(uniqueIndexes);
}

/**
 * Serialize spec metadata after appending one spec object.
 *
 * @param currentValue Existing JSON value.
 * @param spec Mapping spec.
 * @returns Serialized JSON value.
 */
function serializeSpecMetadataValue(
  currentValue: string,
  spec: Conv2DMapping | Pool2DMapping,
): string {
  const parsedSpecs = parseMetadataArray<Conv2DMapping | Pool2DMapping>(
    currentValue,
  );
  if (!parsedSpecs) return JSON.stringify([spec]);

  return JSON.stringify([...parsedSpecs, { ...spec }]);
}

/**
 * Parse a metadata JSON array value safely.
 *
 * @param metadataValue Metadata JSON string.
 * @returns Parsed array when valid, otherwise undefined.
 */
function parseMetadataArray<ItemType>(
  metadataValue: string,
): ItemType[] | undefined {
  try {
    const parsedValue = JSON.parse(metadataValue);
    return Array.isArray(parsedValue) ? (parsedValue as ItemType[]) : undefined;
  } catch {
    return undefined;
  }
}
