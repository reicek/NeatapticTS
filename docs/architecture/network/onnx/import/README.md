# architecture/network/onnx/import

ONNX import orchestration for rebuilding a NeatapticTS runtime network.

This file is the chapter-level tour guide for the import folder. The import
path is intentionally staged so a reader can follow the same questions the
runtime asks while restoring a model:
1. What architecture should be rebuilt?
2. Which runtime factories should own the scaffold?
3. How do dense weights and activations map back onto nodes?
4. Which recurrent and pooling hints need a second pass?

The neighboring files each own one of those stages. Keeping this overview on
the flow file makes the generated folder README read like an import pipeline
instead of an alphabetical pile of helper files.

Example:
```ts
const restored = runOnnxImportFlow(onnxModel);
const output = restored.activate([0.2, 0.8]);
```

## architecture/network/onnx/import/network.onnx.import-flow.utils.ts

### runExternalOnnxImportFlow

```ts
runExternalOnnxImportFlow(
  binaryModel: Uint8Array<ArrayBufferLike>,
): default
```

Execute the external binary ONNX import flow for the first supported dense lane.

High-level behavior:
 1. Decode and normalize one accepted binary `ModelProto` into an importer-owned dense chain.
 2. Fold that chain into the JSON-first model shape used by the existing import flow.
 3. Reuse the same reconstruction pipeline to build the runtime network.

Parameters:
- `binaryModel` - Binary `ModelProto` payload.

Returns: Reconstructed network instance.

### runOnnxImportFlow

```ts
runOnnxImportFlow(
  onnx: OnnxModel,
): default
```

Execute the complete ONNX import flow and reconstruct a runtime network.

High-level behavior:
 1. Extract architecture dimensions and build a perceptron scaffold.
 2. Restore dense parameters and activation functions.
 3. Reconstruct recurrent/pooling metadata and rebuild connection caches.

Parameters:
- `onnx` - ONNX-like model payload to reconstruct.

Returns: Reconstructed network instance.

## architecture/network/onnx/import/network.onnx.runtime-load.types.ts

Import-owned types for ONNX runtime factory loading.

These payloads describe the small runtime bootstrap contract that the import
flow uses to rebuild a perceptron scaffold and attach recurrent layer
constructors without making the parser own a hard-coded constructor shape.

Example:
```ts
const runtimeFactories: OnnxRuntimeFactories = {
  perceptronFactory,
  layerModule,
};
```

### OnnxPerceptronBuildContext

Build context for mapping ONNX layer sizes into a Neataptic MLP factory call.

### OnnxPerceptronSizeValidationContext

Validation context for perceptron size-list checks during ONNX import, supplying sizes, minimum count, and message.

### OnnxRuntimeFactories

Runtime factories consumed during ONNX import network reconstruction, grouping the perceptron and layer module.

### OnnxRuntimeLayerFactory

```ts
OnnxRuntimeLayerFactory(
  size: number,
): default
```

Runtime layer-constructor signature used for recurrent layer reconstruction, accepting size and returning a Layer.

### OnnxRuntimeLayerModule

Runtime layer module shape consumed by ONNX import orchestration, exposing LSTM and GRU factory constructors.

### OnnxRuntimePerceptronFactory

```ts
OnnxRuntimePerceptronFactory(
  sizes: number[],
): default
```

Runtime perceptron factory signature used by ONNX import orchestration, producing a Network from size arguments.

## architecture/network/onnx/import/network.onnx.runtime-load.utils.ts

### buildPerceptronNetwork

```ts
buildPerceptronNetwork(
  buildContext: OnnxPerceptronBuildContext,
): default
```

Build a perceptron network from size-extraction context.

Parameters:
- `buildContext` - Perceptron build context.

Returns: Reconstructed network instance.

### createPerceptronBuildContext

```ts
createPerceptronBuildContext(
  sizes: number[],
): OnnxPerceptronBuildContext
```

Build perceptron-network construction context.

Parameters:
- `sizes` - Layer-size payload.

Returns: Build context.

### createPerceptronFactory

```ts
createPerceptronFactory(): OnnxRuntimePerceptronFactory
```

Create an ONNX import network factory from modern static constructors.

Returns: Perceptron-compatible factory function.

### createPerceptronSizeValidationContext

```ts
createPerceptronSizeValidationContext(
  sizes: number[],
): OnnxPerceptronSizeValidationContext
```

Build perceptron-size validation context.

Parameters:
- `sizes` - Layer-size payload.

Returns: Validation context.

### createRuntimeLayerModule

```ts
createRuntimeLayerModule(): OnnxRuntimeLayerModule
```

Create the runtime layer-module wiring used by ONNX import orchestrators.

Returns: Runtime recurrent-layer module object.

### foldRuntimeFactories

```ts
foldRuntimeFactories(
  perceptronFactory: OnnxRuntimePerceptronFactory,
  layerModule: OnnxRuntimeLayerModule,
): OnnxRuntimeFactories
```

Fold runtime perceptron and layer module into a transport payload.

Parameters:
- `perceptronFactory` - Perceptron factory function.
- `layerModule` - Runtime recurrent-layer constructors.

Returns: Runtime factories payload.

### foldRuntimeLayerModule

```ts
foldRuntimeLayerModule(
  lstmFactory: OnnxRuntimeLayerFactory,
  gruFactory: OnnxRuntimeLayerFactory,
): OnnxRuntimeLayerModule
```

Fold LSTM/GRU factories into a runtime layer module payload.

Parameters:
- `lstmFactory` - Runtime LSTM layer factory.
- `gruFactory` - Runtime GRU layer factory.

Returns: Runtime layer module.

### loadRuntimeFactories

```ts
loadRuntimeFactories(): OnnxRuntimeFactories
```

Resolve runtime constructor factories used by the ONNX import orchestration.

Returns: Perceptron factory and layer module object.

### resolveLayerFactory

```ts
resolveLayerFactory(
  layerKey: keyof OnnxRuntimeLayerModule,
): OnnxRuntimeLayerFactory
```

Resolve one runtime layer factory by module key.

Parameters:
- `layerKey` - Runtime layer key.

Returns: Matching layer factory.

### validatePerceptronSizes

```ts
validatePerceptronSizes(
  validationContext: OnnxPerceptronSizeValidationContext,
): void
```

Validate perceptron size-list constraints.

Parameters:
- `validationContext` - Validation context.

Returns: Nothing. Throws on invalid size-list.

## architecture/network/onnx/import/network.onnx.import-weights.types.ts

Import-owned types for ONNX weight restoration and Conv reconstruction.

This chapter explains the state carried through the importer's heaviest
restoration pass: hidden-size derivation, dense and per-neuron tensor
assignment, and the optional Conv metadata replay that maps flattened ONNX
initializers back onto runtime connections.

Keeping these types near the weight importer makes the generated import README
read like a reconstruction guide instead of scattering the execution model
across the root compatibility barrel.

Example:
```ts
const assignmentContext: OnnxImportWeightAssignmentContext = {
  onnx,
  hiddenLayerSizes,
  metadataProps,
  initializerMap,
  sortedLayerIndices,
  inputNodes,
  hiddenNodes,
  outputNodes,
};
```

### OnnxImportAggregatedLayerAssignmentContext

Context for assigning aggregated dense tensors for one layer, supplying the initializer map and layer node pair.

### OnnxImportAggregatedNeuronAssignmentContext

Context for assigning one aggregated dense target neuron row, carrying previous nodes, target, and tensor refs.

### OnnxImportConvCoordinateAssignmentContext

Context for applying Conv weights and bias at one output coordinate.

### OnnxImportConvKernelAssignmentContext

Context for assigning one concrete Conv kernel connection weight, carrying tensor context, coordinate, and channels.

### OnnxImportConvLayerContext

Context object for reconstructing one Conv layer's imported connectivity weights.

### OnnxImportConvLayerContextBuildParams

Build params for creating one Conv reconstruction layer context, supplying assignment context and Conv metadata.

### OnnxImportConvMetadata

Parsed Conv metadata payload used for optional reconstruction pass, listing Conv layer indices and mapping specs.

### OnnxImportConvNodeSlices

Layer node slices used while applying Conv reconstruction assignments, carrying target and previous layer nodes.

### OnnxImportConvOutputCoordinate

Coordinate for one Conv output neuron traversal position, encoding output channel, row, and column indices.

### OnnxImportConvSourceLayout

Source layout used when replaying Conv weights onto dense source nodes.

### OnnxImportConvTensorContext

Resolved Conv initializer tensors and dimensions for one layer, including channels, kernel height, and width.

### OnnxImportHiddenSizeDerivationContext

Context for deriving hidden layer sizes from initializer tensors and metadata.

### OnnxImportInboundConnectionMap

Inbound connection lookup map keyed by source node for one target neuron.

### OnnxImportLayerNodePair

Node slices for one sequential imported layer assignment pass, carrying current and previous layer node lists.

### OnnxImportLayerNodePairBuildParams

Build params for one sequential layer node-pair slice operation, specifying layer index and sequential position.

### OnnxImportLayerTensorNames

Weight tensor names for one imported layer index, identifying weight and bias initializer name strings.

### OnnxImportLayerWeightBucket

Bucketed ONNX dense/per-neuron tensors for one exported layer index, holding the aggregated and per-neuron lists.

### OnnxImportPerNeuronAssignmentContext

Context for assigning one per-neuron imported target node, carrying previous nodes and weight and bias tensors.

### OnnxImportPerNeuronLayerAssignmentContext

Context for assigning per-neuron tensors for one layer, supplying the initializer map and sequential layer node pair.

### OnnxImportWeightAssignmentBuildParams

Build params for creating shared ONNX import weight-assignment context, supplying network, model, and hidden sizes.

### OnnxImportWeightAssignmentContext

Shared weight-assignment context built once per ONNX import, carrying model, layers, metadata, and initializer map.

## architecture/network/onnx/import/network.onnx.import-weights.utils.ts

Assign weights and biases from ONNX initializers to a newly created network.

### applyAggregatedLayerWeights

```ts
applyAggregatedLayerWeights(
  aggregatedContext: OnnxImportAggregatedLayerAssignmentContext,
): void
```

Apply aggregated dense tensor assignments for one layer.

Parameters:
- `aggregatedContext` - Aggregated assignment context.

Returns: Nothing.

### applyAggregatedNeuronAssignment

```ts
applyAggregatedNeuronAssignment(
  neuronContext: OnnxImportAggregatedNeuronAssignmentContext,
): void
```

Apply aggregated dense row weights and bias for one target neuron.

Parameters:
- `neuronContext` - Aggregated neuron assignment context.

Returns: Nothing.

### applyConvCoordinateAssignment

```ts
applyConvCoordinateAssignment(
  coordinateContext: OnnxImportConvCoordinateAssignmentContext,
): void
```

Apply Conv bias and kernel weights for one output coordinate.

Parameters:
- `coordinateContext` - Conv coordinate assignment context.

Returns: Nothing.

### applyConvLayerReconstruction

```ts
applyConvLayerReconstruction(
  layerContext: OnnxImportConvLayerContext,
): void
```

Apply Conv reconstruction for one validated Conv layer context.

Parameters:
- `layerContext` - Conv layer context.

Returns: Nothing.

### applyDenseWeightAssignments

```ts
applyDenseWeightAssignments(
  assignmentContext: OnnxImportWeightAssignmentContext,
): void
```

Apply dense/per-neuron assignments for all sorted layer indices.

Parameters:
- `assignmentContext` - Shared assignment context.

Returns: Nothing.

### applyOptionalConvReconstruction

```ts
applyOptionalConvReconstruction(
  assignmentContext: OnnxImportWeightAssignmentContext,
): void
```

Apply optional Conv2D reconstruction pass from metadata payloads.

Parameters:
- `assignmentContext` - Shared assignment context.

Returns: Nothing.

### applyPerNeuronAssignment

```ts
applyPerNeuronAssignment(
  perNeuronAssignmentContext: OnnxImportPerNeuronAssignmentContext,
): void
```

Apply one per-neuron weight vector and bias assignment.

Parameters:
- `perNeuronAssignmentContext` - Per-neuron assignment context.

Returns: Nothing.

### applyPerNeuronLayerWeights

```ts
applyPerNeuronLayerWeights(
  perNeuronContext: OnnxImportPerNeuronLayerAssignmentContext,
): void
```

Apply per-neuron tensor assignments for one layer.

Parameters:
- `perNeuronContext` - Per-neuron layer assignment context.

Returns: Nothing.

### assignConvKernelWeight

```ts
assignConvKernelWeight(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): void
```

Assign one Conv kernel weight to the matching inbound neuron connection.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Nothing.

### assignLayerWeights

```ts
assignLayerWeights(
  initializerMap: Record<string, OnnxTensor>,
  nodePair: OnnxImportLayerNodePair,
): void
```

Assign one layer's weights using aggregated or per-neuron tensors.

Parameters:
- `initializerMap` - ONNX initializer map.
- `nodePair` - Current/previous node slices.

Returns: Nothing.

### assignWeightsAndBiases

```ts
assignWeightsAndBiases(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadataProps: OnnxMetadataProperty[] | undefined,
): void
```

Contract for assignWeightsAndBiases.

### buildConvLayerContext

```ts
buildConvLayerContext(
  params: OnnxImportConvLayerContextBuildParams,
): OnnxImportConvLayerContext | null
```

Build one Conv layer reconstruction context.

Parameters:
- `params` - Conv context input params.

Returns: Conv layer context when valid.

### buildConvNeuronLinearIndex

```ts
buildConvNeuronLinearIndex(
  coordinate: OnnxImportConvOutputCoordinate,
  convSpec: Conv2DMapping,
): number
```

Build flattened linear index for one Conv output coordinate.

Parameters:
- `coordinate` - Conv output coordinate.
- `convSpec` - Conv mapping spec.

Returns: Linear neuron index.

### buildConvNodeSlices

```ts
buildConvNodeSlices(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvNodeSlices
```

Build Conv current/previous node slices for one layer context.

Parameters:
- `layerContext` - Conv layer context.

Returns: Node slice payload.

### buildConvTensorContext

```ts
buildConvTensorContext(
  layerContext: OnnxImportConvLayerContext,
): OnnxImportConvTensorContext | null
```

Build validated Conv tensor context for one layer.

Parameters:
- `layerContext` - Conv layer context.

Returns: Conv tensor context when valid.

### buildHiddenLayerSizesFromBuckets

```ts
buildHiddenLayerSizesFromBuckets(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
  sortedLayerIndices: number[],
): number[]
```

Build hidden-layer sizes from weight buckets while excluding output layer.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.
- `sortedLayerIndices` - Ascending layer indices.

Returns: Hidden-layer sizes.

### buildInboundConnectionMap

```ts
buildInboundConnectionMap(
  neuronInternal: NodeInternals,
): OnnxImportInboundConnectionMap
```

Build inbound connection lookup map for one neuron.

Parameters:
- `neuronInternal` - Neuron internals.

Returns: Inbound connection map keyed by source node.

### buildInitializerMap

```ts
buildInitializerMap(
  initializers: OnnxTensor[],
  metadataProps: OnnxMetadataProperty[],
): Record<string, OnnxTensor>
```

Build ONNX initializer map keyed by tensor name.

Parameters:
- `initializers` - ONNX initializer list.

Returns: Tensor map by name.

### buildInputCoordinate

```ts
buildInputCoordinate(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): { inputRow: number; inputColumn: number; } | null
```

Build input-space coordinate for one Conv kernel element.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Input coordinate when in bounds.

### buildInputFeatureLinearIndex

```ts
buildInputFeatureLinearIndex(
  sourceLayout: OnnxImportConvSourceLayout,
  inChannelIndex: number,
  inputRow: number,
  inputColumn: number,
): number
```

Build linear feature index in input feature space.

Parameters:
- `convSpec` - Conv mapping spec.
- `inChannelIndex` - Input channel index.
- `inputRow` - Input row index.
- `inputColumn` - Input column index.

Returns: Linear input feature index.

### buildLayerNodePair

```ts
buildLayerNodePair(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: OnnxImportLayerNodePairBuildParams,
): OnnxImportLayerNodePair
```

Build current/previous node slices for one sequential import layer pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Layer node pair.

### buildLayerTensorNames

```ts
buildLayerTensorNames(
  layerIndex: number,
): OnnxImportLayerTensorNames
```

Build dense weight/bias tensor names for one layer index.

Parameters:
- `layerIndex` - Export layer index.

Returns: Layer tensor names.

### buildPerNeuronTensorNames

```ts
buildPerNeuronTensorNames(
  layerIndex: number,
  neuronIndex: number,
): OnnxImportLayerTensorNames
```

Build per-neuron tensor names for one layer and neuron index.

Parameters:
- `layerIndex` - Export layer index.
- `neuronIndex` - Neuron index in layer.

Returns: Per-neuron tensor names.

### buildWeightAssignmentContext

```ts
buildWeightAssignmentContext(
  params: OnnxImportWeightAssignmentBuildParams,
): OnnxImportWeightAssignmentContext
```

Build the shared assignment context for import weight restoration.

Parameters:
- `params` - Assignment context input params.

Returns: Shared assignment context.

### collectConvKernelCoordinates

```ts
collectConvKernelCoordinates(
  inChannels: number,
  kernelHeight: number,
  kernelWidth: number,
): OnnxConvKernelCoordinate[]
```

Collect all kernel traversal coordinates for one Conv output position.

Parameters:
- `inChannels` - Input channel count.
- `kernelHeight` - Kernel height.
- `kernelWidth` - Kernel width.

Returns: Kernel traversal coordinates.

### collectConvOutputCoordinates

```ts
collectConvOutputCoordinates(
  convSpec: Conv2DMapping,
  outChannels: number,
): OnnxImportConvOutputCoordinate[]
```

Collect all output traversal coordinates for one Conv layer.

Parameters:
- `convSpec` - Conv mapping spec.
- `outChannels` - Output channel count.

Returns: Output traversal coordinates.

### collectLayerWeightBuckets

```ts
collectLayerWeightBuckets(
  initializers: OnnxTensor[],
): Record<string, OnnxImportLayerWeightBucket>
```

Collect ONNX weight tensor buckets grouped by export layer index.

Parameters:
- `initializers` - ONNX initializer tensors.

Returns: Layer-weight buckets keyed by export layer index.

### collectNodesByType

```ts
collectNodesByType(
  nodes: default[],
  nodeType: "hidden" | "input" | "output",
): default[]
```

Collect nodes by runtime node type discriminator.

Parameters:
- `nodes` - Network nodes.
- `nodeType` - Runtime node type.

Returns: Filtered nodes.

### collectSortedLayerIndices

```ts
collectSortedLayerIndices(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
): number[]
```

Collect sorted layer indices from weight buckets.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.

Returns: Ascending export layer indices.

### collectSortedUniqueLayerIndices

```ts
collectSortedUniqueLayerIndices(
  initializerMap: Record<string, OnnxTensor>,
): number[]
```

Collect unique sorted layer indices from initializer weight tensors.

Parameters:
- `initializerMap` - Initializer map keyed by tensor name.

Returns: Unique sorted layer indices.

### deriveHiddenLayerSizes

```ts
deriveHiddenLayerSizes(
  initializers: OnnxTensor[],
  metadataProps: OnnxMetadataProperty[] | undefined,
): number[]
```

Derive hidden-layer sizes from ONNX weight initializers in export order.

Parameters:
- `initializers` - ONNX initializer tensors.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Hidden layer sizes in order.

### hasAggregatedLayerWeights

```ts
hasAggregatedLayerWeights(
  aggregatedContext: OnnxImportAggregatedLayerAssignmentContext,
): boolean
```

Determine whether the layer has aggregated weight tensor data.

Parameters:
- `aggregatedContext` - Aggregated assignment context.

Returns: True when aggregated tensor exists.

### hydrateSharedInitializerAliases

```ts
hydrateSharedInitializerAliases(
  initializerMap: Record<string, OnnxTensor>,
  metadataProps: OnnxMetadataProperty[],
): void
```

Hydrate alias tensor names back into the initializer map for metadata-backed shared initializers.

### parseConvMetadata

```ts
parseConvMetadata(
  metadataProps: OnnxMetadataProperty[],
): OnnxImportConvMetadata | null
```

Parse Conv reconstruction metadata payload.

Parameters:
- `metadataProps` - ONNX metadata properties.

Returns: Parsed Conv metadata.

### parseLayerIndexFromWeightTensor

```ts
parseLayerIndexFromWeightTensor(
  tensorName: string,
): number | null
```

Parse layer index from dense/per-neuron weight tensor name.

Parameters:
- `tensorName` - Tensor name.

Returns: Parsed layer index or null.

### parseMetadataLayerSizes

```ts
parseMetadataLayerSizes(
  metadataProps: OnnxMetadataProperty[],
): number[] | null
```

Parse explicit metadata-driven hidden layer sizes.

Parameters:
- `metadataProps` - ONNX metadata properties.

Returns: Parsed hidden layer sizes when available.

### parseSharedInitializerAliases

```ts
parseSharedInitializerAliases(
  metadataProps: OnnxMetadataProperty[],
): { aliasTensorName: string; canonicalTensorName: string; }[]
```

Parse valid shared-initializer alias metadata records from ONNX metadata.

### parseWeightTensorName

```ts
parseWeightTensorName(
  tensorName: string,
): { layerIndex: string; neuronIndex: number | null; } | null
```

Parse layer/neuron components from a weight tensor name.

Parameters:
- `tensorName` - Tensor name.

Returns: Parsed layer+neuron components when matched.

### readConvKernelWeight

```ts
readConvKernelWeight(
  kernelAssignmentContext: OnnxImportConvKernelAssignmentContext,
): number
```

Read one Conv kernel weight from flattened ONNX tensor payload.

Parameters:
- `kernelAssignmentContext` - Conv kernel assignment context.

Returns: Kernel weight.

### resetInboundConnectionWeights

```ts
resetInboundConnectionWeights(
  neuronInternal: NodeInternals,
): void
```

Reset all inbound weights so Conv reconstruction can write only receptive edges.

Parameters:
- `neuronInternal` - Target neuron internals.

Returns: Nothing.

### resolveCurrentLayerNodes

```ts
resolveCurrentLayerNodes(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: { layerIndex: number; },
): default[]
```

Resolve current layer nodes for one sequential layer assignment pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Current layer nodes.

### resolveLayerHiddenSize

```ts
resolveLayerHiddenSize(
  layerWeightBuckets: Record<string, OnnxImportLayerWeightBucket>,
  layerIndex: number,
): number
```

Resolve one hidden-layer size from its weight bucket.

Parameters:
- `layerWeightBuckets` - Layer-weight buckets.
- `layerIndex` - Export layer index.

Returns: Hidden-layer size.

### resolvePreviousLayerNodes

```ts
resolvePreviousLayerNodes(
  assignmentContext: OnnxImportWeightAssignmentContext,
  params: { layerIndex: number; },
): default[]
```

Resolve previous layer nodes for one sequential layer assignment pass.

Parameters:
- `assignmentContext` - Shared assignment context.
- `params` - Sequential traversal params.

Returns: Previous layer nodes.

### sumHiddenSizesToIndex

```ts
sumHiddenSizesToIndex(
  hiddenLayerSizes: number[],
  exclusiveEndIndex: number,
): number
```

Sum hidden-layer sizes from index `0` to `exclusiveEndIndex`.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.
- `exclusiveEndIndex` - Exclusive end index.

Returns: Prefix sum.

## architecture/network/onnx/import/network.onnx.import-activations.utils.ts

### appendOperationToLayer

```ts
appendOperationToLayer(
  operationsByLayer: OnnxActivationLayerOperations,
  layerIndex: number,
  operation: OnnxActivationOperation,
): void
```

Append one operation to the lookup bucket for a layer.

Parameters:
- `operationsByLayer` - Layer-indexed operation lookup.
- `layerIndex` - Export-layer index.
- `operation` - Supported activation operation.

Returns: Nothing.

### applyHiddenLayerActivations

```ts
applyHiddenLayerActivations(
  context: OnnxActivationAssignmentContext,
): void
```

Apply imported activation operations to hidden layer nodes.

Parameters:
- `context` - Shared assignment context.

Returns: Nothing.

### applyHiddenLayerTraversalActivation

```ts
applyHiddenLayerTraversalActivation(
  traversalContext: HiddenLayerActivationTraversalContext,
): void
```

Apply activation operations for one hidden-layer traversal context.

Parameters:
- `traversalContext` - Hidden-layer traversal context.

Returns: Nothing.

### applyHiddenLayerTraversalContexts

```ts
applyHiddenLayerTraversalContexts(
  traversalContexts: HiddenLayerActivationTraversalContext[],
): void
```

Apply hidden-layer activation assignment for each traversal context.

Parameters:
- `traversalContexts` - Ordered hidden-layer traversal contexts.

Returns: Nothing.

### applyHiddenNeuronActivation

```ts
applyHiddenNeuronActivation(
  traversalContext: HiddenLayerActivationTraversalContext,
  neuronIndex: number,
): void
```

Apply one hidden neuron activation if the target node exists.

Parameters:
- `traversalContext` - Hidden-layer traversal context.
- `neuronIndex` - Neuron index in current hidden layer.

Returns: Nothing.

### applyOutputLayerActivation

```ts
applyOutputLayerActivation(
  context: OnnxActivationAssignmentContext,
): void
```

Apply imported activation to all output nodes.

Parameters:
- `context` - Shared assignment context.

Returns: Nothing.

### asNodeInternals

```ts
asNodeInternals(
  node: unknown,
): NodeInternals
```

Cast one public node instance to runtime node internals.

Parameters:
- `node` - Source node object.

Returns: Runtime node internals.

### assignActivationFunctions

```ts
assignActivationFunctions(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
): void
```

Assign runtime node activation functions from ONNX activation graph operations.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer size list.

Returns: Nothing.

### asSupportedActivationOperation

```ts
asSupportedActivationOperation(
  operationName: string,
): OnnxActivationOperation | null
```

Convert an ONNX op type to a supported activation operation.

Parameters:
- `operationName` - ONNX graph node operation type.

Returns: Supported activation operation or null when unsupported.

### buildActivationAssignmentContext

```ts
buildActivationAssignmentContext(
  sourceNetwork: default,
  sourceOnnx: OnnxModel,
  sourceHiddenLayerSizes: number[],
): OnnxActivationAssignmentContext
```

Build immutable assignment context for hidden/output activation import.

Parameters:
- `sourceNetwork` - Target network to mutate.
- `sourceOnnx` - Source ONNX model.
- `sourceHiddenLayerSizes` - Hidden layer widths in export order.

Returns: Prepared assignment context.

### buildHiddenLayerTraversalContexts

```ts
buildHiddenLayerTraversalContexts(
  context: OnnxActivationAssignmentContext,
): HiddenLayerActivationTraversalContext[]
```

Build traversal contexts for each hidden layer.

Parameters:
- `context` - Shared assignment context.

Returns: Ordered hidden-layer traversal contexts.

### buildHiddenNeuronIndices

```ts
buildHiddenNeuronIndices(
  hiddenLayerSize: number,
): number[]
```

Build contiguous hidden-neuron index list for one layer.

Parameters:
- `hiddenLayerSize` - Hidden-layer width.

Returns: Ordered neuron indices.

### collectNodeInternalsByType

```ts
collectNodeInternalsByType(
  sourceNetwork: default,
  targetType: string,
): NodeInternals[]
```

Collect node internals by runtime node type.

Parameters:
- `sourceNetwork` - Network with runtime nodes.
- `targetType` - Runtime node type to collect.

Returns: Runtime node internals matching the requested type.

### collectOperationsByLayer

```ts
collectOperationsByLayer(
  sourceOnnx: OnnxModel,
): OnnxActivationLayerOperations
```

Collect ONNX activation operations grouped by export-layer index.

Parameters:
- `sourceOnnx` - Source ONNX model.

Returns: Layer-indexed activation operation lookup.

### parseActivationNode

```ts
parseActivationNode(
  nodeName: string,
): OnnxActivationParseResult | null
```

Parse one ONNX activation node name.

Parameters:
- `nodeName` - ONNX graph node name.

Returns: Parsed layer/neuron metadata or null when not a supported activation node.

### resolveActivationFunction

```ts
resolveActivationFunction(
  context: OnnxActivationOperationResolutionContext,
): ((x: number, derivate?: boolean | undefined) => number) & { name?: string | undefined; }
```

Resolve runtime activation function from operation context.

Parameters:
- `context` - Operation-resolution context.

Returns: Runtime activation function.

### resolveHiddenNode

```ts
resolveHiddenNode(
  traversalContext: HiddenLayerActivationTraversalContext,
  neuronIndex: number,
): NodeInternals | undefined
```

Resolve one hidden node by traversal/offset metadata.

Parameters:
- `traversalContext` - Hidden-layer traversal context.
- `neuronIndex` - Neuron index in current hidden layer.

Returns: Hidden node internals when present.

### resolveLayerOperations

```ts
resolveLayerOperations(
  traversalContext: HiddenLayerActivationTraversalContext,
): OnnxActivationOperation[]
```

Resolve operations list for one hidden layer.

Parameters:
- `traversalContext` - Hidden-layer traversal context.

Returns: Ordered operations for target export layer.

### resolveOperationByPriority

```ts
resolveOperationByPriority(
  context: OnnxActivationOperationResolutionContext,
): OnnxActivationOperation
```

Resolve activation operation by neuron-first then layer-default fallback.

Parameters:
- `context` - Operation-resolution context.

Returns: Supported activation operation.

### resolveOutputOperations

```ts
resolveOutputOperations(
  context: OutputLayerActivationContext,
): OnnxActivationOperation[]
```

Resolve output-layer operations from shared lookup.

Parameters:
- `context` - Output-layer assignment context.

Returns: Ordered output-layer operations.

## architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts

Import-owned type surface for ONNX architecture reconstruction orchestration.

These payloads stay close to the orchestration helpers that parse terminal
dimensions, restore recurrent self-connections, and attach pooling metadata.
Keeping them here makes the import chapter explain its own execution state
without forcing the root ONNX compatibility barrel to remain the ownership
home for importer-only details.

### NetworkWithOnnxImportAdvancedGraph

Network instance augmented with optional imported advanced-graph metadata via the _onnxAdvancedGraph field.

### NetworkWithOnnxImportPooling

Network instance augmented with optional imported ONNX pooling metadata via the _onnxPooling field.

### OnnxImportAdvancedGraphCrossLayerConnection

Audit-only cross-layer feed-forward edge carried through Phase 5 import fallback.

### OnnxImportAdvancedGraphMetadata

Parsed advanced-graph metadata attached to imported network instances, grouping merges, residual adds, and blocks.

### OnnxImportArchitectureContext

Shared architecture extraction context with resolved graph dimensions, initializers, and metadata properties.

### OnnxImportArchitectureResult

Parsed architecture dimensions extracted from ONNX import graph payloads, with input, output, and hidden sizes.

### OnnxImportAttentionBlock

Explicit fixed-width self-attention block carried through Phase 5 import fallback.

### OnnxImportConcatMerge

Explicit concat merge carried through Phase 5 import hardening, identifying layer indices and merge tensor names.

### OnnxImportDimensionRecord

Loose ONNX shape-dimension record used by legacy import payload access.

### OnnxImportFlattenConsistencyAudit

Metadata-only audit record comparing a flattened pooled width to the next dense width.

### OnnxImportHiddenLayerSpan

Hidden-layer span payload with one-based layer numbering and global offset.

### OnnxImportLayerConnectionContext

Execution context for assigning one hidden-layer recurrent diagonal tensor, carrying model, nodes, and span.

### OnnxImportPoolingMetadata

Parsed pooling metadata payload attached to imported network instances, listing pool specs and virtual shapes.

### OnnxImportPoolingVirtualShape

Virtual spatial shape derived from Conv and Pool metadata during import.

### OnnxImportRecurrentRestorationContext

Context for recurrent self-connection restoration from ONNX metadata and tensors.

### OnnxImportResidualAdd

Explicit one-hop residual-add merge carried through Phase 5 import hardening.

### OnnxImportSelfConnectionUpsertContext

Context for upserting one hidden node self-connection from recurrent weight.

### OnnxImportSharedInitializerAlias

Audit-only shared initializer alias carried through Phase 5 import fallback.

## architecture/network/onnx/import/network.onnx.import-orchestrators.utils.ts

Restore recurrent self-connections from recurrent metadata and R tensors.
Restoration uses metadata-gated span resolution and diagonal tensor extraction so imported recurrent units recover their self-feedback semantics without guessing hidden-node layout.

### applyLayerSelfConnections

```ts
applyLayerSelfConnections(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): void
```

Apply one hidden layer diagonal recurrent self-weights.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Nothing.

### attachOnnxAdvancedGraphMetadata

```ts
attachOnnxAdvancedGraphMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel | undefined,
): void
```

Attach optional advanced-graph audit metadata from ONNX model metadata.

Phase 5 starts with honest fallback: import keeps rebuilding the layered
baseline, but it can still preserve the exact cross-layer feed-forward edges
the exporter detected so later residual, concat, and attention passes have a
deterministic seam to reuse.

Parameters:
- `network` - Target network.
- `metadata` - ONNX metadata properties to parse for advanced-graph fields.
- `onnx` - Optional source ONNX model for additional context.

Returns: Nothing.

### attachOnnxPoolingMetadata

```ts
attachOnnxPoolingMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
): void
```

Attach optional pooling metadata from ONNX model to network instance.
The importer keeps pooling metadata as additive diagnostics state so later runtime or visualization tooling can reason about spatial stages without modifying core graph wiring.

Parameters:
- `network` - Target network.
- `metadata` - ONNX metadata.

Returns: Nothing.

### attachParsedPoolingMetadata

```ts
attachParsedPoolingMetadata(
  network: default,
  poolingMetadata: OnnxImportPoolingMetadata,
): void
```

Attach parsed pooling metadata to imported network instance.

Parameters:
- `network` - Target network.
- `poolingMetadata` - Parsed pooling metadata payload.

Returns: Nothing.

### buildArchitectureContext

```ts
buildArchitectureContext(
  onnx: OnnxModel,
): OnnxImportArchitectureContext
```

Build architecture extraction context from ONNX graph state.

Parameters:
- `onnx` - Source ONNX model.

Returns: Normalized architecture extraction context.

### buildHiddenLayerSpans

```ts
buildHiddenLayerSpans(
  hiddenLayerSizes: number[],
): OnnxImportHiddenLayerSpan[]
```

Build hidden-layer spans with one-based layer numbering and global offsets.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.

Returns: Hidden-layer span payload list.

### buildImportedLayers

```ts
buildImportedLayers(
  network: default,
  hiddenLayerSizes: number[],
): default[][]
```

Build the imported layer ordering from runtime nodes and hidden-layer widths.

### buildVirtualPoolingShape

```ts
buildVirtualPoolingShape(
  poolingSpec: Pool2DMapping,
  convSpec: Conv2DMapping,
  flattenLayerSet: Set<number>,
): OnnxImportPoolingVirtualShape | null
```

Build one virtual pooled shape from the pre-pool Conv output shape.

Parameters:
- `poolingSpec` - Pool metadata describing the virtual pooling step.
- `convSpec` - Conv metadata describing the pre-pool spatial shape.
- `flattenLayerSet` - Layer indices marked with flatten-after-pool metadata.

Returns: Virtual pooled shape when the metadata is usable.

### calculateSpatialOutputSize

```ts
calculateSpatialOutputSize(
  inputSize: number,
  kernelSize: number,
  strideSize: number,
  leadingPadding: number,
  trailingPadding: number,
): number
```

Calculate one pooled spatial output size from kernel, stride, and padding metadata.

Parameters:
- `inputSize` - Pre-pool spatial size.
- `kernelSize` - Pool kernel size.
- `strideSize` - Pool stride size.
- `leadingPadding` - Leading padding value.
- `trailingPadding` - Trailing padding value.

Returns: Derived output size, or zero when the metadata is unusable.

### collectAvailableConvSpecs

```ts
collectAvailableConvSpecs(
  metadata: OnnxMetadataProperty[],
): Conv2DMapping[]
```

Collect explicit and inferred Conv specs that can anchor pooling shape simulation.

Parameters:
- `metadata` - ONNX metadata entries.

Returns: Layer-indexed Conv specs, preferring explicit specs over inferred ones.

### collectDiagonalRecurrentWeights

```ts
collectDiagonalRecurrentWeights(
  recurrentTensorWeights: number[],
  hiddenLayerSize: number,
): number[]
```

Collect diagonal recurrent weights from flattened layer tensor data.

Parameters:
- `recurrentTensorWeights` - Flattened recurrent tensor weights.
- `hiddenLayerSize` - Hidden-layer width.

Returns: Diagonal recurrent self-weights.

### collectFlattenConsistencyAudit

```ts
collectFlattenConsistencyAudit(
  virtualShapes: OnnxImportPoolingVirtualShape[],
  hiddenLayerSizes: number[],
  outputCount: number,
): OnnxImportFlattenConsistencyAudit[]
```

Collect flatten-consistency audit records for virtual pooled shapes with flatten metadata.

Parameters:
- `virtualShapes` - Derived virtual pooled shapes.
- `hiddenLayerSizes` - Hidden-layer widths from ONNX metadata.
- `outputCount` - Imported output width.

Returns: Metadata-only flatten-consistency audit records.

### collectFusedRecurrentGateWeights

```ts
collectFusedRecurrentGateWeights(
  recurrentTensorWeights: number[],
  unitSize: number,
): number[]
```

Collect diagonal recurrent weights from the recurrent gate block inside a fused tensor.

Parameters:
- `recurrentTensorWeights` - Flattened fused recurrent tensor weights.
- `unitSize` - Fused recurrent unit size.

Returns: Diagonal recurrent self-weights for the recurrent gate slice.

### collectNodesByType

```ts
collectNodesByType(
  nodes: default[],
  nodeType: "hidden" | "input" | "output",
): default[]
```

Collect nodes matching one runtime node-type discriminator.

Parameters:
- `nodes` - Node list.
- `nodeType` - Runtime node type.

Returns: Filtered node list.

### collectPerceptronBoundaryNodes

```ts
collectPerceptronBoundaryNodes(
  nodes: default[],
): default[]
```

Collect input and output boundary nodes for perceptron imports.

Parameters:
- `nodes` - Full network node list.

Returns: Input/output-only node list.

### collectRecurrentLayerSpans

```ts
collectRecurrentLayerSpans(
  restorationContext: OnnxImportRecurrentRestorationContext,
): OnnxImportHiddenLayerSpan[]
```

Resolve recurrent-target hidden-layer spans from metadata + hidden sizes.

Parameters:
- `restorationContext` - Recurrent restoration context.

Returns: Hidden-layer spans requiring recurrent restoration.

### collectVirtualPoolingShapes

```ts
collectVirtualPoolingShapes(
  poolingSpecs: Pool2DMapping[],
  convSpecs: Conv2DMapping[],
  flattenLayers: number[],
): OnnxImportPoolingVirtualShape[]
```

Derive virtual pooled shapes from Conv and Pool metadata without changing weights.

Parameters:
- `poolingSpecs` - Imported pooling metadata specs.
- `convSpecs` - Available explicit/inferred Conv specs.
- `flattenLayers` - Layer indices marked with flatten-after-pool metadata.

Returns: Derived virtual pooled shapes for future consistency checks.

### extractOnnxArchitecture

```ts
extractOnnxArchitecture(
  onnx: OnnxModel,
): OnnxImportArchitectureResult
```

Extract input/output counts and hidden layer sizes from ONNX model.
This architecture probe normalizes graph terminal dimensions and initializer-derived hidden spans into one deterministic result contract used by all downstream reconstruction passes.

Parameters:
- `onnx` - Source ONNX model.

Returns: Parsed architecture dimensions.

### findFusedRecurrentInitializer

```ts
findFusedRecurrentInitializer(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): OnnxTensor | undefined
```

Resolve a fused recurrent tensor for one hidden-layer span when generic `Rk` is absent.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Matching fused recurrent tensor when present.

### findMetadataProperty

```ts
findMetadataProperty(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): OnnxMetadataProperty | undefined
```

Find one ONNX metadata property by key.

Parameters:
- `metadata` - ONNX metadata array.
- `metadataKey` - Metadata key.

Returns: Matching metadata property when present.

### findRecurrentInitializer

```ts
findRecurrentInitializer(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): OnnxTensor | undefined
```

Resolve recurrent initializer tensor for one hidden-layer span.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Recurrent initializer tensor when available.

### inferRecurrentLayerIndicesFromInitializers

```ts
inferRecurrentLayerIndicesFromInitializers(
  hiddenLayerSizes: number[],
  onnx: OnnxModel,
): number[]
```

Infer recurrent layer indices from plain recurrent tensors when metadata is absent.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list used to bound valid layer indices.
- `onnx` - Source ONNX model.

Returns: One-based recurrent layer indices inferred from `Rk` tensors.

### isAdvancedGraphCrossLayerConnection

```ts
isAdvancedGraphCrossLayerConnection(
  value: unknown,
): boolean
```

Validate one parsed cross-layer audit record.

Parameters:
- `value` - Parsed JSON value.

Returns: Whether the value matches the expected metadata shape.

### isAttentionBlock

```ts
isAttentionBlock(
  value: unknown,
): boolean
```

Validate one parsed fixed-width self-attention audit record.

### isResidualAdd

```ts
isResidualAdd(
  value: unknown,
): boolean
```

Validate one parsed residual-add record.

### isSharedInitializerAlias

```ts
isSharedInitializerAlias(
  value: unknown,
): boolean
```

Validate one parsed shared-initializer alias audit record.

### isSingleLayerPerceptronImport

```ts
isSingleLayerPerceptronImport(
  hiddenLayerSizes: number[],
): boolean
```

Determine whether import shape corresponds to a single-layer perceptron.

Parameters:
- `hiddenLayerSizes` - Hidden-layer size list.

Returns: True when no hidden layers exist.

### normalizeRecurrentLayerIndices

```ts
normalizeRecurrentLayerIndices(
  parsedMetadataValue: string | number | boolean | number[] | Record<string, number> | null,
): number[]
```

Normalize recurrent layer indices parsed from metadata JSON.

Parameters:
- `parsedMetadataValue` - Parsed metadata JSON value.

Returns: Recurrent layer indices.

### parseAdvancedGraphCrossLayerConnections

```ts
parseAdvancedGraphCrossLayerConnections(
  metadata: OnnxMetadataProperty[],
): OnnxImportAdvancedGraphCrossLayerConnection[] | null
```

Parse valid cross-layer feed-forward audit metadata.

### parseAdvancedGraphMetadata

```ts
parseAdvancedGraphMetadata(
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel | undefined,
): OnnxImportAdvancedGraphMetadata | null
```

Parse advanced-graph cross-layer metadata from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata entries.

Returns: Parsed advanced-graph metadata, or null when absent or invalid.

### parseAttentionBlockMetadata

```ts
parseAttentionBlockMetadata(
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel | undefined,
): OnnxImportAttentionBlock[] | null
```

Parse valid fixed-width self-attention audit metadata.

### parseOptionalConvSpecs

```ts
parseOptionalConvSpecs(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): Conv2DMapping[]
```

Parse one optional Conv spec metadata field.

Parameters:
- `metadata` - ONNX metadata entries.
- `metadataKey` - Metadata key carrying Conv spec JSON.

Returns: Parsed Conv specs or an empty list when absent/invalid.

### parseOptionalLayerIndicesMetadata

```ts
parseOptionalLayerIndicesMetadata(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): number[]
```

Parse one optional layer-index metadata field.

Parameters:
- `metadata` - ONNX metadata entries.
- `metadataKey` - Metadata key carrying a JSON array of layer indices.

Returns: Parsed layer indices or an empty list when absent/invalid.

### parsePoolingMetadata

```ts
parsePoolingMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
): OnnxImportPoolingMetadata | null
```

Parse pooling metadata payload from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata entries.

Returns: Parsed pooling metadata payload.

### parseRecurrentLayerIndices

```ts
parseRecurrentLayerIndices(
  rawMetadataValue: string,
): number[]
```

Parse recurrent layer indices metadata.

Parameters:
- `rawMetadataValue` - Raw metadata JSON string.

Returns: Normalized recurrent layer indices.

### parseResidualAddMetadata

```ts
parseResidualAddMetadata(
  metadata: OnnxMetadataProperty[],
): OnnxImportResidualAdd[] | null
```

Parse valid one-hop residual-add metadata.

### parseSharedInitializerAliases

```ts
parseSharedInitializerAliases(
  metadata: OnnxMetadataProperty[],
): OnnxImportSharedInitializerAlias[] | null
```

Parse valid shared-initializer alias audit metadata.

### pruneSingleLayerHiddenPlaceholders

```ts
pruneSingleLayerHiddenPlaceholders(
  network: default,
  hiddenLayerSizes: number[],
): void
```

Remove placeholder hidden nodes that arise from single-layer perceptron imports.

Parameters:
- `network` - Target network.
- `hiddenLayerSizes` - Hidden layer sizes.

Returns: Nothing.

### readLastDimensionValue

```ts
readLastDimensionValue(
  dimensions: { dim_value?: number | undefined; }[],
): number
```

Read the terminal ONNX shape dimension value from one shape array.

Parameters:
- `dimensions` - ONNX shape dimensions.

Returns: Terminal `dim_value` payload.

### reconstructFusedRecurrentLayers

```ts
reconstructFusedRecurrentLayers(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: OnnxLayerFactory,
  metadata: OnnxMetadataProperty[],
): void
```

Contract for reconstructFusedRecurrentLayers.

### resolveConsumerLayerWidth

```ts
resolveConsumerLayerWidth(
  consumerLayerIndex: number,
  hiddenLayerSizes: number[],
  outputCount: number,
): number | undefined
```

Resolve the width of the next dense consumer after one flattened pooling site.

Parameters:
- `consumerLayerIndex` - One-based export-layer index for the next dense consumer.
- `hiddenLayerSizes` - Hidden-layer widths from ONNX metadata.
- `outputCount` - Imported output width.

Returns: Consumer width when a dense consumer exists.

### resolveRecurrentLayerIndices

```ts
resolveRecurrentLayerIndices(
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel,
): number[]
```

Resolve recurrent layer indices from ONNX metadata.

Parameters:
- `metadata` - ONNX metadata payload.

Returns: Parsed recurrent layer indices.

### resolveResidualWeight

```ts
resolveResidualWeight(
  residualWeights: number[],
  sourceLayerWidth: number,
  sourceLayerIndex: number,
  targetLayerIndex: number,
): number
```

Resolve one residual branch weight from a row-major target-by-source matrix.

### restoreRecurrentSelfConnections

```ts
restoreRecurrentSelfConnections(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void
```

Contract for restoreRecurrentSelfConnections.

### restoreResidualAddConnections

```ts
restoreResidualAddConnections(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void
```

Restore supported one-hop residual-add skip connections from ONNX metadata.

The import side stays conservative: it only rehydrates skip edges when the
exporter recorded both the residual merge intent and the exact cross-layer
edge list, and when the residual branch tensor is still present.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes from architecture extraction.
- `metadata` - ONNX metadata payload.

Returns: Nothing.

### restoreSingleResidualAddConnections

```ts
restoreSingleResidualAddConnections(
  network: default,
  onnx: OnnxModel,
  importedLayers: default[][],
  crossLayerConnections: OnnxImportAdvancedGraphCrossLayerConnection[],
  residualAdd: OnnxImportResidualAdd,
): void
```

Restore one residual-add branch using the recorded edge list and residual weight tensor.

### sliceLayerHiddenNodes

```ts
sliceLayerHiddenNodes(
  layerConnectionContext: OnnxImportLayerConnectionContext,
): default[]
```

Slice hidden nodes for one hidden-layer span.

Parameters:
- `layerConnectionContext` - Layer connection context.

Returns: Hidden nodes belonging to the span.

### upsertFeedForwardConnection

```ts
upsertFeedForwardConnection(
  sourceNode: default,
  targetNode: default,
  weight: number,
): void
```

Upsert one feed-forward connection between two runtime nodes.

### upsertSelfConnection

```ts
upsertSelfConnection(
  selfConnectionContext: OnnxImportSelfConnectionUpsertContext,
): void
```

Upsert one node self-connection for recurrent import restoration.

Parameters:
- `selfConnectionContext` - Self-connection upsert context.

Returns: Nothing.

## architecture/network/onnx/import/network.onnx.import-fused-recurrent.types.ts

### OnnxFusedGateApplicationContext

Gate-weight application context for one reconstructed fused layer, carrying spec, unit size, and weight arrays.

### OnnxFusedGateRowAssignmentContext

Context for assigning one gate-neuron row from flattened ONNX tensors.

### OnnxFusedLayerNeighborhood

Hidden-layer neighborhood slices around a reconstructed fused layer, including old, previous, and next node lists.

### OnnxFusedLayerReconstructionContext

Execution context for one fused recurrent layer reconstruction, carrying spec, export index, and hidden layer index.

### OnnxFusedLayerRuntime

Runtime interface of a reconstructed fused recurrent layer instance.

The importer only relies on a narrow runtime contract: access to the
reconstructed nodes, an input wiring hook, and an optional output group that
can be reconnected to the next restored layer.

### OnnxFusedRecurrentKind

Supported fused recurrent operator families recognized during ONNX import, currently limited to LSTM and GRU.

### OnnxFusedRecurrentSpec

Fused recurrent family specification used during import reconstruction.

This tells the importer how to interpret one emitted ONNX recurrent family:
how many gates to expect, what order those gates were serialized in, and
which gate owns the self-recurrent diagonal replay.

### OnnxFusedTensorPayload

Fused recurrent tensor payload read from ONNX initializers.

The importer resolves the three recurrent tensor families up front so the
reconstruction pass can focus on wiring and row assignment instead of
repeatedly re-looking up initializers.

### OnnxIncomingWeightAssignmentContext

Context for assigning dense incoming weights for one gate-neuron row.

## architecture/network/onnx/import/network.onnx.import-fused-recurrent.utils.ts

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

### applyGateWeights

```ts
applyGateWeights(
  context: OnnxFusedGateApplicationContext,
): void
```

Apply imported gate parameters to a reconstructed fused layer.

Parameters:
- `context` - Gate application context.

Returns: Nothing.

### assignGateRow

```ts
assignGateRow(
  context: OnnxFusedGateRowAssignmentContext,
): void
```

Assign one gate-neuron row parameters.

Parameters:
- `context` - Gate-row assignment context.

Returns: Nothing.

### assignIncomingWeightAtColumn

```ts
assignIncomingWeightAtColumn(
  context: OnnxIncomingWeightAssignmentContext,
  columnIndex: number,
): void
```

Assign one incoming connection weight by source-column index.

Parameters:
- `context` - Incoming-weight assignment context.
- `columnIndex` - Source column index.

Returns: Nothing.

### assignIncomingWeights

```ts
assignIncomingWeights(
  context: OnnxIncomingWeightAssignmentContext,
): void
```

Assign dense incoming weights for one gate neuron.

Parameters:
- `context` - Incoming-weight assignment context.

Returns: Nothing.

### assignRecurrentDiagonalWeight

```ts
assignRecurrentDiagonalWeight(
  context: OnnxFusedGateRowAssignmentContext,
): void
```

Assign one recurrent diagonal self-weight.

Parameters:
- `context` - Gate-row assignment context.

Returns: Nothing.

### assignRecurrentIncomingWeightAtColumn

```ts
assignRecurrentIncomingWeightAtColumn(
  context: OnnxFusedGateRowAssignmentContext,
  columnIndex: number,
): void
```

Assign one recurrent incoming weight from the native GRU previous-output
carrier into the current gate neuron.

Parameters:
- `context` - Gate-row assignment context.
- `columnIndex` - Recurrent source column index.

Returns: Nothing.

### assignRecurrentWeights

```ts
assignRecurrentWeights(
  context: OnnxFusedGateRowAssignmentContext,
): void
```

Assign recurrent weights for one fused gate row.

Parameters:
- `context` - Gate-row assignment context.

Returns: Nothing.

### buildContiguousGateGroups

```ts
buildContiguousGateGroups(
  fusedNodes: default[],
  gateOrder: string[],
  unitSize: number,
): Record<string, default[]>
```

Build contiguous gate groups from one fused node list.

Parameters:
- `fusedNodes` - Fused node list.
- `gateOrder` - Gate order.
- `unitSize` - Units per gate.

Returns: Gate-name to neuron-list map.

### createFusedLayerRuntime

```ts
createFusedLayerRuntime(
  layerFactory: OnnxLayerFactory,
  spec: OnnxFusedRecurrentSpec,
  unitSize: number,
): OnnxFusedLayerRuntime
```

Create one fused recurrent runtime layer instance.

Parameters:
- `layerFactory` - Dynamic layer module.
- `spec` - Fused family spec.
- `unitSize` - Unit count.

Returns: Runtime fused layer.

### createFusedRecurrentSpecs

```ts
createFusedRecurrentSpecs(): OnnxFusedRecurrentSpec[]
```

Build fused recurrent family specifications.

Returns: Family specifications.

### createHiddenLayerRange

```ts
createHiddenLayerRange(
  hiddenLayerSizes: number[],
  hiddenLayerIndex: number,
): { start: number; end: number; }
```

Create hidden-range boundaries for one hidden-layer index.

Parameters:
- `hiddenLayerSizes` - Hidden-layer widths.
- `hiddenLayerIndex` - Hidden-layer index.

Returns: Start and end boundaries.

### createImmutableSplicedArray

```ts
createImmutableSplicedArray(
  source: TItem[],
  start: number,
  deleteCount: number,
  insertItems: TItem[],
): TItem[]
```

Create an immutable spliced copy, with a compatibility fallback when ES2023
`toSpliced` is typed as optional in ambient declarations.

Parameters:
- `source` - Source array.
- `start` - Start index.
- `deleteCount` - Number of removed items.
- `insertItems` - Items to insert.

Returns: New array containing the splice result.

### createPreviousLayerSourceGroup

```ts
createPreviousLayerSourceGroup(
  previousLayerNodes: default[],
): PreviousLayerSourceGroup
```

Create a source group compatible with both runtime Layer input wiring and
the existing mock fused-layer tests.

Parameters:
- `previousLayerNodes` - Previous-layer node slice.

Returns: Group-like source wrapper.

### deriveLayerNeighborhood

```ts
deriveLayerNeighborhood(
  network: default,
  hiddenLayerSizes: number[],
  hiddenLayerIndex: number,
): OnnxFusedLayerNeighborhood
```

Derive hidden-layer neighborhood slices for replacement traversal.

Parameters:
- `network` - Target network.
- `hiddenLayerSizes` - Hidden-layer widths.
- `hiddenLayerIndex` - Hidden-layer index.

Returns: Layer neighborhood.

### deriveNextLayerNodes

```ts
deriveNextLayerNodes(
  network: default,
  hiddenLayerSizes: number[],
  hiddenLayerIndex: number,
  hiddenNodes: default[],
): default[]
```

Derive next-layer nodes for a hidden layer.

Parameters:
- `network` - Target network.
- `hiddenLayerSizes` - Hidden-layer widths.
- `hiddenLayerIndex` - Hidden-layer index.
- `hiddenNodes` - All hidden nodes.

Returns: Next-layer nodes.

### derivePreviousLayerNodes

```ts
derivePreviousLayerNodes(
  network: default,
  hiddenLayerSizes: number[],
  hiddenLayerIndex: number,
  hiddenNodes: default[],
): default[]
```

Derive previous-layer nodes for a hidden layer.

Parameters:
- `network` - Target network.
- `hiddenLayerSizes` - Hidden-layer widths.
- `hiddenLayerIndex` - Hidden-layer index.
- `hiddenNodes` - All hidden nodes.

Returns: Previous-layer nodes.

### deriveUnitSize

```ts
deriveUnitSize(
  rows: number,
  gateCount: number,
): number | null
```

Derive hidden unit size from recurrent row count and gate count.

Parameters:
- `rows` - Recurrent rows.
- `gateCount` - Gate count.

Returns: Unit size when compatible.

### detachOldLayerConnections

```ts
detachOldLayerConnections(
  network: default,
  neighborhood: OnnxFusedLayerNeighborhood,
): void
```

Detach all connections touching replaced hidden-layer nodes.

Parameters:
- `network` - Target network.
- `neighborhood` - Layer neighborhood.

Returns: Nothing.

### filterNodesByType

```ts
filterNodesByType(
  network: default,
  nodeType: string,
): default[]
```

Filter network nodes by semantic type.

Parameters:
- `network` - Target network.
- `nodeType` - Node type name.

Returns: Filtered node collection.

### findInitializerTensor

```ts
findInitializerTensor(
  onnx: OnnxModel,
  kind: OnnxFusedRecurrentKind,
  suffix: string,
  hiddenLayerIndex: number,
): OnnxTensor | undefined
```

Find one initializer tensor by fused family naming convention.

Parameters:
- `onnx` - Source ONNX model.
- `kind` - Fused family kind.
- `suffix` - Tensor suffix.
- `hiddenLayerIndex` - Hidden-layer index.

Returns: Initializer tensor when found.

### isValidHiddenLayerIndex

```ts
isValidHiddenLayerIndex(
  hiddenLayerSizes: number[],
  hiddenLayerIndex: number,
): boolean
```

Validate hidden-layer index boundaries.

Parameters:
- `hiddenLayerSizes` - Hidden-layer widths.
- `hiddenLayerIndex` - Hidden-layer index.

Returns: True when index is valid.

### parseEmittedLayerIndices

```ts
parseEmittedLayerIndices(
  metadata: OnnxMetadataProperty[],
  spec: OnnxFusedRecurrentSpec,
): number[]
```

Parse exported layer indices from metadata for one fused family.

Parameters:
- `metadata` - ONNX metadata properties.
- `spec` - Fused family spec.

Returns: Export-layer indices.

### parseMetadataJsonArray

```ts
parseMetadataJsonArray(
  metadataValue: string,
): number[]
```

Parse metadata JSON payload as an array of indices.

Parameters:
- `metadataValue` - Serialized metadata payload.

Returns: Parsed index array.

### reconstructAllFusedFamilies

```ts
reconstructAllFusedFamilies(
  scope: FusedRecurrentImportScope,
  specs: OnnxFusedRecurrentSpec[],
): void
```

Reconstruct all fused recurrent families declared by metadata.

Parameters:
- `scope` - Import scope.
- `specs` - Fused family specs.

Returns: Nothing.

### reconstructFusedRecurrentLayers

```ts
reconstructFusedRecurrentLayers(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  layerFactory: OnnxLayerFactory,
  metadata: OnnxMetadataProperty[],
): void
```

Contract for reconstructFusedRecurrentLayers.

### reconstructOneFusedFamily

```ts
reconstructOneFusedFamily(
  scope: FusedRecurrentImportScope,
  spec: OnnxFusedRecurrentSpec,
): void
```

Reconstruct one fused family across all emitted layer indices.

Parameters:
- `scope` - Import scope.
- `spec` - Fused family spec.

Returns: Nothing.

### reconstructOneFusedLayer

```ts
reconstructOneFusedLayer(
  scope: FusedRecurrentImportScope,
  context: OnnxFusedLayerReconstructionContext,
): void
```

Reconstruct one fused layer from metadata and ONNX initializers.

Parameters:
- `scope` - Import scope.
- `context` - Layer reconstruction context.

Returns: Nothing.

### replaceHiddenNodes

```ts
replaceHiddenNodes(
  network: default,
  neighborhood: OnnxFusedLayerNeighborhood,
  replacementNodes: default[],
): void
```

Replace hidden node segment with reconstructed fused layer nodes.

Parameters:
- `network` - Target network.
- `neighborhood` - Layer neighborhood.
- `replacementNodes` - Replacement nodes.

Returns: Nothing.

### resolveFusedTensors

```ts
resolveFusedTensors(
  onnx: OnnxModel,
  context: OnnxFusedLayerReconstructionContext,
): OnnxFusedTensorPayload | null
```

Resolve ONNX fused tensors for one hidden layer.

Parameters:
- `onnx` - Source ONNX model.
- `context` - Layer reconstruction context.

Returns: Tensor payload when fully available.

### resolveGateGroups

```ts
resolveGateGroups(
  fusedNodes: default[],
  spec: OnnxFusedRecurrentSpec,
  unitSize: number,
): { gateGroups: Record<string, default[]>; recurrentSourceNodes: default[]; }
```

Resolve gate groups and recurrent source nodes from one fused runtime layout.

Parameters:
- `fusedNodes` - Fused node list.
- `spec` - Fused family specification.
- `unitSize` - Units per gate.

Returns: Gate groups plus recurrent-source nodes.

### resolveGruGateGroups

```ts
resolveGruGateGroups(
  fusedNodes: default[],
  unitSize: number,
): { gateGroups: Record<string, default[]>; recurrentSourceNodes: default[]; }
```

Resolve GRU gate groups for either the native six-group layout or the
compact three-gate mock layout used by owner-local tests.

Parameters:
- `fusedNodes` - Fused node list.
- `unitSize` - Units per gate.

Returns: Gate-name to neuron-list map plus recurrent-source nodes.

### sumHiddenLayerSizes

```ts
sumHiddenLayerSizes(
  hiddenLayerSizes: number[],
  startIndex: number,
  endIndex: number,
): number
```

Sum hidden-layer sizes over a half-open range.

Parameters:
- `hiddenLayerSizes` - Hidden-layer widths.
- `startIndex` - Inclusive start.
- `endIndex` - Exclusive end.

Returns: Summed size.

### toHiddenLayerIndex

```ts
toHiddenLayerIndex(
  exportLayerIndex: number,
): number
```

Convert export-layer index to hidden-layer index.

Parameters:
- `exportLayerIndex` - Export-layer index.

Returns: Hidden-layer index.

### wireFusedLayer

```ts
wireFusedLayer(
  fusedLayerRuntime: OnnxFusedLayerRuntime,
  previousLayerNodes: default[],
  nextLayerNodes: default[],
): void
```

Wire fused layer between previous and next layer slices.

Parameters:
- `fusedLayerRuntime` - Reconstructed fused layer runtime.
- `previousLayerNodes` - Previous-layer nodes.
- `nextLayerNodes` - Next-layer nodes.

Returns: Nothing.

## architecture/network/onnx/import/network.onnx.import-external.types.ts

### DecodedExternalOnnxAttribute

Decoded ONNX attribute payload preserving scalar, integer, and byte-string forms emitted by the protobuf decoder.
Attribute decoding uses this shape before operation-specific coercion into importer contracts.

### DecodedExternalOnnxDimension

Decoded ONNX dimension payload used by external import parsing to preserve symbolic and numeric shape information from protobuf conversion.
Import normalization relies on this shape to reconstruct rank and axis semantics before tensor compatibility checks run.

### DecodedExternalOnnxGraph

Decoded ONNX graph payload containing decoded graph interfaces, initializer tables, and ordered node records for external import orchestration.
Graph traversal, initializer indexing, and topology validation all begin from this representation.

### DecodedExternalOnnxModel

Decoded ONNX model payload containing optional graph content and operator-set imports used to validate supported external import lanes.
External import entrypoints decode into this shape before compatibility and topology checks proceed.

### DecodedExternalOnnxNode

Decoded ONNX node payload describing operator identity, wiring, and decoded attribute list for importer normalization passes.
Node-level validation and operator support checks consume this schema directly.

### DecodedExternalOnnxOpsetImport

Decoded ONNX operator-set import payload carrying the domain string and version number used by external import compatibility checks.

### DecodedExternalOnnxTensor

Decoded ONNX tensor payload containing name, data type, shape dimensions, and raw or float initializer storage fields.
Initializer extraction and shape-matching code paths depend on this decoded tensor contract.

### DecodedExternalOnnxTensorType

Decoded ONNX tensor-type payload describing element type and optional shape dimensions after external binary decode.
This type bridges raw protobuf decode output and importer-owned tensor validation routines.

### DecodedExternalOnnxValueInfo

Decoded ONNX value-info payload carrying named tensor metadata for graph inputs, outputs, and intermediate value descriptors.
It allows importer passes to align tensor names, element types, and shapes across graph boundaries.

### OnnxDecodedBytes

Union type for raw byte fields emitted by `onnx-proto` object conversion; accepts string, Uint8Array, or number-array representations.

### OnnxDecodedLongLike

Union type for ONNX 64-bit integer fields decoded by `onnx-proto`; includes numeric, string, and toString-capable object forms to handle platform-specific long encoding.

### OnnxExternalDenseChain

Canonical importer-owned dense chain derived from an accepted external binary graph, collecting opset version, IO widths, and an ordered layer list.

### OnnxExternalDenseLayer

Canonical single-layer payload for the external dense import lane, carrying input and output widths, weight values, biases, and the resolved activation operator.

### OnnxExternalImportError

Error raised when an external ONNX binary falls outside the first supported import lane.

### OnnxExternalImportErrorCategory

Named rejection category set for the external import lane; each string label identifies a distinct failure class so callers can route errors without string matching.

## architecture/network/onnx/import/network.onnx.import-concat.utils.ts

### attachOnnxConcatMergeMetadata

```ts
attachOnnxConcatMergeMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel | undefined,
): void
```

Attach validated concat-merge audit metadata to an imported network instance.

Parameters:
- `network` - Target network.
- `metadata` - ONNX metadata payload.
- `onnx` - Source ONNX model used for same-family validation.

Returns: Nothing.

### buildImportedLayers

```ts
buildImportedLayers(
  network: default,
  hiddenLayerSizes: number[],
): default[][]
```

Build the imported layer ordering from runtime nodes and hidden-layer widths.

### collectNodesByType

```ts
collectNodesByType(
  nodes: default[],
  nodeType: "hidden" | "input" | "output",
): default[]
```

Collect nodes matching one runtime node-type discriminator.

### findInitializer

```ts
findInitializer(
  onnx: OnnxModel,
  tensorName: string,
): OnnxTensor | undefined
```

Resolve one initializer by tensor name.

### findMetadataProperty

```ts
findMetadataProperty(
  metadata: OnnxMetadataProperty[],
  metadataKey: string,
): OnnxMetadataProperty | undefined
```

Find one ONNX metadata property by key.

### findNodeByName

```ts
findNodeByName(
  onnx: OnnxModel,
  nodeName: string,
): OnnxNode | undefined
```

Resolve one node by its deterministic export name.

### isConcatMerge

```ts
isConcatMerge(
  value: unknown,
): boolean
```

Validate one parsed concat-merge record.

### matchesExportedConcatMergeSubset

```ts
matchesExportedConcatMergeSubset(
  onnx: OnnxModel,
  concatMerge: OnnxImportConcatMerge,
): boolean
```

Validate the same-family explicit concat subset emitted by this exporter.

### parseConcatMergeMetadata

```ts
parseConcatMergeMetadata(
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel,
): OnnxImportConcatMerge[] | null
```

Parse valid explicit concat-merge metadata.

### resolveConcatMergeWeight

```ts
resolveConcatMergeWeight(
  mergedWeights: number[],
  mergedSourceWidth: number,
  previousLayerWidth: number,
  targetLayerIndex: number,
  sourceLayerIndex: number,
): number
```

Resolve one concat branch weight from the widened row-major target-by-merged-source matrix.

### restoreConcatMergeConnections

```ts
restoreConcatMergeConnections(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
  metadata: OnnxMetadataProperty[],
): void
```

Restore supported concat-merge skip connections from ONNX metadata.

The import side stays narrow and deterministic: it only rebuilds skipped
source-layer fan-in for the explicit concat subset emitted by the exporter,
using the widened dense weight tensor tail while the ordinary adjacent-layer
slice remains assigned by the baseline dense import path.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes from architecture extraction.
- `metadata` - ONNX metadata payload.

Returns: Nothing.

### restoreSingleConcatMergeConnections

```ts
restoreSingleConcatMergeConnections(
  onnx: OnnxModel,
  importedLayers: default[][],
  concatMerge: OnnxImportConcatMerge,
): void
```

Restore one concat-merge branch using the widened dense weight tensor tail.

### upsertFeedForwardConnection

```ts
upsertFeedForwardConnection(
  sourceNode: default,
  targetNode: default,
  weight: number,
): void
```

Upsert one feed-forward connection between two runtime nodes.

## architecture/network/onnx/import/network.onnx.import-external.utils.ts

### normalizeExternalBinaryOnnxModel

```ts
normalizeExternalBinaryOnnxModel(
  binaryModel: Uint8Array<ArrayBufferLike>,
): OnnxModel
```

Normalize the first supported external binary ONNX subset into an importer-owned model.

Parameters:
- `binaryModel` - Binary `ModelProto` payload.

Returns: Canonical JSON-first model that the existing import flow can reconstruct.

### normalizeExternalDenseChain

```ts
normalizeExternalDenseChain(
  binaryModel: Uint8Array<ArrayBufferLike>,
): OnnxExternalDenseChain
```

Normalize the first supported external binary ONNX subset into a canonical dense chain.

Parameters:
- `binaryModel` - Binary `ModelProto` payload.

Returns: Canonical dense-chain payload for importer reconstruction.
