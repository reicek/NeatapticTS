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

Validation context for perceptron size-list checks during ONNX import.

### OnnxRuntimeFactories

Runtime factories consumed during ONNX import network reconstruction.

### OnnxRuntimeLayerFactory

```ts
OnnxRuntimeLayerFactory(
  size: number,
): default
```

Runtime layer-constructor signature used for recurrent layer reconstruction.

### OnnxRuntimeLayerModule

Runtime layer module shape consumed by ONNX import orchestration.

### OnnxRuntimePerceptronFactory

```ts
OnnxRuntimePerceptronFactory(
  sizes: number[],
): default
```

Runtime perceptron factory signature used by ONNX import orchestration.

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

Resolve runtime factories used by ONNX import orchestration.

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

Context for assigning aggregated dense tensors for one layer.

### OnnxImportAggregatedNeuronAssignmentContext

Context for assigning one aggregated dense target neuron row.

### OnnxImportConvCoordinateAssignmentContext

Context for applying Conv weights and bias at one output coordinate.

### OnnxImportConvKernelAssignmentContext

Context for assigning one concrete Conv kernel connection weight.

### OnnxImportConvLayerContext

Context for reconstructing one Conv layer's imported connectivity.

### OnnxImportConvLayerContextBuildParams

Build params for creating one Conv reconstruction layer context.

### OnnxImportConvMetadata

Parsed Conv metadata payload used for optional reconstruction pass.

### OnnxImportConvNodeSlices

Layer node slices used while applying Conv reconstruction assignments.

### OnnxImportConvOutputCoordinate

Coordinate for one Conv output neuron traversal position.

### OnnxImportConvSourceLayout

Source layout used when replaying Conv weights onto dense source nodes.

### OnnxImportConvTensorContext

Resolved Conv initializer tensors and dimensions for one layer.

### OnnxImportHiddenSizeDerivationContext

Context for deriving hidden layer sizes from initializer tensors and metadata.

### OnnxImportInboundConnectionMap

Inbound connection lookup map keyed by source node for one target neuron.

### OnnxImportLayerNodePair

Node slices for one sequential imported layer assignment pass.

### OnnxImportLayerNodePairBuildParams

Build params for one sequential layer node-pair slice operation.

### OnnxImportLayerTensorNames

Weight tensor names for one imported layer index.

### OnnxImportLayerWeightBucket

Bucketed ONNX dense/per-neuron tensors for one exported layer index.

### OnnxImportPerNeuronAssignmentContext

Context for assigning one per-neuron imported target node.

### OnnxImportPerNeuronLayerAssignmentContext

Context for assigning per-neuron tensors for one layer.

### OnnxImportWeightAssignmentBuildParams

Build params for creating shared ONNX import weight-assignment context.

### OnnxImportWeightAssignmentContext

Shared weight-assignment context built once per ONNX import.

## architecture/network/onnx/import/network.onnx.import-weights.utils.ts

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

Assign weights and biases from ONNX initializers to a newly created network.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadataProps` - Optional ONNX metadata properties.

Returns: Nothing.

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

Extract hidden layer sizes from ONNX initializers (weight tensors).

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

### assignActivationFunctions

```ts
assignActivationFunctions(
  network: default,
  onnx: OnnxModel,
  hiddenLayerSizes: number[],
): void
```

Assign node activation functions from ONNX activation nodes.

Parameters:
- `network` - Target network to mutate.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer size list.

Returns: Nothing.

## architecture/network/onnx/import/network.onnx.import-orchestrators.types.ts

Import-owned type surface for ONNX architecture reconstruction orchestration.

These payloads stay close to the orchestration helpers that parse terminal
dimensions, restore recurrent self-connections, and attach pooling metadata.
Keeping them here makes the import chapter explain its own execution state
without forcing the root ONNX compatibility barrel to remain the ownership
home for importer-only details.

### NetworkWithOnnxImportAdvancedGraph

Network instance augmented with optional imported advanced-graph metadata.

### NetworkWithOnnxImportPooling

Network instance augmented with optional imported ONNX pooling metadata.

### OnnxImportAdvancedGraphCrossLayerConnection

Audit-only cross-layer feed-forward edge carried through Phase 5 import fallback.

### OnnxImportAdvancedGraphMetadata

Parsed advanced-graph metadata attached to imported network instances.

### OnnxImportArchitectureContext

Shared architecture extraction context with resolved graph dimensions.

### OnnxImportArchitectureResult

Parsed architecture dimensions extracted from ONNX import graph payloads.

### OnnxImportAttentionBlock

Explicit fixed-width self-attention block carried through Phase 5 import fallback.

### OnnxImportConcatMerge

Explicit concat merge carried through Phase 5 import hardening.

### OnnxImportDimensionRecord

Loose ONNX shape-dimension record used by legacy import payload access.

### OnnxImportFlattenConsistencyAudit

Metadata-only audit record comparing a flattened pooled width to the next dense width.

### OnnxImportHiddenLayerSpan

Hidden-layer span payload with one-based layer numbering and global offset.

### OnnxImportLayerConnectionContext

Execution context for assigning one hidden-layer recurrent diagonal tensor.

### OnnxImportPoolingMetadata

Parsed pooling metadata payload attached to imported network instances.

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
- `metadata` - ONNX metadata.

Returns: Nothing.

### attachOnnxPoolingMetadata

```ts
attachOnnxPoolingMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
): void
```

Attach optional pooling metadata from ONNX model to network instance.

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

Remove placeholder hidden nodes for single-layer perceptron imports.

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

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `layerFactory` - Dynamic layer module.
- `metadata` - ONNX metadata properties.

Returns: Nothing.

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

Restore recurrent self-connections from recurrent metadata and R tensors.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `metadata` - Parsed metadata properties.

Returns: Nothing.

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

Gate-weight application context for one reconstructed fused layer.

### OnnxFusedGateRowAssignmentContext

Context for assigning one gate-neuron row from flattened ONNX tensors.

### OnnxFusedLayerNeighborhood

Hidden-layer neighborhood slices around a reconstructed fused layer.

### OnnxFusedLayerReconstructionContext

Execution context for one fused recurrent layer reconstruction.

### OnnxFusedLayerRuntime

Runtime interface of a reconstructed fused recurrent layer instance.

The importer only relies on a narrow runtime contract: access to the
reconstructed nodes, an input wiring hook, and an optional output group that
can be reconnected to the next restored layer.

### OnnxFusedRecurrentKind

Supported fused recurrent operator families recognized during ONNX import.

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

Reconstruct emitted fused LSTM/GRU layers from ONNX metadata and initializers.

Parameters:
- `network` - Target network.
- `onnx` - Source ONNX model.
- `hiddenLayerSizes` - Hidden layer sizes.
- `layerFactory` - Dynamic layer module.
- `metadata` - ONNX metadata properties.

Returns: Nothing.

## architecture/network/onnx/import/network.onnx.import-concat.utils.ts

### attachOnnxConcatMergeMetadata

```ts
attachOnnxConcatMergeMetadata(
  network: default,
  metadata: OnnxMetadataProperty[],
  onnx: OnnxModel | undefined,
): void
```

Attach validated concat-merge audit metadata onto an imported network.

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
