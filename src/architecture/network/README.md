# architecture/network

## architecture/network/network.types.ts

### ActivateNetworkInternals

Runtime interface for activation internals.

### ActivationFunction

`(x: number, derivate: boolean | undefined) => number`

Runtime activation function signature used by ONNX activation import/export paths.

### ActivationSquashFunction

`(x: number, derivate: boolean | undefined) => number`

Activation function signature used by ONNX layer emission helpers.

### BackwardCandidateTraversalContext

Immutable context for backward candidate traversal.

### BuildAdjacencyContext

Shared immutable inputs used across the adjacency build pipeline.

### CheckpointConfig

Checkpoint callback configuration.

### CompactConnectionRebuildContext

Context for compact-connection reconstruction.

### CompactNodeRebuildContext

Context for compact-node reconstruction.

### CompactPayloadContext

Context carrying compact payload fields.

### CompactSerializedNetworkTuple

Compact tuple payload used by binary-like serialize output.

### ConnectionGene

Crossover connection-gene descriptor.

### ConnectionGeneSelectionContext

Immutable context for selecting inherited genes.

### ConnectionGeneticProps

Extended connection shape used during genetic crossover.

### ConnectionGroupReinitContext

Context for reinitializing connection group weights.

### ConnectionInternals

Internal Connection properties accessed during slab operations.

### ConnectionInternalsWithEnabled

Connection view with optional enabled flag.

### ConnectionSlabView

Shape returned by getConnectionSlab describing the packed SoA view.

### ConnectionSplitResult

Result of replacing a connection with split hidden node.

### ConnectionWeightNoiseProps

Internal runtime properties attached to Connection instances.

### ConnectNetworkInternals

Runtime interface for connect internals.

### Conv2DMapping

Mapping declaration for treating a fully-connected layer as a 2D convolution during export.

### ConvInferenceEvaluationContext

Width and shape evaluation context used by Conv inference helpers.

### ConvInferenceKernelEvaluationContext

Kernel candidate context for one Conv inference evaluation pass.

### ConvInferenceResult

Collected inferred Conv metadata payload.

### ConvInferenceTraversalContext

Traversal context for one hidden layer during Conv inference.

### ConvKernelConsistencyContext

Context for kernel-coordinate consistency checks at one output position.

### ConvLayerPairContext

Context for one resolved Conv mapping layer pair.

### ConvOutputCoordinate

Coordinate for one Conv output neuron position.

### ConvRepresentativeKernelContext

Context for representative Conv kernel collection per output channel.

### ConvSharingValidationContext

Context for validating Conv sharing across all declared mappings.

### ConvSharingValidationResult

Result of Conv sharing validation across declared mappings.

### CostFunction

`(target: number[], output: number[]) => number`

Cost function signature used by training utilities.

### CostFunctionOrObject

Cost function object compatibility shape.

### CostFunctionOrRef

Evolve-side serializable cost-function reference.

### CrossoverContext

Immutable baseline context for one crossover run.

### CrossoverNodeBuildContext

Node-build context derived from crossover baseline.

### DenseActivationContext

Dense activation emission context.

### DenseActivationNodePayload

Strongly typed activation node payload used by dense export helpers.

### DenseGemmNodePayload

Strongly typed Gemm node payload used by dense export helpers.

### DenseGraphNames

Dense graph tensor names.

### DenseInitializerValues

Dense initializer value arrays.

### DenseLayerContext

Dense layer context enriched with resolved activation function.

### DenseLayerParams

Parameters for dense layer emission.

### DenseOrderedNodePayload

Dense node payload union used by ordered append helpers.

### DenseTensorNames

Dense initializer tensor names.

### DenseWeightBuildContext

Context for building dense layer initializers from two adjacent layers.

### DenseWeightBuildResult

Dense layer initializer fold output.

### DenseWeightRow

One collected dense row before fold to flattened initializers.

### DenseWeightRowCollectionContext

Context for collecting one dense row.

### DeterministicChainMutationContext

Context for deterministic-chain add-node mutation.

### DeterministicNetworkInternals

Runtime interface for deterministic internals.

### DiagonalRecurrentBuildContext

Context for building a diagonal recurrent matrix from self-connections.

### DirectionalConnectionContext

Indexed context for directional connection metadata.

### DistinctNodePair

Selected distinct node pair for swap mutation.

### EvolutionaryTargetContext

Context for evolutionary sparsity target computation.

### EvolutionaryTargetResult

Result of evolutionary sparsity target computation.

### EvolutionConfig

Internal normalized evolution config.

### EvolutionFitnessFunction

`(arg0: import("C:/NeatapticTS/src/architecture/network").default & import("C:/NeatapticTS/src/architecture/network").default[]) => number | Promise<void>`

Unified evolution fitness callback shape.

### EvolutionLoopState

Mutable state tracked during evolution loop.

### EvolutionSettings

Scalar evolution settings used by orchestration.

### EvolutionStopConditions

Effective evolution stopping conditions.

### EvolveCostFunction

`(target: number[], output: number[]) => number`

Evolve-side cost function signature.

### EvolveOptions

Evolve options bag.

### ExportNodeIndexAssignmentContext

Context for assigning a stable export index to one node.

### FanOutCollectionContext

Context for fan-out collection: build inputs plus the output count buffer.

### FastSlabNodeRuntime

Node shape required by fast slab activation kernels.

### FitnessSetup

Result of fitness-strategy setup.

### FlattenAfterPoolingContext

Flatten emission context after optional pooling.

### ForwardCandidateTraversalContext

Immutable context for forward candidate traversal.

### FusedRecurrentEmissionExecutionContext

Shared execution context for emitting one fused recurrent layer payload.

### FusedRecurrentGraphNames

Context for ONNX fused recurrent node payload names.

### FusedRecurrentInitializerNames

Context for ONNX fused recurrent initializer names.

### GatingNetworkProps

Internal network properties accessed during gating operations.

### GeneEndpointsContext

Endpoints for one gene traversal step.

### GeneticNetwork

Runtime network shape used by crossover internals.

### GeneTraversalContext

Traversal context for one connection gene.

### GlobalThisWithStructuredClone

GlobalThis extension exposing optional structuredClone.

### GradientClipConfig

Gradient clipping configuration.

### GruEmissionContext

Context for heuristic GRU emission when a layer matches expected shape.

### HiddenLayerActivationTraversalContext

Hidden-layer traversal context for assigning imported activation functions.

### HiddenLayerHeuristicContext

Context for one hidden layer during heuristic recurrent emission.

### IndexedMetadataAppendContext

Append-an-index metadata context for JSON-array metadata keys.

### InputOutputEndpoints

Required endpoint pair for input/output edge seeding.

### JsonConnectionRebuildContext

Context for JSON-connection reconstruction.

### JsonNodeRebuildContext

Context for JSON-node reconstruction.

### LayerActivationContext

Activation analysis context for one layer.

### LayerActivationValidationContext

Activation-homogeneity decision context for one current layer.

### LayerBuildContext

Layer build context used while emitting one ONNX graph layer segment.

### LayerConnectivityValidationContext

Connectivity decision context for one source-target node pair.

### LayerOrderingNodeGroups

Node partitions used by ONNX layered-ordering inference traversal.

### LayerOrderingResolutionContext

Mutable traversal state while resolving hidden-layer ordering.

### LayerRecurrentDecisionContext

Context used to decide recurrent emission branch usage.

### LayerTraversalContext

Layer traversal context with adjacent layers and output classification.

### LayerValidationTraversalContext

Layer-wise validation context for activation and connectivity checks.

### LstmCandidateContext

Candidate context for validating one LSTM-like hidden layer pattern.

### LstmEmissionContext

Context for heuristic LSTM emission when a layer matches expected shape.

### LstmLayerTraversalContext

Traversal context for one hidden layer during LSTM stub collection.

### LstmPatternStub

Heuristic LSTM pattern stub for metadata output.

### MetricsHook

`(m: { iteration: number; error: number; plateauError?: number | undefined; gradNorm: number; }) => void`

Metrics hook signature.

### MixedPrecisionConfig

Mixed-precision configuration.

### MixedPrecisionDynamicConfig

Dynamic mixed-precision configuration.

### MonitoredSmoothingConfig

Config for monitored smoothing computation.

### MovingAverageType

Moving-average strategy identifier.

### MutationHandler

`(method: import("C:/NeatapticTS/src/architecture/network/network.types").MutationMethod | undefined) => void`

Mutation handler function contract.

### MutationMethod

Mutation method descriptor shape.

### MutationMethodObject

Object-only form of mutation method descriptor.

### NeatRuntime

Minimal runtime contract consumed from NEAT in evolve utilities.

### NetworkActivationRuntime

Runtime activation contract used by slab-based execution paths.

### NetworkConstructor

Constructor signature for runtime Network import.

### NetworkGeneticProps

Runtime properties used during genetic operations.

### NetworkInternalsWithDropout

Serialize internals with optional dropout field.

### NetworkJSON

Verbose JSON payload representation.

### NetworkJSONConnection

Verbose JSON connection representation.

### NetworkJSONNode

Verbose JSON node representation.

### NetworkMutationProps

Internal network properties accessed during mutations.

### NetworkPruningProps

Internal network properties accessed during pruning operations.

### NetworkRemoveProps

Internal network properties accessed during remove operations.

### NetworkRuntimeProps

Internal runtime properties attached to Network instances.

### NetworkSlabProps

Internal Network properties for slab operations.

### NetworkStandaloneProps

Internal standalone generation network view.

### NetworkTopoRuntime

Runtime topology contract used to lazily rebuild topological order.

### NetworkWithOnnxImportPooling

Network instance augmented with optional imported ONNX pooling metadata.

### NodeConnectionSnapshotContext

Snapshot of node adjacency prior to removal.

### NodeInternals

Runtime interface for accessing node internal properties.
Nodes have runtime properties for connections, bias, and squash that aren't in the public interface.

### NodeInternalsWithExportIndex

Runtime node internals augmented with optional export index metadata.

### NodePair

Canonical source-target node pair tuple.

### NodeRemovalContext

Immutable context for validated node-removal request.

### NodeWithIndex

Node with generated index for standalone-code emission.

### OffspringMaterializationContext

Immutable context for offspring materialization.

### OnnxActivationAssignmentContext

Shared activation-assignment context for hidden and output traversal.

### OnnxActivationLayerOperations

Layer-indexed activation operator lookup extracted from ONNX graph nodes.

### OnnxActivationOperation

Supported ONNX activation operators recognized during activation import.

### OnnxActivationOperationResolutionContext

Activation operation resolution context for one neuron or layer default.

### OnnxActivationParseResult

Parsed ONNX activation-node naming payload.

### OnnxAttribute

ONNX node attribute.

### OnnxBaseModelBuildContext

Context for constructing a base ONNX model shell.

### OnnxBuildResolvedOptions

Resolved options used by ONNX model build orchestration.

### OnnxConvEmissionContext

Context used after resolving Conv mapping for one layer.

### OnnxConvEmissionParams

Parameters accepted by Conv layer emission.

### OnnxConvKernelCoordinate

Coordinate for one Conv kernel weight lookup.

### OnnxConvParameters

Flattened Conv parameters for ONNX initializers.

### OnnxConvTensorNames

Tensor names generated for Conv parameters.

### OnnxDimension

ONNX tensor type shape dimension.

### OnnxExportOptions

Options controlling ONNX export behavior (Phase 1).

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

### OnnxFusedRecurrentKind

Supported fused recurrent operator families recognized during ONNX import.

### OnnxFusedRecurrentSpec

Fused recurrent family specification used during import reconstruction.

### OnnxFusedTensorPayload

Fused recurrent tensor payload read from ONNX initializers.

### OnnxGraph

### OnnxGraphDimensionBuildContext

Context for constructing input/output ONNX graph dimensions.

### OnnxGraphDimensions

Output dimensions used by ONNX graph input/output value info payloads.

### OnnxImportAggregatedLayerAssignmentContext

Context for assigning aggregated dense tensors for one layer.

### OnnxImportAggregatedNeuronAssignmentContext

Context for assigning one aggregated dense target neuron row.

### OnnxImportArchitectureContext

Shared architecture extraction context with resolved graph dimensions.

### OnnxImportArchitectureResult

Parsed architecture dimensions extracted from ONNX import graph payloads.

### OnnxImportConvCoordinateAssignmentContext

Context for applying Conv weights/bias at one output coordinate.

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

### OnnxImportConvTensorContext

Resolved Conv initializer tensors and dimensions for one layer.

### OnnxImportDimensionRecord

Loose ONNX shape-dimension record used by legacy import payload access.

### OnnxImportHiddenLayerSpan

Hidden-layer span payload with one-based layer numbering and global offset.

### OnnxImportHiddenSizeDerivationContext

Context for deriving hidden layer sizes from initializer tensors and metadata.

### OnnxImportInboundConnectionMap

Inbound connection lookup map keyed by source node for one target neuron.

### OnnxImportLayerConnectionContext

Execution context for assigning one hidden-layer recurrent diagonal tensor.

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

### OnnxImportPoolingMetadata

Parsed pooling metadata payload attached to imported network instances.

### OnnxImportRecurrentRestorationContext

Context for recurrent self-connection restoration from ONNX metadata and tensors.

### OnnxImportSelfConnectionUpsertContext

Context for upserting one hidden node self-connection from recurrent weight.

### OnnxImportWeightAssignmentBuildParams

Build params for creating shared ONNX import weight-assignment context.

### OnnxImportWeightAssignmentContext

Shared weight-assignment context built once per ONNX import.

### OnnxIncomingWeightAssignmentContext

Context for assigning dense incoming weights for one gate-neuron row.

### OnnxLayerEmissionContext

Context for emitting non-input layers during model build.

### OnnxLayerEmissionResult

Result of emitting non-input export layers.

### OnnxLayerFactory

Runtime factory map used to construct dynamic recurrent layer modules.

### OnnxMetadataProperty

Canonical metadata key-value pair used in ONNX model metadata_props.

### OnnxModel

### OnnxModelMetadataContext

Context for applying optional ONNX model metadata.

### OnnxNode

### OnnxPerceptronBuildContext

Build context for mapping ONNX layer sizes into a Neataptic MLP factory call.

### OnnxPerceptronSizeValidationContext

Validation context for perceptron size-list checks during ONNX import.

### OnnxPostProcessingContext

Context for post-processing and export metadata finalization.

### OnnxRecurrentCollectionContext

Context for collecting recurrent layer indices during model build.

### OnnxRecurrentInputValueInfoContext

Context for constructing one recurrent previous-state graph input payload.

### OnnxRecurrentLayerProcessingContext

Execution context for processing one hidden recurrent layer.

### OnnxRecurrentLayerTraversalContext

Traversal context for one hidden layer during recurrent-input collection.

### OnnxRuntimeFactories

Runtime factories consumed during ONNX import network reconstruction.

### OnnxRuntimeLayerFactory

`(size: number) => import("C:/NeatapticTS/src/architecture/layer").default`

Runtime layer-constructor signature used for recurrent layer reconstruction.

### OnnxRuntimeLayerModule

Runtime layer module shape consumed by ONNX import orchestration.

### OnnxRuntimePerceptronFactory

`(sizes: number[]) => import("C:/NeatapticTS/src/architecture/network").default`

Runtime perceptron factory signature used by ONNX import orchestration.

### OnnxShape

ONNX tensor type shape.

### OnnxTensor

### OnnxTensorType

ONNX tensor type.

### OnnxValueInfo

ONNX value info (input/output description).

### OptimizerConfigBase

Base optimizer configuration.

### OptionalLayerOutputParams

Shared parameters for optional pooling/flatten output emission.

### OptionalPoolingAndFlattenParams

Parameters for optional pooling + flatten emission after a layer output.

### OutgoingOrderBuildContext

Context for constructing source-grouped outgoing connection order.

### OutputLayerActivationContext

Output-layer activation assignment context.

### Parent1GeneTraversalContext

Traversal state for parent-1 innovation walk.

### Parent1TraversalSelectionResult

Fold result for parent-1 traversal selection.

### ParentMetrics

Compact parent metrics summary.

### PathSearchContext

Mutable context used while running iterative path search.

### PerNeuronConcatNodePayload

Per-neuron concat node payload.

### PerNeuronGraphNames

Per-neuron graph tensor names.

### PerNeuronLayerContext

Per-neuron layer context alias.

### PerNeuronLayerParams

Parameters for per-neuron layer emission.

### PerNeuronNodeContext

Per-neuron normalized node context.

### PerNeuronSubgraphContext

Per-neuron subgraph emission context.

### PerNeuronTensorNames

Per-neuron initializer tensor names.

### PlateauSmoothingConfig

Config for plateau smoothing computation.

### PlateauSmoothingState

Mutable smoothing state for plateau metric.

### Pool2DMapping

Mapping describing a pooling operation inserted after a given export-layer index.

### PoolingAttributes

Pooling tensor attributes for ONNX node payloads.

### PoolingEmissionContext

Pooling emission context resolved for one layer output.

### PoolKeyMetrics

Per-pool-key allocation & reuse counters (educational / diagnostics).

### PopulationFitnessFunction

`(population: import("C:/NeatapticTS/src/architecture/network").default[]) => Promise<void>`

Fitness signature evaluating full population asynchronously.

### PopulationWorkerEvaluationContext

Shared context for one population worker evaluation run.

### PrimarySmoothingState

Mutable smoothing state for monitored error.

### PruneSelectionContext

Context for selecting prune candidates.

### PruneSelectionResult

Result of prune candidate selection.

### PruningMethod

Pruning strategy identifiers.

### PublishAdjacencyContext

Context for publishing fully built adjacency slabs to internal network state.

### ReconnectEndpointPairContext

Endpoint pair for reconnecting bridged paths.

### RecurrentActivationEmissionContext

Context for selecting and emitting recurrent activation node payload.

### RecurrentGateBlockCollectionContext

Context for collecting one gate parameter block.

### RecurrentGateParameterCollectionResult

Flattened recurrent gate parameter vectors for one fused operator.

### RecurrentGateRow

One recurrent gate row payload before flatten fold.

### RecurrentGateRowCollectionContext

Context for collecting one recurrent gate row (one neuron).

### RecurrentGemmEmissionContext

Context for emitting one Gemm node for recurrent single-step export.

### RecurrentGraphNames

Derived graph names for one recurrent single-step layer payload.

### RecurrentHeuristicEmissionContext

Context for heuristic recurrent operator emission traversal.

### RecurrentInitializerEmissionContext

Context for pushing recurrent initializers into ONNX graph state.

### RecurrentInitializerNames

Initializer tensor names for one single-step recurrent layer.

### RecurrentInitializerValues

Collected initializer vectors for one single-step recurrent layer.

### RecurrentLayerEmissionContext

Derived execution context for single-step recurrent layer emission.

### RecurrentLayerEmissionParams

Parameters for single-step recurrent layer emission.

### RecurrentLayerShape

Minimal recurrent-layer shape used by mutation expanders.

### RecurrentRowCollectionContext

Context for collecting one recurrent matrix row.

### RegrowthExecutionContext

Context for regrowth execution routine.

### RegrowthPlan

Derived regrowth execution plan.

### RegrowthPlanContext

Context for deriving regrowth plan.

### RegularizationConfig

L1/L2 regularization configuration.

### ResolvedNetworkSizeContext

Resolved input/output sizes for rebuild.

### RNGSnapshot

Snapshot payload for RNG state restore flows.

### ScheduleConfig

Schedule callback configuration.

### ScheduledTargetContext

Context for scheduled-pruning target computation.

### ScheduledTargetResult

Result of scheduled-pruning target computation.

### SerializedConnection

Serialized connection representation.

### SerializedNetwork

Serialized network payload used in checkpoint callbacks.

### SerializeNetworkInternals

Runtime interface for accessing network internals during serialization.

### SerializeNodeInternals

Runtime node internals needed for serialization workflows.

### SharedActivationNodeBuildParams

Shared parameters for constructing an activation node payload.

### SharedGemmNodeBuildParams

Shared parameters for constructing a Gemm node payload.

### SingleGenomeFitnessFunction

`(genome: import("C:/NeatapticTS/src/architecture/network").default) => number`

Fitness signature evaluating one genome.

### SlabBuildContext

Immutable inputs required to build or grow connection slab buffers.

### SlabPopulateResult

Result of scanning and populating optional gain/plastic slab arrays.

### SlabWriteArrays

Writable slab arrays targeted during connection serialization.

### SourcePeerConnectionCountContext

Context for source-to-peer connection counting.

### SpecMetadataAppendContext

Append-a-spec metadata context for JSON-array metadata keys.

### StandaloneGenerationContext

Shared mutable state for standalone source generation.

### StartIndicesBuildContext

Context for constructing CSR start offsets from precomputed fan-out counts.

### StatsNetworkProps

Internal network properties used by stats operations.

### SubNodeMutationConfig

Mutation keep-gates option surface used by sub-node removal logic.

### TargetLayerPeerContext

Context for target-layer peer traversal.

### TopologyBuildContext

Mutable context used while building topological ordering.

### TopologyNetworkProps

Internal topology state carrier.

### TrainingConnectionInternals

Runtime connection view used by training internals.

### TrainingNetworkInternals

Runtime network view used by training internals.

### TrainingNodeInternals

Runtime node view used by training internals.

### TrainingOptions

Public training options shape.

### TrainingSample

A single supervised training sample used in evolution scoring.

### TypedArray

Union of slab typed array element container types.

### TypedArrayConstructor

Constructor type for typed arrays used in slabs.

### WeightSamplingRangeContext

Context for sampling one random weight value.

### WeightToleranceComparisonContext

Context for comparing two scalar weights with numeric tolerance.

### WorkerTraversalContext

Worker-local traversal context.

## architecture/network/network.utils.ts

### __trainingInternals

### activateBatch

`(inputs: number[][], training: boolean) => number[][]`

Activate the network over a mini‑batch (array) of input vectors, returning a 2‑D array of outputs.

This helper simply loops, invoking {@link Network.activate} (or its bound variant) for each
sample. It is intentionally naive: no attempt is made to fuse operations across the batch.
For very large batch sizes or performance‑critical paths consider implementing a custom
vectorized backend that exploits SIMD, GPU kernels, or parallel workers.

Input validation occurs per row to surface the earliest mismatch with a descriptive index.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `inputs` - - Array of input vectors; each must have length == network.input.
- `training` - - Whether each activation should keep training traces.

Returns: 2‑D array: outputs[i] is the activation result for inputs[i].

### activateRaw

`(input: number[], training: boolean, maxActivationDepth: number) => number[]`

Thin semantic alias to the network's main activation path.

At present this simply forwards to {@link Network.activate}. The indirection is useful for:
 - Future differentiation between raw (immediate) activation and a mode that performs reuse /
   staged batching logic.
 - Providing a stable exported symbol for external tooling / instrumentation.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `input` - - Input vector (length == network.input).
- `training` - - Whether to retain training traces / gradients (delegated downstream).
- `maxActivationDepth` - - Guard against runaway recursion / cyclic activation attempts.

Returns: Implementation-defined result of Network.activate (typically an output vector).

### applyGradientClippingImpl

`(net: import("C:/NeatapticTS/src/architecture/network").default, cfg: { mode: "norm" | "percentile" | "layerwiseNorm" | "layerwisePercentile"; maxNorm?: number | undefined; percentile?: number | undefined; }) => void`

### canUseFastSlab

`(training: boolean) => boolean`

Public convenience wrapper exposing fast path eligibility.
Mirrors `_canUseFastSlab` internal predicate.

Parameters:
- `training` - Whether caller is performing training (disables fast path if true).

Returns: True when slab fast path predicates hold.

### computeTopoOrder

`() => void`

Topology utilities.

Provides:
 - computeTopoOrder: Kahn-style topological sorting with graceful fallback when cycles detected.
 - hasPath: depth-first reachability query (used to prevent cycle introduction when acyclicity enforced).

Design Notes:
 - We deliberately tolerate cycles by falling back to raw node ordering instead of throwing; this
   allows callers performing interim structural mutations to proceed (e.g. during evolve phases)
   while signaling that the fast acyclic optimizations should not be used.
 - Input nodes are seeded into the queue immediately regardless of in-degree to keep them early in
   the ordering even if an unusual inbound edge was added (defensive redundancy).
 - Self loops are ignored for in-degree accounting and queue progression (they neither unlock new
   nodes nor should they block ordering completion).

### connect

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Create and register one (or multiple) directed connection objects between two nodes.

Some node types (or future composite structures) may return several low‑level connections when
their {@link Node.connect} is invoked (e.g., expanded recurrent templates). For that reason this
function always treats the result as an array and appends each edge to the appropriate collection.

Algorithm outline:
 1. (Acyclic guard) If acyclicity is enforced and the source node appears after the target node in
    the network's node ordering, abort early and return an empty array (prevents back‑edge creation).
 2. Delegate to sourceNode.connect(targetNode, weight) to build the raw Connection object(s).
 3. For each created connection:
      a. If it's a self‑connection: either ignore (acyclic mode) or store in selfconns.
      b. Otherwise store in standard connections array.
 4. If any connection was added, mark structural caches dirty (_topoDirty & _slabDirty) so lazy
    rebuild can occur before the next forward pass.

Complexity:
 - Time: O(k) where k is the number of low‑level connections returned (typically 1).
 - Space: O(k) new Connection instances (delegated to Node.connect).

Edge cases & invariants:
 - Acyclic mode silently refuses back‑edges instead of throwing (makes evolutionary search easier).
 - Self‑connections are skipped entirely when acyclicity is enforced.
 - Weight initialization policy is delegated to Node.connect if not explicitly provided.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `from` - - Source node (emits signal).
- `to` - - Target node (receives signal).
- `weight` - - Optional explicit initial weight value.

### crossOver

`(parentNetwork1: import("C:/NeatapticTS/src/architecture/network").default, parentNetwork2: import("C:/NeatapticTS/src/architecture/network").default, equal: boolean) => import("C:/NeatapticTS/src/architecture/network").default`

Genetic operator: NEAT‑style crossover (legacy merge operator removed).

This module now focuses solely on producing recombinant offspring via {@link crossOver}.
The previous experimental Network.merge has been removed to reduce maintenance surface area
and avoid implying a misleading “sequential composition” guarantee.

### deserialize

`(data: import("C:/NeatapticTS/src/architecture/network/network.types").CompactSerializedNetworkTuple, inputSize: number | undefined, outputSize: number | undefined) => import("C:/NeatapticTS/src/architecture/network").default`

### disconnect

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default) => void`

Remove (at most) one directed connection from source 'from' to target 'to'.

Only a single direct edge is removed because typical graph configurations maintain at most
one logical connection between a given pair of nodes (excluding potential future multi‑edge
semantics). If the target edge is gated we first call {@link Network.ungate} to maintain
gating invariants (ensuring the gater node's internal gate list remains consistent).

Algorithm outline:
 1. Choose the correct list (selfconns vs connections) based on whether from === to.
 2. Linear scan to find the first edge with matching endpoints.
 3. If gated, ungate to detach gater bookkeeping.
 4. Splice the edge out; exit loop (only one expected).
 5. Delegate per‑node cleanup via from.disconnect(to) (clears reverse references, traces, etc.).
 6. Mark structural caches dirty for lazy recomputation.

Complexity:
 - Time: O(m) where m is length of the searched list (connections or selfconns).
 - Space: O(1) extra.

Idempotence: If no such edge exists we still perform node-level disconnect and flag caches dirty –
this conservative approach simplifies callers (they need not pre‑check existence).

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `from` - - Source node.
- `to` - - Target node.

### evolveNetwork

`(set: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingSample[], options: import("C:/NeatapticTS/src/architecture/network/network.types").EvolveOptions) => Promise<{ error: number; iterations: number; time: number; }>`

Evolve (optimize) the current network's topology and weights using a NEAT-like evolutionary loop
until a stopping criterion (target error or max iterations) is met.

High-level process:
 1. Validate dataset shape (input/output vector sizes must match network I/O counts).
 2. Normalize / default option values and construct an internal configuration summary.
 3. Build appropriate fitness evaluation function (single or multi-thread).
 4. Initialize a Neat population (optionally with speciation) seeded by this network.
 5. Iteratively call neat.evolve():
      - Retrieve fittest genome + its fitness.
      - Derive an error metric from fitness (inverse relationship considering complexity penalty).
      - Track best genome overall (elitism) and perform logging/scheduling callbacks.
      - Break if error criterion satisfied or iterations exceeded.
 6. Replace this network's internal structural arrays with the best discovered genome's (in-place upgrade).
 7. Cleanup any worker threads and report final statistics.

Fitness / Error relationship:
  fitness = -error - complexityPenalty  =>  error = -(fitness - complexityPenalty)
We recompute error from the stored fitness plus penalty to ensure consistent reporting.

Resilience strategies:
 - Guard against infinite / NaN errors; after MAX_INF consecutive invalid errors we abort.
 - Fallback for tiny populations: increase mutation aggressiveness to prevent premature convergence.

Parameters:
- `this` - - Bound  {@link Network} instance being evolved in-place.
 *
- `set` - - Supervised dataset (array of {input, output}).
- `options` - - Evolution options (see README / docs). Key fields include:
- iterations: maximum generations (if omitted must supply error target)
- error: target error threshold (if omitted must supply iterations)
- growth: complexity penalty scaling
- amount: number of score evaluations (averaged) per genome
- threads: desired worker count (>=2 enables multi-thread path if available)
- popsize / populationSize: population size
- schedule: { iterations: number, function: (ctx) => void } periodic callback
- log: generation interval for console logging
- clear: whether to call network.clear() after adopting best genome

Returns: Summary object { error, iterations, time(ms) }.

### fastSlabActivate

`(input: number[]) => number[]`

High‑performance forward pass using packed slabs + CSR adjacency.

Fallback Conditions (auto‑detected):
 - Missing slabs / adjacency structures.
 - Topology/gating/stochastic predicates fail (see `_canUseFastSlab`).
 - Any gating present (explicit guard).

Implementation Notes:
 - Reuses internal activation/state buffers to reduce per‑step allocation churn.
 - Applies gain multiplication if optional gain slab exists.
 - Assumes acyclic graph; topological order recomputed on demand if marked dirty.

Parameters:
- `input` - Input vector (length must equal `network.input`).

Returns: Output activations (detached plain array) of length `network.output`.

### fromJSONImpl

`(json: import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON) => import("C:/NeatapticTS/src/architecture/network").default`

### gate

`(node: import("C:/NeatapticTS/src/architecture/node").default, connection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Attach a gater node to a connection so that the connection's effective weight
becomes dynamically modulated by the gater's activation (see {@link Node.gate} for exact math).

Validation / invariants:
 - Throws if the gater node is not part of this network (prevents cross-network corruption).
 - If the connection is already gated, function is a no-op (emits warning when enabled).

Complexity: O(1)

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `node` - - Candidate gater node (must belong to network).
- `connection` - - Connection to gate.

### generateStandalone

`(net: import("C:/NeatapticTS/src/architecture/network").default) => string`

Generate a standalone JavaScript source string that returns an `activate(input:number[])` function.

Implementation Steps:
 1. Validate presence of output nodes (must produce something observable).
 2. Assign stable sequential indices to nodes (used as array offsets in generated code).
 3. Collect initial activation/state values into typed array initializers for warm starting.
 4. For each non-input node, build a line computing S[i] (pre-activation sum with bias) and A[i]
    (post-activation output). Gating multiplies activation by gate activations; self-connection adds
    recurrent term S[i] * weight before activation.
 5. De-duplicate activation functions: each unique squash name is emitted once; references become
    indices into array F of function references for compactness.
 6. Emit an IIFE producing the activate function with internal arrays A (activations) and S (states).

Parameters:
- `net` - Network instance to snapshot.

Returns: Source string (ES5-compatible) – safe to eval in sandbox to obtain activate function.

### getConnectionSlab

`() => import("C:/NeatapticTS/src/architecture/network/slab/network.slab.utils.types").ConnectionSlabView`

Obtain (and lazily rebuild if dirty) the current packed SoA view of connections.

Gain Omission: If the internal gain slab is absent (all gains neutral) a synthetic
neutral array is created and returned (NOT retained) to keep external educational
tooling branch‑free while preserving omission memory savings internally.

Returns: Read‑only style view (do not mutate) containing typed arrays + metadata.

### getCurrentSparsity

`() => number`

Current sparsity fraction relative to the training-time pruning baseline.

### getRegularizationStats

`() => Record<string, unknown> | null`

Obtain the last recorded regularization / stochastic statistics snapshot.

Returns a defensive deep copy so callers can inspect metrics without risking mutation of the
internal `_lastStats` object maintained by the training loop (e.g., during pruning, dropout, or
noise scheduling updates).

Returns: A deep-cloned stats object or null if no stats have been recorded yet.

### getRNGState

`() => number | undefined`

Get the current internal 32‑bit RNG state value.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Unsigned 32‑bit state integer or undefined if generator not yet seeded or was reset.

### getSlabAllocationStats

`() => { pool: { [x: string]: import("C:/NeatapticTS/src/architecture/network/slab/network.slab.utils.types").PoolKeyMetrics; }; fresh: number; pooled: number; }`

Allocation statistics snapshot for slab typed arrays.

Includes:
 - fresh: number of newly constructed typed arrays since process start / metrics reset.
 - pooled: number of arrays served from the pool.
 - pool: per‑key metrics (created, reused, maxRetained) for educational inspection.

NOTE: Stats are cumulative (not auto‑reset); callers may diff successive snapshots.

Returns: Plain object copy (safe to serialize) of current allocator counters.

### hasPath

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default) => boolean`

Depth-first reachability test (avoids infinite loops via visited set).

### maybePrune

`(iteration: number) => void`

Structured and dynamic pruning utilities for networks.

Features:
 - Scheduled pruning during gradient-based training ({@link maybePrune}) with linear sparsity ramp.
 - Evolutionary generation pruning toward a target sparsity ({@link pruneToSparsity}).
 - Two ranking heuristics:
     magnitude: |w|
     snip: |w * g| approximation (g approximated via accumulated delta stats; falls back to |w|)
 - Optional stochastic regrowth during scheduled pruning (dynamic sparse training), preserving acyclic constraints.

Internal State Fields (attached to Network via `any` casting):
 - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 - _rand: deterministic RNG function
 - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting

### mutateImpl

`(method: import("C:/NeatapticTS/src/architecture/network/network.types").MutationMethod | undefined) => void`

Public entry point: apply a single mutation operator to the network.

Parameters:
- `this` - - Network instance.
- `method` - - Mutation enum value or descriptor object.

Returns: Nothing.

### noTraceActivate

`(input: number[]) => number[]`

Perform a forward pass without creating or updating any training / gradient traces.

This is the most allocation‑sensitive activation path. Internally it will attempt
to leverage a compact "fast slab" routine (an optimized, vectorized broadcast over
contiguous activation buffers) when the Network instance indicates that such a path
is currently valid. If that attempt fails (for instance because the slab is stale
after a structural mutation) execution gracefully falls back to a node‑by‑node loop.

Algorithm outline:
 1. (Optional) Refresh cached topological order if the network enforces acyclicity
    and a structural change marked the order as dirty.
 2. Validate the input dimensionality.
 3. Try the fast slab path; if it throws, continue with the standard path.
 4. Acquire a pooled output buffer sized to the number of output neurons.
 5. Iterate all nodes in their internal order:
      - Input nodes: directly assign provided input values.
      - Hidden nodes: compute activation via Node.noTraceActivate (no bookkeeping).
      - Output nodes: compute activation and store it (in sequence) inside the
        pooled output buffer.
 6. Copy the pooled buffer into a fresh array (detaches user from the pool) and
    release the pooled buffer back to the pool.

Complexity considerations:
 - Time: O(N + E) where N = number of nodes, E = number of inbound edges processed
   inside each Node.noTraceActivate call (not explicit here but inside the node).
 - Space: O(O) transient (O = number of outputs) due to the pooled output buffer.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `input` - - Flat numeric vector whose length must equal network.input.

Returns: Array of output neuron activations (length == network.output).

### pruneToSparsity

`(targetSparsity: number, method: import("C:/NeatapticTS/src/architecture/network/network.types").PruningMethod) => void`

Evolutionary (generation-based) pruning toward a target sparsity baseline.
Unlike maybePrune this operates immediately relative to the first invocation's connection count
(stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.

### rebuildConnectionSlab

`(force: boolean) => void`

Build (or refresh) the packed connection slabs for the network synchronously.

ACTIONS
-------
1. Optionally reindex nodes if structural mutations invalidated indices.
2. Grow (geometric) or reuse existing typed arrays to ensure capacity >= active connections.
3. Populate the logical slice [0, connectionCount) with weight/from/to/flag data.
4. Lazily allocate gain & plastic slabs only on first non‑neutral / plastic encounter; omit otherwise.
5. Release previously allocated optional slabs when they revert to neutral / unused (omission optimization).
6. Update internal bookkeeping: logical count, dirty flags, version counter.

PERFORMANCE
-----------
O(C) over active connections with amortized allocation cost due to geometric growth.

Parameters:
- `force` - When true forces rebuild even if network not marked dirty (useful for timing tests).

### rebuildConnectionSlabAsync

`(chunkSize: number) => Promise<void>`

Cooperative asynchronous slab rebuild (Browser only).

Strategy:
 - Perform capacity decision + allocation up front (mirrors sync path).
 - Populate connection data in microtask slices (yield via resolved Promise) to avoid long main‑thread stalls.
 - Adaptive slice sizing for very large graphs if `config.browserSlabChunkTargetMs` set.

Metrics: Increments `_slabAsyncBuilds` for observability.
Fallback: On Node (no `window`) defers to synchronous rebuild for simplicity.

Parameters:
- `chunkSize` - Initial maximum connections per slice (may be reduced adaptively for huge graphs).

Returns: Promise resolving once rebuild completes.

### removeNode

`(node: import("C:/NeatapticTS/src/architecture/node").default) => void`

Node removal utilities.

This module provides a focused implementation for removing a single hidden node from a network
while attempting to preserve overall functional connectivity. The removal procedure mirrors the
legacy Neataptic logic but augments it with clearer documentation and explicit invariants.

High‑level algorithm (removeNode):
 1. Guard: ensure the node exists and is not an input or output (those are structural anchors).
 2. Ungate: detach any connections gated BY the node (we don't currently reassign gater roles).
 3. Snapshot inbound / outbound connections (before mutation of adjacency lists).
 4. Disconnect all inbound, outbound, and self connections.
 5. Physically remove the node from the network's node array.
 6. Simple path repair heuristic: for every former inbound source and outbound target, add a
    direct connection if (a) both endpoints still exist, (b) they are distinct, and (c) no
    direct connection already exists. This keeps forward information flow possibilities.
 7. Mark topology / caches dirty so that subsequent activation / ordering passes rebuild state.

Notes / Limitations:
 - We do NOT attempt to clone weights or distribute the removed node's function across new
   connections (more sophisticated strategies could average or compose weights).
 - Gating effects involving the removed node as a gater are dropped; downstream behavior may
   change—callers relying heavily on gating may want a custom remap strategy.
 - Self connections are simply removed; no attempt is made to emulate recursion via alternative
   structures.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `node` - - Hidden node to remove.

### restoreRNG

`(fn: () => number) => void`

Restore a previously captured RNG function implementation (advanced usage).

This does NOT rehydrate _rngState (it explicitly sets it to undefined). Intended for scenarios
where a caller has customly serialized a full RNG closure or wants to inject a deterministic stub.
If you only need to restore the raw state word produced by {@link snapshotRNG}, prefer
{@link setRNGState} instead.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `fn` - - Function returning a pseudo‑random number in [0,1). Caller guarantees determinism if required.

### serialize

`() => import("C:/NeatapticTS/src/architecture/network/network.types").CompactSerializedNetworkTuple`

Instance-level lightweight serializer used primarily for fast inter-thread transfer.

Parameters:
- `this` - - Bound network instance.

Returns: Compact serialized tuple payload.

### setRNGState

`(state: number) => void`

Explicitly set (override) the internal 32‑bit RNG state without changing the generator function.

This is a low‑level operation; typical clients should call {@link setSeed}. Provided for advanced
replay functionality where the same PRNG algorithm is assumed but you want to resume exactly at a
known state word.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `state` - - Any finite number (only low 32 bits used). Ignored if not numeric.

### setSeed

`(seed: number) => void`

Deterministic pseudo‑random number generation (PRNG) utilities for {@link Network}.

Why this module exists:
 - Facilitates reproducible evolutionary runs / gradient training by allowing explicit seeding.
 - Centralizes RNG state management & snapshot/restore operations (useful for rollbacks or
   deterministic tests around mutation sequences).
 - Keeps the core Network class focused by extracting ancillary RNG concerns.

Implementation notes:
 - Uses a small, fast 32‑bit xorshift / mix style generator (same semantics as the legacy inline version)
   combining an additive Weyl sequence step plus a few avalanche-style integer mixes.
 - Not cryptographically secure. Do not use for security / fairness sensitive applications.
 - Produces floating point numbers in [0,1) with 2^32 (~4.29e9) discrete possible mantissa states.

Public surface:
 - {@link setSeed}: Initialize deterministic generator with a numeric seed.
 - {@link snapshotRNG}: Capture current training step + raw internal RNG state.
 - {@link restoreRNG}: Provide an externally saved RNG function (advanced) & clear stored state.
 - {@link getRNGState} / {@link setRNGState}: Low-level accessors for the internal 32‑bit state word.
 - {@link getRandomFn}: Retrieve the active random() function reference (primarily for tests / tooling).

Design rationale:
 - Storing both a state integer (_rngState) and a function (_rand) allows hot-swapping alternative
   RNG implementations (e.g., for benchmarking or pluggable randomness strategies) without rewriting
   callsites inside Network algorithms.

### snapshotRNG

`() => import("C:/NeatapticTS/src/architecture/network/network.types").RNGSnapshot`

Capture a snapshot of the RNG state together with the network's training step.

Useful for implementing speculative evolutionary mutations where you may revert both the
structural change and the randomness timeline if accepting/rejecting a candidate.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Object containing current training step & 32‑bit RNG state (both possibly undefined if unseeded).

### toJSONImpl

`() => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkJSON`

Verbose JSON export (stable formatVersion).

Parameters:
- `this` - - Bound network instance.

Returns: Verbose structural JSON payload.

### trainImpl

`(net: import("C:/NeatapticTS/src/architecture/network").default, set: { input: number[]; output: number[]; }[], options: import("C:/NeatapticTS/src/architecture/network/network.types").TrainingOptions) => { error: number; iterations: number; time: number; }`

### ungate

`(connection: import("C:/NeatapticTS/src/architecture/connection").default) => void`

Remove gating from a connection, restoring its static weight contribution.

Idempotent: If the connection is not currently gated, the call performs no structural changes
(and optionally logs a warning). After ungating, the connection's weight will be used directly
without modulation by a gater activation.

Complexity: O(n) where n = number of gated connections (indexOf lookup) – typically small.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `connection` - - Connection to ungate.
