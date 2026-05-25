/**
 * NeatapticTS ONNX-like serialization for networks.
 *
 * This module provides the four public entry points:
 * - `exportToONNXBinary()` turns the approved constrained subset into protobuf
 *   `ModelProto` bytes and is the primary runtime-validated artifact for the
 *   current Phase 8 and Phase 9 subset.
 * - `exportToONNX()` turns a runtime `Network` into a plain JSON object
 *   (`OnnxModel`) for same-family debug, roundtrip, and importer-owned workflows.
 * - `importFromONNX()` reconstructs a `Network` from that JSON object.
 * - `importFromONNXBinary()` reconstructs the first honest external binary ONNX
 *   subset through a separate standard-domain `ModelProto` ingress lane.
 *
 * What this format is (and is not):
 * - It exposes two distinct ONNX-adjacent surfaces with different promises.
 * - `exportToONNXBinary()` is the compliance and runtime-evidence surface for the
 *   documented supported subset.
 * - `exportToONNX()` stays **JSON-first** and intentionally resembles ONNX’s
 *   model and graph concepts for same-family inspection and roundtrip import.
 * - Neither surface is a blanket promise of arbitrary ONNX runtime portability
 *   beyond the named subset and validations.
 *
 * How to read this chapter:
 * - Start here for the public surface split and the trust boundary.
 * - Continue into `export/` to see how layered networks become JSON graph payloads.
 * - Continue into `import/` to see how that payload becomes a runtime network again.
 * - Continue into `schema/` for the persisted wire-format shapes.
 * - Continue into `parity/` for the Phase 9 runtime-executed fixture inventory and
 *   binary-first ONNX Runtime comparison seam.
 * - Use `network.onnx.utils.ts` and `network.onnx.utils.types.ts` as compatibility and
 *   bridge surfaces rather than the first place to learn the pipeline.
 *
 * Why the folder is split this way:
 * - The root file keeps the stable entry points and the promise of the format.
 * - The `export/` and `import/` chapters carry the heavier execution details.
 * - The `schema/` chapter keeps the persisted document model separate from runtime logic.
 * - The root utility barrels exist so public ergonomics stay stable while the implementation
 *   can keep moving toward smaller, teachable chapters.
 *
 * Trust boundary:
 * - Treat imported models as **untrusted input**. The importer validates structure, but
 *   you should still apply the same care you would for a generic JSON payload.
 *
 * Example (export → persist → import):
 *
 * ```ts
 * import { exportToONNX, importFromONNX } from './network.onnx';
 *
 * const model = exportToONNX(network, { includeMetadata: true });
 * const jsonText = JSON.stringify(model);
 *
 * const modelRoundTrip = JSON.parse(jsonText);
 * const restored = importFromONNX(modelRoundTrip);
 * ```
 */

import type Network from '../../network/network';
import {
  applyModelMetadata,
  runOnnxExportFlow,
  runOnnxImportFlow,
} from './network.onnx.utils';
import { runExternalOnnxImportFlow } from './import/network.onnx.import-flow.utils';
import type {
  AttentionMapping,
  ConcatMapping,
  OnnxExportOptions,
} from './network.onnx.utils.types';
import type {
  Conv2DMapping,
  OnnxDimension,
  OnnxModel,
  OnnxValueInfo,
  Pool2DMapping,
} from './schema/network.onnx.schema.types';
import { serializeOnnxModelToBinary } from './schema/network.onnx.schema.binary.utils';

export type {
  AttentionMapping,
  ConcatMapping,
  Conv2DMapping,
  OnnxExportOptions,
  OnnxModel,
  Pool2DMapping,
};

const DEFAULT_ONNX_BINARY_PRODUCER_NAME = 'neataptic-ts';
const SYMBOLIC_BINARY_BATCH_DIMENSION_NAME = 'N';
const BINARY_SPATIAL_LAYER_OUTPUT_PATTERN = /^Layer_(\d+)$/;
const BINARY_DENSE_CONSUMER_OPERATIONS = new Set([
  'DynamicQuantizeLinear',
  'Gemm',
  'QuantizeLinear',
]);

/**
 * Export a NeatapticTS network to an ONNX-like **JSON object** (`OnnxModel`).
 *
 * What you get:
 * - A plain object that you can persist with `JSON.stringify()`.
 * - A minimal ONNX-ish graph (`model.graph`) plus optional metadata (`model.metadata_props`).
 *
 * When to use this:
 * - You want a portable same-family snapshot that can be inspected or diffed as JSON.
 * - You want the importer-owned roundtrip surface consumed later via `importFromONNX()`.
 * - You are debugging export structure rather than producing the primary
 *   runtime-validated `.onnx` artifact.
 *
 * Tradeoffs:
 * - The output is ONNX-like, but **not** intended to be universally compatible with all ONNX
 *   runtimes.
 * - This is not the primary runtime-validated artifact for the approved subset;
 *   use `exportToONNXBinary()` when the goal is external validation, parity,
 *   or the main compliant deliverable.
 * - Some advanced features (partial connectivity, mixed activations, recurrent heuristics)
 *   may produce graphs that are primarily meant for this library’s importer.
 * - Phase 6 closes the first optimization wave conservatively: exact unary activation
 *   emission now covers `Softplus`, `Softsign`, `Selu`, `Mish` (opset >= 18), and
 *   `Gelu` with explicit `approximate='tanh'` (opset >= 20), while opset-incompatible
 *   or unsupported activations stay on the honest Identity baseline.
 * - Phase 7 is now active on the first reduced-precision lane: `precision.mode =
 *   'storage-fp16'` packs eligible same-family dense and Conv weight or bias
 *   initializers as float16 payloads and prepends deterministic `Cast -> float32`
 *   bridges so `Gemm` and `Conv` inputs stay type-consistent. Recurrent,
 *   advanced-graph, mixed-activation, and partial-connectivity requests stay on
 *   float32 with explicit metadata fallback reasons instead of silently widening
 *   the supported subset.
 * - Quantization packets are now exporter-owned and calibration-backed. Static
 *   8-bit requests can carry explicit layer-target calibration ranges and emit
 *   deterministic scale or zero-point initializers plus metadata for the
 *   supported same-family dense and spatial subset. The dense-only Phase 7D
 *   lane is now closed for the current explicitly targeted same-family
 *   one-output dense subset: those layers can lower into
 *   `QuantizeLinear -> QLinearMatMul -> DequantizeLinear`, reattach nonzero
 *   bias through an explicit float-domain `Add` bridge, preserve the
 *   exporter-owned unary activation node, and emit a deterministic quantized
 *   weight tensor with `effective_quantization_mode = static-8bit`. Phase 7E
 *   is now also closed for the current explicit Conv subset: supported spatial
 *   paths lower into `QuantizeLinear -> QLinearConv -> DequantizeLinear`, emit
 *   deterministic quantized Conv weight tensors, quantize the fused bias as
 *   one `int32` value per output channel, and return to float32 before pooling,
 *   flatten, reshape, or downstream dense boundaries. Phase 7F is now closed
 *   for the current dense-only dynamic guidance lane: supported same-family
 *   dense paths can either land `metadata-only` guidance or insert
 *   `DynamicQuantizeLinear -> DequantizeLinear` ahead of dense `Gemm` inputs
 *   while keeping the affine and activation compute on the existing float32
 *   path. Wider dense targets, unsupported spatial fallbacks, recurrent,
 *   advanced-graph, mixed-activation, and partial-connectivity requests still
 *   stay on float32 with explicit fallback metadata instead of widening the
 *   supported subset implicitly.
 *
 * Quantized spatial lowering path:
 *
 * ```mermaid
 * flowchart LR
 *   Input[Float spatial input] --> Quantize[QuantizeLinear]
 *   Quantize --> QConv[QLinearConv]
 *   QConv --> Dequantize[DequantizeLinear]
 *   Dequantize --> Activation[Unary activation]
 *   Activation --> Boundary[Pool flatten reshape dense]
 * ```
 *
 * Dynamic dense guidance path:
 *
 * ```mermaid
 * flowchart LR
 *   Input[Float dense input] --> DQL[DynamicQuantizeLinear]
 *   DQL --> DQ[DequantizeLinear]
 *   DQ --> Gemm[Gemm]
 *   Gemm --> Activation[Unary activation]
 * ```
 *
 * The lighter `metadata-only` representation records the same dense-only lane
 * without inserting `DynamicQuantizeLinear` nodes.
 * - The current spatial subset is still conservative: explicit Conv mappings round-trip,
 *   pooling/flatten import remains metadata-driven, and heuristic Conv inference stays
 *   metadata-only unless `autoPromoteInferredConv` is enabled and the inferred dense layer
 *   passes the shared-kernel safety gate for the current proven subset, including
 *   conservative multi-channel layouts, unpooled stacked Conv-like chains, deeper
 *   single-channel post-pool chains whose pooled tensor shape can be derived
 *   sequentially, and deeper pooled multi-channel chains whose pooled tensor shapes
 *   can be derived sequentially while export keeps the pooled source compact per
 *   channel. The only proven flatten-after-pool promotion path is the narrow final
 *   hidden-stage reshape-bridge subset, where export restores the derived pooled
 *   `[C,H,W]` shape before the later Conv. Earlier flattened pooled consumers,
 *   repeated flatten-bridge chains, or downstream dense layers that still depend on
 *   extra non-pooled inputs keep later inferred stages on the honest fallback path.
 * - Export now validates a conservative internal tensor-shape ledger before model
 *   finalization and prunes exporter-owned Identity activation scaffolding only when the
 *   graph stays semantically equivalent for the already-supported same-family subset.
 *
 * High-level algorithm:
 *  1) Normalize/rebuild local connection state for deterministic traversal.
 *  2) Infer an ordered layer view and validate export constraints.
 *  3) Materialize graph nodes/tensors and (optionally) attach metadata.
 *
 * Example (export → JSON text):
 *
 * ```ts
 * const model = exportToONNX(network, { includeMetadata: true });
 * const jsonText = JSON.stringify(model);
 * ```
 *
 * @param network Source network instance to serialize.
 * @param options Export controls (validation strictness and metadata behavior).
 * @returns ONNX-like model object suitable for persistence or re-import.
 * @throws If the network cannot be represented safely under the selected options.
 */
export function exportToONNX(
  network: Network,
  options: OnnxExportOptions = {},
): OnnxModel {
  // Step 1: Delegate complete export orchestration to the centralized flow helper.
  return runOnnxExportFlow(network, options);
}

/**
 * Export a NeatapticTS network to protobuf `ModelProto` bytes.
 *
 * What you get:
 * - A deterministic binary payload whose field layout follows ONNX `ModelProto`.
 * - The primary compliant and runtime-validated artifact for the approved
 *   lower-opset same-family subset.
 * - The same constrained model family as `exportToONNX()`, without widening the
 *   importer or runtime-compatibility promise beyond the documented subset.
 *
 * Important boundary:
 * - This binary surface now has a repo-owned external validation lane for the
 *   current same-family subset: the companion validator decodes and verifies the
 *   `ModelProto` payload and asks ONNX Runtime to accept it under the current
 *   explicit lower-opset contract.
 * - Phase 9 now layers a separate `parity/` seam on top of that load-acceptance
 *   boundary. The parity harness keeps binary `.onnx` as the input artifact,
 *   freezes deterministic baseline float32, storage-fp16, static-8bit dense
 *   qlinear, explicit-Conv static-8bit, and `DynamicQuantizeLinear`
 *   dense-guidance golden fixtures, and now also proves seeded randomized
 *   parity over bounded shapes or input ranges for that same five-lane subset
 *   through an isolated child Node process backed by the raw ONNX Runtime
 *   binding, while keeping the dynamic lane narrow to its explicit
 *   float-to-uint8 scalar-parameter tolerance packet.
 * - That validation lane is still narrower than Phase 9 runtime parity. It does
 *   not widen import support, arbitrary external-consumer support, or cross-opset
 *   execution claims beyond the documented Phase 8 subset.
 *
 * When to use this:
 * - You want the primary `.onnx` artifact for the repo's current compliance and
 *   runtime-evidence claims.
 * - You need the binary input used by the validator and the Phase 9 parity seam.
 * - You want to persist the supported subset in a format that real ONNX tooling
 *   can decode under the documented lower-opset policy.
 *
 * High-level behavior:
 *  1) Build the shared exporter model view for the supported subset.
 *  2) Normalize binary-only graph boundaries and required `ModelProto` headers.
 *  3) Serialize the resulting model into deterministic protobuf bytes.
 *  4) Keep validation separate: use the validator seam when you need explicit
 *     external acceptance evidence for the current supported subset.
 *
 * @param network Source network instance to serialize.
 * @param options Export controls shared with `exportToONNX()`.
 * @returns Binary protobuf `ModelProto` bytes.
 * @throws If the network cannot be represented safely under the selected options.
 */
export function exportToONNXBinary(
  network: Network,
  options: OnnxExportOptions = {},
): Uint8Array {
  // Step 1: Build the shared exporter model view for the supported subset.
  const exportedModel = runOnnxExportFlow(network, options);
  const binaryReadyModel = structuredClone(exportedModel);

  // Step 2: Apply required ModelProto headers without widening metadata_props defaults.
  normalizeBinaryBoundaryTensorRanks(binaryReadyModel, options);

  normalizeBinarySpatialToDenseBoundaries(binaryReadyModel, options);
  normalizeBinaryDeclaredOutputAliases(binaryReadyModel);

  // Step 3: Apply required ModelProto headers without widening metadata_props defaults.
  applyModelMetadata({
    model: binaryReadyModel,
    includeMetadata: true,
    producerName: options.producerName ?? DEFAULT_ONNX_BINARY_PRODUCER_NAME,
    producerVersion: options.producerVersion,
    docString: options.docString,
    opset: options.opset ?? 18,
  });

  // Step 4: Serialize the normalized payload into protobuf bytes.
  return serializeOnnxModelToBinary(binaryReadyModel);
}

function normalizeBinaryBoundaryTensorRanks(
  binaryReadyModel: OnnxModel,
  options: OnnxExportOptions,
): void {
  const firstConvMapping = options.conv2dMappings?.find(
    (convMapping) => convMapping.layerIndex === 1,
  );

  binaryReadyModel.graph.inputs = binaryReadyModel.graph.inputs.map(
    (graphInput, inputIndex) =>
      inputIndex === 0 && firstConvMapping
        ? createBinaryConvInputValueInfo(graphInput, firstConvMapping)
        : prependBinaryBatchDimension(graphInput),
  );
  binaryReadyModel.graph.outputs = binaryReadyModel.graph.outputs.map(
    prependBinaryBatchDimension,
  );
}

function normalizeBinarySpatialToDenseBoundaries(
  binaryReadyModel: OnnxModel,
  options: OnnxExportOptions,
): void {
  const spatialLayerIndices = new Set(
    (options.conv2dMappings ?? []).map((convMapping) => convMapping.layerIndex),
  );

  if (spatialLayerIndices.size === 0) {
    return;
  }

  const emittedFlattenOutputNamesByLayerIndex = new Map<number, string>();

  binaryReadyModel.graph.node = binaryReadyModel.graph.node.flatMap(
    (graphNode) => {
      const requiredFlattenLayerIndices = graphNode.input.flatMap(
        (inputName) => {
          const referencedSpatialLayerIndex =
            resolveReferencedSpatialLayerIndex(inputName, spatialLayerIndices);

          if (
            referencedSpatialLayerIndex === undefined ||
            !BINARY_DENSE_CONSUMER_OPERATIONS.has(graphNode.op_type)
          ) {
            return [];
          }

          return [referencedSpatialLayerIndex];
        },
      );
      const missingFlattenLayerIndices = requiredFlattenLayerIndices.filter(
        (layerIndex) => !emittedFlattenOutputNamesByLayerIndex.has(layerIndex),
      );
      const flattenNodes = missingFlattenLayerIndices.map((layerIndex) => {
        const flattenOutputName = `BinaryFlatten_l${layerIndex}`;
        emittedFlattenOutputNamesByLayerIndex.set(
          layerIndex,
          flattenOutputName,
        );

        return {
          op_type: 'Flatten',
          input: [`Layer_${layerIndex}`],
          output: [flattenOutputName],
          name: `binary_flatten_l${layerIndex}`,
          attributes: [{ name: 'axis', type: 'INT', i: 1 }],
        };
      });

      return [
        ...flattenNodes,
        {
          ...graphNode,
          input: graphNode.input.map((inputName) => {
            const referencedSpatialLayerIndex =
              resolveReferencedSpatialLayerIndex(
                inputName,
                spatialLayerIndices,
              );

            if (
              referencedSpatialLayerIndex === undefined ||
              !BINARY_DENSE_CONSUMER_OPERATIONS.has(graphNode.op_type)
            ) {
              return inputName;
            }

            return emittedFlattenOutputNamesByLayerIndex.get(
              referencedSpatialLayerIndex,
            )!;
          }),
        },
      ];
    },
  );
}

function normalizeBinaryDeclaredOutputAliases(
  binaryReadyModel: OnnxModel,
): void {
  const terminalOutputName = [
    resolveBinaryTerminalOutputName(binaryReadyModel),
    binaryReadyModel.graph.outputs[0]?.name,
    '',
  ].find((tensorName) => tensorName !== undefined)!;

  const materializedTensorNames = new Set<string>([
    ...binaryReadyModel.graph.inputs.map((graphInput) => graphInput.name),
    ...binaryReadyModel.graph.initializer.map(
      (initializerEntry) => initializerEntry.name,
    ),
    ...binaryReadyModel.graph.node.flatMap((graphNode) => graphNode.output),
  ]);

  const missingOutputAliases = binaryReadyModel.graph.outputs
    .filter(
      (graphOutput) =>
        !materializedTensorNames.has(graphOutput.name) &&
        graphOutput.name !== terminalOutputName,
    )
    .map((graphOutput, outputIndex) => {
      materializedTensorNames.add(graphOutput.name);
      return {
        op_type: 'Identity',
        input: [terminalOutputName],
        output: [graphOutput.name],
        name: `binary_output_alias_${outputIndex}`,
      };
    });

  binaryReadyModel.graph.node.push(...missingOutputAliases);
}

function createBinaryConvInputValueInfo(
  onnxValueInfo: OnnxValueInfo,
  convMapping: Conv2DMapping,
): OnnxValueInfo {
  return {
    ...onnxValueInfo,
    type: {
      tensor_type: {
        ...onnxValueInfo.type.tensor_type,
        shape: {
          dim: [
            { dim_param: SYMBOLIC_BINARY_BATCH_DIMENSION_NAME },
            { dim_value: convMapping.inChannels },
            { dim_value: convMapping.inHeight },
            { dim_value: convMapping.inWidth },
          ],
        },
      },
    },
  };
}

function prependBinaryBatchDimension(
  onnxValueInfo: OnnxValueInfo,
): OnnxValueInfo {
  const existingDimensions = onnxValueInfo.type.tensor_type.shape.dim;

  if (isLeadingBinaryBatchDimension(existingDimensions[0])) {
    return onnxValueInfo;
  }

  return {
    ...onnxValueInfo,
    type: {
      tensor_type: {
        ...onnxValueInfo.type.tensor_type,
        shape: {
          dim: [
            { dim_param: SYMBOLIC_BINARY_BATCH_DIMENSION_NAME },
            ...existingDimensions,
          ],
        },
      },
    },
  };
}

function isLeadingBinaryBatchDimension(
  leadingDimension: OnnxDimension | undefined,
): boolean {
  return leadingDimension?.dim_param === SYMBOLIC_BINARY_BATCH_DIMENSION_NAME;
}

function resolveReferencedSpatialLayerIndex(
  inputName: string,
  spatialLayerIndices: Set<number>,
): number | undefined {
  const spatialLayerMatch = inputName.match(
    BINARY_SPATIAL_LAYER_OUTPUT_PATTERN,
  );
  if (!spatialLayerMatch) {
    return undefined;
  }

  const resolvedLayerIndex = Number.parseInt(spatialLayerMatch[1]!, 10);
  return Array.from(spatialLayerIndices).find(
    (layerIndex) => layerIndex === resolvedLayerIndex,
  );
}

function resolveBinaryTerminalOutputName(
  binaryReadyModel: OnnxModel,
): string | undefined {
  return [
    binaryReadyModel.graph.node.at(-1)?.output.at(-1),
    binaryReadyModel.graph.inputs[0]?.name,
  ].find((tensorName): tensorName is string => tensorName !== undefined);
}

/**
 * Reconstruct a NeatapticTS network from an exported `OnnxModel`.
 *
 * Expected input:
 * - A model produced by `exportToONNX()` (same repo/version family), including the
 *   current storage-fp16 subset where eligible weight and bias initializers are
 *   packed as float16 payloads and decoded back into the native runtime during import.
 * - Quantized Phase 7 exports remain export-only for now. The importer does not
 *   yet reconstruct `QLinearMatMul`, `QLinearConv`, `DynamicQuantizeLinear`
 *   guidance boundaries, or other quantized operators back into the native
 *   runtime, so quantized ONNX payloads are outside the current import
 *   contract.
 *
 * Trust boundary:
 * - Do not import untrusted blobs. A malformed model can be extremely large or internally
 *   inconsistent and may cause errors or high memory usage.
 *
 * Current boundary:
 * - `importFromONNX()` continues to consume the repo's JSON-first `OnnxModel` surface.
 * - Binary external import now lands through the separate `importFromONNXBinary()` entrypoint
 *   for the first standard-domain float32 dense `Gemm -> unary activation` lane only.
 * - `importFromONNX()` itself stays scoped to the JSON-first `OnnxModel` surface.
 *
 * High-level behavior:
 *  1) Build a perceptron-shaped scaffold from the payload layer sizes.
 *  2) Assign weights/biases and activation functions.
 *  3) Re-apply recurrent and pooling metadata when present.
 *
 * Example (JSON text → restore):
 *
 * ```ts
 * const model = JSON.parse(jsonText) as OnnxModel;
 * const restored = importFromONNX(model);
 * const output = restored.activate([0.1, 0.9]);
 * ```
 *
 * @param onnx ONNX-like model to reconstruct.
 * @returns Reconstructed network ready for inference/evolution workflows.
 * @throws If the model schema is incompatible or cannot be reconstructed safely.
 */
export function importFromONNX(onnx: OnnxModel): Network {
  // Step 1: Delegate complete import orchestration to the centralized flow helper.
  return runOnnxImportFlow(onnx);
}

/**
 * Import the first supported external binary ONNX subset from protobuf `ModelProto` bytes.
 *
 * Current subset:
 * - Exactly one public input, one public output, and one acyclic dense chain.
 * - Standard-domain `Gemm` plus zero-or-one trailing unary activation per layer.
 * - One final standard-domain `Identity` output alias is accepted when it only republishes
 *   the terminal dense tensor as the declared model output.
 * - Float32 tensors only, canonical `Gemm` attributes only, and initializer-owned affine terms only.
 * - No recurrent, spatial, branching, quantized, custom-domain, or metadata-dependent graphs.
 *
 * High-level behavior:
 *  1) Decode and verify the binary `ModelProto` payload.
 *  2) Normalize one accepted external dense chain into an importer-owned canonical model.
 *  3) Reuse the existing reconstruction flow to rebuild a runtime network.
 *
 * @param binaryModel Binary `ModelProto` payload to reconstruct.
 * @returns Reconstructed network ready for inference workflows.
 * @throws If the binary payload falls outside the first supported external import lane.
 */
export function importFromONNXBinary(binaryModel: Uint8Array): Network {
  // Step 1: Delegate the external binary lane to the dedicated normalization + import flow.
  return runExternalOnnxImportFlow(binaryModel);
}

/**
 * ONNX surfaces for NeatapticTS networks.
 *
 * NeatapticTS exposes two ONNX-adjacent surfaces for a constrained subset of
 * `Network` instances.
 *
 * - `exportToONNXBinary()` is the primary runtime-validated and compliance-grade
 *   artifact for the currently approved subset.
 * - `exportToONNX()` remains the JSON-first debug, same-family roundtrip, and
 *   importer-owned interchange surface.
 *
 * This is primarily meant for:
 * - Saving a trained network snapshot in a portable representation.
 * - Debugging / inspecting network structure as a graph of tensors and nodes.
 * - Interop with tooling that can consume graph-shaped metadata.
 *
 * Formats that exist
 * ------------------
 * In this folder, the public surfaces are deliberately split by job:
 *
 * 1) Serialized binary `ModelProto`
 *    - Persist via `exportToONNXBinary()`.
 *    - This is the primary runtime-validated and compliance-oriented artifact
 *      for the current approved subset.
 *
 * 2) In-memory model object (`OnnxModel`)
 *    - A plain JavaScript object shaped like an ONNX `ModelProto`, but represented as JSON.
 *    - This is what `exportToONNX()` returns and what `importFromONNX()` consumes.
 *
 * 3) Serialized JSON text
 *    - Persist via `JSON.stringify(model)`.
 *    - Human-readable and diffable for the same-family roundtrip surface.
 *
 * The binary surface is now the primary artifact for runtime and compliance
 * evidence on the approved subset. The JSON surface stays stable for importer,
 * debug, and same-family roundtrip workflows.
 *
 * Guarantees & stability
 * ----------------------
 * - Runtime and compliance intent: named external validation and Phase 9 parity
 *   evidence attach to `exportToONNXBinary()` for the approved subset only.
 * - Round-trip intent: `importFromONNX(exportToONNX(network))` aims to reconstruct a
 *   functionally equivalent network for the supported subset.
 * - Compatibility today: the importer is only guaranteed to accept models produced by this
 *   repo’s exporter, the same-family heuristic lanes documented below, and the first
 *   explicit external binary subset accepted by `importFromONNXBinary()`.
 * - External binary boundary: the first standard-domain float32 dense `Gemm -> unary activation`
 *   lane is now implemented behind `importFromONNXBinary()` without widening `importFromONNX()`.
 * - Determinism: export is designed to be deterministic given the same network state and
 *   export options.
 *
 * First external import lane (Phase 9F, current implementation)
 * -------------------------------------------------------------
 *
 * The current external binary lane reconstructs the first explicit subset whose
 * execution semantics come from standard ONNX structure and typed tensors rather
 * than repo-only metadata.
 *
 * | Contract area | Selected rule |
 * | --- | --- |
 * | Ingress boundary | Decode binary `ModelProto` bytes with `onnx-proto`, normalize one accepted graph into an importer-owned canonical dense-chain view, and keep `importFromONNX()` as the JSON-first same-family surface. |
 * | Graph topology | Exactly one graph, one public input, one public output, one acyclic linear path, and no branch or merge topology. |
 * | Domain and opset | Standard domain only with exactly one standard-domain opset import. The first external lane keeps the repo's lower-opset floor at 18, while `Gelu` stays gated to opset >= 20. |
 * | Allowed nodes | `Gemm` plus zero-or-one trailing unary activation per layer: `Identity`, `Relu`, `Sigmoid`, `Tanh`, `Softplus`, `Softsign`, `Selu`, `Mish`, and `Gelu` under the declared opset floor. |
 * | Declared output alias | One final standard-domain `Identity` node may republish the terminal dense tensor as the public output name when it carries no extra semantics. |
 * | Accepted `Gemm` form | Input activation tensor first, weight initializer second, optional bias initializer third, and attributes limited to omitted defaults or the exact affine form `alpha=1`, `beta=1`, `transA=0`, `transB=1`. |
 * | Tensor rules | User-visible input/output tensors and any supporting value-info tensors stay rank 2 with a concrete positive feature width. The leading batch dimension may be fixed `1`, symbolic, or unknown when it does not carry topology meaning. |
 * | Numeric types | Float32 only for the first external lane. Reduced-precision, qlinear, QDQ, and `DynamicQuantizeLinear` external graphs remain outside the claim. |
 * | Affine parameter source | `Gemm` weights and optional bias must come from standard graph initializers. Repo metadata, node names, or `Constant` nodes must not carry required execution semantics. |
 * | Ordering normalization | The importer may topologically reorder only one unambiguous dense chain by tensor dependencies. Ambiguous producer/consumer graphs reject instead of guessing. |
 *
 * Explicit exclusions for that first external lane:
 * - Custom domains, `FunctionProto`, recurrent/state-carrying families, spatial operators,
 *   residual/concat/attention blocks, multiple public inputs/outputs, and any graph that
 *   depends on repo-only metadata for core meaning.
 * - Reduced-precision or quantized external graphs, including storage-fp16, `QLinear*`,
 *   `QuantizeLinear`, `DequantizeLinear`, and `DynamicQuantizeLinear`.
 * - Ambiguous rank/broadcast cases, non-initializer affine parameters, or `Gemm` attribute
 *   forms outside the exact affine contract above.
 *
 * Supported precision subset (current)
 * ------------------------------------
 *
 * | Lane | Current documented opset floor | Supported subset | Granularity / representation | Calibration assumption | Explicit unsupported families |
 * | --- | --- | --- | --- | --- | --- |
 * | `storage-fp16` | current validated export floor: opset 18 | same-family dense and explicit Conv weight or bias initializer storage only | float16 initializer packing plus deterministic `Cast -> float32` bridges before `Gemm` or `Conv` | none | no end-to-end fp16 compute and no recurrent, advanced-graph, mixed-activation, or partial-connectivity widening |
 * | `static-8bit` dense | current validated export floor: opset 18; preserved-unary `Gelu` raises the documented activation floor to opset 20 | explicitly targeted same-family one-output dense layers only | per-tensor activation quantization, deterministic transposed weight tensors, explicit float-domain bias `Add`, and exporter-owned unary activations | requires external calibration ranges for each targeted dense layer | no wider dense outputs, no quantized import, and no residual, concat, attention, recurrent, mixed-activation, or partial-connectivity quantization |
 * | `static-8bit` Conv | current validated export floor: opset 18 | explicit `conv2dMappings` subset only | per-tensor activations, optional per-output-channel weights, one `int32` bias value per output channel, and explicit return to float32 before pool, flatten, reshape, or downstream dense boundaries | requires external calibration ranges for each targeted Conv layer | no heuristic or unsupported spatial promotion and no quantized pooling, flatten, reshape, or downstream dense execution |
 * | `dynamic-uint8` dense guidance | current validated export floor: opset 18 | same-family dense guidance only | `metadata-only` or `DynamicQuantizeLinear -> DequantizeLinear`; affine and activation compute stay on the float32 path | none | no generic dynamic-int8 claim and no Conv, pool, recurrent, advanced-graph, mixed-activation, or partial-connectivity widening |
 *
 * Preserved-unary static dense coverage currently validates `Relu`, `Sigmoid`,
 * `Tanh`, `Softplus`, `Softsign`, and `Selu` on the default opset 18 path,
 * `Mish` at opset >= 18, and `Gelu` at opset >= 20.
 *
 * Across all current Phase 7 lanes, quantized import, arbitrary external
 * reduced-precision graphs, float8, int4, float4, int2, weight-only
 * quantization, and quantization-aware training remain outside the supported
 * subset.
 *
 * Quantized import is still outside the supported subset, and reduced-precision or
 * optimized emission does not imply checker-backed or ONNX Runtime compatibility
 * beyond the binary-first validated subset described above.
 *
 * Supported recurrent subset (current)
 * ------------------------------------
 * - Single-step self-recurrent hidden layers emitted by this repo’s exporter.
 * - Heuristic LSTM/GRU export/import roundtrips from the same version family when the
 *   exported `W`, `R`, and `B` tensors are complete and shape-compatible.
 *
 * Supported spatial subset (current)
 * ----------------------------------
 * - Explicit `conv2dMappings` export real `Conv` nodes and import back into dense runtime
 *   connections for the supported internal roundtrip family.
 * - `pool2dMappings` and `flattenAfterPooling` export real graph nodes, but import currently
 *   re-attaches those semantics as `_onnxPooling` metadata for inspection rather than changing
 *   runtime inference.
 * - `flattenConsistency` is intentionally audit-only for now: it records whether flattened pool
 *   width matches the next dense consumer width without warning, rejecting, or rewriting weights.
 * - Heuristic `conv2d_inferred_specs` stays metadata-only by default. With
 *   `autoPromoteInferredConv`, supported Conv-like patterns can upgrade into real `Conv`
 *   emission only when the dense weights already satisfy the same shared-kernel consistency
 *   rules used by explicit Conv validation. The currently proven subset includes
 *   single-channel layouts, conservative multi-channel layouts, unpooled stacked
 *   Conv-like chains, deeper single-channel post-pool chains when the exporter can
 *   derive each pooled tensor shape sequentially and preserve the compact per-channel
 *   pooled source slice, and deeper pooled multi-channel chains when the exporter
 *   can do the same across repeated pooled stages. The only proven flatten-after-pool
 *   promotion path is the final hidden-stage reshape-bridge subset. Earlier flattened
 *   pooled consumers, repeated flatten-bridge chains, and downstream dense layers that
 *   still carry non-zero weights from extra non-pooled inputs remain metadata-only.
 *
 * Honest fallback boundary
 * ------------------------
 * - Arbitrary external ONNX recurrent graphs are not supported.
 * - One-hop dense-family residual adds now roundtrip for the same-family subset:
 *   export emits an explicit `Add` merge plus `advanced_graph_residual_adds`
 *   metadata, and import rebuilds the skipped feed-forward edges from the
 *   residual branch tensor together with the recorded cross-layer audit edges.
 * - Explicit same-family concat mappings now roundtrip for the narrow Phase 5
 *   subset: export emits a deterministic `Concat -> Gemm` path together with
 *   `advanced_graph_concat_merges` metadata, and import validates that merge,
 *   rebuilds the skipped source-layer fan-in from the widened dense tensor
 *   tail, and preserves the audit payload as `_onnxAdvancedGraph.concatMerges`.
 * - Fixed-width same-family self-attention mappings can now emit a deterministic
 *   shadow subgraph (`Q/K/V`, head split, score `MatMul`, optional scaling,
 *   `Softmax`, value aggregation, head merge, and output projection) together
 *   with `advanced_graph_attention_blocks` metadata. Import validates that
 *   shadow structure and preserves it as `_onnxAdvancedGraph.attentionBlocks`
 *   audit data while runtime inference stays on the dense fallback scaffold.
 * - Other non-adjacent feed-forward edges are still preserved as
 *   `advanced_graph_cross_layer_connections` metadata and re-attached on import
 *   as `_onnxAdvancedGraph` audit data, but merge families outside the explicit
 *   residual and concat subset remain on the honest fallback path.
 * - Exact dense/per-neuron initializer aliases can now reuse one canonical
 *   tensor name when metadata is enabled, but that subset is still same-family
 *   only: near-equal, cross-family, transposed, or otherwise non-exact tensors
 *   remain duplicated rather than being silently tied together.
 * - Export-time optimization remains subset-gated: opset-incompatible activations,
 *   unresolved shape arithmetic, and invalid broadcast or axis combinations fail early
 *   or stay on baseline emission rather than widening support claims implicitly.
 * - If fused recurrent metadata is malformed or required recurrent tensors are missing,
 *   the importer falls back to the base layered reconstruction rather than claiming a
 *   generic recurrent import success.
 *
 * Trust boundary (security)
 * -------------------------
 * Treat an `OnnxModel` like a generic JSON payload: do not import untrusted blobs.
 *
 * Common pitfalls
 * ---------------
 * - Layer shape mismatches: inconsistent tensor shapes/metadata can make import fail.
 * - Relaxed export options can produce graphs that are harder to interpret outside
 *   NeatapticTS.
 * - Conv/pool mappings must match actual layer sizes.
 *
 * Minimal round-trip example
 * --------------------------
 *
 * ```ts
 * import Architect from '../../architect';
 * import { exportToONNX, importFromONNX } from './network.onnx';
 *
 * const network = Architect.perceptron(2, 3, 1);
 *
 * const model = exportToONNX(network, { includeMetadata: true });
 * const jsonText = JSON.stringify(model);
 *
 * const restored = importFromONNX(JSON.parse(jsonText) as typeof model);
 * const output = restored.activate([0.2, 0.8]);
 * ```
 */

/**
 * Default export bundle for the ONNX-like serialization chapter.
 *
 * Bundles the primary ONNX entry points so the network facade can bind them as
 * methods without importing each function individually.
 */
const networkOnnxUtils = {
  exportToONNX,
  importFromONNXBinary,
  importFromONNX,
};
export default networkOnnxUtils;
