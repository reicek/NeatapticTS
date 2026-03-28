/**
 * NeatapticTS ONNX-like serialization for networks.
 *
 * This module provides the two public entry points:
 * - `exportToONNX()` turns a runtime `Network` into a plain JSON object (`OnnxModel`).
 * - `importFromONNX()` reconstructs a `Network` from that JSON object.
 *
 * What this format is (and is not):
 * - It is **JSON-first** and intentionally resembles ONNX’s model/graph concepts.
 * - It is **not** a full ONNX protobuf implementation and is not guaranteed to run on
 *   general ONNX runtimes.
 * - The compatibility promise is primarily **within this repo**: models produced by
 *   `exportToONNX()` should be accepted by `importFromONNX()` (same version family).
 *
 * How to read this chapter:
 * - Start here for the public round-trip API and the trust boundary.
 * - Continue into `export/` to see how layered networks become JSON graph payloads.
 * - Continue into `import/` to see how that payload becomes a runtime network again.
 * - Continue into `schema/` for the persisted wire-format shapes.
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
import { runOnnxExportFlow, runOnnxImportFlow } from './network.onnx.utils';
import type { OnnxExportOptions } from './network.onnx.utils.types';
import type {
  Conv2DMapping,
  OnnxModel,
  Pool2DMapping,
} from './schema/network.onnx.schema.types';

export type { Conv2DMapping, OnnxExportOptions, OnnxModel, Pool2DMapping };

/**
 * Export a NeatapticTS network to an ONNX-like **JSON object** (`OnnxModel`).
 *
 * What you get:
 * - A plain object that you can persist with `JSON.stringify()`.
 * - A minimal ONNX-ish graph (`model.graph`) plus optional metadata (`model.metadata_props`).
 *
 * When to use this:
 * - You want a portable snapshot that can be inspected/diffed as JSON.
 * - You want to reconstruct the network later via `importFromONNX()`.
 *
 * Tradeoffs:
 * - The output is ONNX-like, but **not** intended to be universally compatible with all ONNX
 *   runtimes.
 * - Some advanced features (partial connectivity, mixed activations, recurrent heuristics)
 *   may produce graphs that are primarily meant for this library’s importer.
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
 * Reconstruct a NeatapticTS network from an exported `OnnxModel`.
 *
 * Expected input:
 * - A model produced by `exportToONNX()` (same repo/version family).
 *
 * Trust boundary:
 * - Do not import untrusted blobs. A malformed model can be extremely large or internally
 *   inconsistent and may cause errors or high memory usage.
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
 * ONNX (JSON) serialization for NeatapticTS networks.
 *
 * NeatapticTS provides an **ONNX-like, JSON-first interchange format** for exporting and
 * reconstructing a constrained subset of `Network` instances.
 *
 * This is primarily meant for:
 * - Saving a trained network snapshot in a portable representation.
 * - Debugging / inspecting network structure as a graph of tensors and nodes.
 * - Interop with tooling that can consume graph-shaped metadata.
 *
 * Formats that exist
 * ------------------
 * In this folder, the “format” is really two layers:
 *
 * 1) In-memory model object (`OnnxModel`)
 *    - A plain JavaScript object shaped like an ONNX `ModelProto`, but represented as JSON.
 *    - This is what `exportToONNX()` returns and what `importFromONNX()` consumes.
 *
 * 2) Serialized JSON text
 *    - Persist via `JSON.stringify(model)`.
 *    - Human-readable and diffable.
 *
 * There is no binary/protobuf serializer here. For compactness, apply compression
 * (gzip/brotli) to the JSON string at the application layer.
 *
 * Guarantees & stability
 * ----------------------
 * - Round-trip intent: `importFromONNX(exportToONNX(network))` aims to reconstruct a
 *   functionally equivalent network for the supported subset.
 * - Compatibility: the importer is only guaranteed to accept models produced by this
 *   repo’s exporter.
 * - Determinism: export is designed to be deterministic given the same network state and
 *   export options.
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
export default {
  exportToONNX,
  importFromONNX,
};
