/**
 * One node snapshot inside the worker-friendly inference IR.
 *
 * The IR stores runtime inference data only: stable node indexes, activation
 * lookup ids, recurrent self-loop metadata, and scalar modifiers that change
 * forward-pass output.
 *
 * @example
 * ```ts
 * const irNode: NetworkInferenceIRNode = {
 *   index: 3,
 *   bias: 0.15,
 *   response: 1,
 *   mask: 1,
 *   activationId: 4,
 *   selfWeight: 0,
 *   selfGaterIndex: -1,
 * };
 * ```
 */
export interface NetworkInferenceIRNode {
  /** Stable node index used by every IR edge and activation step. */
  readonly index: number;
  /** Additive bias applied before the activation function. */
  readonly bias: number;
  /** Response multiplier applied to state before squashing. */
  readonly response: number;
  /** Multiplicative activation mask used by inference. */
  readonly mask: number;
  /** Stable index into `INFERENCE_ACTIVATION_TABLE`. */
  readonly activationId: number;
  /** Recurrent self-connection weight, or `0` when absent. */
  readonly selfWeight: number;
  /** Stable gater node index for the self-connection, or `-1` when absent. */
  readonly selfGaterIndex: number;
}

/**
 * One deterministic non-self connection row inside the worker-friendly inference IR.
 * This shape keeps forward edges compact and index-addressable so predictor hot paths can traverse weighted inputs without object graph lookups or runtime connection instances.
 *
 * @example
 * ```ts
 * const irEdge: NetworkInferenceIREdge = {
 *   from: 0,
 *   to: 3,
 *   weight: 0.75,
 *   gaterIndex: -1,
 * };
 * ```
 */
export interface NetworkInferenceIREdge {
  /** Stable source node index. */
  readonly from: number;
  /** Stable target node index. */
  readonly to: number;
  /** Connection weight used during inference. */
  readonly weight: number;
  /** Stable gater node index, or `-1` when ungated. */
  readonly gaterIndex: number;
}

/**
 * Deterministic, worker-consumable snapshot of one network forward pass.
 *
 * This IR is the common substrate for portable, transferable, channel, and
 * shared-memory worker transport strategies. It intentionally stores plain data
 * only so callers can structured-clone it or re-encode it into typed arrays.
 *
 * @example
 * ```ts
 * const inferenceIr = extractNetworkInferenceIR(network);
 * console.log(inferenceIr.activationSteps);
 * ```
 */
export interface NetworkInferenceIR {
  /** Public input-vector width. */
  readonly inputCount: number;
  /** Public output-vector width. */
  readonly outputCount: number;
  /** Stable node snapshots aligned to node index order. */
  readonly nodes: ReadonlyArray<NetworkInferenceIRNode>;
  /** Stable non-self edge snapshots aligned to deterministic connection order. */
  readonly edges: ReadonlyArray<NetworkInferenceIREdge>;
  /** Ordered activation groups that preserve runtime schedule step boundaries. */
  readonly activationSteps: ReadonlyArray<ReadonlyArray<number>>;
  /** Stable output-node indexes aligned to public output order. */
  readonly outputNodeIndices: ReadonlyArray<number>;
}

/**
 * One portable node record used by the structured-clone payload surface.
 *
 * Portable payloads keep human-readable activation names instead of numeric ids
 * so they remain self-describing across worker boundaries and versioned message
 * logs.
 *
 * @example
 * ```ts
 * const portableNode: PortableInferencePayloadNode = {
 *   id: 3,
 *   bias: 0.15,
 *   response: 1,
 *   mask: 1,
 *   activation: 'relu',
 *   selfWeight: 0,
 *   selfGaterIndex: -1,
 * };
 * ```
 */
export interface PortableInferencePayloadNode {
  /** Stable node index used by edges, activation steps, and outputs. */
  readonly id: number;
  /** Additive bias applied before activation. */
  readonly bias: number;
  /** Response multiplier applied before squashing. */
  readonly response: number;
  /** Multiplicative mask applied after squashing. */
  readonly mask: number;
  /** Canonical activation name from `activationTable`. */
  readonly activation: string;
  /** Recurrent self-connection weight, or `0` when absent. */
  readonly selfWeight: number;
  /** Stable gater node index for the self-connection, or `-1` when absent. */
  readonly selfGaterIndex: number;
}

/**
 * One structured-clone-safe edge row used by the portable inference payload strategy.
 * Portable edges preserve the same deterministic topology as IR edges while staying JSON-like and self-describing for debugging, logs, and cross-version message tracing.
 *
 * @example
 * ```ts
 * const portableEdge: PortableInferencePayloadEdge = {
 *   from: 0,
 *   to: 3,
 *   weight: 0.75,
 *   gaterIndex: -1,
 * };
 * ```
 */
export interface PortableInferencePayloadEdge {
  /** Stable source node index. */
  readonly from: number;
  /** Stable target node index. */
  readonly to: number;
  /** Connection weight used during inference. */
  readonly weight: number;
  /** Stable gater node index, or `-1` when ungated. */
  readonly gaterIndex: number;
}

/**
 * Structured-clone-safe inference payload for the universal worker fallback.
 *
 * This payload preserves the exact inference metadata needed to replay a
 * forward pass without shipping live `Network`, `Node`, or `Connection`
 * instances across the worker boundary.
 *
 * @example
 * ```ts
 * const payload = exportPortableInferencePayload(network);
 * console.log(payload.activationTable);
 * ```
 */
export interface PortableInferencePayload {
  /** Schema version for portable payload evolution. */
  readonly version: 1;
  /** Strategy discriminator for public transport callers. */
  readonly strategy: 'portable';
  /** Public input-vector width. */
  readonly inputCount: number;
  /** Public output-vector width. */
  readonly outputCount: number;
  /** Ordered activation groups that preserve runtime schedule boundaries. */
  readonly activationSteps: number[][];
  /** Stable portable node records aligned to node index order. */
  readonly nodes: PortableInferencePayloadNode[];
  /** Stable portable edge records aligned to deterministic connection order. */
  readonly edges: PortableInferencePayloadEdge[];
  /** Stable output-node indexes aligned to public output order. */
  readonly outputNodeIndices: number[];
  /** Self-describing activation shelf used by the payload. */
  readonly activationTable: string[];
}

/** Numeric typed-array shelf used by transferable payload fields. */
type TransferableNumericArray = Float32Array | Float64Array;

/** Numeric precision modes supported by transferable payload export. */
type TransferableNumericPrecision = 'full' | 'f32';

/**
 * Typed-array inference payload for lower-copy worker transport.
 *
 * Transferable payloads keep the same semantic content as portable payloads,
 * but pack it into flat typed arrays so callers can move ownership across
 * worker boundaries without deep-cloning large object graphs.
 *
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network);
 * console.log(payload.nodeIds.length);
 * ```
 */
export interface TransferableInferencePayload {
  /** Schema version for transferable payload evolution. */
  readonly version: 1;
  /** Strategy discriminator for public transport callers. */
  readonly strategy: 'transferable';
  /** Public input-vector width. */
  readonly inputCount: number;
  /** Public output-vector width. */
  readonly outputCount: number;
  /** Start offsets for each activation step inside `activationStepsData`. */
  readonly activationStepsIndex: Int32Array;
  /** Flat activation-step node indexes packed in traversal order. */
  readonly activationStepsData: Int32Array;
  /** Stable node ids aligned to deterministic node order. */
  readonly nodeIds: Int32Array;
  /** Node biases using the configured numeric precision. */
  readonly nodeBiases: TransferableNumericArray;
  /** Node response multipliers using the configured numeric precision. */
  readonly nodeResponses: TransferableNumericArray;
  /** Node activation masks using the configured numeric precision. */
  readonly nodeMasks: TransferableNumericArray;
  /** Stable node activation ids aligned to `INFERENCE_ACTIVATION_TABLE`. */
  readonly nodeActivationIds: Int32Array;
  /** Recurrent self-connection weights using the configured numeric precision. */
  readonly nodeSelfWeights: TransferableNumericArray;
  /** Stable self-gater indexes or `-1` when absent. */
  readonly nodeSelfGaterIndices: Int32Array;
  /** Stable edge source node indexes. */
  readonly edgeFrom: Int32Array;
  /** Stable edge target node indexes. */
  readonly edgeTo: Int32Array;
  /** Edge weights using the configured numeric precision. */
  readonly edgeWeights: TransferableNumericArray;
  /** Stable edge gater indexes or `-1` when ungated. */
  readonly edgeGaterIndices: Int32Array;
  /** Stable output-node indexes aligned to public output order. */
  readonly outputNodeIndices: Int32Array;
  /** Activation shelf length used as a version-drift guard. */
  readonly activationTableLength: number;
}

/**
 * Configuration knobs for transferable payload export when balancing precision, size, and transport throughput.
 * Use these options when a worker boundary prefers lower-copy typed shelves but still needs predictable numeric semantics across browser and Node runtimes.
 *
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network, {
 *   numericPrecision: 'f32',
 * });
 * ```
 */
export interface TransferableInferencePayloadOptions {
  /** Numeric precision used for floating-point typed-array fields. */
  readonly numericPrecision?: TransferableNumericPrecision;
}

/**
 * Persistent worker-backed inference channel.
 *
 * Channels bootstrap one transferable predictor payload exactly once, then keep
 * the worker-side predictor warm across repeated predict or reset requests sent
 * through one dedicated `MessageChannel` port pair.
 *
 * @example
 * ```ts
 * const payload = exportTransferableInferencePayload(network);
 * const channel = openInferenceChannel(payload);
 * const outputValues = await channel.predict([0.25, 0.75]);
 * channel.close();
 * ```
 */
export interface InferenceChannel {
  /** Close the channel and release the underlying worker resources. */
  close(): Promise<void>;
  /** Whether the channel is still open for new requests. */
  readonly isOpen: boolean;
  /** Run one asynchronous prediction against the warm worker-side predictor. */
  predict(input: Float64Array | ReadonlyArray<number>): Promise<Float64Array>;
  /** Reset the worker-side recurrent predictor state. */
  reset(): Promise<void>;
  /** Transport strategy discriminator. */
  readonly strategy: 'channel';
}

/**
 * Configuration for one dedicated persistent inference channel backed by a warm worker predictor.
 * These options tune local request queue pressure and worker delivery strategy so repeated inference calls stay stable under bursty client traffic.
 *
 * @example
 * ```ts
 * const channel = openInferenceChannel(payload, {
 *   maxConcurrentRequests: 4,
 * });
 * ```
 */
export interface InferenceChannelOptions {
  /** Maximum number of requests sent before later calls queue locally. */
  readonly maxConcurrentRequests?: number;
  /** Optional worker entry override for CSP or custom packaging environments. */
  readonly workerUrl?: string;
}

/**
 * Persistent shared-memory inference worker.
 *
 * Shared-memory workers keep one predictor alive inside a dedicated worker and
 * exchange input and output values through `SharedArrayBuffer` shelves instead
 * of cloning or transferring a fresh payload for every request.
 *
 * @example
 * ```ts
 * const worker = openSharedInferenceWorker(payload);
 * const outputValues = await worker.infer([0.25, 0.75]);
 * await worker.release();
 * ```
 */
export interface SharedInferenceWorker {
  /** Wait for the output shelf to become ready after one submitted input. */
  awaitOutput(): Promise<Float64Array>;
  /** Run one full shared-memory inference roundtrip. */
  infer(input: ReadonlyArray<number>): Promise<Float64Array>;
  /** Whether the worker finished bootstrap and can accept new input. */
  readonly isReady: boolean;
  /** Release the worker and shared-memory resources. */
  release(): Promise<void>;
  /** Reset recurrent state inside the shared worker. */
  reset(): Promise<void>;
  /** Write one input vector into the shared input shelf. */
  submitInput(input: ReadonlyArray<number>): void;
  /** Transport strategy discriminator. */
  readonly strategy: 'shared-memory';
}

/**
 * Configuration for one dedicated shared-memory inference worker.
 *
 * Browser hosts default to the module-relative shared worker emitted beside the
 * library files. Provide `workerUrl` when bundling moves that worker asset or
 * when CSP policy disallows the default delivery path.
 *
 * @example
 * ```ts
 * const worker = openSharedInferenceWorker(payload, {
 *   workerUrl: '/assets/shared-inference.worker.js',
 * });
 * ```
 */
export interface SharedInferenceWorkerOptions {
  /** Optional worker entry override for CSP or hosts that relocate the default worker asset. */
  readonly workerUrl?: string;
}

/**
 * Runtime predictor created from an exported worker payload.
 *
 * Predictors intentionally keep mutable activation, recurrent-state, and gain
 * buffers private so callers can reuse the same instance across many
 * predictions without re-allocating worker state.
 *
 * @example
 * ```ts
 * const predictor = createInferencePredictor(payload);
 * const outputValues = predictor.predict([0.25, 0.75]);
 * predictor.reset();
 * ```
 */
export interface InferencePredictor {
  /** Transport strategy that created this predictor. */
  readonly strategy: 'portable' | 'transferable';
  /** Run one inference pass using the current predictor state. */
  predict(input: ReadonlyArray<number>): number[];
  /** Reset recurrent state and gated gains to fresh-network conditions. */
  reset(): void;
}
