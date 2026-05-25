import type { ActivationPrecision, PrecisionConfig } from '../../../config';
import type Network from '../../network/network';
import type Connection from '../../connection';
import type Node from '../../node';

/**
 * Numeric zero sentinel used across slab orchestration and helper pipelines.
 */
export const SLAB_ZERO = 0;

/**
 * Numeric one sentinel used for neutral gain defaults and index math.
 */
export const SLAB_ONE = 1;

/**
 * Capacity growth factor for Node.js runtime slab allocations, scaled conservatively to allow large networks.
 */
export const SLAB_GROWTH_FACTOR_NODE = 1.75;

/**
 * Capacity growth factor for browser runtime slab allocations, tuned for tighter memory environments.
 */
export const SLAB_GROWTH_FACTOR_BROWSER = 1.25;

/**
 * Default async slab rebuild chunk size when no override is provided.
 */
export const SLAB_DEFAULT_ASYNC_CHUNK_SIZE = 50_000;

/**
 * Internal Connection properties accessed during slab build, serialization, and typed-array buffer write operations.
 */
export interface ConnectionInternals {
  _flags: number;
  from: Node & { index: number };
  to: Node & { index: number };
  weight: number;
}

/**
 * Internal Network properties used by slab orchestration for typed-array buffer management and dirty tracking.
 */
export interface NetworkSlabProps {
  _slabDirty?: boolean;
  _nodeIndexDirty?: boolean;
  _connCapacity?: number;
  _useFloat32Weights?: boolean;
  _connWeights?: Float32Array | Float64Array;
  _connFrom?: Uint32Array;
  _connTo?: Uint32Array;
  _connFlags?: Uint8Array;
  _connGain?: Float32Array | Float64Array | null;
  _connPlastic?: Float32Array | Float64Array | null;
  _connUsed?: number;
  _connVersion?: number;
  _outStart?: Uint32Array | null;
  _outOrder?: Uint32Array | null;
  _connCount?: number;
  _adjDirty?: boolean;
  _slabVersion?: number;
  _slabAsyncBuilds?: number;
  _precisionConfig?: PrecisionConfig;
  _activationPrecision?: ActivationPrecision;
  _enforceAcyclic?: boolean;
  _fastA?: Float32Array | Float64Array;
  _fastS?: Float32Array | Float64Array;
  _topoDirty?: boolean;
  _topoOrder?: unknown[];
  _weightNoiseStd?: number;
  _weightNoisePerHidden?: unknown[];
  _stochasticDepth?: unknown[];
  connections: (Connection & ConnectionInternals)[];
}

/**
 * Per-pool-key allocation and reuse counters used for educational diagnostics and memory-pool observability.
 */
export interface PoolKeyMetrics {
  created: number;
  reused: number;
  maxRetained: number;
}

/**
 * Union of slab typed array element container types supported by activation buffer allocation.
 */
export type TypedArray = Float32Array | Float64Array | Uint32Array | Uint8Array;

/**
 * Constructor type for typed arrays used in activation slab allocation and dynamic buffer growth.
 */
export type TypedArrayConstructor =
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | Uint32ArrayConstructor
  | Uint8ArrayConstructor;

/**
 * Runtime activation contract consumed by slab-based forward-pass execution paths for network inference.
 */
export interface NetworkActivationRuntime {
  activate(input: number[], training: boolean): number[];
}

/**
 * Runtime topology contract used to lazily rebuild topological order when the activation cache is dirty.
 */
export interface NetworkTopoRuntime {
  _computeTopoOrder(): void;
}

/**
 * Node shape required by fast slab activation kernels for typed-array forward pass inference.
 */
export interface FastSlabNodeRuntime extends Node {
  index: number;
  bias: number;
  squash(x: number): number;
  activation: number;
  state: number;
}

/**
 * Immutable inputs required to build or grow connection slab buffers.
 */
export interface SlabBuildContext {
  network: Network;
  internalNet: NetworkSlabProps;
  connectionCount: number;
  capacity: number;
  growthFactor: number;
  weightBytes: number;
  weightCtor: Float32ArrayConstructor | Float64ArrayConstructor;
}

/**
 * Result of scanning and populating optional gain and plastic typed-array slab buffers.
 */
export interface SlabPopulateResult {
  anyNonNeutralGain: boolean;
  anyPlastic: boolean;
  gainArray: Float32Array | Float64Array | null;
  plasticArray: Float32Array | Float64Array | null;
}

/**
 * Writable typed-array slab buffers targeted during connection serialization and buffer population.
 */
export interface SlabWriteArrays {
  weightArray: Float32Array | Float64Array;
  fromIndexArray: Uint32Array;
  toIndexArray: Uint32Array;
  flagArray: Uint8Array;
}

/**
 * Packed SoA view returned by getConnectionSlab exposing typed-array weight, index, and flag buffers.
 */
export interface ConnectionSlabView {
  weights: Float32Array | Float64Array;
  from: Uint32Array;
  to: Uint32Array;
  flags: Uint8Array;
  gain: Float32Array | Float64Array | null;
  plastic: Float32Array | Float64Array | null;
  version: number;
  used: number;
  capacity: number;
}

/**
 * Shared immutable inputs used across the CSR adjacency build pipeline for outgoing-order construction.
 */
export type BuildAdjacencyContext = {
  internalNet: NetworkSlabProps;
  nodeCount: number;
  connectionCount: number;
  connectionFromSlab: Uint32Array;
};

/**
 * Context for fan-out collection: build inputs plus the output count buffer.
 */
export type FanOutCollectionContext = {
  buildContext: BuildAdjacencyContext;
  fanOutCounts: Uint32Array;
};

/**
 * Context for constructing CSR start offsets from precomputed fan-out counts.
 */
export type StartIndicesBuildContext = {
  buildContext: BuildAdjacencyContext;
  fanOutCounts: Uint32Array;
};

/**
 * Context for constructing the source-grouped outgoing connection order from precomputed CSR start indices.
 */
export type OutgoingOrderBuildContext = {
  buildContext: BuildAdjacencyContext;
  outgoingStartIndices: Uint32Array;
};

/**
 * Context for publishing fully built adjacency slabs to internal network state.
 */
export type PublishAdjacencyContext = {
  internalNet: NetworkSlabProps;
  outgoingStartIndices: Uint32Array;
  outgoingOrder: Uint32Array;
};
