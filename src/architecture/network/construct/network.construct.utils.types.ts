import type Group from '../../group';
import type Layer from '../../layer';
import type Node from '../../node';
import type Network from '../network';
import type {
  NetworkConstructorOptions,
  NetworkTopologyIntent,
} from '../network.types';

/**
 * Stable node identifiers accepted by the construct-from-parts API.
 *
 * Numeric ids resolve against `node.geneId`. String ids resolve against
 * `node.label` when one was attached through `describe({ label })`.
 */
export type ConstructNodeId = number | string;

/**
 * Mixed primitive inputs accepted by `Network.construct(...)`.
 *
 * The builder flattens nodes out of each composite part while preserving
 * deterministic ordering and validating that referenced edges stay inside the
 * provided part set.
 */
export type ConstructPart = Group | Layer | Node;

/**
 * Extra structural validation switches for the construct-from-parts network materialization pipeline.
 */
export interface ConstructValidationOptions {
  /** Reject parallel edges that share the same source and target nodes. */
  forbidDuplicateEdges?: boolean;
  /** Reject self edges before runtime scheduling is compiled. */
  forbidSelfEdges?: boolean;
  /** Allow output-role nodes to project onward or gate connections instead of acting as pure sinks. */
  allowOutputNodeOutgoingEdges?: boolean;
}

/**
 * Public options for `Network.construct(...)`.
 *
 * This surface deliberately reuses the existing `Network` runtime instead of
 * introducing a second execution engine. Callers provide graph parts, choose
 * acyclic versus recurrent compilation, and optionally pin the public input and
 * output vector ordering explicitly. Public input nodes are always validated as
 * pure sources, and output nodes default to pure sinks unless validation opts
 * into outward feedback or gating explicitly.
 */
export interface ConstructOptions extends Pick<
  NetworkConstructorOptions,
  | 'activationPrecision'
  | 'returnTypedActivations'
  | 'reuseActivationArrays'
  | 'reuseSequenceBuffers'
  | 'seed'
> {
  /** Scheduling mode to compile after materializing the runtime graph. */
  mode?: 'acyclic' | 'recurrent';
  /** Ordered stable ids that define the public input vector contract. */
  inputNodes?: ConstructNodeId[];
  /** Ordered stable ids that define the public output vector contract. */
  outputNodes?: ConstructNodeId[];
  /** Allow hidden-role nodes that carry no incident edges inside the provided parts. */
  allowIsolatedHiddenNodes?: boolean;
  /** Optional edge-level and public I/O boundary validation overrides. */
  validate?: ConstructValidationOptions;
}

/**
 * Lightweight diagnostics returned alongside one constructed network.
 *
 * `activationOrder` resolves into network node indices so downstream tooling can
 * inspect one canonical traversal order without re-reading private scheduling
 * fields.
 */
export interface ConstructDiagnostics {
  /** Total node count after flattening all provided parts. */
  nodeCount: number;
  /** Total connection count including self edges. */
  edgeCount: number;
  /** True when the compiled graph contains at least one directed cycle. */
  detectedCycles: boolean;
  /** Flattened activation traversal in runtime node-index order. */
  activationOrder: number[];
}

/**
 * One detached, JSON-serializable node row in the construct graph snapshot.
 *
 * Carries stable gene identity, semantic role, and public I/O ordering metadata
 * so developer tooling can inspect the materialized graph without accessing
 * mutable `Network` internals directly.
 */
export interface ConstructGraphNodeSummary {
  /** Stable runtime node index after deterministic materialization. */
  index: number;
  /** Stable node gene id. */
  geneId: number;
  /** Optional human-readable label attached through describe(). */
  label: string | null;
  /** Runtime node role used by the public construct contract. */
  role: string;
  /** Ordered public input position when the node participates in the input vector. */
  inputOrder: number | null;
  /** Ordered public output position when the node participates in the output vector. */
  outputOrder: number | null;
}

/**
 * One detached, JSON-serializable edge row in the construct graph snapshot.
 *
 * Carries stable innovation id, source/target identity, current weight and
 * enabled state, gater identity when present, and self-loop classification.
 */
export interface ConstructGraphConnectionSummary {
  /** Stable historical marking for the connection. */
  innovation: number;
  /** Runtime source node index. */
  fromIndex: number;
  /** Runtime target node index. */
  toIndex: number;
  /** Stable source node gene id. */
  fromGeneId: number;
  /** Stable target node gene id. */
  toGeneId: number;
  /** Optional gater node gene id when the edge is gated. */
  gaterGeneId: number | null;
  /** True when the edge loops back onto the same node. */
  isSelfConnection: boolean;
  /** Current enabled state preserved for tooling and export probes. */
  enabled: boolean;
  /** Current edge weight. */
  weight: number;
}

/**
 * Detached construct graph snapshot for developer tooling.
 *
 * The snapshot is intentionally JSON-friendly so callers can log it directly,
 * persist it to diagnostics artifacts, or feed it into visualization tooling
 * without re-reading mutable `Network` internals.
 */
export interface ConstructGraphSnapshot {
  /** Scheduling mode requested during construction. */
  requestedMode: 'acyclic' | 'recurrent';
  /** Public topology intent resolved onto the materialized runtime. */
  topologyIntent: NetworkTopologyIntent;
  /** Ordered public input-role node ids. */
  inputNodeIds: number[];
  /** Ordered public output-role node ids. */
  outputNodeIds: number[];
  /** Canonical activation traversal in runtime node-index order. */
  activationOrder: number[];
  /** Detached node rows in runtime order. */
  nodes: ConstructGraphNodeSummary[];
  /** Detached edge rows in deterministic source-target order. */
  connections: ConstructGraphConnectionSummary[];
}

/**
 * Return payload for `Network.construct(...)`.
 */
export interface ConstructResult {
  /** Fully materialized runtime network. */
  network: Network;
  /** Deterministic graph-assembly diagnostics. */
  diagnostics: ConstructDiagnostics;
  /** Detached construct graph snapshot for developer tooling and JSON logging. */
  graph: ConstructGraphSnapshot;
}
