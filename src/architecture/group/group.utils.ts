/**
 * Structural helper utilities for the Group connect, gate, and disconnect operations.
 *
 * These helpers are extracted from the Group class methods to keep each public
 * method below the complexity threshold. All functions are pure structural
 * utilities — they do not hold state and have no side-effects beyond the
 * explicit mutation of the arguments they receive.
 *
 * @module group.utils
 */
import type Connection from '../connection/connection';
import type Node from '../node/node';
import type Group from './group';
import type Layer from '../layer/layer';
import * as methods from '../../methods/methods';
import { config } from '../../config';
import { GroupOneToOneSizeMismatchError } from './group.errors';

// ─── connect helpers ────────────────────────────────────────────────────────

/**
 * Resolve the default group-connection method when none is provided by the caller.
 *
 * Returns ALL_TO_ALL for distinct source/target groups and ONE_TO_ONE when a
 * group is wired to itself, emitting a console warning for each case when
 * warnings are enabled.
 */
export function resolveDefaultGroupConnectionMethod(
  source: Group,
  target: Group,
): unknown {
  if (source !== target) {
    if (config.warnings) {
      console.warn(
        'No group connection specified, using ALL_TO_ALL by default.',
      );
    }
    return methods.groupConnection.ALL_TO_ALL;
  }
  if (config.warnings) {
    console.warn('Connecting group to itself, using ONE_TO_ONE by default.');
  }
  return methods.groupConnection.ONE_TO_ONE;
}

/**
 * Connect source group to target group using ALL_TO_ALL or ALL_TO_ELSE semantics.
 *
 * Skips self-pairs when ALL_TO_ELSE is requested. Registers every created
 * connection on both groups' bookkeeping lists.
 */
export function connectAllToAll(
  source: Group,
  target: Group,
  method: unknown,
  weight: number | undefined,
): Connection[] {
  const created: Connection[] = [];
  for (let si = 0; si < source.nodes.length; si++) {
    for (let ti = 0; ti < target.nodes.length; ti++) {
      if (
        method === methods.groupConnection.ALL_TO_ELSE &&
        source.nodes[si] === target.nodes[ti]
      ) {
        continue;
      }
      const conn = source.nodes[si].connect(target.nodes[ti], weight);
      source.connections.out.push(conn[0]);
      target.connections.in.push(conn[0]);
      created.push(conn[0]);
    }
  }
  return created;
}

/**
 * Connect source group to target group using ONE_TO_ONE semantics.
 *
 * Throws when the groups differ in size. Registers self-connections on
 * the self shelf when source and target are the same group object.
 */
export function connectOneToOne(
  source: Group,
  target: Group,
  weight: number | undefined,
): Connection[] {
  if (source.nodes.length !== target.nodes.length) {
    throw new GroupOneToOneSizeMismatchError(
      'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.',
    );
  }
  const created: Connection[] = [];
  for (let i = 0; i < source.nodes.length; i++) {
    const conn = source.nodes[i].connect(target.nodes[i], weight);
    if (source === target) {
      source.connections.self.push(conn[0]);
    } else {
      source.connections.out.push(conn[0]);
      target.connections.in.push(conn[0]);
    }
    created.push(conn[0]);
  }
  return created;
}

/**
 * Connect source group to target group, dispatching to the correct connection method.
 *
 * Resolves a default method when none is provided and delegates to ALL_TO_ALL,
 * ALL_TO_ELSE, or ONE_TO_ONE helpers.
 */
export function connectGroupToGroup(
  source: Group,
  target: Group,
  method: unknown,
  weight: number | undefined,
): Connection[] {
  const resolved =
    method === undefined
      ? resolveDefaultGroupConnectionMethod(source, target)
      : method;

  if (
    resolved === methods.groupConnection.ALL_TO_ALL ||
    resolved === methods.groupConnection.ALL_TO_ELSE
  ) {
    return connectAllToAll(source, target, resolved, weight);
  }
  return connectOneToOne(source, target, weight);
}

/**
 * Connect every node in source group to a single target node.
 *
 * Registers each created connection on the source group's outbound list.
 */
export function connectGroupToNode(
  source: Group,
  target: Node,
  weight: number | undefined,
): Connection[] {
  const created: Connection[] = [];
  for (let i = 0; i < source.nodes.length; i++) {
    const conn = source.nodes[i].connect(target, weight);
    source.connections.out.push(conn[0]);
    created.push(conn[0]);
  }
  return created;
}

/**
 * Connect source group to target layer, delegating to the layer's input method.
 */
export function connectGroupToLayer(
  source: Group,
  target: Layer,
  method: unknown,
  weight: number | undefined,
): Connection[] {
  return target.input(source, method, weight);
}

// ─── gate helpers ───────────────────────────────────────────────────────────

/**
 * Collect unique source nodes referenced by a set of connections.
 *
 * Preserves first-seen order so gating index assignment is deterministic.
 */
export function collectUniqueSourceNodes(connections: Connection[]): Node[] {
  const seen: Node[] = [];
  for (let i = 0; i < connections.length; i++) {
    if (!seen.includes(connections[i].from)) {
      seen.push(connections[i].from);
    }
  }
  return seen;
}

/**
 * Apply INPUT gating: assign each connection to the group node at connection-index modulo group size.
 */
export function gateByInput(
  group: Group,
  gatedConnections: Connection[],
): void {
  for (let i = 0; i < gatedConnections.length; i++) {
    const gater = group.nodes[i % group.nodes.length];
    gater.gate(gatedConnections[i]);
  }
}

/**
 * Apply OUTPUT gating: for each source node, gate its matching outbound connections.
 */
export function gateByOutput(
  group: Group,
  gatedConnections: Connection[],
  sourceNodes: Node[],
): void {
  for (let si = 0; si < sourceNodes.length; si++) {
    const node = sourceNodes[si];
    const gater = group.nodes[si % group.nodes.length];
    for (let ci = 0; ci < node.connections.out.length; ci++) {
      const conn = node.connections.out[ci];
      if (gatedConnections.includes(conn)) {
        gater.gate(conn);
      }
    }
  }
}

/**
 * Apply SELF gating: for each source node, gate its self-connection when present in the set.
 */
export function gateBySelf(
  group: Group,
  gatedConnections: Connection[],
  sourceNodes: Node[],
): void {
  for (let si = 0; si < sourceNodes.length; si++) {
    const node = sourceNodes[si];
    const gater = group.nodes[si % group.nodes.length];
    const selfConn = Array.isArray(node.connections.self)
      ? node.connections.self[0]
      : node.connections.self;
    if (gatedConnections.includes(selfConn)) {
      gater.gate(selfConn);
    }
  }
}

// ─── disconnect helpers ──────────────────────────────────────────────────────

/**
 * Remove the first matching outbound connection from a connection list.
 *
 * Walks the list in reverse to avoid index-shift errors during splice. Stops
 * after the first removal because each source/target pair appears at most once.
 */
export function removeOutboundConnection(
  connectionList: Connection[],
  from: Node,
  to: Node,
): void {
  for (let i = connectionList.length - 1; i >= 0; i--) {
    const conn = connectionList[i];
    if (conn.from === from && conn.to === to) {
      connectionList.splice(i, 1);
      break;
    }
  }
}

/**
 * Remove the first matching inbound connection from a connection list.
 *
 * Mirrors `removeOutboundConnection` for the receiving side of an edge.
 */
export function removeInboundConnection(
  connectionList: Connection[],
  from: Node,
  to: Node,
): void {
  for (let i = connectionList.length - 1; i >= 0; i--) {
    const conn = connectionList[i];
    if (conn.from === from && conn.to === to) {
      connectionList.splice(i, 1);
      break;
    }
  }
}

/**
 * Disconnect every source node in the group from every target node in another group.
 *
 * Also removes the disconnected connections from both groups' bookkeeping lists.
 * When twosided is true, the reverse connections are removed as well.
 */
export function disconnectGroupFromGroup(
  source: Group,
  target: Group,
  twosided: boolean,
): void {
  for (let si = 0; si < source.nodes.length; si++) {
    const sourceNode = source.nodes[si];
    for (let ti = 0; ti < target.nodes.length; ti++) {
      const targetNode = target.nodes[ti];
      sourceNode.disconnect(targetNode, twosided);
      removeOutboundConnection(source.connections.out, sourceNode, targetNode);
      removeInboundConnection(target.connections.in, sourceNode, targetNode);
      if (twosided) {
        removeInboundConnection(source.connections.in, targetNode, sourceNode);
        removeOutboundConnection(
          target.connections.out,
          targetNode,
          sourceNode,
        );
      }
    }
  }
}

/**
 * Disconnect every node in the group from a single target node.
 *
 * Removes the disconnected connections from the source group's outbound list.
 * When twosided is true, reverse connections are removed from the inbound list.
 */
export function disconnectGroupFromNode(
  source: Group,
  target: Node,
  twosided: boolean,
): void {
  for (let i = 0; i < source.nodes.length; i++) {
    const sourceNode = source.nodes[i];
    sourceNode.disconnect(target, twosided);
    removeOutboundConnection(source.connections.out, sourceNode, target);
    if (twosided) {
      removeInboundConnection(source.connections.in, target, sourceNode);
    }
  }
}
