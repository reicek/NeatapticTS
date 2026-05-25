import type Network from '../../network/network';
import Node from '../../node';
import Connection from '../../connection';
import { config } from '../../../config';
import { NetworkGatingNodeMembershipError } from './network.gating.errors';

/**
 * Validate that a candidate gater node belongs to the target network before mutating connection gate ownership.
 * This guard prevents cross-network node references that would corrupt reverse gate bookkeeping.
 *
 * @param network - Network performing the gating operation.
 * @param node - Candidate gater node.
 * @returns Nothing.
 * @throws If the node does not belong to the network.
 */
export function assertGaterNodeBelongsToNetwork(
  network: Network,
  node: Node,
): void {
  if (!network.nodes.includes(node)) {
    throw new NetworkGatingNodeMembershipError(
      'Gating node must be part of the network to gate a connection!',
    );
  }
}

/**
 * Determine whether a connection already has a gater node assigned before applying a new gate operation.
 * This check is used to keep gate lists idempotent and avoid duplicate ownership updates.
 *
 * @param connection - Connection candidate.
 * @returns True when a gater is already set; otherwise false.
 */
export function isConnectionAlreadyGated(connection: Connection): boolean {
  return Boolean(connection.gater);
}

/**
 * Emit a warning when a gate operation is skipped due to existing gating.
 *
 * @returns Nothing.
 */
export function warnConnectionAlreadyGated(): void {
  if (config.warnings) {
    console.warn('Connection is already gated. Skipping.');
  }
}

/**
 * Attach a gater to a connection and track that connection in the network gate list used by ungate routines.
 * The helper keeps connection-level and network-level gate bookkeeping synchronized.
 *
 * @param network - Network being updated.
 * @param node - Gater node to attach.
 * @param connection - Connection to gate.
 * @returns Nothing.
 */
export function attachGaterToConnection(
  network: Network,
  node: Node,
  connection: Connection,
): void {
  node.gate(connection);
  network.gates.push(connection);
}

/**
 * Find a connection position within the network global gates list used for deterministic ungate updates.
 * Returning a stable index allows callers to remove tracked gates without scanning extra structures.
 *
 * @param network - Network containing global gate references.
 * @param connection - Connection to locate.
 * @returns Zero-based index in the gates list, or -1 when absent.
 */
export function findGateIndex(
  network: Network,
  connection: Connection,
): number {
  return network.gates.indexOf(connection);
}

/**
 * Emit a warning when an ungate request targets a non-tracked connection.
 *
 * @returns Nothing.
 */
export function warnMissingGateConnection(): void {
  if (config.warnings) {
    console.warn('Attempted to ungate a connection not in the gates list.');
  }
}

/**
 * Remove a gated connection from the network global gate list after ownership checks pass.
 * This mutation helper isolates gate-list updates so higher-level ungate flow stays declarative.
 *
 * @param network - Network being updated.
 * @param gateIndex - Index to remove from the gate list.
 * @returns Nothing.
 */
export function removeGateAtIndex(network: Network, gateIndex: number): void {
  network.gates.splice(gateIndex, 1);
}

/**
 * Remove reverse gater bookkeeping from a connection's gater node when ungating a previously tracked edge.
 * Keeping this step explicit prevents stale gated-connection references on the gater node.
 *
 * @param connection - Connection to detach from its gater.
 * @returns Nothing.
 */
export function detachConnectionFromGater(connection: Connection): void {
  connection.gater?.ungate(connection);
}
