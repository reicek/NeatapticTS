/**
 * Core group chapter for the architecture surface.
 *
 * This folder owns the library's first composite graph primitive: a small
 * collection of nodes that can be activated, propagated, wired, gated, and
 * serialized as one architectural block.
 *
 * Read this chapter in three passes:
 *
 * 1. start with the `Group` class overview to understand how the boundary
 *    wraps many node-level operations into one reusable building block,
 * 2. continue to `connect()` and `gate()` when you need the structural
 *    vocabulary for wiring groups into larger graphs,
 * 3. finish with `disconnect()`, `clear()`, and `toJSON()` when you want the
 *    lifecycle and persistence view for composite primitives.
 *
 * Groups also carry the same two-level story as nodes: construction-time role
 * describes the runtime semantics of their allocated nodes, while
 * `describe({ label, intent, metadata })` lets later tooling remember why this
 * block exists without changing how activation or propagation works.
 */
import Node, {
  type PrimitiveDescriptor,
  type PrimitiveIntent,
  type PrimitiveMetadata,
  type PrimitiveNodeType,
  resolvePrimitiveIntent,
} from '../node/node';
import Connection from '../connection/connection';
import Layer from '../layer/layer';
import { config } from '../../config';
import * as methods from '../../methods/methods';
import {
  GroupGatingMethodRequiredError,
  GroupOneToOneSizeMismatchError,
  GroupSizeMismatchError,
} from './group.errors';

/**
 * Composite node block for architecture construction.
 *
 * A group is the first place where the architecture surface stops talking about
 * one primitive at a time and starts exposing small graph motifs. It owns a set
 * of nodes plus the connection bookkeeping needed to treat that set as one
 * wiring target, one wiring source, and one propagation unit.
 *
 * This makes the boundary useful in three different modes:
 *
 * - dense or structured connection building between graph regions,
 * - collective activation and propagation when a block should act as one unit,
 * - recurrent and gated substructures where node-level behavior is still
 *   needed but orchestration should stay above the single-neuron level.
 *
 * The practical pattern is usually: allocate the group with the right runtime
 * role when the whole block is clearly input- or output-oriented, then add a
 * descriptor only when the boundary should stay visible in diagnostics or
 * later graph assembly.
 *
 * @example
 * ```ts
 * const sensorBlock = new Group(4, 'input');
 * const readoutBlock = new Group(2, 'output');
 *
 * sensorBlock.describe({
 *   label: 'sensorBlock',
 *   metadata: { stage: 'encoder' },
 * });
 * readoutBlock.describe({
 *   label: 'readoutBlock',
 *   metadata: { stage: 'readout' },
 * });
 *
 * sensorBlock.connect(
 *   readoutBlock,
 *   methods.groupConnection.ALL_TO_ALL,
 * );
 * ```
 */
export default class Group {
  /** An array holding all the nodes within this group. */
  nodes: Node[];
  /**
   * Stores connection information related to this group.
   * `in`: Connections coming into nodes in this group from outside.
   * `out`: Connections going out from nodes in this group to outside.
   * `self`: Connections between nodes within this same group.
   */
  connections: {
    in: Connection[];
    out: Connection[];
    self: Connection[];
  };
  /** Optional human-readable descriptor label for architecture tooling. */
  label: string | null;
  /** Optional semantic intent for architecture tooling and diagnostics. */
  intent: PrimitiveIntent | null;
  /** Optional scalar metadata retained on the primitive boundary. */
  metadata: PrimitiveMetadata;

  /**
   * Creates a new group comprised of a specified number of nodes.
   *
   * @param size The quantity of nodes to initialize within this group.
   * @param nodeType Optional primitive role assigned to each allocated node.
   * @returns A live group whose nodes can be wired into larger graph structures.
   */
  constructor(size: number, nodeType: PrimitiveNodeType = 'hidden') {
    this.nodes = [];
    this.connections = {
      in: [],
      out: [],
      self: [],
    };
    this.label = null;
    this.intent = resolvePrimitiveIntent(nodeType);
    this.metadata = { size };

    for (let nodeIndex = 0; nodeIndex < size; nodeIndex++) {
      this.nodes.push(new Node(nodeType));
    }
  }

  /**
   * Attaches optional descriptor metadata to the group boundary.
   *
   * Use this when a group represents a named stage, gate bundle, or other
   * meaningful architecture unit that later diagnostics should recognize
   * without inferring from node order alone.
   *
   * @param descriptor Optional label, intent, and scalar metadata to merge.
   * @returns Nothing.
   *
   * @example
   * ```ts
   * const forgetGate = new Group(8);
   *
   * forgetGate.describe({
   *   label: 'forgetGate',
   *   intent: 'gate',
   *   metadata: { family: 'lstm' },
   * });
   * ```
   */
  describe(descriptor: PrimitiveDescriptor): void {
    if (descriptor.label !== undefined) {
      this.label = descriptor.label;
    }

    if (descriptor.intent !== undefined) {
      this.intent = descriptor.intent;
    }

    if (descriptor.metadata !== undefined) {
      this.metadata = {
        ...this.metadata,
        ...descriptor.metadata,
      };
    }
  }

  /**
   * Activates all nodes in the group.
   *
   * @param value Optional array of input values. Its length must match the number of nodes in the group.
   * @returns Activation value of each node in the group, in order.
   * @throws {Error} If the `value` array length does not match the node count.
   */
  activate(value?: number[]): number[] {
    const values: number[] = [];

    if (value !== undefined && value.length !== this.nodes.length) {
      throw new GroupSizeMismatchError(
        'Array with values should be same as the amount of nodes!',
      );
    }

    for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++) {
      const activation =
        value === undefined
          ? this.nodes[nodeIndex].activate()
          : this.nodes[nodeIndex].activate(value[nodeIndex]);
      values.push(activation);
    }

    return values;
  }

  /**
   * Propagates the error backward through all nodes in the group.
   *
   * @param rate Learning rate to apply during weight updates.
   * @param momentum Momentum factor to apply during weight updates.
   * @param target Optional target values for error calculation. Its length must match the number of nodes.
   * @returns Nothing.
   * @throws {Error} If the `target` array length does not match the node count.
   */
  propagate(rate: number, momentum: number, target?: number[]): void {
    if (target !== undefined && target.length !== this.nodes.length) {
      throw new GroupSizeMismatchError(
        'Array with values should be same as the amount of nodes!',
      );
    }

    for (let nodeIndex = this.nodes.length - 1; nodeIndex >= 0; nodeIndex--) {
      if (target === undefined) {
        this.nodes[nodeIndex].propagate(rate, momentum, true, 0);
      } else {
        this.nodes[nodeIndex].propagate(
          rate,
          momentum,
          true,
          0,
          target[nodeIndex],
        );
      }
    }
  }

  /**
   * Establishes connections from all nodes in this group to a target group, layer, or node.
   *
   * @param target Destination entity to connect to.
   * @param method Connection pattern to use.
   * @param weight Optional fixed weight for all created connections.
   * @returns All connection objects created during this wiring step.
   * @throws {Error} If `ONE_TO_ONE` is used with groups of different sizes.
   */
  connect(
    target: Group | Layer | Node,
    method?: unknown,
    weight?: number,
  ): Connection[] {
    if (target instanceof Group) {
      const resolvedMethod = resolveGroupConnectionMethod(this, target, method);
      return connectGroupToGroup(this, target, resolvedMethod, weight);
    }

    if (target instanceof Layer) {
      return target.input(this, method, weight);
    }

    if (target instanceof Node) {
      return connectGroupToNode(this, target, weight);
    }

    return [];
  }

  /**
   * Configures nodes within this group to act as gates for the specified connection set.
   *
   * @param connections Single connection or list of connections to gate.
   * @param method Gating mechanism to use.
   * @returns Nothing.
   * @throws {Error} If no gating method is specified.
   */
  gate(connections: Connection | Connection[], method: unknown): void {
    if (method === undefined) {
      throw new GroupGatingMethodRequiredError(
        'Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF',
      );
    }

    const gatedConnections = normalizeGatedConnections(connections);
    const sourceNodes = collectUniqueSourceNodes(gatedConnections);
    const gatedConnectionSet = new Set(gatedConnections);

    switch (method) {
      case methods.gating.INPUT:
        gateInputConnections(this, gatedConnections);
        break;

      case methods.gating.OUTPUT:
        gateOutputConnections(this, sourceNodes, gatedConnectionSet);
        break;

      case methods.gating.SELF:
        gateSelfConnections(this, sourceNodes, gatedConnectionSet);
        break;
    }
  }

  /**
   * Sets specific properties for all nodes within the group.
   *
   * @param values Property values to apply to every node.
   * @returns Nothing.
   */
  set(values: {
    bias?: number;
    squash?: (x: number, derivate?: boolean) => number;
    type?: string;
  }): void {
    for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++) {
      if (values.bias !== undefined) {
        this.nodes[nodeIndex].bias = values.bias;
      }
      if (values.squash !== undefined) {
        this.nodes[nodeIndex].squash = values.squash;
      }
      if (values.type !== undefined) {
        this.nodes[nodeIndex].type = values.type;
        this.nodes[nodeIndex].intent = resolvePrimitiveIntent(values.type);
      }
    }

    if (values.type !== undefined) {
      this.intent = resolvePrimitiveIntent(values.type);
    }
  }

  /**
   * Removes connections between nodes in this group and a target group or node.
   *
   * @param target Group or node to disconnect from.
   * @param twosided Whether to also remove reciprocal connections.
   * @returns Nothing.
   */
  disconnect(target: Group | Node, twosided: boolean = false): void {
    if (target instanceof Group) {
      disconnectGroupFromGroup(this, target, twosided);
      return;
    }

    if (target instanceof Node) {
      disconnectGroupFromNode(this, target, twosided);
    }
  }

  /**
   * Resets the state of all nodes in the group.
   *
   * @returns Nothing.
   */
  clear(): void {
    for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++) {
      this.nodes[nodeIndex].clear();
    }
  }

  /**
   * Serializes the group into a JSON-compatible format, avoiding circular references.
   *
   * @returns JSON-friendly representation with node indices and connection counts.
   */
  toJSON() {
    return {
      size: this.nodes.length,
      nodeIndices: this.nodes.map((node) => node.index),
      connections: {
        in: this.connections.in.length,
        out: this.connections.out.length,
        self: this.connections.self.length,
      },
    };
  }
}

function resolveGroupConnectionMethod(
  sourceGroup: Group,
  targetGroup: Group,
  method: unknown,
): unknown {
  if (method !== undefined) {
    return method;
  }

  if (sourceGroup !== targetGroup) {
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

function connectGroupToGroup(
  sourceGroup: Group,
  targetGroup: Group,
  method: unknown,
  weight?: number,
): Connection[] {
  if (
    method === methods.groupConnection.ALL_TO_ALL ||
    method === methods.groupConnection.ALL_TO_ELSE
  ) {
    return createDenseGroupConnections(
      sourceGroup,
      targetGroup,
      method,
      weight,
    );
  }

  if (method === methods.groupConnection.ONE_TO_ONE) {
    return createOneToOneGroupConnections(sourceGroup, targetGroup, weight);
  }

  return [];
}

function createDenseGroupConnections(
  sourceGroup: Group,
  targetGroup: Group,
  method: unknown,
  weight?: number,
): Connection[] {
  const createdConnections: Connection[] = [];

  for (const sourceNode of sourceGroup.nodes) {
    for (const targetNode of targetGroup.nodes) {
      if (shouldSkipGroupPairConnection(method, sourceNode, targetNode)) {
        continue;
      }

      const connection = connectNodePair(sourceNode, targetNode, weight);
      recordOutboundGroupConnection(sourceGroup, targetGroup, connection);
      createdConnections.push(connection);
    }
  }

  return createdConnections;
}

function shouldSkipGroupPairConnection(
  method: unknown,
  sourceNode: Node,
  targetNode: Node,
): boolean {
  return (
    method === methods.groupConnection.ALL_TO_ELSE && sourceNode === targetNode
  );
}

function createOneToOneGroupConnections(
  sourceGroup: Group,
  targetGroup: Group,
  weight?: number,
): Connection[] {
  if (sourceGroup.nodes.length !== targetGroup.nodes.length) {
    throw new GroupOneToOneSizeMismatchError(
      'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.',
    );
  }

  const createdConnections: Connection[] = [];

  for (let nodeIndex = 0; nodeIndex < sourceGroup.nodes.length; nodeIndex++) {
    const connection = connectNodePair(
      sourceGroup.nodes[nodeIndex],
      targetGroup.nodes[nodeIndex],
      weight,
    );

    if (sourceGroup === targetGroup) {
      sourceGroup.connections.self.push(connection);
    } else {
      recordOutboundGroupConnection(sourceGroup, targetGroup, connection);
    }

    createdConnections.push(connection);
  }

  return createdConnections;
}

function connectGroupToNode(
  sourceGroup: Group,
  targetNode: Node,
  weight?: number,
): Connection[] {
  const createdConnections: Connection[] = [];

  for (const sourceNode of sourceGroup.nodes) {
    const connection = connectNodePair(sourceNode, targetNode, weight);
    sourceGroup.connections.out.push(connection);
    createdConnections.push(connection);
  }

  return createdConnections;
}

function connectNodePair(
  sourceNode: Node,
  targetNode: Node,
  weight?: number,
): Connection {
  return sourceNode.connect(targetNode, weight)[0];
}

function recordOutboundGroupConnection(
  sourceGroup: Group,
  targetGroup: Group,
  connection: Connection,
): void {
  sourceGroup.connections.out.push(connection);
  targetGroup.connections.in.push(connection);
}

function normalizeGatedConnections(
  connections: Connection | Connection[],
): Connection[] {
  return Array.isArray(connections) ? connections : [connections];
}

function collectUniqueSourceNodes(gatedConnections: Connection[]): Node[] {
  return Array.from(
    new Set(gatedConnections.map((connection) => connection.from)),
  );
}

function gateInputConnections(
  group: Group,
  gatedConnections: Connection[],
): void {
  gatedConnections.forEach((connection, connectionIndex) => {
    const gater = group.nodes[connectionIndex % group.nodes.length];
    gater.gate(connection);
  });
}

function gateOutputConnections(
  group: Group,
  sourceNodes: Node[],
  gatedConnectionSet: Set<Connection>,
): void {
  sourceNodes.forEach((sourceNode, sourceNodeIndex) => {
    const gater = group.nodes[sourceNodeIndex % group.nodes.length];

    for (const connection of sourceNode.connections.out) {
      if (gatedConnectionSet.has(connection)) {
        gater.gate(connection);
      }
    }
  });
}

function gateSelfConnections(
  group: Group,
  sourceNodes: Node[],
  gatedConnectionSet: Set<Connection>,
): void {
  sourceNodes.forEach((sourceNode, sourceNodeIndex) => {
    const gater = group.nodes[sourceNodeIndex % group.nodes.length];
    const selfConnection = resolvePrimarySelfConnection(sourceNode);

    if (
      selfConnection !== undefined &&
      gatedConnectionSet.has(selfConnection)
    ) {
      gater.gate(selfConnection);
    }
  });
}

function resolvePrimarySelfConnection(
  sourceNode: Node,
): Connection | undefined {
  const selfConnections = (
    sourceNode as Node & { connections: { self: Connection[] | Connection } }
  ).connections.self;

  return Array.isArray(selfConnections) ? selfConnections[0] : selfConnections;
}

function disconnectGroupFromGroup(
  sourceGroup: Group,
  targetGroup: Group,
  twosided: boolean,
): void {
  for (const sourceNode of sourceGroup.nodes) {
    for (const targetNode of targetGroup.nodes) {
      disconnectGroupPair(
        sourceGroup,
        targetGroup,
        sourceNode,
        targetNode,
        twosided,
      );
    }
  }
}

function disconnectGroupPair(
  sourceGroup: Group,
  targetGroup: Group,
  sourceNode: Node,
  targetNode: Node,
  twosided: boolean,
): void {
  sourceNode.disconnect(targetNode, twosided);
  removeFirstMatchingConnection(
    sourceGroup.connections.out,
    sourceNode,
    targetNode,
  );
  removeFirstMatchingConnection(
    targetGroup.connections.in,
    sourceNode,
    targetNode,
  );

  if (!twosided) {
    return;
  }

  removeFirstMatchingConnection(
    sourceGroup.connections.in,
    targetNode,
    sourceNode,
  );
  removeFirstMatchingConnection(
    targetGroup.connections.out,
    targetNode,
    sourceNode,
  );
}

function disconnectGroupFromNode(
  sourceGroup: Group,
  targetNode: Node,
  twosided: boolean,
): void {
  for (const sourceNode of sourceGroup.nodes) {
    sourceNode.disconnect(targetNode, twosided);
    removeFirstMatchingConnection(
      sourceGroup.connections.out,
      sourceNode,
      targetNode,
    );

    if (twosided) {
      removeFirstMatchingConnection(
        sourceGroup.connections.in,
        targetNode,
        sourceNode,
      );
    }
  }
}

function removeFirstMatchingConnection(
  connections: Connection[],
  fromNode: Node,
  toNode: Node,
): void {
  for (
    let connectionIndex = connections.length - 1;
    connectionIndex >= 0;
    connectionIndex--
  ) {
    const connection = connections[connectionIndex];

    if (connection.from === fromNode && connection.to === toNode) {
      connections.splice(connectionIndex, 1);
      return;
    }
  }
}
