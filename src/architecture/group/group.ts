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
    let connections: Connection[] = [];

    if (target instanceof Group) {
      if (method === undefined) {
        if (this !== target) {
          if (config.warnings) {
            console.warn(
              'No group connection specified, using ALL_TO_ALL by default.',
            );
          }
          method = methods.groupConnection.ALL_TO_ALL;
        } else {
          if (config.warnings) {
            console.warn(
              'Connecting group to itself, using ONE_TO_ONE by default.',
            );
          }
          method = methods.groupConnection.ONE_TO_ONE;
        }
      }

      if (
        method === methods.groupConnection.ALL_TO_ALL ||
        method === methods.groupConnection.ALL_TO_ELSE
      ) {
        for (
          let sourceNodeIndex = 0;
          sourceNodeIndex < this.nodes.length;
          sourceNodeIndex++
        ) {
          for (
            let targetNodeIndex = 0;
            targetNodeIndex < target.nodes.length;
            targetNodeIndex++
          ) {
            if (
              method === methods.groupConnection.ALL_TO_ELSE &&
              this.nodes[sourceNodeIndex] === target.nodes[targetNodeIndex]
            ) {
              continue;
            }

            const connection = this.nodes[sourceNodeIndex].connect(
              target.nodes[targetNodeIndex],
              weight,
            );
            this.connections.out.push(connection[0]);
            target.connections.in.push(connection[0]);
            connections.push(connection[0]);
          }
        }
      } else if (method === methods.groupConnection.ONE_TO_ONE) {
        if (this.nodes.length !== target.nodes.length) {
          throw new GroupOneToOneSizeMismatchError(
            'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.',
          );
        }

        for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++) {
          const connection = this.nodes[nodeIndex].connect(
            target.nodes[nodeIndex],
            weight,
          );
          if (this === target) {
            this.connections.self.push(connection[0]);
          } else {
            this.connections.out.push(connection[0]);
            target.connections.in.push(connection[0]);
          }
          connections.push(connection[0]);
        }
      }
    } else if (target instanceof Layer) {
      connections = target.input(this, method, weight);
    } else if (target instanceof Node) {
      for (let nodeIndex = 0; nodeIndex < this.nodes.length; nodeIndex++) {
        const connection = this.nodes[nodeIndex].connect(target, weight);
        this.connections.out.push(connection[0]);
        connections.push(connection[0]);
      }
    }

    return connections;
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

    const gatedConnections = Array.isArray(connections)
      ? connections
      : [connections];
    const sourceNodes: Node[] = [];

    for (
      let connectionIndex = 0;
      connectionIndex < gatedConnections.length;
      connectionIndex++
    ) {
      const connection = gatedConnections[connectionIndex];
      if (!sourceNodes.includes(connection.from)) {
        sourceNodes.push(connection.from);
      }
    }

    switch (method) {
      case methods.gating.INPUT:
        for (
          let connectionIndex = 0;
          connectionIndex < gatedConnections.length;
          connectionIndex++
        ) {
          const connection = gatedConnections[connectionIndex];
          const gater = this.nodes[connectionIndex % this.nodes.length];
          gater.gate(connection);
        }
        break;

      case methods.gating.OUTPUT:
        for (
          let sourceNodeIndex = 0;
          sourceNodeIndex < sourceNodes.length;
          sourceNodeIndex++
        ) {
          const node = sourceNodes[sourceNodeIndex];
          const gater = this.nodes[sourceNodeIndex % this.nodes.length];

          for (
            let connectionIndex = 0;
            connectionIndex < node.connections.out.length;
            connectionIndex++
          ) {
            const connection = node.connections.out[connectionIndex];
            if (gatedConnections.includes(connection)) {
              gater.gate(connection);
            }
          }
        }
        break;

      case methods.gating.SELF:
        for (
          let sourceNodeIndex = 0;
          sourceNodeIndex < sourceNodes.length;
          sourceNodeIndex++
        ) {
          const node = sourceNodes[sourceNodeIndex];
          const gater = this.nodes[sourceNodeIndex % this.nodes.length];
          const selfConnection = Array.isArray(node.connections.self)
            ? node.connections.self[0]
            : node.connections.self;
          if (gatedConnections.includes(selfConnection)) {
            gater.gate(selfConnection);
          }
        }
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
      for (
        let sourceNodeIndex = 0;
        sourceNodeIndex < this.nodes.length;
        sourceNodeIndex++
      ) {
        for (
          let targetNodeIndex = 0;
          targetNodeIndex < target.nodes.length;
          targetNodeIndex++
        ) {
          this.nodes[sourceNodeIndex].disconnect(
            target.nodes[targetNodeIndex],
            twosided,
          );

          for (
            let connectionIndex = this.connections.out.length - 1;
            connectionIndex >= 0;
            connectionIndex--
          ) {
            const connection = this.connections.out[connectionIndex];
            if (
              connection.from === this.nodes[sourceNodeIndex] &&
              connection.to === target.nodes[targetNodeIndex]
            ) {
              this.connections.out.splice(connectionIndex, 1);
              break;
            }
          }

           for (
             let connectionIndex = target.connections.in.length - 1;
             connectionIndex >= 0;
             connectionIndex--
           ) {
             const connection = target.connections.in[connectionIndex];
             if (
               connection.from === this.nodes[sourceNodeIndex] &&
               connection.to === target.nodes[targetNodeIndex]
             ) {
               target.connections.in.splice(connectionIndex, 1);
               break;
             }
           }

          if (twosided) {
            for (
              let connectionIndex = this.connections.in.length - 1;
              connectionIndex >= 0;
              connectionIndex--
            ) {
              const connection = this.connections.in[connectionIndex];
              if (
                connection.from === target.nodes[targetNodeIndex] &&
                connection.to === this.nodes[sourceNodeIndex]
              ) {
                this.connections.in.splice(connectionIndex, 1);
                break;
              }
            }

            for (
              let connectionIndex = target.connections.out.length - 1;
              connectionIndex >= 0;
              connectionIndex--
            ) {
              const connection = target.connections.out[connectionIndex];
              if (
                connection.from === target.nodes[targetNodeIndex] &&
                connection.to === this.nodes[sourceNodeIndex]
              ) {
                target.connections.out.splice(connectionIndex, 1);
                break;
              }
            }
          }
        }
      }
    } else if (target instanceof Node) {
      for (
        let sourceNodeIndex = 0;
        sourceNodeIndex < this.nodes.length;
        sourceNodeIndex++
      ) {
        this.nodes[sourceNodeIndex].disconnect(target, twosided);

        for (
          let connectionIndex = this.connections.out.length - 1;
          connectionIndex >= 0;
          connectionIndex--
        ) {
          const connection = this.connections.out[connectionIndex];
          if (
            connection.from === this.nodes[sourceNodeIndex] &&
            connection.to === target
          ) {
            this.connections.out.splice(connectionIndex, 1);
            break;
          }
        }

        if (twosided) {
          for (
            let connectionIndex = this.connections.in.length - 1;
            connectionIndex >= 0;
            connectionIndex--
          ) {
            const connection = this.connections.in[connectionIndex];
            if (
              connection.from === target &&
              connection.to === this.nodes[sourceNodeIndex]
            ) {
              this.connections.in.splice(connectionIndex, 1);
              break;
            }
          }
        }
      }
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
