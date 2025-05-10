import Node from './node';
import Layer from './layer';
import { config } from '../config';
import * as methods from '../methods/methods';

/**
 * Represents a collection of nodes functioning as a single unit within a network architecture.
 * Groups facilitate operations like collective activation, propagation, and connection management.
 */
export default class Group {
  /**
   * An array holding all the nodes within this group.
   */
  nodes: Node[];
  /**
   * Stores connection information related to this group.
   * `in`: Connections coming into any node in this group from outside.
   * `out`: Connections going out from any node in this group to outside.
   * `self`: Connections between nodes within this same group (e.g., in ONE_TO_ONE connections).
   */
  connections: {
    in: any[]; // Consider using a more specific type like `Connection[]` if available
    out: any[]; // Consider using a more specific type like `Connection[]` if available
    self: any[]; // Consider using a more specific type like `Connection[]` if available
  };

  /**
   * Creates a new group comprised of a specified number of nodes.
   * @param {number} size - The quantity of nodes to initialize within this group.
   */
  constructor(size: number) {
    this.nodes = [];
    this.connections = {
      in: [],
      out: [],
      self: [],
    };

    for (let i = 0; i < size; i++) {
      this.nodes.push(new Node());
    }
  }

  /**
   * Activates all nodes in the group. If input values are provided, they are assigned
   * sequentially to the nodes before activation. Otherwise, nodes activate based on their
   * existing states and incoming connections.
   *
   * @param {number[]} [value] - An optional array of input values. If provided, its length must match the number of nodes in the group.
   * @returns {number[]} An array containing the activation value of each node in the group, in order.
   * @throws {Error} If the `value` array is provided and its length does not match the number of nodes in the group.
   */
  activate(value?: number[]): number[] {
    const values: number[] = [];

    if (value !== undefined && value.length !== this.nodes.length) {
      throw new Error(
        'Array with values should be same as the amount of nodes!'
      );
    }

    for (let i = 0; i < this.nodes.length; i++) {
      const activation =
        value === undefined
          ? this.nodes[i].activate()
          : this.nodes[i].activate(value[i]);
      values.push(activation);
    }

    return values;
  }

  /**
   * Propagates the error backward through all nodes in the group. If target values are provided,
   * the error is calculated against these targets (typically for output layers). Otherwise,
   * the error is calculated based on the error propagated from subsequent layers/nodes.
   *
   * @param {number} rate - The learning rate to apply during weight updates.
   * @param {number} momentum - The momentum factor to apply during weight updates.
   * @param {number[]} [target] - Optional target values for error calculation. If provided, its length must match the number of nodes.
   * @throws {Error} If the `target` array is provided and its length does not match the number of nodes in the group.
   */
  propagate(rate: number, momentum: number, target?: number[]): void {
    if (target !== undefined && target.length !== this.nodes.length) {
      throw new Error(
        'Array with values should be same as the amount of nodes!'
      );
    }

    for (let i = this.nodes.length - 1; i >= 0; i--) {
      if (target === undefined) {
        this.nodes[i].propagate(rate, momentum, true);
      } else {
        this.nodes[i].propagate(rate, momentum, true, target[i]);
      }
    }
  }

  /**
   * Establishes connections from all nodes in this group to a target Group, Layer, or Node.
   * The connection pattern (e.g., all-to-all, one-to-one) can be specified.
   *
   * @param {Group | Layer | Node} target - The destination entity (Group, Layer, or Node) to connect to.
   * @param {methods.groupConnection | methods.connection} [method] - The connection method/type (e.g., `methods.groupConnection.ALL_TO_ALL`, `methods.groupConnection.ONE_TO_ONE`). Defaults depend on the target type and whether it's the same group.
   * @param {number} [weight] - An optional fixed weight to assign to all created connections. If not provided, weights might be initialized randomly or based on node defaults.
   * @returns {any[]} An array containing all the connection objects created. Consider using a more specific type like `Connection[]`.
   * @throws {Error} If `methods.groupConnection.ONE_TO_ONE` is used and the source and target groups have different sizes.
   */
  connect(target: Group | Layer | Node, method?: any, weight?: number): any[] {
    let connections: any[] = [];
    let i, j;

    // Connection to another Group
    if (target instanceof Group) {
      // Determine default connection method if none is provided
      if (method === undefined) {
        if (this !== target) {
          // Default to ALL_TO_ALL if connecting to a different group
          if (config.warnings)
            console.warn(
              'No group connection specified, using ALL_TO_ALL by default.'
            );
          method = methods.groupConnection.ALL_TO_ALL;
        } else {
          // Default to ONE_TO_ONE if connecting to the same group (self-connection)
          if (config.warnings)
            console.warn(
              'Connecting group to itself, using ONE_TO_ONE by default.'
            );
          method = methods.groupConnection.ONE_TO_ONE;
        }
      }
      // Handle ALL_TO_ALL and ALL_TO_ELSE connection methods
      if (
        method === methods.groupConnection.ALL_TO_ALL ||
        method === methods.groupConnection.ALL_TO_ELSE
      ) {
        // Iterate over each node in the source group
        for (i = 0; i < this.nodes.length; i++) {
          // Iterate over each node in the target group
          for (j = 0; j < target.nodes.length; j++) {
            // Skip self-connection if method is ALL_TO_ELSE
            if (
              method === methods.groupConnection.ALL_TO_ELSE &&
              this.nodes[i] === target.nodes[j]
            )
              continue;
            // Create connection from source node to target node
            let connection = this.nodes[i].connect(target.nodes[j], weight);
            // Store the outgoing connection reference in the source group
            this.connections.out.push(connection[0]);
            // Store the incoming connection reference in the target group
            target.connections.in.push(connection[0]);
            // Add the created connection to the list of connections returned by this method
            connections.push(connection[0]);
          }
        }
        // Handle ONE_TO_ONE connection method
      } else if (method === methods.groupConnection.ONE_TO_ONE) {
        // Ensure groups are the same size for ONE_TO_ONE connection
        if (this.nodes.length !== target.nodes.length) {
          throw new Error(
            'Cannot create ONE_TO_ONE connection: source and target groups must have the same size.'
          );
        }

        // Iterate and connect corresponding nodes
        for (i = 0; i < this.nodes.length; i++) {
          let connection = this.nodes[i].connect(target.nodes[i], weight);
          if (this === target) {
            // Store self-connections (within the group)
            this.connections.self.push(connection[0]);
          } else {
            // Store connections between different groups
            this.connections.out.push(connection[0]);
            target.connections.in.push(connection[0]);
          }
          connections.push(connection[0]);
        }
      }
      // Connection to a Layer (delegates to the Layer's input method)
    } else if (target instanceof Layer) {
      connections = target.input(this, method, weight);
      // Connection to a single Node
    } else if (target instanceof Node) {
      // Connect every node in this group to the target node
      for (i = 0; i < this.nodes.length; i++) {
        let connection = this.nodes[i].connect(target, weight);
        // Store outgoing connections
        this.connections.out.push(connection[0]);
        connections.push(connection[0]);
      }
    }

    return connections;
  }

  /**
   * Configures nodes within this group to act as gates for the specified connection(s).
   * Gating allows the output of a node in this group to modulate the flow of signal through the gated connection.
   *
   * @param {any | any[]} connections - A single connection object or an array of connection objects to be gated. Consider using a more specific type like `Connection | Connection[]`.
   * @param {methods.gating} method - The gating mechanism to use (e.g., `methods.gating.INPUT`, `methods.gating.OUTPUT`, `methods.gating.SELF`). Specifies which part of the connection is influenced by the gater node.
   * @throws {Error} If no gating `method` is specified.
   */
  gate(connections: any | any[], method: any): void {
    if (method === undefined) {
      throw new Error(
        'Please specify a gating method: Gating.INPUT, Gating.OUTPUT, or Gating.SELF'
      );
    }

    // Ensure connections is an array for uniform processing
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    // Collect unique source (from) and target (to) nodes from the connections to be gated
    const nodes1: Node[] = []; // Source nodes
    const nodes2: Node[] = []; // Target nodes

    let i, j;
    for (i = 0; i < connections.length; i++) {
      const connection = connections[i];
      if (!nodes1.includes(connection.from)) nodes1.push(connection.from);
      if (!nodes2.includes(connection.to)) nodes2.push(connection.to);
    }

    switch (method) {
      // Gate the input to the target node(s) of the connection(s)
      case methods.gating.INPUT:
        for (let i = 0; i < connections.length; i++) {
          const conn = connections[i];
          const gater = this.nodes[i % this.nodes.length];
          gater.gate(conn);
        }
        break;

      // Gate the output from the source node(s) of the connection(s)
      case methods.gating.OUTPUT:
        for (i = 0; i < nodes1.length; i++) {
          let node = nodes1[i]; // Source node of a connection
          // Select a gater node from this group
          let gater = this.nodes[i % this.nodes.length];

          // Find outgoing connections from the source node that are in the provided list
          for (j = 0; j < node.connections.out.length; j++) {
            let conn = node.connections.out[j];
            if (connections.includes(conn)) {
              // Apply gating from the selected gater node to this connection
              gater.gate(conn);
            }
          }
        }
        break;

      // Gate the self-connection of the node(s) involved
      case methods.gating.SELF:
        for (i = 0; i < nodes1.length; i++) {
          let node = nodes1[i]; // Node with the self-connection
          let gater = this.nodes[i % this.nodes.length];
          // Get the actual self-connection object (first element)
          const selfConn = Array.isArray(node.connections.self)
            ? node.connections.self[0]
            : node.connections.self;
          if (connections.includes(selfConn)) {
            gater.gate(selfConn);
          }
        }
        break;
    }
  }

  /**
   * Sets specific properties (like bias, squash function, or type) for all nodes within the group.
   *
   * @param {{ bias?: number; squash?: any; type?: string }} values - An object containing the properties and their new values. Only provided properties are updated.
   *        `bias`: Sets the bias term for all nodes.
   *        `squash`: Sets the activation function (squashing function) for all nodes.
   *        `type`: Sets the node type (e.g., 'input', 'hidden', 'output') for all nodes.
   */
  set(values: { bias?: number; squash?: any; type?: string }): void {
    for (let i = 0; i < this.nodes.length; i++) {
      if (values.bias !== undefined) {
        this.nodes[i].bias = values.bias;
      }
      this.nodes[i].squash = values.squash || this.nodes[i].squash;
      this.nodes[i].type = values.type || this.nodes[i].type;
    }
  }

  /**
   * Removes connections between nodes in this group and a target Group or Node.
   *
   * @param {Group | Node} target - The Group or Node to disconnect from.
   * @param {boolean} [twosided=false] - If true, also removes connections originating from the `target` and ending in this group. Defaults to false (only removes connections from this group to the target).
   */
  disconnect(target: Group | Node, twosided: boolean = false): void {
    let i, j, k;

    // Disconnecting from another Group
    if (target instanceof Group) {
      // Iterate through nodes in this group
      for (i = 0; i < this.nodes.length; i++) {
        // Iterate through nodes in the target group
        for (j = 0; j < target.nodes.length; j++) {
          // Disconnect individual nodes (handles internal node connection state)
          this.nodes[i].disconnect(target.nodes[j], twosided);

          // Remove the connection reference from this group's outgoing connections list
          for (k = this.connections.out.length - 1; k >= 0; k--) {
            let conn = this.connections.out[k];
            if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
              this.connections.out.splice(k, 1);
              break; // Assume only one connection between two specific nodes
            }
          }

          // If twosided, also remove the reverse connection references from group lists
          if (twosided) {
            // Remove from this group's incoming list
            for (k = this.connections.in.length - 1; k >= 0; k--) {
              let conn = this.connections.in[k];
              if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                this.connections.in.splice(k, 1);
                break; // Assume only one connection
              }
            }
            // Remove from target group's outgoing list
            for (k = target.connections.out.length - 1; k >= 0; k--) {
              let conn = target.connections.out[k];
              if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                target.connections.out.splice(k, 1);
                break; // Assume only one connection
              }
            }
            // Remove from target group's incoming list (forward connection)
            for (k = target.connections.in.length - 1; k >= 0; k--) {
              let conn = target.connections.in[k];
              if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
                target.connections.in.splice(k, 1);
                break; // Assume only one connection
              }
            }
          }
        }
      }
      // Disconnecting from a single Node
    } else if (target instanceof Node) {
      // Iterate through nodes in this group
      for (i = 0; i < this.nodes.length; i++) {
        // Disconnect the node in this group from the target node
        this.nodes[i].disconnect(target, twosided);

        // Remove the connection reference from this group's outgoing connections list
        for (j = this.connections.out.length - 1; j >= 0; j--) {
          let conn = this.connections.out[j];
          if (conn.from === this.nodes[i] && conn.to === target) {
            this.connections.out.splice(j, 1);
            break; // Assume only one connection
          }
        }

        // If twosided, also remove the connection reference from this group's incoming connections list
        if (twosided) {
          for (j = this.connections.in.length - 1; j >= 0; j--) {
            const conn = this.connections.in[j];
            if (conn.from === target && conn.to === this.nodes[i]) {
              this.connections.in.splice(j, 1);
              break; // Assume only one connection
            }
          }
        }
      }
    }
  }

  /**
   * Resets the state of all nodes in the group. This typically involves clearing
   * activation values, state, and propagated errors, preparing the group for a new input pattern,
   * especially relevant in recurrent networks or sequence processing.
   */
  clear(): void {
    for (let i = 0; i < this.nodes.length; i++) {
      this.nodes[i].clear();
    }
  }

  /**
   * Serializes the group into a JSON-compatible format, avoiding circular references.
   * Only includes node indices and connection counts.
   *
   * @returns {object} A JSON-compatible representation of the group.
   */
  toJSON() {
    return {
      size: this.nodes.length,
      nodeIndices: this.nodes.map(n => n.index),
      connections: {
        in: this.connections.in.length,
        out: this.connections.out.length,
        self: this.connections.self.length
      }
    };
  }
}
