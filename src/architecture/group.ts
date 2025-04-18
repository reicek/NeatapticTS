import Node from './node';
import Layer from './layer';
import { config } from '../config';
import * as methods from '../methods/methods';

/**
 * Represents a group of nodes in a neural network.
 */
export default class Group {
  nodes: Node[];
  connections: {
    in: any[];
    out: any[];
    self: any[];
  };

  /**
   * Creates a new group of nodes.
   * @param {number} size - The number of nodes in the group.
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
   * Activates all the nodes in the group.
   * @param {number[]} [value] - Optional input values for the nodes. Must match the number of nodes in the group.
   * @returns {number[]} The activation values of the nodes.
   * @throws {Error} If the input array length does not match the number of nodes.
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
   * Propagates all the nodes in the group.
   * @param {number} rate - The learning rate.
   * @param {number} momentum - The momentum factor.
   * @param {number[]} [target] - Optional target values for the nodes. Must match the number of nodes in the group.
   * @throws {Error} If the target array length does not match the number of nodes.
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
   * Connects the nodes in this group to another group, layer, or node.
   * @param {Group | Layer | Node} target - The target to connect to.
   * @param {any} [method] - The connection method. Defaults to ALL_TO_ALL or ONE_TO_ONE based on the target.
   * @param {number} [weight] - The weight of the connection.
   * @returns {any[]} The created connections.
   */
  connect(target: Group | Layer | Node, method?: any, weight?: number): any[] {
    let connections: any[] = [];
    let i, j;

    if (target instanceof Group) {
      if (method === undefined) {
        if (this !== target) {
          if (config.warnings)
            console.warn('No group connection specified, using ALL_TO_ALL');
          method = methods.connection.ALL_TO_ALL;
        } else {
          if (config.warnings)
            console.warn('No group connection specified, using ONE_TO_ONE');
          method = methods.connection.ONE_TO_ONE;
        }
      }
      if (
        method === methods.connection.ALL_TO_ALL ||
        method === methods.connection.ALL_TO_ELSE
      ) {
        for (i = 0; i < this.nodes.length; i++) {
          for (j = 0; j < target.nodes.length; j++) {
            if (
              method === methods.connection.ALL_TO_ELSE &&
              this.nodes[i] === target.nodes[j]
            )
              continue;
            let connection = this.nodes[i].connect(target.nodes[j], weight);
            this.connections.out.push(connection[0]);
            target.connections.in.push(connection[0]);
            connections.push(connection[0]);
          }
        }
      } else if (method === methods.connection.ONE_TO_ONE) {
        if (this.nodes.length !== target.nodes.length) {
          throw new Error('From and To group must be the same size!');
        }

        for (i = 0; i < this.nodes.length; i++) {
          let connection = this.nodes[i].connect(target.nodes[i], weight);
          this.connections.self.push(connection[0]);
          connections.push(connection[0]);
        }
      }
    } else if (target instanceof Layer) {
      connections = target.input(this, method, weight);
    } else if (target instanceof Node) {
      for (i = 0; i < this.nodes.length; i++) {
        let connection = this.nodes[i].connect(target, weight);
        this.connections.out.push(connection[0]);
        connections.push(connection[0]);
      }
    }

    return connections;
  }

  /**
   * Makes nodes from this group gate the given connection(s).
   * @param {any | any[]} connections - The connection(s) to gate.
   * @param {any} method - The gating method. Must be one of Gating.INPUT, Gating.OUTPUT, or Gating.SELF.
   * @throws {Error} If no gating method is specified.
   */
  gate(connections: any | any[], method: any): void {
    if (method === undefined) {
      throw new Error('Please specify Gating.INPUT, Gating.OUTPUT');
    }

    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    const nodes1: Node[] = [];
    const nodes2: Node[] = [];

    let i, j;
    for (i = 0; i < connections.length; i++) {
      const connection = connections[i];
      if (!nodes1.includes(connection.from)) nodes1.push(connection.from);
      if (!nodes2.includes(connection.to)) nodes2.push(connection.to);
    }

    switch (method) {
      /* Input */
      case methods.gating.INPUT:
        for (i = 0; i < nodes2.length; i++) {
          const node = nodes2[i];
          const gater = this.nodes[i % this.nodes.length];

          for (j = 0; j < node.connections.in.length; j++) {
            const conn = node.connections.in[j];

            if (connections.includes(conn)) {
              gater.gate(conn);
            }
          }
        }
        break;

      /* Output */
      case methods.gating.OUTPUT:
        for (i = 0; i < nodes1.length; i++) {
          let node = nodes1[i];
          let gater = this.nodes[i % this.nodes.length];

          for (j = 0; j < node.connections.out.length; j++) {
            let conn = node.connections.out[j];
            if (connections.includes(conn)) {
              gater.gate(conn);
            }
          }
        }
        break;

      /* Self */
      case methods.gating.SELF:
        for (i = 0; i < nodes1.length; i++) {
          let node = nodes1[i];
          let gater = this.nodes[i % this.nodes.length];

          if (connections.includes(node.connections.self)) {
            gater.gate(node.connections.self);
          }
        }
    }
  }

  /**
   * Sets the value of a property for every node in the group.
   * @param {{ bias?: number; squash?: any; type?: string }} values - The values to set for the nodes.
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
   * Disconnects all nodes in this group from another group or node.
   * @param {Group | Node} target - The target to disconnect from.
   * @param {boolean} [twosided=false] - Whether to disconnect both ways.
   */
  disconnect(target: Group | Node, twosided: boolean = false): void {
    let i, j, k;

    /* If Group */
    if (target instanceof Group) {
      for (i = 0; i < this.nodes.length; i++) {
        for (j = 0; j < target.nodes.length; j++) {
          this.nodes[i].disconnect(target.nodes[j], twosided);

          for (k = this.connections.out.length - 1; k >= 0; k--) {
            let conn = this.connections.out[k];

            if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
              this.connections.out.splice(k, 1);
              break;
            }
          }

          if (twosided) {
            for (k = this.connections.in.length - 1; k >= 0; k--) {
              let conn = this.connections.in[k];

              if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                this.connections.in.splice(k, 1);
                break;
              }
            }
          }
        }
      }
      /* If Node */
    } else if (target instanceof Node) {
      for (i = 0; i < this.nodes.length; i++) {
        this.nodes[i].disconnect(target, twosided);

        for (j = this.connections.out.length - 1; j >= 0; j--) {
          let conn = this.connections.out[j];

          if (conn.from === this.nodes[i] && conn.to === target) {
            this.connections.out.splice(j, 1);
            break;
          }
        }

        if (twosided) {
          for (j = this.connections.in.length - 1; j >= 0; j--) {
            const conn = this.connections.in[j];

            if (conn.from === target && conn.to === this.nodes[i]) {
              this.connections.in.splice(j, 1);
              break;
            }
          }
        }
      }
    }
  }

  /**
   * Clears the context of all nodes in the group.
   */
  clear(): void {
    for (let i = 0; i < this.nodes.length; i++) {
      this.nodes[i].clear();
    }
  }
}
