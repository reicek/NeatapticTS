import Connection from './connection';
import { config } from '../config';
import * as methods from '../methods/methods';

/**
 * Represents a node in a neural network.
 */
export default class Node {
  bias: number;
  squash: (x: number, derivate?: boolean) => number;
  type: string;
  activation: number;
  state: number;
  old: number;
  mask: number;
  previousDeltaBias: number;
  totalDeltaBias: number;
  connections: {
    in: Connection[];
    out: Connection[];
    gated: any[];
    self: any;
  };
  error: {
    responsibility: number;
    projected: number;
    gated: number;
  };
  derivative?: number;
  nodes: Node[];
  gates: any[];
  index?: number;

  /**
   * Creates a new node.
   * @param {string} [type='hidden'] - The type of the node (e.g., 'input', 'output', 'hidden').
   */
  constructor(type: string = 'hidden') {
    this.bias = type === 'input' ? 0 : Math.random() * 0.2 - 0.1;
    this.squash = methods.Activation.logistic || ((x) => x); // Default to logistic or identity function
    this.type = type;

    this.activation = 0;
    this.state = 0;
    this.old = 0;

    // For dropout
    this.mask = 1;

    // For tracking momentum
    this.previousDeltaBias = 0;

    // Batch training
    this.totalDeltaBias = 0;

    this.connections = {
      in: [],
      out: [],
      gated: [],
      self: new Connection(this, this, 0),
    };

    // Data for backpropagation
    this.error = {
      responsibility: 0,
      projected: 0,
      gated: 0,
    };

    this.nodes = [];
    this.gates = [];
  }

  /**
   * Activates the node.
   * @param {number} [input] - The input value for the node.
   * @returns {number} The activation value of the node.
   */
  activate(input?: number): number {
    if (typeof input !== 'undefined') {
      this.activation = input;
      return this.activation;
    }

    this.old = this.state;

    this.state =
      this.connections.self.gain * this.connections.self.weight * this.state +
      this.bias;

    for (const connection of this.connections.in) {
      this.state +=
        connection.from.activation * connection.weight * connection.gain;
    }

    this.activation = this.squash(this.state) * this.mask;
    this.derivative = this.squash(this.state, true);

    const nodes: Node[] = [];
    const influences: number[] = [];

    for (const conn of this.connections.gated) {
      const node = conn.to;
      const index = nodes.indexOf(node);
      if (index > -1) {
        influences[index] += conn.weight * conn.from.activation;
      } else {
        nodes.push(node);
        influences.push(
          conn.weight * conn.from.activation +
            (node.connections.self.gater === this ? node.old : 0)
        );
      }
      conn.gain = this.activation;
    }

    for (const connection of this.connections.in) {
      connection.elegibility =
        this.connections.self.gain *
          this.connections.self.weight *
          connection.elegibility +
        connection.from.activation * connection.gain;

      for (let j = 0; j < nodes.length; j++) {
        const node = nodes[j];
        const influence = influences[j];
        const index = connection.xtrace.nodes.indexOf(node);

        if (index > -1) {
          connection.xtrace.values[index] =
            node.connections.self.gain *
              node.connections.self.weight *
              connection.xtrace.values[index] +
            this.derivative! * connection.elegibility * influence;
        } else {
          connection.xtrace.nodes.push(node);
          connection.xtrace.values.push(
            this.derivative! * connection.elegibility * influence
          );
        }
      }
    }

    return this.activation;
  }

  /**
   * Activates the node without calculating eligibility traces.
   * @param {number} [input] - The input value for the node.
   * @returns {number} The activation value of the node.
   */
  noTraceActivate(input?: number): number {
    if (typeof input !== 'undefined') {
      this.activation = input;
      return this.activation;
    }

    this.state =
      this.connections.self.gain * this.connections.self.weight * this.state +
      this.bias;

    for (const connection of this.connections.in) {
      this.state +=
        connection.from.activation * connection.weight * connection.gain;
    }

    this.activation = this.squash(this.state) * this.mask;
    this.derivative = this.squash(this.state, true);

    for (const connection of this.connections.gated) {
      connection.gain = this.activation;
    }

    return this.activation;
  }

  /**
   * Back-propagates the error through the node.
   * @param {number} rate - The learning rate.
   * @param {number} momentum - The momentum factor.
   * @param {boolean} update - Whether to update weights and biases.
   * @param {number} [target] - The target value for output nodes.
   */
  propagate(
    rate: number,
    momentum: number,
    update: boolean,
    target?: number
  ): void {
    momentum = momentum || 0;
    rate = rate || 0.3;

    let error = 0;

    if (this.type === 'output') {
      this.error.responsibility = this.error.projected =
        target! - this.activation;
    } else {
      for (const connection of this.connections.out) {
        error +=
          connection.to.error.responsibility *
          connection.weight *
          connection.gain;
      }

      this.error.projected = this.derivative! * error;

      error = 0;

      for (const connection of this.connections.gated) {
        const node = connection.to;
        let influence = node.connections.self.gater === this ? node.old : 0;

        influence += connection.weight * connection.from.activation;
        error += node.error.responsibility * influence;
      }

      this.error.gated = this.derivative! * error;
      this.error.responsibility = this.error.projected + this.error.gated;
    }

    if (this.type === 'constant') return;

    for (const connection of this.connections.in) {
      let gradient = this.error.projected * connection.elegibility;

      for (let j = 0; j < connection.xtrace.nodes.length; j++) {
        const node = connection.xtrace.nodes[j];
        const value = connection.xtrace.values[j];
        gradient += node.error.responsibility * value;
      }

      const deltaWeight = rate * gradient * this.mask;
      connection.totalDeltaWeight += deltaWeight;
      if (update) {
        connection.totalDeltaWeight +=
          momentum * connection.previousDeltaWeight;
        connection.weight += connection.totalDeltaWeight;
        connection.previousDeltaWeight = connection.totalDeltaWeight;
        connection.totalDeltaWeight = 0;
      }
    }

    const deltaBias = rate * this.error.responsibility;
    this.totalDeltaBias += deltaBias;
    if (update) {
      this.totalDeltaBias += momentum * this.previousDeltaBias;
      this.bias += this.totalDeltaBias;
      this.previousDeltaBias = this.totalDeltaBias;
      this.totalDeltaBias = 0;
    }
  }

  /**
   * Converts the node to a JSON object.
   * @returns {object} The JSON representation of the node.
   */
  toJSON(): {
    bias: number;
    type: string;
    squash: string | null;
    mask: number;
  } {
    return {
      bias: this.bias,
      type: this.type,
      squash: this.squash ? this.squash.name : null,
      mask: this.mask,
    };
  }

  /**
   * Creates a node from a JSON object.
   * @param {object} json - The JSON object.
   * @returns {Node} The created node.
   */
  static fromJSON(json: {
    bias: number;
    type: string;
    squash: string;
    mask: number;
  }): Node {
    const node = new Node(json.type);
    node.bias = json.bias;
    node.mask = json.mask;
    const squashFunction =
      methods.Activation[json.squash as keyof typeof methods.Activation];
    if (typeof squashFunction === 'function') {
      node.squash = squashFunction as (x: number, derivate?: boolean) => number;
    } else {
      throw new Error(`Invalid squash function: ${json.squash}`);
    }
    return node;
  }

  /**
   * Mutates the node using a given method.
   * @param {any} method - The mutation method.
   */
  mutate(method: any): void {
    if (!method) {
      throw new Error('No mutate method given!');
    } else if (!(method.name in methods.mutation)) {
      throw new Error('This method does not exist!');
    }

    switch (method) {
      case methods.mutation.MOD_ACTIVATION:
        const allowed = method.allowed;
        const index =
          (allowed.indexOf(this.squash) +
            Math.floor(Math.random() * (allowed.length - 1)) +
            1) %
          allowed.length;
        this.squash = allowed[index];
        break;
      case methods.mutation.MOD_BIAS:
        const modification =
          Math.random() * (method.max - method.min) + method.min;
        this.bias += modification;
        break;
    }
  }

  /**
   * Connects the node to a target node or group.
   * @param {Node | { nodes: Node[] }} target - The target node or group.
   * @param {number} [weight] - The weight of the connection.
   * @returns {Connection[]} The connection(s) created.
   */
  connect(target: Node | { nodes: Node[] }, weight?: number): Connection[] {
    const connections: Connection[] = [];
    if ('bias' in target) {
      if (target === this) {
        if (this.connections.self.weight !== 0) {
          throw new Error('This connection already exists!');
        }
        this.connections.self.weight = weight || 1;
        connections.push(this.connections.self);
      } else {
        const connection = new Connection(this, target, weight);
        target.connections.in.push(connection);
        this.connections.out.push(connection);
        connections.push(connection);
      }
    } else {
      for (const node of target.nodes) {
        const connection = new Connection(this, node, weight);
        node.connections.in.push(connection);
        this.connections.out.push(connection);
        connections.push(connection);
      }
    }
    return connections;
  }

  /**
   * Disconnects the node from a target node.
   * @param {Node} target - The target node.
   * @param {boolean} [twosided] - Whether to disconnect both sides.
   */
  disconnect(target: Node, twosided?: boolean): void {
    if (this === target) {
      this.connections.self.weight = 0;
      return;
    }

    this.connections.out = this.connections.out.filter((conn) => {
      if (conn.to === target) {
        target.connections.in = target.connections.in.filter(
          (inConn) => inConn !== conn
        );
        if (conn.gater) conn.gater.ungate(conn);
        return false;
      }
      return true;
    });

    if (twosided) {
      target.disconnect(this);
    }
  }

  /**
   * Gates a connection.
   * @param {Connection | Connection[]} connections - The connection(s) to gate.
   */
  gate(connections: Connection | Connection[]): void {
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    for (const connection of connections) {
      this.connections.gated.push(connection);
      connection.gater = this;
    }
  }

  /**
   * Removes the gates from this node from the given connection(s).
   * @param {Connection | Connection[]} connections - The connection(s) to ungate.
   */
  ungate(connections: Connection | Connection[]): void {
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    for (const connection of connections) {
      const index = this.connections.gated.indexOf(connection);
      if (index !== -1) {
        this.connections.gated.splice(index, 1);
        connection.gater = null;
        connection.gain = 1;
      }
    }
  }

  /**
   * Clears the node's state and connections.
   */
  clear(): void {
    for (const connection of this.connections.in) {
      connection.elegibility = 0;
      connection.xtrace = { nodes: [], values: [] };
    }

    for (const connection of this.connections.gated) {
      connection.gain = 0;
    }

    this.error = { responsibility: 0, projected: 0, gated: 0 };
    this.old = this.state = this.activation = 0;
  }

  /**
   * Checks if this node is projecting to the given node.
   * @param {Node} node - The target node.
   * @returns {boolean} True if projecting, false otherwise.
   */
  isProjectingTo(node: Node): boolean {
    if (node === this && this.connections.self.weight !== 0) return true;

    return this.connections.out.some((conn) => conn.to === node);
  }

  /**
   * Checks if the given node is projecting to this node.
   * @param {Node} node - The source node.
   * @returns {boolean} True if projected, false otherwise.
   */
  isProjectedBy(node: Node): boolean {
    if (node === this && this.connections.self.weight !== 0) return true;

    return this.connections.in.some((conn) => conn.from === node);
  }
}
