import Node from './node';
import Group from './group';
import * as methods from '../methods/methods';

/**
 * Represents a layer in a neural network.
 *
 * A layer is composed of nodes and can connect to other layers, groups, or nodes.
 * Layers support various architectures such as dense, LSTM, GRU, and memory layers.
 */
export default class Layer {
  nodes: Node[];
  connections: { in: any[]; out: any[]; self: any[] };
  output: Group | null;

  constructor() {
    this.output = null;
    this.nodes = [];
    this.connections = { in: [], out: [], self: [] };
  }

  /**
   * Activates all the nodes in the layer.
   *
   * @param {number[]} [value] - Optional array of input values.
   * @returns {number[]} Array of activation values.
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
      let activation: number;
      if (value === undefined) {
        activation = this.nodes[i].activate();
      } else {
        activation = this.nodes[i].activate(value[i]);
      }

      values.push(activation);
    }

    return values;
  }

  /**
   * Propagates errors backward through the layer.
   *
   * @param {number} rate - Learning rate.
   * @param {number} momentum - Momentum factor.
   * @param {number[]} [target] - Optional array of target values.
   * @throws {Error} If the target array length does not match the number of nodes.
   */
  propagate(rate: number, momentum: number, target?: number[]) {
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
   * Connects this layer to a target group, node, or layer.
   *
   * @param {Group | Node | Layer} target - The target to connect to.
   * @param {any} [method] - Connection method (e.g., ALL_TO_ALL).
   * @param {number} [weight] - Optional weight for the connections.
   * @returns {any[]} Array of created connections.
   */
  connect(target: Group | Node | Layer, method?: any, weight?: number): any[] {
    if (!this.output) {
      throw new Error('Layer output is not defined.');
    }

    let connections: any[] = [];
    if (target instanceof Group || target instanceof Node) {
      connections = this.output.connect(target, method, weight);
    } else if (target instanceof Layer) {
      connections = target.input(this, method, weight);
    }

    return connections;
  }

  /**
   * Gates the specified connections using a gating method.
   *
   * @param {any[]} connections - The connections to gate.
   * @param {any} method - The gating method (e.g., INPUT, OUTPUT, SELF).
   */
  gate(connections: any[], method: any) {
    this.output!.gate(connections, method);
  }

  /**
   * Sets the value of a property for every node in the layer.
   *
   * @param {{ bias?: number; squash?: any; type?: string }} values - Object containing properties to set.
   */
  set(values: { bias?: number; squash?: any; type?: string }) {
    for (let i = 0; i < this.nodes.length; i++) {
      let node = this.nodes[i];

      if (node instanceof Node) {
        if (values.bias !== undefined) {
          node.bias = values.bias;
        }
        node.squash = values.squash || node.squash;
        node.type = values.type || node.type;
      } else if (this.isGroup(node)) {
        (node as Group).set(values);
      }
    }
  }

  /**
   * Disconnects this layer from a target group or node.
   *
   * @param {Group | Node} target - The target to disconnect from.
   * @param {boolean} [twosided=false] - Whether to disconnect both ways.
   */
  disconnect(target: Group | Node, twosided?: boolean) {
    twosided = twosided || false;

    let i, j, k;
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
          for (k = this.connections.in.length - 1; k >= 0; k--) {
            let conn = this.connections.in[k];

            if (conn.from === target && conn.to === this.nodes[i]) {
              this.connections.in.splice(k, 1);
              break;
            }
          }
        }
      }
    }
  }

  /**
   * Clears the state of all nodes in the layer.
   */
  clear() {
    for (let i = 0; i < this.nodes.length; i++) {
      this.nodes[i].clear();
    }
  }

  /**
   * Connects the input of this layer to another layer or group.
   *
   * @param {Layer | Group} from - The source layer or group.
   * @param {any} [method] - Connection method (e.g., ALL_TO_ALL).
   * @param {number} [weight] - Optional weight for the connections.
   * @returns {any[]} Array of created connections.
   */
  input(from: Layer | Group, method?: any, weight?: number): any {
    if (from instanceof Layer) from = from.output!;
    method = method || methods.connection.ALL_TO_ALL;
    return from.connect(this.output!, method, weight);
  }

  /**
   * Creates a dense layer with the specified number of nodes.
   *
   * @param {number} size - Number of nodes in the layer.
   * @returns {Layer} The created dense layer.
   */
  static dense(size: number): Layer {
    const layer = new Layer();

    const block = new Group(size);

    layer.nodes.push(...block.nodes); // Push individual nodes from the group
    layer.output = block;

    layer.input = (from: Layer | Group, method?: any, weight?: number) => {
      if (from instanceof Layer) from = from.output!;
      method = method || methods.connection.ALL_TO_ALL;
      return from.connect(block, method, weight);
    };

    return layer;
  }

  /**
   * Creates a long short-term memory (LSTM) layer.
   *
   * @param {number} size - Number of nodes in the layer.
   * @returns {Layer} The created LSTM layer.
   */
  static lstm(size: number): Layer {
    const layer = new Layer();

    const inputGate = new Group(size);
    const forgetGate = new Group(size);
    const memoryCell = new Group(size);
    const outputGate = new Group(size);
    const outputBlock = new Group(size);

    inputGate.set({
      bias: 1,
    });
    forgetGate.set({
      bias: 1,
    });
    outputGate.set({
      bias: 1,
    });

    memoryCell.connect(inputGate, methods.connection.ALL_TO_ALL);
    memoryCell.connect(forgetGate, methods.connection.ALL_TO_ALL);
    memoryCell.connect(outputGate, methods.connection.ALL_TO_ALL);
    const forget = memoryCell.connect(
      memoryCell,
      methods.connection.ONE_TO_ONE
    );
    const output = memoryCell.connect(
      outputBlock,
      methods.connection.ALL_TO_ALL
    );

    forgetGate.gate(forget, methods.gating.SELF);
    outputGate.gate(output, methods.gating.OUTPUT);

    layer.nodes = [
      ...inputGate.nodes,
      ...forgetGate.nodes,
      ...memoryCell.nodes,
      ...outputGate.nodes,
      ...outputBlock.nodes,
    ];

    layer.output = outputBlock;

    layer.input = (from: Layer | Group, method?: any, weight?: number) => {
      if (from instanceof Layer) from = from.output!;
      method = method || methods.connection.ALL_TO_ALL;
      let connections: any[] = [];

      const input = from.connect(memoryCell, method, weight);
      connections = connections.concat(input);

      connections = connections.concat(from.connect(inputGate, method, weight));
      connections = connections.concat(
        from.connect(outputGate, method, weight)
      );
      connections = connections.concat(
        from.connect(forgetGate, method, weight)
      );

      inputGate.gate(input, methods.gating.INPUT);

      return connections;
    };

    return layer;
  }

  /**
   * Creates a gated recurrent unit (GRU) layer.
   *
   * @param {number} size - Number of nodes in the layer.
   * @returns {Layer} The created GRU layer.
   */
  static gru(size: number): Layer {
    const layer = new Layer();

    const updateGate = new Group(size);
    const inverseUpdateGate = new Group(size);
    const resetGate = new Group(size);
    const memoryCell = new Group(size);
    const output = new Group(size);
    const previousOutput = new Group(size);

    previousOutput.set({
      bias: 0,
      squash: methods.Activation.identity,
      type: 'variant', // Corrected type to match JS
    });

    memoryCell.set({
      squash: methods.Activation.tanh,
    });

    inverseUpdateGate.set({
      bias: 0,
      squash: methods.Activation.inverse,
      type: 'variant', // Corrected type to match JS
    });

    updateGate.set({
      bias: 1,
    });

    resetGate.set({
      bias: 0,
    });

    previousOutput.connect(updateGate, methods.connection.ALL_TO_ALL);

    updateGate.connect(inverseUpdateGate, methods.connection.ONE_TO_ONE, 1);

    previousOutput.connect(resetGate, methods.connection.ALL_TO_ALL);

    const reset = previousOutput.connect(
      memoryCell,
      methods.connection.ALL_TO_ALL
    );

    resetGate.gate(reset, methods.gating.OUTPUT);

    const update1 = previousOutput.connect(
      output,
      methods.connection.ALL_TO_ALL
    );
    const update2 = memoryCell.connect(output, methods.connection.ALL_TO_ALL);

    updateGate.gate(update1, methods.gating.OUTPUT);
    inverseUpdateGate.gate(update2, methods.gating.OUTPUT);

    output.connect(previousOutput, methods.connection.ONE_TO_ONE, 1);

    layer.nodes = [
      ...updateGate.nodes,
      ...inverseUpdateGate.nodes,
      ...resetGate.nodes,
      ...memoryCell.nodes,
      ...output.nodes,
      ...previousOutput.nodes,
    ];

    layer.output = output;

    layer.input = (from: Layer | Group, method?: any, weight?: number) => {
      if (from instanceof Layer) from = from.output!;
      method = method || methods.connection.ALL_TO_ALL;
      let connections: any[] = [];

      connections = connections.concat(
        from.connect(updateGate, method, weight)
      );
      connections = connections.concat(from.connect(resetGate, method, weight));
      connections = connections.concat(
        from.connect(memoryCell, method, weight)
      );

      return connections;
    };

    return layer;
  }

  /**
   * Creates a memory layer with the specified size and memory depth.
   *
   * @param {number} size - Number of nodes in each memory block.
   * @param {number} memory - Number of memory blocks.
   * @returns {Layer} The created memory layer.
   */
  static memory(size: number, memory: number): Layer {
    const layer = new Layer();

    let previous: Group | null = null;
    for (let i = 0; i < memory; i++) {
      const block = new Group(size);

      block.set({
        squash: methods.Activation.identity,
        bias: 0,
        type: 'variant', // Corrected type to match JS
      });

      if (previous != null) {
        previous.connect(block, methods.connection.ONE_TO_ONE, 1);
      }

      layer.nodes.push((block as unknown) as Node);
      previous = block;
    }

    layer.nodes.reverse();

    for (let i = 0; i < layer.nodes.length; i++) {
      layer.nodes[i].nodes.reverse();
    }

    const outputGroup = new Group(0);
    for (const group of layer.nodes) {
      outputGroup.nodes = outputGroup.nodes.concat(group.nodes);
    }

    layer.output = outputGroup;

    layer.input = (from: Layer | Group, method?: any, weight?: number) => {
      if (from instanceof Layer) from = from.output!;
      method = method || methods.connection.ALL_TO_ALL;

      if (
        from.nodes.length !== layer.nodes[layer.nodes.length - 1].nodes.length
      ) {
        throw new Error('Previous layer size must be same as memory size');
      }

      return from.connect(
        layer.nodes[layer.nodes.length - 1],
        methods.connection.ONE_TO_ONE,
        1
      );
    };

    return layer;
  }

  /**
   * Determines if the provided object is a `Group`.
   *
   * A `Group` is identified by having a `set` method and a `nodes` property
   * that is an array.
   *
   * @param obj - The object to check.
   * @returns `true` if the object is a `Group`, otherwise `false`.
   */
  private isGroup(obj: any): obj is Group {
    return obj && typeof obj.set === 'function' && Array.isArray(obj.nodes);
  }
}
