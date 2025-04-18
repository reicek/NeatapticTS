import Node from './node';
import Group from './group';
import * as methods from '../methods/methods';

function isGroup(obj: any): obj is Group {
  return obj && typeof obj.set === 'function' && Array.isArray(obj.nodes);
}

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
   * Activates all the nodes in the group
   * @param value Optional array of input values
   * @returns Array of activation values
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
   * Propagates all the nodes in the group
   * @param rate Learning rate
   * @param momentum Momentum factor
   * @param target Optional array of target values
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

  connect(target: Group | Node | Layer, method?: any, weight?: number): any[] {
    let connections: any[] = []; // Initialize connections
    if (target instanceof Group || target instanceof Node) {
      connections = this.output!.connect(target, method, weight);
    } else if (target instanceof Layer) {
      connections = target.input(this, method, weight);
    }

    return connections;
  }

  gate(connections: any[], method: any) {
    this.output!.gate(connections, method);
  }

  /**
   * Sets the value of a property for every node
   * @param values Object containing properties to set (e.g., bias, squash, type)
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
      } else if (isGroup(node)) {
        (node as Group).set(values);
      }
    }
  }

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

  clear() {
    for (let i = 0; i < this.nodes.length; i++) {
      this.nodes[i].clear();
    }
  }

  input(from: Layer | Group, method?: any, weight?: number): any {
    if (from instanceof Layer) from = from.output!;
    method = method || methods.connection.ALL_TO_ALL;
    return from.connect(this.output!, method, weight);
  }

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
      type: 'constant',
    });

    memoryCell.set({
      squash: methods.Activation.tanh,
    });

    inverseUpdateGate.set({
      bias: 0,
      squash: methods.Activation.inverse,
      type: 'constant',
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
      (updateGate as unknown) as Node,
      (inverseUpdateGate as unknown) as Node,
      (resetGate as unknown) as Node,
      (memoryCell as unknown) as Node,
      (output as unknown) as Node,
      (previousOutput as unknown) as Node,
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
   * Memory layer with a specified size and memory depth
   * @param size Number of nodes in each memory block
   * @param memory Number of memory blocks
   * @returns A memory layer
   */
  static memory(size: number, memory: number): Layer {
    const layer = new Layer();

    let previous: Group | null = null;
    for (let i = 0; i < memory; i++) {
      const block = new Group(size);

      block.set({
        squash: methods.Activation.identity,
        bias: 0,
        type: 'constant',
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
}
