import Node from './node';
import Layer from './layer';
import Group from './group';
import Network from './network';
import * as methods from '../methods/methods';

/**
 * Architect class provides static methods to create various types of neural networks.
 */
export default class Architect {
  /**
   * Constructs a network from a given array of connected nodes.
   * Transforms groups and layers into nodes and determines input/output nodes.
   * @param {Array<Group | Layer | Node>} list - Array of connected nodes, groups, or layers.
   * @returns {Network} The constructed network.
   * @throws {Error} If no clear input/output nodes are found.
   */
  static construct(list: Array<Group | Layer | Node>): Network {
    const network = new Network(0, 0);
    let nodes: Node[] = [];

    // Transform all groups and layers into nodes
    for (let i = 0; i < list.length; i++) {
      if (list[i] instanceof Group) {
        nodes.push(...list[i].nodes);
      } else if (list[i] instanceof Layer) {
        for (const group of list[i].nodes) {
          nodes.push(...group.nodes);
        }
      } else if (list[i] instanceof Node) {
        nodes.push(list[i] as Node);
      }
    }

    // Determine input and output nodes
    const inputs: Node[] = [];
    const outputs: Node[] = [];
    for (let i = nodes.length - 1; i >= 0; i--) {
      if (
        nodes[i].type === 'output' ||
        nodes[i].connections.out.length + nodes[i].connections.gated.length ===
          0
      ) {
        nodes[i].type = 'output';
        network.output++;
        outputs.push(nodes[i]);
        nodes.splice(i, 1);
      } else if (nodes[i].type === 'input' || !nodes[i].connections.in.length) {
        nodes[i].type = 'input';
        network.input++;
        inputs.push(nodes[i]);
        nodes.splice(i, 1);
      }
    }

    // Input nodes are always first, output nodes are always last
    nodes = [...inputs, ...nodes, ...outputs];

    if (network.input === 0 || network.output === 0) {
      throw new Error('Given nodes have no clear input/output node!');
    }

    // Add connections and gates to the network
    for (const node of nodes) {
      network.connections.push(...node.connections.out);
      network.gates.push(...node.connections.gated);
      if (node.connections.self.weight !== 0) {
        network.selfconns.push(node.connections.self);
      }
    }

    network.nodes = nodes;
    return network;
  }

  /**
   * Creates a multilayer perceptron (MLP).
   * @param {...number} layers - Sizes of each layer (input, hidden, output).
   * @returns {Network} The constructed MLP network.
   * @throws {Error} If fewer than 3 layers are specified.
   */
  static perceptron(...layers: number[]): Network {
    if (layers.length < 3) {
      throw new Error('You have to specify at least 3 layers');
    }

    const nodes: Group[] = [new Group(layers[0])];

    for (let i = 1; i < layers.length; i++) {
      const layer = new Group(layers[i]);
      nodes.push(layer);
      nodes[i - 1].connect(layer, methods.connection.ALL_TO_ALL);
    }

    return Architect.construct(nodes);
  }

  /**
   * Creates a randomly connected network.
   * @param {number} input - Number of input nodes.
   * @param {number} hidden - Number of hidden nodes.
   * @param {number} output - Number of output nodes.
   * @param {object} [options] - Options for connections, backconnections, selfconnections, and gates.
   * @param {number} [options.connections=hidden*2] - Number of connections.
   * @param {number} [options.backconnections=0] - Number of backconnections.
   * @param {number} [options.selfconnections=0] - Number of self-connections.
   * @param {number} [options.gates=0] - Number of gates.
   * @returns {Network} The constructed random network.
   */
  static random(
    input: number,
    hidden: number,
    output: number,
    options: {
      connections?: number;
      backconnections?: number;
      selfconnections?: number;
      gates?: number;
    } = {}
  ): Network {
    const {
      connections = hidden * 2,
      backconnections = 0,
      selfconnections = 0,
      gates = 0,
    } = options;

    const network = new Network(input, output);

    for (let i = 0; i < hidden; i++) {
      network.mutate(methods.mutation.ADD_NODE);
    }

    for (let i = 0; i < connections - hidden; i++) {
      network.mutate(methods.mutation.ADD_CONN);
    }

    for (let i = 0; i < backconnections; i++) {
      network.mutate(methods.mutation.ADD_BACK_CONN);
    }

    for (let i = 0; i < selfconnections; i++) {
      network.mutate(methods.mutation.ADD_SELF_CONN);
    }

    for (let i = 0; i < gates; i++) {
      network.mutate(methods.mutation.ADD_GATE);
    }

    return network;
  }

  /**
   * Creates a long short-term memory (LSTM) network with additional options.
   * @param {...number | object} layers - Sizes of each layer (input, hidden, output) and optional configuration object.
   * @param {object} [options] - Configuration options for the LSTM network.
   * @param {boolean} [options.memoryToMemory=false] - Connect memory cells to other memory cells.
   * @param {boolean} [options.outputToMemory=false] - Connect output layer to memory cells.
   * @param {boolean} [options.outputToGates=false] - Connect output layer to gates.
   * @param {boolean} [options.inputToOutput=true] - Connect input layer to output layer.
   * @param {boolean} [options.inputToDeep=true] - Connect input layer to deep memory cells.
   * @returns {Network} The constructed LSTM network.
   * @throws {Error} If fewer than 3 layers are specified.
   */
  static lstm(
    ...layers: (
      | number
      | {
          memoryToMemory?: boolean;
          outputToMemory?: boolean;
          outputToGates?: boolean;
          inputToOutput?: boolean;
          inputToDeep?: boolean;
        }
    )[]
  ): Network {
    if (layers.length < 3) {
      throw new Error('You have to specify at least 3 layers');
    }

    const last = layers.pop();
    const outputLayer = new Group(
      typeof last === 'number' ? last : (layers.pop() as number)
    );
    outputLayer.set({ type: 'output' });

    const options = typeof last === 'object' ? last : {};
    const {
      memoryToMemory = false,
      outputToMemory = false,
      outputToGates = false,
      inputToOutput = true,
      inputToDeep = true,
    } = options;

    const inputLayer = new Group(layers.shift() as number);
    inputLayer.set({ type: 'input' });

    const nodes: (Group | Layer)[] = [inputLayer];
    let previous = inputLayer;

    for (const blockSize of layers) {
      const inputGate = new Group(blockSize as number);
      const forgetGate = new Group(blockSize as number);
      const memoryCell = new Group(blockSize as number);
      const outputGate = new Group(blockSize as number);
      const outputBlock = new Group(blockSize as number);

      inputGate.set({ bias: 1 });
      forgetGate.set({ bias: 1 });
      outputGate.set({ bias: 1 });

      const input = previous.connect(memoryCell, methods.connection.ALL_TO_ALL);
      previous.connect(inputGate, methods.connection.ALL_TO_ALL);
      previous.connect(outputGate, methods.connection.ALL_TO_ALL);
      previous.connect(forgetGate, methods.connection.ALL_TO_ALL);

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

      inputGate.gate(input, methods.gating.INPUT);
      forgetGate.gate(forget, methods.gating.SELF);
      outputGate.gate(output, methods.gating.OUTPUT);

      if (inputToDeep && nodes.length > 1) {
        const inputToMemory = inputLayer.connect(
          memoryCell,
          methods.connection.ALL_TO_ALL
        );
        inputGate.gate(inputToMemory, methods.gating.INPUT);
      }

      if (memoryToMemory) {
        const memoryToMemoryConn = memoryCell.connect(
          memoryCell,
          methods.connection.ALL_TO_ELSE
        );
        inputGate.gate(memoryToMemoryConn, methods.gating.INPUT);
      }

      if (outputToMemory) {
        const outputToMemoryConn = outputLayer.connect(
          memoryCell,
          methods.connection.ALL_TO_ALL
        );
        inputGate.gate(outputToMemoryConn, methods.gating.INPUT);
      }

      if (outputToGates) {
        outputLayer.connect(inputGate, methods.connection.ALL_TO_ALL);
        outputLayer.connect(forgetGate, methods.connection.ALL_TO_ALL);
        outputLayer.connect(outputGate, methods.connection.ALL_TO_ALL);
      }

      nodes.push(inputGate, forgetGate, memoryCell, outputGate, outputBlock);
      previous = outputBlock;
    }

    if (inputToOutput) {
      inputLayer.connect(outputLayer, methods.connection.ALL_TO_ALL);
    }

    nodes.push(outputLayer);
    return Architect.construct(nodes);
  }

  /**
   * Creates a gated recurrent unit (GRU) network.
   * @param {...number} layers - Sizes of each layer (input, hidden, output).
   * @returns {Network} The constructed GRU network.
   * @throws {Error} If fewer than 3 layers are specified.
   */
  static gru(...layers: number[]): Network {
    if (layers.length < 3) {
      throw new Error('You have to specify at least 3 layers');
    }

    const inputLayer = new Group(layers.shift()!);
    const outputLayer = new Group(layers.pop()!);

    const nodes: (Group | Layer)[] = [inputLayer];
    let previous = inputLayer;

    for (const blockSize of layers) {
      const layer = Layer.gru(blockSize);
      previous.connect(layer);
      nodes.push(layer);
      previous = layer;
    }

    previous.connect(outputLayer);
    nodes.push(outputLayer);

    return Architect.construct(nodes);
  }

  /**
   * Creates a Hopfield network.
   * @param {number} size - Size of the network.
   * @returns {Network} The constructed Hopfield network.
   */
  static hopfield(size: number): Network {
    const input = new Group(size);
    const output = new Group(size);

    input.connect(output, methods.connection.ALL_TO_ALL);

    input.set({ type: 'input' });
    output.set({ squash: methods.Activation.step, type: 'output' });

    return Architect.construct([input, output]);
  }

  /**
   * Creates a NARX (Nonlinear AutoRegressive with eXogenous inputs) network.
   * @param {number} inputSize - Number of input nodes.
   * @param {number | number[]} hiddenLayers - Number of hidden layers or an array of sizes.
   * @param {number} outputSize - Number of output nodes.
   * @param {number} previousInput - Number of previous input steps to remember.
   * @param {number} previousOutput - Number of previous output steps to remember.
   * @returns {Network} The constructed NARX network.
   */
  static narx(
    inputSize: number,
    hiddenLayers: number | number[],
    outputSize: number,
    previousInput: number,
    previousOutput: number
  ): Network {
    if (!Array.isArray(hiddenLayers)) {
      hiddenLayers = [hiddenLayers];
    }

    const input = Layer.dense(inputSize);
    const inputMemory = Layer.memory(inputSize, previousInput);
    const output = Layer.dense(outputSize);
    const outputMemory = Layer.memory(outputSize, previousOutput);

    const hidden: Layer[] = [];
    const nodes: (Layer | Group)[] = [input, outputMemory];

    for (const size of hiddenLayers) {
      const hiddenLayer = Layer.dense(size);
      hidden.push(hiddenLayer);
      nodes.push(hiddenLayer);
      if (hidden.length > 1) {
        hidden[hidden.length - 2].connect(
          hiddenLayer,
          methods.connection.ALL_TO_ALL
        );
      }
    }

    nodes.push(inputMemory, output);

    input.connect(hidden[0], methods.connection.ALL_TO_ALL);
    input.connect(inputMemory, methods.connection.ONE_TO_ONE, 1);
    inputMemory.connect(hidden[0], methods.connection.ALL_TO_ALL);
    hidden[hidden.length - 1].connect(output, methods.connection.ALL_TO_ALL);
    output.connect(outputMemory, methods.connection.ONE_TO_ONE, 1);
    outputMemory.connect(hidden[0], methods.connection.ALL_TO_ALL);

    input.set({ type: 'input' });
    output.set({ type: 'output' });

    return Architect.construct(nodes);
  }
}
