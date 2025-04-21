import Node from './node';
import Layer from './layer';
import Group from './group';
import Network from './network';
import * as methods from '../methods/methods';
import Connection from './connection'; // Added import for Connection

/**
 * Architect class provides static methods to create various types of neural networks.
 *
 * This class includes methods inspired by the Instinct algorithm, such as random network
 * generation, LSTM, GRU, and NARX networks. These methods leverage advanced mutation
 * strategies and gating mechanisms to create adaptable and efficient networks.
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
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
      const item = list[i];
      if (item instanceof Group) {
        nodes.push(...item.nodes);
      } else if (item instanceof Layer) {
        // Iterate through the items within the layer's nodes array
        // Check if item.nodes contains Groups or Nodes directly
        if (item.nodes.every((n) => n instanceof Node)) {
          nodes.push(...(item.nodes as Node[]));
        } else {
          // Handle cases where layer.nodes might contain Groups (like in old LSTM impl)
          for (const layerNode of item.nodes) {
            if (layerNode instanceof Group) {
              nodes.push(...layerNode.nodes);
            } else if (layerNode instanceof Node) {
              nodes.push(layerNode);
            }
          }
        }
      } else if (item instanceof Node) {
        nodes.push(item);
      }
    }

    // Determine input and output sizes from node types
    let inputSize = 0;
    let outputSize = 0;
    // Use a Set to avoid adding the same node multiple times if it appears in different groups/layers
    const uniqueNodes = [...new Set(nodes)];
    uniqueNodes.forEach((node) => {
      if (node.type === 'input') inputSize++;
      else if (node.type === 'output') outputSize++;
    });

    // Assign determined sizes to the network
    network.input = inputSize;
    network.output = outputSize;

    if (network.input === 0 || network.output === 0) {
      // Check if types were set on the Layer/Group level instead
      list.forEach((item) => {
        if (item instanceof Layer || item instanceof Group) {
          item.nodes.forEach((node) => {
            if (node.type === 'input') network.input++;
            else if (node.type === 'output') network.output++;
          });
        }
      });
      // Recalculate unique nodes based on potentially updated input/output counts
      const finalNodes = [...new Set(nodes)];
      network.nodes = finalNodes;
      if (network.input === 0 || network.output === 0) {
        throw new Error('Given nodes have no clear input/output node!');
      }
    } else {
      // Assign unique nodes to the network if sizes were determined initially
      network.nodes = uniqueNodes;
    }

    // Add connections and gates to the network
    network.connections = [];
    network.gates = [];
    network.selfconns = [];

    network.nodes.forEach((node) => {
      // Ensure connections.out is iterable and contains Connection objects
      if (node.connections && Array.isArray(node.connections.out)) {
        node.connections.out.forEach((conn) => {
          if (conn instanceof Connection) {
            // Check if it's a Connection object
            network.connections.push(conn);
          }
        });
      }

      // Ensure connections.gated is iterable and contains Connection objects
      if (node.connections && Array.isArray(node.connections.gated)) {
        node.connections.gated.forEach((conn) => {
          if (conn instanceof Connection) {
            // Check if it's a Connection object
            network.gates.push(conn);
          }
        });
      }

      // Ensure self connection exists and is a Connection object
      if (
        node.connections &&
        node.connections.self instanceof Connection &&
        node.connections.self.weight !== 0
      ) {
        network.selfconns.push(node.connections.self);
      }
    });

    // Filter out duplicate connections that might arise from group/layer processing
    network.connections = [...new Set(network.connections)];
    network.gates = [...new Set(network.gates)];
    network.selfconns = [...new Set(network.selfconns)];

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

    // Use Layer.dense for consistency
    const inputLayer = Layer.dense(layers[0]);
    inputLayer.set({ type: 'input' });

    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    for (let i = 1; i < layers.length; i++) {
      const currentLayer = Layer.dense(layers[i]);
      if (i === layers.length - 1) {
        currentLayer.set({ type: 'output' }); // Set output type for the last layer
      }
      (previousLayer as Layer).connect(
        currentLayer,
        methods.connection.ALL_TO_ALL
      );
      nodes.push(currentLayer);
      previousLayer = currentLayer;
    }

    return Architect.construct(nodes);
  }

  /**
   * Creates a randomly connected network.
   *
   * This method implements the random network generation described in the Instinct algorithm,
   * allowing for the creation of networks with configurable connections, backconnections,
   * self-connections, and gates.
   *
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
   *
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
   * Creates a long short-term memory (LSTM) network using Layer.lstm.
   * @param {...number | object} layers - Sizes of each layer (input, hidden, output) and optional configuration object.
   * @param {object} [options] - Configuration options for the LSTM network.
   * @param {boolean} [options.inputToOutput=true] - Connect input layer to output layer.
   * @returns {Network} The constructed LSTM network.
   * @throws {Error} If fewer than 3 layers are specified.
   */
  static lstm(...layers: (number | { inputToOutput?: boolean })[]): Network {
    if (layers.length < 3) {
      throw new Error('You have to specify at least 3 layers');
    }

    const last = layers.pop();
    const outputLayerSize =
      typeof last === 'number' ? last : (layers.pop() as number);
    const options = typeof last === 'object' ? last : {};
    const { inputToOutput = true } = options;

    const inputLayerSize = layers.shift() as number;
    const inputLayer = Layer.dense(inputLayerSize); // Use Layer.dense for input
    inputLayer.set({ type: 'input' });

    const outputLayer = Layer.dense(outputLayerSize); // Use Layer.dense for output
    outputLayer.set({ type: 'output' });

    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    // Create LSTM layers using Layer.lstm
    for (const layerSize of layers) {
      if (typeof layerSize !== 'number') continue; // Skip options object if present mid-array
      const lstmLayer = Layer.lstm(layerSize);
      (previousLayer as Layer).connect(lstmLayer); // Connect previous layer's output to this LSTM layer's input
      nodes.push(lstmLayer);
      previousLayer = lstmLayer;
    }

    // Connect the last LSTM layer to the output layer
    (previousLayer as Layer).connect(outputLayer);
    nodes.push(outputLayer);

    // Optional connection from input directly to output
    if (inputToOutput) {
      inputLayer.connect(outputLayer, methods.connection.ALL_TO_ALL);
    }

    // Construct the network
    const network = Architect.construct(nodes);

    // Explicitly set input/output size for the final network object
    network.input = inputLayerSize;
    network.output = outputLayerSize;

    return network;
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

    const inputLayerSize = layers.shift()!;
    const outputLayerSize = layers.pop()!;

    const inputLayer = Layer.dense(inputLayerSize);
    inputLayer.set({ type: 'input' }); // Set input type

    const outputLayer = Layer.dense(outputLayerSize);
    outputLayer.set({ type: 'output' }); // Set output type

    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    for (const blockSize of layers) {
      const gruLayer = Layer.gru(blockSize);
      (previousLayer as Layer).connect(gruLayer);
      nodes.push(gruLayer);
      previousLayer = gruLayer;
    }

    (previousLayer as Layer).connect(outputLayer);
    nodes.push(outputLayer);

    const network = Architect.construct(nodes);

    // Explicitly set input/output size for the final network object
    network.input = inputLayerSize;
    network.output = outputLayerSize;

    return network;
  }

  /**
   * Creates a Hopfield network.
   * @param {number} size - Size of the network.
   * @returns {Network} The constructed Hopfield network.
   */
  static hopfield(size: number): Network {
    // Use Layer.dense for consistency
    const inputLayer = Layer.dense(size);
    const outputLayer = Layer.dense(size);

    inputLayer.connect(outputLayer, methods.connection.ALL_TO_ALL);

    inputLayer.set({ type: 'input' });
    outputLayer.set({ squash: methods.Activation.step, type: 'output' });

    return Architect.construct([inputLayer, outputLayer]);
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
    const nodes: (Layer | Group)[] = [input, outputMemory]; // Start with input and output memory

    let previousHiddenLayer: Layer | Group = input; // Connect input to the first hidden layer

    // Connect input memory to the first hidden layer as well
    inputMemory.connect(hidden[0], methods.connection.ALL_TO_ALL);

    for (const size of hiddenLayers) {
      const hiddenLayer = Layer.dense(size);
      hidden.push(hiddenLayer);
      nodes.push(hiddenLayer);
      // Connect previous layer (input or previous hidden) to current hidden layer
      (previousHiddenLayer as Layer).connect(
        hiddenLayer,
        methods.connection.ALL_TO_ALL
      );
      previousHiddenLayer = hiddenLayer; // Update previous layer for the next iteration
    }

    nodes.push(inputMemory, output); // Add input memory and output layer to the list for construction

    // Connect input to input memory
    input.connect(inputMemory, methods.connection.ONE_TO_ONE, 1);

    // Connect the last hidden layer to the output layer
    hidden[hidden.length - 1].connect(output, methods.connection.ALL_TO_ALL);

    // Connect output to output memory
    output.connect(outputMemory, methods.connection.ONE_TO_ONE, 1);

    // Connect output memory back to the first hidden layer
    outputMemory.connect(hidden[0], methods.connection.ALL_TO_ALL);

    input.set({ type: 'input' });
    output.set({ type: 'output' });

    const network = Architect.construct(nodes);

    // Explicitly set input/output size for the final network object
    network.input = inputSize;
    network.output = outputSize;

    return network;
  }
}
