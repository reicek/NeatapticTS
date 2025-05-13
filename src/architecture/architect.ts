import Node from './node';
import Layer from './layer';
import Group from './group';
import Network from './network';
import * as methods from '../methods/methods';
import Connection from './connection'; // Ensure Connection is imported for type checking

/**
 * Provides static methods for constructing various predefined neural network architectures.
 *
 * The Architect class simplifies the creation of common network types like Multi-Layer Perceptrons (MLPs),
 * Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRUs), and more complex structures
 * inspired by neuro-evolutionary algorithms. It leverages the underlying `Layer`, `Group`, and `Node`
 * components to build interconnected `Network` objects.
 *
 * Methods often utilize helper functions from `Layer` (e.g., `Layer.dense`, `Layer.lstm`) and
 * connection strategies from `methods.groupConnection`.
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation} - Some methods like `random` are inspired by concepts discussed here.
 */
export default class Architect {
  /**
   * Constructs a Network instance from an array of interconnected Layers, Groups, or Nodes.
   *
   * This method processes the input list, extracts all unique nodes, identifies connections,
   * gates, and self-connections, and determines the network's input and output sizes based
   * on the `type` property ('input' or 'output') set on the nodes. It uses Sets internally
   * for efficient handling of unique elements during construction.
   *
   * @param {Array<Group | Layer | Node>} list - An array containing the building blocks (Nodes, Layers, Groups) of the network, assumed to be already interconnected.
   * @returns {Network} A Network object representing the constructed architecture.
   * @throws {Error} If the input/output nodes cannot be determined (e.g., no nodes are marked with type 'input' or 'output').
   * @throws {Error} If the constructed network has zero input or output nodes after processing the list.
   */
  static construct(list: Array<Group | Layer | Node>): Network {
    // Initialize a new Network with placeholder input/output sizes (0, 0).
    // These will be determined during the construction process.
    const network = new Network(0, 0);
    // Use Sets for efficient storage and retrieval of unique nodes and connections.
    const uniqueNodes = new Set<Node>();
    const connections = new Set<Connection>(); // Regular forward connections
    const gates = new Set<Connection>(); // Gating connections
    const selfconns = new Set<Connection>(); // Self-connections (node to itself)
    let inputSize = 0; // Counter for nodes identified as input nodes
    let outputSize = 0; // Counter for nodes identified as output nodes
    let foundTypes = false; // Flag to track if any node had its 'type' property set.

    // Iterate through the provided list of Layers, Groups, or Nodes.
    for (const item of list) {
      let currentNodes: Node[] = [];
      // Extract nodes based on the type of the item (Group, Layer, or Node).
      if (item instanceof Group) {
        currentNodes = item.nodes;
      } else if (item instanceof Layer) {
        // Layers can potentially contain Groups (though typically contain Nodes).
        // Flatten the structure to get individual nodes.
        for (const layerNode of item.nodes) {
          if (layerNode instanceof Group) {
            currentNodes.push(...layerNode.nodes);
          } else if (layerNode instanceof Node) {
            currentNodes.push(layerNode);
          }
        }
      } else if (item instanceof Node) {
        // If the item is already a Node, add it directly.
        currentNodes = [item];
      }

      // Process each node extracted from the current item.
      for (const node of currentNodes) {
        // Add the node to the set of unique nodes if it hasn't been added yet.
        if (!uniqueNodes.has(node)) {
          uniqueNodes.add(node);

          // Check the node's type to determine if it's an input or output node.
          // The 'type' property must be explicitly set on the nodes beforehand.
          if (node.type === 'input') {
            inputSize++;
            foundTypes = true; // Mark that we found at least one node with a type.
          } else if (node.type === 'output') {
            outputSize++;
            foundTypes = true; // Mark that we found at least one node with a type.
          }

          // Collect all outgoing, gated, and self-connections associated with this node.
          // Ensure connections are valid Connection objects before adding to Sets.
          if (node.connections) {
            if (Array.isArray(node.connections.out)) {
              node.connections.out.forEach((conn) => {
                if (conn instanceof Connection) connections.add(conn);
              });
            }
            if (Array.isArray(node.connections.gated)) {
              node.connections.gated.forEach((conn) => {
                if (conn instanceof Connection) gates.add(conn);
              });
            }
            // Add self-connection only if it exists (array is not empty) and has a non-zero weight.
            if (
              node.connections.self.length > 0 && // Check if array has elements
              node.connections.self[0] instanceof Connection && // Check type of first element
              node.connections.self[0].weight !== 0 // Access weight of first element
            ) {
              selfconns.add(node.connections.self[0]); // Add the Connection object
            }
          }
        }
      }
    }

    // After processing all items, check if input and output sizes were determined.
    if (inputSize > 0 && outputSize > 0) {
      network.input = inputSize;
      network.output = outputSize;
    } else {
      // If no nodes were explicitly typed as 'input' or 'output', or if either count is zero,
      // the network structure is ambiguous or incomplete.
      if (!foundTypes || inputSize === 0 || outputSize === 0) {
        throw new Error(
          'Could not determine input/output nodes. Ensure nodes have their `type` property set to "input" or "output".'
        );
      }
      // Note: A previous fallback mechanism existed here but was removed for stricter type enforcement.
      // Layers/Groups themselves don't inherently define network I/O; individual nodes must be typed.
    }

    // Populate the network object with the collected nodes and connections.
    network.nodes = Array.from(uniqueNodes);
    network.connections = Array.from(connections);
    network.gates = Array.from(gates);
    network.selfconns = Array.from(selfconns);

    // Final validation to ensure the network is viable.
    if (network.input === 0 || network.output === 0) {
      // This check is somewhat redundant due to the earlier error throw, but serves as a safeguard.
      throw new Error('Constructed network has zero input or output nodes.');
    }

    return network;
  }

  /**
   * Creates a standard Multi-Layer Perceptron (MLP) network.
   * An MLP consists of an input layer, one or more hidden layers, and an output layer,
   * fully connected layer by layer.
   *
   * @param {...number} layers - A sequence of numbers representing the size (number of nodes) of each layer, starting with the input layer, followed by hidden layers, and ending with the output layer. Must include at least input, one hidden, and output layer sizes.
   * @returns {Network} The constructed MLP network.
   * @throws {Error} If fewer than 3 layer sizes (input, hidden, output) are provided.
   */
  static perceptron(...layers: number[]): Network {
    if (layers.length < 3) {
      throw new Error(
        'Invalid MLP configuration: You must specify at least 3 layer sizes (input, hidden, output).'
      );
    }

    // Compute minimum hidden size
    const inputSize = layers[0];
    const outputSize = layers[layers.length - 1];
    const minHidden = Math.min(inputSize, outputSize) + 1;

    // Create the input layer using Layer.dense for a standard fully connected layer.
    const inputLayer = Layer.dense(inputSize);
    // Mark nodes in this layer as network inputs.
    inputLayer.set({ type: 'input' });

    // Initialize the list of network components (layers/groups) and track the previous layer for connection.
    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    // Create hidden layers and the output layer.
    for (let i = 1; i < layers.length; i++) {
      // For hidden layers, enforce minimum size
      let layerSize = layers[i];
      if (i !== layers.length - 1 && layerSize < minHidden) {
        layerSize = minHidden;
      }
      const currentLayer = Layer.dense(layerSize);
      // Mark the final layer's nodes as network outputs.
      if (i === layers.length - 1) {
        currentLayer.set({ type: 'output' });
      }
      // Connect the previous layer to the current layer using a full mesh connection.
      (previousLayer as Layer).connect(
        currentLayer,
        methods.groupConnection.ALL_TO_ALL // Every node in previousLayer connects to every node in currentLayer.
      );
      nodes.push(currentLayer); // Add the new layer to the list of network components.
      previousLayer = currentLayer; // Update the reference to the previous layer.
    }

    // Construct the final Network object from the assembled layers.
    return Architect.construct(nodes);
  }

  /**
   * Creates a randomly structured network based on specified node counts and connection options.
   *
   * This method allows for the generation of networks with a less rigid structure than MLPs.
   * It initializes a network with input and output nodes and then iteratively adds hidden nodes
   * and various types of connections (forward, backward, self) and gates using mutation methods.
   * This approach is inspired by neuro-evolution techniques where network topology evolves.
   *
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
   *
   * @param {number} input - The number of input nodes.
   * @param {number} hidden - The number of hidden nodes to add.
   * @param {number} output - The number of output nodes.
   * @param {object} [options] - Optional configuration for the network structure.
   * @param {number} [options.connections=hidden*2] - The target number of forward connections to add (in addition to initial hidden node connections). Defaults to `hidden * 2`.
   * @param {number} [options.backconnections=0] - The target number of recurrent (backward) connections to add. Defaults to 0.
   * @param {number} [options.selfconnections=0] - The target number of self-connections (node connecting to itself) to add. Defaults to 0.
   * @param {number} [options.gates=0] - The target number of gating connections to add. Defaults to 0.
   * @returns {Network} The constructed network with a randomized topology.
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
    // Set default values for optional parameters if not provided.
    const {
      connections = hidden * 2, // Default connections aim for reasonable density.
      backconnections = 0,
      selfconnections = 0,
      gates = 0,
    } = options;

    // Initialize a base network with the specified input and output sizes.
    // Input and output nodes are created automatically by the Network constructor.
    const network = new Network(input, output);

    // Add the specified number of hidden nodes using the ADD_NODE mutation.
    // This mutation typically adds a node by splitting an existing connection.
    for (let i = 0; i < hidden; i++) {
      network.mutate(methods.mutation.ADD_NODE);
    }

    // Add forward connections using the ADD_CONN mutation.
    // This mutation adds a connection between two previously unconnected nodes.
    // Note: The initial hidden node additions also create connections, so we add `connections - hidden` more.
    for (let i = 0; i < connections - hidden; i++) {
      network.mutate(methods.mutation.ADD_CONN);
    }

    // Add recurrent (backward) connections using the ADD_BACK_CONN mutation.
    for (let i = 0; i < backconnections; i++) {
      network.mutate(methods.mutation.ADD_BACK_CONN);
    }

    // Add self-connections using the ADD_SELF_CONN mutation.
    for (let i = 0; i < selfconnections; i++) {
      network.mutate(methods.mutation.ADD_SELF_CONN);
    }

    // Add gating connections using the ADD_GATE mutation.
    // This adds a connection where one node controls the flow through another connection.
    for (let i = 0; i < gates; i++) {
      network.mutate(methods.mutation.ADD_GATE);
    }

    // Return the network with the generated topology.
    return network;
  }

  /**
   * Creates a Long Short-Term Memory (LSTM) network.
   * LSTMs are a type of recurrent neural network (RNN) capable of learning long-range dependencies.
   * This constructor uses `Layer.lstm` to create the core LSTM blocks.
   *
   * @param {...(number | object)} layerArgs - A sequence of arguments defining the network structure:
   *   - Numbers represent the size (number of units) of each layer: input layer size, hidden LSTM layer sizes..., output layer size.
   *   - An optional configuration object can be provided as the last argument.
   * @param {object} [options] - Configuration options (if passed as the last argument).
   * @param {boolean} [options.inputToOutput=true] - If true, creates direct connections from the input layer to the output layer, bypassing the LSTM layers. Defaults to true.
   * @returns {Network} The constructed LSTM network.
   * @throws {Error} If fewer than 3 numerical layer sizes (input, hidden, output) are provided.
   * @throws {Error} If any layer size argument is not a positive finite number.
   */
  static lstm(...layerArgs: (number | { inputToOutput?: boolean })[]): Network {
    let options: { inputToOutput?: boolean } = {};
    let layers: number[] = [];

    // Check if the last argument is an options object.
    if (
      layerArgs.length > 0 &&
      typeof layerArgs[layerArgs.length - 1] === 'object' &&
      layerArgs[layerArgs.length - 1] !== null &&
      !Array.isArray(layerArgs[layerArgs.length - 1])
    ) {
      // Pop the options object from the arguments array.
      options = layerArgs.pop() as { inputToOutput?: boolean };
    }

    // Validate that the remaining arguments are positive numbers representing layer sizes.
    if (
      !layerArgs.every(
        (arg): arg is number =>
          typeof arg === 'number' && Number.isFinite(arg) && arg > 0
      )
    ) {
      throw new Error(
        'Invalid LSTM layer arguments: All layer sizes must be positive finite numbers.'
      );
    }
    layers = layerArgs as number[]; // Type assertion is safe after validation.

    // Ensure at least input, one hidden (LSTM), and output layers are specified.
    if (layers.length < 3) {
      throw new Error(
        'Invalid LSTM configuration: You must specify at least 3 layer sizes (input, hidden..., output).'
      );
    }

    // Apply default value for the inputToOutput option if not provided.
    const { inputToOutput = true } = options;

    // Extract input and output layer sizes. The remaining numbers in 'layers' are hidden layer sizes.
    const inputLayerSize = layers.shift()!; // Non-null assertion is safe due to length check.
    const outputLayerSize = layers.pop()!; // Non-null assertion is safe due to length check.

    // Create the input layer.
    const inputLayer = Layer.dense(inputLayerSize);
    inputLayer.set({ type: 'input' }); // Mark nodes as network inputs.

    // Create the output layer.
    const outputLayer = Layer.dense(outputLayerSize);
    outputLayer.set({ type: 'output' }); // Mark nodes as network outputs.

    // Initialize the list of network components and track the previous layer.
    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    // Create the hidden LSTM layers.
    for (const layerSize of layers) {
      // Iterate through the specified hidden layer sizes.
      // Create an LSTM layer (which is internally a Group of nodes: input, forget, output, memory cells).
      const lstmLayer = Layer.lstm(layerSize);
      // Connect the previous layer to the LSTM layer. The default connection typically targets the input gates.
      (previousLayer as Layer).connect(lstmLayer);
      nodes.push(lstmLayer); // Add the LSTM layer group to the network components.
      previousLayer = lstmLayer; // Update the reference to the previous layer.
    }

    // Connect the last hidden/LSTM layer to the output layer.
    (previousLayer as Layer).connect(outputLayer); // Default connection.
    nodes.push(outputLayer); // Add the output layer to the list.

    // Optionally, add direct connections from the input layer to the output layer.
    if (inputToOutput) {
      inputLayer.connect(outputLayer, methods.groupConnection.ALL_TO_ALL);
    }

    // Construct the final Network object from the assembled layers and groups.
    const network = Architect.construct(nodes);

    // Explicitly set the input and output sizes on the final Network object,
    // as the construct method relies on node types which might not cover all cases perfectly,
    // especially with complex groups like LSTMs.
    network.input = inputLayerSize;
    network.output = outputLayerSize;

    return network;
  }

  /**
   * Creates a Gated Recurrent Unit (GRU) network.
   * GRUs are another type of recurrent neural network, similar to LSTMs but often simpler.
   * This constructor uses `Layer.gru` to create the core GRU blocks.
   *
   * @param {...number} layers - A sequence of numbers representing the size (number of units) of each layer: input layer size, hidden GRU layer sizes..., output layer size. Must include at least input, one hidden, and output layer sizes.
   * @returns {Network} The constructed GRU network.
   * @throws {Error} If fewer than 3 layer sizes (input, hidden, output) are provided.
   */
  static gru(...layers: number[]): Network {
    // Ensure at least input, one hidden (GRU), and output layers are specified.
    if (layers.length < 3) {
      throw new Error(
        'Invalid GRU configuration: You must specify at least 3 layer sizes (input, hidden..., output).'
      );
    }

    // Extract input and output layer sizes.
    const inputLayerSize = layers.shift()!;
    const outputLayerSize = layers.pop()!;
    // 'layers' now contains only hidden GRU layer sizes.

    // Create the input layer.
    const inputLayer = Layer.dense(inputLayerSize);
    inputLayer.set({ type: 'input' }); // Mark nodes as network inputs.

    // Create the output layer.
    const outputLayer = Layer.dense(outputLayerSize);
    outputLayer.set({ type: 'output' }); // Mark nodes as network outputs.

    // Initialize the list of network components and track the previous layer.
    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    // Create the hidden GRU layers.
    for (const blockSize of layers) {
      // Iterate through the specified hidden layer sizes.
      // Create a GRU layer (internally a Group of nodes: update gate, reset gate, hidden state).
      const gruLayer = Layer.gru(blockSize);
      // Connect the previous layer to the GRU layer. Default connection targets appropriate gates.
      (previousLayer as Layer).connect(gruLayer);
      nodes.push(gruLayer); // Add the GRU layer group to the network components.
      previousLayer = gruLayer; // Update the reference to the previous layer.
    }

    // Connect the last hidden/GRU layer to the output layer.
    (previousLayer as Layer).connect(outputLayer);
    nodes.push(outputLayer); // Add the output layer to the list.

    // Construct the final Network object.
    const network = Architect.construct(nodes);

    // Explicitly set the input and output sizes on the final Network object for clarity and robustness.
    network.input = inputLayerSize;
    network.output = outputLayerSize;

    return network;
  }

  /**
   * Creates a Hopfield network.
   * Hopfield networks are a form of recurrent neural network often used for associative memory tasks.
   * This implementation creates a simple, fully connected structure.
   *
   * @param {number} size - The number of nodes in the network (input and output layers will have this size).
   * @returns {Network} The constructed Hopfield network.
   */
  static hopfield(size: number): Network {
    // Create input and output layers of the specified size.
    const inputLayer = Layer.dense(size);
    const outputLayer = Layer.dense(size);

    // Create a full connection between the input and output layers.
    // Note: Traditional Hopfield networks often have connections within a single layer,
    // but this structure represents a common feedforward variant or interpretation.
    // For a classic Hopfield, one might connect a layer to itself (ALL_TO_ALL excluding self).
    inputLayer.connect(outputLayer, methods.groupConnection.ALL_TO_ALL);

    // Mark the input layer nodes.
    inputLayer.set({ type: 'input' });
    // Mark the output layer nodes and set their activation function to a step function, typical for Hopfield networks.
    outputLayer.set({ squash: methods.Activation.step, type: 'output' });

    // Construct the network from the two layers.
    return Architect.construct([inputLayer, outputLayer]);
  }

  /**
   * Creates a Nonlinear AutoRegressive network with eXogenous inputs (NARX).
   * NARX networks are recurrent networks often used for time series prediction.
   * They predict the next value of a time series based on previous values of the series
   * and previous values of external (exogenous) input series.
   *
   * @param {number} inputSize - The number of input nodes for the exogenous inputs at each time step.
   * @param {number | number[]} hiddenLayers - The size of the hidden layer(s). Can be a single number for one hidden layer, or an array of numbers for multiple hidden layers. Use 0 or [] for no hidden layers.
   * @param {number} outputSize - The number of output nodes (predicting the time series).
   * @param {number} previousInput - The number of past time steps of the exogenous input to feed back into the network.
   * @param {number} previousOutput - The number of past time steps of the network's own output to feed back into the network (autoregressive part).
   * @returns {Network} The constructed NARX network.
   */
  static narx(
    inputSize: number,
    hiddenLayers: number | number[],
    outputSize: number,
    previousInput: number, // Input delay taps
    previousOutput: number // Output delay taps
  ): Network {
    // Ensure hiddenLayers is an array, even if a single number or zero is provided.
    if (!Array.isArray(hiddenLayers)) {
      hiddenLayers = hiddenLayers > 0 ? [hiddenLayers] : []; // Convert number to array or empty array if 0.
    }

    // Create the main input layer for current exogenous inputs.
    const input = Layer.dense(inputSize);
    // Create a memory layer to hold 'previousInput' past values of the input.
    const inputMemory = Layer.memory(inputSize, previousInput);
    // Create the main output layer.
    const output = Layer.dense(outputSize);
    // Create a memory layer to hold 'previousOutput' past values of the output.
    const outputMemory = Layer.memory(outputSize, previousOutput);

    // Mark input and output layers appropriately.
    input.set({ type: 'input' });
    output.set({ type: 'output' });

    // Connect the main input layer to its corresponding memory layer.
    // A weight of 1 ensures the current input is stored for the next time step.
    input.connect(inputMemory, methods.groupConnection.ONE_TO_ONE, 1);
    // Connect the main output layer to its corresponding memory layer.
    // A weight of 1 ensures the current output is stored for the next time step.
    output.connect(outputMemory, methods.groupConnection.ONE_TO_ONE, 1);

    const hidden: Layer[] = []; // Array to hold created hidden layers.
    let previousLayer: Layer | Group = input; // Start connections from the input layer.
    // Initialize the list of network components. Memory layers are included early.
    const nodes: (Layer | Group)[] = [input, inputMemory, outputMemory];

    // This layer will receive inputs from the main input AND the memory layers.
    // It's either the first hidden layer or the output layer if no hidden layers exist.
    let firstProcessingLayer: Layer | Group;

    // Create hidden layers if specified.
    if (hiddenLayers.length > 0) {
      for (let i = 0; i < hiddenLayers.length; i++) {
        const size = hiddenLayers[i];
        const hiddenLayer = Layer.dense(size);
        hidden.push(hiddenLayer);
        nodes.push(hiddenLayer); // Add hidden layer to the network components.

        // Connect the previous layer (input or preceding hidden layer) to the current hidden layer.
        (previousLayer as Layer).connect(
          hiddenLayer,
          methods.groupConnection.ALL_TO_ALL
        );
        previousLayer = hiddenLayer; // Update previous layer for the next connection.

        // Identify the first hidden layer as the target for memory inputs.
        if (i === 0) {
          firstProcessingLayer = hiddenLayer;
        }
      }
      // Connect the last hidden layer to the output layer.
      (previousLayer as Layer).connect(
        output,
        methods.groupConnection.ALL_TO_ALL
      );
    } else {
      // No hidden layers: connect the main input layer directly to the output layer.
      input.connect(output, methods.groupConnection.ALL_TO_ALL);
      // In this case, the output layer is the first processing layer receiving memory inputs.
      firstProcessingLayer = output;
    }

    nodes.push(output); // Add the output layer to the list of components.

    // Connect the memory layers to the first processing layer (first hidden layer or output layer).
    // These connections provide the historical context (past inputs and outputs).
    // Use ALL_TO_ALL connection: every memory node connects to every node in the target layer.
    inputMemory.connect(
      firstProcessingLayer!,
      methods.groupConnection.ALL_TO_ALL
    ); // Non-null assertion safe due to logic above.
    outputMemory.connect(
      firstProcessingLayer!,
      methods.groupConnection.ALL_TO_ALL
    ); // Non-null assertion safe due to logic above.

    // Construct the final Network object.
    const network = Architect.construct(nodes);

    // Explicitly set the input and output sizes for the final network object.
    // Input size corresponds to the exogenous input dimension.
    // Output size corresponds to the predicted time series dimension.
    network.input = inputSize;
    network.output = outputSize;

    return network;
  }

  /**
   * Enforces the minimum hidden layer size rule on a network.
   * 
   * This ensures that all hidden layers have at least min(input, output) + 1 nodes,
   * which is a common heuristic to ensure networks have adequate representation capacity.
   * 
   * @param {Network} network - The network to enforce minimum hidden layer sizes on
   * @returns {Network} The same network with properly sized hidden layers
   */
  static enforceMinimumHiddenLayerSizes(network: Network): Network {
    if (!network.layers || network.layers.length <= 2) {
      // No hidden layers to resize
      return network;
    }

    // Calculate minimum size for hidden layers
    const minSize = Math.min(network.input, network.output) + 1;
    
    // Adjust all hidden layers (skip input and output layers)
    for (let i = 1; i < network.layers.length - 1; i++) {
      const hiddenLayer = network.layers[i];
      const currentSize = hiddenLayer.nodes.length;
      
      if (currentSize < minSize) {
        console.log(`Resizing hidden layer ${i} from ${currentSize} to ${minSize} nodes`);
        
        // Create the additional nodes needed
        for (let j = currentSize; j < minSize; j++) {
          const newNode = new Node('hidden');
          hiddenLayer.nodes.push(newNode);
          
          // Add node to network's node list
          network.nodes.push(newNode);
          
          // Connect to previous layer
          if (i > 0 && network.layers[i-1].output) {
            for (const prevNode of network.layers[i-1].output.nodes) {
              const connections = prevNode.connect(newNode);
              // Fix: Spread the connections array into individual connections
              network.connections.push(...connections);
            }
          }
          
          // Connect to next layer
          if (i < network.layers.length - 1 && network.layers[i+1].output) {
            for (const nextNode of network.layers[i+1].output.nodes) {
              const connections = newNode.connect(nextNode);
              // Fix: Spread the connections array into individual connections
              network.connections.push(...connections);
            }
          }
          
          // If this layer has an output group, add the node to it
          if (hiddenLayer.output && Array.isArray(hiddenLayer.output.nodes)) {
            hiddenLayer.output.nodes.push(newNode);
          }
        }
      }
    }
    
    return network;
  }
}
