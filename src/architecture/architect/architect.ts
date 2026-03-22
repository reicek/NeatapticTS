/**
 * Core architect chapter for the architecture surface.
 *
 * This folder owns the top-level builder entrypoint that turns folderized
 * node, connection, group, layer, and network chapters into named neural
 * network presets.
 *
 * Read this chapter in three passes:
 *
 * 1. start with `construct()` to see how pre-wired primitives become one
 *    `Network` instance,
 * 2. continue to `perceptron()` and `random()` when you want feed-forward and
 *    topology-search-friendly builders,
 * 3. finish with `lstm()`, `gru()`, `hopfield()`, and `narx()` when you need
 *    recurrent presets built from the lower-level architecture chapters.
 */
import Node from '../node/node';
import Layer from '../layer';
import Group from '../group/group';
import Network from '../network';
import * as methods from '../../methods/methods';
import Connection from '../connection/connection';

/**
 * Provides static methods for constructing predefined neural network
 * architectures.
 *
 * `Architect` is the point where the low-level graph primitives stop being
 * raw building blocks and start becoming named network recipes. It assembles
 * nodes, groups, and layers into complete graphs, then normalizes the final
 * `Network` surface so callers can activate, train, serialize, or evolve the
 * result without manually wiring each primitive.
 *
 * This boundary matters when you want one of three things:
 *
 * - a deterministic builder for common feed-forward shapes,
 * - a quick way to sample or mutate topology-oriented starting graphs,
 * - recurrent presets that reuse the same lower-level chapters instead of
 *   hiding a separate graph implementation.
 *
 * @example
 * ```ts
 * const network = Architect.perceptron(2, 4, 1);
 * const output = network.activate([0, 1]);
 * ```
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
 * Some methods like `random` are inspired by concepts discussed here.
 */
export default class Architect {
  /**
   * Constructs a network instance from an array of interconnected layers,
   * groups, or nodes.
   *
   * This method is the bridge between manual graph assembly and a runnable
   * `Network`. It walks the supplied primitives, collects the unique nodes and
   * connections they reference, infers input/output counts from node types, and
   * folds the result into one normalized network object.
   *
   * @param list Building blocks that are already interconnected.
   * @returns A network representing the supplied architecture.
   * @throws {Error} If the input/output nodes cannot be determined.
   * @throws {Error} If the constructed network has zero input or output nodes.
   */
  static construct(list: Array<Group | Layer | Node>): Network {
    const network = new Network(0, 0);
    const uniqueNodes = new Set<Node>();
    const connections = new Set<Connection>();
    const gates = new Set<Connection>();
    const selfconns = new Set<Connection>();
    let inputSize = 0;
    let outputSize = 0;
    let foundTypes = false;

    for (const item of list) {
      let currentNodes: Node[] = [];

      if (item instanceof Group) {
        currentNodes = item.nodes;
      } else if (item instanceof Layer) {
        for (const layerNode of item.nodes) {
          if (layerNode instanceof Group) {
            currentNodes.push(...layerNode.nodes);
          } else if (layerNode instanceof Node) {
            currentNodes.push(layerNode);
          }
        }
      } else if (item instanceof Node) {
        currentNodes = [item];
      }

      for (const node of currentNodes) {
        if (!uniqueNodes.has(node)) {
          uniqueNodes.add(node);

          if (node.type === 'input') {
            inputSize++;
            foundTypes = true;
          } else if (node.type === 'output') {
            outputSize++;
            foundTypes = true;
          }

          if (node.connections) {
            if (Array.isArray(node.connections.out)) {
              node.connections.out.forEach((connection) => {
                if (connection instanceof Connection) {
                  connections.add(connection);
                }
              });
            }

            if (Array.isArray(node.connections.gated)) {
              node.connections.gated.forEach((connection) => {
                if (connection instanceof Connection) {
                  gates.add(connection);
                }
              });
            }

            if (
              node.connections.self.length > 0 &&
              node.connections.self[0] instanceof Connection &&
              node.connections.self[0].weight !== 0
            ) {
              selfconns.add(node.connections.self[0]);
            }
          }
        }
      }
    }

    if (inputSize > 0 && outputSize > 0) {
      network.input = inputSize;
      network.output = outputSize;
    } else if (!foundTypes || inputSize === 0 || outputSize === 0) {
      throw new Error(
        'Could not determine input/output nodes. Ensure nodes have their `type` property set to "input" or "output".',
      );
    }

    network.nodes = Array.from(uniqueNodes);
    network.connections = Array.from(connections);
    network.gates = Array.from(gates);
    network.selfconns = Array.from(selfconns);

    if (network.input === 0 || network.output === 0) {
      throw new Error('Constructed network has zero input or output nodes.');
    }

    return network;
  }

  /**
   * Creates a standard multi-layer perceptron network.
   *
   * The returned network is marked with the public `feed-forward` topology
   * intent so acyclic enforcement and slab fast-path eligibility stay aligned
   * with the builder users already chose.
   *
   * @param layers Layer sizes starting with input, followed by hidden layers,
   * and ending with output.
   * @returns The constructed MLP network.
   * @throws {Error} If fewer than three layer sizes are provided.
   */
  static perceptron(...layers: number[]): Network {
    if (layers.length < 3) {
      throw new Error(
        'Invalid MLP configuration: You must specify at least 3 layer sizes (input, hidden, output).',
      );
    }

    const inputSize = layers[0];
    const outputSize = layers[layers.length - 1];
    const minHidden = Math.min(inputSize, outputSize) + 1;

    const inputLayer = Layer.dense(inputSize);
    inputLayer.set({ type: 'input' });

    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
      let layerSize = layers[layerIndex];
      if (layerIndex !== layers.length - 1 && layerSize < minHidden) {
        layerSize = minHidden;
      }

      const currentLayer = Layer.dense(layerSize);
      if (layerIndex === layers.length - 1) {
        currentLayer.set({ type: 'output' });
      }

      (previousLayer as Layer).connect(
        currentLayer,
        methods.groupConnection.ALL_TO_ALL,
      );
      nodes.push(currentLayer);
      previousLayer = currentLayer;
    }

    const network = Architect.construct(nodes);

    (network as unknown as { layers: Layer[] }).layers = nodes.filter(
      (node) => node instanceof Layer,
    );
    network.setTopologyIntent('feed-forward');

    return network;
  }

  /**
   * Creates a randomly structured network based on node counts and connection
   * options.
   *
   * @param input The number of input nodes.
   * @param hidden The number of hidden nodes to add.
   * @param output The number of output nodes.
   * @param options Optional configuration for connection counts and gates.
   * @returns The constructed randomized network.
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
    } = {},
  ): Network {
    const {
      connections = hidden * 2,
      backconnections = 0,
      selfconnections = 0,
      gates = 0,
    } = options;

    const network = new Network(input, output);

    for (let hiddenNodeIndex = 0; hiddenNodeIndex < hidden; hiddenNodeIndex++) {
      network.mutate(methods.mutation.ADD_NODE as never);
    }

    for (
      let connectionIndex = 0;
      connectionIndex < connections - hidden;
      connectionIndex++
    ) {
      network.mutate(methods.mutation.ADD_CONN as never);
    }

    for (
      let connectionIndex = 0;
      connectionIndex < backconnections;
      connectionIndex++
    ) {
      network.mutate(methods.mutation.ADD_BACK_CONN as never);
    }

    for (
      let connectionIndex = 0;
      connectionIndex < selfconnections;
      connectionIndex++
    ) {
      network.mutate(methods.mutation.ADD_SELF_CONN as never);
    }

    for (let gateIndex = 0; gateIndex < gates; gateIndex++) {
      network.mutate(methods.mutation.ADD_GATE as never);
    }

    return network;
  }

  /**
   * Creates a Long Short-Term Memory network.
   *
   * @param layerArgs Layer sizes plus an optional trailing options object.
   * @returns The constructed LSTM network.
   * @throws {Error} If fewer than three numerical layer sizes are provided.
   * @throws {Error} If any layer size is not a positive finite number.
   */
  static lstm(...layerArgs: (number | { inputToOutput?: boolean })[]): Network {
    let options: { inputToOutput?: boolean } = {};

    if (
      layerArgs.length > 0 &&
      typeof layerArgs[layerArgs.length - 1] === 'object' &&
      layerArgs[layerArgs.length - 1] !== null &&
      !Array.isArray(layerArgs[layerArgs.length - 1])
    ) {
      options = layerArgs.pop() as { inputToOutput?: boolean };
    }

    if (
      !layerArgs.every(
        (argument): argument is number =>
          typeof argument === 'number' &&
          Number.isFinite(argument) &&
          argument > 0,
      )
    ) {
      throw new Error(
        'Invalid LSTM layer arguments: All layer sizes must be positive finite numbers.',
      );
    }

    const layers = layerArgs as number[];

    if (layers.length < 3) {
      throw new Error(
        'Invalid LSTM configuration: You must specify at least 3 layer sizes (input, hidden..., output).',
      );
    }

    const { inputToOutput = true } = options;
    const inputLayerSize = layers.shift()!;
    const outputLayerSize = layers.pop()!;

    const inputLayer = Layer.dense(inputLayerSize);
    inputLayer.set({ type: 'input' });

    const outputLayer = Layer.dense(outputLayerSize);
    outputLayer.set({ type: 'output' });

    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    for (const layerSize of layers) {
      const lstmLayer = Layer.lstm(layerSize);
      (previousLayer as Layer).connect(lstmLayer);
      nodes.push(lstmLayer);
      previousLayer = lstmLayer;
    }

    (previousLayer as Layer).connect(outputLayer);
    nodes.push(outputLayer);

    if (inputToOutput) {
      inputLayer.connect(outputLayer, methods.groupConnection.ALL_TO_ALL);
    }

    const network = Architect.construct(nodes);
    network.input = inputLayerSize;
    network.output = outputLayerSize;

    return network;
  }

  /**
   * Creates a Gated Recurrent Unit network.
   *
   * @param layers Layer sizes starting with input and ending with output.
   * @returns The constructed GRU network.
   * @throws {Error} If fewer than three layer sizes are provided.
   */
  static gru(...layers: number[]): Network {
    if (layers.length < 3) {
      throw new Error(
        'Invalid GRU configuration: You must specify at least 3 layer sizes (input, hidden..., output).',
      );
    }

    const inputLayerSize = layers.shift()!;
    const outputLayerSize = layers.pop()!;

    const inputLayer = Layer.dense(inputLayerSize);
    inputLayer.set({ type: 'input' });

    const outputLayer = Layer.dense(outputLayerSize);
    outputLayer.set({ type: 'output' });

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
    network.input = inputLayerSize;
    network.output = outputLayerSize;

    return network;
  }

  /**
   * Creates a Hopfield network.
   *
   * @param size The number of nodes in the network.
   * @returns The constructed Hopfield network.
   */
  static hopfield(size: number): Network {
    const inputLayer = Layer.dense(size);
    const outputLayer = Layer.dense(size);

    inputLayer.connect(outputLayer, methods.groupConnection.ALL_TO_ALL);

    inputLayer.set({ type: 'input' });
    outputLayer.set({ squash: methods.Activation.step, type: 'output' });

    return Architect.construct([inputLayer, outputLayer]);
  }

  /**
   * Creates a Nonlinear AutoRegressive network with eXogenous inputs.
   *
   * @param inputSize The exogenous input size at each time step.
   * @param hiddenLayers Hidden layer sizes, or zero / empty for none.
   * @param outputSize The prediction output size.
   * @param previousInput The number of delayed input steps.
   * @param previousOutput The number of delayed output steps.
   * @returns The constructed NARX network.
   */
  static narx(
    inputSize: number,
    hiddenLayers: number | number[],
    outputSize: number,
    previousInput: number,
    previousOutput: number,
  ): Network {
    if (!Array.isArray(hiddenLayers)) {
      hiddenLayers = hiddenLayers > 0 ? [hiddenLayers] : [];
    }

    const input = Layer.dense(inputSize);
    const inputMemory = Layer.memory(inputSize, previousInput);
    const output = Layer.dense(outputSize);
    const outputMemory = Layer.memory(outputSize, previousOutput);

    input.set({ type: 'input' });
    output.set({ type: 'output' });

    input.connect(inputMemory, methods.groupConnection.ONE_TO_ONE, 1);
    output.connect(outputMemory, methods.groupConnection.ONE_TO_ONE, 1);

    const hidden: Layer[] = [];
    let previousLayer: Layer | Group = input;
    const nodes: (Layer | Group)[] = [input, inputMemory, outputMemory];
    let firstProcessingLayer: Layer | Group;

    if (hiddenLayers.length > 0) {
      for (let layerIndex = 0; layerIndex < hiddenLayers.length; layerIndex++) {
        const layerSize = hiddenLayers[layerIndex];
        const hiddenLayer = Layer.dense(layerSize);
        hidden.push(hiddenLayer);
        nodes.push(hiddenLayer);

        (previousLayer as Layer).connect(
          hiddenLayer,
          methods.groupConnection.ALL_TO_ALL,
        );
        previousLayer = hiddenLayer;

        if (layerIndex === 0) {
          firstProcessingLayer = hiddenLayer;
        }
      }

      (previousLayer as Layer).connect(
        output,
        methods.groupConnection.ALL_TO_ALL,
      );
    } else {
      input.connect(output, methods.groupConnection.ALL_TO_ALL);
      firstProcessingLayer = output;
    }

    nodes.push(output);

    inputMemory.connect(
      firstProcessingLayer!,
      methods.groupConnection.ALL_TO_ALL,
    );
    outputMemory.connect(
      firstProcessingLayer!,
      methods.groupConnection.ALL_TO_ALL,
    );

    const network = Architect.construct(nodes);
    network.input = inputSize;
    network.output = outputSize;

    return network;
  }

  /**
   * Enforces the minimum hidden layer size rule on a network.
   *
   * @param network The network to normalize.
   * @returns The same network with hidden layers grown to the minimum size when needed.
   */
  static enforceMinimumHiddenLayerSizes(network: Network): Network {
    if (!network.layers || network.layers.length <= 2) {
      return network;
    }

    const minSize = Math.min(network.input, network.output) + 1;

    for (
      let layerIndex = 1;
      layerIndex < network.layers.length - 1;
      layerIndex++
    ) {
      const hiddenLayer = network.layers[layerIndex];
      const currentSize = hiddenLayer.nodes.length;

      if (currentSize < minSize) {
        for (let nodeIndex = currentSize; nodeIndex < minSize; nodeIndex++) {
          const newNode = new Node('hidden');
          hiddenLayer.nodes.push(newNode);
          network.nodes.push(newNode);

          if (layerIndex > 0 && network.layers[layerIndex - 1].output) {
            for (const previousNode of network.layers[layerIndex - 1].output!
              .nodes) {
              const connections = previousNode.connect(newNode);
              network.connections.push(...connections);
            }
          }

          if (
            layerIndex < network.layers.length - 1 &&
            network.layers[layerIndex + 1].output
          ) {
            for (const nextNode of network.layers[layerIndex + 1].output!
              .nodes) {
              const connections = newNode.connect(nextNode);
              network.connections.push(...connections);
            }
          }

          if (hiddenLayer.output && Array.isArray(hiddenLayer.output.nodes)) {
            hiddenLayer.output.nodes.push(newNode);
          }
        }
      }
    }

    return network;
  }
}
