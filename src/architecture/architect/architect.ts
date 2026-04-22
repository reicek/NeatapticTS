/**
 * Core architect chapter for the architecture surface.
 *
 * This folder owns the top-level builder entrypoint that turns folderized
 * node, connection, group, layer, and network chapters into named neural
 * network presets.
 *
 * One practical selection guide is to choose the smallest builder that matches
 * the kind of memory your task actually needs:
 *
 * - `perceptron()` for static input-to-output mappings and dense feed-forward baselines,
 * - `randomSparse()` when evolution should begin from a lighter graph with room to grow,
 * - `narx()` when a short explicit window of past inputs and outputs is enough,
 * - `gru()` and `lstm()` when the state itself should be learned inside gated recurrent blocks.
 *
 * Read this chapter in three passes:
 *
 * 1. start with `construct()` to see how pre-wired primitives become one
 *    `Network` instance,
 * 2. continue to `perceptron()`, `randomSparse()`, and `random()` when you
 *    want feed-forward and topology-search-friendly builders,
 * 3. finish with `lstm()`, `gru()`, `hopfield()`, and `narx()` when you need
 *    recurrent presets built from the lower-level architecture chapters.
 */
import Node from '../node/node';
import Layer from '../layer/layer';
import Group from '../group/group';
import Network from '../network/network';
import {
  appendTemporalDescriptorSet,
  buildGruTemporalDescriptorSet,
  buildLstmTemporalDescriptorSet,
  buildNarxMemoryTemporalDescriptorSet,
  splitGruLayerNodes,
  splitLstmLayerNodes,
} from '../network/network.temporal.extensions.utils';
import type { MutationMethod } from '../network/network.types';
import * as methods from '../../methods/methods';
import Connection from '../connection/connection';
import {
  ArchitectInputOutputTypeResolutionError,
  ArchitectInvalidGruConfigurationError,
  ArchitectInvalidGruLayerArgumentsError,
  ArchitectInvalidLstmConfigurationError,
  ArchitectInvalidLstmLayerArgumentsError,
  ArchitectInvalidPerceptronConfigurationError,
  ArchitectInvalidRandomSparseConfigurationError,
  ArchitectZeroInputOutputNodesError,
} from './architect.errors';

type ArchitectRecurrentShortcutOptions = {
  inputToOutput?: boolean;
};

type ArchitectRandomSparseOptions = {
  seed?: number;
  connections?: number;
  backConnections?: number;
  selfConnections?: number;
  gates?: number;
};

type ArchitectLegacyRandomOptions = {
  seed?: number;
  connections?: number;
  backconnections?: number;
  selfconnections?: number;
  gates?: number;
};

type NormalizedArchitectRandomSparseOptions = {
  seed?: number;
  connections: number;
  backConnections: number;
  selfConnections: number;
  gates: number;
};

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
 * Some methods like `randomSparse` are inspired by concepts discussed here.
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
    } else {
      throw new ArchitectInputOutputTypeResolutionError(
        'Could not determine input/output nodes. Ensure nodes have their `type` property set to "input" or "output".',
      );
    }

    network.nodes = Array.from(uniqueNodes);
    network.connections = Array.from(connections);
    network.gates = Array.from(gates);
    network.selfconns = Array.from(selfconns);
    network.refreshExplicitIORoles();

    if (network.input === 0 || network.output === 0) {
      throw new ArchitectZeroInputOutputNodesError(
        'Constructed network has zero input or output nodes.',
      );
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
    * Recommended starting knobs:
    *
    * - gradient training: start with `iterations` in the low thousands, `rate`
    *   around `0.1` to `0.3`, and `momentum` around `0.9` for small normalized tasks,
    * - evolution: use this as the deterministic baseline profile with a fixed
    *   `seed`, `popsize` around `50` to `150`, and moderate `mutationRate`.
    *
    * Common pitfalls:
    *
    * - expecting a feed-forward graph to remember prior timesteps,
    * - keeping the exact same sample order every epoch on small i.i.d. datasets
    *   when some order randomization would reduce training bias.
   *
   * @param layers Layer sizes starting with input, followed by hidden layers,
   * and ending with output.
   * @returns The constructed MLP network.
   * @throws {Error} If fewer than three layer sizes are provided.
    * @example
    * ```ts
    * const network = Architect.perceptron(2, 4, 1);
    *
    * network.train(
    *   [
    *     { input: [0, 0], output: [0] },
    *     { input: [0, 1], output: [1] },
    *     { input: [1, 0], output: [1] },
    *     { input: [1, 1], output: [0] },
    *   ],
    *   {
    *     iterations: 3_000,
    *     rate: 0.3,
    *     momentum: 0.9,
    *   },
    * );
    *
    * const output = network.activate([1, 0])[0];
    * ```
   */
  static perceptron(...layers: number[]): Network {
    if (layers.length < 3) {
      throw new ArchitectInvalidPerceptronConfigurationError(
        'Invalid MLP configuration: You must specify at least 3 layer sizes (input, hidden, output).',
      );
    }

    const inputSize = layers[0];
    const outputSize = layers.at(-1)!;
    const minHidden = Math.min(inputSize, outputSize) + 1;

    const inputLayer = Layer.dense(inputSize, 'input');

    const nodes: (Layer | Group)[] = [inputLayer];
    let previousLayer: Layer | Group = inputLayer;

    for (let layerIndex = 1; layerIndex < layers.length; layerIndex++) {
      let layerSize = layers[layerIndex];
      if (layerIndex !== layers.length - 1 && layerSize < minHidden) {
        layerSize = minHidden;
      }

      const currentLayer =
        layerIndex === layers.length - 1
          ? Layer.dense(layerSize, 'output')
          : Layer.dense(layerSize);

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
   * This compatibility wrapper preserves the historical `random()` surface
   * while forwarding to the stricter `randomSparse()` builder that uses the
   * Phase 2 sparse-profile vocabulary.
   *
   * @param input The number of input nodes.
   * @param hidden The number of hidden nodes to add.
   * @param output The number of output nodes.
   * @param options Optional legacy configuration using lowercase back/self keys.
   * @returns The constructed randomized network.
   */
  static random(
    input: number,
    hidden: number,
    output: number,
    options: ArchitectLegacyRandomOptions = {},
  ): Network {
    return Architect.randomSparse(input, hidden, output, {
      seed: options.seed,
      connections: options.connections,
      backConnections: options.backconnections,
      selfConnections: options.selfconnections,
      gates: options.gates,
    });
  }

  /**
   * Creates a sparse random starting graph for topology-oriented search.
   *
   * This builder is the explicit sparse-profile entrypoint for the public
   * architecture set. It accepts camelCase option names, validates the request
   * up front, and throws a clear error as soon as one requested structural edit
   * cannot be satisfied instead of silently leaving the graph underspecified.
    *
    * Recommended starting knobs:
    *
    * - topology search: keep `connections` only a little above the hidden-node
    *   count, then add `backConnections`, `selfConnections`, and `gates`
    *   gradually as the task proves it benefits from richer recurrence,
    * - deterministic replay: set `seed` whenever you want to compare topology
    *   changes instead of random initializers,
    * - NEAT seeding: pair this preset with `popsize` around `100` and moderate
    *   `mutationRate` so evolution still has structural headroom left.
    *
    * Common pitfalls:
    *
    * - requesting more structural edits than the graph can satisfy,
    * - starting from a graph that is already so dense that topology search has
    *   little left to discover.
   *
   * @param input The number of input nodes.
   * @param hidden The number of hidden nodes to add.
   * @param output The number of output nodes.
   * @param options Optional sparse-structure counts for forward connections,
    * back connections, self connections, gates, and an optional deterministic seed.
   * @returns The constructed sparse random network.
   * @throws {Error} If the dimensions are invalid or the requested sparse
   * structure cannot be fully created.
   *
   * @example
   * ```ts
    * const network = Architect.randomSparse(3, 6, 1, {
    *   connections: 10,
    *   backConnections: 1,
   *   selfConnections: 1,
    *   seed: 7,
   * });
    *
    * const output = network.activate([0.2, -0.1, 0.8])[0];
   * ```
   */
  static randomSparse(
    input: number,
    hidden: number,
    output: number,
    options: ArchitectRandomSparseOptions = {},
  ): Network {
    validateRandomSparseDimensions(input, hidden, output);
    const normalizedOptions = normalizeRandomSparseOptions(hidden, options);
    const network = new Network(input, output, {
      seed: normalizedOptions.seed,
    });

    for (let hiddenNodeIndex = 0; hiddenNodeIndex < hidden; hiddenNodeIndex++) {
      applyRandomSparseMutationOrThrow(
        network,
        methods.mutation.ADD_NODE as never,
        () => network.nodes.length,
        `Invalid RandomSparse configuration: unable to add hidden node ${hiddenNodeIndex + 1} of ${hidden}.`,
      );
    }

    for (
      let connectionIndex = 0;
      connectionIndex < Math.max(0, normalizedOptions.connections - hidden);
      connectionIndex++
    ) {
      applyRandomSparseMutationOrThrow(
        network,
        methods.mutation.ADD_CONN as never,
        () => network.connections.length,
        `Invalid RandomSparse configuration: unable to add forward connection ${connectionIndex + 1} of ${Math.max(0, normalizedOptions.connections - hidden)} because no unused forward pairs remain.`,
      );
    }

    for (
      let connectionIndex = 0;
      connectionIndex < normalizedOptions.backConnections;
      connectionIndex++
    ) {
      applyRandomSparseMutationOrThrow(
        network,
        methods.mutation.ADD_BACK_CONN as never,
        () => network.connections.length,
        `Invalid RandomSparse configuration: unable to add back connection ${connectionIndex + 1} of ${normalizedOptions.backConnections}.`,
      );
    }

    for (
      let connectionIndex = 0;
      connectionIndex < normalizedOptions.selfConnections;
      connectionIndex++
    ) {
      applyRandomSparseMutationOrThrow(
        network,
        methods.mutation.ADD_SELF_CONN as never,
        () => network.selfconns.length,
        `Invalid RandomSparse configuration: unable to add self connection ${connectionIndex + 1} of ${normalizedOptions.selfConnections}.`,
      );
    }

    for (let gateIndex = 0; gateIndex < normalizedOptions.gates; gateIndex++) {
      applyRandomSparseMutationOrThrow(
        network,
        methods.mutation.ADD_GATE as never,
        () => network.gates.length,
        `Invalid RandomSparse configuration: unable to add gate ${gateIndex + 1} of ${normalizedOptions.gates}.`,
      );
    }

    return network;
  }

  /**
   * Creates a Long Short-Term Memory network.
   *
   * This builder keeps the LSTM graph explicit: each recurrent block is
   * assembled from gate groups, a memory-cell group, and an output block using
   * the same primitive wiring surface used elsewhere in the architecture layer.
   *
   * The optional `inputToOutput` shortcut preserves the historical builder
   * behavior by default. Disable it when you want the public preset to route
   * information strictly through the recurrent block stack.
    *
    * Recommended starting knobs:
    *
    * - start with one small block such as `Architect.lstm(input, 4, output)`
    *   before stacking multiple recurrent stages,
    * - supervised fine-tuning: keep `rate` low, usually around `0.001` to
    *   `0.01`, and feed one ordered sequence at a time,
    * - evolution: keep `seed` fixed for comparisons and prefer `NARX` first
    *   when a short explicit window already explains the task.
    *
    * Common pitfalls:
    *
    * - forgetting that `inputToOutput` defaults to `true`,
    * - flattening unrelated episodes into one activation stream without calling
    *   `clear()`,
    * - shuffling timesteps inside a sequence and destroying the temporal story
    *   the recurrent block is supposed to learn.
   *
   * @param layerArgs Layer sizes plus an optional trailing options object.
   * @returns The constructed LSTM network.
   * @throws {Error} If fewer than three numerical layer sizes are provided.
   * @throws {Error} If one or more layer sizes are not positive finite numbers.
    * @example
    * ```ts
    * const network = Architect.lstm(1, 4, 1, { inputToOutput: false });
    *
    * const firstPass = [0.1, 0.4, 0.2].map(
    *   (value) => network.activate([value])[0],
    * );
    *
    * network.clear();
    *
    * const secondPass = [0.1, 0.4, 0.2].map(
    *   (value) => network.activate([value])[0],
    * );
    *
    * console.log(firstPass, secondPass);
    * ```
   */
  static lstm(...layerArgs: (number | ArchitectRecurrentShortcutOptions)[]): Network {
    let options: ArchitectRecurrentShortcutOptions = {};
    const trailingArgument = layerArgs.at(-1);

    if (
      layerArgs.length > 0 &&
      typeof trailingArgument === 'object' &&
      trailingArgument !== null &&
      !Array.isArray(trailingArgument)
    ) {
      options = layerArgs.pop() as ArchitectRecurrentShortcutOptions;
    }

    if (
      !layerArgs.every(
        (argument): argument is number =>
          typeof argument === 'number' &&
          Number.isFinite(argument) &&
          argument > 0,
      )
    ) {
      throw new ArchitectInvalidLstmLayerArgumentsError(
        'Invalid LSTM layer arguments: All layer sizes must be positive finite numbers.',
      );
    }

    const layers = layerArgs as number[];

    if (layers.length < 3) {
      throw new ArchitectInvalidLstmConfigurationError(
        'Invalid LSTM configuration: You must specify at least 3 layer sizes (input, hidden..., output).',
      );
    }

    const { inputToOutput = true } = options;
    const inputLayerSize = layers.shift()!;
    const outputLayerSize = layers.pop()!;

    const inputLayer = Layer.dense(inputLayerSize, 'input');

    const outputLayer = Layer.dense(outputLayerSize, 'output');

    const nodes: (Layer | Group)[] = [inputLayer];
    const recurrentLayers: Array<{ layer: Layer; size: number }> = [];
    let previousLayer: Layer | Group = inputLayer;

    for (const layerSize of layers) {
      const lstmLayer = Layer.lstm(layerSize);
      (previousLayer as Layer).connect(lstmLayer);
      nodes.push(lstmLayer);
      recurrentLayers.push({ layer: lstmLayer, size: layerSize });
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
    recurrentLayers.forEach(({ layer, size }) => {
      const roleNodes = splitLstmLayerNodes(layer.nodes, size)!;
      appendTemporalDescriptorSet(
        network,
        buildLstmTemporalDescriptorSet(network, roleNodes),
      );
    });

    return network;
  }

  /**
   * Creates a Gated Recurrent Unit network.
   *
   * This builder keeps the GRU graph explicit: update, inverse-update, reset,
   * memory, output, and previous-output groups are all wired from the public
   * primitive surface rather than hidden behind a fused recurrent runtime.
   *
   * The optional `inputToOutput` shortcut is disabled by default so existing
   * GRU topologies keep their historical shape. Enable it when you want a
   * direct input-to-readout path in addition to the recurrent block stack.
    *
    * Recommended starting knobs:
    *
    * - start with one small recurrent block before stacking multiple GRU stages,
    * - supervised fine-tuning: keep `rate` low, usually around `0.001` to
    *   `0.01`, and preserve sequence order,
    * - evolution: compare runs with a fixed `seed`, moderate `mutationRate`,
    *   and only enable `inputToOutput` when the task clearly benefits from a
    *   direct readout shortcut.
    *
    * Common pitfalls:
    *
    * - forgetting to call `clear()` between independent sequences or episodes,
    * - enabling the shortcut before checking whether the recurrent block alone
    *   already solves the task,
    * - shuffling timesteps rather than whole sequences.
   *
   * @param layerArgs Layer sizes plus an optional trailing options object.
   * @returns The constructed GRU network.
   * @throws {Error} If one or more layer sizes are not positive finite numbers.
   * @throws {Error} If fewer than three layer sizes are provided.
    * @example
    * ```ts
    * const network = Architect.gru(1, 3, 1, { inputToOutput: true });
    *
    * const outputs = [0.1, 0.4, 0.2].map(
    *   (value) => network.activate([value])[0],
    * );
    *
    * network.clear();
    *
    * console.log(outputs);
    * ```
   */
  static gru(...layerArgs: (number | ArchitectRecurrentShortcutOptions)[]): Network {
    let options: ArchitectRecurrentShortcutOptions = {};
    const trailingArgument = layerArgs.at(-1);

    if (
      layerArgs.length > 0 &&
      typeof trailingArgument === 'object' &&
      trailingArgument !== null &&
      !Array.isArray(trailingArgument)
    ) {
      options = layerArgs.pop() as ArchitectRecurrentShortcutOptions;
    }

    if (
      !layerArgs.every(
        (argument): argument is number =>
          typeof argument === 'number' &&
          Number.isFinite(argument) &&
          argument > 0,
      )
    ) {
      throw new ArchitectInvalidGruLayerArgumentsError(
        'Invalid GRU layer arguments: All layer sizes must be positive finite numbers.',
      );
    }

    const layers = layerArgs as number[];

    if (layers.length < 3) {
      throw new ArchitectInvalidGruConfigurationError(
        'Invalid GRU configuration: You must specify at least 3 layer sizes (input, hidden..., output).',
      );
    }

    const { inputToOutput = false } = options;

    const inputLayerSize = layers.shift()!;
    const outputLayerSize = layers.pop()!;

    const inputLayer = Layer.dense(inputLayerSize, 'input');

    const outputLayer = Layer.dense(outputLayerSize, 'output');

    const nodes: (Layer | Group)[] = [inputLayer];
    const recurrentLayers: Array<{ layer: Layer; size: number }> = [];
    let previousLayer: Layer | Group = inputLayer;

    for (const blockSize of layers) {
      const gruLayer = Layer.gru(blockSize);
      (previousLayer as Layer).connect(gruLayer);
      nodes.push(gruLayer);
      recurrentLayers.push({ layer: gruLayer, size: blockSize });
      previousLayer = gruLayer;
    }

    (previousLayer as Layer).connect(outputLayer);
    nodes.push(outputLayer);

    if (inputToOutput) {
      inputLayer.connect(outputLayer, methods.groupConnection.ALL_TO_ALL);
    }

    const network = Architect.construct(nodes);
    network.input = inputLayerSize;
    network.output = outputLayerSize;
    recurrentLayers.forEach(({ layer, size }) => {
      const roleNodes = splitGruLayerNodes(layer.nodes, size)!;
      appendTemporalDescriptorSet(
        network,
        buildGruTemporalDescriptorSet(network, roleNodes),
      );
    });

    return network;
  }

  /**
   * Creates a Hopfield network.
   *
   * @param size The number of nodes in the network.
   * @returns The constructed Hopfield network.
   */
  static hopfield(size: number): Network {
    const inputLayer = Layer.dense(size, 'input');
    const outputLayer = Layer.dense(size, 'output');

    inputLayer.connect(outputLayer, methods.groupConnection.ALL_TO_ALL);

    outputLayer.set({ squash: methods.Activation.step });

    return Architect.construct([inputLayer, outputLayer]);
  }

  /**
   * Creates a Nonlinear AutoRegressive network with eXogenous inputs.
   *
   * This is the smallest stateful preset in the public builder surface. The
   * main processing path receives the current exogenous input plus two explicit
   * delay lines: one for recent inputs and one for recent outputs. That keeps
   * the temporal story readable for debugging, visualization, and evolution
   * work without immediately jumping to gated cells.
   *
   * Both delay lines are built from `Layer.memory(...)` blocks. Those memory
   * blocks use identity activation, zero bias, and one-to-one unit carry links,
   * so remembered values behave like a deterministic rolling window rather than
   * a learned recurrent cell.
  *
  * Recommended starting knobs:
  *
  * - start with short shelves such as `previousInput` and `previousOutput`
  *   in the `1` to `3` range,
  * - prefer `narx()` before gated builders when the task is really "predict
  *   from a small rolling window",
  * - keep sequence order intact during fine-tuning or evaluation and reset the
  *   carried state between independent runs.
  *
  * Common pitfalls:
  *
  * - forgetting that the delay lines intentionally carry state until
  *   `clear()` is called,
  * - shuffling timesteps from different sequences together,
  * - using long memory shelves when a shorter explicit window would be easier
  *   to debug, visualize, and evolve.
   *
   * Clear-state guidance: call `network.clear()` before starting a new
   * independent sequence, episode, or evaluation run. If you keep activating
   * the same runtime without clearing it, the delay lines intentionally carry
   * their terminal state into the next activation stream.
   *
   * @param inputSize The exogenous input size at each time step.
   * @param hiddenLayers Hidden layer sizes, or zero / empty for none.
   * @param outputSize The prediction output size.
   * @param previousInput The number of delayed input steps.
   * @param previousOutput The number of delayed output steps.
   * @returns The constructed NARX network.
   * @example
   * ```ts
   * const network = Architect.narx(1, [4], 1, 2, 1);
  * const firstSequence = [[0.2], [0.7], [0.4]];
  * const secondSequence = [[1], [0], [0]];
   *
  * const firstOutputs = firstSequence.map((input) => network.activate(input)[0]);
   *
   * network.clear();
   *
  * const secondOutputs = secondSequence.map((input) => network.activate(input)[0]);
  *
  * console.log(firstOutputs, secondOutputs);
   * ```
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

    const input = Layer.dense(inputSize, 'input');
    const inputMemory = Layer.memory(inputSize, previousInput);
    const output = Layer.dense(outputSize, 'output');
    const outputMemory = Layer.memory(outputSize, previousOutput);

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
    appendTemporalDescriptorSet(
      network,
      buildNarxMemoryTemporalDescriptorSet(
        network,
        'input',
        resolveMemoryBlockNodes(inputMemory),
      ),
    );
    appendTemporalDescriptorSet(
      network,
      buildNarxMemoryTemporalDescriptorSet(
        network,
        'output',
        resolveMemoryBlockNodes(outputMemory),
      ),
    );

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

function resolveMemoryBlockNodes(memoryLayer: Layer): Node[][] {
  return (memoryLayer.nodes as unknown as Group[])
    .filter((layerNode) => layerNode instanceof Group)
    .map((memoryBlock) => memoryBlock.nodes);
}

function validateRandomSparseDimensions(
  input: number,
  hidden: number,
  output: number,
): void {
  assertNonNegativeInteger(input, 'input', true);
  assertNonNegativeInteger(hidden, 'hidden');
  assertNonNegativeInteger(output, 'output', true);
}

function normalizeRandomSparseOptions(
  hidden: number,
  options: ArchitectRandomSparseOptions,
): NormalizedArchitectRandomSparseOptions {
  const normalizedOptions = {
    seed: options.seed,
    connections: options.connections ?? hidden * 2,
    backConnections: options.backConnections ?? 0,
    selfConnections: options.selfConnections ?? 0,
    gates: options.gates ?? 0,
  };

  assertFiniteSeed(normalizedOptions.seed);

  assertNonNegativeInteger(normalizedOptions.connections, 'connections');
  assertNonNegativeInteger(
    normalizedOptions.backConnections,
    'backConnections',
  );
  assertNonNegativeInteger(
    normalizedOptions.selfConnections,
    'selfConnections',
  );
  assertNonNegativeInteger(normalizedOptions.gates, 'gates');

  return normalizedOptions;
}

function assertFiniteSeed(seed: number | undefined): void {
  if (seed !== undefined && !Number.isFinite(seed)) {
    throw new ArchitectInvalidRandomSparseConfigurationError(
      'Invalid RandomSparse configuration: seed must be a finite number.',
    );
  }
}

function assertNonNegativeInteger(
  value: number,
  label: string,
  requirePositive = false,
): void {
  const isValidInteger = Number.isInteger(value);
  const passesLowerBound = requirePositive ? value > 0 : value >= 0;

  if (!isValidInteger || !passesLowerBound) {
    throw new ArchitectInvalidRandomSparseConfigurationError(
      `Invalid RandomSparse configuration: ${label} must be ${requirePositive ? 'a positive integer' : 'a non-negative integer'}.`,
    );
  }
}

function applyRandomSparseMutationOrThrow(
  network: Network,
  method: MutationMethod,
  resolveCount: () => number,
  errorMessage: string,
): void {
  const countBeforeMutation = resolveCount();
  network.mutate(method as never);
  const countAfterMutation = resolveCount();

  if (countAfterMutation === countBeforeMutation) {
    throw new ArchitectInvalidRandomSparseConfigurationError(errorMessage);
  }
}
