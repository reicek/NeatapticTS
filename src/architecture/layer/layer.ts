/**
 * Core layer chapter for the architecture surface.
 *
 * This folder owns the public `Layer` class that turns node-level primitives
 * into reusable feed-forward and recurrent building blocks. The helper files in
 * this chapter keep activation, connection, propagation, and factory policies
 * focused, while this file preserves the orchestration story that readers and
 * callers actually meet first.
 *
 * If `Node` is the single-neuron chapter and `Network` is the whole-graph
 * chapter, `Layer` is the middle shelf that lets builders talk in model-sized
 * blocks. Dense stages, recurrent cells, normalization passes, and memory-
 * shaped motifs all need more intent than a raw list of nodes, but far less
 * ceremony than constructing an entire network by hand.
 *
 * That middle shelf matters for two audiences at once. Callers want one place
 * to say "make a dense block" or "connect this stage to that stage" without
 * manually pushing node arrays around. Maintainers want activation, wiring,
 * propagation, and factory mechanics separated so the public API can stay easy
 * to read while the underlying policies continue to evolve. This folderized
 * chapter is how the repo serves both goals at the same time.
 *
 * One useful mental model is to place `Layer` between `Group` and `Network`.
 * `Group` exposes a reusable cluster of nodes plus wiring vocabulary. `Layer`
 * adds the stronger promise that the cluster represents a recognizable model
 * stage with standard entrypoints such as `activate()`, `propagate()`,
 * `connect()`, and factory constructors. `Network` then chains many of those
 * stages into a runnable, mutable graph.
 *
 * A second mental model is to treat `layer.ts` as a public facade over several
 * helper shelves. Readers should start here because this file owns the stable
 * orchestration story. The helper files exist to keep concerns narrow: one set
 * handles activation and propagation, another handles wiring and guards, and
 * the factory helpers explain how dense, recurrent, normalization, and
 * experimental layer families are assembled.
 *
 * The descriptor surface follows the same rule as the rest of the primitive
 * chapter: factory helpers already stamp default family metadata when they can,
 * so callers usually only reach for `describe(...)` when a block needs a stable
 * human-facing name such as `encoder`, `memoryShelf`, or `policyHead`.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Node[Node primitives]:::base --> Group[Group blocks]:::base
 *   Group --> Layer[Layer model stage]:::accent
 *   Layer --> Network[Network orchestration]:::base
 *   Layer --> Architect[Architect presets]:::base
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   LayerFacade[Layer facade]:::accent --> Runtime[activate propagate clear gate input]:::base
 *   LayerFacade --> Wiring[connection and guard helpers]:::base
 *   LayerFacade --> Factories[dense recurrent normalization experimental factories]:::base
 * ```
 *
 * For background on the broader modeling idea, see Wikipedia contributors,
 * [Artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network).
 * This chapter is narrower: it documents the library boundary that packages
 * those ideas into reusable blocks and helper-owned policies.
 *
 * Read this chapter in three passes:
 *
 * 1. start with the `Layer` class overview to understand why the boundary
 *    exists above raw nodes and groups,
 * 2. continue to `activate()`, `propagate()`, `connect()`, and `input()` when
 *    you need the runtime wiring surface,
 * 3. finish with the static factory methods when you want named layer shapes
 *    such as dense, recurrent, normalization, or experimental blocks.
 *
 * Example: wire a small dense stack using the same block-level API that higher
 * level builders depend on.
 *
 * ```ts
 * const input = Layer.dense(2, 'input');
 * const hidden = Layer.dense(4);
 * const output = Layer.dense(1, 'output');
 *
 * input.describe({ label: 'sensorStage' });
 * hidden.describe({ label: 'hiddenStage' });
 * output.describe({ label: 'policyHead' });
 *
 * input.connect(hidden);
 * hidden.connect(output);
 *
 * input.activate([0, 1]);
 * hidden.activate();
 * const values = output.activate();
 * ```
 *
 * Example: swap in a richer factory-built block without changing the top-level
 * layer-to-layer wiring vocabulary.
 *
 * ```ts
 * const recurrent = Layer.lstm(8);
 * const readout = Layer.dense(2);
 *
 * recurrent.describe({ label: 'controllerCore' });
 * readout.describe({ label: 'readoutHead', intent: 'output' });
 *
 * recurrent.connect(readout);
 * ```
 */

import Node from '../node/node';
import {
  type PrimitiveDescriptor,
  type PrimitiveIntent,
  type PrimitiveMetadata,
  type PrimitiveNodeType,
  resolvePrimitiveIntent,
} from '../node/node';
import Connection from '../connection/connection';
import Group from '../group/group';
import {
  activateLayer as activateLayerUtils,
  createAttentionLayer as createAttentionLayerUtils,
  createBatchNormLayer as createBatchNormLayerUtils,
  createConv1dLayer as createConv1dLayerUtils,
  createDenseLayer as createDenseLayerUtils,
  createGruLayer as createGruLayerUtils,
  createLayerNormLayer as createLayerNormLayerUtils,
  createLstmLayer as createLstmLayerUtils,
  createMemoryLayer as createMemoryLayerUtils,
  clearLayer as clearLayerUtils,
  connectLayer as connectLayerUtils,
  disconnectLayer as disconnectLayerUtils,
  gateLayer as gateLayerUtils,
  inputLayer as inputLayerUtils,
  propagateLayer as propagateLayerUtils,
} from './layer.utils';
import { isGroup as isGroupUtils } from './layer.guard.utils';
import type { LayerLike } from './layer.utils.types';

const DEFAULT_LAYER_DROPOUT = 0;
const DEFAULT_LAYER_TRAINING_MODE = false;
const DEFAULT_TWO_SIDED_DISCONNECT = false;
const DEFAULT_CONV1D_STRIDE = 1;
const DEFAULT_CONV1D_PADDING = 0;
const DEFAULT_ATTENTION_HEADS = 1;
const NODE_INDEX_START = 0;
const NODE_INDEX_STEP = 1;

function isLayerInstance(candidate: unknown): candidate is LayerLike {
  return candidate instanceof Layer;
}

/**
 * Public block-level facade for layer-oriented architecture building.
 *
 * `Layer` is the boundary readers reach for when a graph region should behave
 * like one model stage rather than a loose collection of neurons. It
 * normalizes common operations across dense, recurrent, normalization, and
 * experimental factories so higher-level chapters can compose readable graphs
 * without depending on each helper shelf's private implementation details.
 *
 * This makes the class useful when you want to:
 *
 * - assemble a network from named blocks instead of raw nodes,
 * - reuse the same activation, wiring, and propagation vocabulary across layer
 *   families,
 * - keep factory-specific mechanics below the public API.
 *
 * Dense, recurrent, normalization, convolution, attention, and memory helpers
 * can all stamp their own default family metadata. That means the public layer
 * API stays low ceremony: choose the right factory first, then add
 * `describe(...)` only if a later reader benefits from a clearer boundary name.
 */
export default class Layer {
  /**
   * An array containing all the nodes (neurons or groups) that constitute this layer.
   * The order of nodes might be relevant depending on the layer type and its connections.
   */
  nodes: Node[];

  /**
   * Stores connection information related to this layer. This is often managed
   * by the network or higher-level structures rather than directly by the layer itself.
   * `in`: Incoming connections to the layer's nodes.
   * `out`: Outgoing connections from the layer's nodes.
   * `self`: Self-connections within the layer's nodes.
   */
  connections: { in: Connection[]; out: Connection[]; self: Connection[] };

  /**
   * Represents the primary output group of nodes for this layer.
   * This group is typically used when connecting this layer *to* another layer or group.
   * It might be null if the layer is not yet fully constructed or is an input layer.
   */
  output: Group | null;
  /** Optional human-readable descriptor label for architecture tooling. */
  label: string | null;
  /** Optional semantic intent for architecture tooling and diagnostics. */
  intent: PrimitiveIntent | null;
  /** Optional scalar metadata retained on the primitive boundary. */
  metadata: PrimitiveMetadata;

  /**
   * Dropout rate for this layer (0 to 1). If > 0, all nodes in the layer are masked together during training.
   * Layer-level dropout takes precedence over node-level dropout for nodes in this layer.
   */
  dropout: number = DEFAULT_LAYER_DROPOUT;

  /**
   * Initializes a new Layer instance.
   */
  constructor() {
    this.output = null;
    this.nodes = [];
    this.connections = { in: [], out: [], self: [] };
    this.label = null;
    this.intent = null;
    this.metadata = {};
  }

  /**
   * Attaches optional descriptor metadata to the layer boundary.
   *
   * Use this when a layer represents a named stage such as a readout block,
   * memory shelf, or recurrent cell family that later diagnostics should
   * understand without re-deriving meaning from the internal node order.
   *
   * @param descriptor Optional label, intent, and scalar metadata to merge.
   * @returns Nothing.
    *
    * @example
    * ```ts
    * const readout = Layer.dense(2, 'output');
    *
    * readout.describe({
    *   label: 'readoutHead',
    *   metadata: { stage: 'policy' },
    * });
    * ```
   */
  describe(descriptor: PrimitiveDescriptor): void {
    if (descriptor.label !== undefined) {
      this.label = descriptor.label;
    }

    if (descriptor.intent !== undefined) {
      this.intent = descriptor.intent;
    }

    if (descriptor.metadata !== undefined) {
      this.metadata = {
        ...this.metadata,
        ...descriptor.metadata,
      };
    }
  }

  /**
   * Activates all nodes within the layer, computing their output values.
   *
   * If an input `value` array is provided, it's used as the initial activation
   * for the corresponding nodes in the layer. Otherwise, nodes compute their
   * activation based on their incoming connections.
   *
   * During training, layer-level dropout is applied, masking all nodes in the layer together.
   * During inference, all masks are set to 1.
   *
   * @param value An optional array of activation values to set for the layer's nodes. The length must match the number of nodes.
   * @param training A boolean indicating whether the layer is in training mode. Defaults to false.
   * @returns An array containing the activation value of each node in the layer after activation.
   * @throws {Error} If the provided `value` array's length does not match the number of nodes in the layer.
   */
  activate(
    value?: number[],
    training: boolean = DEFAULT_LAYER_TRAINING_MODE,
  ): number[] {
    // Step 1: Delegate activation to the orchestrated utils flow.
    return activateLayerUtils(
      { nodes: this.nodes, dropout: this.dropout },
      value,
      training,
    );
  }

  /**
   * Propagates the error backward through all nodes in the layer.
   *
   * This is a core step in the backpropagation algorithm used for training.
   * If a `target` array is provided (typically for the output layer), it's used
   * to calculate the initial error for each node. Otherwise, nodes calculate
   * their error based on the error propagated from subsequent layers.
   *
  * @param rate The learning rate, controlling the step size of weight adjustments.
  * @param momentum The momentum factor, used to smooth weight updates and escape local minima.
  * @param target An optional array of target values (expected outputs) for the layer's nodes. The length must match the number of nodes.
   * @throws {Error} If the provided `target` array's length does not match the number of nodes in the layer.
   */
  propagate(rate: number, momentum: number, target?: number[]) {
    // Step 1: Delegate propagation to the orchestrated utils flow.
    propagateLayerUtils({ nodes: this.nodes }, rate, momentum, target);
  }

  /**
   * Connects this layer's output to a target component (Layer, Group, or Node).
   *
   * This method delegates the connection logic primarily to the layer's `output` group
   * or the target layer's `input` method. It establishes the forward connections
   * necessary for signal propagation.
   *
  * @param target The destination Layer, Group, or Node to connect to.
  * @param method The connection method (e.g., `ALL_TO_ALL`, `ONE_TO_ONE`) defining the connection pattern. See `methods.groupConnection`.
  * @param weight An optional fixed weight to assign to all created connections.
   * @returns An array containing the newly created connection objects.
   * @throws {Error} If the layer's `output` group is not defined.
   */
  connect(
    target: Group | Node | LayerLike,
    method?: unknown,
    weight?: number,
  ): Connection[] {
    // Step 1: Delegate connection to the orchestrated utils flow.
    return connectLayerUtils(
      {
        connections: this.connections,
        isLayer: isLayerInstance,
        layer: this,
        nodes: this.nodes,
        output: this.output,
      },
      target,
      method,
      weight,
    );
  }

  /**
   * Applies gating to a set of connections originating from this layer's output group.
   *
   * Gating allows the activity of nodes in this layer (specifically, the output group)
   * to modulate the flow of information through the specified `connections`.
   *
  * @param connections An array of connection objects to be gated.
  * @param method The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.
   * @throws {Error} If the layer's `output` group is not defined.
   */
  gate(connections: Connection[], method: unknown): void {
    // Step 1: Delegate gating to the orchestrated utils flow.
    gateLayerUtils(
      {
        connections: this.connections,
        isLayer: isLayerInstance,
        layer: this,
        nodes: this.nodes,
        output: this.output,
      },
      connections,
      method,
    );
  }

  /**
   * Configures properties for all nodes within the layer.
   *
   * Allows batch setting of common node properties like bias, activation function (`squash`),
   * or node type. If a node within the `nodes` array is actually a `Group` (e.g., in memory layers),
   * the configuration is applied recursively to the nodes within that group.
   *
  * @param values An object containing the properties and their values to set.
   *                 Example: `{ bias: 0.5, squash: methods.Activation.ReLU }`
   */
  set(values: {
    bias?: number;
    squash?: (x: number, derivate?: boolean) => number;
    type?: string;
  }): void {
    for (
      let nodeIndex = NODE_INDEX_START;
      nodeIndex < this.nodes.length;
      nodeIndex += NODE_INDEX_STEP
    ) {
      const node = this.nodes[nodeIndex];

      if (node instanceof Node) {
        if (values.bias !== undefined) {
          node.bias = values.bias;
        }
        if (values.squash !== undefined) {
          node.squash = values.squash;
        }
        if (values.type !== undefined) {
          node.type = values.type;
          node.intent = resolvePrimitiveIntent(values.type);
        }
      } else if (isGroupUtils(node)) {
        (node as Group).set(values);
      }
    }

    if (values.type !== undefined) {
      this.intent = resolvePrimitiveIntent(values.type);
    }
  }

  /**
   * Removes connections between this layer's nodes and a target Group or Node.
   *
  * @param target The Group or Node to disconnect from.
  * @param twosided If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.
   */
  disconnect(target: Group | Node, twosided?: boolean) {
    // Step 1: Delegate disconnection to the orchestrated utils flow.
    disconnectLayerUtils(
      {
        connections: this.connections,
        isLayer: isLayerInstance,
        layer: this,
        nodes: this.nodes,
        output: this.output,
      },
      target,
      twosided ?? DEFAULT_TWO_SIDED_DISCONNECT,
    );
  }

  /**
   * Resets the activation state of all nodes within the layer.
   * This is typically done before processing a new input sequence or sample.
   */
  clear() {
    // Step 1: Delegate clearing to the orchestrated utils flow.
    clearLayerUtils({
      connections: this.connections,
      isLayer: isLayerInstance,
      layer: this,
      nodes: this.nodes,
      output: this.output,
    });
  }

  /**
   * Handles the connection logic when this layer is the *target* of a connection.
   *
   * It connects the output of the `from` layer or group to this layer's primary
   * input mechanism (which is often the `output` group itself, but depends on the layer type).
   * This method is usually called by the `connect` method of the source layer/group.
   *
  * @param from The source Layer or Group connecting *to* this layer.
  * @param method The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
  * @param weight An optional fixed weight for the connections.
   * @returns An array containing the newly created connection objects.
   * @throws {Error} If the layer's `output` group (acting as input target here) is not defined.
   */
  input(
    from: LayerLike | Group,
    method?: unknown,
    weight?: number,
  ): Connection[] {
    // Step 1: Delegate input wiring to the orchestrated utils flow.
    return inputLayerUtils(
      {
        connections: this.connections,
        isLayer: isLayerInstance,
        layer: this,
        nodes: this.nodes,
        output: this.output,
      },
      from,
      method,
      weight,
    );
  }

  /**
   * Creates a standard fully connected (dense) layer.
   *
   * All nodes in the source layer/group will connect to all nodes in this layer
   * when using the default `ALL_TO_ALL` connection method via `layer.input()`.
    * Dense layers also stamp default descriptor metadata (`family: 'dense'`) so
    * later tooling can recognize the block even when the caller never names it.
   *
  * @param size The number of nodes (neurons) in this layer.
  * @param nodeType Optional primitive role assigned to the dense block.
   * @returns A new Layer instance configured as a dense layer.
    *
    * @example
    * ```ts
    * const output = Layer.dense(2, 'output');
    *
    * output.describe({ label: 'policyHead' });
    * ```
   */
  static dense(
    size: number,
    nodeType: PrimitiveNodeType = 'hidden',
  ): Layer {
    // Step 1: Delegate dense layer creation to the utils orchestrator.
    return createDenseLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
      nodeType,
    );
  }

  /**
   * Creates a Long Short-Term Memory (LSTM) layer.
   *
   * LSTMs are a type of recurrent neural network (RNN) cell capable of learning
   * long-range dependencies. This implementation uses standard LSTM architecture
   * with input, forget, and output gates, and a memory cell.
   *
  * @param size The number of LSTM units (and nodes in each gate/cell group).
   * @returns A new Layer instance configured as an LSTM layer.
   */
  static lstm(size: number): Layer {
    // Step 1: Delegate LSTM layer creation to the utils orchestrator.
    return createLstmLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
    );
  }

  /**
   * Creates a Gated Recurrent Unit (GRU) layer.
   *
   * GRUs are another type of recurrent neural network cell, often considered
   * simpler than LSTMs but achieving similar performance on many tasks.
   * They use an update gate and a reset gate to manage information flow.
   *
  * @param size The number of GRU units (and nodes in each gate/cell group).
   * @returns A new Layer instance configured as a GRU layer.
   */
  static gru(size: number): Layer {
    // Step 1: Delegate GRU layer creation to the utils orchestrator.
    return createGruLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
    );
  }

  /**
   * Creates a Memory layer, designed to hold state over a fixed number of time steps.
   *
   * This layer consists of multiple groups (memory blocks), each holding the state
   * from a previous time step. The input connects to the most recent block, and
   * information propagates backward through the blocks. The layer's output
   * concatenates the states of all memory blocks.
   *
  * @param size The number of nodes in each memory block (must match the input size).
  * @param memory The number of time steps to remember (number of memory blocks).
   * @returns A new Layer instance configured as a Memory layer.
   * @throws {Error} If the connecting layer's size doesn't match the memory block `size`.
   */
  static memory(size: number, memory: number): Layer {
    // Step 1: Delegate Memory layer creation to the utils orchestrator.
    return createMemoryLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
      memory,
    );
  }

  /**
   * Creates a batch normalization layer.
   * Applies batch normalization to the activations of the nodes in this layer during activation.
  * @param size The number of nodes in this layer.
   * @returns A new Layer instance configured as a batch normalization layer.
   */
  static batchNorm(size: number): Layer {
    // Step 1: Delegate batch norm creation to the utils orchestrator.
    return createBatchNormLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
    );
  }

  /**
   * Creates a layer normalization layer.
   * Applies layer normalization to the activations of the nodes in this layer during activation.
  * @param size The number of nodes in this layer.
   * @returns A new Layer instance configured as a layer normalization layer.
   */
  static layerNorm(size: number): Layer {
    // Step 1: Delegate layer norm creation to the utils orchestrator.
    return createLayerNormLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
    );
  }

  /**
   * Creates a 1D convolutional layer (stub implementation).
  * @param size Number of output nodes (filters).
  * @param kernelSize Size of the convolution kernel.
  * @param stride Stride of the convolution (default 1).
  * @param padding Padding (default 0).
   * @returns A new Layer instance representing a 1D convolutional layer.
   */
  static conv1d(
    size: number,
    kernelSize: number,
    stride: number = DEFAULT_CONV1D_STRIDE,
    padding: number = DEFAULT_CONV1D_PADDING,
  ): Layer {
    // Step 1: Delegate conv1d creation to the utils orchestrator.
    return createConv1dLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
      kernelSize,
      stride,
      padding,
    );
  }

  /**
   * Creates a multi-head self-attention layer (stub implementation).
  * @param size Number of output nodes.
  * @param heads Number of attention heads (default 1).
   * @returns A new Layer instance representing an attention layer.
   */
  static attention(
    size: number,
    heads: number = DEFAULT_ATTENTION_HEADS,
  ): Layer {
    // Step 1: Delegate attention creation to the utils orchestrator.
    return createAttentionLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: isLayerInstance,
      },
      size,
      heads,
    );
  }
}
