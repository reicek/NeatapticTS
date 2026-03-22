import Node from './node';
import Connection from './connection/connection';
import Group from './group/group';
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
} from './layer/layer.utils';
import { isGroup as isGroupUtils } from './layer/layer.guard.utils';
import type { LayerLike } from './layer/layer.utils.types';

const DEFAULT_LAYER_DROPOUT = 0;
const DEFAULT_LAYER_TRAINING_MODE = false;
const DEFAULT_TWO_SIDED_DISCONNECT = false;
const DEFAULT_CONV1D_STRIDE = 1;
const DEFAULT_CONV1D_PADDING = 0;
const DEFAULT_ATTENTION_HEADS = 1;
const NODE_INDEX_START = 0;
const NODE_INDEX_STEP = 1;

/**
 * Represents a functional layer within a neural network architecture.
 *
 * Layers act as organizational units for nodes, facilitating the creation of
 * complex network structures like Dense, LSTM, GRU, or Memory layers.
 * They manage the collective behavior of their nodes, including activation,
 * propagation, and connection to other network components.
 */
export default class Layer {
  /**
   * An array containing all the nodes (neurons or groups) that constitute this layer.
   * The order of nodes might be relevant depending on the layer type and its connections.
   */
  nodes: Node[]; // Note: While typed as Node[], can contain Group instances in practice for memory layers.

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
    this.connections = { in: [], out: [], self: [] }; // Initialize connection tracking
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
   * @param value - An optional array of activation values to set for the layer's nodes. The length must match the number of nodes.
   * @param training - A boolean indicating whether the layer is in training mode. Defaults to false.
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
   * @param rate - The learning rate, controlling the step size of weight adjustments.
   * @param momentum - The momentum factor, used to smooth weight updates and escape local minima.
   * @param target - An optional array of target values (expected outputs) for the layer's nodes. The length must match the number of nodes.
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
   * @param target - The destination Layer, Group, or Node to connect to.
   * @param method - The connection method (e.g., `ALL_TO_ALL`, `ONE_TO_ONE`) defining the connection pattern. See `methods.groupConnection`.
   * @param weight - An optional fixed weight to assign to all created connections.
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
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
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
   * @param connections - An array of connection objects to be gated.
   * @param method - The gating method (e.g., `INPUT`, `OUTPUT`, `SELF`) specifying how the gate influences the connection. See `methods.gating`.
   * @throws {Error} If the layer's `output` group is not defined.
   */
  gate(connections: Connection[], method: unknown): void {
    // Step 1: Delegate gating to the orchestrated utils flow.
    gateLayerUtils(
      {
        connections: this.connections,
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
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
   * @param values - An object containing the properties and their values to set.
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
        // Apply settings directly to Node instances
        if (values.bias !== undefined) {
          node.bias = values.bias;
        }
        if (values.squash !== undefined) {
          node.squash = values.squash;
        }
        if (values.type !== undefined) {
          node.type = values.type;
        }
      } else if (isGroupUtils(node)) {
        // If it's a Group (possible in memory layers), apply settings recursively
        (node as Group).set(values);
      }
    }
  }

  /**
   * Removes connections between this layer's nodes and a target Group or Node.
   *
   * @param target - The Group or Node to disconnect from.
   * @param twosided - If true, removes connections in both directions (from this layer to target, and from target to this layer). Defaults to false.
   */
  disconnect(target: Group | Node, twosided?: boolean) {
    // Step 1: Delegate disconnection to the orchestrated utils flow.
    disconnectLayerUtils(
      {
        connections: this.connections,
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
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
      isLayer: (candidate): candidate is Layer => candidate instanceof Layer,
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
   * @param from - The source Layer or Group connecting *to* this layer.
   * @param method - The connection method (e.g., `ALL_TO_ALL`). Defaults to `ALL_TO_ALL`.
   * @param weight - An optional fixed weight for the connections.
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
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
        layer: this,
        nodes: this.nodes,
        output: this.output,
      },
      from,
      method,
      weight,
    );
  }

  // Static Layer Factory Methods

  /**
   * Creates a standard fully connected (dense) layer.
   *
   * All nodes in the source layer/group will connect to all nodes in this layer
   * when using the default `ALL_TO_ALL` connection method via `layer.input()`.
   *
   * @param size - The number of nodes (neurons) in this layer.
   * @returns A new Layer instance configured as a dense layer.
   */
  static dense(size: number): Layer {
    // Step 1: Delegate dense layer creation to the utils orchestrator.
    return createDenseLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
      },
      size,
    );
  }

  /**
   * Creates a Long Short-Term Memory (LSTM) layer.
   *
   * LSTMs are a type of recurrent neural network (RNN) cell capable of learning
   * long-range dependencies. This implementation uses standard LSTM architecture
   * with input, forget, and output gates, and a memory cell.
   *
   * @param size - The number of LSTM units (and nodes in each gate/cell group).
   * @returns A new Layer instance configured as an LSTM layer.
   */
  static lstm(size: number): Layer {
    // Step 1: Delegate LSTM layer creation to the utils orchestrator.
    return createLstmLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
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
   * @param size - The number of GRU units (and nodes in each gate/cell group).
   * @returns A new Layer instance configured as a GRU layer.
   */
  static gru(size: number): Layer {
    // Step 1: Delegate GRU layer creation to the utils orchestrator.
    return createGruLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
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
   * @param size - The number of nodes in each memory block (must match the input size).
   * @param memory - The number of time steps to remember (number of memory blocks).
   * @returns A new Layer instance configured as a Memory layer.
   * @throws {Error} If the connecting layer's size doesn't match the memory block `size`.
   */
  static memory(size: number, memory: number): Layer {
    // Step 1: Delegate Memory layer creation to the utils orchestrator.
    return createMemoryLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
      },
      size,
      memory,
    );
  }

  /**
   * Creates a batch normalization layer.
   * Applies batch normalization to the activations of the nodes in this layer during activation.
   * @param size - The number of nodes in this layer.
   * @returns A new Layer instance configured as a batch normalization layer.
   */
  static batchNorm(size: number): Layer {
    // Step 1: Delegate batch norm creation to the utils orchestrator.
    return createBatchNormLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
      },
      size,
    );
  }

  /**
   * Creates a layer normalization layer.
   * Applies layer normalization to the activations of the nodes in this layer during activation.
   * @param size - The number of nodes in this layer.
   * @returns A new Layer instance configured as a layer normalization layer.
   */
  static layerNorm(size: number): Layer {
    // Step 1: Delegate layer norm creation to the utils orchestrator.
    return createLayerNormLayerUtils<Layer>(
      {
        createLayer: () => new Layer(),
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
      },
      size,
    );
  }

  /**
   * Creates a 1D convolutional layer (stub implementation).
   * @param size - Number of output nodes (filters).
   * @param kernelSize - Size of the convolution kernel.
   * @param stride - Stride of the convolution (default 1).
   * @param padding - Padding (default 0).
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
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
      },
      size,
      kernelSize,
      stride,
      padding,
    );
  }

  /**
   * Creates a multi-head self-attention layer (stub implementation).
   * @param size - Number of output nodes.
   * @param heads - Number of attention heads (default 1).
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
        isLayer: (candidate): candidate is LayerLike =>
          candidate instanceof Layer,
      },
      size,
      heads,
    );
  }
}
