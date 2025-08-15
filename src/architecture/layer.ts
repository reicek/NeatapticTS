import Node from './node';
import Group from './group';
import * as methods from '../methods/methods';
import { activationArrayPool } from './activationArrayPool';

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
  connections: { in: any[]; out: any[]; self: any[] };

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
  dropout: number = 0;

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
  activate(value?: number[], training: boolean = false): number[] {
    const out = activationArrayPool.acquire(this.nodes.length);

    // Input validation
    if (value !== undefined && value.length !== this.nodes.length) {
      throw new Error(
        'Array with values should be same as the amount of nodes!'
      );
    }

    // --- Layer-level dropout logic ---
    let layerMask = 1;
    if (training && this.dropout > 0) {
      // Fix: Use comparison with dropout rate directly to ensure both 0 and 1 masks occur
      layerMask = Math.random() >= this.dropout ? 1 : 0;
      this.nodes.forEach((node) => {
        node.mask = layerMask;
      });
    } else {
      // In inference or no dropout, ensure all masks are 1
      this.nodes.forEach((node) => {
        node.mask = 1;
      });
    }

    // Activate each node
    for (let i = 0; i < this.nodes.length; i++) {
      let activation: number;
      if (value === undefined) {
        activation = this.nodes[i].activate();
      } else {
        activation = this.nodes[i].activate(value[i]);
      }
      (out as any)[i] = activation;
    }
    const cloned = Array.from(out as any) as number[];
    activationArrayPool.release(out);
    return cloned; // Return the activation values of all nodes
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
    // Input validation
    if (target !== undefined && target.length !== this.nodes.length) {
      throw new Error(
        'Array with values should be same as the amount of nodes!'
      );
    }

    // Propagate error backward through nodes (iterate in reverse order)
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      if (target === undefined) {
        this.nodes[i].propagate(rate, momentum, true, 0);
      } else {
        this.nodes[i].propagate(rate, momentum, true, 0, target[i]);
      }
    }
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
  connect(target: Group | Node | Layer, method?: any, weight?: number): any[] {
    // Ensure the output group is defined before connecting
    if (!this.output) {
      throw new Error(
        'Layer output is not defined. Cannot connect from this layer.'
      );
    }

    let connections: any[] = [];
    if (target instanceof Layer) {
      // Delegate connection ONLY to the target layer's input method
      connections = target.input(this, method, weight);
    } else if (target instanceof Group || target instanceof Node) {
      // Connect the layer's output group to the target Group or Node
      connections = this.output.connect(target, method, weight);
    }

    return connections;
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
  gate(connections: any[], method: any) {
    // Ensure the output group is defined before gating
    if (!this.output) {
      throw new Error(
        'Layer output is not defined. Cannot gate from this layer.'
      );
    }
    // Delegate gating to the output group
    this.output.gate(connections, method);
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
  set(values: { bias?: number; squash?: any; type?: string }) {
    for (let i = 0; i < this.nodes.length; i++) {
      let node = this.nodes[i];

      if (node instanceof Node) {
        // Apply settings directly to Node instances
        if (values.bias !== undefined) {
          node.bias = values.bias;
        }
        // Use provided squash function or keep the existing one
        node.squash = values.squash || node.squash;
        // Use provided type or keep the existing one
        node.type = values.type || node.type;
      } else if (this.isGroup(node)) {
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
    twosided = twosided || false; // Default to false if not provided

    let i, j, k;
    // Determine if the target is a Group or a single Node
    if (target instanceof Group) {
      // Iterate through all nodes in this layer and the target group
      for (i = 0; i < this.nodes.length; i++) {
        for (j = 0; j < target.nodes.length; j++) {
          // Disconnect individual nodes
          this.nodes[i].disconnect(target.nodes[j], twosided);

          // Clean up connection tracking within the layer object (outgoing)
          for (k = this.connections.out.length - 1; k >= 0; k--) {
            let conn = this.connections.out[k];
            if (conn.from === this.nodes[i] && conn.to === target.nodes[j]) {
              this.connections.out.splice(k, 1);
              break; // Assume only one connection between two nodes here
            }
          }

          // Clean up connection tracking (incoming) if twosided
          if (twosided) {
            for (k = this.connections.in.length - 1; k >= 0; k--) {
              let conn = this.connections.in[k];
              if (conn.from === target.nodes[j] && conn.to === this.nodes[i]) {
                this.connections.in.splice(k, 1);
                break; // Assume only one connection
              }
            }
          }
        }
      }
    } else if (target instanceof Node) {
      // Iterate through all nodes in this layer
      for (i = 0; i < this.nodes.length; i++) {
        // Disconnect from the target node
        this.nodes[i].disconnect(target, twosided);

        // Clean up connection tracking (outgoing)
        for (j = this.connections.out.length - 1; j >= 0; j--) {
          let conn = this.connections.out[j];
          if (conn.from === this.nodes[i] && conn.to === target) {
            this.connections.out.splice(j, 1);
            break; // Assume only one connection
          }
        }

        // Clean up connection tracking (incoming) if twosided
        if (twosided) {
          for (k = this.connections.in.length - 1; k >= 0; k--) {
            let conn = this.connections.in[k];
            if (conn.from === target && conn.to === this.nodes[i]) {
              this.connections.in.splice(k, 1);
              break; // Assume only one connection
            }
          }
        }
      }
    }
  }

  /**
   * Resets the activation state of all nodes within the layer.
   * This is typically done before processing a new input sequence or sample.
   */
  clear() {
    for (let i = 0; i < this.nodes.length; i++) {
      this.nodes[i].clear(); // Delegate clearing to individual nodes/groups
    }
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
  input(from: Layer | Group, method?: any, weight?: number): any[] {
    // If connecting from another Layer, use its output group as the source
    if (from instanceof Layer) from = from.output!;
    // Default connection method if not specified
    method = method || methods.groupConnection.ALL_TO_ALL;
    // Ensure this layer's target group (output) is defined
    if (!this.output) {
      throw new Error('Layer output (acting as input target) is not defined.');
    }
    // Connect the source group 'from' to this layer's 'output' group
    return from.connect(this.output, method, weight);
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
    // Initialize a new Layer
    const layer = new Layer();

    // Create a single group containing all nodes for this layer
    const block = new Group(size);

    // Add the nodes from the group to the layer's node list
    layer.nodes.push(...block.nodes);
    // Set the group as the primary output (and input target) for this layer
    layer.output = block;

    // Override the default input method to connect directly to the 'block' group
    layer.input = (
      from: Layer | Group,
      method?: any,
      weight?: number
    ): any[] => {
      if (from instanceof Layer) from = from.output!; // Use output group of source layer
      method = method || methods.groupConnection.ALL_TO_ALL; // Default connection
      // Connect the source 'from' to this layer's 'block'
      return from.connect(block, method, weight);
    };

    return layer;
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
    // Initialize a new Layer
    const layer = new Layer();

    // Create the core components (groups of nodes) of the LSTM cell
    const inputGate = new Group(size); // Controls flow of new information into the cell
    const forgetGate = new Group(size); // Controls what information to throw away from the cell state
    const memoryCell = new Group(size); // Stores the internal cell state over time
    const outputGate = new Group(size); // Controls what parts of the cell state to output
    const outputBlock = new Group(size); // Final output of the LSTM unit for this time step

    // Set initial biases for gates (common practice to initialize near 1 or 0)
    inputGate.set({ bias: 1 });
    forgetGate.set({ bias: 1 });
    outputGate.set({ bias: 1 });
    // Set initial bias for memory cell and output block to 0 (modern practice)
    memoryCell.set({ bias: 0 });
    outputBlock.set({ bias: 0 });

    // Internal connections within the LSTM unit
    // Connections to gates influence their activation
    memoryCell.connect(inputGate, methods.groupConnection.ALL_TO_ALL);
    memoryCell.connect(forgetGate, methods.groupConnection.ALL_TO_ALL);
    memoryCell.connect(outputGate, methods.groupConnection.ALL_TO_ALL);
    // Recurrent connection from memory cell back to itself (gated by forget gate)
    memoryCell.connect(memoryCell, methods.groupConnection.ONE_TO_ONE);
    // Connection from memory cell to the final output block (gated by output gate)
    const output = memoryCell.connect(
      outputBlock,
      methods.groupConnection.ALL_TO_ALL
    );

    // Apply gating mechanisms
    // Output gate controls the connection from the memory cell to the output block
    outputGate.gate(output, methods.gating.OUTPUT);

    // Apply forget gate to self-connections directly
    memoryCell.nodes.forEach((node, i) => {
      // Find the self-connection on the node
      const selfConnection = node.connections.self.find(
        (conn) => conn.to === node && conn.from === node
      );
      if (selfConnection) {
        // Assign the corresponding forget gate node as the gater
        selfConnection.gater = forgetGate.nodes[i];
        // Ensure the gater node knows about the connection it gates
        if (!forgetGate.nodes[i].connections.gated.includes(selfConnection)) {
          forgetGate.nodes[i].connections.gated.push(selfConnection);
        }
      } else {
        // This case should ideally not happen if connect worked correctly
        console.warn(
          `LSTM Warning: No self-connection found for memory cell node ${i}`
        );
      }
    });

    // Aggregate all nodes from the internal groups into the layer's node list
    layer.nodes = [
      ...inputGate.nodes,
      ...forgetGate.nodes,
      ...memoryCell.nodes,
      ...outputGate.nodes,
      ...outputBlock.nodes,
    ];

    // Set the final output block as the layer's primary output
    layer.output = outputBlock;

    // Define how external inputs connect to this LSTM layer
    layer.input = (
      from: Layer | Group,
      method?: any,
      weight?: number
    ): any[] => {
      if (from instanceof Layer) from = from.output!; // Use output group of source layer
      method = method || methods.groupConnection.ALL_TO_ALL; // Default connection
      let connections: any[] = [];

      // Connect external input to the memory cell (candidate values) and all three gates
      const input = from.connect(memoryCell, method, weight); // Input to cell calculation
      connections = connections.concat(input);
      connections = connections.concat(from.connect(inputGate, method, weight)); // Input to Input Gate
      connections = connections.concat(
        from.connect(outputGate, method, weight)
      ); // Input to Output Gate
      connections = connections.concat(
        from.connect(forgetGate, method, weight)
      ); // Input to Forget Gate

      // Input gate controls the influence of the external input on the memory cell state update
      inputGate.gate(input, methods.gating.INPUT);

      return connections; // Return all created connections
    };

    return layer;
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
    // Initialize a new Layer
    const layer = new Layer();

    // Create the core components (groups of nodes) of the GRU cell
    const updateGate = new Group(size); // Determines how much of the previous state to keep
    const inverseUpdateGate = new Group(size); // Computes (1 - updateGate output)
    const resetGate = new Group(size); // Determines how much of the previous state to forget
    const memoryCell = new Group(size); // Calculates candidate activation
    const output = new Group(size); // Final output of the GRU unit for this time step
    const previousOutput = new Group(size); // Stores the output from the previous time step

    // Configure node properties for specific components
    previousOutput.set({
      bias: 0,
      squash: methods.Activation.identity, // Pass through previous output directly
      type: 'variant', // Custom type identifier
    });
    memoryCell.set({
      squash: methods.Activation.tanh, // Tanh activation for candidate state
    });
    inverseUpdateGate.set({
      bias: 0,
      squash: methods.Activation.inverse, // Activation computes 1 - input
      type: 'variant', // Custom type identifier
    });
    updateGate.set({ bias: 1 }); // Initialize update gate bias (common practice)
    resetGate.set({ bias: 0 }); // Initialize reset gate bias

    // Internal connections within the GRU unit
    // Previous output influences gates
    previousOutput.connect(updateGate, methods.groupConnection.ALL_TO_ALL);
    previousOutput.connect(resetGate, methods.groupConnection.ALL_TO_ALL);

    // Update gate feeds into inverse update gate
    updateGate.connect(
      inverseUpdateGate,
      methods.groupConnection.ONE_TO_ONE,
      1
    ); // Weight of 1 for direct inversion

    // Previous output, gated by reset gate, influences memory cell candidate calculation
    const reset = previousOutput.connect(
      memoryCell,
      methods.groupConnection.ALL_TO_ALL
    );
    resetGate.gate(reset, methods.gating.OUTPUT); // Reset gate controls this connection

    // Calculate final output: combination of previous output and candidate activation, controlled by update gate
    const update1 = previousOutput.connect(
      output,
      methods.groupConnection.ALL_TO_ALL
    ); // Connection from previous output
    const update2 = memoryCell.connect(
      output,
      methods.groupConnection.ALL_TO_ALL
    ); // Connection from candidate activation

    // Apply gating by update gate and its inverse
    updateGate.gate(update1, methods.gating.OUTPUT); // Update gate controls influence of previous output
    inverseUpdateGate.gate(update2, methods.gating.OUTPUT); // Inverse update gate controls influence of candidate activation

    // Store the current output for the next time step
    output.connect(previousOutput, methods.groupConnection.ONE_TO_ONE, 1); // Direct copy with weight 1

    // Aggregate all nodes into the layer's node list
    layer.nodes = [
      ...updateGate.nodes,
      ...inverseUpdateGate.nodes,
      ...resetGate.nodes,
      ...memoryCell.nodes,
      ...output.nodes,
      ...previousOutput.nodes,
    ];

    // Set the 'output' group as the layer's primary output
    layer.output = output;

    // Define how external inputs connect to this GRU layer
    layer.input = (
      from: Layer | Group,
      method?: any,
      weight?: number
    ): any[] => {
      if (from instanceof Layer) from = from.output!; // Use output group of source layer
      method = method || methods.groupConnection.ALL_TO_ALL; // Default connection
      let connections: any[] = [];

      // Connect external input to update gate, reset gate, and memory cell candidate calculation
      connections = connections.concat(
        from.connect(updateGate, method, weight)
      );
      connections = connections.concat(from.connect(resetGate, method, weight));
      connections = connections.concat(
        from.connect(memoryCell, method, weight)
      );

      return connections; // Return all created connections
    };

    return layer;
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
    // Initialize a new Layer
    const layer = new Layer();

    let previous: Group | null = null; // Keep track of the previously created block
    // Create 'memory' number of blocks
    for (let i = 0; i < memory; i++) {
      const block = new Group(size); // Each block has 'size' nodes

      // Configure memory block nodes: linear activation, no bias
      block.set({
        squash: methods.Activation.identity,
        bias: 0,
        type: 'variant', // Custom type identifier
      });

      // Connect the previous block to the current block (propagates state backward)
      if (previous != null) {
        // ONE_TO_ONE connection with weight 1 copies state directly
        previous.connect(block, methods.groupConnection.ONE_TO_ONE, 1);
      }

      // Add the *Group* itself to the layer's nodes list (unlike other layer types)
      // This requires the `set` method to handle Groups internally.
      layer.nodes.push((block as unknown) as Node); // Cast needed due to `nodes: Node[]` type hint
      previous = block; // Update previous block reference
    }

    // Reverse the order of blocks so index 0 is the oldest memory
    layer.nodes.reverse();

    // Optional: Reverse nodes within each block if needed (depends on desired output order)
    // for (let i = 0; i < layer.nodes.length; i++) {
    //   layer.nodes[i].nodes.reverse(); // Assuming nodes property exists and is mutable
    // }

    // Create a single output group that concatenates nodes from all memory blocks
    const outputGroup = new Group(0); // Start with an empty group
    for (const group of layer.nodes) {
      // Iterate through the blocks (which are Groups)
      // Check if the item is actually a group before accessing nodes
      if (this.prototype.isGroup(group)) {
        outputGroup.nodes = outputGroup.nodes.concat(group.nodes);
      } else {
        // Handle cases where a Node might be directly in layer.nodes, though unlikely for memory layer
        console.warn(
          'Unexpected Node type found directly in Memory layer nodes list during output group creation.'
        );
      }
    }
    // Set the concatenated group as the layer's output
    layer.output = outputGroup;

    // Define how external inputs connect to this Memory layer
    layer.input = (
      from: Layer | Group,
      method?: any,
      weight?: number
    ): any[] => {
      if (from instanceof Layer) from = from.output!; // Use output group of source layer
      // Method is typically ignored here as we force ONE_TO_ONE to the last block
      method = method || methods.groupConnection.ALL_TO_ALL; // Keep for signature consistency

      // Get the most recent memory block (last element after reversal)
      const inputBlock = layer.nodes[layer.nodes.length - 1];
      // Ensure the input block is a Group before accessing its nodes
      if (!this.prototype.isGroup(inputBlock)) {
        throw new Error('Memory layer input block is not a Group.');
      }

      // Validate that the input size matches the memory block size
      if (from.nodes.length !== inputBlock.nodes.length) {
        throw new Error(
          `Previous layer size (${from.nodes.length}) must be same as memory size (${inputBlock.nodes.length})`
        );
      }

      // Connect the external input directly to the most recent memory block
      // ONE_TO_ONE with weight 1 copies the input into the block's state
      return from.connect(inputBlock, methods.groupConnection.ONE_TO_ONE, 1);
    };

    return layer;
  }

  /**
   * Creates a batch normalization layer.
   * Applies batch normalization to the activations of the nodes in this layer during activation.
   * @param size - The number of nodes in this layer.
   * @returns A new Layer instance configured as a batch normalization layer.
   */
  static batchNorm(size: number): Layer {
    const layer = Layer.dense(size);
    (layer as any).batchNorm = true;
    // Override activate to apply batch normalization
    const baseActivate = layer.activate.bind(layer);
    layer.activate = function (
      value?: number[],
      training: boolean = false
    ): number[] {
      const activations = baseActivate(value, training);
      // Compute mean and variance
      const mean = activations.reduce((a, b) => a + b, 0) / activations.length;
      const variance =
        activations.reduce((a, b) => a + (b - mean) ** 2, 0) /
        activations.length;
      const epsilon = 1e-5;
      // Normalize
      return activations.map((a) => (a - mean) / Math.sqrt(variance + epsilon));
    };
    return layer;
  }

  /**
   * Creates a layer normalization layer.
   * Applies layer normalization to the activations of the nodes in this layer during activation.
   * @param size - The number of nodes in this layer.
   * @returns A new Layer instance configured as a layer normalization layer.
   */
  static layerNorm(size: number): Layer {
    const layer = Layer.dense(size);
    (layer as any).layerNorm = true;
    // Override activate to apply layer normalization
    const baseActivate = layer.activate.bind(layer);
    layer.activate = function (
      value?: number[],
      training: boolean = false
    ): number[] {
      const activations = baseActivate(value, training);
      // Compute mean and variance (per sample, but here per layer)
      const mean = activations.reduce((a, b) => a + b, 0) / activations.length;
      const variance =
        activations.reduce((a, b) => a + (b - mean) ** 2, 0) /
        activations.length;
      const epsilon = 1e-5;
      // Normalize
      return activations.map((a) => (a - mean) / Math.sqrt(variance + epsilon));
    };
    return layer;
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
    stride: number = 1,
    padding: number = 0
  ): Layer {
    const layer = new Layer();
    layer.nodes = Array.from({ length: size }, () => new Node());
    layer.output = new Group(size);
    // Store conv params for future use
    (layer as any).conv1d = { kernelSize, stride, padding };
    // Placeholder: actual convolution logic would be in a custom activate method
    layer.activate = function (value?: number[]): number[] {
      // For now, just pass through or slice input as a stub
      if (!value) return this.nodes.map((n) => n.activate());
      // Simple stub: take the first 'size' values
      return value.slice(0, size);
    };
    return layer;
  }

  /**
   * Creates a multi-head self-attention layer (stub implementation).
   * @param size - Number of output nodes.
   * @param heads - Number of attention heads (default 1).
   * @returns A new Layer instance representing an attention layer.
   */
  static attention(size: number, heads: number = 1): Layer {
    const layer = new Layer();
    layer.nodes = Array.from({ length: size }, () => new Node());
    layer.output = new Group(size);
    (layer as any).attention = { heads };
    // Placeholder: actual attention logic would be in a custom activate method
    layer.activate = function (value?: number[]): number[] {
      // For now, just average the input as a stub
      if (!value) return this.nodes.map((n) => n.activate());
      const avg = value.reduce((a, b) => a + b, 0) / value.length;
      return Array(size).fill(avg);
    };
    return layer;
  }

  /**
   * Type guard to check if an object is likely a `Group`.
   *
   * This is a duck-typing check based on the presence of expected properties
   * (`set` method and `nodes` array). Used internally where `layer.nodes`
   * might contain `Group` instances (e.g., in `Memory` layers).
   *
   * @param obj - The object to inspect.
   * @returns `true` if the object has `set` and `nodes` properties matching a Group, `false` otherwise.
   */
  private isGroup(obj: any): obj is Group {
    // Check for existence and type of key properties
    return !!obj && typeof obj.set === 'function' && Array.isArray(obj.nodes);
  }
}
