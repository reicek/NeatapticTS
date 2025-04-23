import Connection from './connection';
import { config } from '../config';
import * as methods from '../methods/methods';

/**
 * Represents a node (neuron) in a neural network graph.
 *
 * Nodes are the fundamental processing units. They receive inputs, apply an activation function,
 * and produce an output. Nodes can be of type 'input', 'hidden', or 'output'. Hidden and output
 * nodes have biases and activation functions, which can be mutated during neuro-evolution.
 * This class also implements mechanisms for backpropagation, including support for momentum (NAG),
 * L2 regularization, dropout, and eligibility traces for recurrent connections.
 *
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-1-nodes Instinct Algorithm - Section 1.1 Nodes}
 */
export default class Node {
  /**
   * The bias value of the node. Added to the weighted sum of inputs before activation.
   * Input nodes typically have a bias of 0.
   */
  bias: number;
  /**
   * The activation function (squashing function) applied to the node's state.
   * Maps the internal state to the node's output (activation).
   * @param x The node's internal state (sum of weighted inputs + bias).
   * @param derivate If true, returns the derivative of the function instead of the function value.
   * @returns The activation value or its derivative.
   */
  squash: (x: number, derivate?: boolean) => number;
  /**
   * The type of the node: 'input', 'hidden', or 'output'.
   * Determines behavior (e.g., input nodes don't have biases modified typically, output nodes calculate error differently).
   */
  type: string;
  /**
   * The output value of the node after applying the activation function. This is the value transmitted to connected nodes.
   */
  activation: number;
  /**
   * The internal state of the node (sum of weighted inputs + bias) before the activation function is applied.
   */
  state: number;
  /**
   * The node's state from the previous activation cycle. Used for recurrent self-connections.
   */
  old: number;
  /**
   * A mask factor (typically 0 or 1) used for implementing dropout. If 0, the node's output is effectively silenced.
   */
  mask: number;
  /**
   * The change in bias applied in the previous training iteration. Used for calculating momentum.
   */
  previousDeltaBias: number;
  /**
   * Accumulates changes in bias over a mini-batch during batch training. Reset after each weight update.
   */
  totalDeltaBias: number;
  /**
   * Stores incoming, outgoing, gated, and self-connections for this node.
   */
  connections: {
    /** Incoming connections to this node. */
    in: Connection[];
    /** Outgoing connections from this node. */
    out: Connection[];
    /** Connections gated by this node's activation. */
    gated: Connection[];
    /** The recurrent self-connection. */
    self: Connection;
  };
  /**
   * Stores error values calculated during backpropagation.
   */
  error: {
    /** The node's responsibility for the network error, calculated based on projected and gated errors. */
    responsibility: number;
    /** Error projected back from nodes this node connects to. */
    projected: number;
    /** Error projected back from connections gated by this node. */
    gated: number;
  };
  /**
   * The derivative of the activation function evaluated at the node's current state. Used in backpropagation.
   */
  derivative?: number;
  /**
   * Temporary store for nodes influenced by gated connections during activation. Used for eligibility trace calculation.
   * @deprecated Consider refactoring eligibility trace logic if this becomes complex.
   */
  nodes: Node[];
  /**
   * Stores connections that this node gates. Seems redundant with `connections.gated`.
   * @deprecated Review usage; likely replaceable by `connections.gated`.
   */
  gates: Connection[];
  /**
   * Optional index, potentially used to identify the node's position within a layer or network structure. Not used internally by the Node class itself.
   */
  index?: number;

  /**
   * Creates a new node.
   * @param type The type of the node ('input', 'hidden', or 'output'). Defaults to 'hidden'.
   */
  constructor(type: string = 'hidden') {
    // Initialize bias: 0 for input nodes, small random value for others.
    this.bias = type === 'input' ? 0 : Math.random() * 0.2 - 0.1;
    // Set activation function. Default to logistic or identity if logistic is not available.
    this.squash = methods.Activation.logistic || ((x) => x);
    this.type = type;

    // Initialize state and activation values.
    this.activation = 0;
    this.state = 0;
    this.old = 0;

    // Initialize mask for dropout (default is no dropout).
    this.mask = 1;

    // Initialize momentum tracking variables.
    this.previousDeltaBias = 0;

    // Initialize batch training accumulator.
    this.totalDeltaBias = 0;

    // Initialize connection storage.
    this.connections = {
      in: [],
      out: [],
      gated: [],
      // Self-connection initialized with weight 0 (inactive).
      self: new Connection(this, this, 0),
    };

    // Initialize error tracking variables for backpropagation.
    this.error = {
      responsibility: 0,
      projected: 0,
      gated: 0,
    };

    // Temporary arrays used during activation/propagation.
    this.nodes = []; // Stores nodes influenced by this node's gating activity.
    this.gates = []; // Likely unused/redundant.
  }

  /**
   * Activates the node, calculating its output value based on inputs and state.
   * This method also calculates eligibility traces (`xtrace`) used for training recurrent connections.
   *
   * The activation process involves:
   * 1. Calculating the node's internal state (`this.state`) based on:
   *    - Incoming connections' weighted activations.
   *    - The recurrent self-connection's weighted state from the previous timestep (`this.old`).
   *    - The node's bias.
   * 2. Applying the activation function (`this.squash`) to the state to get the activation (`this.activation`).
   * 3. Applying the dropout mask (`this.mask`).
   * 4. Calculating the derivative of the activation function.
   * 5. Updating the gain of connections gated by this node.
   * 6. Calculating and updating eligibility traces for incoming connections.
   *
   * @param input Optional input value. If provided, sets the node's activation directly (used for input nodes).
   * @returns The calculated activation value of the node.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-3-activation Instinct Algorithm - Section 1.3 Activation}
   */
  activate(input?: number): number {
    // Direct activation for input nodes or if an input value is provided.
    if (typeof input !== 'undefined') {
      this.activation = input;
      return this.activation;
    }

    // Store the previous state for recurrent connections.
    this.old = this.state;

    // Calculate the new state: self-connection contribution + bias.
    // Note: Uses 'this.old' (state from the previous timestep) for the self-connection.
    this.state =
      this.connections.self.gain * this.connections.self.weight * this.old +
      this.bias;

    // Add contributions from incoming connections.
    for (const connection of this.connections.in) {
      this.state +=
        connection.from.activation * connection.weight * connection.gain;
    }

    // Apply activation function and dropout mask.
    this.activation = this.squash(this.state) * this.mask;
    // Calculate the derivative for backpropagation.
    this.derivative = this.squash(this.state, true);

    // Reset temporary storage for gated node influences.
    const nodes: Node[] = [];
    const influences: number[] = [];

    // Update gain for connections gated by this node.
    for (const conn of this.connections.gated) {
      const node = conn.to; // The node whose connection is being gated.
      const index = nodes.indexOf(node);

      // Calculate the influence this node exerts on the gated connection's state.
      // This is needed for calculating the gated error during backpropagation.
      if (index > -1) {
        // Accumulate influence if the target node is already listed (multiple connections gated).
        influences[index] += conn.weight * conn.from.activation;
      } else {
        // Store the target node and its initial influence.
        nodes.push(node);
        influences.push(
          conn.weight * conn.from.activation +
            // Include the state of the target node's self-connection if it's also gated by this node.
            (node.connections.self.gater === this ? node.old : 0)
        );
      }
      // Set the gain of the gated connection to this node's activation.
      conn.gain = this.activation;
    }

    // Update eligibility traces (xtraces) for incoming connections.
    // Eligibility traces are used to assign credit/blame across time in recurrent networks.
    for (const connection of this.connections.in) {
      // Update the eligibility trace for the connection itself.
      // Formula: trace = decay * trace + input_activation * connection_gain
      connection.elegibility =
        this.connections.self.gain * // Decay factor from self-connection
          this.connections.self.weight *
          connection.elegibility +
        connection.from.activation * connection.gain; // New influence

      // Update extended eligibility traces (xtraces) related to gated connections.
      // These track how the current node's activation influences nodes further down the line via gating.
      for (let j = 0; j < nodes.length; j++) {
        const node = nodes[j]; // The node influenced by gating.
        const influence = influences[j]; // The calculated influence of this node on the 'node'.
        const index = connection.xtrace.nodes.indexOf(node);

        // Update the xtrace value for the influenced node.
        // Formula similar to eligibility trace, incorporating the derivative and influence.
        if (index > -1) {
          // Update existing xtrace value.
          connection.xtrace.values[index] =
            node.connections.self.gain * // Decay from influenced node's self-connection
              node.connections.self.weight *
              connection.xtrace.values[index] +
            this.derivative! * connection.elegibility * influence; // New influence component
        } else {
          // Add a new xtrace entry if this node wasn't previously tracked.
          connection.xtrace.nodes.push(node);
          connection.xtrace.values.push(
            this.derivative! * connection.elegibility * influence
          );
        }
      }
    }

    return this.activation;
  }

  /**
   * Activates the node without calculating eligibility traces (`xtrace`).
   * This is a performance optimization used during inference (when the network
   * is just making predictions, not learning) as trace calculations are only needed for training.
   *
   * @param input Optional input value. If provided, sets the node's activation directly (used for input nodes).
   * @returns The calculated activation value of the node.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#1-3-activation Instinct Algorithm - Section 1.3 Activation}
   */
  noTraceActivate(input?: number): number {
    // Direct activation for input nodes or if an input value is provided.
    if (typeof input !== 'undefined') {
      this.activation = input;
      return this.activation;
    }

    // Calculate the new state: self-connection contribution + bias.
    // Note: Uses 'this.state' (current state) for self-connection, differing from `activate`.
    // This might be a subtle point depending on the exact recurrent update rule desired.
    // Consider if `this.old` should be used here too for consistency if traces are sometimes used.
    this.state =
      this.connections.self.gain * this.connections.self.weight * this.state +
      this.bias;

    // Add contributions from incoming connections.
    for (const connection of this.connections.in) {
      this.state +=
        connection.from.activation * connection.weight * connection.gain;
    }

    // Apply activation function and dropout mask.
    this.activation = this.squash(this.state) * this.mask;
    // Calculate the derivative (still needed if gating is used, even without traces).
    this.derivative = this.squash(this.state, true);

    // Update gain for connections gated by this node.
    for (const connection of this.connections.gated) {
      connection.gain = this.activation;
    }

    // No eligibility trace calculations are performed in this method.

    return this.activation;
  }

  /**
   * Back-propagates the error signal through the node and calculates weight/bias updates.
   *
   * This method implements the backpropagation algorithm, including:
   * 1. Calculating the node's error responsibility based on errors from subsequent nodes (`projected` error)
   *    and errors from connections it gates (`gated` error).
   * 2. Calculating the gradient for each incoming connection's weight using eligibility traces (`xtrace`).
   * 3. Calculating the change (delta) for weights and bias, incorporating:
   *    - Learning rate.
   *    - L2 regularization (weight decay).
   *    - Momentum (using Nesterov Accelerated Gradient - NAG).
   * 4. Optionally applying the calculated updates immediately or accumulating them for batch training.
   *
   * @param rate The learning rate (controls the step size of updates).
   * @param momentum The momentum factor (helps accelerate learning and overcome local minima). Uses NAG.
   * @param update If true, apply the calculated weight/bias updates immediately. If false, accumulate them in `totalDelta*` properties for batch updates.
   * @param regularization The L2 regularization factor (lambda). Penalizes large weights to prevent overfitting. Defaults to 0 (no regularization).
   * @param target The target output value for this node. Only used if the node is of type 'output'.
   */
  propagate(
    rate: number,
    momentum: number,
    update: boolean,
    regularization: number = 0, // L2 regularization factor (lambda)
    target?: number
  ): void {
    // Nesterov Accelerated Gradient (NAG): Apply momentum update *before* calculating the gradient.
    // This "lookahead" step estimates the future position and calculates the gradient there.
    if (update && momentum > 0) {
      // Apply previous momentum step to weights (lookahead).
      for (const connection of this.connections.in) {
        connection.weight += momentum * connection.previousDeltaWeight;
      }
      // Apply previous momentum step to bias (lookahead).
      this.bias += momentum * this.previousDeltaBias;
    }

    // Calculate the node's error signal (delta).
    let error = 0;

    // 1. Calculate error responsibility.
    if (this.type === 'output') {
      // For output nodes, the projected error is the difference between target and activation.
      // Responsibility is the same as projected error for output nodes (no gating error contribution needed here).
      this.error.responsibility = this.error.projected =
        target! - this.activation; // target should always be defined for output nodes during training.
    } else {
      // For hidden nodes:
      // Calculate projected error: sum of errors from outgoing connections, weighted by connection weights and gains.
      for (const connection of this.connections.out) {
        error +=
          connection.to.error.responsibility * // Error responsibility of the node this connection points to.
          connection.weight * // Weight of the connection.
          connection.gain; // Gain of the connection (usually 1, unless gated).
      }
      // Projected error = derivative * sum of weighted errors from the next layer.
      this.error.projected = this.derivative! * error;

      // Calculate gated error: sum of errors from connections gated by this node.
      error = 0; // Reset error accumulator.
      for (const connection of this.connections.gated) {
        const node = connection.to; // The node whose connection is gated.
        // Calculate the influence this node's activation had on the gated connection's state.
        let influence = node.connections.self.gater === this ? node.old : 0; // Influence via self-connection gating.
        influence += connection.weight * connection.from.activation; // Influence via regular connection gating.

        // Add the gated node's responsibility weighted by the influence.
        error += node.error.responsibility * influence;
      }
      // Gated error = derivative * sum of weighted responsibilities from gated connections.
      this.error.gated = this.derivative! * error;

      // Total error responsibility = projected error + gated error.
      this.error.responsibility = this.error.projected + this.error.gated;
    }

    // Nodes marked as 'constant' (if used) should not have their weights/biases updated.
    if (this.type === 'constant') return;

    // 2. Calculate gradients and update weights/biases for incoming connections.
    for (const connection of this.connections.in) {
      // Calculate the gradient for the connection weight.
      // Base gradient: error projected back * eligibility trace of the connection.
      let gradient = this.error.projected * connection.elegibility;

      // Add contributions from extended eligibility traces (xtraces) for gated connections.
      for (let j = 0; j < connection.xtrace.nodes.length; j++) {
        const node = connection.xtrace.nodes[j]; // Node influenced via gating.
        const value = connection.xtrace.values[j]; // Corresponding xtrace value.
        // Add the influenced node's responsibility weighted by the xtrace value.
        gradient += node.error.responsibility * value;
      }

      // Calculate the weight change (delta) for this connection.
      // Delta = learning_rate * (gradient * mask - regularization_term)
      // Regularization term: lambda * weight (L2 regularization)
      const deltaWeight =
        rate * (gradient * this.mask - regularization * connection.weight);

      // Accumulate delta for batch training.
      connection.totalDeltaWeight += deltaWeight;

      if (update) {
        // Apply the update immediately (if not batch training or end of batch).

        // Calculate the final weight update including momentum (NAG style).
        // The gradient was calculated at the lookahead position.
        const currentDeltaWeight =
          connection.totalDeltaWeight + // Current batch gradient contribution.
          momentum * connection.previousDeltaWeight; // Momentum contribution.

        // Apply the update (NAG):
        // 1. Revert the lookahead momentum step applied at the beginning.
        if (momentum > 0) {
          connection.weight -= momentum * connection.previousDeltaWeight;
        }
        // 2. Apply the full calculated delta (gradient + momentum).
        connection.weight += currentDeltaWeight;

        // Store the current delta for the next iteration's momentum calculation.
        connection.previousDeltaWeight = currentDeltaWeight;
        // Reset the batch accumulator.
        connection.totalDeltaWeight = 0;
      }
    }

    // Calculate bias change (delta). Regularization typically doesn't apply to bias.
    // Delta = learning_rate * error_responsibility
    const deltaBias = rate * this.error.responsibility;
    // Accumulate delta for batch training.
    this.totalDeltaBias += deltaBias;

    if (update) {
      // Apply the update immediately (if not batch training or end of batch).

      // Calculate the final bias update including momentum (NAG style).
      const currentDeltaBias =
        this.totalDeltaBias + // Current batch gradient contribution.
        momentum * this.previousDeltaBias; // Momentum contribution.

      // Apply the update (NAG):
      // 1. Revert the lookahead momentum step.
      if (momentum > 0) {
        this.bias -= momentum * this.previousDeltaBias;
      }
      // 2. Apply the full calculated delta.
      this.bias += currentDeltaBias;

      // Store the current delta for the next iteration's momentum calculation.
      this.previousDeltaBias = currentDeltaBias;
      // Reset the batch accumulator.
      this.totalDeltaBias = 0;
    }
  }

  /**
   * Converts the node's essential properties to a JSON object for serialization.
   * Does not include state, activation, error, or connection information, as these
   * are typically transient or reconstructed separately.
   * @returns A JSON representation of the node's configuration.
   */
  toJSON(): {
    bias: number;
    type: string;
    squash: string | null;
    mask: number;
  } {
    return {
      bias: this.bias,
      type: this.type,
      // Store the name of the squash function for reconstruction.
      squash: this.squash ? this.squash.name : null,
      mask: this.mask,
    };
  }

  /**
   * Creates a Node instance from a JSON object.
   * @param json The JSON object containing node configuration.
   * @returns A new Node instance configured according to the JSON object.
   * @throws {Error} If the squash function name in the JSON is invalid.
   */
  static fromJSON(json: {
    bias: number;
    type: string;
    squash: string;
    mask: number;
  }): Node {
    const node = new Node(json.type);
    node.bias = json.bias;
    node.mask = json.mask;
    // Look up the activation function by name in the methods.Activation enum/object.
    const squashFunction =
      methods.Activation[json.squash as keyof typeof methods.Activation];
    if (typeof squashFunction === 'function') {
      node.squash = squashFunction as (x: number, derivate?: boolean) => number;
    } else {
      // Handle cases where the function name might be invalid or not found.
      throw new Error(
        `Invalid or unknown squash function name: ${json.squash}`
      );
    }
    return node;
  }

  /**
   * Applies a mutation method to the node. Used in neuro-evolution.
   *
   * This allows modifying the node's properties, such as its activation function or bias,
   * based on predefined mutation methods.
   *
   * @param method A mutation method object, typically from `methods.mutation`. It should define the type of mutation and its parameters (e.g., allowed functions, modification range).
   * @throws {Error} If the mutation method is invalid, not provided, or not found in `methods.mutation`.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6#3-mutation Instinct Algorithm - Section 3 Mutation}
   */
  mutate(method: any): void {
    // Validate the provided mutation method.
    if (!method) {
      throw new Error('Mutation method cannot be null or undefined.');
    }
    // Ensure the method exists in the defined mutation methods.
    // Note: This check assumes `method` itself is the function, comparing its name.
    // If `method` is an object describing the mutation, the check might need adjustment.
    if (!(method.name in methods.mutation)) {
      throw new Error(`Unknown mutation method: ${method.name}`);
    }

    // Apply the specified mutation.
    switch (method) {
      case methods.mutation.MOD_ACTIVATION:
        // Mutate the activation function.
        if (!method.allowed || method.allowed.length === 0) {
          console.warn(
            'MOD_ACTIVATION mutation called without allowed functions specified.'
          );
          return;
        }
        const allowed = method.allowed;
        // Find the index of the current squash function.
        const currentIndex = allowed.indexOf(this.squash);
        // Select a new function randomly from the allowed list, ensuring it's different.
        let newIndex = currentIndex;
        if (allowed.length > 1) {
          newIndex =
            (currentIndex +
              Math.floor(Math.random() * (allowed.length - 1)) +
              1) %
            allowed.length;
        }
        this.squash = allowed[newIndex];
        break;
      case methods.mutation.MOD_BIAS:
        // Mutate the bias value.
        const min = method.min ?? -1; // Default min modification
        const max = method.max ?? 1; // Default max modification
        // Add a random modification within the specified range [min, max).
        const modification = Math.random() * (max - min) + min;
        this.bias += modification;
        break;
      // Add cases for other mutation types if needed.
      default:
        // This case might be redundant if the initial check catches unknown methods.
        throw new Error(`Unsupported mutation method: ${method.name}`);
    }
  }

  /**
   * Creates a connection from this node to a target node or all nodes in a group.
   *
   * @param target The target Node or a group object containing a `nodes` array.
   * @param weight The weight for the new connection(s). If undefined, a default or random weight might be assigned by the Connection constructor (currently defaults to 0, consider changing).
   * @returns An array containing the newly created Connection object(s).
   * @throws {Error} If the target is undefined.
   * @throws {Error} If trying to create a self-connection when one already exists (weight is not 0).
   */
  connect(target: Node | { nodes: Node[] }, weight?: number): Connection[] {
    const connections: Connection[] = [];
    if (!target) {
      throw new Error('Cannot connect to an undefined target.');
    }

    // Check if the target is a single Node.
    if ('bias' in target) {
      // Simple check if target looks like a Node instance.
      const targetNode = target as Node;
      if (targetNode === this) {
        // Handle self-connection.
        if (this.connections.self.weight !== 0) {
          // Avoid creating duplicate self-connections if one already exists (weight != 0).
          console.warn('Self-connection already exists.');
          // Optionally return the existing connection: return [this.connections.self];
          return []; // Or return empty array if no action is taken.
        }
        // Activate the self-connection by setting its weight.
        this.connections.self.weight = weight ?? 1; // Default self-connection weight to 1 if not provided.
        connections.push(this.connections.self);
      } else {
        // Handle connection to a different node.
        // Check if connection already exists (optional, can be expensive).
        // if (this.isProjectingTo(targetNode)) {
        //   console.warn(`Connection from Node ${this.index ?? '?'} to Node ${targetNode.index ?? '?'} already exists.`);
        //   return []; // Or find and return existing connection.
        // }
        const connection = new Connection(this, targetNode, weight);
        // Add connection to the target's incoming list and this node's outgoing list.
        targetNode.connections.in.push(connection);
        this.connections.out.push(connection);
        connections.push(connection);
      }
    } else if ('nodes' in target && Array.isArray(target.nodes)) {
      // Handle connection to a group of nodes.
      for (const node of target.nodes) {
        // Create connection for each node in the group.
        const connection = new Connection(this, node, weight);
        node.connections.in.push(connection);
        this.connections.out.push(connection);
        connections.push(connection);
      }
    } else {
      // Handle invalid target type.
      throw new Error(
        'Invalid target type for connection. Must be a Node or a group { nodes: Node[] }.'
      );
    }
    return connections;
  }

  /**
   * Removes the connection from this node to the target node.
   *
   * @param target The target node to disconnect from.
   * @param twosided If true, also removes the connection from the target node back to this node (if it exists). Defaults to false.
   */
  disconnect(target: Node, twosided: boolean = false): void {
    // Handle self-connection disconnection.
    if (this === target) {
      this.connections.self.weight = 0; // Deactivate self-connection by setting weight to 0.
      // Consider also resetting gain/gater if applicable.
      // this.connections.self.gain = 1;
      // if (this.connections.self.gater) this.connections.self.gater.ungate(this.connections.self);
      return;
    }

    // Filter out the connection to the target node from the outgoing list.
    this.connections.out = this.connections.out.filter((conn) => {
      if (conn.to === target) {
        // Remove the connection from the target node's incoming list.
        target.connections.in = target.connections.in.filter(
          (inConn) => inConn !== conn // Filter by reference.
        );
        // If the connection was gated, ungate it properly.
        if (conn.gater) {
          conn.gater.ungate(conn);
        }
        return false; // Remove from this.connections.out.
      }
      return true; // Keep other connections.
    });

    // If twosided is true, recursively call disconnect on the target node.
    if (twosided) {
      target.disconnect(this, false); // Pass false to avoid infinite recursion.
    }
  }

  /**
   * Makes this node gate the provided connection(s).
   * The connection's gain will be controlled by this node's activation value.
   *
   * @param connections A single Connection object or an array of Connection objects to be gated.
   */
  gate(connections: Connection | Connection[]): void {
    // Ensure connections is an array.
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    for (const connection of connections) {
      if (!connection || !connection.from || !connection.to) {
        console.warn('Attempted to gate an invalid or incomplete connection.');
        continue;
      }
      // Check if this node is already gating this connection.
      if (connection.gater === this) {
        console.warn('Node is already gating this connection.');
        continue;
      }
      // Check if the connection is already gated by another node.
      if (connection.gater !== null) {
        console.warn(
          'Connection is already gated by another node. Ungate first.'
        );
        // Optionally, automatically ungate from the previous gater:
        // connection.gater.ungate(connection);
        continue; // Skip gating if already gated by another.
      }

      // Add the connection to this node's list of gated connections.
      this.connections.gated.push(connection);
      // Set the gater property on the connection itself.
      connection.gater = this;
      // Gain will be updated during activation. Initialize?
      // connection.gain = this.activation; // Or 0? Or leave as is? Depends on desired initial state.
    }
  }

  /**
   * Removes this node's gating control over the specified connection(s).
   * Resets the connection's gain to 1 and removes it from the `connections.gated` list.
   *
   * @param connections A single Connection object or an array of Connection objects to ungate.
   */
  ungate(connections: Connection | Connection[]): void {
    // Ensure connections is an array.
    if (!Array.isArray(connections)) {
      connections = [connections];
    }

    for (const connection of connections) {
      if (!connection) continue; // Skip null/undefined entries

      // Find the connection in the gated list.
      const index = this.connections.gated.indexOf(connection);
      if (index !== -1) {
        // Remove from the gated list.
        this.connections.gated.splice(index, 1);
        // Reset the connection's gater property.
        connection.gater = null;
        // Reset the connection's gain to its default value (usually 1).
        connection.gain = 1;
      } else {
        // Optional: Warn if trying to ungate a connection not gated by this node.
        // console.warn("Attempted to ungate a connection not gated by this node, or already ungated.");
      }
    }
  }

  /**
   * Clears the node's dynamic state information.
   * Resets activation, state, previous state, error signals, and eligibility traces.
   * Useful for starting a new activation sequence (e.g., for a new input pattern).
   */
  clear(): void {
    // Reset eligibility traces for all incoming connections.
    for (const connection of this.connections.in) {
      connection.elegibility = 0;
      connection.xtrace = { nodes: [], values: [] };
    }

    // Reset gain for connections gated by this node.
    // Should gain be reset to 1 or 0? Resetting to 0 seems more consistent
    // with a 'cleared' state before activation. However, gain=1 is the 'ungated' state.
    // Let's reset to 0, assuming activation will set it correctly.
    for (const connection of this.connections.gated) {
      connection.gain = 0;
    }

    // Reset error values.
    this.error = { responsibility: 0, projected: 0, gated: 0 };
    // Reset state, activation, and old state.
    this.old = this.state = this.activation = 0;
    // Note: Does not reset bias, mask, or previousDeltaBias/totalDeltaBias as these
    // usually persist across activations or are handled by the training process.
  }

  /**
   * Checks if this node has a direct outgoing connection to the given node.
   * Considers both regular outgoing connections and the self-connection.
   *
   * @param node The potential target node.
   * @returns True if this node projects to the target node, false otherwise.
   */
  isProjectingTo(node: Node): boolean {
    // Check self-connection (only if weight is non-zero, indicating it's active).
    if (node === this && this.connections.self.weight !== 0) return true;

    // Check regular outgoing connections.
    return this.connections.out.some((conn) => conn.to === node);
  }

  /**
   * Checks if the given node has a direct outgoing connection to this node.
   * Considers both regular incoming connections and the self-connection.
   *
   * @param node The potential source node.
   * @returns True if the given node projects to this node, false otherwise.
   */
  isProjectedBy(node: Node): boolean {
    // Check self-connection (only if weight is non-zero).
    if (node === this && this.connections.self.weight !== 0) return true;

    // Check regular incoming connections.
    return this.connections.in.some((conn) => conn.from === node);
  }
}
