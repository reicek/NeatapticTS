import Node from './node';
import Connection from './connection';
import Multi from '../multithreading/multi';
import * as methods from '../methods/methods';
import mutation from '../methods/mutation'; // Fix the import for mutation
import { config } from '../config';

// Helper function to strip coverage code - Remove Semicolon Requirement & Add Cleanup
const stripCoverage = (code: string): string => {
  // 1. Remove Istanbul ignore comments
  code = code.replace(/\/\*\s*istanbul\s+ignore\s+[\s\S]*?\*\//g, '');

  // 2. Remove coverage counter increments (No trailing semicolon required)
  code = code.replace(/cov_[\w$]+\(\)\.(s|f|b)\[\d+\](\[\d+\])?\+\+/g, ''); // Removed trailing ';'

  // 3. Remove simple coverage function calls (No trailing semicolon required)
  code = code.replace(/cov_[\w$]+\(\)/g, ''); // Removed trailing ';'

  // 4. Remove sourceMappingURL comments
  code = code.replace(/^\s*\/\/# sourceMappingURL=.*\s*$/gm, '');

  // 5. Cleanup potential leftover artifacts from comma operator usage
  // Example: ( , false) -> ( false) or ( true , ) -> ( true )
  code = code.replace(/\(\s*,\s*/g, '( '); // Remove leading comma inside parentheses
  code = code.replace(/\s*,\s*\)/g, ' )'); // Remove trailing comma inside parentheses

  // 6. Trim whitespace from start/end
  code = code.trim();

  // 7. Remove empty statements potentially left behind
  code = code.replace(/^\s*;\s*$/gm, ''); // Remove lines containing only a semicolon
  code = code.replace(/;{2,}/g, ';'); // Replace multiple semicolons with one
  // Remove lines that might now only contain commas or whitespace
  code = code.replace(/^\s*[,;]?\s*$/gm, '');

  return code;
};

/**
 * Represents a neural network with nodes and connections.
 *
 * The network supports feedforward and recurrent architectures. It includes advanced
 * features such as dropout, backpropagation, and mutation for neuro-evolution.
 *
 * @see Instinct Algorithm - Section 1.3 Activation
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
 */
export default class Network {
  input: number; // Number of input nodes
  output: number; // Number of output nodes
  score?: number; // Optional score property for genetic algorithms
  nodes: Node[]; // List of all nodes in the network
  connections: Connection[]; // List of all connections between nodes
  gates: Connection[]; // List of gated connections
  selfconns: Connection[]; // List of self-connections
  dropout: number = 0; // Dropout rate for training

  /**
   * Creates a new network with the specified number of input and output nodes.
   * @param {number} input - Number of input nodes.
   * @param {number} output - Number of output nodes.
   * @throws {Error} If input or output size is not provided.
   */
  constructor(input: number, output: number) {
    // Validate input and output sizes
    if (typeof input === 'undefined' || typeof output === 'undefined') {
      throw new Error('No input or output size given');
    }

    // Initialize network properties
    this.input = input;
    this.output = output;
    this.nodes = [];
    this.connections = [];
    this.gates = [];
    this.selfconns = [];
    this.dropout = 0;

    // Create input and output nodes
    for (let i = 0; i < this.input + this.output; i++) {
      const type = i < this.input ? 'input' : 'output'; // Determine node type
      this.nodes.push(new Node(type)); // Add node to the network
    }

    // Connect input nodes to output nodes
    for (let i = 0; i < this.input; i++) {
      for (let j = this.input; j < this.input + this.output; j++) {
        const weight = Math.random() * this.input * Math.sqrt(2 / this.input); // Initialize weight
        this.connect(this.nodes[i], this.nodes[j], weight); // Create connection
      }
    }
  }

  /**
   * Activates the network with the given input and returns the output.
   *
   * This method implements the activation process described in the Instinct algorithm,
   * where self-connections are handled differently by multiplying the node's previous
   * state. Activation functions are applied to the node's state to compute the final
   * activation value.
   *
   * @see Instinct Algorithm - Section 1.3 Activation
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6}
   * @param {number[]} input - Array of input values.
   * @param {boolean} [training=false] - Whether the network is in training mode.
   * @returns {number[]} Array of output values.
   */
  activate(input: number[], training = false): number[] {
    const output: number[] = [];

    this.nodes.forEach((node, index) => {
      if (node.type === 'input') {
        node.activate(input[index]); // Activate input nodes with input values
      } else if (node.type === 'output') {
        const activation = node.activate(); // Activate output nodes
        output.push(activation); // Collect output values
      } else {
        if (training) node.mask = Math.random() < this.dropout ? 0 : 1; // Apply dropout during training
        node.activate(); // Activate hidden nodes
      }
    });

    return output; // Return output values
  }

  /**
   * Activates the network without calculating eligibility traces.
   *
   * This is useful for testing or inference where backpropagation is not needed.
   *
   * @param {number[]} input - Array of input values.
   * @returns {number[]} Array of output values.
   */
  noTraceActivate(input: number[]): number[] {
    const output: number[] = [];

    this.nodes.forEach((node, index) => {
      if (node.type === 'input') {
        node.noTraceActivate(input[index]); // Activate input nodes without traces
      } else if (node.type === 'output') {
        const activation = node.noTraceActivate(); // Activate output nodes
        output.push(activation); // Collect output values
      } else {
        node.noTraceActivate(); // Activate hidden nodes without traces
      }
    });

    return output; // Return output values
  }

  /**
   * Propagates the error backward through the network.
   *
   * This method adjusts weights and biases based on the error gradient using the
   * specified learning rate and momentum.
   *
   * @param {number} rate - Learning rate.
   * @param {number} momentum - Momentum factor.
   * @param {boolean} update - Whether to update weights.
   * @param {number[]} target - Target output values.
   * @throws {Error} If the target length does not match the output size.
   */
  propagate(
    rate: number,
    momentum: number,
    update: boolean,
    target: number[]
  ): void {
    if (!target || target.length !== this.output) {
      throw new Error(
        'Output target length should match network output length'
      ); // Validate target size
    }

    let targetIndex = target.length;

    // Propagate error for output nodes
    for (
      let i = this.nodes.length - 1;
      i >= this.nodes.length - this.output;
      i--
    ) {
      this.nodes[i].propagate(rate, momentum, update, target[--targetIndex]);
    }

    // Propagate error for hidden nodes
    for (let i = this.nodes.length - this.output - 1; i >= this.input; i--) {
      this.nodes[i].propagate(rate, momentum, update);
    }
  }

  /**
   * Clears the state of all nodes in the network.
   *
   * This resets activations, states, and eligibility traces.
   */
  clear(): void {
    this.nodes.forEach((node) => node.clear());
  }

  /**
   * Mutates the network using the specified mutation method.
   *
   * Supported mutation methods include adding/removing nodes or connections, modifying weights, etc.
   *
   * @param {any} method - Mutation method to apply.
   * @throws {Error} If no valid mutation method is provided.
   */
  mutate(method: any): void {
    if (!method) {
      throw new Error('No (correct) mutate method given!');
    }

    let connection, node, index, possible, available, pair, randomConn;

    switch (method) {
      case mutation.ADD_NODE:
        connection = this.connections[
          Math.floor(Math.random() * this.connections.length)
        ];
        const gater = connection.gater;
        this.disconnect(connection.from, connection.to);

        node = new Node('hidden');
        node.mutate(mutation.MOD_ACTIVATION);

        const toIndex = this.nodes.indexOf(connection.to);
        const minBound = Math.min(toIndex, this.nodes.length - this.output);
        this.nodes.splice(minBound, 0, node);

        const newConn1 = this.connect(connection.from, node)[0];
        const newConn2 = this.connect(node, connection.to)[0];

        if (gater) {
          this.gate(gater, Math.random() >= 0.5 ? newConn1 : newConn2);
        }
        break;

      case mutation.SUB_NODE:
        if (this.nodes.length === this.input + this.output) {
          if (config.warnings) console.warn('No more nodes left to remove!');
          break;
        }

        index = Math.floor(
          Math.random() * (this.nodes.length - this.output - this.input) +
            this.input
        );
        this.remove(this.nodes[index]);
        break;

      case mutation.ADD_CONN:
        available = [];
        for (let i = 0; i < this.nodes.length - this.output; i++) {
          const node1 = this.nodes[i];
          for (
            let j = Math.max(i + 1, this.input);
            j < this.nodes.length;
            j++
          ) {
            const node2 = this.nodes[j];
            if (!node1.isProjectingTo(node2)) available.push([node1, node2]);
          }
        }

        if (available.length === 0) {
          // No warning to avoid overloading the console
          break;
        }

        pair = available[Math.floor(Math.random() * available.length)];
        this.connect(pair[0], pair[1]);
        break;

      case mutation.SUB_CONN:
        possible = this.connections.filter(
          (conn) =>
            conn.from.connections.out.length > 1 &&
            conn.to.connections.in.length > 1 &&
            this.nodes.indexOf(conn.to) > this.nodes.indexOf(conn.from)
        );

        if (possible.length === 0) {
          // No warning to avoid overloading the console
          break;
        }

        randomConn = possible[Math.floor(Math.random() * possible.length)];
        this.disconnect(randomConn.from, randomConn.to);
        break;

      case mutation.MOD_WEIGHT:
        const allConnections = this.connections.concat(this.selfconns);
        connection =
          allConnections[Math.floor(Math.random() * allConnections.length)];
        const modification =
          Math.random() * (method.max - method.min) + method.min;
        connection.weight += modification;
        break;

      case mutation.MOD_BIAS:
        index = Math.floor(
          Math.random() * (this.nodes.length - this.input) + this.input
        );
        node = this.nodes[index];
        node.mutate(method);
        break;

      case mutation.MOD_ACTIVATION:
        if (
          !method.mutateOutput &&
          this.input + this.output === this.nodes.length
        ) {
          console.warn('No nodes that allow mutation of activation function');
          break;
        }

        index = Math.floor(
          Math.random() *
            (this.nodes.length -
              (method.mutateOutput ? 0 : this.output) -
              this.input) +
            this.input
        );
        node = this.nodes[index];
        node.mutate(method);
        break;

      case mutation.ADD_SELF_CONN:
        possible = this.nodes.filter(
          (node) => node.connections.self.weight === 0
        );

        if (possible.length === 0) {
          console.warn('No more self-connections to add!');
          break;
        }

        node = possible[Math.floor(Math.random() * possible.length)];
        this.connect(node, node);
        break;

      case mutation.SUB_SELF_CONN:
        if (this.selfconns.length === 0) {
          console.warn('No more self-connections to remove!');
          break;
        }

        connection = this.selfconns[
          Math.floor(Math.random() * this.selfconns.length)
        ];
        this.disconnect(connection.from, connection.to);
        break;

      case mutation.ADD_GATE:
        const allConns = this.connections.concat(this.selfconns);
        possible = allConns.filter((conn) => conn.gater === null);

        if (possible.length === 0) {
          if (config.warnings) console.warn('No more connections to gate!');
          break;
        }

        index = Math.floor(
          Math.random() * (this.nodes.length - this.input) + this.input
        ); // Select a random node that is not an input node.
        node = this.nodes[index];
        connection = possible[Math.floor(Math.random() * possible.length)]; // Select a random connection from the list of possible connections.
        this.gate(node, connection); // Gate the selected connection with the selected node.
        break;

      case mutation.SUB_GATE:
        if (this.gates.length === 0) {
          console.warn('No more connections to ungate!'); // Warn if there are no gated connections to remove.
          break;
        }

        index = Math.floor(Math.random() * this.gates.length); // Select a random gated connection.
        const gatedConn = this.gates[index];
        this.ungate(gatedConn); // Remove the gate from the selected connection.
        break;

      case mutation.ADD_BACK_CONN:
        available = [];
        for (let i = this.input; i < this.nodes.length; i++) {
          // Iterate over all nodes except input nodes.
          const node1 = this.nodes[i];
          for (let j = this.input; j < i; j++) {
            // Iterate over nodes that come before the current node to ensure back connections.
            const node2 = this.nodes[j];
            if (!node1.isProjectingTo(node2)) available.push([node1, node2]); // Add to the list if no connection exists between the nodes.
          }
        }

        if (available.length === 0) {
          // No warning to avoid overloading the console
          break;
        }

        pair = available[Math.floor(Math.random() * available.length)];
        this.connect(pair[0], pair[1]);
        break;

      case mutation.SUB_BACK_CONN:
        possible = this.connections.filter(
          (conn) =>
            conn.from.connections.out.length > 1 &&
            conn.to.connections.in.length > 1 &&
            this.nodes.indexOf(conn.from) > this.nodes.indexOf(conn.to)
        );

        if (possible.length === 0) {
          // No warning to avoid overloading the console
          break;
        }

        randomConn = possible[Math.floor(Math.random() * possible.length)];
        this.disconnect(randomConn.from, randomConn.to);
        break;

      case mutation.SWAP_NODES:
        if (
          (method.mutateOutput && this.nodes.length - this.input < 2) ||
          (!method.mutateOutput &&
            this.nodes.length - this.input - this.output < 2)
        ) {
          // No warning to avoid overloading the console
          break;
        }

        const node1Index = Math.floor(
          Math.random() *
            (this.nodes.length -
              (method.mutateOutput ? 0 : this.output) -
              this.input) +
            this.input
        );
        const node2Index = Math.floor(
          Math.random() *
            (this.nodes.length -
              (method.mutateOutput ? 0 : this.output) -
              this.input) +
            this.input
        );

        const node1 = this.nodes[node1Index];
        const node2 = this.nodes[node2Index];

        const tempBias = node1.bias;
        const tempSquash = node1.squash;

        node1.bias = node2.bias;
        node1.squash = node2.squash;
        node2.bias = tempBias;
        node2.squash = tempSquash;
        break;
    }
  }

  /**
   * Connects two nodes in the network.
   *
   * @param {Node} from - The source node.
   * @param {Node} to - The target node.
   * @param {number} [weight] - Optional weight for the connection.
   * @returns {Connection[]} Array of created connections.
   */
  connect(from: Node, to: Node, weight?: number): Connection[] {
    const connections = from.connect(to, weight);

    for (const connection of connections) {
      if (from !== to) {
        this.connections.push(connection);
      } else {
        this.selfconns.push(connection);
      }
    }

    return connections;
  }

  /**
   * Gates a connection with a node.
   *
   * @param {Node} node - The gating node.
   * @param {Connection} connection - The connection to gate.
   * @throws {Error} If the node is not part of the network or the connection is already gated.
   */
  gate(node: Node, connection: Connection): void {
    if (!this.nodes.includes(node)) {
      throw new Error('This node is not part of the network!');
    }
    if (connection.gater) {
      if (config.warnings) console.warn('This connection is already gated!');
      return;
    }
    node.gate(connection);
    this.gates.push(connection);
  }

  /**
   * Removes a node from the network.
   *
   * This also reconnects its input and output nodes to maintain the network's structure.
   *
   * @param {Node} node - The node to remove.
   * @throws {Error} If the node does not exist in the network.
   */
  remove(node: Node): void {
    const index = this.nodes.indexOf(node);

    if (index === -1) {
      throw new Error('This node does not exist in the network!');
    }

    // Keep track of gaters
    const gaters: Node[] = [];

    // Remove self-connections from this.selfconns
    this.disconnect(node, node);

    // Get all its input nodes
    const inputs: Node[] = [];
    for (let i = node.connections.in.length - 1; i >= 0; i--) {
      const connection = node.connections.in[i];
      if (
        mutation.SUB_NODE.keep_gates &&
        connection.gater !== null &&
        connection.gater !== node
      ) {
        gaters.push(connection.gater);
      }
      inputs.push(connection.from);
      this.disconnect(connection.from, node);
    }

    // Get all its output nodes
    const outputs: Node[] = [];
    for (let i = node.connections.out.length - 1; i >= 0; i--) {
      const connection = node.connections.out[i];
      if (
        mutation.SUB_NODE.keep_gates &&
        connection.gater !== null &&
        connection.gater !== node
      ) {
        gaters.push(connection.gater);
      }
      outputs.push(connection.to);
      this.disconnect(node, connection.to);
    }

    // Connect the input nodes to the output nodes (if not already connected)
    const connections: Connection[] = [];
    for (const input of inputs) {
      for (const output of outputs) {
        if (!input.isProjectingTo(output)) {
          const conn = this.connect(input, output);
          connections.push(conn[0]);
        }
      }
    }

    // Gate random connections with gaters
    for (const gater of gaters) {
      if (connections.length === 0) break;

      const connIndex = Math.floor(Math.random() * connections.length);
      this.gate(gater, connections[connIndex]);
      connections.splice(connIndex, 1);
    }

    // Remove gated connections gated by this node
    for (let i = node.connections.gated.length - 1; i >= 0; i--) {
      const conn = node.connections.gated[i];
      this.ungate(conn);
    }

    // Remove self-connection
    this.disconnect(node, node);

    // Remove the node from this.nodes
    this.nodes.splice(index, 1);
  }

  /**
   * Disconnects two nodes in the network.
   *
   * @param {Node} from - The source node.
   * @param {Node} to - The target node.
   */
  disconnect(from: Node, to: Node): void {
    // Determine the correct connections array
    const connections = from === to ? this.selfconns : this.connections;

    // Find and remove the connection
    for (let i = 0; i < connections.length; i++) {
      const connection = connections[i];
      if (connection.from === from && connection.to === to) {
        // If the connection is gated, ungate it
        if (connection.gater !== null) {
          this.ungate(connection);
        }
        connections.splice(i, 1);
        break;
      }
    }

    // Remove the connection from the nodes
    from.disconnect(to);
  }

  /**
   * Removes the gate from a connection.
   *
   * @param {Connection} connection - The connection to ungate.
   * @throws {Error} If the connection is not gated.
   */
  ungate(connection: Connection): void {
    const index = this.gates.indexOf(connection);
    if (index === -1) {
      throw new Error('This connection is not gated!');
    }

    this.gates.splice(index, 1);
    connection.gater?.ungate(connection);
  }

  /**
   * Trains the network on a dataset for one epoch.
   *
   * This is a private method used internally by the `train` method.
   *
   * @param {{ input: number[]; output: number[] }[]} set - Training dataset.
   * @param {number} batchSize - Size of the training batch.
   * @param {number} currentRate - Current learning rate.
   * @param {number} momentum - Momentum factor.
   * @param {(target: number[], output: number[]) => number} costFunction - Cost function to calculate error.
   * @returns {number} Average error for the epoch.
   * @private
   */
  private _trainSet(
    set: { input: number[]; output: number[] }[],
    batchSize: number,
    currentRate: number,
    momentum: number,
    costFunction: (target: number[], output: number[]) => number
  ): number {
    let errorSum = 0;
    for (let i = 0; i < set.length; i++) {
      const input = set[i].input;
      const target = set[i].output;

      const update = !!((i + 1) % batchSize === 0 || i + 1 === set.length);

      const output = this.activate(input, true);
      this.propagate(currentRate, momentum, update, target);

      errorSum += costFunction(target, output);
    }
    return errorSum / set.length;
  }

  /**
   * Trains the network on a dataset.
   *
   * @param {{ input: number[]; output: number[] }[]} set - Training dataset.
   * @param {any} options - Training options.
   * @returns {{ error: number; iterations: number; time: number }} Training results.
   * @throws {Error} If the dataset size does not match the network input/output size.
   */
  train(
    set: { input: number[]; output: number[] }[],
    options: any
  ): { error: number; iterations: number; time: number } {
    if (
      set[0].input.length !== this.input ||
      set[0].output.length !== this.output
    ) {
      throw new Error(
        'Dataset input/output size should match network input/output size!'
      ); // Validate dataset size
    }

    options = options || {};

    // Warning messages
    if (config.warnings) {
      // Check for warnings configuration
      if (typeof options.rate === 'undefined') {
        console.warn('Using default learning rate, please define a rate!'); // Warn if rate is not defined
      }
      if (typeof options.iterations === 'undefined') {
        console.warn(
          'No target iterations given, running until error is reached!'
        ); // Warn if iterations are not defined
      }
    }

    // Read training options
    let targetError = options.error || 0.05; // Default target error
    const cost = options.cost || methods.Cost.mse; // Default cost function
    const baseRate = options.rate || 0.3; // Default learning rate
    const dropout = options.dropout || 0; // Default dropout rate
    const momentum = options.momentum || 0; // Default momentum
    const batchSize = options.batchSize || 1; // Default batch size
    const ratePolicy = options.ratePolicy || methods.Rate.fixed(); // Default rate policy
    const shuffle = options.shuffle || false; // Default shuffle option

    // Validate batch size
    if (batchSize > set.length) {
      throw new Error('Batch size must be smaller or equal to dataset length!');
    } else if (
      typeof options.iterations === 'undefined' &&
      typeof options.error === 'undefined'
    ) {
      throw new Error(
        'At least one of the following options must be specified: error, iterations'
      );
    } else if (typeof options.error === 'undefined') {
      targetError = -1; // run until iterations
    } else if (typeof options.iterations === 'undefined') {
      options.iterations = 0; // run until target error
    }

    this.dropout = dropout; // Set dropout rate

    let trainSet = set;
    let testSet: { input: number[]; output: number[] }[] = [];
    if (options.crossValidate) {
      const numTrain = Math.ceil(
        (1 - options.crossValidate.testSize) * set.length
      ); // Split dataset for cross-validation
      trainSet = set.slice(0, numTrain);
      testSet = set.slice(numTrain);
    }

    let error = 1;
    let iteration = 0;
    const start = Date.now();

    while (
      error > targetError &&
      (options.iterations === 0 || iteration < options.iterations)
    ) {
      if (options.crossValidate && error <= options.crossValidate.testError)
        break;

      iteration++;

      // Update the rate
      const currentRate = ratePolicy(baseRate, iteration);

      // Checks if cross validation is enabled
      if (options.crossValidate) {
        this._trainSet(trainSet, batchSize, currentRate, momentum, cost); // Train on training set
        if (options.clear) this.clear(); // Clear network state if required
        error = this.test(testSet, cost).error; // Test on validation set
        if (options.clear) this.clear(); // Clear network state if required
      } else {
        error = this._trainSet(set, batchSize, currentRate, momentum, cost); // Train on full dataset
        if (options.clear) this.clear(); // Clear network state if required
      }

      // Checks for options such as scheduled logs and shuffling
      if (shuffle) {
        for (let i = set.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [set[i], set[j]] = [set[j], set[i]]; // Shuffle dataset
        }
      }

      if (options.log && iteration % options.log === 0) {
        console.log(
          `Iteration: ${iteration}, Error: ${error}, Rate: ${currentRate}`
        ); // Log progress
      }

      if (options.schedule && iteration % options.schedule.iterations === 0) {
        options.schedule.function({ error, iteration }); // Execute scheduled function
      }
    }

    if (options.clear) this.clear();

    if (dropout) {
      this.nodes.forEach((node) => {
        if (node.type === 'hidden' || node.type === 'varant') {
          node.mask = 1 - this.dropout; // Reset dropout mask
        }
      });
    }

    return { error, iterations: iteration, time: Date.now() - start }; // Return training results
  }

  /**
   * Evolves the network to minimize error on a dataset.
   *
   * This method uses genetic algorithms to optimize the network structure and weights.
   *
   * @param {{ input: number[]; output: number[] }[]} set - Training dataset.
   * @param {any} options - Evolution options.
   * @returns {Promise<{ error: number; iterations: number; time: number }>} Evolution results.
   * @see Instinct Algorithm - Section 4 Constraints
   */
  async evolve(
    set: { input: number[]; output: number[] }[],
    options: any
  ): Promise<{ error: number; iterations: number; time: number }> {
    // Validate dataset size
    if (
      set[0].input.length !== this.input ||
      set[0].output.length !== this.output
    ) {
      throw new Error(
        'Dataset input/output size should match network input/output size!'
      );
    }

    options = options || {};
    let targetError =
      typeof options.error !== 'undefined' ? options.error : 0.05;
    let growth =
      typeof options.growth !== 'undefined' ? options.growth : 0.0001;
    let cost = options.cost || methods.Cost.mse;
    let amount = options.amount || 1;

    // Determine threads
    let threads = options.threads;
    if (typeof threads === 'undefined') {
      if (typeof window === 'undefined' && typeof navigator === 'undefined') {
        threads = require('os').cpus().length;
      } else if (typeof navigator !== 'undefined') {
        threads = navigator.hardwareConcurrency;
      } else {
        threads = 1;
      }
    }

    const start = Date.now();

    if (
      typeof options.iterations === 'undefined' &&
      typeof options.error === 'undefined'
    ) {
      throw new Error(
        'At least one of the following options must be specified: error, iterations'
      );
    } else if (typeof options.error === 'undefined') {
      targetError = -1; // run until iterations
    } else if (typeof options.iterations === 'undefined') {
      options.iterations = 0; // run until target error
    }

    let fitnessFunction: any;
    let workers: any[] = [];
    if (threads === 1) {
      fitnessFunction = (genome: Network) => {
        let score = 0;
        for (let i = 0; i < amount; i++) {
          score -= genome.test(set, cost).error;
        }
        score -=
          (genome.nodes.length -
            genome.input -
            genome.output +
            genome.connections.length +
            genome.gates.length) *
          growth;
        score = isNaN(score) ? -Infinity : score;
        return score / amount;
      };
    } else {
      const converted = Multi.serializeDataSet(set);
      // Create workers
      for (let i = 0; i < threads; i++) {
        if (typeof navigator !== 'undefined') {
          workers.push(
            await Multi.getBrowserTestWorker().then(
              (TestWorker) => new TestWorker(converted, cost)
            )
          );
        } else {
          workers.push(
            await Multi.getNodeTestWorker().then(
              (TestWorker) => new TestWorker(converted, cost)
            )
          );
        }
      }
      fitnessFunction = (population: Network[]) =>
        new Promise<void>((resolve) => {
          const queue = population.slice();
          let done = 0;
          const startWorker = (worker: any) => {
            if (!queue.length) {
              if (++done === threads) resolve();
              return;
            }
            const genome = queue.shift();
            if (typeof genome === 'undefined') {
              startWorker(worker);
              return;
            }
            worker.evaluate(genome).then((result: number) => {
              if (typeof genome !== 'undefined' && typeof result === 'number') {
                genome.score =
                  -result -
                  (genome.nodes.length -
                    genome.input -
                    genome.output +
                    genome.connections.length +
                    genome.gates.length) *
                    growth;
                genome.score = isNaN(result) ? -Infinity : genome.score;
              }
              startWorker(worker);
            });
          };
          workers.forEach(startWorker);
        });
      options.fitnessPopulation = true;
    }

    // Use NEAT instance for evolution
    options.network = this;
    // Dynamically import Neat to avoid circular dependency
    const { default: Neat } = await import('../neat');
    const neat = new Neat(this.input, this.output, fitnessFunction, options);

    let error = -Infinity;
    let bestFitness = -Infinity;
    let bestGenome: any = undefined;

    while (
      error < -targetError &&
      (options.iterations === 0 || neat.generation < options.iterations)
    ) {
      const fittest = await neat.evolve();
      const fitness = fittest.score;
      error =
        (typeof fitness === 'number' ? fitness : 0) +
        (fittest.nodes.length -
          fittest.input -
          fittest.output +
          fittest.connections.length +
          fittest.gates.length) *
          growth;

      if (typeof fitness === 'number' && fitness > bestFitness) {
        bestFitness = fitness;
        bestGenome = fittest;
      }

      if (options.log && neat.generation % options.log === 0) {
        console.log(
          'iteration',
          neat.generation,
          'fitness',
          fitness,
          'error',
          -error
        );
      }

      if (
        options.schedule &&
        neat.generation % options.schedule.iterations === 0
      ) {
        options.schedule.function({
          fitness: fitness,
          error: -error,
          iteration: neat.generation,
        });
      }
    }

    if (threads > 1) {
      for (let i = 0; i < workers.length; i++) workers[i].terminate?.();
    }

    if (typeof bestGenome !== 'undefined') {
      this.nodes = bestGenome.nodes;
      this.connections = bestGenome.connections;
      this.selfconns = bestGenome.selfconns;
      this.gates = bestGenome.gates;
      if (options.clear) this.clear();
    }

    return {
      error: -error,
      iterations: neat.generation,
      time: Date.now() - start,
    };
  }

  /**
   * Tests the network on a dataset.
   *
   * @param {{ input: number[]; output: number[] }[]} set - Test dataset.
   * @param {any} cost - Cost function to evaluate error.
   * @returns {{ error: number; time: number }} Test results.
   */
  test(
    set: { input: number[]; output: number[] }[],
    cost?: any
  ): { error: number; time: number } {
    // Check if dropout is enabled, set correct mask
    if (this.dropout) {
      this.nodes.forEach((node) => {
        if (node.type === 'hidden' || node.type === 'varant') {
          node.mask = 1 - this.dropout;
        }
      });
    }

    const start = Date.now();
    let error = 0;
    const costFn = cost || methods.Cost.mse;

    set.forEach((data) => {
      const output = this.noTraceActivate(data.input);
      error += costFn(data.output, output);
    });

    return { error: error / set.length, time: Date.now() - start };
  }

  /**
   * Serializes the network into a compact format for efficient transfer.
   *
   * @returns {any[]} Serialized network data.
   */
  serialize(): any[] {
    const activations = this.nodes.map((node) => node.activation);
    const states = this.nodes.map((node) => node.state);
    const squashes = this.nodes.map((node) => node.squash.name);
    const connections = this.connections.concat(this.selfconns).map((conn) => ({
      from: conn.from.index,
      to: conn.to.index,
      weight: conn.weight,
      gater: conn.gater ? conn.gater.index : null,
    }));
    return [activations, states, squashes, connections];
  }

  /**
   * Creates a network from serialized data.
   *
   * @param {any[]} data - Serialized network data.
   * @returns {Network} The deserialized network.
   */
  static deserialize(data: any[]): Network {
    const [activations, states, squashes, connections] = data;
    const network = new Network(
      activations.length,
      states.length - activations.length
    );

    network.nodes.forEach((node, index) => {
      node.activation = activations[index];
      node.state = states[index];
      node.squash = methods.Activation[
        squashes[index] as keyof typeof methods.Activation
      ] as (x: number, derivate?: boolean) => number;
    });

    connections.forEach(
      (conn: {
        from: number;
        to: number;
        weight: number;
        gater: number | null;
      }) => {
        const fromNode: Node = network.nodes[conn.from];
        const toNode: Node = network.nodes[conn.to];
        const connection: Connection = network.connect(
          fromNode,
          toNode,
          conn.weight
        )[0];
        if (conn.gater !== null) {
          network.gate(network.nodes[conn.gater], connection);
        }
      }
    );

    return network;
  }

  /**
   * Creates a network from a JSON object.
   *
   * @param {any} json - JSON representation of the network.
   * @returns {Network} The created network.
   */
  static fromJSON(json: any): Network {
    const network = new Network(json.input, json.output);
    network.dropout = json.dropout;
    // Deserialize nodes
    network.nodes = json.nodes.map((nodeJSON: any) => Node.fromJSON(nodeJSON));
    // Set squash functions explicitly
    json.nodes.forEach((nodeJSON: any, index: number) => {
      network.nodes[index].squash = methods.Activation[
        nodeJSON.squash as keyof typeof methods.Activation
      ] as (x: number, derivate?: boolean) => number;
    });
    // Deserialize connections
    network.connections = [];
    network.selfconns = [];
    network.gates = [];
    json.connections.forEach((connJSON: any) => {
      if (
        typeof connJSON.from === 'undefined' ||
        typeof connJSON.to === 'undefined' ||
        typeof network.nodes[connJSON.from] === 'undefined' ||
        typeof network.nodes[connJSON.to] === 'undefined'
      ) {
        return;
      }
      const connection = network.connect(
        network.nodes[connJSON.from],
        network.nodes[connJSON.to]
      )[0];
      connection.weight = connJSON.weight;
      if (
        connJSON.gater !== null &&
        typeof connJSON.gater === 'number' &&
        connJSON.gater < network.nodes.length
      ) {
        network.gate(network.nodes[connJSON.gater], connection);
      }
    });
    return network;
  }

  /**
   * Merges two networks into one.
   *
   * The output of the first network is connected to the input of the second network.
   *
   * @param {Network} network1 - The first network.
   * @param {Network} network2 - The second network.
   * @returns {Network} The merged network.
   * @throws {Error} If the output size of the first network does not match the input size of the second network.
   */
  static merge(network1: Network, network2: Network): Network {
    if (network1.output !== network2.input) {
      throw new Error(
        'Output size of network1 must match input size of network2!'
      );
    }

    const merged = new Network(network1.input, network2.output);
    merged.nodes = [...network1.nodes, ...network2.nodes.slice(network2.input)];
    merged.connections = [
      ...network1.connections,
      ...network2.connections.map((conn) => {
        if (conn.from.type === 'input') {
          conn.from =
            network1.nodes[
              network1.nodes.length - network2.nodes.indexOf(conn.from) - 1
            ];
        }
        return conn;
      }),
    ];

    return merged;
  }

  /**
   * Creates an offspring network by crossing over two parent networks.
   *
   * This method implements the crossover logic described in the Instinct algorithm.
   *
   * @param {Network} network1 - The first parent network.
   * @param {Network} network2 - The second parent network.
   * @param {boolean} [equal] - Whether the offspring should have an equal number of nodes from both parent networks.
   * @returns {Network} The offspring network.
   * @throws {Error} If the input/output sizes of the parent networks do not match.
   * @see Instinct Algorithm - Section 2 Crossover
   */
  static crossOver(
    network1: Network,
    network2: Network,
    equal?: boolean
  ): Network {
    if (
      network1.input !== network2.input ||
      network1.output !== network2.output
    ) {
      throw new Error(
        'Networks must have the same input and output sizes to cross over.'
      );
    }

    const offspring = new Network(network1.input, network1.output);
    offspring.connections = [];
    offspring.nodes = [];

    const score1 = network1.score || 0;
    const score2 = network2.score || 0;

    let size;
    if (equal || score1 === score2) {
      const max = Math.max(network1.nodes.length, network2.nodes.length);
      const min = Math.min(network1.nodes.length, network2.nodes.length);
      size = Math.floor(Math.random() * (max - min + 1) + min);
    } else if (score1 > score2) {
      size = network1.nodes.length;
    } else {
      size = network2.nodes.length;
    }

    const outputSize = network1.output;

    network1.nodes.forEach((node, index) => (node.index = index));
    network2.nodes.forEach((node, index) => (node.index = index));

    for (let i = 0; i < size; i++) {
      let node;
      if (i < size - outputSize) {
        const random = Math.random();
        node = random >= 0.5 ? network1.nodes[i] : network2.nodes[i];
        const other = random < 0.5 ? network1.nodes[i] : network2.nodes[i];
        if (!node || node.type === 'output') node = other;
      } else {
        node =
          Math.random() >= 0.5
            ? network1.nodes[network1.nodes.length + i - size]
            : network2.nodes[network2.nodes.length + i - size];
      }

      const newNode = new Node(node.type);
      newNode.bias = node.bias;
      newNode.squash = node.squash;
      offspring.nodes.push(newNode);
    }

    const n1conns: Record<string, any> = {};
    const n2conns: Record<string, any> = {};

    network1.connections.forEach((conn) => {
      n1conns[Connection.innovationID(conn.from.index, conn.to.index)] = {
        weight: conn.weight,
        from: conn.from.index,
        to: conn.to.index,
        gater: conn.gater ? conn.gater.index : -1,
      };
    });

    network1.selfconns.forEach((conn) => {
      n1conns[Connection.innovationID(conn.from.index, conn.to.index)] = {
        weight: conn.weight,
        from: conn.from.index,
        to: conn.to.index,
        gater: conn.gater ? conn.gater.index : -1,
      };
    });

    network2.connections.forEach((conn) => {
      n2conns[Connection.innovationID(conn.from.index, conn.to.index)] = {
        weight: conn.weight,
        from: conn.from.index,
        to: conn.to.index,
        gater: conn.gater ? conn.gater.index : -1,
      };
    });

    network2.selfconns.forEach((conn) => {
      n2conns[Connection.innovationID(conn.from.index, conn.to.index)] = {
        weight: conn.weight,
        from: conn.from.index,
        to: conn.to.index,
        gater: conn.gater ? conn.gater.index : -1,
      };
    });

    const connections: any[] = [];
    const keys1 = Object.keys(n1conns);
    const keys2 = Object.keys(n2conns);

    keys1.forEach((key) => {
      if (n2conns[key]) {
        connections.push(Math.random() >= 0.5 ? n1conns[key] : n2conns[key]);
        delete n2conns[key];
      } else if (score1 >= score2 || equal) {
        connections.push(n1conns[key]);
      }
    });

    if (score2 >= score1 || equal) {
      keys2.forEach((key) => {
        if (n2conns[key]) connections.push(n2conns[key]);
      });
    }

    connections.forEach((connData) => {
      if (connData.to < size && connData.from < size) {
        const from = offspring.nodes[connData.from];
        const to = offspring.nodes[connData.to];
        const conn = offspring.connect(from, to)[0];
        conn.weight = connData.weight;

        if (connData.gater !== -1 && connData.gater < size) {
          offspring.gate(offspring.nodes[connData.gater], conn);
        }
      }
    });

    return offspring;
  }

  /**
   * Generates a standalone JavaScript function for the network.
   * This function can be used independently of the library.
   * @returns {string} Standalone function as a string.
   */
  standalone(): string {
    const present: { [key: string]: string } = {}; // Map name to function string
    const squashDefinitions: string[] = []; // Store definitions of functions
    const functionMap: { [key: string]: number } = {}; // Map name to index in F
    let functionIndexCounter = 0;

    const activations: number[] = [];
    const states: number[] = [];
    const lines: string[] = [];

    // Predefined activation functions (ensure these match your methods.Activation)
    const predefined: { [key: string]: string } = {
      LOGISTIC: 'function LOGISTIC(x){ return 1 / (1 + Math.exp(-x)); }',
      TANH: 'function TANH(x){ return Math.tanh(x); }',
      RELU: 'function RELU(x){ return x > 0 ? x : 0; }',
      IDENTITY: 'function IDENTITY(x){ return x; }',
      STEP: 'function STEP(x){ return x > 0 ? 1 : 0; }',
      SOFTSIGN: 'function SOFTSIGN(x){ return x / (1 + Math.abs(x)); }',
      SINUSOID: 'function SINUSOID(x){ return Math.sin(x); }',
      GAUSSIAN: 'function GAUSSIAN(x){ return Math.exp(-Math.pow(x, 2)); }',
      BENT_IDENTITY:
        'function BENT_IDENTITY(x){ return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x; }',
      BIPOLAR: 'function BIPOLAR(x){ return x > 0 ? 1 : -1; }',
      BIPOLAR_SIGMOID:
        'function BIPOLAR_SIGMOID(x){ return 2 / (1 + Math.exp(-x)) - 1; }',
      HARD_TANH:
        'function HARD_TANH(x){ return Math.max(-1, Math.min(1, x)); }',
      ABSOLUTE: 'function ABSOLUTE(x){ return Math.abs(x); }',
      INVERSE: 'function INVERSE(x){ return 1 - x; }',
      // Corrected SELU definition
      SELU:
        'function SELU(x){ var alpha = 1.6732632423543772848170429916717; var scale = 1.0507009873554804934193349852946; var fx = x > 0 ? x : alpha * Math.exp(x) - alpha; return fx * scale; }',
      SOFTPLUS: 'function SOFTPLUS(x){ return Math.log(1 + Math.exp(x)); }',
      // Add other predefined functions used in your methods.Activation here
    };

    // Set up input activations and states
    for (let i = 0; i < this.input; i++) {
      const node = this.nodes[i];
      activations.push(node.activation);
      states.push(node.state);
    }

    lines.push('for(var i = 0; i < input.length; i++) A[i] = input[i];');

    // Assign indices for all nodes
    this.nodes.forEach((node, index) => {
      node.index = index;
    });

    // Set up hidden/output nodes
    for (let i = this.input; i < this.nodes.length; i++) {
      const node = this.nodes[i];
      activations.push(node.activation);
      states.push(node.state);

      const squash = node.squash as any;
      const squashName = squash.name || `anonymous_squash_${i}`;

      if (!(squashName in present)) {
        let funcStr: string;
        if (predefined[squashName]) {
          funcStr = predefined[squashName]; // Use predefined definition
          // *** CRITICAL: Even predefined strings might be affected if coverage runs *after* this code generation step ***
          // Although unlikely, let's strip the predefined string too, just in case.
          funcStr = stripCoverage(funcStr);
        } else {
          // Fallback for custom/unknown functions
          funcStr = squash.toString();
          // Attempt to clean up the stringified function definition
          funcStr = stripCoverage(funcStr); // Strip coverage here (using enhanced function)

          if (!funcStr.startsWith('function') && funcStr.includes('=>')) {
            // Handle arrow functions (basic conversion)
            funcStr = `function ${squashName}${funcStr.substring(
              funcStr.indexOf('(')
            )}`;
          } else if (
            !funcStr.startsWith('function') &&
            !funcStr.includes('=>')
          ) {
            // Handle cases where 'function' keyword might be missing
            if (!funcStr.includes('{')) {
              console.warn(
                `Standalone: Could not get definition for squash function '${squashName}'. Using name directly.`
              );
              funcStr = `function ${squashName}(){ /* unknown */ }`; // Placeholder
            } else {
              // Assume it's just missing 'function name'
              funcStr = `function ${squashName}${funcStr.substring(
                funcStr.indexOf('(')
              )}`;
            }
          } else if (
            funcStr.startsWith('function') &&
            !funcStr.includes(squashName)
          ) {
            // Ensure the function name is correct if it was anonymous
            funcStr = `function ${squashName}${funcStr.substring(
              funcStr.indexOf('(')
            )}`;
          }
        }
        present[squashName] = funcStr;
        squashDefinitions.push(present[squashName]); // Add definition to list
        functionMap[squashName] = functionIndexCounter++; // Assign index
      }
      const functionIndex = functionMap[squashName];

      // ... (rest of the incoming connection logic remains the same) ...
      const incoming: string[] = [];
      for (const conn of node.connections.in) {
        let computation = `A[${conn.from.index}] * ${conn.weight}`;
        if (conn.gater && typeof conn.gater.index !== 'undefined') {
          // Check gater index exists
          computation += ` * A[${conn.gater.index}]`;
        }
        incoming.push(computation);
      }

      if (node.connections.self.weight) {
        let computation = `S[${i}] * ${node.connections.self.weight}`;
        if (
          node.connections.self.gater &&
          typeof node.connections.self.gater.index !== 'undefined'
        ) {
          // Check gater index exists
          computation += ` * A[${node.connections.self.gater.index}]`;
        }
        incoming.push(computation);
      }

      const incomingCalculation =
        incoming.length > 0 ? incoming.join(' + ') : '0';
      lines.push(`S[${i}] = ${incomingCalculation} + ${node.bias};`);
      const maskValue = typeof node.mask !== 'undefined' ? node.mask : 1;
      lines.push(
        `A[${i}] = F[${functionIndex}](S[${i}])${
          maskValue !== 1 ? ` * ${maskValue}` : ''
        };`
      );
    }

    // ... (rest of the output generation remains the same) ...
    const output = [];
    for (let i = this.nodes.length - this.output; i < this.nodes.length; i++) {
      // Ensure index exists before pushing
      if (typeof i !== 'undefined' && i < this.nodes.length) {
        output.push(`A[${i}]`);
      } else {
        console.warn(`Standalone: Invalid output node index ${i}`);
      }
    }
    lines.push(`return [${output.join(',')}];`);

    // Construct the final standalone function string
    let total = '';
    total += `(function(){\n`;
    // Define the functions first
    total += `${squashDefinitions.join('\n')}\n`;
    // Create the F array mapping indices to function names (references)
    const fArrayContent = Object.entries(functionMap)
      .sort(([, a], [, b]) => a - b)
      .map(([name]) => name)
      .join(',');
    total += `var F = [${fArrayContent}];\n`;
    total += `var A = [${activations.join(',')}];\n`;
    total += `var S = [${states.join(',')}];\n`;
    total += `function activate(input){\n`;
    // Add safety check for input length
    total += `if (input.length !== ${this.input}) { throw new Error('Invalid input size. Expected ${this.input}, got ' + input.length); }\n`;
    total += `${lines.join('\n')}\n}\n`;
    total += `return activate;\n`;
    total += `})()`;

    // Final strip just in case something was missed (belt and suspenders)
    return stripCoverage(total); // Apply enhanced stripping to the final result
  }

  /**
   * Sets the value of a property for every node in the network.
   *
   * @param {{ bias?: number; squash?: any }} values - Values to set.
   */
  set(values: { bias?: number; squash?: any }): void {
    this.nodes.forEach((node) => {
      if (values.bias !== undefined) node.bias = values.bias;
      if (values.squash !== undefined) node.squash = values.squash;
    });
  }

  /**
   * Converts the network to a JSON object.
   *
   * This includes nodes, connections, and other properties.
   *
   * @returns {object} JSON representation of the network.
   */
  toJSON(): object {
    const json: any = {
      nodes: [], // Array to store serialized nodes.
      connections: [], // Array to store serialized connections.
      input: this.input, // Number of input nodes.
      output: this.output, // Number of output nodes.
      dropout: this.dropout, // Dropout rate of the network.
    };

    this.nodes.forEach((node, index) => {
      node.index = index; // Assign an index to each node for serialization.
      const nodeJSON: any = node.toJSON(); // Serialize the node.
      nodeJSON.squash = node.squash.name; // Serialize the name of the squash function.
      json.nodes.push(nodeJSON); // Add the serialized node to the JSON object.

      if (node.connections.self.weight !== 0) {
        // Check if the node has a self-connection.
        const selfConn: any = node.connections.self.toJSON(); // Serialize the self-connection.
        selfConn.from = index; // Set the source of the self-connection.
        selfConn.to = index; // Set the target of the self-connection.
        selfConn.gater = node.connections.self.gater
          ? node.connections.self.gater.index
          : null; // Serialize the gater if it exists.
        json.connections.push(selfConn); // Add the self-connection to the JSON object.
      }
    });

    this.connections.forEach((conn: Connection) => {
      const connJSON: any = { weight: conn.weight };
      (conn as Connection).from &&
        (connJSON.from = (conn as Connection).from.index); // Explicitly cast to Connection
      (conn as Connection).to && (connJSON.to = (conn as Connection).to.index); // Explicitly cast to Connection
      (conn as Connection).gater &&
        (connJSON.gater = (conn as Connection).gater.index); // Explicitly cast to Connection
      json.connections.push(connJSON);
    });

    return json;
  }
}
