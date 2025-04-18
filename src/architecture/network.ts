import Node from './node';
import Connection from './connection';
import Multi from '../multithreading/multi';
import * as methods from '../methods/methods';
import mutation from '../methods/mutation'; // Fix the import for mutation
import { config } from '../config';

/**
 * Represents a neural network with nodes and connections.
 * The network supports feedforward and recurrent architectures.
 */
export default class Network {
  input: number; // Number of input nodes
  output: number; // Number of output nodes
  score?: number; // Optional score property for genetic algorithms
  nodes: Node[]; // List of all nodes in the network
  connections: Connection[]; // List of all connections between nodes
  gates: any[]; // List of gated connections
  selfconns: any[]; // List of self-connections
  dropout: number = 0; // Dropout rate for training

  /**
   * Creates a new network with the specified number of input and output nodes.
   * @param {number} input - Number of input nodes.
   * @param {number} output - Number of output nodes.
   * @throws Will throw an error if input or output size is not provided.
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
   * @param {number[]} input - Array of input values.
   * @param {boolean} [training=false] - Whether the network is in training mode.
   * @returns {number[]} - Array of output values.
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
   * This is useful for testing or inference where backpropagation is not needed.
   * @param {number[]} input - Array of input values.
   * @returns {number[]} - Array of output values.
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
   * @param {number} rate - Learning rate.
   * @param {number} momentum - Momentum factor.
   * @param {boolean} update - Whether to update weights.
   * @param {number[]} target - Target output values.
   * @throws Will throw an error if the target length does not match the output size.
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
   * This resets activations, states, and eligibility traces.
   */
  clear(): void {
    this.nodes.forEach((node) => node.clear());
  }

  /**
   * Mutates the network using the specified mutation method.
   * Supported mutation methods include adding/removing nodes or connections, modifying weights, etc.
   * @param {any} method - Mutation method to apply.
   * @throws Will throw an error if no valid mutation method is provided.
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
          console.warn('No more connections to be made!');
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
          console.warn('No connections to remove!');
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
          console.warn('No more connections to be made!'); // Warn if no back connections can be added.
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
          console.warn('No connections to remove!');
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
          console.warn(
            'No nodes that allow swapping of bias and activation function'
          );
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
   * @param {Node} from - The source node.
   * @param {Node} to - The target node.
   * @param {number} [weight] - Optional weight for the connection.
   * @returns {Connection[]} - Array of created connections.
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
   * @param {Node} node - The gating node.
   * @param {Connection} connection - The connection to gate.
   * @throws Will throw an error if the node is not part of the network or the connection is already gated.
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
   * This also reconnects its input and output nodes to maintain the network's structure.
   * @param {Node} node - The node to remove.
   * @throws Will throw an error if the node does not exist in the network.
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
   * @param {Connection} connection - The connection to ungate.
   * @throws Will throw an error if the connection is not gated.
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
   * This is a private method used internally by the `train` method.
   * @param {{ input: number[]; output: number[] }[]} set - Training dataset.
   * @param {number} batchSize - Size of the training batch.
   * @param {number} currentRate - Current learning rate.
   * @param {number} momentum - Momentum factor.
   * @param {(target: number[], output: number[]) => number} costFunction - Cost function to calculate error.
   * @returns {number} - Average error for the epoch.
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
   * @param {{ input: number[]; output: number[] }[]} set - Training dataset.
   * @param {any} options - Training options.
   * @returns {{ error: number; iterations: number; time: number }} - Training results.
   * @throws Will throw an error if the dataset size does not match the network input/output size.
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
    const targetError = options.error || 0.05; // Default target error
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
      (!options.iterations || iteration < options.iterations)
    ) {
      if (shuffle) {
        for (let i = set.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [set[i], set[j]] = [set[j], set[i]]; // Shuffle dataset
        }
      }

      const currentRate = ratePolicy(baseRate, iteration); // Adjust learning rate

      if (options.crossValidate) {
        this._trainSet(trainSet, batchSize, currentRate, momentum, cost); // Train on training set
        if (options.clear) this.clear(); // Clear network state if required
        error = this.test(testSet, cost).error; // Test on validation set
        if (options.clear) this.clear(); // Clear network state if required

        if (error <= options.crossValidate.testError) {
          // Stop if test error is below threshold
          break;
        }
      } else {
        error = this._trainSet(set, batchSize, currentRate, momentum, cost); // Train on full dataset
        if (options.clear) this.clear(); // Clear network state if required
      }

      if (options.log && iteration % options.log === 0) {
        console.log(
          `Iteration: ${iteration}, Error: ${error}, Rate: ${currentRate}`
        ); // Log progress
      }

      if (options.schedule && iteration % options.schedule.iterations === 0) {
        options.schedule.function({ error, iteration }); // Execute scheduled function
      }

      iteration++;
    }

    if (dropout) {
      this.nodes.forEach((node) => {
        if (node.type === 'hidden') {
          node.mask = 1 - this.dropout; // Reset dropout mask
        }
      });
    }

    return { error, iterations: iteration, time: Date.now() - start }; // Return training results
  }

  /**
   * Evolves the network to minimize error on a dataset.
   * Uses genetic algorithms to optimize the network structure and weights.
   * @param {{ input: number[]; output: number[] }[]} set - Training dataset.
   * @param {any} options - Evolution options.
   * @returns {Promise<{ error: number; iterations: number; time: number }>} - Evolution results.
   */
  async evolve(
    set: { input: number[]; output: number[] }[],
    options: any
  ): Promise<{ error: number; iterations: number; time: number }> {
    const start = Date.now();
    const targetError = options.error || 0.05;
    const cost =
      options.cost ||
      ((target: number[], output: number[]) =>
        target.reduce((sum, t, i) => sum + Math.pow(t - output[i], 2), 0) /
        target.length);
    const growth = options.growth || 0.0001;
    const amount = options.amount || 1;

    let bestFitness = -Infinity;
    let bestGenome: Network | undefined;
    let error = -Infinity;

    // Multithreading logic (if applicable)
    const threads =
      options.threads ||
      (typeof navigator !== 'undefined'
        ? navigator.hardwareConcurrency
        : require('os').cpus().length);
    if (threads > 1) {
      const converted = Multi.serializeDataSet(set); // Fix reference to Multi
      const workers = await Promise.all(
        Array.from(
          { length: threads },
          () =>
            typeof navigator !== 'undefined'
              ? Multi.getBrowserTestWorker().then(
                  (TestWorker) => new TestWorker(converted, cost)
                ) // Fix workers reference
              : Multi.getNodeTestWorker().then(
                  (TestWorker) => new TestWorker(converted, cost)
                ) // Fix workers reference
        )
      );

      options.fitnessPopulation = true; // Set fitnessPopulation for multithreading

      const fitnessFunction = (population: Network[]) =>
        new Promise<void>((resolve) => {
          const queue = [...population];
          let done = 0;

          const startWorker = (worker: any) => {
            if (!queue.length) {
              if (++done === threads) resolve();
              return;
            }

            const genome = queue.shift();
            worker.evaluate(genome).then((result: number) => {
              if (genome) {
                genome.score =
                  -result -
                  (genome.nodes.length -
                    genome.input -
                    genome.output +
                    genome.connections.length +
                    genome.gates.length) *
                    growth;
              }
              startWorker(worker);
            });
          };

          workers.forEach(startWorker);
        });
    }

    while (
      error < -targetError &&
      (!options.iterations || options.iterations-- > 0)
    ) {
      const fitness =
        -this.test(set, cost).error -
        (this.nodes.length -
          this.input -
          this.output +
          this.connections.length +
          this.gates.length) *
          growth;
      if (fitness > bestFitness) {
        bestFitness = fitness;
        bestGenome = this;
      }
      error = fitness;
    }

    if (bestGenome) {
      this.nodes = bestGenome.nodes;
      this.connections = bestGenome.connections;
      this.selfconns = bestGenome.selfconns;
      this.gates = bestGenome.gates;
    }

    return {
      error: -error,
      iterations: options.iterations,
      time: Date.now() - start,
    };
  }

  /**
   * Serializes the network into a compact format for efficient transfer.
   * This method is not part of the original JavaScript implementation.
   * @returns {any[]} - Serialized network data.
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
   * This method is not part of the original JavaScript implementation.
   * @param {any[]} data - Serialized network data.
   * @returns {Network} - The deserialized network.
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
   * Merges two networks into one.
   * This method is not part of the original JavaScript implementation.
   * The output of the first network is connected to the input of the second network.
   * @param {Network} network1 - The first network.
   * @param {Network} network2 - The second network.
   * @returns {Network} - The merged network.
   * @throws Will throw an error if the output size of the first network does not match the input size of the second network.
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
   * @param {Network} network1 - The first parent network.
   * @param {Network} network2 - The second parent network.
   * @returns {Network} - The offspring network.
   * @throws Will throw an error if the input/output sizes of the parent networks do not match.
   */
  static crossOver(network1: Network, network2: Network): Network {
    if (
      network1.input !== network2.input ||
      network1.output !== network2.output
    ) {
      throw new Error(
        'Networks must have the same input and output sizes to cross over.'
      );
    }

    const offspring = new Network(network1.input, network1.output);

    offspring.nodes = network1.nodes.map((node, index) =>
      Math.random() > 0.5 ? node : network2.nodes[index]
    );

    offspring.connections = network1.connections.map((conn, index) =>
      Math.random() > 0.5 ? conn : network2.connections[index]
    );

    return offspring;
  }

  /**
   * Tests the network on a dataset.
   * @param {{ input: number[]; output: number[] }[]} set - Test dataset.
   * @param {any} cost - Cost function to evaluate error.
   * @returns {{ error: number; time: number }} - Test results.
   */
  test(
    set: { input: number[]; output: number[] }[],
    cost: any
  ): { error: number; time: number } {
    // Check if dropout is enabled, set correct mask
    if (this.dropout) {
      this.nodes.forEach((node) => {
        if (node.type === 'hidden') {
          node.mask = 1 - this.dropout;
        }
      });
    }

    const start = Date.now();
    let error = 0;

    set.forEach((data) => {
      const output = this.noTraceActivate(data.input);
      error += cost(data.output, output);
    });

    return { error: error / set.length, time: Date.now() - start };
  }

  /**
   * Generates a graph representation of the network.
   * This can be used for visualization purposes.
   * @param {number} width - Width of the graph.
   * @param {number} height - Height of the graph.
   * @returns {object} - Graph representation.
   */
  graph(width: number, height: number): object {
    let input = 0,
      output = 0;
    const json: any = {
      nodes: [],
      links: [],
      constraints: [
        { type: 'alignment', axis: 'x', offsets: [] },
        { type: 'alignment', axis: 'y', offsets: [] },
      ],
    };

    this.nodes.forEach((node, i) => {
      if (node.type === 'input') {
        json.constraints[0].offsets.push({
          node: i,
          offset: (input++ * (0.8 * width)) / (this.input - 1),
        });
        json.constraints[1].offsets.push({ node: i, offset: 0 });
      } else if (node.type === 'output') {
        json.constraints[0].offsets.push({
          node: i,
          offset: (output++ * (0.8 * width)) / (this.output - 1),
        });
        json.constraints[1].offsets.push({ node: i, offset: -0.8 * height });
      }
      json.nodes.push({
        id: i,
        name:
          node.type === 'hidden' ? node.squash.name : node.type.toUpperCase(),
        activation: node.activation,
        bias: node.bias,
      });
    });

    this.connections.concat(this.selfconns).forEach((conn) => {
      if (!conn.gater) {
        json.links.push({
          source: this.nodes.indexOf(conn.from),
          target: this.nodes.indexOf(conn.to),
          weight: conn.weight,
        });
      } else {
        const gaterIndex = json.nodes.length;
        json.nodes.push({
          id: gaterIndex,
          activation: conn.gater.activation,
          name: 'GATE',
        });
        json.links.push({
          source: this.nodes.indexOf(conn.from),
          target: gaterIndex,
          weight: conn.weight / 2,
        });
        json.links.push({
          source: gaterIndex,
          target: this.nodes.indexOf(conn.to),
          weight: conn.weight / 2,
        });
        json.links.push({
          source: this.nodes.indexOf(conn.gater),
          target: gaterIndex,
          weight: conn.gater.activation,
          gate: true,
        });
      }
    });

    return json;
  }

  /**
   * Sets the value of a property for every node in the network.
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
   * This includes nodes, connections, and other properties.
   * @returns {object} - JSON representation of the network.
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

  /**
   * Creates a network from a JSON object.
   * @param {any} json - JSON representation of the network.
   * @returns {Network} - The created network.
   */
  static fromJSON(json: any): Network {
    const network = new Network(json.input, json.output); // Create new network
    network.dropout = json.dropout; // Set dropout rate
    network.nodes = json.nodes.map((nodeJSON: any) => Node.fromJSON(nodeJSON)); // Deserialize nodes

    // Deserialize squash functions
    json.nodes.forEach((nodeJSON: any, index: number) => {
      network.nodes[index].squash = methods.Activation[
        nodeJSON.squash as keyof typeof methods.Activation
      ] as (x: number, derivate?: boolean) => number; // Ensure correct type
    });

    json.connections.forEach((connJSON: any) => {
      const connection = network.connect(
        network.nodes[connJSON.from],
        network.nodes[connJSON.to]
      )[0]; // Create connection
      connection.weight = connJSON.weight; // Set connection weight

      if (connJSON.gater !== null) {
        network.gate(network.nodes[connJSON.gater], connection); // Gate connection if applicable
      }
    });

    return network; // Return deserialized network
  }

  /**
   * Generates a standalone JavaScript function for the network.
   * This function can be used independently of the library.
   * @returns {string} - Standalone function as a string.
   */
  standalone(): string {
    const present: string[] = [];
    const activations: number[] = [];
    const states: number[] = [];
    const lines: string[] = [];
    const functions: string[] = [];

    this.nodes.forEach((node, index) => {
      activations.push(node.activation);
      states.push(node.state);

      if (node.type !== 'input') {
        const funcIndex = present.indexOf(node.squash.name);
        const squashIndex =
          funcIndex === -1 ? present.push(node.squash.name) - 1 : funcIndex;

        const incoming = node.connections.in.map((conn) => {
          let computation = `A[${conn.from.index}] * ${conn.weight}`;
          if (conn.gater) computation += ` * A[${conn.gater.index}]`;
          return computation;
        });

        if (node.connections.self.weight) {
          let selfComputation = `S[${index}] * ${node.connections.self.weight}`;
          if (node.connections.self.gater) {
            selfComputation += ` * A[${node.connections.self.gater.index}]`;
          }
          incoming.push(selfComputation);
        }

        lines.push(`S[${index}] = ${incoming.join(' + ')} + ${node.bias};`);
        lines.push(
          `A[${index}] = F[${squashIndex}](S[${index}])${
            node.mask !== undefined ? ` * ${node.mask}` : ''
          };`
        ); // Fixed mask handling
      }
    });

    const outputIndices = this.nodes
      .slice(-this.output)
      .map((_, i) => `A[${this.nodes.length - this.output + i}]`);
    lines.push(`return [${outputIndices.join(',')}];`);

    const squashFunctions = this.nodes
      .map((node) => node.squash)
      .filter((squash, index, self) => self.indexOf(squash) === index)
      .map((squash) => squash.toString());

    return `
      var F = [${squashFunctions.join(',')}];
      var A = [${activations.join(',')}];
      var S = [${states.join(',')}];
      function activate(input) {
        ${lines.join('\n')}
      }
    `;
  }
}
