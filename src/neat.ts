import Network from './architecture/network';
import * as methods from './methods/methods';
import NodeType from './architecture/node'; // Import the Node type with a different name to avoid conflicts

type Options = {
  equal?: boolean;
  clear?: boolean;
  popsize?: number;
  elitism?: number;
  provenance?: number;
  mutationRate?: number;
  mutationAmount?: number;
  fitnessPopulation?: boolean;
  selection?: any;
  crossover?: any[];
  mutation?: any;
  network?: Network;
  maxNodes?: number;
  maxConns?: number;
  maxGates?: number;
  mutationSelection?: (genome: any) => any;
  allowRecurrent?: boolean; // Add allowRecurrent option
  hiddenLayerMultiplier?: number; // Add hiddenLayerMultiplier option
  minHidden?: number; // Add minHidden option for minimum hidden nodes in evolved networks
  seed?: number; // Optional seed for deterministic evolution
};

export default class Neat {
  input: number;
  output: number;
  fitness: (network: Network) => number;
  options: Options;
  population: Network[] = [];
  generation: number = 0;
  // Deterministic RNG state (lazy init)
  private _rngState?: number;
  private _rng?: () => number;

  /**
   * Initializes a new instance of the Neat class.
   * @param input - Number of input nodes in the network.
   * @param output - Number of output nodes in the network.
   * @param fitness - Fitness function to evaluate the performance of networks.
   * @param options - Configuration options for the evolutionary process.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
   */
  constructor(
    input: number,
    output: number,
    fitness: (network: Network) => number,
    options: Options = {}
  ) {
    this.input = input;
    this.output = output;
    this.fitness = fitness;
    this.options = options;

    this.options.equal = this.options.equal || false;
    this.options.clear = this.options.clear || false;
    this.options.popsize = this.options.popsize || 50;
    this.options.elitism = this.options.elitism || 0;
    this.options.provenance = this.options.provenance || 0;
    this.options.mutationRate = this.options.mutationRate || 0.7;
    this.options.mutationAmount = this.options.mutationAmount || 1;
    this.options.fitnessPopulation = this.options.fitnessPopulation || false;
    this.options.selection = this.options.selection || methods.selection.POWER;
    this.options.crossover = this.options.crossover || [
      methods.crossover.SINGLE_POINT,
      methods.crossover.TWO_POINT,
      methods.crossover.UNIFORM,
      methods.crossover.AVERAGE,
    ];
    // Initialize allowRecurrent first, defaulting to false if not specified
    this.options.allowRecurrent = typeof this.options.allowRecurrent === 'boolean' ? this.options.allowRecurrent : false;

    // Set mutation methods based on allowRecurrent, if not explicitly provided in options
    if (this.options.mutation === undefined) {
      if (this.options.allowRecurrent) {
        this.options.mutation = methods.mutation.ALL; // Use all mutations if recurrent is allowed
      } else {
        this.options.mutation = methods.mutation.FFW; // Default to FFW for non-recurrent
      }
    }
    
    this.options.maxNodes = this.options.maxNodes || Infinity;
    this.options.maxConns = this.options.maxConns || Infinity;
    this.options.maxGates = this.options.maxGates || Infinity;

  this.createPool(this.options.network || null);
  }

  /**
   * Gets the minimum hidden layer size for a network based on input/output sizes.
   * Uses the formula: max(input, output) x multiplier (default random 2-5)
   * Allows deterministic override for testing.
   * @param multiplierOverride Optional fixed multiplier for deterministic tests
   * @returns The minimum number of hidden nodes required in each hidden layer
   */
  getMinimumHiddenSize(multiplierOverride?: number): number {
    let hiddenLayerMultiplier: number;
    if (typeof multiplierOverride === 'number') {
      hiddenLayerMultiplier = multiplierOverride;
    } else if (typeof this.options.hiddenLayerMultiplier === 'number') {
      hiddenLayerMultiplier = this.options.hiddenLayerMultiplier;
    } else {
      const rng = this._getRNG();
  hiddenLayerMultiplier = Math.floor(rng() * (4 - 2 + 1)) + 2; // 2 to 4
    }
    return Math.max(this.input, this.output) * hiddenLayerMultiplier;
  }

  private _getRNG(): () => number {
    if (this._rng) return this._rng;
    if (typeof this.options.seed === 'number') {
      this._rngState = this.options.seed >>> 0;
      this._rng = () => {
        this._rngState = (this._rngState! + 0x6D2B79F5) >>> 0;
        let r = Math.imul(this._rngState! ^ (this._rngState! >>> 15), 1 | this._rngState!);
        r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
      };
      return this._rng;
    }
    this._rng = Math.random;
    return this._rng;
  }
  
  /**
   * Checks if a network meets the minimum hidden node requirements.
   * Returns information about hidden layer sizes without modifying the network.
   * @param network The network to check
   * @param multiplierOverride Optional fixed multiplier for deterministic tests
   * @returns Object containing information about hidden layer compliance
   */
  checkHiddenSizes(network: Network, multiplierOverride?: number): { 
    compliant: boolean; 
    minRequired: number;
    hiddenLayerSizes: number[];
  } {
    const minHidden = this.getMinimumHiddenSize(multiplierOverride);
    const result = {
      compliant: true,
      minRequired: minHidden,
      hiddenLayerSizes: [] as number[]
    };
    
    // Check networks with explicit layers
    if (network.layers && network.layers.length >= 3) {
      // Go through hidden layers (skip input layer [0] and output layer [length-1])
      for (let i = 1; i < network.layers.length - 1; i++) {
        const layer = network.layers[i];
        if (!layer || !Array.isArray(layer.nodes)) {
          result.hiddenLayerSizes.push(0);
          result.compliant = false;
          continue;
        }
        
        const layerSize = layer.nodes.length;
        result.hiddenLayerSizes.push(layerSize);
        
        if (layerSize < minHidden) {
          result.compliant = false;
        }
      }
    } else {
      // Flat/legacy network: check total hidden node count
      const hiddenCount = network.nodes.filter(n => n.type === 'hidden').length;
      result.hiddenLayerSizes.push(hiddenCount);
      
      if (hiddenCount < minHidden) {
        result.compliant = false;
      }
    }
    
    return result;
  }

  /**
   * Ensures that the network has at least min(input, output) + 1 hidden nodes in each hidden layer.
   * This prevents bottlenecks in networks where hidden layers might be too small.
   * For layered networks: Ensures each hidden layer has at least the minimum size.
   * For non-layered networks: Reorganizes into proper layers with the minimum size.
   * @param network The network to check and modify
   * @param multiplierOverride Optional fixed multiplier for deterministic tests
   */
  private ensureMinHiddenNodes(network: Network, multiplierOverride?: number) {
    const maxNodes = this.options.maxNodes || Infinity;
    const minHidden = Math.min(this.getMinimumHiddenSize(multiplierOverride), maxNodes - network.nodes.filter(n => n.type !== 'hidden').length);

    const inputNodes = network.nodes.filter(n => n.type === 'input');
    const outputNodes = network.nodes.filter(n => n.type === 'output');
    let hiddenNodes = network.nodes.filter(n => n.type === 'hidden');

    if (inputNodes.length === 0 || outputNodes.length === 0) {
      console.warn('Network is missing input or output nodes. Cannot ensure minimum hidden nodes.');
      return;
    }

    // Only add hidden nodes if needed, do not disconnect/reconnect existing ones
    const existingCount = hiddenNodes.length;
    for (let i = existingCount; i < minHidden && network.nodes.length < maxNodes; i++) {
      const NodeClass = require('./architecture/node').default;
      const newNode = new NodeClass('hidden');
      network.nodes.push(newNode);
      hiddenNodes.push(newNode);
    }

    // Ensure each hidden node has at least one input and one output connection
    for (const hiddenNode of hiddenNodes) {
      // At least one input connection (from input or another hidden)
      if (hiddenNode.connections.in.length === 0) {
        const candidates = inputNodes.concat(hiddenNodes.filter(n => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const source = candidates[Math.floor(rng() * candidates.length)];
          try { network.connect(source, hiddenNode); } catch {}
        }
      }
      // At least one output connection (to output or another hidden)
      if (hiddenNode.connections.out.length === 0) {
        const candidates = outputNodes.concat(hiddenNodes.filter(n => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const target = candidates[Math.floor(rng() * candidates.length)];
          try { network.connect(hiddenNode, target); } catch {}
        }
      }
    }

    // Ensure network.connections is consistent with per-node connections after all changes
    Network.rebuildConnections(network);
  }

  // Helper method to check if a connection exists between two nodes
  private hasConnectionBetween(network: Network, from: NodeType, to: NodeType): boolean {
    return network.connections.some(conn => conn.from === from && conn.to === to);
  }

  /**
   * Ensures that all input nodes have at least one outgoing connection,
   * all output nodes have at least one incoming connection,
   * and all hidden nodes have at least one incoming and one outgoing connection.
   * This prevents dead ends and blind I/O neurons.
   * @param network The network to check and fix
   */
  private ensureNoDeadEnds(network: Network) {
    const inputNodes = network.nodes.filter(n => n.type === 'input');
    const outputNodes = network.nodes.filter(n => n.type === 'output');
    const hiddenNodes = network.nodes.filter(n => n.type === 'hidden');

    // Helper to check if a node has a connection in a direction
    const hasOutgoing = (node: any) => node.connections && node.connections.out && node.connections.out.length > 0;
    const hasIncoming = (node: any) => node.connections && node.connections.in && node.connections.in.length > 0;

    // 1. Ensure all input nodes have at least one outgoing connection
    for (const inputNode of inputNodes) {
      if (!hasOutgoing(inputNode)) {
        // Try to connect to a random hidden or output node
        const candidates = hiddenNodes.length > 0 ? hiddenNodes : outputNodes;
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const target = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(inputNode, target);
          } catch (e: any) {
            // Ignore duplicate connection errors
          }
        }
      }
    }

    // 2. Ensure all output nodes have at least one incoming connection
    for (const outputNode of outputNodes) {
      if (!hasIncoming(outputNode)) {
        // Try to connect from a random hidden or input node
        const candidates = hiddenNodes.length > 0 ? hiddenNodes : inputNodes;
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const source = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(source, outputNode);
          } catch (e: any) {
            // Ignore duplicate connection errors
          }
        }
      }
    }

    // 3. Ensure all hidden nodes have at least one incoming and one outgoing connection
    for (const hiddenNode of hiddenNodes) {
      if (!hasIncoming(hiddenNode)) {
        // Try to connect from input or another hidden node
        const candidates = inputNodes.concat(hiddenNodes.filter(n => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const source = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(source, hiddenNode);
          } catch (e: any) {
            // Ignore duplicate connection errors
          }
        }
      }
      if (!hasOutgoing(hiddenNode)) {
        // Try to connect to output or another hidden node
        const candidates = outputNodes.concat(hiddenNodes.filter(n => n !== hiddenNode));
        if (candidates.length > 0) {
          const rng = this._getRNG();
          const target = candidates[Math.floor(rng() * candidates.length)];
          try {
            network.connect(hiddenNode, target);
          } catch (e: any) {
            // Ignore duplicate connection errors
          }
        }
      }
    }
  }

  /**
   * Evaluates the fitness of the current population.
   * If `fitnessPopulation` is true, evaluates the entire population at once.
   * Otherwise, evaluates each genome individually.
   * @returns A promise that resolves when evaluation is complete.
   */
  async evaluate(): Promise<void> {
    if (this.options.fitnessPopulation) {
      if (this.options.clear) {
        this.population.forEach((genome) => genome.clear());
      }
      await this.fitness(this.population as any);
    } else {
      for (const genome of this.population) {
        if (this.options.clear) genome.clear();
        genome.score = await this.fitness(genome);
      }
    }
  }

  /**
   * Evolves the population by selecting, mutating, and breeding genomes.
   * Implements elitism, provenance, and crossover to create the next generation.
   * @returns The fittest network from the current generation.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
   */
  async evolve(): Promise<Network> {
    if (this.population[this.population.length - 1].score === undefined) {
      await this.evaluate();
    }
    this.sort();

    const fittest = Network.fromJSON(this.population[0].toJSON());
    fittest.score = this.population[0].score;

    const newPopulation: Network[] = [];

    // Elitism (clamped to available population)
    const elitismCount = Math.max(0, Math.min(this.options.elitism || 0, this.population.length));
    for (let i = 0; i < elitismCount; i++) {
      const elite = this.population[i];
      if (elite) newPopulation.push(elite);
    }

    // Provenance (clamp so total does not exceed desired popsize)
    const desiredPop = Math.max(0, this.options.popsize || 0);
    const remainingSlotsAfterElites = Math.max(0, desiredPop - newPopulation.length);
    const provenanceCount = Math.max(0, Math.min(this.options.provenance || 0, remainingSlotsAfterElites));
    for (let i = 0; i < provenanceCount; i++) {
      if (this.options.network) {
        newPopulation.push(Network.fromJSON(this.options.network.toJSON()));
      } else {
        newPopulation.push(new Network(this.input, this.output, { minHidden: this.options.minHidden }));
      }
    }

    // Breed the next individuals (fill up to desired popsize)
    const toBreed = Math.max(0, desiredPop - newPopulation.length);
    for (let i = 0; i < toBreed; i++) {
      newPopulation.push(this.getOffspring());
    }

    // Ensure minimum hidden nodes to avoid bottlenecks
    for (const genome of newPopulation) {
      if (!genome) continue;
      this.ensureMinHiddenNodes(genome);
      this.ensureNoDeadEnds(genome); // Ensure no dead ends or blind I/O
    }

    this.population = newPopulation; // Replace population instead of appending
    this.mutate();

    this.population.forEach((genome) => (genome.score = undefined));

    this.generation++;

    return fittest;
  }

  /**
   * Creates the initial population of networks.
   * If a base network is provided, clones it to create the population.
   * @param network - The base network to clone, or null to create new networks.
   */
  createPool(network: Network | null): void {
    this.population = [];
    for (let i = 0; i < (this.options.popsize || 50); i++) {
      const copy = network
        ? Network.fromJSON(network.toJSON())
        : new Network(this.input, this.output, { minHidden: this.options.minHidden });
      copy.score = undefined;
      this.ensureNoDeadEnds(copy); // Ensure no dead ends or blind I/O
      this.population.push(copy);
    }
  }

  /**
   * Generates an offspring by crossing over two parent networks.
   * Uses the crossover method described in the Instinct algorithm.
   * @returns A new network created from two parents.
   * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
   */
  getOffspring(): Network {
    const parent1 = this.getParent();
    const parent2 = this.getParent();
    const offspring = Network.crossOver(parent1, parent2, this.options.equal || false);
    // Ensure the offspring has the minimum required hidden nodes
    this.ensureMinHiddenNodes(offspring);
    this.ensureNoDeadEnds(offspring); // Ensure no dead ends or blind I/O
    return offspring;
  }

  /**
   * Selects a mutation method for a given genome based on constraints.
   * Ensures that the mutation respects the maximum nodes, connections, and gates.
   * @param genome - The genome to mutate.
   * @returns The selected mutation method or null if no valid method is available.
   */
  selectMutationMethod(genome: Network): any {
    const mutationMethod = this.options.mutation![
  Math.floor(this._getRNG()() * this.options.mutation!.length)
    ];

    if (
      mutationMethod === methods.mutation.ADD_NODE &&
      genome.nodes.length >= (this.options.maxNodes || Infinity)
    ) {
      return null;
    }

    if (
      mutationMethod === methods.mutation.ADD_CONN &&
      genome.connections.length >= (this.options.maxConns || Infinity)
    ) {
      return null;
    }

    if (
      mutationMethod === methods.mutation.ADD_GATE &&
      genome.gates.length >= (this.options.maxGates || Infinity)
    ) {
      return null;
    }

    if (
      !this.options.allowRecurrent &&
      (mutationMethod === methods.mutation.ADD_BACK_CONN ||
        mutationMethod === methods.mutation.ADD_SELF_CONN)
    ) {
      return null; // Skip recurrent mutations if not allowed
    }

    return mutationMethod;
  }

  /**
   * Applies mutations to the population based on the mutation rate and amount.
   * Each genome is mutated using the selected mutation methods.
   * Slightly increases the chance of ADD_CONN mutation for more connectivity.
   */
  mutate(): void {
    for (const genome of this.population) {
  if (this._getRNG()() <= (this.options.mutationRate || 0.7)) {
        for (let j = 0; j < (this.options.mutationAmount || 1); j++) {
          const mutationMethod = this.selectMutationMethod(genome);
          if (mutationMethod) {
            genome.mutate(mutationMethod);
            // Slightly increase the chance of ADD_CONN mutation for more connectivity
            if (this._getRNG()() < 0.5) {
              genome.mutate(methods.mutation.ADD_CONN);
            }
          }
          // If mutationMethod is null, do not call any mutation (including fallback)
        }
      }
    }
  }

  /**
   * Sorts the population in descending order of fitness scores.
   * Ensures that the fittest genomes are at the start of the population array.
   */
  sort(): void {
    this.population.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  }

  /**
   * Selects a parent genome for breeding based on the selection method.
   * Supports multiple selection strategies, including POWER, FITNESS_PROPORTIONATE, and TOURNAMENT.
   * @returns The selected parent genome.
   * @throws Error if tournament size exceeds population size.
   */
  getParent(): Network {
    const selection = this.options.selection;
    const selectionName = selection?.name;
    switch (selectionName) {
      case 'POWER':
        if (
          this.population[0]?.score !== undefined &&
          this.population[1]?.score !== undefined &&
          this.population[0].score < this.population[1].score
        ) {
          this.sort();
        }
        const index = Math.floor(
          Math.pow(this._getRNG()(), selection.power || 1) *
            this.population.length
        );
        return this.population[index];
      case 'FITNESS_PROPORTIONATE':
        let totalFitness = 0;
        let minimalFitness = 0;
        this.population.forEach((genome) => {
          minimalFitness = Math.min(minimalFitness, genome.score ?? 0);
          totalFitness += genome.score ?? 0;
        });
        minimalFitness = Math.abs(minimalFitness);
        totalFitness += minimalFitness * this.population.length;

  const random = this._getRNG()() * totalFitness;
        let value = 0;
        for (const genome of this.population) {
          value += (genome.score ?? 0) + minimalFitness;
          if (random < value) return genome;
        }
        return this.population[
          Math.floor(this._getRNG()() * this.population.length)
        ];
      case 'TOURNAMENT':
        if (selection.size > this.options.popsize!) {
          throw new Error('Tournament size must be less than population size.');
        }
        const tournament = [];
        for (let i = 0; i < selection.size; i++) {
          tournament.push(
            this.population[Math.floor(this._getRNG()() * this.population.length)]
          );
        }
        tournament.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
        for (let i = 0; i < tournament.length; i++) {
          if (
            this._getRNG()() < selection.probability ||
            i === tournament.length - 1
          ) {
            return tournament[i];
          }
        }
        break;
      default:
        // fallback for legacy or custom selection objects
        if (selection === methods.selection.POWER) {
          // ...repeat POWER logic...
          if (
            this.population[0]?.score !== undefined &&
            this.population[1]?.score !== undefined &&
            this.population[0].score < this.population[1].score
          ) {
            this.sort();
          }
          const index = Math.floor(
            Math.pow(this._getRNG()(), selection.power || 1) *
              this.population.length
          );
          return this.population[index];
        }
        if (selection === methods.selection.FITNESS_PROPORTIONATE) {
          // ...repeat FITNESS_PROPORTIONATE logic...
          let totalFitness = 0;
          let minimalFitness = 0;
          this.population.forEach((genome) => {
            minimalFitness = Math.min(minimalFitness, genome.score ?? 0);
            totalFitness += genome.score ?? 0;
          });
          minimalFitness = Math.abs(minimalFitness);
          totalFitness += minimalFitness * this.population.length;

          const random = this._getRNG()() * totalFitness;
          let value = 0;
          for (const genome of this.population) {
            value += (genome.score ?? 0) + minimalFitness;
            if (random < value) return genome;
          }
          return this.population[
            Math.floor(this._getRNG()() * this.population.length)
          ];
        }
        if (selection === methods.selection.TOURNAMENT) {
          // ...repeat TOURNAMENT logic...
          if (selection.size > this.options.popsize!) {
            throw new Error('Tournament size must be less than population size.');
          }
          const tournament = [];
          for (let i = 0; i < selection.size; i++) {
            tournament.push(
              this.population[Math.floor(this._getRNG()() * this.population.length)]
            );
          }
          tournament.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
          for (let i = 0; i < tournament.length; i++) {
            if (
              this._getRNG()() < selection.probability ||
              i === tournament.length - 1
            ) {
              return tournament[i];
            }
          }
        }
        break;
    }
    return this.population[0]; // Default fallback
  }

  /**
   * Retrieves the fittest genome from the population.
   * Ensures that the population is evaluated and sorted before returning the result.
   * @returns The fittest genome in the population.
   */
  getFittest(): Network {
    if (this.population[this.population.length - 1].score === undefined) {
      this.evaluate();
    }
    if (
      this.population[1] &&
      (this.population[0].score ?? 0) < (this.population[1].score ?? 0)
    ) {
      this.sort();
    }
    return this.population[0];
  }

  /**
   * Calculates the average fitness score of the population.
   * Ensures that the population is evaluated before calculating the average.
   * @returns The average fitness score of the population.
   */
  getAverage(): number {
    if (this.population[this.population.length - 1].score === undefined) {
      this.evaluate();
    }
    const totalScore = this.population.reduce(
      (sum, genome) => sum + (genome.score ?? 0),
      0
    );
    return totalScore / this.population.length;
  }

  /**
   * Exports the current population as an array of JSON objects.
   * Useful for saving the state of the population for later use.
   * @returns An array of JSON representations of the population.
   */
  export(): any[] {
    return this.population.map((genome) => genome.toJSON());
  }

  /**
   * Imports a population from an array of JSON objects.
   * Replaces the current population with the imported one.
   * @param json - An array of JSON objects representing the population.
   */
  import(json: any[]): void {
    this.population = json.map((genome) => Network.fromJSON(genome));
    this.options.popsize = this.population.length;
  }
}
