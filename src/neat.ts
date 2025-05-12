import Network from './architecture/network';
import * as methods from './methods/methods';

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
};

export default class Neat {
  input: number;
  output: number;
  fitness: (network: Network) => number;
  options: Options;
  population: Network[] = [];
  generation: number = 0;

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
    this.options.mutationRate = this.options.mutationRate || 0.3;
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

    // Elitism
    for (let i = 0; i < (this.options.elitism || 0); i++) {
      newPopulation.push(this.population[i]);
    }

    // Provenance
    for (let i = 0; i < (this.options.provenance || 0); i++) {
      if (this.options.network) {
        newPopulation.push(Network.fromJSON(this.options.network.toJSON()));
      } else {
        newPopulation.push(new Network(this.input, this.output));
      }
    }

    // Breed the next individuals
    for (
      let i = 0;
      i <
      this.options.popsize! -
        (this.options.elitism || 0) -
        (this.options.provenance || 0);
      i++
    ) {
      newPopulation.push(this.getOffspring());
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
        : new Network(this.input, this.output);
      copy.score = undefined;
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
    return Network.crossOver(parent1, parent2, this.options.equal || false);
  }

  /**
   * Selects a mutation method for a given genome based on constraints.
   * Ensures that the mutation respects the maximum nodes, connections, and gates.
   * @param genome - The genome to mutate.
   * @returns The selected mutation method or null if no valid method is available.
   */
  selectMutationMethod(genome: Network): any {
    const mutationMethod = this.options.mutation![
      Math.floor(Math.random() * this.options.mutation!.length)
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
   */
  mutate(): void {
    for (const genome of this.population) {
      if (Math.random() <= (this.options.mutationRate || 0.3)) {
        for (let j = 0; j < (this.options.mutationAmount || 1); j++) {
          const mutationMethod = this.selectMutationMethod(genome);
          if (mutationMethod) genome.mutate(mutationMethod);
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
          Math.pow(Math.random(), selection.power || 1) *
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

        const random = Math.random() * totalFitness;
        let value = 0;
        for (const genome of this.population) {
          value += (genome.score ?? 0) + minimalFitness;
          if (random < value) return genome;
        }
        return this.population[
          Math.floor(Math.random() * this.population.length)
        ];
      case 'TOURNAMENT':
        if (selection.size > this.options.popsize!) {
          throw new Error('Tournament size must be less than population size.');
        }
        const tournament = [];
        for (let i = 0; i < selection.size; i++) {
          tournament.push(
            this.population[Math.floor(Math.random() * this.population.length)]
          );
        }
        tournament.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
        for (let i = 0; i < tournament.length; i++) {
          if (
            Math.random() < selection.probability ||
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
            Math.pow(Math.random(), selection.power || 1) *
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

          const random = Math.random() * totalFitness;
          let value = 0;
          for (const genome of this.population) {
            value += (genome.score ?? 0) + minimalFitness;
            if (random < value) return genome;
          }
          return this.population[
            Math.floor(Math.random() * this.population.length)
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
              this.population[Math.floor(Math.random() * this.population.length)]
            );
          }
          tournament.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
          for (let i = 0; i < tournament.length; i++) {
            if (
              Math.random() < selection.probability ||
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
