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
};

export default class Neat {
  input: number;
  output: number;
  fitness: (network: Network) => number;
  options: Options;

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
  }
}
