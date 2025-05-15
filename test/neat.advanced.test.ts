import Neat from '../src/neat';
import Network from '../src/architecture/network';
import * as methods from '../src/methods/methods';
import Node from '../src/architecture/node';
import Connection from '../src/architecture/connection';

describe('Neat advanced coverage', () => {
  describe('constructor', () => {
    test('should set default options when not provided', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness);
      // Assert
      expect(neat.options.popsize).toBe(50);
    });
    test('should use provided options', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, { popsize: 10, elitism: 2 });
      // Assert
      expect(neat.options.popsize).toBe(10);
      expect(neat.options.elitism).toBe(2);
    });
    test('should set default popsize if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.popsize).toBe(50);
    });
    test('should set default equal if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.equal).toBe(false);
    });
    test('should set default clear if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.clear).toBe(false);
    });
    test('should set default elitism if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.elitism).toBe(0);
    });
    test('should set default provenance if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.provenance).toBe(0);
    });
    test('should set default mutationRate if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.mutationRate).toBe(0.7);
    });
    test('should set default mutationAmount if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.mutationAmount).toBe(1);
    });
    test('should set default fitnessPopulation if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.fitnessPopulation).toBe(false);
    });
    test('should set default selection if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.selection).toBeDefined();
    });
    test('should set default crossover if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.crossover).toBeDefined();
    });
    test('should set default mutation if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.mutation).toBeDefined();
    });
    test('should set default maxNodes if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.maxNodes).toBe(Infinity);
    });
    test('should set default maxConns if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.maxConns).toBe(Infinity);
    });
    test('should set default maxGates if options is empty', () => {
      // Arrange
      const fitness = jest.fn();
      // Act
      const neat = new Neat(2, 1, fitness, {});
      // Assert
      expect(neat.options.maxGates).toBe(Infinity);
    });
  });

  describe('createPool', () => {
    test('should create population with base network', () => {
      // Arrange
      const fitness = jest.fn();
      const base = new Network(2, 1);
      const neat = new Neat(2, 1, fitness, { popsize: 2, network: base });
      // Act
      neat.createPool(base);
      // Assert
      expect(neat.population.length).toBe(2);
    });
    test('should create population without base network', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      // Act
      neat.createPool(null);
      // Assert
      expect(neat.population.length).toBe(2);
    });
  });

  describe('selectMutationMethod', () => {
    test('should return null if ADD_NODE and maxNodes reached', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { mutation: [methods.mutation.ADD_NODE], maxNodes: 1 });
      const genome = new Network(2, 1);
      // Add a Node instance to reach the maxNodes constraint
      genome.nodes.push(new Node());
      // Act
      const result = neat.selectMutationMethod(genome);
      // Assert
      expect(result).toBeNull();
    });
    test('should return null if ADD_CONN and maxConns reached', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { mutation: [methods.mutation.ADD_CONN], maxConns: 1 });
      const genome = new Network(2, 1);
      // Add a Connection instance to reach the maxConns constraint
      genome.connections.push(new Connection(new Node(), new Node()));
      // Act
      const result = neat.selectMutationMethod(genome);
      // Assert
      expect(result).toBeNull();
    });
    test('should return null if ADD_GATE and maxGates reached', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { mutation: [methods.mutation.ADD_GATE], maxGates: 1 });
      const genome = new Network(2, 1);
      // Add a Connection instance to reach the maxGates constraint
      genome.gates.push(new Connection(new Node(), new Node()));
      // Act
      const result = neat.selectMutationMethod(genome);
      // Assert
      expect(result).toBeNull();
    });
    test('should return mutation method if constraints not reached', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { mutation: [methods.mutation.FFW] });
      const genome = new Network(2, 1);
      // Act
      const result = neat.selectMutationMethod(genome);
      // Assert
      expect(result).toBe(methods.mutation.FFW);
    });
  });

  describe('getParent', () => {
    test('should use POWER selection', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { selection: methods.selection.POWER });
      neat.population = [new Network(2, 1), new Network(2, 1)];
      neat.population[0].score = 2;
      neat.population[1].score = 1;
      // Act
      const parent = neat.getParent();
      // Assert
      expect(neat.population).toContain(parent);
    });
    test('should use FITNESS_PROPORTIONATE selection', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { selection: methods.selection.FITNESS_PROPORTIONATE });
      neat.population = [new Network(2, 1), new Network(2, 1)];
      neat.population[0].score = 2;
      neat.population[1].score = 1;
      // Act
      const parent = neat.getParent();
      // Assert
      expect(neat.population).toContain(parent);
    });
    test('should use TOURNAMENT selection', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { selection: { ...methods.selection.TOURNAMENT, size: 2, probability: 1 }, popsize: 2 });
      neat.population = [new Network(2, 1), new Network(2, 1)];
      neat.population[0].score = 2;
      neat.population[1].score = 1;
      // Act
      const parent = neat.getParent();
      // Assert
      expect(neat.population).toContain(parent);
    });
    test('should throw if tournament size > popsize', () => {
      // Arrange
      const fitness = jest.fn();
      // popsize = 2, selection.size = 3
      const neat = new Neat(2, 1, fitness, { selection: { ...methods.selection.TOURNAMENT, size: 3, probability: 1 }, popsize: 2 });
      neat.population = [new Network(2, 1), new Network(2, 1)];
      // Act & Assert
      expect(() => {
        neat.getParent();
      }).toThrow('Tournament size must be less than population size.');
    });
  });

  describe('evaluate', () => {
    test('should call fitness for each genome if fitnessPopulation is false', async () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = undefined;
      neat.population[1].score = undefined;
      // Act
      await neat.evaluate();
      // Assert
      expect(fitness).toHaveBeenCalledTimes(2);
    });
    test('should call fitness once for population if fitnessPopulation is true', async () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2, fitnessPopulation: true });
      // Act
      await neat.evaluate();
      // Assert
      expect(fitness).toHaveBeenCalledTimes(1);
    });
    test('should clear genomes if clear is true', async () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2, fitnessPopulation: true, clear: true });
      const clearSpy = jest.spyOn(neat.population[0], 'clear');
      // Act
      await neat.evaluate();
      // Assert
      expect(clearSpy).toHaveBeenCalled();
    });
  });

  describe('evolve', () => {
    test('should call evaluate if last genome score is undefined', async () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[1].score = undefined;
      const evalSpy = jest.spyOn(neat, 'evaluate');
      // Act
      await neat.evolve();
      // Assert
      expect(evalSpy).toHaveBeenCalled();
    });
    test('should increment generation', async () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = 1;
      neat.population[1].score = 2;
      // Act
      await neat.evolve();
      // Assert
      expect(neat.generation).toBe(1);
    });
  });

  describe('getOffspring', () => {
    test('should call Network.crossOver with two parents', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      const crossSpy = jest.spyOn(Network, 'crossOver').mockReturnValue(new Network(2, 1));
      // Act
      neat.getOffspring();
      // Assert
      expect(crossSpy).toHaveBeenCalled();
    });
  });

  describe('mutate', () => {
    test('should call mutate on genome if mutationMethod is valid', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 1, mutation: [methods.mutation.FFW], mutationRate: 1 });
      const mutateSpy = jest.spyOn(neat.population[0], 'mutate');
      // Act
      neat.mutate();
      // Assert
      expect(mutateSpy).toHaveBeenCalled();
    });
    test('should not call mutate if mutationMethod is null', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 1, mutation: [methods.mutation.ADD_NODE], mutationRate: 1, maxNodes: 0 });
      const mutateSpy = jest.spyOn(neat.population[0], 'mutate');
      // Force selectMutationMethod to always return null
      jest.spyOn(neat, 'selectMutationMethod').mockReturnValue(null);
      // Act
      neat.mutate();
      // Assert
      expect(mutateSpy).not.toHaveBeenCalled();
    });
  });

  describe('import/export', () => {
    test('should export and import population', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = 1;
      neat.population[1].score = 2;
      // Act
      const exported = neat.export();
      neat.import(exported);
      // Assert
      expect(neat.population.length).toBe(2);
    });
    test('should export empty population as empty array', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 0 });
      neat.population = [];
      // Act
      const exported = neat.export();
      // Assert
      expect(exported).toEqual([]);
    });
    test('should import empty array as empty population', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      // Act
      neat.import([]);
      // Assert
      expect(neat.population.length).toBe(0);
    });
  });

  describe('sort', () => {
    test('should sort population by score descending', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = 1;
      neat.population[1].score = 2;
      // Act
      neat.sort();
      // Assert
      expect(neat.population[0].score).toBe(2);
    });
    test('should handle empty population', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 0 });
      neat.population = [];
      // Act
      neat.sort();
      // Assert
      expect(neat.population).toEqual([]);
    });
  });

  describe('getFittest', () => {
    test('should return fittest genome', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = 1;
      neat.population[1].score = 2;
      // Act
      const fittest = neat.getFittest();
      // Assert
      expect(fittest.score).toBe(2);
    });
    test('should call evaluate if last genome score is undefined', () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[1].score = undefined;
      const evalSpy = jest.spyOn(neat, 'evaluate');
      // Act
      neat.getFittest();
      // Assert
      expect(evalSpy).toHaveBeenCalled();
    });
    test('should call sort if population[1] exists and population[0] is less fit', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = 1;
      neat.population[1].score = 2;
      const sortSpy = jest.spyOn(neat, 'sort');
      // Act
      neat.getFittest();
      // Assert
      expect(sortSpy).toHaveBeenCalled();
    });
  });

  describe('getAverage', () => {
    test('should return average score', () => {
      // Arrange
      const fitness = jest.fn();
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[0].score = 1;
      neat.population[1].score = 3;
      // Act
      const avg = neat.getAverage();
      // Assert
      expect(avg).toBe(2);
    });
    test('should call evaluate if last genome score is undefined', () => {
      // Arrange
      const fitness = jest.fn().mockReturnValue(1); // Spy
      const neat = new Neat(2, 1, fitness, { popsize: 2 });
      neat.population[1].score = undefined;
      const evalSpy = jest.spyOn(neat, 'evaluate');
      // Act
      neat.getAverage();
      // Assert
      expect(evalSpy).toHaveBeenCalled();
    });
  });
});
