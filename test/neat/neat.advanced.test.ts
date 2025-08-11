import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import * as methods from '../../src/methods/methods';
import Node from '../../src/architecture/node';
import Connection from '../../src/architecture/connection';

describe('Neat advanced coverage', () => {
  describe('constructor', () => {
    describe('when no options are provided', () => {
      it('should set default popsize to 50', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness);
        // Assert
        expect(neat.options.popsize).toBe(50);
      });
    });
    describe('when options are provided', () => {
      describe('popsize is set', () => {
        it('should use provided popsize', () => {
          // Arrange
          const fitness = jest.fn();
          // Act
          const neat = new Neat(2, 1, fitness, { popsize: 10 });
          // Assert
          expect(neat.options.popsize).toBe(10);
        });
      });
      describe('elitism is set', () => {
        it('should use provided elitism', () => {
          // Arrange
          const fitness = jest.fn();
          // Act
          const neat = new Neat(2, 1, fitness, { elitism: 2 });
          // Assert
          expect(neat.options.elitism).toBe(2);
        });
      });
    });
    describe('when options is empty', () => {
      it('should set default popsize to 50', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.popsize).toBe(50);
      });
      it('should set default equal to false', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.equal).toBe(false);
      });
      it('should set default clear to false', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.clear).toBe(false);
      });
      it('should set default elitism to 0', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.elitism).toBe(0);
      });
      it('should set default provenance to 0', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.provenance).toBe(0);
      });
      it('should set default mutationRate to 0.7', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.mutationRate).toBe(0.7);
      });
      it('should set default mutationAmount to 1', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.mutationAmount).toBe(1);
      });
      it('should set default fitnessPopulation to false', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.fitnessPopulation).toBe(false);
      });
      it('should set default selection to be defined', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.selection).toBeDefined();
      });
      it('should set default crossover to be defined', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.crossover).toBeDefined();
      });
      it('should set default mutation to be defined', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.mutation).toBeDefined();
      });
      it('should set default maxNodes to Infinity', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.maxNodes).toBe(Infinity);
      });
      it('should set default maxConns to Infinity', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.maxConns).toBe(Infinity);
      });
      it('should set default maxGates to Infinity', () => {
        // Arrange
        const fitness = jest.fn();
        // Act
        const neat = new Neat(2, 1, fitness, {});
        // Assert
        expect(neat.options.maxGates).toBe(Infinity);
      });
    });
  });

  describe('createPool', () => {
    describe('when base network is provided', () => {
      it('should create population with base network', () => {
        // Arrange
        const fitness = jest.fn();
        const base = new Network(2, 1);
        const neat = new Neat(2, 1, fitness, { popsize: 2, network: base });
        // Act
        neat.createPool(base);
        // Assert
        expect(neat.population.length).toBe(2);
      });
    });
    describe('when base network is not provided', () => {
      it('should create population without base network', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, { popsize: 2 });
        // Act
        neat.createPool(null);
        // Assert
        expect(neat.population.length).toBe(2);
      });
    });
  });

  describe('selectMutationMethod', () => {
    describe('when ADD_NODE and maxNodes reached', () => {
      it('should return null', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          mutation: [methods.mutation.ADD_NODE],
          maxNodes: 1,
        });
        const genome = new Network(2, 1);
        genome.nodes.push(new Node());
        // Act
        const result = neat.selectMutationMethod(genome);
        // Assert
        expect(result).toBeNull();
      });
    });
    describe('when ADD_CONN and maxConns reached', () => {
      it('should return null', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          mutation: [methods.mutation.ADD_CONN],
          maxConns: 1,
        });
        const genome = new Network(2, 1);
        genome.connections.push(new Connection(new Node(), new Node()));
        // Act
        const result = neat.selectMutationMethod(genome);
        // Assert
        expect(result).toBeNull();
      });
    });
    describe('when ADD_GATE and maxGates reached', () => {
      it('should return null', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          mutation: [methods.mutation.ADD_GATE],
          maxGates: 1,
        });
        const genome = new Network(2, 1);
        genome.gates.push(new Connection(new Node(), new Node()));
        // Act
        const result = neat.selectMutationMethod(genome);
        // Assert
        expect(result).toBeNull();
      });
    });
    describe('when constraints not reached', () => {
      it('should return mutation method', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          mutation: [methods.mutation.FFW],
        });
        const genome = new Network(2, 1);
        // Act
        const result = neat.selectMutationMethod(genome);
        // Assert
        expect(result).toBe(methods.mutation.FFW);
      });
    });
  });

  describe('getParent', () => {
    describe('when selection is POWER', () => {
      it('should return a parent from the population', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          selection: methods.selection.POWER,
        });
        neat.population = [new Network(2, 1), new Network(2, 1)];
        neat.population[0].score = 2;
        neat.population[1].score = 1;
        // Act
        const parent = neat.getParent();
        // Assert
        expect(neat.population).toContain(parent);
      });
    });
    describe('when selection is FITNESS_PROPORTIONATE', () => {
      it('should return a parent from the population', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          selection: methods.selection.FITNESS_PROPORTIONATE,
        });
        neat.population = [new Network(2, 1), new Network(2, 1)];
        neat.population[0].score = 2;
        neat.population[1].score = 1;
        // Act
        const parent = neat.getParent();
        // Assert
        expect(neat.population).toContain(parent);
      });
    });
    describe('when selection is TOURNAMENT', () => {
      it('should return a parent from the population', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          selection: {
            ...methods.selection.TOURNAMENT,
            size: 2,
            probability: 1,
          },
          popsize: 2,
        });
        neat.population = [new Network(2, 1), new Network(2, 1)];
        neat.population[0].score = 2;
        neat.population[1].score = 1;
        // Act
        const parent = neat.getParent();
        // Assert
        expect(neat.population).toContain(parent);
      });
      it('should throw if tournament size is greater than population size', () => {
        // Arrange
        const fitness = jest.fn();
        // popsize = 2, selection.size = 3
        const neat = new Neat(2, 1, fitness, {
          selection: {
            ...methods.selection.TOURNAMENT,
            size: 3,
            probability: 1,
          },
          popsize: 2,
        });
        neat.population = [new Network(2, 1), new Network(2, 1)];
        // Act & Assert
        expect(() => {
          neat.getParent();
        }).toThrow('Tournament size must be less than population size.');
      });
    });
  });

  describe('evaluate', () => {
    describe('when fitnessPopulation is false', () => {
      it('should call fitness for each genome', async () => {
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
    });
    describe('when fitnessPopulation is true', () => {
      it('should call fitness once for population', async () => {
        // Arrange
        const fitness = jest.fn().mockReturnValue(1); // Spy
        const neat = new Neat(2, 1, fitness, {
          popsize: 2,
          fitnessPopulation: true,
        });
        // Act
        await neat.evaluate();
        // Assert
        expect(fitness).toHaveBeenCalledTimes(1);
      });
      it('should clear genomes if clear is true', async () => {
        // Arrange
        const fitness = jest.fn().mockReturnValue(1); // Spy
        const neat = new Neat(2, 1, fitness, {
          popsize: 2,
          fitnessPopulation: true,
          clear: true,
        });
        const clearSpy = jest.spyOn(neat.population[0], 'clear');
        // Act
        await neat.evaluate();
        // Assert
        expect(clearSpy).toHaveBeenCalled();
      });
    });
  });

  describe('evolve', () => {
    describe('when last genome score is undefined', () => {
      it('should call evaluate', async () => {
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
    });
    describe('when all genome scores are defined', () => {
      it('should increment generation', async () => {
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
  });

  describe('getOffspring', () => {
    describe('when called', () => {
      it('should call Network.crossOver with two parents', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, { popsize: 2 });
        const crossSpy = jest
          .spyOn(Network, 'crossOver')
          .mockReturnValue(new Network(2, 1));
        // Act
        neat.getOffspring();
        // Assert
        expect(crossSpy).toHaveBeenCalled();
      });
    });
  });

  describe('mutate', () => {
    describe('when mutationMethod is valid', () => {
      it('should call mutate on genome', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          popsize: 1,
          mutation: [methods.mutation.FFW],
          mutationRate: 1,
        });
        const mutateSpy = jest.spyOn(neat.population[0], 'mutate');
        // Act
        neat.mutate();
        // Assert
        expect(mutateSpy).toHaveBeenCalled();
      });
    });
    describe('when mutationMethod is null', () => {
      it('should not call mutate', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, {
          popsize: 1,
          mutation: [methods.mutation.ADD_NODE],
          mutationRate: 1,
          maxNodes: 0,
        });
        const mutateSpy = jest.spyOn(neat.population[0], 'mutate');
        // Force selectMutationMethod to always return null
        jest.spyOn(neat, 'selectMutationMethod').mockReturnValue(null);
        // Act
        neat.mutate();
        // Assert
        expect(mutateSpy).not.toHaveBeenCalled();
      });
    });
  });

  describe('import/export', () => {
    describe('when exporting and importing population', () => {
      it('should export and import population', () => {
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
    });
    describe('when exporting empty population', () => {
      it('should export empty population as empty array', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, { popsize: 0 });
        neat.population = [];
        // Act
        const exported = neat.export();
        // Assert
        expect(exported).toEqual([]);
      });
    });
    describe('when importing empty array', () => {
      it('should import empty array as empty population', () => {
        // Arrange
        const fitness = jest.fn();
        const neat = new Neat(2, 1, fitness, { popsize: 2 });
        // Act
        neat.import([]);
        // Assert
        expect(neat.population.length).toBe(0);
      });
    });
  });

  describe('sort', () => {
    describe('when population has scores', () => {
      it('should sort population by score descending', () => {
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
    });
    describe('when population is empty', () => {
      it('should handle empty population', () => {
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
  });

  describe('getFittest', () => {
    describe('when population has scores', () => {
      it('should return fittest genome', () => {
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
    });
    describe('when last genome score is undefined', () => {
      it('should call evaluate', () => {
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
    });
    describe('when population[1] exists and population[0] is less fit', () => {
      it('should call sort', () => {
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
  });

  describe('getAverage', () => {
    describe('when population has scores', () => {
      it('should return average score', () => {
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
    });
    describe('when last genome score is undefined', () => {
      it('should call evaluate', () => {
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
});
