import { crossover } from '../../src/methods/crossover';

describe('Crossover Methods', () => {
  describe('SINGLE_POINT', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(crossover.SINGLE_POINT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(crossover.SINGLE_POINT.name).toBe('SINGLE_POINT');
      });
    });
    describe('Scenario: Config property', () => {
      describe('Config presence', () => {
        it('should have config property', () => {
          // Assert
          expect(crossover.SINGLE_POINT.config).toBeDefined();
        });
      });
      describe('Config type', () => {
        it('should have config as an array', () => {
          // Assert
          expect(Array.isArray(crossover.SINGLE_POINT.config)).toBe(true);
        });
      });
      describe('Config length', () => {
        it('should have config array of length 1', () => {
          // Assert
          expect(crossover.SINGLE_POINT.config).toHaveLength(1);
        });
      });
      describe('Config value', () => {
        it('should have correct config value', () => {
          // Assert
          expect(crossover.SINGLE_POINT.config[0]).toBeCloseTo(0.4);
        });
        it('config value is greater than or equal to 0', () => {
          // Arrange
          const configVal = crossover.SINGLE_POINT.config[0];
          // Assert
          expect(configVal).toBeGreaterThanOrEqual(0);
        });
        it('config value is less than or equal to 1', () => {
          // Arrange
          const configVal = crossover.SINGLE_POINT.config[0];
          // Assert
          expect(configVal).toBeLessThanOrEqual(1);
        });
      });
      describe('Negative scenarios', () => {
        it('should not have config of length 0', () => {
          // Assert
          expect(crossover.SINGLE_POINT.config.length).not.toBe(0);
        });
        it('should not have config of length greater than 1', () => {
          // Assert
          expect(crossover.SINGLE_POINT.config.length).not.toBeGreaterThan(1);
        });
      });
    });
  });

  describe('TWO_POINT', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(crossover.TWO_POINT).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(crossover.TWO_POINT.name).toBe('TWO_POINT');
      });
    });
    describe('Scenario: Config property', () => {
      describe('Config presence', () => {
        it('should have config property', () => {
          // Assert
          expect(crossover.TWO_POINT.config).toBeDefined();
        });
      });
      describe('Config type', () => {
        it('should have config as an array', () => {
          // Assert
          expect(Array.isArray(crossover.TWO_POINT.config)).toBe(true);
        });
      });
      describe('Config length', () => {
        it('should have config array of length 2', () => {
          // Assert
          expect(crossover.TWO_POINT.config).toHaveLength(2);
        });
      });
      describe('Config values', () => {
        it('should have correct config value 0', () => {
          // Assert
          expect(crossover.TWO_POINT.config[0]).toBeCloseTo(0.4);
        });
        it('should have correct config value 1', () => {
          // Assert
          expect(crossover.TWO_POINT.config[1]).toBeCloseTo(0.9);
        });
        it('config value 0 is greater than or equal to 0', () => {
          // Arrange
          const configVal1 = crossover.TWO_POINT.config[0];
          // Assert
          expect(configVal1).toBeGreaterThanOrEqual(0);
        });
        it('config value 0 is less than or equal to 1', () => {
          // Arrange
          const configVal1 = crossover.TWO_POINT.config[0];
          // Assert
          expect(configVal1).toBeLessThanOrEqual(1);
        });
        it('config value 1 is greater than or equal to 0', () => {
          // Arrange
          const configVal2 = crossover.TWO_POINT.config[1];
          // Assert
          expect(configVal2).toBeGreaterThanOrEqual(0);
        });
        it('config value 1 is less than or equal to 1', () => {
          // Arrange
          const configVal2 = crossover.TWO_POINT.config[1];
          // Assert
          expect(configVal2).toBeLessThanOrEqual(1);
        });
        it('config value 0 is less than or equal to config value 1', () => {
          // Arrange
          const configVal1 = crossover.TWO_POINT.config[0];
          const configVal2 = crossover.TWO_POINT.config[1];
          // Assert
          expect(configVal1).toBeLessThanOrEqual(configVal2);
        });
      });
      describe('Negative scenarios', () => {
        it('should not have config of length 0', () => {
          // Assert
          expect(crossover.TWO_POINT.config.length).not.toBe(0);
        });
        it('should not have config of length greater than 2', () => {
          // Assert
          expect(crossover.TWO_POINT.config.length).not.toBeGreaterThan(2);
        });
      });
    });
  });

  describe('UNIFORM', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(crossover.UNIFORM).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(crossover.UNIFORM.name).toBe('UNIFORM');
      });
    });
    describe('Scenario: Config property', () => {
      it('should not have config property', () => {
        // Assert
        expect((crossover.UNIFORM as any).config).toBeUndefined();
      });
      it('should not have config property defined', () => {
        // Assert
        expect('config' in crossover.UNIFORM).toBe(false);
      });
    });
  });

  describe('AVERAGE', () => {
    describe('Scenario: Existence', () => {
      it('should exist', () => {
        // Assert
        expect(crossover.AVERAGE).toBeDefined();
      });
    });
    describe('Scenario: Name property', () => {
      it('should have correct name property', () => {
        // Assert
        expect(crossover.AVERAGE.name).toBe('AVERAGE');
      });
    });
    describe('Scenario: Config property', () => {
      it('should not have config property', () => {
        // Assert
        expect((crossover.AVERAGE as any).config).toBeUndefined();
      });
      it('should not have config property defined', () => {
        // Assert
        expect('config' in crossover.AVERAGE).toBe(false);
      });
    });
  });
});
