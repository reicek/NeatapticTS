import Rate from './rate';

describe('Rate', () => {
  describe('fixed()', () => {
    describe('given a positive base rate', () => {
      describe('when the iteration is at the start of training', () => {
        it('returns the base rate unchanged', () => {
          // Arrange
          const schedule = Rate.fixed();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBe(baseRate);
        });
      });

      describe('when the iteration is far into training', () => {
        it('still returns the base rate unchanged', () => {
          // Arrange
          const schedule = Rate.fixed();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 100);

          // Assert
          expect(resolvedRate).toBe(baseRate);
        });
      });
    });

    describe('given a negative base rate', () => {
      describe('when the schedule is evaluated', () => {
        it('preserves the negative base rate', () => {
          // Arrange
          const schedule = Rate.fixed();
          const baseRate = -0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBe(baseRate);
        });
      });
    });
  });

  describe('step()', () => {
    describe('given custom decay settings', () => {
      describe('when the iteration is before the first decay step', () => {
        it('returns the base rate', () => {
          // Arrange
          const schedule = Rate.step(0.9, 10);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 9);

          // Assert
          expect(resolvedRate).toBe(baseRate);
        });
      });

      describe('when the iteration reaches the first decay step', () => {
        it('returns the first decayed rate', () => {
          // Arrange
          const schedule = Rate.step(0.9, 10);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 10);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.09);
        });
      });

      describe('when two decay steps have elapsed', () => {
        it('returns the second decayed rate', () => {
          // Arrange
          const schedule = Rate.step(0.9, 10);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 25);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.081);
        });
      });
    });

    describe('given the default decay settings', () => {
      describe('when the iteration reaches the default decay boundary', () => {
        it('uses the default factor and step size', () => {
          // Arrange
          const schedule = Rate.step();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 100);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.09);
        });
      });
    });

    describe('given a negative base rate', () => {
      describe('when the schedule is evaluated', () => {
        it('clamps the returned rate to zero', () => {
          // Arrange
          const schedule = Rate.step(0.9, 10);

          // Act
          const resolvedRate = schedule(-0.1, 0);

          // Assert
          expect(resolvedRate).toBe(0);
        });
      });
    });
  });

  describe('exp()', () => {
    describe('given a custom decay factor', () => {
      describe('when the iteration is zero', () => {
        it('returns the base rate', () => {
          // Arrange
          const schedule = Rate.exp(0.95);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBe(baseRate);
        });
      });

      describe('when the iteration advances', () => {
        it('applies exponential decay', () => {
          // Arrange
          const schedule = Rate.exp(0.95);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 10);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate * 0.95 ** 10);
        });
      });
    });

    describe('given the default decay factor', () => {
      describe('when the iteration advances', () => {
        it('uses the default exponential decay factor', () => {
          // Arrange
          const schedule = Rate.exp();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 10);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate * 0.999 ** 10);
        });
      });
    });
  });

  describe('inv()', () => {
    describe('given custom decay settings', () => {
      describe('when the iteration is zero', () => {
        it('returns the base rate', () => {
          // Arrange
          const schedule = Rate.inv(0.01, 1.5);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate, 5);
        });
      });

      describe('when the iteration increases', () => {
        it('returns the inverse-decayed rate', () => {
          // Arrange
          const schedule = Rate.inv(0.01, 1.5);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 10);

          // Assert
          expect(resolvedRate).toBeCloseTo(
            baseRate / (1 + 0.01 * 10 ** 1.5),
            5,
          );
        });
      });
    });

    describe('given the default decay settings', () => {
      describe('when the iteration increases', () => {
        it('uses the default inverse decay formula', () => {
          // Arrange
          const schedule = Rate.inv();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 10);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate / (1 + 0.001 * 10 ** 2), 5);
        });
      });
    });
  });

  describe('cosineAnnealing()', () => {
    describe('given custom cycle settings', () => {
      describe('when the cycle starts', () => {
        it('returns the base rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealing(100, 0.01);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate);
        });
      });

      describe('when the cycle is halfway complete', () => {
        it('returns the midpoint between base and minimum rates', () => {
          // Arrange
          const schedule = Rate.cosineAnnealing(100, 0.01);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 50);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.055);
        });
      });

      describe('when a new cycle begins', () => {
        it('restarts at the base rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealing(100, 0.01);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 100);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate);
        });
      });
    });

    describe('given the default minimum rate', () => {
      describe('when the cycle is halfway complete', () => {
        it('returns half the base rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealing();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 500);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.05);
        });
      });
    });
  });

  describe('cosineAnnealingWarmRestarts()', () => {
    describe('given a growing restart schedule', () => {
      describe('when the first cycle starts', () => {
        it('returns the base rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealingWarmRestarts(5, 0.001, 2);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate, 10);
        });
      });

      describe('when the first cycle is nearly complete', () => {
        it('returns a value close to the minimum rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealingWarmRestarts(5, 0.001, 2);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 4);

          // Assert
          expect(resolvedRate).toBeLessThanOrEqual(baseRate * 0.51);
        });
      });

      describe('when the second cycle starts', () => {
        it('restarts near the base rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealingWarmRestarts(5, 0.001, 2);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 5);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate, 2);
        });
      });

      describe('when the second cycle is in progress', () => {
        it('stays above the minimum rate', () => {
          // Arrange
          const schedule = Rate.cosineAnnealingWarmRestarts(5, 0.001, 2);
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 8);

          // Assert
          expect(resolvedRate).toBeGreaterThan(0.001);
        });
      });
    });

    describe('given no arguments', () => {
      describe('when the first cycle starts', () => {
        it('returns the base rate using all default cycle parameters', () => {
          // Arrange
          const schedule = Rate.cosineAnnealingWarmRestarts();
          const baseRate = 0.1;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert: cosine peak at iteration 0 equals baseRate
          expect(resolvedRate).toBeCloseTo(baseRate, 10);
        });
      });
    });
  });

  describe('linearWarmupDecay()', () => {
    describe('given an explicit warmup and end rate', () => {
      describe('when warmup begins', () => {
        it('starts at zero', () => {
          // Arrange
          const schedule = Rate.linearWarmupDecay(20, 5, 0.01);
          const baseRate = 0.2;

          // Act
          const resolvedRate = schedule(baseRate, 0);

          // Assert
          expect(resolvedRate).toBeCloseTo(0, 5);
        });
      });

      describe('when warmup ends', () => {
        it('reaches the base rate', () => {
          // Arrange
          const schedule = Rate.linearWarmupDecay(20, 5, 0.01);
          const baseRate = 0.2;

          // Act
          const resolvedRate = schedule(baseRate, 5);

          // Assert
          expect(resolvedRate).toBeCloseTo(baseRate, 5);
        });
      });

      describe('when the full schedule ends', () => {
        it('reaches the configured end rate', () => {
          // Arrange
          const schedule = Rate.linearWarmupDecay(20, 5, 0.01);
          const baseRate = 0.2;

          // Act
          const resolvedRate = schedule(baseRate, 20);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.01, 5);
        });
      });

      describe('when the schedule is in the decay phase', () => {
        it('linearly interpolates between base and end rates', () => {
          // Arrange
          const schedule = Rate.linearWarmupDecay(20, 5, 0.01);
          const baseRate = 0.2;

          // Act
          const resolvedRate = schedule(baseRate, 10);

          // Assert
          expect(resolvedRate).toBeCloseTo(0.1366666666666667, 10);
        });
      });
    });

    describe('given a non-positive total step count', () => {
      describe('when the schedule is created', () => {
        it('throws the total-step error', () => {
          // Arrange
          const createSchedule = () => Rate.linearWarmupDecay(0);

          // Act
          const thrownError = captureError(createSchedule);

          // Assert
          expect(thrownError?.message).toBe('totalSteps must be > 0');
        });
      });
    });
  });

  describe('reduceOnPlateau()', () => {
    describe('given patience, factor, and minimum rate settings', () => {
      describe('when monitoring begins', () => {
        it('starts at the base rate', () => {
          // Arrange
          const resolvedRates = runPlateauSchedule({
            baseRate: 0.05,
            errors: [0.5],
            options: { patience: 2, factor: 0.5, minRate: 0.005 },
          });

          // Act
          const resolvedRate = resolvedRates[0];

          // Assert
          expect(resolvedRate).toBeCloseTo(0.05, 10);
        });
      });

      describe('when patience has not been exhausted', () => {
        it('keeps the base rate unchanged', () => {
          // Arrange
          const resolvedRates = runPlateauSchedule({
            baseRate: 0.05,
            errors: [0.5, 0.45],
            options: { patience: 2, factor: 0.5, minRate: 0.005 },
          });

          // Act
          const resolvedRate = resolvedRates[1];

          // Assert
          expect(resolvedRate).toBe(0.05);
        });
      });

      describe('when the plateau exceeds patience', () => {
        it('reduces the learning rate', () => {
          // Arrange
          const resolvedRates = runPlateauSchedule({
            baseRate: 0.05,
            errors: [0.5, 0.45, 0.46, 0.47],
            options: { patience: 2, factor: 0.5, minRate: 0.005 },
          });

          // Act
          const resolvedRate = resolvedRates[3];

          // Assert
          expect(resolvedRate).toBe(0.025);
        });
      });

      describe('when a new best error is observed after a reduction', () => {
        it('keeps the reduced rate instead of cutting again immediately', () => {
          // Arrange
          const resolvedRates = runPlateauSchedule({
            baseRate: 0.05,
            errors: [0.5, 0.45, 0.46, 0.47, 0.4],
            options: { patience: 2, factor: 0.5, minRate: 0.005 },
          });

          // Act
          const resolvedRate = resolvedRates[4];

          // Assert
          expect(resolvedRate).toBe(0.025);
        });
      });
    });

    describe('given a minimum rate floor', () => {
      describe('when repeated plateaus would drop below the floor', () => {
        it('stops reducing at the configured minimum rate', () => {
          // Arrange
          const resolvedRates = runPlateauSchedule({
            baseRate: 0.05,
            errors: [0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
            options: { patience: 1, factor: 0.5, minRate: 0.01 },
          });

          // Act
          const resolvedRate = resolvedRates.at(-1);

          // Assert
          expect(resolvedRate).toBe(0.01);
        });
      });
    });

    describe('given verbose mode is enabled', () => {
      describe('when the schedule is evaluated', () => {
        it('preserves the current learning rate behavior', () => {
          // Arrange
          const resolvedRates = runPlateauSchedule({
            baseRate: 0.05,
            errors: [0.5],
            options: { verbose: true },
          });

          // Act
          const resolvedRate = resolvedRates[0];

          // Assert
          expect(resolvedRate).toBe(0.05);
        });
      });
    });
  });
});

interface PlateauScheduleInput {
  baseRate: number;
  errors: number[];
  options: {
    patience?: number;
    factor?: number;
    minDelta?: number;
    cooldown?: number;
    minRate?: number;
    verbose?: boolean;
  };
}

function runPlateauSchedule({
  baseRate,
  errors,
  options,
}: PlateauScheduleInput): number[] {
  const schedule = Rate.reduceOnPlateau(options);
  return errors.map((errorValue, iteration) =>
    schedule(baseRate, iteration, errorValue),
  );
}

function captureError(fn: () => unknown): Error | undefined {
  try {
    fn();
  } catch (error: unknown) {
    return error instanceof Error ? error : new Error(String(error));
  }
  return undefined;
}
