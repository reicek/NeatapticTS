import { Architect, Network, methods } from '../src/neataptic';

describe('Neat', () => {
  test('AND', async () => {
    jest.setTimeout(40000); // Adjusted timeout

    const trainingSet = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];

    const network = new Network(2, 1);
    const results = await network.evolve(trainingSet, {
      mutation: methods.mutation.FFW,
      equal: true,
      elitism: 10,
      mutationRate: 0.5,
      error: 0.03,
      threads: 1,
    });

    expect(results.error).toBeLessThan(0.03); // Use expect
  });

  test('XOR', async () => {
    jest.setTimeout(40000); // Adjusted timeout

    const trainingSet = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];

    const network = new Network(2, 1);
    const results = await network.evolve(trainingSet, {
      mutation: methods.mutation.FFW,
      equal: true,
      elitism: 10,
      mutationRate: 0.5,
      error: 0.03,
      threads: 1,
    });

    expect(results.error).toBeLessThan(0.03); // Use expect
  });

  test('XNOR', async () => {
    jest.setTimeout(60000); // Adjusted timeout

    const trainingSet = [
      { input: [0, 0], output: [1] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];

    const network = new Network(2, 1);
    const results = await network.evolve(trainingSet, {
      mutation: methods.mutation.FFW,
      equal: true,
      elitism: 10,
      mutationRate: 0.5,
      error: 0.03,
      threads: 1,
    });

    expect(results.error).toBeLessThan(0.03); // Use expect
  });
});
