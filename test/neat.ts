import { Architect, Network, methods } from '../src/neataptic';

// Test suite for the main Neataptic evolution capabilities.
describe('Neat', () => {
  // Test case for the logical AND function.
  test('AND', async () => {
    // Increase the timeout for this asynchronous test, as evolution can take time.
    jest.setTimeout(50000); // Adjusted timeout

    // Define the training set for the AND function.
    // Input arrays represent the two inputs, output array represents the expected result.
    const trainingSet = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];

    // Create a new network with 2 input neurons and 1 output neuron.
    const network = new Network(2, 1);
    // Evolve the network to learn the AND function using the provided training set and options.
    const results = await network.evolve(trainingSet, {
      mutation: methods.mutation.FFW, // Use feed-forward compatible mutations.
      equal: true, // Optimize for equality in fitness calculation (useful for classification tasks).
      elitism: 10, // Keep the top 10 best networks from one generation to the next.
      mutationRate: 0.5, // Set the probability of mutation.
      error: 0.01, // Target error threshold to stop evolution (tightened).
      threads: 1, // Use a single thread for evolution (can be increased for parallel processing).
    });

    // Assert that the final error after evolution is below the target threshold.
    expect(results.error).toBeGreaterThanOrEqual(0); // Error should be non-negative
    expect(results.error).toBeLessThan(0.01); // Use expect with tightened threshold
  });

  // Test case for the logical XOR function.
  test('XOR', async () => {
    // Increase the timeout for this asynchronous test.
    jest.setTimeout(50000); // Adjusted timeout

    // Define the training set for the XOR function.
    const trainingSet = [
      { input: [0, 0], output: [0] },
      { input: [0, 1], output: [1] },
      { input: [1, 0], output: [1] },
      { input: [1, 1], output: [0] },
    ];

    // Create a new network with 2 input neurons and 1 output neuron.
    const network = new Network(2, 1);
    // Evolve the network to learn the XOR function.
    const results = await network.evolve(trainingSet, {
      mutation: methods.mutation.FFW, // Use feed-forward compatible mutations.
      equal: true, // Optimize for equality in fitness calculation.
      elitism: 10, // Keep the top 10 best networks.
      mutationRate: 0.5, // Set the probability of mutation.
      error: 0.01, // Target error threshold (tightened).
      threads: 1, // Use a single thread.
    });

    // Assert that the final error is below the target threshold.
    expect(results.error).toBeGreaterThanOrEqual(0); // Error should be non-negative
    expect(results.error).toBeLessThan(0.01); // Use expect with tightened threshold
  });

  // Test case for the logical XNOR function.
  test('XNOR', async () => {
    // Increase the timeout significantly for this potentially harder problem.
    jest.setTimeout(70000); // Adjusted timeout

    // Define the training set for the XNOR function.
    const trainingSet = [
      { input: [0, 0], output: [1] },
      { input: [0, 1], output: [0] },
      { input: [1, 0], output: [0] },
      { input: [1, 1], output: [1] },
    ];

    // Create a new network with 2 input neurons and 1 output neuron.
    const network = new Network(2, 1);
    // Evolve the network to learn the XNOR function.
    const results = await network.evolve(trainingSet, {
      mutation: methods.mutation.FFW, // Use feed-forward compatible mutations.
      equal: true, // Optimize for equality in fitness calculation.
      elitism: 10, // Keep the top 10 best networks.
      mutationRate: 0.5, // Set the probability of mutation.
      error: 0.01, // Target error threshold (tightened).
      threads: 1, // Use a single thread.
    });

    // Assert that the final error is below the target threshold.
    expect(results.error).toBeGreaterThanOrEqual(0); // Error should be non-negative
    expect(results.error).toBeLessThan(0.01); // Use expect with tightened threshold
  });
});
