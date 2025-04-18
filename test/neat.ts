/* Import */
import { assert } from 'chai';
import { Network, methods } from '../src/neataptic';
import mocha from 'mocha';

/*******************************************************************************************
                      Tests the effectiveness of evolution
*******************************************************************************************/

describe('Neat', function () {
  /**
   * Tests the evolution of a network to learn the AND gate.
   */
  it('AND', async function () {
    this.timeout(40000); // Use this.timeout

    // Training set for the AND gate
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

    assert.isBelow(results.error, 0.03, 'Error should be below 0.03');
  });

  /**
   * Tests the evolution of a network to learn the XOR gate.
   */
  it('XOR', async function () {
    this.timeout(40000); // Use this.timeout

    // Training set for the XOR gate
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

    assert.isBelow(results.error, 0.03, 'Error should be below 0.03');
  });

  /**
   * Tests the evolution of a network to learn the XNOR gate.
   */
  it('XNOR', async function () {
    this.timeout(60000); // Use this.timeout

    // Training set for the XNOR gate
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

    assert.isBelow(results.error, 0.03, 'Error should be below 0.03');
  });
});
