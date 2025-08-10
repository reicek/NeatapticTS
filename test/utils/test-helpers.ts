import { Network, Architect } from '../../src/neataptic';

/**
 * Creates a network with identical structure but different weight initialization
 */
export function cloneNetworkStructure(network: Network): Network {
  const inputs = network.nodes.filter((n) => n.type === 'input').length;
  const outputs = network.nodes.filter((n) => n.type === 'output').length;
  return new Network(inputs, outputs);
}

/**
 * Creates networks with identical weights for comparing training methods
 */
export function createIdenticalNetworks(
  count: number,
  inputs: number,
  hiddens: number,
  outputs: number
): Network[] {
  const networks: Network[] = [];

  // Create first network
  const template = Architect.perceptron(inputs, hiddens, outputs);
  networks.push(template);

  // Clone the network structure and weights
  for (let i = 1; i < count; i++) {
    const clone = new Network(inputs, outputs);

    // Add connections to match template
    for (let j = 0; j < template.connections.length; j++) {
      const conn = template.connections[j];
      const fromIndex = template.nodes.indexOf(conn.from);
      const toIndex = template.nodes.indexOf(conn.to);

      clone.connect(clone.nodes[fromIndex], clone.nodes[toIndex], conn.weight);
    }

    networks.push(clone);
  }

  return networks;
}

/**
 * Creates a dataset for a specific problem
 */
export function createDataset(
  type: 'XOR' | 'AND' | 'OR' | 'SIN',
  size: number = 4
): { input: number[]; output: number[] }[] {
  switch (type) {
    case 'XOR':
      return [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [0] },
      ];
    case 'AND':
      return [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [0] },
        { input: [1, 0], output: [0] },
        { input: [1, 1], output: [1] },
      ];
    case 'OR':
      return [
        { input: [0, 0], output: [0] },
        { input: [0, 1], output: [1] },
        { input: [1, 0], output: [1] },
        { input: [1, 1], output: [1] },
      ];
    case 'SIN':
      const dataset: { input: number[]; output: number[] }[] = [];
      for (let i = 0; i < size; i++) {
        const x = Math.random() * Math.PI * 2;
        dataset.push({
          input: [x],
          output: [Math.sin(x)],
        });
      }
      return dataset;
  }
}

/**
 * Mock console.warn during a test and assert warning was called
 */
export function expectWarning(fn: () => void, expectedWarning: string): void {
  const originalWarn = console.warn;
  const mockWarn = jest.fn();
  console.warn = mockWarn;

  try {
    fn();
  } finally {
    console.warn = originalWarn;
  }

  expect(mockWarn).toHaveBeenCalledWith(
    expect.stringContaining(expectedWarning)
  );
}

/**
 * Creates deep networks for testing
 */
export const createDeepNetworks = (): { original: Network; clone: Network } => {
  const original = Architect.perceptron(2, 4, 3, 1);

  // Create a clone with additional structure
  const clone = new Network(original.input, original.output);
  // Use mutate to add a node instead of nonexistent addNode method
  clone.mutate(require('../../src/methods/mutation').default.ADD_NODE);

  // Add test properties - this should work due to global interface declaration
  (original as any).testProp = 'original';
  (clone as any).testProp = 'clone';

  return { original, clone };
};

/**
 * Creates testing samples
 */
export const createTestingSamples = (
  count: number = 10
): { input: number[]; output: number[] }[] => {
  const samples: { input: number[]; output: number[] }[] = [];
  for (let i = 0; i < count; i++) {
    samples.push({
      input: [Math.random(), Math.random()],
      output: [Math.random()],
    });
  }
  return samples;
};

/**
 * Compares two networks
 */
export const compareNetworks = (
  net1: Network,
  net2: Network,
  matchingOutput?: boolean,
  expectIdentical: boolean = true
): void => {
  // Validate basic properties
  expect(net1.input).toEqual(net2.input);
  expect(net1.output).toEqual(net2.output);
  expect(net1.nodes.length).toEqual(
    expectIdentical ? net2.nodes.length : expect.any(Number)
  );

  // Compare structure if expected to be identical
  if (expectIdentical) {
    expect(net1.connections.length).toEqual(net2.connections.length);
    expect(net1.gates.length).toEqual(net2.gates.length);
    expect(net1.selfconns.length).toEqual(net2.selfconns.length);
  }

  // Test functionality if requested
  if (matchingOutput === true) {
    const testInput = [Math.random(), Math.random()];
    const output1 = net1.activate(testInput);
    const output2 = net2.activate(testInput);

    // Allow small numerical differences
    output1.forEach((val, i) => {
      expect(val).toBeCloseTo(output2[i], 5);
    });
  }
};
