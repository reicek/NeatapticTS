import { Network } from '../../src/browser-entry.ts';

const HELLO_NETWORK_INPUT_VALUES = [0.25, 0.75] as const;
const HELLO_NETWORK_HIDDEN_LAYER_SIZES = [3] as const;
const HELLO_NETWORK_HIDDEN_BIASES = [-0.1, 0.2, 0.05] as const;
const HELLO_NETWORK_OUTPUT_BIAS = -0.15;
const HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS = [
  [0.8, -0.4, 0.2],
  [0.1, 0.6, -0.3],
] as const;
const HELLO_NETWORK_HIDDEN_TO_OUTPUT_WEIGHTS = [0.7, -0.5, 0.9] as const;

/** Public summary returned by the Hello Network starter example. */
export interface HelloNetworkExampleResult {
  architecture: {
    hiddenLayerSizes: number[];
    inputCount: number;
    outputCount: number;
    topologyIntent: string;
  };
  inputValues: number[];
  outputValues: number[];
}

/**
 * Runs the smallest starter example in the learning path.
 *
 * The example deliberately keeps one compact story: build a tiny layered
 * network, pin one deterministic parameter set so the walkthrough stays
 * reproducible, then run a single activation pass through the public `Network`
 * facade.
 *
 * @returns Structured summary of the architecture and one inference pass.
 *
 * @example
 * ```ts
 * import { runHelloNetworkExample } from './index';
 *
 * const summary = runHelloNetworkExample();
 * console.log(summary.outputValues[0]);
 * ```
 */
export function runHelloNetworkExample(): HelloNetworkExampleResult {
  const network = Network.createMLP(
    HELLO_NETWORK_INPUT_VALUES.length,
    [...HELLO_NETWORK_HIDDEN_LAYER_SIZES],
    1,
  );
  const inputValues = [...HELLO_NETWORK_INPUT_VALUES];

  // Step 1: Pin a small deterministic parameter set so every run tells the same story.
  configureDeterministicParameters(network);

  // Step 2: Run one inference pass through the public network facade.
  const outputValues = network.activate(inputValues).map(roundOutputValue);

  // Step 3: Return the tiny architecture summary that the starter path teaches first.
  return {
    architecture: {
      hiddenLayerSizes: network.describeArchitecture().hiddenLayerSizes,
      inputCount: network.inputNodeIds.length,
      outputCount: network.outputNodeIds.length,
      topologyIntent: network.getTopologyIntent(),
    },
    inputValues,
    outputValues,
  };

  /**
   * Applies one fixed parameter layout so the walkthrough remains reproducible.
   *
   * @param targetNetwork - Newly created feed-forward network to configure.
   * @returns Nothing.
   */
  function configureDeterministicParameters(targetNetwork: Network): void {
    const inputNodes = targetNetwork.nodes
      .filter((node) => node.type === 'input')
      .toSorted(
        (leftNode, rightNode) =>
          resolveNodeIndex(leftNode) - resolveNodeIndex(rightNode),
      );
    const hiddenNodes = targetNetwork.nodes
      .filter((node) => node.type === 'hidden')
      .toSorted(
        (leftNode, rightNode) =>
          resolveNodeIndex(leftNode) - resolveNodeIndex(rightNode),
      );
    const outputNode = targetNetwork.nodes.find(
      (node) => node.type === 'output',
    );

    if (inputNodes.length !== 2 || hiddenNodes.length !== 3) {
      throw new Error(
        'Hello Network example expected a 2 -> 3 -> 1 feed-forward shape.',
      );
    }

    if (!outputNode) {
      throw new Error('Hello Network example expected one output node.');
    }

    const sortedInputNodeIndices = inputNodes.map(resolveNodeIndex);
    const sortedHiddenNodeIds = hiddenNodes.map(resolveNodeIndex);
    const outputNodeId = resolveNodeIndex(outputNode);
    const weightByEdgeKey = new Map<string, number>([
      [
        createEdgeKey(sortedInputNodeIndices[0], sortedHiddenNodeIds[0]),
        HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS[0][0],
      ],
      [
        createEdgeKey(sortedInputNodeIndices[0], sortedHiddenNodeIds[1]),
        HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS[0][1],
      ],
      [
        createEdgeKey(sortedInputNodeIndices[0], sortedHiddenNodeIds[2]),
        HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS[0][2],
      ],
      [
        createEdgeKey(sortedInputNodeIndices[1], sortedHiddenNodeIds[0]),
        HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS[1][0],
      ],
      [
        createEdgeKey(sortedInputNodeIndices[1], sortedHiddenNodeIds[1]),
        HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS[1][1],
      ],
      [
        createEdgeKey(sortedInputNodeIndices[1], sortedHiddenNodeIds[2]),
        HELLO_NETWORK_INPUT_TO_HIDDEN_WEIGHTS[1][2],
      ],
      [
        createEdgeKey(sortedHiddenNodeIds[0], outputNodeId),
        HELLO_NETWORK_HIDDEN_TO_OUTPUT_WEIGHTS[0],
      ],
      [
        createEdgeKey(sortedHiddenNodeIds[1], outputNodeId),
        HELLO_NETWORK_HIDDEN_TO_OUTPUT_WEIGHTS[1],
      ],
      [
        createEdgeKey(sortedHiddenNodeIds[2], outputNodeId),
        HELLO_NETWORK_HIDDEN_TO_OUTPUT_WEIGHTS[2],
      ],
    ]);

    hiddenNodes.forEach((hiddenNode, hiddenNodeIndex) => {
      hiddenNode.bias = HELLO_NETWORK_HIDDEN_BIASES[hiddenNodeIndex];
    });
    outputNode.bias = HELLO_NETWORK_OUTPUT_BIAS;

    targetNetwork.connections.forEach((connection) => {
      const sourceNodeIndex = resolveNodeIndex(connection.from);
      const targetNodeIndex = resolveNodeIndex(connection.to);
      const deterministicWeight = weightByEdgeKey.get(
        createEdgeKey(sourceNodeIndex, targetNodeIndex),
      );

      if (deterministicWeight === undefined) {
        throw new Error(
          `Hello Network example found unexpected edge ${sourceNodeIndex}->${targetNodeIndex}.`,
        );
      }

      connection.weight = deterministicWeight;
    });
  }
}

/**
 * Formats the Hello Network summary as a short console-friendly block.
 *
 * @param exampleResult - Structured summary returned by `runHelloNetworkExample`.
 * @returns Readable multi-line text for one starter-example run.
 */
export function formatHelloNetworkExampleResult(
  exampleResult: HelloNetworkExampleResult,
): string {
  const architectureSignature = `${exampleResult.architecture.inputCount} -> [${exampleResult.architecture.hiddenLayerSizes.join(', ')}] -> ${exampleResult.architecture.outputCount}`;

  return [
    'Hello Network',
    `Architecture: ${architectureSignature} (${exampleResult.architecture.topologyIntent})`,
    `Input values: ${exampleResult.inputValues.join(', ')}`,
    `Output values: ${exampleResult.outputValues.join(', ')}`,
  ].join('\n');
}

/**
 * Rounds one activation value to a small readable precision.
 *
 * @param outputValue - Raw activation output from the network.
 * @returns Rounded activation value.
 */
function roundOutputValue(outputValue: number): number {
  return Number(outputValue.toFixed(6));
}

/**
 * Resolves one node index or throws when the example cannot trust the graph.
 *
 * @param nodeWithIndex - Node-like value carrying an optional index.
 * @returns Stable numeric node index.
 */
function resolveNodeIndex(nodeWithIndex: { index?: number }): number {
  if (typeof nodeWithIndex.index !== 'number') {
    throw new Error(
      'Hello Network example expected every node to expose an index.',
    );
  }

  return nodeWithIndex.index;
}

/**
 * Builds one stable edge key for the deterministic parameter map.
 *
 * @param sourceNodeIndex - Source node index.
 * @param targetNodeIndex - Target node index.
 * @returns Stable key for one directed edge.
 */
function createEdgeKey(
  sourceNodeIndex: number,
  targetNodeIndex: number,
): string {
  return `${sourceNodeIndex}->${targetNodeIndex}`;
}
