import { Architect, Layer, methods, type Network } from '../src/neataptic';
import {
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from './flappy_bird/constants/constants';

const ASCII_MAZE_INPUT_SIZE = 6;
const ASCII_MAZE_OUTPUT_SIZE = 4;
const ASCII_MAZE_MLP_HIDDEN_LAYER_SIZES = [6, 6] as const;
const ASCII_MAZE_RANDOM_SPARSE_HIDDEN_SIZE = 8;
const ASCII_MAZE_RANDOM_SPARSE_CONNECTIONS = 16;
const ASCII_MAZE_SEQUENCE_BLOCK_SIZES = [6] as const;
const ASCII_MAZE_NARX_INPUT_MEMORY = 1;
const ASCII_MAZE_NARX_OUTPUT_MEMORY = 1;
const FLAPPY_RANDOM_SPARSE_HIDDEN_SIZE = 8;
const FLAPPY_RANDOM_SPARSE_CONNECTIONS = 16;
const FLAPPY_SEQUENCE_BLOCK_SIZES = [16, 8] as const;
const FLAPPY_NARX_INPUT_MEMORY = 3;
const FLAPPY_NARX_OUTPUT_MEMORY = 3;

/** Stable demo identifiers supported by the shared example profile contract. */
export type ExampleDemoId = 'ascii-maze' | 'flappy-bird';

/** Public builder families exposed through the shared example profile contract. */
export type ExampleArchitectureBuilderFamily =
  | 'GRU'
  | 'LSTM'
  | 'MLP'
  | 'NARX'
  | 'RandomSparse';

/** Stable ids for the shared example architecture profiles. */
export type ExampleArchitectureProfileId =
  | 'gru'
  | 'lstm'
  | 'mlp'
  | 'narx'
  | 'random-sparse';

const EXAMPLE_ARCHITECTURE_PROFILE_ORDER: readonly ExampleArchitectureProfileId[] =
  ['mlp', 'random-sparse', 'narx', 'gru', 'lstm'] as const;

/** Demo approval flags keyed by the shared example demo ids. */
export type ExampleArchitectureApprovalMap = Record<ExampleDemoId, boolean>;

/** Demo-specific MLP configuration resolved from one shared profile id. */
export interface ExampleMlpArchitectureConfiguration {
  family: 'MLP';
  hiddenLayerSizes: number[];
  input: number;
  output: number;
}

/** Demo-specific sparse-graph configuration resolved from one shared profile id. */
export interface ExampleRandomSparseArchitectureConfiguration {
  backConnections: number;
  connections: number;
  family: 'RandomSparse';
  gates: number;
  hidden: number;
  input: number;
  output: number;
  selfConnections: number;
}

/** Demo-specific NARX configuration resolved from one shared profile id. */
export interface ExampleNarxArchitectureConfiguration {
  family: 'NARX';
  hiddenLayerSizes: number[];
  input: number;
  inputMemory: number;
  output: number;
  outputMemory: number;
}

/** Demo-specific recurrent block configuration resolved from one shared profile id. */
export interface ExampleSequenceArchitectureConfiguration {
  blockSizes: number[];
  family: 'GRU' | 'LSTM';
  input: number;
  inputToOutput?: boolean;
  output: number;
}

/** Family-specific size parameters exposed by the shared example profile contract. */
export type ExampleArchitectureProfileConfiguration =
  | ExampleMlpArchitectureConfiguration
  | ExampleNarxArchitectureConfiguration
  | ExampleRandomSparseArchitectureConfiguration
  | ExampleSequenceArchitectureConfiguration;

/**
 * Resolved shared example architecture profile.
 *
 * The profile id and family stay stable across demos, while the concrete size
 * parameters are resolved per demo so Flappy Bird and ASCII Maze can speak the
 * same architecture vocabulary without forcing identical input/output shapes.
 */
export interface ExampleArchitectureProfile {
  approvalByDemoId: ExampleArchitectureApprovalMap;
  approvedForDemo: boolean;
  configuration: ExampleArchitectureProfileConfiguration;
  demoId: ExampleDemoId;
  family: ExampleArchitectureBuilderFamily;
  id: ExampleArchitectureProfileId;
  label: string;
  recurrent: boolean;
  shortDescription: string;
}

type ExampleArchitectureProfileDefinition = {
  approvalByDemoId: ExampleArchitectureApprovalMap;
  family: ExampleArchitectureBuilderFamily;
  id: ExampleArchitectureProfileId;
  label: string;
  recurrent: boolean;
  resolveConfiguration: (
    demoId: ExampleDemoId,
  ) => ExampleArchitectureProfileConfiguration;
  shortDescription: string;
};

/** Default shared profile id for Flappy Bird until the browser selector lands. */
export const DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID: ExampleArchitectureProfileId =
  'random-sparse';

/** Default shared profile id for ASCII Maze when a builder-backed seed is requested. */
export const DEFAULT_ASCII_MAZE_ARCHITECTURE_PROFILE_ID: ExampleArchitectureProfileId =
  'random-sparse';

const EXAMPLE_ARCHITECTURE_PROFILE_DEFINITIONS: Record<
  ExampleArchitectureProfileId,
  ExampleArchitectureProfileDefinition
> = {
  'random-sparse': {
    id: 'random-sparse',
    family: 'RandomSparse',
    label: 'Sparse',
    shortDescription:
      'Sparse topology-search baseline that starts with fewer explicit edges.',
    recurrent: false,
    approvalByDemoId: {
      'ascii-maze': true,
      'flappy-bird': true,
    },
    resolveConfiguration: (demoId) =>
      demoId === 'flappy-bird'
        ? {
            family: 'RandomSparse',
            input: FLAPPY_NETWORK_INPUT_SIZE,
            hidden: FLAPPY_RANDOM_SPARSE_HIDDEN_SIZE,
            output: FLAPPY_NETWORK_OUTPUT_SIZE,
            connections: FLAPPY_RANDOM_SPARSE_CONNECTIONS,
            backConnections: 0,
            selfConnections: 0,
            gates: 0,
          }
        : {
            family: 'RandomSparse',
            input: ASCII_MAZE_INPUT_SIZE,
            hidden: ASCII_MAZE_RANDOM_SPARSE_HIDDEN_SIZE,
            output: ASCII_MAZE_OUTPUT_SIZE,
            connections: ASCII_MAZE_RANDOM_SPARSE_CONNECTIONS,
            backConnections: 0,
            selfConnections: 0,
            gates: 0,
          },
  },
  narx: {
    id: 'narx',
    family: 'NARX',
    label: 'NARX',
    shortDescription:
      'Delay-line state profile for short-horizon sequence and control tasks.',
    recurrent: true,
    approvalByDemoId: {
      'ascii-maze': true,
      'flappy-bird': true,
    },
    resolveConfiguration: (demoId) =>
      demoId === 'flappy-bird'
        ? {
            family: 'NARX',
            input: FLAPPY_NETWORK_INPUT_SIZE,
            hiddenLayerSizes: [...FLAPPY_SEQUENCE_BLOCK_SIZES],
            output: FLAPPY_NETWORK_OUTPUT_SIZE,
            inputMemory: FLAPPY_NARX_INPUT_MEMORY,
            outputMemory: FLAPPY_NARX_OUTPUT_MEMORY,
          }
        : {
            family: 'NARX',
            input: ASCII_MAZE_INPUT_SIZE,
            hiddenLayerSizes: [...ASCII_MAZE_SEQUENCE_BLOCK_SIZES],
            output: ASCII_MAZE_OUTPUT_SIZE,
            inputMemory: ASCII_MAZE_NARX_INPUT_MEMORY,
            outputMemory: ASCII_MAZE_NARX_OUTPUT_MEMORY,
          },
  },
  gru: {
    id: 'gru',
    family: 'GRU',
    label: 'GRU',
    shortDescription:
      'Pedagogical recurrent block profile for tasks that benefit from carried state.',
    recurrent: true,
    approvalByDemoId: {
      'ascii-maze': true,
      'flappy-bird': true,
    },
    resolveConfiguration: (demoId) =>
      demoId === 'flappy-bird'
        ? {
            family: 'GRU',
            input: FLAPPY_NETWORK_INPUT_SIZE,
            blockSizes: [...FLAPPY_SEQUENCE_BLOCK_SIZES],
            inputToOutput: true,
            output: FLAPPY_NETWORK_OUTPUT_SIZE,
          }
        : {
            family: 'GRU',
            input: ASCII_MAZE_INPUT_SIZE,
            blockSizes: [...ASCII_MAZE_SEQUENCE_BLOCK_SIZES],
            output: ASCII_MAZE_OUTPUT_SIZE,
          },
  },
  lstm: {
    id: 'lstm',
    family: 'LSTM',
    label: 'LSTM',
    shortDescription:
      'Pedagogical gated-memory profile for tasks that need longer carry-over state.',
    recurrent: true,
    approvalByDemoId: {
      'ascii-maze': true,
      'flappy-bird': true,
    },
    resolveConfiguration: (demoId) =>
      demoId === 'flappy-bird'
        ? {
            family: 'LSTM',
            input: FLAPPY_NETWORK_INPUT_SIZE,
            blockSizes: [...FLAPPY_SEQUENCE_BLOCK_SIZES],
            output: FLAPPY_NETWORK_OUTPUT_SIZE,
          }
        : {
            family: 'LSTM',
            input: ASCII_MAZE_INPUT_SIZE,
            blockSizes: [...ASCII_MAZE_SEQUENCE_BLOCK_SIZES],
            output: ASCII_MAZE_OUTPUT_SIZE,
          },
  },
  mlp: {
    id: 'mlp',
    family: 'MLP',
    label: 'MLP',
    shortDescription:
      'Dense feed-forward baseline used as the current approved reference profile.',
    recurrent: false,
    approvalByDemoId: {
      'ascii-maze': true,
      'flappy-bird': true,
    },
    resolveConfiguration: (demoId) =>
      demoId === 'flappy-bird'
        ? {
            family: 'MLP',
            input: FLAPPY_NETWORK_INPUT_SIZE,
            hiddenLayerSizes: [...FLAPPY_NETWORK_HIDDEN_LAYER_SIZES],
            output: FLAPPY_NETWORK_OUTPUT_SIZE,
          }
        : {
            family: 'MLP',
            input: ASCII_MAZE_INPUT_SIZE,
            hiddenLayerSizes: [...ASCII_MAZE_MLP_HIDDEN_LAYER_SIZES],
            output: ASCII_MAZE_OUTPUT_SIZE,
          },
  },
};

/**
 * Resolve one shared example architecture profile into demo-specific size parameters.
 *
 * @param demoId - Demo requesting the profile.
 * @param profileId - Stable shared profile id.
 * @returns Resolved profile metadata and demo-specific builder configuration.
 */
export function resolveExampleArchitectureProfile(
  demoId: ExampleDemoId,
  profileId: ExampleArchitectureProfileId,
): ExampleArchitectureProfile {
  const profileDefinition = EXAMPLE_ARCHITECTURE_PROFILE_DEFINITIONS[profileId];

  return {
    id: profileDefinition.id,
    family: profileDefinition.family,
    label: profileDefinition.label,
    shortDescription: profileDefinition.shortDescription,
    recurrent: profileDefinition.recurrent,
    approvalByDemoId: profileDefinition.approvalByDemoId,
    approvedForDemo: profileDefinition.approvalByDemoId[demoId],
    demoId,
    configuration: profileDefinition.resolveConfiguration(demoId),
  };
}

/**
 * List the currently approved shared example architecture profiles for one demo.
 *
 * @param demoId - Demo requesting the currently approved profiles.
 * @returns Resolved profiles whose approval flag is currently enabled for the demo.
 */
export function getApprovedExampleArchitectureProfiles(
  demoId: ExampleDemoId,
): ExampleArchitectureProfile[] {
  return EXAMPLE_ARCHITECTURE_PROFILE_ORDER.map((profileId) =>
    resolveExampleArchitectureProfile(demoId, profileId),
  ).filter((profile) => profile.approvedForDemo);
}

/**
 * Build one demo-ready seed network from the shared example profile contract.
 *
 * @param demoId - Demo requesting the seed network.
 * @param profileId - Stable shared profile id.
 * @returns Builder-backed seed network ready for demo or test use.
 */
export function buildExampleArchitectureProfileNetwork(
  demoId: ExampleDemoId,
  profileId: ExampleArchitectureProfileId,
): Network {
  const resolvedProfile = resolveExampleArchitectureProfile(demoId, profileId);
  const { configuration } = resolvedProfile;

  switch (configuration.family) {
    case 'MLP':
      return demoId === 'ascii-maze'
        ? buildSparseStagewiseMlpNetwork(
            configuration.input,
            configuration.hiddenLayerSizes,
            configuration.output,
          )
        : buildExactMlpNetwork(
            configuration.input,
            configuration.hiddenLayerSizes,
            configuration.output,
          );

    case 'RandomSparse':
      return Architect.randomSparse(
        configuration.input,
        configuration.hidden,
        configuration.output,
        {
          connections: configuration.connections,
          backConnections: configuration.backConnections,
          selfConnections: configuration.selfConnections,
          gates: configuration.gates,
        },
      );

    case 'NARX':
      return Architect.narx(
        configuration.input,
        configuration.hiddenLayerSizes,
        configuration.output,
        configuration.inputMemory,
        configuration.outputMemory,
      );

    case 'GRU':
      return Architect.gru(
        configuration.input,
        ...configuration.blockSizes,
        configuration.output,
        { inputToOutput: configuration.inputToOutput },
      );

    case 'LSTM':
      return Architect.lstm(
        configuration.input,
        ...configuration.blockSizes,
        configuration.output,
      );
  }
}

/**
 * Build an exact feed-forward MLP with caller-provided hidden layer sizes.
 *
 * Unlike `Architect.perceptron`, this helper does not enforce a minimum hidden
 * width, so educational presets can intentionally stay very small.
 */
function buildExactMlpNetwork(
  inputSize: number,
  hiddenLayerSizes: number[],
  outputSize: number,
): Network {
  const inputLayer = Layer.dense(inputSize, 'input');
  const outputLayer = Layer.dense(outputSize, 'output');
  const hiddenLayers = hiddenLayerSizes.map((hiddenLayerSize) =>
    Layer.dense(hiddenLayerSize),
  );
  const orderedLayers = [inputLayer, ...hiddenLayers, outputLayer];

  for (
    let layerIndex = 0;
    layerIndex < orderedLayers.length - 1;
    layerIndex += 1
  ) {
    orderedLayers[layerIndex].connect(
      orderedLayers[layerIndex + 1],
      methods.groupConnection.ALL_TO_ALL,
    );
  }

  const network = Architect.construct(orderedLayers);
  network.setTopologyIntent('feed-forward');
  return network;
}

/**
 * Build a sparse feed-forward MLP seed that keeps the staged hidden shape.
 *
 * The ASCII Maze demo uses this lighter backbone so NEAT can grow routing
 * structure instead of inheriting a fully dense mesh on generation zero.
 */
function buildSparseStagewiseMlpNetwork(
  inputSize: number,
  hiddenLayerSizes: number[],
  outputSize: number,
): Network {
  const inputLayer = Layer.dense(inputSize, 'input');
  const outputLayer = Layer.dense(outputSize, 'output');
  const hiddenLayers = hiddenLayerSizes.map((hiddenLayerSize) =>
    Layer.dense(hiddenLayerSize),
  );
  const orderedLayers = [inputLayer, ...hiddenLayers, outputLayer];

  for (
    let layerIndex = 0;
    layerIndex < orderedLayers.length - 1;
    layerIndex += 1
  ) {
    connectLayerPairWithSparseBackbone(
      orderedLayers[layerIndex],
      orderedLayers[layerIndex + 1],
    );
  }

  const network = Architect.construct(orderedLayers);
  network.setTopologyIntent('feed-forward');
  return network;
}

function connectLayerPairWithSparseBackbone(
  sourceLayer: Layer,
  targetLayer: Layer,
): void {
  if (sourceLayer.nodes.length === 0 || targetLayer.nodes.length === 0) {
    return;
  }

  const backboneConnectionCount = Math.max(
    sourceLayer.nodes.length,
    targetLayer.nodes.length,
  );

  for (
    let connectionIndex = 0;
    connectionIndex < backboneConnectionCount;
    connectionIndex += 1
  ) {
    const sourceNode =
      sourceLayer.nodes[connectionIndex % sourceLayer.nodes.length];
    const targetNode =
      targetLayer.nodes[connectionIndex % targetLayer.nodes.length];
    sourceNode.connect(targetNode);
  }
}
