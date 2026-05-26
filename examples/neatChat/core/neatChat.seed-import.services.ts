import {
  Architect,
  fromParameterVector,
  toParameterVector,
  type ParameterVector,
} from '../../../src/neataptic.ts';
import {
  NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
  NEATCHAT_SPECIAL_TOKENS,
} from './neatChat.constants';
import { createNeatChatEpisodicMemoryBank } from './neatChat.memory.services';
import { buildNeatChatVocabulary } from './neatChat.session.services';
import { exportNeatChatSessionV2 } from './neatChat.snapshot.v2.services';
import { NeatChatSeedImportError } from './neatChat.seed-import.errors';
import type {
  NeatChatExternalSeedDescriptor,
  NeatChatExternalSeedLayerWeights,
  NeatChatSeedMetadata,
  NeatChatSupportedSeedFamily,
} from './neatChat.seed-import.types';
import type {
  NeatChatSession,
  NeatChatSessionSnapshotV2,
} from './neatChat.types';

type SeedImportNetwork = Parameters<typeof fromParameterVector>[0];

type ResolvedSeedDescriptor = {
  readonly family: NeatChatSupportedSeedFamily;
  readonly vocabSize: number;
  readonly hiddenSize: number;
  readonly layer: NeatChatExternalSeedLayerWeights;
  readonly linearWeight?: readonly (readonly number[])[];
  readonly linearBias?: readonly number[];
};

type ResolvedNetworkShape = {
  readonly inputCount: number;
  readonly outputCount: number;
  readonly hiddenSize: number;
  readonly biasCount: number;
  readonly weightStart: number;
};

const MIN_SUPPORTED_VOCAB_SIZE = 300;
const MAX_SUPPORTED_VOCAB_SIZE = 3000;
const MIN_SUPPORTED_HIDDEN_SIZE = 8;
const MAX_SUPPORTED_HIDDEN_SIZE = 128;
const GRU_GATE_COUNT = 3;
const LSTM_GATE_COUNT = 4;
const GRU_ROLE_COUNT = 6;
const LSTM_ROLE_COUNT = 5;
const ZERO_VALUE = 0;
const UNIT_VALUE = 1;
const EXTERNAL_PARAMETER_VECTOR_SOURCE = 'external-parameter-vector';
const GRU_RESET_GATE_INDEX = 0;
const GRU_UPDATE_GATE_INDEX = 1;
const GRU_NEW_MEMORY_GATE_INDEX = 2;
const LSTM_INPUT_GATE_INDEX = 0;
const LSTM_FORGET_GATE_INDEX = 1;
const LSTM_CELL_GATE_INDEX = 2;
const LSTM_OUTPUT_GATE_INDEX = 3;
const DISTILLATION_SUGGESTION =
  'Direct import supports only single-layer GRU/LSTM checkpoints that fit the documented one-hot recurrent subset. Use the supervised-distillation path when operators, layers, or dimensions fall outside that contract.';

/**
 * Validates whether an external recurrent descriptor stays inside the direct-import subset.
 *
 * The supported subset for direct parameter-vector conversion requires:
 * - `family`: `'gru'` or `'lstm'` (bidirectional and multi-layer are not supported)
 * - `vocabSize`: 300–3000 (must match the one-hot input and output width)
 * - `hiddenSize`: 8–128 (must fit within practical browser evaluation budgets)
 * - `layers`: exactly one recurrent layer in PyTorch gate-row order
 * - no attention heads, embedding matrices, layer normalization, positional encoding,
 *   or dropout operators that survive export
 *
 * Descriptors outside this subset should use the supervised-distillation path instead:
 * train a native `Architect.gru` or `Architect.lstm` builder network to match the
 * teacher checkpoint's token distribution on a representative corpus.
 *
 * @param descriptor - External recurrent seed descriptor to validate.
 * @throws {NeatChatSeedImportError} When the descriptor falls outside the direct-import subset.
 *   `code` is `'UNSUPPORTED_OPERATOR'` for unsupported families or layer counts, and
 *   `'DIMENSION_MISMATCH'` for out-of-range vocabulary or hidden sizes.
 *
 * @example
 * ```ts
 * import { validateNeatChatSeedFamily, NeatChatSeedImportError } from './index';
 *
 * // Passes silently for a supported single-layer GRU descriptor.
 * validateNeatChatSeedFamily({
 *   family: 'gru',
 *   vocabSize: 512,
 *   hiddenSize: 24,
 *   layers: [{ weightIh: [...], weightHh: [...], biasIh: [...], biasHh: [...] }],
 * });
 *
 * // Throws NeatChatSeedImportError for unsupported families.
 * try {
 *   validateNeatChatSeedFamily({ family: 'transformer', vocabSize: 512, hiddenSize: 24, layers: [] });
 * } catch (err) {
 *   if (err instanceof NeatChatSeedImportError) {
 *     console.log(err.code);                  // 'UNSUPPORTED_OPERATOR'
 *     console.log(err.distillationSuggestion); // guidance for the distillation path
 *   }
 * }
 * ```
 */
export function validateNeatChatSeedFamily(
  descriptor: NeatChatExternalSeedDescriptor,
): void {
  resolveExternalSeedDescriptor(descriptor, {
    requireSupportedRange: true,
  });
}

/**
 * Maps one compatible external recurrent descriptor into a native parameter vector.
 *
 * The conversion reads PyTorch `state_dict()` weight matrices in gate-row order and places
 * them deterministically into the NeatapticTS GRU or LSTM builder layout. Non-mappable
 * entries — structural one-to-one edges, peephole connections, and native gated paths with
 * no direct PyTorch equivalent — are pinned to deterministic neutral values (zero or
 * identity weight) so the result is always reproducible for the same descriptor.
 *
 * **GRU approximations**: the `h_t` output node uses sigmoid rather than the standard
 * passthrough blend; `previousOutput → output` gated paths carry learnable weights rather
 * than unit weights. These are NeatapticTS topology choices that differ from PyTorch GRU.
 *
 * **LSTM approximations**: NeatapticTS LSTM does not wire `outputBlock` back to gate groups;
 * there is no `weight_hh` equivalent. Only `weight_ih` and combined (`biasIh + biasHh`) gate
 * biases are directly mappable. Peephole connections (`memoryCell → gates`) are present in
 * the native topology but absent from standard PyTorch LSTM; they are set to zero.
 *
 * @param network - Native GRU or LSTM network whose parameter layout should receive the mapped values.
 * @param descriptor - External recurrent descriptor exported in PyTorch gate-row order.
 * @returns Parameter vector aligned with the native network's deterministic layout.
 * @throws {NeatChatSeedImportError} When the descriptor dimensions are incompatible with the native layout.
 *
 * @example
 * ```ts
 * import { Architect } from '../../../src/neataptic';
 * import { fromParameterVector } from '../../../src/neataptic';
 * import { mapExternalRecurrentWeightsToParameterVector } from './index';
 *
 * const network = Architect.gru(512, 24, 512);
 * const descriptor = {
 *   family: 'gru' as const,
 *   vocabSize: 512,
 *   hiddenSize: 24,
 *   layers: [{ weightIh, weightHh, biasIh, biasHh }],
 * };
 * const parameterVector = mapExternalRecurrentWeightsToParameterVector(network, descriptor);
 * fromParameterVector(network, parameterVector);
 * // The native GRU network now holds the mapped weights.
 * ```
 */
export function mapExternalRecurrentWeightsToParameterVector(
  network: SeedImportNetwork,
  descriptor: NeatChatExternalSeedDescriptor,
): ParameterVector {
  // Step 1: Validate the external descriptor and the native target layout.
  const resolvedDescriptor = resolveExternalSeedDescriptor(descriptor, {
    requireSupportedRange: false,
  });
  const baselineParameterVector = toParameterVector(network);
  const networkShape = resolveCompatibleNetworkShape(
    network,
    baselineParameterVector,
    resolvedDescriptor,
  );

  // Step 2: Fill a deterministic value buffer from the authoritative mapping recipe.
  const mappedValues = new Float64Array(baselineParameterVector.values.length);

  if (resolvedDescriptor.family === 'gru') {
    applyGruMapping(mappedValues, networkShape, resolvedDescriptor);
  } else {
    applyLstmMapping(mappedValues, networkShape, resolvedDescriptor);
  }

  // Step 3: Fold the mapped values back onto the native layout metadata.
  return {
    layout: baselineParameterVector.layout,
    values: mappedValues,
  };
}

/**
 * Builds an importable NEATchat v2 snapshot from one compatible external descriptor.
 *
 * This is the entry point for the full offline conversion flow:
 *
 * ```
 * external PyTorch checkpoint
 *   → NeatChatExternalSeedDescriptor JSON
 *   → native Architect.gru / Architect.lstm network
 *   → mapExternalRecurrentWeightsToParameterVector
 *   → NeatChatSessionSnapshotV2 with extensions.neatchat.seedMetadata
 * ```
 *
 * The produced snapshot can be loaded through the standard `importNeatChatSessionV2`
 * path and keeps full compatibility with checkpoint, worker-payload, and parameter-vector
 * contracts. Seed provenance is recorded in `snapshot.extensions.neatchat.seedMetadata`
 * with `conversionSource: 'external-parameter-vector'`.
 *
 * Placeholder retained terms are synthesized deterministically when the descriptor does
 * not carry vocabulary strings. The same descriptor therefore always produces the same
 * snapshot structure and parameter values.
 *
 * For descriptors outside the supported subset, build a native builder network of the
 * target dimensions and train it on a representative corpus using the teacher checkpoint's
 * token distribution. Save the result as a `NeatChatSessionSnapshotV2` with
 * `conversionSource: 'distillation'` in `extensions.neatchat.seedMetadata`.
 *
 * @param descriptor - External recurrent descriptor exported in PyTorch gate-row order.
 * @returns v2 session snapshot containing the mapped native seed network and seed metadata.
 * @throws {NeatChatSeedImportError} When the descriptor falls outside the supported direct-import subset.
 *
 * @example
 * ```ts
 * import {
 *   buildSeedSnapshotFromExternalWeights,
 *   importNeatChatSessionV2,
 * } from './index';
 *
 * const snapshot = buildSeedSnapshotFromExternalWeights({
 *   family: 'gru',
 *   vocabSize: 512,
 *   hiddenSize: 24,
 *   layers: [{ weightIh, weightHh, biasIh, biasHh }],
 * });
 *
 * // Seed provenance is recorded in the extension bag.
 * console.log(snapshot.extensions.neatchat.seedMetadata?.conversionSource);
 * // 'external-parameter-vector'
 *
 * // The snapshot is loadable through the standard session path.
 * const session = importNeatChatSessionV2(snapshot);
 * ```
 */
export function buildSeedSnapshotFromExternalWeights(
  descriptor: NeatChatExternalSeedDescriptor,
): NeatChatSessionSnapshotV2 {
  // Step 1: Validate the supported direct-import subset and build the native shell.
  const resolvedDescriptor = resolveExternalSeedDescriptor(descriptor, {
    requireSupportedRange: true,
  });
  const network = createNativeSeedNetwork(resolvedDescriptor);
  const mappedParameterVector = mapExternalRecurrentWeightsToParameterVector(
    network,
    descriptor,
  );

  // Step 2: Apply the mapped weights, then export a standard v2 NEATchat snapshot.
  fromParameterVector(network, mappedParameterVector);
  const seedSession = createSyntheticSeedSession(network, resolvedDescriptor);
  const snapshot = exportNeatChatSessionV2(seedSession);
  const seedMetadata = createSeedMetadata(resolvedDescriptor);

  // Step 3: Extend the v2 snapshot metadata without bypassing the standard snapshot owner.
  return {
    ...snapshot,
    extensions: {
      ...snapshot.extensions,
      neatchat: {
        ...snapshot.extensions.neatchat,
        seedMetadata,
      },
    },
  };
}

function resolveExternalSeedDescriptor(
  descriptor: NeatChatExternalSeedDescriptor,
  options: {
    readonly requireSupportedRange: boolean;
  },
): ResolvedSeedDescriptor {
  const family = resolveSupportedSeedFamily(descriptor.family);
  const vocabSize = resolvePositiveInteger(descriptor.vocabSize, 'vocabSize');
  const hiddenSize = resolvePositiveInteger(
    descriptor.hiddenSize,
    'hiddenSize',
  );

  if (options.requireSupportedRange) {
    assertSupportedDirectImportRange(vocabSize, hiddenSize);
  }

  const layer = resolveSingleLayerWeights(
    descriptor.layers,
    family,
    vocabSize,
    hiddenSize,
  );

  return {
    family,
    vocabSize,
    hiddenSize,
    layer,
    linearWeight: resolveOptionalLinearWeight(
      descriptor.linearWeight,
      vocabSize,
      hiddenSize,
    ),
    linearBias: resolveOptionalLinearBias(descriptor.linearBias, vocabSize),
  };
}

function resolveSupportedSeedFamily(
  family: string,
): NeatChatSupportedSeedFamily {
  if (family === 'gru' || family === 'lstm') {
    return family;
  }

  throw createUnsupportedOperatorError(
    `Unsupported NEATchat seed family: ${family}.`,
  );
}

function resolvePositiveInteger(value: number, fieldName: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new NeatChatSeedImportError(
      `NEATchat seed descriptor ${fieldName} must be a positive integer.`,
      'DIMENSION_MISMATCH',
    );
  }

  return value;
}

function assertSupportedDirectImportRange(
  vocabSize: number,
  hiddenSize: number,
): void {
  if (
    vocabSize < MIN_SUPPORTED_VOCAB_SIZE ||
    vocabSize > MAX_SUPPORTED_VOCAB_SIZE
  ) {
    throw new NeatChatSeedImportError(
      `NEATchat direct import supports vocabSize ${String(MIN_SUPPORTED_VOCAB_SIZE)}-${String(MAX_SUPPORTED_VOCAB_SIZE)}.`,
      'DIMENSION_MISMATCH',
    );
  }

  if (
    hiddenSize < MIN_SUPPORTED_HIDDEN_SIZE ||
    hiddenSize > MAX_SUPPORTED_HIDDEN_SIZE
  ) {
    throw new NeatChatSeedImportError(
      `NEATchat direct import supports hiddenSize ${String(MIN_SUPPORTED_HIDDEN_SIZE)}-${String(MAX_SUPPORTED_HIDDEN_SIZE)}.`,
      'DIMENSION_MISMATCH',
    );
  }
}

function resolveSingleLayerWeights(
  layers: readonly NeatChatExternalSeedLayerWeights[],
  family: NeatChatSupportedSeedFamily,
  vocabSize: number,
  hiddenSize: number,
): NeatChatExternalSeedLayerWeights {
  if (layers.length !== 1) {
    throw createUnsupportedOperatorError(
      'NEATchat direct import supports single-layer GRU and LSTM descriptors only.',
    );
  }

  const [layer] = layers;
  const gateCount = family === 'gru' ? GRU_GATE_COUNT : LSTM_GATE_COUNT;
  const expectedGateRowCount = gateCount * hiddenSize;

  assertMatrixShape(
    layer.weightIh,
    expectedGateRowCount,
    vocabSize,
    'layers[0].weightIh',
  );
  assertMatrixShape(
    layer.weightHh,
    expectedGateRowCount,
    hiddenSize,
    'layers[0].weightHh',
  );
  assertVectorShape(layer.biasIh, expectedGateRowCount, 'layers[0].biasIh');
  assertVectorShape(layer.biasHh, expectedGateRowCount, 'layers[0].biasHh');

  return layer;
}

function assertMatrixShape(
  matrix: readonly (readonly number[])[],
  expectedRowCount: number,
  expectedColumnCount: number,
  fieldName: string,
): void {
  if (matrix.length !== expectedRowCount) {
    throw new NeatChatSeedImportError(
      `NEATchat seed descriptor ${fieldName} must contain ${String(expectedRowCount)} rows.`,
      'DIMENSION_MISMATCH',
    );
  }

  matrix.forEach((row, rowIndex) => {
    if (row.length !== expectedColumnCount) {
      throw new NeatChatSeedImportError(
        `NEATchat seed descriptor ${fieldName}[${String(rowIndex)}] must contain ${String(expectedColumnCount)} columns.`,
        'DIMENSION_MISMATCH',
      );
    }

    row.forEach((value, columnIndex) => {
      if (!Number.isFinite(value)) {
        throw new NeatChatSeedImportError(
          `NEATchat seed descriptor ${fieldName}[${String(rowIndex)}][${String(columnIndex)}] must be finite.`,
          'DIMENSION_MISMATCH',
        );
      }
    });
  });
}

function assertVectorShape(
  vector: readonly number[],
  expectedLength: number,
  fieldName: string,
): void {
  if (vector.length !== expectedLength) {
    throw new NeatChatSeedImportError(
      `NEATchat seed descriptor ${fieldName} must contain ${String(expectedLength)} values.`,
      'DIMENSION_MISMATCH',
    );
  }

  vector.forEach((value, valueIndex) => {
    if (!Number.isFinite(value)) {
      throw new NeatChatSeedImportError(
        `NEATchat seed descriptor ${fieldName}[${String(valueIndex)}] must be finite.`,
        'DIMENSION_MISMATCH',
      );
    }
  });
}

function resolveOptionalLinearWeight(
  linearWeight: readonly (readonly number[])[] | undefined,
  vocabSize: number,
  hiddenSize: number,
): readonly (readonly number[])[] | undefined {
  if (linearWeight == null) {
    return undefined;
  }

  assertMatrixShape(linearWeight, vocabSize, hiddenSize, 'linearWeight');
  return linearWeight;
}

function resolveOptionalLinearBias(
  linearBias: readonly number[] | undefined,
  vocabSize: number,
): readonly number[] | undefined {
  if (linearBias == null) {
    return undefined;
  }

  assertVectorShape(linearBias, vocabSize, 'linearBias');
  return linearBias;
}

function resolveCompatibleNetworkShape(
  network: SeedImportNetwork,
  baselineParameterVector: ParameterVector,
  descriptor: ResolvedSeedDescriptor,
): ResolvedNetworkShape {
  const inputCount = network.inputNodeIds.length;
  const outputCount = network.outputNodeIds.length;
  const biasCount = baselineParameterVector.layout.entries.filter(
    (entry) => entry.kind === 'bias',
  ).length;
  const hiddenSize =
    descriptor.family === 'gru'
      ? resolveHiddenSizeFromBiasCount(
          biasCount,
          inputCount,
          outputCount,
          GRU_ROLE_COUNT,
          descriptor.family,
        )
      : resolveHiddenSizeFromBiasCount(
          biasCount,
          inputCount,
          outputCount,
          LSTM_ROLE_COUNT,
          descriptor.family,
        );
  const weightCount = baselineParameterVector.layout.entries.length - biasCount;
  const expectedWeightCount =
    descriptor.family === 'gru'
      ? 5 * hiddenSize * hiddenSize +
        2 * hiddenSize +
        3 * inputCount * hiddenSize +
        hiddenSize * outputCount
      : 4 * hiddenSize * hiddenSize +
        hiddenSize +
        4 * inputCount * hiddenSize +
        hiddenSize * outputCount +
        inputCount * outputCount;

  if (
    inputCount !== descriptor.vocabSize ||
    outputCount !== descriptor.vocabSize
  ) {
    throw new NeatChatSeedImportError(
      'NEATchat seed descriptor vocabSize must match the native network input and output width.',
      'DIMENSION_MISMATCH',
    );
  }

  if (hiddenSize !== descriptor.hiddenSize) {
    throw new NeatChatSeedImportError(
      'NEATchat seed descriptor hiddenSize must match the native recurrent block width.',
      'DIMENSION_MISMATCH',
    );
  }

  if (weightCount !== expectedWeightCount) {
    throw createUnsupportedOperatorError(
      `Native ${descriptor.family.toUpperCase()} layout does not match the Workstream 2 direct-import recipe.`,
    );
  }

  return {
    inputCount,
    outputCount,
    hiddenSize,
    biasCount,
    weightStart: biasCount,
  };
}

function resolveHiddenSizeFromBiasCount(
  biasCount: number,
  inputCount: number,
  outputCount: number,
  roleCount: number,
  family: NeatChatSupportedSeedFamily,
): number {
  const hiddenSize = (biasCount - inputCount - outputCount) / roleCount;

  if (!Number.isInteger(hiddenSize) || hiddenSize <= 0) {
    throw createUnsupportedOperatorError(
      `Native ${family.toUpperCase()} bias layout does not match the Workstream 2 mapping contract.`,
    );
  }

  return hiddenSize;
}

function applyGruMapping(
  mappedValues: Float64Array,
  networkShape: ResolvedNetworkShape,
  descriptor: ResolvedSeedDescriptor,
): void {
  const { inputCount, outputCount, hiddenSize, weightStart } = networkShape;
  const finalOutputBiasStart = inputCount;
  const updateGateBiasStart = inputCount + outputCount;
  const resetGateBiasStart = updateGateBiasStart + 2 * hiddenSize;
  const memoryCellBiasStart = resetGateBiasStart + hiddenSize;

  const blockAStart = weightStart;
  const blockBStart = blockAStart + hiddenSize * hiddenSize;
  const blockCStart = blockBStart + hiddenSize * hiddenSize;
  const blockDStart = blockCStart + hiddenSize;
  const blockEStart = blockDStart + hiddenSize * hiddenSize;
  const blockFStart = blockEStart + hiddenSize * hiddenSize;
  const blockGStart = blockFStart + hiddenSize * hiddenSize;
  const blockHStart = blockGStart + hiddenSize;
  const blockIStart = blockHStart + inputCount * hiddenSize;
  const blockJStart = blockIStart + inputCount * hiddenSize;
  const blockKStart = blockJStart + inputCount * hiddenSize;

  applyOptionalLinearBias(
    mappedValues,
    finalOutputBiasStart,
    outputCount,
    descriptor.linearBias,
  );

  fillCombinedBiasBlock(
    mappedValues,
    updateGateBiasStart,
    descriptor.layer,
    GRU_UPDATE_GATE_INDEX,
    hiddenSize,
  );
  fillCombinedBiasBlock(
    mappedValues,
    resetGateBiasStart,
    descriptor.layer,
    GRU_RESET_GATE_INDEX,
    hiddenSize,
  );
  fillCombinedBiasBlock(
    mappedValues,
    memoryCellBiasStart,
    descriptor.layer,
    GRU_NEW_MEMORY_GATE_INDEX,
    hiddenSize,
  );

  setOneToOneBlock(mappedValues, blockCStart, hiddenSize, UNIT_VALUE);
  setIdentityMatrixBlock(mappedValues, blockEStart, hiddenSize, UNIT_VALUE);
  setIdentityMatrixBlock(mappedValues, blockFStart, hiddenSize, UNIT_VALUE);
  setOneToOneBlock(mappedValues, blockGStart, hiddenSize, UNIT_VALUE);

  fillRecurrentGateWeightBlock(
    mappedValues,
    blockAStart,
    descriptor.layer.weightHh,
    GRU_UPDATE_GATE_INDEX,
    hiddenSize,
  );
  fillRecurrentGateWeightBlock(
    mappedValues,
    blockBStart,
    descriptor.layer.weightHh,
    GRU_RESET_GATE_INDEX,
    hiddenSize,
  );
  fillRecurrentGateWeightBlock(
    mappedValues,
    blockDStart,
    descriptor.layer.weightHh,
    GRU_NEW_MEMORY_GATE_INDEX,
    hiddenSize,
  );
  fillInputGateWeightBlock(
    mappedValues,
    blockHStart,
    descriptor.layer.weightIh,
    GRU_UPDATE_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  fillInputGateWeightBlock(
    mappedValues,
    blockIStart,
    descriptor.layer.weightIh,
    GRU_RESET_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  fillInputGateWeightBlock(
    mappedValues,
    blockJStart,
    descriptor.layer.weightIh,
    GRU_NEW_MEMORY_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  applyOptionalLinearWeight(
    mappedValues,
    blockKStart,
    descriptor.linearWeight,
    hiddenSize,
    outputCount,
  );
}

function applyLstmMapping(
  mappedValues: Float64Array,
  networkShape: ResolvedNetworkShape,
  descriptor: ResolvedSeedDescriptor,
): void {
  const { inputCount, outputCount, hiddenSize, weightStart } = networkShape;
  const finalOutputBiasStart = inputCount;
  const inputGateBiasStart = inputCount + outputCount;
  const forgetGateBiasStart = inputGateBiasStart + hiddenSize;
  const memoryCellBiasStart = forgetGateBiasStart + hiddenSize;
  const outputGateBiasStart = memoryCellBiasStart + hiddenSize;

  const blockAStart = weightStart;
  const blockBStart = blockAStart + hiddenSize * hiddenSize;
  const blockCStart = blockBStart + hiddenSize * hiddenSize;
  const blockDStart = blockCStart + hiddenSize * hiddenSize;
  const blockEStart = blockDStart + hiddenSize;
  const blockFStart = blockEStart + hiddenSize * hiddenSize;
  const blockGStart = blockFStart + inputCount * hiddenSize;
  const blockHStart = blockGStart + inputCount * hiddenSize;
  const blockIStart = blockHStart + inputCount * hiddenSize;
  const blockJStart = blockIStart + inputCount * hiddenSize;

  applyOptionalLinearBias(
    mappedValues,
    finalOutputBiasStart,
    outputCount,
    descriptor.linearBias,
  );

  fillCombinedBiasBlock(
    mappedValues,
    inputGateBiasStart,
    descriptor.layer,
    LSTM_INPUT_GATE_INDEX,
    hiddenSize,
  );
  fillCombinedBiasBlock(
    mappedValues,
    forgetGateBiasStart,
    descriptor.layer,
    LSTM_FORGET_GATE_INDEX,
    hiddenSize,
  );
  fillCombinedBiasBlock(
    mappedValues,
    memoryCellBiasStart,
    descriptor.layer,
    LSTM_CELL_GATE_INDEX,
    hiddenSize,
  );
  fillCombinedBiasBlock(
    mappedValues,
    outputGateBiasStart,
    descriptor.layer,
    LSTM_OUTPUT_GATE_INDEX,
    hiddenSize,
  );

  setOneToOneBlock(mappedValues, blockDStart, hiddenSize, UNIT_VALUE);
  setIdentityMatrixBlock(mappedValues, blockEStart, hiddenSize, UNIT_VALUE);

  fillInputGateWeightBlock(
    mappedValues,
    blockFStart,
    descriptor.layer.weightIh,
    LSTM_CELL_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  fillInputGateWeightBlock(
    mappedValues,
    blockGStart,
    descriptor.layer.weightIh,
    LSTM_INPUT_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  fillInputGateWeightBlock(
    mappedValues,
    blockHStart,
    descriptor.layer.weightIh,
    LSTM_OUTPUT_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  fillInputGateWeightBlock(
    mappedValues,
    blockIStart,
    descriptor.layer.weightIh,
    LSTM_FORGET_GATE_INDEX,
    hiddenSize,
    inputCount,
  );
  applyOptionalLinearWeight(
    mappedValues,
    blockJStart,
    descriptor.linearWeight,
    hiddenSize,
    outputCount,
  );
}

function fillCombinedBiasBlock(
  mappedValues: Float64Array,
  startIndex: number,
  layer: NeatChatExternalSeedLayerWeights,
  gateIndex: number,
  hiddenSize: number,
): void {
  for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex += 1) {
    const gateRowIndex = gateIndex * hiddenSize + hiddenIndex;
    mappedValues[startIndex + hiddenIndex] =
      layer.biasIh[gateRowIndex]! + layer.biasHh[gateRowIndex]!;
  }
}

function fillRecurrentGateWeightBlock(
  mappedValues: Float64Array,
  startIndex: number,
  weightMatrix: readonly (readonly number[])[],
  gateIndex: number,
  hiddenSize: number,
): void {
  for (let sourceIndex = 0; sourceIndex < hiddenSize; sourceIndex += 1) {
    for (let targetIndex = 0; targetIndex < hiddenSize; targetIndex += 1) {
      const gateRowIndex = gateIndex * hiddenSize + targetIndex;
      mappedValues[startIndex + sourceIndex * hiddenSize + targetIndex] =
        weightMatrix[gateRowIndex]![sourceIndex]!;
    }
  }
}

function fillInputGateWeightBlock(
  mappedValues: Float64Array,
  startIndex: number,
  weightMatrix: readonly (readonly number[])[],
  gateIndex: number,
  hiddenSize: number,
  inputCount: number,
): void {
  for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
    for (let targetIndex = 0; targetIndex < hiddenSize; targetIndex += 1) {
      const gateRowIndex = gateIndex * hiddenSize + targetIndex;
      mappedValues[startIndex + inputIndex * hiddenSize + targetIndex] =
        weightMatrix[gateRowIndex]![inputIndex]!;
    }
  }
}

function applyOptionalLinearWeight(
  mappedValues: Float64Array,
  startIndex: number,
  linearWeight: readonly (readonly number[])[] | undefined,
  hiddenSize: number,
  outputCount: number,
): void {
  if (linearWeight == null) {
    return;
  }

  for (let hiddenIndex = 0; hiddenIndex < hiddenSize; hiddenIndex += 1) {
    for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
      mappedValues[startIndex + hiddenIndex * outputCount + outputIndex] =
        linearWeight[outputIndex]![hiddenIndex]!;
    }
  }
}

function applyOptionalLinearBias(
  mappedValues: Float64Array,
  startIndex: number,
  outputCount: number,
  linearBias: readonly number[] | undefined,
): void {
  if (linearBias == null) {
    return;
  }

  for (let outputIndex = 0; outputIndex < outputCount; outputIndex += 1) {
    mappedValues[startIndex + outputIndex] = linearBias[outputIndex]!;
  }
}

function setOneToOneBlock(
  mappedValues: Float64Array,
  startIndex: number,
  count: number,
  value: number,
): void {
  for (let itemIndex = 0; itemIndex < count; itemIndex += 1) {
    mappedValues[startIndex + itemIndex] = value;
  }
}

function setIdentityMatrixBlock(
  mappedValues: Float64Array,
  startIndex: number,
  size: number,
  diagonalValue: number,
): void {
  for (let sourceIndex = 0; sourceIndex < size; sourceIndex += 1) {
    for (let targetIndex = 0; targetIndex < size; targetIndex += 1) {
      mappedValues[startIndex + sourceIndex * size + targetIndex] =
        sourceIndex === targetIndex ? diagonalValue : ZERO_VALUE;
    }
  }
}

function createNativeSeedNetwork(
  descriptor: ResolvedSeedDescriptor,
): SeedImportNetwork {
  if (descriptor.family === 'gru') {
    return Architect.gru(
      descriptor.vocabSize,
      descriptor.hiddenSize,
      descriptor.vocabSize,
    );
  }

  return Architect.lstm(
    descriptor.vocabSize,
    descriptor.hiddenSize,
    descriptor.vocabSize,
  );
}

function createSyntheticSeedSession(
  network: SeedImportNetwork,
  descriptor: ResolvedSeedDescriptor,
): NeatChatSession {
  const vocabulary = buildNeatChatVocabulary(
    createDeterministicRetainedTerms(descriptor.vocabSize),
  );

  return {
    vocabulary,
    network,
    exchanges: [],
    learnedExchangeCount: 0,
    learnedTokenPairCount: 0,
    seededTokenPairCount: 0,
    contextWindowTokenCount: NEATCHAT_DEFAULT_CONTEXT_WINDOW_TOKEN_COUNT,
    replayBufferExchangeCount: 0,
    pendingCandidates: [],
    candidateLog: [],
    memoryBank: createNeatChatEpisodicMemoryBank(),
    routingLog: [],
  };
}

function createDeterministicRetainedTerms(vocabSize: number): string[] {
  const retainedTermCount = vocabSize - NEATCHAT_SPECIAL_TOKENS.length;

  if (retainedTermCount <= 0) {
    throw new NeatChatSeedImportError(
      'NEATchat direct import requires vocabSize to exceed the reserved special-token count.',
      'DIMENSION_MISMATCH',
    );
  }

  return Array.from(
    { length: retainedTermCount },
    (_, termIndex) =>
      `external_seed_term_${String(termIndex).padStart(4, '0')}`,
  );
}

function createSeedMetadata(
  descriptor: ResolvedSeedDescriptor,
): NeatChatSeedMetadata {
  return {
    family: descriptor.family,
    vocabSize: descriptor.vocabSize,
    hiddenSize: descriptor.hiddenSize,
    conversionSource: EXTERNAL_PARAMETER_VECTOR_SOURCE,
  };
}

function createUnsupportedOperatorError(
  message: string,
): NeatChatSeedImportError {
  return new NeatChatSeedImportError(
    message,
    'UNSUPPORTED_OPERATOR',
    DISTILLATION_SUGGESTION,
  );
}
