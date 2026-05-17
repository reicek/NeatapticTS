import { Buffer } from 'node:buffer';
import { spawnSync } from 'node:child_process';
import { Tensor } from 'onnxruntime-common';
import Network from '../../network';
import { exportToONNXBinary } from '../network.onnx';
import type { Conv2DMapping } from '../network.onnx';
import type {
  OnnxRuntimeParityExecutedFixtureDescriptor,
  OnnxRuntimeParityExecutedResult,
  OnnxRuntimeParityFixtureDescriptor,
  OnnxRuntimeParityLane,
  OnnxRuntimeParityRandomizedRunOptions,
  OnnxRuntimeParityRandomizedSampleResult,
  OnnxRuntimeParityResult,
} from './network.onnx.parity.types';

type NumericRange = {
  minimum: number;
  maximum: number;
};

type DeterministicParameterSeed = {
  weightStart: number;
  weightStep: number;
  biasStart: number;
  biasStep: number;
};

type RandomizedFixtureFactory = (
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
) => OnnxRuntimeParityExecutedFixtureDescriptor;

const MINIMUM_PHASE_9_RANDOMIZED_SAMPLE_COUNT = 1;
const MAXIMUM_PHASE_9_RANDOMIZED_SAMPLE_COUNT = 4;
const SEEDED_RANDOM_MULTIPLIER = 1_664_525;
const SEEDED_RANDOM_INCREMENT = 1_013_904_223;
const SEEDED_RANDOM_DIVISOR = 0x1_00000000;
const SEEDED_RANDOM_DECIMAL_PLACES = 6;
const RANDOMIZED_DENSE_INPUT_WIDTH_RANGE = { minimum: 2, maximum: 4 };
const RANDOMIZED_DENSE_HIDDEN_WIDTH_RANGE = { minimum: 1, maximum: 4 };
const RANDOMIZED_DENSE_INPUT_VALUE_RANGE = { minimum: -0.9, maximum: 0.9 };
const RANDOMIZED_CONV_INPUT_WIDTH_RANGE = { minimum: 3, maximum: 4 };
const RANDOMIZED_CONV_OUTPUT_CHANNEL_RANGE = { minimum: 1, maximum: 2 };
const RANDOMIZED_CONV_OUTPUT_WIDTH_RANGE = { minimum: 2, maximum: 3 };
const RANDOMIZED_CONV_INPUT_VALUE_RANGE = { minimum: -1, maximum: 1 };
const RANDOMIZED_DENSE_PARAMETER_SEED_BOUNDS = {
  weightStart: { minimum: -0.35, maximum: -0.05 },
  weightStep: { minimum: 0.02, maximum: 0.09 },
  biasStart: { minimum: -0.05, maximum: 0.05 },
  biasStep: { minimum: 0.005, maximum: 0.02 },
} as const;
const RANDOMIZED_CONV_PARAMETER_SEED_BOUNDS = {
  weightStart: { minimum: -0.25, maximum: -0.05 },
  weightStep: { minimum: 0.005, maximum: 0.02 },
  biasStart: { minimum: -0.02, maximum: 0.02 },
  biasStep: { minimum: 0.003, maximum: 0.01 },
} as const;
const RANDOMIZED_PARITY_FACTORY_BY_LANE: Record<
  OnnxRuntimeParityLane,
  RandomizedFixtureFactory
> = {
  'baseline-float32-dense': createRandomizedDenseFixture,
  'storage-fp16-dense': createRandomizedDenseFixture,
  'static-8bit-dense-qlinear': createRandomizedDenseFixture,
  'static-8bit-conv-qlinear': createRandomizedStatic8BitConvFixture,
  'dynamic-uint8-dense-guidance': createRandomizedDenseFixture,
};

type OnnxRuntimeParitySubprocessPayload = {
  binaryModelBase64: string;
  runtimeInputValues: number[];
  runtimeInputDimensions?: readonly number[];
};

type OnnxRuntimeParitySubprocessResult = {
  inputNames: string[];
  outputNames: string[];
  runtimeOutput: number[];
};

const RUNTIME_PARITY_SUBPROCESS_SCRIPT = String.raw`
const { readFileSync } = require('node:fs');
const { Tensor } = require('onnxruntime-common');
const bindingModule = require('onnxruntime-node/dist/binding.js');

const payload = JSON.parse(readFileSync(0, 'utf8'));

class RuntimeParityTensor {
  constructor(arg0, arg1, arg2) {
    return createRuntimeParityTensor(arg0, arg1, arg2);
  }
}

bindingModule.binding.initOrtOnce(2, RuntimeParityTensor, true);

const inferenceSession = new bindingModule.binding.InferenceSession();
const binaryModel = Buffer.from(payload.binaryModelBase64, 'base64');

try {
  inferenceSession.loadModel(
    binaryModel.buffer,
    binaryModel.byteOffset,
    binaryModel.byteLength,
    {},
  );

  const inputMetadata = inferenceSession.inputMetadata.map(resolveMetadataEntry);
  const outputMetadata = inferenceSession.outputMetadata.map(resolveMetadataEntry);
  const inputName = inputMetadata[0].name;
  const outputName = outputMetadata[0].name;
  const inputDimensions =
    payload.runtimeInputDimensions ??
    resolveRuntimeTensorDimensions(inputMetadata[0].shape);
  const outputDimensions = resolveRuntimeTensorDimensions(outputMetadata[0].shape);
  const outputElementCount = outputDimensions.reduce(
    (runningProduct, dimensionValue) => runningProduct * dimensionValue,
    1,
  );

  const runtimeFeeds = {
    [inputName]: new Tensor(Float32Array.from(payload.runtimeInputValues), inputDimensions),
  };
  const runtimeFetches = {
    [outputName]: new Tensor(new Float32Array(outputElementCount), outputDimensions),
  };
  const runtimeValues = inferenceSession.run(runtimeFeeds, runtimeFetches, {});
  const runtimeOutput = Array.from(
    runtimeValues[outputName].data,
    (value) => Number(value),
  );

  process.stdout.write(
    JSON.stringify({
      inputNames: [inputName],
      outputNames: [outputName],
      runtimeOutput,
    }),
  );
} finally {
  inferenceSession.dispose();
}

function resolveMetadataEntry(metadataEntry) {
  return {
    name: metadataEntry.name,
    shape: metadataEntry.shape.map((shapeDimension, shapeIndex) => {
      if (shapeDimension === -1) {
        return metadataEntry.symbolicDimensions[shapeIndex];
      }

      return shapeDimension;
    }),
  };
}

function resolveRuntimeTensorDimensions(runtimeShape) {
  return runtimeShape.map((shapeDimension) => {
    if (typeof shapeDimension === 'number' && shapeDimension > 0) {
      return shapeDimension;
    }

    return 1;
  });
}

function createRuntimeParityTensor(arg0, arg1, arg2) {
  if (arg0 === 'float32') {
    if (isArrayBufferLike(arg1)) {
      return new Tensor('float32', new Float32Array(arg1), arg2);
    }

    if (ArrayBuffer.isView(arg1) && !(arg1 instanceof Float32Array)) {
      return new Tensor(
        'float32',
        new Float32Array(
          arg1.buffer,
          arg1.byteOffset,
          arg1.byteLength / Float32Array.BYTES_PER_ELEMENT,
        ),
        arg2,
      );
    }
  }

  return new Tensor(arg0, arg1, arg2);
}

function isArrayBufferLike(tensorData) {
  return (
    tensorData instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' &&
      tensorData instanceof SharedArrayBuffer) ||
    Object.prototype.toString.call(tensorData) === '[object ArrayBuffer]' ||
    Object.prototype.toString.call(tensorData) === '[object SharedArrayBuffer]'
  );
}
`;

const PHASE_9_RUNTIME_PARITY_INVENTORY = [
  createBaselineFloat32DenseFixture(),
  createStorageFp16DenseFixture(),
  createStatic8BitDenseQlinearFixture(),
  createStatic8BitConvQlinearFixture(),
  createDynamicUint8DenseGuidanceFixture(),
] as const satisfies readonly OnnxRuntimeParityFixtureDescriptor[];

/**
 * Return the frozen Phase 9A runtime-parity inventory.
 *
 * This inventory is the narrow stop line between the closed Phase 8 binary
 * acceptance lane and the later Phase 9B golden parity tranche. Keeping the
 * list centralized prevents new runtime lanes from slipping in through ad hoc
 * tests.
 *
 * @returns The current Phase 9A fixture inventory in deterministic order.
 */
export function getPhase9RuntimeParityInventory():
  readonly OnnxRuntimeParityFixtureDescriptor[] {
  return PHASE_9_RUNTIME_PARITY_INVENTORY;
}

/**
 * Resolve one named Phase 9 runtime-parity fixture.
 *
 * @param fixtureId Named fixture identifier from the frozen Phase 9A inventory.
 * @returns The matching deterministic fixture descriptor.
 */
export function findPhase9RuntimeParityFixture(
  fixtureId: OnnxRuntimeParityFixtureDescriptor['id'],
): OnnxRuntimeParityFixtureDescriptor {
  const fixtureDescriptor = PHASE_9_RUNTIME_PARITY_INVENTORY.find(
    (inventoryEntry) => inventoryEntry.id === fixtureId,
  );

  if (!fixtureDescriptor) {
    throw new Error(`Unknown ONNX runtime parity fixture: ${fixtureId}`);
  }

  return fixtureDescriptor;
}

/**
 * Export, execute, and compare one Phase 9 runtime-parity fixture.
 *
 * The harness always starts from binary `.onnx` bytes so Phase 9 stays
 * binary-first rather than drifting back to the JSON-first roundtrip lane.
 * Skipped fixtures are returned with their explicit reason instead of being
 * executed implicitly.
 *
 * @param fixtureDescriptor Deterministic runtime-parity fixture descriptor.
 * @returns Either a skipped result or an executed comparison packet.
 */
export async function runOnnxRuntimeParityFixture(
  fixtureDescriptor: OnnxRuntimeParityExecutedFixtureDescriptor,
): Promise<OnnxRuntimeParityExecutedResult>;
export async function runOnnxRuntimeParityFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
): Promise<OnnxRuntimeParityResult>;
export async function runOnnxRuntimeParityFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
): Promise<OnnxRuntimeParityResult> {
  if (fixtureDescriptor.executionMode === 'skip') {
    return {
      fixture: fixtureDescriptor,
      skipped: true,
      skipReason: fixtureDescriptor.skipReason!,
    };
  }

  const sourceNetwork = fixtureDescriptor.createNetwork();
  const binaryModel = exportToONNXBinary(
    sourceNetwork,
    fixtureDescriptor.exportOptions,
  );
  // Step 1: Evaluate the native runtime on the deterministic fixture sample.
  sourceNetwork.clear();
  const nativeOutput = sourceNetwork.activate([
    ...fixtureDescriptor.nativeInputValues,
  ]) as number[];

  // Step 2: Execute the binary model in an isolated ONNX Runtime process.
  const { inputNames, outputNames, runtimeOutput } =
    executeRuntimeParityInSubprocess(binaryModel, fixtureDescriptor);

  // Step 3: Compare the native and external outputs against the explicit lane tolerance.
  return buildExecutedResult(
    fixtureDescriptor,
    inputNames,
    outputNames,
    nativeOutput,
    runtimeOutput,
  );
}

/**
 * Execute reproducible randomized parity samples for one approved Phase 9 lane.
 *
 * Phase 9C reuses the existing binary-first fixture seam, but broadens the
 * evidence from one named golden sample into several seeded randomized cases.
 * The runner keeps the subset narrow by generating only lane-approved shapes.
 *
 * @param fixtureDescriptor Approved Phase 9 runtime-parity fixture descriptor.
 * @param randomizedRunOptions Seed and sample-count packet for this randomized pass.
 * @returns Flattened executed results for each seeded randomized sample.
 */
export async function runSeededOnnxRuntimeParitySamples(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  randomizedRunOptions: OnnxRuntimeParityRandomizedRunOptions,
): Promise<OnnxRuntimeParityRandomizedSampleResult[]> {
  const validatedSampleCount = resolveValidatedRandomizedSampleCount(
    randomizedRunOptions.sampleCount,
  );

  if (fixtureDescriptor.executionMode !== 'execute') {
    throw new Error(
      `Randomized parity requires an executed fixture: ${fixtureDescriptor.id}`,
    );
  }

  const sampleSeedGenerator = createSeededUnitIntervalGenerator(
    randomizedRunOptions.seed,
  );
  const randomizedSampleResults: OnnxRuntimeParityRandomizedSampleResult[] = [];

  // Step 1: Generate one deterministic sample seed per requested randomized case.
  const sampleIndexes = Array.from(
    { length: validatedSampleCount },
    (_unusedEntry, sampleIndex) => sampleIndex,
  );

  // Step 2: Materialize lane-approved randomized fixtures and reuse the Phase 9A runner.
  for (const sampleIndex of sampleIndexes) {
    const sampleSeed = Math.floor(sampleSeedGenerator() * SEEDED_RANDOM_DIVISOR);
    const randomizedFixtureDescriptor = createRandomizedParityFixture(
      fixtureDescriptor,
      sampleSeed,
      sampleIndex,
    );
    const parityResult = await runOnnxRuntimeParityFixture(
      randomizedFixtureDescriptor,
    );

    randomizedSampleResults.push({
      fixtureId: parityResult.fixture.id,
      lane: parityResult.fixture.lane,
      sampleIndex,
      nativeInputValues: [...randomizedFixtureDescriptor.nativeInputValues],
      runtimeInputValues: [
        ...(randomizedFixtureDescriptor.runtimeInputValues ??
          randomizedFixtureDescriptor.nativeInputValues),
      ],
      runtimeInputDimensions: randomizedFixtureDescriptor.runtimeInputDimensions,
      nativeOutput: parityResult.nativeOutput,
      runtimeOutput: parityResult.runtimeOutput,
      meanSquaredError: parityResult.meanSquaredError,
      maxAbsoluteDifference: parityResult.maxAbsoluteDifference,
      isWithinTolerance: parityResult.isWithinTolerance,
    });
  }

  return randomizedSampleResults;
}

/**
 * Execute one ONNX Runtime parity case in a fresh child Node process.
 *
 * This keeps the raw binding isolated from the Jest process so Phase 8 runtime
 * validation and Phase 9 parity execution do not compete over native addon
 * initialization order.
 *
 * @param binaryModel Binary `.onnx` payload produced by `exportToONNXBinary()`.
 * @param fixtureDescriptor Deterministic runtime-parity descriptor.
 * @returns Runtime output packet emitted by the child process.
 */
function executeRuntimeParityInSubprocess(
  binaryModel: Uint8Array,
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
): OnnxRuntimeParitySubprocessResult {
  const parityPayload: OnnxRuntimeParitySubprocessPayload = {
    binaryModelBase64: Buffer.from(binaryModel).toString('base64'),
    runtimeInputValues: [
      ...(fixtureDescriptor.runtimeInputValues ?? fixtureDescriptor.nativeInputValues),
    ],
    runtimeInputDimensions: fixtureDescriptor.runtimeInputDimensions,
  };
  const subprocessResult = spawnSync(
    process.execPath,
    ['-e', RUNTIME_PARITY_SUBPROCESS_SCRIPT],
    {
      input: JSON.stringify(parityPayload),
      encoding: 'utf8',
    },
  );

  if (subprocessResult.error) {
    throw subprocessResult.error;
  }

  if (subprocessResult.status !== 0) {
    throw new Error(resolveRuntimeSubprocessError(subprocessResult));
  }

  return JSON.parse(subprocessResult.stdout) as OnnxRuntimeParitySubprocessResult;
}

/**
 * Resolve the best available child-process failure text.
 *
 * @param subprocessResult Child-process execution result.
 * @returns Human-readable error message.
 */
function resolveRuntimeSubprocessError(
  subprocessResult: { stderr: string; stdout: string },
): string {
  const stderrText = subprocessResult.stderr.trim();
  if (stderrText.length > 0) {
    return stderrText;
  }

  const stdoutText = subprocessResult.stdout.trim();
  if (stdoutText.length > 0) {
    return stdoutText;
  }

  return 'ONNX runtime parity subprocess failed without error output.';
}

/**
 * Create one tensor instance that matches the raw binding's float32 output quirks.
 *
 * The current Node binding can surface float32 outputs through `ArrayBuffer` or
 * non-`Float32Array` views. This helper normalizes those payloads before they
 * reach the standard `Tensor` constructor.
 *
 * @param arg0 Raw tensor type or data argument.
 * @param arg1 Raw tensor data argument.
 * @param arg2 Optional tensor dimensions.
 * @returns A standard ONNX Runtime tensor instance.
 */
export function createRuntimeParityTensor(
  arg0: unknown,
  arg1?: unknown,
  arg2?: readonly number[],
): Tensor {
  if (arg0 === 'float32') {
    if (isArrayBufferLike(arg1)) {
      return new Tensor('float32', new Float32Array(arg1), arg2);
    }

    if (ArrayBuffer.isView(arg1) && !(arg1 instanceof Float32Array)) {
      const bufferView = arg1;
      return new Tensor(
        'float32',
        new Float32Array(
          bufferView.buffer,
          bufferView.byteOffset,
          bufferView.byteLength / Float32Array.BYTES_PER_ELEMENT,
        ),
        arg2,
      );
    }
  }

  return new Tensor(arg0 as never, arg1 as never, arg2 as never);
}

/**
 * Create the deterministic Phase 9A baseline float32 dense fixture.
 *
 * @returns Baseline same-family dense runtime-parity descriptor.
 */
function createBaselineFloat32DenseFixture(): OnnxRuntimeParityFixtureDescriptor {
  return {
    id: 'baseline-float32-dense',
    lane: 'baseline-float32-dense',
    executionMode: 'execute',
    tolerance: {
      maximumMeanSquaredError: 1e-12,
      maximumAbsoluteDifference: 1e-6,
    },
    nativeInputValues: [0.25, -0.5, 0.9],
    createNetwork: () => {
      const network = Network.createMLP(3, [4], 1);
      assignDeterministicParameters(network, {
        weightStart: -0.35,
        weightStep: 0.05,
        biasStart: -0.02,
        biasStep: 0.01,
      });
      return network;
    },
  };
}

/**
 * Create the deterministic storage-fp16 dense fixture descriptor.
 *
 * @returns Storage-fp16 cast-bridge runtime-parity descriptor.
 */
function createStorageFp16DenseFixture(): OnnxRuntimeParityFixtureDescriptor {
  return {
    id: 'storage-fp16-dense',
    lane: 'storage-fp16-dense',
    executionMode: 'execute',
    tolerance: {
      maximumMeanSquaredError: 1e-6,
      maximumAbsoluteDifference: 1e-3,
    },
    exportOptions: {
      includeMetadata: true,
      precision: {
        mode: 'storage-fp16',
      },
    },
    nativeInputValues: [0.2, -0.4],
    createNetwork: () => {
      const network = Network.createMLP(2, [1], 1);
      assignDeterministicParameters(network, {
        weightStart: -0.2,
        weightStep: 0.075,
        biasStart: 0.03125,
        biasStep: 0.015625,
      });
      return network;
    },
  };
}

/**
 * Create the deterministic one-output static-8bit dense fixture descriptor.
 *
 * @returns Static-8bit dense qlinear runtime-parity descriptor.
 */
function createStatic8BitDenseQlinearFixture(): OnnxRuntimeParityFixtureDescriptor {
  return {
    id: 'static-8bit-dense-qlinear',
    lane: 'static-8bit-dense-qlinear',
    executionMode: 'execute',
    tolerance: {
      maximumMeanSquaredError: 5e-3,
      maximumAbsoluteDifference: 0.075,
    },
    exportOptions: {
      includeMetadata: true,
      quantization: {
        mode: 'static-8bit',
        targets: ['dense'],
        calibration: {
          source: 'external',
          layerTargets: [
            {
              target: 'dense',
              layerIndex: 2,
              inputRange: { min: 0, max: 1 },
              outputRange: { min: 0, max: 0.75 },
            },
          ],
        },
        representation: 'qlinear',
      },
    },
    nativeInputValues: [0.25, -0.75],
    createNetwork: () => {
      const network = Network.createMLP(2, [2], 1);
      assignDeterministicParameters(network, {
        weightStart: -0.5,
        weightStep: 0.125,
        biasStart: -0.0625,
        biasStep: 0.03125,
      });
      return network;
    },
  };
}

/**
 * Create the deterministic explicit-Conv static-8bit fixture descriptor.
 *
 * @returns Static-8bit explicit-Conv runtime-parity descriptor.
 */
function createStatic8BitConvQlinearFixture(): OnnxRuntimeParityFixtureDescriptor {
  const convScenario = createConvGroundworkScenario();

  return {
    id: 'static-8bit-conv-qlinear',
    lane: 'static-8bit-conv-qlinear',
    executionMode: 'execute',
    tolerance: {
      maximumMeanSquaredError: 1e-2,
      maximumAbsoluteDifference: 0.1,
    },
    exportOptions: {
      includeMetadata: true,
      conv2dMappings: convScenario.mappings,
      quantization: {
        mode: 'static-8bit',
        targets: ['conv'],
        calibration: {
          source: 'external',
          layerTargets: [
            {
              target: 'conv',
              layerIndex: 1,
              inputRange: { min: -1, max: 1 },
              outputRange: { min: -0.5, max: 0.75 },
            },
          ],
        },
      },
    },
    nativeInputValues: convScenario.nativeInputValues,
    runtimeInputValues: convScenario.runtimeInputValues,
    runtimeInputDimensions: [1, 1, 3, 3],
    createNetwork: convScenario.createNetwork,
  };
}

/**
 * Create the deterministic DynamicQuantizeLinear dense-guidance descriptor.
 *
 * @returns Dynamic-guidance runtime-parity descriptor with an explicit tolerance packet.
 */
function createDynamicUint8DenseGuidanceFixture(): OnnxRuntimeParityFixtureDescriptor {
  return {
    id: 'dynamic-uint8-dense-guidance',
    lane: 'dynamic-uint8-dense-guidance',
    executionMode: 'execute',
    tolerance: {
      maximumMeanSquaredError: 1e-3,
      maximumAbsoluteDifference: 0.05,
    },
    exportOptions: {
      includeMetadata: true,
      quantization: {
        mode: 'dynamic-uint8',
        target: 'dense',
        representation: 'DynamicQuantizeLinear',
      },
    },
    nativeInputValues: [0.125, -0.375],
    createNetwork: () => {
      const network = Network.createMLP(2, [2], 1);
      assignDeterministicParameters(network, {
        weightStart: -0.25,
        weightStep: 0.0625,
        biasStart: -0.03125,
        biasStep: 0.015625,
      });
      return network;
    },
  };
}

/**
 * Apply deterministic scalar parameters to a network fixture.
 *
 * @param network Target network fixture.
 * @param parameterSeed Arithmetic progression used for weights and biases.
 */
function assignDeterministicParameters(
  network: Network,
  parameterSeed: DeterministicParameterSeed,
): void {
  network.connections.forEach((connectionEntry, connectionIndex) => {
    connectionEntry.weight =
      parameterSeed.weightStart + connectionIndex * parameterSeed.weightStep;
  });

  const nonInputNodes = network.nodes.filter(
    (nodeEntry) => nodeEntry.type !== 'input',
  );
  nonInputNodes.forEach((nodeEntry, nodeIndex) => {
    nodeEntry.bias = parameterSeed.biasStart + nodeIndex * parameterSeed.biasStep;
  });
}

/**
 * Build the executed runtime-parity result packet for one fixture.
 *
 * @param fixtureDescriptor Executed deterministic fixture descriptor.
 * @param inputNames Resolved runtime input names.
 * @param outputNames Resolved runtime output names.
 * @param nativeOutput Output from `Network.activate()`.
 * @param runtimeOutput Output from ONNX Runtime.
 * @returns Executed comparison packet for the current fixture.
 */
function buildExecutedResult(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  inputNames: readonly string[],
  outputNames: readonly string[],
  nativeOutput: number[],
  runtimeOutput: number[],
): OnnxRuntimeParityExecutedResult {
  const meanSquaredError = calculateMeanSquaredError(nativeOutput, runtimeOutput);
  const maxAbsoluteDifference = calculateMaxAbsoluteDifference(
    nativeOutput,
    runtimeOutput,
  );

  return {
    fixture: fixtureDescriptor,
    skipped: false,
    inputNames: [...inputNames],
    outputNames: [...outputNames],
    nativeOutput,
    runtimeOutput,
    meanSquaredError,
    maxAbsoluteDifference,
    isWithinTolerance:
      meanSquaredError <= fixtureDescriptor.tolerance.maximumMeanSquaredError &&
      maxAbsoluteDifference <=
        fixtureDescriptor.tolerance.maximumAbsoluteDifference,
  };
}

/**
 * Calculate mean squared error for one parity comparison.
 *
 * @param nativeOutput Native runtime output.
 * @param runtimeOutput ONNX Runtime output.
 * @returns Mean squared error across the compared output vector.
 */
function calculateMeanSquaredError(
  nativeOutput: readonly number[],
  runtimeOutput: readonly number[],
): number {
  const squaredDifferenceSum = nativeOutput.reduce(
    (runningSum, nativeValue, outputIndex) => {
      const runtimeValue = runtimeOutput[outputIndex]!;
      const outputDifference = nativeValue - runtimeValue;
      return runningSum + outputDifference * outputDifference;
    },
    0,
  );

  return squaredDifferenceSum / nativeOutput.length;
}

/**
 * Calculate the maximum absolute scalar difference for one parity comparison.
 *
 * @param nativeOutput Native runtime output.
 * @param runtimeOutput ONNX Runtime output.
 * @returns Maximum absolute difference across the compared output vector.
 */
function calculateMaxAbsoluteDifference(
  nativeOutput: readonly number[],
  runtimeOutput: readonly number[],
): number {
  return nativeOutput.reduce((runningMaximum, nativeValue, outputIndex) => {
    const runtimeValue = runtimeOutput[outputIndex]!;
    return Math.max(runningMaximum, Math.abs(nativeValue - runtimeValue));
  }, 0);
}

/**
 * Create the deterministic explicit-Conv groundwork scenario reused by Phase 9.
 *
 * @returns Deterministic network factory, mappings, and sample inputs for the explicit-Conv lane.
 */
function createConvGroundworkScenario(): {
  createNetwork: () => Network;
  mappings: Conv2DMapping[];
  nativeInputValues: number[];
  runtimeInputValues: number[];
} {
  const inputChannels = 1;
  const inputHeight = 3;
  const inputWidth = 3;
  const kernelHeight = 2;
  const kernelWidth = 2;
  const strideHeight = 1;
  const strideWidth = 1;
  const outputChannels = 2;
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  const inputSize = inputChannels * inputHeight * inputWidth;
  const hiddenSize = outputChannels * outputHeight * outputWidth;
  const outputSize = 3;
  const sampleInputValues = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];

  return {
    createNetwork: () => {
      const network = Network.createMLP(inputSize, [hiddenSize], outputSize);
      assignDeterministicParameters(network, {
        weightStart: -0.2,
        weightStep: 0.01,
        biasStart: -0.015,
        biasStep: 0.005,
      });
      return network;
    },
    mappings: [
      {
        layerIndex: 1,
        inHeight: inputHeight,
        inWidth: inputWidth,
        inChannels: inputChannels,
        kernelHeight,
        kernelWidth,
        strideHeight,
        strideWidth,
        padTop: 0,
        padBottom: 0,
        padLeft: 0,
        padRight: 0,
        outHeight: outputHeight,
        outWidth: outputWidth,
        outChannels: outputChannels,
      },
    ],
    nativeInputValues: sampleInputValues,
    runtimeInputValues: sampleInputValues,
  };
}

/**
 * Resolve runtime feed dimensions from the fixture packet or session metadata.
 *
 * @param runtimeInputValues Numeric runtime feed values.
 * @param inputShape Runtime metadata shape reported by ONNX Runtime.
 * @returns Numeric input dimensions suitable for the feed tensor constructor.
 */
export function resolveRuntimeTensorDimensions(
  tensorShape: readonly (number | string)[],
): number[] {
  return tensorShape.map((shapeDimension) =>
    typeof shapeDimension === 'number' ? shapeDimension : 1,
  );
}

/**
 * Check whether one runtime payload should be treated as an ArrayBuffer source.
 *
 * @param tensorData Raw tensor payload from the native binding.
 * @returns True when the payload is a plain or shared ArrayBuffer-like object.
 */
function isArrayBufferLike(tensorData: unknown): tensorData is ArrayBuffer {
  return (
    tensorData instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' &&
      tensorData instanceof SharedArrayBuffer) ||
    Object.prototype.toString.call(tensorData) === '[object ArrayBuffer]' ||
    Object.prototype.toString.call(tensorData) === '[object SharedArrayBuffer]'
  );
}

/**
 * Resolve the allowed randomized sample count for Phase 9C.
 *
 * @param sampleCount Requested randomized sample count.
 * @returns Validated sample count within the supported Phase 9C bounds.
 */
function resolveValidatedRandomizedSampleCount(sampleCount: number): number {
  if (
    !Number.isInteger(sampleCount) ||
    sampleCount < MINIMUM_PHASE_9_RANDOMIZED_SAMPLE_COUNT ||
    sampleCount > MAXIMUM_PHASE_9_RANDOMIZED_SAMPLE_COUNT
  ) {
    throw new Error(
      `Phase 9 randomized parity sampleCount must be an integer between ${MINIMUM_PHASE_9_RANDOMIZED_SAMPLE_COUNT} and ${MAXIMUM_PHASE_9_RANDOMIZED_SAMPLE_COUNT}.`,
    );
  }

  return sampleCount;
}

/**
 * Materialize one lane-approved randomized fixture descriptor.
 *
 * @param fixtureDescriptor Base execute fixture from the frozen Phase 9 inventory.
 * @param sampleSeed Deterministic seed for this randomized sample.
 * @returns Randomized fixture descriptor that stays within the declared lane boundary.
 */
function createRandomizedParityFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
): OnnxRuntimeParityExecutedFixtureDescriptor {
  const randomizedFixtureFactory =
    RANDOMIZED_PARITY_FACTORY_BY_LANE[fixtureDescriptor.lane];

  return randomizedFixtureFactory(fixtureDescriptor, sampleSeed, sampleIndex);
}

/**
 * Create one seeded same-family dense randomized fixture.
 *
 * @param fixtureDescriptor Base dense execute fixture.
 * @param sampleSeed Deterministic seed for this randomized case.
 * @returns Lane-compatible randomized dense fixture descriptor.
 */
function createRandomizedDenseFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
): OnnxRuntimeParityExecutedFixtureDescriptor {
  const sampleGenerator = createSeededUnitIntervalGenerator(sampleSeed);
  const inputWidth = cycleIntegerRangeBySampleIndex(
    sampleIndex,
    RANDOMIZED_DENSE_INPUT_WIDTH_RANGE,
  );
  const hiddenWidth = cycleIntegerRangeBySampleIndex(
    sampleIndex + 1,
    RANDOMIZED_DENSE_HIDDEN_WIDTH_RANGE,
  );
  const parameterSeed = createRandomizedParameterSeed(
    sampleGenerator,
    RANDOMIZED_DENSE_PARAMETER_SEED_BOUNDS,
  );
  const nativeInputValues = createRandomizedNumericVector(
    sampleGenerator,
    inputWidth,
    RANDOMIZED_DENSE_INPUT_VALUE_RANGE,
  );

  return {
    id: fixtureDescriptor.id,
    lane: fixtureDescriptor.lane,
    executionMode: 'execute',
    tolerance: fixtureDescriptor.tolerance,
    exportOptions: fixtureDescriptor.exportOptions
      ? structuredClone(fixtureDescriptor.exportOptions)
      : undefined,
    nativeInputValues,
    createNetwork: () => {
      const network = Network.createMLP(inputWidth, [hiddenWidth], 1);
      assignDeterministicParameters(network, parameterSeed);
      return network;
    },
  };
}

/**
 * Create one seeded explicit-Conv static-8bit randomized fixture.
 *
 * @param fixtureDescriptor Base explicit-Conv execute fixture.
 * @param sampleSeed Deterministic seed for this randomized case.
 * @returns Lane-compatible randomized explicit-Conv fixture descriptor.
 */
function createRandomizedStatic8BitConvFixture(
  fixtureDescriptor: OnnxRuntimeParityFixtureDescriptor,
  sampleSeed: number,
  sampleIndex: number,
): OnnxRuntimeParityExecutedFixtureDescriptor {
  const sampleGenerator = createSeededUnitIntervalGenerator(sampleSeed);
  const inputHeight = cycleIntegerRangeBySampleIndex(
    sampleIndex,
    RANDOMIZED_CONV_INPUT_WIDTH_RANGE,
  );
  const inputWidth = cycleIntegerRangeBySampleIndex(
    sampleIndex + 1,
    RANDOMIZED_CONV_INPUT_WIDTH_RANGE,
  );
  const outputChannels = cycleIntegerRangeBySampleIndex(
    sampleIndex,
    RANDOMIZED_CONV_OUTPUT_CHANNEL_RANGE,
  );
  const outputHeight = inputHeight - 1;
  const outputWidth = inputWidth - 1;
  const inputSize = inputHeight * inputWidth;
  const hiddenSize = outputChannels * outputHeight * outputWidth;
  const outputSize = cycleIntegerRangeBySampleIndex(
    sampleIndex + 1,
    RANDOMIZED_CONV_OUTPUT_WIDTH_RANGE,
  );
  const parameterSeed = createRandomizedParameterSeed(
    sampleGenerator,
    RANDOMIZED_CONV_PARAMETER_SEED_BOUNDS,
  );
  const nativeInputValues = createRandomizedNumericVector(
    sampleGenerator,
    inputSize,
    RANDOMIZED_CONV_INPUT_VALUE_RANGE,
  );

  return {
    id: fixtureDescriptor.id,
    lane: fixtureDescriptor.lane,
    executionMode: 'execute',
    tolerance: fixtureDescriptor.tolerance,
    exportOptions: {
      includeMetadata: true,
      conv2dMappings: [
        {
          layerIndex: 1,
          inHeight: inputHeight,
          inWidth: inputWidth,
          inChannels: 1,
          kernelHeight: 2,
          kernelWidth: 2,
          strideHeight: 1,
          strideWidth: 1,
          padTop: 0,
          padBottom: 0,
          padLeft: 0,
          padRight: 0,
          outHeight: outputHeight,
          outWidth: outputWidth,
          outChannels: outputChannels,
        },
      ],
      quantization: {
        mode: 'static-8bit',
        targets: ['conv'],
        calibration: {
          source: 'external',
          layerTargets: [
            {
              target: 'conv',
              layerIndex: 1,
              inputRange: { min: -1, max: 1 },
              outputRange: { min: -0.5, max: 0.75 },
            },
          ],
        },
      },
    },
    nativeInputValues,
    runtimeInputValues: nativeInputValues,
    runtimeInputDimensions: [1, 1, inputHeight, inputWidth],
    createNetwork: () => {
      const network = Network.createMLP(inputSize, [hiddenSize], outputSize);
      assignDeterministicParameters(network, parameterSeed);
      return network;
    },
  };
}

/**
 * Create one reproducible unit-interval generator from a 32-bit seed.
 *
 * @param initialSeed Seed used for the linear congruential generator.
 * @returns Deterministic floating-point generator in the range [0, 1).
 */
function createSeededUnitIntervalGenerator(initialSeed: number): () => number {
  let currentSeed = initialSeed >>> 0;

  return () => {
    currentSeed =
      (Math.imul(currentSeed, SEEDED_RANDOM_MULTIPLIER) +
        SEEDED_RANDOM_INCREMENT) >>>
      0;
    return currentSeed / SEEDED_RANDOM_DIVISOR;
  };
}

/**
 * Cycle deterministically through an inclusive integer range by sample index.
 *
 * @param sampleIndex Zero-based randomized sample index.
 * @param numericRange Inclusive integer range.
 * @returns Deterministic integer derived from the sample position.
 */
function cycleIntegerRangeBySampleIndex(
  sampleIndex: number,
  numericRange: NumericRange,
): number {
  const rangeWidth = numericRange.maximum - numericRange.minimum + 1;

  return numericRange.minimum + (sampleIndex % rangeWidth);
}

/**
 * Sample one rounded floating-point value inside a numeric range.
 *
 * @param sampleGenerator Deterministic unit-interval generator.
 * @param numericRange Inclusive numeric range.
 * @returns Rounded floating-point sample inside the declared range.
 */
function sampleNumberInRange(
  sampleGenerator: () => number,
  numericRange: NumericRange,
): number {
  const rawValue =
    numericRange.minimum +
    sampleGenerator() * (numericRange.maximum - numericRange.minimum);

  return Number(rawValue.toFixed(SEEDED_RANDOM_DECIMAL_PLACES));
}

/**
 * Create a reproducible numeric vector for one randomized parity sample.
 *
 * @param sampleGenerator Deterministic unit-interval generator.
 * @param vectorLength Target vector length.
 * @param numericRange Numeric range for each sampled value.
 * @returns Deterministic numeric vector for the sample input.
 */
function createRandomizedNumericVector(
  sampleGenerator: () => number,
  vectorLength: number,
  numericRange: NumericRange,
): number[] {
  return Array.from({ length: vectorLength }, () =>
    sampleNumberInRange(sampleGenerator, numericRange),
  );
}

/**
 * Create a reproducible arithmetic parameter seed packet.
 *
 * @param sampleGenerator Deterministic unit-interval generator.
 * @param parameterSeedBounds Numeric ranges for each parameter-seed component.
 * @returns Deterministic scalar parameter packet for one randomized network.
 */
function createRandomizedParameterSeed(
  sampleGenerator: () => number,
  parameterSeedBounds: {
    weightStart: NumericRange;
    weightStep: NumericRange;
    biasStart: NumericRange;
    biasStep: NumericRange;
  },
): DeterministicParameterSeed {
  return {
    weightStart: sampleNumberInRange(
      sampleGenerator,
      parameterSeedBounds.weightStart,
    ),
    weightStep: sampleNumberInRange(
      sampleGenerator,
      parameterSeedBounds.weightStep,
    ),
    biasStart: sampleNumberInRange(
      sampleGenerator,
      parameterSeedBounds.biasStart,
    ),
    biasStep: sampleNumberInRange(sampleGenerator, parameterSeedBounds.biasStep),
  };
}