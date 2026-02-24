import * as methods from '../../../methods/methods';
import type Network from '../../network';

type TestSample = { input: number[]; output: number[] };
type CostFunction = (target: number[], output: number[]) => number;
type TestNetworkResult = { error: number; time: number };

/**
 * Evaluate a dataset and return average error and elapsed time.
 *
 * @param this Bound network instance.
 * @param set Evaluation samples.
 * @param cost Optional cost function override.
 * @returns Mean error and evaluation duration.
 */
export function testNetwork(
  this: Network,
  set: TestSample[],
  cost?: CostFunction,
): TestNetworkResult {
  // Step 1: Validate dataset shape and dimensions.
  validateTestSet(this, set);

  // Step 2: Resolve the cost function used during evaluation.
  const costFunction = resolveCostFunction(cost);

  // Step 3: Configure deterministic inference state.
  const startTime = Date.now();
  const previousDropout = disableDropoutForTesting(this);
  resetHiddenMasks(this);

  // Step 4: Evaluate samples and always restore runtime state.
  try {
    const cumulativeError = evaluateSamples(this, set, costFunction);
    return createTestResult(cumulativeError, set.length, startTime);
  } finally {
    restoreDropout(this, previousDropout);
  }
}

/**
 * Validate that the evaluation set exists and each sample matches network dimensions.
 *
 * @param network Bound network instance.
 * @param testSet Evaluation sample set.
 */
function validateTestSet(network: Network, testSet: TestSample[]): void {
  validateTestSetPresence(testSet);
  validateAllSampleDimensions(network, testSet);
}

/**
 * Validate that the test set is a non-empty array.
 *
 * @param testSet Evaluation sample set.
 */
function validateTestSetPresence(testSet: TestSample[]): void {
  if (!Array.isArray(testSet) || testSet.length === 0) {
    throw new Error('Test set is empty or not an array.');
  }
}

/**
 * Validate input and output dimensions for every sample.
 *
 * @param network Bound network instance.
 * @param testSet Evaluation sample set.
 */
function validateAllSampleDimensions(
  network: Network,
  testSet: TestSample[],
): void {
  for (const sample of testSet) {
    validateSampleInputDimensions(network, sample);
    validateSampleOutputDimensions(network, sample);
  }
}

/**
 * Validate one sample input vector size.
 *
 * @param network Bound network instance.
 * @param sample Evaluation sample.
 */
function validateSampleInputDimensions(
  network: Network,
  sample: TestSample,
): void {
  if (!Array.isArray(sample.input) || sample.input.length !== network.input) {
    throw new Error(
      `Test sample input size mismatch: expected ${network.input}, got ${
        sample.input ? sample.input.length : 'undefined'
      }`,
    );
  }
}

/**
 * Validate one sample output vector size.
 *
 * @param network Bound network instance.
 * @param sample Evaluation sample.
 */
function validateSampleOutputDimensions(
  network: Network,
  sample: TestSample,
): void {
  if (
    !Array.isArray(sample.output) ||
    sample.output.length !== network.output
  ) {
    throw new Error(
      `Test sample output size mismatch: expected ${network.output}, got ${
        sample.output ? sample.output.length : 'undefined'
      }`,
    );
  }
}

/**
 * Resolve evaluation cost function with a stable default.
 *
 * @param cost Optional cost override.
 * @returns Cost function used for test evaluation.
 */
function resolveCostFunction(cost?: CostFunction): CostFunction {
  return cost ?? methods.Cost.mse;
}

/**
 * Force hidden-node masks to active state for deterministic testing.
 *
 * @param network Bound network instance.
 */
function resetHiddenMasks(network: Network): void {
  for (const node of network.nodes) {
    if (node.type === 'hidden') {
      node.mask = 1;
    }
  }
}

/**
 * Disable dropout while preserving previous runtime dropout value.
 *
 * @param network Bound network instance.
 * @returns Previous dropout value.
 */
function disableDropoutForTesting(network: Network): number {
  const previousDropout = network.dropout;
  if (network.dropout > 0) {
    network.dropout = 0;
  }
  return previousDropout;
}

/**
 * Restore dropout value after test evaluation.
 *
 * @param network Bound network instance.
 * @param previousDropout Dropout value to restore.
 */
function restoreDropout(network: Network, previousDropout: number): void {
  network.dropout = previousDropout;
}

/**
 * Evaluate all test samples and accumulate total cost.
 *
 * @param network Bound network instance.
 * @param testSet Evaluation sample set.
 * @param costFunction Cost function used for scoring.
 * @returns Cumulative error across all samples.
 */
function evaluateSamples(
  network: Network,
  testSet: TestSample[],
  costFunction: CostFunction,
): number {
  let cumulativeError = 0;
  for (const sample of testSet) {
    cumulativeError += evaluateSingleSample(network, sample, costFunction);
  }
  return cumulativeError;
}

/**
 * Evaluate a single sample and return its cost.
 *
 * @param network Bound network instance.
 * @param sample Evaluation sample.
 * @param costFunction Cost function used for scoring.
 * @returns Error for the sample.
 */
function evaluateSingleSample(
  network: Network,
  sample: TestSample,
  costFunction: CostFunction,
): number {
  const output = network.noTraceActivate(sample.input);
  return costFunction(sample.output, output);
}

/**
 * Build the final test result payload.
 *
 * @param cumulativeError Cumulative sample error.
 * @param sampleCount Number of evaluated samples.
 * @param startTime Evaluation start timestamp.
 * @returns Mean error and elapsed duration.
 */
function createTestResult(
  cumulativeError: number,
  sampleCount: number,
  startTime: number,
): TestNetworkResult {
  return {
    error: cumulativeError / sampleCount,
    time: Date.now() - startTime,
  };
}
