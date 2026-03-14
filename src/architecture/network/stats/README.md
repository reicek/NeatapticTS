# architecture/network/stats

Network statistics accessors.

Currently exposes a single helper for retrieving the most recent regularization / stochasticity
metrics snapshot recorded during training or evaluation. The internal `_lastStats` field (on the
Network instance, typed as any) is expected to be populated elsewhere in the training loop with
values such as:
 - l1Penalty, l2Penalty
 - dropoutApplied (fraction of units dropped last pass)
 - weightNoiseStd (effective std dev used if noise injected)
 - sparsityRatio, prunedConnections
 - any custom user extensions (object is not strictly typed to allow experimentation)

Design decision: We return a deep copy to prevent external mutation of internal accounting state.
If the object is large and copying becomes a bottleneck, future versions could offer a freeze
option or incremental diff interface.

## architecture/network/stats/network.stats.utils.ts

### getRegularizationStats

`() => Record<string, unknown> | null`

Obtain the last recorded regularization / stochastic statistics snapshot.

Returns a defensive deep copy so callers can inspect metrics without risking mutation of the
internal `_lastStats` object maintained by the training loop (e.g., during pruning, dropout, or
noise scheduling updates).

Returns: A deep-cloned stats object or null if no stats have been recorded yet.

### testNetwork

`(set: TestSample[], cost: CostFunction | undefined) => TestNetworkResult`

Evaluate a dataset and return average error and elapsed time.

Parameters:
- `this` - Bound network instance.
- `set` - Evaluation samples.
- `cost` - Optional cost function override.

Returns: Mean error and evaluation duration.

## architecture/network/stats/network.stats.test.utils.ts

### createTestResult

`(cumulativeError: number, sampleCount: number, startTime: number) => TestNetworkResult`

Build the final test result payload.

Parameters:
- `cumulativeError` - Cumulative sample error.
- `sampleCount` - Number of evaluated samples.
- `startTime` - Evaluation start timestamp.

Returns: Mean error and elapsed duration.

### disableDropoutForTesting

`(network: import("src/architecture/network").default) => number`

Disable dropout while preserving previous runtime dropout value.

Parameters:
- `network` - Bound network instance.

Returns: Previous dropout value.

### evaluateSamples

`(network: import("src/architecture/network").default, testSet: TestSample[], costFunction: CostFunction) => number`

Evaluate all test samples and accumulate total cost.

Parameters:
- `network` - Bound network instance.
- `testSet` - Evaluation sample set.
- `costFunction` - Cost function used for scoring.

Returns: Cumulative error across all samples.

### evaluateSingleSample

`(network: import("src/architecture/network").default, sample: TestSample, costFunction: CostFunction) => number`

Evaluate a single sample and return its cost.

Parameters:
- `network` - Bound network instance.
- `sample` - Evaluation sample.
- `costFunction` - Cost function used for scoring.

Returns: Error for the sample.

### resetHiddenMasks

`(network: import("src/architecture/network").default) => void`

Force hidden-node masks to active state for deterministic testing.

Parameters:
- `network` - Bound network instance.

### resolveCostFunction

`(cost: CostFunction | undefined) => CostFunction`

Resolve evaluation cost function with a stable default.

Parameters:
- `cost` - Optional cost override.

Returns: Cost function used for test evaluation.

### restoreDropout

`(network: import("src/architecture/network").default, previousDropout: number) => void`

Restore dropout value after test evaluation.

Parameters:
- `network` - Bound network instance.
- `previousDropout` - Dropout value to restore.

### testNetwork

`(set: TestSample[], cost: CostFunction | undefined) => TestNetworkResult`

Evaluate a dataset and return average error and elapsed time.

Parameters:
- `this` - Bound network instance.
- `set` - Evaluation samples.
- `cost` - Optional cost function override.

Returns: Mean error and evaluation duration.

### validateAllSampleDimensions

`(network: import("src/architecture/network").default, testSet: TestSample[]) => void`

Validate input and output dimensions for every sample.

Parameters:
- `network` - Bound network instance.
- `testSet` - Evaluation sample set.

### validateSampleInputDimensions

`(network: import("src/architecture/network").default, sample: TestSample) => void`

Validate one sample input vector size.

Parameters:
- `network` - Bound network instance.
- `sample` - Evaluation sample.

### validateSampleOutputDimensions

`(network: import("src/architecture/network").default, sample: TestSample) => void`

Validate one sample output vector size.

Parameters:
- `network` - Bound network instance.
- `sample` - Evaluation sample.

### validateTestSet

`(network: import("src/architecture/network").default, testSet: TestSample[]) => void`

Validate that the evaluation set exists and each sample matches network dimensions.

Parameters:
- `network` - Bound network instance.
- `testSet` - Evaluation sample set.

### validateTestSetPresence

`(testSet: TestSample[]) => void`

Validate that the test set is a non-empty array.

Parameters:
- `testSet` - Evaluation sample set.
