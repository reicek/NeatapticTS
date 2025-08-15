# multithreading

## multithreading/multi.ts

### multi

Multi-threading utilities for neural network operations.

This class provides methods for serializing datasets, activating serialized networks,
and testing serialized datasets. These utilities align with the Instinct algorithm's
emphasis on efficient evaluation and mutation of neural networks in parallel environments.

### Multi

Multi-threading utilities for neural network operations.

This class provides methods for serializing datasets, activating serialized networks,
and testing serialized datasets. These utilities align with the Instinct algorithm's
emphasis on efficient evaluation and mutation of neural networks in parallel environments.

### default

#### absolute

`(x: number) => number`

Absolute activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### activateSerializedNetwork

`(input: number[], A: number[], S: number[], data: number[], F: Function[]) => number[]`

Activates a serialized network.

Parameters:
- `` - - The input values.
- `` - - The activations array.
- `` - - The states array.
- `` - - The serialized network data.
- `` - - The activation functions.

Returns: The output values.

#### activations

A list of compiled activation functions in a specific order.

#### bentIdentity

`(x: number) => number`

Bent Identity activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### bipolar

`(x: number) => number`

Bipolar activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### bipolarSigmoid

`(x: number) => number`

Bipolar Sigmoid activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### deserializeDataSet

`(serializedSet: number[]) => { input: number[]; output: number[]; }[]`

Deserializes a dataset from a flat array.

Parameters:
- `` - - The serialized dataset.

Returns: The deserialized dataset as an array of input-output pairs.

#### gaussian

`(x: number) => number`

Gaussian activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### getBrowserTestWorker

`() => Promise<typeof import("D:/code-practice/NeatapticTS/src/multithreading/workers/browser/testworker").TestWorker>`

Gets the browser test worker.

Returns: The browser test worker.

#### getNodeTestWorker

`() => Promise<typeof import("D:/code-practice/NeatapticTS/src/multithreading/workers/node/testworker").TestWorker>`

Gets the node test worker.

Returns: The node test worker.

#### hardTanh

`(x: number) => number`

Hard Tanh activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### identity

`(x: number) => number`

Identity activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### inverse

`(x: number) => number`

Inverse activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### logistic

`(x: number) => number`

Logistic activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### relu

`(x: number) => number`

Rectified Linear Unit (ReLU) activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### selu

`(x: number) => number`

Scaled Exponential Linear Unit (SELU) activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### serializeDataSet

`(dataSet: { input: number[]; output: number[]; }[]) => number[]`

Serializes a dataset into a flat array.

Parameters:
- `` - - The dataset to serialize.

Returns: The serialized dataset.

#### sinusoid

`(x: number) => number`

Sinusoid activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### softplus

`(x: number) => number`

Softplus activation function. - Added

Parameters:
- `` - - The input value.

Returns: The activated value.

#### softsign

`(x: number) => number`

Softsign activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### step

`(x: number) => number`

Step activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### tanh

`(x: number) => number`

Hyperbolic tangent activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### testSerializedSet

`(set: { input: number[]; output: number[]; }[], cost: (expected: number[], actual: number[]) => number, A: number[], S: number[], data: number[], F: Function[]) => number`

Tests a serialized dataset using a cost function.

Parameters:
- `` - - The serialized dataset as an array of input-output pairs.
- `` - - The cost function.
- `` - - The activations array.
- `` - - The states array.
- `` - - The serialized network data.
- `` - - The activation functions.

Returns: The average error.

#### workers

Workers for multi-threading
