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

`(inputValue: number) => number`

Absolute activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### activateSerializedNetwork

`(inputValues: number[], activationValues: number[], stateValues: number[], serializedNetwork: number[], activationFunctions: import("C:/NeatapticTS/src/multithreading/types").ActivationFn[]) => number[]`

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

`(inputValue: number) => number`

Bent Identity activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### bipolar

`(inputValue: number) => number`

Bipolar activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### bipolarSigmoid

`(inputValue: number) => number`

Bipolar Sigmoid activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### deserializeDataSet

`(serializedSet: number[]) => import("C:/NeatapticTS/src/multithreading/types").SerializedSample[]`

Deserializes a dataset from a flat array.

Parameters:
- `` - - The serialized dataset.

Returns: The deserialized dataset as an array of input-output pairs.

#### gaussian

`(inputValue: number) => number`

Gaussian activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### getBrowserTestWorker

`() => Promise<import("C:/NeatapticTS/src/multithreading/types").TestWorkerConstructor>`

Gets the browser test worker.

Returns: The browser test worker.

#### getNodeTestWorker

`() => Promise<import("C:/NeatapticTS/src/multithreading/types").TestWorkerConstructor>`

Gets the node test worker.

Returns: The node test worker.

#### hardTanh

`(inputValue: number) => number`

Hard Tanh activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### identity

`(inputValue: number) => number`

Identity activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### inverse

`(inputValue: number) => number`

Inverse activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### logistic

`(inputValue: number) => number`

Logistic activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### relu

`(inputValue: number) => number`

Rectified Linear Unit (ReLU) activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### selu

`(inputValue: number) => number`

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

`(inputValue: number) => number`

Sinusoid activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### softplus

`(inputValue: number) => number`

Softplus activation function. - Added

Parameters:
- `` - - The input value.

Returns: The activated value.

#### softsign

`(inputValue: number) => number`

Softsign activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### step

`(inputValue: number) => number`

Step activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### tanh

`(inputValue: number) => number`

Hyperbolic tangent activation function.

Parameters:
- `` - - The input value.

Returns: The activated value.

#### testSerializedSet

`(serializedSampleSet: import("C:/NeatapticTS/src/multithreading/types").SerializedSample[], cost: (expected: number[], actual: number[]) => number, activationValues: number[], stateValues: number[], serializedNetwork: number[], activationFunctions: import("C:/NeatapticTS/src/multithreading/types").ActivationFn[]) => number`

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

## multithreading/types.ts

### types

Shared types for multithreading helpers and test workers.

### ActivationFn

`(x: number) => number`

Shared types for multithreading helpers and test workers.

### SerializableNetwork

### SerializedSample

### TestWorkerConstructor

### TestWorkerInstance

## multithreading/multi.utils.ts

### absoluteActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Absolute activation.

### activateSerializedNetwork

`(inputValues: number[], activationValues: number[], stateValues: number[], serializedNetwork: number[], activationFunctions: import("C:/NeatapticTS/src/multithreading/types").ActivationFn[]) => number[]`

Activates a serialized network and produces outputs.

Parameters:
- `inputValues` - - Inputs to feed into the network.
- `activationValues` - - Mutable activation register shared across runs.
- `stateValues` - - Mutable state register shared across runs.
- `serializedNetwork` - - Flat encoded network data.
- `activationFunctions` - - Ordered activation functions.

Returns: Activated outputs.

### ACTIVATION_FUNCTIONS

### bentIdentityActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Bent identity activation.

### bipolarActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Bipolar activation.

### bipolarSigmoidActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Bipolar sigmoid activation.

### deserializeDataSet

`(serializedSet: number[]) => import("C:/NeatapticTS/src/multithreading/types").SerializedSample[]`

Deserializes a dataset from its flat representation.

Parameters:
- `serializedSet` - - Flat serialized dataset array.

Returns: Array of input/output sample pairs.

### gaussianActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Gaussian activation.

### hardTanhActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Hard tanh activation.

### identityActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Identity activation.

### inverseActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Inverse activation.

### logisticActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Logistic activation.

### reluActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: ReLU activation.

### seluActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: SELU activation.

### serializeDataSet

`(dataSet: { input: number[]; output: number[]; }[]) => number[]`

Serializes a dataset into a flat numeric array.

Parameters:
- `dataSet` - - Collection of samples with input and output arrays.

Returns: Flat serialized representation [inputCount, outputCount, ...samples].

### sinusoidActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Sinusoid activation.

### softplusActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Softplus activation.

### softsignActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Softsign activation.

### stepActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Step activation.

### tanhActivation

`(value: number) => number`

Parameters:
- `value` - - Input value.

Returns: Hyperbolic tangent activation.

### testSerializedSet

`(serializedSampleSet: import("C:/NeatapticTS/src/multithreading/types").SerializedSample[], costFunction: (expected: number[], actual: number[]) => number, activationValues: number[], stateValues: number[], serializedNetwork: number[], activationFunctions: import("C:/NeatapticTS/src/multithreading/types").ActivationFn[]) => number`

Tests a serialized dataset using a cost function.

Parameters:
- `serializedSampleSet` - - Serialized dataset samples.
- `costFunction` - - Cost function comparing expected and actual outputs.
- `activationValues` - - Mutable activation register.
- `stateValues` - - Mutable state register.
- `serializedNetwork` - - Serialized network data.
- `activationFunctions` - - Activation functions to apply.

Returns: Average cost or NaN when invalid input.
