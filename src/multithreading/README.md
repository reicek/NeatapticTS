# multithreading

Shared types for multithreading helpers and test workers.

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

```ts
absolute(
  inputValue: number,
): number
```

Absolute activation function.

Returns: The activated value.

#### activateSerializedNetwork

```ts
activateSerializedNetwork(
  inputValues: number[],
  activationValues: number[],
  stateValues: number[],
  serializedNetwork: number[],
  activationFunctions: ActivationFn[],
): number[]
```

Activates a serialized network.

Returns: The output values.

#### activations

A list of compiled activation functions in a specific order.

#### bentIdentity

```ts
bentIdentity(
  inputValue: number,
): number
```

Bent Identity activation function.

Returns: The activated value.

#### bipolar

```ts
bipolar(
  inputValue: number,
): number
```

Bipolar activation function.

Returns: The activated value.

#### bipolarSigmoid

```ts
bipolarSigmoid(
  inputValue: number,
): number
```

Bipolar Sigmoid activation function.

Returns: The activated value.

#### deserializeDataSet

```ts
deserializeDataSet(
  serializedSet: number[],
): SerializedSample[]
```

Deserializes a dataset from a flat array.

Returns: The deserialized dataset as an array of input-output pairs.

#### gaussian

```ts
gaussian(
  inputValue: number,
): number
```

Gaussian activation function.

Returns: The activated value.

#### getBrowserTestWorker

```ts
getBrowserTestWorker(): Promise<TestWorkerConstructor>
```

Gets the browser test worker.

Returns: The browser test worker.

#### getNodeTestWorker

```ts
getNodeTestWorker(): Promise<TestWorkerConstructor>
```

Gets the node test worker.

Returns: The node test worker.

#### hardTanh

```ts
hardTanh(
  inputValue: number,
): number
```

Hard Tanh activation function.

Returns: The activated value.

#### identity

```ts
identity(
  inputValue: number,
): number
```

Identity activation function.

Returns: The activated value.

#### inverse

```ts
inverse(
  inputValue: number,
): number
```

Inverse activation function.

Returns: The activated value.

#### logistic

```ts
logistic(
  inputValue: number,
): number
```

Logistic activation function.

Returns: The activated value.

#### relu

```ts
relu(
  inputValue: number,
): number
```

Rectified Linear Unit (ReLU) activation function.

Returns: The activated value.

#### selu

```ts
selu(
  inputValue: number,
): number
```

Scaled Exponential Linear Unit (SELU) activation function.

Returns: The activated value.

#### serializeDataSet

```ts
serializeDataSet(
  dataSet: { input: number[]; output: number[]; }[],
): number[]
```

Serializes a dataset into a flat array.

Returns: The serialized dataset.

#### sinusoid

```ts
sinusoid(
  inputValue: number,
): number
```

Sinusoid activation function.

Returns: The activated value.

#### softplus

```ts
softplus(
  inputValue: number,
): number
```

Softplus activation function. - Added

Returns: The activated value.

#### softsign

```ts
softsign(
  inputValue: number,
): number
```

Softsign activation function.

Returns: The activated value.

#### step

```ts
step(
  inputValue: number,
): number
```

Step activation function.

Returns: The activated value.

#### tanh

```ts
tanh(
  inputValue: number,
): number
```

Hyperbolic tangent activation function.

Returns: The activated value.

#### testSerializedSet

```ts
testSerializedSet(
  serializedSampleSet: SerializedSample[],
  cost: (expected: number[], actual: number[]) => number,
  activationValues: number[],
  stateValues: number[],
  serializedNetwork: number[],
  activationFunctions: ActivationFn[],
): number
```

Tests a serialized dataset using a cost function.

Returns: The average error.

#### workers

Workers for multi-threading

## multithreading/types.ts

### ActivationFn

```ts
ActivationFn(
  x: number,
): number
```

Shared types for multithreading helpers and test workers.

### SerializableNetwork

### SerializedSample

### TestWorkerConstructor

### TestWorkerInstance

## multithreading/multi.utils.ts

### absoluteActivation

```ts
absoluteActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Absolute activation.

### activateSerializedNetwork

```ts
activateSerializedNetwork(
  inputValues: number[],
  activationValues: number[],
  stateValues: number[],
  serializedNetwork: number[],
  activationFunctions: ActivationFn[],
): number[]
```

Activates a serialized network and produces outputs.

Parameters:
- `inputValues` - - Inputs to feed into the network.
- `activationValues` - - Mutable activation register shared across runs.
- `stateValues` - - Mutable state register shared across runs.
- `serializedNetwork` - - Flat encoded network data.
- `activationFunctions` - - Ordered activation functions.

Returns: Activated outputs.

### ACTIVATION_FUNCTIONS

Returns: Activation functions ordered for serialization compatibility.

### bentIdentityActivation

```ts
bentIdentityActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Bent identity activation.

### bipolarActivation

```ts
bipolarActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Bipolar activation.

### bipolarSigmoidActivation

```ts
bipolarSigmoidActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Bipolar sigmoid activation.

### deserializeDataSet

```ts
deserializeDataSet(
  serializedSet: number[],
): SerializedSample[]
```

Deserializes a dataset from its flat representation.

Parameters:
- `serializedSet` - - Flat serialized dataset array.

Returns: Array of input/output sample pairs.

### gaussianActivation

```ts
gaussianActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Gaussian activation.

### hardTanhActivation

```ts
hardTanhActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Hard tanh activation.

### identityActivation

```ts
identityActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Identity activation.

### inverseActivation

```ts
inverseActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Inverse activation.

### logisticActivation

```ts
logisticActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Logistic activation.

### reluActivation

```ts
reluActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: ReLU activation.

### seluActivation

```ts
seluActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: SELU activation.

### serializeDataSet

```ts
serializeDataSet(
  dataSet: { input: number[]; output: number[]; }[],
): number[]
```

Serializes a dataset into a flat numeric array.

Parameters:
- `dataSet` - - Collection of samples with input and output arrays.

Returns: Flat serialized representation [inputCount, outputCount, ...samples].

### sinusoidActivation

```ts
sinusoidActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Sinusoid activation.

### softplusActivation

```ts
softplusActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Softplus activation.

### softsignActivation

```ts
softsignActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Softsign activation.

### stepActivation

```ts
stepActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Step activation.

### tanhActivation

```ts
tanhActivation(
  value: number,
): number
```

Parameters:
- `value` - - Input value.

Returns: Hyperbolic tangent activation.

### testSerializedSet

```ts
testSerializedSet(
  serializedSampleSet: SerializedSample[],
  costFunction: (expected: number[], actual: number[]) => number,
  activationValues: number[],
  stateValues: number[],
  serializedNetwork: number[],
  activationFunctions: ActivationFn[],
): number
```

Tests a serialized dataset using a cost function.

Parameters:
- `serializedSampleSet` - - Serialized dataset samples.
- `costFunction` - - Cost function comparing expected and actual outputs.
- `activationValues` - - Mutable activation register.
- `stateValues` - - Mutable state register.
- `serializedNetwork` - - Serialized network data.
- `activationFunctions` - - Activation functions to apply.

Returns: Average cost or NaN when invalid input.
