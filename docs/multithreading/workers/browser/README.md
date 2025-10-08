# multithreading/workers/browser

## multithreading/workers/browser/testworker.ts

### CostFunction

Interface for cost function used in worker evaluation.

### SerializableNetwork

Interface for serializable network used in worker evaluation.

### TestWorker

TestWorker class for handling network evaluations in a browser environment using Web Workers.

This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
neural networks in parallel environments. The use of Web Workers allows for offloading
computationally expensive tasks, such as network evaluation, to separate threads.

#### _createBlobString

`(cost: CostFunction) => string`

Creates a string representation of the worker's blob.

Parameters:
- `` - - The cost function to be used by the worker.

Returns: The blob string.

#### evaluate

`(network: SerializableNetwork) => Promise<number>`

Evaluates a network using the worker process.

Parameters:
- `` - - The network to evaluate.

Returns: A promise that resolves to the evaluation result.

#### terminate

`() => void`

Terminates the worker process and revokes the object URL.
