# multithreading/workers/node

## multithreading/workers/node/worker.ts

### WorkerMessage

Handles messages sent to the worker process.

This function listens for messages sent to the worker process and performs one of two actions:
1. If the message contains serialized activations, states, and connections, it evaluates the network using the dataset.
2. If the message contains a dataset and cost function, it initializes the worker with the provided data.

Parameters:
- `e` - - The message object sent to the worker process. It can contain:
- `set`: Serialized dataset to initialize the worker. This is an array of objects with `input` and `output` properties.
- `cost`: The name of the cost function to use. This should match a key in the `methods.Cost` object.
- `activations`: Serialized activation values for the network.
- `states`: Serialized state values for the network.
- `conns`: Serialized connection data for the network.

## multithreading/workers/node/testworker.ts

### CostFunction

Interface for cost function used in worker evaluation.

### SerializableNetwork

Interface for serializable network used in worker evaluation.

### TestWorker

TestWorker class for handling network evaluations in a Node.js environment using Worker Threads.

This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
neural networks in parallel environments. The use of Worker Threads allows for offloading
computationally expensive tasks, such as network evaluation, to separate threads.

#### evaluate

`(network: SerializableNetwork) => Promise<number>`

Evaluates a neural network using the worker process.

The network is serialized and sent to the worker for evaluation. The worker
sends back the evaluation result, which is returned as a promise.

Parameters:
- `` - - The neural network to evaluate. It must implement a `serialize` method.

Returns: A promise that resolves to the evaluation result.

#### terminate

`() => void`

Terminates the worker process.

This method ensures that the worker process is properly terminated to free up system resources.
