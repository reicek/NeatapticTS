# multithreading/workers/node

## multithreading/workers/node/testworker.ts

### TestWorker

TestWorker class for handling network evaluations in a Node.js environment using Worker Threads.

This implementation aligns with the Instinct algorithm's emphasis on efficient evaluation of
neural networks in parallel environments. The use of Worker Threads allows for offloading
computationally expensive tasks, such as network evaluation, to separate threads.

#### evaluate

`(network: any) => Promise<number>`

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
