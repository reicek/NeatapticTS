# multithreading/workers

## multithreading/workers/workers.ts

### workers

Utility class for managing workers in both Node.js and browser environments.

### Workers

Utility class for managing workers in both Node.js and browser environments.

#### getBrowserTestWorker

`() => Promise<typeof import("C:/NeatapticTS/src/multithreading/workers/browser/testworker").TestWorker>`

Loads the browser test worker dynamically.

Returns: A promise that resolves to the browser TestWorker class.

#### getNodeTestWorker

`() => Promise<typeof import("C:/NeatapticTS/src/multithreading/workers/node/testworker").TestWorker>`

Loads the Node.js test worker dynamically.

Returns: A promise that resolves to the Node.js TestWorker class.
