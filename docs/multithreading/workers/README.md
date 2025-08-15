# multithreading/workers

## multithreading/workers/workers.ts

### Workers

Utility class for managing workers in both Node.js and browser environments.

#### getBrowserTestWorker

`() => Promise<any>`

Loads the browser test worker dynamically.

Returns: A promise that resolves to the browser TestWorker class.

#### getNodeTestWorker

`() => Promise<any>`

Loads the Node.js test worker dynamically.

Returns: A promise that resolves to the Node.js TestWorker class.
