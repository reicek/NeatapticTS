# multithreading/workers

Utility class for managing workers in both Node.js and browser environments.

## multithreading/workers/workers.ts

### Workers

Utility class for managing workers in both Node.js and browser environments.

#### getBrowserTestWorker

```ts
getBrowserTestWorker(): Promise<typeof TestWorker>
```

Loads the browser test worker dynamically.

Returns: A promise that resolves to the browser TestWorker class.

#### getNodeTestWorker

```ts
getNodeTestWorker(): Promise<typeof TestWorker>
```

Loads the Node.js test worker dynamically.

Returns: A promise that resolves to the Node.js TestWorker class.
