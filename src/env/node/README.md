# env/node

## env/node/worker-loader.ts

### getBrowserTestWorker

```ts
getBrowserTestWorker(): Promise<TestWorkerConstructor>
```

Resolve the browser worker wrapper from the Node-oriented environment shelf.

Keeping this available preserves the current Jest and jsdom validation path,
where browser worker code is exercised inside a Node host runtime.

Returns: Browser worker constructor.

### getNodeTestWorker

```ts
getNodeTestWorker(): Promise<TestWorkerConstructor>
```

Resolve the Node worker wrapper from the Node-oriented environment shelf.

Returns: Node worker constructor.
