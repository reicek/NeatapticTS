# env/browser

## env/browser/worker-loader.ts

### getBrowserTestWorker

```ts
getBrowserTestWorker(): Promise<TestWorkerConstructor>
```

Resolve the browser worker wrapper from the browser-oriented environment shelf.

Returns: Browser worker constructor.

### getNodeTestWorker

```ts
getNodeTestWorker(): Promise<TestWorkerConstructor>
```

Reject Node worker loading from the browser-oriented environment shelf.

Browser builds should fail with a clear environment error rather than trying
to pull `child_process` or `path` into the bundle.

Returns: Rejected promise describing the unsupported Node worker request.
