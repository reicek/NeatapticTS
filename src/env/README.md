# env

Default environment adapter entry for repo builds and Node-oriented tests.

Browser-target bundlers should alias this module to the browser adapter so
Node-only worker code is compiled out at build time rather than selected at
runtime.

## env/index.ts

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
