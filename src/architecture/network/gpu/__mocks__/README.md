# architecture/network/gpu/__mocks__

Reusable WebGPU mock helpers for owner-local GPU tests.

The returned devices are intentionally shallow: they record the calls the
implementation is expected to make without pulling in a real WebGPU backend.
Optionally, a mock device can emulate a CPU forward pass so that parity
tests can compare GPU read-back values against the CPU source of truth.

## architecture/network/gpu/__mocks__/gpu.mock.ts

### computeMockForwardPass

```ts
computeMockForwardPass(
  entries: { binding: number; resource: { buffer: GPUBuffer; }; }[],
  nodeCount: number,
): Float32Array<ArrayBufferLike>
```

Compute the pre-activation values for every non-input node.

This mirrors the struct-packed gather-and-activate kernel: each node's
pre-activation value is the sum of incoming weights times the current source
activations (read from the bound node buffer) plus the node's bias. Input
nodes are never processed by a dispatch, so they retain the values uploaded
by the caller.

### computeMockLevels

```ts
computeMockLevels(
  connections: MockConnection[],
  nodeCount: number,
): Uint32Array<ArrayBufferLike>
```

Compute a simple topological level for every node from the connection list.

Level 0 contains nodes with no incoming connections; every other node gets
one level above its highest predecessor. This is sufficient for the mock
because GPU-eligible networks are feed-forward and the real kernel already
skips disabled connections via its flags check.

### createMockGPUDevice

```ts
createMockGPUDevice(
  options: any,
): MockGPUDevice
```

Build a fake WebGPU device that records every call made by the inference path.

Parameters:
- `options` - Optional limits, emulator network, or output generator.

### createMockNavigatorGPU

```ts
createMockNavigatorGPU(
  options: { adapter?: any; device?: any; } | undefined,
): Navigator & { gpu: GPU; }
```

Build a fake `navigator` object that exposes a mock `gpu` property.

### MockBuffer

Narrow a generic GPUBuffer to the mock's internal backing-store shape.

### MockConnection

Connection struct mirrored from the WGSL `Connection` layout.

### MockGenerateOutput

```ts
MockGenerateOutput(
  context: MockGenerateOutputContext,
): number[] | Float32Array<ArrayBufferLike>
```

Optional callback that can write deterministic output values into the mock.

### MockGenerateOutputContext

Context supplied to the optional per-dispatch output generator.

### MockGPUDevice

### MockGPUDeviceOptions

### MockGPURecordings

### MockParams

Per-dispatch params mirrored from the WGSL `Params` layout.

### readBoundConnections

```ts
readBoundConnections(
  entries: { binding: number; resource: { buffer: GPUBuffer; }; }[],
): MockConnection[]
```

Read the bound connection buffer as an array of connection structs.

### readBoundFloatArray

```ts
readBoundFloatArray(
  entries: { binding: number; resource: { buffer: GPUBuffer; }; }[],
  binding: number,
): Float32Array<ArrayBufferLike>
```

Read a bound buffer as a typed array, returning an empty array if the buffer
is not a mock-backed GPUBuffer.

### readBoundParams

```ts
readBoundParams(
  entries: { binding: number; resource: { buffer: GPUBuffer; }; }[],
): MockParams | undefined
```

Read the bound params uniform as a typed struct.
