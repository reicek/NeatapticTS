# architecture/network/gpu/__mocks__

Reusable WebGPU mock helpers for owner-local GPU tests.

The returned devices are intentionally shallow: they record the calls the
implementation is expected to make without pulling in a real WebGPU backend.
Optionally, a mock device can emulate a CPU forward pass so that parity
tests can compare GPU read-back values against the CPU source of truth.

## architecture/network/gpu/__mocks__/gpu.mock.ts

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
