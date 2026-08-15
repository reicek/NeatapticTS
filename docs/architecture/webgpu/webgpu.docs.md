# WebGPU master reference for NeatapticTS

> Synthesized from three sources captured 2026-06-30: the WebGPU community landing page (`webgpu.org`), the W3C WebGPU specification (`https://www.w3.org/TR/webgpu/`), and the MDN WebGPU API guide (`https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API`).
>
> This document distills the parts of WebGPU that matter for general-purpose GPU compute (GPGPU) inference: adapters, devices, buffers, bind groups, compute pipelines, command encoders, queues, WGSL essentials, limits, error handling, and browser support.

## 1. API model overview

WebGPU is a low-level API that exposes the GPU through a **logical device** abstraction. The browser talks to native GPU APIs (Metal, Direct3D 12, Vulkan) on the app's behalf. The API is exposed to JavaScript in secure contexts (`Window` and `Worker`) as `navigator.gpu`.

Core object types:

- `GPU` — entry point (`navigator.gpu`).
- `GPUAdapter` — represents a physical GPU + driver that the implementation is willing to expose.
- `GPUDevice` — a logical, compartmentalized view of GPU resources; the main workhorse.
- `GPUQueue` — the device's command submission channel.
- `GPUBuffer` — raw GPU memory block.
- `GPUTexture` / `GPUTextureView` — image-like resources (mostly graphics-related, not used for our compute path).
- `GPUShaderModule` — container for WGSL source.
- `GPUComputePipeline` — compiled compute stage.
- `GPUBindGroupLayout` / `GPUBindGroup` — shader resource binding descriptors and runtime bindings.
- `GPUCommandEncoder` / `GPUComputePassEncoder` / `GPUCommandBuffer` — the command recording and submission machinery.

## 2. Adapter and device lifecycle

### 2.1 Feature detection and adapter request

```js
if (!navigator.gpu) {
  throw new Error('WebGPU not supported on this context.');
}

const adapter = await navigator.gpu.requestAdapter({
  powerPreference: 'high-performance', // or 'low-power'
  forceFallbackAdapter: false,
});

if (!adapter) {
  throw new Error('No suitable WebGPU adapter found.');
}
```

- WebGPU only runs in secure contexts (HTTPS or localhost).
- `requestAdapter` is exposed on `navigator.gpu` (and `WorkerNavigator.gpu` in workers).
- The optional `featureLevel: 'compatibility'` requests compatibility mode, which guarantees only a smaller baseline of features/limits. Core mode is preferred for compute.

### 2.2 Requesting a logical device

```js
const device = await adapter.requestDevice({
  requiredFeatures: [],
  requiredLimits: {
    // e.g. 'maxStorageBufferBindingSize': adapter.limits.maxStorageBufferBindingSize,
  },
  defaultQueue: {},
});

// Handle asynchronous loss.
device.lost.then((info) => {
  console.warn('WebGPU device lost:', info.reason, info.message);
});
```

- `GPUDevice` exposes `features`, `limits`, `adapterInfo`, and `queue`.
- `device.destroy()` releases all resources created from this device.
- Once lost, the device is unusable; the application must request a new adapter/device pair.

## 3. Buffer usage and data movement

### 3.1 Creating buffers

```js
const buffer = device.createBuffer({
  size: 1024,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  mappedAtCreation: false,
});
```

Common `GPUBufferUsage` flags for compute:

- `STORAGE` — read/write storage buffer in a shader.
- `UNIFORM` — read-only uniform buffer.
- `COPY_SRC` / `COPY_DST` — source/destination for buffer-to-buffer or buffer-to-texture copies.
- `MAP_READ` / `MAP_WRITE` — CPU-mappable buffers.

### 3.2 Writing and reading buffer data

GPU-side write (recommended for upload):

```js
device.queue.writeBuffer(buffer, 0, typedArray);
```

CPU-side mapping for readback:

```js
const staging = device.createBuffer({
  size: resultSize,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// In a command encoder, copy result buffer to staging.
encoder.copyBufferToBuffer(resultBuffer, 0, staging, 0, resultSize);

// Submit, then map asynchronously.
await device.queue.submit([encoder.finish()]);
await staging.mapAsync(GPUMapMode.READ);
const copy = staging.getMappedRange(0, resultSize).slice();
staging.unmap();
const data = new Float32Array(copy);
```

Mapping rules:

- `mapAsync` returns a `Promise` and transitions the buffer through `pending` → `mapped`.
- Only one map mode at a time (`READ` or `WRITE`).
- A buffer must have the matching `MAP_READ` / `MAP_WRITE` usage and must not be used on the GPU while mapped.
- `getMappedRange` returns an `ArrayBuffer` view; `unmap` must be called before GPU use resumes.

## 4. WGSL compute shader essentials

WebGPU Shading Language (WGSL) is the only shading language. A compute shader entry point is annotated with `@compute` and a `@workgroup_size`.

```wgsl
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
  let idx = global_id.x;
  if (idx >= arrayLength(&output)) { return; }
  output[idx] = output[idx] * 2.0;
}
```

Key concepts:

- **Workgroup size** (`@workgroup_size(x, y, z)`): threads within one workgroup. The product `x*y*z` must be `<= maxComputeInvocationsPerWorkgroup`, and each dimension `<= maxComputeWorkgroupSizeX/Y/Z`.
- **Dispatch**: `passEncoder.dispatchWorkgroups(x, y, z)`. Total threads in dimension X (`x * workgroup_size_x`) must be `<= maxComputeWorkgroupsPerDimension`.
- **Builtins**: `global_invocation_id`, `local_invocation_id`, `workgroup_id`, `num_workgroups`, `workgroup_size`.
- **Buffer binding**: `@group(N) @binding(M) var<storage, read|read_write|write> name: ...` or `var<uniform>`.
- **Types**: `f32`, `i32`, `u32`, `vec2/3/4<f32>`, `vec3u`, `array<T>`, `matCxR<T>`. No 64-bit floats.
- **Math intrinsics**: `exp`, `log`, `pow`, `sqrt`, `sin`, `cos`, `tanh`, `abs`, `clamp`, `min`, `max`, `select`, etc. See the WGSL spec for the complete list.

## 5. Bind groups and pipeline layout

A compute pipeline must know which buffers to use. Bind groups declare that mapping.

```js
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    {
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' }, // or 'read-only-storage', 'uniform'
    },
  ],
});

const pipelineLayout = device.createPipelineLayout({
  bindGroupLayouts: [bindGroupLayout],
});

const bindGroup = device.createBindGroup({
  layout: bindGroupLayout,
  entries: [
    {
      binding: 0,
      resource: { buffer: outputBuffer },
    },
  ],
});
```

Layout compatibility:

- The pipeline's layout (or auto-generated layout) must be compatible with the bind group layout used to create the bind group.
- `layout: 'auto'` lets WebGPU infer the layout from the shader, but explicit layouts are recommended for stable shader/buffer contracts.
- A pipeline can use multiple bind groups (`@group(0)`, `@group(1)`, ... up to `maxBindGroups - 1`).

## 6. Compute pipeline and command lifecycle

The full lifecycle from device to execution:

```js
// 1. Create shader module.
const shaderModule = device.createShaderModule({ code: wgslSource });

// 2. Create compute pipeline.
const pipeline = device.createComputePipeline({
  layout: pipelineLayout,
  compute: { module: shaderModule, entryPoint: 'main' },
});

// 3. Record commands.
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
pass.end();

// 4. Submit.
device.queue.submit([encoder.finish()]);
```

Async pipeline creation (`createComputePipelineAsync`) compiles on a background thread and is preferred to avoid jank.

## 7. Error model

WebGPU errors are **asynchronous and contagious**:

- Validation errors are caught on the GPU process/process side, not at the JavaScript call site.
- A call that fails returns an invalid object; subsequent operations that depend on it also become invalid.
- Use error scopes (`pushErrorScope` / `popErrorScope`) to catch validation errors.
- Listen to `uncapturederror` events for errors not captured by scopes (including `GPUOutOfMemoryError`).
- `device.lost` resolves if the device is lost (e.g. GPU reset, tab backgrounded, driver crash).

```js
device.pushErrorScope('validation');
const buffer = device.createBuffer({ size: huge, usage: ... });
const error = await device.popErrorScope();
if (error) console.error(error.message);

device.addEventListener('uncapturederror', event => {
  console.error('Uncaptured WebGPU error:', event.error.message);
});
```

## 8. Limits that matter for GPGPU inference

Relevant default/core limits for large neural-network workloads:

| Limit                               | Typical default / concern                                    |
| ----------------------------------- | ------------------------------------------------------------ |
| `maxStorageBuffersPerShaderStage`   | How many storage buffers one shader can bind simultaneously. |
| `maxStorageBufferBindingSize`       | Maximum byte size of a single storage buffer binding.        |
| `maxBufferSize`                     | Maximum total buffer allocation.                             |
| `maxBindGroups`                     | How many `@group` slots are available.                       |
| `maxBindingsPerBindGroup`           | How many `@binding` slots per group.                         |
| `maxComputeWorkgroupStorageSize`    | Shared memory (`var<workgroup>`) limit.                      |
| `maxComputeInvocationsPerWorkgroup` | `x*y*z` threads per workgroup.                               |
| `maxComputeWorkgroupSizeX/Y/Z`      | Per-dimension workgroup size.                                |
| `maxComputeWorkgroupsPerDimension`  | Per-dimension dispatch grid size.                            |
| `minStorageBufferOffsetAlignment`   | Alignment for dynamic offsets.                               |

These limits are exposed on `adapter.limits` and `device.limits`. Requesting a higher value than the adapter supports throws a validation error.

## 9. Browser compatibility and testing

- **Chrome/Edge**: stable since Chrome 113 (with continued feature rollout).
- **Firefox**: enabled by default in recent releases; implementation status tracked in the WebGPU Implementation Status wiki.
- **Safari**: supported in Safari Technology Preview / recent macOS releases.
- **Workers**: supported via `WorkerNavigator.gpu` in dedicated workers.
- **Secure context required**: `https://`, `localhost`, or `file://` depending on browser.
- **Headless/CI**: browser-based smoke tests require a real browser with WebGPU. Node.js does not ship a built-in WebGPU backend, though Dawn bindings exist for native/Node usage outside the web platform. For CI, mocked device tests plus browser smoke validation are the recommended first pass.
- **Feature detection**: check `navigator.gpu`, then `adapter.features.has('...')` for optional features such as `shader-f16`, `timestamp-query`, `subgroups`, etc.

## 10. GPGPU patterns for batch compute

For batch inference across many agents:

1. Keep a single `GPUDevice` and reuse one compiled compute pipeline.
2. Use one large storage buffer per logical tensor/Slab (weights, CSR indices, activations, outputs).
3. Encode all dispatches for one frame/tick into one `GPUCommandEncoder` and submit once.
4. Use `writeBuffer` to upload only changed data each tick.
5. Keep an output buffer GPU-resident; copy to a mappable staging buffer only when CPU needs the result.
6. Use batched dispatch dimensions to run one network per workgroup or one neuron/edge per thread, depending on topology.

## 11. References

- WebGPU community page: https://webgpu.org/
- W3C WebGPU specification: https://www.w3.org/TR/webgpu/
- MDN WebGPU API guide: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- WGSL specification: https://gpuweb.github.io/gpuweb/wgsl/
- WebGPU samples: https://webgpu.github.io/webgpu-samples/
- WebGPU Fundamentals: https://webgpufundamentals.org/
- WebGPU best practices: https://toji.dev/webgpu-best-practices/
