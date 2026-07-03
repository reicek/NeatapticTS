# architecture/network/gpu

WebGPU single-network activation seam.

This folder implements the optional GPU fast path for `Network.activate`.
Neural networks in NeatapticTS normally run on the CPU, which is the
deterministic source of truth for evolution, replay, and cross-machine
benchmarks. When the runtime has a working WebGPU device and the network is
eligible, the same forward pass can be dispatched to the GPU for higher
throughput. The seam is deliberately thin: this module wires together the
kernel compiler, buffer upload, and device probe so each piece stays focused.

The high-level entry point is `activateGPU`. Most callers should use the
public `Network.activate(..., { useGPU: true })` overload instead, because
it automatically falls back to the CPU path when the network or device is
ineligible.

GPU output is expected to agree with the CPU path within an absolute tolerance
of `5e-1` and a mean absolute error of `≤ 1e-1`. Use the CPU path for
deterministic replay and cross-machine regression tests.

```mermaid
flowchart TD
    A["Network.activate(input, { useGPU: true })"] --> B{isGPUEligible?}
    B -->|yes| C[Upload slab to GPU]
    C --> D[Compile / cache kernel]
    D --> E[Dispatch compute]
    E --> F[Read back outputs]
    F --> G([Float32Array])
    B -->|no| H[CPU activate]
    H --> G
```

## architecture/network/gpu/network.gpu.activate.ts

### activateGPU

```ts
activateGPU(
  device: GPUDevice,
  network: default,
  inputs: number[] | Float32Array<ArrayBufferLike>,
): Promise<Float32Array<ArrayBufferLike>>
```

Run a single-network forward pass on the supplied WebGPU device.

Parameters:
- `device` - WebGPU device used to run the forward kernel.
- `network` - Network whose fast-slab topology will be uploaded.
- `inputs` - Input vector of length `network.input`.

Returns: A promise resolving to a Float32Array of output-node values.

Example:

```ts
const adapter = await navigator.gpu.requestAdapter({
  powerPreference: 'high-performance',
});
const device = await adapter?.requestDevice();
if (device) {
  const output = await activateGPU(device, network, [0.5, -0.2]);
}
```

### dispatchActivationKernel

```ts
dispatchActivationKernel(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
  pipeline: GPUComputePipeline,
): Promise<void>
```

Create the bind group for the supplied compiled pipeline and dispatch the
kernel for every node in the network.

### GPUCommandEncoderCopy

Local extension of the ambient GPU command encoder so we can copy a storage
buffer to a mappable staging buffer. The WebGPU ambient types in this repo are
intentionally minimal; the cast is justified because `copyBufferToBuffer` is
part of the actual WebGPU API surface.

### prepareActivationContext

```ts
prepareActivationContext(
  network: default,
): { index: number; restore: () => void; }
```

Temporarily annotate the first node's squash with its worker-registry index
so `compileActivationKernel` can generate the correct WGSL switch, then restore
the original value.

We only set the index during compilation because `uploadNetworkToGPU` uses an
empty supported-activation set in its eligibility check and would otherwise
reject networks whose nodes carry any index. Restoring the original value
keeps the mutation scoped to this seam.

Parameters:
- `network` - Network whose first node squash will be temporarily annotated.

Returns: A context object with the resolved index and a `restore()` callback.

### readOutputValues

```ts
readOutputValues(
  device: GPUDevice,
  network: default,
  bufferSet: GPUBufferSet,
): Promise<Float32Array<ArrayBufferLike>>
```

Copy the output-node slice of the GPU output buffer to a mappable staging
buffer, await the mapping, and return a detached Float32Array copy.

### resolveActivationIndex

```ts
resolveActivationIndex(
  squash: (value: number, derivate?: boolean | undefined) => number,
): number | undefined
```

Look up the worker-registry activation index for a built-in squash function.

The lookup is intentionally robust across module-loading boundaries and mock
environments where the same activation may be imported from different source
files and therefore fails a strict `===` comparison. It first tries the
runtime-registry symbol key, then falls back to the function name.

Parameters:
- `squash` - Activation function attached to a node.

Returns: The corresponding worker index, or `undefined` when the function is
not part of the canonical registry.

### writeInputValues

```ts
writeInputValues(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
  inputs: Float32Array<ArrayBufferLike>,
): void
```

Write the input vector into the first `network.input` slots of the GPU
output buffer.

## architecture/network/gpu/network.gpu.fallback.ts

Transparent CPU fallback and GPU eligibility predicate.

This module decides whether the public `Network.activate(..., { useGPU: true })`
overload can safely take the WebGPU fast path, and provides a standalone
dispatch helper that falls back to the CPU path automatically when the GPU
path is unavailable. Keeping the eligibility decision in one place ensures
the public CPU seam and any direct GPU dispatch helpers agree on when the
GPU path is safe to use.

### dispatchActivation

```ts
dispatchActivation(
  network: default,
  inputs: number[] | Float32Array<ArrayBufferLike>,
  device: any,
): Promise<Float32Array<ArrayBufferLike>>
```

Transparent single-network activation seam.

Dispatches to the WebGPU fast path when the network and device are eligible,
otherwise falls back to the CPU `network.activate()` implementation. Both
paths return a `Float32Array` of the same output length so callers do not
need to know which path was taken.

Eligibility is evaluated by `isGPUEligible`: it rejects missing or
lost devices, networks with float32 weights disabled, and structurally
unsupported networks.

Parameters:
- `network` - Network to activate.
- `inputs` - Input vector of length `network.input`.
- `device` - Optional WebGPU device. When null, missing, lost, or the
network is ineligible, the CPU path is used.

Returns: Promise resolving to the network output.

Example:

```ts
const adapter = await navigator.gpu.requestAdapter({
  powerPreference: 'high-performance',
});
const device = await adapter?.requestDevice();
const output = await dispatchActivation(network, [0.5, -0.2], device);
// output is Float32Array from GPU if eligible, otherwise from CPU
```

### isGPUEligible

```ts
isGPUEligible(
  network: default,
  device: any,
): boolean
```

Shared GPU eligibility predicate used by the single-network fallback seam and
by `Network.activate`. A network is eligible only when:

- a usable WebGPU device is present and has not been lost,
- the network stores slab weights in float32 (the GPU kernel is f32-only),
- `canUseGPU` reports the network is structurally eligible.

This keeps the fallback decision in one place so the public CPU seam and the
standalone dispatch seam agree on when the GPU path is safe to use.

Parameters:
- `network` - Network to evaluate for GPU inference.
- `device` - WebGPU device, or null/undefined when WebGPU is unavailable.

Returns: Type guard that narrows device to GPUDevice when true.

Example:

```ts
const adapter = await navigator.gpu.requestAdapter({
  powerPreference: 'high-performance',
});
const device = await adapter?.requestDevice();
if (isGPUEligible(network, device)) {
  // device is narrowed to GPUDevice here
  const output = await activateGPU(device, network, inputs);
}
```

## architecture/network/gpu/network.gpu.capability.ts

### canUseGPU

```ts
canUseGPU(
  network: default,
  device: any,
  supportedActivations: ReadonlySet<number>,
): boolean
```

Minimum GPU eligibility predicate.

Decides whether a network is structurally eligible for the GPU inference
path without requiring a real WebGPU backend. It rejects missing devices,
networks with gating connections, networks that contain self-connections,
and nodes whose activation index is both present and unsupported. Buffer
sizing is only coarsely estimated so the predicate can run against mock
devices as well as real hardware.

This predicate is one input to `isGPUEligible`, which also checks
device readiness and the float32 slab flag. Most callers should use
`isGPUEligible` rather than calling `canUseGPU` directly.

Parameters:
- `network` - Network to evaluate for GPU inference.
- `device` - WebGPU device, or null when WebGPU is unavailable.
- `supportedActivations` - Worker-registry activation indices the GPU
kernel supports. Nodes without an explicit index are skipped so networks
built from high-level constructors can still be evaluated; nodes without a
squash function are also skipped so `activateGPU` can report the
missing-squash error with its own message.

Returns: True when the network is structurally eligible for the GPU path.

Example:

```ts
const supported = new Set<number>([0, 1, 2, 3]);
const eligible = canUseGPU(network, device, supported);
```

## architecture/network/gpu/network.gpu.device.ts

Module-local tracking of which WebGPU devices have reported lost.

WebGPU only exposes loss through an async `device.lost` promise, so the
runtime keeps a WeakMap that is flipped to `true` when that promise resolves.
This allows `isDeviceReady` to give a synchronous answer without polling the
GPU process.

### isDeviceReady

```ts
isDeviceReady(
  device: any,
): boolean
```

Returns true when the supplied WebGPU device is present and has not been
reported lost.

`null` and `undefined` inputs are treated as not-ready so callers can safely
chain GPU probing with CPU fallback logic. Device loss is tracked through
the module-local WeakMap populated by `requestGPUDevice` and by lazy
attachment on the first call to this function.

Parameters:
- `device` - WebGPU device to check, or a falsy value when no GPU exists.

Returns: `true` only when a non-null device is available and not lost.

### requestGPUDevice

```ts
requestGPUDevice(): Promise<any>
```

Request a high-performance WebGPU device suitable for compute inference.

Probes `navigator.gpu`, requests a `high-performance` adapter, then asks
the adapter for a device whose limits match the adapter's reported limits for
`maxStorageBufferBindingSize` and `maxBufferSize`. Returns `null` safely in
non-browser environments, when WebGPU is unavailable, when no adapter can be
obtained, or when device creation is rejected.

Returns: A ready-to-use `GPUDevice`, or `null` when WebGPU cannot be used.

Example:

```ts
const device = await requestGPUDevice();
if (device) {
  network.gpuDevice = device;
}
```

### trackDevice

```ts
trackDevice(
  device: GPUDevice,
): void
```

Attach a one-shot listener to `device.lost` so the module can later answer
whether the device is still usable.

## architecture/network/gpu/network.gpu.kernel.ts

### buildGPUPipeline

```ts
buildGPUPipeline(
  device: GPUDevice,
  shaderModule: GPUShaderModule,
  bindGroupLayout: GPUBindGroupLayout,
): GPUComputePipeline
```

Build a compute pipeline from a shader module and a bind-group layout.

The pipeline uses the `forward` compute entry point and a pipeline layout
built from the supplied bind-group layout. It is the factory used by
`compileActivationKernel` to materialize the compiled GPU path.

Parameters:
- `device` - WebGPU device used to create the pipeline layout and pipeline.
- `shaderModule` - Shader module containing the `forward` entry point.
- `bindGroupLayout` - Layout describing the kernel's storage buffers.

Returns: A compute pipeline configured for the forward-pass kernel.

Example:

```ts
const pipeline = buildGPUPipeline(device, shaderModule, bindGroupLayout);
```

### compileActivationKernel

```ts
compileActivationKernel(
  device: GPUDevice,
  network: default,
): GPUComputePipeline
```

Compile (or reuse) the activation compute pipeline for a network topology.

The pipeline is created once per unique topology and cached on the supplied
device. Recompilations with identical topology but different weights reuse
the cached `GPUComputePipeline`, avoiding redundant compile stalls during
live inference. The shader module, bind-group layout, and pipeline creation
calls remain observable through a mock device for unit testing.

Parameters:
- `device` - WebGPU device used to compile the compute pipeline.
- `network` - Network whose topology and activation index drive the kernel.

Returns: A compute pipeline configured for the `forward` entry point.

Example:

```ts
const pipeline = compileActivationKernel(device, network);
```

### computeTopologyKey

```ts
computeTopologyKey(
  network: default,
  activationIndex: number,
): string
```

Compute a deterministic key that identifies the network topology.

The key includes node and connection counts plus the CSR from/to arrays when
they are available. Networks that differ only in weights therefore share a
key, which is exactly the condition that lets the pipeline cache reuse the
same compiled shader.

Parameters:
- `network` - Network whose topology will be hashed.
- `activationIndex` - Activation index that changes the generated shader.

Returns: A stable string key for the pipeline cache.

### createActivationKernel

```ts
createActivationKernel(
  network: default,
): string
```

Generate the WGSL source for the activation kernel of a supported network.

The returned source is a real, bindable compute shader: it declares the
seven storage-buffer bindings, one f32 activation function per supported
worker index, and a `forward` entry point that dispatches one thread per
node. Unsupported activations or ineligible topologies are rejected before
any source is emitted.

Parameters:
- `network` - Network whose activation index and topology are inspected.

Returns: Non-empty WGSL source string.

Example:

```ts
const source = createActivationKernel(network);
const module = device.createShaderModule({ code: source });
```

### createBindGroupLayout

```ts
createBindGroupLayout(
  device: GPUDevice,
): GPUBindGroupLayout
```

Create the bind-group layout used by the GPU forward-pass kernel.

The layout exposes seven storage-buffer entries in the exact order expected
by the slab-to-GPU upload contract: connection weights, source node ids,
target node ids, connection flags, CSR output-start offsets, CSR output
order, and the per-node output buffer.

Parameters:
- `device` - WebGPU device used to create the layout.

Returns: A bind-group layout with seven storage-buffer entries.

Example:

```ts
const bindGroupLayout = createBindGroupLayout(device);
```

### generateActivationSource

```ts
generateActivationSource(
  activationIndex: number,
  nodeCount: number,
): string
```

Build the WGSL source for the forward-pass activation kernel.

The shader exposes the seven storage-buffer bindings from the slab upload
contract, declares a real f32 activation function for every supported worker
index, and dispatches one thread per node in topological order. The current
kernel applies the network's canonical activation index to the per-node output
buffer; full weighted fan-out is intentionally left for the parity slice that
owns the complete forward pass.

Parameters:
- `activationIndex` - Activation index that must appear as a switch case.
- `nodeCount` - Number of nodes in the network; bounds the dispatch.

Returns: WGSL source string.

### getDevicePipelineCache

```ts
getDevicePipelineCache(
  device: GPUDevice,
): Map<string, GPUComputePipeline>
```

Return the per-device pipeline cache map, creating it on first use.

### InternalSlabNetwork

Internal slab-backed shape used only to read CSR source/target arrays for
topology hashing. Optional because some callers pass network shapes that do
not carry slab state.

### mixHash

```ts
mixHash(
  hash: number,
  value: number,
): number
```

Mix one integer into a simple 32-bit rolling hash.

### readActivationIndex

```ts
readActivationIndex(
  network: default,
): number
```

Read the activation index that the network would use from its first
computation node.

The fast-slab CPU path assigns a stable activation-function index to every
node's `squash` function. The GPU kernel mirrors that index in a WGSL switch.

Parameters:
- `network` - Network whose first node's activation index will drive the
kernel switch.

Returns: The activation index stored on the first node's squash function.

### SUPPORTED_ACTIVATION_INDICES

Ordered subset of worker activation indices that the first WGSL kernel
supports.

These positions must stay in sync with `ACTIVATION_FUNCTIONS` in
`src/multithreading/multi.utils.ts` because DNA, workers, and the GPU kernel
all use the same numeric index for the same activation.

## architecture/network/gpu/network.gpu.buffer.ts

### createGPUBuffer

```ts
createGPUBuffer(
  device: GPUDevice,
  byteLength: number,
  label: string,
  usage: number,
): GPUBuffer
```

Create a WebGPU storage buffer that can receive `queue.writeBuffer` uploads.

Every buffer produced by the upload path must be usable as a read-only
(or read-write) storage binding and as a copy destination. This helper
validates the request against the device's binding and buffer size limits
before delegating to `device.createBuffer`.

Parameters:
- `device` - WebGPU device used to allocate the buffer.
- `byteLength` - Desired buffer size in bytes. Must be finite and
non-negative.
- `label` - Debug label attached to the buffer.
- `usage` - Additional usage flags merged with the mandatory
`STORAGE | COPY_DST` bits. Defaults to no extra flags.

Returns: A freshly created `GPUBuffer` with the mandatory usage bits set.

### destroyGPUBufferSet

```ts
destroyGPUBufferSet(
  device: GPUDevice,
  bufferSet: GPUBufferSet,
): void
```

Destroy every GPU buffer in a previously uploaded buffer set.

Parameters:
- `device` - WebGPU device that owns the buffers (unused by this helper,
kept in the signature for API symmetry).
- `bufferSet` - Buffer set returned by `uploadNetworkToGPU`.

### GPUBufferSet

GPU-side buffer handles and metadata produced by uploading a network slab.

The implementation creates one WebGPU buffer per slab/activation array
and records `nodeCount`/`connectionCount` so the compute pipeline can size its
dispatches without re-reading CPU structures.

### NetworkSlabInternals

Internal network state used to read the CSR adjacency arrays produced by
the fast-slab path. The cast is intentional: GPU upload is a consumer of the
same private layout that slab activation uses.

### uploadNetworkToGPU

```ts
uploadNetworkToGPU(
  device: GPUDevice,
  network: default,
): GPUBufferSet
```

Upload a network's fast-slab structures to WebGPU buffers.

The upload path reuses the existing CPU slab arrays without
re-serialization: it creates one `GPUBuffer` per slab/CSR array via
`createGPUBuffer`, writes each slab exactly once with
`queue.writeBuffer`, and returns the buffer handles plus node/connection
counts. The buffer order matches `GPU_BUFFER_BINDING` so the compute
kernel can bind them with stable indices.

Parameters:
- `device` - Mock or real WebGPU device used to allocate buffers.
- `network` - Network whose fast-slab layout will be uploaded.

Returns: Handles for the uploaded slab buffers and network metadata.

## architecture/network/gpu/network.gpu.batched.ts

Batched WebGPU activation for multi-agent evaluation.

This module evaluates many networks in a single GPU dispatch, which is useful
when the racing-curriculum worker or another demo needs to score a whole
generation at once. Networks with the same topology share compiled pipelines,
and the output is returned as a row-major matrix with one row per network.

The seam remains opt-in: callers must supply a usable `GPUDevice` and every
network must pass the same structural eligibility checks used by the
single-network GPU path. Ineligible networks or missing hardware fall back
to per-network CPU activation through `evaluateRacingGeneration` or a
caller-local fallback.

### batchActivate

```ts
batchActivate(
  device: GPUDevice,
  networks: default[],
  inputMatrix: Float32Array<ArrayBufferLike>,
): Promise<BatchedGPUResult>
```

Batched GPU activation for multi-agent evaluation.

Uploads the input matrix and every network's fast-slab topology to the GPU,
reuses compiled pipelines for networks that share topology, dispatches all
networks in a single compute pass, and reads back one output row per network
into a row-major result matrix. The CPU path remains the default; this seam
is opt-in and gated by `canUseGPU`.

Parameters:
- `device` - WebGPU device used to run the forward kernel.
- `networks` - Networks to evaluate as a batch. All networks must have the
same input and output dimensions.
- `inputMatrix` - Flattened row-major inputs, length
`networks.length * networks[0].input`.

Returns: Promise resolving to a row-major output matrix.

Example:

```ts
const networks = Array.from({ length: 4 }, () => Network.createMLP(2, [3], 1));
const inputs = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
const { outputs, rowCount, colCount } = await batchActivate(device, networks, inputs);
```

### BatchedGPUResult

Result shape returned by a batched GPU activation pass.

The output matrix is stored in row-major order so that downstream consumers
(such as the racing-curriculum worker controller) can slice one row per
agent without extra re-layout.

### createBindGroup

```ts
createBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bufferSet: GPUBufferSet,
): GPUBindGroup
```

Create the bind group for the supplied compiled pipeline and uploaded buffer
set.

Parameters:
- `device` - WebGPU device used to create the bind group.
- `pipeline` - Compiled activation pipeline.
- `bufferSet` - Uploaded network slab buffers.

Returns: A bind group wired to the seven storage-buffer bindings.

### GPUCommandEncoderCopy

Local extension of the ambient GPU command encoder so we can copy a storage
buffer to a mappable staging buffer. The WebGPU ambient types in this repo
are intentionally minimal; the cast is justified because `copyBufferToBuffer`
is part of the actual WebGPU API surface.

### prepareActivationContext

```ts
prepareActivationContext(
  network: default,
): { restore: () => void; }
```

Temporarily annotate the first node's squash with its worker-registry index
so `compileActivationKernel` can generate the correct WGSL switch, then
restore the original value.

Parameters:
- `network` - Network whose first node squash will be temporarily annotated.

Returns: A context object with a `restore()` callback.

### resolveActivationIndex

```ts
resolveActivationIndex(
  squash: (value: number, derivate?: boolean | undefined) => number,
): number | undefined
```

Look up the worker-registry activation index for a built-in squash function.

The lookup is intentionally robust across module-loading boundaries and mock
environments where the same activation may be imported from different source
files and therefore fails a strict `===` comparison. It first tries strict
identity, then the runtime-registry symbol key, then falls back to the
function name.

Parameters:
- `squash` - Activation function attached to a node.

Returns: The corresponding worker index, or `undefined` when the function is
not part of the canonical registry.

### validateBatchInputs

```ts
validateBatchInputs(
  device: GPUDevice,
  networks: default[],
  inputMatrix: Float32Array<ArrayBufferLike>,
): void
```

Validate the batching contract before any GPU work is issued.

Parameters:
- `device` - WebGPU device that will run the dispatch.
- `networks` - Networks to evaluate as a batch.
- `inputMatrix` - Flattened row-major input matrix.

### validateNetworkShapes

```ts
validateNetworkShapes(
  networks: default[],
): void
```

Ensure every network in the batch has the same input and output dimensions.

The result matrix is row-major with one column count for the entire batch, so
mixed shapes would corrupt the layout.

Parameters:
- `networks` - Networks to validate.

## architecture/network/gpu/network.gpu.racing.ts

### evaluateOnCPU

```ts
evaluateOnCPU(
  networks: default[],
  inputMatrix: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBufferLike>
```

Fall back to per-network CPU activation and stack the results.

The output matrix is laid out row-major, with one row per network. Each row
contains the output values returned by `network.activate()` for the
corresponding input slice.

Parameters:
- `networks` - Networks to evaluate in CPU mode.
- `inputMatrix` - Flattened row-major input matrix.

Returns: Row-major Float32Array of stacked network outputs.

### evaluateRacingGeneration

```ts
evaluateRacingGeneration(
  networks: default[],
  inputMatrix: Float32Array<ArrayBufferLike>,
  device: any,
  options: RacingBatchOptions | undefined,
): Promise<Float32Array<ArrayBufferLike>>
```

Racing-curriculum generation evaluation seam.

Evaluates a generation of controller networks against a row-major input
matrix. Uses the batched GPU path when the batch size is above the threshold,
every network is GPU-eligible, and a valid device is supplied; otherwise falls
back to per-network CPU `network.activate()` calls.

Parameters:
- `networks` - One network per car / genome in the generation.
- `inputMatrix` - Flattened row-major inputs, length
`networks.length * networks[0].input`.
- `device` - WebGPU device, or null when GPU inference is unavailable.
- `options` - Threshold and policy options.

Returns: Promise resolving to a row-major output matrix of length
`networks.length * networks[0].output`.

### isDeviceUsable

```ts
isDeviceUsable(
  device: any,
): boolean
```

Check whether a supplied GPU device is present and has not been lost.

See the matching helper in `network.gpu.fallback` for the rationale:
real devices report loss asynchronously through `device.lost`, while this
predicate gives a synchronous yes/no answer for the current call site.

Parameters:
- `device` - Device to inspect, or null/undefined when WebGPU is absent.

Returns: True when the device is present and not marked lost.

### RacingBatchOptions

Options controlling batch evaluation in the racing-curriculum worker seam.

### shouldUseGPUPath

```ts
shouldUseGPUPath(
  networks: default[],
  device: GPUDevice,
  threshold: number,
): boolean
```

Decide whether the racing generation can use the batched GPU path.

All of the following must hold:
1. The batch size is strictly greater than the configured threshold.
2. Every network in the batch is structurally GPU-eligible.

Callers must already have verified the device is usable with
`isDeviceUsable` before invoking this predicate.

Parameters:
- `networks` - Generation of networks to evaluate.
- `device` - Verified WebGPU device.
- `threshold` - Minimum batch size that justifies GPU dispatch.

Returns: True when the batched GPU path should be used.

## architecture/network/gpu/network.gpu.types.ts

Public type surface for the WebGPU inference fast path.

The canonical WebGPU interfaces are declared ambiently in
`gpu.types.d.ts` so they are available to both source files and automated
tests without a runtime dependency. This module re-exports the subset of
those names used by the device probe and capability predicate, giving
callers a single local import surface if they prefer explicit module
references over global declarations.

### GPU_BUFFER_BINDING

Stable WebGPU binding indices for the slab-to-GPU upload contract.

The compute kernel's bind group layout and WGSL declarations must use these
exact indices so the shader reads the uploaded arrays in the order produced
by `uploadNetworkToGPU`. Keeping the mapping in one exported table
prevents drift between the upload path and the kernel.

Example:

```ts
const binding = GPU_BUFFER_BINDING.weights; // 0
```

### GPU_BUFFER_BINDING_COUNT

Number of storage-buffer bindings used by the GPU forward-pass kernel.

This count matches the length of `GPU_BUFFER_BINDING` and the number
of entries in the kernel bind-group layout.

### GPUActivationPipelineCache

Pipeline cache scoped to one WebGPU device.

Maps a deterministic topology key to the compiled compute pipeline so that
identical network topologies share one pipeline even when their weights differ.

### GPUAdapterType

Public type surface for the WebGPU inference fast path.

The canonical WebGPU interfaces are declared ambiently in
`gpu.types.d.ts` so they are available to both source files and automated
tests without a runtime dependency. This module re-exports the subset of
those names used by the device probe and capability predicate, giving
callers a single local import surface if they prefer explicit module
references over global declarations.

### GPUBufferName

Names of the slab buffers that participate in the GPU upload contract.

Each name maps to a stable binding index in `GPU_BUFFER_BINDING`.

### GPUDeviceType

### GPUKernelTopologyKey

Deterministic string key that identifies a network topology for the pipeline
cache.

### GPURequestAdapterOptionsType

### GPUSupportedLimitsType

## architecture/network/gpu/network.gpu.activation.wgsl.ts

WGSL activation-function registry for the WebGPU inference fast path.

This module maps the compact numeric activation indices used by the worker
serialization contract (see `src/multithreading/multi.utils.ts`) to their f32
WGSL implementations. Only the first-kernel subset is implemented here;
unsupported activations are deliberately omitted from the generated switch.

### ActivationFunctionEntry

Description of one supported activation function in WGSL form.

### buildActivationFunctionBody

```ts
buildActivationFunctionBody(
  index: 0 | 2 | 1 | 4 | 3 | 5 | 12 | 10 | 13 | 9 | 11,
): string
```

Build the single-statement WGSL body for a supported activation index.

### buildActivationRegistry

```ts
buildActivationRegistry(): readonly ActivationFunctionEntry[]
```

Build the canonical registry of WGSL activation functions.

Returns: A read-only array of supported activation entries. The order matches
`SUPPORTED_ACTIVATION_INDICES` so callers can emit a deterministic switch.

### formatActivationFunctionsWgsl

```ts
formatActivationFunctionsWgsl(
  registry: readonly ActivationFunctionEntry[],
): string
```

Format the registry as a block of WGSL function declarations.

Parameters:
- `registry` - Activation entries from `buildActivationRegistry`.

Returns: WGSL source containing one `fn activation_<index>(x: f32) -> f32`
declaration per supported index.

### SUPPORTED_ACTIVATION_INDICES

Ordered subset of worker activation indices that the first WGSL kernel
supports.

These positions must stay in sync with `ACTIVATION_FUNCTIONS` in
`src/multithreading/multi.utils.ts` because DNA, workers, and the GPU kernel
all use the same numeric index for the same activation.
