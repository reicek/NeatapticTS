# MDN WebGPU API condensed notes

> Source: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API — captured 2026-06-30. This file contains raw condensed notes from MDN's practical WebGPU guide.

#

Skip to main content
Skip to search

# WebGPU API

Limited availability
This feature is not Baseline because it does not work in some of the most widely-used browsers.
Tell us why.
Learn more
See full compatibility
Secure context:
secure contexts
supporting browsers
WebGPU API
WebGL

## Concepts and usage

Concepts and usage
WebGL
OpenGL ES 2.0
`<canvas>`
GLSL
Three.js
Babylon.js
PlayCanvas
However, WebGL has some fundamental issues that needed addressing:
Microsoft's Direct3D 12
Apple's Metal
The Khronos Group's Vulkan

- WebGL is based wholly around the use case of drawing graphics and rendering them to a canvas. It does not handle general-purpose GPU (GPGPU) computations very well. GPGPU computations are becoming more and more important for many different use cases, for example those based on machine learning models.
- 3D graphics apps are becoming increasingly demanding, both in terms of the number of objects to be rendered simultaneously, and usage of new rendering features.
  WebGPU addresses these issues, providing an updated general-purpose architecture compatible with modern GPU APIs, which feels more "webby". It supports graphic rendering, but also has first-class support for GPGPU computations. Rendering of individual objects is significantly cheaper on the CPU side, and it supports modern GPU rendering features such as compute-based particles and post-processing filters like color effects, sharpening, and depth-of-field simulation. In addition, it can handle expensive computations such as culling and skinned model transformation directly on the GPU.

## General model

General model
There are several layers of abstraction between a device GPU and a web browser running the WebGPU API. It is useful to understand these as you begin to learn WebGPU:
Physical devices have GPUs. Most devices only have one GPU, but some have more than one. Different GPU types are available:

- Integrated GPUs, which live on the same board as the CPU and share its memory.
- Discrete GPUs, which live on their own board, separate from the CPU.
- Software "GPUs", implemented on the CPU.
  Note:
  A native GPU API, which is part of the OS (for example, Metal on macOS), is a programming interface allowing native applications to use the capabilities of the GPU. API instructions are sent to the GPU (and responses received) via a driver. It is possible for a system to have multiple native OS APIs and drivers available to communicate with the GPU, although the above diagram assumes a device with only one native API/driver.
  A browser's WebGPU implementation handles communicating with the GPU via a native GPU API driver. A WebGPU adapter effectively represents a physical GPU and driver available on the underlying system, in your code.
  A logical device is an abstraction via which a single web app can access GPU capabilities in a compartmentalized way. Logical devices are required to provide multiplexing capabilities. A physical device's GPU is used by many applications and processes concurrently, including potentially many web apps. Each web app needs to be able to access WebGPU in isolation for security and logic reasons.

## Accessing a device

Accessing a device
`GPUDevice`
`Navigator.gpu`
`WorkerNavigator.gpu`
`GPU`
`GPU.requestAdapter()`
compatibility mode
`GPUAdapter.requestDevice()`
Putting this together with some feature detection checks, the above process could be achieved as follows:
js
`async function init() { if (!navigator.gpu) { throw Error("WebGPU not supported."); } let adapter; try { adapter = await navigator.gpu.requestAdapter(); } catch (error) { console.error(error); } if (!adapter) { throw Error("Couldn't request WebGPU adapter."); } const device = await adapter.requestDevice(); // … }`

### WebGPU compatibility mode

WebGPU compatibility mode
`GPUAdapter`
`GPUAdapter`
`featureLevel`
`compatibility`
`GPU.requestAdapter()`
js
`const adapter = await navigator.gpu.requestAdapter({ featureLevel: "compatibility", });`
WebGPU Compatibility Mode
`GPUAdapter`
`GPUDevice`
`core-features-and-limits`
`GPUSupportedFeatures`
`core-features-and-limits`
js
`const isCore = device.features.has("core-features-and-limits");`
Using compatibility mode only if necessary

## Pipelines and shaders: WebGPU app structure

Pipelines and shaders: WebGPU app structure
A pipeline is a logical structure containing programmable stages that are completed to get your program's work done. WebGPU is currently able to handle two types of pipeline:
`<canvas>`
A vertex stage, in which a vertex shader takes positioning data fed into the GPU and uses it to position a series of vertices in 3D space by applying specified effects like rotation, translation, or perspective. The vertices are then assembled into primitives such as triangles (the basic building block of rendered graphics) and rasterized by the GPU to figure out what pixels each one should cover on the drawing canvas.
A fragment stage, in which a fragment shader computes the color for each pixel covered by the primitives produced by the vertex shader. These computations frequently use inputs such as images (in the form of textures) that provide surface details and the position and color of virtual lights.
A compute pipeline is for general computation. A compute pipeline contains a single compute stage in which a compute shader takes general data, processes it in parallel across a specified number of workgroups, then returns the result in one or more buffers. The buffers can contain any kind of data.
WebGPU Shading Language
There are several different ways in which you could architect a WebGPU app, but the process will likely contain the following steps:
Create shader modules
Get and configure the canvas context
`webgpu`
`<canvas>`
Create resources containing your data
Create pipelines
Run a compute/rendering pass

- Create a command encoder that can encode a set of commands to be passed to the GPU to execute.
- Create a pass encoder object on which compute/render commands are issued.
- Run commands to specify which pipelines to use, what buffer(s) to get the required data from, how many drawing operations to run (in the case of render pipelines), etc.
- Finalize the command list and encapsulate it in a command buffer.
- Submit the command buffer to the GPU via the logical device's command queue.
  basic compute pipeline

## Basic render pipeline

Basic render pipeline
basic render demo
`<canvas>`

### Create shader modules

Create shader modules
`@vertex`
`@fragment`
js
`const shaders = ` struct VertexOut { @builtin(position) position : vec4f, @location(0) color : vec4f } @vertex fn vertex_main(@location(0) position: vec4f, @location(1) color: vec4f) -> VertexOut { var output : VertexOut; output.position = position; output.color = color; return output; } @fragment fn fragment_main(fragData: VertexOut) -> @location(0) vec4f { return fragData.color; } `;`
Note:
`<script>`
`Node.textContent`
`text/wgsl`
`GPUShaderModule`
`GPUDevice.createShaderModule()`
js
`const shaderModule = device.createShaderModule({ code: shaders, });`

### Get and configure the canvas context

Get and configure the canvas context
`<canvas>`
`HTMLCanvasElement.getContext()`
`webgpu`
`GPUCanvasContext`
`GPUCanvasContext.configure()`
`GPUDevice`
js
`const canvas = document.querySelector("#gpuCanvas"); const context = canvas.getContext("webgpu"); context.configure({ device, format: navigator.gpu.getPreferredCanvasFormat(), alphaMode: "premultiplied", });`
Note:
`GPU.getPreferredCanvasFormat()`
`bgra8unorm`
`rgba8unorm`

### Create a buffer and write our triangle data into it

Create a buffer and write our triangle data into it
`Float32Array`
js
`const vertices = new Float32Array([ 0.0, 0.6, 0, 1, 1, 0, 0, 1, -0.5, -0.6, 0, 1, 0, 1, 0, 1, 0.5, -0.6, 0, 1, 0, 0, 1, 1, ]);`
`GPUBuffer`
`GPUBuffer`
`GPUDevice.createBuffer()`
`vertices`
`VERTEX`
`COPY_DST`
js
`const vertexBuffer = device.createBuffer({ size: vertices.byteLength, // make it big enough to store vertices in usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST, });`
`GPUBuffer`
compute pipeline example
`GPUQueue.writeBuffer()`
js
`device.queue.writeBuffer(vertexBuffer, 0, vertices, 0, vertices.length);`

### Define and create the render pipeline

Define and create the render pipeline
Now we've got our data into a buffer, the next part of the setup is to actually create our pipeline, ready to be used for rendering.
`vertices`
`float32x4`
`vec4<f32>`
`arrayStride`
`stepMode`
js
`const vertexBuffers = [ { attributes: [ { shaderLocation: 0, // position offset: 0, format: "float32x4", }, { shaderLocation: 1, // color offset: 16, format: "float32x4", }, ], arrayStride: 32, stepMode: "vertex", }, ];`
`GPUShaderModule`
`shaderModule`
`vertexBuffers`
`primitive`
`layout`
`auto`
`layout`
`GPUPipelineLayout`
`GPUDevice.createPipelineLayout()`
Basic compute pipeline
`auto`
js
`const pipelineDescriptor = { vertex: { module: shaderModule, entryPoint: "vertex_main", buffers: vertexBuffers, }, fragment: { module: shaderModule, entryPoint: "fragment_main", targets: [ { format: navigator.gpu.getPreferredCanvasFormat(), }, ], }, primitive: { topology: "triangle-list", }, layout: "auto", };`
`GPURenderPipeline`
`pipelineDescriptor`
`GPUDevice.createRenderPipeline()`
js
`const renderPipeline = device.createRenderPipeline(pipelineDescriptor);`

### Running a rendering pass

Running a rendering pass
`<canvas>`
`GPUCommandEncoder`
`GPUDevice.createCommandEncoder()`
js
`const commandEncoder = device.createCommandEncoder();`
`GPURenderPassEncoder`
`GPUCommandEncoder.beginRenderPass()`
`colorAttachments`
`<canvas>`
`context.getCurrentTexture().createView()`

- That the view should be "cleared" to a specified color once loaded and before any drawing takes place. This is what causes the blue background behind the triangle.
- That the value of the current rendering pass should be stored for this color attachment.
  js
  `const clearColor = { r: 0.0, g: 0.5, b: 1.0, a: 1.0 }; const renderPassDescriptor = { colorAttachments: [ { clearValue: clearColor, loadOp: "clear", storeOp: "store", view: context.getCurrentTexture().createView(), }, ], }; const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);`
  Now we can invoke methods of the rendering pass encoder to draw our triangle:
  `GPURenderPassEncoder.setPipeline()`
  `renderPipeline`
  `GPURenderPassEncoder.setVertexBuffer()`
  `vertexBuffer`
  `vertexBuffers`
  `GPURenderPassEncoder.draw()`
  `vertexBuffer`
  `3`
  js
  `passEncoder.setPipeline(renderPipeline); passEncoder.setVertexBuffer(0, vertexBuffer); passEncoder.draw(3);`
  To finish encoding the sequence of commands and issue them to the GPU, three more steps are needed.
  `GPURenderPassEncoder.end()`
  `GPUCommandEncoder.finish()`
  `GPUCommandBuffer`
  `GPUCommandBuffer`
  `GPUQueue`
  `GPUDevice.queue`
  `GPUCommandBuffer`
  `GPUQueue.submit()`
  These three steps can be achieved via the following two lines:
  js
  `passEncoder.end(); device.queue.submit([commandEncoder.finish()]);`

## Basic compute pipeline

Basic compute pipeline
basic compute demo
`GPUDevice`
`GPUShaderModule`
`GPUDevice.createShaderModule()`
`@compute`
js
`// Define global buffer size const NUM_ELEMENTS = 1000; const BUFFER_SIZE = NUM_ELEMENTS * 4; // Buffer size, in bytes const shader = ` @group(0) @binding(0) var<storage, read_write> output: array<f32>; @compute @workgroup_size(64) fn main( @builtin(global_invocation_id) global_id : vec3u, @builtin(local_invocation_id) local_id : vec3u, ) { // Avoid accessing the buffer out of bounds if (global_id.x >= ${NUM_ELEMENTS}) { return; } output[global_id.x] = f32(global_id.x) * 1000. + f32(local_id.x); } `;`

### Create buffers to handle our data

Create buffers to handle our data
`GPUBuffer`
`output`
`stagingBuffer`
`output`
`output`
`stagingBuffer`
js
`const output = device.createBuffer({ size: BUFFER_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, }); const stagingBuffer = device.createBuffer({ size: BUFFER_SIZE, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, });`

### Create a bind group layout

Create a bind group layout
`GPUBindGroupLayout`
`GPUDevice.createBindGroupLayout()`
`@binding(0)`
`storage`
js
`const bindGroupLayout = device.createBindGroupLayout({ entries: [ { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage", }, }, ], });`
`GPUBindGroup`
`GPUDevice.createBindGroup()`
`output`
js
`const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: [ { binding: 0, resource: { buffer: output, }, }, ], });`
Note:
`GPUComputePipeline.getBindGroupLayout()`
`GPURenderPipeline.getBindGroupLayout()`

### Create a compute pipeline

Create a compute pipeline
`GPUDevice.createComputePipeline()`
`layout`
`bindGroupLayout`
`GPUDevice.createPipelineLayout()`
js
`const computePipeline = device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout], }), compute: { module: shaderModule, entryPoint: "main", }, });`
One difference here from the render pipeline layout is that we are not specifying a primitive type, as we are not drawing anything.

### Running a compute pass

Running a compute pass
`GPUCommandEncoder.beginComputePass()`
`GPUComputePassEncoder.setPipeline()`
`GPUComputePassEncoder.setBindGroup()`
`bindGroup`
`GPUComputePassEncoder.dispatchWorkgroups()`
`GPURenderPassEncoder.end()`
js
`passEncoder.setPipeline(computePipeline); passEncoder.setBindGroup(0, bindGroup); passEncoder.dispatchWorkgroups(Math.ceil(NUM_ELEMENTS / 64)); passEncoder.end();`

### Reading the results back to JavaScript

Reading the results back to JavaScript
`GPUQueue.submit()`
`output`
`stagingBuffer`
`GPUCommandEncoder.copyBufferToBuffer()`
js
`// Copy output buffer to staging buffer commandEncoder.copyBufferToBuffer( output, 0, // Source offset stagingBuffer, 0, // Destination offset BUFFER_SIZE, // Length, in bytes ); // End frame by passing array of command buffers to command queue for execution device.queue.submit([commandEncoder.finish()]);`
`stagingBuffer`
`GPUBuffer.mapAsync()`
`GPUBuffer.getMappedRange()`
`stagingBuffer`
js
`// map staging buffer to read results back to JS await stagingBuffer.mapAsync( GPUMapMode.READ, 0, // Offset BUFFER_SIZE, // Length, in bytes ); const copyArrayBuffer = stagingBuffer.getMappedRange(0, BUFFER_SIZE); const data = copyArrayBuffer.slice(); stagingBuffer.unmap(); console.log(new Float32Array(data));`

## GPU error handling

GPU error handling
WebGPU calls are validated asynchronously in the GPU process. If errors are found, the problem call is marked as invalid on the GPU side. If another call is made that relies on the return value of an invalidated call, that object will also be marked as invalid, and so on. For this reason, errors in WebGPU are referred to as "contagious".
`GPUDevice`
`GPUDevice.pushErrorScope()`
`GPUDevice.popErrorScope()`
`Promise`
`GPUInternalError`
`GPUOutOfMemoryError`
`GPUValidationError`
`null`
`GPUDevice.createBindGroup()`

- Non-obvious, for example combinations of descriptor properties that produce validation errors. There is no point telling you to make sure you use the correct descriptor object structure. That is both obvious and vague.
- Developer-controlled. Some of the error criteria are purely based on internals and not really relevant to web developers.
  Object validity and destroyed-ness
  Errors
  WebGPU Error Handling best practices
  Note:
  `getError()`

## Interfaces

Interfaces

### Entry point for the API

Entry point for the API
`Navigator.gpu`
`WorkerNavigator.gpu`
`GPU`
`GPU`
`GPUAdapter`
`GPUAdapter`
`GPUDevice`
`GPUAdapterInfo`
Contains identifying information about an adapter.

### Configuring GPUDevices

Configuring GPUDevices
`GPUDevice`
Represents a logical GPU device. This is the main interface through which the majority of WebGPU functionality is accessed.
`GPUSupportedFeatures`
setlike
`GPUAdapter`
`GPUDevice`
`GPUSupportedLimits`
`GPUAdapter`
`GPUDevice`

### Configuring a rendering <canvas>

`<canvas>`
`HTMLCanvasElement.getContext()`
`"webgpu"`
`contextType`
`getContext()`
`"webgpu"`
`contextType`
`GPUCanvasContext`
`GPUCanvasContext.configure()`
`GPUCanvasContext`
`<canvas>`

### Representing pipeline resources

Representing pipeline resources
`GPUBuffer`
Represents a block of memory that can be used to store raw data to use in GPU operations.
`GPUExternalTexture`
`HTMLVideoElement`
`GPUSampler`
Controls how shaders transform and filter texture resource data.
`GPUShaderModule`
A reference to an internal shader module object, a container for WGSL shader code that can be submitted to the GPU to execution by a pipeline.
`GPUTexture`
A container used to store 1D, 2D, or 3D arrays of data, such as images, to use in GPU rendering operations.
`GPUTextureView`
`GPUTexture`

### Representing pipelines

Representing pipelines
`GPUBindGroup`
`GPUBindGroupLayout`
`GPUBindGroup`
`GPUBindGroupLayout`
`GPUBindGroup`
`GPUComputePipeline`
`GPUComputePassEncoder`
`GPUPipelineLayout`
`GPUBindGroupLayout`
`GPUBindGroup`
`GPUBindGroupLayout`
`GPURenderPipeline`
`GPURenderPassEncoder`
`GPURenderBundleEncoder`

### Encoding and submitting commands to the GPU

Encoding and submitting commands to the GPU
`GPUCommandBuffer`
`GPUQueue`
`GPUCommandEncoder`
Represents a command encoder, used to encode commands to be issued to the GPU.
`GPUComputePassEncoder`
`GPUComputePipeline`
`GPUCommandEncoder`
`GPUQueue`
controls execution of encoded commands on the GPU.
`GPURenderBundle`
`GPURenderBundleEncoder`
`GPURenderBundleEncoder`
`GPURenderPassEncoder`
`executeBundles()`
`GPURenderPassEncoder`
`GPURenderPipeline`
`GPUCommandEncoder`

### Running queries on rendering passes

Running queries on rendering passes
`GPUQuerySet`
Used to record the results of queries on passes, such as occlusion or timestamp queries.

### Debugging errors

Debugging errors
`GPUCompilationInfo`
`GPUCompilationMessage`
`GPUCompilationMessage`
Represents a single informational, warning, or error message generated by the GPU shader module compiler.
`GPUDeviceLostInfo`
`GPUDevice.lost`
`Promise`
`GPUError`
`GPUDevice.popErrorScope`
`uncapturederror`
`GPUInternalError`
`GPUDevice.popErrorScope`
`GPUDevice`
`uncapturederror`
`GPUOutOfMemoryError`
`GPUDevice.popErrorScope`
`GPUDevice`
`uncapturederror`
`GPUPipelineError`
`Promise`
`GPUDevice.createComputePipelineAsync()`
`GPUDevice.createRenderPipelineAsync()`
`GPUUncapturedErrorEvent`
`GPUDevice`
`uncapturederror`
`GPUValidationError`
`GPUDevice.popErrorScope`
`GPUDevice`
`uncapturederror`

## Security requirements

Security requirements
secure context

## Examples

Examples
Basic compute demo
Basic render demo
WebGPU samples

## Specifications

Specifications
Specification

## Browser compatibility

Browser compatibility

## See also

See also
WebGPU best practices
WebGPU explainer
WebGPU — All of the cores, none of the canvas

## Help improve MDN

Learn how to contribute
May 5, 2026
MDN contributors
View this page on GitHub
Report a problem with this content
