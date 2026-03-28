# architecture/activationArrayPool

Core activation-array-pool chapter for the architecture surface.

This folder owns the reusable output buffers that support high-frequency
activation paths. It sits beside the network and layer chapters because the
pool is not a new runtime primitive; it is the memory policy that decides
when temporary activation storage should be reused instead of reallocated.

Read this chapter in three passes:

1. start with `ActivationArray` to see which buffer shapes the runtime can
   safely recycle,
2. continue to `activationArrayPool.acquire()` and `.release()` when you need
   the hot-path allocation story,
3. finish with `stats()`, `setMaxPerBucket()`, and `prewarm()` when you want
   observability and capacity control.

## architecture/activationArrayPool/activationArrayPool.ts

### ActivationArray

Allowed activation array shapes for pooling.

The runtime prefers typed arrays when float32 mode is enabled, but keeps
plain numeric arrays available for code paths that expect standard JS array
behavior.

### ActivationArrayPool

A size-bucketed pool of activation arrays.

Buckets map array length to stacks of reusable buffers. Acquire returns a
zeroed buffer, either by recycling an existing one or by allocating a new
array when the requested bucket is empty.
