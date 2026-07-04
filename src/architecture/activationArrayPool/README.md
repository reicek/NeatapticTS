# architecture/activationArrayPool

Core activation-array-pool chapter for the architecture surface.

This folder owns the reusable output buffers that support high-frequency
activation paths. It sits beside the network and layer chapters because the
pool is not a new runtime primitive; it is the memory policy that decides
when temporary activation storage should be reused instead of reallocated.

Think of this boundary as scratch-space policy for activation-heavy code.
Network execution repeatedly asks for short-lived arrays whose shapes often
repeat from sample to sample. Reallocating those arrays every pass adds
garbage-collector work without changing the model's behavior. This chapter
keeps the reusable buffer story explicit while leaving graph semantics to
the node, layer, and network chapters.

The key design choice is that pooling happens by array length and resolved
activation precision, not by caller identity. A network slab path and a
layer helper can share the same retained storage as long as they request the
same length and precision and release the buffer after use. Acquire always
returns a zeroed buffer, so the pool promises clean scratch memory rather
than cached activations or memoized results.

That distinction matters when you are debugging correctness. This folder is
not a semantic cache and it does not preserve intermediate values for later
reads. It only keeps array shells alive long enough to reduce allocation
churn in hot loops, worker batches, and repeated forward passes.

```mermaid
flowchart LR
  classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
  classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;

  Request[Activation request]:::base --> Acquire[acquire by length]:::accent
  Acquire -->|bucket hit| Reuse[Zeroed reused buffer]:::base
  Acquire -->|bucket miss| Fresh[Freshly allocated buffer]:::base
  Reuse --> Execute[Layer or network activation]:::base
  Fresh --> Execute
  Execute --> Release[release buffer]:::accent
  Release --> Bucket[Length bucket]:::base
```

For background on the broader software pattern, see Wikipedia contributors,
[Object pool pattern](https://en.wikipedia.org/wiki/Object_pool_pattern).
This chapter applies that idea to activation scratch space, not to long-lived
model parameters.

Read this chapter in three passes:

1. start with `ActivationArray` to see which buffer shapes the runtime can
   safely recycle,
2. continue to `activationArrayPool.acquire()` and `.release()` when you need
   the hot-path allocation story,
3. finish with `stats()`, `setMaxPerBucket()`, `prewarm()`, and `compact()`
   when you want observability and capacity control.

Example: reuse one scratch buffer around a custom activation-heavy loop.

```ts
const activationArray = activationArrayPool.acquire(128);
// fill and consume the buffer here
activationArrayPool.release(activationArray);
```

Example: prewarm one common bucket before a large evaluation batch.

```ts
activationArrayPool.setMaxPerBucket(32);
activationArrayPool.prewarm(256, 8);
const stats = activationArrayPool.stats();
```

## architecture/activationArrayPool/activationArrayPool.ts

### ActivationArray

Allowed activation array shapes for pooling.

The runtime prefers typed arrays when float32 mode is enabled, but keeps
plain numeric arrays available for code paths that expect standard JS array
behavior.

### ActivationArrayPool

A size-bucketed pool of activation arrays.

Buckets map array length plus resolved precision to stacks of reusable
buffers. Acquire returns a zeroed buffer, either by recycling an existing
one or by allocating a new array when the requested bucket is empty.

### createActivationArray

```ts
createActivationArray(
  size: number,
  precisionFlags: { float32Mode: boolean; },
  activationPrecision: ActivationPrecision | undefined,
): ActivationArray
```

Create one fresh activation buffer using the shared precision owner.

Parameters:
- `size` - Required activation-array length.
- `precisionFlags` - Config-like precision flags for the current runtime.

Returns: Fresh activation buffer with the resolved precision policy.

### createActivationArrayBucketKey

```ts
createActivationArrayBucketKey(
  size: number,
  activationPrecision: ActivationPrecision,
): string
```

Build the pool key for one activation-array bucket.

Parameters:
- `size` - Required activation-array length.
- `activationPrecision` - Resolved precision for this bucket.

Returns: Stable bucket key string.

### resolveActivationArrayPrecision

```ts
resolveActivationArrayPrecision(
  activationArray: ActivationArray,
): ActivationPrecision
```

Infer the retained precision for one released activation array.

Parameters:
- `activationArray` - Activation buffer being returned to the pool.

Returns: Precision bucket that owns this array.
