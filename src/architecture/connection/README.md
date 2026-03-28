# architecture/connection

Core connection chapter for the architecture surface.

This folder owns the library's directed edge primitive: the mutable link that
carries weight, gain modulation, gating metadata, eligibility traces,
optimizer scratch state, and evolutionary innovation bookkeeping between two
nodes.

Read this chapter in three passes:

1. start with the `Connection` class overview to understand which edge fields
   are always present versus allocated lazily,
2. continue to the serialization and innovation helpers when you need to see
   how a connection survives genome alignment, pooling, and persistence,
3. finish with the virtualized accessors when you want the memory-shaping
   details behind gating, plasticity, and optimizer moments.

Example:

```ts
const source = new Node('input');
const target = new Node('output');
const edge = new Connection(source, target, 0.42);

edge.gain = 1.5;
edge.enabled = true;
```

## architecture/connection/connection.ts

### Connection

Connection (Synapse / Edge)
===========================
Directed weighted link between two nodes. The connection keeps the everyday
graph fields (`from`, `to`, `weight`, `innovation`) directly on the instance,
then pushes rarer capabilities behind symbol-backed accessors so large
populations do not pay object-shape costs for features they are not using.

This makes the boundary useful in three different modes:

- ordinary feed-forward links that only need endpoints and weight,
- gated or plastic links that gradually opt into extra runtime state,
- optimizer-heavy training paths that need moment buffers without turning
  every connection into a bloated record.

Example:

```ts
const source = new Node('input');
const target = new Node('output');
const edge = new Connection(source, target, 0.42);

edge.gain = 1.5;
edge.enabled = true;
```

### ConnectionSymbolProps

Internal interface for accessing symbol-keyed properties on Connection instances.
Used for type-safe access to dynamic symbol properties.

### default

#### _flags

Packed state flags (private for future-proofing hidden class):
bit0 => enabled gene expression (1 = active)
bit1 => DropConnect active mask (1 = not dropped this forward pass)
bit2 => hasGater (1 = symbol field present)
bit3 => plastic (plasticityRate > 0)
bits4+ reserved.

#### acquire

```ts
acquire(
  from: default,
  to: default,
  weight: number | undefined,
): default
```

Acquire a connection from the internal pool, or construct a fresh one when the pool is empty.
This is the low-allocation path used by topology mutation and other edge-churn heavy flows.

Parameters:
- `from` - Source node.
- `to` - Target node.
- `weight` - Optional initial weight.

Returns: Reinitialized connection instance.

#### dcMask

DropConnect active mask: `1` means active for this stochastic pass, `0` means dropped.

#### dropConnectActiveMask

Convenience alias for DropConnect mask with clearer naming.

#### eligibility

Standard eligibility trace (e.g., for RTRL / policy gradient credit assignment).

#### enabled

Whether the gene is currently expressed and participates in the forward pass.

#### firstMoment

First moment estimate used by Adam-family optimizers.

#### from

The source (pre-synaptic) node supplying activation.

#### gain

Multiplicative modulation applied after weight. Neutral gain `1` is omitted from storage.

#### gater

Optional gating node whose activation modulates effective weight.

#### gradientAccumulator

Generic gradient accumulator used by RMSProp and AdaGrad.

#### hasGater

Whether a gater node is assigned to modulate this connection's effective weight.

#### infinityNorm

Adamax infinity norm accumulator.

#### innovation

Unique historical marking (auto-increment) for evolutionary alignment.

#### innovationID

```ts
innovationID(
  sourceNodeId: number,
  targetNodeId: number,
): number
```

Deterministic Cantor pairing function for a `(sourceNodeId, targetNodeId)` pair.
Use it when you need a stable edge identifier without relying on the mutable
auto-increment counter.

Parameters:
- `sourceNodeId` - Source node integer id or index.
- `targetNodeId` - Target node integer id or index.

Returns: Unique non-negative integer derived from the ordered pair.

Example:

```ts
const id = Connection.innovationID(2, 5);
```

#### lookaheadShadowWeight

Lookahead slow-weight snapshot.

#### maxSecondMoment

AMSGrad maximum of past second-moment estimates.

#### plastic

Whether this connection participates in plastic adaptation.

#### plasticityRate

Per-connection plasticity rate. `0` means the connection is not plastic.

#### previousDeltaWeight

Last applied delta weight (used by classic momentum).

#### release

```ts
release(
  conn: default,
): void
```

Return a connection instance to the internal pool for later reuse.
Treat the instance as surrendered after calling this method.

Parameters:
- `conn` - The connection instance to recycle.

Returns: Nothing.

#### resetInnovationCounter

```ts
resetInnovationCounter(
  value: number,
): void
```

Reset the monotonic innovation counter used for newly constructed or pooled connections.
You usually call this at the start of an experiment or before rebuilding a whole population.

Parameters:
- `value` - New starting value.

Returns: Nothing.

#### secondMoment

Second raw moment estimate used by Adam-family optimizers.

#### secondMomentum

Secondary momentum buffer used by Lion-style updates.

#### to

The target (post-synaptic) node receiving activation.

#### toJSON

```ts
toJSON(): { from: number | undefined; to: number | undefined; weight: number; gain: number; innovation: number; enabled: boolean; gater?: number | undefined; }
```

Serialize to a minimal JSON-friendly shape used by genome and network save flows.
Undefined node indices are preserved so callers can resolve or remap them later.

Returns: Object with node indices, weight, gain, innovation id, enabled flag, and gater index when one exists.

Example:

```ts
const json = connection.toJSON();
// => { from: 0, to: 3, weight: 0.12, gain: 1, innovation: 57, enabled: true }
```

#### totalDeltaWeight

Accumulated (batched) delta weight awaiting an apply step.

#### weight

Scalar multiplier applied to the source activation (prior to gain modulation).

#### xtrace

Extended trace structure for modulatory / eligibility propagation algorithms. Parallel arrays for cache-friendly iteration.
