# Network Slab Packing Design (Phase 3 Scaffold)

Purpose: Define target structure-of-arrays (SoA) / slab layout for high‑density memory representation of network connection & node state prior to implementation.

## Goals

1. Reduce per-connection object overhead (header + property slots) by packing hot fields into contiguous typed arrays.
2. Improve cache locality for forward passes & mutation scans.
3. Enable optional object mode (default today) for educational clarity; slab mode is opt‑in & pay‑for‑use.
4. Provide deterministic layout to support future persistence & hashing.

## Proposed Layout (Initial Draft)

Connection Slab (length = capacity):

- weights: Float32Array
- gains: Float32Array (optional; lazily allocated only if any gain !=1)
- flags: Uint8Array (bit0 enabled, bit1 dcMask, bit2 hasGater, bit3 reserved, bit4 plastic, bit5 spare, bit6 spare, bit7 spare)
- fromIndex: Uint32Array
- toIndex: Uint32Array
- gaterIndex: Uint32Array (optional; parallel to flags.hasGater; omit if none)
- innovation: Uint32Array (or Uint32 + overflow guard)
- opt_firstMoment / opt_secondMoment / opt_cache ... (deferred; separate struct or interleaved? TBD based on access patterns)

Node Slab (length = nodeCapacity):

- activation: Float32Array
- state: Float32Array
- bias: Float32Array (could be Float32 even if weights become f16; bias precision retention beneficial)
- type: Uint8Array (enum mapping: 0=input,1=hidden,2=output,3=constant)
- mask: Float32Array (or pack dropout mask into flags if strictly 0/1 → Uint8)
- derivative: Float32Array (lazy; allocate only if training path active)

## Allocation Strategy

- Geometric growth factors: Node (1.5×), Connection (1.75× Node / 1.25× Browser variant) with upper cap to avoid huge jumps.
- Free-list for holes after prune; periodic compaction threshold configurable (e.g., >20% holes triggers pack).

## Interop With Object Mode

- `Network` maintains either arrays of objects OR slab handles + lightweight proxy objects created on demand for APIs that expect instances.
- Slab mode flagged via `config.useSlabMode` (future) captured in `memoryStats().flags`.
- Transition path: build network in object mode, call `network.materializeSlabs()` to migrate.

## Plasticity / Extended State

- Optional parallel slabs (Float32Array) for plasticity rates / accumulators only allocated when feature flag enabled.
- Plastic flags integrated into connection flags byte.

## Hashing / Provenance

- Dist bundle provenance already tracked (Phase 1). Slab mode will add: layoutVersion, capacity, activeCounts inside benchmark artifact.

## Open Questions

1. Do we co-locate rarely-updated fields (innovation) with hot arrays? (Probably yes; innovation accesses are infrequent.)
2. Gater indices: store -1 sentinel in shared Uint32Array vs separate optional array? (Trade-off memory vs branch; likely single array with 0xFFFFFFFF sentinel.)
3. Mixed precision: weights f16 / bias f32 path – convert on read or convert during forward? (Benchmark.)
4. Optimizer state: separate structure keyed by connection index to reduce cache pollution when inactive.

## Next Steps

- Implement capacity planner & initial slab structs.
- Add conversion tests: object → slab → object round trip parity (activations, forward outputs).
- Update `memoryStats()` to report slab bytes (raw + overhead) and active vs capacity counts.

(Scaffold only – implementation lives in future PR.)
