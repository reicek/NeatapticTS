# utils

## utils/memory.ts

### memoryStats

`(targetNetworks: import("C:/NeatapticTS/src/utils/memory").NetworkView | import("C:/NeatapticTS/src/utils/memory").NetworkView[] | undefined) => import("C:/NeatapticTS/src/utils/memory").MemoryStats`

### MemoryStats

Memory instrumentation utilities (Phase 0).

Educational overview:
These helpers expose a *heuristic* snapshot of memory usage for the
evolutionary population and internal pools. The goal is to help learners
reason about how design choices (slab storage, pooling, typed arrays)
influence memory footprint *without* incurring heavy introspection costs.

Design principles:
- Lightweight: Avoid deep graph walks or JSON serialization.
- Pay-for-use: If no networks are registered the function returns a small, fast object.
- Cross‑environment: Works in both Browser and Node via feature detection.
- Extensible: Shape deliberately includes draft sections for later precise accounting phases.

### NetworkView

Minimal view of a network used for memory heuristics. Only properties
accessed by this module are declared. This keeps coupling light while
enabling typed local variables instead of `any` everywhere.

### registerTrackedNetwork

`(network: import("C:/NeatapticTS/src/utils/memory").NetworkView | null | undefined) => void`

### resetMemoryTracking

`() => void`

### SlabAllocStats

Minimal slab allocator stats shape used here. The real shape may
include additional fields; we only rely on fresh/pooled counts.

### unregisterTrackedNetwork

`(network: import("C:/NeatapticTS/src/utils/memory").NetworkView) => void`
