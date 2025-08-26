# utils

## utils/memory.ts

### memoryStats

`(targetNetworks: any) => import("D:/code-practice/NeatapticTS/src/utils/memory").MemoryStats`

Capture heuristic memory statistics for one or more networks with snapshot of active config flags.

Parameters:
- `targetNetworks` - Optional single network or array. If omitted, uses registered networks.

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

### registerTrackedNetwork

`(network: any) => void`

Register a network for inclusion in future `memoryStats()` calls made
without explicit parameters.

Duplicate registrations are ignored; insertion order is preserved which is
useful for deterministic test snapshots.

Parameters:
- `network` - Network instance (loosely typed to defer strict coupling).

### resetMemoryTracking

`() => void`

Reset internal tracking registry (and, in later phases, ancillary counters).

Educational: Calling this does NOT free memory — it simply clears the list
of networks that will be included when `memoryStats()` is invoked without
arguments. Use it between benchmark runs to isolate scenarios.

### unregisterTrackedNetwork

`(network: any) => void`

Remove a previously registered network from the tracking registry.
No-op if the network is not currently registered.

Parameters:
- `network` - Network instance.
