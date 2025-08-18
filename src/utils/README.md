# utils

## utils/memory.ts

### memoryStats

`(targetNetworks: any) => any`

Capture heuristic memory statistics for one or more networks with snapshot of active config flags.

Parameters:
- `targetNetworks` - Optional single network or array. If omitted, uses registered networks.

### registerTrackedNetwork

`(network: any) => void`

Register a network instance for memory tracking (optional depending on design).
Allows memoryStats() to iterate over registered networks when no explicit instance passed.

Parameters:
- `` - Network instance (type narrowed later).

### resetMemoryTracking

`() => void`

Reset internal counters / high-water marks used for memory tracking.
Placeholder for future pool tracking reset.

### unregisterTrackedNetwork

`(network: any) => void`

Unregister a previously registered network instance.

Parameters:
- `` - Network instance.
