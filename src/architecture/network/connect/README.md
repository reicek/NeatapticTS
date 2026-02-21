# architecture/network/connect

## architecture/network/connect/network.connect.utils.ts

### connect

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default, weight: number | undefined) => import("C:/NeatapticTS/src/architecture/connection").default[]`

Create and register one (or multiple) directed connection objects between two nodes.

Some node types (or future composite structures) may return several low‑level connections when
their {@link Node.connect} is invoked (e.g., expanded recurrent templates). For that reason this
function always treats the result as an array and appends each edge to the appropriate collection.

Algorithm outline:
 1. (Acyclic guard) If acyclicity is enforced and the source node appears after the target node in
    the network's node ordering, abort early and return an empty array (prevents back‑edge creation).
 2. Delegate to sourceNode.connect(targetNode, weight) to build the raw Connection object(s).
 3. For each created connection:
      a. If it's a self‑connection: either ignore (acyclic mode) or store in selfconns.
      b. Otherwise store in standard connections array.
 4. If any connection was added, mark structural caches dirty (_topoDirty & _slabDirty) so lazy
    rebuild can occur before the next forward pass.

Complexity:
 - Time: O(k) where k is the number of low‑level connections returned (typically 1).
 - Space: O(k) new Connection instances (delegated to Node.connect).

Edge cases & invariants:
 - Acyclic mode silently refuses back‑edges instead of throwing (makes evolutionary search easier).
 - Self‑connections are skipped entirely when acyclicity is enforced.
 - Weight initialization policy is delegated to Node.connect if not explicitly provided.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `from` - - Source node (emits signal).
- `to` - - Target node (receives signal).
- `weight` - - Optional explicit initial weight value.

### disconnect

`(from: import("C:/NeatapticTS/src/architecture/node").default, to: import("C:/NeatapticTS/src/architecture/node").default) => void`

Remove (at most) one directed connection from source 'from' to target 'to'.

Only a single direct edge is removed because typical graph configurations maintain at most
one logical connection between a given pair of nodes (excluding potential future multi‑edge
semantics). If the target edge is gated we first call {@link Network.ungate} to maintain
gating invariants (ensuring the gater node's internal gate list remains consistent).

Algorithm outline:
 1. Choose the correct list (selfconns vs connections) based on whether from === to.
 2. Linear scan to find the first edge with matching endpoints.
 3. If gated, ungate to detach gater bookkeeping.
 4. Splice the edge out; exit loop (only one expected).
 5. Delegate per‑node cleanup via from.disconnect(to) (clears reverse references, traces, etc.).
 6. Mark structural caches dirty for lazy recomputation.

Complexity:
 - Time: O(m) where m is length of the searched list (connections or selfconns).
 - Space: O(1) extra.

Idempotence: If no such edge exists we still perform node-level disconnect and flag caches dirty –
this conservative approach simplifies callers (they need not pre‑check existence).

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `from` - - Source node.
- `to` - - Target node.
