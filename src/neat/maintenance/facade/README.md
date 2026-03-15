# neat/maintenance/facade

Public maintenance facade helpers for the stable `Neat` entrypoint.

The mutation chapter owns the actual topology-repair mechanics, but the
stable `Neat` class still exposes a tiny maintenance surface for callers that
want to enforce a minimum hidden-node budget, repair dead ends, or inspect
the configured hidden-node floor. Keeping that wrapper layer in
`maintenance/facade/` makes the ownership story match the newer chaptered
layout used by the RNG, pruning, selection, and telemetry facades.

Invariant: this boundary only maintains baseline structural viability for a
single network. It does not change mutation operator selection, speciation,
or public import paths.

## neat/maintenance/facade/maintenance.facade.ts

### ensureMinHiddenNodes

```ts
ensureMinHiddenNodes(
  host: NeatMaintenanceFacadeHost,
  network: default,
  multiplierOverride: number | undefined,
): Promise<void>
```

Ensure a network satisfies the configured minimum hidden-node policy.

The underlying mutation helper may add hidden nodes and wire them into the
graph so later mutation and evaluation steps start from a minimally viable
structure.

Parameters:
- `host` - - `Neat` instance exposing mutation constraints and innovation tables.
- `network` - - Network whose hidden-node floor should be enforced.
- `multiplierOverride` - - Optional one-off multiplier overriding the configured policy.

Returns: Promise that resolves after any required topology repair finishes.

### ensureNoDeadEnds

```ts
ensureNoDeadEnds(
  host: NeatMaintenanceFacadeHost,
  network: default,
): void
```

Repair input, output, and hidden nodes that have become structural dead ends.

This preserves the historical best-effort behavior of `neat.ensureNoDeadEnds()`:
if the underlying repair helper throws, the public facade suppresses that
failure so maintenance stays additive rather than fatal.

Parameters:
- `host` - - `Neat` instance exposing mutation constraints and innovation tables.
- `network` - - Network whose endpoint connectivity should be repaired.

Returns: Nothing. The network is patched in place when repairs are possible.

### getMinimumHiddenSize

```ts
getMinimumHiddenSize(
  host: NeatMaintenanceFacadeHost,
  multiplierOverride: number | undefined,
): number
```

Compute the minimum hidden-node target for the current `Neat` configuration.

The public `Neat` facade historically exposed this as a read-only policy
helper. Keeping it beside the repair wrappers makes the generated docs tell a
clearer story: one boundary defines the target size, and the neighboring
helpers enforce it on concrete networks.

Parameters:
- `host` - - `Neat` instance exposing input/output counts and maintenance options.
- `multiplierOverride` - - Optional one-off multiplier overriding the configured policy.

Returns: Minimum hidden-node count implied by explicit or multiplier-based settings.

Example:

```ts
const minimumHidden = neat.getMinimumHiddenSize();
await neat.ensureMinHiddenNodes(network);
console.log(minimumHidden, network.nodes.length);
```

### NeatMaintenanceFacadeHost

Narrow `Neat` host surface required by the public maintenance facade.

This boundary keeps the contract focused on topology-maintenance state:
mutation constraints, innovation bookkeeping, endpoint counts, and the
legacy helper used to compute a minimum hidden-node target.
