# neat/mutation

Root orchestration for NEAT mutation operations.

This chapter keeps the public mutation flow readable: mutate every genome,
reuse structural innovations for add-node and add-connection operators,
repair minimum hidden-node structure, and preserve the stable mutation-method
selection surface used by the main `Neat` controller.

The neighboring `flow/`, `select/`, `add-node/`, `add-conn/`, and
`repair/` chapters own the narrower mechanics.

## neat/mutation/mutation.ts

### DEFAULT_CONNECTION_WEIGHT

Default connection weight used for bootstrap and split in-edges.

### DEFAULT_GENE_ID

Default gene id value when a node has no gene id.

### DEFAULT_INNOVATION_ID

Default innovation id value when a connection has none.

### ensureMinHiddenNodes

```ts
ensureMinHiddenNodes(
  network: GenomeWithMetadata,
  multiplierOverride: number | undefined,
): Promise<void>
```

Ensure the network has a minimum number of hidden nodes and connectivity.

### ensureNoDeadEnds

```ts
ensureNoDeadEnds(
  network: GenomeWithMetadata,
): void
```

Ensure there are no dead-end nodes (input/output isolation) in the network.

### mutate

```ts
mutate(): Promise<void>
```

Mutate every genome in the population according to configured policies.

This is the high-level mutation driver used by NeatapticTS. It iterates the
current population and, depending on the configured mutation rate and
(optional) adaptive mutation controller, applies one or more mutation
operators to each genome. The sibling `maintenance/facade/` chapter keeps the
stable `Neat` class wrappers for maintenance-oriented callers, while this
mutation chapter continues to own the actual repair and mutation mechanics.

Educational notes:
- Adaptive mutation allows per-genome mutation rates/amounts to evolve so
  that successful genomes can reduce or increase plasticity over time.
- Structural mutations (ADD_NODE, ADD_CONN, etc.) may update global
  innovation bookkeeping; this function attempts to reuse specialized
  helper routines that preserve innovation ids across the population.

Example:

```ts
// called on a Neat instance after a generation completes
neat.mutate();
```

### mutateAddConnReuse

```ts
mutateAddConnReuse(
  genome: GenomeWithMetadata,
): void
```

Add a connection between two previously unconnected nodes, reusing a
stable innovation id per unordered node pair when possible.

Notes on behavior:
- The search space consists of node pairs (from, to) where `from` is not
  already projecting to `to` and respects the input/output ordering used by
  the genome representation.
- When a historical innovation exists for the unordered pair, the
  previously assigned innovation id is reused to keep different genomes
  compatible for downstream crossover and speciation.

Steps:
- Build a list of all legal (from,to) pairs that don't currently have a
  connection.
- Prefer pairs which already have a recorded innovation id (reuse
  candidates) to maximize reuse; otherwise use the full set.
- If the genome enforces acyclicity, simulate whether adding the connection
  would create a cycle; abort if it does.
- Create the connection and set its innovation id, either from the
  historical table or by allocating a new global innovation id.

Parameters:
- `genome` - - genome to modify in-place

### mutateAddNodeReuse

```ts
mutateAddNodeReuse(
  genome: GenomeWithMetadata,
): Promise<void>
```

Split a randomly chosen enabled connection and insert a hidden node.

This routine attempts to reuse a historical "node split" innovation record
so that identical splits across different genomes share the same
innovation ids. This preservation of innovation information is important
for NEAT-style speciation and genome alignment.

Method steps (high-level):
- If the genome has no connections, connect an input to an output to
  bootstrap connectivity.
- Filter enabled connections and choose one at random.
- Disconnect the chosen connection and either reuse an existing split
  innovation record or create a new hidden node + two connecting
  connections (in->new, new->out) assigning new innovation ids.
- Insert the newly created node into the genome's node list at the
  deterministic position to preserve ordering for downstream algorithms.

Example:

```ts
neat._mutateAddNodeReuse(genome);
```

Parameters:
- `genome` - - genome to modify in-place

### selectMutationMethod

```ts
selectMutationMethod(
  genome: GenomeWithMetadata,
  rawReturnForTest: boolean,
): Promise<MutationMethod | MutationMethod[] | null>
```

Select a mutation method respecting structural constraints and adaptive controllers.
Mirrors legacy implementation from `neat.ts` to preserve test expectations.
`rawReturnForTest` retains historical behavior where the full FFW array is
returned for identity checks in tests.
