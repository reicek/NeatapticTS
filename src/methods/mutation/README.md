# methods/mutation

Mutation policy shelf for neuroevolution runs.

This chapter belongs in `methods/` rather than `architecture/network/mutate/`
because it does not execute one mutation against one concrete graph. It
defines the reusable operator vocabulary that higher-level controllers pick
from before any specific network is touched. The network chapter later uses
that vocabulary to dispatch real edits.

Read the shelf in five families. Growth operators add structure. Pruning
operators remove it. Parameter operators retune weights and biases without
rewriting topology. Behavior operators change activation or gating policy.
Memory operators add recurrent building blocks when the search should be
allowed to invent stateful behavior.

Those families matter because mutation is where an evolutionary run decides
whether it is mostly refining a plausible graph or still exploring new
architectures. A shelf dominated by `MOD_WEIGHT` and `MOD_BIAS` behaves like
local numeric search. A shelf that also allows `ADD_NODE`, `ADD_CONN`, and
gating or recurrent operators gives the run permission to change what the
network can represent at all.

`ALL` and `FFW` are the two convenience summaries at the bottom of the
chapter. `ALL` keeps the widest search surface, including recurrence and
memory additions. `FFW` keeps the feedforward-safe subset for runs that must
remain acyclic.

```mermaid
flowchart TD
  Mutation[Mutation shelf] --> Grow[Grow structure]
  Mutation --> Prune[Prune structure]
  Mutation --> Tune[Tune parameters]
  Mutation --> Shape[Reshape behavior]
  Mutation --> Memory[Add memory blocks]
```

For compact background on why mutation pressure matters in evolutionary
search, see Wikipedia contributors,
[Mutation (genetic algorithm)](https://en.wikipedia.org/wiki/Mutation_(genetic_algorithm)).

Example: keep a feedforward-safe shelf for searches that must remain simple
and acyclic.

```ts
const feedforwardOnly = mutation.FFW;
```

Example: widen the shelf when structural exploration is part of the goal.

```ts
const structuralExploration = [
  mutation.ADD_CONN,
  mutation.ADD_NODE,
  mutation.MOD_WEIGHT,
  mutation.ADD_GATE,
];
```

## methods/mutation/mutation.ts

### ALL

Named export of the `ALL` mutation list for direct import.

### FFW

Named export of the `FFW` mutation list for direct import.

### MOD_TIME_CONSTANT

Named export of the `MOD_TIME_CONSTANT` config for direct import.

This operator retunes a node's CTRNN `timeConstant` without changing
topology. Larger values give the neuron slower, more inertial activation
dynamics; smaller values produce near-instant response. It complements
structural memory operators such as `ADD_LSTM_NODE` and `ADD_GRU_NODE`
because it modifies temporal behavior on existing nodes.

Runtime use: a mutation controller picks `MOD_TIME_CONSTANT` from the shelf;
the actual perturbation is applied by {@link mutateTimeConstant}; the
perturbed node then integrates via `applyCtrnnActivation`.

Example:

```ts
const broadShelf = mutation.ALL;
expect(broadShelf.map((m) => m.name)).toContain('MOD_TIME_CONSTANT');
```

### mutateTimeConstant

```ts
mutateTimeConstant(
  node: TimeConstantBearer,
  rng: () => number,
): void
```

Perturbs a node's `timeConstant` by a Gaussian N(0, TIME_CONSTANT_SIGMA)
perturbation drawn from the supplied RNG via the Box-Muller transform,
clamping to a positive minimum so the CTRNN integration remains stable.

Using the supplied RNG (rather than a deterministic hash) preserves
evolutionary diversity: clone populations with different RNG states produce
different perturbations, while the same RNG state remains reproducible.

Parameters:
- `node` - A node instance with a `timeConstant` property.
- `rng` - Uniform RNG returning values in [0, 1). Defaults to `Math.random`.

### MutationConfig

Configuration shape for one mutation operator.

Each mutation method carries a small policy object describing what kind of
structural or parametric change it performs and the narrow knobs that shape
that change. Read the fields as metadata for the evolutionary controller,
not as a full runtime implementation.

### TimeConstantBearer

Minimal interface for nodes that carry an evolvable `timeConstant` property.
