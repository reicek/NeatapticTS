# neat/nge-main-agent

## neat/nge-main-agent/neat.nge-main-agent.types.ts

### NgeMainAgentAdult

Main agent adult state after maturation.

### NgeMainAgentAttentionHead

Attention-head motif descriptor for threat prioritization.

This is a typed tag used by the main agent allowlist; the actual realization
logic lives in later NGE phases.

### NgeMainAgentEmbryo

Main agent embryo state.

The embryo is the smallest materialized stage. It carries the full motif
allowlist as archetypes but keeps node and edge counts tiny so that later
growth has headroom within the tier budget.

### NgeMainAgentEpisodicSlot

Episodic slot motif descriptor for spawn-pattern memory.

This is a typed tag used by the main agent allowlist; the actual realization
logic lives in later NGE phases.

### NgeMainAgentEquilibrium

Equilibrium candidate produced by adult optimization.

### NgeMainAgentGatedRecurrentCell

Gated recurrent cell motif descriptor for aim/strafe state retention.

This is a typed tag used by the main agent allowlist; the actual realization
logic lives in later NGE phases.

### NgeMainAgentJuvenile

Main agent juvenile state during local growth and focus gating.

### NgeMainAgentLifecycleConfig

Configuration governing the main agent lifecycle and tier budget.

The same config plus the same seed must always advance through the lifecycle
in the same order, so the seed is stored here rather than inferred from the
environment.

### NgeMainAgentLifecycleStage

Ordered lifecycle stages for the NGE main agent.

The stage machine is intentionally distinct from the Racing Curriculum
juvenile/adult stage contract so that main-agent morphogenesis can evolve its
own cadence without being coupled to a different demo's lifecycle policy.

### NgeMainAgentLifecycleState

Shared state fields across all main-agent lifecycle stages.

### NgeMainAgentReproducing

Main agent reproducing stage, ready to emit the next generation's embryo.

### NgeMainAgentTopologyBudget

Topology budget enforced at every lifecycle transition.

The budget is always positive and never exceeds the configured tier cap.

## neat/nge-main-agent/neat.nge-main-agent.adult.ts

### evaluateAdultTopologyBudget

```ts
evaluateAdultTopologyBudget(
  adult: NgeMainAgentAdult,
  config: NgeMainAgentLifecycleConfig,
): { withinBudget: boolean; }
```

Evaluate whether an adult topology is within the configured tier budget.

Parameters:
- `adult` - Adult state to evaluate.
- `config` - Lifecycle config with the tier budget.

Returns: An evaluation object whose `withinBudget` flag is true when the adult respects both caps.

### pruneAdultTopology

```ts
pruneAdultTopology(
  juvenile: NgeMainAgentJuvenile,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult
```

Prune a juvenile topology down to adult limits while remaining within budget.

The adult stage may reduce or cap node and edge counts, but it never allows
them to exceed the configured tier budget.

Parameters:
- `juvenile` - Juvenile state to prune.
- `config` - Lifecycle config with the tier budget cap.

Returns: Adult state whose counts are bounded by the tier budget.

### runAdultEquilibrium

```ts
runAdultEquilibrium(
  adult: NgeMainAgentAdult,
  _config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEquilibrium
```

Run adult optimization until an equilibrium candidate is stable.

This first-pass implementation treats the pruned adult as already stable so
that the lifecycle runner can be tested end-to-end. Later phases will replace
this with iterative equilibrium detection.

Parameters:
- `adult` - Adult state to optimize.
- `_config` - Lifecycle config (reserved for future equilibrium parameters).

Returns: Equilibrium candidate wrapping the stable adult.

## neat/nge-main-agent/neat.nge-main-agent.embryo.ts

### buildMainAgentEmbryo

```ts
buildMainAgentEmbryo(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentEmbryo
```

Build a deterministic main-agent embryo state.

The embryo carries the three allowed motif archetypes, a node and edge count
within the tier budget, and the canonical schema version A.1.0. The same
config always produces the same embryo, which is required for reproducible
generation barriers.

Parameters:
- `config` - Lifecycle config with seed and tier budget.

Returns: A deterministic embryo state ready for juvenile growth.

### computeEmbryoTopologyBudget

```ts
computeEmbryoTopologyBudget(
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentTopologyBudget
```

Compute the topology budget for an embryo, capping it at the tier budget.

The embryo is intentionally small relative to the tier cap so that juvenile
growth and adult pruning have meaningful headroom without risking overflow.

Parameters:
- `config` - Lifecycle config with the configured tier cap.

Returns: A positive topology budget bounded by the tier cap.

### resolveMainAgentMotifAllowlist

```ts
resolveMainAgentMotifAllowlist(): readonly ("DenseFeedForward" | "AttentionHead" | "GatedRecurrentCell" | "EpisodicSlot" | "ModulatorBroadcaster" | "GatingRouter")[]
```

Resolve the exact three existing catalogue motifs allowed for the main agent.

The main-agent motif set is deliberately minimal and does not introduce any
new computation types or schema versions. All returned values are already
present in {@link NEAT_GENOME_COMPUTATION_TYPE_CATALOGUE}.

Returns: A readonly array containing AttentionHead, GatedRecurrentCell, and EpisodicSlot.

## neat/nge-main-agent/neat.nge-main-agent.juvenile.ts

### growJuvenileTopology

```ts
growJuvenileTopology(
  embryo: NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentJuvenile
```

Grow the embryo topology into a juvenile state while staying within budget.

Juvenile growth is deterministic and never shrinks the network below the
embryo size. The growth factor is derived from the embryo's seed so that the
same embryo always produces the same juvenile.

Parameters:
- `embryo` - Embryo state to grow.
- `config` - Lifecycle config with the tier budget cap.

Returns: Juvenile state with node and edge counts within the tier budget.

### transitionJuvenileToAdult

```ts
transitionJuvenileToAdult(
  juvenile: NgeMainAgentJuvenile | NgeMainAgentEmbryo,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentAdult
```

Mature a juvenile or embryo state into an adult state.

The transition is a deterministic stage change that preserves the input
topology. Adult pruning is handled separately by {@link pruneAdultTopology}.

Parameters:
- `juvenile` - Juvenile or embryo state to mature.
- `config` - Lifecycle config with the deterministic seed.

Returns: Adult state with the same topology as the input.

## neat/nge-main-agent/neat.nge-main-agent.lifecycle.ts

### advanceMainAgentLifecycle

```ts
advanceMainAgentLifecycle(
  state: NgeMainAgentLifecycleState,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentLifecycleState
```

Advance one main-agent lifecycle state to the next stage.

The stage machine follows the fixed order
Embryo → Juvenile → Adult → Reproducing → Embryo. The generation counter
increments on every call so that the full cycle is observable and
deterministic for the same config.

Parameters:
- `state` - Current lifecycle state.
- `config` - Lifecycle config that supplies the deterministic seed.

Returns: The next lifecycle state with the stage advanced and generation incremented.

### createMainAgentLifecycleRunner

```ts
createMainAgentLifecycleRunner(
  config: NgeMainAgentLifecycleConfig,
): (state: NgeMainAgentLifecycleState) => NgeMainAgentLifecycleState
```

Create a deterministic lifecycle runner bound to the supplied config.

The runner is a pure function: the same input state always yields the same
output state, which makes generation replay and barrier tests stable.

Parameters:
- `config` - Lifecycle config to bind to every runner invocation.

Returns: A function that advances a lifecycle state using the bound config.

## neat/nge-main-agent/neat.nge-main-agent.reproduction.ts

### transitionAdultToReproducing

```ts
transitionAdultToReproducing(
  adult: NgeMainAgentAdult,
  equilibrium: NgeMainAgentEquilibrium,
  config: NgeMainAgentLifecycleConfig,
): NgeMainAgentReproducing
```

Transition a stable adult equilibrium into the reproducing stage.

The reproducing state inherits its topology from the stable equilibrium
adult. It is the final stage before the lifecycle runner loops back to embryo
for the next generation.

Parameters:
- `adult` - Adult state entering reproduction.
- `equilibrium` - Stable equilibrium candidate produced by adult optimization.
- `config` - Lifecycle config with the deterministic seed.

Returns: Reproducing state ready to emit the next generation.
