# neat/nge-collective

Shared type definitions for the NGE collective multi-agent system (Phase G).

These types are the shared vocabulary for the shared-field, evaluation, and metrics
sub-modules. Import individual sub-modules directly for their implementation functions.

## Type relationships

```mermaid
graph TD
  SF["SharedField\nwidth · height · cells: Float32Array"] -->|"bound into"| CEC
  CEC["CollectiveEvaluationContext\nagentCount · generationTick · field"] -->|"passed to"| CTR
  CTR["CollectiveTickResult\nagentFitness[ ] · evaluationOrder[ ]"]
  AE["AgentEvaluator\n(agentIndex, field) => number"] -->|"invoked by"| CEC
  OSP["OpponentSnapshotPool\ncapacity · snapshots[ ]"] -->|"contains"| OS
  OS["OpponentSnapshot\nagentId · snapshot · frozenAt"]
```

### Key invariants

- `SharedField.cells` is row-major: cell `(x, y)` maps to index `y × width + x`.
- `CollectiveEvaluationContext.generationTick` starts at `0` and is immutable within a tick;
  it advances only via `resetCollectiveEvaluationState`.
- `CollectiveTickResult.agentFitness` and `evaluationOrder` are parallel arrays of equal length.
- `OpponentSnapshot.snapshot` is a deep-cloned, frozen payload — post-registration mutations
  to the original object are never reflected.

## neat/nge-collective/neat.nge-collective.types.ts

### AgentEvaluator

```ts
AgentEvaluator(
  agentIndex: number,
  field: SharedField,
): number
```

Evaluator function invoked once per agent per collective tick.

Because `writeCell` mutates the shared field in-place, sequential evaluators
within the same tick can observe writes committed by prior agents.

Parameters:
- `agentIndex` - Zero-based index of the agent being evaluated.
- `field` - Live shared field; sequential evaluators see writes from prior agents.

Returns: Scalar fitness value for the evaluated agent.

### CollectiveEvaluationContext

Persistent state carried across collective evaluation ticks.
Returned by `createCollectiveEvaluationContext` and updated via `resetCollectiveEvaluationState`.

### CollectiveTickResult

Result produced by one collective evaluation tick via `runCollectiveEvaluationTick`.

### OpponentSnapshot

A frozen point-in-time snapshot of an opponent agent for rolling tournament evaluation.
Payloads are deep-cloned at registration time so post-registration mutations are not reflected.

### OpponentSnapshotPool

Rolling circular buffer of opponent snapshots with a fixed capacity.
The oldest snapshot is evicted in FIFO order when the pool reaches capacity.

### SharedField

A row-major Float32Array-backed 2D pheromone/signal field shared across all agents
in one collective evaluation tick.

Cell `(x, y)` maps to flat index `y * width + x`.
All cells are initialized to `0` on creation.

Example:

```ts
const field: SharedField = createSharedField(10, 10);
```

## neat/nge-collective/neat.nge-collective.ts

NGE Collective Intelligence boundary — Phase G public surface.

This module is the entry point for all multi-agent collective evaluation primitives
introduced in NGE Phase G. It re-exports three cooperating sub-modules as a single
cohesive boundary so consumers need only one import path.
*Design contract:** classic NEAT behavior is entirely unaffected when this module
is not imported. No global state is mutated at import time.

## Architecture

```mermaid
graph TD
  A["shared-field\nFloat32Array-backed 2D grid\ndecay · diffusion · read/write"] --> B
  B["evaluation\nCollectiveEvaluationContext\nrunCollectiveEvaluationTick\nresetCollectiveEvaluationState"] --> C
  C["metrics\ncomputeRoleDivergenceMetric\ncreateOpponentSnapshotPool\naddOpponentSnapshot"]
  B -->|"passes SharedField\nby reference"| A
```

## Sub-modules

### shared-field
A `Float32Array`-backed 2D grid (`SharedField`) shared across all agents during one
evaluation tick. Agents write signals with `writeCell` and read them with `readCell`.
Because the backing array is passed by reference, sequential evaluators within the same
tick observe each other's writes — exactly the stigmergy contract required by the ant-hive
and racing benchmarks. Between ticks, `applyDecay` and `applyDiffusion` evolve field
dynamics; `clearField` resets it for the next generation.

### evaluation
`createCollectiveEvaluationContext` initialises a generation counter and binds the shared
field. `runCollectiveEvaluationTick` invokes each `AgentEvaluator` in declared order and
returns per-agent fitness plus the evaluation order. `resetCollectiveEvaluationState`
advances the generation tick and zeros the field, ready for the next generation.

### metrics
`computeRoleDivergenceMetric` measures structural divergence between two agents using the
L1 (Manhattan) distance over their module-size distributions — a zero score means identical
compositions. `createOpponentSnapshotPool` and `addOpponentSnapshot` maintain a rolling
fixed-capacity buffer of deep-cloned opponent payloads for tournament evaluation; the oldest
snapshot is evicted FIFO when the pool reaches capacity.

## Determinism contract
Same agent count + same evaluator array + same initial field ⇒ identical
`CollectiveTickResult` for every call. `applyDecay` and `applyDiffusion` are both
deterministic pure functions that produce new `SharedField` instances without mutating
the source.

Example:

```ts
import {
  createSharedField,
  writeCell,
  readCell,
  applyDecay,
  createCollectiveEvaluationContext,
  runCollectiveEvaluationTick,
  resetCollectiveEvaluationState,
  computeRoleDivergenceMetric,
  createOpponentSnapshotPool,
  addOpponentSnapshot,
} from './neat.nge-collective';

// 1. Create a 10×10 pheromone field shared across 3 agents.
const field = createSharedField(10, 10);
const context = createCollectiveEvaluationContext(3, field);

// 2. Run one evaluation tick — agent 0 writes a signal; agent 1 reads it.
const result = runCollectiveEvaluationTick(context, [
  (_idx, f) => { writeCell(f, 0, 0, 1.0); return 10; },
  (_idx, f) => readCell(f, 0, 0) * 5,   // sees agent 0's write
  () => 0,
]);
// result.agentFitness === [10, 5, 0]

// 3. Advance to the next generation with decay applied.
const decayed = applyDecay(context.field, 0.95);
const nextContext = resetCollectiveEvaluationState({ ...context, field: decayed });

// 4. Measure role divergence between two agents' module distributions.
const divergence = computeRoleDivergenceMetric([10, 5, 0], [0, 5, 10]); // 20

// 5. Maintain a rolling opponent pool for tournament selection.
let pool = createOpponentSnapshotPool(5);
pool = addOpponentSnapshot(pool, 'agent:alpha', { fitness: 42 }, nextContext.generationTick);
```

### addOpponentSnapshot

```ts
addOpponentSnapshot(
  pool: OpponentSnapshotPool,
  agentId: string,
  payload: Record<string, unknown>,
  frozenAt: number,
): OpponentSnapshotPool
```

Adds a new snapshot to the opponent pool, evicting the oldest if the pool is at capacity.

The payload is deep-cloned via `safeStructuredClone` at registration time so that
post-registration mutations to the original object are **not** reflected in the stored snapshot.

Returns a **new** `OpponentSnapshotPool`; the original pool is **not** mutated.

Parameters:
- `pool` - Current snapshot pool.
- `agentId` - Stable identifier for the agent being snapshotted.
- `payload` - Arbitrary agent state to freeze at registration time.
- `frozenAt` - Generation tick at which this snapshot is registered.

Returns: A new pool containing the added snapshot, rotated if the pool was at capacity.

Example:

```ts
const updated = addOpponentSnapshot(pool, 'agent:alpha', { fitness: 42 }, 3);
// updated.snapshots[0].agentId === 'agent:alpha'
// updated.snapshots[0].frozenAt === 3
```

### AgentEvaluator

```ts
AgentEvaluator(
  agentIndex: number,
  field: SharedField,
): number
```

Evaluator function invoked once per agent per collective tick.

Because `writeCell` mutates the shared field in-place, sequential evaluators
within the same tick can observe writes committed by prior agents.

Parameters:
- `agentIndex` - Zero-based index of the agent being evaluated.
- `field` - Live shared field; sequential evaluators see writes from prior agents.

Returns: Scalar fitness value for the evaluated agent.

### applyDecay

```ts
applyDecay(
  field: SharedField,
  factor: number,
): SharedField
```

Applies exponential pheromone decay to every cell in the field.
Returns a **new** `SharedField`; the original is **not** mutated.

Each cell's new value is: `oldValue × factor`.

Parameters:
- `field` - Source field to decay.
- `factor` - Decay multiplier in `[0, 1]`. A value of `0.95` retains 95% per tick.

Returns: A new `SharedField` with decayed cell values.

Example:

```ts
const decayed = applyDecay(field, 0.95); // all cells × 0.95
```

### applyDiffusion

```ts
applyDiffusion(
  field: SharedField,
  rate: number,
): SharedField
```

Applies one lateral diffusion step to the shared field.
Returns a **new** `SharedField`; the original is **not** mutated.

Each cell donates `rate × cellValue / neighborCount` to each of its 4-connected
grid neighbors and retains `cellValue × (1 − rate)`. Deterministic: identical
inputs always produce byte-identical outputs.

Parameters:
- `field` - Source field to diffuse.
- `rate` - Diffusion rate in `[0, 1]`. A value of `0.5` spreads half the cell's value.

Returns: A new `SharedField` with diffused cell values.

Example:

```ts
const diffused = applyDiffusion(field, 0.1); // 10% of each cell spreads to neighbors
```

### clearField

```ts
clearField(
  field: SharedField,
): SharedField
```

Zeros every cell in the field and returns a **new** `SharedField`.
The original field is **not** mutated.

Parameters:
- `field` - Source field to clear.

Returns: A new `SharedField` with all cells set to `0`.

Example:

```ts
const clean = clearField(field); // all cells === 0
```

### CollectiveEvaluationContext

Persistent state carried across collective evaluation ticks.
Returned by `createCollectiveEvaluationContext` and updated via `resetCollectiveEvaluationState`.

### CollectiveTickResult

Result produced by one collective evaluation tick via `runCollectiveEvaluationTick`.

### computeRoleDivergenceMetric

```ts
computeRoleDivergenceMetric(
  distributionA: number[],
  distributionB: number[],
): number
```

Computes a role-divergence metric between two module-size distributions using
the L1 (Manhattan) distance — the sum of absolute per-slot differences.

A value of `0` indicates identical distributions. Larger values indicate greater
structural divergence between the two agents' module compositions.

Parameters:
- `distributionA` - Ordered module-size counts for agent A.
- `distributionB` - Ordered module-size counts for agent B.

Returns: Non-negative divergence score (`0` when distributions are equal).

Example:

```ts
computeRoleDivergenceMetric([10, 5], [10, 5]); // 0
computeRoleDivergenceMetric([20, 0], [0, 20]); // 40
```

### createCollectiveEvaluationContext

```ts
createCollectiveEvaluationContext(
  agentCount: number,
  field: SharedField,
): CollectiveEvaluationContext
```

Creates a new collective evaluation context with an initial generation tick of zero.

Parameters:
- `agentCount` - Number of agents participating in collective evaluation.
- `field` - Shared field visible to all agent evaluators within the current tick.

Returns: A new `CollectiveEvaluationContext` ready for the first evaluation tick.

Example:

```ts
const field = createSharedField(10, 10);
const context = createCollectiveEvaluationContext(4, field);
// context.agentCount === 4, context.generationTick === 0
```

### createOpponentSnapshotPool

```ts
createOpponentSnapshotPool(
  capacity: number,
): OpponentSnapshotPool
```

Creates an empty opponent snapshot pool with the given capacity.

The pool is a rolling circular buffer: when at capacity, the oldest snapshot is
evicted in FIFO order to make room for each new addition.

Parameters:
- `capacity` - Maximum number of snapshots retained at any time.

Returns: A new `OpponentSnapshotPool` with an empty snapshot list.

Example:

```ts
const pool = createOpponentSnapshotPool(5);
// pool.capacity === 5, pool.snapshots.length === 0
```

### createSharedField

```ts
createSharedField(
  width: number,
  height: number,
): SharedField
```

Creates a new zeroed shared field with the given dimensions.

The backing store is a `Float32Array` of `width × height` elements, all initialized to `0`.
Cell `(x, y)` maps to flat index `y * width + x` (row-major order).

Parameters:
- `width` - Number of columns in the 2D grid.
- `height` - Number of rows in the 2D grid.

Returns: A new `SharedField` with all cells initialized to `0`.

Example:

```ts
const field = createSharedField(10, 10); // 100-element Float32Array
```

### OpponentSnapshot

A frozen point-in-time snapshot of an opponent agent for rolling tournament evaluation.
Payloads are deep-cloned at registration time so post-registration mutations are not reflected.

### OpponentSnapshotPool

Rolling circular buffer of opponent snapshots with a fixed capacity.
The oldest snapshot is evicted in FIFO order when the pool reaches capacity.

### readCell

```ts
readCell(
  field: SharedField,
  x: number,
  y: number,
): number
```

Reads the value stored at cell `(x, y)` in the shared field.

Parameters:
- `field` - The shared field to read from.
- `x` - Column index (0-based).
- `y` - Row index (0-based).

Returns: The stored cell value, or `0` if the index is out of bounds.

Example:

```ts
const value = readCell(field, 1, 1); // 0.75
```

### resetCollectiveEvaluationState

```ts
resetCollectiveEvaluationState(
  context: CollectiveEvaluationContext,
): CollectiveEvaluationContext
```

Resets the collective evaluation state after a completed generation.

Returns a **new** `CollectiveEvaluationContext` with the generation tick incremented by one
and the shared field zeroed via `clearField`. The original context is **not** mutated.

Parameters:
- `context` - Context from the generation that just completed.

Returns: A new context ready for the next generation's evaluation tick.

Example:

```ts
const nextContext = resetCollectiveEvaluationState(context);
// nextContext.generationTick === context.generationTick + 1
// all cells in nextContext.field === 0
```

### runCollectiveEvaluationTick

```ts
runCollectiveEvaluationTick(
  context: CollectiveEvaluationContext,
  evaluators: AgentEvaluator[],
): CollectiveTickResult
```

Runs one sequential collective evaluation tick.

Evaluators are called in declared order `[0, 1, ..., N-1]`. Because the shared field is
passed by reference and `writeCell` mutates in-place, each evaluator observes writes
committed by all earlier evaluators within the same tick.

Parameters:
- `context` - Current collective evaluation context carrying field and agent count.
- `evaluators` - Ordered array of evaluator functions, one per agent.

Returns: Tick result containing per-agent fitness values and the evaluation order.

Example:

```ts
const result = runCollectiveEvaluationTick(context, [
  (_idx, field) => { writeCell(field, 0, 0, 1.0); return 10; },
  (_idx, field) => readCell(field, 0, 0) * 5,
]);
// result.agentFitness === [10, 5]
```

### SharedField

A row-major Float32Array-backed 2D pheromone/signal field shared across all agents
in one collective evaluation tick.

Cell `(x, y)` maps to flat index `y * width + x`.
All cells are initialized to `0` on creation.

Example:

```ts
const field: SharedField = createSharedField(10, 10);
```

### writeCell

```ts
writeCell(
  field: SharedField,
  x: number,
  y: number,
  value: number,
): SharedField
```

Writes a value to the cell at `(x, y)` in the shared field.

Mutates the backing `Float32Array` **in-place** so that sequential evaluators
within the same collective tick observe each other's writes via the shared reference.
Returns the same `field` reference for fluent chaining.

Parameters:
- `field` - The shared field to write into.
- `x` - Column index (0-based).
- `y` - Row index (0-based).
- `value` - Value to store at the target cell.

Returns: The same `field` reference (mutation is in-place).

Example:

```ts
const field = createSharedField(3, 3);
writeCell(field, 1, 1, 0.75); // field.cells[4] === 0.75
```

## neat/nge-collective/neat.nge-collective.errors.ts

Error classes for the NGE collective multi-agent system (Phase G).

These errors are raised when shared-field operations receive invalid inputs or
when a collective evaluation tick cannot proceed due to an evaluator mismatch.

Both classes forward an optional `{ cause }` to the base `Error` constructor so
that callers can chain underlying exceptions for full stack attribution.

Example:

```ts
import { NgeCollective_FieldDimensionError, NgeCollective_EvaluationError } from './neat.nge-collective.errors';

// Raised when width or height is non-positive or non-integer.
throw new NgeCollective_FieldDimensionError('width must be a positive integer');

// Raised when evaluator array length does not match registered agent count.
throw new NgeCollective_EvaluationError('no evaluator registered for agent 2');
```

### NgeCollective_EvaluationError

Error raised when a collective evaluation tick cannot proceed due to a
mismatched evaluator count or a missing evaluator for a registered agent.

Example:

```ts
throw new NgeCollective_EvaluationError('no evaluator registered for agent 2');
```

### NgeCollective_FieldDimensionError

Error raised when a shared-field operation receives invalid grid dimensions.

Example:

```ts
throw new NgeCollective_FieldDimensionError('width must be a positive integer');
```

## neat/nge-collective/neat.nge-collective.metrics.ts

### addOpponentSnapshot

```ts
addOpponentSnapshot(
  pool: OpponentSnapshotPool,
  agentId: string,
  payload: Record<string, unknown>,
  frozenAt: number,
): OpponentSnapshotPool
```

Adds a new snapshot to the opponent pool, evicting the oldest if the pool is at capacity.

The payload is deep-cloned via `safeStructuredClone` at registration time so that
post-registration mutations to the original object are **not** reflected in the stored snapshot.

Returns a **new** `OpponentSnapshotPool`; the original pool is **not** mutated.

Parameters:
- `pool` - Current snapshot pool.
- `agentId` - Stable identifier for the agent being snapshotted.
- `payload` - Arbitrary agent state to freeze at registration time.
- `frozenAt` - Generation tick at which this snapshot is registered.

Returns: A new pool containing the added snapshot, rotated if the pool was at capacity.

Example:

```ts
const updated = addOpponentSnapshot(pool, 'agent:alpha', { fitness: 42 }, 3);
// updated.snapshots[0].agentId === 'agent:alpha'
// updated.snapshots[0].frozenAt === 3
```

### computeRoleDivergenceMetric

```ts
computeRoleDivergenceMetric(
  distributionA: number[],
  distributionB: number[],
): number
```

Computes a role-divergence metric between two module-size distributions using
the L1 (Manhattan) distance — the sum of absolute per-slot differences.

A value of `0` indicates identical distributions. Larger values indicate greater
structural divergence between the two agents' module compositions.

Parameters:
- `distributionA` - Ordered module-size counts for agent A.
- `distributionB` - Ordered module-size counts for agent B.

Returns: Non-negative divergence score (`0` when distributions are equal).

Example:

```ts
computeRoleDivergenceMetric([10, 5], [10, 5]); // 0
computeRoleDivergenceMetric([20, 0], [0, 20]); // 40
```

### createOpponentSnapshotPool

```ts
createOpponentSnapshotPool(
  capacity: number,
): OpponentSnapshotPool
```

Creates an empty opponent snapshot pool with the given capacity.

The pool is a rolling circular buffer: when at capacity, the oldest snapshot is
evicted in FIFO order to make room for each new addition.

Parameters:
- `capacity` - Maximum number of snapshots retained at any time.

Returns: A new `OpponentSnapshotPool` with an empty snapshot list.

Example:

```ts
const pool = createOpponentSnapshotPool(5);
// pool.capacity === 5, pool.snapshots.length === 0
```

### OpponentSnapshot

A frozen point-in-time snapshot of an opponent agent for rolling tournament evaluation.
Payloads are deep-cloned at registration time so post-registration mutations are not reflected.

### OpponentSnapshotPool

Rolling circular buffer of opponent snapshots with a fixed capacity.
The oldest snapshot is evicted in FIFO order when the pool reaches capacity.

## neat/nge-collective/neat.nge-collective.constants.ts

Named constants for the NGE collective multi-agent system (Phase G).

These defaults control pheromone field dynamics and snapshot pool sizing.
All values are opt-in; classic NEAT behavior is unaffected when the
collective runtime is not active.

Consumers may override any constant by passing explicit values to the
relevant factory or operator function. The constants exist to document
the recommended starting points, not to impose fixed behaviour.

### Tuning guidance

| Constant | Increase effect | Decrease effect |
|---|---|---|
| `NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR` | Signals persist longer, agents rely on older traces | Signals fade quickly, agents must reinforce paths more often |
| `NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE` | Signals spread wider, less spatial specificity | Signals stay local, stronger spatial gradients |
| `NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY` | More diverse opponent history; higher memory cost | Recency-biased selection; lower memory cost |

### NGE_COLLECTIVE_DEFAULT_DECAY_FACTOR

Default pheromone/signal decay factor applied to all cells each simulation tick.
A value of `0.95` means cells retain 95% of their value before the next diffusion step.
Lower values make signals fade faster, encouraging agents to reinforce paths more frequently.

### NGE_COLLECTIVE_DEFAULT_DIFFUSION_RATE

Default lateral diffusion rate applied to all cells each simulation tick.
A value of `0.1` means each cell donates 10% of its value spread evenly across
its 4-connected grid neighbors per tick.

### NGE_COLLECTIVE_DEFAULT_SNAPSHOT_POOL_CAPACITY

Default opponent snapshot pool capacity.
Controls how many historical opponent snapshots are retained for rolling tournament selection.
Older snapshots are evicted in FIFO order when the pool is at capacity.

### NGE_COLLECTIVE_INITIAL_GENERATION_TICK

Initial generation tick value assigned to every new `CollectiveEvaluationContext`.
Starts at zero and increments by one each time `resetCollectiveEvaluationState` is called.

## neat/nge-collective/neat.nge-collective.evaluation.ts

### CollectiveEvaluationContext

Persistent state carried across collective evaluation ticks.
Returned by `createCollectiveEvaluationContext` and updated via `resetCollectiveEvaluationState`.

### CollectiveTickResult

Result produced by one collective evaluation tick via `runCollectiveEvaluationTick`.

### createCollectiveEvaluationContext

```ts
createCollectiveEvaluationContext(
  agentCount: number,
  field: SharedField,
): CollectiveEvaluationContext
```

Creates a new collective evaluation context with an initial generation tick of zero.

Parameters:
- `agentCount` - Number of agents participating in collective evaluation.
- `field` - Shared field visible to all agent evaluators within the current tick.

Returns: A new `CollectiveEvaluationContext` ready for the first evaluation tick.

Example:

```ts
const field = createSharedField(10, 10);
const context = createCollectiveEvaluationContext(4, field);
// context.agentCount === 4, context.generationTick === 0
```

### resetCollectiveEvaluationState

```ts
resetCollectiveEvaluationState(
  context: CollectiveEvaluationContext,
): CollectiveEvaluationContext
```

Resets the collective evaluation state after a completed generation.

Returns a **new** `CollectiveEvaluationContext` with the generation tick incremented by one
and the shared field zeroed via `clearField`. The original context is **not** mutated.

Parameters:
- `context` - Context from the generation that just completed.

Returns: A new context ready for the next generation's evaluation tick.

Example:

```ts
const nextContext = resetCollectiveEvaluationState(context);
// nextContext.generationTick === context.generationTick + 1
// all cells in nextContext.field === 0
```

### runCollectiveEvaluationTick

```ts
runCollectiveEvaluationTick(
  context: CollectiveEvaluationContext,
  evaluators: AgentEvaluator[],
): CollectiveTickResult
```

Runs one sequential collective evaluation tick.

Evaluators are called in declared order `[0, 1, ..., N-1]`. Because the shared field is
passed by reference and `writeCell` mutates in-place, each evaluator observes writes
committed by all earlier evaluators within the same tick.

Parameters:
- `context` - Current collective evaluation context carrying field and agent count.
- `evaluators` - Ordered array of evaluator functions, one per agent.

Returns: Tick result containing per-agent fitness values and the evaluation order.

Example:

```ts
const result = runCollectiveEvaluationTick(context, [
  (_idx, field) => { writeCell(field, 0, 0, 1.0); return 10; },
  (_idx, field) => readCell(field, 0, 0) * 5,
]);
// result.agentFitness === [10, 5]
```

## neat/nge-collective/neat.nge-collective.shared-field.ts

### applyDecay

```ts
applyDecay(
  field: SharedField,
  factor: number,
): SharedField
```

Applies exponential pheromone decay to every cell in the field.
Returns a **new** `SharedField`; the original is **not** mutated.

Each cell's new value is: `oldValue × factor`.

Parameters:
- `field` - Source field to decay.
- `factor` - Decay multiplier in `[0, 1]`. A value of `0.95` retains 95% per tick.

Returns: A new `SharedField` with decayed cell values.

Example:

```ts
const decayed = applyDecay(field, 0.95); // all cells × 0.95
```

### applyDiffusion

```ts
applyDiffusion(
  field: SharedField,
  rate: number,
): SharedField
```

Applies one lateral diffusion step to the shared field.
Returns a **new** `SharedField`; the original is **not** mutated.

Each cell donates `rate × cellValue / neighborCount` to each of its 4-connected
grid neighbors and retains `cellValue × (1 − rate)`. Deterministic: identical
inputs always produce byte-identical outputs.

Parameters:
- `field` - Source field to diffuse.
- `rate` - Diffusion rate in `[0, 1]`. A value of `0.5` spreads half the cell's value.

Returns: A new `SharedField` with diffused cell values.

Example:

```ts
const diffused = applyDiffusion(field, 0.1); // 10% of each cell spreads to neighbors
```

### clearField

```ts
clearField(
  field: SharedField,
): SharedField
```

Zeros every cell in the field and returns a **new** `SharedField`.
The original field is **not** mutated.

Parameters:
- `field` - Source field to clear.

Returns: A new `SharedField` with all cells set to `0`.

Example:

```ts
const clean = clearField(field); // all cells === 0
```

### collectNeighborCoordinates

```ts
collectNeighborCoordinates(
  col: number,
  row: number,
  width: number,
  height: number,
): [number, number][]
```

Collects the 4-connected neighbor coordinates for a given grid cell,
filtering out coordinates that fall outside the grid boundaries.

Parameters:
- `col` - Column index of the source cell.
- `row` - Row index of the source cell.
- `width` - Grid width used as the column boundary.
- `height` - Grid height used as the row boundary.

Returns: Array of valid `[col, row]` neighbor coordinate pairs.

### createSharedField

```ts
createSharedField(
  width: number,
  height: number,
): SharedField
```

Creates a new zeroed shared field with the given dimensions.

The backing store is a `Float32Array` of `width × height` elements, all initialized to `0`.
Cell `(x, y)` maps to flat index `y * width + x` (row-major order).

Parameters:
- `width` - Number of columns in the 2D grid.
- `height` - Number of rows in the 2D grid.

Returns: A new `SharedField` with all cells initialized to `0`.

Example:

```ts
const field = createSharedField(10, 10); // 100-element Float32Array
```

### readCell

```ts
readCell(
  field: SharedField,
  x: number,
  y: number,
): number
```

Reads the value stored at cell `(x, y)` in the shared field.

Parameters:
- `field` - The shared field to read from.
- `x` - Column index (0-based).
- `y` - Row index (0-based).

Returns: The stored cell value, or `0` if the index is out of bounds.

Example:

```ts
const value = readCell(field, 1, 1); // 0.75
```

### SharedField

A row-major Float32Array-backed 2D pheromone/signal field shared across all agents
in one collective evaluation tick.

Cell `(x, y)` maps to flat index `y * width + x`.
All cells are initialized to `0` on creation.

Example:

```ts
const field: SharedField = createSharedField(10, 10);
```

### writeCell

```ts
writeCell(
  field: SharedField,
  x: number,
  y: number,
  value: number,
): SharedField
```

Writes a value to the cell at `(x, y)` in the shared field.

Mutates the backing `Float32Array` **in-place** so that sequential evaluators
within the same collective tick observe each other's writes via the shared reference.
Returns the same `field` reference for fluent chaining.

Parameters:
- `field` - The shared field to write into.
- `x` - Column index (0-based).
- `y` - Row index (0-based).
- `value` - Value to store at the target cell.

Returns: The same `field` reference (mutation is in-place).

Example:

```ts
const field = createSharedField(3, 3);
writeCell(field, 1, 1, 0.75); // field.cells[4] === 0.75
```
