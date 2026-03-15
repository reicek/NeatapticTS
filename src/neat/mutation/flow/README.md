# neat/mutation/flow

Flow helpers for the mutation root orchestration.

This chapter owns the per-genome mutation loop: initialize adaptive state,
resolve effective rates and counts, dispatch operators, and keep operator
statistics in sync with the actual structural outcome.

## neat/mutation/flow/mutation.flow.ts

### applyAddConnMutation

```ts
applyAddConnMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply an ADD_CONN mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyAddNodeMutation

```ts
applyAddNodeMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply an ADD_NODE mutation with reuse and weight nudging.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### applyMutationOperator

```ts
applyMutationOperator(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): void
```

Apply a mutation operator to a genome and invalidate caches as needed.

Parameters:
- `genome` - - genome to mutate
- `mutationMethod` - - mutation operator to apply
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: void

### captureStructuralSizes

```ts
captureStructuralSizes(
  genome: GenomeWithMetadata,
): { beforeNodes: number; beforeConns: number; }
```

Capture structural sizes used to evaluate operator success.

Parameters:
- `genome` - - genome to inspect

Returns: structural size snapshot

### initializeAdaptiveMutation

```ts
initializeAdaptiveMutation(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Initialize per-genome adaptive mutation parameters if configured.

Parameters:
- `genome` - - genome to initialize
- `internal` - - neat controller context

Returns: void

### maybeAddExtraConnection

```ts
maybeAddExtraConnection(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void
```

Optionally add an extra connection to increase exploration.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context

Returns: void

### mutateGenome

```ts
mutateGenome(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): Promise<void>
```

Mutate a single genome based on configured mutation policies.

Parameters:
- `genome` - - genome to mutate
- `internal` - - neat controller context
- `methods` - - mutation methods module

Returns: Promise resolving after mutation attempts complete

### resolveEffectiveAmount

```ts
resolveEffectiveAmount(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number
```

Resolve the effective mutation amount for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation amount

### resolveEffectiveRate

```ts
resolveEffectiveRate(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): number
```

Resolve the effective mutation rate for a genome.

Parameters:
- `genome` - - genome to resolve for
- `internal` - - neat controller context

Returns: effective mutation rate

### selectConcreteMutationMethod

```ts
selectConcreteMutationMethod(
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): Promise<MutationMethod | null>
```

Select a concrete mutation method, resolving any legacy arrays.

Parameters:
- `genome` - - genome to select for
- `internal` - - neat controller context

Returns: resolved mutation method or null

### shouldInvalidateCaches

```ts
shouldInvalidateCaches(
  mutationMethod: MutationMethod,
  methods: { mutation: unknown; },
): boolean
```

Determine whether a mutation method invalidates cached structures.

Parameters:
- `mutationMethod` - - mutation operator to inspect
- `methods` - - mutation methods module

Returns: true when caches should be invalidated

### shouldMutateGenome

```ts
shouldMutateGenome(
  effectiveRate: number,
  internal: NeatControllerForMutation,
): boolean
```

Decide whether a genome should be mutated based on probability.

Parameters:
- `effectiveRate` - - effective mutation probability
- `internal` - - neat controller context

Returns: true when the genome should be mutated

### updateOperatorStatsIfNeeded

```ts
updateOperatorStatsIfNeeded(
  genome: GenomeWithMetadata,
  mutationMethod: MutationMethod,
  beforeSizes: { beforeNodes: number; beforeConns: number; },
  internal: NeatControllerForMutation,
): void
```

Update operator statistics when adaptation is enabled.

Parameters:
- `genome` - - genome used to compute after-sizes
- `mutationMethod` - - operator being recorded
- `beforeSizes` - - structural sizes captured before mutation
- `internal` - - neat controller context

Returns: void
