# neat/mutation/select

Mutation-method selection helpers.

This chapter owns policy resolution for mutation operator choice: normalize
legacy pools, bias the pool for simplify/complexify phases, boost successful
operators, and optionally hand final choice to the operator bandit.

## neat/mutation/select/mutation.select.ts

### applyOperatorAdaptationForSelect

```ts
applyOperatorAdaptationForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[]
```

Apply operator adaptation weighting to the pool when enabled.

Parameters:
- `pool` - - base pool
- `internal` - - neat controller context

Returns: augmented pool

### applyOperatorBanditForSelect

```ts
applyOperatorBanditForSelect(
  pool: MutationMethod[],
  fallbackMethod: MutationMethod,
  internal: NeatControllerForMutation,
): MutationMethod
```

Apply operator bandit selection if enabled.

Parameters:
- `pool` - - operator pool
- `fallbackMethod` - - method used when bandit is disabled
- `internal` - - neat controller context

Returns: selected method

### applyPhasedComplexityForSelect

```ts
applyPhasedComplexityForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod[]
```

Apply phased complexity adjustments to the pool when enabled.

Parameters:
- `pool` - - base operator pool
- `internal` - - neat controller context

Returns: pool with phased complexity adjustments

### isBlockedByRecurrentPolicyForSelect

```ts
isBlockedByRecurrentPolicyForSelect(
  mutationMethod: MutationMethod,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): boolean
```

Check whether a mutation is blocked by recurrent connection policy.

Parameters:
- `mutationMethod` - - mutation operator to check
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isBlockedByStructuralLimitsForSelect

```ts
isBlockedByStructuralLimitsForSelect(
  mutationMethod: MutationMethod,
  genome: GenomeWithMetadata,
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
): boolean
```

Check whether a mutation is blocked by structural limits.

Parameters:
- `mutationMethod` - - mutation operator to check
- `genome` - - genome to inspect
- `internal` - - neat controller context
- `methods` - - methods module

Returns: true when the mutation should be blocked

### isLegacyFFWPoolForSelect

```ts
isLegacyFFWPoolForSelect(
  configuredPool: MutationMethod[],
  methods: { mutation: unknown; },
): boolean
```

Check whether a pool matches the legacy FFW operator ordering.

Parameters:
- `configuredPool` - - configured operator pool
- `methods` - - methods module

Returns: true when the pool matches FFW

### isOperatorNamePrefixedForSelect

```ts
isOperatorNamePrefixedForSelect(
  method: MutationMethod,
  prefix: string,
): boolean
```

Check whether an operator name uses a specific prefix.

Parameters:
- `method` - - mutation operator
- `prefix` - - name prefix to match

Returns: true when the operator name matches the prefix

### normalizeMutationPoolForSelect

```ts
normalizeMutationPoolForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
  rawReturnForTest: boolean,
): MutationMethod[]
```

Normalize the configured mutation pool to a flat operator list.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW for tests

Returns: normalized mutation pool

### resolveFFWPolicyForSelect

```ts
resolveFFWPolicyForSelect(
  internal: NeatControllerForMutation,
  methods: { mutation: unknown; },
  rawReturnForTest: boolean,
): MutationMethod | MutationMethod[] | null
```

Resolve legacy FFW policy behavior, including test-specific returns.

Parameters:
- `internal` - - neat controller context
- `methods` - - methods module
- `rawReturnForTest` - - whether to return raw FFW array for tests

Returns: mutation method or null when not handled

### sampleFromPoolForSelect

```ts
sampleFromPoolForSelect(
  pool: MutationMethod[],
  internal: NeatControllerForMutation,
): MutationMethod | null
```

Sample a random method from the pool.

Parameters:
- `pool` - - operator pool
- `internal` - - neat controller context

Returns: sampled method or null
