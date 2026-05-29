# neat/nge-assimilation

Error thrown when one assimilation candidate would exceed the configured structural budget.

## neat/nge-assimilation/neat.nge-assimilation.types.ts

### NgeAssimilationCandidate

Typed Phase D input assembled from one adult equilibrium signal and one module-local structural delta.
This boundary is structural only: it must never carry raw network weights.

### NgeAssimilationModuleDelta

Structural-prior delta captured for one realized module once the adult phase reaches equilibrium.
Each numeric field carries both the current DNA value and the equilibrium-informed target value.

### NgeAssimilationPolicy

Resolved policy bag controlling one Phase D structural-prior write-back attempt.

### NgeAssimilationResult

Outcome returned by one owner-local assimilation attempt for a single module candidate.

## neat/nge-assimilation/neat.nge-assimilation.ts

### assimilateEquilibriumCandidate

```ts
assimilateEquilibriumCandidate(
  candidate: NgeAssimilationCandidate,
  policy: NgeAssimilationPolicy,
): NgeAssimilationResult
```

Orchestrate one equilibrium-candidate assimilation pass inside the owner-local Phase D boundary.

Parameters:
- `candidate` - Structured equilibrium candidate prepared by the adult boundary.
- `policy` - Resolved policy controlling validation, budget handling, and encoding behavior.

Returns: The public result packet for the current assimilation attempt.

## neat/nge-assimilation/neat.nge-assimilation.errors.ts

### AssimilationBudgetError

Error thrown when one assimilation candidate would exceed the configured structural budget.

### AssimilationSchemaError

Error thrown when one assimilation candidate fails schema or envelope validation.

## neat/nge-assimilation/neat.nge-assimilation.constants.ts

### ASSIMILATION_ENCODING_MODES

Supported encoding modes for owner-local assimilation serialization.

### DEFAULT_ASSIMILATION_WRITE_BACK_RATE

Default fraction of one structural gap applied during one assimilation write-back pass.

### DEFAULT_BUDGET_GUARD_ENABLED

Default budget-guard switch for Phase D assimilation updates.

## neat/nge-assimilation/neat.nge-assimilation.writeback.ts

### applyAssimilationWriteback

```ts
applyAssimilationWriteback(
  candidate: NgeAssimilationCandidate,
  policy: NgeAssimilationPolicy,
): NgeAssimilationResult
```

Apply one slow, deterministic structural-prior write-back pass for a single equilibrium candidate.

Parameters:
- `candidate` - One owner-local equilibrium candidate carrying only structural deltas.
- `policy` - Resolved write-back policy for the current assimilation pass.

Returns: The accepted per-module structural-prior update for the DNA-facing shelf.

## neat/nge-assimilation/neat.nge-assimilation.utils.ts

### buildAssimilationResult

```ts
buildAssimilationResult(
  candidate: NgeAssimilationCandidate,
  status: "accepted" | "budget-exceeded" | "schema-invalid",
  policy: NgeAssimilationPolicy,
  updatedModuleDelta: NgeAssimilationModuleDelta | null,
  telemetryOptions: { lossy?: boolean | undefined; },
): NgeAssimilationResult
```

Fold the canonical public result payload for one assimilation attempt.

Parameters:
- `candidate` - Candidate tied to the current owner-local assimilation pass.
- `status` - Terminal status for the current pass.
- `policy` - Resolved policy for the current assimilation pass.
- `updatedModuleDelta` - Updated structural-prior payload, or `null` when rejected.
- `telemetryOptions` - Optional telemetry overrides for the current result.

Returns: The normalized result returned by the assimilation facade.

### buildAssimilationTelemetry

```ts
buildAssimilationTelemetry(
  policy: NgeAssimilationPolicy,
  lossy: boolean,
): { lossy: boolean; budgetGuardEnabled: boolean; }
```

Build the public telemetry payload emitted by the owner-local assimilation boundary.

Parameters:
- `policy` - Resolved policy for the current assimilation pass.
- `lossy` - Whether lossy compression was used during write-back.

Returns: The normalized telemetry packet for the caller-facing result.

### validateAssimilationCandidate

```ts
validateAssimilationCandidate(
  candidate: NgeAssimilationCandidate,
): AssimilationSchemaError | null
```

Validate that one equilibrium candidate can safely enter the owner-local assimilation path.

Parameters:
- `candidate` - Candidate assembled from the adult equilibrium boundary.

Returns: A schema error when the envelope is inconsistent, otherwise `null`.
