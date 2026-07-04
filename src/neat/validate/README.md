# neat/validate

## neat/validate/neat.validate.ts

### assertValidGenomeContract

```ts
assertValidGenomeContract(
  genome: NeatGenome,
): void
```

Assert that one strict genome contract satisfies the Step 7.1 structural
identity rules.

Parameters:
- `genome` - Strict structural genome contract.

Returns: Nothing.

### assertValidNativeGenome

```ts
assertValidNativeGenome(
  genome: default,
): void
```

Assert that a native NEAT genome satisfies the proper-NEAT validation rules.

This is the fail-fast convenience wrapper for dev/test callers that prefer an
exception over manually checking `report.isValid`.

Parameters:
- `genome` - Network-shaped native genome candidate.

Returns: Nothing.

Example:

```ts
assertValidNativeGenome(genome);
```

### NativeGenomeValidationIssue

One validator finding describing a broken native-genome invariant.

`path` points to the runtime field that triggered the issue so test and
debug callers can move directly from the report to the malformed state.

### NativeGenomeValidationIssueCode

Stable issue codes reported by the proper-NEAT native-genome validator.

The codes are intentionally narrow so regression tests can assert the exact
invariant that failed without string-matching full human-readable messages.

### NativeGenomeValidationReport

Validation summary for one native NEAT genome.

The report is designed for dev/test use before speciation, compatibility, or
crossover work begins so malformed native genomes fail loudly and locally.

### NeatNativeGenomeValidationError

Raised when a native NEAT genome violates proper-NEAT identity invariants.

The validator itself returns a rich report for tests and tooling, while this
error class powers fail-fast assertion helpers that want exception semantics.

#### issues

Structured validator findings attached to the thrown error.

### RuntimeConnectionListName

Native-genome validation for the proper-NEAT contract.

This validator is the controller's "trust but verify" seam.

Most internal algorithms assume a *native* genome has already committed to
explicit identity fields:

- every node has a finite `geneId`
- every connection has a finite `innovation`
- endpoints and gaters resolve to runtime nodes
- topology intent agrees with the actual edge orientation

When those invariants are violated, downstream failures can be extremely
indirect (compatibility walks, crossover materialization, speciation, etc.).
`validateNativeGenome()` pushes that failure back toward the write path.

This is also an export/replay safety boundary: checkpoints should not
normalize malformed native state into a snapshot that appears portable.

### validateGenomeContract

```ts
validateGenomeContract(
  genome: NeatGenome,
): NeatGenomeValidationReport
```

Validate one strict genome contract without requiring a live `Network`
phenotype.

Parameters:
- `genome` - Strict structural genome contract.

Returns: Structured genome validation report.

### validateNativeGenome

```ts
validateNativeGenome(
  genome: default,
): NativeGenomeValidationReport
```

Validate one native NEAT genome before compatibility, speciation, or
crossover code consumes it.

Native genomes are expected to carry explicit node gene ids, explicit
connection innovations, resolvable endpoint and gater references, and a
topology-intent contract that agrees with the low-level runtime flags and the
actual edge orientation in memory.

Use this helper in dev and test flows when you want malformed native genomes
to fail near the write path that created them rather than much later inside a
compatibility walk or crossover materialization pass.

Parameters:
- `genome` - Network-shaped native genome candidate to inspect.

Returns: Structured report listing all discovered invariant violations.

Example:

```ts
const report = validateNativeGenome(genome);

if (!report.isValid) {
  console.log(report.issues);
}
```

## neat/validate/neat.validate.types.ts

### NativeGenomeValidationIssue

One validator finding describing a broken native-genome invariant.

`path` points to the runtime field that triggered the issue so test and
debug callers can move directly from the report to the malformed state.

### NativeGenomeValidationIssueCode

Stable issue codes reported by the proper-NEAT native-genome validator.

The codes are intentionally narrow so regression tests can assert the exact
invariant that failed without string-matching full human-readable messages.

### NativeGenomeValidationReport

Validation summary for one native NEAT genome.

The report is designed for dev/test use before speciation, compatibility, or
crossover work begins so malformed native genomes fail loudly and locally.

## neat/validate/neat.validate.errors.ts

### NeatNativeGenomeValidationError

Raised when a native NEAT genome violates proper-NEAT identity invariants.

The validator itself returns a rich report for tests and tooling, while this
error class powers fail-fast assertion helpers that want exception semantics.

#### issues

Structured validator findings attached to the thrown error.
