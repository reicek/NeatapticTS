# neat/init

Constructor bootstrap helpers for the shared NEAT controller facade.

This chapter isolates the public `Neat` startup sequence so the main facade
can stay focused on the long-lived class surface while constructor-time
policy remains readable in one place. The helper deliberately preserves the
legacy ordering that existing tests rely on: apply defaults, prepare internal
controller state, attempt initial pool creation, then switch on lineage and
deterministic RNG access.

## neat/init/neat.init.ts

### initializeNeatConstructor

```ts
initializeNeatConstructor(
  host: NeatInitializationHost,
  request: InitializeNeatConstructorRequest,
): void
```

Apply the legacy constructor bootstrap sequence behind the public `Neat`
facade.

This helper exists to keep [src/neat.ts](src/neat.ts) focused on the public
class surface while preserving the exact startup order that current tests and
migrated helpers rely on: mutate the caller-supplied options bag in place,
initialize controller state, optionally create the starting population, then
enable lineage tracking and bind the RNG accessor.

Parameters:
- `host` - - `Neat` instance receiving constructor-time side effects.
- `request` - - Mutable options bag, raw constructor options, and public
default values exported by the facade.

Returns: Nothing. The helper mutates `host` and `request.optionBag` in place.

Example:

```ts
initializeNeatConstructor(this, {
  optionBag: this.options,
  rawOptions: options,
  defaults: publicDefaults,
});
```
