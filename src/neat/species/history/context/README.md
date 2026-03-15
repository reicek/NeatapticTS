# neat/species/history/context

## neat/species/history/context/species.history.context.ts

### ResolvedSpeciesHistoryContext

Resolved host data needed to serve a species-history read.

The public `getSpeciesHistory()` facade should not need to know how the NEAT
controller stores its history buffer or which internal fields are required to
support optional extended-history augmentation. This context keeps that
plumbing in one place.

### resolveSpeciesHistoryContext

```ts
resolveSpeciesHistoryContext(
  host: NeatLike,
): ResolvedSpeciesHistoryContext
```

Resolve the stored history buffer, augmentation context, and options needed
by the public species-history read path.

Parameters:
- `host` - - NEAT host exposing species history, species records, fallback innovation logic, and options.

Returns: Normalized history-read context for the species facade.

Example:

```ts
const historyContext = resolveSpeciesHistoryContext(neat);
console.log(historyContext.speciesHistory.length);
```
