# neat/species/history/read

Bounded orchestration for species-history reads.

The public `species/` chapter promises a simple read surface: ask for the
recent species story and receive generation-stamped history rows. This file
is the narrow orchestration seam that keeps that promise without leaking
controller storage details into the facade.

The read path has three stages:

1. resolve one normalized context from the host's stored history buffer,
   optional augmentation inputs, and current options,
2. ask the species core policy gate whether extended history should be
   enriched for this read,
3. return the stable history buffer, optionally after in-place backfill of
   missing extended fields.

That split is intentional. `context/` owns how raw host fields are gathered,
`core/` owns whether extended augmentation should run and how it fills missing
metrics, and this file owns the small read-time orchestration that composes
those two responsibilities into one public helper.

## neat/species/history/read/species.history.read.ts

### getSpeciesHistory

```ts
getSpeciesHistory(
  host: NeatLike,
): SpeciesHistoryEntry[]
```

Read the recorded species history for a NEAT host.

This history-read chapter keeps the public species facade thin by owning the
only remaining orchestration in the history path: resolve the stored history
context, optionally backfill extended metrics, then return the normalized
history buffer.

Read this helper as the controller-facing handoff between raw storage and a
usable reporting surface. It does not compute species history from scratch,
and it does not force every caller to understand internal history buffers or
augmentation rules. Instead it resolves the normalized read context once,
lets the policy layer decide whether optional enrichment is warranted, and
then returns the history rows in the shape the public facade expects.

The orchestration stays deliberately small:

1. gather the normalized history-read context,
2. backfill extended fields only when the current options allow it,
3. return the same history array as the public read result.

That makes the helper easy to trust during telemetry, debugging, and export
flows. Callers always get the recorded history buffer, and the only variable
behavior is whether missing extended fields are enriched before that buffer
is handed back.

Parameters:
- `host` - - NEAT host exposing species history, species records, fallback innovation logic, and options.

Returns: Generation-stamped species history snapshots.

Example:

```ts
const historyEntries = getSpeciesHistory(neat);

if (historyEntries.at(-1)?.innovationRange) {
  console.log('Extended history fields were available for this read.');
}

console.log(historyEntries.at(-1));
```
