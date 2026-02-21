# architecture/network/stats

## architecture/network/stats/network.stats.utils.ts

### getRegularizationStats

`() => Record<string, unknown> | null`

Obtain the last recorded regularization / stochastic statistics snapshot.

Returns a defensive deep copy so callers can inspect metrics without risking mutation of the
internal `_lastStats` object maintained by the training loop (e.g., during pruning, dropout, or
noise scheduling updates).

Returns: A deep-cloned stats object or null if no stats have been recorded yet.
