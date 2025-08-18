/**
 * benchmark.report.ts
 *
 * Placeholder aggregation logic for benchmark metrics prior to real statistical computation.
 */

/** Raw measurement placeholder describing a single captured timing/memory event. */
export interface RawBenchMeasurement {
  mode: 'src' | 'dist';
  scenario: string;
  size: number;
  metrics: Record<string, number>;
}

/** Aggregated group summarising statistical descriptors for a (mode,scenario,size) key. */
export interface BenchAggregateGroup {
  mode: 'src' | 'dist';
  scenario: string;
  size: number;
  count: number;
}

/**
 * Aggregate raw measurements into groups with central tendency statistics.
 * Phase 0 Implementation: compute mean, median (p50), p95, and standard deviation for each numeric metric key.
 * The function is generic: it discovers metric keys by unioning all record metric objects.
 *
 * @param records Raw measurement records.
 * @returns Array of aggregate groups (order stable by insertion of first occurrence).
 */
/**
 * Aggregate raw measurement records by (mode, scenario, size) computing basic statistics
 * for each numeric metric encountered. All metric keys are imported from the union of
 * input rows to keep this function generic and forward‑compatible.
 */
export function aggregateBenchMeasurements(
  records: RawBenchMeasurement[]
): BenchAggregateGroup[] {
  if (!records.length) return [];
  const groupMap = new Map<
    string,
    BenchAggregateGroup & { _metricSamples: Record<string, number[]> }
  >();
  for (const rec of records) { // Iterate each raw record (Act: grouping)
    const groupKey = `${rec.mode}|${rec.scenario}|${rec.size}`;
    let group = groupMap.get(groupKey);
    if (!group) {
      group = {
        mode: rec.mode,
        scenario: rec.scenario,
        size: rec.size,
        count: 0,
        _metricSamples: {},
      };
      groupMap.set(groupKey, group);
    }
    group.count++;
    // Accumulate metric samples
    for (const [metricName, metricValue] of Object.entries(rec.metrics || {})) {
      if (!group._metricSamples[metricName])
        group._metricSamples[metricName] = [];
      group._metricSamples[metricName].push(metricValue);
    }
  }
  // Finalize statistics
  const finalized: BenchAggregateGroup[] = [];
  for (const group of groupMap.values()) { // Finalize each group (Act: statistic computation)
    const stats: Record<string, number> = {};
    for (const [metricName, samples] of Object.entries(group._metricSamples)) {
      samples.sort((a, b) => a - b);
      const n = samples.length;
      const mean = samples.reduce((s, v) => s + v, 0) / n;
      const p50 = samples[Math.floor(0.5 * (n - 1))];
      const p95 = samples[Math.floor(0.95 * (n - 1))];
      const variance = samples.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
      const stdDev = Math.sqrt(variance);
      stats[`${metricName}Mean`] = mean;
      stats[`${metricName}P50`] = p50;
      stats[`${metricName}P95`] = p95;
      stats[`${metricName}Std`] = stdDev;
    }
    // Spread base group shape
    finalized.push({
      mode: group.mode,
      scenario: group.scenario,
      size: group.size,
      count: group.count,
      // Cast: we intentionally restrict exported interface; extended stats could be exposed later.
      ...(stats as any),
    });
  }
  return finalized;
}

// Tests ------------------------------------------------------------------------------------------

describe('benchmark.report placeholder', () => {
  describe('aggregateBenchMeasurements()', () => {
    it('returns empty array for empty input', () => {
      // Arrange & Act
      const out = aggregateBenchMeasurements([]);
      // Assert
      expect(out.length).toBe(0);
    });
    const identicalKeyRecords: RawBenchMeasurement[] = [
      { mode: 'src', scenario: 'build', size: 1000, metrics: { constructMs: 10 } },
      { mode: 'src', scenario: 'build', size: 1000, metrics: { constructMs: 20 } },
    ];
    const aggregated = aggregateBenchMeasurements(identicalKeyRecords);
    it('produces one group for identical key records', () => {
      expect(aggregated.length).toBe(1);
    });
    it('exposes constructMsMean statistic', () => {
      const g: any = aggregated[0];
      expect(typeof g.constructMsMean).toBe('number');
    });
    it('exposes constructMsP50 statistic', () => {
      const g: any = aggregated[0];
      expect(typeof g.constructMsP50).toBe('number');
    });
  });
});
