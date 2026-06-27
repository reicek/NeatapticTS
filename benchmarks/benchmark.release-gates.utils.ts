/**
 * Benchmark release-gate names enforced during the Phase 10 hardening pass.
 */
export type BenchmarkReleaseGateName =
  'variance' | 'memory' | 'determinism' | 'audit';

/**
 * Failure emitted when one benchmark release gate does not pass.
 */
export interface BenchmarkReleaseGateFailure {
  name: BenchmarkReleaseGateName;
  message: string;
}

/**
 * Final outcome of the benchmark release-gate evaluation.
 */
export interface BenchmarkReleaseGateResult {
  failures: BenchmarkReleaseGateFailure[];
  passed: boolean;
}

interface BenchmarkArtifactVarianceEntry {
  buildMsCvPct: number;
  fwdAvgMsCvPct: number;
  mode: string;
  samples: number;
  size: number;
}

interface BenchmarkArtifactAggregateEntry {
  bytesPerConnMean?: number;
  mode?: string;
  scenario?: string;
  size: number;
}

interface BenchmarkArtifactHistorySummaryEntry {
  bytesPerConnMean?: number;
  size: number;
}

interface BenchmarkArtifactHistoryEntry {
  commit?: string;
  distBundle?: {
    bytes?: number;
    exists?: boolean;
    hash?: string;
  };
  generatedAt?: string;
  summary?: BenchmarkArtifactHistorySummaryEntry[];
}

interface BenchmarkArtifactDeterminismReplay {
  checks?: Array<{
    passed?: boolean;
    scenario?: string;
  }>;
  generatedAt?: string;
  passed?: boolean;
}

/**
 * Minimal benchmark artifact contract required by the Phase 10 release gates.
 */
export interface BenchmarkReleaseGateArtifact {
  aggregated?: BenchmarkArtifactAggregateEntry[];
  determinismReplay?: BenchmarkArtifactDeterminismReplay;
  generatedAt?: string;
  history?: BenchmarkArtifactHistoryEntry[];
  meta?: {
    distBundle?: {
      bytes?: number;
      exists?: boolean;
      hash?: string;
    };
    varianceRepeatsLarge?: number;
  };
  variance?: BenchmarkArtifactVarianceEntry[];
}

/**
 * Maximum allowed bytes-per-connection growth versus the rolling history median.
 */
export const MEMORY_REGRESSION_DELTA_PCT_THRESHOLD = 5;

/**
 * Maximum allowed build coefficient of variation for the monitored large benchmark sizes.
 */
export const VARIANCE_BUILD_CV_PCT_THRESHOLD = 20;

/**
 * Maximum allowed forward-pass coefficient of variation for the monitored large benchmark sizes.
 */
export const VARIANCE_FORWARD_CV_PCT_THRESHOLD = 35;

/**
 * Large synthetic benchmark sizes that must carry stable variance evidence.
 */
export const VARIANCE_MONITORED_SIZES = [100000, 200000] as const;

/**
 * Evaluate the benchmark artifact against the Phase 10 hard release gates.
 *
 * @param artifact - Parsed benchmark artifact to validate.
 * @returns Consolidated release-gate result.
 *
 * @example
 * ```ts
 * const gateResult = evaluateBenchmarkReleaseGates(artifact);
 * if (!gateResult.passed) {
 *   throw new Error(gateResult.failures.map((failure) => failure.message).join('\n'));
 * }
 * ```
 */
export function evaluateBenchmarkReleaseGates(
  artifact: BenchmarkReleaseGateArtifact,
): BenchmarkReleaseGateResult {
  const failures = [
    ...evaluateVarianceFailures(artifact),
    ...evaluateMemoryFailures(artifact),
    ...evaluateDeterminismFailures(artifact),
    ...evaluateAuditFailures(artifact),
  ];

  return {
    failures,
    passed: failures.length === 0,
  };
}

function evaluateVarianceFailures(
  artifact: BenchmarkReleaseGateArtifact,
): BenchmarkReleaseGateFailure[] {
  const configuredRepeats = artifact.meta?.varianceRepeatsLarge ?? 0;
  const varianceEntries = Array.isArray(artifact.variance)
    ? artifact.variance
    : [];

  if (configuredRepeats <= 1) {
    return [
      {
        message:
          'Variance gate requires meta.varianceRepeatsLarge greater than 1 so large-size stability is actually measured.',
        name: 'variance',
      },
    ];
  }

  return VARIANCE_MONITORED_SIZES.flatMap((monitoredSize) => {
    const matchingEntry = varianceEntries.find(
      (entry) => entry.size === monitoredSize,
    );

    if (!matchingEntry) {
      return [
        {
          message: `Variance gate is missing the monitored size ${monitoredSize}.`,
          name: 'variance',
        },
      ];
    }

    if (matchingEntry.samples < configuredRepeats) {
      return [
        {
          message: `Variance gate recorded only ${matchingEntry.samples} samples for ${monitoredSize} but requires ${configuredRepeats}.`,
          name: 'variance',
        },
      ];
    }

    if (
      matchingEntry.buildMsCvPct > VARIANCE_BUILD_CV_PCT_THRESHOLD ||
      matchingEntry.fwdAvgMsCvPct > VARIANCE_FORWARD_CV_PCT_THRESHOLD
    ) {
      return [
        {
          message:
            `Variance gate exceeded build=${VARIANCE_BUILD_CV_PCT_THRESHOLD}% or ` +
            `forward=${VARIANCE_FORWARD_CV_PCT_THRESHOLD}% at ${monitoredSize} ` +
            `(build=${matchingEntry.buildMsCvPct.toFixed(2)}%, forward=${matchingEntry.fwdAvgMsCvPct.toFixed(2)}%).`,
          name: 'variance',
        },
      ];
    }

    return [];
  });
}

function evaluateMemoryFailures(
  artifact: BenchmarkReleaseGateArtifact,
): BenchmarkReleaseGateFailure[] {
  const currentAggregatedEntries = Array.isArray(artifact.aggregated)
    ? artifact.aggregated
    : [];
  const priorHistoryEntries = Array.isArray(artifact.history)
    ? artifact.history.slice(0, -1)
    : [];

  return currentAggregatedEntries.flatMap((currentEntry) => {
    const currentBytesPerConnection = currentEntry.bytesPerConnMean;

    if (typeof currentBytesPerConnection !== 'number') {
      return [];
    }

    const priorSamples = priorHistoryEntries
      .flatMap((historyEntry) => historyEntry.summary ?? [])
      .filter((summaryEntry) => summaryEntry.size === currentEntry.size)
      .map((summaryEntry) => summaryEntry.bytesPerConnMean)
      .filter((sample): sample is number => typeof sample === 'number');

    if (priorSamples.length === 0) {
      return [];
    }

    const medianBaseline = computeMedian(priorSamples);
    const deltaPct =
      ((currentBytesPerConnection - medianBaseline) / medianBaseline) * 100;

    if (deltaPct <= MEMORY_REGRESSION_DELTA_PCT_THRESHOLD) {
      return [];
    }

    return [
      {
        message:
          `Memory gate detected a ${deltaPct.toFixed(2)}% bytes-per-connection regression at ${currentEntry.size} ` +
          `(baseline=${medianBaseline.toFixed(2)}, current=${currentBytesPerConnection.toFixed(2)}).`,
        name: 'memory',
      },
    ];
  });
}

function evaluateDeterminismFailures(
  artifact: BenchmarkReleaseGateArtifact,
): BenchmarkReleaseGateFailure[] {
  const determinismReplay = artifact.determinismReplay;
  const determinismChecks = determinismReplay?.checks ?? [];

  if (!determinismReplay) {
    return [
      {
        message:
          'Determinism gate requires a persisted determinismReplay section.',
        name: 'determinism',
      },
    ];
  }

  if (!determinismReplay.generatedAt) {
    return [
      {
        message:
          'Determinism gate requires determinismReplay.generatedAt for auditability.',
        name: 'determinism',
      },
    ];
  }

  if (!determinismReplay.passed) {
    return [
      {
        message:
          'Determinism gate requires determinismReplay.passed to be true.',
        name: 'determinism',
      },
    ];
  }

  if (!determinismChecks.length) {
    return [
      {
        message:
          'Determinism gate requires at least one persisted replay check.',
        name: 'determinism',
      },
    ];
  }

  const failedCheck = determinismChecks.find((check) => !check.passed);

  if (!failedCheck) {
    return [];
  }

  return [
    {
      message: `Determinism gate failed for scenario ${failedCheck.scenario ?? 'unknown'}.`,
      name: 'determinism',
    },
  ];
}

function evaluateAuditFailures(
  artifact: BenchmarkReleaseGateArtifact,
): BenchmarkReleaseGateFailure[] {
  const historyEntries = Array.isArray(artifact.history)
    ? artifact.history
    : [];
  const latestHistoryEntry = historyEntries.at(-1);
  const currentDistBundle = artifact.meta?.distBundle;

  if (!artifact.generatedAt) {
    return [
      {
        message: 'Audit gate requires a top-level generatedAt timestamp.',
        name: 'audit',
      },
    ];
  }

  if (!latestHistoryEntry) {
    return [
      {
        message: 'Audit gate requires at least one history snapshot.',
        name: 'audit',
      },
    ];
  }

  if (
    !latestHistoryEntry.generatedAt ||
    latestHistoryEntry.generatedAt !== artifact.generatedAt
  ) {
    return [
      {
        message:
          'Audit gate requires the latest history snapshot timestamp to match the current artifact timestamp.',
        name: 'audit',
      },
    ];
  }

  if (!latestHistoryEntry.commit) {
    return [
      {
        message:
          'Audit gate requires the latest history snapshot to record a commit.',
        name: 'audit',
      },
    ];
  }

  if (
    !currentDistBundle?.exists ||
    !currentDistBundle.hash ||
    !currentDistBundle.bytes
  ) {
    return [
      {
        message:
          'Audit gate requires meta.distBundle with exists, hash, and bytes populated.',
        name: 'audit',
      },
    ];
  }

  if (!latestHistoryEntry.distBundle?.hash) {
    return [
      {
        message:
          'Audit gate requires the latest history snapshot to persist the dist bundle hash.',
        name: 'audit',
      },
    ];
  }

  if (latestHistoryEntry.distBundle.hash !== currentDistBundle.hash) {
    return [
      {
        message:
          'Audit gate requires the latest history snapshot dist bundle hash to match meta.distBundle.hash.',
        name: 'audit',
      },
    ];
  }

  return [];
}

function computeMedian(samples: number[]): number {
  const sortedSamples = samples.toSorted((leftSample, rightSample) => {
    return leftSample - rightSample;
  });
  const middleIndex = Math.floor((sortedSamples.length - 1) / 2);

  return sortedSamples[middleIndex] ?? 0;
}
