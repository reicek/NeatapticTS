/** One synthetic browser benchmark measurement row. */
export interface BrowserBenchRecord {
  size: number;
  buildMs: number;
  fwdAvgMs: number;
  fwdTotalMs: number;
  conn: number;
  nodes: number;
  iterations: number;
}

/** One transport roundtrip comparison row for browser-ready inference payloads. */
export interface BrowserTransportComparisonRecord {
  scenario: string;
  connectionBudget: number;
  populationSize: number;
  iterations: number;
  jsonPrepareAvgMs: number;
  jsonRoundtripAvgMs: number;
  transferablePrepareAvgMs: number;
  transferableRoundtripAvgMs: number;
  jsonApproxBytes: number;
  transferableBytes: number;
  transferableBufferCount: number;
}

/** One async slab-build probe row used for browser responsiveness evidence. */
export interface BrowserAsyncBuildRecord {
  scenario: string;
  connectionBudget: number;
  requestedChunkSize: number;
  chunkTargetMs: number | null;
  frameBudgetMs: number;
  elapsedMs: number;
  microtaskYieldCount: number;
  asyncBuildsDelta: number;
  macrotaskHeartbeatCount: number;
  averageMacrotaskGapMs: number;
  animationFrameCount: number;
  maxAnimationFrameGapMs: number;
  maxMacrotaskGapMs: number;
}

/** Optional browser memory snapshot collected alongside the benchmark payload. */
export interface BrowserPerformanceMemorySnapshot {
  usedJSHeapSize?: number;
  totalJSHeapSize?: number;
  jsHeapSizeLimit?: number;
  userAgentSpecificMemoryBytes?: number;
}

/** Serializable payload attached to `window.__NEATAPTIC_BENCH__`. */
export interface BrowserBenchPayload {
  mode: string;
  generatedAt: string;
  asyncBuildComparisons: BrowserAsyncBuildRecord[];
  performanceMemory: BrowserPerformanceMemorySnapshot | null;
  results: BrowserBenchRecord[];
  transportComparisons: BrowserTransportComparisonRecord[];
}

/** Narrow performance API contract used by the browser harness. */
export interface BrowserBenchPerformanceApi {
  now(): number;
  memory?: {
    usedJSHeapSize?: number;
    totalJSHeapSize?: number;
    jsHeapSizeLimit?: number;
  };
  measureUserAgentSpecificMemory?: () => Promise<{
    bytes: number;
  }>;
}

/** Optional overrides that keep the harness easy to test outside a real browser. */
export interface BrowserBenchHarnessOptions {
  asyncBuildProbeRunner?: BrowserBenchAsyncBuildProbeRunner;
  enableTransportComparisons?: boolean;
  generatedAtFactory?: () => string;
  mode?: string;
  networkFactory?: BrowserBenchNetworkFactory;
  performanceApi?: BrowserBenchPerformanceApi;
  randomSource?: () => number;
  sizes?: readonly number[];
  transferListResolver?: BrowserBenchTransferListResolver;
  transferPayloadExporter?: BrowserBenchTransferPayloadExporter;
  transportScenarios?: readonly TransportScenario[];
}

type BrowserTransferableNetworkPayload = Record<string, unknown>;

interface BrowserGenerationSummaryMessage {
  type: 'generation-ready';
  payload: {
    architectureProfileId: string;
    generation: number;
    bestFitness: number;
    bestNetworkJson?: Record<string, unknown>;
    bestNetworkPayload?: BrowserTransferableNetworkPayload;
    populationNetworksJson?: Record<string, unknown>[];
    populationNetworkPayloads?: BrowserTransferableNetworkPayload[];
  };
}

interface TransportScenario {
  scenario: string;
  connectionBudget: number;
  populationSize: number;
  iterations: number;
}

interface BrowserBenchConnection {
  from: object;
  to: object;
}

interface BrowserBenchNetwork {
  activate(inputValues: number[]): number[];
  connections: BrowserBenchConnection[];
  disconnect(fromNode: object, toNode: object): void;
  input: number;
  nodes: { length: number };
  score?: number;
  toJSON(): Record<string, unknown>;
}

interface ResolvedBrowserBenchHarnessOptions {
  asyncBuildProbeRunner?: BrowserBenchAsyncBuildProbeRunner;
  enableTransportComparisons: boolean;
  generatedAtFactory: () => string;
  mode: string;
  networkFactory: BrowserBenchNetworkFactory;
  performanceApi: BrowserBenchPerformanceApi;
  randomSource: () => number;
  sizes: readonly number[];
  transferListResolver?: BrowserBenchTransferListResolver;
  transferPayloadExporter?: BrowserBenchTransferPayloadExporter;
  transportScenarios: readonly TransportScenario[];
}

type BrowserBenchNetworkFactory = (
  inputCount: number,
  outputCount: number,
) => BrowserBenchNetwork;
type BrowserBenchAsyncBuildProbeRunner = () => Promise<
  BrowserAsyncBuildRecord[]
>;
type BrowserBenchTransferPayloadExporter = (
  network: BrowserBenchNetwork,
) => BrowserTransferableNetworkPayload;
type BrowserBenchTransferListResolver = (
  payload: BrowserTransferableNetworkPayload,
) => ArrayBuffer[];

const DEFAULT_BENCH_SIZES = [1_000, 10_000, 50_000, 100_000] as const;
const DEFAULT_TRANSPORT_SCENARIOS: readonly TransportScenario[] = [
  {
    scenario: 'best-only',
    connectionBudget: 1_500,
    populationSize: 1,
    iterations: 5,
  },
  {
    scenario: 'generation-population-12',
    connectionBudget: 1_500,
    populationSize: 12,
    iterations: 3,
  },
] as const;
const TEXT_ENCODER = new TextEncoder();

/**
 * Collects any browser-exposed memory readings that are currently available.
 *
 * @param performanceApi Narrow performance API abstraction.
 * @returns Heap snapshot data when at least one reading is present, else `null`.
 */
export async function collectBrowserPerformanceMemorySnapshot(
  performanceApi: BrowserBenchPerformanceApi,
): Promise<BrowserPerformanceMemorySnapshot | null> {
  const memorySnapshot: BrowserPerformanceMemorySnapshot = {};

  // Step 1: Copy Chromium heap readings when the browser exposes them.
  if (performanceApi.memory?.usedJSHeapSize !== undefined) {
    memorySnapshot.usedJSHeapSize = performanceApi.memory.usedJSHeapSize;
  }
  if (performanceApi.memory?.totalJSHeapSize !== undefined) {
    memorySnapshot.totalJSHeapSize = performanceApi.memory.totalJSHeapSize;
  }
  if (performanceApi.memory?.jsHeapSizeLimit !== undefined) {
    memorySnapshot.jsHeapSizeLimit = performanceApi.memory.jsHeapSizeLimit;
  }

  // Step 2: Add the UA-specific memory reading when the browser supports it.
  const userAgentSpecificMemoryBytes = await performanceApi
    .measureUserAgentSpecificMemory?.()
    .then((browserSpecificSnapshot) => browserSpecificSnapshot.bytes)
    .catch(() => undefined);
  if (userAgentSpecificMemoryBytes !== undefined) {
    memorySnapshot.userAgentSpecificMemoryBytes = userAgentSpecificMemoryBytes;
  }

  return Object.keys(memorySnapshot).length === 0 ? null : memorySnapshot;
}

/**
 * Runs the browser benchmark harness and returns the serializable payload.
 *
 * @param options Optional overrides for deterministic testing and alternate runners.
 * @returns Browser benchmark payload attached to the page by `bench-entry.ts`.
 */
export async function runBrowserBenchPayload(
  options: BrowserBenchHarnessOptions = {},
): Promise<BrowserBenchPayload> {
  const resolvedOptions = resolveBrowserBenchHarnessOptions(options);
  const benchmarkRecords = resolvedOptions.sizes.map((size) => {
    return runSyntheticBenchmark(size, resolvedOptions);
  });
  const asyncBuildComparisons =
    (await resolvedOptions.asyncBuildProbeRunner?.()) ?? [];
  const transportComparisons = resolvedOptions.enableTransportComparisons
    ? await runTransportComparisons(resolvedOptions)
    : [];
  const performanceMemory = await collectBrowserPerformanceMemorySnapshot(
    resolvedOptions.performanceApi,
  );

  return {
    mode: resolvedOptions.mode,
    generatedAt: resolvedOptions.generatedAtFactory(),
    asyncBuildComparisons,
    performanceMemory,
    results: benchmarkRecords,
    transportComparisons,
  };
}

function resolveBrowserBenchHarnessOptions(
  options: BrowserBenchHarnessOptions,
): ResolvedBrowserBenchHarnessOptions {
  return {
    asyncBuildProbeRunner: options.asyncBuildProbeRunner,
    enableTransportComparisons: options.enableTransportComparisons ?? true,
    generatedAtFactory:
      options.generatedAtFactory ?? (() => new Date().toISOString()),
    mode: options.mode ?? '__UNDEF__',
    networkFactory:
      options.networkFactory ??
      (() => {
        throw new Error(
          'runBrowserBenchPayload requires a networkFactory in non-browser adapter contexts.',
        );
      }),
    performanceApi:
      options.performanceApi ?? (performance as BrowserBenchPerformanceApi),
    randomSource: options.randomSource ?? (() => Math.random()),
    sizes: options.sizes ?? DEFAULT_BENCH_SIZES,
    transferListResolver: options.transferListResolver,
    transferPayloadExporter: options.transferPayloadExporter,
    transportScenarios:
      options.transportScenarios ?? DEFAULT_TRANSPORT_SCENARIOS,
  };
}

function runSyntheticBenchmark(
  size: number,
  options: ResolvedBrowserBenchHarnessOptions,
): BrowserBenchRecord {
  const { net, buildMs } = buildSyntheticNetwork(size, options);
  const iterations = resolveIterationBudget(size);
  const { totalMs, avgMs } = measureForward(net, iterations, options);

  return {
    size,
    buildMs: Number(buildMs.toFixed(3)),
    fwdAvgMs: Number(avgMs.toFixed(4)),
    fwdTotalMs: Number(totalMs.toFixed(3)),
    conn: net.connections.length,
    nodes: net.nodes.length,
    iterations,
  };
}

function buildSyntheticNetwork(
  size: number,
  options: ResolvedBrowserBenchHarnessOptions,
): {
  net: BrowserBenchNetwork;
  buildMs: number;
} {
  const buildStartedAt = options.performanceApi.now();
  const inputCount = Math.max(1, Math.floor(Math.sqrt(size)));
  const outputCount = Math.max(1, Math.ceil(size / inputCount));
  const net = options.networkFactory(inputCount, outputCount);

  while (net.connections.length > size) {
    const connectionIndex = Math.floor(
      options.randomSource() * net.connections.length,
    );
    const connection = net.connections[connectionIndex];
    if (!connection) {
      break;
    }
    net.disconnect(connection.from, connection.to);
  }

  return {
    net,
    buildMs: options.performanceApi.now() - buildStartedAt,
  };
}

function resolveIterationBudget(size: number): number {
  if (size >= 100_000) {
    return 2;
  }
  if (size >= 50_000) {
    return 3;
  }
  return 5;
}

function measureForward(
  net: BrowserBenchNetwork,
  iterations: number,
  options: ResolvedBrowserBenchHarnessOptions,
): { totalMs: number; avgMs: number } {
  const inputVector = Array.from({ length: net.input }, () => {
    return options.randomSource();
  });
  const forwardStartedAt = options.performanceApi.now();

  for (
    let iterationIndex = 0;
    iterationIndex < iterations;
    iterationIndex += 1
  ) {
    net.activate(inputVector);
  }

  const totalMs = options.performanceApi.now() - forwardStartedAt;

  return {
    totalMs,
    avgMs: totalMs / iterations,
  };
}

async function runTransportComparisons(
  options: ResolvedBrowserBenchHarnessOptions,
): Promise<BrowserTransportComparisonRecord[]> {
  if (!options.transferListResolver || !options.transferPayloadExporter) {
    return [];
  }

  const transferListResolver = options.transferListResolver;
  const transferPayloadExporter = options.transferPayloadExporter;

  const comparisonRecords: BrowserTransportComparisonRecord[] = [];

  for (const transportScenario of options.transportScenarios) {
    const scenarioNetworks = Array.from(
      { length: transportScenario.populationSize },
      () =>
        buildSyntheticNetwork(transportScenario.connectionBudget, options).net,
    );
    comparisonRecords.push(
      await measureTransportScenario(
        scenarioNetworks,
        transportScenario,
        options,
        transferPayloadExporter,
        transferListResolver,
      ),
    );
  }

  return comparisonRecords;
}

async function measureTransportScenario(
  networks: readonly BrowserBenchNetwork[],
  transportScenario: TransportScenario,
  options: ResolvedBrowserBenchHarnessOptions,
  transferPayloadExporter: BrowserBenchTransferPayloadExporter,
  transferListResolver: BrowserBenchTransferListResolver,
): Promise<BrowserTransportComparisonRecord> {
  let jsonPrepareTotalMs = 0;
  let jsonRoundtripTotalMs = 0;
  let transferablePrepareTotalMs = 0;
  let transferableRoundtripTotalMs = 0;
  let jsonApproxBytes = 0;
  let transferableBytes = 0;
  let transferableBufferCount = 0;

  for (
    let iterationIndex = 0;
    iterationIndex < transportScenario.iterations;
    iterationIndex += 1
  ) {
    const jsonBuild = buildJsonGenerationSummaryMessage(
      networks,
      options.performanceApi,
    );
    jsonPrepareTotalMs += jsonBuild.prepareMs;
    jsonRoundtripTotalMs += await measureMessageRoundtrip({
      message: jsonBuild.message,
      resolveEchoTransferList: () => [],
    });
    jsonApproxBytes = jsonBuild.approxBytes;

    const transferableBuild = buildTransferableGenerationSummaryMessage(
      networks,
      options.performanceApi,
      transferPayloadExporter,
      transferListResolver,
    );
    transferablePrepareTotalMs += transferableBuild.prepareMs;
    transferableRoundtripTotalMs += await measureMessageRoundtrip({
      message: transferableBuild.message,
      transferList: transferableBuild.transferList,
      resolveEchoTransferList: (message) => {
        return resolveGenerationSummaryTransferList(
          message,
          transferListResolver,
        );
      },
    });
    transferableBytes = transferableBuild.transferBytes;
    transferableBufferCount = transferableBuild.transferList.length;
  }

  return {
    scenario: transportScenario.scenario,
    connectionBudget: transportScenario.connectionBudget,
    populationSize: transportScenario.populationSize,
    iterations: transportScenario.iterations,
    jsonPrepareAvgMs: Number(
      (jsonPrepareTotalMs / transportScenario.iterations).toFixed(4),
    ),
    jsonRoundtripAvgMs: Number(
      (jsonRoundtripTotalMs / transportScenario.iterations).toFixed(4),
    ),
    transferablePrepareAvgMs: Number(
      (transferablePrepareTotalMs / transportScenario.iterations).toFixed(4),
    ),
    transferableRoundtripAvgMs: Number(
      (transferableRoundtripTotalMs / transportScenario.iterations).toFixed(4),
    ),
    jsonApproxBytes,
    transferableBytes,
    transferableBufferCount,
  };
}

function buildJsonGenerationSummaryMessage(
  networks: readonly BrowserBenchNetwork[],
  performanceApi: BrowserBenchPerformanceApi,
): {
  message: BrowserGenerationSummaryMessage;
  prepareMs: number;
  approxBytes: number;
} {
  const prepareStartedAt = performanceApi.now();
  const bestNetwork = networks[0];
  const message: BrowserGenerationSummaryMessage = {
    type: 'generation-ready',
    payload: {
      architectureProfileId: 'benchmark',
      generation: 1,
      bestFitness: Number(bestNetwork?.score ?? 0),
      bestNetworkJson: bestNetwork?.toJSON(),
      populationNetworksJson: networks.map((network) => network.toJSON()),
    },
  };

  return {
    message,
    prepareMs: performanceApi.now() - prepareStartedAt,
    approxBytes: TEXT_ENCODER.encode(JSON.stringify(message)).byteLength,
  };
}

function buildTransferableGenerationSummaryMessage(
  networks: readonly BrowserBenchNetwork[],
  performanceApi: BrowserBenchPerformanceApi,
  transferPayloadExporter: BrowserBenchTransferPayloadExporter,
  transferListResolver: BrowserBenchTransferListResolver,
): {
  message: BrowserGenerationSummaryMessage;
  prepareMs: number;
  transferList: ArrayBuffer[];
  transferBytes: number;
} {
  const prepareStartedAt = performanceApi.now();
  const bestNetwork = networks[0];
  const bestNetworkPayload = bestNetwork
    ? transferPayloadExporter(bestNetwork)
    : undefined;
  const populationNetworkPayloads = networks.map((network) => {
    return transferPayloadExporter(network);
  });
  const message: BrowserGenerationSummaryMessage = {
    type: 'generation-ready',
    payload: {
      architectureProfileId: 'benchmark',
      generation: 1,
      bestFitness: Number(bestNetwork?.score ?? 0),
      bestNetworkPayload,
      populationNetworkPayloads,
    },
  };
  const transferList = resolveGenerationSummaryTransferList(
    message,
    transferListResolver,
  );

  return {
    message,
    prepareMs: performanceApi.now() - prepareStartedAt,
    transferList,
    transferBytes: transferList.reduce((totalBytes, transferableBuffer) => {
      return totalBytes + transferableBuffer.byteLength;
    }, 0),
  };
}

function resolveGenerationSummaryTransferList(
  message: BrowserGenerationSummaryMessage,
  transferListResolver: BrowserBenchTransferListResolver,
): ArrayBuffer[] {
  const bestNetworkTransferList = message.payload.bestNetworkPayload
    ? transferListResolver(message.payload.bestNetworkPayload)
    : [];
  const populationTransferList =
    message.payload.populationNetworkPayloads?.flatMap(
      (populationNetworkPayload) =>
        transferListResolver(populationNetworkPayload),
    ) ?? [];

  return [...bestNetworkTransferList, ...populationTransferList];
}

function measureMessageRoundtrip(options: {
  message: BrowserGenerationSummaryMessage;
  transferList?: Transferable[];
  resolveEchoTransferList: (
    message: BrowserGenerationSummaryMessage,
  ) => Transferable[];
}): Promise<number> {
  return new Promise((resolve, reject) => {
    const messageChannel = new MessageChannel();
    const startedAt = performance.now();

    const closePorts = () => {
      messageChannel.port1.close();
      messageChannel.port2.close();
    };

    messageChannel.port2.onmessage = (event) => {
      try {
        const echoedMessage = event.data as BrowserGenerationSummaryMessage;
        messageChannel.port2.postMessage(
          echoedMessage,
          options.resolveEchoTransferList(echoedMessage),
        );
      } catch (error) {
        closePorts();
        reject(error);
      }
    };

    messageChannel.port1.onmessage = () => {
      const elapsedMs = performance.now() - startedAt;
      closePorts();
      resolve(elapsedMs);
    };

    try {
      messageChannel.port1.postMessage(
        options.message,
        options.transferList ?? [],
      );
    } catch (error) {
      closePorts();
      reject(error);
    }
  });
}
