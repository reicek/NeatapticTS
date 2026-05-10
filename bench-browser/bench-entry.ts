/*
 * Browser Benchmark Entry (Phase 4 harness extraction)
 * Runs the reusable browser harness and exposes the serialized payload on the page.
 */
// @ts-ignore dist build assumed present after npm run build
import {
  exportTransferableInferencePayload,
  getTransferList,
  Network,
  Node,
  config,
} from '../dist/neataptic.js';
import { rebuildConnectionSlabAsync } from '../dist/architecture/network/network.utils.js';
import { runBrowserBenchPayload } from './harness';

interface WindowWithBrowserBench extends Window {
  __BENCH_MODE__?: string;
  __NEATAPTIC_BENCH__?: Awaited<ReturnType<typeof runBrowserBenchPayload>>;
}

interface BrowserAsyncProbeNetwork extends Network {
  _nodeIndexDirty?: boolean;
  _slabAsyncBuilds?: number;
  _topoDirty?: boolean;
}

interface ProbeScenario {
  scenario: string;
  connectionBudget: number;
  requestedChunkSize: number;
  chunkTargetMs: number;
}

const ASYNC_BUILD_PROBE_SCENARIOS: readonly ProbeScenario[] = [
  {
    scenario: 'dense-110k-target-1ms',
    connectionBudget: 110_000,
    requestedChunkSize: 10_000,
    chunkTargetMs: 1,
  },
] as const;
const BROWSER_FRAME_BUDGET_MS = 16.7;

void runBrowserBenchPayload({
  asyncBuildProbeRunner: runAsyncBuildProbeComparisons,
  mode: (window as WindowWithBrowserBench).__BENCH_MODE__ ?? '__UNDEF__',
  networkFactory: (inputCount, outputCount) => {
    return new Network(inputCount, outputCount);
  },
  transferListResolver: (payload) => getTransferList(payload),
  transferPayloadExporter: (network) => exportTransferableInferencePayload(network),
}).then((benchPayload) => {
  (window as WindowWithBrowserBench).__NEATAPTIC_BENCH__ = benchPayload;
});

console.log('[NEATAPTIC_BROWSER_BENCH] ready');

async function runAsyncBuildProbeComparisons() {
  const asyncBuildComparisonRecords = [];

  for (const probeScenario of ASYNC_BUILD_PROBE_SCENARIOS) {
    asyncBuildComparisonRecords.push(await measureAsyncBuildScenario(probeScenario));
  }

  return asyncBuildComparisonRecords;
}

async function measureAsyncBuildScenario(probeScenario: ProbeScenario) {
  const asyncProbeNetwork = buildAsyncProbeNetwork(
    probeScenario.connectionBudget,
  );
  const originalChunkTargetMs = config.browserSlabChunkTargetMs;
  const originalPromiseThen = Promise.prototype.then;
  const asyncBuildsBefore = asyncProbeNetwork._slabAsyncBuilds ?? 0;
  let microtaskYieldCount = 0;
  let macrotaskHeartbeatCount = 0;
  let animationFrameCount = 0;
  let maxAnimationFrameGapMs = 0;
  let maxMacrotaskGapMs = 0;
  let lastAnimationFrameAt = 0;
  let lastHeartbeatAt = 0;
  let rebuildCompletedAt: number | null = null;

  const patchedThen: typeof Promise.prototype.then = function patchedThen(
    this: Promise<unknown>,
    onfulfilled,
    onrejected,
  ) {
    microtaskYieldCount += 1;
    return originalPromiseThen.call(this, onfulfilled, onrejected);
  };

  try {
    Promise.prototype.then = patchedThen;
    config.browserSlabChunkTargetMs = probeScenario.chunkTargetMs;
    lastAnimationFrameAt = performance.now();
    lastHeartbeatAt = performance.now();

    const animationFramePromise =
      typeof requestAnimationFrame !== 'function'
        ? Promise.resolve()
        : new Promise<void>((resolve) => {
            const pumpAnimationFrame = (timestamp: number) => {
              const effectiveTimestamp =
                rebuildCompletedAt === null
                  ? timestamp
                  : Math.min(timestamp, rebuildCompletedAt);
              maxAnimationFrameGapMs = Math.max(
                maxAnimationFrameGapMs,
                effectiveTimestamp - lastAnimationFrameAt,
              );
              lastAnimationFrameAt = effectiveTimestamp;

              if (rebuildCompletedAt !== null) {
                resolve();
                return;
              }

              animationFrameCount += 1;
              requestAnimationFrame(pumpAnimationFrame);
            };

            requestAnimationFrame(pumpAnimationFrame);
          });

    const heartbeatPromise = new Promise<void>((resolve) => {
      const pumpHeartbeat = () => {
        const now = performance.now();
        const effectiveNow =
          rebuildCompletedAt === null ? now : Math.min(now, rebuildCompletedAt);
        maxMacrotaskGapMs = Math.max(
          maxMacrotaskGapMs,
          effectiveNow - lastHeartbeatAt,
        );
        lastHeartbeatAt = effectiveNow;

        if (rebuildCompletedAt !== null) {
          resolve();
          return;
        }

        macrotaskHeartbeatCount += 1;
        setTimeout(pumpHeartbeat, 0);
      };

      setTimeout(pumpHeartbeat, 0);
    });

    const startedAt = performance.now();
    await rebuildConnectionSlabAsync.call(
      asyncProbeNetwork,
      probeScenario.requestedChunkSize,
    );
    const elapsedMs = performance.now() - startedAt;
    rebuildCompletedAt = performance.now();
    await Promise.all([heartbeatPromise, animationFramePromise]);
    const averageMacrotaskGapMs =
      macrotaskHeartbeatCount > 0 ? elapsedMs / macrotaskHeartbeatCount : elapsedMs;

    return {
      scenario: probeScenario.scenario,
      connectionBudget: probeScenario.connectionBudget,
      requestedChunkSize: probeScenario.requestedChunkSize,
      chunkTargetMs: probeScenario.chunkTargetMs,
      frameBudgetMs: BROWSER_FRAME_BUDGET_MS,
      elapsedMs: Number(elapsedMs.toFixed(3)),
      microtaskYieldCount,
      asyncBuildsDelta: (asyncProbeNetwork._slabAsyncBuilds ?? 0) - asyncBuildsBefore,
      macrotaskHeartbeatCount,
      averageMacrotaskGapMs: Number(averageMacrotaskGapMs.toFixed(3)),
      animationFrameCount,
      maxAnimationFrameGapMs: Number(maxAnimationFrameGapMs.toFixed(3)),
      maxMacrotaskGapMs: Number(maxMacrotaskGapMs.toFixed(3)),
    };
  } finally {
    rebuildCompletedAt ??= performance.now();
    Promise.prototype.then = originalPromiseThen;
    config.browserSlabChunkTargetMs = originalChunkTargetMs;
  }
}

function buildAsyncProbeNetwork(
  connectionBudget: number,
): BrowserAsyncProbeNetwork {
  const inputCount = 20;
  const outputCount = 5;
  const hiddenCount = Math.max(
    1,
    Math.ceil(connectionBudget / (inputCount + outputCount)),
  );
  const asyncProbeNetwork = new Network(inputCount, outputCount, {
    enforceAcyclic: true,
  }) as BrowserAsyncProbeNetwork;

  for (
    let hiddenIndex = 0;
    hiddenIndex < hiddenCount;
    hiddenIndex += 1
  ) {
    const hiddenNode = new Node('hidden');
    asyncProbeNetwork.nodes.push(hiddenNode);

    for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
      asyncProbeNetwork.connect(
        asyncProbeNetwork.nodes[inputIndex],
        hiddenNode,
        buildDeterministicWeight(inputIndex, hiddenIndex),
      );
    }

    for (
      let outputIndex = asyncProbeNetwork.nodes.length - outputCount;
      outputIndex < asyncProbeNetwork.nodes.length;
      outputIndex += 1
    ) {
      asyncProbeNetwork.connect(
        hiddenNode,
        asyncProbeNetwork.nodes[outputIndex],
        buildDeterministicWeight(hiddenIndex, outputIndex, 3),
      );
    }
  }

  asyncProbeNetwork._topoDirty = true;
  asyncProbeNetwork._nodeIndexDirty = true;

  return asyncProbeNetwork;
}

function buildDeterministicWeight(
  fromIndex: number,
  toIndex: number,
  offset = 0,
) {
  return (((fromIndex + toIndex + offset) % 9) - 4) / 20;
}
