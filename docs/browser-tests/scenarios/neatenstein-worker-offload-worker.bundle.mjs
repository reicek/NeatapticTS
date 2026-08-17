// examples/neatenstein/browser-entry/constants.ts
var NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION = "neatenstein-frame-v1";
var NEATENSTEIN_GPU_COLUMN_COUNT = 320;
var NEATENSTEIN_WORKER_COLUMN_COUNT = 240;
var NEATENSTEIN_CPU_COLUMN_COUNT = 160;

// examples/neatenstein/browser-entry/renderer/frame.ts
var nextRequestId = 0;
function buildNeatensteinRenderFrame(state, columnCount) {
  const id = nextRequestId;
  nextRequestId += 1;
  return {
    format: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    requestId: id,
    canvasWidth: state.canvasWidth,
    canvasHeight: state.canvasHeight,
    columnCount,
    simTick: state.simTick,
    wallDistances: new Float32Array(columnCount),
    wallSides: new Uint8Array(columnCount),
    zBuffer: new Float32Array(columnCount),
    enemyScreenX: new Float32Array(columnCount),
    enemyScale: new Float32Array(columnCount),
    projectileScreenX: new Float32Array(columnCount)
  };
}

// examples/neatenstein/browser-entry/worker/display.worker.ts
var currentTier = null;
var workerCanvas = null;
var latestState = null;
var rafScheduled = false;
function resolveColumnCount(tier) {
  switch (tier) {
    case "gpu":
      return NEATENSTEIN_GPU_COLUMN_COUNT;
    case "worker":
      return NEATENSTEIN_WORKER_COLUMN_COUNT;
    case "cpu":
    default:
      return NEATENSTEIN_CPU_COLUMN_COUNT;
  }
}
function buildAndPostFrame() {
  if (!latestState || !currentTier) {
    return;
  }
  const columnCount = resolveColumnCount(currentTier);
  const frame = buildNeatensteinRenderFrame(latestState, columnCount);
  if (currentTier === "worker" && workerCanvas) {
    const ctx = workerCanvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, latestState.canvasWidth, latestState.canvasHeight);
    }
  }
  self.postMessage({ type: "frame", frame });
}
function rafTick() {
  if (currentTier === "cpu" || currentTier === "gpu") {
    buildAndPostFrame();
    self.requestAnimationFrame(rafTick);
  }
}
function startCpuGpuLoop() {
  if (!rafScheduled) {
    rafScheduled = true;
    self.requestAnimationFrame(rafTick);
  }
}
self.onmessage = (event) => {
  const data = event.data;
  if (!data || typeof data !== "object") {
    return;
  }
  if (data.type === "init") {
    const tier = data.tier;
    currentTier = tier;
    if (data.canvas) {
      workerCanvas = data.canvas;
    }
    self.postMessage({
      type: "initialized",
      tier,
      version: data.version ?? NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION
    });
    if (tier === "cpu" || tier === "gpu") {
      startCpuGpuLoop();
    }
  } else if (data.type === "simState") {
    latestState = data.state;
    buildAndPostFrame();
  }
};
//# sourceMappingURL=neatenstein-worker-offload-worker.bundle.mjs.map
