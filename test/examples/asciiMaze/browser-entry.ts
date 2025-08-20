import { BrowserTerminalUtility } from './browserTerminalUtility';
import { createBrowserLogger } from './browserLogger';
import { DashboardManager } from './dashboardManager';
import { EvolutionEngine } from './evolutionEngine';
import * as mazes from './mazes';

// Small helper to create UI controls (start/stop) — minimal, unobtrusive
// No UI controls in embedded mode; the demo will run automatically.

// Exported start function for the bundle.
export async function start(containerId = 'ascii-maze-output') {
  const host = document.getElementById(containerId);
  const archiveEl = host
    ? (host.querySelector('#ascii-maze-archive') as HTMLElement)
    : null;
  const liveEl = host
    ? (host.querySelector('#ascii-maze-live') as HTMLElement)
    : null;

  // clearer will clear only the live area; archive remains
  const clearFn = BrowserTerminalUtility.createTerminalClearer(
    liveEl ?? undefined
  );
  const liveLogFn = createBrowserLogger(liveEl ?? undefined);
  const archiveLogFn = createBrowserLogger(archiveEl ?? undefined);

  // DashboardManager will use live logger for ongoing redraws and archive logger to append solved blocks
  const dashboard = new DashboardManager(
    clearFn,
    liveLogFn as any,
    archiveLogFn as any
  );

  // keep a global reference so the control can call it
  (window as any).asciiMazeStart = async () => {
    // Run mazes in the same curriculum order as the e2e test and mirror its
    // evolution settings where practical. We intentionally disable the
    // post-evolution backprop refinement (lamarckian iterations = 0) for
    // the browser demo as requested.
    const order = [
      'tiny',
      'spiralSmall',
      'spiral',
      'small',
      'medium',
      'medium2',
      'large',
      'minotaur',
    ];

    // Carry the winning network forward between phases (curriculum transfer)
    let lastBestNetwork: any = undefined;

  for (const key of order) {
      const maze = (mazes as any)[key] as string[];
      if (!Array.isArray(maze)) continue; // skip missing exports

      // Per-phase settings copied from the e2e test with lamarckianIterations=0
      let agentMaxSteps = 1000;
      let maxGenerations = 500;
      switch (key) {
        case 'tiny':
          agentMaxSteps = 100;
          maxGenerations = 200;
          break;
        case 'spiralSmall':
          agentMaxSteps = 100;
          maxGenerations = 200;
          break;
        case 'spiral':
          agentMaxSteps = 150;
          maxGenerations = 300;
          break;
        case 'small':
          agentMaxSteps = 50;
          maxGenerations = 300;
          break;
        case 'medium':
          agentMaxSteps = 250;
          maxGenerations = 400;
          break;
        case 'medium2':
          agentMaxSteps = 300;
          maxGenerations = 400;
          break;
        case 'large':
          agentMaxSteps = 400;
          maxGenerations = 500;
          break;
        case 'minotaur':
          agentMaxSteps = 700;
          maxGenerations = 600;
          break;
      }

      try {
        const result = await EvolutionEngine.runMazeEvolution({
          mazeConfig: { maze },
          agentSimConfig: { maxSteps: agentMaxSteps },
          evolutionAlgorithmConfig: {
            allowRecurrent: true,
            popSize: 40,
            // Run indefinitely until solved; remove stagnation pressure for demo clarity
            maxStagnantGenerations: Number.POSITIVE_INFINITY,
            minProgressToPass: 99,
            maxGenerations: Number.POSITIVE_INFINITY,
            stopOnlyOnSolve: false,
            autoPauseOnSolve: false,
            // Disable Lamarckian/backprop refinement for browser runs per request
            lamarckianIterations: 0,
            lamarckianSampleSize: 0,
            // seed previous winner if available
            initialBestNetwork: lastBestNetwork,
          },
          reportingConfig: {
            dashboardManager: dashboard,
            logEvery: 1,
            label: `browser-${key}`,
          },
        });
        try { console.log('[asciiMaze] maze solved', key, (result as any)?.bestResult?.progress); } catch {}
        if (result && (result as any).bestNetwork)
          lastBestNetwork = (result as any).bestNetwork;
      } catch (e) {
        console.error('Error while running maze', key, e);
      }
    }
  };

  // auto-start for convenience
  (window as any).asciiMazeStart();

  // Setup a cooperative pause flag and UI wiring so users can pause/resume the entire curriculum.
  try {
    // Ensure the paused flag exists on window
    (window as any).asciiMazePaused = false;

    const playPauseBtn = document.getElementById(
      'ascii-maze-playpause'
    ) as HTMLButtonElement | null;
    const statusEl = document.getElementById('ascii-maze-status');
    const updateUI = () => {
      const paused = !!(window as any).asciiMazePaused;
      if (playPauseBtn) {
        playPauseBtn.textContent = paused ? 'Resume' : 'Pause';
        playPauseBtn.style.background = paused ? '#39632C' : '#2C3963';
        playPauseBtn.setAttribute('aria-pressed', String(!paused));
        playPauseBtn.disabled = false;
      }
    };

    if (playPauseBtn) {
      playPauseBtn.addEventListener('click', () => {
        (window as any).asciiMazePaused = !(window as any).asciiMazePaused;
        if (statusEl) {
          statusEl.textContent = (window as any).asciiMazePaused
            ? 'Paused – evolution halted'
            : 'Running continuously';
        }
        updateUI();
      });
      playPauseBtn.addEventListener('update-ui', updateUI as any);
    }
    // initialize UI state
    updateUI();
  } catch {
    // ignore DOM wiring errors in non-browser envs
  }
}

// If loaded directly in a script tag, automatically start
if (typeof window !== 'undefined' && (window as any).document) {
  // Delay to allow DOM insertion
  setTimeout(() => start(), 20);
}
