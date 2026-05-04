import {
  runSequenceResetExample,
  type SequenceResetExampleResult,
} from './index';
import mermaid from 'mermaid';

type BrowserHostContainer = string | HTMLElement;
type SequenceResetStart = (container?: BrowserHostContainer) => Promise<void>;

let hasInitializedMermaid = false;
let mermaidRenderCount = 0;

declare global {
  interface Window {
    sequenceReset?: {
      start: SequenceResetStart;
    };
    sequenceResetStart?: SequenceResetStart;
  }
}

/**
 * Starts the browser-hosted Sequence Reset starter demo.
 *
 * @param container - Host element or element id.
 * @returns Promise resolved after the walkthrough renders into the container.
 */
export async function start(
  container: BrowserHostContainer = 'sequence-reset-output',
): Promise<void> {
  const hostElement = resolveHostElement(container);

  hostElement.innerHTML = buildLoadingMarkup();
  await waitForAnimationFrame();

  try {
    const exampleResult = runSequenceResetExample();
    hostElement.innerHTML = buildSequenceResetMarkup(exampleResult);
    await renderDiagramIntoHost(
      hostElement,
      '[data-sequence-reset-diagram]',
      buildSequenceResetDiagram(),
    );
  } catch (error) {
    hostElement.innerHTML = buildErrorMarkup(
      error instanceof Error ? error.message : String(error),
    );
  }

  const rerunButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-sequence-reset-rerun]',
  );
  rerunButton?.addEventListener('click', () => {
    void start(hostElement);
  });
}

function buildSequenceResetMarkup(result: SequenceResetExampleResult): string {
  const resetMatches = result.afterResetOutputs.every(
    (v, i) => v === result.stateAccumulationOutputs[i],
  );
  const carryoverDiffers = result.carryoverOutputs.some(
    (v, i) => v !== result.stateAccumulationOutputs[i],
  );

  const tableRowsMarkup = result.inputSequence
    .map((inputValue, stepIndex) => {
      const run1 = result.stateAccumulationOutputs[stepIndex].toFixed(5);
      const run2 = result.afterResetOutputs[stepIndex].toFixed(5);
      const run3 = result.carryoverOutputs[stepIndex].toFixed(5);
      const run2MatchClass =
        run2 === run1
          ? 'starter-demo-row starter-demo-row--correct'
          : 'starter-demo-row starter-demo-row--incorrect';
      const run3DiffClass =
        run3 !== run1 ? 'starter-demo-row starter-demo-row--correct' : '';
      const run3Tooltip =
        run3 !== run1
          ? `State carryover changes the output at step ${stepIndex + 1}.`
          : `State carryover does not change the rounded output at step ${stepIndex + 1}; later-step reconvergence is expected in this fixed setup.`;
      return `<tr class="${run2MatchClass}"><td title="Sequence step ${stepIndex + 1} in the repeated five-step input series.">${stepIndex + 1}</td><td title="Input value presented at step ${stepIndex + 1}.">${inputValue.toFixed(3)}</td><td title="Baseline output from fresh state at step ${stepIndex + 1}.">${run1}</td><td title="Output after calling clear before replaying the same sequence.">${run2}</td><td class="${run3DiffClass}" title="${run3Tooltip}">${run3}</td></tr>`;
    })
    .join('');

  return `<section class="starter-demo-panel">
<div class="starter-demo-grid">
  <article class="starter-demo-card"><span class="starter-demo-label">Topology</span><strong>${result.architecture.topologyType}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Architecture</span><strong>${result.architecture.inputCount} → ${result.architecture.lstmHiddenSize} → ${result.architecture.outputCount} (LSTM)</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">clear() match</span><strong>${resetMatches ? 'yes' : 'no'}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Carryover effect</span><strong>${carryoverDiffers ? 'yes' : 'no'}</strong></article>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Three-Run Comparison</h2>
      <p>Run 1 uses fresh state. Run 2 follows <code>clear()</code> — outputs match Run 1 exactly. Run 3 skips <code>clear()</code> — accumulated state from Run 2 changes the outputs.</p>
    </div>
    <button type="button" class="starter-demo-button" data-sequence-reset-rerun>Run Again</button>
  </div>
  <table class="starter-demo-table">
    <thead>
      <tr>
        <th scope="col" title="Position in the repeated five-step sequence.">Step</th>
        <th scope="col" title="Input value delivered to the LSTM at that step.">Input</th>
        <th scope="col" title="Output from the first pass that starts with zero internal state.">Run 1 fresh</th>
        <th scope="col" title="Output from the replay after clear resets recurrent state.">Run 2 after clear</th>
        <th scope="col" title="Output from the replay that keeps the previous state instead of resetting it.">Run 3 no clear</th>
      </tr>
    </thead>
    <tbody>${tableRowsMarkup}</tbody>
  </table>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>How To Interpret The Three Runs</h2>
      <p>Sequence Reset shows that recurrent networks do not only react to the current input; they also react to whatever state they carried forward from earlier steps.</p>
    </div>
  </div>
  <ul>
    <li><strong>Run 1 (fresh)</strong> is the baseline trajectory from zero internal state.</li>
    <li><strong>Run 2 (after clear)</strong> should match Run 1 at every step, proving that <code>clear()</code> resets memory.</li>
    <li><strong>Run 3 (no clear)</strong> can differ at early steps because memory from Run 2 leaks into the next sequence.</li>
    <li>If Run 2 does not match Run 1, reset behavior is broken for this setup.</li>
    <li>If some later Run 3 steps match Run 1 again, that means the trajectories reconverged, not that reset failed.</li>
    <li>Important idea: recurrent networks are history-sensitive, so resetting state is part of correct evaluation workflow.</li>
  </ul>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Network Diagram</h2>
      <p>This diagram highlights the recurrent structure: one input fan-in, four memory-carrying units, and one output readout.</p>
    </div>
  </div>
  <div class="mermaid-diagram" data-processed="true" data-sequence-reset-diagram></div>
</div>
</section>`;
}

function buildSequenceResetDiagram(): string {
  return `flowchart LR
  input["Input step"] --> memory1["LSTM memory 1"]
  input --> memory2["LSTM memory 2"]
  input --> memory3["LSTM memory 3"]
  input --> memory4["LSTM memory 4"]
  memory1 --> output["Output"]
  memory2 --> output
  memory3 --> output
  memory4 --> output
  memory1 --> memory1
  memory2 --> memory2
  memory3 --> memory3
  memory4 --> memory4`;
}

function buildLoadingMarkup(): string {
  return `<section class="starter-demo-surface"><p class="starter-demo-loading">Building the LSTM and running three sequence passes in the browser...</p></section>`;
}

function buildErrorMarkup(errorMessage: string): string {
  return `<section class="starter-demo-surface"><h2>Run failed</h2><p class="starter-demo-error">${errorMessage}</p><button type="button" class="starter-demo-button" data-sequence-reset-rerun>Try Again</button></section>`;
}

function resolveHostElement(container: BrowserHostContainer): HTMLElement {
  if (typeof container === 'string') {
    const resolvedElement = document.getElementById(container);

    if (!resolvedElement) {
      throw new Error(
        `Sequence Reset browser demo could not find #${container}.`,
      );
    }

    return resolvedElement;
  }

  return container;
}

function waitForAnimationFrame(): Promise<void> {
  return new Promise((resolve) => {
    requestAnimationFrame(() => {
      resolve();
    });
  });
}

async function renderDiagramIntoHost(
  hostElement: HTMLElement,
  selector: string,
  diagramDefinition: string,
): Promise<void> {
  const diagramHost = hostElement.querySelector<HTMLElement>(selector);

  if (!diagramHost) {
    return;
  }

  if (!hasInitializedMermaid) {
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'loose',
      theme: 'dark',
    });
    hasInitializedMermaid = true;
  }

  const { svg } = await mermaid.render(
    `sequence-reset-diagram-${(mermaidRenderCount += 1)}`,
    diagramDefinition,
  );

  diagramHost.innerHTML = svg;
}

if (typeof window !== 'undefined') {
  window.sequenceResetStart = start;
  window.sequenceReset = { start };
}
