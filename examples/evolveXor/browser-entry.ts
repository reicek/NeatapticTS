import { runEvolveXorExample, type EvolveXorExampleResult } from './index';
import mermaid from 'mermaid';

type BrowserHostContainer = string | HTMLElement;
type EvolveXorStart = (container?: BrowserHostContainer) => Promise<void>;

const activeRunsByHost = new WeakMap<HTMLElement, Promise<void>>();
let hasInitializedMermaid = false;
let mermaidRenderCount = 0;

declare global {
  interface Window {
    evolveXor?: {
      start: EvolveXorStart;
    };
    evolveXorStart?: EvolveXorStart;
  }
}

/**
 * Starts the browser-hosted Evolve XOR starter demo.
 *
 * @param container - Host element or element id.
 * @returns Promise resolved after the seeded browser run renders its results.
 */
export async function start(
  container: BrowserHostContainer = 'evolve-xor-output',
): Promise<void> {
  const hostElement = resolveHostElement(container);
  const activeRun = activeRunsByHost.get(hostElement);

  if (activeRun) {
    return activeRun;
  }

  const runPromise = renderBrowserRun(hostElement).finally(() => {
    activeRunsByHost.delete(hostElement);
  });

  activeRunsByHost.set(hostElement, runPromise);
  return runPromise;
}

async function renderBrowserRun(hostElement: HTMLElement): Promise<void> {
  hostElement.innerHTML = buildLoadingMarkup();
  await waitForAnimationFrame();

  try {
    const exampleResult = await runEvolveXorExample();
    hostElement.innerHTML = buildEvolveXorMarkup(exampleResult);
    await renderDiagramIntoHost(
      hostElement,
      '[data-evolve-xor-diagram]',
      buildEvolveXorDiagram(),
    );
  } catch (error) {
    hostElement.innerHTML = buildErrorMarkup(
      error instanceof Error ? error.message : String(error),
    );
  }

  const rerunButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-evolve-xor-rerun]',
  );
  rerunButton?.addEventListener('click', () => {
    void start(hostElement);
  });
}

function buildEvolveXorMarkup(exampleResult: EvolveXorExampleResult): string {
  const predictionRowsMarkup = exampleResult.predictions
    .map((predictionSummary) => {
      const resolvedDecision = predictionSummary.outputValue >= 0.5 ? 1 : 0;
      const rowClassName =
        resolvedDecision === predictionSummary.expectedOutput
          ? 'starter-demo-row starter-demo-row--correct'
          : 'starter-demo-row starter-demo-row--incorrect';

      return `<tr class="${rowClassName}"><td title="First XOR input for this row.">${predictionSummary.inputValues[0]}</td><td title="Second XOR input for this row.">${predictionSummary.inputValues[1]}</td><td title="Correct XOR label for the input pair ${predictionSummary.inputValues.join(', ')}.">${predictionSummary.expectedOutput}</td><td title="Raw network response ${predictionSummary.outputValue.toFixed(5)} before thresholding.">${predictionSummary.outputValue.toFixed(5)}</td><td title="Decision after applying the 0.5 cutoff to the raw output.">${resolvedDecision}</td></tr>`;
    })
    .join('');

  return `<section class="starter-demo-panel">
<div class="starter-demo-grid">
  <article class="starter-demo-card"><span class="starter-demo-label">Solved</span><strong>${exampleResult.solved ? 'yes' : 'no'}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Final generation</span><strong>${exampleResult.finalGeneration}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Best score</span><strong>${exampleResult.bestScore.toFixed(5)}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Prediction rows</span><strong>${exampleResult.predictions.length}</strong></article>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Prediction Table</h2>
      <p>The browser run evolves a small feed-forward controller, then renders each XOR row so the final champion stays inspectable.</p>
    </div>
    <button type="button" class="starter-demo-button" data-evolve-xor-rerun>Run Again</button>
  </div>
  <table class="starter-demo-table">
    <thead>
      <tr>
        <th scope="col" title="First input value in the XOR truth table row.">Input A</th>
        <th scope="col" title="Second input value in the XOR truth table row.">Input B</th>
        <th scope="col" title="Correct XOR label: 1 when inputs differ, 0 when they match.">Expected</th>
        <th scope="col" title="Raw network prediction before converting it into a 0 or 1 decision.">Output</th>
        <th scope="col" title="Predicted class after applying the 0.5 decision threshold.">Decision</th>
      </tr>
    </thead>
    <tbody>${predictionRowsMarkup}</tbody>
  </table>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>How To Read XOR Progress</h2>
      <p>Evolve XOR shows how a feed-forward NEAT run moves from a population of guesses toward a network that separates matching and non-matching input pairs.</p>
    </div>
  </div>
  <ul>
    <li><strong>Expected</strong> is the true XOR label for each input pair.</li>
    <li><strong>Output</strong> is the network's raw response between 0 and 1.</li>
    <li><strong>Decision</strong> applies a 0.5 cutoff: output >= 0.5 becomes 1, otherwise 0.</li>
    <li>Green-tinted rows mean the decision matches the expected label; magenta-tinted rows mean it misses.</li>
    <li><strong>Best score</strong> is higher when absolute prediction error is lower. A solved run is typically near 4.0 across all four rows.</li>
    <li><strong>Final generation</strong> tells you how long evolution needed before stopping under this run's solved-or-budget rule.</li>
  </ul>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Network Diagram</h2>
      <p>This diagram shows a compact feed-forward shape that combines both inputs before producing the final XOR decision.</p>
    </div>
  </div>
  <div class="mermaid-diagram" data-processed="true" data-evolve-xor-diagram></div>
</div>
</section>`;
}

function buildEvolveXorDiagram(): string {
  return `flowchart LR
  inputA["Input A"] --> hidden1["Hidden unit 1"]
  inputA --> hidden2["Hidden unit 2"]
  inputB["Input B"] --> hidden1
  inputB --> hidden2
  hidden1 --> output["XOR output"]
  hidden2 --> output`;
}

function buildErrorMarkup(errorMessage: string): string {
  return `<section class="starter-demo-surface"><h2>Run failed</h2><p class="starter-demo-error">${errorMessage}</p><button type="button" class="starter-demo-button" data-evolve-xor-rerun>Try Again</button></section>`;
}

function buildLoadingMarkup(): string {
  return `<section class="starter-demo-surface"><p class="starter-demo-loading">Running the seeded XOR evolution loop in the browser. This can take a moment because the example really evaluates and evolves the population before it prints the table.</p></section>`;
}

function resolveHostElement(container: BrowserHostContainer): HTMLElement {
  if (typeof container === 'string') {
    const resolvedElement = document.getElementById(container);

    if (!resolvedElement) {
      throw new Error(`Evolve XOR browser demo could not find #${container}.`);
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
    `evolve-xor-diagram-${(mermaidRenderCount += 1)}`,
    diagramDefinition,
  );

  diagramHost.innerHTML = svg;
}

if (typeof window !== 'undefined') {
  window.evolveXorStart = start;
  window.evolveXor = { start };
}
