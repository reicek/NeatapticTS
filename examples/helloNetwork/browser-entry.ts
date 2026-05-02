import {
  runHelloNetworkExample,
  type HelloNetworkExampleResult,
} from './index';
import mermaid from 'mermaid';

type BrowserHostContainer = string | HTMLElement;
type HelloNetworkStart = (container?: BrowserHostContainer) => Promise<void>;

let hasInitializedMermaid = false;
let mermaidRenderCount = 0;

declare global {
  interface Window {
    helloNetwork?: {
      start: HelloNetworkStart;
    };
    helloNetworkStart?: HelloNetworkStart;
  }
}

/**
 * Starts the browser-hosted Hello Network starter demo.
 *
 * @param container - Host element or element id.
 * @returns Promise resolved after the deterministic walkthrough renders.
 */
export async function start(
  container: BrowserHostContainer = 'hello-network-output',
): Promise<void> {
  const hostElement = resolveHostElement(container);

  hostElement.innerHTML = buildLoadingMarkup(
    'Building one tiny feed-forward network in the browser...',
  );
  await waitForAnimationFrame();

  const exampleResult = runHelloNetworkExample();
  hostElement.innerHTML = buildHelloNetworkMarkup(exampleResult);
  await renderDiagramIntoHost(
    hostElement,
    '[data-hello-network-diagram]',
    buildHelloNetworkDiagram(),
  );

  const rerunButton = hostElement.querySelector<HTMLButtonElement>(
    '[data-hello-network-rerun]',
  );
  rerunButton?.addEventListener('click', () => {
    void start(hostElement);
  });
}

function buildHelloNetworkMarkup(
  exampleResult: HelloNetworkExampleResult,
): string {
  return `<section class="starter-demo-panel">
<div class="starter-demo-grid">
  <article class="starter-demo-card"><span class="starter-demo-label">Topology</span><strong>${exampleResult.architecture.topologyIntent}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Architecture</span><strong>${exampleResult.architecture.inputCount} -> [${exampleResult.architecture.hiddenLayerSizes.join(', ')}] -> ${exampleResult.architecture.outputCount}</strong></article>
  <article class="starter-demo-card"><span class="starter-demo-label">Output count</span><strong>${exampleResult.outputValues.length}</strong></article>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Inference Table</h2>
      <p>One deterministic forward pass rendered directly in the browser.</p>
    </div>
    <button type="button" class="starter-demo-button" data-hello-network-rerun>Run Again</button>
  </div>
  <table class="starter-demo-table">
    <thead>
      <tr>
        <th scope="col" title="First input value for the single forward pass.">Input A</th>
        <th scope="col" title="Second input value for the single forward pass.">Input B</th>
        <th scope="col" title="Network response after signals pass through the hidden layer.">Output</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td title="Input A is ${formatNumber(exampleResult.inputValues[0])} for this deterministic run.">${formatNumber(exampleResult.inputValues[0])}</td>
        <td title="Input B is ${formatNumber(exampleResult.inputValues[1])} for this deterministic run.">${formatNumber(exampleResult.inputValues[1])}</td>
        <td title="The network returns ${formatNumber(exampleResult.outputValues[0])} for this exact input pair.">${formatNumber(exampleResult.outputValues[0])}</td>
      </tr>
    </tbody>
  </table>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>How To Read This Result</h2>
      <p>Hello Network shows one complete inference pass: two input values enter a fixed feed-forward network, and one output value comes back.</p>
    </div>
  </div>
  <ul>
    <li><strong>Input A / Input B</strong> are the two numbers fed into the network for this single pass.</li>
    <li><strong>Output</strong> is the network response after signals move through the hidden layer once.</li>
    <li>In this starter setup, output values stay in a probability-like range between 0 and 1.</li>
    <li>You should expect this run to stay near <strong>0.58307</strong> every time because the example pins fixed weights and biases.</li>
    <li>If output changes between reruns, it usually means the example code changed, not random training drift.</li>
  </ul>
</div>
<div class="starter-demo-surface">
  <div class="starter-demo-header">
    <div>
      <h2>Network Diagram</h2>
      <p>This diagram shows the exact layered structure used in the walkthrough: two inputs, three hidden nodes, and one output node.</p>
    </div>
  </div>
  <div class="mermaid-diagram" data-processed="true" data-hello-network-diagram></div>
</div>
</section>`;
}

function buildHelloNetworkDiagram(): string {
  return `flowchart LR
  inputA["Input A"] --> hidden1["Hidden 1"]
  inputA --> hidden2["Hidden 2"]
  inputA --> hidden3["Hidden 3"]
  inputB["Input B"] --> hidden1
  inputB --> hidden2
  inputB --> hidden3
  hidden1 --> output["Output"]
  hidden2 --> output
  hidden3 --> output`;
}

function buildLoadingMarkup(statusMessage: string): string {
  return `<section class="starter-demo-surface"><p class="starter-demo-loading">${statusMessage}</p></section>`;
}

function formatNumber(value: number): string {
  return value.toFixed(5);
}

function resolveHostElement(container: BrowserHostContainer): HTMLElement {
  if (typeof container === 'string') {
    const resolvedElement = document.getElementById(container);

    if (!resolvedElement) {
      throw new Error(
        `Hello Network browser demo could not find #${container}.`,
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
    `hello-network-diagram-${(mermaidRenderCount += 1)}`,
    diagramDefinition,
  );

  diagramHost.innerHTML = svg;
}

if (typeof window !== 'undefined') {
  window.helloNetworkStart = start;
  window.helloNetwork = { start };
}
