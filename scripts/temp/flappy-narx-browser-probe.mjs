import puppeteer from 'puppeteer';

const PROBE_URL = 'http://127.0.0.1:8080/examples/flappy_bird/index.html';
const MAX_WAIT_MS = resolvePositiveIntegerEnvValue(
  process.env.FLAPPY_MAX_WAIT_MS,
  300_000,
);
const POLL_INTERVAL_MS = resolvePositiveIntegerEnvValue(
  process.env.FLAPPY_POLL_INTERVAL_MS,
  2_000,
);
const TARGET_PIPE_COUNT = resolvePositiveIntegerEnvValue(
  process.env.FLAPPY_TARGET_PIPE_COUNT,
  10,
);
const TARGET_ARCHITECTURE_LABEL = process.env.FLAPPY_PROFILE_LABEL ?? 'NARX';
const TARGET_ARCHITECTURE_PROFILE_ID =
  process.env.FLAPPY_PROFILE_ID ?? 'narx';

const browser = await puppeteer.launch({
  headless: 'new',
  defaultViewport: {
    width: 1440,
    height: 960,
  },
});

try {
  const page = await browser.newPage();
  const recurrentDebugEvents = [];

  page.on('console', (message) => {
    const text = message.text();
    const recurrentDebugPrefix = '[flappy-recurrent-debug] ';

    if (!text.startsWith(recurrentDebugPrefix)) {
      return;
    }

    try {
      recurrentDebugEvents.push(
        JSON.parse(text.slice(recurrentDebugPrefix.length)),
      );
    } catch {
      recurrentDebugEvents.push({ parseError: text });
    }
  });

  await page.goto(PROBE_URL, { waitUntil: 'networkidle2' });
  await page.waitForFunction(
    (targetArchitectureLabel) =>
      [...document.querySelectorAll('button')].some((buttonElement) =>
        buttonElement.textContent?.includes(targetArchitectureLabel),
      ),
    { timeout: 30_000 },
    TARGET_ARCHITECTURE_LABEL,
  );

  const architectureLabels = await page.$$eval('button', (buttonElements) =>
    buttonElements.map((buttonElement) => buttonElement.textContent?.trim() ?? ''),
  );
  const targetButtonIndex = architectureLabels.findIndex((labelText) =>
    labelText.includes(TARGET_ARCHITECTURE_LABEL),
  );

  if (targetButtonIndex === -1) {
    throw new Error(
      `Could not find the ${TARGET_ARCHITECTURE_LABEL} architecture selector button.`,
    );
  }

  const architectureButtons = await page.$$('button');
  await architectureButtons[targetButtonIndex].click();

  const deadlineMs = Date.now() + MAX_WAIT_MS;
  let finalStats = undefined;

  while (Date.now() < deadlineMs) {
    finalStats = await readHudStats(page);
    const bestMaxPipesValue = Number.parseInt(
      finalStats.bestRun.maxPipes ?? '0',
      10,
    );
    const currentMaxPipesValue = Number.parseInt(
      finalStats.currentRun.maxPipes ?? '0',
      10,
    );
    const currentArchitecture =
      finalStats.currentRun.architecture ?? finalStats.bestRun.architecture;
    const lastObservedGeneration = resolveLastObservedGeneration(
      recurrentDebugEvents,
    );

    if (
      currentArchitecture?.includes(TARGET_ARCHITECTURE_LABEL) &&
      Math.max(bestMaxPipesValue, currentMaxPipesValue) >= TARGET_PIPE_COUNT
    ) {
      console.log(
        JSON.stringify(
          {
            success: true,
            architectureLabels,
            currentArchitecture,
            bestMaxPipesValue,
            currentMaxPipesValue,
            targetPipeCount: TARGET_PIPE_COUNT,
            targetArchitectureLabel: TARGET_ARCHITECTURE_LABEL,
            lastObservedGeneration,
            finalStats,
          },
          null,
          2,
        ),
      );
      process.exit(0);
    }

    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }

  console.log(
    JSON.stringify(
      {
        success: false,
          reason:
            'Timed out while waiting for the target architecture to reach the target pipe count.',
        architectureLabels,
          targetPipeCount: TARGET_PIPE_COUNT,
          targetArchitectureLabel: TARGET_ARCHITECTURE_LABEL,
        finalStats,
          lastObservedGeneration: resolveLastObservedGeneration(
            recurrentDebugEvents,
          ),
      },
      null,
      2,
    ),
  );
  process.exit(1);
} finally {
  await browser.close();
}

async function readHudStats(page) {
  return page.evaluate(() => {
    const statsBySection = {
      currentRun: {},
      bestRun: {},
      status: undefined,
      birds: undefined,
    };
    let activeSection = '';

    for (const rowElement of document.querySelectorAll('table tr')) {
      const cellElements = [...rowElement.querySelectorAll('th, td')];
      if (cellElements.length === 1) {
        activeSection = cellElements[0]?.textContent?.trim() ?? '';
        continue;
      }

      if (cellElements.length !== 2) {
        continue;
      }

      const labelText = cellElements[0]?.textContent?.trim() ?? '';
      const valueText = cellElements[1]?.textContent?.trim() ?? '';

      if (activeSection.startsWith('Current run')) {
        if (labelText === 'Max pipes') {
          statsBySection.currentRun.maxPipes = valueText;
        }
        if (labelText === 'NN architecture') {
          statsBySection.currentRun.architecture = valueText;
        }
      }

      if (activeSection.startsWith('Best run')) {
        if (labelText === 'Max pipes') {
          statsBySection.bestRun.maxPipes = valueText;
        }
        if (labelText === 'NN architecture') {
          statsBySection.bestRun.architecture = valueText;
        }
      }

      if (labelText === 'Status') {
        statsBySection.status = valueText;
      }

      if (labelText === 'Birds') {
        statsBySection.birds = valueText;
      }
    }

    return statsBySection;
  });
}

function resolveLastObservedGeneration(recurrentDebugEvents) {
  return recurrentDebugEvents.reduce((bestGeneration, debugEvent) => {
    if (
      debugEvent?.architectureProfileId !== TARGET_ARCHITECTURE_PROFILE_ID
    ) {
      return bestGeneration;
    }

    return Math.max(bestGeneration, Number(debugEvent.generation ?? -1));
  }, -1);
}

function resolvePositiveIntegerEnvValue(rawValue, fallbackValue) {
  const parsedValue = Number.parseInt(rawValue ?? '', 10);

  return Number.isFinite(parsedValue) && parsedValue > 0
    ? parsedValue
    : fallbackValue;
}