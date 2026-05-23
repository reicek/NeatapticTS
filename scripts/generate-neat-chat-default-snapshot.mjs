import { spawnSync } from 'node:child_process';

import { build } from 'esbuild';

const SUPPORTED_NPM_CONFIG_ARGUMENTS = {
  contextwindowtokencount: 'contextWindowTokenCount',
  extrareinforcementpasses: 'extraReinforcementPasses',
  exportname: 'exportName',
  generationseed: 'generationSeed',
  maxcasesperphase: 'maxCasesPerPhase',
  maxsourcelines: 'maxSourceLines',
  outputfilename: 'outputFileName',
  progress: 'progress',
  topwordlimit: 'topWordLimit',
  validationlinecount: 'validationLineCount',
};

const bundledGeneratorPath =
  'dist-docs/scripts/generate-neat-chat-default-snapshot.js';
const forwardedGenerationArguments = resolveForwardedGenerationArguments();

await build({
  entryPoints: ['examples/neatChat/generate-default-pretrained-session-snapshot.ts'],
  bundle: true,
  platform: 'node',
  format: 'esm',
  outfile: bundledGeneratorPath,
});

const generationResult = spawnSync(
  process.execPath,
  [bundledGeneratorPath, ...forwardedGenerationArguments],
  {
    stdio: 'inherit',
  },
);

if (generationResult.error) {
  throw generationResult.error;
}

process.exit(generationResult.status ?? 1);

function resolveForwardedGenerationArguments() {
  const cliArguments = process.argv.slice(2);
  const resolvedArguments = [...cliArguments];
  const forwardedOptionNames = new Set(
    cliArguments
      .map(resolveArgumentOptionName)
      .filter((optionName) => optionName !== null),
  );

  for (const [npmConfigName, optionName] of Object.entries(
    SUPPORTED_NPM_CONFIG_ARGUMENTS,
  ).toSorted(([leftName], [rightName]) => leftName.localeCompare(rightName))) {
    if (forwardedOptionNames.has(optionName)) {
      continue;
    }

    const npmConfigValue = process.env[`npm_config_${npmConfigName}`];

    if (npmConfigValue === undefined) {
      continue;
    }

    resolvedArguments.push(`--${optionName}=${npmConfigValue}`);
  }

  return resolvedArguments;
}

function resolveArgumentOptionName(argument) {
  if (!argument.startsWith('--')) {
    return null;
  }

  const separatorIndex = argument.indexOf('=');

  if (separatorIndex < 0) {
    return null;
  }

  return argument.slice(2, separatorIndex);
}