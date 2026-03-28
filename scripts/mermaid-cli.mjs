import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const EXPORT_COMMAND = 'export';
const VALIDATE_COMMAND = 'validate';
const SUPPORTED_COMMANDS = new Set([EXPORT_COMMAND, VALIDATE_COMMAND]);

await main();

async function main() {
  const [command = VALIDATE_COMMAND, ...rawArgs] = process.argv.slice(2);
  if (!SUPPORTED_COMMANDS.has(command)) {
    printUsageAndExit(`Unsupported Mermaid CLI command: ${command}`);
  }

  const parsedArguments = parseArguments(rawArgs);
  const inputPath = parsedArguments.named.input ?? parsedArguments.named.i;
  if (!inputPath) {
    printUsageAndExit('Missing required --input argument.');
  }

  const cliPath = await resolveMermaidCliPath();
  if (command === VALIDATE_COMMAND) {
    await runValidateCommand(cliPath, inputPath, parsedArguments);
    return;
  }

  const outputPath = parsedArguments.named.output ?? parsedArguments.named.o;
  if (!outputPath) {
    printUsageAndExit('Missing required --output argument for export.');
  }

  await runExportCommand(cliPath, inputPath, outputPath, parsedArguments);
}

function parseArguments(rawArgs) {
  const named = {};
  const passthrough = [];

  for (let argumentIndex = 0; argumentIndex < rawArgs.length; argumentIndex += 1) {
    const argument = rawArgs[argumentIndex];
    if (!argument.startsWith('--')) {
      passthrough.push(argument);
      continue;
    }

    const normalizedArgument = argument.slice(2);
    const [name, inlineValue] = normalizedArgument.split('=', 2);
    if (inlineValue !== undefined) {
      named[name] = inlineValue;
      continue;
    }

    const nextArgument = rawArgs[argumentIndex + 1];
    if (!nextArgument || nextArgument.startsWith('--')) {
      named[name] = 'true';
      continue;
    }

    named[name] = nextArgument;
    argumentIndex += 1;
  }

  return { named, passthrough };
}

async function resolveMermaidCliPath() {
  return path.resolve(
    process.cwd(),
    'node_modules',
    '@mermaid-js',
    'mermaid-cli',
    'src',
    'cli.js',
  );
}

async function runValidateCommand(cliPath, inputPath, parsedArguments) {
  const tempDirectoryPath = await mkdtemp(
    path.join(os.tmpdir(), 'neatapticts-mermaid-'),
  );
  const tempOutputPath = path.join(tempDirectoryPath, 'diagram.svg');
  const mermaidCliInvocation = await buildMermaidCliInvocation(
    parsedArguments,
    [
      '--input',
      inputPath,
      '--output',
      tempOutputPath,
      ...buildPassthroughArguments(parsedArguments, {
        excludedNames: new Set(['input', 'i', 'output', 'o']),
      }),
    ],
  );

  try {
    await mkdir(path.dirname(tempOutputPath), { recursive: true });
    await runMermaidCli(cliPath, mermaidCliInvocation.argumentsToPass);
    console.log(`[mermaid] Valid diagram: ${inputPath}`);
  } finally {
    await mermaidCliInvocation.cleanup();
    await rm(tempDirectoryPath, { recursive: true, force: true });
  }
}

async function runExportCommand(cliPath, inputPath, outputPath, parsedArguments) {
  await mkdir(path.dirname(path.resolve(outputPath)), { recursive: true });
  const mermaidCliInvocation = await buildMermaidCliInvocation(
    parsedArguments,
    [
      '--input',
      inputPath,
      '--output',
      outputPath,
      ...buildPassthroughArguments(parsedArguments, {
        excludedNames: new Set(['input', 'i', 'output', 'o']),
      }),
    ],
  );

  try {
    await runMermaidCli(cliPath, mermaidCliInvocation.argumentsToPass);
    console.log(`[mermaid] Exported diagram to ${outputPath}`);
  } finally {
    await mermaidCliInvocation.cleanup();
  }
}

function buildPassthroughArguments(parsedArguments, options) {
  const namedArguments = Object.entries(parsedArguments.named)
    .filter(([name]) => !options.excludedNames.has(name))
    .flatMap(([name, value]) => {
      if (value === 'true') {
        return [`--${name}`];
      }

      return [`--${name}`, value];
    });

  return [...namedArguments, ...parsedArguments.passthrough];
}

async function runMermaidCli(cliPath, argumentsToPass) {
  await new Promise((resolve, reject) => {
    const childProcess = spawn(process.execPath, [cliPath, ...argumentsToPass], {
      stdio: 'inherit',
    });

    childProcess.once('exit', (exitCode) => {
      if (exitCode === 0) {
        resolve(undefined);
        return;
      }

      reject(new Error(`Mermaid CLI exited with code ${exitCode ?? 'null'}.`));
    });
    childProcess.once('error', reject);
  });
}

async function buildMermaidCliInvocation(parsedArguments, baseArguments) {
  if (!shouldInjectCiLinuxNoSandbox(parsedArguments)) {
    return {
      argumentsToPass: baseArguments,
      cleanup: async () => {},
    };
  }

  const tempDirectoryPath = await mkdtemp(
    path.join(os.tmpdir(), 'neatapticts-mermaid-puppeteer-'),
  );
  const puppeteerConfigFilePath = path.join(
    tempDirectoryPath,
    'puppeteer-config.json',
  );

  await writeFile(
    puppeteerConfigFilePath,
    JSON.stringify({
      args: ['--no-sandbox', '--disable-setuid-sandbox'],
    }),
  );

  return {
    argumentsToPass: [
      '--puppeteerConfigFile',
      puppeteerConfigFilePath,
      ...baseArguments,
    ],
    cleanup: async () => {
      await rm(tempDirectoryPath, { recursive: true, force: true });
    },
  };
}

function shouldInjectCiLinuxNoSandbox(parsedArguments) {
  return (
    process.platform === 'linux' &&
    (process.env.CI === 'true' || process.env.GITHUB_ACTIONS === 'true') &&
    !hasExplicitPuppeteerConfig(parsedArguments)
  );
}

function hasExplicitPuppeteerConfig(parsedArguments) {
  return (
    parsedArguments.named.puppeteerConfigFile !== undefined ||
    parsedArguments.named.p !== undefined ||
    parsedArguments.passthrough.includes('--puppeteerConfigFile') ||
    parsedArguments.passthrough.includes('-p')
  );
}

function printUsageAndExit(message) {
  console.error(`[mermaid] ${message}`);
  console.error(
    [
      'Usage:',
      '  npm run docs:mermaid:validate -- --input path/to/diagram.mmd [extra mmdc args]',
      '  npm run docs:mermaid:export -- --input path/to/diagram.mmd --output path/to/diagram.svg [extra mmdc args]',
    ].join('\n'),
  );
  process.exit(1);
}