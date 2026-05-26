/**
 * Mermaid CLI wrapper entry point.
 *
 * Accepts `validate` or `export` as the first argument followed by
 * `--input`/`--output` options and any extra mmdc flags.
 *
 * This file is intentionally orchestration-only.  All argument parsing,
 * context resolution, Puppeteer config injection, and command execution
 * live in the `scripts/mermaid-cli/` subfolder.
 *
 * Compiled output: `dist-docs/scripts/mermaid-cli.js`
 * Invoked by: `npm run docs:mermaid:validate` and `npm run docs:mermaid:export`
 */

import { buildCommandContext } from './mermaid-cli/mermaid-cli.context.js';
import { runCommand } from './mermaid-cli/mermaid-cli.commands.js';

const commandContext = buildCommandContext(process.argv.slice(2));
await runCommand(commandContext);
