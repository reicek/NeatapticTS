/**
 * @module package-contract.test
 * @description The manifest contract home for the OpenTelemetry package boundary.
 *
 * Slice: OT-V1-1 of `plans/opentelemetry-slice-telemetry.plans.md` (vertical slice A).
 * Acceptance criteria pinned by this home:
 *   - AC-V1-A (the vertical-A package contract), which carries forward
 *     AC-010-A (the manifest), AC-010-B (the excluded names), AC-010-C (the
 *     consumer-install coherence), AC-010-D (one identity per package),
 *     AC-010-E (the two engine floors) and AC-010-F (the root `overrides`
 *     block is ABSENT - the overrides-pinning lane is REPLACED by an
 *     overrides-ABSENCE lane, not left behind).
 *
 * The subject under test is the LIVE manifest, `package.json`, plus the live
 * source tree. Every lane reads the working tree at call time, never a
 * snapshot, so a future edit that violates the contract fails this home.
 *
 * THE CONTRACT THIS HOME DEMANDS:
 *   - the six RULED stable OpenTelemetry packages are declared as peer
 *     dependencies and every one of them is marked optional in
 *     `peerDependenciesMeta`;
 *   - the SAME six are installed as dev dependencies;
 *   - the published runtime dependency list is untouched;
 *   - NO remote exporter, NO experimental package and NO auto-instrumentation
 *     package is declared anywhere in the manifest;
 *   - the RETIRED trace SDK declaration is gone from EVERY dependency
 *     section, and the root `overrides` block is GONE - `overrides` is a
 *     root-only npm mechanism that never travels to a consumer, so it is not
 *     a permitted repair for a peer conflict;
 *   - the four package names that do not exist on npm are absent;
 *   - no source file under `src/observability/` imports from `src/neat/telemetry/`;
 *   - `src/neataptic.ts` does not re-export the observability boundary;
 *   - the `telemetry:*` npm scripts are registered and point at real files;
 *   - the six packages are installed in this repo and satisfy their declared range;
 *   - this home is actually reachable through the declared validation command.
 *
 * WHY THE CONSUMER-INSTALL LANE IS A REAL INSTALL AND NOT A SIMULATION:
 * the published peer ranges are only coherent if a DOWNSTREAM CONSUMER can
 * actually install them, and the only honest way to know that is to let npm
 * resolve the tree. The lane therefore spawns a REAL `npm install --dry-run`
 * over the ranges the manifest publishes, in a throwaway consumer package in
 * a temporary directory. The companion lane is a SELF-CONTAINED interval
 * comparator: this repo declares NO `semver` dependency, so the comparator is
 * written out here, it understands a small and explicit interval grammar, and
 * it FAILS LOUDLY on any grammar it does not understand instead of quietly
 * answering "true".
 *
 * WHY THE EXCLUSION LANES ARE RULES AND NOT ONLY LISTS: a list of forbidden
 * names only catches the names that were known when the list was written. The
 * rules catch their CATEGORIES AS THIS REPO SCOPES THEM - a remote exporter
 * under the `@opentelemetry/exporter-` name prefix, an auto-instrumentation
 * package under any declared name containing the substring
 * `instrumentation`, an experimental 0.x release under the `@opentelemetry/`
 * scope, searched across the four dependency sections and the `overrides`
 * block - so a future edit that adds one of them fails instead of quietly
 * widening the published surface. Each rule is scoped exactly as written: a
 * package published outside those shapes is NOT caught, and this home does
 * not claim to catch it.
 *
 * WHY THE SOURCE LANES ARE NOT VACUOUS TODAY: `src/observability/` does not
 * exist yet, so the live cross-import scan has nothing to walk. That lane is
 * therefore proven on a synthetic import pair, through BOTH halves of the
 * detector: the specifier extractor must still see the observability-to-NEAT-
 * telemetry import, and the path classifier must still classify it as a
 * violation - so the pair fails the moment EITHER half stops working. The
 * boundary lane is proven against the LIVE tree instead: it requires the
 * NEAT-algorithm telemetry directory to exist on disk.
 *
 * @see plans/completed/opentelemetry-slice-telemetry.plans.md - "The package and configuration plan"
 */

import { describe, it, expect } from '@jest/globals';
import { spawnSync } from 'node:child_process';
import {
  existsSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '..', '..');
const MANIFEST_PATH = path.join(REPO_ROOT, 'package.json');
const JEST_CONFIG_PATH = path.join(REPO_ROOT, 'jest.config.mjs');
const SRC_DIR = path.join(REPO_ROOT, 'src');
const OBSERVABILITY_DIR = path.join(SRC_DIR, 'observability');
const NEAT_TELEMETRY_DIR = path.join(SRC_DIR, 'neat', 'telemetry');
const PUBLIC_ENTRYPOINT = path.join(SRC_DIR, 'neataptic.ts');

/**
 * The six RULED stable OpenTelemetry packages and the versions the plan pins
 * (PD-4). The declared RANGE may be a caret range - the repo uses caret ranges
 * everywhere - but the range BASE must be the planned version, so the manifest
 * can never drift onto a different release line.
 *
 * PD-4 replaced the trace SDK base package with the trace SDK itself: the
 * base package is RETIRED and its 2.x line is incompatible with the
 * `@opentelemetry/api` 1.10.0 the rest of the tree requires.
 */
const REQUIRED_PACKAGES = Object.freeze([
  { name: '@opentelemetry/api', version: '1.10.0' },
  { name: '@opentelemetry/context-async-hooks', version: '3.1.1' },
  { name: '@opentelemetry/resources', version: '3.1.1' },
  { name: '@opentelemetry/sdk-logs', version: '3.1.1' },
  { name: '@opentelemetry/sdk-metrics', version: '3.1.1' },
  { name: '@opentelemetry/sdk-trace', version: '3.1.1' },
]);

/**
 * The package names PD-4 RETIRES. They must be absent from EVERY dependency
 * section of the manifest, and a retired name that only survives in one
 * section still installs, so the check is section-by-section.
 */
const RETIRED_DECLARATION_NAMES = Object.freeze([
  '@opentelemetry/sdk-trace-base',
]);

/**
 * The published runtime dependency list. It is FROZEN: the plan states that the
 * published package list must not change and that `dependencies` is untouched.
 */
const RUNTIME_DEPENDENCY_BASELINE = Object.freeze([
  '@huggingface/tokenizers',
  '@libsql/client',
  '@mermaid-js/layout-elk',
  '@mermaid-js/parser',
  'bs-logger',
  'dompurify',
  'glob',
  'seedrandom',
  'undici',
]);

/**
 * The four package names that do not exist on npm (404). The plan lists them
 * with the scope omitted after the first one; they are the scoped names below.
 */
const NON_EXISTENT_PACKAGE_NAMES = Object.freeze([
  '@opentelemetry/sdk-metrics-node',
  '@opentelemetry/sdk-metrics-browser',
  '@opentelemetry/sdk-logs-node',
  '@opentelemetry/sdk-logs-web',
]);

/**
 * The exact names that are deliberately not taken: the 0.x experimental
 * package, the remote OTLP exporters, the Prometheus exporter and the
 * auto-instrumentation bundle.
 */
const EXCLUDED_EXACT_NAMES = Object.freeze([
  '@opentelemetry/sdk-node',
  '@opentelemetry/exporter-trace-otlp-http',
  '@opentelemetry/exporter-metrics-otlp-http',
  '@opentelemetry/exporter-logs-otlp-http',
  '@opentelemetry/exporter-prometheus',
  'auto-instrumentations-node',
]);

/**
 * The engine floors the plan states. The ROOT floor stays where it is; the
 * telemetry path declares and enforces its OWN, higher floor because the
 * OpenTelemetry 3.x line requires it.
 */
const ROOT_NODE_FLOOR = '>= 22.0.0';
const TELEMETRY_NODE_FLOOR = '>= 22.15.0';

/**
 * The machine-stable reason token the telemetry path reports when the
 * runtime is below its own floor. A token, not a sentence: it has to be
 * greppable and stable across releases.
 */
const REASON_NODE_BELOW_TELEMETRY_FLOOR = 'node_below_telemetry_floor';

/**
 * Reads the live manifest. Called inside every lane so a lane always asserts
 * against the current working tree.
 *
 * @returns {Record<string, any>} The parsed manifest.
 */
function readManifest() {
  return JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
}

/**
 * Collects every package name the manifest declares, across every dependency
 * section AND the `overrides` block - an override is a declaration too, and a
 * remote exporter smuggled in there would install exactly like a dependency.
 *
 * @param {Record<string, any>} manifest The manifest.
 * @returns {string[]} The declared names.
 */
function collectDeclaredNames(manifest) {
  const names = [];
  for (const section of [
    'dependencies',
    'devDependencies',
    'peerDependencies',
    'optionalDependencies',
  ]) {
    for (const name of Object.keys(manifest[section] ?? {})) names.push(name);
  }
  for (const [name, value] of Object.entries(manifest.overrides ?? {})) {
    names.push(name);
    if (value !== null && typeof value === 'object') {
      for (const nested of Object.keys(value)) names.push(nested);
    }
  }
  return names;
}

/**
 * Collects every declared package together with the spec it is declared at.
 * An `overrides` entry is a declaration too: a string value is a version spec
 * and an object value nests one level of `name -> spec` entries. An override
 * subtree root carries no spec of its own - its release is the one the tree
 * resolves to - so it is collected with an empty spec.
 *
 * @param {Record<string, any>} manifest The manifest.
 * @returns {{ name: string, spec: string }[]} The declared packages.
 */
function collectDeclaredSpecs(manifest) {
  const specs = [];
  for (const section of [
    'dependencies',
    'devDependencies',
    'peerDependencies',
    'optionalDependencies',
  ]) {
    for (const [name, spec] of Object.entries(manifest[section] ?? {})) {
      specs.push({ name, spec: String(spec) });
    }
  }
  for (const [name, value] of Object.entries(manifest.overrides ?? {})) {
    if (value !== null && typeof value === 'object') {
      specs.push({ name, spec: '' });
      for (const [nestedName, nestedSpec] of Object.entries(value)) {
        specs.push({ name: nestedName, spec: String(nestedSpec) });
      }
    } else {
      specs.push({ name, spec: String(value) });
    }
  }
  return specs;
}

/**
 * Decides whether a declared spec pins a 0.x release - the experimental
 * OpenTelemetry line the plan refuses. A spec that names no release (an npm
 * `$ref`, a peer range, an empty override root) is not judged here.
 *
 * @param {string} spec The declared spec.
 * @returns {boolean} True when the spec pins a 0.x release.
 */
function isExperimentalSpec(spec) {
  const match = /^\s*[\^~=]?\s*v?(\d+)\.(\d+)\.(\d+)/.exec(String(spec));
  if (match === null) return false;
  return Number.parseInt(match[1], 10) === 0;
}

/**
 * Strips the version range off a declared dependency spec, leaving the base
 * version. `^1.10.0` and `1.10.0` both yield `1.10.0`.
 *
 * @param {string} spec The declared spec.
 * @returns {string} The base version.
 */
function baseVersion(spec) {
  return String(spec).trim().replace(/^[\^~=><\s]+/, '');
}

/**
 * Parses a `x.y.z` version into its numeric parts.
 *
 * @param {string} version The version.
 * @returns {{ major: number, minor: number, patch: number }} The parsed version.
 */
function parseVersion(version) {
  const [major = 0, minor = 0, patch = 0] = String(version)
    .split('.')
    .map((part) => Number.parseInt(part, 10) || 0);
  return { major, minor, patch };
}

/**
 * Decides whether an installed version satisfies a caret range.
 *
 * @param {string} range The declared range.
 * @param {string} installed The installed version.
 * @returns {boolean} True when the installed version satisfies the range.
 */
function satisfiesCaretRange(range, installed) {
  const base = parseVersion(baseVersion(range));
  const actual = parseVersion(installed);
  if (actual.major !== base.major) return false;
  if (actual.minor > base.minor) return true;
  if (actual.minor < base.minor) return false;
  return actual.patch >= base.patch;
}

/**
 * Lists the TypeScript source files under a directory. A missing directory
 * yields an empty list - the boundary may not exist yet.
 *
 * @param {string} directory The directory.
 * @returns {string[]} The absolute file paths.
 */
function listTypeScriptFiles(directory) {
  if (!existsSync(directory)) return [];
  const files = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...listTypeScriptFiles(full));
    else if (entry.isFile() && entry.name.endsWith('.ts')) files.push(full);
  }
  return files;
}

/**
 * Removes the comments from a source file so a prose mention of an import is
 * never mistaken for an import. A `//` that follows a `:` is kept, so a URL
 * inside a string literal survives.
 *
 * @param {string} source The source text.
 * @returns {string} The source text without comments.
 */
function stripComments(source) {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1');
}

const SPECIFIER_PATTERNS = Object.freeze([
  /\bfrom\s*['"]([^'"\n]+)['"]/g,
  /\bimport\s*\(\s*['"]([^'"\n]+)['"]\s*\)/g,
  /\brequire\s*\(\s*['"]([^'"\n]+)['"]\s*\)/g,
  /\bimport\s+['"]([^'"\n]+)['"]/g,
]);

/**
 * Extracts every module specifier a source file names.
 *
 * @param {string} source The source text.
 * @returns {string[]} The module specifiers.
 */
function extractModuleSpecifiers(source) {
  const code = stripComments(source);
  const specifiers = new Set();
  for (const pattern of SPECIFIER_PATTERNS) {
    pattern.lastIndex = 0;
    let match = pattern.exec(code);
    while (match !== null) {
      specifiers.add(match[1]);
      match = pattern.exec(code);
    }
  }
  return [...specifiers];
}

/**
 * Resolves a relative module specifier against the file that names it. A
 * bare package specifier resolves to nothing - it cannot name a repo path.
 *
 * @param {string} fromFile The file that names the specifier.
 * @param {string} specifier The module specifier.
 * @returns {string | null} The resolved path, or null for a bare specifier.
 */
function resolveRelativeSpecifier(fromFile, specifier) {
  if (!specifier.startsWith('./') && !specifier.startsWith('../')) return null;
  return path.resolve(path.dirname(fromFile), specifier);
}

/**
 * Decides whether a path lies inside a directory.
 *
 * @param {string} directory The directory.
 * @param {string} target The path.
 * @returns {boolean} True when the path lies inside the directory.
 */
function isInsideDirectory(directory, target) {
  const relative = path.relative(directory, target);
  return (
    relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
  );
}

/**
 * Lists the relative specifiers a file uses that resolve into a directory.
 *
 * @param {string} fromFile The file.
 * @param {string} directory The forbidden directory.
 * @returns {string[]} The offending specifiers.
 */
function findSpecifiersResolvingInto(fromFile, directory) {
  const source = readFileSync(fromFile, 'utf8');
  return extractModuleSpecifiers(source).filter((specifier) => {
    const resolved = resolveRelativeSpecifier(fromFile, specifier);
    return resolved !== null && isInsideDirectory(directory, resolved);
  });
}

/**
 * Extracts the repo-relative file paths a shell command names.
 *
 * @param {string} command The command.
 * @returns {string[]} The repo-relative paths.
 */
function extractScriptPaths(command) {
  const paths = [];
  for (const token of String(command).split(/\s+/)) {
    const cleaned = token.includes('=')
      ? token.slice(token.lastIndexOf('=') + 1)
      : token;
    if (/^[\w.-]+(?:\/[\w.-]+)*\.(?:mjs|cjs|js|ts|json)$/.test(cleaned)) {
      paths.push(cleaned);
    }
  }
  return paths;
}

/**
 * Reads the installed manifest of a package in this repo.
 *
 * @param {string} name The package name.
 * @returns {Record<string, any> | null} The installed manifest, or null.
 */
function readInstalledManifest(name) {
  const installed = path.join(
    REPO_ROOT,
    'node_modules',
    ...name.split('/'),
    'package.json',
  );
  if (!existsSync(installed)) return null;
  return JSON.parse(readFileSync(installed, 'utf8'));
}

/**
 * Lists every INSTALLED `@opentelemetry/*` package in this repo, INCLUDING the
 * nested duplicates npm leaves under a package's own `node_modules`. A nested
 * duplicate is a second copy of the same package at a second version, which
 * is exactly what AC-010-D forbids.
 *
 * @returns {{ name: string, version: string, peerDependencies: Record<string, string> }[]} The installed packages.
 */
function listInstalledOpenTelemetryPackages() {
  const scopeRoot = path.join(REPO_ROOT, 'node_modules', '@opentelemetry');
  const found = new Map();
  const walk = (directory) => {
    if (!existsSync(directory)) return;
    for (const entry of readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const child = path.join(directory, entry.name);
      const manifestPath = path.join(child, 'package.json');
      if (existsSync(manifestPath)) {
        try {
          const parsed = JSON.parse(readFileSync(manifestPath, 'utf8'));
          const key = `${parsed.name ?? entry.name}@${parsed.version ?? 'unknown'}`;
          if (!found.has(key)) {
            found.set(key, {
              name: parsed.name ?? entry.name,
              version: parsed.version ?? 'unknown',
              peerDependencies: parsed.peerDependencies ?? {},
            });
          }
        } catch {
          // An unreadable manifest is not an OpenTelemetry package.
        }
      }
      walk(path.join(child, 'node_modules', '@opentelemetry'));
    }
  };
  walk(scopeRoot);
  return [...found.values()];
}

/**
 * The npm CLI entry point used by the consumer-install dry run. The dry run
 * is a REAL npm install, so it is spawned through the npm CLI JavaScript file
 * with the current Node binary: that is portable across platforms and needs
 * no shell.
 *
 * @returns {string} The absolute path of the npm CLI entry point.
 */
function resolveNpmCli() {
  const fromEnvironment = process.env.npm_execpath;
  if (
    typeof fromEnvironment === 'string' &&
    fromEnvironment.length > 0 &&
    existsSync(fromEnvironment)
  ) {
    return fromEnvironment;
  }
  return path.join(
    path.dirname(process.execPath),
    'node_modules',
    'npm',
    'bin',
    'npm-cli.js',
  );
}

/**
 * Runs a REAL consumer-install dry run over the ranges the manifest
 * publishes.
 *
 * The consumer is a throwaway package in a temporary directory that declares
 * the published ranges as its own `dependencies`, exactly as a downstream
 * consumer would. `--dry-run` resolves the whole tree and reports the
 * resolution without writing it, so a peer conflict surfaces as an ERESOLVE
 * and a non-zero exit code. Nothing in this repo is mutated.
 *
 * @param {Record<string, string>} ranges The published ranges to declare.
 * @returns {{ status: number | null, output: string }} The dry-run outcome.
 */
function runConsumerInstallDryRun(ranges) {
  const directory = mkdtempSync(path.join(tmpdir(), 'otel-consumer-'));
  writeFileSync(
    path.join(directory, 'package.json'),
    `${JSON.stringify(
      {
        name: 'otel-consumer-install-probe',
        version: '0.0.0',
        private: true,
        dependencies: ranges,
      },
      null,
      2,
    )}\n`,
    'utf8',
  );
  const result = spawnSync(
    process.execPath,
    [
      resolveNpmCli(),
      'install',
      '--dry-run',
      '--ignore-scripts',
      '--no-audit',
      '--no-fund',
      '--no-package-lock',
    ],
    { cwd: directory, encoding: 'utf8', timeout: 240000 },
  );
  return {
    status: result.status,
    output: `${result.stdout ?? ''}${result.stderr ?? ''}`,
  };
}

const SEMVER_LITERAL_PATTERN = /^(\d+)\.(\d+)\.(\d+)$/;

/**
 * Parses a STRICT `x.y.z` version literal. A literal that is not exactly
 * three numeric parts is a grammar this comparator does not understand, and
 * it throws rather than guessing.
 *
 * @param {string} version The version literal.
 * @returns {{ major: number, minor: number, patch: number }} The parsed version.
 */
function parseSemver(version) {
  const text = String(version).trim().replace(/^v/i, '');
  const match = SEMVER_LITERAL_PATTERN.exec(text);
  if (match === null) {
    throw new Error(`unsupported version literal: ${version}`);
  }
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
  };
}

/**
 * Orders two parsed versions.
 *
 * @param {{ major: number, minor: number, patch: number }} left The left version.
 * @param {{ major: number, minor: number, patch: number }} right The right version.
 * @returns {number} A negative, zero or positive ordering.
 */
function compareSemver(left, right) {
  if (left.major !== right.major) return left.major - right.major;
  if (left.minor !== right.minor) return left.minor - right.minor;
  return left.patch - right.patch;
}

/**
 * Parses ONE term of an interval. The grammar is deliberately small:
 * `>=`, `>`, `<=`, `<`, `=`, `^`, `~` and a bare `x.y.z`. Anything else -
 * a wildcard, an `||` union, a `~>` pessimistic operator - is a grammar this
 * comparator does not understand, and it THROWS. A comparator that answered
 * "true" for a grammar it did not parse would make every lane built on it
 * vacuous.
 *
 * @param {string} term The interval term.
 * @returns {Record<string, any>} The parsed term.
 */
function parseIntervalTerm(term) {
  const text = String(term).trim();
  if (text.length === 0) {
    throw new Error('unsupported interval grammar: an empty term');
  }
  const match = /^(\^|~|>=|<=|>|<|=)?\s*(.+)$/.exec(text);
  if (match === null) {
    throw new Error(`unsupported interval grammar: ${text}`);
  }
  const operator = match[1] ?? '=';
  const literal = match[2].trim();
  if (/[xX*]/.test(literal)) {
    throw new Error(`unsupported interval grammar: ${text}`);
  }
  const version = parseSemver(literal);
  if (operator === '^') {
    const upper =
      version.major > 0
        ? { major: version.major + 1, minor: 0, patch: 0 }
        : version.minor > 0
          ? { major: 0, minor: version.minor + 1, patch: 0 }
          : { major: 0, minor: 0, patch: version.patch + 1 };
    return {
      lower: version,
      lowerInclusive: true,
      upper,
      upperInclusive: false,
    };
  }
  if (operator === '~') {
    return {
      lower: version,
      lowerInclusive: true,
      upper: { major: version.major, minor: version.minor + 1, patch: 0 },
      upperInclusive: false,
    };
  }
  if (operator === '>=') return { lower: version, lowerInclusive: true };
  if (operator === '>') return { lower: version, lowerInclusive: false };
  if (operator === '<=') return { upper: version, upperInclusive: true };
  if (operator === '<') return { upper: version, upperInclusive: false };
  return { exact: version };
}

/**
 * Parses a whole interval. A whitespace-ANDed list of terms is the only
 * conjunction this comparator understands.
 *
 * @param {string} range The interval.
 * @returns {Record<string, any>[]} The parsed terms.
 */
function parseInterval(range) {
  const text = String(range).trim();
  if (text.length === 0) {
    throw new Error('unsupported interval grammar: an empty range');
  }
  if (text.includes('|')) {
    throw new Error(`unsupported interval grammar: ${text}`);
  }
  const terms = text.split(/\s+/).filter((term) => term.length > 0);
  if (terms.length === 0) {
    throw new Error(`unsupported interval grammar: ${text}`);
  }
  return terms.map(parseIntervalTerm);
}

/**
 * Decides whether a version satisfies an interval.
 *
 * @param {string} version The version.
 * @param {string} range The interval.
 * @returns {boolean} True when the version satisfies the interval.
 */
function satisfiesInterval(version, range) {
  const target = parseSemver(version);
  return parseInterval(range).every((term) => {
    if (term.exact !== undefined) {
      return compareSemver(target, term.exact) === 0;
    }
    if (term.lower !== undefined) {
      const comparison = compareSemver(target, term.lower);
      if (term.lowerInclusive ? comparison < 0 : comparison <= 0) return false;
    }
    if (term.upper !== undefined) {
      const comparison = compareSemver(target, term.upper);
      if (term.upperInclusive ? comparison > 0 : comparison >= 0) return false;
    }
    return true;
  });
}

/**
 * The tolerant form of `satisfiesInterval` used by the lanes that assert on
 * a range this comparator may not be able to parse. An unparseable range
 * ADMITS NOTHING, so a range the comparator cannot read fails the lane by
 * assertion instead of throwing an unrelated error.
 *
 * @param {string} range The interval.
 * @param {string} version The version.
 * @returns {boolean} True when the range admits the version.
 */
function admits(range, version) {
  try {
    return satisfiesInterval(version, range);
  } catch {
    return false;
  }
}

/**
 * Reads the committed lockfile.
 *
 * @returns {Record<string, any>} The parsed lockfile.
 */
function readLockfile() {
  return JSON.parse(
    readFileSync(path.join(REPO_ROOT, 'package-lock.json'), 'utf8'),
  );
}

/**
 * Every lockfile path that resolves the given package name, INCLUDING the
 * nested duplicates npm records under a package's own `node_modules`.
 *
 * @param {Record<string, any>} lockfile The lockfile.
 * @param {string} packageName The package name.
 * @returns {string[]} The matching lockfile paths.
 */
function lockfileIdentityPaths(lockfile, packageName) {
  const needle = `node_modules/${packageName}`;
  return Object.keys(lockfile.packages ?? {}).filter(
    (key) => key === needle || key.endsWith(`/${needle}`),
  );
}

/**
 * Lists the runtime exports of an installed package by importing it in a
 * child process, so the home never links an OpenTelemetry package into its
 * own module graph.
 *
 * @param {string} name The package name.
 * @returns {{ status: number | null, exports: string[], output: string }} The export probe.
 */
function probePackageExports(name) {
  const script = [
    'const mod = await import(',
    JSON.stringify(name),
    ');',
    'process.stdout.write(JSON.stringify(Object.keys(mod)));',
  ].join('');
  const result = spawnSync(
    process.execPath,
    ['--input-type=module', '-e', script],
    { cwd: REPO_ROOT, encoding: 'utf8', timeout: 120000 },
  );
  let exported = [];
  try {
    exported = JSON.parse(result.stdout ?? '[]');
  } catch {
    exported = [];
  }
  return {
    status: result.status,
    exports: exported,
    output: `${result.stdout ?? ''}${result.stderr ?? ''}`,
  };
}

describe('AC-V1-A / AC-010-A: the manifest declares the six RULED OpenTelemetry packages', () => {
  it('declares each of the six packages as a peer dependency', () => {
    const manifest = readManifest();
    const peers = manifest.peerDependencies ?? {};
    for (const { name, version } of REQUIRED_PACKAGES) {
      expect(peers[name]).toBeDefined();
      expect(baseVersion(peers[name])).toBe(version);
    }
  });

  it('marks every peer dependency optional in peerDependenciesMeta', () => {
    const manifest = readManifest();
    const meta = manifest.peerDependenciesMeta ?? {};
    for (const { name } of REQUIRED_PACKAGES) {
      expect(meta[name]).toBeDefined();
      expect(meta[name].optional).toBe(true);
    }
  });

  it('declares exactly the six packages and no other peer dependency', () => {
    const manifest = readManifest();
    expect(Object.keys(manifest.peerDependencies ?? {}).sort()).toEqual(
      REQUIRED_PACKAGES.map(({ name }) => name).sort(),
    );
    expect(Object.keys(manifest.peerDependenciesMeta ?? {}).sort()).toEqual(
      REQUIRED_PACKAGES.map(({ name }) => name).sort(),
    );
  });

  it('installs the same six packages as dev dependencies', () => {
    const manifest = readManifest();
    const dev = manifest.devDependencies ?? {};
    for (const { name, version } of REQUIRED_PACKAGES) {
      expect(dev[name]).toBeDefined();
      expect(baseVersion(dev[name])).toBe(version);
    }
  });

  it('declares exactly the six OpenTelemetry packages in devDependencies', () => {
    const manifest = readManifest();
    const dev = manifest.devDependencies ?? {};
    const scoped = Object.keys(dev)
      .filter((name) => name.startsWith('@opentelemetry/'))
      .sort();
    expect(scoped).toEqual(
      REQUIRED_PACKAGES.map(({ name }) => name).sort(),
    );
  });

  it('leaves the published runtime dependency list untouched', () => {
    const manifest = readManifest();
    expect(Object.keys(manifest.dependencies ?? {}).sort()).toEqual(
      [...RUNTIME_DEPENDENCY_BASELINE].sort(),
    );
    expect(
      Object.keys(manifest.dependencies ?? {}).filter((name) =>
        name.startsWith('@opentelemetry/'),
      ),
    ).toEqual([]);
    // `optionalDependencies` is a runtime install channel into every
    // consumer's tree, and no other lane pins it: it is asserted here so a
    // package can never appear there without the contract noticing.
    expect(Object.keys(manifest.optionalDependencies ?? {}).sort()).toEqual(
      [],
    );
  });

  it('declares no remote exporter package', () => {
    const manifest = readManifest();
    const exporters = collectDeclaredNames(manifest).filter((name) =>
      name.startsWith('@opentelemetry/exporter-'),
    );
    expect(exporters).toEqual([]);
  });

  it('declares no experimental package', () => {
    const manifest = readManifest();
    const declared = collectDeclaredSpecs(manifest);
    // The plan refuses the 0.x line: `@opentelemetry/sdk-node` (0.301.2) and
    // the 0.x exporters are "0.x experimental". The rule is the RELEASE LINE,
    // not the name, so a 0.x package is caught whatever it is called.
    const experimental = declared
      .filter(({ name }) => name.startsWith('@opentelemetry/'))
      .filter(({ spec }) => isExperimentalSpec(spec))
      .map(({ name }) => name);
    expect(experimental).toEqual([]);
    // The plan's named experimental package is refused at any version.
    expect(declared.map(({ name }) => name)).not.toContain(
      '@opentelemetry/sdk-node',
    );
  });

  it('declares no auto-instrumentation package', () => {
    const manifest = readManifest();
    const instrumentation = collectDeclaredNames(manifest).filter((name) =>
      name.includes('instrumentation'),
    );
    expect(instrumentation).toEqual([]);
  });

  it('carries NO root overrides block - an override is a root-only repair that never reaches a consumer', () => {
    const manifest = readManifest();
    // PD-4 REPLACES the overrides-pinning lane with an overrides-ABSENCE
    // lane. `overrides` is a root-only npm mechanism: npm does not record it
    // in `package-lock.json` and npm does not propagate it into a package
    // that depends on this one, so an override can make THIS repo install
    // while leaving every downstream consumer broken. The published peer
    // ranges have to be coherent ON THEIR OWN, and the consumer-install dry
    // run is the lane that proves it.
    expect(Object.keys(manifest.overrides ?? {})).toEqual([]);
    expect(manifest.overrides).toBeUndefined();
  });

  it('leaves no OpenTelemetry override key behind anywhere in the manifest', () => {
    const manifest = readManifest();
    const scoped = Object.keys(manifest.overrides ?? {}).filter((name) =>
      name.startsWith('@opentelemetry/'),
    );
    expect(scoped).toEqual([]);
  });
});

describe('AC-010-B: the excluded names and the boundary separation', () => {
  it('declares none of the excluded package names', () => {
    const manifest = readManifest();
    const declared = collectDeclaredNames(manifest);
    for (const name of EXCLUDED_EXACT_NAMES) {
      expect(declared).not.toContain(name);
    }
  });

  it('declares none of the four package names that do not exist on npm', () => {
    const manifest = readManifest();
    const declared = collectDeclaredNames(manifest);
    for (const name of NON_EXISTENT_PACKAGE_NAMES) {
      expect(declared).not.toContain(name);
    }
  });

  it('asserts that no source file under src/observability imports from src/neat/telemetry', () => {
    const files = listTypeScriptFiles(OBSERVABILITY_DIR);
    const violations = files.flatMap((file) =>
      findSpecifiersResolvingInto(file, NEAT_TELEMETRY_DIR).map(
        (specifier) => `${path.relative(REPO_ROOT, file)} -> ${specifier}`,
      ),
    );
    expect(violations).toEqual([]);
  });

  it('asserts that src/neataptic.ts does not re-export the observability boundary', () => {
    const offending = findSpecifiersResolvingInto(
      PUBLIC_ENTRYPOINT,
      OBSERVABILITY_DIR,
    );
    expect(offending).toEqual([]);
  });

  it('proves the boundary detector classifies an observability-to-NEAT-telemetry import as a violation', () => {
    const syntheticFile = path.join(
      OBSERVABILITY_DIR,
      'telemetry',
      'synthetic.ts',
    );
    const forbidden = resolveRelativeSpecifier(
      syntheticFile,
      '../../neat/telemetry/facade/telemetry.facade',
    );
    const allowed = resolveRelativeSpecifier(
      syntheticFile,
      '../../architecture/network',
    );
    expect(forbidden).not.toBeNull();
    expect(isInsideDirectory(NEAT_TELEMETRY_DIR, forbidden)).toBe(true);
    expect(isInsideDirectory(NEAT_TELEMETRY_DIR, allowed)).toBe(false);
    // The EXTRACTION half of the detector must be exercised too: the three
    // assertions above prove only the path classifier, not the specifier
    // extractor that the cross-import lane depends on.
    const syntheticSource = [
      "import { telemetryFacade } from '../../neat/telemetry/facade/telemetry.facade';",
      "import { network } from '../../architecture/network';",
    ].join('\n');
    const extracted = extractModuleSpecifiers(syntheticSource);
    expect(extracted).toContain('../../neat/telemetry/facade/telemetry.facade');
    expect(extracted).toContain('../../architecture/network');
    expect(extracted).toHaveLength(2);
  });

  it('keeps the observability boundary a directory of its own, distinct from the NEAT-algorithm telemetry', () => {
    // The premise is the LIVE tree, not two path constants: the boundary this
    // lane protects only means something while the NEAT-algorithm telemetry
    // directory it protects against exists.
    expect(existsSync(NEAT_TELEMETRY_DIR)).toBe(true);
    expect(OBSERVABILITY_DIR).not.toBe(NEAT_TELEMETRY_DIR);
    expect(isInsideDirectory(NEAT_TELEMETRY_DIR, OBSERVABILITY_DIR)).toBe(false);
    expect(isInsideDirectory(OBSERVABILITY_DIR, NEAT_TELEMETRY_DIR)).toBe(false);
  });
});

describe('the telemetry script surface and the installed package set', () => {
  it('registers telemetry:* npm scripts that point at real files', () => {
    const manifest = readManifest();
    const telemetryScripts = Object.entries(manifest.scripts ?? {}).filter(
      ([name]) => name.startsWith('telemetry:'),
    );
    // The slice's deliverable is two NAMED scripts: both must be present, so
    // deleting one of them turns the home red. The SET is deliberately NOT
    // pinned to exactly those two, because the vertical slice is allowed to
    // register its own scripts - only the named ones and the file targets
    // are contract.
    const names = telemetryScripts.map(([name]) => name);
    expect(names).toContain('telemetry:probe-attribution');
    expect(names).toContain('telemetry:test');
    // And `telemetry:probe-attribution` must still point at the probe script
    // itself, not merely at some existing file.
    expect(manifest.scripts?.['telemetry:probe-attribution']).toContain(
      'scripts/telemetry/probe-attribution-sources.mjs',
    );
    for (const [name, command] of telemetryScripts) {
      const referenced = extractScriptPaths(command);
      expect(referenced.length).toBeGreaterThan(0);
      for (const referencedPath of referenced) {
        expect({
          script: name,
          path: referencedPath,
          exists: existsSync(path.join(REPO_ROOT, referencedPath)),
        }).toEqual({ script: name, path: referencedPath, exists: true });
      }
    }
  });

  it('installs the six packages in this repo at a version that satisfies the declared range', () => {
    const manifest = readManifest();
    const dev = manifest.devDependencies ?? {};
    for (const { name } of REQUIRED_PACKAGES) {
      const installed = readInstalledManifest(name);
      expect(installed).not.toBeNull();
      expect(satisfiesCaretRange(dev[name], installed.version)).toBe(true);
    }
  });

  it('runs this home through the declared validation command', () => {
    const manifest = readManifest();
    expect(manifest.scripts['jest:mjs']).toContain(
      '--selectProjects telemetry-scripts',
    );
    const jestConfig = readFileSync(JEST_CONFIG_PATH, 'utf8');
    expect(jestConfig).toContain('telemetry-scripts');
    expect(jestConfig).toContain('**/scripts/telemetry/**/*.test.mjs');
  });
});

describe('AC-V1-A / AC-010-C: the published peer ranges are coherent for a consumer', () => {
  it('publishes peer ranges that a REAL consumer install can resolve', () => {
    const manifest = readManifest();
    const peers = manifest.peerDependencies ?? {};
    // The dry run installs EXACTLY what the manifest publishes - not the
    // ruled set the contract wishes for - so the lane reports the state of
    // the published contract and not of an aspiration. (The published SET is
    // pinned by its own lane in AC-010-A above.)
    const ranges = { ...peers };
    expect(Object.keys(ranges).length).toBeGreaterThan(0);
    for (const range of Object.values(ranges)) {
      expect(typeof range).toBe('string');
      expect(range.length).toBeGreaterThan(0);
    }
    // The dry run is a REAL `npm install --dry-run` in a throwaway consumer
    // package in a temporary directory. A peer conflict surfaces as an
    // ERESOLVE and a non-zero exit code. The lane is a PURE ASSERTION: the
    // npm error code is reported inside the received object so a failure is
    // diagnosable without a thrown error.
    const result = runConsumerInstallDryRun(ranges);
    expect({
      exitCode: result.status,
      npmErrorCode: (result.output.match(/npm error code \w+/) ?? [null])[0],
    }).toEqual({ exitCode: 0, npmErrorCode: null });
  }, 300000);

  it('installs an api version that every installed OpenTelemetry package admits', () => {
    const apiVersion = readInstalledManifest('@opentelemetry/api')?.version;
    expect(apiVersion).toBeDefined();
    const incompatible = [];
    for (const installed of listInstalledOpenTelemetryPackages()) {
      const range = installed.peerDependencies?.['@opentelemetry/api'];
      if (range === undefined) continue;
      if (!admits(range, apiVersion)) {
        incompatible.push(
          `${installed.name}@${installed.version} peers api ${range}`,
        );
      }
    }
    expect(incompatible).toEqual([]);
  });

  it('fails LOUDLY on an interval grammar it does not understand', () => {
    // The comparator is self-contained, so it must never answer "true" for a
    // grammar it did not parse: a silent "true" would make the lane above
    // vacuous.
    for (const grammar of [
      '1.x',
      '*',
      '>=1.10.0 || <1.9.0',
      '~>1.2.3',
      'latest',
    ]) {
      expect(() => satisfiesInterval('1.10.0', grammar)).toThrow(
        /unsupported interval grammar|unsupported version literal/,
      );
    }
    // And the grammar it claims to understand is understood.
    expect(satisfiesInterval('1.10.0', '>=1.3.0 <1.10.0')).toBe(false);
    expect(satisfiesInterval('1.10.0', '>=1.3.0 <1.11.0')).toBe(true);
    expect(satisfiesInterval('3.1.1', '^3.1.1')).toBe(true);
    expect(satisfiesInterval('3.2.0', '^3.1.1')).toBe(true);
    expect(satisfiesInterval('4.0.0', '^3.1.1')).toBe(false);
    expect(satisfiesInterval('3.1.2', '~3.1.1')).toBe(true);
    expect(satisfiesInterval('3.2.0', '~3.1.1')).toBe(false);
    expect(satisfiesInterval('1.10.0', '=1.10.0')).toBe(true);
  });

  it('keeps the interval comparator self-contained - the repo declares no range library', () => {
    const manifest = readManifest();
    for (const section of [
      'dependencies',
      'devDependencies',
      'peerDependencies',
      'optionalDependencies',
    ]) {
      expect(Object.keys(manifest[section] ?? {})).not.toContain('semver');
    }
    // The home must not have grown an import of a range library either.
    const homeSource = readFileSync(
      path.join(HERE, 'package-contract.test.mjs'),
      'utf8',
    );
    expect(homeSource).not.toMatch(/from\s+['"]semver['"]/);
  });
});

describe('AC-V1-A / AC-010-D: one identity per OpenTelemetry package in the lockfile', () => {
  it('resolves exactly ONE @opentelemetry/core in the lockfile', () => {
    const lockfile = readLockfile();
    expect(lockfileIdentityPaths(lockfile, '@opentelemetry/core')).toEqual([
      'node_modules/@opentelemetry/core',
    ]);
  });

  it('resolves exactly ONE @opentelemetry/resources in the lockfile', () => {
    const lockfile = readLockfile();
    expect(lockfileIdentityPaths(lockfile, '@opentelemetry/resources')).toEqual([
      'node_modules/@opentelemetry/resources',
    ]);
  });

  it('carries no nested duplicate of any @opentelemetry package in the lockfile', () => {
    const lockfile = readLockfile();
    const nested = Object.keys(lockfile.packages ?? {}).filter((key) =>
      /^node_modules\/@opentelemetry\/[^/]+\/node_modules\//.test(key),
    );
    expect(nested).toEqual([]);
  });

  it('resolves the trace SDK at the ruled release in the lockfile', () => {
    const lockfile = readLockfile();
    const entry = lockfile.packages?.['node_modules/@opentelemetry/sdk-trace'];
    expect(entry).toBeDefined();
    expect(entry.version).toBe('3.1.1');
  });

  it('resolves no retired package in the lockfile', () => {
    const lockfile = readLockfile();
    const resolved = Object.keys(lockfile.packages ?? {}).filter((key) =>
      RETIRED_DECLARATION_NAMES.some(
        (name) =>
          key === `node_modules/${name}` ||
          key.endsWith(`/node_modules/${name}`),
      ),
    );
    expect(resolved).toEqual([]);
  });

  it('records the ruled six in the lockfile root entry', () => {
    const lockfile = readLockfile();
    const root = lockfile.packages?.[''];
    expect(root).toBeDefined();
    const dev = root.devDependencies ?? {};
    const scoped = Object.keys(dev)
      .filter((name) => name.startsWith('@opentelemetry/'))
      .sort();
    expect(scoped).toEqual(
      REQUIRED_PACKAGES.map(({ name }) => name).sort(),
    );
  });
});

describe('AC-V1-A / AC-010-E: the two engine floors', () => {
  it('keeps the root engine floor at 22.0.0', () => {
    const manifest = readManifest();
    const engines = manifest.engines?.node;
    expect(engines).toBeDefined();
    expect(admits(engines, '22.0.0')).toBe(true);
    expect(admits(engines, '21.99.99')).toBe(false);
  });

  it('declares a telemetry engine floor strictly above the root floor', () => {
    const manifest = readManifest();
    const rootEngines = manifest.engines?.node;
    expect(rootEngines).toBeDefined();
    // The root floor still admits the runtime the telemetry path refuses, so
    // the telemetry path has to police the gap for itself.
    expect(admits(rootEngines, '22.15.0')).toBe(true);
    // The telemetry path declares its OWN floor, and the reason it refuses a
    // runtime is a machine-stable token, not a sentence.
    const files = listTypeScriptFiles(OBSERVABILITY_DIR);
    expect(files.length).toBeGreaterThan(0);
    const sources = files.map((file) => readFileSync(file, 'utf8')).join('\n');
    expect(sources).toContain('22.15.0');
    expect(sources).toContain(REASON_NODE_BELOW_TELEMETRY_FLOOR);
  });

  it('installs every non-api ruled package at a release that requires the telemetry floor', () => {
    const offenders = [];
    for (const { name } of REQUIRED_PACKAGES) {
      if (name === '@opentelemetry/api') continue;
      const installed = readInstalledManifest(name);
      if (installed === null) {
        offenders.push(`${name} is not installed`);
        continue;
      }
      const engines = installed.engines?.node;
      if (engines === undefined) {
        offenders.push(`${name}@${installed.version} declares no engines.node`);
        continue;
      }
      if (!admits(engines, '22.15.0') || admits(engines, '22.14.0')) {
        offenders.push(`${name}@${installed.version} engines.node=${engines}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});

describe('AC-V1-A / PD-4: the retired trace SDK is gone', () => {
  it('declares no retired package name in any dependency section', () => {
    const manifest = readManifest();
    const offenders = [];
    for (const section of [
      'dependencies',
      'devDependencies',
      'peerDependencies',
      'optionalDependencies',
      'overrides',
    ]) {
      const declared = Object.keys(manifest[section] ?? {});
      for (const name of RETIRED_DECLARATION_NAMES) {
        if (declared.includes(name)) offenders.push(`${section}:${name}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('exposes no retired tracer provider factory from the installed packages', () => {
    // The needle is assembled at runtime so this home does not itself carry
    // the literal it forbids.
    const needle = ['Basic', 'Tracer', 'Provider'].join('');
    const offenders = [];
    for (const installed of listInstalledOpenTelemetryPackages()) {
      const probe = probePackageExports(installed.name);
      if (probe.status !== 0) {
        offenders.push(`${installed.name}@${installed.version}: not importable`);
        continue;
      }
      if (probe.exports.includes(needle)) {
        offenders.push(`${installed.name}@${installed.version}`);
      }
    }
    expect(offenders).toEqual([]);
  });

  it('installs each ruled package at exactly the release the plan pins', () => {
    const offenders = [];
    for (const { name, version } of REQUIRED_PACKAGES) {
      const installed = readInstalledManifest(name);
      const actual = installed?.version ?? null;
      if (actual !== version) {
        offenders.push(`${name}: ${actual} !== ${version}`);
      }
    }
    expect(offenders).toEqual([]);
  });
});
