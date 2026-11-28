/**
 * The READER of the LOCAL telemetry settings: the file
 * `logs/telemetry/settings.json` under the repository root, which the local
 * enablement command owns and which the exporters OBEY.
 *
 * WHAT THIS MODULE IS: the one place where the local signal layer learns
 * what the operator decided on THIS machine - whether the layer is enabled
 * and where its sinks live.
 *
 * WHY THE SETTINGS GOVERN THE SINKS: the layer used to hard-code its sink
 * root and ignore the settings entirely, so the local settings were a note
 * nobody read and `enabled: true` changed nothing. The settings are now the
 * ONLY source of the sink root and of the enabled flag, and the enablement
 * command writes them, so the file the operator edits is the file the
 * exporters obey.
 *
 * WHY AN ABSOLUTE SINK PATH IS REFUSED: an absolute machine path in a
 * persisted artefact is a privacy violation, so a settings file that carries
 * one is treated as carrying no usable sink root and the declared default
 * root is used instead. The same reasoning that makes [path traversal]
 * (https://en.wikipedia.org/wiki/Path_traversal) a security bug in a
 * filesystem API applies here: a configuration file that may name an
 * arbitrary absolute location is a capability nobody needs to hand out.
 *
 * WHY THE READ IS FAIL-OPEN: the signal layer is optional and fail-open by
 * contract, in the [fail-safe](https://en.wikipedia.org/wiki/Fail-safe)
 * sense of the term - the optional subsystem is the one that gives way, not
 * the work it observes. A settings file that cannot be read, cannot be
 * parsed, or names no usable sink MUST NOT silence the layer that is
 * supposed to be optional. Only an EXPLICIT `enabled: false` in a PARSEABLE
 * settings file stops the emission.
 *
 * THE THREE STATES, AND WHAT EACH ONE MEANS:
 *
 * - `OK`: the file is there and it parsed into a plain JSON object. The
 *   parsed fields are the settings, and the missing ones fall back to the
 *   declared defaults.
 * - `MISSING`: there is no settings file on this machine. The declared
 *   defaults apply, and the layer stays enabled.
 * - `CORRUPT`: the file is there but it cannot be trusted - it cannot be
 *   read, it is not JSON, or it parses into something that is not a plain
 *   object (an array or a scalar). The declared defaults apply, and the
 *   layer stays enabled.
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   read[Read logs/telemetry/settings.json]:::base --> present{File present?}:::accent
 *   present -->|No| missing[MISSING, declared defaults apply]:::base
 *   present -->|Yes| parseable{Parses into a plain object?}:::accent
 *   parseable -->|No| corrupt[CORRUPT, declared defaults apply]:::base
 *   parseable -->|Yes| ok[OK, the parsed fields are the settings]:::base
 *
 *   ok --> flag{Explicit enabled false?}:::accent
 *   flag -->|Yes| silent[Layer stays silent]:::base
 *   flag -->|No| live[Layer may emit]:::base
 *
 *   missing --> live
 *   corrupt --> live
 * ```
 *
 * LAYER INDEPENDENCE: this module imports NOTHING from the repo-owned
 * measurement tooling in `scripts/telemetry` and NOTHING from the
 * NEAT-algorithm telemetry tree in `src/neat/telemetry`. The local settings
 * are a LOCAL artefact of the signal layer, not a measurement of a slice.
 *
 * @example
 * import { resolveTelemetryEnabled, resolveTelemetrySinkRoot } from './telemetry.local-settings';
 *
 * resolveTelemetryEnabled('/repo');
 * // => true
 * resolveTelemetrySinkRoot('/repo');
 * // => '/repo/logs/telemetry'
 *
 * @module telemetry.local-settings
 */

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

import { TELEMETRY_SINK_ROOT } from './telemetry.policy';

/**
 * The repo-relative path of the LOCAL settings file.
 *
 * The path is RELATIVE to the repository root on purpose: an absolute
 * machine path is exactly what a persisted artefact must never carry, so
 * even the location of the settings file is expressed relative to the tree
 * it configures.
 *
 * @type {string}
 * @example
 * TELEMETRY_SETTINGS_RELPATH;
 * // => 'logs/telemetry/settings.json'
 */
export const TELEMETRY_SETTINGS_RELPATH = 'logs/telemetry/settings.json';

/**
 * The shape of the local settings the reader understands. Every field is
 * OPTIONAL: a settings file that carries less than this is still readable,
 * and the reader falls back to the declared defaults for what it lacks.
 *
 * `exporter.path` is RELATIVE to the directory of the settings file. An
 * absolute value is refused by the reader rather than honoured.
 *
 * @interface LocalTelemetrySettings
 * @property {number} [schema_version] - The schema version of the settings.
 * @property {boolean} [enabled] - Whether the local signal layer is enabled.
 * @property {Readonly<Record<string, boolean>>} [signals] - The per-signal flags.
 * @property {{ type?: string, path?: string }} [exporter] - The exporter configuration.
 */
export interface LocalTelemetrySettings {
  readonly schema_version?: number;
  readonly enabled?: boolean;
  readonly signals?: Readonly<Record<string, boolean>>;
  readonly exporter?: { readonly type?: string; readonly path?: string };
}

/**
 * The three states of the local settings, mirroring the states of the
 * repo-owned classifier: a settings file that is there and parsable, one
 * that is not there and one that cannot be trusted.
 *
 * @type {'OK' | 'MISSING' | 'CORRUPT'}
 */
export type LocalTelemetrySettingsStatus = 'OK' | 'MISSING' | 'CORRUPT';

/**
 * The result of reading the local settings: the state of the file, the
 * parsed settings when they are usable, and the absolute path the read was
 * made against.
 *
 * @interface LocalTelemetrySettingsRead
 * @property {LocalTelemetrySettingsStatus} status - The state of the settings.
 * @property {LocalTelemetrySettings | null} settings - The parsed settings, or null.
 * @property {string} path - The absolute path of the settings file.
 */
export interface LocalTelemetrySettingsRead {
  readonly status: LocalTelemetrySettingsStatus;
  readonly settings: LocalTelemetrySettings | null;
  readonly path: string;
}

/**
 * Resolves the absolute path of the local settings file for a repository
 * root, without touching the filesystem.
 *
 * @param {string} repositoryRoot - The repository root.
 * @returns {string} The absolute settings path.
 * @example
 * resolveSettingsPath('/repo');
 * // => '/repo/logs/telemetry/settings.json'
 */
export function resolveSettingsPath(repositoryRoot: string): string {
  return path.join(repositoryRoot, ...TELEMETRY_SETTINGS_RELPATH.split('/'));
}

/**
 * Tells whether a parsed value is a plain JSON object, so an ARRAY or a
 * scalar can never be mistaken for a settings object.
 *
 * @param {unknown} value - The parsed value.
 * @returns {boolean} True when the value is a plain object.
 * @example
 * isPlainObject({});
 * // => true
 */
function isPlainObject(value: unknown): boolean {
  if (value === null) return false;
  if (typeof value !== 'object') return false;
  return !Array.isArray(value);
}

/**
 * Reads the local settings and tells the three states apart.
 *
 * The read NEVER throws and NEVER rejects: an unreadable file, a file that
 * is not JSON, and a file that parses into something other than a plain
 * object all resolve to `CORRUPT`, and an absent file resolves to
 * `MISSING`. Callers get a state to reason about, not an exception to
 * handle.
 *
 * @param {string} repositoryRoot - The repository root.
 * @returns {LocalTelemetrySettingsRead} The read result.
 * @example
 * readLocalTelemetrySettings('/repo');
 * // => { status: 'OK', settings: { ... }, path: '/repo/logs/telemetry/settings.json' }
 * readLocalTelemetrySettings('/repo-without-settings');
 * // => { status: 'MISSING', settings: null, path: '/repo-without-settings/logs/telemetry/settings.json' }
 */
export function readLocalTelemetrySettings(
  repositoryRoot: string,
): LocalTelemetrySettingsRead {
  const settingsPath = resolveSettingsPath(repositoryRoot);
  if (!existsSync(settingsPath)) {
    return { status: 'MISSING', settings: null, path: settingsPath };
  }
  let text: string;
  try {
    text = readFileSync(settingsPath, 'utf8');
  } catch {
    return { status: 'CORRUPT', settings: null, path: settingsPath };
  }
  try {
    const parsed: unknown = JSON.parse(text);
    if (!isPlainObject(parsed)) {
      return { status: 'CORRUPT', settings: null, path: settingsPath };
    }
    return {
      status: 'OK',
      settings: parsed as LocalTelemetrySettings,
      path: settingsPath,
    };
  } catch {
    return { status: 'CORRUPT', settings: null, path: settingsPath };
  }
}

/**
 * Resolves the sink root of the local signal layer.
 *
 * The sink root is the `exporter.path` of the local settings, resolved
 * against the directory of the settings file. A settings file that carries
 * no usable relative path - because it is missing, corrupt, carries no
 * exporter or carries an ABSOLUTE path - resolves to the DECLARED default
 * root instead.
 *
 * @param {string} repositoryRoot - The repository root.
 * @returns {string} The absolute sink directory.
 * @example
 * resolveTelemetrySinkRoot('/repo');
 * // => '/repo/logs/telemetry'
 */
export function resolveTelemetrySinkRoot(repositoryRoot: string): string {
  const read = readLocalTelemetrySettings(repositoryRoot);
  const configured = read.settings?.exporter?.path;
  if (read.status === 'OK' && typeof configured === 'string') {
    const relative = configured.trim();
    if (relative.length > 0 && !path.isAbsolute(relative)) {
      return path.resolve(path.dirname(read.path), relative);
    }
  }
  return path.join(repositoryRoot, ...TELEMETRY_SINK_ROOT.split('/'));
}

/**
 * Resolves whether the local signal layer is enabled.
 *
 * ONLY an explicit `enabled: false` in a PARSEABLE settings file disables
 * the layer: a missing or corrupt settings file keeps the declared default,
 * which is ENABLED, because the layer is optional and fail-open.
 *
 * @param {string} repositoryRoot - The repository root.
 * @returns {boolean} True when the layer may emit.
 * @example
 * resolveTelemetryEnabled('/repo');
 * // => true
 */
export function resolveTelemetryEnabled(repositoryRoot: string): boolean {
  const read = readLocalTelemetrySettings(repositoryRoot);
  if (read.status !== 'OK' || read.settings === null) return true;
  return read.settings.enabled !== false;
}
