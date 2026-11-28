/**
 * GREEN home for the LOCAL telemetry settings reader (slice OT-V2-3 of
 * `plans/opentelemetry-slice-telemetry.plans.md`).
 *
 * WHAT THIS HOME DEMANDS: the local settings file
 * `logs/telemetry/settings.json` is the file the local enablement command
 * writes and the file the exporters OBEY, so the reader is proved over the
 * REAL module, against REAL settings files on disk:
 *
 *   - the three states are told apart: a settings file that is there and
 *     parsable, one that is not there, and one that cannot be trusted -
 *     unparsable JSON, a JSON array, a JSON scalar and a file that cannot
 *     be read at all;
 *   - the sink root is the `exporter.path` of the settings, resolved
 *     against the DIRECTORY of the settings file;
 *   - an ABSOLUTE `exporter.path` is REFUSED, because an absolute machine
 *     path in a persisted artefact is a privacy violation, and the declared
 *     default root is used instead;
 *   - an empty or non-string `exporter.path` falls back to the declared
 *     default root as well;
 *   - only an EXPLICIT `enabled: false` in a PARSEABLE settings file
 *     disables the layer: a missing or corrupt settings file keeps the
 *     fail-open default, which is ENABLED.
 *
 * WHY EVERY SETTINGS FILE IS A REAL FILE ON DISK: a reader that resolved a
 * mocked settings object would not prove that a file nobody can parse is
 * treated as a file nobody can trust.
 *
 * @see plans/opentelemetry-slice-telemetry.plans.md - "Vertical slice A"
 */

import {
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import * as path from 'node:path';

import {
  TELEMETRY_SETTINGS_RELPATH,
  readLocalTelemetrySettings,
  resolveSettingsPath,
  resolveTelemetryEnabled,
  resolveTelemetrySinkRoot,
} from './telemetry.local-settings';

/**
 * Creates a throwaway repository root whose settings directory exists but
 * whose settings file does NOT.
 *
 * @returns The absolute path of the repository root.
 */
function createTempRepositoryRoot(): string {
  const root = mkdtempSync(path.join(tmpdir(), 'telemetry-local-settings-'));
  mkdirSync(path.join(root, 'logs', 'telemetry'), { recursive: true });
  return root;
}

/**
 * Writes the raw bytes of a settings file into a throwaway repository root.
 *
 * @param root The repository root.
 * @param content The raw file content.
 * @returns The absolute path of the settings file.
 */
function writeSettings(root: string, content: string): string {
  const settingsPath = path.join(root, ...TELEMETRY_SETTINGS_RELPATH.split('/'));
  writeFileSync(settingsPath, content, 'utf8');
  return settingsPath;
}

/**
 * The POSIX-style form of a path, so the lanes compare the SAME shape on
 * every platform.
 *
 * @param value The path.
 * @returns The path with forward slashes.
 */
function toPosix(value: string): string {
  return value.replace(/\\/g, '/');
}

describe('OT-V2-3: the local telemetry settings reader', () => {
  const roots: string[] = [];

  afterEach(() => {
    while (roots.length > 0) {
      const root = roots.pop();
      if (root) rmSync(root, { recursive: true, force: true });
    }
  });

  /**
   * Creates a throwaway repository root and remembers it for the cleanup.
   *
   * @returns The absolute path of the repository root.
   */
  function newRoot(): string {
    const root = createTempRepositoryRoot();
    roots.push(root);
    return root;
  }

  describe('where the settings live', () => {
    it('reads them from the telemetry sink root of the repository root', () => {
      const root = newRoot();
      expect(toPosix(resolveSettingsPath(root))).toBe(
        `${toPosix(root)}/logs/telemetry/settings.json`,
      );
    });
  });

  describe('the three states of the settings', () => {
    it('reports MISSING when there is no settings file', () => {
      const root = newRoot();
      const read = readLocalTelemetrySettings(root);
      expect(read.status).toBe('MISSING');
      expect(read.settings).toBeNull();
      expect(toPosix(read.path)).toBe(
        `${toPosix(root)}/logs/telemetry/settings.json`,
      );
    });

    it('reports OK and the parsed settings when the file parses', () => {
      const root = newRoot();
      writeSettings(
        root,
        JSON.stringify({ schema_version: 1, enabled: true, exporter: { path: '.' } }),
      );
      const read = readLocalTelemetrySettings(root);
      expect(read.status).toBe('OK');
      expect(read.settings).toEqual({
        schema_version: 1,
        enabled: true,
        exporter: { path: '.' },
      });
    });

    it('reports CORRUPT when the file is not JSON at all', () => {
      const root = newRoot();
      writeSettings(root, '{ this is not json');
      const read = readLocalTelemetrySettings(root);
      expect(read.status).toBe('CORRUPT');
      expect(read.settings).toBeNull();
    });

    it('reports CORRUPT when the file holds a JSON array', () => {
      const root = newRoot();
      writeSettings(root, '[1, 2, 3]');
      expect(readLocalTelemetrySettings(root).status).toBe('CORRUPT');
    });

    it('reports CORRUPT when the file holds a JSON null', () => {
      const root = newRoot();
      writeSettings(root, 'null');
      expect(readLocalTelemetrySettings(root).status).toBe('CORRUPT');
    });

    it('reports CORRUPT when the file holds a JSON scalar', () => {
      const root = newRoot();
      writeSettings(root, '42');
      expect(readLocalTelemetrySettings(root).status).toBe('CORRUPT');
    });

    it('reports CORRUPT when the settings file cannot be read', () => {
      const root = newRoot();
      const settingsPath = path.join(root, ...TELEMETRY_SETTINGS_RELPATH.split('/'));
      mkdirSync(settingsPath, { recursive: true });
      const read = readLocalTelemetrySettings(root);
      expect(read.status).toBe('CORRUPT');
      expect(read.settings).toBeNull();
    });
  });

  describe('the sink root the settings name', () => {
    it('resolves the relative exporter path against the settings directory', () => {
      const root = newRoot();
      writeSettings(
        root,
        JSON.stringify({ exporter: { path: 'sinks' } }),
      );
      expect(toPosix(resolveTelemetrySinkRoot(root))).toBe(
        `${toPosix(root)}/logs/telemetry/sinks`,
      );
    });

    it('resolves the canonical dot path to the settings directory itself', () => {
      const root = newRoot();
      writeSettings(root, JSON.stringify({ exporter: { path: '.' } }));
      expect(toPosix(resolveTelemetrySinkRoot(root))).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });

    it('REFUSES an absolute exporter path and uses the declared default root', () => {
      const root = newRoot();
      writeSettings(
        root,
        JSON.stringify({ exporter: { path: path.join(root, 'elsewhere') } }),
      );
      expect(toPosix(resolveTelemetrySinkRoot(root))).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });

    it('uses the declared default root when the exporter path is empty', () => {
      const root = newRoot();
      writeSettings(root, JSON.stringify({ exporter: { path: '   ' } }));
      expect(toPosix(resolveTelemetrySinkRoot(root))).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });

    it('uses the declared default root when the exporter path is not a string', () => {
      const root = newRoot();
      writeSettings(root, JSON.stringify({ exporter: { path: 42 } }));
      expect(toPosix(resolveTelemetrySinkRoot(root))).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });

    it('uses the declared default root when the settings are missing', () => {
      const root = newRoot();
      expect(toPosix(resolveTelemetrySinkRoot(root))).toBe(
        `${toPosix(root)}/logs/telemetry`,
      );
    });
  });

  describe('whether the layer is enabled', () => {
    it('is enabled when the settings say so explicitly', () => {
      const root = newRoot();
      writeSettings(root, JSON.stringify({ enabled: true }));
      expect(resolveTelemetryEnabled(root)).toBe(true);
    });

    it('is DISABLED when the settings say enabled false', () => {
      const root = newRoot();
      writeSettings(root, JSON.stringify({ enabled: false }));
      expect(resolveTelemetryEnabled(root)).toBe(false);
    });

    it('is enabled when the settings carry no enabled flag at all', () => {
      const root = newRoot();
      writeSettings(root, JSON.stringify({ schema_version: 1 }));
      expect(resolveTelemetryEnabled(root)).toBe(true);
    });

    it('is enabled when the settings are missing, because the layer fails open', () => {
      const root = newRoot();
      expect(resolveTelemetryEnabled(root)).toBe(true);
    });

    it('is enabled when the settings cannot be trusted, because the layer fails open', () => {
      const root = newRoot();
      writeSettings(root, 'not json');
      expect(resolveTelemetryEnabled(root)).toBe(true);
    });
  });
});
