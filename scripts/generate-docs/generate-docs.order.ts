/*
 * Owns `docs.order.json` loading, validation, and cached lookup.
 *
 * Rendering code only needs a validated config view; this chapter keeps JSON
 * parsing, warning messages, and cache lifecycle separate from markdown logic.
 */

import fs from 'fs-extra';
import * as path from 'path';

import {
  DOCS_ORDER_CONFIG_FILE_NAME,
  DOCS_ORDER_SUPPORTED_KEYS,
  WORKSPACE_ROOT_DIR,
} from './generate-docs.constants.js';
import type {
  DirectoryDocsOrderConfig,
  GenerateDocsState,
  LoadedDirectoryDocsOrderConfig,
} from './generate-docs.types.js';

/**
 * Loads and caches the optional per-folder docs ordering config.
 *
 * Phase 1 only validates and stores config so later rendering steps can reuse
 * the same validated data without re-reading the file.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path that may contain docs config.
 * @returns Parsed and validated config when present.
 */
export async function loadDirectoryDocsOrderConfig(
  state: GenerateDocsState,
  directoryPath: string,
): Promise<LoadedDirectoryDocsOrderConfig | undefined> {
  const normalizedDirectoryPath = path.resolve(directoryPath);
  if (state.directoryDocsOrderConfigCache.has(normalizedDirectoryPath)) {
    return state.directoryDocsOrderConfigCache.get(normalizedDirectoryPath)!;
  }

  const pendingConfig = readDirectoryDocsOrderConfig(
    normalizedDirectoryPath,
  ).then((loadedConfig) => {
    state.resolvedDirectoryDocsOrderConfigCache.set(
      normalizedDirectoryPath,
      loadedConfig,
    );
    return loadedConfig;
  });

  state.directoryDocsOrderConfigCache.set(
    normalizedDirectoryPath,
    pendingConfig,
  );
  return pendingConfig;
}

/**
 * Returns the cached docs ordering config for a directory.
 *
 * @param state - Shared docs-generator state.
 * @param directoryPath - Absolute directory path.
 * @returns Cached config when already loaded.
 */
export function getCachedDirectoryDocsOrderConfig(
  state: GenerateDocsState,
  directoryPath: string,
): LoadedDirectoryDocsOrderConfig | undefined {
  return state.resolvedDirectoryDocsOrderConfigCache.get(
    path.resolve(directoryPath),
  );
}

/**
 * Writes a scoped warning for one docs ordering config file.
 *
 * @param configPath - Absolute config path.
 * @param message - Warning message.
 * @returns Nothing.
 */
export function warnDirectoryDocsOrderConfig(
  configPath: string,
  message: string,
): void {
  const relativeConfigPath = path
    .relative(WORKSPACE_ROOT_DIR, configPath)
    .replace(/\\/g, '/');
  console.warn(`[docs] ${relativeConfigPath}: ${message}`);
}

/**
 * Reads and validates one directory's `docs.order.json` file when present.
 *
 * @param directoryPath - Absolute directory path.
 * @returns Parsed config or undefined when absent or invalid.
 */
async function readDirectoryDocsOrderConfig(
  directoryPath: string,
): Promise<LoadedDirectoryDocsOrderConfig | undefined> {
  const configPath = path.join(directoryPath, DOCS_ORDER_CONFIG_FILE_NAME);
  if (!(await fs.pathExists(configPath))) {
    return undefined;
  }

  try {
    const rawConfigText = await fs.readFile(configPath, 'utf8');
    const parsedConfig: unknown = JSON.parse(rawConfigText);
    const validatedConfig = validateDirectoryDocsOrderConfig(
      parsedConfig,
      configPath,
    );
    return validatedConfig
      ? { configPath, config: validatedConfig }
      : undefined;
  } catch (error) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Failed to read ${DOCS_ORDER_CONFIG_FILE_NAME}: ${getErrorMessage(error)}`,
    );
    return undefined;
  }
}

/**
 * Validates the supported phase-1 docs ordering config keys.
 *
 * @param parsedConfig - Raw parsed JSON value.
 * @param configPath - Absolute config path used in warnings.
 * @returns Sanitized config when at least one supported value is valid.
 */
function validateDirectoryDocsOrderConfig(
  parsedConfig: unknown,
  configPath: string,
): DirectoryDocsOrderConfig | undefined {
  if (!isRecord(parsedConfig)) {
    warnDirectoryDocsOrderConfig(
      configPath,
      'Config must be a JSON object. Ignoring file.',
    );
    return undefined;
  }

  for (const configKey of Object.keys(parsedConfig)) {
    if (!DOCS_ORDER_SUPPORTED_KEYS.includes(configKey as never)) {
      warnDirectoryDocsOrderConfig(
        configPath,
        `Unknown key "${configKey}". Supported keys: ${DOCS_ORDER_SUPPORTED_KEYS.join(', ')}`,
      );
    }
  }

  const validatedConfig: DirectoryDocsOrderConfig = {};
  const introFile = normalizeOptionalStringConfigValue(
    parsedConfig.introFile,
    'introFile',
    configPath,
  );
  if (introFile) {
    validatedConfig.introFile = introFile;
  }

  const fileOrder = normalizeStringArrayConfigValue(
    parsedConfig.fileOrder,
    'fileOrder',
    configPath,
  );
  if (fileOrder) {
    validatedConfig.fileOrder = fileOrder;
  }

  const symbolOrder = normalizeSymbolOrderConfigValue(
    parsedConfig.symbolOrder,
    configPath,
  );
  if (symbolOrder) {
    validatedConfig.symbolOrder = symbolOrder;
  }

  const folderOrder = normalizeStringArrayConfigValue(
    parsedConfig.folderOrder,
    'folderOrder',
    configPath,
  );
  if (folderOrder) {
    validatedConfig.folderOrder = folderOrder;
  }

  const hiddenFiles = normalizeStringArrayConfigValue(
    parsedConfig.hiddenFiles,
    'hiddenFiles',
    configPath,
  );
  if (hiddenFiles) {
    validatedConfig.hiddenFiles = hiddenFiles;
  }

  const hiddenSymbols = normalizeStringArrayConfigValue(
    parsedConfig.hiddenSymbols,
    'hiddenSymbols',
    configPath,
  );
  if (hiddenSymbols) {
    validatedConfig.hiddenSymbols = hiddenSymbols;
  }

  return Object.keys(validatedConfig).length > 0 ? validatedConfig : {};
}

/**
 * Normalizes a single optional string config field.
 *
 * @param value - Raw config value.
 * @param fieldName - Config field name.
 * @param configPath - Absolute config path used in warnings.
 * @returns Trimmed string when valid.
 */
function normalizeOptionalStringConfigValue(
  value: unknown,
  fieldName: string,
  configPath: string,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (typeof value !== 'string' || value.trim().length === 0) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Field "${fieldName}" must be a non-empty string. Ignoring value.`,
    );
    return undefined;
  }

  return value.trim();
}

/**
 * Normalizes one string-array config field.
 *
 * @param value - Raw config value.
 * @param fieldName - Config field name.
 * @param configPath - Absolute config path used in warnings.
 * @returns Unique non-empty strings when valid.
 */
function normalizeStringArrayConfigValue(
  value: unknown,
  fieldName: string,
  configPath: string,
): string[] | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (!Array.isArray(value)) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Field "${fieldName}" must be an array of non-empty strings. Ignoring value.`,
    );
    return undefined;
  }

  const normalizedEntries = value
    .filter((entry): entry is string => typeof entry === 'string')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);

  if (normalizedEntries.length !== value.length) {
    warnDirectoryDocsOrderConfig(
      configPath,
      `Field "${fieldName}" must contain only non-empty strings. Dropping invalid entries.`,
    );
  }

  return normalizedEntries.length > 0
    ? [...new Set(normalizedEntries)]
    : undefined;
}

/**
 * Normalizes the per-file symbol ordering map.
 *
 * @param value - Raw config value.
 * @param configPath - Absolute config path used in warnings.
 * @returns Sanitized symbol-order map when valid.
 */
function normalizeSymbolOrderConfigValue(
  value: unknown,
  configPath: string,
): Record<string, string[]> | undefined {
  if (value === undefined) {
    return undefined;
  }

  if (!isRecord(value)) {
    warnDirectoryDocsOrderConfig(
      configPath,
      'Field "symbolOrder" must be an object keyed by file name. Ignoring value.',
    );
    return undefined;
  }

  const normalizedSymbolOrderEntries = Object.entries(value)
    .map(([fileName, symbolNames]) => {
      const normalizedFileName = fileName.trim();
      if (normalizedFileName.length === 0) {
        warnDirectoryDocsOrderConfig(
          configPath,
          'Field "symbolOrder" contains an empty file key. Dropping entry.',
        );
        return undefined;
      }

      const normalizedSymbols = normalizeStringArrayConfigValue(
        symbolNames,
        `symbolOrder.${normalizedFileName}`,
        configPath,
      );
      if (!normalizedSymbols) {
        return undefined;
      }

      return [normalizedFileName, normalizedSymbols] as const;
    })
    .filter(
      (entry): entry is readonly [string, string[]] => entry !== undefined,
    );

  return normalizedSymbolOrderEntries.length > 0
    ? Object.fromEntries(normalizedSymbolOrderEntries)
    : undefined;
}

/**
 * Narrows unknown values to simple object records.
 *
 * @param value - Value to inspect.
 * @returns True when the value is a plain record-like object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Extracts a readable message from unknown thrown values.
 *
 * @param error - Thrown value.
 * @returns Human-readable error text.
 */
function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}
