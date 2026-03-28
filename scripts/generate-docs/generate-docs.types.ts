/*
 * Shared contracts for the folder-based docs generator.
 *
 * The split keeps traversal, ordering, and markdown emission in separate
 * modules, so these interfaces act as the small vocabulary that lets those
 * chapters collaborate without reaching into each other's implementation.
 */

import type { Project } from 'ts-morph';

/**
 * Configures one docs-generation target.
 */
export interface DocsTargetConfig {
  name: string;
  sourceDir: string;
  docsDir: string;
  rootDocsDir: string;
  rootReadmeSource?: string;
  rootReadmeDestination?: string;
  excludeRootSourceFiles?: boolean;
  includeFolderIndex?: boolean;
  publishedRootDir?: string;
  preservePublishedEntries?: readonly string[];
}

/**
 * Renders one documented parameter description for README output.
 */
export interface RenderedParameter {
  name: string;
  type?: string;
  doc?: string;
}

/**
 * Stores the normalized docs-ready view of one exported or documented symbol.
 */
export interface RenderedSymbol {
  kind: string;
  name: string;
  filePath: string;
  parent?: string;
  signature?: string;
  jsdoc: {
    summary?: string;
    description?: string;
    params?: RenderedParameter[];
    returns?: string;
    deprecated?: string;
    examples?: string[];
  };
}

/**
 * Represents one folder inside the generated docs index tree.
 */
export interface FolderIndexNode {
  name: string;
  path: string;
  sourceDirectoryPath: string;
  children: Map<string, FolderIndexNode>;
  fileCount?: number;
}

/**
 * Captures the supported `docs.order.json` controls for one directory.
 */
export interface DirectoryDocsOrderConfig {
  introFile?: string;
  fileOrder?: string[];
  symbolOrder?: Record<string, string[]>;
  folderOrder?: string[];
  hiddenFiles?: string[];
  hiddenSymbols?: string[];
}

/**
 * Stores one validated directory-order config plus its source path.
 */
export interface LoadedDirectoryDocsOrderConfig {
  configPath: string;
  config: DirectoryDocsOrderConfig;
}

/**
 * Represents the promoted file-summary content for a source file.
 */
export interface RenderedFileSummary {
  description?: string;
  examples?: string[];
}

/**
 * Groups rendered symbols by absolute directory path, then by absolute file path.
 */
export type DirectorySymbolMap = Map<string, Map<string, RenderedSymbol[]>>;

/**
 * Owns mutable state shared across one docs-generation run.
 */
export interface GenerateDocsState {
  project: Project;
  resolvedDirectoryDocsOrderConfigCache: Map<
    string,
    LoadedDirectoryDocsOrderConfig | undefined
  >;
  directoryDocsOrderConfigCache: Map<
    string,
    Promise<LoadedDirectoryDocsOrderConfig | undefined>
  >;
}
