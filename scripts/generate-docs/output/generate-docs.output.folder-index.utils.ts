/*
 * Builds the generated folder index markdown for docs output.
 *
 * This chapter owns tree construction and recursive index rendering so the
 * output root can stay focused on orchestrating file writes.
 */

import * as path from 'path';

import type {
  DirectorySymbolMap,
  DocsTargetConfig,
  FolderIndexNode,
  GenerateDocsState,
} from '../generate-docs.types.js';
import {
  resolveSortedFolderIndexChildNames,
  resolveVisibleDirectoryFiles,
} from './generate-docs.output.ordering.utils.js';

/**
 * Builds markdown for the generated folder index.
 *
 * @param state - Shared docs-generator state.
 * @param directorySymbolMap - Normalized directory symbol map.
 * @param target - Target configuration.
 * @returns Folder index markdown.
 */
export function buildFolderIndexMarkdown(
  state: GenerateDocsState,
  directorySymbolMap: DirectorySymbolMap,
  target: DocsTargetConfig,
): string {
  const rootNode: FolderIndexNode = {
    name: 'src',
    path: 'src',
    sourceDirectoryPath: target.sourceDir,
    children: new Map(),
    fileCount: 0,
  };

  const relativeDirectories = [...directorySymbolMap.keys()]
    .map((directoryPath) => path.relative(target.sourceDir, directoryPath))
    .filter((relativeDirectory) => !relativeDirectory.startsWith('..'));

  for (const relativeDirectory of relativeDirectories) {
    insertFolderIndexNode(
      state,
      rootNode,
      relativeDirectory,
      target,
      directorySymbolMap,
    );
  }

  const lines = [
    '# Docs Index',
    '',
    'Auto-generated index of source folders (click to open folder README).',
    '',
  ];

  renderFolderIndexNode(state, rootNode, 0, lines);
  return `${lines.join('\n')}\n`;
}

/**
 * Inserts one relative directory into the folder index tree.
 *
 * @param state - Shared docs-generator state.
 * @param rootNode - Root folder index node.
 * @param relativeDirectory - Relative directory from the target source root.
 * @param target - Target configuration.
 * @param directorySymbolMap - Directory symbol map used for file counts.
 * @returns Nothing.
 */
function insertFolderIndexNode(
  state: GenerateDocsState,
  rootNode: FolderIndexNode,
  relativeDirectory: string,
  target: DocsTargetConfig,
  directorySymbolMap: DirectorySymbolMap,
): void {
  const absoluteDirectoryPath = path.join(
    target.sourceDir,
    relativeDirectory === '' ? '' : relativeDirectory,
  );
  const fileSymbolMap = directorySymbolMap.get(absoluteDirectoryPath);

  if (relativeDirectory === '') {
    if (fileSymbolMap) {
      rootNode.fileCount = resolveVisibleDirectoryFiles(
        state,
        absoluteDirectoryPath,
        [...fileSymbolMap.keys()],
      ).length;
    }
    return;
  }

  const pathParts = relativeDirectory.replace(/\\/g, '/').split('/');

  let currentNode = rootNode;
  let accumulatedPath = rootNode.path;
  let accumulatedSourceDirectoryPath = rootNode.sourceDirectoryPath;

  for (const pathPart of pathParts) {
    accumulatedPath = `${accumulatedPath}/${pathPart}`;
    accumulatedSourceDirectoryPath = path.join(
      accumulatedSourceDirectoryPath,
      pathPart,
    );

    if (!currentNode.children.has(pathPart)) {
      currentNode.children.set(pathPart, {
        name: pathPart,
        path: accumulatedPath,
        sourceDirectoryPath: accumulatedSourceDirectoryPath,
        children: new Map(),
      });
    }

    currentNode = currentNode.children.get(pathPart)!;
  }

  if (fileSymbolMap) {
    currentNode.fileCount = resolveVisibleDirectoryFiles(
      state,
      absoluteDirectoryPath,
      [...fileSymbolMap.keys()],
    ).length;
  }
}

/**
 * Renders one folder index node and its children recursively.
 *
 * @param state - Shared docs-generator state.
 * @param node - Current folder index node.
 * @param level - Nesting level.
 * @param lines - Output line buffer.
 * @returns Nothing.
 */
function renderFolderIndexNode(
  state: GenerateDocsState,
  node: FolderIndexNode,
  level: number,
  lines: string[],
): void {
  const indent = '  '.repeat(Math.max(0, level));
  const label = node.path === 'src' && level === 0 ? 'src (root)' : node.name;
  const countSuffix = node.fileCount
    ? ` — ${node.fileCount} file${node.fileCount > 1 ? 's' : ''}`
    : '';

  lines.push(`${indent}- [${label}](${node.path}/README.md)${countSuffix}`);

  const childNames = resolveSortedFolderIndexChildNames(state, node);
  for (const childName of childNames) {
    renderFolderIndexNode(
      state,
      node.children.get(childName)!,
      level + 1,
      lines,
    );
  }
}
