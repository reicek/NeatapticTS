/*
 * Public output boundary for the folderized docs generator.
 *
 * This root keeps the exported emitters orchestration-first while narrower
 * helper chapters inside the folder own README composition, folder-index
 * rendering, ordering policy, and config warning details.
 */

import fs from 'fs-extra';
import * as path from 'path';

import {
  DOCS_DIR,
  FOLDER_INDEX_FILE_NAME,
} from '../generate-docs.constants.js';
import { loadDirectoryDocsOrderConfig } from '../generate-docs.order.js';
import type {
  DirectorySymbolMap,
  DocsTargetConfig,
  GenerateDocsState,
} from '../generate-docs.types.js';
import { buildFolderIndexMarkdown } from './generate-docs.output.folder-index.utils.js';
import { buildDirectoryReadme } from './generate-docs.output.readme.utils.js';

/**
 * Emits directory README files to both docs output and the source tree.
 *
 * @param state - Shared docs-generator state.
 * @param directorySymbolMap - Normalized directory symbol map.
 * @param target - Target configuration.
 * @returns Nothing.
 */
export async function emitDirectoryDocs(
  state: GenerateDocsState,
  directorySymbolMap: DirectorySymbolMap,
  target: DocsTargetConfig,
): Promise<void> {
  for (const [directoryPath, fileSymbolMap] of directorySymbolMap) {
    const relativeDirectory = path.relative(target.sourceDir, directoryPath);
    if (relativeDirectory.startsWith('..')) {
      continue;
    }

    const outputDirectory =
      relativeDirectory === ''
        ? target.rootDocsDir
        : path.join(target.docsDir, relativeDirectory);
    const sourceDirectory =
      relativeDirectory === ''
        ? target.sourceDir
        : path.join(target.sourceDir, relativeDirectory);

    await loadDirectoryDocsOrderConfig(state, sourceDirectory);

    const markdown = buildDirectoryReadme(
      state,
      relativeDirectory,
      fileSymbolMap,
      target.sourceDir,
    );

    await fs.ensureDir(outputDirectory);
    await writeFileIfChanged(path.join(outputDirectory, 'README.md'), markdown);
    await writeFileIfChanged(path.join(sourceDirectory, 'README.md'), markdown);
  }
}

/**
 * Emits the folder index for targets that want a docs landing page.
 *
 * @param state - Shared docs-generator state.
 * @param directorySymbolMap - Normalized directory symbol map.
 * @param target - Target configuration.
 * @returns Nothing.
 */
export async function emitFolderIndex(
  state: GenerateDocsState,
  directorySymbolMap: DirectorySymbolMap,
  target: DocsTargetConfig,
): Promise<void> {
  const relativeDirectories = [...directorySymbolMap.keys()]
    .map((directoryPath) => path.relative(target.sourceDir, directoryPath))
    .filter((relativeDirectory) => !relativeDirectory.startsWith('..'));

  const directoriesWithConfigSupport = [
    target.sourceDir,
    ...relativeDirectories.map((relativeDirectory) =>
      path.join(target.sourceDir, relativeDirectory),
    ),
  ];
  await Promise.all(
    [...new Set(directoriesWithConfigSupport)].map((directoryPath) =>
      loadDirectoryDocsOrderConfig(state, directoryPath),
    ),
  );

  const markdown = buildFolderIndexMarkdown(state, directorySymbolMap, target);
  await writeFileIfChanged(
    path.join(DOCS_DIR, FOLDER_INDEX_FILE_NAME),
    markdown,
  );
}

/**
 * Writes a file only when its content changed.
 *
 * @param filePath - Output file path.
 * @param content - Desired file content.
 * @returns Nothing.
 */
async function writeFileIfChanged(
  filePath: string,
  content: string,
): Promise<void> {
  if (await fs.pathExists(filePath)) {
    const previousContent = await fs.readFile(filePath, 'utf8');
    if (previousContent === content) {
      return;
    }
  }

  await fs.writeFile(filePath, content, 'utf8');
}
