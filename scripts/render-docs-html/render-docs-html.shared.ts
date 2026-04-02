/*
 * Shared path, link, and escaping helpers for generated docs rendering.
 *
 * Navigation, page rendering, and content-link rewriting all depend on the
 * same docs-relative and repo-relative path semantics. Centralizing them here
 * keeps those rules consistent across the folder boundary.
 */

import path from 'path';
import fs from 'fs-extra';

export const DOCS_DIR = path.resolve('docs');

const GITHUB_REPOSITORY_BLOB_BASE_URL =
  'https://github.com/reicek/NeatapticTS/blob/develop';
const GITHUB_REPOSITORY_TREE_BASE_URL =
  'https://github.com/reicek/NeatapticTS/tree/develop';

/** Escapes text for safe HTML insertion. */
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Creates deterministic heading ids and anchor slugs. */
export function slugify(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .replace(/-{2,}/g, '-');
}

/** Builds a docs-relative link between two generated page directories. */
export function buildRelativeDocsHref(
  currentDir: string,
  targetDir: string,
): string {
  const relativeLink =
    path.posix.relative(currentDir || '.', targetDir || '.') || '.';
  return `${relativeLink === '.' ? '.' : relativeLink}/index.html`;
}

/** Checks whether a generated docs page exists or will be emitted in this run. */
export function hasPublishedDocsPage(
  relDir: string,
  generatedPageDirectories: ReadonlySet<string>,
): boolean {
  if (generatedPageDirectories.has(relDir)) {
    return true;
  }

  return fs.existsSync(path.resolve(DOCS_DIR, relDir, 'index.html'));
}

/** Returns true for protocol, protocol-relative, or same-page anchor hrefs. */
export function isExternalOrAnchorHref(href: string): boolean {
  return (
    /^[a-z][a-z0-9+.-]*:/i.test(href) ||
    href.startsWith('//') ||
    href.startsWith('#')
  );
}

/** Normalizes a repo-relative path to portable forward-slash form. */
export function normalizeRepoRelativeHref(href: string): string {
  return href
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\//, '')
    .replace(/^\//, '')
    .replace(/\/{2,}/g, '/');
}

/**
 * Rewrites markdown-generated content links so docs pages link to compiled docs
 * or repository URLs instead of raw unresolved markdown-relative paths.
 */
export function rewriteCompiledDocsContentLinks(
  htmlBody: string,
  currentDir: string,
  generatedPageDirectories: ReadonlySet<string>,
): string {
  return htmlBody.replace(
    /\bhref=(['"])([^'"]+)\1/g,
    (_match, quote: string, href: string) => {
      return `href=${quote}${resolveCompiledDocsContentHref(
        href,
        currentDir,
        generatedPageDirectories,
      )}${quote}`;
    },
  );
}

/** Resolves one content href to a compiled docs page or repository URL. */
function resolveCompiledDocsContentHref(
  href: string,
  currentDir: string,
  generatedPageDirectories: ReadonlySet<string>,
): string {
  if (!href || isExternalOrAnchorHref(href)) {
    return href;
  }

  const docsRelativeTarget = normalizeDocsRelativeTarget(currentDir, href);
  const docsTarget = resolveDocsTargetFromDocsRelativePath(docsRelativeTarget);
  if (
    docsTarget !== undefined &&
    hasPublishedDocsPage(docsTarget, generatedPageDirectories)
  ) {
    return buildRelativeDocsHref(currentDir, docsTarget);
  }

  const explicitRepositoryPath = resolveExplicitRepositoryPath(href);
  if (explicitRepositoryPath) {
    return resolveRepositoryPathHref(
      explicitRepositoryPath,
      currentDir,
      generatedPageDirectories,
    );
  }

  const contextualRepositoryPath = resolveContextualRepositoryPath(
    currentDir,
    href,
  );
  if (contextualRepositoryPath) {
    return resolveRepositoryPathHref(
      contextualRepositoryPath,
      currentDir,
      generatedPageDirectories,
    );
  }

  return href;
}

/** Normalizes a docs-relative target from the current generated page directory. */
function normalizeDocsRelativeTarget(currentDir: string, href: string): string {
  return normalizeRepoRelativeHref(
    path.posix.normalize(path.posix.join(currentDir || '.', href)),
  );
}

/** Resolves a docs page directory from a docs-relative markdown target. */
function resolveDocsTargetFromDocsRelativePath(
  docsRelativePath: string,
): string | undefined {
  const normalizedPath = docsRelativePath.replace(/\/$/, '');
  if (normalizedPath === 'README.md') {
    return '';
  }

  if (normalizedPath.endsWith('/README.md')) {
    return normalizedPath.slice(0, -'/README.md'.length);
  }

  if (normalizedPath.endsWith('/index.html')) {
    return normalizedPath.slice(0, -'/index.html'.length);
  }

  if (normalizedPath === 'index.html') {
    return '';
  }

  return normalizedPath.endsWith('.md') || normalizedPath.endsWith('.html')
    ? undefined
    : normalizedPath;
}

/** Resolves repo-root hrefs that are already expressed as repository paths. */
function resolveExplicitRepositoryPath(href: string): string | undefined {
  const normalizedHref = normalizeRepoRelativeHref(href);
  if (!normalizedHref) {
    return undefined;
  }

  if (
    normalizedHref === 'docs' ||
    normalizedHref.startsWith('docs/') ||
    normalizedHref === 'src' ||
    normalizedHref.startsWith('src/') ||
    normalizedHref === 'examples' ||
    normalizedHref.startsWith('examples/')
  ) {
    return normalizedHref;
  }

  return undefined;
}

/**
 * Resolves repo-relative source paths for content links that are expressed
 * relative to the current generated docs page.
 */
function resolveContextualRepositoryPath(
  currentDir: string,
  href: string,
): string | undefined {
  const currentSourceDirectory = resolveCurrentSourceDirectory(currentDir);
  if (!currentSourceDirectory) {
    return undefined;
  }

  return normalizeRepoRelativeHref(
    path.posix.normalize(path.posix.join(currentSourceDirectory, href)),
  );
}

/** Maps the current generated docs page back to its source directory. */
function resolveCurrentSourceDirectory(currentDir: string): string | undefined {
  if (currentDir === '') {
    return 'src';
  }

  const exampleDocsMatch = /^examples\/([^/]+)\/docs(?:\/(.*))?$/.exec(
    currentDir,
  );
  if (exampleDocsMatch) {
    const [, exampleDirName, subpath = ''] = exampleDocsMatch;
    return normalizeRepoRelativeHref(
      path.posix.join('examples', exampleDirName, subpath),
    );
  }

  if (currentDir === 'examples') {
    return 'examples';
  }

  if (currentDir.startsWith('examples/')) {
    return normalizeRepoRelativeHref(currentDir);
  }

  return normalizeRepoRelativeHref(path.posix.join('src', currentDir));
}

/** Resolves a repo path to a compiled docs href or GitHub repository URL. */
function resolveRepositoryPathHref(
  repositoryPath: string,
  currentDir: string,
  generatedPageDirectories: ReadonlySet<string>,
): string {
  const docsTarget =
    resolvePublishedDocsTargetFromRepositoryPath(repositoryPath);
  if (
    docsTarget !== undefined &&
    hasPublishedDocsPage(docsTarget, generatedPageDirectories)
  ) {
    return buildRelativeDocsHref(currentDir, docsTarget);
  }

  return buildGitHubRepositoryHref(repositoryPath);
}

/** Maps a repo path to the matching generated docs page directory when one exists. */
function resolvePublishedDocsTargetFromRepositoryPath(
  repositoryPath: string,
): string | undefined {
  const normalizedPath = normalizeRepoRelativeHref(repositoryPath).replace(
    /\/$/,
    '',
  );

  if (normalizedPath === 'docs' || normalizedPath === 'docs/index.html') {
    return '';
  }

  if (normalizedPath.startsWith('docs/')) {
    return normalizedPath.slice('docs/'.length).replace(/\/index\.html$/, '');
  }

  if (
    normalizedPath === 'src' ||
    normalizedPath === 'src/' ||
    normalizedPath === 'src/README.md'
  ) {
    return '';
  }

  if (
    normalizedPath.startsWith('src/') &&
    normalizedPath.endsWith('/README.md')
  ) {
    return normalizedPath.slice('src/'.length, -'/README.md'.length);
  }

  if (
    normalizedPath === 'examples' ||
    normalizedPath === 'examples/' ||
    normalizedPath === 'examples/README.md'
  ) {
    return 'examples';
  }

  const exampleReadmeMatch = /^examples\/([^/]+)\/README\.md$/.exec(normalizedPath);
  if (exampleReadmeMatch) {
    return `examples/${exampleReadmeMatch[1]}/docs`;
  }

  const nestedExampleReadmeMatch =
    /^examples\/([^/]+)\/(.+)\/README\.md$/.exec(normalizedPath);
  if (nestedExampleReadmeMatch) {
    return `examples/${nestedExampleReadmeMatch[1]}/docs/${nestedExampleReadmeMatch[2]}`;
  }

  return undefined;
}

/** Builds a GitHub blob or tree link for a repo-relative path. */
function buildGitHubRepositoryHref(repoRelativePath: string): string {
  const normalizedPath = normalizeRepoRelativeHref(repoRelativePath).replace(
    /\/$/,
    '',
  );
  const absoluteRepositoryPath = path.resolve(normalizedPath);
  const pathExistsInRepository = fs.existsSync(absoluteRepositoryPath);
  const urlBase =
    pathExistsInRepository && fs.statSync(absoluteRepositoryPath).isFile()
      ? GITHUB_REPOSITORY_BLOB_BASE_URL
      : path.posix.extname(normalizedPath)
        ? GITHUB_REPOSITORY_BLOB_BASE_URL
        : GITHUB_REPOSITORY_TREE_BASE_URL;

  return `${urlBase}/${normalizedPath}`;
}
