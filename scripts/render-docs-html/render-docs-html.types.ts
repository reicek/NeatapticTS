/*
 * Shared contracts for the HTML docs renderer boundary.
 *
 * The renderer now spans page discovery, Mermaid validation, navigation, and
 * asset bootstrapping. These types keep the public chapter interfaces aligned
 * without pushing low-level implementation details back into the root script.
 */

/** Markdown page discovered under the generated docs tree. */
export interface PageMeta {
  abs: string;
  relDir: string;
  title: string;
  markdown: string;
}

/** One Mermaid fenced block collected from a README surface. */
export interface MermaidBlockReference {
  readmePath: string;
  blockNumber: number;
  diagram: string;
}

interface SidebarRenderHelpers {
  generatedPageDirectories: ReadonlySet<string>;
  hasPublishedDocsPage: (
    relDir: string,
    generatedPageDirectories: ReadonlySet<string>,
  ) => boolean;
  buildRelativeDocsHref: (currentDir: string, targetDir: string) => string;
  escapeHtml: (value: string) => string;
}

/** Shared render context for sidebar fragments. */
export interface SidebarRenderContext extends SidebarRenderHelpers {
  currentDir: string;
}

/** Render context for the full docs sidebar. */
export interface DocsSidebarRenderContext extends SidebarRenderContext {
  pages: readonly PageMeta[];
}
