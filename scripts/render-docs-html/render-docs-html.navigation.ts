/*
 * Owns the generated docs sidebar contract.
 *
 * This chapter turns the published docs page set into the left-nav learning
 * path: examples first for onboarding, grouped docs pages for reference
 * reading, and a depth-aware rail that remains visually continuous.
 */

import type {
  DocsSidebarRenderContext,
  PageMeta,
  SidebarRenderContext,
} from './render-docs-html.types.js';

interface ExampleNavLink {
  dir: string;
  label: string;
}

interface ExampleNavDemo {
  dir: string;
  label: string;
  eyebrow: string;
  description: string;
  readingOrder: readonly ExampleNavLink[];
  deepDives: readonly ExampleNavLink[];
}

interface PipeNavEntry {
  label: string;
  targetDir: string;
  depth: number;
  isCurrent: boolean;
  cssClass?: string;
}

const EXAMPLES_OVERVIEW_DIR = 'examples';
const SIDEBAR_GROUP_ORDER = [
  'root',
  'architecture',
  'methods',
  'neat',
  'multithreading',
] as const;

const EXAMPLE_DEMOS: readonly ExampleNavDemo[] = [
  {
    dir: 'examples/flappy_bird',
    label: 'Flappy Bird',
    eyebrow: 'Fast control and playback',
    description:
      'Start here if you want the clearest end-to-end tour through evaluation fairness, deterministic simulation, worker playback, and browser inspection.',
    readingOrder: [
      { dir: 'examples/flappy_bird/docs', label: 'Docs overview' },
      { dir: 'examples/flappy_bird/docs/trainer', label: 'Trainer' },
      { dir: 'examples/flappy_bird/docs/evaluation', label: 'Evaluation' },
      { dir: 'examples/flappy_bird/docs/environment', label: 'Environment' },
      {
        dir: 'examples/flappy_bird/docs/simulation-shared',
        label: 'Simulation shared',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry',
        label: 'Browser entry',
      },
      {
        dir: 'examples/flappy_bird/docs/flappy-evolution-worker',
        label: 'Evolution worker',
      },
      { dir: 'examples/flappy_bird', label: 'Open browser demo' },
    ],
    deepDives: [
      { dir: 'examples/flappy_bird/docs/constants', label: 'Constants' },
      {
        dir: 'examples/flappy_bird/docs/evaluation/rollout',
        label: 'Evaluation rollout',
      },
      {
        dir: 'examples/flappy_bird/docs/simulation-shared/observation',
        label: 'Observation model',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/host',
        label: 'Host shell',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/host/resize',
        label: 'Resize behavior',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/runtime',
        label: 'Browser runtime',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/network-view',
        label: 'Network view',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/visualization',
        label: 'Visualization',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/worker-channel',
        label: 'Worker channel',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback',
        label: 'Playback orchestration',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback/background',
        label: 'Playback background',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback/background/ground-grid',
        label: 'Ground grid',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback/frame-render',
        label: 'Frame render',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback/snapshot',
        label: 'Snapshot decoding',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback/trail',
        label: 'Bird trail',
      },
      {
        dir: 'examples/flappy_bird/docs/browser-entry/playback/worker-channel',
        label: 'Playback worker channel',
      },
    ],
  },
  {
    dir: 'examples/asciiMaze',
    label: 'ASCII Maze',
    eyebrow: 'Compact policy and shaping',
    description:
      'Read this after Flappy Bird if you want the counterpoint: tiny observations, reward shaping, curriculum transfer, and telemetry-first search.',
    readingOrder: [
      { dir: 'examples/asciiMaze/docs', label: 'Docs overview' },
      {
        dir: 'examples/asciiMaze/docs/evolutionEngine',
        label: 'Evolution engine',
      },
      {
        dir: 'examples/asciiMaze/docs/mazeMovement',
        label: 'Maze movement',
      },
      {
        dir: 'examples/asciiMaze/docs/dashboardManager',
        label: 'Dashboard manager',
      },
      {
        dir: 'examples/asciiMaze/docs/browser-entry',
        label: 'Browser entry',
      },
      { dir: 'examples/asciiMaze', label: 'Open browser demo' },
    ],
    deepDives: [
      {
        dir: 'examples/asciiMaze/docs/mazeMovement/policy',
        label: 'Movement policy',
      },
      {
        dir: 'examples/asciiMaze/docs/mazeMovement/runtime',
        label: 'Movement runtime',
      },
      {
        dir: 'examples/asciiMaze/docs/mazeMovement/shaping',
        label: 'Movement shaping',
      },
      {
        dir: 'examples/asciiMaze/docs/mazeMovement/finalization',
        label: 'Episode finalization',
      },
      {
        dir: 'examples/asciiMaze/docs/dashboardManager/live',
        label: 'Live dashboard',
      },
      {
        dir: 'examples/asciiMaze/docs/dashboardManager/archive',
        label: 'Archive dashboard',
      },
      {
        dir: 'examples/asciiMaze/docs/dashboardManager/telemetry',
        label: 'Telemetry surfaces',
      },
    ],
  },
];

/** Renders the compact examples list used on the root-page table of contents. */
export function buildExamplesTocLinksHtml(
  context: SidebarRenderContext,
): string {
  const entries: PipeNavEntry[] = [];

  if (
    context.hasPublishedDocsPage(
      EXAMPLES_OVERVIEW_DIR,
      context.generatedPageDirectories,
    )
  ) {
    entries.push({
      label: 'Examples overview',
      targetDir: EXAMPLES_OVERVIEW_DIR,
      depth: 0,
      isCurrent: context.currentDir === EXAMPLES_OVERVIEW_DIR,
    });
  }

  entries.push(
    ...EXAMPLE_DEMOS.filter((demo) =>
      context.hasPublishedDocsPage(
        `${demo.dir}/docs`,
        context.generatedPageDirectories,
      ),
    ).map((demo) => ({
      label: demo.label,
      targetDir: `${demo.dir}/docs`,
      depth: 0,
      isCurrent: context.currentDir === `${demo.dir}/docs`,
    })),
  );

  return buildPipeNavListHtml(context, entries, 'pipe-nav-list-compact');
}

/** Renders the full left docs sidebar with the examples-first learning path. */
export function buildDocsSidebarHtml(
  context: DocsSidebarRenderContext,
): string {
  const groupsMap = new Map<string, { name: string; items: PageMeta[] }>();

  for (const page of context.pages) {
    const segment = page.relDir.split('/')[0] || 'root';
    if (segment === 'examples') {
      continue;
    }

    if (!groupsMap.has(segment)) {
      groupsMap.set(segment, { name: segment, items: [] });
    }

    groupsMap.get(segment)?.items.push(page);
  }

  const examplesSidebarHtml = buildExamplesSidebarHtml(context);
  const groupsHtml = Array.from(groupsMap.values())
    .sort((leftGroup, rightGroup) => {
      const leftOrder = SIDEBAR_GROUP_ORDER.indexOf(
        leftGroup.name as (typeof SIDEBAR_GROUP_ORDER)[number],
      );
      const rightOrder = SIDEBAR_GROUP_ORDER.indexOf(
        rightGroup.name as (typeof SIDEBAR_GROUP_ORDER)[number],
      );
      const leftRank = leftOrder === -1 ? Number.MAX_SAFE_INTEGER : leftOrder;
      const rightRank =
        rightOrder === -1 ? Number.MAX_SAFE_INTEGER : rightOrder;
      return (
        leftRank - rightRank || leftGroup.name.localeCompare(rightGroup.name)
      );
    })
    .map((group) => {
      const items = group.items.toSorted((leftPage, rightPage) =>
        leftPage.relDir.localeCompare(rightPage.relDir),
      );

      if (group.name === 'root') {
        return buildPipeNavListHtml(
          context,
          buildGroupPipeEntries(items, context.currentDir).filter(
            (entry) => entry.targetDir === '',
          ),
          'pipe-nav-list-root',
        );
      }

      return `<li class="group"><div class="g-head">${context.escapeHtml(
        formatSidebarGroupName(group.name),
      )}</div>${buildPipeNavListHtml(
        context,
        buildGroupPipeEntries(items, context.currentDir),
      )}</li>`;
    })
    .join('');

  return `<ul class="sidebar-sections">${examplesSidebarHtml}${groupsHtml}</ul>`;
}

/** Renders the example showcase cards that anchor the sidebar learning path. */
function buildExamplesSidebarHtml(context: SidebarRenderContext): string {
  const examplesOverviewLink = buildPipeNavListHtml(
    context,
    context.hasPublishedDocsPage(
      EXAMPLES_OVERVIEW_DIR,
      context.generatedPageDirectories,
    )
      ? [
          {
            label: 'Examples overview',
            targetDir: EXAMPLES_OVERVIEW_DIR,
            depth: 0,
            isCurrent: context.currentDir === EXAMPLES_OVERVIEW_DIR,
            cssClass: 'nav-priority-link',
          },
        ]
      : [],
    'pipe-nav-list-priority',
  );

  const showExpandedExampleShowcase =
    context.currentDir === '' ||
    context.currentDir === EXAMPLES_OVERVIEW_DIR ||
    context.currentDir.startsWith('examples/');
  const compactExamplesLinksHtml = showExpandedExampleShowcase
    ? ''
    : buildPipeNavListHtml(
        context,
        EXAMPLE_DEMOS.filter((demo) =>
          context.hasPublishedDocsPage(
            `${demo.dir}/docs`,
            context.generatedPageDirectories,
          ),
        ).map((demo) => ({
          label: demo.label,
          targetDir: `${demo.dir}/docs`,
          depth: 0,
          isCurrent: context.currentDir === `${demo.dir}/docs`,
        })),
        'pipe-nav-list-compact',
      );

  const demoCardsHtml = showExpandedExampleShowcase
    ? EXAMPLE_DEMOS.map((demo) => {
        const readingOrderLinksHtml = buildPipeNavListHtml(
          context,
          demo.readingOrder
            .filter((entry) =>
              context.hasPublishedDocsPage(
                entry.dir,
                context.generatedPageDirectories,
              ),
            )
            .map((entry) => ({
              label: entry.label,
              targetDir: entry.dir,
              depth: 0,
              isCurrent: context.currentDir === entry.dir,
            })),
        );
        const deepDiveLinksHtml = buildPipeNavListHtml(
          context,
          demo.deepDives
            .filter((entry) =>
              context.hasPublishedDocsPage(
                entry.dir,
                context.generatedPageDirectories,
              ),
            )
            .map((entry) => ({
              label: entry.label,
              targetDir: entry.dir,
              depth: 1,
              isCurrent: context.currentDir === entry.dir,
            })),
        );
        const isCurrentDemo =
          context.currentDir === demo.dir ||
          context.currentDir.startsWith(`${demo.dir}/`) ||
          context.currentDir.startsWith(`${demo.dir}/docs`);

        if (!readingOrderLinksHtml && !deepDiveLinksHtml) {
          return '';
        }

        return `<li class="nav-demo-card${isCurrentDemo ? ' is-current-demo' : ''}"><div class="nav-demo-meta"><div class="nav-demo-eyebrow">${context.escapeHtml(
          demo.eyebrow,
        )}</div><div class="nav-demo-title">${context.escapeHtml(
          demo.label,
        )}</div><p class="nav-demo-description">${context.escapeHtml(
          demo.description,
        )}</p></div><div class="nav-demo-cluster"><div class="nav-cluster-title">Recommended path</div>${readingOrderLinksHtml}</div>${
          deepDiveLinksHtml
            ? `<div class="nav-demo-cluster nav-demo-cluster-secondary"><div class="nav-cluster-title">Deep dives</div>${deepDiveLinksHtml}</div>`
            : ''
        }</li>`;
      })
        .filter(Boolean)
        .join('')
    : '';

  if (!examplesOverviewLink && !compactExamplesLinksHtml && !demoCardsHtml) {
    return '';
  }

  return `<li class="group group-priority"><div class="g-head">Start With Examples</div><p class="group-copy">The demos are the fastest way to learn the library in the order a first-time reader can actually absorb.</p><div class="nav-priority-links">${examplesOverviewLink}${compactExamplesLinksHtml}</div>${
    demoCardsHtml ? `<ul class="nav-demo-showcase">${demoCardsHtml}</ul>` : ''
  }</li>`;
}

function buildGroupPipeEntries(
  items: readonly PageMeta[],
  currentDir: string,
): PipeNavEntry[] {
  return items
    .toSorted((leftPage, rightPage) =>
      leftPage.relDir.localeCompare(rightPage.relDir),
    )
    .map((page) => {
      const relativeSegments = page.relDir.split('/').slice(1);
      return {
        label: formatGroupEntryLabel(relativeSegments),
        targetDir: page.relDir,
        depth: Math.max(0, relativeSegments.length - 1),
        isCurrent: page.relDir === currentDir,
      } satisfies PipeNavEntry;
    });
}

/**
 * Formats sidebar labels so overview pages read clearly and deep pages are not
 * reduced to ambiguous repeated leaf names.
 */
function formatGroupEntryLabel(relativeSegments: readonly string[]): string {
  if (relativeSegments.length === 0) {
    return 'Overview';
  }

  const leafLabel = formatSidebarGroupName(
    relativeSegments.at(-1) ?? 'Overview',
  );
  if (relativeSegments.length === 1) {
    return leafLabel;
  }

  const parentLabel = formatSidebarGroupName(relativeSegments.at(-2) ?? '');
  return `${parentLabel} / ${leafLabel}`;
}

/** Formats a sidebar group or segment name into a human-readable label. */
function formatSidebarGroupName(segment: string): string {
  return segment
    .split(/[-_]/g)
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/** Renders the fixed-width rail cells that anchor one sidebar row. */
function buildPipeRailCellsHtml(depth: number, isFooter = false): string {
  const railDepth = Math.max(0, depth);

  return Array.from({ length: railDepth + 1 }, (_, columnIndex) => {
    const isLastColumn = columnIndex === railDepth;
    const classNames = [
      'nav-rail-cell',
      isLastColumn ? 'nav-rail-cell-last' : '',
      isFooter ? 'nav-rail-cell-footer' : '',
    ]
      .filter(Boolean)
      .join(' ');

    return `<span class="${classNames}"></span>`;
  }).join('');
}

/** Renders one rail-based pipe navigation list. */
function buildPipeNavListHtml(
  context: SidebarRenderContext,
  entries: readonly PipeNavEntry[],
  listClass = '',
): string {
  if (entries.length === 0) {
    return '';
  }

  const itemsHtml = entries
    .map((entry) => {
      const href = context.buildRelativeDocsHref(
        context.currentDir,
        entry.targetDir,
      );
      const classNames = [
        'pipe-nav-item',
        entry.cssClass,
        entry.isCurrent ? 'current' : '',
      ]
        .filter(Boolean)
        .join(' ');

      return `<li class="${classNames}"><a href="${href}"><span class="nav-rail-track" aria-hidden="true">${buildPipeRailCellsHtml(
        entry.depth,
      )}</span><span class="nav-link-label">${context.escapeHtml(
        entry.label,
      )}</span></a></li>`;
    })
    .join('');

  const listClasses = ['pipe-nav-list', listClass].filter(Boolean).join(' ');
  const lastEntryDepth = entries.at(-1)?.depth ?? 0;

  return `<ul class="${listClasses}">${itemsHtml}<li class="pipe-nav-footer" aria-hidden="true"><span class="nav-rail-track">${buildPipeRailCellsHtml(
    lastEntryDepth,
    true,
  )}</span></li></ul>`;
}
