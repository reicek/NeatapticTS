/*
 * Converts every README.md inside docs/ (including root copy) into an index.html in the same directory.
 * Usage: npm run docs:html
 */
import fg from 'fast-glob';
import path from 'path';
import fs from 'fs-extra';
import { marked } from 'marked';
const DOCS_DIR = path.resolve('docs');
function slugify(s) {
    return s
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .replace(/-{2,}/g, '-');
}
async function main() {
    const readmes = await fg(['**/README.md'], { cwd: DOCS_DIR, absolute: true });
    const pages = [];
    for (const mdFile of readmes) {
        const md = await fs.readFile(mdFile, 'utf8');
        const title = md.match(/^#\s+(.+)$/m)?.[1] ||
            path.relative(DOCS_DIR, path.dirname(mdFile)) ||
            'Documentation';
        const relDir = path
            .relative(DOCS_DIR, path.dirname(mdFile))
            .replace(/\\/g, '/');
        pages.push({ abs: mdFile, relDir, title });
    }
    // Build nav list; group top-level folders similar to original sections.
    const navHtmlFor = (currentDir) => {
        const groupsMap = new Map();
        for (const p of pages) {
            const seg = p.relDir.split('/')[0] || 'root';
            if (!groupsMap.has(seg))
                groupsMap.set(seg, { name: seg, items: [] });
            groupsMap.get(seg).items.push(p);
        }
        const order = ['root', 'architecture', 'methods', 'neat', 'multithreading', 'examples'];
        const makeLink = (page) => {
            const isCurrent = page.relDir === currentDir;
            const relLink = path.posix.relative(currentDir || '.', page.relDir || '.') || '.';
            const href = (relLink === '.' ? '.' : relLink) + '/index.html';
            const label = page.relDir === '' ? 'Overview' : page.relDir.replace(/\\/g, '/');
            return `<li${isCurrent ? ' class="current"' : ''}><a href="${href}">${label}${isCurrent ? '' : ''}</a></li>`;
        };
        // Add asciiMaze example explicitly if present
        const asciiExample = () => {
            try {
                const copiedExampleAbs = path.resolve(DOCS_DIR, 'examples', 'asciiMaze', 'index.html');
                if (fs.existsSync(copiedExampleAbs)) {
                    const relTargetDir = 'examples/asciiMaze';
                    const relLink = path.posix.relative(currentDir || '.', relTargetDir) || '.';
                    const href = (relLink === '.' ? '.' : relLink) + '/index.html';
                    return `<li><a href="${href}">examples/asciiMaze</a></li>`;
                }
            }
            catch { /* ignore */ }
            return '';
        };
        const groupsHtml = Array.from(groupsMap.values())
            .sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name))
            .map(g => {
            const items = g.items.sort((a, b) => a.relDir.localeCompare(b.relDir));
            if (g.name === 'root')
                return makeLink(items.find(i => i.relDir === ''));
            return `<li class="group"><div class="g-head">${g.name}</div><ul>${items.map(makeLink).join('')}${g.name === 'examples' ? asciiExample() : ''}</ul></li>`;
        })
            .join('');
        return `<ul class="sidebar-sections">${groupsHtml}</ul>`;
    };
    for (const meta of pages) {
        const md = await fs.readFile(meta.abs, 'utf8');
        // Extract headings for TOC (## file, ### symbol)
        const fileHeadings = [];
        const lines = md.split(/\r?\n/);
        let currentFile = null;
        for (const line of lines) {
            const fileMatch = /^##\s+(.+\.ts)\s*$/.exec(line);
            if (fileMatch) {
                const fileName = fileMatch[1];
                const anchor = slugify(fileName);
                currentFile = { file: fileName, anchor, symbols: [] };
                fileHeadings.push(currentFile);
                continue;
            }
            const symMatch = /^###\s+([A-Za-z0-9_]+)\s*$/.exec(line);
            if (symMatch && currentFile) {
                const sym = symMatch[1];
                currentFile.symbols.push({ name: sym, anchor: slugify(sym) });
            }
        }
        // Configure marked renderer with deterministic heading IDs so anchors match our TOC.
        const renderer = new marked.Renderer();
        const originalHeading = renderer.heading?.bind(renderer);
        renderer.heading = (text, level, raw) => {
            // raw is the unescaped heading text; use it for id to align with our parsing.
            const id = slugify(raw.trim());
            return `<h${level} id="${id}">${text}</h${level}>`;
        };
        marked.use({ renderer });
        const htmlBody = marked.parse(md, { async: false });
        const toc = fileHeadings.length
            ? `<div class="page-toc"><h2>Files</h2>${fileHeadings
                .map((f) => `<div class=\"toc-file\"><a href=\"#${f.anchor}\">${f.file}</a>${f.symbols.length
                ? `<ul>${f.symbols
                    .map((s) => `<li><a href=#${s.anchor}>${s.name}</a></li>`)
                    .join('')}</ul>`
                : ''}</div>`)
                .join('')}</div>`
            : '';
        const outFile = path.join(path.dirname(meta.abs), 'index.html');
        const relToRoot = path.relative(path.dirname(meta.abs), DOCS_DIR).replace(/\\/g, '/');
        const cssHref = (relToRoot ? relToRoot + '/' : '') + 'assets/theme.css';
        const page = `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>${meta.title} – NeatapticTS Docs</title><meta name="viewport" content="width=device-width,initial-scale=1">\n<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700&family=Open+Sans:wght@400;600&display=swap" rel="stylesheet">\n<link rel="stylesheet" href="${cssHref}"></head><body class="${meta.relDir === '' ? 'is-root' : ''}">\n<header class="topbar"><div class="inner"><div class="brand"><a href="${relToRoot || '.'}/index.html">NeatapticTS</a></div><nav class="main-nav"><a href="${relToRoot || '.'}/index.html">Home</a><a href="${relToRoot || '.'}/index.html" class="active">Docs</a><a href="https://github.com/reicek/NeatapticTS" target="_blank" rel="noopener">GitHub</a></nav></div></header>\n<div class="layout"><aside class="sidebar">${navHtmlFor(meta.relDir)}</aside><main class="content">${htmlBody}<footer class="site-footer">Generated from source JSDoc • <a href="https://github.com/reicek/NeatapticTS">GitHub</a></footer></main><aside class="toc">${toc}</aside></div></body></html>`;
        await fs.writeFile(outFile, page, 'utf8');
    }
    console.log('HTML docs generated.');
}
main().catch((e) => {
    console.error(e);
    process.exit(1);
});
//# sourceMappingURL=render-docs-html.js.map