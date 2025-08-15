"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
/*
 * Converts every README.md inside docs/ (including root copy) into an index.html in the same directory.
 * Usage: npm run docs:html
 */
const fast_glob_1 = __importDefault(require("fast-glob"));
const path_1 = __importDefault(require("path"));
const fs_extra_1 = __importDefault(require("fs-extra"));
const marked_1 = require("marked");
const DOCS_DIR = path_1.default.resolve('docs');
function slugify(s) {
    return s
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .replace(/-{2,}/g, '-');
}
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        var _a, _b;
        const readmes = yield (0, fast_glob_1.default)(['**/README.md'], { cwd: DOCS_DIR, absolute: true });
        const pages = [];
        for (const mdFile of readmes) {
            const md = yield fs_extra_1.default.readFile(mdFile, 'utf8');
            const title = ((_a = md.match(/^#\s+(.+)$/m)) === null || _a === void 0 ? void 0 : _a[1]) || (path_1.default.relative(DOCS_DIR, path_1.default.dirname(mdFile)) || 'Documentation');
            const relDir = path_1.default.relative(DOCS_DIR, path_1.default.dirname(mdFile)).replace(/\\/g, '/');
            pages.push({ abs: mdFile, relDir, title });
        }
        // Build flat nav list (could be enhanced to a tree)
        const navHtmlFor = (currentDir) => {
            const links = pages
                .sort((a, b) => a.relDir.localeCompare(b.relDir))
                .map(p => {
                const label = p.relDir === '' ? 'root' : p.relDir + '/';
                const isCurrent = p.relDir === currentDir;
                const relLink = path_1.default.posix.relative(currentDir || '.', p.relDir || '.') || '.'; // relative folder path
                const href = (relLink === '.' ? '.' : relLink) + '/index.html';
                return `<li${isCurrent ? ' class="current"' : ''}><a href="${href}">${label}</a></li>`;
            }).join('\n');
            // If the asciiMaze example exists in the repo, append a nav link so generated docs include it
            try {
                const asciiExampleAbs = path_1.default.resolve('test', 'examples', 'asciiMaze', 'index.html');
                if (fs_extra_1.default.existsSync(asciiExampleAbs)) {
                    const absTargetDir = path_1.default.relative(DOCS_DIR, path_1.default.dirname(asciiExampleAbs)).replace(/\\/g, '/');
                    const relLink = path_1.default.posix.relative(currentDir || '.', absTargetDir || '.') || '.';
                    const href = (relLink === '.' ? '.' : relLink) + '/index.html';
                    const extra = `<li><a href="${href}">examples/asciiMaze/</a></li>`;
                    return `<ul class="doc-nav">${links}\n${extra}</ul>`;
                }
            }
            catch (e) {
                /* ignore extra nav */
            }
            return `<ul class="doc-nav">${links}</ul>`;
        };
        for (const meta of pages) {
            const md = yield fs_extra_1.default.readFile(meta.abs, 'utf8');
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
            const renderer = new marked_1.marked.Renderer();
            const originalHeading = (_b = renderer.heading) === null || _b === void 0 ? void 0 : _b.bind(renderer);
            renderer.heading = (text, level, raw, slugger) => {
                // raw is the unescaped heading text; use it for id to align with our parsing.
                const id = slugify(raw.trim());
                return `<h${level} id="${id}">${text}</h${level}>`;
            };
            marked_1.marked.use({ renderer });
            const htmlBody = marked_1.marked.parse(md, { async: false });
            const toc = fileHeadings.length ? `<div class="page-toc"><h2>Files</h2>${fileHeadings.map(f => `<div class=\"toc-file\"><a href=\"#${f.anchor}\">${f.file}</a>${f.symbols.length ? `<ul>${f.symbols.map(s => `<li><a href=#${s.anchor}>${s.name}</a></li>`).join('')}</ul>` : ''}</div>`).join('')}</div>` : '';
            const outFile = path_1.default.join(path_1.default.dirname(meta.abs), 'index.html');
            const page = `<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>${meta.title}</title>
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;margin:0 auto;padding:0 20px 60px;line-height:1.55;background:#fff;color:#222;display:grid;grid-template-columns:260px 1fr 280px;grid-gap:32px;}
nav.site{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:24px 0 40px;}
nav.site h1{font-size:1.05rem;margin:0 0 .75rem;font-weight:600;}
.doc-nav{list-style:none;margin:0;padding:0;font-size:.85rem;}
.doc-nav li{margin:2px 0;}
 .doc-nav a{display:block;padding:4px 8px;border-radius:4px;color:#2c3963;text-decoration:none;}
 .doc-nav li.current>a{background:#2c3963;color:#fff;font-weight:600;}
 .doc-nav a:hover{background:#e4e8f3;}
main{padding:40px 0;}
aside.page-index{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:32px 0 40px;font-size:.85rem;}
aside.page-index h2{font-size:.9rem;margin:0 0 .75rem;text-transform:uppercase;letter-spacing:.5px;color:#444;}
aside.page-index .toc-file{margin:0 0 .5rem;}
aside.page-index ul{list-style:none;margin:.25rem 0 .5rem .25rem;padding:0;}
aside.page-index li{margin:0;}
 aside.page-index a{color:#444;text-decoration:none;}
 aside.page-index a:hover{color:#2c3963;}
pre{background:#1e1e1e;color:#eee;padding:12px;border-radius:6px;overflow:auto;}
code{background:#f5f5f5;padding:2px 4px;border-radius:4px;font-size:90%;}
pre code{background:transparent;padding:0;font-size:90%;}
 a{color:#2c3963;text-decoration:none;}a:hover{text-decoration:underline;}
h1,h2,h3,h4{scroll-margin-top:70px;}
blockquote{border-left:4px solid #ddd;margin:1em 0;padding:.5em 1em;color:#555;}
table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}
footer{margin-top:64px;font-size:.75rem;color:#666;}
@media (max-width:1100px){body{grid-template-columns:220px 1fr;}aside.page-index{display:none;} }
@media (max-width:800px){body{grid-template-columns:1fr;}nav.site{position:relative;top:auto;max-height:none;order:2;}main{order:1;padding-top:24px;}}
</style></head><body>
<nav class="site">
  <h1>Docs Index</h1>
  ${navHtmlFor(meta.relDir)}
</nav>
<main>
${htmlBody}
<footer>Generated from JSDoc. <a href="https://github.com/reicek/NeatapticTS">GitHub</a></footer>
</main>
<aside class="page-index">${toc}</aside>
</body></html>`;
            yield fs_extra_1.default.writeFile(outFile, page, 'utf8');
        }
        console.log('HTML docs generated.');
    });
}
main().catch(e => { console.error(e); process.exit(1); });
//# sourceMappingURL=render-docs-html.js.map