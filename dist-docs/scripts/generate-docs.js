"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
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
 * Generates per-folder README.md aggregating exported symbols' JSDoc.
 * - Copies root README.md into docs/ (manual content retained)
 * - Skips generating README for src root (leave top-level README manual)
 * Usage: npm run docs:folders
 */
const ts_morph_1 = require("ts-morph");
const fast_glob_1 = __importDefault(require("fast-glob"));
const path = __importStar(require("path"));
const fs_extra_1 = __importDefault(require("fs-extra"));
const SRC_DIR = path.resolve('src');
const DOCS_DIR = path.resolve('docs');
const ROOT_README_SRC = path.resolve('README.md');
const ROOT_README_DEST = path.join(DOCS_DIR, 'README.md');
const project = new ts_morph_1.Project({
    tsConfigFilePath: path.resolve('tsconfig.json'),
    skipAddingFilesFromTsConfig: true
});
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t, _u, _v, _w, _x, _y, _z, _0, _1;
        yield fs_extra_1.default.ensureDir(DOCS_DIR);
        // Copy root README (manual) into docs
        if (yield fs_extra_1.default.pathExists(ROOT_README_SRC)) {
            yield fs_extra_1.default.copyFile(ROOT_README_SRC, ROOT_README_DEST);
        }
        const filePaths = yield (0, fast_glob_1.default)(['**/*.ts'], { cwd: SRC_DIR, absolute: true, ignore: ['**/*.d.ts'] });
        for (const p of filePaths) {
            if (/\.test\.ts$/i.test(p))
                continue; // skip test specification files
            project.addSourceFileAtPath(p);
        }
        const all = project.getSourceFiles();
        const sourceFiles = all.filter((sf) => {
            const fp = sf.getFilePath();
            return !fp.endsWith('.d.ts') && !/node_modules/.test(fp);
        });
        console.log(`[docs] Loaded ${sourceFiles.length} source files (raw: ${all.length})`);
        const dirMap = new Map();
        for (const sf of sourceFiles) {
            const exported = sf.getExportedDeclarations();
            // capture file-level/module JSDoc if present
            const fileJsDocs = ((_b = (_a = sf).getJsDocs) === null || _b === void 0 ? void 0 : _b.call(_a)) || [];
            const fileDocPrimary = fileJsDocs[0];
            const fileDocDesc = (_d = (_c = fileDocPrimary === null || fileDocPrimary === void 0 ? void 0 : fileDocPrimary.getDescription) === null || _c === void 0 ? void 0 : _c.call(fileDocPrimary)) === null || _d === void 0 ? void 0 : _d.trim();
            // ensure fileMap exists early so we can add file-level doc
            const dir = path.dirname(sf.getFilePath());
            let fileMap = dirMap.get(dir);
            if (!fileMap) {
                fileMap = new Map();
                dirMap.set(dir, fileMap);
            }
            if (fileDocDesc) {
                const fileSummarySymbol = {
                    kind: 'File',
                    name: '__file_summary__',
                    filePath: sf.getFilePath(),
                    jsdoc: { description: fileDocDesc }
                };
                const arr0 = fileMap.get(sf.getFilePath()) || [];
                arr0.push(fileSummarySymbol);
                fileMap.set(sf.getFilePath(), arr0);
                console.log(`[docs] Added file summary for ${sf.getBaseName()}`);
            }
            if (exported.size) {
                console.log(`[docs] ${sf.getBaseName()} exports: ${exported.size}`);
                for (const [, decls] of exported) {
                    for (const decl of decls) {
                        const sym = decl.getSymbol();
                        if (!sym)
                            continue;
                        const rendered = renderSymbol(sym, decl.getKindName(), sf.getFilePath());
                        if (rendered) {
                            const arr = fileMap.get(sf.getFilePath()) || [];
                            arr.push(rendered);
                            fileMap.set(sf.getFilePath(), arr);
                        }
                        // special-case: exported variable with object literal initializer (e.g. Activation = { ... })
                        try {
                            if (decl.getKindName && decl.getKindName() === 'VariableDeclaration') {
                                const init = (_f = (_e = decl).getInitializer) === null || _f === void 0 ? void 0 : _f.call(_e);
                                if (init && init.getKindName && init.getKindName() === 'ObjectLiteralExpression') {
                                    const props = ((_g = init.getProperties) === null || _g === void 0 ? void 0 : _g.call(init)) || [];
                                    for (const p of props) {
                                        const propName = ((_h = p.getName) === null || _h === void 0 ? void 0 : _h.call(p)) || (p.getSymbol && ((_k = (_j = p.getSymbol()) === null || _j === void 0 ? void 0 : _j.getName) === null || _k === void 0 ? void 0 : _k.call(_j)));
                                        const renderedProp = renderDeclaration(p, String(propName), ((_l = p.getKindName) === null || _l === void 0 ? void 0 : _l.call(p)) || 'Property', sf.getFilePath(), sym.getName());
                                        if (renderedProp) {
                                            const arr2 = fileMap.get(sf.getFilePath()) || [];
                                            arr2.push(renderedProp);
                                            fileMap.set(sf.getFilePath(), arr2);
                                        }
                                    }
                                }
                            }
                        }
                        catch (e) { /* ignore introspection errors */ }
                        // special-case: exported class -> include members
                        try {
                            if (decl.getKindName && decl.getKindName() === 'ClassDeclaration') {
                                const members = ((_o = (_m = decl).getMembers) === null || _o === void 0 ? void 0 : _o.call(_m)) || [];
                                for (const m of members) {
                                    const memberName = (_p = m.getName) === null || _p === void 0 ? void 0 : _p.call(m);
                                    if (!memberName)
                                        continue;
                                    const renderedMember = renderDeclaration(m, String(memberName), ((_q = m.getKindName) === null || _q === void 0 ? void 0 : _q.call(m)) || 'ClassMember', sf.getFilePath(), sym.getName());
                                    if (renderedMember) {
                                        const arr3 = fileMap.get(sf.getFilePath()) || [];
                                        arr3.push(renderedMember);
                                        fileMap.set(sf.getFilePath(), arr3);
                                    }
                                }
                            }
                        }
                        catch (e) { /* ignore */ }
                    }
                }
            }
            // Additionally capture top-level, non-exported declarations if they have JSDoc
            try {
                const stmts = ((_s = (_r = sf).getStatements) === null || _s === void 0 ? void 0 : _s.call(_r)) || [];
                for (const st of stmts) {
                    const decls = (st.getDeclarations && st.getDeclarations()) || [st];
                    for (const d of decls) {
                        const jsdocs = ((_u = (_t = d).getJsDocs) === null || _u === void 0 ? void 0 : _u.call(_t)) || [];
                        const hasExport = (d.getSymbol && ((_x = (_w = (_v = d.getSymbol()) === null || _v === void 0 ? void 0 : _v.getDeclarations) === null || _w === void 0 ? void 0 : _w.call(_v)) === null || _x === void 0 ? void 0 : _x.some((x) => x.isExported && x.isExported()))) || false;
                        if (!jsdocs.length)
                            continue; // only include documented non-exported declarations
                        // skip if already added via exported processing
                        const name = ((_y = d.getName) === null || _y === void 0 ? void 0 : _y.call(d)) || (d.getSymbol && ((_0 = (_z = d.getSymbol()) === null || _z === void 0 ? void 0 : _z.getName) === null || _0 === void 0 ? void 0 : _0.call(_z)));
                        if (!name)
                            continue;
                        const already = (fileMap.get(sf.getFilePath()) || []).some(s => s.name === name || s.name === `${name}` || s.name === `${name}`);
                        if (already)
                            continue;
                        const rendered = renderDeclaration(d, String(name), ((_1 = d.getKindName) === null || _1 === void 0 ? void 0 : _1.call(d)) || 'Declaration', sf.getFilePath());
                        if (rendered) {
                            const arr4 = fileMap.get(sf.getFilePath()) || [];
                            arr4.push(rendered);
                            fileMap.set(sf.getFilePath(), arr4);
                        }
                    }
                }
            }
            catch (e) { /* ignore */ }
        }
        for (const [dir, fileMap] of dirMap) {
            // Tidy & deduplicate symbols per file before rendering
            for (const [filePath, symbols] of fileMap) {
                const seen = new Map();
                const deduped = [];
                for (const s of symbols) {
                    // normalize name
                    const normName = normalizeName(s, filePath);
                    const key = `${s.parent || ''}::${normName}::${s.signature || ''}`;
                    const existing = seen.get(key);
                    if (existing) {
                        // merge jsdoc fields conservatively
                        existing.jsdoc.description = existing.jsdoc.description || s.jsdoc.description;
                        existing.jsdoc.summary = existing.jsdoc.summary || s.jsdoc.summary;
                        existing.jsdoc.deprecated = existing.jsdoc.deprecated || s.jsdoc.deprecated;
                        if (s.jsdoc.params) {
                            existing.jsdoc.params = (existing.jsdoc.params || []).slice();
                            for (const p of s.jsdoc.params) {
                                if (!existing.jsdoc.params.some(ep => ep.name === p.name))
                                    existing.jsdoc.params.push(p);
                            }
                        }
                        if (!existing.signature && s.signature)
                            existing.signature = s.signature;
                    }
                    else {
                        const clone = Object.assign(Object.assign({}, s), { name: normName, jsdoc: Object.assign(Object.assign({}, s.jsdoc), { params: s.jsdoc.params ? s.jsdoc.params.slice() : undefined }) });
                        seen.set(key, clone);
                        deduped.push(clone);
                    }
                }
                fileMap.set(filePath, deduped);
            }
            const relDir = path.relative(SRC_DIR, dir); // '' means src root
            if (relDir.startsWith('..'))
                continue; // outside src
            const outDir = relDir === '' ? path.join(DOCS_DIR, 'src') : path.join(DOCS_DIR, relDir);
            yield fs_extra_1.default.ensureDir(outDir);
            const outFile = path.join(outDir, 'README.md');
            const md = buildDirectoryReadme(relDir, fileMap);
            yield writeIfChanged(outFile, md);
            // Also emit directly into the src folder tree so GitHub shows it inline when browsing code.
            const srcTargetDir = path.join(SRC_DIR, relDir);
            const srcReadme = path.join(srcTargetDir, 'README.md');
            yield emitSourceReadme(srcReadme, md);
        }
        const rootNode = { name: 'src', path: 'src', children: new Map(), fileCount: 0 };
        const relDirs = [...dirMap.keys()].map(d => path.relative(SRC_DIR, d)).filter(d => !d.startsWith('..'));
        for (const rel of relDirs) {
            const clean = rel === '' ? 'src' : rel.replace(/\\/g, '/');
            const parts = clean.split('/');
            let node = rootNode;
            let acc = '';
            for (const part of parts) {
                acc = acc ? `${acc}/${part}` : part;
                if (!node.children.has(part)) {
                    node.children.set(part, { name: part, path: acc, children: new Map() });
                }
                node = node.children.get(part);
            }
            // attach file count if available
            const abs = path.join(SRC_DIR, rel === '' ? '' : rel);
            const fm = dirMap.get(abs);
            if (fm)
                node.fileCount = fm.size;
        }
        const lines = ['# Docs Index', '', 'Auto-generated index of source folders (click to open folder README).', ''];
        function renderNode(n, level) {
            const indent = '  '.repeat(Math.max(0, level));
            const label = (n.path === 'src' && level === 0) ? 'src (root)' : n.name;
            const link = `${n.path}/README.md`;
            const count = n.fileCount ? ` — ${n.fileCount} file${n.fileCount > 1 ? 's' : ''}` : '';
            lines.push(`${indent}- [${label}](${link})${count}`);
            const childNames = [...n.children.keys()].sort();
            for (const k of childNames)
                renderNode(n.children.get(k), level + 1);
        }
        // Render top-level root and its children (skip a duplicate 'src' nesting)
        renderNode(rootNode, 0);
        yield fs_extra_1.default.writeFile(path.join(DOCS_DIR, 'FOLDERS.md'), lines.join('\n') + '\n', 'utf8');
        console.log('Per-folder README generation complete.');
    });
}
function renderSymbol(sym, fallbackKind, filePath) {
    var _a, _b, _c;
    const decl = sym.getDeclarations()[0];
    if (!decl)
        return null;
    const kind = decl.getKindName();
    const allow = /ClassDeclaration|FunctionDeclaration|InterfaceDeclaration|EnumDeclaration|TypeAliasDeclaration|VariableDeclaration/;
    if (!allow.test(kind))
        return null;
    // ts-morph Declaration with JSDoc support; cast to any to access getJsDocs generically
    const jsDocs = ((_b = (_a = decl).getJsDocs) === null || _b === void 0 ? void 0 : _b.call(_a)) || [];
    if (jsDocs.some((j) => j.getTags().some((t) => t.getTagName() === 'internal')))
        return null;
    const primary = jsDocs[0];
    const fullDesc = primary === null || primary === void 0 ? void 0 : primary.getDescription().trim();
    const summary = (_c = fullDesc === null || fullDesc === void 0 ? void 0 : fullDesc.split(/\r?\n\r?\n/)[0]) === null || _c === void 0 ? void 0 : _c.trim();
    const tags = (primary === null || primary === void 0 ? void 0 : primary.getTags()) || [];
    const paramsTags = tags.filter(t => t.getTagName() === 'param');
    const returnsTag = tags.find(t => t.getTagName() === 'returns' || t.getTagName() === 'return');
    const deprecatedTag = tags.find(t => t.getTagName() === 'deprecated');
    let signature;
    try {
        const sig = decl.getType().getCallSignatures()[0];
        if (sig) {
            const params = sig.getParameters().map((p) => {
                const decls = p.getDeclarations();
                const t = p.getTypeAtLocation(decls[0] || decl).getText();
                return `${p.getName()}: ${t}`;
            }).join(', ');
            const ret = sig.getReturnType().getText();
            signature = `(${params}) => ${ret}`;
        }
    }
    catch ( /* ignore */_d) { /* ignore */ }
    const params = paramsTags.map(t => {
        const text = t.getText();
        const match = text.match(/@param\s+(\w+)/);
        const name = (match === null || match === void 0 ? void 0 : match[1]) || '';
        const raw = t.getComment();
        const doc = Array.isArray(raw) ? raw.map(r => r.getText ? r.getText() : String(r)).join(' ').trim() : (raw || undefined);
        return { name, doc };
    });
    return {
        kind: fallbackKind || kind,
        name: sym.getName(),
        parent: undefined,
        filePath,
        signature,
        jsdoc: {
            summary,
            description: fullDesc,
            params: params.length ? params : undefined,
            returns: (() => { const r = returnsTag === null || returnsTag === void 0 ? void 0 : returnsTag.getComment(); return typeof r === 'string' ? r.trim() : undefined; })(),
            deprecated: (() => { const r = deprecatedTag === null || deprecatedTag === void 0 ? void 0 : deprecatedTag.getComment(); return typeof r === 'string' ? r.trim() : undefined; })()
        }
    };
}
// Render a declaration-like object (node) which may not have a symbol
function renderDeclaration(decl, forcedName, forcedKind, filePath, parentName) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o;
    if (!decl)
        return null;
    try {
        const kind = forcedKind || ((_a = decl.getKindName) === null || _a === void 0 ? void 0 : _a.call(decl)) || ((_b = decl.getKind) === null || _b === void 0 ? void 0 : _b.call(decl)) || 'Declaration';
        const name = forcedName || ((_c = decl.getName) === null || _c === void 0 ? void 0 : _c.call(decl)) || (decl.getSymbol && ((_e = (_d = decl.getSymbol()) === null || _d === void 0 ? void 0 : _d.getName) === null || _e === void 0 ? void 0 : _e.call(_d))) || (parentName ? `${parentName}.${forcedName}` : '');
        const jsDocs = ((_f = decl.getJsDocs) === null || _f === void 0 ? void 0 : _f.call(decl)) || [];
        if (!jsDocs.length)
            return null;
        if (jsDocs.some((j) => j.getTags().some((t) => t.getTagName() === 'internal')))
            return null;
        const primary = jsDocs[0];
        const fullDesc = (_h = (_g = primary === null || primary === void 0 ? void 0 : primary.getDescription) === null || _g === void 0 ? void 0 : _g.call(primary)) === null || _h === void 0 ? void 0 : _h.trim();
        const summary = (_j = fullDesc === null || fullDesc === void 0 ? void 0 : fullDesc.split(/\r?\n\r?\n/)[0]) === null || _j === void 0 ? void 0 : _j.trim();
        const tags = (primary === null || primary === void 0 ? void 0 : primary.getTags()) || [];
        const paramsTags = tags.filter(t => t.getTagName() === 'param');
        const returnsTag = tags.find(t => t.getTagName() === 'returns' || t.getTagName() === 'return');
        const deprecatedTag = tags.find(t => t.getTagName() === 'deprecated');
        let signature;
        try {
            const type = ((_k = decl.getType) === null || _k === void 0 ? void 0 : _k.call(decl)) || (decl.getSymbol && ((_m = (_l = decl.getSymbol()) === null || _l === void 0 ? void 0 : _l.getType) === null || _m === void 0 ? void 0 : _m.call(_l)));
            const sig = (_o = type === null || type === void 0 ? void 0 : type.getCallSignatures) === null || _o === void 0 ? void 0 : _o.call(type)[0];
            if (sig) {
                const params = sig.getParameters().map((p) => {
                    const decls = p.getDeclarations();
                    const t = p.getTypeAtLocation(decls[0] || decl).getText();
                    return `${p.getName()}: ${t}`;
                }).join(', ');
                const ret = sig.getReturnType().getText();
                signature = `(${params}) => ${ret}`;
            }
        }
        catch ( /* ignore */_p) { /* ignore */ }
        const params = paramsTags.map(t => {
            var _a, _b;
            const text = ((_a = t.getText) === null || _a === void 0 ? void 0 : _a.call(t)) || '';
            const match = text.match(/@param\s+(\w+)/);
            const name = (match === null || match === void 0 ? void 0 : match[1]) || '';
            const raw = (_b = t.getComment) === null || _b === void 0 ? void 0 : _b.call(t);
            const doc = Array.isArray(raw) ? raw.map(r => r.getText ? r.getText() : String(r)).join(' ').trim() : (raw || undefined);
            return { name, doc };
        });
        return {
            kind,
            name,
            parent: parentName || undefined,
            filePath: filePath || '',
            signature,
            jsdoc: {
                summary,
                description: fullDesc,
                params: params.length ? params : undefined,
                returns: (() => { var _a; const r = (_a = returnsTag === null || returnsTag === void 0 ? void 0 : returnsTag.getComment) === null || _a === void 0 ? void 0 : _a.call(returnsTag); return typeof r === 'string' ? r.trim() : undefined; })(),
                deprecated: (() => { var _a; const r = (_a = deprecatedTag === null || deprecatedTag === void 0 ? void 0 : deprecatedTag.getComment) === null || _a === void 0 ? void 0 : _a.call(deprecatedTag); return typeof r === 'string' ? r.trim() : undefined; })()
            }
        };
    }
    catch (e) {
        return null;
    }
}
// Normalize a symbol name: prefer meaningful names over 'default', fallback to file basename
function normalizeName(s, filePath) {
    let name = (s.name || '').toString();
    if (!name || name === 'default' || name === '__file_summary__') {
        // try to derive from file path or signature
        const base = path.basename(filePath || '', '.ts');
        if (s.kind === 'File' || name === '__file_summary__')
            return base;
        // if parent exists, qualify with parent
        if (s.parent)
            return `${s.parent}.${base}`;
        // try to extract from signature
        if (s.signature)
            return `${base}${s.signature.split(')')[0]})`;
        return base;
    }
    // strip trailing redundant '()' or 'function ' prefixes
    name = name.replace(/^function\s+/, '').replace(/\(\)$/, '');
    return name;
}
function buildDirectoryReadme(relDir, fileMap) {
    var _a, _b, _c;
    const title = relDir.replace(/\\/g, '/');
    const lines = [
        `# ${title}`,
        ''
    ];
    const filesSorted = [...fileMap.keys()].sort();
    for (const file of filesSorted) {
        const relFile = path.relative(SRC_DIR, file).replace(/\\/g, '/');
        lines.push(`## ${relFile}`, '');
        const symbols = fileMap.get(file).slice().sort((a, b) => (a.parent || a.name).localeCompare(b.parent || b.name) || a.name.localeCompare(b.name));
        // extract file summary if present
        const fileSummaryIdx = symbols.findIndex(s => s.name === '__file_summary__' && s.kind === 'File');
        if (fileSummaryIdx >= 0) {
            const fsym = symbols.splice(fileSummaryIdx, 1)[0];
            if (fsym.jsdoc.description)
                lines.push(fsym.jsdoc.description, '');
        }
        // Group by parent: top-level (no parent) and parent groups
        const topLevel = symbols.filter(s => !s.parent);
        const byParent = new Map();
        for (const s of symbols.filter(s => s.parent)) {
            const arr = byParent.get(s.parent) || [];
            arr.push(s);
            byParent.set(s.parent, arr);
        }
        // render top-level symbols
        for (const s of topLevel) {
            lines.push(`### ${s.name}`);
            if (s.signature)
                lines.push('', '`' + s.signature + '`');
            if (s.jsdoc.description)
                lines.push('', s.jsdoc.description);
            if (s.jsdoc.deprecated)
                lines.push('', `**Deprecated:** ${s.jsdoc.deprecated}`);
            if ((_a = s.jsdoc.params) === null || _a === void 0 ? void 0 : _a.length) {
                lines.push('', 'Parameters:');
                for (const p of s.jsdoc.params)
                    lines.push(`- \`${p.name}\`${p.doc ? ' - ' + p.doc : ''}`);
            }
            if (s.jsdoc.returns)
                lines.push('', `Returns: ${s.jsdoc.returns}`);
            lines.push('');
            // if this top-level has grouped children (byParent keyed by this name), render them nested
            const children = byParent.get(s.name);
            if (children) {
                children.sort((a, b) => a.name.localeCompare(b.name));
                for (const c of children) {
                    lines.push(`#### ${c.name}`);
                    if (c.signature)
                        lines.push('', '`' + c.signature + '`');
                    if (c.jsdoc.description)
                        lines.push('', c.jsdoc.description);
                    if (c.jsdoc.deprecated)
                        lines.push('', `**Deprecated:** ${c.jsdoc.deprecated}`);
                    if ((_b = c.jsdoc.params) === null || _b === void 0 ? void 0 : _b.length) {
                        lines.push('', 'Parameters:');
                        for (const p of c.jsdoc.params)
                            lines.push(`- \`${p.name}\`${p.doc ? ' - ' + p.doc : ''}`);
                    }
                    if (c.jsdoc.returns)
                        lines.push('', `Returns: ${c.jsdoc.returns}`);
                    lines.push('');
                }
            }
        }
        // Render any parent groups that didn't have a top-level parent symbol (e.g., object literal properties grouped under exported var)
        for (const [parentName, group] of byParent) {
            if (topLevel.some(t => t.name === parentName))
                continue; // already rendered under its parent item
            lines.push(`### ${parentName}`, '');
            group.sort((a, b) => a.name.localeCompare(b.name));
            for (const c of group) {
                lines.push(`#### ${c.name}`);
                if (c.signature)
                    lines.push('', '`' + c.signature + '`');
                if (c.jsdoc.description)
                    lines.push('', c.jsdoc.description);
                if (c.jsdoc.deprecated)
                    lines.push('', `**Deprecated:** ${c.jsdoc.deprecated}`);
                if ((_c = c.jsdoc.params) === null || _c === void 0 ? void 0 : _c.length) {
                    lines.push('', 'Parameters:');
                    for (const p of c.jsdoc.params)
                        lines.push(`- \`${p.name}\`${p.doc ? ' - ' + p.doc : ''}`);
                }
                if (c.jsdoc.returns)
                    lines.push('', `Returns: ${c.jsdoc.returns}`);
                lines.push('');
            }
        }
    }
    return lines.join('\n').trim() + '\n';
}
function writeIfChanged(file, content) {
    return __awaiter(this, void 0, void 0, function* () {
        if (yield fs_extra_1.default.pathExists(file)) {
            const prev = yield fs_extra_1.default.readFile(file, 'utf8');
            if (prev === content)
                return;
        }
        yield fs_extra_1.default.writeFile(file, content, 'utf8');
    });
}
// Write README into src folders, but avoid overwriting a manual README unless previously generated.
function emitSourceReadme(file, content) {
    return __awaiter(this, void 0, void 0, function* () {
        // Always overwrite to keep docs in sync (educational repo preference: no banner / frictionless reading)
        if (yield fs_extra_1.default.pathExists(file)) {
            const prev = yield fs_extra_1.default.readFile(file, 'utf8');
            if (prev === content)
                return;
        }
        yield fs_extra_1.default.writeFile(file, content, 'utf8');
    });
}
main().catch(e => {
    console.error(e);
    process.exit(1);
});
//# sourceMappingURL=generate-docs.js.map