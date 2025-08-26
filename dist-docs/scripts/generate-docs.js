import { Project } from 'ts-morph';
import fg from 'fast-glob';
import * as path from 'path';
import fs from 'fs-extra';
const SRC_DIR = path.resolve('src');
const DOCS_DIR = path.resolve('docs');
const ROOT_README_SRC = path.resolve('README.md');
const ROOT_README_DEST = path.join(DOCS_DIR, 'README.md');
const project = new Project({
    tsConfigFilePath: path.resolve('tsconfig.json'),
    skipAddingFilesFromTsConfig: true
});
async function main() {
    await fs.ensureDir(DOCS_DIR);
    if (await fs.pathExists(ROOT_README_SRC)) {
        await fs.copyFile(ROOT_README_SRC, ROOT_README_DEST);
    }
    const filePaths = await fg(['**/*.ts'], { cwd: SRC_DIR, absolute: true, ignore: ['**/*.d.ts'] });
    for (const p of filePaths) {
        if (/\.test\.ts$/i.test(p))
            continue;
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
        const fileJsDocs = sf.getJsDocs?.() || [];
        const fileDocPrimary = fileJsDocs[0];
        const fileDocDesc = fileDocPrimary?.getDescription?.()?.trim();
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
                    try {
                        if (decl.getKindName && decl.getKindName() === 'VariableDeclaration') {
                            const init = decl.getInitializer?.();
                            if (init && init.getKindName && init.getKindName() === 'ObjectLiteralExpression') {
                                const props = init.getProperties?.() || [];
                                for (const p of props) {
                                    const propName = p.getName?.() || (p.getSymbol && p.getSymbol()?.getName?.());
                                    const renderedProp = renderDeclaration(p, String(propName), p.getKindName?.() || 'Property', sf.getFilePath(), sym.getName());
                                    if (renderedProp) {
                                        const arr2 = fileMap.get(sf.getFilePath()) || [];
                                        arr2.push(renderedProp);
                                        fileMap.set(sf.getFilePath(), arr2);
                                    }
                                }
                            }
                        }
                    }
                    catch (e) { }
                    try {
                        if (decl.getKindName && decl.getKindName() === 'ClassDeclaration') {
                            const members = decl.getMembers?.() || [];
                            for (const m of members) {
                                const memberName = m.getName?.();
                                if (!memberName)
                                    continue;
                                const renderedMember = renderDeclaration(m, String(memberName), m.getKindName?.() || 'ClassMember', sf.getFilePath(), sym.getName());
                                if (renderedMember) {
                                    const arr3 = fileMap.get(sf.getFilePath()) || [];
                                    arr3.push(renderedMember);
                                    fileMap.set(sf.getFilePath(), arr3);
                                }
                            }
                        }
                    }
                    catch (e) { }
                }
            }
        }
        try {
            const stmts = sf.getStatements?.() || [];
            for (const st of stmts) {
                const decls = (st.getDeclarations && st.getDeclarations()) || [st];
                for (const d of decls) {
                    const jsdocs = d.getJsDocs?.() || [];
                    const hasExport = (d.getSymbol && d.getSymbol()?.getDeclarations?.()?.some((x) => x.isExported && x.isExported())) || false;
                    if (!jsdocs.length)
                        continue;
                    const name = d.getName?.() || (d.getSymbol && d.getSymbol()?.getName?.());
                    if (!name)
                        continue;
                    const already = (fileMap.get(sf.getFilePath()) || []).some(s => s.name === name || s.name === `${name}` || s.name === `${name}`);
                    if (already)
                        continue;
                    const rendered = renderDeclaration(d, String(name), d.getKindName?.() || 'Declaration', sf.getFilePath());
                    if (rendered) {
                        const arr4 = fileMap.get(sf.getFilePath()) || [];
                        arr4.push(rendered);
                        fileMap.set(sf.getFilePath(), arr4);
                    }
                }
            }
        }
        catch (e) { }
    }
    for (const [dir, fileMap] of dirMap) {
        for (const [filePath, symbols] of fileMap) {
            const seen = new Map();
            const deduped = [];
            for (const s of symbols) {
                const normName = normalizeName(s, filePath);
                const key = `${s.parent || ''}::${normName}::${s.signature || ''}`;
                const existing = seen.get(key);
                if (existing) {
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
                    const clone = { ...s, name: normName, jsdoc: { ...s.jsdoc, params: s.jsdoc.params ? s.jsdoc.params.slice() : undefined } };
                    seen.set(key, clone);
                    deduped.push(clone);
                }
            }
            fileMap.set(filePath, deduped);
        }
        const relDir = path.relative(SRC_DIR, dir);
        if (relDir.startsWith('..'))
            continue;
        const outDir = relDir === '' ? path.join(DOCS_DIR, 'src') : path.join(DOCS_DIR, relDir);
        await fs.ensureDir(outDir);
        const outFile = path.join(outDir, 'README.md');
        const md = buildDirectoryReadme(relDir, fileMap);
        await writeIfChanged(outFile, md);
        const srcTargetDir = path.join(SRC_DIR, relDir);
        const srcReadme = path.join(srcTargetDir, 'README.md');
        await emitSourceReadme(srcReadme, md);
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
    renderNode(rootNode, 0);
    await fs.writeFile(path.join(DOCS_DIR, 'FOLDERS.md'), lines.join('\n') + '\n', 'utf8');
    console.log('Per-folder README generation complete.');
}
function renderSymbol(sym, fallbackKind, filePath) {
    const decl = sym.getDeclarations()[0];
    if (!decl)
        return null;
    const kind = decl.getKindName();
    const allow = /ClassDeclaration|FunctionDeclaration|InterfaceDeclaration|EnumDeclaration|TypeAliasDeclaration|VariableDeclaration/;
    if (!allow.test(kind))
        return null;
    const jsDocs = decl.getJsDocs?.() || [];
    if (jsDocs.some((j) => j.getTags().some((t) => t.getTagName() === 'internal')))
        return null;
    const primary = jsDocs[0];
    const fullDesc = primary?.getDescription().trim();
    const summary = fullDesc?.split(/\r?\n\r?\n/)[0]?.trim();
    const tags = primary?.getTags() || [];
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
    catch { }
    const params = paramsTags.map(t => {
        const text = t.getText();
        const match = text.match(/@param\s+(\w+)/);
        const name = match?.[1] || '';
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
            returns: (() => { const r = returnsTag?.getComment(); return typeof r === 'string' ? r.trim() : undefined; })(),
            deprecated: (() => { const r = deprecatedTag?.getComment(); return typeof r === 'string' ? r.trim() : undefined; })()
        }
    };
}
function renderDeclaration(decl, forcedName, forcedKind, filePath, parentName) {
    if (!decl)
        return null;
    try {
        const kind = forcedKind || decl.getKindName?.() || decl.getKind?.() || 'Declaration';
        const name = forcedName || decl.getName?.() || (decl.getSymbol && decl.getSymbol()?.getName?.()) || (parentName ? `${parentName}.${forcedName}` : '');
        const jsDocs = decl.getJsDocs?.() || [];
        if (!jsDocs.length)
            return null;
        if (jsDocs.some((j) => j.getTags().some((t) => t.getTagName() === 'internal')))
            return null;
        const primary = jsDocs[0];
        const fullDesc = primary?.getDescription?.()?.trim();
        const summary = fullDesc?.split(/\r?\n\r?\n/)[0]?.trim();
        const tags = primary?.getTags() || [];
        const paramsTags = tags.filter(t => t.getTagName() === 'param');
        const returnsTag = tags.find(t => t.getTagName() === 'returns' || t.getTagName() === 'return');
        const deprecatedTag = tags.find(t => t.getTagName() === 'deprecated');
        let signature;
        try {
            const type = decl.getType?.() || (decl.getSymbol && decl.getSymbol()?.getType?.());
            const sig = type?.getCallSignatures?.()[0];
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
        catch { }
        const params = paramsTags.map(t => {
            const text = t.getText?.() || '';
            const match = text.match(/@param\s+(\w+)/);
            const name = match?.[1] || '';
            const raw = t.getComment?.();
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
                returns: (() => { const r = returnsTag?.getComment?.(); return typeof r === 'string' ? r.trim() : undefined; })(),
                deprecated: (() => { const r = deprecatedTag?.getComment?.(); return typeof r === 'string' ? r.trim() : undefined; })()
            }
        };
    }
    catch (e) {
        return null;
    }
}
function normalizeName(s, filePath) {
    let name = (s.name || '').toString();
    if (!name || name === 'default' || name === '__file_summary__') {
        const base = path.basename(filePath || '', '.ts');
        if (s.kind === 'File' || name === '__file_summary__')
            return base;
        if (s.parent)
            return `${s.parent}.${base}`;
        if (s.signature)
            return `${base}${s.signature.split(')')[0]})`;
        return base;
    }
    name = name.replace(/^function\s+/, '').replace(/\(\)$/, '');
    return name;
}
function buildDirectoryReadme(relDir, fileMap) {
    const title = relDir.replace(/\\/g, '/');
    const lines = [
        `# ${title}`,
        ''
    ];
    const filesSorted = [...fileMap.keys()].sort();
    for (const file of filesSorted) {
        const relFile = path.relative(SRC_DIR, file).replace(/\\/g, '/');
        lines.push(`## ${relFile}`, '');
        const symbols = fileMap.get(file).toSorted((a, b) => (a.parent || a.name).localeCompare(b.parent || b.name) || a.name.localeCompare(b.name));
        const fileSummaryIdx = symbols.findIndex(s => s.name === '__file_summary__' && s.kind === 'File');
        if (fileSummaryIdx >= 0) {
            const fsym = symbols.splice(fileSummaryIdx, 1)[0];
            if (fsym.jsdoc.description)
                lines.push(fsym.jsdoc.description, '');
        }
        const topLevel = symbols.filter(s => !s.parent);
        const byParent = new Map();
        for (const s of symbols.filter(s => s.parent)) {
            const arr = byParent.get(s.parent) || [];
            arr.push(s);
            byParent.set(s.parent, arr);
        }
        for (const s of topLevel) {
            lines.push(`### ${s.name}`);
            if (s.signature)
                lines.push('', '`' + s.signature + '`');
            if (s.jsdoc.description)
                lines.push('', s.jsdoc.description);
            if (s.jsdoc.deprecated)
                lines.push('', `**Deprecated:** ${s.jsdoc.deprecated}`);
            if (s.jsdoc.params?.length) {
                lines.push('', 'Parameters:');
                for (const p of s.jsdoc.params)
                    lines.push(`- \`${p.name}\`${p.doc ? ' - ' + p.doc : ''}`);
            }
            if (s.jsdoc.returns)
                lines.push('', `Returns: ${s.jsdoc.returns}`);
            lines.push('');
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
                    if (c.jsdoc.params?.length) {
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
        for (const [parentName, group] of byParent) {
            if (topLevel.some(t => t.name === parentName))
                continue;
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
                if (c.jsdoc.params?.length) {
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
async function writeIfChanged(file, content) {
    if (await fs.pathExists(file)) {
        const prev = await fs.readFile(file, 'utf8');
        if (prev === content)
            return;
    }
    await fs.writeFile(file, content, 'utf8');
}
async function emitSourceReadme(file, content) {
    if (await fs.pathExists(file)) {
        const prev = await fs.readFile(file, 'utf8');
        if (prev === content)
            return;
    }
    await fs.writeFile(file, content, 'utf8');
}
main().catch(e => {
    console.error(e);
    process.exit(1);
});
//# sourceMappingURL=generate-docs.js.map