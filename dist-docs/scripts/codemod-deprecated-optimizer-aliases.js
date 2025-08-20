#!/usr/bin/env ts-node
/**
 * Codemod: Replace deprecated optimizer alias & node gating properties.
 * Usage: npx ts-node scripts/codemod-deprecated-optimizer-aliases.ts [--dry]
 */
import { Project, SyntaxKind } from 'ts-morph';
import path from 'path';
import fs from 'fs';
const DRY = process.argv.includes('--dry');
const rootTsconfig = path.join(process.cwd(), 'tsconfig.json');
if (!fs.existsSync(rootTsconfig)) {
    console.error('tsconfig.json not found at repo root. Aborting.');
    process.exit(1);
}
const mapping = {
    opt_m: 'firstMoment',
    opt_v: 'secondMoment',
    opt_cache: 'gradientAccumulator',
    opt_vhat: 'maxSecondMoment',
    opt_u: 'infinityNorm',
    opt_m2: 'secondMomentum',
    _la_shadowWeight: 'lookaheadShadowWeight',
};
// node.gates & node.nodes handled specially
const project = new Project({ tsConfigFilePath: rootTsconfig });
const sourceFiles = project.getSourceFiles(['src/**/*.ts', 'test/**/*.ts']);
let changedFiles = 0;
let totalRewrites = 0;
for (const sf of sourceFiles) {
    let fileChanged = false;
    sf.forEachDescendant((node) => {
        if (node.getKind() !== SyntaxKind.PropertyAccessExpression)
            return;
        const pae = node; // PropertyAccessExpression
        const name = pae.getName?.();
        if (!name)
            return;
        // Simple mapping replacements
        if (mapping[name]) {
            pae.getNameNode().replaceWithText(mapping[name]);
            pae.replaceWithText(pae.getExpression().getText() + '.' + mapping[name]);
            fileChanged = true;
            totalRewrites++;
            return;
        }
        // node.gates -> node.connections.gated (exclude Network.gates)
        if (name === 'gates') {
            // Best-effort: check type text includes 'Node' (heuristic)
            try {
                const typeText = pae.getExpression().getType().getText();
                if (typeText.includes('Node')) {
                    pae.replaceWithText(pae.getExpression().getText() + '.connections.gated');
                    fileChanged = true;
                    totalRewrites++;
                }
            }
            catch { }
            return;
        }
        // node.nodes -> [] (comment to prompt manual migration)
        if (name === 'nodes') {
            try {
                const typeText = pae.getExpression().getType().getText();
                if (typeText.includes('Node')) {
                    pae.replaceWithText('/* deprecated node.nodes */ []');
                    fileChanged = true;
                    totalRewrites++;
                }
            }
            catch { }
        }
    });
    if (fileChanged) {
        changedFiles++;
    }
}
if (!DRY)
    project.saveSync();
console.log(`Codemod complete. Files changed: ${changedFiles}, rewrites: ${totalRewrites}${DRY ? ' (dry run)' : ''}`);
//# sourceMappingURL=codemod-deprecated-optimizer-aliases.js.map