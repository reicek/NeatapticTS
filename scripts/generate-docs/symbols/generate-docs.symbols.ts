/*
 * Public symbols boundary for the folderized docs generator.
 *
 * This root exposes the stable symbol-facing surface while the detailed
 * collection, normalization, JSDoc parsing, and signature work live in
 * narrower helper chapters inside this folder.
 */

export { collectDirectorySymbols } from './generate-docs.symbols.collection.utils.js';
export { dedupeDirectorySymbols } from './generate-docs.symbols.normalize.utils.js';
export { renderSignatureBlock } from './generate-docs.symbols.signature.utils.js';
