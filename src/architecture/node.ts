/**
 * Flat compatibility facade for the `Node` primitive.
 *
 * Read the `node/` chapter when you want the activation, propagation, and
 * trace semantics; use this file when you only need the stable root export.
 */
export { default, type PrimitiveNodeType } from './node/node';
export {
	type PrimitiveDescriptor,
	type PrimitiveIntent,
	type PrimitiveMetadata,
	type PrimitiveMetadataValue,
	resolvePrimitiveIntent,
} from './node/node';
