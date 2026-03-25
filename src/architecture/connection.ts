/**
 * Flat compatibility facade for the `Connection` primitive.
 *
 * The `connection/` chapter owns the concrete edge behavior, while this root
 * file keeps the public architecture import short and stable.
 */
export { default } from './connection/connection';
