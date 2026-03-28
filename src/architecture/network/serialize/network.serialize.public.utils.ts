import type Network from '../../network/network';

type CloneableNetworkConstructor = {
  fromJSON: (json: Record<string, unknown>) => Network;
};

/**
 * Public serialization-facing helpers that stay above the lower-level payload builders.
 *
 * This file owns small convenience methods that callers expect on `Network`
 * itself, while delegating the real persistence work to the serialize chapter.
 */

/**
 * Create a deep copy of one network through the verbose JSON round-trip.
 *
 * This keeps cloning behavior aligned with the same versioned payload contract
 * used by `toJSON()` and `fromJSON()`, so clone semantics stay stable as the
 * serialization chapter evolves.
 *
 * @param this Target network instance.
 * @returns Deep-cloned network instance.
 */
export function cloneImpl(this: Network): Network {
  const networkConstructor = this
    .constructor as unknown as CloneableNetworkConstructor;
  return networkConstructor.fromJSON(this.toJSON());
}
