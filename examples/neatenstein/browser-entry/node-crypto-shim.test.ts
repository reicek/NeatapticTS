import { describe, expect, it } from '@jest/globals';

import { createHash } from './node-crypto-shim';
import { SHA256_ALGORITHM, SHA256_ENCODING } from './constants';

/**
 * Known SHA-256 test vectors from NIST FIPS 180-4 and common references.
 * Verified against `node:crypto` output.
 */
const SHA256_EMPTY = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
const SHA256_ABC = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
const SHA256_HELLO = '2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824';

describe('node-crypto-shim', (): void => {
  describe('createHash', (): void => {
    it('produces the correct SHA-256 digest for the empty string', (): void => {
      const hash = createHash(SHA256_ALGORITHM);
      const result = hash.digest(SHA256_ENCODING);
      expect(result).toBe(SHA256_EMPTY);
    });

    it('produces the correct SHA-256 digest for a single update', (): void => {
      const hash = createHash(SHA256_ALGORITHM);
      hash.update('abc');
      expect(hash.digest(SHA256_ENCODING)).toBe(SHA256_ABC);
    });

    it('produces the correct SHA-256 digest for "hello"', (): void => {
      const hash = createHash(SHA256_ALGORITHM);
      hash.update('hello');
      expect(hash.digest(SHA256_ENCODING)).toBe(SHA256_HELLO);
    });

    it('supports chained multi-update calls that concatenate input', (): void => {
      const hash = createHash(SHA256_ALGORITHM);
      hash.update('he').update('llo');
      expect(hash.digest(SHA256_ENCODING)).toBe(SHA256_HELLO);
    });

    it('returns the same hash instance from update for chaining', (): void => {
      const hash = createHash(SHA256_ALGORITHM);
      const returned = hash.update('data');
      expect(returned).toBe(hash);
    });

    it('throws when an unsupported algorithm is requested', (): void => {
      expect(() => createHash('md5')).toThrow('sha256');
    });

    it('throws when an unsupported encoding is requested', (): void => {
      const hash = createHash(SHA256_ALGORITHM);
      hash.update('data');
      expect(() => hash.digest('base64' as 'hex')).toThrow('hex');
    });
  });
});