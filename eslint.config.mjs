import { defineConfig, globalIgnores  } from 'eslint/config';
import typescriptEslint from '@typescript-eslint/eslint-plugin';
import preferArrow from 'eslint-plugin-prefer-arrow';
import tsParser from '@typescript-eslint/parser';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import js from '@eslint/js';
import { FlatCompat } from '@eslint/eslintrc';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const compat = new FlatCompat({
  baseDirectory: __dirname,
  recommendedConfig: js.configs.recommended,
  allConfig: js.configs.all,
});

export default defineConfig([
  globalIgnores([
    'ascii_maze_snapshots/',
    'bench-browser/',
    'coverage/',
    'dist/',
    'dist-docs/',
    'docs/',
    'node_modules/',
    'plans/',
    'scripts/',
  ]),
  {
    extends: compat.extends(
      'eslint:recommended',
      'plugin:@typescript-eslint/recommended'
    ),

    plugins: {
      '@typescript-eslint': typescriptEslint,
      'prefer-arrow': preferArrow,
    },

    languageOptions: {
      parser: tsParser,
    },

    rules: {
      'no-var': 'error',
      'prefer-const': 'error',
      'prefer-arrow/prefer-arrow-functions': 'error',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
    },
  },
]);
