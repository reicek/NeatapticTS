// Pure ESM Jest configuration.
// We avoid CommonJS entirely; Jest 29+ supports loading ESM config (.mjs).

const experimentalVmModulesFlag = '--experimental-vm-modules';
if (!process.env.NODE_OPTIONS?.includes(experimentalVmModulesFlag)) {
  process.env.NODE_OPTIONS = `${process.env.NODE_OPTIONS ?? ''} ${experimentalVmModulesFlag}`.trim();
}

/** @type {import('jest').Config} */
const config = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  // Use jsdom for asciiMaze browser example tests (pattern matching)
  testEnvironmentOptions: {},
  projects: [
    {
      displayName: 'default',
      testMatch: [
        '**/src/**/*.test.ts',
        '**/benchmarks/**/*.test.ts',
        '**/examples/**/*.test.ts',
      ],
      preset: 'ts-jest/presets/default-esm',
      testEnvironment: 'node',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
        ]
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 300000,
      collectCoverageFrom: [
        'src/**/*.ts',
        '!src/**/*.d.ts',
        '!src/**/*.test.ts',
      ],
      coverageReporters: ['lcov', 'text', 'html', 'json-summary'],
      testPathIgnorePatterns: [
        '/node_modules/',
        '/dist/',
        '/examples/asciiMaze/',
         '/examples/starter-examples.smoke.test.ts',
      ],
    },
    {
      displayName: 'asciiMaze-browser',
      testMatch: [
        '**/examples/asciiMaze/**/*.test.ts',
      ],
        preset: 'ts-jest/presets/default-esm',
        testEnvironment: 'jsdom',
      extensionsToTreatAsEsm: ['.ts'],
      transform: {
        '^.+\\.ts$': [
          'ts-jest',
          { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
        ]
      },
      setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
      testTimeout: 3000000,
    },
      {
        displayName: 'starter-examples',
        testMatch: [
          '**/examples/starter-examples.smoke.test.ts',
        ],
        preset: 'ts-jest/presets/default-esm',
        testEnvironment: 'node',
        extensionsToTreatAsEsm: ['.ts'],
        transform: {
          '^.+\\.ts$': [
            'ts-jest',
            { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
          ]
        },
        setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
        testTimeout: 300000,
      },
      {
        displayName: 'semantic-index-scripts',
        testMatch: [
          '**/scripts/semantic-index/**/*.test.ts',
        ],
        preset: 'ts-jest/presets/default-esm',
        testEnvironment: 'node',
        extensionsToTreatAsEsm: ['.ts'],
        transform: {
          '^.+\\.ts$': [
            'ts-jest',
            { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
          ]
        },
        setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
        testTimeout: 300000,
      },
      {
        displayName: 'agent-customization-scripts',
        testMatch: [
          '**/scripts/agent-customization/**/*.test.ts',
        ],
        preset: 'ts-jest/presets/default-esm',
        testEnvironment: 'node',
        extensionsToTreatAsEsm: ['.ts'],
        transform: {
          '^.+\\.ts$': [
            'ts-jest',
            { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
          ]
        },
        setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
        testTimeout: 300000,
      },
      {
        displayName: 'mcp-semantic-scripts',
        testMatch: [
          '**/scripts/mcp-semantic/**/*.test.ts',
        ],
        preset: 'ts-jest/presets/default-esm',
        testEnvironment: 'node',
        extensionsToTreatAsEsm: ['.ts'],
        transform: {
          '^.+\\.ts$': [
            'ts-jest',
            { useESM: true, tsconfig: 'tsconfig.test.json', diagnostics: true }
          ]
        },
        setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
        testTimeout: 300000,
      },
  ],
  testMatch: [
    '**/src/**/*.test.ts',
    '**/benchmarks/**/*.test.ts',
    '**/examples/**/*.test.ts',
  ],
  moduleFileExtensions: ['ts', 'js', 'mjs', 'cjs', 'json'],
  extensionsToTreatAsEsm: ['.ts'],
  transform: {
    '^.+\\.ts$': [
      'ts-jest',
      {
        useESM: true,
        tsconfig: 'tsconfig.test.json',
        diagnostics: true
      }
    ]
  },
  setupFilesAfterEnv: ['<rootDir>/testing/jest-setup.ts'],
  testTimeout: 300000,
  collectCoverageFrom: ['src/**/*.ts', '!src/**/*.d.ts', '!src/**/*.test.ts'],
  coverageReporters: ['lcov', 'text', 'html', 'json-summary'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/']
};

export default config;
