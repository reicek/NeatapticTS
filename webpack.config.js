import { fileURLToPath } from 'url';
import path from 'path';

// Define __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const config = {
  mode: 'production',
  entry: './src/neataptic.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'index.js',
    library: 'Neataptic',
    libraryTarget: 'umd',
    globalObject: 'this',
    clean: true,
    environment: {
      arrowFunction: true,
      bigIntLiteral: true,
      const: true,
      destructuring: true,
      dynamicImport: true,
      forOf: true,
      module: true,
      optionalChaining: true,
      templateLiteral: true
    }
  },
  resolve: {
    extensions: ['.ts', '.js'], // Resolve both TypeScript and JavaScript files
    fallback: {
      child_process: false, // Ensure compatibility for browser builds
      os: false,
      path: false,
    },
  },
  module: {
    rules: [
      {
        test: /\.ts$/, // Match TypeScript files
        exclude: /node_modules/, // Exclude node_modules directory
        use: {
          loader: 'ts-loader', // Use ts-loader to transpile TypeScript
          options: {
            transpileOnly: true, // Skip type checking for faster builds
            compilerOptions: {
              module: 'ESNext',
              moduleResolution: 'Bundler',
              target: 'ES2023',
              sourceMap: true,
            },
          },
        },
      },
    ],
  },
  experiments: {
    topLevelAwait: true
  },
  devtool: 'source-map'
};

export default config;
