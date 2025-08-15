import { fileURLToPath } from 'url';
import path from 'path';

// Define __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const config = {
  mode: 'production', // Use 'production' for optimized builds
  entry: './src/neataptic.ts', // Entry point for the application
  output: {
    path: path.resolve(__dirname, 'dist'), // Output directory
    filename: 'index.js', // Output file name
    library: 'Neataptic', // Expose the library globally
    libraryTarget: 'umd', // Universal Module Definition for compatibility with various module systems
    globalObject: 'this', // Ensure compatibility with both Node.js and browser environments
    clean: true, // Clean the output directory before emitting files
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
              module: 'esnext', // Use ESNext module system for output
              target: 'esnext', // Use ESNext target for output
              sourceMap: true, // Enable source maps for debugging
            },
          },
        },
      },
    ],
  },
  devtool: 'source-map', // Enable source maps for debugging
};

export default config;
