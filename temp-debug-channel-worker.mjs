function printError(error) {
  console.log('type:', typeof error);
  console.log('prototype:', Object.getPrototypeOf(error));
  console.log('names:', Object.getOwnPropertyNames(error));
  console.log('symbols:', Object.getOwnPropertySymbols(error));

  for (const propertyName of Object.getOwnPropertyNames(error)) {
    console.log(propertyName + ':', error[propertyName]);
  }

  for (const propertySymbol of Object.getOwnPropertySymbols(error)) {
    console.log(propertySymbol.toString() + ':', error[propertySymbol]);

    if (typeof error[propertySymbol] === 'function') {
      console.log(
        propertySymbol.toString() + '() :',
        error[propertySymbol](),
      );
    }
  }

  console.dir(error, { depth: 6, showHidden: true });
}

process.on('uncaughtException', (error) => {
  printError(error);
  process.exit(1);
});

try {
  await import('./src/architecture/network/worker-payload/network.worker-payload.channel.worker.ts');
  console.log('loaded');
} catch (error) {
  printError(error);
  process.exit(1);
}
