import {
  createNeatChatExampleContract,
  createNeatChatSeedNetwork,
  formatNeatChatExampleContract,
} from './index';

const exampleContract = createNeatChatExampleContract();
const seedNetworkResult = createNeatChatSeedNetwork({ vocabularySize: 100 });

console.log(
  formatNeatChatExampleContract(exampleContract, seedNetworkResult.summary),
);
