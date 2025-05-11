import MistralClient from '@mistralai/mistralai';

export const createMistralClient = (apiKey: string) => {
  return new MistralClient(apiKey);
};