import { v4 as uuidv4 } from 'uuid';

export const config = {
  // MongoDB
  mongoUri: process.env.MONGO_URI,
  databaseName: process.env.MONGO_URI,

  // OpenAI
  openAiApiKey: process.env.OPENAI_API_KEY,
  embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
  openAiBaseUrl: 'https://api.openai.com/v1',
  maxEmbeddingsPerBatch: 50000,
  orgTokenLimit: 3000000,
  openAiRateLimit: parseInt(process.env.OPENAI_RATE_LIMIT || '3500', 10),

  // process tracking
  processId: process.env.PROCESS_ID || `embed-process-${uuidv4()}`,

  // Weaviate
  weaviateHost: process.env.WEAVIATE_HOST,
  weaviateApiKey: process.env.WEAVIATE_APIKEY,
};
