import { v4 as uuidv4 } from 'uuid';

export const config = {
  // MongoDB
  mongoUri: process.env.MONGO_URI,
  databaseName: process.env.DB_NAME || 'CORE_POC_DEV',

  // OpenAI
  openAiApiKey: process.env.OPENAI_API_KEY,
  embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
  openAiBaseUrl: 'https://api.openai.com/v1',

  // process tracking
  processId: process.env.PROCESS_ID || `embed-process-${uuidv4()}`,

  // Weaviate
  weaviateHost: process.env.WEAVIATE_API_HOST,
  weaviateApiKey: process.env.WEAVIATE_API_KEY
};
