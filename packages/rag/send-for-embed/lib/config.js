import { v4 as uuidv4 } from 'uuid';

export const MAX_BATCH_SIZE = 50000;

export const config = {
    // MongoDB
    mongoUri: process.env.MONGO_URI,
    databaseName: process.env.DB_NAME || 'CORE_POC_DEV',

    // OpenAI
    openAiApiKey: process.env.OPENAI_API_KEY,
    embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
    openAiBaseUrl: 'https://api.openai.com/v1',
    maxTokenPerBatch: 8191,
    maxJsonlRows: 50000,   // OpenAI hard limit per batch

    // batching
    batchSize: Math.min(
        parseInt(process.env.BATCH_SIZE || '50000', 10),
        MAX_BATCH_SIZE
    ),

    // process tracking
    processId: process.env.PROCESS_ID || `embed-process-${uuidv4()}`
};