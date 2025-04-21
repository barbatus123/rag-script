import { MongoClient } from 'mongodb';
import { config } from './config.js';
import { logger } from './logger.js';

let cachedClient;
export async function getMongo() {
  if (cachedClient?.topology?.isConnected?.()) return cachedClient;
  cachedClient = new MongoClient(config.mongoUri, {
    maxPoolSize: 10,
    retryWrites: true,
  });
  try {
    await cachedClient.connect();
    logger.info({ uri: config.mongoUri.split('@').pop() }, 'Mongo connected');
  } catch (err) {
    logger.error({ err }, 'Mongo connection failed');
    throw err;
  }
  return cachedClient;
}

export function collections(db) {
  return {
    chunks: db.collection('MATRIX_CHUNKED'),
    embIndex: db.collection('AI_EMBEDDING'),
    ragReady: db.collection('AI_FOR_RAG'),
    signal: db.collection('SIGNAL_RAG'),
    temp: db.collection('AI_TEMP'),
  };
}
