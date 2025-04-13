import { MongoClient } from 'mongodb';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
import tiktoken from 'tiktoken';
import winston from 'winston';

// Configure logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.Console()
    ]
});

// Constants
const timeLimit = 850000; // 850 seconds (14min 10sec) to leave buffer before the 15min limit
const MAX_REQUESTS_PER_MINUTE = 3500;
const REQUEST_DELAY = 60000 / MAX_REQUESTS_PER_MINUTE; // Delay between requests in ms
const MAX_BATCH_SIZE = 8; // Reduced batch size to manage memory better
const MAX_DOCUMENTS_PER_INSERT = 50; // Maximum documents to insert at once

// Configuration
const config = {
    // MongoDB configuration
    mongoUri: process.env.MONGO_URI,
    databaseName: process.env.DB_NAME || 'CORE_POC_DEV',

    // OpenAI configuration
    openAiApiKey: process.env.OPENAI_API_KEY,
    embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
    batchSize: Math.min(parseInt(process.env.BATCH_SIZE || '8'), MAX_BATCH_SIZE), // Reduced default batch size

    // Process ID (from args or generate new one)
    processId: process.env.PROCESS_ID || `embed-process-${uuidv4()}`
};

// Initialize OpenAI client
const openaiClient = axios.create({
    baseURL: 'https://api.openai.com/v1',
    headers: {
        'Authorization': `Bearer ${config.openAiApiKey}`,
        'Content-Type': 'application/json'
    }
});

// Initialize tokenizer
const enc = tiktoken.get_encoding('cl100k_base');

async function getCheckpoint(db) {
    const checkpoint = await db.collection('AI_TEMP').findOne({
        type: 'embedding_checkpoint',
        processId: config.processId
    });
    return checkpoint || { lastProcessedId: null, batchId: null };
}

async function saveCheckpoint(db, lastProcessedId, batchId) {
    await db.collection('AI_TEMP').updateOne(
        {
            type: 'embedding_checkpoint',
            processId: config.processId
        },
        {
            $set: {
                lastProcessedId,
                batchId,
                timestamp: new Date(),
                processId: config.processId
            }
        },
        { upsert: true }
    );
}

async function countTokens(text) {
    return enc.encode(text).length;
}

async function getEmbeddings(texts) {
    try {
        const response = await openaiClient.post('/embeddings', {
            model: config.embeddingModel,
            input: texts
        });
        return response.data.data.map(item => item.embedding);
    } catch (error) {
        logger.error(`OpenAI API error: ${error.response?.data?.error?.message || error.message}`);
        throw error;
    }
}

async function insertEmbeddingsInBatches(db, embeddingDocs) {
    // Split documents into smaller batches for insertion
    for (let i = 0; i < embeddingDocs.length; i += MAX_DOCUMENTS_PER_INSERT) {
        const batch = embeddingDocs.slice(i, i + MAX_DOCUMENTS_PER_INSERT);
        try {
            await db.collection('AI_EMBEDDING').insertMany(batch);
            logger.info(`Inserted batch of ${batch.length} documents`);
        } catch (error) {
            if (error.code === 12501) { // MongoDB space quota error
                logger.warn('Space quota exceeded, reducing batch size and retrying');
                // Try inserting one document at a time
                for (const doc of batch) {
                    try {
                        await db.collection('AI_EMBEDDING').insertOne(doc);
                    } catch (singleDocError) {
                        logger.error(`Failed to insert document: ${singleDocError.message}`);
                        throw singleDocError;
                    }
                }
            } else {
                throw error;
            }
        }
    }
}

async function processBatch(chunks, batchId, db) {
    const texts = chunks.map(chunk => chunk.html_content);
    const embeddings = await getEmbeddings(texts);

    const embeddingDocs = chunks.map((chunk, index) => ({
        batch_id: batchId,
        vector_id: `${chunk.source}__p${chunk.page_counter}__c${chunk.chunk_id}`,
        html_content: chunk.html_content,
        vector: embeddings[index],
        timestamp: new Date(),
        processId: config.processId
    }));

    await insertEmbeddingsInBatches(db, embeddingDocs);
    logger.info(`Processed batch ${batchId} with ${chunks.length} chunks for process ${config.processId}`);
}

async function sendForEmbed() {
    const startTime = Date.now();
    const client = new MongoClient(config.mongoUri);

    try {
        await client.connect();
        const db = client.db(config.databaseName);

        // Get checkpoint
        const checkpoint = await getCheckpoint(db);
        logger.info(`Starting from checkpoint: ${JSON.stringify(checkpoint)} for process ${config.processId}`);

        // Query chunks
        const query = checkpoint.lastProcessedId
            ? { _id: { $gt: checkpoint.lastProcessedId } }
            : {};

        const chunksCursor = db.collection('MATRIX_CHUNKED')
            .find(query)
            .sort({ _id: 1 })
            .batchSize(config.batchSize);

        let currentBatch = [];
        let currentBatchId = checkpoint.batchId || uuidv4();
        let lastProcessedId = checkpoint.lastProcessedId;
        let processedCount = 0;

        while (await chunksCursor.hasNext()) {
            // Check timeout
            if (Date.now() - startTime > timeLimit) {
                logger.info(`Approaching timeout, saving checkpoint for process ${config.processId}`);
                await saveCheckpoint(db, lastProcessedId, currentBatchId);
                return {
                    status: 'timeout',
                    message: 'Process will continue from checkpoint',
                    processedCount,
                    processId: config.processId
                };
            }

            const chunk = await chunksCursor.next();
            lastProcessedId = chunk._id;
            processedCount++;

            // Verify token count
            const tokenCount = await countTokens(chunk.html_content);
            if (tokenCount > 500) {
                logger.warn(`Chunk ${chunk._id} exceeds token limit: ${tokenCount} tokens`);
                continue;
            }

            currentBatch.push(chunk);

            if (currentBatch.length === config.batchSize) {
                await processBatch(currentBatch, currentBatchId, db);
                currentBatch = [];
                currentBatchId = uuidv4();

                // Respect rate limit
                await new Promise(resolve => setTimeout(resolve, REQUEST_DELAY));
            }
        }

        // Process remaining chunks
        if (currentBatch.length > 0) {
            await processBatch(currentBatch, currentBatchId, db);
        }

        // Clear checkpoint if all chunks are processed
        await db.collection('AI_TEMP').deleteOne({
            type: 'embedding_checkpoint',
            processId: config.processId
        });
        logger.info(`All chunks processed successfully for process ${config.processId}`);

        return {
            status: 'success',
            message: 'All chunks processed',
            processedCount,
            processId: config.processId
        };

    } catch (error) {
        logger.error(`Error processing chunks for process ${config.processId}:`, error);
        throw error;
    } finally {
        await client.close();
    }
}

// Digital Ocean Functions entry point
export async function main(req, res) {
    try {
        const result = await sendForEmbed();
        res.status(200).json(result);
    } catch (error) {
        logger.error('Function execution failed:', error);
        res.status(500).json({
            error: error.message,
            processId: config.processId
        });
    }
}