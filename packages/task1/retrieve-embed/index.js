import { MongoClient } from 'mongodb';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';
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
const MAX_BATCH_SIZE = 8; // Maximum chunks per batch
const CHECKPOINT_INTERVAL = 100; // Save checkpoint every 100 documents

// Configuration
const config = {
    // MongoDB configuration
    mongoUri: process.env.MONGO_URI,
    databaseName: process.env.DB_NAME || 'CORE_POC_DEV',

    // OpenAI configuration
    openAiApiKey: process.env.OPENAI_API_KEY,
    embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
    batchSize: Math.min(parseInt(process.env.BATCH_SIZE || '8'), MAX_BATCH_SIZE),

    // Process ID (from args or generate new one)
    processId: process.env.PROCESS_ID
};

// Initialize OpenAI client
const openaiClient = axios.create({
    baseURL: 'https://api.openai.com/v1',
    headers: {
        'Authorization': `Bearer ${config.openAiApiKey}`,
        'Content-Type': 'application/json'
    }
});

async function getLastUnfinishedProcessId(db) {
    const lastProcess = await db.collection('AI_TEMP')
        .findOne(
            {
                type: 'retrieve_checkpoint',
                processedCount: { $exists: true } // Ensure it's a process that has started
            },
            { sort: { timestamp: -1 } }
        );

    if (lastProcess) {
        logger.info(`Found last unfinished process ID: ${lastProcess.processId}`);
        return lastProcess.processId;
    }

    return null;
}

async function getProcessId(db) {
    // First try to use the process ID from environment variable
    if (config.processId) {
        logger.info(`Using process ID from environment: ${config.processId}`);
        return config.processId;
    }

    // If no process ID in env, try to get the last unfinished process
    const lastProcessId = await getLastUnfinishedProcessId(db);
    if (lastProcessId) {
        return lastProcessId;
    }

    // If no unfinished process exists, create a new one
    const newProcessId = `retrieve-process-${uuidv4()}`;
    logger.info(`No previous process found, created new process ID: ${newProcessId}`);
    return newProcessId;
}

async function getCheckpoint(db, processId) {
    const checkpoint = await db.collection('AI_TEMP').findOne({
        type: 'retrieve_checkpoint',
        processId: processId
    });
    return checkpoint || {
        lastProcessedId: null,
        batchId: null,
        processedCount: 0,
        lastSource: null
    };
}

async function saveCheckpoint(db, checkpoint, processId) {
    await db.collection('AI_TEMP').updateOne(
        {
            type: 'retrieve_checkpoint',
            processId: processId
        },
        {
            $set: {
                ...checkpoint,
                timestamp: new Date(),
                processId: processId
            }
        },
        { upsert: true }
    );
}

async function getPendingBatches(db, lastProcessedId) {
    const query = lastProcessedId
        ? { _id: { $gt: lastProcessedId } }
        : {};

    return await db.collection('AI_EMBEDDING')
        .aggregate([
            { $match: { ...query, timestamp: { $exists: false } } },
            { $group: { _id: '$batch_id' } },
            { $sort: { _id: 1 } }
        ])
        .toArray();
}

async function getBatchDocuments(db, batchId) {
    return await db.collection('AI_EMBEDDING')
        .find({ batch_id: batchId, timestamp: { $exists: false } })
        .toArray();
}

async function updateEmbeddingDocuments(db, documents, processId) {
    const bulkOps = documents.map(doc => ({
        updateOne: {
            filter: { _id: doc._id },
            update: {
                $set: {
                    timestamp: new Date(),
                    processId: processId
                }
            }
        }
    }));

    if (bulkOps.length > 0) {
        await db.collection('AI_EMBEDDING').bulkWrite(bulkOps);
    }
}

async function getOriginalChunk(db, vectorId) {
    const [source, pageCounter, chunkId] = vectorId.split('__').map(part => part.replace(/^[pc]/, ''));
    return await db.collection('MATRIX_CHUNKED').findOne({
        source,
        page_counter: parseInt(pageCounter),
        chunk_id: parseInt(chunkId)
    });
}

async function createRagDocument(embeddingDoc, originalChunk) {
    return {
        id: embeddingDoc.vector_id,
        vector: embeddingDoc.vector,
        properties: {
            chunk_id: originalChunk.chunk_id,
            page_counter: originalChunk.page_counter,
            source: originalChunk.source,
            rag_timestamp: null, // Will be updated after Weaviate insertion
            html_content: originalChunk.html_content,
            metadata: originalChunk.metadata
        }
    };
}

async function getEmbeddingsFromOpenAI(texts) {
    try {
        const response = await openaiClient.post('/embeddings', {
            input: texts,
            model: config.embeddingModel
        });

        return response.data.data.map(item => item.embedding);
    } catch (error) {
        logger.error('Error getting embeddings from OpenAI:', error);
        throw error;
    }
}

async function processBatch(db, batchId, checkpoint, processId) {
    logger.info(`Processing batch ${batchId} for process ${processId}`);

    const documents = await getBatchDocuments(db, batchId);
    if (documents.length === 0) {
        logger.info(`No documents found for batch ${batchId}`);
        return checkpoint;
    }

    try {
        // Get original chunks for the batch
        const originalChunks = await Promise.all(
            documents.map(doc => getOriginalChunk(db, doc.vector_id))
        );

        // Filter out any missing chunks
        const validChunks = originalChunks.filter(chunk => chunk !== null);
        if (validChunks.length === 0) {
            logger.warn(`No valid chunks found for batch ${batchId}`);
            return checkpoint;
        }

        // Get embeddings from OpenAI
        const texts = validChunks.map(chunk => chunk.html_content);
        const embeddings = await getEmbeddingsFromOpenAI(texts);

        // Update documents with embeddings and timestamps
        const updatedDocuments = documents.map((doc, index) => {
            if (index < embeddings.length) {
                return {
                    ...doc,
                    vector: embeddings[index],
                    timestamp: new Date(),
                    processId: processId
                };
            }
            return doc;
        });

        // Update documents in MongoDB
        await updateEmbeddingDocuments(db, updatedDocuments, processId);

        // Update checkpoint
        const lastDocument = updatedDocuments[updatedDocuments.length - 1];
        checkpoint.lastProcessedId = lastDocument._id;
        checkpoint.processedCount += updatedDocuments.length;
        checkpoint.lastSource = lastDocument.source;

        // Save checkpoint periodically
        if (checkpoint.processedCount % CHECKPOINT_INTERVAL === 0) {
            await saveCheckpoint(db, checkpoint, processId);
            logger.info(`Saved checkpoint at ${checkpoint.processedCount} documents`);
        }

        // Process each document for RAG
        for (let i = 0; i < updatedDocuments.length; i++) {
            const embeddingDoc = updatedDocuments[i];
            const originalChunk = validChunks[i];

            try {
                // Create RAG document
                const ragDoc = await createRagDocument(embeddingDoc, originalChunk);

                // Insert into AI_FOR_RAG collection
                await db.collection('AI_FOR_RAG').insertOne(ragDoc);
                logger.info(`Inserted RAG document for ${embeddingDoc.vector_id}`);
            } catch (error) {
                logger.error(`Error processing document ${embeddingDoc.vector_id}:`, error);
            }
        }

        return checkpoint;
    } catch (error) {
        logger.error(`Error processing batch ${batchId}:`, error);
        throw error;
    }
}

async function retrieveEmbed() {
    const startTime = Date.now();
    const client = new MongoClient(config.mongoUri);

    try {
        await client.connect();
        const db = client.db(config.databaseName);

        // Get process ID based on priority
        const processId = await getProcessId(db);

        // Get checkpoint
        let checkpoint = await getCheckpoint(db, processId);
        logger.info(`Starting from checkpoint: ${JSON.stringify(checkpoint)}`);

        // Get pending batches
        const pendingBatches = await getPendingBatches(db, checkpoint.lastProcessedId);
        logger.info(`Found ${pendingBatches.length} pending batches`);

        for (const batch of pendingBatches) {
            // Check timeout
            if (Date.now() - startTime > timeLimit) {
                logger.info(`Approaching timeout, saving checkpoint for process ${processId}`);
                await saveCheckpoint(db, checkpoint, processId);
                return {
                    status: 'timeout',
                    message: 'Process will continue from checkpoint',
                    processedCount: checkpoint.processedCount,
                    processId: processId
                };
            }

            checkpoint = await processBatch(db, batch._id, checkpoint, processId);
        }

        // Clear checkpoint if all batches are processed
        await db.collection('AI_TEMP').deleteOne({
            type: 'retrieve_checkpoint',
            processId: processId
        });
        logger.info(`All batches processed successfully for process ${processId}`);

        return {
            status: 'success',
            message: 'All batches processed',
            processedCount: checkpoint.processedCount,
            processId: processId
        };

    } catch (error) {
        logger.error(`Error processing batches for process ${processId}:`, error);
        throw error;
    } finally {
        await client.close();
    }
}

// Digital Ocean Functions entry point
export async function main(req, res) {
    try {
        const result = await retrieveEmbed();
        res.status(200).json(result);
    } catch (error) {
        logger.error('Function execution failed:', error);
        res.status(500).json({
            error: error.message,
            processId: processId // Note: This will be undefined if main() failed before setting processId
        });
    }
}