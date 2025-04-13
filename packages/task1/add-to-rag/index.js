import { MongoClient } from 'mongodb';
import { WeaviateClient } from 'weaviate-ts-client';
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
const MAX_DOCUMENTS_PER_BATCH = 20; // Reduced batch size to manage memory better
const MAX_VECTORS_PER_REQUEST = 10; // Maximum vectors to process in one Weaviate request
const CHECKPOINT_INTERVAL = 100; // Save checkpoint every 100 documents

// Collection mapping
const COLLECTION_MAPPING = {
    'website_structure': 'WebsiteStructure',
    'script_example': 'ScriptExample',
    'sdk_documentation': 'SDKDocumentation'
};

// Configuration
const config = {
    // MongoDB configuration
    mongoUri: process.env.MONGO_URI,
    databaseName: process.env.DB_NAME || 'CORE_POC_DEV',

    // Weaviate configuration
    weaviateUrl: process.env.WEAVIATE_URL,
    weaviateApiKey: process.env.WEAVIATE_API_KEY,
    weaviateCluster: process.env.WEAVIATE_CLUSTER || 'ladeston-core-cluster',

    // Process ID (from args or generate new one)
    processId: process.env.PROCESS_ID
};

// Initialize Weaviate client
const weaviateClient = new WeaviateClient({
    scheme: 'https',
    host: config.weaviateUrl,
    apiKey: config.weaviateApiKey,
    headers: {
        'X-OpenAI-Api-Key': config.openAiApiKey
    }
});

async function getLastUnfinishedProcessId(db) {
    const lastProcess = await db.collection('AI_TEMP')
        .findOne(
            {
                type: 'rag_checkpoint',
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
    const newProcessId = `rag-process-${uuidv4()}`;
    logger.info(`No previous process found, created new process ID: ${newProcessId}`);
    return newProcessId;
}

async function getCheckpoint(db, processId) {
    const checkpoint = await db.collection('AI_TEMP').findOne({
        type: 'rag_checkpoint',
        processId: processId
    });
    return checkpoint || {
        lastProcessedId: null,
        processedCount: 0,
        lastSource: null,
        lastDataType: null
    };
}

async function saveCheckpoint(db, checkpoint, processId) {
    await db.collection('AI_TEMP').updateOne(
        {
            type: 'rag_checkpoint',
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

async function getPendingDocuments(db, lastProcessedId, limit) {
    const query = lastProcessedId
        ? { _id: { $gt: lastProcessedId } }
        : {};

    return await db.collection('AI_FOR_RAG')
        .find({
            ...query,
            'properties.rag_timestamp': null
        })
        .sort({ _id: 1 })
        .limit(limit)
        .toArray();
}

async function updateRagTimestamp(db, documentId, timestamp) {
    await db.collection('AI_FOR_RAG').updateOne(
        { _id: documentId },
        { $set: { 'properties.rag_timestamp': timestamp } }
    );
}

async function createWeaviateObjects(documents) {
    const results = [];
    const batches = [];

    // Group documents by collection type
    for (const doc of documents) {
        const collectionName = COLLECTION_MAPPING[doc.properties.metadata.data_type];
        if (!collectionName) {
            throw new Error(`Unknown data type: ${doc.properties.metadata.data_type}`);
        }

        if (!batches[collectionName]) {
            batches[collectionName] = [];
        }
        batches[collectionName].push(doc);
    }

    // Process each collection type
    for (const [collectionName, docs] of Object.entries(batches)) {
        // Split into smaller batches for memory management
        for (let i = 0; i < docs.length; i += MAX_VECTORS_PER_REQUEST) {
            const batch = docs.slice(i, i + MAX_VECTORS_PER_REQUEST);
            const batchResults = await Promise.all(
                batch.map(async (doc) => {
                    const dataObject = {
                        chunk_id: doc.properties.chunk_id,
                        page_counter: doc.properties.page_counter,
                        source: doc.properties.source,
                        rag_timestamp: new Date().toISOString(),
                        html_content: doc.properties.html_content,
                        metadata_data_type: doc.properties.metadata.data_type,
                        metadata_timestamp: doc.properties.metadata.timestamp,
                        metadata_crawl_depth: doc.properties.metadata.crawl_depth,
                        metadata_title: doc.properties.metadata.title
                    };

                    try {
                        await weaviateClient.data
                            .creator()
                            .withClassName(collectionName)
                            .withProperties(dataObject)
                            .withVector(doc.vector)
                            .withId(doc.id)
                            .do();

                        logger.info(`Inserted document ${doc.id} into Weaviate collection ${collectionName}`);
                        return { success: true, doc };
                    } catch (error) {
                        logger.error(`Failed to insert document ${doc.id} into Weaviate:`, error);
                        return { success: false, doc, error };
                    }
                })
            );
            results.push(...batchResults);
        }
    }

    return results;
}

async function checkLastDocumentForSource(db, source) {
    const lastDocument = await db.collection('AI_FOR_RAG')
        .findOne(
            { 'properties.source': source },
            { sort: { 'properties.page_counter': -1, 'properties.chunk_id': -1 } }
        );

    const pendingDocuments = await db.collection('AI_FOR_RAG')
        .countDocuments({
            'properties.source': source,
            'properties.rag_timestamp': null
        });

    return {
        isLastDocument: lastDocument && pendingDocuments === 0,
        dataType: lastDocument?.properties?.metadata?.data_type
    };
}

async function createSignalRag(db, source, dataType) {
    const collectionName = COLLECTION_MAPPING[dataType];
    if (!collectionName) {
        throw new Error(`Unknown data type: ${dataType}`);
    }

    await db.collection('SIGNAL_RAG').insertOne({
        source,
        collection: collectionName,
        timestamp: new Date()
    });

    logger.info(`Created signal for source ${source} in collection ${collectionName}`);
}

async function processDocuments(db) {
    // Get process ID based on priority
    const processId = await getProcessId(db);

    let checkpoint = await getCheckpoint(db, processId);
    logger.info(`Starting from checkpoint: ${JSON.stringify(checkpoint)}`);

    const startTime = Date.now();
    let currentSource = null;

    while (true) {
        // Check time limit
        if (Date.now() - startTime > timeLimit) {
            logger.info(`Approaching time limit for process ${processId}`);
            await saveCheckpoint(db, checkpoint, processId);
            return checkpoint.processedCount;
        }

        // Get next batch of documents
        const documents = await getPendingDocuments(db, checkpoint.lastProcessedId, MAX_DOCUMENTS_PER_BATCH);
        if (documents.length === 0) {
            break;
        }

        logger.info(`Processing batch of ${documents.length} documents for process ${processId}`);

        // Process documents in smaller batches for Weaviate
        const results = await createWeaviateObjects(documents);

        // Update MongoDB and handle signals
        for (const result of results) {
            if (result.success) {
                const doc = result.doc;

                // Check if we're starting a new source
                if (currentSource !== doc.properties.source) {
                    currentSource = doc.properties.source;
                    logger.info(`Processing documents for source: ${currentSource}`);
                }

                // Update timestamp in MongoDB
                const timestamp = new Date();
                await updateRagTimestamp(db, doc._id, timestamp);

                // Update checkpoint
                checkpoint.lastProcessedId = doc._id;
                checkpoint.processedCount++;
                checkpoint.lastSource = doc.properties.source;
                checkpoint.lastDataType = doc.properties.metadata.data_type;

                // Save checkpoint periodically
                if (checkpoint.processedCount % CHECKPOINT_INTERVAL === 0) {
                    await saveCheckpoint(db, checkpoint, processId);
                    logger.info(`Saved checkpoint at ${checkpoint.processedCount} documents`);
                }

                // Check if this is the last document for this source
                const { isLastDocument, dataType } = await checkLastDocumentForSource(db, currentSource);
                if (isLastDocument) {
                    await createSignalRag(db, currentSource, dataType);
                }
            } else {
                logger.error(`Failed to process document ${result.doc.id}:`, result.error);
            }
        }
    }

    // Clear checkpoint if all documents are processed
    await db.collection('AI_TEMP').deleteOne({
        type: 'rag_checkpoint',
        processId: processId
    });

    return checkpoint.processedCount;
}

async function addToRAG() {
    const client = new MongoClient(config.mongoUri);

    try {
        await client.connect();
        const db = client.db(config.databaseName);

        const processedCount = await processDocuments(db);

        return {
            status: 'success',
            message: 'All documents processed',
            processedCount,
            processId: config.processId
        };

    } catch (error) {
        logger.error(`Error processing documents for process ${config.processId}:`, error);
        throw error;
    } finally {
        await client.close();
    }
}

// Digital Ocean Functions entry point
export async function main(req, res) {
    try {
        const result = await addToRAG();
        res.status(200).json(result);
    } catch (error) {
        logger.error('Function execution failed:', error);
        res.status(500).json({
            error: error.message,
            processId: config.processId
        });
    }
}