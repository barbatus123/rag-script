import { MongoClient } from 'mongodb';
import axios from 'axios';

/**
 * RetrieveEmbed - Retrieves embedding results from OpenAI and merges them with original chunks
 * This function will be triggered via Cron
 */
export async function main(params) {    // Configuration
    const config = {
        mongoUri: process.env.MONGO_URI,
        openAiApiKey: process.env.OPENAI_API_KEY,
        embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
        batchProcessLimit: parseInt(process.env.BATCH_PROCESS_LIMIT || '10'), // Max batches to process in one run
        retryAttempts: parseInt(process.env.RETRY_ATTEMPTS || '3'),
        retryDelay: parseInt(process.env.RETRY_DELAY || '1000') // ms
    };

    let client = null;
    const processedBatches = [];
    const failedBatches = [];

    try {
        // Connect to MongoDB
        client = new MongoClient(config.mongoUri);

        await client.connect();
        console.log('[RetrieveEmbed] Connected to MongoDB');

        const db = client.db();
        const embeddingCollection = db.collection('AI_EMBEDDING');
        const chunkedCollection = db.collection('MATRIX_CHUNKED');
        const forRagCollection = db.collection('AI_FOR_RAG');

        // Find batches that need to be retrieved (no timestamp)
        // Also filter out batches with error flag that have been retried too many times
        const batchesToRetrieve = await embeddingCollection.aggregate([
            {
                $match: {
                    timestamp: null,
                    $or: [
                        { retry_count: { $exists: false } },
                        { retry_count: { $lt: config.retryAttempts } }
                    ]
                }
            },
            { $group: { _id: "$batch_id" } },
            { $limit: config.batchProcessLimit }
        ]).toArray();

        if (batchesToRetrieve.length === 0) {
            console.log('[RetrieveEmbed] No batches to retrieve');
            return {
                success: true,
                message: 'No batches to retrieve',
                stats: { processed: 0, failed: 0 }
            };
        }

        const batchIds = batchesToRetrieve.map(b => b._id);
        console.log(`[RetrieveEmbed] Found ${batchIds.length} batches to retrieve: ${batchIds.join(', ')}`);

        // Process each batch
        for (const batchData of batchesToRetrieve) {
            const batchId = batchData._id;
            console.log(`[RetrieveEmbed] Processing batch ${batchId}`);

            try {
                // Get all documents for this batch
                const batchDocuments = await embeddingCollection.find({ batch_id: batchId }).toArray();

                if (batchDocuments.length === 0) {
                    console.log(`[RetrieveEmbed] No documents found for batch ${batchId}`);
                    continue;
                }

                console.log(`[RetrieveEmbed] Found ${batchDocuments.length} documents in batch ${batchId}`);

                // Create input array for OpenAI embedding API
                const contentArray = batchDocuments.map(doc => doc.html_content);

                // Retrieve embeddings from OpenAI with retries
                let embeddings = null;
                let attempts = 0;
                let success = false;

                while (!success && attempts < config.retryAttempts) {
                    attempts++;
                    try {
                        // Send request to OpenAI
                        const response = await axios.post(
                            'https://api.openai.com/v1/embeddings',
                            {
                                model: config.embeddingModel,
                                input: contentArray
                            },
                            {
                                headers: {
                                    'Authorization': `Bearer ${config.openAiApiKey}`,
                                    'Content-Type': 'application/json'
                                },
                                timeout: 60000 // 60 seconds timeout for larger batches
                            }
                        );

                        embeddings = response.data.data;
                        success = true;

                        console.log(`[RetrieveEmbed] Successfully retrieved embeddings for batch ${batchId}`);
                    } catch (error) {
                        const status = error.response?.status;

                        // Handle different error types
                        if (status === 429) {
                            // Rate limit error - back off and retry
                            const retryAfter = error.response.headers['retry-after']
                                ? parseInt(error.response.headers['retry-after']) * 1000
                                : config.retryDelay * Math.pow(2, attempts);

                            console.warn(`[RetrieveEmbed] Rate limit hit, retrying after ${retryAfter}ms`);
                            await new Promise(resolve => setTimeout(resolve, retryAfter));
                        } else if (status >= 500) {
                            // Server error - retry with exponential backoff
                            const backoffTime = config.retryDelay * Math.pow(2, attempts);
                            console.warn(`[RetrieveEmbed] OpenAI server error (${status}), retrying after ${backoffTime}ms`);
                            await new Promise(resolve => setTimeout(resolve, backoffTime));
                        } else {
                            // Other errors - log and retry with delay
                            console.error(`[RetrieveEmbed] Error retrieving embeddings for batch ${batchId}:`, error.response?.data || error.message);

                            if (attempts < config.retryAttempts) {
                                const backoffTime = config.retryDelay * Math.pow(2, attempts);
                                console.warn(`[RetrieveEmbed] Retrying in ${backoffTime}ms (attempt ${attempts + 1}/${config.retryAttempts})`);
                                await new Promise(resolve => setTimeout(resolve, backoffTime));
                            } else {
                                console.error(`[RetrieveEmbed] Failed to retrieve embeddings for batch ${batchId} after ${config.retryAttempts} attempts`);

                                // Update retry count for all documents in the batch
                                await embeddingCollection.updateMany(
                                    { batch_id: batchId },
                                    {
                                        $inc: { retry_count: 1 },
                                        $set: { last_error: error.message, last_retry: new Date() }
                                    }
                                );

                                failedBatches.push(batchId);
                                throw new Error(`Failed to retrieve embeddings after ${config.retryAttempts} attempts: ${error.message}`);
                            }
                        }
                    }
                }

                if (!embeddings) {
                    console.error(`[RetrieveEmbed] No embeddings retrieved for batch ${batchId}`);
                    failedBatches.push(batchId);
                    continue;
                }

                // Process documents
                const timestamp = new Date();

                for (let i = 0; i < batchDocuments.length; i++) {
                    const doc = batchDocuments[i];
                    const embedding = embeddings[i].embedding;

                    try {
                        // Update the document with the embedding and timestamp
                        await embeddingCollection.updateOne(
                            { batch_id: batchId, vector_id: doc.vector_id },
                            {
                                $set: {
                                    vector: embedding,
                                    timestamp: timestamp
                                }
                            }
                        );

                        console.log(`[RetrieveEmbed] Updated embedding for ${doc.vector_id}`);

                        // Parse vector_id to get original document info
                        const vectorIdParts = doc.vector_id.split('__');
                        const source = vectorIdParts[0];
                        const pageCounter = parseInt(vectorIdParts[1].substring(1));
                        const chunkId = parseInt(vectorIdParts[2].substring(1));

                        // Get the original document from MATRIX_CHUNKED
                        const originalDoc = await chunkedCollection.findOne({
                            source: source,
                            page_counter: pageCounter,
                            chunk_id: chunkId
                        });

                        if (!originalDoc) {
                            console.error(`[RetrieveEmbed] Original document not found for ${doc.vector_id}`);
                            continue;
                        }

                        // Merge data and insert into AI_FOR_RAG
                        const mergedDoc = {
                            id: doc.vector_id,
                            vector: embedding,
                            properties: {
                                chunk_id: originalDoc.chunk_id,
                                page_counter: originalDoc.page_counter,
                                source: originalDoc.source,
                                rag_timestamp: null, // Will be updated after insertion in Weaviate
                                html_content: originalDoc.html_content,
                                metadata: originalDoc.metadata
                            }
                        };

                        // Upsert into AI_FOR_RAG
                        await forRagCollection.updateOne(
                            { id: doc.vector_id },
                            { $set: mergedDoc },
                            { upsert: true }
                        );

                        console.log(`[RetrieveEmbed] Merged and inserted document for ${doc.vector_id}`);
                    } catch (error) {
                        console.error(`[RetrieveEmbed] Error processing document ${doc.vector_id}:`, error.message);
                    }
                }

                processedBatches.push(batchId);
                console.log(`[RetrieveEmbed] Successfully processed batch ${batchId}`);

            } catch (error) {
                console.error(`[RetrieveEmbed] Error processing batch ${batchId}:`, error.message);
                failedBatches.push(batchId);
            }
        }

        return {
            success: true,
            message: `Retrieved embeddings for ${processedBatches.length} batches (${failedBatches.length} failed)`,
            stats: {
                processed: processedBatches.length,
                failed: failedBatches.length,
                processedBatches,
                failedBatches
            }
        };
    } catch (error) {
        console.error('[RetrieveEmbed] Fatal error:', error);
        return {
            success: false,
            error: error.message,
            stats: {
                processed: processedBatches.length,
                failed: failedBatches.length,
                processedBatches,
                failedBatches
            }
        };
    } finally {
        if (client) {
            await client.close();
            console.log('[RetrieveEmbed] MongoDB connection closed');
        }
    }
}