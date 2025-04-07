import { MongoClient } from 'mongodb';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';

export async function main(params) {    // Configuration
    const config = {
        mongoUri: process.env.MONGO_URI,
        openAiApiKey: process.env.OPENAI_API_KEY,
        maxTokensPerBatch: 8000, // Slightly below limit for safety
        embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
        batchSize: parseInt(process.env.BATCH_SIZE || '16'), // Number of chunks per batch
        retryAttempts: parseInt(process.env.RETRY_ATTEMPTS || '3'),
        retryDelay: parseInt(process.env.RETRY_DELAY || '1000') // ms
    };

    let client = null;

    try {
        // Connect to MongoDB
        client = new MongoClient(config.mongoUri);

        await client.connect();
        console.log('[SendForEmbed] Connected to MongoDB');

        const db = client.db();
        const chunkedCollection = db.collection('MATRIX_CHUNKED');
        const embeddingCollection = db.collection('AI_EMBEDDING');

        // Get all chunks that need embedding (not already in AI_EMBEDDING)
        const processedVectorIds = await embeddingCollection.distinct('vector_id');

        // Get all chunks
        const chunks = await chunkedCollection.find({}).toArray();

        const chunksToProcess = [];
        let skippedCount = 0;

        // Filter chunks that have already been processed
        for (const chunk of chunks) {
            const vectorId = `${chunk.source}__p${chunk.page_counter}__c${chunk.chunk_id}`;

            // Skip if already processed
            if (processedVectorIds.includes(vectorId)) {
                skippedCount++;
                continue;
            }

            chunksToProcess.push({
                vectorId,
                content: chunk.html_content,
                chunk
            });
        }

        if (chunksToProcess.length === 0) {
            console.log('[SendForEmbed] No new chunks found to process');
            return {
                success: true,
                message: 'No new chunks to process',
                stats: { processed: 0, skipped: skippedCount }
            };
        }

        console.log(`[SendForEmbed] Found ${chunksToProcess.length} chunks to process (${skippedCount} skipped)`);

        // Create batches
        const batches = [];
        for (let i = 0; i < chunksToProcess.length; i += config.batchSize) {
            batches.push(chunksToProcess.slice(i, i + config.batchSize));
        }

        console.log(`[SendForEmbed] Created ${batches.length} batches for processing`);

        // Process each batch
        for (const [index, batch] of batches.entries()) {
            // Generate batch ID
            const batchId = uuidv4();

            console.log(`[SendForEmbed] Processing batch ${index + 1}/${batches.length}: ID ${batchId}, ${batch.length} chunks`);

            // Store batch info in MongoDB Collection2 [AI_EMBEDDING]
            const embeddingDocs = batch.map(item => ({
                batch_id: batchId,
                vector_id: item.vectorId,
                html_content: item.content,
                vector: [], // Empty for now, will be updated after retrieval
                timestamp: null // Empty for now, will be updated after retrieval
            }));

            // Insert records into AI_EMBEDDING collection
            await embeddingCollection.insertMany(embeddingDocs);

            // Send batch to OpenAI (to be retrieved later by SCRIPT2)
            // In a production environment, we're just registering the batch for later retrieval
            try {
                // Note: This call is just for validation that the OpenAI API is working
                // SCRIPT2 will get the actual embeddings later
                await axios.post(
                    'https://api.openai.com/v1/embeddings',
                    {
                        model: config.embeddingModel,
                        input: [batch[0].content] // Just send first item as a test
                    },
                    {
                        headers: {
                            'Authorization': `Bearer ${config.openAiApiKey}`,
                            'Content-Type': 'application/json'
                        }
                    }
                );

                console.log(`[SendForEmbed] Successfully validated OpenAI API for batch ${batchId}`);
            } catch (error) {
                console.error(`[SendForEmbed] Error validating OpenAI API for batch ${batchId}:`, error.message);

                // Mark batch with error flag
                await embeddingCollection.updateMany(
                    { batch_id: batchId },
                    { $set: { error: error.message } }
                );
            }
        }

        return {
            success: true,
            message: `Registered ${chunksToProcess.length} chunks in ${batches.length} batches for embedding (${skippedCount} skipped)`,
            stats: {
                processed: chunksToProcess.length,
                skipped: skippedCount,
                batches: batches.length
            }
        };
    } catch (error) {
        console.error('[SendForEmbed] Fatal error:', error);
        return {
            success: false,
            error: error.message
        };
    } finally {
        if (client) {
            await client.close();
            console.log('[SendForEmbed] MongoDB connection closed');
        }
    }
}