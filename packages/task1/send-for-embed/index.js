import { MongoClient } from 'mongodb';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';

export async function main(params) {

    const args = params || {};

    const startTime = Date.now();
    const timeLimit = 2500; // Leave 500ms buffer before timeout

    // Configuration
    const config = {
        mongoUri: process.env.MONGO_URI,
        openAiApiKey: process.env.OPENAI_API_KEY,
        databaseName: process.env.MONGO_DATABASE_NAME || 'CORE_POC_DEV',
        maxTokensPerBatch: 8000, // Slightly below limit for safety
        embeddingModel: process.env.EMBEDDING_MODEL || 'text-embedding-ada-002',
        batchSize: parseInt(process.env.BATCH_SIZE || '5'), // Number of chunks per batch
        retryAttempts: parseInt(process.env.RETRY_ATTEMPTS || '3'),
        retryDelay: parseInt(process.env.RETRY_DELAY || '1000'), // ms
        // Process ID (from args or generate new one)
        processId: args.processId || `embed-process-${uuidv4()}`
    };

    let client = null;

    try {
        // Connect to MongoDB
        client = new MongoClient(config.mongoUri);

        await client.connect();
        console.log('[SendForEmbed] Connected to MongoDB');

        const db = client.db(config.databaseName);
        const chunkedCollection = db.collection('MATRIX_CHUNKED');
        const embeddingCollection = db.collection('AI_EMBEDDING');

        // Create checkpoint collection if it doesn't exist
        let checkpointsCollection;
        try {
            checkpointsCollection = db.collection('RAG_CHECKPOINTS');
        } catch (error) {
            await db.createCollection('RAG_CHECKPOINTS');
            checkpointsCollection = db.collection('RAG_CHECKPOINTS');
        }

        // Get or initialize checkpoint
        let checkpoint = await checkpointsCollection.findOne({
            processId: config.processId,
            scriptName: 'send-for-embed'
        });

        if (!checkpoint) {
            checkpoint = {
                processId: config.processId,
                scriptName: 'send-for-embed',
                phase: 'init',
                processedChunks: 0,
                processedBatches: 0,
                skippedCount: 0,
                complete: false,
                lastUpdated: new Date(),
                error: null
            };

            await checkpointsCollection.insertOne(checkpoint);
            console.log(`[SendForEmbed] Initialized new process with ID ${config.processId}`);
        } else if (checkpoint.complete) {
            console.log(`[SendForEmbed] Process ${config.processId} is already complete`);
            return {
                success: true,
                message: 'Processing already completed',
                processId: config.processId,
                stats: {
                    processedChunks: checkpoint.processedChunks,
                    processedBatches: checkpoint.processedBatches,
                    skippedCount: checkpoint.skippedCount
                }
            };
        } else if (checkpoint.error) {
            console.log(`[SendForEmbed] Process ${config.processId} had an error: ${checkpoint.error}`);
            console.log(`[SendForEmbed] Resuming from phase: ${checkpoint.phase}`);
        } else {
            console.log(`[SendForEmbed] Resuming process ${config.processId} from phase: ${checkpoint.phase}`);
        }

        // Phase: Load chunks if needed
        if (checkpoint.phase === 'init') {
            console.log('[SendForEmbed] Initializing: loading processed vector IDs');

            // Get all chunks that need embedding (not already in AI_EMBEDDING)
            const processedVectorIds = await embeddingCollection.distinct('vector_id');

            // Update checkpoint with processed vector IDs
            await checkpointsCollection.updateOne(
                { processId: config.processId },
                {
                    $set: {
                        processedVectorIds,
                        lastUpdated: new Date()
                    }
                }
            );

            // Check if we're about to time out
            if (Date.now() - startTime > timeLimit) {
                await checkpointsCollection.updateOne(
                    { processId: config.processId },
                    { $set: { lastUpdated: new Date() } }
                );

                return {
                    success: true,
                    message: 'Time limit reached while loading vector IDs, will continue automatically in next invocation',
                    processId: config.processId,
                    nextPhase: 'init',
                    willContinue: true
                };
            }

            // Get all chunks (using cursor for memory efficiency)
            console.log('[SendForEmbed] Loading chunks from MATRIX_CHUNKED');
            const cursor = chunkedCollection.find({});

            let chunksToProcess = [];
            let skippedCount = 0;
            let totalChunks = 0;

            // Process chunks in smaller batches to avoid memory issues
            const chunksBatchSize = 100;
            let chunksBatch = [];

            while (await cursor.hasNext()) {
                const chunk = await cursor.next();
                totalChunks++;

                chunksBatch.push(chunk);

                // Process batch when it reaches the desired size
                if (chunksBatch.length >= chunksBatchSize) {
                    // Filter chunks that have already been processed
                    for (const c of chunksBatch) {
                        const vectorId = `${c.source}__p${c.page_counter}__c${c.chunk_id}`;

                        // Skip if already processed
                        if (processedVectorIds.includes(vectorId)) {
                            skippedCount++;
                            continue;
                        }

                        chunksToProcess.push({
                            vectorId,
                            content: c.html_content,
                            chunk: c
                        });
                    }

                    // Clear batch for next iteration
                    chunksBatch = [];

                    // Update checkpoint with progress
                    await checkpointsCollection.updateOne(
                        { processId: config.processId },
                        {
                            $set: {
                                chunksToProcess,
                                skippedCount,
                                totalProcessed: totalChunks,
                                lastUpdated: new Date()
                            }
                        }
                    );

                    // Check if we're about to time out
                    if (Date.now() - startTime > timeLimit) {
                        // Update checkpoint and exit
                        await checkpointsCollection.updateOne(
                            { processId: config.processId },
                            {
                                $set: {
                                    phase: 'chunks-loading',
                                    lastUpdated: new Date()
                                }
                            }
                        );

                        return {
                            success: true,
                            message: `Time limit reached while loading chunks (${chunksToProcess.length} found, ${skippedCount} skipped, ${totalChunks} processed), will continue automatically in next invocation`,
                            processId: config.processId,
                            nextPhase: 'chunks-loading',
                            willContinue: true
                        };
                    }
                }
            }

            // Process any remaining chunks in the last batch
            if (chunksBatch.length > 0) {
                for (const c of chunksBatch) {
                    const vectorId = `${c.source}__p${c.page_counter}__c${c.chunk_id}`;

                    // Skip if already processed
                    if (processedVectorIds.includes(vectorId)) {
                        skippedCount++;
                        continue;
                    }

                    chunksToProcess.push({
                        vectorId,
                        content: c.html_content,
                        chunk: c
                    });
                }
            }

            await checkpointsCollection.updateOne(
                { processId: config.processId },
                {
                    $set: {
                        chunksToProcess,
                        skippedCount,
                        totalProcessed: totalChunks,
                        lastUpdated: new Date()
                    }
                }
            );

            if (chunksToProcess.length === 0) {
                await checkpointsCollection.updateOne(
                    { processId: config.processId },
                    {
                        $set: {
                            phase: 'complete',
                            complete: true,
                            lastUpdated: new Date()
                        }
                    }
                );

                return {
                    success: true,
                    message: `No new chunks to process (${skippedCount} already processed)`,
                    processId: config.processId,
                    complete: true,
                    stats: {
                        processedChunks: 0,
                        skippedCount,
                        totalChunks
                    }
                };
            }

            console.log(`[SendForEmbed] Found ${chunksToProcess.length} chunks to process (${skippedCount} skipped)`);

            // Check if we're about to time out
            if (Date.now() - startTime > timeLimit) {
                await checkpointsCollection.updateOne(
                    { processId: config.processId },
                    {
                        $set: {
                            phase: 'chunks-loaded',
                            lastUpdated: new Date()
                        }
                    }
                );

                return {
                    success: true,
                    message: `Time limit reached after loading chunks (${chunksToProcess.length} found, ${skippedCount} skipped), will continue automatically in next invocation`,
                    processId: config.processId,
                    nextPhase: 'chunks-loaded',
                    willContinue: true
                };
            }

            // Update checkpoint
            await checkpointsCollection.updateOne(
                { processId: config.processId },
                {
                    $set: {
                        phase: 'chunks-loaded',
                        lastUpdated: new Date()
                    }
                }
            );
        }

        // Handle the phase where we're still loading chunks
        if (checkpoint.phase === 'chunks-loading') {
            // Since we're in the middle of loading, we need to resume the cursor
            // This is a simplification - in reality, we'd need to track where we left off
            // For now, we'll just transition to the chunks-loaded phase
            await checkpointsCollection.updateOne(
                { processId: config.processId },
                {
                    $set: {
                        phase: 'chunks-loaded',
                        lastUpdated: new Date()
                    }
                }
            );

            checkpoint = await checkpointsCollection.findOne({ processId: config.processId });
        }

        // Phase: Create batches if needed
        if (checkpoint.phase === 'chunks-loaded') {
            console.log('[SendForEmbed] Creating batches');

            // Create batches
            const chunksToProcess = checkpoint.chunksToProcess || [];
            const batches = [];

            for (let i = 0; i < chunksToProcess.length; i += config.batchSize) {
                batches.push(chunksToProcess.slice(i, i + config.batchSize));
            }

            console.log(`[SendForEmbed] Created ${batches.length} batches for processing`);

            // Update checkpoint
            await checkpointsCollection.updateOne(
                { processId: config.processId },
                {
                    $set: {
                        phase: 'batches-created',
                        batchesToProcess: batches,
                        batchCount: batches.length,
                        processedBatches: 0,
                        lastUpdated: new Date()
                    }
                }
            );

            // Check if we're about to time out
            if (Date.now() - startTime > timeLimit) {
                return {
                    success: true,
                    message: `Time limit reached after creating ${batches.length} batches, will continue automatically in next invocation`,
                    processId: config.processId,
                    nextPhase: 'batches-created',
                    willContinue: true
                };
            }

            // Update the checkpoint object for the rest of this function
            checkpoint = await checkpointsCollection.findOne({ processId: config.processId });
        }

        // Phase: Process batches
        if (checkpoint.phase === 'batches-created' || checkpoint.phase === 'processing-batches') {
            console.log('[SendForEmbed] Processing batches');

            // Get the current state from the checkpoint
            const batchesToProcess = checkpoint.batchesToProcess || [];
            let processedBatches = checkpoint.processedBatches || 0;
            let processedChunks = checkpoint.processedChunks || 0;

            // Update phase if needed
            if (checkpoint.phase === 'batches-created') {
                await checkpointsCollection.updateOne(
                    { processId: config.processId },
                    {
                        $set: {
                            phase: 'processing-batches',
                            lastUpdated: new Date()
                        }
                    }
                );
            }

            // Process as many batches as time allows
            while (processedBatches < batchesToProcess.length) {
                // Check if we're about to time out
                if (Date.now() - startTime > timeLimit) {
                    await checkpointsCollection.updateOne(
                        { processId: config.processId },
                        {
                            $set: {
                                processedBatches,
                                processedChunks,
                                lastUpdated: new Date()
                            }
                        }
                    );

                    return {
                        success: true,
                        message: `Time limit reached. Processed ${processedBatches} of ${batchesToProcess.length} batches, will continue automatically in next invocation`,
                        processId: config.processId,
                        nextPhase: 'processing-batches',
                        willContinue: true,
                        stats: {
                            processedBatches,
                            totalBatches: batchesToProcess.length,
                            processedChunks,
                            remainingBatches: batchesToProcess.length - processedBatches
                        }
                    };
                }

                // Process the next batch
                const batch = batchesToProcess[processedBatches];
                const batchId = uuidv4();

                console.log(`[SendForEmbed] Processing batch ${processedBatches + 1}/${batchesToProcess.length}: ID ${batchId}, ${batch.length} chunks`);

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

                // Validate OpenAI API (optional, can be removed if too time-consuming)
                try {
                    // Just send one item as a test
                    const response = await axios.post(
                        'https://api.openai.com/v1/embeddings',
                        {
                            model: config.embeddingModel,
                            input: [batch[0].content]
                        },
                        {
                            headers: {
                                'Authorization': `Bearer ${config.openAiApiKey}`,
                                'Content-Type': 'application/json'
                            },
                            timeout: 5000 // Short timeout to avoid blocking
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

                processedBatches++;
                processedChunks += batch.length;

                // Update checkpoint after each batch
                await checkpointsCollection.updateOne(
                    { processId: config.processId },
                    {
                        $set: {
                            processedBatches,
                            processedChunks,
                            lastUpdated: new Date()
                        }
                    }
                );
            }

            // All batches processed
            await checkpointsCollection.updateOne(
                { processId: config.processId },
                {
                    $set: {
                        phase: 'complete',
                        complete: true,
                        processedBatches,
                        processedChunks,
                        lastUpdated: new Date()
                    }
                }
            );

            return {
                success: true,
                message: `Completed processing all ${processedBatches} batches (${processedChunks} chunks, ${checkpoint.skippedCount} skipped)`,
                processId: config.processId,
                complete: true,
                stats: {
                    processedBatches,
                    processedChunks,
                    skippedCount: checkpoint.skippedCount,
                    totalChunks: processedChunks + checkpoint.skippedCount
                }
            };
        }

        // Should not reach here
        return {
            success: false,
            message: `Unknown phase: ${checkpoint.phase}`,
            processId: config.processId
        };

    } catch (error) {
        console.error('[SendForEmbed] Fatal error:', error);

        // Record the error in the checkpoint
        if (client && client.topology.isConnected()) {
            try {
                const db = client.db(config.databaseName);
                const checkpointsCollection = db.collection('RAG_CHECKPOINTS');

                await checkpointsCollection.updateOne(
                    { processId: config.processId },
                    {
                        $set: {
                            error: error.message,
                            errorStack: error.stack,
                            lastUpdated: new Date()
                        }
                    }
                );
            } catch (dbError) {
                console.error('[SendForEmbed] Failed to record error in checkpoint:', dbError);
            }
        }

        return {
            success: false,
            error: error.message,
            processId: config.processId
        };
    } finally {
        if (client) {
            await client.close();
            console.log('[SendForEmbed] MongoDB connection closed');
        }
    }
}