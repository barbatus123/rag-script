import { MongoClient } from 'mongodb';
import weaviate from 'weaviate-ts-client';

/**
 * AddToRAG - Inserts data from MongoDB Collection3 [AI_FOR_RAG] into Weaviate vector DB
 */
export async function main(params) {    // Configuration
    const config = {
        mongoUri: process.env.MONGO_URI,
        weaviateScheme: process.env.WEAVIATE_SCHEME || 'https',
        weaviateHost: process.env.WEAVIATE_HOST,
        weaviateApiKey: process.env.WEAVIATE_API_KEY,
        processLimit: parseInt(process.env.PROCESS_LIMIT || '100'), // Max documents to process in one run
        retryAttempts: parseInt(process.env.RETRY_ATTEMPTS || '3'),
        retryDelay: parseInt(process.env.RETRY_DELAY || '1000') // ms
    };

    // Collection mappings
    const collectionMappings = {
        'website_structure': 'WebsiteStructure',
        'script_example': 'ScriptExample',
        'sdk_documentation': 'SDKDocumentation'
    };

    let client = null;
    let weaviateClient = null;

    // Stats tracking
    const stats = {
        processed: 0,
        failed: 0,
        signalsSent: 0,
        sourceStats: {}
    };

    try {
        // Connect to MongoDB
        client = new MongoClient(config.mongoUri);

        await client.connect();
        console.log('[AddToRAG] Connected to MongoDB');

        // Initialize Weaviate client
        weaviateClient = weaviate.client({
            scheme: config.weaviateScheme,
            host: config.weaviateHost,
            apiKey: new weaviate.ApiKey(config.weaviateApiKey)
        });

        // Test Weaviate connection
        await weaviateClient.schema.getter().do();
        console.log('[AddToRAG] Connected to Weaviate');

        const db = client.db();
        const forRagCollection = db.collection('AI_FOR_RAG');
        const signalRagCollection = db.collection('SIGNAL_RAG');

        // Find documents in AI_FOR_RAG without a timestamp
        const docsToImportCount = await forRagCollection.countDocuments({ 'properties.rag_timestamp': null });

        if (docsToImportCount === 0) {
            console.log('[AddToRAG] No documents to import to Weaviate');
            return {
                success: true,
                message: 'No documents to import',
                stats
            };
        }

        console.log(`[AddToRAG] Found ${docsToImportCount} documents to import to Weaviate (processing up to ${config.processLimit})`);

        // Get documents to import with a limit
        const docsToImport = await forRagCollection
            .find({ 'properties.rag_timestamp': null })
            .limit(config.processLimit)
            .toArray();

        // Group documents by source and data type for tracking
        const docsBySource = {};
        const sourceTotal = {};

        for (const doc of docsToImport) {
            const source = doc.properties.source;
            const dataType = doc.properties.metadata.data_type;

            if (!docsBySource[source]) {
                docsBySource[source] = {};
                sourceTotal[source] = await forRagCollection.countDocuments({ 'properties.source': source });

                stats.sourceStats[source] = {
                    total: sourceTotal[source],
                    processed: 0,
                    failed: 0,
                    completed: false
                };
            }

            if (!docsBySource[source][dataType]) {
                docsBySource[source][dataType] = [];
            }

            docsBySource[source][dataType].push(doc);
        }

        // Process documents by source and data type
        const sources = Object.keys(docsBySource);

        for (const source of sources) {
            console.log(`[AddToRAG] Processing source: ${source}`);

            const dataTypes = Object.keys(docsBySource[source]);

            for (const dataType of dataTypes) {
                // Map dataType to Weaviate collection
                const weaviateCollection = collectionMappings[dataType];

                if (!weaviateCollection) {
                    console.error(`[AddToRAG] Unknown data_type: ${dataType} for source ${source}`);
                    continue;
                }

                console.log(`[AddToRAG] Processing data type: ${dataType} => ${weaviateCollection}`);

                const docs = docsBySource[source][dataType];

                // Check if Weaviate collection exists and create schema if not
                try {
                    const schemaRes = await weaviateClient.schema.getter().do();
                    const collectionExists = schemaRes.classes && schemaRes.classes.some(c => c.class === weaviateCollection);

                    if (!collectionExists) {
                        console.log(`[AddToRAG] Creating schema for collection ${weaviateCollection}`);

                        // Create schema based on collection type
                        const schemaDefinition = {
                            class: weaviateCollection,
                            description: `A chunk of ${dataType.replace('_', ' ')} including vector and metadata for RAG`,
                            vectorizer: 'none',
                            properties: [
                                {name: 'chunk_id', dataType: ['int']},
                                {name: 'page_counter', dataType: ['int']},
                                {name: 'source', dataType: ['text']},
                                {name: 'rag_timestamp', dataType: ['date']},
                                {name: 'html_content', dataType: ['text']},
                                {name: 'metadata_data_type', dataType: ['text']},
                                {name: 'metadata_timestamp', dataType: ['date']},
                                {name: 'metadata_crawl_depth', dataType: ['int']},
                                {name: 'metadata_title', dataType: ['text']}
                            ]
                        };

                        await weaviateClient.schema.classCreator().withClass(schemaDefinition).do();
                        console.log(`[AddToRAG] Created schema for collection ${weaviateCollection}`);
                    }
                } catch (error) {
                    console.error(`[AddToRAG] Error checking/creating schema for ${weaviateCollection}:`, error.message);
                    continue;
                }

                // Process each document individually
                for (const doc of docs) {
                    try {
                        // Prepare data object for Weaviate
                        const weaviateObj = {
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

                        // Try to insert into Weaviate with retries
                        let success = false;
                        let attempts = 0;

                        while (!success && attempts < config.retryAttempts) {
                            attempts++;
                            try {
                                // Insert into Weaviate
                                await weaviateClient.data
                                    .creator()
                                    .withClassName(weaviateCollection)
                                    .withId(doc.id)
                                    .withProperties(weaviateObj)
                                    .withVector(doc.vector)
                                    .do();

                                success = true;

                                // Update timestamp in AI_FOR_RAG
                                await forRagCollection.updateOne(
                                    { id: doc.id },
                                    { $set: { 'properties.rag_timestamp': new Date() } }
                                );

                                // Update stats
                                stats.processed++;
                                stats.sourceStats[source].processed++;

                                console.log(`[AddToRAG] Inserted document ${doc.id} into Weaviate collection ${weaviateCollection}`);
                            } catch (error) {
                                console.error(`[AddToRAG] Error inserting document ${doc.id} into Weaviate (attempt ${attempts}):`, error.message);

                                if (attempts < config.retryAttempts) {
                                    const backoffTime = config.retryDelay * Math.pow(2, attempts - 1);
                                    console.warn(`[AddToRAG] Retrying in ${backoffTime}ms (attempt ${attempts + 1}/${config.retryAttempts})`);
                                    await new Promise(resolve => setTimeout(resolve, backoffTime));
                                } else {
                                    console.error(`[AddToRAG] Failed to insert document ${doc.id} after ${config.retryAttempts} attempts`);
                                    stats.failed++;
                                    stats.sourceStats[source].failed++;
                                }
                            }
                        }
                    } catch (error) {
                        console.error(`[AddToRAG] Error processing document ${doc.id}:`, error.message);
                        stats.failed++;
                        stats.sourceStats[source].failed++;
                    }
                }
            }

            // Check if all documents for this source have been processed
            const remaining = await forRagCollection.countDocuments({
                'properties.source': source,
                'properties.rag_timestamp': null
            });

            // If no remaining documents, insert signal in SIGNAL_RAG
            if (remaining === 0) {
                console.log(`[AddToRAG] All documents for source ${source} have been processed`);
                stats.sourceStats[source].completed = true;

                // Determine the collection based on the most common data type for this source
                let primaryCollection = '';
                let maxCount = 0;

                for (const dataType of Object.keys(docsBySource[source])) {
                    const count = docsBySource[source][dataType].length;
                    if (count > maxCount) {
                        maxCount = count;
                        primaryCollection = collectionMappings[dataType];
                    }
                }

                // Insert signal in SIGNAL_RAG
                await signalRagCollection.insertOne({
                    source: source,
                    collection: primaryCollection,
                    timestamp: new Date()
                });

                stats.signalsSent++;
                console.log(`[AddToRAG] Inserted signal in SIGNAL_RAG for source ${source} with collection ${primaryCollection}`);
            } else {
                console.log(`[AddToRAG] ${remaining} documents remaining for source ${source}`);
            }
        }

        return {
            success: true,
            message: `Imported ${stats.processed} documents to Weaviate (${stats.failed} failed, ${stats.signalsSent} signals sent)`,
            stats
        };
    } catch (error) {
        console.error('[AddToRAG] Fatal error:', error);
        return {
            success: false,
            error: error.message,
            stack: error.stack,
            stats
        };
    } finally {
        if (client) {
            await client.close();
            console.log('[AddToRAG] MongoDB connection closed');
        }
    }
}