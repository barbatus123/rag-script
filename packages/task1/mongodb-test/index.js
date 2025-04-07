import { MongoClient } from 'mongodb';

/**
 * Test function to check MongoDB permissions for the embedding schema
 */
export async function main(args) {
    // Configuration
    const config = {
        mongoUri: process.env.MONGO_URI || 'mongodb://localhost:27017'
    };

    let client = null;

    try {
        // Connect to MongoDB
        console.log('[Test] Attempting to connect to MongoDB...');
        client = new MongoClient(config.mongoUri);
        await client.connect();
        console.log('[Test] Successfully connected to MongoDB');

        const db = client.db();

        // Test 1: List all collections
        console.log('[Test] Checking available collections...');
        const collections = await db.listCollections().toArray();
        console.log(`[Test] Found ${collections.length} collections:`);
        collections.forEach(collection => {
            console.log(`- ${collection.name}`);
        });

        // Test 2: Check if required collections exist
        const requiredCollections = ['MATRIX_CHUNKED', 'AI_EMBEDDING', 'AI_FOR_RAG', 'SIGNAL_RAG'];
        console.log('[Test] Checking required collections...');

        for (const collectionName of requiredCollections) {
            const exists = collections.some(c => c.name === collectionName);
            console.log(`- ${collectionName}: ${exists ? 'EXISTS' : 'MISSING'}`);
        }

        // Test 3: Check read permissions on MATRIX_CHUNKED
        console.log('[Test] Testing read permissions on MATRIX_CHUNKED...');
        try {
            const count = await db.collection('MATRIX_CHUNKED').countDocuments({});
            console.log(`[Test] Can read MATRIX_CHUNKED - found ${count} documents`);

            // Get a sample document if available
            if (count > 0) {
                const sample = await db.collection('MATRIX_CHUNKED').findOne({});
                console.log('[Test] Sample document structure:', JSON.stringify(sample, null, 2));
            }
        } catch (error) {
            console.error('[Test] Failed to read from MATRIX_CHUNKED:', error.message);
        }

        // Test 4: Check write permissions on AI_EMBEDDING
        console.log('[Test] Testing write permissions on AI_EMBEDDING...');
        try {
            const testId = `test_${Date.now()}`;
            const result = await db.collection('AI_EMBEDDING').insertOne({
                batch_id: testId,
                vector_id: testId,
                html_content: 'Test content',
                vector: [],
                timestamp: null,
                test: true
            });

            console.log('[Test] Successfully wrote to AI_EMBEDDING:', result.acknowledged);

            // Clean up the test document
            await db.collection('AI_EMBEDDING').deleteOne({ batch_id: testId });
            console.log('[Test] Successfully deleted test document from AI_EMBEDDING');
        } catch (error) {
            console.error('[Test] Failed to write to AI_EMBEDDING:', error.message);
        }

        return {
            success: true,
            message: 'MongoDB permissions test completed',
            results: {
                collectionsFound: collections.map(c => c.name),
                requiredCollections: requiredCollections.map(name => {
                    return {
                        name,
                        exists: collections.some(c => c.name === name)
                    };
                })
            }
        };
    } catch (error) {
        console.error('[Test] Fatal error:', error);
        return {
            success: false,
            error: error.message
        };
    } finally {
        if (client) {
            await client.close();
            console.log('[Test] MongoDB connection closed');
        }
    }
}